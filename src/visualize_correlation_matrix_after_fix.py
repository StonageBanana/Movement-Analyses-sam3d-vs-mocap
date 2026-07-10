"""Correlation matrix (heatmap): all 24 comparable signals (18 joint
positions + 6 flexion angles; pelvis excluded as the trivial reference
point) x all 10 trials, Pearson r vs. mocap, AFTER the pelvis-frame fix
(see visualize_hip_diagnostics.py / visualize_correlation_before_after.py).
Joint positions are unaffected by the fix (it only touches how hip/knee/
ankle flexion angles are projected); shown alongside the angles so the full,
final per-joint picture is visible in one place.

All correlations are real-time, per-frame Pearson r (matching
compare_metrics.py's methodology), not gait-cycle-normalized.

Usage: .venv\\Scripts\\python.exe src\\visualize_correlation_matrix_after_fix.py
"""

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent))
from joint_mapping import CANONICAL_JOINTS
from mocap.angles import (
    compute_joint_angles_from_joints,
    wrap_around_center,
    _sagittal_angle_deg,
    _wrap_around_own_circular_mean,
    joint_only_pelvis_frame,
)
from compare_metrics import MOCAP_UP, load_static_angle_offsets

ANALYSIS_DIR = Path(__file__).resolve().parent.parent
ALIGNED_FUSED_DIR = ANALYSIS_DIR / "output" / "aligned_fused"
OUT_DIR = ANALYSIS_DIR / "output" / "diagnostics"

ANGLE_NAMES = ["hip_flexion_left", "hip_flexion_right", "knee_flexion_left",
               "knee_flexion_right", "ankle_flexion_left", "ankle_flexion_right"]
POSITION_JOINTS = [j for j in CANONICAL_JOINTS if j != "pelvis"]
ALL_SIGNALS = POSITION_JOINTS + ANGLE_NAMES

TRIALS = ["walking_1", "walking_2", "walking_3", "running_1", "running_2",
          "dance_move_1", "feet_movements_1", "random_1", "squats_1", "squats_2"]


def _corr(a, b):
    valid = ~np.isnan(a) & ~np.isnan(b)
    if valid.sum() < 10 or np.std(a[valid]) == 0 or np.std(b[valid]) == 0:
        return float("nan")
    return float(stats.pearsonr(a[valid], b[valid])[0])


def corr_for_trial(trial: str) -> dict:
    d = np.load(ALIGNED_FUSED_DIR / f"{trial}.npz")
    mocap_joints = {j: d[f"mocap__{j}"] for j in CANONICAL_JOINTS}
    fused_joints = {j: d[f"aligned__{j}"] for j in CANONICAL_JOINTS}

    out = {}
    pelvis_y_mocap = mocap_joints["pelvis"][:, 1]
    pelvis_y_fused = fused_joints["pelvis"][:, 1]
    for j in POSITION_JOINTS:
        mocap_sig = mocap_joints[j][:, 1] - pelvis_y_mocap
        fused_sig = fused_joints[j][:, 1] - pelvis_y_fused
        out[j] = _corr(mocap_sig, fused_sig)

    static_offsets = load_static_angle_offsets()
    mocap_angles_raw = compute_joint_angles_from_joints(mocap_joints, MOCAP_UP)
    x_hat_m, z_hat_m = joint_only_pelvis_frame(mocap_joints["hip_left"], mocap_joints["hip_right"], MOCAP_UP)

    for name in ANGLE_NAMES:
        side = name.split("_")[-1]
        offset = static_offsets[name]
        mocap_sig = mocap_angles_raw[name] - offset

        hip, knee, ankle = fused_joints[f"hip_{side}"], fused_joints[f"knee_{side}"], fused_joints[f"ankle_{side}"]
        thigh_ang_c = _wrap_around_own_circular_mean(_sagittal_angle_deg(knee - hip, x_hat_m, z_hat_m))
        shank_ang_c = _wrap_around_own_circular_mean(_sagittal_angle_deg(ankle - knee, x_hat_m, z_hat_m))
        foot_ang_c = _wrap_around_own_circular_mean(_sagittal_angle_deg(
            fused_joints[f"toe_{side}"] - fused_joints[f"heel_{side}"], x_hat_m, z_hat_m))
        if name.startswith("hip"):
            after_raw = thigh_ang_c
        elif name.startswith("knee"):
            after_raw = wrap_around_center(shank_ang_c - thigh_ang_c, 0.0)
        else:
            after_raw = wrap_around_center(foot_ang_c - shank_ang_c, 0.0)
        after_sig = mocap_sig + wrap_around_center(after_raw - offset - mocap_sig, 0.0)
        out[name] = _corr(mocap_sig, after_sig)

    return out


def main():
    out_dir = OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    matrix = np.full((len(ALL_SIGNALS), len(TRIALS)), np.nan)
    for tc, trial in enumerate(TRIALS):
        res = corr_for_trial(trial)
        for rc, sig in enumerate(ALL_SIGNALS):
            matrix[rc, tc] = res[sig]

    col_mean = np.nanmean(matrix, axis=1)
    order = np.argsort(-col_mean)
    signals_sorted = [ALL_SIGNALS[i] for i in order]
    matrix_sorted = matrix[order].T  # transpose: rows=trials, columns=signals

    fig, ax = plt.subplots(figsize=(15, 5.5))
    im = ax.imshow(matrix_sorted, cmap="RdYlGn", vmin=-1, vmax=1, aspect="auto")
    labels = [f"{s} [angle]" if s in ANGLE_NAMES else s for s in signals_sorted]
    ax.set_xticks(range(len(signals_sorted)))
    ax.set_xticklabels(labels, rotation=90, ha="center", fontsize=8.5)
    ax.set_yticks(range(len(TRIALS)))
    ax.set_yticklabels(TRIALS, fontsize=8.5)
    for r in range(matrix_sorted.shape[0]):
        for c in range(matrix_sorted.shape[1]):
            v = matrix_sorted[r, c]
            if not np.isnan(v):
                ax.text(c, r, f"{v:.2f}", ha="center", va="center", fontsize=6.5,
                        color="black" if abs(v) < 0.6 else "white")
    ax.set_title("Correlation vs. Mocap, After Pelvis-Frame Fix\n(all 18 joint positions + 6 flexion angles, all 10 trials)")
    fig.colorbar(im, ax=ax, label="Pearson r", shrink=0.8)
    plt.tight_layout()
    out_path = out_dir / "07_correlation_matrix_after_fix.png"
    plt.savefig(out_path, dpi=130)
    plt.close(fig)

    print(f"{'signal':22s}" + "".join(f"{t[:10]:>12s}" for t in TRIALS) + f"{'mean':>10s}")
    for i, s in enumerate(signals_sorted):
        print(f"{s:22s}" + "".join(f"{v:12.2f}" for v in matrix_sorted[:, i]) + f"{col_mean[order[i]]:10.2f}")
    print(f"\nSaved -> {out_path}")


if __name__ == "__main__":
    main()

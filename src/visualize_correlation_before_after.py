"""Single before/after correlation chart, all 24 comparable signals (18 joint
positions + 6 flexion angles; pelvis excluded as the trivial reference
point), for one trial. "Before" = the original computation (joint positions
as-is; angles projected onto the fused skeleton's own, misoriented pelvis
frame). "After" = joint positions are unchanged (the frame fix doesn't touch
them at all -- shown so the fix's scope is visible, not just claimed); angles
are re-projected onto mocap's own pelvis frame (see
visualize_hip_diagnostics.py's frame-correction test).

All correlations here are real-time, per-frame Pearson r (matching
compare_metrics.py's methodology) -- not gait-cycle-normalized -- so this is
directly comparable to Phase 4/6's own numbers, not to the separate
gait-cycle classification in visualize_gait_cycles.py.

Usage: .venv\\Scripts\\python.exe src\\visualize_correlation_before_after.py <trial>
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

BEFORE_COLOR = "#c0392b"
AFTER_COLOR = "#1a7a3c"

ANGLE_NAMES = ["hip_flexion_left", "hip_flexion_right", "knee_flexion_left",
               "knee_flexion_right", "ankle_flexion_left", "ankle_flexion_right"]
POSITION_JOINTS = [j for j in CANONICAL_JOINTS if j != "pelvis"]


def _corr(a, b):
    valid = ~np.isnan(a) & ~np.isnan(b)
    if valid.sum() < 10 or np.std(a[valid]) == 0 or np.std(b[valid]) == 0:
        return float("nan")
    r, _ = stats.pearsonr(a[valid], b[valid])
    return float(r)


def main(trial: str):
    out_dir = OUT_DIR / trial
    out_dir.mkdir(parents=True, exist_ok=True)

    d = np.load(ALIGNED_FUSED_DIR / f"{trial}.npz")
    mocap_joints = {j: d[f"mocap__{j}"] for j in CANONICAL_JOINTS}
    fused_joints = {j: d[f"aligned__{j}"] for j in CANONICAL_JOINTS}

    results = {}

    # --- positions: unaffected by the frame fix (shown to prove scope) -----
    pelvis_y_mocap = mocap_joints["pelvis"][:, 1]
    pelvis_y_fused = fused_joints["pelvis"][:, 1]
    for j in POSITION_JOINTS:
        mocap_sig = mocap_joints[j][:, 1] - pelvis_y_mocap
        fused_sig = fused_joints[j][:, 1] - pelvis_y_fused
        r = _corr(mocap_sig, fused_sig)
        results[j] = {"before": r, "after": r}  # identical -- fix doesn't touch positions

    # --- angles: own (misoriented) frame vs. mocap's frame ------------------
    static_offsets = load_static_angle_offsets()
    mocap_angles_raw = compute_joint_angles_from_joints(mocap_joints, MOCAP_UP)
    fused_angles_raw = compute_joint_angles_from_joints(fused_joints, MOCAP_UP)
    x_hat_m, z_hat_m = joint_only_pelvis_frame(mocap_joints["hip_left"], mocap_joints["hip_right"], MOCAP_UP)

    for name in ANGLE_NAMES:
        side = name.split("_")[-1]
        offset = static_offsets[name]
        mocap_sig = mocap_angles_raw[name] - offset

        before_raw = fused_angles_raw[name]
        before_sig = mocap_sig + wrap_around_center(before_raw - offset - mocap_sig, 0.0)
        r_before = _corr(mocap_sig, before_sig)

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
        r_after = _corr(mocap_sig, after_sig)

        results[name] = {"before": r_before, "after": r_after}

    # --- plot ---------------------------------------------------------------
    order = sorted(results.keys(), key=lambda k: results[k]["after"])
    y = np.arange(len(order))
    before_vals = [results[k]["before"] for k in order]
    after_vals = [results[k]["after"] for k in order]
    is_angle = [k in ANGLE_NAMES for k in order]

    fig, ax = plt.subplots(figsize=(9, 10))
    height = 0.38
    ax.barh(y + height / 2, before_vals, height=height, color=BEFORE_COLOR, alpha=0.75,
            label="Before (own/fused pelvis frame)")
    ax.barh(y - height / 2, after_vals, height=height, color=AFTER_COLOR, alpha=0.85,
            label="After (mocap's pelvis frame) -- positions unchanged")
    ax.set_yticks(y)
    labels = [f"{k}  [angle]" if a else k for k, a in zip(order, is_angle)]
    ax.set_yticklabels(labels, fontsize=8.5)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.axvline(0.7, color="gray", linestyle=":", linewidth=0.8, label="r = 0.7 (\"accurately follows\")")
    ax.set_xlabel("Pearson r vs. mocap (real time, per-frame)")
    ax.set_title(f"{trial} -- Correlation Before vs. After Pelvis-Frame Fix\n(all 18 joint positions + 6 flexion angles)")
    ax.set_xlim(-1, 1)
    ax.legend(fontsize=8.5, loc="lower right")
    plt.tight_layout()
    out_path = out_dir / "05_correlation_before_after.png"
    plt.savefig(out_path, dpi=120)
    plt.close(fig)

    print(f"{'signal':22s} {'before':>8s} {'after':>8s}")
    for k in order:
        print(f"{k:22s} {results[k]['before']:8.2f} {results[k]['after']:8.2f}")
    print(f"\nSaved -> {out_path}")


if __name__ == "__main__":
    trial = sys.argv[1] if len(sys.argv) > 1 else "walking_2"
    main(trial)

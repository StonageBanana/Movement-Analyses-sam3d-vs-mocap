"""Gait-cycle-normalized knee flexion (mean +/- std vs. %% gait cycle, same
clinical style as visualize_gait_cycles.py) shown before vs. after the
pelvis-frame fix (see visualize_hip_diagnostics.py /
visualize_correlation_before_after.py). "Before" projects the fused
skeleton's shank/thigh vectors onto its own (misoriented) pelvis frame;
"after" projects the same vectors onto mocap's own frame -- isolating the
effect of that one fix on the classic gait-cycle plot the rest of this
project's angle comparisons are modeled on.

Usage: .venv\\Scripts\\python.exe src\\visualize_knee_gait_before_after.py <trial>
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
from gait_cycle import detect_cycle_boundaries, mean_std_curve

ANALYSIS_DIR = Path(__file__).resolve().parent.parent
ALIGNED_FUSED_DIR = ANALYSIS_DIR / "output" / "aligned_fused"
OUT_DIR = ANALYSIS_DIR / "output" / "diagnostics"

MOCAP_COLOR = "#1f4e8c"
BEFORE_COLOR = "#c0392b"
AFTER_COLOR = "#1a7a3c"


def main(trial: str):
    out_dir = OUT_DIR / trial
    out_dir.mkdir(parents=True, exist_ok=True)

    d = np.load(ALIGNED_FUSED_DIR / f"{trial}.npz")
    time = d["mocap_time_overlap"]
    mocap_joints = {j: d[f"mocap__{j}"] for j in CANONICAL_JOINTS}
    fused_joints = {j: d[f"aligned__{j}"] for j in CANONICAL_JOINTS}

    cycles = detect_cycle_boundaries(time, mocap_joints)
    print(f"{trial}: detected {len(cycles)} usable gait cycles "
          f"(median duration {np.median([e - s for s, e in cycles]):.2f}s)")

    static_offsets = load_static_angle_offsets()
    mocap_angles_raw = compute_joint_angles_from_joints(mocap_joints, MOCAP_UP)
    fused_angles_raw = compute_joint_angles_from_joints(fused_joints, MOCAP_UP)
    x_hat_m, z_hat_m = joint_only_pelvis_frame(mocap_joints["hip_left"], mocap_joints["hip_right"], MOCAP_UP)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    for ax, side in zip(axes, ["left", "right"]):
        name = f"knee_flexion_{side}"
        offset = static_offsets[name]
        mocap_sig = mocap_angles_raw[name] - offset

        before_sig = mocap_sig + wrap_around_center(fused_angles_raw[name] - offset - mocap_sig, 0.0)

        hip, knee, ankle = fused_joints[f"hip_{side}"], fused_joints[f"knee_{side}"], fused_joints[f"ankle_{side}"]
        thigh_ang_c = _wrap_around_own_circular_mean(_sagittal_angle_deg(knee - hip, x_hat_m, z_hat_m))
        shank_ang_c = _wrap_around_own_circular_mean(_sagittal_angle_deg(ankle - knee, x_hat_m, z_hat_m))
        after_raw = wrap_around_center(shank_ang_c - thigh_ang_c, 0.0)
        after_sig = mocap_sig + wrap_around_center(after_raw - offset - mocap_sig, 0.0)

        m_mean, m_std, n = mean_std_curve(time, mocap_sig, cycles)
        b_mean, b_std, _ = mean_std_curve(time, before_sig, cycles)
        a_mean, a_std, _ = mean_std_curve(time, after_sig, cycles)

        def r_of(mean_a, mean_b):
            valid = ~np.isnan(mean_a) & ~np.isnan(mean_b)
            if valid.sum() < 10:
                return float("nan")
            return stats.pearsonr(mean_a[valid], mean_b[valid])[0]

        r_before, r_after = r_of(m_mean, b_mean), r_of(m_mean, a_mean)

        x = np.linspace(0, 100, len(m_mean))
        ax.plot(x, m_mean, color=MOCAP_COLOR, linewidth=2, label="Mocap (ground truth)")
        ax.fill_between(x, m_mean - m_std, m_mean + m_std, color=MOCAP_COLOR, alpha=0.2)
        ax.plot(x, b_mean, color=BEFORE_COLOR, linewidth=2, label=f"Fused, before fix (r={r_before:+.2f})")
        ax.fill_between(x, b_mean - b_std, b_mean + b_std, color=BEFORE_COLOR, alpha=0.15)
        ax.plot(x, a_mean, color=AFTER_COLOR, linewidth=2, label=f"Fused, after fix (r={r_after:+.2f})")
        ax.fill_between(x, a_mean - a_std, a_mean + a_std, color=AFTER_COLOR, alpha=0.15)

        ax.set_title(f"Knee Flexion {side.title()}  (n={n} cycles)", fontsize=12)
        ax.set_xlabel("Gait Cycle (%)")
        ax.set_ylabel("Angle (deg)")
        ax.set_xlim(0, 100)
        ax.legend(fontsize=9)
        print(f"  {name}: r_before={r_before:+.2f}  r_after={r_after:+.2f}")

    fig.suptitle(f"{trial} -- Knee Flexion vs. %% Gait Cycle, Before vs. After Pelvis-Frame Fix", fontsize=14)
    plt.tight_layout()
    out_path = out_dir / "06_knee_flexion_gait_before_after.png"
    plt.savefig(out_path, dpi=120)
    plt.close(fig)
    print(f"\nSaved -> {out_path}")


if __name__ == "__main__":
    trial = sys.argv[1] if len(sys.argv) > 1 else "walking_2"
    main(trial)

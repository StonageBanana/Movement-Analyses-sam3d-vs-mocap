"""Gait-cycle-normalized flexion angles (%% gait cycle vs. angle, clinical
style) with all four sources overlaid: mocap (ground truth), view1, view2,
and fused -- all using the pelvis-frame fix, now hardcoded into
mocap.angles.compute_joint_angles_from_joints's `frame_joints` parameter
(see CLAUDE.md's "Five real bugs" #5 and compare_metrics.py). Without the
fix, view1/view2/fused's hip/knee flexion curves are phase-inverted against
mocap; this is only meaningful to compare all four sources together *after*
that correction.

Usage: .venv\\Scripts\\python.exe src\\visualize_all_sources_gait_cycle.py <trial>
"""

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from joint_mapping import CANONICAL_JOINTS
from mocap.angles import compute_joint_angles_from_joints, wrap_around_center
from compare_metrics import MOCAP_UP, load_static_angle_offsets
from gait_cycle import detect_cycle_boundaries, mean_std_curve

ANALYSIS_DIR = Path(__file__).resolve().parent.parent
ALIGNED_DIR = ANALYSIS_DIR / "output" / "aligned"
ALIGNED_FUSED_DIR = ANALYSIS_DIR / "output" / "aligned_fused"
OUT_DIR = ANALYSIS_DIR / "output" / "diagnostics"

ANGLE_ORDER = ["hip_flexion_left", "hip_flexion_right", "knee_flexion_left",
               "knee_flexion_right", "ankle_flexion_left", "ankle_flexion_right"]

COLORS = {"mocap": "#1f4e8c", "view1": "#e08214", "view2": "#8073ac", "fused": "#1a7a3c"}


# gait_cycle.detect_cycle_boundaries' default max_period=2.5s is tuned for
# walking/running strides; squats' rep period runs ~2.5-2.9s (confirmed via
# real-time inspection), which the default silently rejects (0 cycles found,
# not an error) -- widen it per-trial-category instead of raising the shared
# default for every trial.
MAX_PERIOD_OVERRIDE = {"squats_1": 6.0, "squats_2": 6.0}


def load_source(trial: str, condition: str):
    if condition == "fused":
        d = np.load(ALIGNED_FUSED_DIR / f"{trial}.npz")
    else:
        d = np.load(ALIGNED_DIR / f"{trial}__{condition}.npz")
    time = d["mocap_time_overlap"]
    mocap_joints = {j: d[f"mocap__{j}"] for j in CANONICAL_JOINTS}
    est_joints = {j: d[f"aligned__{j}"] for j in CANONICAL_JOINTS}
    return time, mocap_joints, est_joints


def main(trial: str):
    out_dir = OUT_DIR / trial
    out_dir.mkdir(parents=True, exist_ok=True)
    static_offsets = load_static_angle_offsets()

    sources = {}
    for cond in ["view1", "view2", "fused"]:
        time, mocap_joints, est_joints = load_source(trial, cond)
        mocap_angles_raw = compute_joint_angles_from_joints(mocap_joints, MOCAP_UP)
        est_angles_raw = compute_joint_angles_from_joints(est_joints, MOCAP_UP, frame_joints=mocap_joints)
        max_period = MAX_PERIOD_OVERRIDE.get(trial, 2.5)
        cycles = detect_cycle_boundaries(time, mocap_joints, max_period=max_period)
        sources[cond] = dict(time=time, mocap_joints=mocap_joints, est_joints=est_joints,
                              mocap_angles_raw=mocap_angles_raw, est_angles_raw=est_angles_raw,
                              cycles=cycles)
        print(f"{trial} [{cond}]: {len(cycles)} cycles, median {np.median([e - s for s, e in cycles]):.2f}s")

    fig, axes = plt.subplots(2, 3, figsize=(17, 9))
    for ax, name in zip(axes.flat, ANGLE_ORDER):
        offset = static_offsets[name]

        # ground truth curve: use fused's own mocap (representative; all
        # three sources' mocap subsets are the same trial, only the overlap
        # window's edges differ slightly)
        ref = sources["fused"]
        mocap_sig = ref["mocap_angles_raw"][name] - offset
        m_mean, m_std, n_ref = mean_std_curve(ref["time"], mocap_sig, ref["cycles"])
        x = np.linspace(0, 100, len(m_mean))
        ax.plot(x, m_mean, color=COLORS["mocap"], linewidth=2.5, label="Mocap (ground truth)", zorder=5)
        ax.fill_between(x, m_mean - m_std, m_mean + m_std, color=COLORS["mocap"], alpha=0.15, zorder=1)

        for cond in ["view1", "view2", "fused"]:
            s = sources[cond]
            mocap_sig_s = s["mocap_angles_raw"][name] - offset
            raw = s["est_angles_raw"][name]
            sig = mocap_sig_s + wrap_around_center(raw - offset - mocap_sig_s, 0.0)
            mean_c, _, n_c = mean_std_curve(s["time"], sig, s["cycles"])
            ax.plot(x, mean_c, color=COLORS[cond], linewidth=1.8, label=f"{cond} (n={n_c})", alpha=0.9)

        ax.set_title(name.replace("_", " ").title(), fontsize=12)
        ax.set_xlabel("Gait Cycle (%)")
        ax.set_ylabel("Angle (deg)")
        ax.set_xlim(0, 100)
        ax.legend(fontsize=8)

    fig.suptitle(f"{trial} -- View1 vs. View2 vs. Fused vs. Mocap, After Pelvis-Frame Fix\n"
                 f"(%% Gait Cycle vs. Angle, mean +/- std)", fontsize=14)
    plt.tight_layout()
    out_path = out_dir / "09_all_sources_gait_cycle.png"
    plt.savefig(out_path, dpi=120)
    plt.close(fig)
    print(f"\nSaved -> {out_path}")


if __name__ == "__main__":
    trial = sys.argv[1] if len(sys.argv) > 1 else "walking_2"
    main(trial)

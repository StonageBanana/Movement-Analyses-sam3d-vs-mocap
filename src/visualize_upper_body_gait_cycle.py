"""Gait-cycle-normalized upper-body joint positions (%% gait cycle vs.
vertical position relative to pelvis), all four sources overlaid: mocap
(ground truth), view1, view2, fused. Same style as
visualize_all_sources_gait_cycle.py's angle plots, but for the 8 upper-body
joints -- neck, head, both shoulders/elbows/wrists -- which have no
computed flexion angle in this pipeline (only hip/knee/ankle do), so
position (as in visualize_gait_cycles.py's joint_*.png) is the comparable
metric. Position isn't affected by the pelvis-frame angle fix, so no
frame_joints correction is needed here.

Usage: .venv\\Scripts\\python.exe src\\visualize_upper_body_gait_cycle.py <trial>
"""

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from joint_mapping import CANONICAL_JOINTS
from gait_cycle import detect_cycle_boundaries, mean_std_curve

ANALYSIS_DIR = Path(__file__).resolve().parent.parent
ALIGNED_DIR = ANALYSIS_DIR / "output" / "aligned"
ALIGNED_FUSED_DIR = ANALYSIS_DIR / "output" / "aligned_fused"
OUT_DIR = ANALYSIS_DIR / "output" / "diagnostics"

UPPER_BODY_JOINTS = ["neck", "head", "shoulder_left", "shoulder_right",
                     "elbow_left", "elbow_right", "wrist_left", "wrist_right"]

COLORS = {"mocap": "#1f4e8c", "view1": "#e08214", "view2": "#8073ac", "fused": "#1a7a3c"}

# see visualize_all_sources_gait_cycle.py -- squats' rep period (~2.5-2.9s)
# exceeds gait_cycle.py's walking/running-tuned default max_period=2.5s.
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

    sources = {}
    for cond in ["view1", "view2", "fused"]:
        time, mocap_joints, est_joints = load_source(trial, cond)
        max_period = MAX_PERIOD_OVERRIDE.get(trial, 2.5)
        cycles = detect_cycle_boundaries(time, mocap_joints, max_period=max_period)
        sources[cond] = dict(time=time, mocap_joints=mocap_joints, est_joints=est_joints, cycles=cycles)
        print(f"{trial} [{cond}]: {len(cycles)} cycles, median {np.median([e - s for s, e in cycles]):.2f}s")

    fig, axes = plt.subplots(2, 4, figsize=(21, 9))
    for ax, j in zip(axes.flat, UPPER_BODY_JOINTS):
        ref = sources["fused"]
        pelvis_y_mocap = ref["mocap_joints"]["pelvis"][:, 1]
        mocap_sig = ref["mocap_joints"][j][:, 1] - pelvis_y_mocap
        m_mean, m_std, n_ref = mean_std_curve(ref["time"], mocap_sig, ref["cycles"])
        x = np.linspace(0, 100, len(m_mean))
        ax.plot(x, m_mean, color=COLORS["mocap"], linewidth=2.5, label="Mocap (ground truth)", zorder=5)
        ax.fill_between(x, m_mean - m_std, m_mean + m_std, color=COLORS["mocap"], alpha=0.15, zorder=1)

        for cond in ["view1", "view2", "fused"]:
            s = sources[cond]
            pelvis_y = s["mocap_joints"]["pelvis"][:, 1]
            pelvis_y_est = s["est_joints"]["pelvis"][:, 1]
            sig = s["est_joints"][j][:, 1] - pelvis_y_est
            mean_c, _, n_c = mean_std_curve(s["time"], sig, s["cycles"])
            ax.plot(x, mean_c, color=COLORS[cond], linewidth=1.8, label=f"{cond} (n={n_c})", alpha=0.9)

        ax.set_title(j.replace("_", " ").title(), fontsize=12)
        ax.set_xlabel("Gait Cycle (%)")
        ax.set_ylabel("Position (mm, rel. to pelvis)")
        ax.set_xlim(0, 100)
        ax.legend(fontsize=7.5)

    fig.suptitle(f"{trial} -- Upper Body: View1 vs. View2 vs. Fused vs. Mocap\n"
                 f"(%% Gait Cycle vs. Vertical Position, mean +/- std)", fontsize=14)
    plt.tight_layout()
    out_path = out_dir / "10_upper_body_gait_cycle.png"
    plt.savefig(out_path, dpi=120)
    plt.close(fig)
    print(f"\nSaved -> {out_path}")


if __name__ == "__main__":
    trial = sys.argv[1] if len(sys.argv) > 1 else "walking_1"
    main(trial)

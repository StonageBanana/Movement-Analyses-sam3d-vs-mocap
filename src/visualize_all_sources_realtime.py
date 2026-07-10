"""Real-time (not gait-cycle-normalized) flexion angles across the *entire*
trial duration, all four sources overlaid: mocap (ground truth), view1,
view2, fused -- using the pelvis-frame fix hardcoded into
mocap.angles.compute_joint_angles_from_joints's `frame_joints` parameter.

Unlike visualize_all_sources_gait_cycle.py, this does not fold the signal
into %% gait cycle -- appropriate for trials with no real repeating cycle
(random movement, dance, feet movements), where gait-cycle normalization
produces a statistically meaningless average over 2-3 arbitrary "cycles".
Plots the whole video's timeline instead, so every action in the trial is
visible in its actual place in time.

Usage: .venv\\Scripts\\python.exe src\\visualize_all_sources_realtime.py <trial>
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

ANALYSIS_DIR = Path(__file__).resolve().parent.parent
ALIGNED_DIR = ANALYSIS_DIR / "output" / "aligned"
ALIGNED_FUSED_DIR = ANALYSIS_DIR / "output" / "aligned_fused"
OUT_DIR = ANALYSIS_DIR / "output" / "diagnostics"

ANGLE_ORDER = ["hip_flexion_left", "hip_flexion_right", "knee_flexion_left",
               "knee_flexion_right", "ankle_flexion_left", "ankle_flexion_right"]

COLORS = {"mocap": "#1f4e8c", "view1": "#e08214", "view2": "#8073ac", "fused": "#1a7a3c"}


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
        sources[cond] = dict(time=time, mocap_angles_raw=mocap_angles_raw, est_angles_raw=est_angles_raw)
        print(f"{trial} [{cond}]: {len(time):d} frames, {time[-1]-time[0]:.1f}s duration")

    fig, axes = plt.subplots(2, 3, figsize=(19, 9), sharex=False)
    for ax, name in zip(axes.flat, ANGLE_ORDER):
        offset = static_offsets[name]

        ref = sources["fused"]
        mocap_sig = ref["mocap_angles_raw"][name] - offset
        ax.plot(ref["time"], mocap_sig, color=COLORS["mocap"], linewidth=1.8,
                 label="Mocap (ground truth)", zorder=5)

        for cond in ["view1", "view2", "fused"]:
            s = sources[cond]
            mocap_sig_s = s["mocap_angles_raw"][name] - offset
            raw = s["est_angles_raw"][name]
            sig = mocap_sig_s + wrap_around_center(raw - offset - mocap_sig_s, 0.0)
            ax.plot(s["time"], sig, color=COLORS[cond], linewidth=1.1, label=cond, alpha=0.85)

        ax.set_title(name.replace("_", " ").title(), fontsize=12)
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Angle (deg)")
        ax.legend(fontsize=8)

    fig.suptitle(f"{trial} -- View1 vs. View2 vs. Fused vs. Mocap, Full Trial Timeline\n"
                 f"(real time, not gait-cycle-normalized -- after pelvis-frame fix)", fontsize=14)
    plt.tight_layout()
    out_path = out_dir / "11_all_sources_realtime_full_trial.png"
    plt.savefig(out_path, dpi=120)
    plt.close(fig)
    print(f"\nSaved -> {out_path}")


if __name__ == "__main__":
    trial = sys.argv[1] if len(sys.argv) > 1 else "random_1"
    main(trial)

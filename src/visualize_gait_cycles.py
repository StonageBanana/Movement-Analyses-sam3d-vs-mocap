"""Gait-cycle-normalized comparison plots (mocap vs. fused), in the classic
clinical style: mean +/- 1 std curve vs. % gait cycle. Uses output/aligned_fused
(already on mocap's timeline/frame, post Phase 6) so no new alignment happens
here -- purely a different way of visualizing already-computed, already-
validated data. Produces:

- 6 joint-angle plots (hip/knee/ankle flexion, left+right) -- same style as
  the reference clinical figure this was modeled on.
- 19 joint-position plots (each joint's vertical position relative to the
  pelvis, capturing the same "articulation over the gait cycle" idea for
  joints that don't have a defined flexion angle).
- A classification table (Pearson r between mocap's and fused's mean
  normalized curves) for all 25 signals, printed to console and saved as JSON.

Usage: .venv\\Scripts\\python.exe src\\visualize_gait_cycles.py <trial>
"""

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent))
from joint_mapping import CANONICAL_JOINTS
from mocap.angles import compute_joint_angles_from_joints, wrap_around_center
from compare_metrics import ANGLE_NAMES, MOCAP_UP, load_static_angle_offsets
from gait_cycle import detect_cycle_boundaries, mean_std_curve

ANALYSIS_DIR = Path(__file__).resolve().parent.parent
ALIGNED_FUSED_DIR = ANALYSIS_DIR / "output" / "aligned_fused"
OUT_DIR = ANALYSIS_DIR / "output" / "gait_cycle"

MOCAP_COLOR = "#1f4e8c"
EST_COLOR = "#c0392b"


def plot_comparison(ax, mocap_mean, mocap_std, est_mean, est_std, title, ylabel, n_cycles):
    x = np.linspace(0, 100, len(mocap_mean))
    ax.plot(x, mocap_mean, color=MOCAP_COLOR, linewidth=2, label="Mocap (ground truth)")
    ax.fill_between(x, mocap_mean - mocap_std, mocap_mean + mocap_std, color=MOCAP_COLOR, alpha=0.2)
    ax.plot(x, est_mean, color=EST_COLOR, linewidth=2, label="Fused (SAM 3D Body)")
    ax.fill_between(x, est_mean - est_std, est_mean + est_std, color=EST_COLOR, alpha=0.2)
    ax.set_title(f"{title}  (n={n_cycles} cycles)", fontsize=11)
    ax.set_xlabel("Gait Cycle (%)")
    ax.set_ylabel(ylabel)
    ax.set_xlim(0, 100)
    ax.legend(fontsize=8)


def curve_similarity(mocap_mean, est_mean):
    valid = ~np.isnan(mocap_mean) & ~np.isnan(est_mean)
    if valid.sum() < 10 or np.std(mocap_mean[valid]) == 0 or np.std(est_mean[valid]) == 0:
        return float("nan")
    r, _ = stats.pearsonr(mocap_mean[valid], est_mean[valid])
    return float(r)


def classify(r: float) -> str:
    if np.isnan(r):
        return "insufficient data"
    if r >= 0.7:
        return "accurately follows the curve"
    if r >= 0.3:
        return "partially follows the curve"
    return "does not follow the curve"


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

    results = {}

    # --- 6 joint angles, matching the reference figure's style -----------
    static_offsets = load_static_angle_offsets()
    mocap_angles_raw = compute_joint_angles_from_joints(mocap_joints, MOCAP_UP)
    fused_angles_raw = compute_joint_angles_from_joints(fused_joints, MOCAP_UP)

    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    angle_order = ["hip_flexion_left", "hip_flexion_right", "knee_flexion_left",
                   "knee_flexion_right", "ankle_flexion_left", "ankle_flexion_right"]
    for ax, name in zip(axes.flat, angle_order):
        offset = static_offsets[name]
        mocap_sig = mocap_angles_raw[name] - offset
        fused_sig = mocap_sig + wrap_around_center(fused_angles_raw[name] - mocap_angles_raw[name], 0.0)
        m_mean, m_std, n = mean_std_curve(time, mocap_sig, cycles)
        f_mean, f_std, _ = mean_std_curve(time, fused_sig, cycles)
        r = curve_similarity(m_mean, f_mean)
        results[name] = {"pearson_r": r, "classification": classify(r), "n_cycles": n}
        plot_comparison(ax, m_mean, m_std, f_mean, f_std, name.replace("_", " ").title(), "Angle (deg)", n)
    fig.suptitle(f"Gait Cycle Comparison -- {trial} -- Joint Angles", fontsize=14)
    plt.tight_layout()
    out_path = out_dir / "00_all_angles.png"
    plt.savefig(out_path, dpi=120)
    plt.close(fig)
    print(f"  Saved {out_path}")

    # --- individual angle figures (one per angle, for easy deletion) -----
    for name in angle_order:
        offset = static_offsets[name]
        mocap_sig = mocap_angles_raw[name] - offset
        fused_sig = mocap_sig + wrap_around_center(fused_angles_raw[name] - mocap_angles_raw[name], 0.0)
        m_mean, m_std, n = mean_std_curve(time, mocap_sig, cycles)
        f_mean, f_std, _ = mean_std_curve(time, fused_sig, cycles)
        fig, ax = plt.subplots(figsize=(6, 5))
        plot_comparison(ax, m_mean, m_std, f_mean, f_std, name.replace("_", " ").title(), "Angle (deg)", n)
        plt.tight_layout()
        out_path = out_dir / f"angle_{name}.png"
        plt.savefig(out_path, dpi=120)
        plt.close(fig)

    # --- all 19 joints, vertical position relative to pelvis -------------
    pelvis_y_mocap = mocap_joints["pelvis"][:, 1]
    pelvis_y_fused = fused_joints["pelvis"][:, 1]
    for j in CANONICAL_JOINTS:
        mocap_sig = mocap_joints[j][:, 1] - pelvis_y_mocap
        fused_sig = fused_joints[j][:, 1] - pelvis_y_fused
        m_mean, m_std, n = mean_std_curve(time, mocap_sig, cycles)
        f_mean, f_std, _ = mean_std_curve(time, fused_sig, cycles)
        r = curve_similarity(m_mean, f_mean)
        results[j] = {"pearson_r": r, "classification": classify(r), "n_cycles": n}

        fig, ax = plt.subplots(figsize=(6, 5))
        plot_comparison(ax, m_mean, m_std, f_mean, f_std, f"{j} (vertical, rel. to pelvis)", "Position (mm)", n)
        plt.tight_layout()
        out_path = out_dir / f"joint_{j}.png"
        plt.savefig(out_path, dpi=120)
        plt.close(fig)

    (out_dir / "classification.json").write_text(json.dumps(results, indent=2))

    print(f"\n{'signal':25s} {'pearson_r':>10s}   classification")
    for name, r in sorted(results.items(), key=lambda kv: (kv[1]["pearson_r"] if not np.isnan(kv[1]["pearson_r"]) else -99), reverse=True):
        print(f"{name:25s} {r['pearson_r']:>10.2f}   {r['classification']}")

    print(f"\nAll figures saved -> {out_dir}")


if __name__ == "__main__":
    trial = sys.argv[1] if len(sys.argv) > 1 else "walking_2"
    main(trial)

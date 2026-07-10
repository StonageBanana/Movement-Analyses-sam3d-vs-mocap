"""Does the subject's locomotion speed affect SAM 3D Body's overall tracking
accuracy? Uses the 5 locomotion trials (walking_1/2/3, running_1/2) --
squats/dance/etc. have no well-defined travel speed. Speed is measured from
mocap's own data (ground truth), not from the estimate, so it's an
independent variable.

Speed metric: NOT literal translation speed -- the subject's pelvis moves
only ~70-90mm horizontally over the entire 60+ second trial in every one of
these 5 trials (confirmed by direct inspection), meaning the "walking" and
"running" here are performed in a confined capture volume without covering
real distance, so m/s-style path-length speed is meaningless (it came out
~0.16-0.17 m/s and indistinguishable across all 5 trials on first attempt).
Instead, speed is measured as cadence: the dominant frequency (Welch PSD,
0.3-4 Hz band) of the broadband vertical signal (mean Y-position across all
19 joints -- same signal gait_cycle.py uses for cycle detection), in
cycles/min. This cleanly separates the 5 trials (walking_1/2 ~97, walking_3
~111, running_1/2 ~150 cycles/min) where the path-length approach could not.

Accuracy metrics (whole-subject, not per-joint), all computed on the
already-validated Phase 6 (fused) data, angles using the pelvis-frame fix
from visualize_hip_diagnostics.py:
- overall_mpjpe_mm: Phase 6's own mpjpe_overall_mm (position accuracy;
  unaffected by the angle-frame fix, included as the primary whole-body
  accuracy figure).
- angle_rmse_deg: mean RMSE across all 6 flexion angles, post-fix.
- angle_r: mean Pearson r across all 6 flexion angles, post-fix.

Usage: .venv\\Scripts\\python.exe src\\analyze_speed_vs_accuracy.py
"""

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
from scipy.signal import welch

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

TRIALS = ["walking_1", "walking_2", "walking_3", "running_1", "running_2"]
LABELS = {"walking_1": "walking_1", "walking_2": "walking_2", "walking_3": "walking_3",
          "running_1": "running_1", "running_2": "running_2"}
IS_RUNNING = {"walking_1": False, "walking_2": False, "walking_3": False,
              "running_1": True, "running_2": True}


def cadence_per_min(time: np.ndarray, mocap_joints: dict) -> float:
    stacked = np.stack([mocap_joints[j] for j in CANONICAL_JOINTS], axis=0)
    broadband = np.nanmean(stacked, axis=0)[:, 1]
    valid = ~np.isnan(broadband)
    filled = np.interp(time, time[valid], broadband[valid])
    dt = np.median(np.diff(time))
    freqs, psd = welch(filled - filled.mean(), fs=1.0 / dt, nperseg=2048)
    band = (freqs > 0.3) & (freqs < 4.0)
    peak_freq = freqs[band][np.argmax(psd[band])]
    return float(peak_freq * 60.0)


def angle_metrics_after_fix(mocap_joints, fused_joints):
    static_offsets = load_static_angle_offsets()
    mocap_angles_raw = compute_joint_angles_from_joints(mocap_joints, MOCAP_UP)
    x_hat_m, z_hat_m = joint_only_pelvis_frame(mocap_joints["hip_left"], mocap_joints["hip_right"], MOCAP_UP)

    rmses, rs = [], []
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

        d = wrap_around_center(after_raw - offset - mocap_sig, 0.0)
        valid = ~np.isnan(d)
        rmses.append(float(np.sqrt(np.mean(d[valid] ** 2))))

        after_sig = mocap_sig + d
        v2 = valid & ~np.isnan(mocap_sig)
        if v2.sum() > 10 and np.std(mocap_sig[v2]) > 0 and np.std(after_sig[v2]) > 0:
            rs.append(float(stats.pearsonr(mocap_sig[v2], after_sig[v2])[0]))
    return float(np.mean(rmses)), float(np.mean(rs))


def main():
    phase6 = json.load(open(ANALYSIS_DIR / "output" / "metrics" / "phase6_metrics.json"))
    pt = {x["trial"]: x for x in phase6["per_trial"]}

    rows = []
    for trial in TRIALS:
        d = np.load(ALIGNED_FUSED_DIR / f"{trial}.npz")
        time = d["mocap_time_overlap"]
        mocap_joints = {j: d[f"mocap__{j}"] for j in CANONICAL_JOINTS}
        fused_joints = {j: d[f"aligned__{j}"] for j in CANONICAL_JOINTS}

        speed = cadence_per_min(time, mocap_joints)
        mpjpe = pt[trial]["mpjpe_overall_mm"]
        angle_rmse, angle_r = angle_metrics_after_fix(mocap_joints, fused_joints)

        rows.append({"trial": trial, "cadence": speed, "mpjpe_mm": mpjpe,
                     "angle_rmse_deg": angle_rmse, "angle_r": angle_r, "running": IS_RUNNING[trial]})
        print(f"{trial:12s} cadence={speed:.1f} cycles/min   MPJPE={mpjpe:.1f}mm   "
              f"angle_RMSE={angle_rmse:.1f} deg   angle_r={angle_r:+.2f}")

    speeds = np.array([r["cadence"] for r in rows])
    mpjpes = np.array([r["mpjpe_mm"] for r in rows])
    rmses = np.array([r["angle_rmse_deg"] for r in rows])
    rs = np.array([r["angle_r"] for r in rows])
    colors = ["#1f4e8c" if not r["running"] else "#c0392b" for r in rows]

    r_mpjpe, _ = stats.pearsonr(speeds, mpjpes)
    r_rmse, _ = stats.pearsonr(speeds, rmses)
    r_r, _ = stats.pearsonr(speeds, rs)

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    for ax, y, title, ylabel, r_val in [
        (axes[0], mpjpes, "Position Error vs. Speed", "MPJPE (mm, lower = better)", r_mpjpe),
        (axes[1], rmses, "Angle Error vs. Speed", "Mean flexion-angle RMSE (deg, lower = better)", r_rmse),
        (axes[2], rs, "Angle Shape-Match vs. Speed", "Mean flexion-angle Pearson r (higher = better)", r_r),
    ]:
        ax.scatter(speeds, y, c=colors, s=90, zorder=3)
        for row, xv, yv in zip(rows, speeds, y):
            ax.annotate(row["trial"], (xv, yv), textcoords="offset points", xytext=(6, 6), fontsize=8)
        if len(speeds) > 1:
            fit = np.polyfit(speeds, y, 1)
            xs = np.linspace(speeds.min(), speeds.max(), 50)
            ax.plot(xs, np.polyval(fit, xs), color="gray", linestyle="--", alpha=0.6)
        ax.set_title(f"{title}\n(Pearson r = {r_val:+.2f}, n=5)", fontsize=10.5)
        ax.set_xlabel("Cadence (cycles/min, from mocap vertical-bounce frequency)")
        ax.set_ylabel(ylabel)

    handles = [plt.Line2D([0], [0], marker="o", color="w", markerfacecolor="#1f4e8c", markersize=9, label="walking"),
               plt.Line2D([0], [0], marker="o", color="w", markerfacecolor="#c0392b", markersize=9, label="running")]
    fig.legend(handles=handles, loc="upper center", ncol=2, fontsize=9, bbox_to_anchor=(0.5, 1.04))
    fig.suptitle("Does Locomotion Speed Affect Tracking Accuracy? (post pelvis-frame fix)", fontsize=13, y=1.1)
    plt.tight_layout()
    out_path = OUT_DIR / "08_speed_vs_accuracy.png"
    plt.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"\nOverall Pearson r (speed vs metric): MPJPE={r_mpjpe:+.2f}  angle_RMSE={r_rmse:+.2f}  angle_r={r_r:+.2f}")
    print(f"Saved -> {out_path}")


if __name__ == "__main__":
    main()

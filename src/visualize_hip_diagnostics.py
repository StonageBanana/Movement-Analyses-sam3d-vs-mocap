"""Real-time (non-gait-cycle-normalized) diagnostics isolating *why* the hip
and knee flexion angles show a systematic, cross-trial-consistent inversion
against mocap (see investigation in the per_joint_comparison work). Produces,
per trial:

- 01_hip_separation.png: hip_left-to-hip_right distance over time, mocap vs.
  fused -- directly visualizes the ~38% compression of the estimated hip
  width relative to true anthropometry.
- 02_pelvis_axis_error.png: angle (deg) between mocap's and fused's
  hip-left-minus-hip-right direction vector, over time -- a persistent
  non-zero value here means the sagittal-plane reference frame that every
  hip/knee/ankle flexion angle is projected into is itself misoriented.
- 03_hip_knee_flexion_realtime.png: hip_flexion_left/right and
  knee_flexion_left/right, mocap vs. fused, plotted against real time
  (seconds) rather than %% gait cycle -- shows the phase relationship
  directly, ruling out any cycle-normalization artifact.
- 04_frame_corrected_hip_knee_flexion.png: same four angles, but the
  fused skeleton's thigh/shank vectors are projected onto MOCAP's own
  (correctly-oriented) pelvis frame instead of the fused skeleton's own
  -- isolating whether the angle error comes from the misoriented pelvis
  axis (02_pelvis_axis_error.png) specifically, rather than from the
  underlying hip/knee/ankle joint positions themselves being wrong.

This uses output/aligned_fused (already-validated, already-aligned data) --
no new alignment or fusion happens here.

Usage: .venv\\Scripts\\python.exe src\\visualize_hip_diagnostics.py <trial>
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

MOCAP_COLOR = "#1f4e8c"
EST_COLOR = "#c0392b"


def _unit(v):
    return v / np.linalg.norm(v, axis=-1, keepdims=True)


def main(trial: str):
    out_dir = OUT_DIR / trial
    out_dir.mkdir(parents=True, exist_ok=True)

    d = np.load(ALIGNED_FUSED_DIR / f"{trial}.npz")
    time = d["mocap_time_overlap"]
    mocap_joints = {j: d[f"mocap__{j}"] for j in CANONICAL_JOINTS}
    fused_joints = {j: d[f"aligned__{j}"] for j in CANONICAL_JOINTS}

    # --- 1. hip separation over time -------------------------------------
    hw_m = np.linalg.norm(mocap_joints["hip_left"] - mocap_joints["hip_right"], axis=-1)
    hw_f = np.linalg.norm(fused_joints["hip_left"] - fused_joints["hip_right"], axis=-1)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(time, hw_m, color=MOCAP_COLOR, linewidth=1.5, label="Mocap (ground truth)")
    ax.plot(time, hw_f, color=EST_COLOR, linewidth=1.5, label="Fused (SAM 3D Body)")
    ax.axhline(np.nanmean(hw_m), color=MOCAP_COLOR, linestyle="--", alpha=0.5,
               label=f"Mocap mean = {np.nanmean(hw_m):.0f}mm")
    ax.axhline(np.nanmean(hw_f), color=EST_COLOR, linestyle="--", alpha=0.5,
               label=f"Fused mean = {np.nanmean(hw_f):.0f}mm ({100*np.nanmean(hw_f)/np.nanmean(hw_m):.0f}% of true)")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("hip_left <-> hip_right distance (mm)")
    ax.set_title(f"{trial} -- Hip Separation Over Time")
    ax.legend(fontsize=9)
    plt.tight_layout()
    plt.savefig(out_dir / "01_hip_separation.png", dpi=120)
    plt.close(fig)

    # --- 2. pelvis mediolateral axis orientation error --------------------
    y_m = _unit(mocap_joints["hip_left"] - mocap_joints["hip_right"])
    y_f = _unit(fused_joints["hip_left"] - fused_joints["hip_right"])
    cosang = np.sum(y_m * y_f, axis=-1)
    axis_err = np.rad2deg(np.arccos(np.clip(cosang, -1, 1)))

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(time, axis_err, color="#6a3d9a", linewidth=1.5)
    ax.axhline(np.nanmean(axis_err), color="black", linestyle="--", alpha=0.6,
               label=f"mean = {np.nanmean(axis_err):.1f} deg (std = {np.nanstd(axis_err):.1f} deg)")
    ax.axhline(0, color="gray", linestyle=":", alpha=0.5, label="0 deg = perfectly aligned")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Angle between mocap and fused pelvis axes (deg)")
    ax.set_title(f"{trial} -- Pelvis Left-Right Axis Orientation Error")
    ax.set_ylim(0, 180)
    ax.legend(fontsize=9)
    plt.tight_layout()
    plt.savefig(out_dir / "02_pelvis_axis_error.png", dpi=120)
    plt.close(fig)

    # --- 3. hip/knee flexion, real time, mocap vs fused --------------------
    static_offsets = load_static_angle_offsets()
    mocap_angles_raw = compute_joint_angles_from_joints(mocap_joints, MOCAP_UP)
    fused_angles_raw = compute_joint_angles_from_joints(fused_joints, MOCAP_UP)

    fig, axes = plt.subplots(2, 2, figsize=(13, 8), sharex=True)
    names = ["hip_flexion_left", "hip_flexion_right", "knee_flexion_left", "knee_flexion_right"]
    for ax, name in zip(axes.flat, names):
        offset = static_offsets[name]
        mocap_sig = mocap_angles_raw[name] - offset
        fused_sig = mocap_sig + wrap_around_center(fused_angles_raw[name] - mocap_angles_raw[name], 0.0)
        ax.plot(time, mocap_sig, color=MOCAP_COLOR, linewidth=1.2, label="Mocap (ground truth)")
        ax.plot(time, fused_sig, color=EST_COLOR, linewidth=1.2, label="Fused (SAM 3D Body)")
        ax.set_title(name.replace("_", " ").title(), fontsize=11)
        ax.set_ylabel("Angle (deg)")
        ax.legend(fontsize=8)
    for ax in axes[-1]:
        ax.set_xlabel("Time (s)")
    fig.suptitle(f"{trial} -- Hip/Knee Flexion, Real Time (not gait-cycle-normalized)", fontsize=14)
    plt.tight_layout()
    plt.savefig(out_dir / "03_hip_knee_flexion_realtime.png", dpi=120)
    plt.close(fig)

    # --- 4. frame-corrected hip/knee flexion -------------------------------
    # Recompute the fused skeleton's angles using MOCAP's own pelvis frame
    # (x_hat, z_hat) instead of the fused skeleton's own -- if the fused
    # skeleton's knee/ankle/hip *positions* are fundamentally sound and only
    # the shared pelvis reference axis was wrong, projecting the same thigh/
    # shank vectors onto the correct frame should recover a positive
    # correlation with mocap.
    x_hat_m, z_hat_m = joint_only_pelvis_frame(mocap_joints["hip_left"], mocap_joints["hip_right"], MOCAP_UP)

    fig, axes = plt.subplots(2, 2, figsize=(13, 8), sharex=True)
    for ax, name in zip(axes.flat, names):
        side = name.split("_")[-1]
        hip, knee, ankle = fused_joints[f"hip_{side}"], fused_joints[f"knee_{side}"], fused_joints[f"ankle_{side}"]
        thigh_ang_c = _wrap_around_own_circular_mean(_sagittal_angle_deg(knee - hip, x_hat_m, z_hat_m))
        shank_ang_c = _wrap_around_own_circular_mean(_sagittal_angle_deg(ankle - knee, x_hat_m, z_hat_m))
        corrected_raw = thigh_ang_c if name.startswith("hip") else wrap_around_center(shank_ang_c - thigh_ang_c, 0.0)

        offset = static_offsets[name]
        mocap_sig = mocap_angles_raw[name] - offset
        original_sig = mocap_sig + wrap_around_center(fused_angles_raw[name] - offset - mocap_sig, 0.0)
        corrected_sig = mocap_sig + wrap_around_center(corrected_raw - offset - mocap_sig, 0.0)

        valid = ~np.isnan(mocap_sig) & ~np.isnan(corrected_sig)
        r_corrected, _ = stats.pearsonr(mocap_sig[valid], corrected_sig[valid])
        bias = np.nanmean(corrected_sig - mocap_sig)
        corrected_debiased = corrected_sig - bias  # for visual overlay only -- r is bias-invariant

        ax.plot(time, mocap_sig, color=MOCAP_COLOR, linewidth=1.3, label="Mocap (ground truth)")
        ax.plot(time, original_sig, color=EST_COLOR, linewidth=1.0, alpha=0.4, label="Fused, own frame (original)")
        ax.plot(time, corrected_debiased, color="#1a7a3c", linewidth=1.3,
                label=f"Fused, mocap's frame, bias removed (r={r_corrected:+.2f})")
        ax.set_title(name.replace("_", " ").title(), fontsize=11)
        ax.set_ylabel("Angle (deg)")
        ax.legend(fontsize=7.5)
    for ax in axes[-1]:
        ax.set_xlabel("Time (s)")
    fig.suptitle(f"{trial} -- Hip/Knee Flexion, Pelvis-Frame-Corrected vs. Original", fontsize=14)
    plt.tight_layout()
    plt.savefig(out_dir / "04_frame_corrected_hip_knee_flexion.png", dpi=120)
    plt.close(fig)

    print(f"Saved 4 diagnostic figures -> {out_dir}")


if __name__ == "__main__":
    trial = sys.argv[1] if len(sys.argv) > 1 else "walking_2"
    main(trial)

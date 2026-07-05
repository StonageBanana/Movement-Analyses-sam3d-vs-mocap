"""Phase 3 orchestrator: for every trial/view whose SAM 3D Body output
already exists, align it to the mocap ground truth (temporal sync via
cross-correlation, spatial fit via Umeyama similarity transform) and save
the result. Safe to re-run as more Phase 2 outputs land in the background --
already-aligned trial/views are skipped.
"""

import json
import sys
from pathlib import Path

import numpy as np
from scipy.signal import savgol_filter

sys.path.insert(0, str(Path(__file__).resolve().parent))
from align import umeyama_alignment, apply_similarity, cross_correlate_lag, resample_joints_to_times
from joint_mapping import CANONICAL_JOINTS, sam3d_canonical_joints, detect_vertical_axis_generic

SAM3D_REPO = Path(__file__).resolve().parent.parent / "third_party" / "sam-3d-body"
sys.path.insert(0, str(SAM3D_REPO))
from sam_3d_body.metadata.mhr70 import mhr_names  # noqa: E402

ANALYSIS_DIR = Path(__file__).resolve().parent.parent
MANIFEST_PATH = ANALYSIS_DIR / "manifest.json"
MOCAP_DIR = ANALYSIS_DIR / "output" / "mocap"
SAM3D_DIR = ANALYSIS_DIR / "output" / "sam3d"
OUTPUT_DIR = ANALYSIS_DIR / "output" / "aligned"


def load_mocap_joints(trial: str):
    d = np.load(MOCAP_DIR / f"{trial}.npz")
    joints = {name: d[f"joint__{name}"] for name in CANONICAL_JOINTS}
    time = d["time"]
    return joints, time


def _smooth(arr: np.ndarray, window: int = 9, polyorder: int = 2) -> np.ndarray:
    n = arr.shape[0]
    w = min(window, n if n % 2 == 1 else n - 1)
    if w <= polyorder or w < 3:
        return arr.copy()
    return savgol_filter(arr, window_length=w, polyorder=polyorder, axis=0)


def load_sam3d_joints(trial: str, view: str, video_fps: float):
    d = np.load(SAM3D_DIR / f"{trial}__{view}.npz")
    # pred_cam_t wasn't smoothed when Phase 2 saved it (only the joint
    # arrays were) -- smooth it the same way here for consistency, since it
    # now feeds directly into every joint's position.
    cam_t_smoothed = _smooth(d["pred_cam_t"])
    joints = sam3d_canonical_joints(d["pred_keypoints_3d_smoothed"], mhr_names, cam_t_smoothed)
    time = d["frame_idx"] / video_fps
    return joints, time


def vertical_component(joints: dict, up_axis: np.ndarray) -> np.ndarray:
    # A single joint's vertical trajectory is strongly periodic for
    # repetitive actions (squats, walking, running), which makes
    # cross-correlation prone to cycle-slip (locking onto a secondary peak
    # one period away -- confirmed on squats_2, where a neck-only signal
    # and a knee-only signal disagreed on the lag by ~2.4s). Averaging
    # across many joints gives a broadband signal -- still periodic, but a
    # much less clean/aliasing-prone waveform -- and nanmean skips
    # occluded markers per frame instead of failing on any single joint's
    # gaps (e.g. pelvis is NaN in ~25% of squats_2 frames).
    stacked = np.stack([joints[j] for j in CANONICAL_JOINTS], axis=0)  # (J, F, 3)
    mean_pos = np.nanmean(stacked, axis=0)  # (F, 3)
    return mean_pos @ up_axis


def align_trial_view(trial: str, view: str, video_fps: float) -> dict:
    mocap_joints, mocap_time = load_mocap_joints(trial)
    sam3d_joints, sam3d_time = load_sam3d_joints(trial, view, video_fps)

    mocap_up = np.array([0.0, 1.0, 0.0])  # mocap .trc is Y-up (confirmed in Phase 1)
    sam3d_up = detect_vertical_axis_generic(sam3d_joints["head"], sam3d_joints["toe_left"])

    mocap_vert = vertical_component(mocap_joints, mocap_up)
    sam3d_vert = vertical_component(sam3d_joints, sam3d_up)

    lag = cross_correlate_lag(mocap_time, mocap_vert, sam3d_time, sam3d_vert)
    sam3d_time_shifted = sam3d_time + lag

    sam3d_resampled, valid_mask = resample_joints_to_times(sam3d_time_shifted, sam3d_joints, mocap_time)
    mocap_overlap = {name: arr[valid_mask] for name, arr in mocap_joints.items()}

    # Occluded mocap markers (common during e.g. deep squats) leave NaN gaps;
    # exclude those rows from the fit, or a single NaN corrupts the whole
    # covariance matrix and SVD fails to converge.
    src_pts = np.concatenate([sam3d_resampled[j] for j in CANONICAL_JOINTS], axis=0)
    dst_pts = np.concatenate([mocap_overlap[j] for j in CANONICAL_JOINTS], axis=0)
    fit_mask = ~np.isnan(src_pts).any(axis=-1) & ~np.isnan(dst_pts).any(axis=-1)
    R, scale, t = umeyama_alignment(src_pts[fit_mask], dst_pts[fit_mask])

    aligned = {name: apply_similarity(arr, R, scale, t) for name, arr in sam3d_resampled.items()}

    residuals = np.concatenate(
        [np.linalg.norm(aligned[j] - mocap_overlap[j], axis=-1) for j in CANONICAL_JOINTS]
    )

    return {
        "aligned_joints": aligned,
        "mocap_overlap_joints": mocap_overlap,
        "mocap_time_overlap": mocap_time[valid_mask],
        "lag_seconds": lag,
        "scale": scale,
        "R": R,
        "t": t,
        "mean_residual_mm": float(np.nanmean(residuals)),
        "median_residual_mm": float(np.nanmedian(residuals)),
        "n_overlap_frames": int(valid_mask.sum()),
    }


def main():
    manifest = json.loads(MANIFEST_PATH.read_text())
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for trial, entry in manifest["trials"].items():
        if not entry["paired"]:
            continue
        for view in ("view1", "view2"):
            sam3d_path = SAM3D_DIR / f"{trial}__{view}.npz"
            out_path = OUTPUT_DIR / f"{trial}__{view}.npz"
            if not sam3d_path.exists() or out_path.exists():
                continue

            video_fps = entry["views"][view]["fps"]
            print(f"Aligning {trial} / {view} (fps={video_fps}) ...")
            try:
                result = align_trial_view(trial, view, video_fps)
            except Exception as e:
                print(f"  FAILED: {e}")
                continue

            print(f"  lag={result['lag_seconds']:.3f}s, scale={result['scale']:.4f}, "
                  f"overlap={result['n_overlap_frames']} frames, "
                  f"mean_residual={result['mean_residual_mm']:.1f}mm, "
                  f"median_residual={result['median_residual_mm']:.1f}mm")

            np.savez_compressed(
                out_path,
                mocap_time_overlap=result["mocap_time_overlap"],
                lag_seconds=result["lag_seconds"],
                scale=result["scale"],
                R=result["R"],
                t=result["t"],
                mean_residual_mm=result["mean_residual_mm"],
                median_residual_mm=result["median_residual_mm"],
                **{f"aligned__{k}": v for k, v in result["aligned_joints"].items()},
                **{f"mocap__{k}": v for k, v in result["mocap_overlap_joints"].items()},
            )
            print(f"  Saved -> {out_path}")


if __name__ == "__main__":
    main()

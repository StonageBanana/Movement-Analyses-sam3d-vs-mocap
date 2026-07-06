"""Phase 6: align the fused (Phase 5) trajectory to mocap ground truth --
temporal sync (cross-correlation) + spatial fit (Umeyama similarity
transform), exactly the same method Phase 3 used per view, just applied
once more to the fused output instead of each raw SAM3D view. This is the
*only* place mocap touches the fused dataset -- Phase 5's fusion itself
used zero mocap/calibration (view1 and view2 were aligned to each other
only), so this step is what actually scores it.

Safe to re-run as more Phase 5 outputs land -- already-aligned trials are
skipped.
"""

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from align import umeyama_alignment, apply_similarity, cross_correlate_lag_candidates, resample_joints_to_times
from joint_mapping import CANONICAL_JOINTS, detect_vertical_axis_generic

ANALYSIS_DIR = Path(__file__).resolve().parent.parent
MANIFEST_PATH = ANALYSIS_DIR / "manifest.json"
MOCAP_DIR = ANALYSIS_DIR / "output" / "mocap"
FUSED_DIR = ANALYSIS_DIR / "output" / "fused"
OUTPUT_DIR = ANALYSIS_DIR / "output" / "aligned_fused"


def load_mocap_joints(trial: str):
    d = np.load(MOCAP_DIR / f"{trial}.npz")
    joints = {name: d[f"joint__{name}"] for name in CANONICAL_JOINTS}
    time = d["time"]
    return joints, time


def load_fused_joints(trial: str):
    d = np.load(FUSED_DIR / f"{trial}.npz")
    joints = {name: d[f"fused__{name}"] for name in CANONICAL_JOINTS}
    time = d["time"]
    return joints, time


def vertical_component(joints: dict, up_axis: np.ndarray) -> np.ndarray:
    # Same broadband-signal approach as Phase 3 (mean across all canonical
    # joints, not one periodic joint) to avoid cycle-slip in the
    # cross-correlation for repetitive actions -- see align.cross_correlate_lag.
    stacked = np.stack([joints[j] for j in CANONICAL_JOINTS], axis=0)  # (J, F, 3)
    mean_pos = np.nanmean(stacked, axis=0)  # (F, 3)
    return mean_pos @ up_axis


def _fit_at_lag(lag: float, mocap_joints: dict, mocap_time, fused_joints: dict, fused_time) -> dict:
    fused_time_shifted = fused_time + lag
    fused_resampled, valid_mask = resample_joints_to_times(fused_time_shifted, fused_joints, mocap_time)
    mocap_overlap = {name: arr[valid_mask] for name, arr in mocap_joints.items()}

    # Occluded mocap markers leave NaN gaps (same as Phase 3); exclude those
    # rows from the fit, or a single NaN corrupts the whole covariance
    # matrix and SVD fails to converge.
    src_pts = np.concatenate([fused_resampled[j] for j in CANONICAL_JOINTS], axis=0)
    dst_pts = np.concatenate([mocap_overlap[j] for j in CANONICAL_JOINTS], axis=0)
    fit_mask = ~np.isnan(src_pts).any(axis=-1) & ~np.isnan(dst_pts).any(axis=-1)
    R, scale, t = umeyama_alignment(src_pts[fit_mask], dst_pts[fit_mask])

    aligned = {name: apply_similarity(arr, R, scale, t) for name, arr in fused_resampled.items()}

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


def align_trial(trial: str) -> dict:
    mocap_joints, mocap_time = load_mocap_joints(trial)
    fused_joints, fused_time = load_fused_joints(trial)

    mocap_up = np.array([0.0, 1.0, 0.0])  # mocap .trc is Y-up (confirmed in Phase 1)
    fused_up = detect_vertical_axis_generic(fused_joints["head"], fused_joints["toe_left"])

    mocap_vert = vertical_component(mocap_joints, mocap_up)
    fused_vert = vertical_component(fused_joints, fused_up)

    # Try every strong correlation-peak candidate against the real spatial
    # fit and keep whichever minimizes the residual, rather than trusting
    # the strongest correlation peak alone (see
    # align.cross_correlate_lag_candidates's docstring for why).
    candidates = cross_correlate_lag_candidates(mocap_time, mocap_vert, fused_time, fused_vert)
    results = [_fit_at_lag(lag, mocap_joints, mocap_time, fused_joints, fused_time) for lag, _ in candidates]
    best = min(results, key=lambda r: r["mean_residual_mm"])
    if best["lag_seconds"] != candidates[0][0]:
        strongest = next(r["mean_residual_mm"] for r in results if r["lag_seconds"] == candidates[0][0])
        print(f"    NOTE: strongest correlation peak (lag={candidates[0][0]:.2f}s) gave a worse fit "
              f"than lag={best['lag_seconds']:.2f}s ({best['mean_residual_mm']:.0f}mm vs {strongest:.0f}mm) "
              f"-- picked the lower-residual lag instead")
    return best


def main():
    manifest = json.loads(MANIFEST_PATH.read_text())
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for trial, entry in manifest["trials"].items():
        if not entry["paired"]:
            continue
        fused_path = FUSED_DIR / f"{trial}.npz"
        out_path = OUTPUT_DIR / f"{trial}.npz"
        if not fused_path.exists() or out_path.exists():
            continue

        print(f"Aligning fused {trial} ...")
        try:
            result = align_trial(trial)
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

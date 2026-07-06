"""Phase 5: fuse view1 and view2's independent SAM 3D Body reconstructions
into a single per-trial joint trajectory, using the two RGB views *only* --
no mocap, no camera calibration. This mirrors what a real two-camera
deployment would have to do (the .xcp calibration only covers the mocap
infrared cameras, not the RGB ones -- confirmed in Phase 0). Mocap is used
strictly afterward, in Phase 6, to score the fused result.

Per trial:
1. Temporal sync: cross-correlate a broadband vertical signal between
   view1 and view2's own SAM 3D Body output (same technique Phase 3 uses
   for view-vs-mocap sync, just view-vs-view here) to find their relative
   lag, then resample view2 onto view1's own frame timeline.
2. Whole-trial spatial fit: one Umeyama similarity transform (rotation +
   scale + translation) mapping view2's canonical joints onto view1's,
   fit across every temporally-overlapping frame -- brings both views into
   one shared coordinate frame/scale for the whole trial. view1 is an
   arbitrary but fixed reference for this step; it doesn't bias the later
   comparison since Phase 6 re-aligns the fused result to mocap from
   scratch regardless of which view anchored this step.
3. Per-frame GPA refinement: with both views root-centered (subtract each
   view's own pelvis position, per frame), a two-shape Generalized
   Procrustes fit (rotation + uniform scale -- no translation needed,
   already centered) between the root-relative skeletons corrects
   residual frame-level pose disagreement beyond the trial's fixed
   inter-camera relationship -- and does so symmetrically (both views
   rotate/scale toward their consensus shape; neither is treated as
   ground truth).
4. Fuse: average the two now-mutually-aligned views per frame -- root
   position (translation) and root-relative joint offsets (shape)
   separately, then recombine. Also save each joint's post-alignment
   inter-view disagreement (||view1-view2||) per frame, as an implicit
   occlusion/confidence signal for weighted aggregation in Phase 6/7 (not
   used to bias the fused position here -- with only two, mutually
   symmetric estimates, disagreement alone can't tell you *which* view is
   wrong, only that this joint/frame is uncertain).
"""

import json
import sys
from pathlib import Path

import numpy as np
from scipy.signal import savgol_filter

sys.path.insert(0, str(Path(__file__).resolve().parent))
from align import umeyama_alignment, apply_similarity, cross_correlate_lag_candidates, resample_joints_to_times
from joint_mapping import CANONICAL_JOINTS, sam3d_canonical_joints, detect_vertical_axis_generic

SAM3D_REPO = Path(__file__).resolve().parent.parent / "third_party" / "sam-3d-body"
sys.path.insert(0, str(SAM3D_REPO))
from sam_3d_body.metadata.mhr70 import mhr_names  # noqa: E402

ANALYSIS_DIR = Path(__file__).resolve().parent.parent
MANIFEST_PATH = ANALYSIS_DIR / "manifest.json"
SAM3D_DIR = ANALYSIS_DIR / "output" / "sam3d"
OUTPUT_DIR = ANALYSIS_DIR / "output" / "fused"

ROOT_JOINT = "pelvis"


def _smooth(arr: np.ndarray, window: int = 9, polyorder: int = 2) -> np.ndarray:
    n = arr.shape[0]
    w = min(window, n if n % 2 == 1 else n - 1)
    if w <= polyorder or w < 3:
        return arr.copy()
    return savgol_filter(arr, window_length=w, polyorder=polyorder, axis=0)


def load_view_joints(trial: str, view: str, video_fps: float):
    d = np.load(SAM3D_DIR / f"{trial}__{view}.npz")
    # pred_cam_t wasn't smoothed when Phase 2 saved it -- smooth it here for
    # consistency, same as run_alignment.py does for the mocap-side alignment.
    cam_t_smoothed = _smooth(d["pred_cam_t"])
    joints = sam3d_canonical_joints(d["pred_keypoints_3d_smoothed"], mhr_names, cam_t_smoothed)
    time = d["frame_idx"] / video_fps
    up = detect_vertical_axis_generic(joints["head"], joints["toe_left"])
    return joints, time, up


def vertical_component(joints: dict, up: np.ndarray) -> np.ndarray:
    stacked = np.stack([joints[j] for j in CANONICAL_JOINTS], axis=0)  # (J, F, 3)
    return np.nanmean(stacked, axis=0) @ up


def _procrustes_rotation_scale(source: np.ndarray, target: np.ndarray):
    """Rotation + uniform scale (no translation) mapping `source` onto
    `target`; both must already be centered at the origin. Same derivation
    as align.umeyama_alignment with the translation term dropped (and the
    same U/V roles verified there: R = V @ S @ U^T, not U @ S @ V^T)."""
    sigma = (source.T @ target) / len(source)
    U, D, Vt = np.linalg.svd(sigma)
    S = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        S[-1, -1] = -1.0
    R = Vt.T @ S @ U.T
    var_s = np.mean(np.sum(source ** 2, axis=1))
    scale = np.trace(np.diag(D) @ S) / var_s if var_s > 1e-9 else 1.0
    return R, scale


def _rms_size(shape: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.sum(shape ** 2, axis=1))))


def _rescale_to_rms(shape: np.ndarray, target_rms: float) -> np.ndarray:
    current = _rms_size(shape)
    return shape if current < 1e-9 else shape * (target_rms / current)


def gpa_two_shapes(x1: np.ndarray, x2: np.ndarray, max_iter: int = 10, tol: float = 1e-4):
    """Generalized Procrustes Analysis between two already root-centered
    (J,3) shapes -- iteratively align both to their running mean shape
    (rotation + scale) until the mean stops moving. Returns both shapes
    re-expressed in the converged mean shape's frame -- neither view is
    treated as ground truth, both rotate/scale toward the consensus.

    The running mean shape's overall size is re-anchored to the fixed
    target `0.5*(rms(x1)+rms(x2))` after every update. Without this, GPA
    has a well-known degeneracy: since averaging two not-yet-aligned
    shapes always yields an equal-or-*smaller* RMS size (only equal if
    they're already perfectly aligned), the mean shape shrinks a little
    every iteration, which then pulls the next fit's scale down further --
    a slow monotonic collapse toward zero that never actually converges
    (verified: `shift` never dropped below `tol` even after 10 iterations,
    with scale still visibly drifting). Re-anchoring the size breaks that
    feedback loop; with it, this converges in ~2 iterations."""
    target_rms = 0.5 * (_rms_size(x1) + _rms_size(x2))
    mean_shape = _rescale_to_rms(0.5 * (x1 + x2), target_rms)
    x1_aligned, x2_aligned = x1, x2
    converged = False
    for _ in range(max_iter):
        R1, s1 = _procrustes_rotation_scale(x1, mean_shape)
        R2, s2 = _procrustes_rotation_scale(x2, mean_shape)
        x1_aligned = s1 * (R1 @ x1.T).T
        x2_aligned = s2 * (R2 @ x2.T).T
        new_mean = _rescale_to_rms(0.5 * (x1_aligned + x2_aligned), target_rms)
        shift = float(np.mean(np.linalg.norm(new_mean - mean_shape, axis=-1)))
        mean_shape = new_mean
        if shift < tol:
            converged = True
            break
    return x1_aligned, x2_aligned, converged


def _rigid_fit_at_lag(lag: float, j1: dict, t1, j2: dict, t2):
    """Whole-trial rigid+scale fit (view2 -> view1) for one candidate lag --
    used to score candidates before committing to the expensive per-frame
    GPA loop for just the winner."""
    t2_shifted = t2 + lag
    j2_resampled, valid_mask = resample_joints_to_times(t2_shifted, j2, t1)
    j1_overlap = {name: arr[valid_mask] for name, arr in j1.items()}
    time_overlap = t1[valid_mask]

    src_pts = np.concatenate([j2_resampled[j] for j in CANONICAL_JOINTS], axis=0)
    dst_pts = np.concatenate([j1_overlap[j] for j in CANONICAL_JOINTS], axis=0)
    fit_mask = ~np.isnan(src_pts).any(axis=-1) & ~np.isnan(dst_pts).any(axis=-1)
    R, scale, t = umeyama_alignment(src_pts[fit_mask], dst_pts[fit_mask])
    fitted = apply_similarity(src_pts[fit_mask], R, scale, t)
    mean_residual = float(np.mean(np.linalg.norm(fitted - dst_pts[fit_mask], axis=-1)))

    j2_aligned = {name: apply_similarity(arr, R, scale, t) for name, arr in j2_resampled.items()}
    return {
        "lag": lag, "mean_residual": mean_residual, "R": R, "scale": scale, "t": t,
        "j1_overlap": j1_overlap, "j2_aligned": j2_aligned, "time_overlap": time_overlap,
    }


def fuse_trial(trial: str, fps1: float, fps2: float) -> dict:
    j1, t1, up1 = load_view_joints(trial, "view1", fps1)
    j2, t2, up2 = load_view_joints(trial, "view2", fps2)

    v1 = vertical_component(j1, up1)
    v2 = vertical_component(j2, up2)

    # Try every strong correlation-peak candidate against the real spatial
    # fit and keep whichever minimizes the residual, rather than trusting
    # the strongest correlation peak alone -- confirmed on real data that
    # the strongest peak can be a cycle-slip (see
    # align.cross_correlate_lag_candidates's docstring for how this was found).
    candidates = cross_correlate_lag_candidates(t1, v1, t2, v2)
    fits = [_rigid_fit_at_lag(lag, j1, t1, j2, t2) for lag, _ in candidates]
    best = min(fits, key=lambda f: f["mean_residual"])
    if best["lag"] != candidates[0][0]:
        strongest_residual = next(f["mean_residual"] for f in fits if f["lag"] == candidates[0][0])
        print(f"    NOTE: strongest correlation peak (lag={candidates[0][0]:.2f}s) gave a worse fit "
              f"than lag={best['lag']:.2f}s ({best['mean_residual']:.3f} vs {strongest_residual:.3f}) "
              f"-- picked the lower-residual lag instead")

    lag = best["lag"]
    R, scale, t = best["R"], best["scale"], best["t"]
    j1_overlap = best["j1_overlap"]
    j2_aligned = best["j2_aligned"]
    time_overlap = best["time_overlap"]

    n_frames = len(time_overlap)
    fused = {name: np.full((n_frames, 3), np.nan) for name in CANONICAL_JOINTS}
    disagreement = {name: np.full(n_frames, np.nan) for name in CANONICAL_JOINTS}
    n_unconverged = 0

    for f in range(n_frames):
        root1 = j1_overlap[ROOT_JOINT][f]
        root2 = j2_aligned[ROOT_JOINT][f]
        if np.isnan(root1).any() or np.isnan(root2).any():
            continue

        x1 = np.stack([j1_overlap[j][f] - root1 for j in CANONICAL_JOINTS])
        x2 = np.stack([j2_aligned[j][f] - root2 for j in CANONICAL_JOINTS])
        if np.isnan(x1).any() or np.isnan(x2).any():
            continue

        x1_al, x2_al, converged = gpa_two_shapes(x1, x2)
        if not converged:
            n_unconverged += 1
        fused_local = 0.5 * (x1_al + x2_al)
        root_f = 0.5 * (root1 + root2)

        for i, name in enumerate(CANONICAL_JOINTS):
            fused[name][f] = root_f + fused_local[i]
            disagreement[name][f] = np.linalg.norm(x1_al[i] - x2_al[i])

    return {
        "time": time_overlap,
        "lag_seconds": lag,
        "scale_view2_to_view1": scale,
        "R_view2_to_view1": R,
        "t_view2_to_view1": t,
        "fused_joints": fused,
        "disagreement": disagreement,
        "n_frames": n_frames,
        "n_unconverged": n_unconverged,
    }


def main():
    manifest = json.loads(MANIFEST_PATH.read_text())
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for trial, entry in manifest["trials"].items():
        if not entry["paired"]:
            continue
        out_path = OUTPUT_DIR / f"{trial}.npz"
        if out_path.exists():
            continue
        v1_path = SAM3D_DIR / f"{trial}__view1.npz"
        v2_path = SAM3D_DIR / f"{trial}__view2.npz"
        if not (v1_path.exists() and v2_path.exists()):
            continue

        print(f"Fusing {trial} ...")
        try:
            result = fuse_trial(trial, entry["views"]["view1"]["fps"], entry["views"]["view2"]["fps"])
        except Exception as e:
            print(f"  FAILED: {e}")
            continue

        all_disagreement = np.concatenate([result["disagreement"][j] for j in CANONICAL_JOINTS])
        print(f"  lag={result['lag_seconds']:.3f}s  scale(view2->view1)={result['scale_view2_to_view1']:.4f}  "
              f"n_frames={result['n_frames']}  n_unconverged={result['n_unconverged']}  "
              f"mean_inter_view_disagreement={np.nanmean(all_disagreement):.4f} (model units, ~meters)")

        np.savez_compressed(
            out_path,
            time=result["time"],
            lag_seconds=result["lag_seconds"],
            scale_view2_to_view1=result["scale_view2_to_view1"],
            R_view2_to_view1=result["R_view2_to_view1"],
            t_view2_to_view1=result["t_view2_to_view1"],
            n_unconverged=result["n_unconverged"],
            **{f"fused__{k}": v for k, v in result["fused_joints"].items()},
            **{f"disagreement__{k}": v for k, v in result["disagreement"].items()},
        )
        print(f"  Saved -> {out_path}")


if __name__ == "__main__":
    main()

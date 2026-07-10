"""Phase 4: independent per-view comparison of SAM 3D Body against mocap
ground truth. Reads output/aligned/*.npz (Phase 3's output) only -- no new
model inference. For every trial/view, computes:

- MPJPE: mean per-joint position error (mm), broken down per joint, using
  Phase 3's whole-trial alignment.
- PA-MPJPE: mean per-joint position error after an *additional* per-frame
  Procrustes (rotation + scale + translation) fit -- a separate, tighter
  alignment than Phase 3's single whole-trial fit, isolating pose/shape
  error from any residual per-frame drift.
- Joint angles: RMSE, MAE, Pearson r and Bland-Altman bias/limits of
  agreement between mocap- and SAM3D-derived hip/knee/ankle flexion angles.
  Both sides' angles are computed with the same joint-only method
  (mocap.angles.compute_joint_angles_from_joints) for methodological parity,
  then calibrated against the mocap `static` trial's own offset (decision:
  lean on the mocap-derived zero-reference for interpretability -- this is
  a common additive shift applied to both sides equally, so it cannot
  change the mocap-vs-SAM3D difference/bias itself, only the absolute
  scale angles are reported on).

Results are aggregated by action category (trial name with its trailing
"_<n>" stripped) and by view. Safe to re-run any time.
"""

import json
import re
import sys
from pathlib import Path

import numpy as np
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent))
from align import umeyama_alignment, apply_similarity
from joint_mapping import CANONICAL_JOINTS
from mocap.angles import compute_joint_angles_from_joints, circular_mean_deg, wrap_around_center

ANALYSIS_DIR = Path(__file__).resolve().parent.parent
MANIFEST_PATH = ANALYSIS_DIR / "manifest.json"
MOCAP_DIR = ANALYSIS_DIR / "output" / "mocap"
ALIGNED_DIR = ANALYSIS_DIR / "output" / "aligned"
OUTPUT_DIR = ANALYSIS_DIR / "output" / "metrics"

MOCAP_UP = np.array([0.0, 1.0, 0.0])  # mocap .trc is Y-up; Phase 3 aligns SAM3D into this same frame

ANGLE_NAMES = [
    "hip_flexion_left", "hip_flexion_right",
    "knee_flexion_left", "knee_flexion_right",
    "ankle_flexion_left", "ankle_flexion_right",
]

# A per-frame Procrustes fit needs enough non-collinear points to be
# well-conditioned; below this, skip the frame (NaN) rather than trust an
# unstable fit from a near-degenerate point set.
MIN_JOINTS_FOR_PA = 6


def action_category(trial: str) -> str:
    return re.sub(r"_\d+$", "", trial)


def load_static_angle_offsets() -> dict:
    """Circular-mean angle value per joint during the `static` (relaxed
    standing) trial -- the shared zero-reference subtracted from both
    mocap- and SAM3D-derived angles below."""
    d = np.load(MOCAP_DIR / "static.npz")
    joints = {name: d[f"joint__{name}"] for name in CANONICAL_JOINTS}
    angles = compute_joint_angles_from_joints(joints, MOCAP_UP)
    return {name: circular_mean_deg(angles[name]) for name in ANGLE_NAMES}


def mpjpe_per_joint(aligned: dict, mocap: dict) -> dict:
    return {
        j: float(np.nanmean(np.linalg.norm(aligned[j] - mocap[j], axis=-1)))
        for j in CANONICAL_JOINTS
    }


def pa_mpjpe_per_frame(aligned: dict, mocap: dict) -> np.ndarray:
    """Per-frame Procrustes-aligned mean joint error -- a fresh
    rotation+scale+translation fit for each individual frame's 19-joint
    pose (as opposed to Phase 3's one fit for the whole trial), isolating
    shape/articulation error from any residual whole-trial misalignment."""
    src = np.stack([aligned[j] for j in CANONICAL_JOINTS], axis=1)  # (F, J, 3)
    dst = np.stack([mocap[j] for j in CANONICAL_JOINTS], axis=1)   # (F, J, 3)
    n_frames = src.shape[0]
    errs = np.full(n_frames, np.nan)
    for f in range(n_frames):
        valid = ~np.isnan(src[f]).any(axis=-1) & ~np.isnan(dst[f]).any(axis=-1)
        if valid.sum() < MIN_JOINTS_FOR_PA:
            continue
        R, scale, t = umeyama_alignment(src[f][valid], dst[f][valid])
        fitted = apply_similarity(src[f][valid], R, scale, t)
        errs[f] = np.mean(np.linalg.norm(fitted - dst[f][valid], axis=-1))
    return errs


def angle_metrics(mocap_angle: np.ndarray, sam3d_angle: np.ndarray, unreliable: bool) -> dict:
    # Wrap the *difference* (not the raw signals) around zero so an
    # independent wrap-point choice on either side can't show up as a
    # spurious ~360 deg jump in the comparison.
    diff = wrap_around_center(sam3d_angle - mocap_angle, 0.0)
    valid = ~np.isnan(diff)
    n = int(valid.sum())
    result = {"n_valid": n, "unreliable": unreliable}
    if n < 2:
        return result

    d = diff[valid]
    m = mocap_angle[valid]
    # Correlate against mocap + diff (SAM3D's value re-expressed on
    # mocap's own numeric branch), not the raw independently-wrapped
    # sam3d_angle -- otherwise two signals that only differ by which side
    # of an arbitrary +-180 deg cut they each landed on (very possible
    # here, since "thigh straight down" -- the neutral standing pose -- is
    # exactly +-180 deg by construction of this angle) can register as a
    # strong *spurious* correlation of the wrong sign, even though the
    # actual frame-by-frame angular difference (the same `diff` used for
    # RMSE/bias above) is small and well-behaved.
    s_rewrapped = m + d
    if np.std(m) > 0 and np.std(s_rewrapped) > 0:
        r, _ = stats.pearsonr(m, s_rewrapped)
    else:
        r = float("nan")
    bias = float(np.mean(d))
    sd = float(np.std(d))

    result.update({
        "rmse_deg": float(np.sqrt(np.mean(d ** 2))),
        "mae_deg": float(np.mean(np.abs(d))),
        "pearson_r": float(r),
        "bland_altman_bias_deg": bias,
        "bland_altman_loa_lower_deg": bias - 1.96 * sd,
        "bland_altman_loa_upper_deg": bias + 1.96 * sd,
    })
    return result


def process_trial_view(trial: str, view: str, static_offsets: dict) -> dict:
    d = np.load(ALIGNED_DIR / f"{trial}__{view}.npz")
    aligned = {j: d[f"aligned__{j}"] for j in CANONICAL_JOINTS}
    mocap = {j: d[f"mocap__{j}"] for j in CANONICAL_JOINTS}

    mpjpe = mpjpe_per_joint(aligned, mocap)
    pa_errs = pa_mpjpe_per_frame(aligned, mocap)

    mocap_angles_raw = compute_joint_angles_from_joints(mocap, MOCAP_UP)
    # frame_joints=mocap: SAM3D's own hip_left/hip_right vector is a fixed,
    # ~44%-width, ~106deg-misoriented distortion of the true pelvis axis
    # (see mocap.angles.compute_joint_angles_from_joints docstring) -- use
    # mocap's own, correctly-oriented frame to project the estimate's
    # thigh/shank/foot vectors instead of the estimate's own frame.
    sam3d_angles_raw = compute_joint_angles_from_joints(aligned, MOCAP_UP, frame_joints=mocap)
    mocap_unreliable = set(mocap_angles_raw.pop("_unreliable"))
    sam3d_unreliable = set(sam3d_angles_raw.pop("_unreliable"))

    angles = {}
    for name in ANGLE_NAMES:
        offset = static_offsets[name]
        mocap_a = mocap_angles_raw[name] - offset
        sam3d_a = sam3d_angles_raw[name] - offset
        unreliable = name in mocap_unreliable or name in sam3d_unreliable
        angles[name] = angle_metrics(mocap_a, sam3d_a, unreliable)

    return {
        "trial": trial,
        "view": view,
        "category": action_category(trial),
        "mpjpe_per_joint_mm": mpjpe,
        "mpjpe_overall_mm": float(np.mean(list(mpjpe.values()))),
        "pa_mpjpe_mm": float(np.nanmean(pa_errs)) if np.any(~np.isnan(pa_errs)) else float("nan"),
        "pa_mpjpe_n_frames": int(np.sum(~np.isnan(pa_errs))),
        "angles": angles,
    }


def aggregate(results: list, key_fn) -> dict:
    groups = {}
    for r in results:
        groups.setdefault(key_fn(r), []).append(r)

    out = {}
    for key, group in groups.items():
        per_joint = {
            j: float(np.mean([r["mpjpe_per_joint_mm"][j] for r in group]))
            for j in CANONICAL_JOINTS
        }
        angle_summary = {}
        for name in ANGLE_NAMES:
            entries = [
                r["angles"][name] for r in group
                if not r["angles"][name]["unreliable"] and "rmse_deg" in r["angles"][name]
            ]
            if not entries:
                angle_summary[name] = {"n_trials": 0, "note": "all unreliable or insufficient overlap"}
                continue
            angle_summary[name] = {
                "n_trials": len(entries),
                "rmse_deg": float(np.mean([e["rmse_deg"] for e in entries])),
                "mae_deg": float(np.mean([e["mae_deg"] for e in entries])),
                "pearson_r": float(np.nanmean([e["pearson_r"] for e in entries])),
                "bland_altman_bias_deg": float(np.mean([e["bland_altman_bias_deg"] for e in entries])),
            }
        out[key] = {
            "n_trial_views": len(group),
            "mpjpe_overall_mm": float(np.mean([r["mpjpe_overall_mm"] for r in group])),
            "pa_mpjpe_mm": float(np.nanmean([r["pa_mpjpe_mm"] for r in group])),
            "mpjpe_per_joint_mm": per_joint,
            "angles": angle_summary,
        }
    return out


def main():
    manifest = json.loads(MANIFEST_PATH.read_text())
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    static_offsets = load_static_angle_offsets()
    print("Static-trial angle calibration offsets (deg):")
    for name, off in static_offsets.items():
        print(f"  {name:22s} {off:+7.1f}")

    results = []
    for trial, entry in manifest["trials"].items():
        if not entry["paired"]:
            continue
        for view in ("view1", "view2"):
            path = ALIGNED_DIR / f"{trial}__{view}.npz"
            if not path.exists():
                continue
            print(f"\nProcessing {trial} / {view} ...")
            r = process_trial_view(trial, view, static_offsets)
            results.append(r)
            print(f"  MPJPE={r['mpjpe_overall_mm']:.1f}mm  "
                  f"PA-MPJPE={r['pa_mpjpe_mm']:.1f}mm (n={r['pa_mpjpe_n_frames']})")
            for name in ANGLE_NAMES:
                m = r["angles"][name]
                if "rmse_deg" not in m:
                    print(f"    {name:22s} insufficient overlap")
                    continue
                flag = " [UNRELIABLE]" if m["unreliable"] else ""
                print(f"    {name:22s} RMSE={m['rmse_deg']:5.1f} deg  "
                      f"bias={m['bland_altman_bias_deg']:+.1f} deg  r={m['pearson_r']:.2f}{flag}")

    by_view = aggregate(results, lambda r: r["view"])
    by_category = aggregate(results, lambda r: r["category"])
    by_category_view = aggregate(results, lambda r: f"{r['category']}__{r['view']}")
    overall = aggregate(results, lambda r: "overall")

    summary = {
        "per_trial_view": results,
        "by_view": by_view,
        "by_category": by_category,
        "by_category_view": by_category_view,
        "overall": overall,
    }

    out_path = OUTPUT_DIR / "phase4_metrics.json"
    out_path.write_text(json.dumps(summary, indent=2, default=str))
    print(f"\nSaved -> {out_path}")

    print("\n=== Overall (all trials, both views) ===")
    o = overall["overall"]
    print(f"MPJPE: {o['mpjpe_overall_mm']:.1f}mm   PA-MPJPE: {o['pa_mpjpe_mm']:.1f}mm")
    for name, m in o["angles"].items():
        if m["n_trials"] == 0:
            print(f"  {name:22s} -- {m['note']}")
        else:
            print(f"  {name:22s} RMSE={m['rmse_deg']:5.1f} deg  MAE={m['mae_deg']:5.1f} deg  "
                  f"r={m['pearson_r']:.2f}  bias={m['bland_altman_bias_deg']:+.1f} deg  (n={m['n_trials']})")

    print("\n=== By action category ===")
    for cat, m in by_category.items():
        print(f"  {cat:18s} MPJPE={m['mpjpe_overall_mm']:6.1f}mm  "
              f"PA-MPJPE={m['pa_mpjpe_mm']:6.1f}mm  (n={m['n_trial_views']})")

    print("\n=== By view ===")
    for v, m in by_view.items():
        print(f"  {v:10s} MPJPE={m['mpjpe_overall_mm']:6.1f}mm  "
              f"PA-MPJPE={m['pa_mpjpe_mm']:6.1f}mm  (n={m['n_trial_views']})")


if __name__ == "__main__":
    main()

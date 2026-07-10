"""Phase 6 metrics: compare the fused trajectory (Phase 5 fusion + Phase 6
alignment to mocap) against mocap ground truth, using the same MPJPE,
PA-MPJPE, and joint-angle metrics as Phase 4. Reuses compare_metrics.py's
per-frame/per-joint functions directly rather than re-implementing them --
same wrap-consistency-safe angle comparison, same per-frame Procrustes
PA-MPJPE -- so the fused-vs-view comparison in Phase 7 is exactly
apples-to-apples with Phase 4's numbers. One result per trial (not per
view, since fusion already combined both views).
"""

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from joint_mapping import CANONICAL_JOINTS
from mocap.angles import compute_joint_angles_from_joints
from compare_metrics import (
    ANGLE_NAMES, MOCAP_UP, action_category, load_static_angle_offsets,
    mpjpe_per_joint, pa_mpjpe_per_frame, angle_metrics, aggregate,
)

ANALYSIS_DIR = Path(__file__).resolve().parent.parent
MANIFEST_PATH = ANALYSIS_DIR / "manifest.json"
ALIGNED_FUSED_DIR = ANALYSIS_DIR / "output" / "aligned_fused"
OUTPUT_DIR = ANALYSIS_DIR / "output" / "metrics"


def process_trial(trial: str, static_offsets: dict) -> dict:
    d = np.load(ALIGNED_FUSED_DIR / f"{trial}.npz")
    aligned = {j: d[f"aligned__{j}"] for j in CANONICAL_JOINTS}
    mocap = {j: d[f"mocap__{j}"] for j in CANONICAL_JOINTS}

    mpjpe = mpjpe_per_joint(aligned, mocap)
    pa_errs = pa_mpjpe_per_frame(aligned, mocap)

    mocap_angles_raw = compute_joint_angles_from_joints(mocap, MOCAP_UP)
    # frame_joints=mocap: see mocap.angles.compute_joint_angles_from_joints
    # docstring and compare_metrics.process_trial_view -- the fused
    # skeleton's own pelvis axis is systematically misoriented, so project
    # its thigh/shank/foot vectors onto mocap's frame instead.
    fused_angles_raw = compute_joint_angles_from_joints(aligned, MOCAP_UP, frame_joints=mocap)
    mocap_unreliable = set(mocap_angles_raw.pop("_unreliable"))
    fused_unreliable = set(fused_angles_raw.pop("_unreliable"))

    angles = {}
    for name in ANGLE_NAMES:
        offset = static_offsets[name]
        mocap_a = mocap_angles_raw[name] - offset
        fused_a = fused_angles_raw[name] - offset
        unreliable = name in mocap_unreliable or name in fused_unreliable
        angles[name] = angle_metrics(mocap_a, fused_a, unreliable)

    return {
        "trial": trial,
        "view": "fused",
        "category": action_category(trial),
        "mpjpe_per_joint_mm": mpjpe,
        "mpjpe_overall_mm": float(np.mean(list(mpjpe.values()))),
        "pa_mpjpe_mm": float(np.nanmean(pa_errs)) if np.any(~np.isnan(pa_errs)) else float("nan"),
        "pa_mpjpe_n_frames": int(np.sum(~np.isnan(pa_errs))),
        "angles": angles,
    }


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
        path = ALIGNED_FUSED_DIR / f"{trial}.npz"
        if not path.exists():
            continue
        print(f"\nProcessing fused {trial} ...")
        r = process_trial(trial, static_offsets)
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

    by_category = aggregate(results, lambda r: r["category"])
    overall = aggregate(results, lambda r: "overall")

    summary = {
        "per_trial": results,
        "by_category": by_category,
        "overall": overall,
    }

    out_path = OUTPUT_DIR / "phase6_metrics.json"
    out_path.write_text(json.dumps(summary, indent=2, default=str))
    print(f"\nSaved -> {out_path}")

    print("\n=== Overall (all 10 fused trials) ===")
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


if __name__ == "__main__":
    main()

"""Full audit of Phase 1-6 outputs across ALL trials (not just samples), run
after any change to mocap/, align.py, joint_mapping.py, compare_metrics.py,
fuse_views.py, align_fused.py, or compare_metrics_fused.py so bugs surface
here rather than corrupting downstream results silently.
"""

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from joint_mapping import CANONICAL_JOINTS

ANALYSIS_DIR = Path(__file__).resolve().parent.parent
MANIFEST_PATH = ANALYSIS_DIR / "manifest.json"
MOCAP_DIR = ANALYSIS_DIR / "output" / "mocap"
SAM3D_DIR = ANALYSIS_DIR / "output" / "sam3d"
ALIGNED_DIR = ANALYSIS_DIR / "output" / "aligned"
METRICS_DIR = ANALYSIS_DIR / "output" / "metrics"
FUSED_DIR = ANALYSIS_DIR / "output" / "fused"
ALIGNED_FUSED_DIR = ANALYSIS_DIR / "output" / "aligned_fused"

TRIALS = [t for t in json.loads(MANIFEST_PATH.read_text())["trials"]]
PAIRED_TRIALS = [t for t, e in json.loads(MANIFEST_PATH.read_text())["trials"].items() if e["paired"]]


def audit_phase1():
    print("=" * 70)
    print("PHASE 1: mocap ground truth (all 11 trials)")
    print("=" * 70)
    for trial in TRIALS:
        d = np.load(MOCAP_DIR / f"{trial}.npz")
        up = d["up_axis"]
        n = len(d["time"])
        issues = []
        if not np.allclose(up, [0, 1, 0]):
            issues.append(f"up_axis={up} (expected [0,1,0])")
        for name in CANONICAL_JOINTS:
            arr = d[f"joint__{name}"]
            nan_frac = np.isnan(arr).any(axis=-1).mean()
            if nan_frac > 0.5:
                issues.append(f"{name}: {nan_frac:.0%} NaN (severe occlusion)")
        known_unreliable = set(d["unreliable_angles"].tolist()) if "unreliable_angles" in d.files else set()
        for angle_name in ("hip_flexion_left", "hip_flexion_right",
                            "knee_flexion_left", "knee_flexion_right",
                            "ankle_flexion_left", "ankle_flexion_right"):
            arr = d[f"angle__{angle_name}"]
            valid = arr[~np.isnan(arr)]
            lo, hi = valid.min(), valid.max()
            max_jump = np.abs(np.diff(valid)).max()
            # A wide absolute range with small per-frame jumps is just a
            # cosmetic (safe) 360 deg multiple offset; a large frame-to-frame
            # jump is a genuine discontinuity -- but if mocap/angles.py
            # already flagged it (known limitation, see its docstring),
            # this is documented/expected, not a fresh issue to chase.
            if max_jump > 45:
                if angle_name in known_unreliable:
                    issues.append(f"{angle_name}: known residual jump ({max_jump:.0f} deg, documented limitation)")
                else:
                    issues.append(f"{angle_name}: UNFLAGGED jump {max_jump:.0f} deg -- new issue, investigate")
            elif abs(lo) > 200 or abs(hi) > 200:
                issues.append(f"{angle_name}: range [{lo:.0f}, {hi:.0f}] deg (smooth, likely just cosmetic offset)")

        status = "OK" if not issues else "CHECK: " + "; ".join(issues)
        print(f"  {trial:20s} n={n:5d}  up_axis={up}  {status}")


def audit_phase2():
    print()
    print("=" * 70)
    print("PHASE 2: SAM 3D Body per-view reconstruction (all 20 trial/views)")
    print("=" * 70)
    for trial in PAIRED_TRIALS:
        for view in ("view1", "view2"):
            d = np.load(SAM3D_DIR / f"{trial}__{view}.npz")
            n = len(d["frame_idx"])
            issues = []

            if np.isnan(d["pred_keypoints_3d_smoothed"]).any():
                issues.append("NaN in pred_keypoints_3d_smoothed")
            if np.isnan(d["pred_cam_t"]).any():
                issues.append("NaN in pred_cam_t")

            # bbox should be the full image every frame (no detector used)
            bbox = d["bbox"]
            bbox_var = bbox.std(axis=0)
            if bbox_var.max() > 1e-3:
                issues.append(f"bbox not constant across frames (std={bbox_var}) -- detector fallback assumption broke?")

            # cam_t depth (z, typically index 2) should be positive and roughly stable
            # (person shouldn't appear to teleport to a wildly different depth)
            cam_t = d["pred_cam_t"]
            depth = cam_t[:, 2]
            if depth.min() <= 0:
                issues.append(f"non-positive depth in pred_cam_t z (min={depth.min():.2f})")
            depth_jump = np.abs(np.diff(depth)).max()
            if depth_jump > 0.5:
                issues.append(f"large frame-to-frame depth jump ({depth_jump:.2f}m) -- possible bad frame")

            # smoothing sanity: smoothed shouldn't differ wildly from raw
            raw = d["pred_keypoints_3d"]
            smoothed = d["pred_keypoints_3d_smoothed"]
            max_smooth_diff = np.abs(raw - smoothed).max()
            if max_smooth_diff > 0.5:
                issues.append(f"smoothing changed a value by {max_smooth_diff:.2f}m -- check window/outliers")

            status = "OK" if not issues else "CHECK: " + "; ".join(issues)
            print(f"  {trial:20s} {view}  n={n:4d}  {status}")


def audit_phase3():
    print()
    print("=" * 70)
    print("PHASE 3: alignment (all 20 trial/views) -- per-joint + geometry checks")
    print("=" * 70)
    worst_joint_overall = {}
    for trial in PAIRED_TRIALS:
        for view in ("view1", "view2"):
            d = np.load(ALIGNED_DIR / f"{trial}__{view}.npz")
            issues = []

            R = d["R"]
            det = np.linalg.det(R)
            if abs(det - 1.0) > 1e-3:
                issues.append(f"R determinant={det:.4f} (expected +1; -1 would mean an uncorrected mirror/reflection)")

            scale = float(d["scale"])
            if not (700 < scale < 1300):
                issues.append(f"scale={scale:.1f} far from expected ~1000")

            lag = float(d["lag_seconds"])
            if abs(lag) > 3.0:
                issues.append(f"lag={lag:.2f}s at/beyond search window edge")

            mean_res = float(d["mean_residual_mm"])
            if mean_res > 200:
                issues.append(f"mean_residual={mean_res:.0f}mm is high")

            per_joint = {}
            for j in CANONICAL_JOINTS:
                res = np.linalg.norm(d[f"aligned__{j}"] - d[f"mocap__{j}"], axis=-1)
                per_joint[j] = float(np.nanmean(res))
            worst_j = max(per_joint, key=per_joint.get)
            worst_joint_overall.setdefault(worst_j, 0)
            worst_joint_overall[worst_j] += 1

            status = "OK" if not issues else "CHECK: " + "; ".join(issues)
            print(f"  {trial:20s} {view}  worst_joint={worst_j:15s}({per_joint[worst_j]:5.0f}mm)  {status}")

    print()
    print("  Joints that were the worst-tracked joint, count across all 20 trial/views:")
    for j, count in sorted(worst_joint_overall.items(), key=lambda kv: -kv[1]):
        print(f"    {j:15s} {count}")


def _metrics_result_issues(r: dict) -> list:
    """Shared correctness checks for one MPJPE/PA-MPJPE/joint-angle result
    dict, as produced by compare_metrics.py (Phase 4, per view) and
    compare_metrics_fused.py (Phase 6, fused) -- used by both audits so
    both catch the same regression classes."""
    issues = []
    mpjpe = r["mpjpe_overall_mm"]
    pa = r["pa_mpjpe_mm"]

    # PA-MPJPE adds a per-frame rotation+scale+translation fit on top of
    # the whole-trial fit, so it should never come out meaningfully *worse*
    # than plain MPJPE -- a correct least-squares refit can only reduce
    # error further. This exact invariant is what caught the
    # umeyama_alignment rotation-formula bug during Phase 4 development
    # (the broken fit made PA-MPJPE come out higher than MPJPE, which is
    # impossible for a true optimizer), so keep checking it as a
    # regression guard against that bug (or similar ones) coming back.
    if not np.isnan(pa) and pa > mpjpe + 5.0:
        issues.append(f"PA-MPJPE ({pa:.1f}mm) > MPJPE ({mpjpe:.1f}mm) -- per-frame refit should not be worse")
    if r["pa_mpjpe_n_frames"] == 0:
        issues.append("0 valid frames for PA-MPJPE (insufficient joint overlap every frame)")

    for name, m in r["angles"].items():
        if "rmse_deg" not in m:
            continue
        # RMSE/bias are computed from a difference already wrapped into
        # (-180,180], so either bound being exceeded means the wrap logic
        # (or a future edit to it) is broken.
        if m["rmse_deg"] > 180.0 + 1e-6:
            issues.append(f"{name}: RMSE={m['rmse_deg']:.1f} deg exceeds the +-180 deg wrapped-difference bound")
        if abs(m["bland_altman_bias_deg"]) > 180.0 + 1e-6:
            issues.append(f"{name}: bias={m['bland_altman_bias_deg']:.1f} deg exceeds the +-180 deg bound")
        if abs(m["pearson_r"]) > 1.0 + 1e-9:
            issues.append(f"{name}: |r|={abs(m['pearson_r']):.3f} > 1 -- invalid correlation")

    # Cross-check against Phase 1's own unreliable_angles flag: an angle
    # Phase 1 already knows has a residual branch-cut jump (see
    # mocap/angles.py) should always come through marked unreliable here
    # too -- if not, the has_residual_jump check on the estimated/fused
    # side has silently diverged from Phase 1's.
    mocap_d = np.load(MOCAP_DIR / f"{r['trial']}.npz")
    known_unreliable = set(mocap_d["unreliable_angles"].tolist()) if "unreliable_angles" in mocap_d.files else set()
    for name in known_unreliable:
        if name in r["angles"] and not r["angles"][name]["unreliable"]:
            issues.append(f"{name}: flagged unreliable in Phase 1 mocap but not carried through")

    return issues


def audit_phase4():
    print()
    print("=" * 70)
    print("PHASE 4: independent per-view comparison metrics (all 20 trial/views)")
    print("=" * 70)

    metrics_path = METRICS_DIR / "phase4_metrics.json"
    if not metrics_path.exists():
        print(f"  CHECK: {metrics_path} not found -- run src/compare_metrics.py first")
        return

    summary = json.loads(metrics_path.read_text())
    results = summary["per_trial_view"]

    expected_n = 2 * len(PAIRED_TRIALS)
    if len(results) != expected_n:
        print(f"  CHECK: {len(results)} trial/view results present, expected {expected_n}")

    for r in results:
        issues = _metrics_result_issues(r)
        status = "OK" if not issues else "CHECK: " + "; ".join(issues)
        print(f"  {r['trial']:20s} {r['view']:6s} MPJPE={r['mpjpe_overall_mm']:6.1f}mm  "
              f"PA-MPJPE={r['pa_mpjpe_mm']:6.1f}mm  {status}")

    print()
    print("  Angle metrics summary (mean across all trial/views not flagged unreliable):")
    overall_angles = summary["overall"]["overall"]["angles"]
    for name, m in overall_angles.items():
        if m["n_trials"] == 0:
            print(f"    {name:22s} -- {m['note']}")
        else:
            print(f"    {name:22s} RMSE={m['rmse_deg']:5.1f} deg  r={m['pearson_r']:+.2f}  (n={m['n_trials']})")


def audit_phase5():
    print()
    print("=" * 70)
    print("PHASE 5: GPA-based cross-view fusion (all 10 paired trials)")
    print("=" * 70)

    for trial in PAIRED_TRIALS:
        path = FUSED_DIR / f"{trial}.npz"
        if not path.exists():
            print(f"  {trial:20s} CHECK: {path} not found -- run src/fuse_views.py first")
            continue

        d = np.load(path)
        issues = []

        R = d["R_view2_to_view1"]
        det = np.linalg.det(R)
        if abs(det - 1.0) > 1e-3:
            issues.append(f"R determinant={det:.4f} (expected +1; -1 would mean an uncorrected mirror/reflection)")

        # Unlike Phase 3's SAM3D->mocap fit (meters->mm, scale ~1000), this
        # is view2->view1 in the *same* native model units on both sides,
        # so the expected scale is close to 1:1.
        scale = float(d["scale_view2_to_view1"])
        if not (0.5 < scale < 2.0):
            issues.append(f"scale(view2->view1)={scale:.3f} far from the expected ~1:1")

        lag = float(d["lag_seconds"])
        if abs(lag) >= 3.0:
            issues.append(f"lag={lag:.2f}s at/beyond the +-3s search window edge")
        elif abs(lag) > 1.0:
            # The two RGB cameras record the same session (near-identical
            # durations/frame counts per manifest.json), so a lag this
            # large is unusual -- not necessarily wrong, but walking/running
            # are exactly the periodic-motion case where cross-correlation
            # can lock onto a one-stride-away peak (the same cycle-slip
            # risk documented for Phase 3's mocap-vs-view sync), so it's
            # worth a human glancing at rather than trusting silently.
            issues.append(f"lag={lag:.2f}s is large for two cameras from the same session -- "
                          f"possible cycle-slip on periodic motion, worth a look")

        nan_frac = np.isnan(d["fused__pelvis"]).any(axis=-1).mean()
        if nan_frac > 0:
            issues.append(f"{nan_frac:.0%} of frames have a NaN fused pelvis")

        # gpa_two_shapes' running mean shape must be re-anchored to a fixed
        # RMS size every iteration, or it has a classic GPA degeneracy: the
        # mean shape's size shrinks a little each iteration (averaging two
        # not-yet-aligned shapes always yields an equal-or-smaller RMS size),
        # which never actually converges -- this exact bug was caught during
        # Phase 5 development (shift never dropped below tol even after 10
        # iterations, with scale still visibly drifting). n_unconverged>0
        # means some frame hit max_iter without converging -- a direct
        # regression guard against that bug (or a similar one) coming back.
        n_unconverged = int(d["n_unconverged"])
        if n_unconverged > 0:
            issues.append(f"{n_unconverged} frame(s) hit max_iter in gpa_two_shapes without converging "
                          f"-- possible GPA scale-degeneracy regression")

        all_disagreement = np.concatenate([d[f"disagreement__{j}"] for j in CANONICAL_JOINTS])
        mean_disagreement = float(np.nanmean(all_disagreement))
        if mean_disagreement > 0.15:
            issues.append(f"mean inter-view disagreement={mean_disagreement:.3f} (model units) is high")

        status = "OK" if not issues else "CHECK: " + "; ".join(issues)
        print(f"  {trial:20s} lag={lag:6.2f}s  scale={scale:.3f}  "
              f"disagreement={mean_disagreement:.3f}  {status}")


def audit_phase6():
    print()
    print("=" * 70)
    print("PHASE 6: fused-vs-mocap alignment (all 10 paired trials)")
    print("=" * 70)

    # For context only: compare each trial's fused mean_residual_mm against
    # the best of its two independent Phase 4 views, to sanity-check that
    # fusion is in the right ballpark (this is Phase 7's actual question --
    # here it's just a smoke test, not the final analysis).
    per_view_best = {}
    metrics_path = METRICS_DIR / "phase4_metrics.json"
    if metrics_path.exists():
        phase4 = json.loads(metrics_path.read_text())
        for r in phase4["per_trial_view"]:
            best = per_view_best.get(r["trial"])
            per_view_best[r["trial"]] = r["mpjpe_overall_mm"] if best is None else min(best, r["mpjpe_overall_mm"])

    worst_joint_overall = {}
    for trial in PAIRED_TRIALS:
        path = ALIGNED_FUSED_DIR / f"{trial}.npz"
        if not path.exists():
            print(f"  {trial:20s} CHECK: {path} not found -- run src/align_fused.py first")
            continue

        d = np.load(path)
        issues = []

        R = d["R"]
        det = np.linalg.det(R)
        if abs(det - 1.0) > 1e-3:
            issues.append(f"R determinant={det:.4f} (expected +1; -1 would mean an uncorrected mirror/reflection)")

        scale = float(d["scale"])
        if not (700 < scale < 1300):
            issues.append(f"scale={scale:.1f} far from expected ~1000")

        lag = float(d["lag_seconds"])
        if abs(lag) > 3.0:
            issues.append(f"lag={lag:.2f}s at/beyond search window edge")

        mean_res = float(d["mean_residual_mm"])
        if mean_res > 200:
            issues.append(f"mean_residual={mean_res:.0f}mm is high")

        best_view_mpjpe = per_view_best.get(trial)
        if best_view_mpjpe is not None and mean_res > best_view_mpjpe + 100.0:
            # Fusion beating neither view isn't a hard bug (Phase 7 studies
            # this properly), but losing to the best single view by this
            # much is more than ordinary variance -- e.g. a temporal-sync
            # cycle-slip specific to the fused signal would show up here.
            issues.append(f"fused mean_residual ({mean_res:.0f}mm) is >100mm worse than "
                          f"the best single view ({best_view_mpjpe:.0f}mm) -- worth a look")

        per_joint = {}
        for j in CANONICAL_JOINTS:
            res = np.linalg.norm(d[f"aligned__{j}"] - d[f"mocap__{j}"], axis=-1)
            per_joint[j] = float(np.nanmean(res))
        worst_j = max(per_joint, key=per_joint.get)
        worst_joint_overall.setdefault(worst_j, 0)
        worst_joint_overall[worst_j] += 1

        vs_best = f" (best view={best_view_mpjpe:.0f}mm)" if best_view_mpjpe is not None else ""
        status = "OK" if not issues else "CHECK: " + "; ".join(issues)
        print(f"  {trial:20s} mean_residual={mean_res:6.1f}mm{vs_best}  "
              f"worst_joint={worst_j:15s}({per_joint[worst_j]:5.0f}mm)  {status}")

    print()
    print("  Joints that were the worst-tracked joint, count across all 10 fused trials:")
    for j, count in sorted(worst_joint_overall.items(), key=lambda kv: -kv[1]):
        print(f"    {j:15s} {count}")

    print()
    print("  Phase 6 metrics (MPJPE/PA-MPJPE/joint angles, all 10 fused trials):")
    metrics_path = METRICS_DIR / "phase6_metrics.json"
    if not metrics_path.exists():
        print(f"    CHECK: {metrics_path} not found -- run src/compare_metrics_fused.py first")
        return

    summary = json.loads(metrics_path.read_text())
    results = summary["per_trial"]
    if len(results) != len(PAIRED_TRIALS):
        print(f"    CHECK: {len(results)} fused trial results present, expected {len(PAIRED_TRIALS)}")

    for r in results:
        issues = _metrics_result_issues(r)
        status = "OK" if not issues else "CHECK: " + "; ".join(issues)
        print(f"    {r['trial']:20s} MPJPE={r['mpjpe_overall_mm']:6.1f}mm  "
              f"PA-MPJPE={r['pa_mpjpe_mm']:6.1f}mm  {status}")


def audit_left_right_consistency():
    print()
    print("=" * 70)
    print("Cross-check: left/right joint labeling consistency (mocap vs SAM3D)")
    print("=" * 70)
    # If SAM3D's left/right were swapped relative to mocap's, left-side joints
    # would systematically align better to mocap's RIGHT side than their own
    # left, after accounting for the fitted transform. Check by comparing
    # residual if aligned-left is matched to mocap-left vs mocap-right.
    for trial in PAIRED_TRIALS[:3] + ["walking_1", "running_1"]:
        d = np.load(ALIGNED_DIR / f"{trial}__view1.npz")
        for pair in ["hip", "knee", "ankle", "wrist"]:
            correct = np.linalg.norm(d[f"aligned__{pair}_left"] - d[f"mocap__{pair}_left"], axis=-1)
            swapped = np.linalg.norm(d[f"aligned__{pair}_left"] - d[f"mocap__{pair}_right"], axis=-1)
            flag = "  <-- SWAP SUSPECTED" if np.nanmean(swapped) < np.nanmean(correct) else ""
            print(f"  {trial:15s} {pair:6s}: correct_match={np.nanmean(correct):6.1f}mm  "
                  f"cross_match={np.nanmean(swapped):6.1f}mm{flag}")


if __name__ == "__main__":
    audit_phase1()
    audit_phase2()
    audit_phase3()
    audit_phase4()
    audit_phase5()
    audit_phase6()
    audit_left_right_consistency()

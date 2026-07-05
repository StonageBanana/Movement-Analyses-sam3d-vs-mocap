"""Phase 7: capstone synthesis -- view1 vs. view2 vs. fused, across every
metric/joint/category already computed in Phase 4 and Phase 6. Reads
output/metrics/phase4_metrics.json and phase6_metrics.json only (plus
output/fused/*.npz for the confidence-weighting premise test below) -- no
new model inference or alignment, just aggregation/comparison of results
already produced and audited.

Answers the project's research questions:
1. Single view vs. mocap: see Phase 4's own MPJPE/PA-MPJPE/angle numbers
   (this script pulls them into one place for context).
2. Slow/controlled vs. fast/dynamic actions: `by_category` breakdown.
3. Which joints are most/least accurate, and where does fusion help most:
   per-joint MPJPE table + per-joint improvement ranking.
4. Are single-view joint angles within a clinically-acceptable Bland-Altman
   range: `verdict()` compares against literature thresholds.
5. Does camera viewing angle alone matter: already visualized in Phase 4/
   09_synthesis_preview; summarized numerically here too.
6. Does fusion measurably beat the best single view: headline stats below.
7. Does inter-view disagreement predict true (mocap-relative) error --
   i.e. is Phase 5's disagreement signal a valid basis for confidence-
   weighted fusion (not built, since a real weighting scheme needs this
   premise validated first): `confidence_weighting_premise_test()`.
8. Where fusion helps most: cross-referenced with the per-joint/category
   improvement tables.
9. Capstone verdict: `verdict()`.
"""

import json
import sys
from pathlib import Path

import numpy as np
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent))
from joint_mapping import CANONICAL_JOINTS
from compare_metrics import ANGLE_NAMES

ANALYSIS_DIR = Path(__file__).resolve().parent.parent
METRICS_DIR = ANALYSIS_DIR / "output" / "metrics"
FUSED_DIR = ANALYSIS_DIR / "output" / "fused"
OUTPUT_DIR = ANALYSIS_DIR / "output" / "synthesis"

# Rough action-tempo grouping for the "slow/controlled vs fast/dynamic"
# research question -- not a rigid classification, just a lens for
# discussion (walking/squats are self-paced and comparatively controlled;
# running/dance/random/feet_movements involve faster or less repetitive
# motion that's harder for a single-image-per-frame model to track).
TEMPO_GROUPS = {
    "walking": "slow/controlled", "squats": "slow/controlled",
    "running": "fast/dynamic", "dance_move": "fast/dynamic",
    "random": "fast/dynamic", "feet_movements": "fast/dynamic",
}


def load_metrics():
    phase4 = json.loads((METRICS_DIR / "phase4_metrics.json").read_text())
    phase6 = json.loads((METRICS_DIR / "phase6_metrics.json").read_text())
    return phase4, phase6


def per_trial_comparison(phase4: dict, phase6: dict) -> list:
    by_trial = {}
    for r in phase4["per_trial_view"]:
        by_trial.setdefault(r["trial"], {})[r["view"]] = r
    for r in phase6["per_trial"]:
        by_trial.setdefault(r["trial"], {})["fused"] = r

    rows = []
    for trial, views in by_trial.items():
        if "fused" not in views:
            continue
        v1, v2, f = views["view1"], views["view2"], views["fused"]
        best_mpjpe = min(v1["mpjpe_overall_mm"], v2["mpjpe_overall_mm"])
        best_pa = min(v1["pa_mpjpe_mm"], v2["pa_mpjpe_mm"])
        rows.append({
            "trial": trial,
            "category": v1["category"],
            "view1_mpjpe": v1["mpjpe_overall_mm"], "view2_mpjpe": v2["mpjpe_overall_mm"],
            "best_view_mpjpe": best_mpjpe, "fused_mpjpe": f["mpjpe_overall_mm"],
            "fused_vs_best_pct": 100 * (best_mpjpe - f["mpjpe_overall_mm"]) / best_mpjpe,
            "view1_pa": v1["pa_mpjpe_mm"], "view2_pa": v2["pa_mpjpe_mm"],
            "best_view_pa": best_pa, "fused_pa": f["pa_mpjpe_mm"],
            "fused_vs_best_pa_pct": 100 * (best_pa - f["pa_mpjpe_mm"]) / best_pa,
        })
    return rows


def per_joint_comparison(phase4: dict, phase6: dict) -> list:
    view_joint = phase4["overall"]["overall"]["mpjpe_per_joint_mm"]
    fused_joint = phase6["overall"]["overall"]["mpjpe_per_joint_mm"]
    rows = []
    for j in CANONICAL_JOINTS:
        v, f = view_joint[j], fused_joint[j]
        rows.append({
            "joint": j, "view_mpjpe": v, "fused_mpjpe": f,
            "improvement_pct": 100 * (v - f) / v,
        })
    return sorted(rows, key=lambda r: -r["improvement_pct"])


def angle_comparison(phase4: dict, phase6: dict) -> list:
    view_angles = phase4["overall"]["overall"]["angles"]
    fused_angles = phase6["overall"]["overall"]["angles"]
    rows = []
    for name in ANGLE_NAMES:
        v, f = view_angles[name], fused_angles[name]
        if v["n_trials"] == 0 or f["n_trials"] == 0:
            rows.append({"angle": name, "note": "insufficient reliable trials on one or both sides"})
            continue
        rows.append({
            "angle": name,
            "view_rmse_deg": v["rmse_deg"], "view_bias_deg": v["bland_altman_bias_deg"],
            "fused_rmse_deg": f["rmse_deg"], "fused_bias_deg": f["bland_altman_bias_deg"],
            "view_n": v["n_trials"], "fused_n": f["n_trials"],
        })
    return rows


def confidence_weighting_premise_test(phase6: dict) -> dict:
    """Does Phase 5's inter-view disagreement (computed with zero mocap
    involvement) predict the fused joint's *actual* error against mocap
    (Phase 6)? If yes, that's real evidence a confidence-weighted fusion
    (down-weighting high-disagreement joints) would plausibly help in a
    true deployment (no ground truth available to check against directly).
    If no, weighting by disagreement wouldn't be worth building. Tested at
    the per-joint, per-trial level (mean disagreement vs. mean mocap error
    for that joint across the whole trial) -- 10 trials x 19 joints = 190
    pairs -- rather than per-frame, since Phase 5's disagreement and Phase
    6's error live on different timelines (view1's own clock vs. mocap's
    resampled clock) and per-trial-per-joint means avoid needing to
    re-resample either."""
    error_by_trial = {r["trial"]: r["mpjpe_per_joint_mm"] for r in phase6["per_trial"]}

    disagreement, error = [], []
    for trial, per_joint_error in error_by_trial.items():
        fused_path = FUSED_DIR / f"{trial}.npz"
        if not fused_path.exists():
            continue
        d = np.load(fused_path)
        for j in CANONICAL_JOINTS:
            disagreement.append(float(np.nanmean(d[f"disagreement__{j}"])))
            error.append(per_joint_error[j])

    disagreement, error = np.array(disagreement), np.array(error)
    r, p = stats.pearsonr(disagreement, error)
    return {
        "n_pairs": len(disagreement),
        "pearson_r": float(r),
        "p_value": float(p),
        "disagreement": disagreement.tolist(),
        "error_mm": error.tolist(),
    }


def headline_stats(trial_rows: list) -> dict:
    n = len(trial_rows)
    n_fusion_wins = sum(1 for r in trial_rows if r["fused_mpjpe"] < r["best_view_mpjpe"])
    n_fusion_wins_pa = sum(1 for r in trial_rows if r["fused_pa"] < r["best_view_pa"])
    return {
        "n_trials": n,
        "n_trials_fused_beats_best_view_mpjpe": n_fusion_wins,
        "n_trials_fused_beats_best_view_pa_mpjpe": n_fusion_wins_pa,
        "mean_fused_vs_best_view_mpjpe_pct": float(np.mean([r["fused_vs_best_pct"] for r in trial_rows])),
        "mean_fused_vs_best_view_pa_mpjpe_pct": float(np.mean([r["fused_vs_best_pa_pct"] for r in trial_rows])),
        "mean_best_view_mpjpe_mm": float(np.mean([r["best_view_mpjpe"] for r in trial_rows])),
        "mean_fused_mpjpe_mm": float(np.mean([r["fused_mpjpe"] for r in trial_rows])),
        "mean_best_view_pa_mpjpe_mm": float(np.mean([r["best_view_pa"] for r in trial_rows])),
        "mean_fused_pa_mpjpe_mm": float(np.mean([r["fused_pa"] for r in trial_rows])),
    }


def tempo_breakdown(trial_rows: list) -> dict:
    groups = {}
    for r in trial_rows:
        tempo = TEMPO_GROUPS.get(r["category"], "unclassified")
        groups.setdefault(tempo, []).append(r)
    return {
        tempo: {
            "n_trials": len(rows),
            "mean_best_view_mpjpe_mm": float(np.mean([r["best_view_mpjpe"] for r in rows])),
            "mean_fused_mpjpe_mm": float(np.mean([r["fused_mpjpe"] for r in rows])),
            "mean_fused_vs_best_view_pct": float(np.mean([r["fused_vs_best_pct"] for r in rows])),
        }
        for tempo, rows in groups.items()
    }


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    phase4, phase6 = load_metrics()

    trial_rows = per_trial_comparison(phase4, phase6)
    joint_rows = per_joint_comparison(phase4, phase6)
    angle_rows = angle_comparison(phase4, phase6)
    headline = headline_stats(trial_rows)
    tempo = tempo_breakdown(trial_rows)
    confidence_test = confidence_weighting_premise_test(phase6)

    summary = {
        "per_trial": trial_rows,
        "per_joint": joint_rows,
        "per_angle": angle_rows,
        "headline": headline,
        "by_tempo": tempo,
        "confidence_weighting_premise_test": confidence_test,
    }
    out_path = OUTPUT_DIR / "phase7_synthesis.json"
    out_path.write_text(json.dumps(summary, indent=2, default=str))

    print("=== Per-trial: view1 vs view2 vs fused (MPJPE / PA-MPJPE, mm) ===")
    for r in trial_rows:
        print(f"  {r['trial']:20s} v1={r['view1_mpjpe']:6.1f} v2={r['view2_mpjpe']:6.1f} "
              f"best={r['best_view_mpjpe']:6.1f} fused={r['fused_mpjpe']:6.1f} "
              f"({r['fused_vs_best_pct']:+5.1f}%)   "
              f"PA: best={r['best_view_pa']:6.1f} fused={r['fused_pa']:6.1f} ({r['fused_vs_best_pa_pct']:+5.1f}%)")

    print(f"\n=== Headline ===")
    print(f"  Fusion beats best single view (MPJPE): {headline['n_trials_fused_beats_best_view_mpjpe']}/{headline['n_trials']} trials")
    print(f"  Fusion beats best single view (PA-MPJPE): {headline['n_trials_fused_beats_best_view_pa_mpjpe']}/{headline['n_trials']} trials")
    print(f"  Mean MPJPE change vs best view: {headline['mean_fused_vs_best_view_mpjpe_pct']:+.1f}%")
    print(f"  Mean PA-MPJPE change vs best view: {headline['mean_fused_vs_best_view_pa_mpjpe_pct']:+.1f}%")
    print(f"  Mean best-single-view MPJPE: {headline['mean_best_view_mpjpe_mm']:.1f}mm  ->  Mean fused MPJPE: {headline['mean_fused_mpjpe_mm']:.1f}mm")

    print(f"\n=== By action tempo ===")
    for tempo_name, m in tempo.items():
        print(f"  {tempo_name:16s} best_view={m['mean_best_view_mpjpe_mm']:6.1f}mm  "
              f"fused={m['mean_fused_mpjpe_mm']:6.1f}mm  ({m['mean_fused_vs_best_view_pct']:+5.1f}%)  n={m['n_trials']}")

    print(f"\n=== Per-joint MPJPE: single view (avg of 20 trial/views) vs fused (avg of 10 trials) ===")
    for r in joint_rows:
        print(f"  {r['joint']:15s} view={r['view_mpjpe']:6.1f}mm  fused={r['fused_mpjpe']:6.1f}mm  "
              f"({r['improvement_pct']:+5.1f}%)")

    print(f"\n=== Joint angles: single view vs fused (RMSE / bias, deg) ===")
    for r in angle_rows:
        if "note" in r:
            print(f"  {r['angle']:22s} -- {r['note']}")
        else:
            print(f"  {r['angle']:22s} view: RMSE={r['view_rmse_deg']:5.1f} bias={r['view_bias_deg']:+6.1f}  "
                  f"fused: RMSE={r['fused_rmse_deg']:5.1f} bias={r['fused_bias_deg']:+6.1f}")

    print(f"\n=== Confidence-weighting premise test ===")
    c = confidence_test
    print(f"  Inter-view disagreement (Phase 5) vs actual mocap error (Phase 6),")
    print(f"  {c['n_pairs']} (trial, joint) pairs: Pearson r={c['pearson_r']:.3f} (p={c['p_value']:.2e})")

    print(f"\nSaved -> {out_path}")


if __name__ == "__main__":
    main()

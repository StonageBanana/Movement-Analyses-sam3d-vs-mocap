"""Phase 8: final, presentation-ready visualization + reporting pass, as
distinct from the mid-session output/overview/ checkpoint plots. Produces:

- Joint-angle time-series overlays (mocap vs view1 vs view2 vs fused) for
  a representative trial per action tempo.
- Pooled Bland-Altman plots per joint angle, one per condition (view1,
  view2, fused), aggregating every non-unreliable trial's frames -- the
  standard population-level Bland-Altman presentation, not a per-trial one.
- A consolidated MPJPE summary (per-category, per-joint).
- A skeleton "filmstrip" across one full squat cycle (mocap + fused) as a
  static stand-in for the plan's optional 3D animation.
- report_data.json: every number behind every figure, in one place.

Reads only already-computed output/aligned, output/aligned_fused,
output/metrics, output/mocap, and output/synthesis files -- no new
alignment or model inference.
"""

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from joint_mapping import CANONICAL_JOINTS
from mocap.angles import compute_joint_angles_from_joints, wrap_around_center
from compare_metrics import ANGLE_NAMES, MOCAP_UP, load_static_angle_offsets

ANALYSIS_DIR = Path(__file__).resolve().parent.parent
MOCAP_DIR = ANALYSIS_DIR / "output" / "mocap"
ALIGNED_DIR = ANALYSIS_DIR / "output" / "aligned"
ALIGNED_FUSED_DIR = ANALYSIS_DIR / "output" / "aligned_fused"
METRICS_DIR = ANALYSIS_DIR / "output" / "metrics"
SYNTHESIS_PATH = ANALYSIS_DIR / "output" / "synthesis" / "phase7_synthesis.json"
OUT_DIR = ANALYSIS_DIR / "output" / "final"

CONDITIONS = ["view1", "view2", "fused"]
CONDITION_COLORS = {"view1": "steelblue", "view2": "indianred", "fused": "darkorange"}

CANON_LINKS = [
    ("neck", "head"),
    ("neck", "shoulder_left"), ("neck", "shoulder_right"),
    ("shoulder_left", "elbow_left"), ("elbow_left", "wrist_left"),
    ("shoulder_right", "elbow_right"), ("elbow_right", "wrist_right"),
    ("neck", "pelvis"),
    ("pelvis", "hip_left"), ("pelvis", "hip_right"),
    ("hip_left", "knee_left"), ("knee_left", "ankle_left"),
    ("hip_right", "knee_right"), ("knee_right", "ankle_right"),
    ("ankle_left", "heel_left"), ("ankle_left", "toe_left"),
    ("ankle_right", "heel_right"), ("ankle_right", "toe_right"),
]

TRIALS = [t.stem for t in ALIGNED_FUSED_DIR.glob("*.npz")]


def _load_condition(trial: str, condition: str):
    path = (ALIGNED_DIR / f"{trial}__{condition}.npz") if condition != "fused" else (ALIGNED_FUSED_DIR / f"{trial}.npz")
    return np.load(path)


def angle_series(trial: str, condition: str, static_offsets: dict) -> dict:
    """Returns {angle_name: (time, mocap_angle_deg, est_angle_deg, unreliable)}."""
    d = _load_condition(trial, condition)
    mocap = {j: d[f"mocap__{j}"] for j in CANONICAL_JOINTS}
    est = {j: d[f"aligned__{j}"] for j in CANONICAL_JOINTS}
    mocap_angles = compute_joint_angles_from_joints(mocap, MOCAP_UP)
    est_angles = compute_joint_angles_from_joints(est, MOCAP_UP)
    mocap_unreliable = set(mocap_angles.pop("_unreliable"))
    est_unreliable = set(est_angles.pop("_unreliable"))
    time = d["mocap_time_overlap"]

    out = {}
    for name in ANGLE_NAMES:
        offset = static_offsets[name]
        out[name] = (
            time,
            mocap_angles[name] - offset,
            est_angles[name] - offset,
            name in mocap_unreliable or name in est_unreliable,
        )
    return out


def fig_angle_timeseries(trial: str, static_offsets: dict, angle_names=("hip_flexion_left", "knee_flexion_left")):
    fig, axes = plt.subplots(len(angle_names), 1, figsize=(12, 3.2 * len(angle_names)), sharex=True)
    axes = np.atleast_1d(axes)
    for ax, name in zip(axes, angle_names):
        mocap_plotted = False
        for cond in CONDITIONS:
            t, mocap_a, est_a, unreliable = angle_series(trial, cond, static_offsets)[name]
            if not mocap_plotted:
                ax.plot(t, mocap_a, "k-", linewidth=2, label="mocap (ground truth)")
                mocap_plotted = True
            flag = " [unreliable]" if unreliable else ""
            style = "-" if cond == "fused" else "--"
            ax.plot(t, est_a, style, color=CONDITION_COLORS[cond], linewidth=1.6, label=f"{cond}{flag}")
        ax.set_ylabel(f"{name}\n(deg)")
        ax.legend(fontsize=8, ncol=4, loc="upper right")
    axes[-1].set_xlabel("time (s, mocap clock)")
    fig.suptitle(f"Phase 8: joint angle time series -- {trial}")
    plt.tight_layout()
    out = OUT_DIR / f"angle_timeseries_{trial}.png"
    plt.savefig(out, dpi=120)
    plt.close(fig)
    print(f"Saved {out}")


def pooled_bland_altman(angle_name: str, condition: str, static_offsets: dict, trials: list):
    means, diffs, n_trials_used = [], [], 0
    for trial in trials:
        try:
            t, mocap_a, est_a, unreliable = angle_series(trial, condition, static_offsets)[angle_name]
        except FileNotFoundError:
            continue
        if unreliable:
            continue
        diff = wrap_around_center(est_a - mocap_a, 0.0)
        valid = ~np.isnan(diff)
        if valid.sum() < 2:
            continue
        m = mocap_a[valid]
        d = diff[valid]
        s_rewrapped = m + d  # est re-expressed on mocap's own numeric branch (see compare_metrics.angle_metrics)
        means.append(0.5 * (m + s_rewrapped))
        diffs.append(d)
        n_trials_used += 1
    if not means:
        return None
    return np.concatenate(means), np.concatenate(diffs), n_trials_used


def fig_bland_altman(angle_name: str, static_offsets: dict, trials: list) -> dict:
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), sharey=True)
    result_summary = {}
    for ax, cond in zip(axes, CONDITIONS):
        result = pooled_bland_altman(angle_name, cond, static_offsets, trials)
        if result is None:
            ax.text(0.5, 0.5, "insufficient reliable data", ha="center", transform=ax.transAxes)
            ax.set_title(cond)
            result_summary[cond] = None
            continue
        mean_arr, diff_arr, n_trials_used = result
        bias, sd = float(diff_arr.mean()), float(diff_arr.std())
        loa_lo, loa_hi = bias - 1.96 * sd, bias + 1.96 * sd

        ax.scatter(mean_arr, diff_arr, s=4, alpha=0.12, color=CONDITION_COLORS[cond])
        ax.axhline(bias, color="red", linewidth=1.2, label=f"bias={bias:+.1f} deg")
        ax.axhline(loa_lo, color="gray", linestyle="--", linewidth=1)
        ax.axhline(loa_hi, color="gray", linestyle="--", linewidth=1, label=f"95% LoA [{loa_lo:.1f}, {loa_hi:.1f}]")
        ax.set_title(f"{cond} (n={n_trials_used} trials, {len(diff_arr)} frames)")
        ax.set_xlabel("mean of mocap & estimate (deg)")
        ax.legend(fontsize=7, loc="upper right")
        result_summary[cond] = {
            "bias_deg": bias, "sd_deg": sd, "loa_lower_deg": loa_lo, "loa_upper_deg": loa_hi,
            "n_trials": n_trials_used, "n_frames": int(len(diff_arr)),
        }
    axes[0].set_ylabel("estimate - mocap (deg)")
    fig.suptitle(f"Phase 8: Bland-Altman -- {angle_name} (pooled across all reliable trials)")
    plt.tight_layout()
    out = OUT_DIR / f"bland_altman_{angle_name}.png"
    plt.savefig(out, dpi=120)
    plt.close(fig)
    print(f"Saved {out}")
    return result_summary


def fig_mpjpe_summary(phase4: dict, phase6: dict):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    categories = list(phase4["by_category"].keys())
    x = np.arange(len(categories))
    width = 0.25
    for i, cond in enumerate(("view1", "view2")):
        vals = [phase4["by_category_view"][f"{c}__{cond}"]["mpjpe_overall_mm"] for c in categories]
        ax1.bar(x + (i - 1) * width, vals, width, label=cond, color=CONDITION_COLORS[cond])
    fused_vals = [phase6["by_category"][c]["mpjpe_overall_mm"] for c in categories]
    ax1.bar(x + width, fused_vals, width, label="fused", color=CONDITION_COLORS["fused"])
    ax1.set_xticks(x)
    ax1.set_xticklabels(categories, rotation=30, ha="right")
    ax1.set_ylabel("MPJPE (mm)")
    ax1.set_title("MPJPE by action category")
    ax1.legend()

    view_joint = phase4["overall"]["overall"]["mpjpe_per_joint_mm"]
    fused_joint = phase6["overall"]["overall"]["mpjpe_per_joint_mm"]
    joints = sorted(CANONICAL_JOINTS, key=lambda j: view_joint[j])
    y = np.arange(len(joints))
    ax2.barh(y - 0.2, [view_joint[j] for j in joints], 0.4, label="single view (avg)", color="steelblue")
    ax2.barh(y + 0.2, [fused_joint[j] for j in joints], 0.4, label="fused", color=CONDITION_COLORS["fused"])
    ax2.set_yticks(y)
    ax2.set_yticklabels(joints, fontsize=8)
    ax2.set_xlabel("MPJPE (mm)")
    ax2.set_title("MPJPE by joint")
    ax2.legend()

    fig.suptitle("Phase 8: MPJPE summary -- single view vs fused")
    plt.tight_layout()
    out = OUT_DIR / "mpjpe_summary.png"
    plt.savefig(out, dpi=120)
    plt.close(fig)
    print(f"Saved {out}")


def _skeleton(ax, joints: dict, frame_idx: int, color: str, alpha: float = 1.0):
    for a, b in CANON_LINKS:
        pa, pb = joints[a][frame_idx], joints[b][frame_idx]
        if np.isnan(pa).any() or np.isnan(pb).any():
            continue
        ax.plot([pa[0], pb[0]], [pa[1], pb[1]], "-", color=color, linewidth=1.5, alpha=alpha)


def fig_skeleton_filmstrip(trial: str = "squats_1", n_frames: int = 5, cycle_window=(1.5, 4.6)):
    """cycle_window: (start, end) seconds on mocap's clock spanning one full
    standing->deep-squat->standing cycle -- for squats_1 specifically,
    confirmed against its own pelvis-height trajectory (standing ~1035mm at
    t=1.5s, deepest ~598mm at t=3.6s, back to standing ~1040mm by t=4.6s)."""
    df = np.load(ALIGNED_FUSED_DIR / f"{trial}.npz")
    fused = {j: df[f"aligned__{j}"] for j in CANONICAL_JOINTS}
    mocap = {j: df[f"mocap__{j}"] for j in CANONICAL_JOINTS}
    t_overlap = df["mocap_time_overlap"]
    sample_times = np.linspace(cycle_window[0], cycle_window[1], n_frames)

    fig, axes = plt.subplots(1, n_frames, figsize=(3.2 * n_frames, 5.5), sharey=True)
    for ax, target_t in zip(axes, sample_times):
        idx = int(np.argmin(np.abs(t_overlap - target_t)))
        _skeleton(ax, mocap, idx, "black")
        _skeleton(ax, fused, idx, CONDITION_COLORS["fused"])
        ax.set_aspect("equal")
        ax.invert_yaxis()
        ax.set_title(f"t={t_overlap[idx]:.2f}s")
        ax.set_xticks([])
    axes[0].set_ylabel("Y / up (mm)")
    fig.legend(
        handles=[plt.Line2D([], [], color="black", label="mocap"),
                 plt.Line2D([], [], color=CONDITION_COLORS["fused"], label="fused")],
        loc="upper center", ncol=2, fontsize=10, bbox_to_anchor=(0.5, 1.06),
    )
    fig.suptitle(f"Phase 8: {trial} -- fused vs mocap across one squat cycle", y=1.15)
    plt.tight_layout()
    out = OUT_DIR / f"skeleton_filmstrip_{trial}.png"
    plt.savefig(out, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    static_offsets = load_static_angle_offsets()
    phase4 = json.loads((METRICS_DIR / "phase4_metrics.json").read_text())
    phase6 = json.loads((METRICS_DIR / "phase6_metrics.json").read_text())
    synthesis = json.loads(SYNTHESIS_PATH.read_text())

    representative_trials = ["squats_1", "walking_1", "running_1"]
    for trial in representative_trials:
        fig_angle_timeseries(trial, static_offsets, ("hip_flexion_left", "knee_flexion_left", "hip_flexion_right"))

    bland_altman_summary = {}
    for name in ANGLE_NAMES:
        bland_altman_summary[name] = fig_bland_altman(name, static_offsets, TRIALS)

    fig_mpjpe_summary(phase4, phase6)
    fig_skeleton_filmstrip("squats_1")

    report_data = {
        "representative_trials": representative_trials,
        "bland_altman": bland_altman_summary,
        "mpjpe_overall": {
            "view_avg": phase4["overall"]["overall"]["mpjpe_overall_mm"],
            "fused": phase6["overall"]["overall"]["mpjpe_overall_mm"],
        },
        "pa_mpjpe_overall": {
            "view_avg": phase4["overall"]["overall"]["pa_mpjpe_mm"],
            "fused": phase6["overall"]["overall"]["pa_mpjpe_mm"],
        },
        "mpjpe_by_category": {
            c: {
                "view1": phase4["by_category_view"][f"{c}__view1"]["mpjpe_overall_mm"],
                "view2": phase4["by_category_view"][f"{c}__view2"]["mpjpe_overall_mm"],
                "fused": phase6["by_category"][c]["mpjpe_overall_mm"],
            }
            for c in phase4["by_category"]
        },
        "mpjpe_by_joint": {
            j: {"view_avg": phase4["overall"]["overall"]["mpjpe_per_joint_mm"][j],
                "fused": phase6["overall"]["overall"]["mpjpe_per_joint_mm"][j]}
            for j in CANONICAL_JOINTS
        },
        "phase7_headline": synthesis["headline"],
        "phase7_confidence_weighting_premise_test": {
            k: v for k, v in synthesis["confidence_weighting_premise_test"].items()
            if k not in ("disagreement", "error_mm")
        },
    }
    out_path = OUT_DIR / "report_data.json"
    out_path.write_text(json.dumps(report_data, indent=2, default=str))
    print(f"\nSaved -> {out_path}")


if __name__ == "__main__":
    main()

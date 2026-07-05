"""Checkpoint visualizations for Phase 4 (independent per-view metrics),
Phase 5 (cross-view fusion), and Phase 6 (fused-vs-mocap). Companion to
visualize_overview.py (which covers Phase 1-3). Produces PNGs in
output/overview/. No new computation -- everything is read back from
already-saved output/metrics, output/fused, and output/aligned_fused .npz
files.
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

ANALYSIS_DIR = Path(__file__).resolve().parent.parent
MANIFEST_PATH = ANALYSIS_DIR / "manifest.json"
MOCAP_DIR = ANALYSIS_DIR / "output" / "mocap"
ALIGNED_DIR = ANALYSIS_DIR / "output" / "aligned"
FUSED_DIR = ANALYSIS_DIR / "output" / "fused"
ALIGNED_FUSED_DIR = ANALYSIS_DIR / "output" / "aligned_fused"
METRICS_DIR = ANALYSIS_DIR / "output" / "metrics"
OUT_DIR = ANALYSIS_DIR / "output" / "overview"

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


def fig_phase4_view_comparison():
    """Grouped bar chart: MPJPE and PA-MPJPE for view1 vs view2, per trial --
    answers "does viewing angle alone change accuracy?" at a glance."""
    summary = json.loads((METRICS_DIR / "phase4_metrics.json").read_text())
    by_trial = {}
    for r in summary["per_trial_view"]:
        by_trial.setdefault(r["trial"], {})[r["view"]] = r
    trials = list(by_trial.keys())

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
    x = np.arange(len(trials))
    width = 0.35
    for ax, metric, title in ((ax1, "mpjpe_overall_mm", "MPJPE"), (ax2, "pa_mpjpe_mm", "PA-MPJPE")):
        v1 = [by_trial[t]["view1"][metric] for t in trials]
        v2 = [by_trial[t]["view2"][metric] for t in trials]
        ax.bar(x - width / 2, v1, width, label="view1", color="steelblue")
        ax.bar(x + width / 2, v2, width, label="view2", color="indianred")
        ax.set_ylabel(f"{title} (mm)")
        ax.set_xticks(x)
        ax.set_xticklabels(trials, rotation=30, ha="right")
        ax.legend()
        ax.set_title(f"Phase 4: {title} per trial, view1 vs view2")
    plt.tight_layout()
    out = OUT_DIR / "05_phase4_view_comparison.png"
    plt.savefig(out, dpi=110)
    plt.close(fig)
    print(f"Saved {out}")


def _plot_skeleton(ax, joints: dict, frame_idx: int, color: str, label: str):
    for a, b in CANON_LINKS:
        pa, pb = joints[a][frame_idx], joints[b][frame_idx]
        if np.isnan(pa).any() or np.isnan(pb).any():
            continue
        ax.plot([pa[0], pb[0]], [pa[1], pb[1]], "-", color=color, linewidth=1.5)
    pts = np.stack([joints[j][frame_idx] for j in CANONICAL_JOINTS])
    valid = ~np.isnan(pts).any(axis=-1)
    ax.scatter(pts[valid, 0], pts[valid, 1], s=12, color=color, label=label, zorder=3)


def fig_phase5_skeleton_overlay(trial: str = "squats_1"):
    """view1 (aligned to mocap), view2 (aligned to mocap), mocap, and the
    fused skeleton (aligned to mocap), all overlaid on the deepest-squat
    frame -- per the Phase 5 plan's own verification criterion: the fused
    pose should sit visually "between" the two views, never wildly outside
    either."""
    mocap_full = np.load(MOCAP_DIR / f"{trial}.npz")
    target_time = mocap_full["time"][np.nanargmin(mocap_full["joint__pelvis"][:, 1])]

    d1 = np.load(ALIGNED_DIR / f"{trial}__view1.npz")
    d2 = np.load(ALIGNED_DIR / f"{trial}__view2.npz")
    df = np.load(ALIGNED_FUSED_DIR / f"{trial}.npz")

    fig, ax = plt.subplots(figsize=(7, 9))
    sources = [
        (d1, "aligned__", "steelblue", "view1 (aligned)"),
        (d2, "aligned__", "indianred", "view2 (aligned)"),
        (df, "aligned__", "darkorange", "fused (aligned)"),
        (d1, "mocap__", "black", "mocap (ground truth)"),
    ]
    for d, prefix, color, label in sources:
        idx = int(np.argmin(np.abs(d["mocap_time_overlap"] - target_time)))
        joints = {j: d[f"{prefix}{j}"] for j in CANONICAL_JOINTS}
        _plot_skeleton(ax, joints, idx, color, label)

    ax.set_xlabel("X (mm)")
    ax.set_ylabel("Y / up (mm)")
    ax.set_aspect("equal")
    ax.invert_yaxis()
    ax.legend(loc="upper right", fontsize=9)
    ax.set_title(f"{trial}: view1 vs view2 vs fused vs mocap, deepest-squat frame\n"
                 f"(t~{target_time:.2f}s, mocap clock)")
    plt.tight_layout()
    out = OUT_DIR / f"06_phase5_skeleton_overlay_{trial}.png"
    plt.savefig(out, dpi=110)
    plt.close(fig)
    print(f"Saved {out}")


def fig_phase5_disagreement_timeseries(trial: str = "squats_1"):
    """Per-joint inter-view disagreement (post-GPA-alignment ||view1-view2||)
    over the trial, for a handful of representative joints -- shows where/
    when the two views agree least (occlusion, fast motion, etc.)."""
    d = np.load(FUSED_DIR / f"{trial}.npz")
    t = d["time"]
    joints_to_show = ["hip_right", "knee_right", "ankle_right", "wrist_left", "toe_right"]

    fig, ax = plt.subplots(figsize=(12, 5))
    for j in joints_to_show:
        ax.plot(t, d[f"disagreement__{j}"], label=j)
    ax.set_xlabel("time (s, view1 clock)")
    ax.set_ylabel("inter-view disagreement (model units, ~m)")
    ax.set_title(f"Phase 5: per-joint inter-view disagreement over time -- {trial}")
    ax.legend(fontsize=9)
    plt.tight_layout()
    out = OUT_DIR / f"07_phase5_disagreement_{trial}.png"
    plt.savefig(out, dpi=110)
    plt.close(fig)
    print(f"Saved {out}")


def fig_phase6_trajectory(trial: str = "squats_1", joint: str = "pelvis"):
    """mocap vs view1 vs view2 vs fused for one joint's vertical trajectory
    -- the fused line should track mocap at least as well as the better of
    the two view lines."""
    d1 = np.load(ALIGNED_DIR / f"{trial}__view1.npz")
    d2 = np.load(ALIGNED_DIR / f"{trial}__view2.npz")
    df = np.load(ALIGNED_FUSED_DIR / f"{trial}.npz")

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(d1["mocap_time_overlap"], d1[f"mocap__{joint}"][:, 1], "k-", linewidth=2, label="mocap (ground truth)")
    ax.plot(d1["mocap_time_overlap"], d1[f"aligned__{joint}"][:, 1], "--", color="steelblue", label="view1 (aligned)")
    ax.plot(d2["mocap_time_overlap"], d2[f"aligned__{joint}"][:, 1], "--", color="indianred", label="view2 (aligned)")
    ax.plot(df["mocap_time_overlap"], df[f"aligned__{joint}"][:, 1], "-", color="darkorange", linewidth=2,
            label="fused (aligned)")
    ax.set_xlabel("time (s, mocap clock)")
    ax.set_ylabel(f"{joint} vertical (mm)")
    ax.legend(fontsize=9)
    ax.set_title(f"Phase 6: {trial} -- mocap vs view1 vs view2 vs fused")
    plt.tight_layout()
    out = OUT_DIR / f"08_phase6_trajectory_{trial}.png"
    plt.savefig(out, dpi=110)
    plt.close(fig)
    print(f"Saved {out}")


def fig_synthesis_preview():
    """The core "does fusion help" chart -- view1 vs view2 vs fused MPJPE/
    PA-MPJPE per trial. A preview of Phase 7's actual analysis, built
    directly from the Phase 4 and Phase 6 metrics already computed."""
    phase4 = json.loads((METRICS_DIR / "phase4_metrics.json").read_text())
    phase6 = json.loads((METRICS_DIR / "phase6_metrics.json").read_text())

    by_trial = {}
    for r in phase4["per_trial_view"]:
        by_trial.setdefault(r["trial"], {})[r["view"]] = r
    for r in phase6["per_trial"]:
        by_trial.setdefault(r["trial"], {})["fused"] = r
    trials = [t for t in by_trial if "fused" in by_trial[t]]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(13, 9))
    x = np.arange(len(trials))
    width = 0.26
    for ax, metric, title in ((ax1, "mpjpe_overall_mm", "MPJPE"), (ax2, "pa_mpjpe_mm", "PA-MPJPE")):
        v1 = [by_trial[t]["view1"][metric] for t in trials]
        v2 = [by_trial[t]["view2"][metric] for t in trials]
        vf = [by_trial[t]["fused"][metric] for t in trials]
        ax.bar(x - width, v1, width, label="view1", color="steelblue")
        ax.bar(x, v2, width, label="view2", color="indianred")
        ax.bar(x + width, vf, width, label="fused", color="darkorange")
        ax.set_ylabel(f"{title} (mm)")
        ax.set_xticks(x)
        ax.set_xticklabels(trials, rotation=30, ha="right")
        ax.legend()
        ax.set_title(f"view1 vs view2 vs fused: {title} per trial")
    fig.suptitle("Phase 7 preview: does two-view fusion improve on the best single view?", y=1.0)
    plt.tight_layout()
    out = OUT_DIR / "09_synthesis_preview.png"
    plt.savefig(out, dpi=110)
    plt.close(fig)
    print(f"Saved {out}")


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig_phase4_view_comparison()
    fig_phase5_skeleton_overlay("squats_1")
    fig_phase5_disagreement_timeseries("squats_1")
    fig_phase6_trajectory("squats_1", "pelvis")
    fig_synthesis_preview()


if __name__ == "__main__":
    main()

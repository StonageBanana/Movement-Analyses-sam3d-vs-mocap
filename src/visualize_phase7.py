"""Checkpoint visualizations for Phase 7 (capstone synthesis). Produces
PNGs in output/overview/. No new computation -- reads back
output/synthesis/phase7_synthesis.json only.
"""

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ANALYSIS_DIR = Path(__file__).resolve().parent.parent
SYNTHESIS_PATH = ANALYSIS_DIR / "output" / "synthesis" / "phase7_synthesis.json"
OUT_DIR = ANALYSIS_DIR / "output" / "overview"


def fig_per_joint_improvement(summary: dict):
    rows = summary["per_joint"]
    joints = [r["joint"] for r in rows]
    pct = [r["improvement_pct"] for r in rows]
    colors = ["darkorange" if p >= 0 else "gray" for p in pct]

    fig, ax = plt.subplots(figsize=(10, 7))
    y = np.arange(len(joints))
    ax.barh(y, pct, color=colors)
    ax.set_yticks(y)
    ax.set_yticklabels(joints)
    ax.invert_yaxis()
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("MPJPE improvement from fusion vs. single view (%)")
    ax.set_title("Phase 7: where does fusion help most?\n"
                  "(distal joints improve most; hip_left/hip_right barely move)")
    plt.tight_layout()
    out = OUT_DIR / "10_phase7_joint_improvement.png"
    plt.savefig(out, dpi=110)
    plt.close(fig)
    print(f"Saved {out}")


def fig_confidence_premise(summary: dict):
    c = summary["confidence_weighting_premise_test"]
    dis = np.array(c["disagreement"])
    err = np.array(c["error_mm"])
    joints = [r["joint"] for r in summary["per_joint"]]
    n_trials = len(dis) // len(joints)
    dis = dis.reshape(n_trials, len(joints))
    err = err.reshape(n_trials, len(joints))

    fig, ax = plt.subplots(figsize=(8, 7))
    for i, j in enumerate(joints):
        marker = "*" if "hip" in j else "o"
        size = 90 if "hip" in j else 35
        ax.scatter(dis[:, i], err[:, i], label=j if "hip" in j else None, marker=marker, s=size, alpha=0.8)
    ax.set_xlabel("mean inter-view disagreement (Phase 5, model units)")
    ax.set_ylabel("mean actual error vs mocap (Phase 6, mm)")
    ax.set_title(f"Phase 7: does disagreement predict true error?\n"
                 f"r={c['pearson_r']:.2f} (p={c['p_value']:.2f}) across {c['n_pairs']} (trial,joint) pairs -- "
                 f"hips highlighted (low disagreement, hip_right high error)")
    ax.legend(fontsize=9)
    plt.tight_layout()
    out = OUT_DIR / "11_phase7_confidence_premise.png"
    plt.savefig(out, dpi=110)
    plt.close(fig)
    print(f"Saved {out}")


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    summary = json.loads(SYNTHESIS_PATH.read_text())
    fig_per_joint_improvement(summary)
    fig_confidence_premise(summary)


if __name__ == "__main__":
    main()

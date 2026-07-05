"""Checkpoint visualizations covering everything produced so far:
Phase 1 (mocap ground truth), Phase 2 (SAM 3D Body per-view reconstruction),
and Phase 3 (temporal/spatial alignment). Produces PNGs in output/overview/.
No new inference is run -- everything is read back from already-saved
output/mocap, output/sam3d, and output/aligned .npz files.
"""

import json
import sys
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from joint_mapping import CANONICAL_JOINTS

SAM3D_REPO = Path(__file__).resolve().parent.parent / "third_party" / "sam-3d-body"
sys.path.insert(0, str(SAM3D_REPO))
from sam_3d_body.metadata.mhr70 import mhr_names  # noqa: E402

ANALYSIS_DIR = Path(__file__).resolve().parent.parent
MANIFEST_PATH = ANALYSIS_DIR / "manifest.json"
MOCAP_DIR = ANALYSIS_DIR / "output" / "mocap"
SAM3D_DIR = ANALYSIS_DIR / "output" / "sam3d"
ALIGNED_DIR = ANALYSIS_DIR / "output" / "aligned"
OUT_DIR = ANALYSIS_DIR / "output" / "overview"

JOINT_IDX = {name: i for i, name in enumerate(mhr_names)}
SKELETON_LINKS = [
    ("neck", "left-shoulder"), ("neck", "right-shoulder"),
    ("left-shoulder", "left-elbow"), ("left-elbow", "left-wrist"),
    ("right-shoulder", "right-elbow"), ("right-elbow", "right-wrist"),
    ("left-hip", "right-hip"),
    ("neck", "left-hip"), ("neck", "right-hip"),
    ("left-hip", "left-knee"), ("left-knee", "left-ankle"),
    ("right-hip", "right-knee"), ("right-knee", "right-ankle"),
    ("left-ankle", "left-heel"), ("left-ankle", "left-big-toe-tip"),
    ("right-ankle", "right-heel"), ("right-ankle", "right-big-toe-tip"),
    ("nose", "left-eye"), ("nose", "right-eye"),
]


def fig_data_overview(manifest: dict):
    trials = [t for t, e in manifest["trials"].items() if e["paired"]]
    mocap_frames, sam3d_frames_v1, sam3d_frames_v2 = [], [], []
    for t in trials:
        mocap_frames.append(len(np.load(MOCAP_DIR / f"{t}.npz")["time"]))
        sam3d_frames_v1.append(len(np.load(SAM3D_DIR / f"{t}__view1.npz")["frame_idx"]))
        sam3d_frames_v2.append(len(np.load(SAM3D_DIR / f"{t}__view2.npz")["frame_idx"]))

    x = np.arange(len(trials))
    width = 0.28
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.bar(x - width, mocap_frames, width, label="mocap @ 100Hz")
    ax.bar(x, sam3d_frames_v1, width, label="SAM3D view1 (subsampled 10fps)")
    ax.bar(x + width, sam3d_frames_v2, width, label="SAM3D view2 (subsampled 10fps)")
    ax.set_xticks(x)
    ax.set_xticklabels(trials, rotation=30, ha="right")
    ax.set_ylabel("frame count")
    ax.set_title("Phase 1/2 data volume per trial")
    ax.legend()
    plt.tight_layout()
    out = OUT_DIR / "01_data_overview.png"
    plt.savefig(out, dpi=110)
    plt.close(fig)
    print(f"Saved {out}")


def draw_skeleton_2d(img_bgr, keypoints_2d):
    img = img_bgr.copy()
    for a, b in SKELETON_LINKS:
        pa, pb = keypoints_2d[JOINT_IDX[a]], keypoints_2d[JOINT_IDX[b]]
        cv2.line(img, (int(pa[0]), int(pa[1])), (int(pb[0]), int(pb[1])), (0, 255, 0), 2)
    for idx in JOINT_IDX.values():
        x, y = keypoints_2d[idx][:2]
        cv2.circle(img, (int(x), int(y)), 4, (0, 0, 255), -1)
    return img


def fig_skeleton_overlays(manifest: dict, trial: str = "walking_1"):
    fig, axes = plt.subplots(1, 2, figsize=(10, 12))
    for ax, view in zip(axes, ("view1", "view2")):
        d = np.load(SAM3D_DIR / f"{trial}__{view}.npz")
        mid = len(d["frame_idx"]) // 2
        frame_idx = int(d["frame_idx"][mid])
        kp2d = d["pred_keypoints_2d"][mid]

        video_path = manifest["trials"][trial]["views"][view]["path"]
        cap = cv2.VideoCapture(video_path)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ok, frame_bgr = cap.read()
        cap.release()
        assert ok

        overlay = draw_skeleton_2d(frame_bgr, kp2d)
        ax.imshow(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
        ax.set_title(f"{trial} / {view} (frame {frame_idx})")
        ax.axis("off")
    plt.tight_layout()
    out = OUT_DIR / f"02_skeleton_overlay_{trial}.png"
    plt.savefig(out, dpi=110)
    plt.close(fig)
    print(f"Saved {out}")


def fig_trajectory_comparison(trial: str, view: str = "view1"):
    d = np.load(ALIGNED_DIR / f"{trial}__{view}.npz")
    t = d["mocap_time_overlap"]

    fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    for ax, joint in zip(axes, ["pelvis", "knee_left"]):
        mocap = d[f"mocap__{joint}"]
        aligned = d[f"aligned__{joint}"]
        ax.plot(t, mocap[:, 1], "b-", label="mocap (ground truth)")
        ax.plot(t, aligned[:, 1], "r--", label="SAM 3D Body (aligned)")
        ax.set_ylabel(f"{joint} vertical (mm)")
        ax.legend(fontsize=8)
    axes[-1].set_xlabel("time (s, mocap clock)")
    lag = float(d["lag_seconds"])
    scale = float(d["scale"])
    mean_res = float(d["mean_residual_mm"])
    fig.suptitle(f"{trial} / {view} -- lag={lag:.2f}s, scale={scale:.1f}, mean_residual={mean_res:.0f}mm")
    plt.tight_layout()
    out = OUT_DIR / f"03_trajectory_{trial}_{view}.png"
    plt.savefig(out, dpi=110)
    plt.close(fig)
    print(f"Saved {out}")


def fig_alignment_summary(manifest: dict):
    trials = [t for t, e in manifest["trials"].items() if e["paired"]]
    rows = []
    for t in trials:
        for view in ("view1", "view2"):
            d = np.load(ALIGNED_DIR / f"{t}__{view}.npz")
            rows.append((f"{t}\n{view}", float(d["scale"]), float(d["mean_residual_mm"])))

    labels = [r[0] for r in rows]
    scales = [r[1] for r in rows]
    residuals = [r[2] for r in rows]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8))
    x = np.arange(len(labels))
    ax1.bar(x, scales, color="steelblue")
    ax1.axhline(1000, color="black", linestyle=":", label="expected (m->mm)")
    ax1.set_ylabel("recovered scale")
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, rotation=90, fontsize=7)
    ax1.legend()
    ax1.set_title("Phase 3 alignment: recovered scale per trial/view")

    ax2.bar(x, residuals, color="indianred")
    ax2.set_ylabel("mean residual (mm)")
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, rotation=90, fontsize=7)
    ax2.set_title("Phase 3 alignment: mean joint-position residual per trial/view")

    plt.tight_layout()
    out = OUT_DIR / "04_alignment_summary.png"
    plt.savefig(out, dpi=110)
    plt.close(fig)
    print(f"Saved {out}")


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    manifest = json.loads(MANIFEST_PATH.read_text())

    fig_data_overview(manifest)
    fig_skeleton_overlays(manifest, trial="walking_1")
    fig_skeleton_overlays(manifest, trial="squats_1")
    fig_trajectory_comparison("squats_1", "view1")   # clean lag, no ambiguity warning
    fig_trajectory_comparison("walking_1", "view1")  # flagged ambiguous lag
    fig_alignment_summary(manifest)


if __name__ == "__main__":
    main()

"""Renders the raw source video for a trial with SAM 3D Body's own 2D
keypoint predictions (output/sam3d/*.npz's pred_keypoints_2d, already in
full-image pixel coordinates -- no camera calibration needed) drawn on top,
one output video per camera view.

SAM 3D Body only ran on every FRAME_STEP-th frame (see run_sam3d.py), so the
skeleton is held at its last known position between sampled frames -- output
stays at the source's native fps (full smooth video), the skeleton updates
at the model's actual ~10fps.

Reuses the same SKELETON_LINKS / draw_skeleton_2d convention already
established in visualize_overview.py's fig_skeleton_overlays (single-frame
checkpoint plot) -- here applied to every frame of a full video instead of
one midpoint frame.
"""

import json
import sys
from pathlib import Path

import cv2
import numpy as np

SAM3D_REPO = Path(__file__).resolve().parent.parent / "third_party" / "sam-3d-body"
sys.path.insert(0, str(SAM3D_REPO))
from sam_3d_body.metadata.mhr70 import mhr_names  # noqa: E402

ANALYSIS_DIR = Path(__file__).resolve().parent.parent
MANIFEST_PATH = ANALYSIS_DIR / "manifest.json"
SAM3D_DIR = ANALYSIS_DIR / "output" / "sam3d"
OUT_DIR = ANALYSIS_DIR / "output" / "overlay"

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


def draw_skeleton_2d(img_bgr, keypoints_2d):
    img = img_bgr.copy()
    for a, b in SKELETON_LINKS:
        pa, pb = keypoints_2d[JOINT_IDX[a]], keypoints_2d[JOINT_IDX[b]]
        cv2.line(img, (int(pa[0]), int(pa[1])), (int(pb[0]), int(pb[1])), (0, 255, 0), 2)
    for idx in JOINT_IDX.values():
        x, y = keypoints_2d[idx][:2]
        cv2.circle(img, (int(x), int(y)), 4, (0, 0, 255), -1)
    return img


def render_overlay_video(video_path: str, sam3d_npz: Path, out_path: Path):
    d = np.load(sam3d_npz)
    frame_idx = d["frame_idx"]
    kp2d_all = d["pred_keypoints_2d"]

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"Failed to open VideoWriter for {out_path}")

    sample_ptr = 0
    current_kp2d = None
    for frame_num in range(total_frames):
        ok, frame_bgr = cap.read()
        if not ok:
            break
        while sample_ptr < len(frame_idx) and frame_idx[sample_ptr] <= frame_num:
            current_kp2d = kp2d_all[sample_ptr]
            sample_ptr += 1
        frame_out = draw_skeleton_2d(frame_bgr, current_kp2d) if current_kp2d is not None else frame_bgr
        writer.write(frame_out)

    cap.release()
    writer.release()
    print(f"Saved {out_path} ({total_frames} frames @ {fps:.1f}fps)")


def main(trial: str = "walking_2"):
    manifest = json.loads(MANIFEST_PATH.read_text())
    for view in ("view1", "view2"):
        video_path = manifest["trials"][trial]["views"][view]["path"]
        sam3d_npz = SAM3D_DIR / f"{trial}__{view}.npz"
        out_path = OUT_DIR / f"{trial}__{view}_overlay.mp4"
        render_overlay_video(video_path, sam3d_npz, out_path)


if __name__ == "__main__":
    trial = sys.argv[1] if len(sys.argv) > 1 else "walking_2"
    main(trial)

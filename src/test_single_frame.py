"""One-off smoke test: run SAM 3D Body on a single frame from squats_1 view1,
with no detector/segmentor/FOV-estimator (full-image bbox, default intrinsics),
and save a rendered overlay for visual inspection."""

import os
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

SAM3D_REPO = Path(__file__).resolve().parent.parent / "third_party" / "sam-3d-body"
sys.path.insert(0, str(SAM3D_REPO))

from sam_3d_body import load_sam_3d_body, SAM3DBodyEstimator
from sam_3d_body.metadata.mhr70 import mhr_names

# tools/vis_utils pulls in pyrender (mesh rendering) and detectron2 (label
# rendering) for its full visualize_sample_together(); we only need a quick
# 2D keypoint sanity check, so draw our own minimal skeleton overlay instead.
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
    for name, idx in JOINT_IDX.items():
        if idx >= len(keypoints_2d):
            continue
        x, y = keypoints_2d[idx][:2]
        cv2.circle(img, (int(x), int(y)), 4, (0, 0, 255), -1)
    return img

CHECKPOINT = SAM3D_REPO / "checkpoints" / "sam-3d-body-vith" / "model.ckpt"
MHR_PATH = SAM3D_REPO / "checkpoints" / "sam-3d-body-vith" / "assets" / "mhr_model.pt"
VIDEO_PATH = Path(
    r"D:\Kushal 2020\FAU MTech\SEMESTER 4\seminar-rma\FAU-Seminar-Research-in-Movement-Analysis"
    r"\Segmented Videos _1\Squats_1.mov"
)
OUT_DIR = Path(__file__).resolve().parent.parent / "output" / "smoke_test"


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)

    cap = cv2.VideoCapture(str(VIDEO_PATH))
    ok, frame_bgr = cap.read()
    for _ in range(100):  # grab a frame ~3s in, mid-motion rather than frame 0
        ok, frame_bgr = cap.read()
    cap.release()
    assert ok, "failed to read frame"
    print("frame shape (H,W,C):", frame_bgr.shape)

    model, model_cfg = load_sam_3d_body(str(CHECKPOINT), device=device, mhr_path=str(MHR_PATH))
    estimator = SAM3DBodyEstimator(
        sam_3d_body_model=model,
        model_cfg=model_cfg,
        human_detector=None,
        human_segmentor=None,
        fov_estimator=None,
    )

    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    if torch.cuda.is_available():
        print("VRAM before inference (MB):", torch.cuda.memory_allocated() / 1e6)

    outputs = estimator.process_one_image(frame_rgb, bboxes=None, use_mask=False)

    if torch.cuda.is_available():
        print("VRAM peak during inference (MB):", torch.cuda.max_memory_allocated() / 1e6)

    print("num people detected:", len(outputs))
    out0 = outputs[0]
    for k, v in out0.items():
        if isinstance(v, np.ndarray):
            print(f"  {k}: shape={v.shape} dtype={v.dtype}")
        else:
            print(f"  {k}: {type(v)} = {v}")

    kp2d = out0["pred_keypoints_2d"]
    print("pred_keypoints_2d shape:", kp2d.shape, "range x:", kp2d[:, 0].min(), kp2d[:, 0].max(),
          "range y:", kp2d[:, 1].min(), kp2d[:, 1].max())
    rend_img = draw_skeleton_2d(frame_bgr, kp2d)
    out_path = OUT_DIR / "squats_1_view1_frame100.jpg"
    cv2.imwrite(str(out_path), rend_img.astype(np.uint8))
    print("Saved overlay to", out_path)


if __name__ == "__main__":
    main()

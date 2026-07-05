"""Phase 2: run SAM 3D Body on every frame (subsampled every FRAME_STEP-th
frame) of both camera views for all paired trials.

No human detector/segmentor/FOV-estimator is used: the videos are
single-person, lab-recorded, person-filling-frame, so process_one_image()
falls back to a full-image bounding box (confirmed correct via visual
smoke test on squats_1). Detectron2 was skipped entirely -- this repo has
no C++ build tools, and the full-image-bbox fallback makes it unnecessary
for this dataset.

Resumable: a trial/view combo already saved to output/sam3d/ is skipped, so
an interrupted multi-hour run can just be restarted.
"""

import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from scipy.signal import savgol_filter
from tqdm import tqdm

ANALYSIS_DIR = Path(__file__).resolve().parent.parent
SAM3D_REPO = ANALYSIS_DIR / "third_party" / "sam-3d-body"
sys.path.insert(0, str(SAM3D_REPO))

from sam_3d_body import load_sam_3d_body, SAM3DBodyEstimator  # noqa: E402

MANIFEST_PATH = ANALYSIS_DIR / "manifest.json"
CHECKPOINT = SAM3D_REPO / "checkpoints" / "sam-3d-body-vith" / "model.ckpt"
MHR_PATH = SAM3D_REPO / "checkpoints" / "sam-3d-body-vith" / "assets" / "mhr_model.pt"
OUTPUT_DIR = ANALYSIS_DIR / "output" / "sam3d"

FRAME_STEP = 3  # process every 3rd frame (~10fps effective on 30fps source)
EMPTY_CACHE_EVERY = 20  # frames, to manage the tight 4GB VRAM budget


def build_estimator():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)
    model, model_cfg = load_sam_3d_body(str(CHECKPOINT), device=device, mhr_path=str(MHR_PATH))
    return SAM3DBodyEstimator(
        sam_3d_body_model=model,
        model_cfg=model_cfg,
        human_detector=None,
        human_segmentor=None,
        fov_estimator=None,
    )


def process_video(estimator, video_path: str, frame_step: int = FRAME_STEP):
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    records = {
        "frame_idx": [],
        "pred_joint_coords": [],
        "pred_keypoints_3d": [],
        "pred_keypoints_2d": [],
        "global_rot": [],
        "pred_cam_t": [],
        "focal_length": [],
        "bbox": [],
    }

    frame_i = 0
    n_processed = 0
    n_failed = 0
    pbar = tqdm(total=total_frames // frame_step + 1, desc=Path(video_path).stem)
    while True:
        ok, frame_bgr = cap.read()
        if not ok:
            break
        if frame_i % frame_step == 0:
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            try:
                outputs = estimator.process_one_image(frame_rgb, bboxes=None, use_mask=False)
                out0 = outputs[0]
                records["frame_idx"].append(frame_i)
                records["pred_joint_coords"].append(out0["pred_joint_coords"])
                records["pred_keypoints_3d"].append(out0["pred_keypoints_3d"])
                records["pred_keypoints_2d"].append(out0["pred_keypoints_2d"])
                records["global_rot"].append(out0["global_rot"])
                records["pred_cam_t"].append(out0["pred_cam_t"])
                records["focal_length"].append(out0["focal_length"])
                records["bbox"].append(out0["bbox"])
            except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
                n_failed += 1
                tqdm.write(f"  frame {frame_i}: inference failed ({e}); skipping")
                torch.cuda.empty_cache()

            n_processed += 1
            if n_processed % EMPTY_CACHE_EVERY == 0:
                torch.cuda.empty_cache()
            pbar.update(1)
        frame_i += 1
    pbar.close()
    cap.release()

    if n_failed:
        print(f"  WARNING: {n_failed} frame(s) failed inference and were skipped")

    return {k: np.array(v) for k, v in records.items()}


def smooth_trajectory(arr: np.ndarray, window: int = 9, polyorder: int = 2) -> np.ndarray:
    """Savitzky-Golay smoothing along the frame axis (axis 0), per joint/coord.
    Falls back to no smoothing if there aren't enough frames for the window."""
    n = arr.shape[0]
    w = min(window, n if n % 2 == 1 else n - 1)
    if w <= polyorder or w < 3:
        return arr.copy()
    flat = arr.reshape(n, -1)
    smoothed = savgol_filter(flat, window_length=w, polyorder=polyorder, axis=0)
    return smoothed.reshape(arr.shape)


def list_available_jobs(manifest: dict):
    jobs = []
    for trial, entry in manifest["trials"].items():
        if not entry["paired"]:
            continue
        for view in ("view1", "view2"):
            jobs.append((trial, view, entry["views"][view]["path"]))
    return jobs


def prompt_selection(jobs: list):
    print("Available trial/view videos:")
    for i, (trial, view, _path) in enumerate(jobs):
        done = "[done]" if (OUTPUT_DIR / f"{trial}__{view}.npz").exists() else ""
        print(f"  [{i:2d}] {trial:20s} {view}  {done}")

    while True:
        choice = input("\nEnter a number, or '<trial> <view>' (e.g. 'walking_1 view1'): ").strip()
        if choice.isdigit() and 0 <= int(choice) < len(jobs):
            return jobs[int(choice)]
        parts = choice.split()
        if len(parts) == 2:
            match = next((j for j in jobs if j[0] == parts[0] and j[1] == parts[1]), None)
            if match:
                return match
        print("  Not recognized, try again.")


def run_one(estimator, trial: str, view: str, video_path: str):
    out_path = OUTPUT_DIR / f"{trial}__{view}.npz"
    if out_path.exists():
        overwrite = input(f"{out_path.name} already exists -- overwrite? [y/N]: ").strip().lower()
        if overwrite != "y":
            print("Skipped.")
            return

    print(f"\n=== {trial} / {view} ===")
    t0 = time.time()
    records = process_video(estimator, video_path)
    elapsed = time.time() - t0
    n = len(records["frame_idx"])
    print(f"  {n} frames processed in {elapsed/60:.1f} min ({elapsed/max(n,1):.2f} s/frame)")

    smoothed_joint_coords = smooth_trajectory(records["pred_joint_coords"])
    smoothed_keypoints_3d = smooth_trajectory(records["pred_keypoints_3d"])

    np.savez_compressed(
        out_path,
        frame_idx=records["frame_idx"],
        pred_joint_coords=records["pred_joint_coords"],
        pred_joint_coords_smoothed=smoothed_joint_coords,
        pred_keypoints_3d=records["pred_keypoints_3d"],
        pred_keypoints_3d_smoothed=smoothed_keypoints_3d,
        pred_keypoints_2d=records["pred_keypoints_2d"],
        global_rot=records["global_rot"],
        pred_cam_t=records["pred_cam_t"],
        focal_length=records["focal_length"],
        bbox=records["bbox"],
        frame_step=FRAME_STEP,
    )
    print(f"  Saved -> {out_path}")


def main():
    manifest = json.loads(MANIFEST_PATH.read_text())
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    jobs = list_available_jobs(manifest)
    trial, view, video_path = prompt_selection(jobs)

    estimator = build_estimator()
    run_one(estimator, trial, view, video_path)


if __name__ == "__main__":
    main()

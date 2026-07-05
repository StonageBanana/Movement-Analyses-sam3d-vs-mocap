"""Phase 0: scan the mocap and two video folders and build manifest.json.

Maps each of the 11 trials to its mocap files and (where present) its two
camera-view video files, with fps/frame-count probed directly from each
video via OpenCV (no external ffmpeg/ffprobe binary required).
"""

import json
from pathlib import Path

import cv2

PROJECT_ROOT = Path(r"D:\Kushal 2020\FAU MTech\SEMESTER 4\seminar-rma\FAU-Seminar-Research-in-Movement-Analysis")
MEASUREMENT_DIR = PROJECT_ROOT / "Measurement"
VIEW_DIRS = {
    "view1": PROJECT_ROOT / "Segmented Videos _1",
    "view2": PROJECT_ROOT / "Segmented Videos_2",
}
ANALYSIS_DIR = Path(__file__).resolve().parent.parent
OUTPUT_MANIFEST = ANALYSIS_DIR / "manifest.json"

TRIALS = [
    "dance_move_1",
    "feet_movements_1",
    "random_1",
    "running_1",
    "running_2",
    "squats_1",
    "squats_2",
    "static",
    "walking_1",
    "walking_2",
    "walking_3",
]


def probe_video(path: Path) -> dict:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {path}")
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    return {
        "path": str(path),
        "fps": round(fps, 3),
        "frame_count": frame_count,
        "duration_sec": round(frame_count / fps, 2) if fps else None,
        "width": width,
        "height": height,
    }


def video_filename_for_trial(trial: str) -> str:
    # Video files are capitalized (e.g. "Walking_1.mov") while mocap trial
    # names are lowercase (e.g. "walking_1") -- map explicitly rather than
    # guessing casing rules.
    return trial[0].upper() + trial[1:] + ".mov"


def main():
    manifest = {"trials": {}}
    missing = []

    for trial in TRIALS:
        entry = {
            "trial": trial,
            "mocap": {
                "trc": str(MEASUREMENT_DIR / f"{trial}.trc"),
                "c3d": str(MEASUREMENT_DIR / f"{trial}.c3d"),
                "csv": str(MEASUREMENT_DIR / f"{trial}.csv"),
                "mot": str(MEASUREMENT_DIR / f"{trial}.mot"),
                "xcp": str(MEASUREMENT_DIR / f"{trial}.xcp"),
            },
            "views": {},
            "paired": False,
        }

        for key, path in entry["mocap"].items():
            if not Path(path).exists():
                missing.append(path)

        if trial == "static":
            manifest["trials"][trial] = entry
            continue

        video_name = video_filename_for_trial(trial)
        views_ok = True
        for view_key, view_dir in VIEW_DIRS.items():
            video_path = view_dir / video_name
            if not video_path.exists():
                missing.append(str(video_path))
                views_ok = False
                continue
            entry["views"][view_key] = probe_video(video_path)

        entry["paired"] = views_ok
        manifest["trials"][trial] = entry

    manifest["summary"] = {
        "total_trials": len(TRIALS),
        "paired_trials": sum(1 for t in manifest["trials"].values() if t["paired"]),
        "unpaired_trials": [t for t, v in manifest["trials"].items() if not v["paired"]],
        "missing_files": missing,
    }

    OUTPUT_MANIFEST.write_text(json.dumps(manifest, indent=2))
    print(f"Wrote manifest to {OUTPUT_MANIFEST}")
    print(f"Paired trials: {manifest['summary']['paired_trials']}/{len(TRIALS)}")
    print(f"Unpaired: {manifest['summary']['unpaired_trials']}")
    if missing:
        print(f"WARNING: {len(missing)} missing files:")
        for m in missing:
            print(f"  - {m}")


if __name__ == "__main__":
    main()

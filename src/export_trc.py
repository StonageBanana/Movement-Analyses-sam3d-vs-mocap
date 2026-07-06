"""Export aligned SAM3D/fused joint trajectories as .trc files, in the same
ASCII format as the raw mocap .trc files, so they can be loaded directly
alongside the real ground-truth mocap trajectories in Vicon Nexus,
Visual3D, OpenSim, or similar for a direct side-by-side comparison.

Only the *aligned* outputs are exported -- output/aligned/*.npz (Phase 3,
per view) and output/aligned_fused/*.npz (Phase 6, fused) -- since those
are already in mocap's own coordinate frame, scale (mm), and 100Hz
timeline. The raw, un-aligned output/sam3d/*.npz and output/fused/*.npz
live in an arbitrary scale/frame and wouldn't overlay meaningfully with
mocap, so they're intentionally not exported here.

The canonical 19 joints (pelvis, neck, head, hip/knee/ankle/heel/toe/
shoulder/elbow/wrist x left/right) are written as 19 "markers" -- frame
numbers/timestamps match the real mocap file's own numbering (frame 1 =
t=0s, 100Hz), so loading the exported .trc next to the real trial's .trc
in the same viewer lines up frame-for-frame.
"""

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from joint_mapping import CANONICAL_JOINTS

ANALYSIS_DIR = Path(__file__).resolve().parent.parent
MANIFEST_PATH = ANALYSIS_DIR / "manifest.json"
ALIGNED_DIR = ANALYSIS_DIR / "output" / "aligned"
ALIGNED_FUSED_DIR = ANALYSIS_DIR / "output" / "aligned_fused"
OUTPUT_DIR = ANALYSIS_DIR / "output" / "trc_export"

DATA_RATE = 100.0  # aligned outputs are already resampled onto mocap's 100Hz clock


def _fmt(v: float) -> str:
    return "" if np.isnan(v) else f"{v:.6f}"


def write_trc(out_path: Path, joints: dict, time: np.ndarray):
    n_frames = len(time)
    n_markers = len(CANONICAL_JOINTS)
    frame_numbers = np.round(time * DATA_RATE).astype(int) + 1  # frame 1 == mocap's own t=0

    header = [
        f"PathFileType\t3\t(X/Y/Z)\t{out_path.name}",
        "DataRate\tCameraRate\tNumFrames\tNumMarkers\tUnits\tOrigDataRate\tOrigDataStartFrame\tOrigNumFrames",
        f"{DATA_RATE:.1f}\t{DATA_RATE:.1f}\t{n_frames}\t{n_markers}\tmm\t{DATA_RATE:.1f}\t{int(frame_numbers[0])}\t{n_frames}",
        "Frame#\tTime\t" + "".join(f"{name}\t\t\t" for name in CANONICAL_JOINTS),
        "\t\t" + "".join(f"X{i}\tY{i}\tZ{i}\t" for i in range(1, n_markers + 1)),
    ]

    # newline="\r\n": Vicon/OpenSim .trc files use CRLF line endings.
    with open(out_path, "w", newline="\r\n") as f:
        f.write("\n".join(header) + "\n")
        for fi in range(n_frames):
            row = [str(int(frame_numbers[fi])), f"{time[fi]:.5f}"]
            for name in CANONICAL_JOINTS:
                x, y, z = joints[name][fi]
                row.extend([_fmt(x), _fmt(y), _fmt(z)])
            f.write("\t".join(row) + "\n")


def export_view(trial: str, view: str) -> Path:
    d = np.load(ALIGNED_DIR / f"{trial}__{view}.npz")
    joints = {j: d[f"aligned__{j}"] for j in CANONICAL_JOINTS}
    time = d["mocap_time_overlap"]
    out_path = OUTPUT_DIR / f"{trial}__{view}_aligned.trc"
    write_trc(out_path, joints, time)
    return out_path


def export_fused(trial: str) -> Path:
    d = np.load(ALIGNED_FUSED_DIR / f"{trial}.npz")
    joints = {j: d[f"aligned__{j}"] for j in CANONICAL_JOINTS}
    time = d["mocap_time_overlap"]
    out_path = OUTPUT_DIR / f"{trial}__fused_aligned.trc"
    write_trc(out_path, joints, time)
    return out_path


def main():
    manifest = json.loads(MANIFEST_PATH.read_text())
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    n_written = 0
    for trial, entry in manifest["trials"].items():
        if not entry["paired"]:
            continue
        print(f"{trial}:")
        for view in ("view1", "view2"):
            path = ALIGNED_DIR / f"{trial}__{view}.npz"
            if not path.exists():
                print(f"  SKIP {view} (no aligned output yet)")
                continue
            out = export_view(trial, view)
            print(f"  Saved {out.name}")
            n_written += 1
        fused_path = ALIGNED_FUSED_DIR / f"{trial}.npz"
        if fused_path.exists():
            out = export_fused(trial)
            print(f"  Saved {out.name}")
            n_written += 1
        else:
            print("  SKIP fused (no aligned_fused output yet)")

    print(f"\n{n_written} .trc files written -> {OUTPUT_DIR}")


if __name__ == "__main__":
    main()

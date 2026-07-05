"""Phase 1: parse mocap ground truth for every trial in manifest.json.

For each trial: parse .trc marker trajectories, derive canonical joint
centers (mocap.joints) and simplified clinical joint angles (mocap.angles),
and save the result to output/mocap/{trial}.npz. Also cross-checks one
trial's .trc against its binary .c3d, and sanity-checks knee flexion range.
"""

import itertools
import json
import sys
from pathlib import Path

import ezc3d
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from mocap.trc_io import parse_trc
from mocap.vsk_io import parse_vsk
from mocap.joints import compute_canonical_joints
from mocap.angles import compute_joint_angles, circular_mean_deg

ANALYSIS_DIR = Path(__file__).resolve().parent.parent
MANIFEST_PATH = ANALYSIS_DIR / "manifest.json"
VSK_PATH = ANALYSIS_DIR.parent / "FAU-Seminar-Research-in-Movement-Analysis" / "Measurement" / "SAM3D.vsk"
OUTPUT_DIR = ANALYSIS_DIR / "output" / "mocap"


def process_trial(trial: str, trc_path: str, anthro: dict) -> dict:
    trc = parse_trc(trc_path)
    if trc.num_markers != 39:
        print(f"  WARNING: {trial} has {trc.num_markers} markers, expected 39")

    joints = compute_canonical_joints(trc.markers, anthro)
    angles = compute_joint_angles(trc.markers, joints)

    up_axis = joints.pop("_up_axis")
    return {
        "trial": trial,
        "data_rate": trc.data_rate,
        "time": trc.time,
        "frames": trc.frames,
        "up_axis": up_axis,
        "joints": joints,
        "angles": angles,
        "raw_marker_count": trc.num_markers,
        "raw_frame_count": len(trc.frames),
    }


def save_result(result: dict, out_path: Path, unreliable_angles: list):
    flat = {
        "time": result["time"],
        "frames": result["frames"],
        "up_axis": result["up_axis"],
        "unreliable_angles": np.array(unreliable_angles, dtype=str),
    }
    for name, arr in result["joints"].items():
        flat[f"joint__{name}"] = arr
    for name, arr in result["angles"].items():
        flat[f"angle__{name}"] = arr
    np.savez_compressed(out_path, **flat)


def _best_axis_mapping(trc_xyz: np.ndarray, c3d_xyz: np.ndarray):
    """.trc and .c3d exports can use different (but equally valid) lab axis
    conventions (e.g. Y-up vs Z-up). Search all 48 axis permutation/sign
    combinations and return the one that makes trc_xyz match c3d_xyz, so the
    cross-check reports genuine numeric agreement rather than a false
    mismatch caused by axis convention alone."""
    best = None
    for perm in itertools.permutations(range(3)):
        for signs in itertools.product([1, -1], repeat=3):
            mapped = trc_xyz[:, list(perm)] * np.array(signs)
            err = np.abs(mapped - c3d_xyz).max()
            if best is None or err < best[0]:
                best = (err, perm, signs)
    return best  # (max_abs_error, perm, signs)


def cross_check_c3d(trial: str, trc_path: str, c3d_path: str):
    print(f"\nCross-checking {trial}: .trc vs .c3d")
    trc = parse_trc(trc_path)
    c3d = ezc3d.c3d(c3d_path)

    c3d_labels = [l.strip() for l in c3d["parameters"]["POINT"]["LABELS"]["value"]]
    c3d_rate = c3d["parameters"]["POINT"]["RATE"]["value"][0]
    c3d_points = c3d["data"]["points"]  # (4, n_markers, n_frames), mm assumed

    print(f"  .trc: {trc.num_markers} markers @ {trc.data_rate}Hz, {len(trc.frames)} frames")
    print(f"  .c3d: {len(c3d_labels)} labels @ {c3d_rate}Hz, {c3d_points.shape[2]} frames")

    common = [m for m in trc.marker_names if m in c3d_labels]
    print(f"  Common marker names: {len(common)}/{len(trc.marker_names)}")
    if not common:
        return

    # Sample several markers x several frames, stacked, to find the axis
    # permutation/sign that reconciles the two files (not just one lucky match).
    frame_idxs = [i for i in (50, 100, 500, 1000) if i < c3d_points.shape[2] and i < len(trc.time)]
    trc_stack, c3d_stack = [], []
    for marker in common[:10]:
        c3d_idx = c3d_labels.index(marker)
        for fi in frame_idxs:
            trc_stack.append(trc.markers[marker][fi])
            c3d_stack.append(c3d_points[:3, c3d_idx, fi])
    trc_stack = np.array(trc_stack)
    c3d_stack = np.array(c3d_stack)

    err, perm, signs = _best_axis_mapping(trc_stack, c3d_stack)
    axis_names = ["X", "Y", "Z"]
    mapping_desc = ", ".join(
        f"c3d_{axis_names[i]} = {'+' if s > 0 else '-'}trc_{axis_names[p]}"
        for i, (p, s) in enumerate(zip(perm, signs))
    )
    print(f"  Best axis mapping across {len(trc_stack)} (marker,frame) samples: {mapping_desc}")
    print(f"  Max abs residual after mapping: {err:.4f} mm "
          f"({'MATCH -- same data, different axis convention' if err < 1.0 else 'MISMATCH -- investigate'})")


def main():
    manifest = json.loads(MANIFEST_PATH.read_text())
    anthro = parse_vsk(str(VSK_PATH))
    print(f"Loaded anthropometrics from {VSK_PATH.name}: "
          f"LeftLegLength={anthro.get('LeftLegLength')}, "
          f"RightLegLength={anthro.get('RightLegLength')}, "
          f"InterAsisDistance={anthro.get('InterAsisDistance')}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Pass 1: parse every trial and compute raw (uncalibrated) joints/angles.
    results = {}
    for trial, entry in manifest["trials"].items():
        print(f"\nProcessing {trial} ...")
        result = process_trial(trial, entry["mocap"]["trc"], anthro)
        results[trial] = result
        print(f"  {result['raw_marker_count']} markers, {result['raw_frame_count']} frames "
              f"@ {result['data_rate']}Hz")

    # Pass 2: calibrate every angle against the `static` trial's mean value,
    # so 0 deg means "this subject's own relaxed standing pose", not an
    # idealized anatomical zero -- corrects the systematic per-marker offset
    # inherent to using raw KNE/ANK markers as joint-center proxies.
    static_offsets = {
        name: circular_mean_deg(arr) for name, arr in results["static"]["angles"].items()
        if name != "_unreliable"
    }
    print(f"\nStatic-trial calibration offsets (deg): {static_offsets}")

    knee_ranges = {}
    unreliable_report = {}
    for trial, result in results.items():
        # _unreliable is metadata (list of angle names with a residual
        # branch-cut jump, see mocap/angles.py), not a numeric angle array --
        # pull it out before the calibration loop below, which only makes
        # sense for the actual angle arrays.
        unreliable_report[trial] = result["angles"].pop("_unreliable")

        for name in result["angles"]:
            calibrated = result["angles"][name] - static_offsets[name]
            # Cosmetic only: shift by a single constant multiple of 360 deg
            # (same shift for every frame in the trial) so numbers land in a
            # conventional range. Safe because it's one constant per
            # trial/joint, not a per-frame decision -- it cannot introduce a
            # new discontinuity, and doesn't fix a genuinely-discontinuous
            # signal (e.g. ankle_flexion_right in dynamic trials) either.
            k = np.round(np.nanmean(calibrated) / 360.0)
            result["angles"][name] = calibrated - k * 360.0

        out_path = OUTPUT_DIR / f"{trial}.npz"
        save_result(result, out_path, unreliable_report[trial])

        kf_l = result["angles"]["knee_flexion_left"]
        kf_r = result["angles"]["knee_flexion_right"]
        knee_ranges[trial] = (float(np.nanmin(kf_l)), float(np.nanmax(kf_l)),
                               float(np.nanmin(kf_r)), float(np.nanmax(kf_r)))

    # Verification 1: cross-check one trial against its binary .c3d
    walking_entry = manifest["trials"]["walking_1"]
    cross_check_c3d("walking_1", walking_entry["mocap"]["trc"], walking_entry["mocap"]["c3d"])

    # Verification 2: knee flexion plausibility (walking should peak ~50-70 deg,
    # static should sit near 0 now that it defines the calibration offset)
    print("\nCalibrated knee flexion range per trial (deg): [L_min, L_max, R_min, R_max]")
    for trial, rng in knee_ranges.items():
        print(f"  {trial:20s} {rng[0]:7.1f} {rng[1]:7.1f} {rng[2]:7.1f} {rng[3]:7.1f}")

    print("\nAngle series flagged unreliable (residual branch-cut jump -- see mocap/angles.py):")
    for trial, flagged in unreliable_report.items():
        if len(flagged):
            print(f"  {trial:20s} {list(flagged)}")


if __name__ == "__main__":
    main()

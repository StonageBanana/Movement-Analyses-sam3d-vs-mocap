"""Parse Vicon .trc marker trajectory files (ASCII, X/Y/Z per marker)."""

from dataclasses import dataclass

import numpy as np


@dataclass
class TrcData:
    data_rate: float
    num_frames: int
    num_markers: int
    units: str
    marker_names: list
    frames: np.ndarray  # (N,)
    time: np.ndarray  # (N,)
    markers: dict  # name -> (N, 3) array, mm


def _to_float(v: str) -> float:
    try:
        return float(v)
    except ValueError:
        return np.nan


def parse_trc(path: str) -> TrcData:
    with open(path, "r") as f:
        lines = f.readlines()

    header_keys = lines[1].rstrip("\n").split("\t")
    header_vals = lines[2].rstrip("\n").split("\t")
    header = dict(zip(header_keys, header_vals))

    data_rate = float(header["DataRate"])
    num_frames = int(header["NumFrames"])
    num_markers = int(header["NumMarkers"])
    units = header["Units"]

    marker_name_row = lines[3].rstrip("\n").split("\t")
    marker_names = [m for m in marker_name_row[2:] if m.strip() != ""]
    if len(marker_names) != num_markers:
        raise ValueError(
            f"{path}: header says {num_markers} markers but found {len(marker_names)} names"
        )

    frames, times = [], []
    marker_data = {name: [] for name in marker_names}

    for line in lines[5:]:
        line = line.rstrip("\n")
        if not line.strip():
            continue
        parts = line.split("\t")
        frames.append(int(float(parts[0])))
        times.append(float(parts[1]))
        vals = parts[2 : 2 + num_markers * 3]
        for i, name in enumerate(marker_names):
            x = vals[i * 3] if i * 3 < len(vals) else ""
            y = vals[i * 3 + 1] if i * 3 + 1 < len(vals) else ""
            z = vals[i * 3 + 2] if i * 3 + 2 < len(vals) else ""
            marker_data[name].append([_to_float(x), _to_float(y), _to_float(z)])

    markers = {name: np.array(vals, dtype=float) for name, vals in marker_data.items()}

    return TrcData(
        data_rate=data_rate,
        num_frames=num_frames,
        num_markers=num_markers,
        units=units,
        marker_names=marker_names,
        frames=np.array(frames),
        time=np.array(times),
        markers=markers,
    )

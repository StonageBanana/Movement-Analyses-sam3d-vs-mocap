"""Gait-cycle detection and normalization -- the "Normalization" step that
was identified as absent from the rest of the pipeline (Phases 1-8 compare
mocap and estimates frame-by-frame at matched real time instants, which is
more precise than cycle normalization when frame-accurate correspondence
already exists, but doesn't produce the classic "mean curve vs. % gait
cycle" clinical plot). This module adds that plot type specifically.

Cycle boundaries are detected from the *broadband* vertical signal (mean
position across all 19 canonical joints) -- the same signal already used
for temporal cross-correlation in align.py, chosen there for exactly the
same reason it's reused here: a single joint's trajectory (even heel/ankle
vertical position, the traditional heel-strike signal) is noisier and more
prone to spurious extra local minima than the broadband average.
"""

import numpy as np
from scipy.signal import find_peaks, savgol_filter

from joint_mapping import CANONICAL_JOINTS


def detect_cycle_boundaries(time: np.ndarray, joints: dict, min_period=0.9, max_period=2.5,
                             outlier_tolerance=0.3):
    """Cycle boundaries = local minima of the broadband vertical signal
    (mean position across all joints, Y component) -- one per full gait
    cycle. Cycles whose duration deviates from the trial's own median by
    more than `outlier_tolerance` (fractional) are dropped, since a missed
    detection shows up as a cycle roughly 1.5-2x the true period -- keeping
    it would corrupt the normalized average with a doubled-up stride.

    `min_period=0.9` is deliberately close to a real full-stride period
    (~1.2-1.3s for this subject's walking trials) -- a looser value (e.g.
    0.6s) lets the peak finder lock onto individual left/right half-steps
    instead of full strides, which silently mixes opposite-phase halves
    into the same "cycle" and corrupts every downstream average (confirmed:
    this exact mistake initially produced nonsensical negative correlations
    for angles independently known to track well).

    Returns a list of (t_start, t_end) tuples, one per accepted cycle.
    """
    stacked = np.stack([joints[j] for j in CANONICAL_JOINTS], axis=0)
    broadband = np.nanmean(stacked, axis=0)[:, 1]
    valid = ~np.isnan(broadband)
    filled = np.interp(time, time[valid], broadband[valid])
    smooth = savgol_filter(filled, 21, 3)

    dt = np.median(np.diff(time))
    minima, _ = find_peaks(-smooth, distance=int(min_period / dt), prominence=5)
    boundary_times = time[minima]

    durations = np.diff(boundary_times)
    median_dur = np.median(durations)
    cycles = []
    for i, dur in enumerate(durations):
        if abs(dur - median_dur) <= outlier_tolerance * median_dur and min_period <= dur <= max_period:
            cycles.append((boundary_times[i], boundary_times[i + 1]))
    return cycles


def normalize_signal_to_cycles(time: np.ndarray, signal: np.ndarray, cycles: list, n_points: int = 101):
    """Resample `signal` onto a common 0-100% grid for each cycle in
    `cycles`. Returns (n_accepted_cycles, n_points) -- cycles where the
    signal is entirely NaN inside that window are skipped."""
    grid = np.linspace(0, 100, n_points)
    valid_mask = ~np.isnan(signal)
    rows = []
    for t_start, t_end in cycles:
        in_cycle = (time >= t_start) & (time <= t_end)
        if in_cycle.sum() < 4 or not valid_mask[in_cycle].any():
            continue
        t_cycle = time[in_cycle]
        pct = (t_cycle - t_start) / (t_end - t_start) * 100
        sig_cycle = signal[in_cycle]
        cv = valid_mask[in_cycle]
        if cv.sum() < 4:
            continue
        resampled = np.interp(grid, pct[cv], sig_cycle[cv])
        rows.append(resampled)
    return np.array(rows)  # (n_cycles, n_points)


def mean_std_curve(time: np.ndarray, signal: np.ndarray, cycles: list, n_points: int = 101):
    """(mean, std, n_cycles_used) of `signal` normalized across `cycles`."""
    rows = normalize_signal_to_cycles(time, signal, cycles, n_points)
    if len(rows) == 0:
        nan = np.full(n_points, np.nan)
        return nan, nan, 0
    return np.nanmean(rows, axis=0), np.nanstd(rows, axis=0), len(rows)

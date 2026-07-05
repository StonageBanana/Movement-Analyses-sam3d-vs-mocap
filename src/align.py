"""Phase 3: align a SAM 3D Body view's canonical joint trajectory to the
mocap ground truth's timeline and coordinate frame.

Temporal: cross-correlate a shared derived signal (vertical pelvis motion)
to find the lag between the video's own timeline and mocap's, since the two
capture systems aren't hardware-synced.

Spatial: fit ONE similarity transform (rotation + uniform scale +
translation -- the Umeyama/Procrustes solution) per trial using every
temporally-corresponding frame and canonical joint, then apply it to the
whole sequence. A single dedicated calibration clip (e.g. the `static`
trial) isn't usable here since `static` has no video -- so instead of a
per-clip reference pose, the whole trial's own overlap is used as the
fitting set, which is the standard approach in monocular-pose-vs-mocap
validation literature (e.g. Human3.6M PA-MPJPE protocol) and is better
conditioned than a single reference frame anyway.
"""

import numpy as np


def umeyama_alignment(source: np.ndarray, target: np.ndarray):
    """Similarity transform (R, scale, t) mapping source -> target in a
    least-squares sense: target ~= scale * (R @ source.T).T + t.
    source, target: (N, 3)."""
    mu_s, mu_t = source.mean(axis=0), target.mean(axis=0)
    sc, tc = source - mu_s, target - mu_t

    # M = sc.T @ tc = sum_i (source_i-mu_s)(target_i-mu_t)^T. The rotation
    # maximizing trace(R @ M) (equivalently minimizing ||scale*R*source+t -
    # target||^2) is R = V @ U^T where M = U @ D @ V^T (*not* U @ V^T --
    # verified directly: with the U/Vt order below, the fit could produce
    # *higher* squared error than the identity transform, which is
    # impossible for a true least-squares optimum since identity is itself
    # a feasible candidate rotation).
    sigma = (sc.T @ tc) / len(source)
    U, D, Vt = np.linalg.svd(sigma)

    S = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        S[-1, -1] = -1.0

    R = Vt.T @ S @ U.T
    var_s = np.mean(np.sum(sc**2, axis=1))
    scale = np.trace(np.diag(D) @ S) / var_s
    t = mu_t - scale * (R @ mu_s)
    return R, scale, t


def apply_similarity(points: np.ndarray, R: np.ndarray, scale: float, t: np.ndarray) -> np.ndarray:
    """points: (..., 3) -> transformed (..., 3)."""
    shape = points.shape
    flat = points.reshape(-1, 3)
    out = scale * (R @ flat.T).T + t
    return out.reshape(shape)


def cross_correlate_lag(time_a, signal_a, time_b, signal_b, dt=None, max_lag_seconds=3.0):
    """Find the lag (in the same time units as time_a/time_b) that best
    aligns signal_b onto signal_a. Positive lag means signal_b happens
    `lag` seconds after signal_a (shift b backward, or equivalently
    interpret b's timeline as starting `lag` later than a's).

    Both series are resampled onto a common uniform grid before
    cross-correlating (they may come from different native frame rates).

    Repetitive actions (squats, walking, running) make single-joint signals
    prone to "cycle-slip": a strong secondary correlation peak at +/- one
    (half-)period of the movement, which can outscore the true lag. Two
    mitigations: (1) the search is restricted to +/- max_lag_seconds, since
    the two capture systems here are started within a few seconds of each
    other, not tens of seconds apart; (2) callers should pass a broadband
    signal (e.g. averaged across many joints) rather than one periodic
    joint, which is far less likely to have a clean aliasing peak.
    """
    t0 = max(time_a[0], time_b[0])
    t1 = min(time_a[-1], time_b[-1])
    if dt is None:
        dt = min(np.median(np.diff(time_a)), np.median(np.diff(time_b)))
    grid = np.arange(t0, t1, dt)

    # np.interp can't skip NaN source samples (occluded mocap markers), so
    # interpolate over them using only the valid (non-NaN) samples first.
    valid_a = ~np.isnan(signal_a)
    valid_b = ~np.isnan(signal_b)
    a = np.interp(grid, time_a[valid_a], signal_a[valid_a])
    b = np.interp(grid, time_b[valid_b], signal_b[valid_b])
    a = (a - a.mean()) / (a.std() + 1e-8)
    b = (b - b.mean()) / (b.std() + 1e-8)

    corr = np.correlate(a, b, mode="full")
    lags = np.arange(-len(b) + 1, len(a)) * dt

    window = np.abs(lags) <= max_lag_seconds
    corr_win, lags_win = corr[window], lags[window]
    best_lag = lags_win[np.argmax(corr_win)]

    # Diagnostic: flag when a competing peak (outside a small neighborhood
    # of the chosen one) comes within 5% of the winning correlation --
    # likely cycle-slip ambiguity worth a human glancing at. 0.5s excludes
    # the natural "shoulder" of the same smooth peak (a few dt away is not
    # a separate peak) while still catching real cycle-slip candidates,
    # which for gait-like actions land roughly a stride-length apart.
    far = np.abs(lags_win - best_lag) > 0.5
    if far.any() and corr_win[far].max() > 0.95 * corr_win.max():
        runner_up = lags_win[far][np.argmax(corr_win[far])]
        print(f"    WARNING: ambiguous lag -- chosen {best_lag:.2f}s vs "
              f"competing peak at {runner_up:.2f}s (>95% as strong); likely periodic aliasing")

    return best_lag


def resample_joints_to_times(src_times, joints: dict, dst_times) -> dict:
    """Linearly interpolate every (F,3) array in `joints` from src_times
    onto dst_times. Frames of dst_times outside src_times' range are
    dropped (returned joints are truncated accordingly, along with a
    boolean mask of which dst_times were kept)."""
    valid = (dst_times >= src_times[0]) & (dst_times <= src_times[-1])
    dst_valid = dst_times[valid]
    out = {}
    for name, arr in joints.items():
        resampled = np.stack(
            [np.interp(dst_valid, src_times, arr[:, d]) for d in range(3)], axis=-1
        )
        out[name] = resampled
    return out, valid

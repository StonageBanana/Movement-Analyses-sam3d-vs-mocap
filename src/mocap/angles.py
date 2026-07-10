"""Simplified sagittal-plane joint angles (hip/knee/ankle flexion-extension).

This is a deliberate simplification of a full Grood & Suntay joint
coordinate system: each segment vector is projected onto the pelvis's own
sagittal plane (anterior x_hat / vertical z_hat) and reported as an angle
from vertical. It is not clinically calibrated to the ISB standard, but the
same method is applied to both the mocap ground truth and the SAM 3D Body
skeletons later in the pipeline, so the two sides of the comparison stay
consistent even though the absolute convention is simplified.

For at least one trial/side, this subject's ankle angle sits close to the
+/-180 boundary at some point in the motion, so a fixed (-180,180] wrap
puts the discontinuity right where real data lives. Each trial's own angle
signal is instead wrapped around its own circular mean, so the
discontinuity falls wherever that trial's data is sparsest rather than at
an arbitrary fixed point. This resolves hip/knee flexion and ankle_flexion
cleanly for every trial and side except ankle_flexion_right in dynamic
trials (running/squats/random/feet_movements/dance), where this subject's
right ankle genuinely approaches a full +/-180 deg excursion at some
points -- wide enough that no single wrap window avoids a branch-cut
crossing. np.unwrap (plain, and with median-filtered crossing detection)
was tried and made this worse or didn't resolve it: real motion crossing
the branch cut is indistinguishable enough from marker-noise-driven false
crossings that unwrap's cumulative correction is unreliable either way
(seen drifting to -2000+ degrees in the worst case). This is accepted as a
known, narrow limitation: ankle_flexion_right should be treated as
unreliable specifically in those dynamic trials, downstream (flagged
explicitly in Phase 4/7 rather than silently trusted) -- the other five
angle series (both hips, both knees, ankle_flexion_left) are unaffected.
"""

import numpy as np

from .joints import pelvis_frame, _normalize


MIN_SEGMENT_LENGTH_MM = 20.0  # below this, treat the segment as a marker glitch


def _sagittal_angle_deg(vec, x_hat, z_hat):
    """Angle of `vec` from vertical (z_hat), measured in the sagittal
    plane spanned by (x_hat, z_hat). 0 deg = vertical, positive = tilted
    anterior. NaN where the segment is near-degenerate (a marker-tracking
    glitch -- e.g. toe/heel markers briefly collapsing to ~the same point --
    rather than a real anatomical configuration), since atan2 becomes
    numerically unstable there."""
    vx = np.sum(vec * x_hat, axis=-1)
    vz = np.sum(vec * z_hat, axis=-1)
    ang = np.rad2deg(np.arctan2(vx, vz))
    degenerate = np.linalg.norm(vec, axis=-1) < MIN_SEGMENT_LENGTH_MM
    ang = np.where(degenerate, np.nan, ang)
    return ang


def circular_mean_deg(angle_deg: np.ndarray) -> float:
    rad = np.deg2rad(angle_deg)
    return float(np.rad2deg(np.arctan2(np.nanmean(np.sin(rad)), np.nanmean(np.cos(rad)))))


def wrap_around_center(angle_deg: np.ndarray, center_deg: float) -> np.ndarray:
    """Wrap into (center-180, center+180] instead of the fixed (-180,180],
    so the discontinuity can be placed away from where the data lives."""
    return ((angle_deg - center_deg + 180.0) % 360.0) - 180.0 + center_deg


def _wrap_around_own_circular_mean(angle_deg: np.ndarray) -> np.ndarray:
    """Wrap a single trial's own angle signal around its own circular mean
    -- self-contained per trial, so the branch cut lands in whatever
    angular region that trial's specific motion visits least."""
    if np.isnan(angle_deg).all():
        return angle_deg
    center = circular_mean_deg(angle_deg)
    return wrap_around_center(angle_deg, center)


JUMP_THRESHOLD_DEG = 45.0


def has_residual_jump(angle_deg: np.ndarray) -> bool:
    """True if this series still has a frame-to-frame jump bigger than
    JUMP_THRESHOLD_DEG after wrapping -- i.e. a genuine branch-cut crossing
    too wide for any single wrap window to avoid (see module docstring).
    Used downstream to flag ankle_flexion_right rather than silently trust
    it in the trials where this occurs."""
    valid = angle_deg[~np.isnan(angle_deg)]
    if len(valid) < 2:
        return False
    return bool(np.max(np.abs(np.diff(valid))) > JUMP_THRESHOLD_DEG)


def compute_joint_angles(markers: dict, joints: dict) -> dict:
    up = joints["_up_axis"]
    _, x_hat, _, z_hat = pelvis_frame(markers, up)

    angles = {}
    for side in ("left", "right"):
        hip = joints[f"hip_{side}"]
        knee = joints[f"knee_{side}"]
        ankle = joints[f"ankle_{side}"]
        toe = joints[f"toe_{side}"]
        heel = joints[f"heel_{side}"]

        thigh_vec = knee - hip
        shank_vec = ankle - knee
        foot_vec = toe - heel

        thigh_ang = _sagittal_angle_deg(thigh_vec, x_hat, z_hat)
        shank_ang = _sagittal_angle_deg(shank_vec, x_hat, z_hat)
        foot_ang = _sagittal_angle_deg(foot_vec, x_hat, z_hat)

        # Each is a difference of two independent atan2 results (already
        # each in (-180,180]), so the difference itself can span up to
        # (-360,360) before wrapping -- wrap it around this trial's own
        # circular mean rather than a fixed range (see module docstring).
        # 0 deg (approx) = thigh hanging vertical from the pelvis = neutral hip.
        angles[f"hip_flexion_{side}"] = _wrap_around_own_circular_mean(thigh_ang)
        # 0 deg = straight leg (thigh and shank collinear); positive = knee bent.
        angles[f"knee_flexion_{side}"] = _wrap_around_own_circular_mean(shank_ang - thigh_ang)
        # Reported relative to the shank; not zero-referenced to a neutral
        # standing foot position (would require the static trial).
        angles[f"ankle_flexion_{side}"] = _wrap_around_own_circular_mean(foot_ang - shank_ang)

    angles["_unreliable"] = [
        name for name, arr in angles.items() if has_residual_jump(arr)
    ]
    return angles


def joint_only_pelvis_frame(hip_left: np.ndarray, hip_right: np.ndarray, up: np.ndarray):
    """Pelvis frame derived from joint centers alone, for use where raw
    ASIS/PSIS markers aren't available (e.g. SAM 3D Body's estimated
    skeleton, in Phase 4). y_hat (toward subject's left) comes from the hip
    joint centers, same as `pelvis_frame`; z_hat is taken directly as the
    (fixed, per-trial) global up axis rather than the pelvis-tilt-sensitive
    axis `pelvis_frame` derives from ASIS/PSIS (no anterior/posterior marker
    pair exists here to derive one); x_hat = y_hat x z_hat is then anterior
    by the same standard anatomical right-handed identity (left x up =
    forward) that `pelvis_frame` relies on once its own z_hat is
    sign-corrected against up. Used identically for mocap and SAM 3D joints
    in Phase 4 so both sides of that comparison share exactly the same
    frame definition -- unlike Phase 1, where only mocap has the markers
    for the pelvis-tilt-sensitive frame.
    """
    y_hat = _normalize(hip_left - hip_right)
    z_hat = np.broadcast_to(up, y_hat.shape)
    x_hat = _normalize(np.cross(y_hat, z_hat))
    return x_hat, z_hat


def compute_joint_angles_from_joints(joints: dict, up: np.ndarray, frame_joints: dict = None) -> dict:
    """Same sagittal hip/knee/ankle flexion-extension angles as
    `compute_joint_angles`, but derived purely from canonical joint centers
    -- works identically for mocap's joints and SAM 3D Body's canonical
    joints (see `joint_only_pelvis_frame`), so Phase 4 can compare the two
    sides using one shared angle definition instead of Phase 1's
    marker-only one.

    `frame_joints`: optional dict supplying hip_left/hip_right used only to
    derive the pelvis reference frame (x_hat/z_hat), instead of `joints`'
    own. SAM 3D Body's own hip_left-hip_right vector is a fixed, ~44%-width,
    ~106-degree-misoriented distortion of the true pelvis axis in this
    dataset (confirmed identical before vs. after Phase 3/5's alignment, and
    consistent across all 10 trials regardless of activity -- see project
    notes), which corrupts every hip/knee/ankle flexion angle even though
    the underlying thigh/shank/foot vectors (computed from `joints`, not
    `frame_joints`) are largely fine. Passing mocap's joints here for the
    SAM3D/fused side re-projects those vectors onto mocap's own, correctly-
    oriented axis and recovers the correlation with mocap (e.g.
    hip_flexion_right: r=-0.87 -> +0.85 on a validation trial). Defaults to
    `joints` itself, preserving the original (mocap-vs-mocap, or any other
    self-consistent) behavior."""
    frame_source = frame_joints if frame_joints is not None else joints
    x_hat, z_hat = joint_only_pelvis_frame(frame_source["hip_left"], frame_source["hip_right"], up)

    angles = {}
    for side in ("left", "right"):
        hip = joints[f"hip_{side}"]
        knee = joints[f"knee_{side}"]
        ankle = joints[f"ankle_{side}"]
        toe = joints[f"toe_{side}"]
        heel = joints[f"heel_{side}"]

        thigh_vec = knee - hip
        shank_vec = ankle - knee
        foot_vec = toe - heel

        thigh_ang = _sagittal_angle_deg(thigh_vec, x_hat, z_hat)
        shank_ang = _sagittal_angle_deg(shank_vec, x_hat, z_hat)
        foot_ang = _sagittal_angle_deg(foot_vec, x_hat, z_hat)

        angles[f"hip_flexion_{side}"] = _wrap_around_own_circular_mean(thigh_ang)
        angles[f"knee_flexion_{side}"] = _wrap_around_own_circular_mean(shank_ang - thigh_ang)
        angles[f"ankle_flexion_{side}"] = _wrap_around_own_circular_mean(foot_ang - shank_ang)

    angles["_unreliable"] = [
        name for name, arr in angles.items() if has_residual_jump(arr)
    ]
    return angles

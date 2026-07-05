"""Canonical joint set shared between the mocap ground truth (Phase 1) and
SAM 3D Body's output (Phase 2), so both sides can be compared/aligned with
the same joint names.

Mocap side already produces this exact joint set (mocap/joints.py).
SAM 3D Body side is derived here from pred_keypoints_3d (the 70 *named*
MHR joints, via sam_3d_body.metadata.mhr70) rather than the raw 127-joint
pred_joint_coords, which has no accessible name list in this repo. There is
no pelvis/head joint in the mhr70 set, so both are approximated the same
way the mocap side approximates them: pelvis = hip midpoint, head = mean of
face landmarks.

IMPORTANT: pred_keypoints_3d is root-relative (verified empirically -- the
hip midpoint's own position has ~0 variance across an entire squat
sequence, i.e. it sits at a fixed canonical root). The model's actual
estimate of how the root moves through the scene is in the separate
pred_cam_t field, which must be added back to every joint or all relative
(non-global-translation) motion looks fine while genuine whole-body
translation (e.g. rising/sinking during a squat) is silently dropped.
"""

import numpy as np

CANONICAL_JOINTS = [
    "pelvis", "neck", "head",
    "hip_left", "hip_right",
    "knee_left", "knee_right",
    "ankle_left", "ankle_right",
    "heel_left", "heel_right",
    "toe_left", "toe_right",
    "shoulder_left", "shoulder_right",
    "elbow_left", "elbow_right",
    "wrist_left", "wrist_right",
]


def sam3d_canonical_joints(pred_keypoints_3d: np.ndarray, mhr_names: list, pred_cam_t: np.ndarray) -> dict:
    """pred_keypoints_3d: (F, 70, 3), root-relative. pred_cam_t: (F, 3), the
    root's translation in camera space -- added to every joint so the
    result reflects true global motion, not just relative articulation.
    Returns {joint_name: (F,3)}."""
    idx = {name: i for i, name in enumerate(mhr_names)}

    def pt(name):
        return pred_keypoints_3d[:, idx[name], :] + pred_cam_t

    head = np.mean(
        np.stack([pt("nose"), pt("left-eye"), pt("right-eye"), pt("left-ear"), pt("right-ear")]),
        axis=0,
    )
    return {
        "pelvis": 0.5 * (pt("left-hip") + pt("right-hip")),
        "neck": pt("neck"),
        "head": head,
        "hip_left": pt("left-hip"),
        "hip_right": pt("right-hip"),
        "knee_left": pt("left-knee"),
        "knee_right": pt("right-knee"),
        "ankle_left": pt("left-ankle"),
        "ankle_right": pt("right-ankle"),
        "heel_left": pt("left-heel"),
        "heel_right": pt("right-heel"),
        "toe_left": pt("left-big-toe-tip"),
        "toe_right": pt("right-big-toe-tip"),
        "shoulder_left": pt("left-shoulder"),
        "shoulder_right": pt("right-shoulder"),
        "elbow_left": pt("left-elbow"),
        "elbow_right": pt("right-elbow"),
        "wrist_left": pt("left-wrist"),
        "wrist_right": pt("right-wrist"),
    }


def detect_vertical_axis_generic(head_pos: np.ndarray, feet_pos: np.ndarray) -> np.ndarray:
    """Same data-driven approach as mocap.joints.detect_vertical_axis, but
    generic over any (F,3) head/feet position series (used for SAM 3D
    Body's own, unknown coordinate convention)."""
    diff = np.nanmean(head_pos - feet_pos, axis=0)
    axis = int(np.argmax(np.abs(diff)))
    sign = 1.0 if diff[axis] > 0 else -1.0
    up = np.zeros(3)
    up[axis] = sign
    return up

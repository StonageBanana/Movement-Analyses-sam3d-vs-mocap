"""Derive canonical joint centers from the 39-marker PlugInGait set.

Hip joint centers use the Davis et al. (1991) regression as implemented in
Vicon Plug-in Gait / Visual3D (coefficients cross-checked against the
LeftAsisTrocanterDistance/RightAsisTrocanterDistance expressions embedded in
this subject's own SAM3D.vsk file, which independently confirm the
0.1288*LegLength - 48.56 term). All other joint centers use the standard
simplified convention of taking the relevant PlugInGait marker directly
(shoulder/elbow/knee/ankle) or the midpoint of a marker pair (wrist) --
the same simplification the estimated (SAM 3D Body) skeleton will be mapped
to in Phase 4, so both sides of the later comparison use consistent joint
definitions even where they diverge from a full clinical model.
"""

import numpy as np

MARKER_RADIUS_MM = 7.0  # standard Vicon marker radius default (14mm diameter)


def _normalize(v, axis=-1):
    n = np.linalg.norm(v, axis=axis, keepdims=True)
    n = np.where(n == 0, 1.0, n)
    return v / n


def detect_vertical_axis(markers: dict) -> np.ndarray:
    """Pick the world axis (unit vector) most consistent with "up", by
    comparing mean head-marker height to mean foot-marker height across
    all three axes and picking the axis with the largest, correctly
    signed separation."""
    head = np.nanmean(
        np.stack([markers["LFHD"], markers["RFHD"], markers["LBHD"], markers["RBHD"]]), axis=0
    )
    feet = np.nanmean(
        np.stack([markers["LTOE"], markers["RTOE"], markers["LHEE"], markers["RHEE"]]), axis=0
    )
    diff = np.nanmean(head - feet, axis=0)  # (3,), head-minus-feet per axis
    axis = int(np.argmax(np.abs(diff)))
    sign = 1.0 if diff[axis] > 0 else -1.0
    up = np.zeros(3)
    up[axis] = sign
    return up


def pelvis_frame(markers: dict, up: np.ndarray):
    """Per-frame pelvis coordinate system: origin at ASIS midpoint,
    y_hat toward subject's left, z_hat up, x_hat anterior."""
    lasi, rasi = markers["LASI"], markers["RASI"]
    lpsi, rpsi = markers["LPSI"], markers["RPSI"]

    o_asis = 0.5 * (lasi + rasi)
    o_psis = 0.5 * (lpsi + rpsi)

    y_hat = _normalize(lasi - rasi)
    v = o_asis - o_psis
    z_raw = np.cross(y_hat, v)
    z_hat = _normalize(z_raw)
    flip = np.sign(np.sum(z_hat * up, axis=-1, keepdims=True))
    flip = np.where(flip == 0, 1.0, flip)
    z_hat = z_hat * flip
    x_hat = _normalize(np.cross(y_hat, z_hat))

    return o_asis, x_hat, y_hat, z_hat


def _davis_hip_offset(leg_length_mm: float, dist_asis: np.ndarray, side_sign: float):
    """Davis et al. (1991) hip joint centre regression.

    side_sign: +1 for right hip, -1 for left hip (matches the S term).
    dist_asis: (N,) per-frame inter-ASIS distance in mm.
    Returns (hip_x, hip_y, hip_z) each (N,), in the pelvis frame
    (x=anterior, y=toward-this-side-lateral magnitude, z=up).
    """
    c = 0.115 * leg_length_mm - 15.3
    theta = np.deg2rad(28.4)
    beta = np.deg2rad(18.0)
    x_dis = 0.1288 * leg_length_mm - 48.56  # ASIS-to-trochanter, matches SAM3D.vsk expression

    hip_x = -side_sign * (c * np.sin(theta) - 0.5 * dist_asis)
    hip_y = (-x_dis - MARKER_RADIUS_MM) * np.cos(beta) + c * np.cos(theta) * np.sin(beta)
    hip_z = (-x_dis - MARKER_RADIUS_MM) * np.sin(beta) - c * np.cos(theta) * np.cos(beta)
    hip_y = np.full_like(dist_asis, hip_y)
    hip_z = np.full_like(dist_asis, hip_z)
    return hip_x, hip_y, hip_z


def compute_hip_joint_centers(markers: dict, anthro: dict, up: np.ndarray):
    lasi, rasi = markers["LASI"], markers["RASI"]
    dist_asis = np.linalg.norm(lasi - rasi, axis=-1)
    o_asis, x_hat, y_hat, z_hat = pelvis_frame(markers, up)

    left_ll = anthro.get("LeftLegLength", 800.0)
    right_ll = anthro.get("RightLegLength", 800.0)

    hx_r, hy_r, hz_r = _davis_hip_offset(right_ll, dist_asis, side_sign=+1.0)
    hjc_right = o_asis + hx_r[:, None] * x_hat - hy_r[:, None] * y_hat + hz_r[:, None] * z_hat

    hx_l, hy_l, hz_l = _davis_hip_offset(left_ll, dist_asis, side_sign=-1.0)
    hjc_left = o_asis + hx_l[:, None] * x_hat + hy_l[:, None] * y_hat + hz_l[:, None] * z_hat

    return hjc_left, hjc_right


def compute_canonical_joints(markers: dict, anthro: dict) -> dict:
    """Return {joint_name: (N,3)} for the canonical joint set used
    throughout the rest of the pipeline (matched later to SAM 3D Body's
    MHR joints in Phase 3/4)."""
    up = detect_vertical_axis(markers)
    hjc_left, hjc_right = compute_hip_joint_centers(markers, anthro, up)

    pelvis = 0.25 * (
        markers["LASI"] + markers["RASI"] + markers["LPSI"] + markers["RPSI"]
    )
    head = 0.25 * (
        markers["LFHD"] + markers["RFHD"] + markers["LBHD"] + markers["RBHD"]
    )
    neck = 0.5 * (markers["C7"] + markers["CLAV"])

    joints = {
        "pelvis": pelvis,
        "neck": neck,
        "head": head,
        "hip_left": hjc_left,
        "hip_right": hjc_right,
        "knee_left": markers["LKNE"],
        "knee_right": markers["RKNE"],
        "ankle_left": markers["LANK"],
        "ankle_right": markers["RANK"],
        "heel_left": markers["LHEE"],
        "heel_right": markers["RHEE"],
        "toe_left": markers["LTOE"],
        "toe_right": markers["RTOE"],
        "shoulder_left": markers["LSHO"],
        "shoulder_right": markers["RSHO"],
        "elbow_left": markers["LELB"],
        "elbow_right": markers["RELB"],
        "wrist_left": 0.5 * (markers["LWRA"] + markers["LWRB"]),
        "wrist_right": 0.5 * (markers["RWRA"] + markers["RWRB"]),
    }
    joints["_up_axis"] = up
    return joints

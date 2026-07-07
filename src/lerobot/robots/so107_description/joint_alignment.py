"""SO-107 motor-space -> URDF-joint-space alignment (the "layer-2" delta).

The SO-107 URDF was exported from CAD, so its joint-zero does not coincide
with the motors' calibrated zero, and some joint axes are inverted. The
per-joint ``(sign, offset_deg)`` below realigns motor readings to the URDF:

    urdf_angle_deg = sign * motor_angle_deg + offset_deg

This is calibration data, not a model constant — a different physical SO-107
build would measure slightly different values. It lives here only as the
*default* the GUI seeds a new SO-107 robot profile with; the profile's
``urdf_calibration`` section is the editable source of truth. A well-authored
URDF (e.g. the vendored SO-101) needs only the identity alignment and carries
no file like this.

The left and right arms of a bimanual SO-107 have *different* alignments
(some axes are mirror-mounted) — which is exactly why this cannot be baked
into a single shared URDF.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# Motor order matches the SO-107 follower motor list and the URDF joints S1..S7.
MOTOR_NAMES: tuple[str, ...] = (
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "forearm_roll",
    "wrist_flex",
    "wrist_roll",
    "gripper",
)
URDF_JOINT_NAMES: tuple[str, ...] = ("S1", "S2", "S3", "S4", "S5", "S6", "S7")

# IK target frame: the URDF link `L6_1`, with a static SE(3) `TIP_OFFSET` to
# the virtual EE point. The virtual EE is "where L7_1 sits when S7=0" — i.e.
# a tip frame independent of the gripper joint, so the user's gripper-pos
# command can drive S7 without fighting the IK. The offset comes straight
# from the URDF: it is the L6_1 -> L7_1 joint origin (S7 at zero), pure
# translation (no rotation in the URDF `origin rpy`). This decouples the
# 6-DOF IK from the gripper DOF that the controller overwrites post-IK.
#
# Replaces the previous "target L7_1 directly" setup, which over-constrained
# the IK and showed up as 5-8 deg rotation drift on 2D paths — see
# `teleoperators/quest_vr/TODO.md` ("Tighter tracking on moving targets").
URDF_ANCHOR_FRAME: str = "L6_1"
TIP_OFFSET: np.ndarray = np.array(
    [
        [1.0, 0.0, 0.0, 0.0202],
        [0.0, 1.0, 0.0, -0.0250],
        [0.0, 0.0, 1.0, -0.0168],
        [0.0, 0.0, 0.0, 1.0],
    ],
    dtype=float,
)


@dataclass(frozen=True)
class JointAlignment:
    """Per-joint ``(sign, offset_deg)``. ``urdf_deg = sign * motor_deg + offset_deg``."""

    sign: float
    offset_deg: float


# Discovered empirically per physical arm (the two arms are partly mirror-mounted).
RIGHT_ARM_ALIGNMENT: dict[str, JointAlignment] = {
    "shoulder_pan": JointAlignment(sign=-1.0, offset_deg=+0.00),
    "shoulder_lift": JointAlignment(sign=+1.0, offset_deg=-90.00),
    "elbow_flex": JointAlignment(sign=+1.0, offset_deg=+78.87),
    "forearm_roll": JointAlignment(sign=-1.0, offset_deg=-3.71),
    "wrist_flex": JointAlignment(sign=-1.0, offset_deg=+0.00),
    "wrist_roll": JointAlignment(sign=-1.0, offset_deg=+90.00),
    "gripper": JointAlignment(sign=-1.0, offset_deg=+0.00),
}
LEFT_ARM_ALIGNMENT: dict[str, JointAlignment] = {
    "shoulder_pan": JointAlignment(sign=-1.0, offset_deg=+0.00),
    "shoulder_lift": JointAlignment(sign=+1.0, offset_deg=-90.00),
    "elbow_flex": JointAlignment(sign=+1.0, offset_deg=+78.87),
    "forearm_roll": JointAlignment(sign=-1.0, offset_deg=+0.00),
    "wrist_flex": JointAlignment(sign=+1.0, offset_deg=+0.00),
    "wrist_roll": JointAlignment(sign=-1.0, offset_deg=+90.00),
    "gripper": JointAlignment(sign=+1.0, offset_deg=-90.00),
}

# Canonical "ready" pose in URDF joint-space degrees (S1..S7). Shoulder down,
# elbow at +60°, wrist at -40° — far from URDF joint limits (±π) so the IK
# can manoeuvre freely without joint-limit struggle, and well-centered in
# the kinematic reach. Used as the seed for IK trajectory tests, the
# trajectory-closure benchmark, and the virtual follower's connect() default.
READY_POSE_URDF_DEG: np.ndarray = np.array([0.0, -90.0, 60.0, 0.0, -40.0, 0.0, 0.0])


def motor_pose_from_urdf(urdf_deg: np.ndarray, alignment: dict[str, JointAlignment]) -> np.ndarray:
    """Invert the per-joint ``urdf = sign*motor + offset`` alignment.

    Given a URDF-space joint vector in :data:`MOTOR_NAMES` order and an
    arm alignment dict, returns the motor-space vector that produces it.
    Used to seed a controller / virtual arm from a pose that's natural to
    name in URDF coords (e.g. :data:`READY_POSE_URDF_DEG`).
    """
    sign = np.array([alignment[m].sign for m in MOTOR_NAMES], dtype=float)
    offset = np.array([alignment[m].offset_deg for m in MOTOR_NAMES], dtype=float)
    return (urdf_deg - offset) / sign

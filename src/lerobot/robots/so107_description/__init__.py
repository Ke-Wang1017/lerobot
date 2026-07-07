"""SO-107 robot description (URDF + meshes) for visualization and kinematics.

The SO-107 URDF was exported from CAD (onshape-to-robot) with generic joint
names ``S1..S7`` and a joint-zero that does not match the motor calibration
zero. The motor->URDF alignment (sign + offset per joint) lives in
:mod:`joint_alignment` and is consumed as the *default* layer-2 calibration
by the URDF visualization.
"""

from __future__ import annotations

from pathlib import Path

_PKG_DIR = Path(__file__).resolve().parent


def get_urdf_path() -> Path:
    """Absolute path to the SO-107 URDF file."""
    p = _PKG_DIR / "urdf" / "SO107.urdf"
    assert p.exists(), f"SO-107 URDF missing at {p}"
    return p


def get_meshes_dir() -> Path:
    """Absolute path to the directory containing the SO-107 link meshes."""
    p = _PKG_DIR / "meshes"
    assert p.is_dir(), f"SO-107 meshes dir missing at {p}"
    return p


def _viz_spec() -> dict:
    """URDF-visualization metadata, discovered by ``lerobot.gui.urdf_viz``.

    The SO-107 URDF uses generic ``S1..S7`` joint names and a CAD-export
    joint-zero, so it carries a non-identity per-arm alignment (layer 2).
    """
    from .joint_alignment import (
        LEFT_ARM_ALIGNMENT,
        MOTOR_NAMES,
        RIGHT_ARM_ALIGNMENT,
        URDF_JOINT_NAMES,
    )

    return {
        "name": "SO-107",
        "motors": MOTOR_NAMES,
        "urdf_joints": URDF_JOINT_NAMES,
        "urdf_file": "SO107.urdf",
        "alignment": {
            "left_": {m: (a.sign, a.offset_deg) for m, a in LEFT_ARM_ALIGNMENT.items()},
            "right_": {m: (a.sign, a.offset_deg) for m, a in RIGHT_ARM_ALIGNMENT.items()},
        },
        # EE link for trajectory visualization — the gripper tip
        # downstream of S7. This is the "tool point" a user reads off
        # the trace, even though it picks up some gripper-open/close
        # jiggle. Trades the gripper-independent cleanliness of L6 for
        # the more intuitive "where will the gripper be?" reading.
        "ee_link": "L7_1",
    }


VIZ_SPEC = _viz_spec()

__all__ = ["get_urdf_path", "get_meshes_dir", "VIZ_SPEC"]

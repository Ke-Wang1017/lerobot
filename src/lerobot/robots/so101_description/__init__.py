"""SO-101 (standard SO-ARM) robot description (URDF + meshes).

Vendored for kinematics and visualization — the unmodified 6-DOF SO-ARM.
Its URDF (``so101_new_calib``) is authored to match the motor calibration,
so the URDF visualization needs no per-joint alignment. See ``PROVENANCE.md``
for the upstream source.
"""

from __future__ import annotations

from pathlib import Path

_PKG_DIR = Path(__file__).resolve().parent


def get_urdf_path() -> Path:
    """Absolute path to the SO-101 URDF file.

    Precondition: ``urdf/so101.urdf`` has been vendored (see ``PROVENANCE.md``).
    """
    p = _PKG_DIR / "urdf" / "so101.urdf"
    assert p.exists(), f"SO-101 URDF missing at {p} — see {_PKG_DIR / 'PROVENANCE.md'}"
    return p


def get_meshes_dir() -> Path:
    """Absolute path to the directory containing the SO-101 link meshes.

    Precondition: link meshes have been vendored (see ``PROVENANCE.md``).
    """
    p = _PKG_DIR / "meshes"
    assert p.is_dir(), f"SO-101 meshes dir missing at {p}"
    return p


# URDF-visualization metadata, discovered by ``lerobot.gui.urdf_viz``. The
# URDF joint names equal the motor names and the URDF is calibration-aligned,
# so no per-joint alignment is needed (identity).
VIZ_SPEC = {
    "name": "SO-101",
    "motors": ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"),
    "urdf_joints": ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"),
    "urdf_file": "so101.urdf",
    "alignment": None,
    # EE link for trajectory visualization. Pick the wrist anchor rather
    # than the gripper-downstream link so the trace doesn't jiggle as the
    # operator opens/closes the gripper.
    "ee_link": "gripper_frame_link",
}

__all__ = ["get_urdf_path", "get_meshes_dir", "VIZ_SPEC"]

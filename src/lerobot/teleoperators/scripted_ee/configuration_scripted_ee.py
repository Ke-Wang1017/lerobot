# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Config for the scripted bimanual EE teleop (benchmark trajectory source).

Single shape per teleop instance — heart / circle / square. The shape is
ramped into (smooth move from seed to anchor), traced for ``n_waypoints``
ticks at the configured ``loop_hz``, ramped back out, then the teleop
flags ``is_exhausted`` so the loop driver exits cleanly. Used as the
intent source for the on-hardware Cartesian-IK trajectory benchmark, in
place of the standalone bench script's inline scripted shim — same
production path Quest VR teleop uses (``attach_teleop`` installs the IK
transform; predictive follower builds the Cartesian adapter).
"""

from __future__ import annotations

from dataclasses import dataclass, field

from ..config import TeleoperatorConfig


@TeleoperatorConfig.register_subclass("scripted_bimanual_ee")
@dataclass
class ScriptedBimanualEETeleopConfig(TeleoperatorConfig):
    """Bimanual scripted Cartesian-EE-delta trajectory source.

    Emits one Cartesian-EE-delta dict per tick (the action_features
    shape :class:`QuestVRTeleop` produces), driving each arm's
    ``CartesianIKController`` through ``ramp_in → shape → ramp_out``.
    Both arms get the **same** delta in robot-base frame; the user
    stages the bimanual arms so their workspaces accept that motion.

    Tick advancement is wall-clock based off ``loop_hz`` so the shape's
    trajectory timing is preserved whether the consumer polls at 30 Hz
    (script loop / lerobot-teleoperate) or 90 Hz (Cartesian-IK adapter).
    """

    shape: str = field(
        default="heart",
        metadata={
            "description": (
                "Shape to trace. heart / circle / square trace closed shapes. "
                "static_hold drives to (forward = +size_m, lateral = 0) and holds "
                "there for ``n_waypoints`` ticks — diagnostic for steady-state "
                "action↔state behaviour (does state catch up to action when "
                "action is held constant; how long does it take)."
            ),
            "enum": ["heart", "circle", "square", "static_hold"],
        },
    )
    size_m: float = field(
        default=0.05,
        metadata={
            "description": (
                "Shape size in meters. Heart = bounding-box max extent; "
                "circle = radius; square = side length; static_hold = "
                "forward offset of the held pose."
            ),
        },
    )
    n_waypoints: int = field(
        default=256,
        metadata={
            "description": (
                "Waypoints in the shape portion. 256 matches the math viz "
                "default and gives ~4-5 cm/s peak EE speed at 30 Hz."
            ),
        },
    )
    ramp_ticks: int = field(
        default=30,
        metadata={
            "description": "Linear-ramp ticks from seed to shape anchor (and back). 30 = 1 s at 30 Hz.",
        },
    )
    loop_hz: float = field(
        default=30.0,
        metadata={
            "description": "Wall-clock tick rate the trajectory unfolds at.",
        },
    )
    # Anchor offset (in robot base frame) that pushes the shape's closest-
    # to-base point away from the base + up off the desk. Default 5 cm
    # forward + 5 cm up matches the bench's defaults.
    offset_forward_m: float = field(
        default=0.05,
        metadata={"description": "Forward (away from base) offset, m."},
    )
    offset_up_m: float = field(
        default=0.05,
        metadata={"description": "Upward offset above the seed, m."},
    )
    # World-frame axes the shape's local (forward, lateral, up) directions
    # map onto. Defaults assume a typical SO-107 bimanual setup: arms
    # mounted side-by-side facing -y, gripper-up is +z. "forward" = -y,
    # "lateral" = +x.
    forward_axis: tuple[float, float, float] = field(
        default=(0.0, -1.0, 0.0),
        metadata={
            "description": "Unit vector for 'forward' (away from operator/base) in robot base frame.",
        },
    )
    lateral_axis: tuple[float, float, float] = field(
        default=(1.0, 0.0, 0.0),
        metadata={
            "description": "Unit vector for 'lateral' (in-plane perpendicular to forward) in robot base frame.",
        },
    )
    # Gripper target while the trajectory runs. Half-open by default.
    gripper_value: float = field(
        default=50.0,
        metadata={"description": "Constant motor-space gripper target while tracing."},
    )

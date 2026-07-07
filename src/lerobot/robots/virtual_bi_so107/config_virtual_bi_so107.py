#!/usr/bin/env python

# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from dataclasses import dataclass, field

from ..config import RobotConfig


@RobotConfig.register_subclass("virtual_bi_so107")
@dataclass
class VirtualBiSO107FollowerConfig(RobotConfig):
    """Configuration for the motor-less, perfect-tracker bimanual SO-107.

    Mirrors the public surface of ``BiSO107FollowerConfig`` only in the
    abstractions a teleop / record loop actually cares about (the
    embodiment is bimanual SO-107). Every motor-bus / camera / calibration
    knob is dropped because the virtual robot has no buses, no cameras,
    and no calibration state to load.

    The config exists so the GUI's robot dropdown picks the virtual
    follower up automatically (``RobotConfig.get_known_choices()``); a
    user can then select it from the run tab and pair it with any
    bimanual teleop (``ScriptedBimanualEETeleop`` today, ``Quest VR`` in
    a follow-up PR) to verify the Cartesian-IK pipeline without
    hardware.

    The IK-tuning fields mirror :class:`BiSO107FollowerConfig`'s so the
    virtual robot can be used as a no-hardware testbench for IK feel
    changes (raise ``ik_posture_cost`` here, A/B against a recorded
    trajectory, then port the same value to the real robot's profile).
    """

    # See BiSO107FollowerConfig.ik_posture_cost / ik_max_iters for the
    # full rationale. Kept in sync so a virtual<->real A/B is one field
    # copy, not a refactor.
    ik_posture_cost: float = field(
        default=0.05,
        metadata={
            "description": (
                "PostureTask weight relative to FrameTask (which is 1.0). "
                "Default 0.05 makes posture a null-space tiebreaker. Raise "
                "(e.g. 0.3) for stronger 'stay near previous pose' — tighter "
                "joint continuity near reach limits / singularities, at the "
                "cost of small EE tracking lag."
            ),
        },
    )
    ik_max_iters: int = field(
        default=50,
        metadata={
            "description": (
                "QP iteration budget per IK call. 50 closes per-call lag on "
                "moving teleop targets; lower (10–20) for more 'stick to "
                "seed' feel at the cost of moving-target tracking accuracy."
            ),
        },
    )

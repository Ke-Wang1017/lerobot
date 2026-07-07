#!/usr/bin/env python

# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Bimanual SO-107 leader with per-arm background read threads.

Drop-in replacement for :class:`BiSO107Leader`. Same caller-facing
interface (``get_action`` returns the prefixed ``left_*`` / ``right_*``
dict, ``send_feedback`` routes per-arm, etc.); the only behavioral
difference is that each per-arm leader is a
:class:`SO107LeaderHighRate` with its own bus-read thread, so each
``get_action()`` call returns the per-arm cached pose without
touching either serial bus.

The two per-arm threads run independently — they share nothing
(separate buses, separate caches, separate locks). Each arm's
controller (when bound via ``robot.attach_teleop``) polls its own
arm's cache, giving the predictive follower distinct intent samples
at control rate without bus contention or LSQ-bias.
"""

from __future__ import annotations

import logging

from lerobot.teleoperators.so107_leader_highrate import (
    SO107LeaderHighRate,
    SO107LeaderHighRateConfig,
)

from ..bi_so107_leader.bi_so107_leader import BiSO107Leader
from ..teleoperator import Teleoperator
from .config_bi_so107_leader_highrate import BiSO107LeaderHighRateConfig

logger = logging.getLogger(__name__)


class BiSO107LeaderHighRate(BiSO107Leader):
    """Bimanual SO-107 leader composed of two SO107LeaderHighRate arms."""

    config_class = BiSO107LeaderHighRateConfig
    name = "bi_so107_leader_highrate"

    def __init__(self, config: BiSO107LeaderHighRateConfig):
        # Bypass BiSO107Leader.__init__ — it constructs plain SO107Leader
        # instances. We want SO107LeaderHighRate, so call Teleoperator
        # init directly and rebuild the per-arm setup.
        Teleoperator.__init__(self, config)
        self.config = config

        left_arm_config = SO107LeaderHighRateConfig(
            id=f"{config.id}_left" if config.id else None,
            calibration_dir=config.calibration_dir,
            port=config.left_arm_port,
            gripper_bounce=config.gripper_bounce,
            intervention_enabled=config.intervention_enabled,  # left arm owns the keyboard listener
            read_rate_hz=config.read_rate_hz,
        )
        right_arm_config = SO107LeaderHighRateConfig(
            id=f"{config.id}_right" if config.id else None,
            calibration_dir=config.calibration_dir,
            port=config.right_arm_port,
            gripper_bounce=config.gripper_bounce,
            intervention_enabled=False,  # no keyboard listener on right arm
            read_rate_hz=config.read_rate_hz,
        )

        self.left_arm = SO107LeaderHighRate(left_arm_config)
        self.right_arm = SO107LeaderHighRate(right_arm_config)

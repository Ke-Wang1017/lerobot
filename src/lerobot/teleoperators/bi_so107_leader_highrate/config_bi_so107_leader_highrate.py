#!/usr/bin/env python

# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

from dataclasses import dataclass

from ..bi_so107_leader.config_bi_so107_leader import BiSO107LeaderConfig
from ..config import TeleoperatorConfig


@TeleoperatorConfig.register_subclass("bi_so107_leader_highrate")
@dataclass
class BiSO107LeaderHighRateConfig(BiSO107LeaderConfig):
    """Bimanual SO-107 leader with per-arm background read threads.

    Inherits every field of :class:`BiSO107LeaderConfig` (left/right
    ports, gripper_bounce, intervention_enabled) and adds ``read_rate_hz``
    which is propagated to each per-arm :class:`SO107LeaderHighRate`.
    Both arms run independent read threads at the same rate. The two
    threads contend on nothing (separate buses).
    """

    # Per-arm background poll rate. Default matches the predictive
    # follower's control_rate_hz so each consumer tick sees a fresh
    # sample. See SO107LeaderHighRateConfig for the rationale.
    read_rate_hz: float = 200.0

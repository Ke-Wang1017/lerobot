#!/usr/bin/env python

# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

from dataclasses import dataclass

from ..config import TeleoperatorConfig
from ..highrate.config import HighRateLeaderConfig
from ..so_leader.config_so_leader import SO107LeaderConfig


@TeleoperatorConfig.register_subclass("so107_leader_highrate")
@dataclass
class SO107LeaderHighRateConfig(HighRateLeaderConfig, SO107LeaderConfig):
    """SO-107 Leader configuration with a background bus-read thread.

    Inherits every field of :class:`SO107LeaderConfig` (port,
    use_degrees, gripper_bounce, intervention_enabled) and adds the
    ``read_rate_hz`` knob from :class:`HighRateLeaderConfig`.
    ``get_action()`` returns the cached pose without touching the bus,
    so callers (the 30 Hz record loop AND the predictive controller's
    200 Hz tick when attached) all see fresh samples without contending
    on a serial port.

    The point of the high-rate read is for the **consumer** to have
    access to distinct intent samples at a rate close to the leader's
    natural bandwidth. With the default ``read_rate_hz=200``, a
    consumer polling at 200 Hz gets ~14 distinct samples in a 70 ms
    velocity-LSQ window vs. the ~2 distinct samples it gets when the
    leader is sampled at the record loop's 30 Hz. The LSQ velocity
    estimate's under-shoot bias collapses.
    """

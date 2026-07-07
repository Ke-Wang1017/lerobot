#!/usr/bin/env python

# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
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

from dataclasses import dataclass

from ..config import TeleoperatorConfig
from ..highrate.config import HighRateLeaderConfig
from ..so_leader.config_so_leader import SOLeaderTeleopConfig


@TeleoperatorConfig.register_subclass("so_leader_highrate")
@dataclass
class SOLeaderHighRateConfig(HighRateLeaderConfig, SOLeaderTeleopConfig):
    """6-motor SO-leader configuration with a background bus-read thread.

    Covers SO-100 and SO-101 hardware — same as the plain
    :class:`SOLeaderTeleopConfig` (a.k.a. ``so100_leader`` /
    ``so101_leader``) but adds the ``read_rate_hz`` knob from
    :class:`HighRateLeaderConfig`.
    """

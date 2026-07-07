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

"""Config fields for the high-rate leader background-poller.

A dataclass mixin: any leader config that wants the high-rate read
inherits both its base teleop config AND this class, e.g.::

    @TeleoperatorConfig.register_subclass("so107_leader_highrate")
    @dataclass
    class SO107LeaderHighRateConfig(HighRateLeaderConfig, SO107LeaderConfig):
        pass

Field metadata is surfaced to the GUI as hover tooltips.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class HighRateLeaderConfig:
    """Background-poller tunables. Inherit alongside the base leader config."""

    read_rate_hz: float = field(
        default=200.0,
        metadata={
            "description": (
                "Background poll rate (Hz) for reading leader Present_Position. "
                "200 Hz matches the predictive follower's control_rate_hz so "
                "the consumer's velocity estimator sees one fresh sample per "
                "tick. Don't go higher than the Feetech bus can sustain "
                "(~170-200 Hz with P=48 sync_read of 6-7 motors per arm) — "
                "going higher just produces serial timeouts."
            ),
        },
    )

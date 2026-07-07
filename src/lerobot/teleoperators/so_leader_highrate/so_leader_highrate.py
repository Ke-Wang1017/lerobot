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

"""SO-100 / SO-101 leader with a background bus-read thread.

Same as :class:`SO107LeaderHighRate` but for the 6-motor SO-100 / SO-101
hardware (no ``forearm_roll``). All thread / cache logic comes from
:class:`HighRateLeaderMixin`; this class just nails the variant name +
config type to the 6-motor SO-leader.
"""

from __future__ import annotations

from ..highrate.mixin import HighRateLeaderMixin
from ..so_leader.so_leader import SOLeader
from .config_so_leader_highrate import SOLeaderHighRateConfig


class SOLeaderHighRate(HighRateLeaderMixin, SOLeader):
    """6-motor SO leader that polls the leader bus in a background thread."""

    config_class = SOLeaderHighRateConfig
    name = "so_leader_highrate"

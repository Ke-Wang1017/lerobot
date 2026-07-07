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

from dataclasses import dataclass

from ..config import TeleoperatorConfig


@dataclass
class SOLeaderConfig:
    """Base configuration class for SO Leader teleoperators."""

    # Port to connect to the arm
    port: str

    # Whether to use degrees for angles
    use_degrees: bool = True


@TeleoperatorConfig.register_subclass("so101_leader")
@TeleoperatorConfig.register_subclass("so100_leader")
@dataclass
class SOLeaderTeleopConfig(TeleoperatorConfig, SOLeaderConfig):
    pass


@TeleoperatorConfig.register_subclass("so107_leader")
@dataclass
class SO107LeaderConfig(TeleoperatorConfig, SOLeaderConfig):
    """SO-107 Leader configuration with gripper bounce and intervention support."""

    # Gripper bounce back to neutral position (50% open)
    gripper_bounce: bool = False
    # Enable intervention mode (press SPACE to toggle during policy execution)
    intervention_enabled: bool = False


SO100LeaderConfig = SOLeaderTeleopConfig
SO101LeaderConfig = SOLeaderTeleopConfig

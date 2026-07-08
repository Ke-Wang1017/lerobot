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

"""The robot layer ships only the abstract interface + plugin factory.

Concrete robot drivers live in external plugin packages; the factory resolves any
registered ``RobotConfig`` subclass via ``make_device_from_device_class`` and raises a
clean ``ValueError`` when no implementation can be located. These tests pin that contract
so it keeps working for external sim/real robot plugins (see examples/dagger_sim_aloha.py).
"""

from dataclasses import dataclass

import draccus
import pytest

from lerobot.robots import Robot, RobotConfig, make_robot_from_config


@RobotConfig.register_subclass("factory_iface_unresolvable_robot")
@dataclass(kw_only=True)
class _UnresolvableRobotConfig(RobotConfig):
    """A registered config with no importable ``_UnresolvableRobot`` device class."""


def test_robot_interface_is_plugin_ready():
    # The public surface is the abstract base + config registry + factory.
    assert issubclass(RobotConfig, draccus.ChoiceRegistry)
    assert isinstance(Robot, type)
    assert callable(make_robot_from_config)


def test_make_robot_from_config_raises_for_unresolvable_type():
    # No in-tree concrete robots: an unregistered/unresolvable type fails cleanly.
    with pytest.raises(ValueError):
        make_robot_from_config(_UnresolvableRobotConfig())

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

"""The teleoperator layer keeps only device-agnostic human-input teleops in-tree.

``keyboard``/``gamepad``/``quest_vr`` (intervention devices for validation / DAgger) ship
in-tree; concrete leader arms live in external plugins. The factory dispatches the in-tree
input devices and falls back to the plugin loader (clean ``ValueError`` on failure).
"""

from dataclasses import dataclass

import draccus
import pytest

from lerobot.teleoperators import Teleoperator, TeleoperatorConfig, make_teleoperator_from_config


@TeleoperatorConfig.register_subclass("factory_iface_unresolvable_teleop")
@dataclass(kw_only=True)
class _UnresolvableTeleopConfig(TeleoperatorConfig):
    """A registered config with no importable ``_UnresolvableTeleop`` device class."""


def test_teleop_interface_is_plugin_ready():
    assert issubclass(TeleoperatorConfig, draccus.ChoiceRegistry)
    assert isinstance(Teleoperator, type)
    assert callable(make_teleoperator_from_config)


def test_make_teleoperator_from_config_raises_for_unresolvable_type():
    with pytest.raises(ValueError):
        make_teleoperator_from_config(_UnresolvableTeleopConfig())


def test_keyboard_intervention_device_is_in_tree():
    # Human-input teleop stays in-tree for validation/DAgger interventions.
    pytest.importorskip("pynput", reason="pynput is required (install lerobot[pynput-dep])")
    from lerobot.teleoperators.keyboard import KeyboardTeleop, KeyboardTeleopConfig

    teleop = make_teleoperator_from_config(KeyboardTeleopConfig())
    assert isinstance(teleop, KeyboardTeleop)

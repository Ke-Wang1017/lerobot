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

"""The camera layer ships only the abstract interface + plugin factory.

Concrete camera backends (OpenCV, RealSense, ...) live in external plugin packages;
``make_cameras_from_configs`` resolves any registered ``CameraConfig`` via the plugin
loader and raises a clean ``ValueError`` when no backend can be located.
"""

from dataclasses import dataclass

import draccus
import pytest

from lerobot.cameras import Camera, CameraConfig, make_cameras_from_configs


@CameraConfig.register_subclass("factory_iface_unresolvable_camera")
@dataclass(kw_only=True)
class _UnresolvableCameraConfig(CameraConfig):
    """A registered config with no importable ``_UnresolvableCamera`` backend class."""


def test_camera_interface_is_plugin_ready():
    assert issubclass(CameraConfig, draccus.ChoiceRegistry)
    assert isinstance(Camera, type)
    assert callable(make_cameras_from_configs)


def test_empty_config_dict_returns_empty_mapping():
    assert make_cameras_from_configs({}) == {}


def test_make_cameras_from_configs_raises_for_unresolvable_type():
    with pytest.raises(ValueError):
        make_cameras_from_configs({"cam": _UnresolvableCameraConfig()})

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
"""Tests for the read-only Robots MCP tool surface.

Pure reads: enumerate JSON profile files under tmp dirs, scan pyserial
device tree (mocked at the boundary). No motor connections, no port
opening, no camera streams.
"""

from __future__ import annotations

import asyncio
import json
from unittest.mock import MagicMock, patch

import pytest

from lerobot.gui.api import robot as robot_module
from lerobot.mcp.server import build_server


@pytest.fixture
def mcp(tmp_path):
    return build_server(dataset_root=tmp_path / "_root")


def _call(mcp_server, name, args):
    _, structured = asyncio.run(mcp_server.call_tool(name, args))
    return structured


def _write_profile(d, name: str, type_: str = "bi_so107_follower", port: str = "/dev/ttyACM0"):
    d.mkdir(parents=True, exist_ok=True)
    (d / f"{name}.json").write_text(
        json.dumps(
            {
                "name": name,
                "type": type_,
                "fields": {"port": port, "baudrate": 1000000},
                "cameras": {},
            }
        )
    )


# ── list_robot_profiles / get_robot_profile ───────────────────────────────


class TestRobotProfileReads:
    def test_empty_dir_returns_zero(self, mcp, tmp_path):
        robots_dir = tmp_path / "robots"
        robots_dir.mkdir()
        with patch.object(robot_module, "ROBOT_PROFILES_DIR", robots_dir):
            result = _call(mcp, "list_robot_profiles", {})
        assert result == {"profiles": [], "total": 0}

    def test_lists_profiles_with_name_and_type(self, mcp, tmp_path):
        robots_dir = tmp_path / "robots"
        _write_profile(robots_dir, "left_arm", type_="bi_so107_follower")
        _write_profile(robots_dir, "right_arm", type_="bi_so107_follower")
        with patch.object(robot_module, "ROBOT_PROFILES_DIR", robots_dir):
            result = _call(mcp, "list_robot_profiles", {})
        assert result["total"] == 2
        names = {p["name"] for p in result["profiles"]}
        assert names == {"left_arm", "right_arm"}
        types = {p["type"] for p in result["profiles"]}
        assert types == {"bi_so107_follower"}

    def test_get_existing_profile_returns_full_json(self, mcp, tmp_path):
        robots_dir = tmp_path / "robots"
        _write_profile(robots_dir, "left_arm", port="/dev/ttyACM2")
        with patch.object(robot_module, "ROBOT_PROFILES_DIR", robots_dir):
            result = _call(mcp, "get_robot_profile", {"name": "left_arm"})
        assert result["name"] == "left_arm"
        assert result["type"] == "bi_so107_follower"
        assert result["fields"]["port"] == "/dev/ttyACM2"

    def test_get_unknown_profile_raises(self, mcp, tmp_path):
        robots_dir = tmp_path / "robots"
        robots_dir.mkdir()
        with patch.object(robot_module, "ROBOT_PROFILES_DIR", robots_dir):
            with pytest.raises(Exception, match="not found"):
                _call(mcp, "get_robot_profile", {"name": "ghost"})


# ── list_teleop_profiles / get_teleop_profile ─────────────────────────────


class TestTeleopProfileReads:
    def test_empty_dir_returns_zero(self, mcp, tmp_path):
        teleops_dir = tmp_path / "teleops"
        teleops_dir.mkdir()
        with patch.object(robot_module, "TELEOP_PROFILES_DIR", teleops_dir):
            result = _call(mcp, "list_teleop_profiles", {})
        assert result == {"profiles": [], "total": 0}

    def test_lists_profiles(self, mcp, tmp_path):
        teleops_dir = tmp_path / "teleops"
        _write_profile(teleops_dir, "quest_vr", type_="quest_vr")
        with patch.object(robot_module, "TELEOP_PROFILES_DIR", teleops_dir):
            result = _call(mcp, "list_teleop_profiles", {})
        assert result["total"] == 1
        assert result["profiles"][0]["name"] == "quest_vr"

    def test_get_unknown_teleop_raises(self, mcp, tmp_path):
        teleops_dir = tmp_path / "teleops"
        teleops_dir.mkdir()
        with patch.object(robot_module, "TELEOP_PROFILES_DIR", teleops_dir):
            with pytest.raises(Exception, match="not found"):
                _call(mcp, "get_teleop_profile", {"name": "ghost"})


# ── list_ports ────────────────────────────────────────────────────────────


class TestListPorts:
    def test_returns_pyserial_results(self, mcp):
        """When pyserial is available, list_ports.comports() is called and
        results are filtered to USB serial.
        """
        fake_port = MagicMock(
            device="/dev/ttyACM0",
            description="Feetech STM32 Virtual COM Port",
            manufacturer="Feetech",
            vid=0x1A86,
            pid=0x7523,
        )
        with patch("serial.tools.list_ports.comports", return_value=[fake_port]):
            result = _call(mcp, "list_ports", {})
        assert result["total"] == 1
        entry = result["ports"][0]
        assert entry["path"] == "/dev/ttyACM0"
        assert entry["manufacturer"] == "Feetech"
        assert "1a86:7523" in entry["vid_pid"]

    def test_filters_legacy_ttys_on_linux(self, mcp):
        """ttyS* (legacy 16550), tty0, ttyprintk etc. should be filtered out."""
        usb = MagicMock(
            device="/dev/ttyACM1",
            description="USB",
            manufacturer="Acme",
            vid=0x0001,
            pid=0x0002,
        )
        legacy = MagicMock(
            device="/dev/ttyS0",
            description="legacy",
            manufacturer=None,
            vid=None,
            pid=None,
        )
        with patch("serial.tools.list_ports.comports", return_value=[legacy, usb]):
            result = _call(mcp, "list_ports", {})
        paths = [p["path"] for p in result["ports"]]
        assert "/dev/ttyACM1" in paths
        assert "/dev/ttyS0" not in paths


# ── get_all_port_assignments ──────────────────────────────────────────────


class TestPortAssignments:
    def test_empty_returns_zero(self, mcp, tmp_path):
        robots_dir = tmp_path / "robots"
        teleops_dir = tmp_path / "teleops"
        robots_dir.mkdir()
        teleops_dir.mkdir()
        with (
            patch.object(robot_module, "ROBOT_PROFILES_DIR", robots_dir),
            patch.object(robot_module, "TELEOP_PROFILES_DIR", teleops_dir),
        ):
            result = _call(mcp, "get_all_port_assignments", {})
        assert result == {"assignments": [], "total": 0}

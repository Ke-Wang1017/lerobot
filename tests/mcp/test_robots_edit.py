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
"""Tests for the edit-tier Robots MCP tool surface.

Pure JSON file CRUD under tmp dirs. No motor connections, no port
opening, no camera streams. Verifies the standard contract:

- Outcome-transparent responses (`overwrote`, `previous_type`,
  `previous_port`, `changed`, `removed_type`, etc.).
- Error cases raise ValueError with clean messages.
- assign_port_to_arm modifies one field and preserves the rest.
"""

from __future__ import annotations

import asyncio
import json
from unittest.mock import patch

import pytest

from lerobot.gui.api import robot as robot_module
from lerobot.mcp.server import build_server


@pytest.fixture
def dirs(tmp_path):
    """Throwaway profile dirs patched into the gui.api.robot module."""
    robots_dir = tmp_path / "robots"
    teleops_dir = tmp_path / "teleops"
    robots_dir.mkdir()
    teleops_dir.mkdir()
    with (
        patch.object(robot_module, "ROBOT_PROFILES_DIR", robots_dir),
        patch.object(robot_module, "TELEOP_PROFILES_DIR", teleops_dir),
    ):
        yield robots_dir, teleops_dir


@pytest.fixture
def mcp(tmp_path, dirs):
    return build_server(dataset_root=tmp_path / "_root")


def _call(mcp_server, name, args):
    _, structured = asyncio.run(mcp_server.call_tool(name, args))
    return structured


def _seed_profile(d, name: str, type_: str = "bi_so107_follower", port: str = "/dev/ttyACM0"):
    (d / f"{name}.json").write_text(
        json.dumps(
            {
                "name": name,
                "type": type_,
                "fields": {"port": port, "baudrate": 1000000},
                "cameras": {},
                "rest_position": {},
            }
        )
    )


# ── Robot CRUD ────────────────────────────────────────────────────────────


class TestCreateRobotProfile:
    def test_creates_new_profile(self, mcp, dirs):
        robots_dir, _ = dirs
        result = _call(
            mcp,
            "create_robot_profile",
            {
                "name": "left_arm",
                "type": "bi_so107_follower",
                "fields": {"port": "/dev/ttyACM0", "baudrate": 1000000},
            },
        )
        assert result["status"] == "ok"
        assert result["created"] is True
        assert result["name"] == "left_arm"
        assert (robots_dir / "left_arm.json").exists()

    def test_fails_on_collision(self, mcp, dirs):
        robots_dir, _ = dirs
        _seed_profile(robots_dir, "left_arm")
        with pytest.raises(Exception, match="already exists"):
            _call(
                mcp,
                "create_robot_profile",
                {"name": "left_arm", "type": "bi_so107_follower"},
            )


class TestUpdateRobotProfile:
    def test_updates_existing_carries_overwrote_metadata(self, mcp, dirs):
        robots_dir, _ = dirs
        _seed_profile(robots_dir, "left_arm", type_="bi_so107_follower")
        result = _call(
            mcp,
            "update_robot_profile",
            {
                "name": "left_arm",
                "type": "bi_so107_follower_predictive",
                "fields": {"port": "/dev/ttyACM1"},
            },
        )
        assert result["status"] == "ok"
        assert result["overwrote"] is True
        assert result["previous_type"] == "bi_so107_follower"
        assert result["previous_fields_count"] == 2  # port + baudrate
        # On-disk reflects the new type
        on_disk = json.loads((robots_dir / "left_arm.json").read_text())
        assert on_disk["type"] == "bi_so107_follower_predictive"

    def test_update_missing_name_creates_with_overwrote_false(self, mcp, dirs):
        robots_dir, _ = dirs
        result = _call(
            mcp,
            "update_robot_profile",
            {"name": "fresh", "type": "bi_so107_follower"},
        )
        assert result["overwrote"] is False
        assert result["previous_type"] is None
        assert (robots_dir / "fresh.json").exists()


class TestRenameRobotProfile:
    def test_renames(self, mcp, dirs):
        robots_dir, _ = dirs
        _seed_profile(robots_dir, "old_name")
        result = _call(
            mcp,
            "rename_robot_profile",
            {"old_name": "old_name", "new_name": "new_name"},
        )
        assert result["renamed"] is True
        assert result["from"] == "old_name"
        assert result["to"] == "new_name"
        assert not (robots_dir / "old_name.json").exists()
        assert (robots_dir / "new_name.json").exists()
        # JSON contents updated too
        on_disk = json.loads((robots_dir / "new_name.json").read_text())
        assert on_disk["name"] == "new_name"

    def test_missing_old_name_raises(self, mcp, dirs):
        with pytest.raises(Exception, match="not found"):
            _call(
                mcp,
                "rename_robot_profile",
                {"old_name": "ghost", "new_name": "anything"},
            )

    def test_collision_with_new_name_raises(self, mcp, dirs):
        robots_dir, _ = dirs
        _seed_profile(robots_dir, "a")
        _seed_profile(robots_dir, "b")
        with pytest.raises(Exception, match="already exists"):
            _call(mcp, "rename_robot_profile", {"old_name": "a", "new_name": "b"})


class TestDeleteRobotProfile:
    def test_deletes_and_echoes_what_was_removed(self, mcp, dirs):
        robots_dir, _ = dirs
        _seed_profile(robots_dir, "doomed", type_="bi_so107_follower")
        result = _call(mcp, "delete_robot_profile", {"name": "doomed"})
        assert result["deleted"] is True
        assert result["name"] == "doomed"
        assert result["removed_type"] == "bi_so107_follower"
        assert result["removed_fields_count"] == 2
        assert not (robots_dir / "doomed.json").exists()

    def test_missing_raises(self, mcp, dirs):
        with pytest.raises(Exception, match="not found"):
            _call(mcp, "delete_robot_profile", {"name": "ghost"})


# ── Teleop CRUD ───────────────────────────────────────────────────────────


class TestTeleopCrud:
    def test_create_teleop_writes_to_teleop_dir(self, mcp, dirs):
        robots_dir, teleops_dir = dirs
        result = _call(
            mcp,
            "create_teleop_profile",
            {"name": "quest_vr", "type": "quest_vr"},
        )
        assert result["created"] is True
        assert (teleops_dir / "quest_vr.json").exists()
        # Did NOT land in the robots dir
        assert not (robots_dir / "quest_vr.json").exists()

    def test_delete_teleop_isolated_from_robots(self, mcp, dirs):
        robots_dir, teleops_dir = dirs
        _seed_profile(teleops_dir, "doomed")
        _seed_profile(robots_dir, "doomed")  # same name, different dir
        _call(mcp, "delete_teleop_profile", {"name": "doomed"})
        assert not (teleops_dir / "doomed.json").exists()
        # Robot profile with the same name is untouched
        assert (robots_dir / "doomed.json").exists()


# ── assign_port_to_arm ────────────────────────────────────────────────────


class TestAssignPort:
    def test_changes_port_field_preserves_other_fields(self, mcp, dirs):
        robots_dir, _ = dirs
        _seed_profile(robots_dir, "left_arm", port="/dev/ttyACM0")
        result = _call(
            mcp,
            "assign_port_to_arm",
            {"profile_name": "left_arm", "port": "/dev/ttyACM2"},
        )
        assert result["status"] == "ok"
        assert result["previous_port"] == "/dev/ttyACM0"
        assert result["new_port"] == "/dev/ttyACM2"
        assert result["changed"] is True
        on_disk = json.loads((robots_dir / "left_arm.json").read_text())
        assert on_disk["fields"]["port"] == "/dev/ttyACM2"
        # Other fields preserved
        assert on_disk["fields"]["baudrate"] == 1000000

    def test_no_op_reports_changed_false(self, mcp, dirs):
        robots_dir, _ = dirs
        _seed_profile(robots_dir, "left_arm", port="/dev/ttyACM0")
        result = _call(
            mcp,
            "assign_port_to_arm",
            {"profile_name": "left_arm", "port": "/dev/ttyACM0"},
        )
        assert result["previous_port"] == "/dev/ttyACM0"
        assert result["new_port"] == "/dev/ttyACM0"
        assert result["changed"] is False

    def test_custom_field_name(self, mcp, dirs):
        """Bi-arm profiles have port_left / port_right instead of plain port."""
        robots_dir, _ = dirs
        (robots_dir / "bi_arm.json").write_text(
            json.dumps(
                {
                    "name": "bi_arm",
                    "type": "bi_so107_follower",
                    "fields": {"port_left": "/dev/ttyACM0", "port_right": "/dev/ttyACM1"},
                    "cameras": {},
                    "rest_position": {},
                }
            )
        )
        result = _call(
            mcp,
            "assign_port_to_arm",
            {
                "profile_name": "bi_arm",
                "port": "/dev/ttyACM5",
                "field_name": "port_left",
            },
        )
        assert result["field"] == "port_left"
        assert result["new_port"] == "/dev/ttyACM5"
        on_disk = json.loads((robots_dir / "bi_arm.json").read_text())
        assert on_disk["fields"]["port_left"] == "/dev/ttyACM5"
        # port_right unchanged
        assert on_disk["fields"]["port_right"] == "/dev/ttyACM1"

    def test_unknown_profile_raises(self, mcp, dirs):
        with pytest.raises(Exception, match="not found"):
            _call(
                mcp,
                "assign_port_to_arm",
                {"profile_name": "ghost", "port": "/dev/ttyACM0"},
            )

    def test_invalid_profile_kind_raises(self, mcp, dirs):
        with pytest.raises(Exception, match="profile_kind must be"):
            _call(
                mcp,
                "assign_port_to_arm",
                {
                    "profile_name": "left_arm",
                    "port": "/dev/ttyACM0",
                    "profile_kind": "made_up",
                },
            )


# ── cross-surface shared state ────────────────────────────────────────────


class TestCrossSurfaceSharedState:
    """The MCP edit tools write to the same `~/.config/lerobot/`
    directory the GUI's Robot tab does. Profiles created via MCP show
    up in `list_robot_profiles` (which reads the same dir).
    """

    def test_mcp_create_visible_via_mcp_list(self, mcp, dirs):
        _call(
            mcp,
            "create_robot_profile",
            {"name": "via_mcp", "type": "bi_so107_follower"},
        )
        listed = _call(mcp, "list_robot_profiles", {})
        names = {p["name"] for p in listed["profiles"]}
        assert "via_mcp" in names

    def test_full_lifecycle_create_update_rename_delete(self, mcp, dirs):
        """End-to-end: create → update → rename → delete. Each step's
        on-disk state is verified.
        """
        robots_dir, _ = dirs
        _call(
            mcp,
            "create_robot_profile",
            {"name": "step1", "type": "bi_so107_follower"},
        )
        assert (robots_dir / "step1.json").exists()
        _call(
            mcp,
            "update_robot_profile",
            {
                "name": "step1",
                "type": "bi_so107_follower",
                "fields": {"port": "/dev/ttyACM7"},
            },
        )
        on_disk = json.loads((robots_dir / "step1.json").read_text())
        assert on_disk["fields"]["port"] == "/dev/ttyACM7"
        _call(
            mcp,
            "rename_robot_profile",
            {"old_name": "step1", "new_name": "step1_renamed"},
        )
        assert (robots_dir / "step1_renamed.json").exists()
        assert not (robots_dir / "step1.json").exists()
        _call(mcp, "delete_robot_profile", {"name": "step1_renamed"})
        assert not (robots_dir / "step1_renamed.json").exists()

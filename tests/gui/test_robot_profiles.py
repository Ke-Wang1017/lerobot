"""Tests for robot/teleop profile CRUD and schema introspection."""

import dataclasses
import json
from pathlib import Path
from typing import Optional
from unittest.mock import MagicMock, patch

import pytest
from fastapi import HTTPException

from lerobot.gui.api.robot import (
    ProfileData,
    _collect_all_port_assignments,
    _delete_profile,
    _introspect_fields,
    _list_profiles,
    _read_profile,
    _rename_profile,
    _stringify_type,
    _write_profile,
)

# ============================================================================
# _stringify_type
# ============================================================================


class TestStringifyType:
    """Tests for type annotation to string conversion."""

    def test_simple_types(self):
        assert _stringify_type(int) == "int"
        assert _stringify_type(float) == "float"
        assert _stringify_type(str) == "str"
        assert _stringify_type(bool) == "bool"

    def test_optional_type(self):
        # Deliberately testing Optional[X] handling, not the | None form.
        result = _stringify_type(Optional[str])  # noqa: UP045
        # Should not contain 'typing.' prefix
        assert "typing." not in result
        assert "str" in result

    def test_pathlib_type(self):
        result = _stringify_type(Path)
        assert "pathlib." not in result
        assert "Path" in result

    def test_class_string_cleaned(self):
        # Simulates "<class 'int'>" style annotations
        result = _stringify_type("<class 'int'>")
        assert result == "int"

    def test_complex_typing(self):
        result = _stringify_type(dict[str, int])
        assert "typing." not in result


# ============================================================================
# _introspect_fields
# ============================================================================


@dataclasses.dataclass
class _MockConfig:
    port: str
    baudrate: int = 115200
    gripper_bounce: bool = False
    calibration_dir: str = "/cal"  # Should be skipped
    cameras: dict = dataclasses.field(default_factory=dict)  # Should be skipped


@dataclasses.dataclass
class _MockConfigRequired:
    port: str  # Required — no default
    name: str  # Required — no default


@dataclasses.dataclass
class _MockConfigFactory:
    items: list = dataclasses.field(default_factory=list)


class TestIntrospectFields:
    """Tests for dataclass field extraction."""

    def test_extracts_fields(self):
        fields = _introspect_fields(_MockConfig)
        names = [f["name"] for f in fields]
        assert "port" in names
        assert "baudrate" in names
        assert "gripper_bounce" in names

    def test_skips_cameras_and_calibration_dir(self):
        fields = _introspect_fields(_MockConfig)
        names = [f["name"] for f in fields]
        assert "cameras" not in names
        assert "calibration_dir" not in names

    def test_required_fields_detected(self):
        fields = _introspect_fields(_MockConfigRequired)
        port_field = next(f for f in fields if f["name"] == "port")
        assert port_field["required"] is True

    def test_optional_fields_detected(self):
        fields = _introspect_fields(_MockConfig)
        baud_field = next(f for f in fields if f["name"] == "baudrate")
        assert baud_field["required"] is False
        assert baud_field["default"] == 115200

    def test_bool_default(self):
        fields = _introspect_fields(_MockConfig)
        gb_field = next(f for f in fields if f["name"] == "gripper_bounce")
        assert gb_field["default"] is False

    def test_factory_default_is_none(self):
        """Fields with default_factory should show default=None (we don't call the factory)."""
        fields = _introspect_fields(_MockConfigFactory)
        items_field = next(f for f in fields if f["name"] == "items")
        assert items_field["required"] is False
        assert items_field["default"] is None

    def test_type_str_populated(self):
        fields = _introspect_fields(_MockConfig)
        port_field = next(f for f in fields if f["name"] == "port")
        assert "str" in port_field["type_str"]

    def test_empty_dataclass(self):
        @dataclasses.dataclass
        class _Empty:
            pass

        fields = _introspect_fields(_Empty)
        assert fields == []


# ============================================================================
# Profile CRUD helpers
# ============================================================================


class TestProfileCRUD:
    """Tests for profile file operations using tmp_path."""

    def test_write_and_read_profile(self, tmp_path):
        profile = ProfileData(type="test_robot", name="my_robot", fields={"port": "/dev/ttyACM0"})
        _write_profile(tmp_path, profile)

        data = _read_profile(tmp_path, "my_robot")
        assert data["type"] == "test_robot"
        assert data["name"] == "my_robot"
        assert data["fields"]["port"] == "/dev/ttyACM0"

    def test_list_profiles(self, tmp_path):
        _write_profile(tmp_path, ProfileData(type="type_a", name="alpha"))
        _write_profile(tmp_path, ProfileData(type="type_b", name="beta"))

        profiles = _list_profiles(tmp_path)
        names = [p["name"] for p in profiles]
        assert "alpha" in names
        assert "beta" in names
        assert len(profiles) == 2

    def test_list_profiles_empty_dir(self, tmp_path):
        profiles = _list_profiles(tmp_path)
        assert profiles == []

    def test_list_profiles_skips_corrupt_json(self, tmp_path):
        (tmp_path / "good.json").write_text(json.dumps({"name": "good", "type": "t"}))
        (tmp_path / "bad.json").write_text("not valid json {{{")

        profiles = _list_profiles(tmp_path)
        assert len(profiles) == 1
        assert profiles[0]["name"] == "good"

    def test_read_profile_not_found(self, tmp_path):
        with pytest.raises(HTTPException) as exc_info:
            _read_profile(tmp_path, "nonexistent")
        assert exc_info.value.status_code == 404

    def test_delete_profile(self, tmp_path):
        _write_profile(tmp_path, ProfileData(type="t", name="deleteme"))
        assert (tmp_path / "deleteme.json").exists()

        _delete_profile(tmp_path, "deleteme")
        assert not (tmp_path / "deleteme.json").exists()

    def test_delete_profile_not_found(self, tmp_path):
        with pytest.raises(HTTPException) as exc_info:
            _delete_profile(tmp_path, "ghost")
        assert exc_info.value.status_code == 404

    def test_rename_profile(self, tmp_path):
        _write_profile(tmp_path, ProfileData(type="t", name="old_name", fields={"x": 1}))

        _rename_profile(tmp_path, "old_name", "new_name")

        assert not (tmp_path / "old_name.json").exists()
        assert (tmp_path / "new_name.json").exists()

        data = json.loads((tmp_path / "new_name.json").read_text())
        assert data["name"] == "new_name"
        assert data["fields"]["x"] == 1

    def test_rename_profile_collision(self, tmp_path):
        _write_profile(tmp_path, ProfileData(type="t", name="existing"))
        _write_profile(tmp_path, ProfileData(type="t", name="other"))

        with pytest.raises(HTTPException) as exc_info:
            _rename_profile(tmp_path, "other", "existing")
        assert exc_info.value.status_code == 409

    def test_rename_profile_source_not_found(self, tmp_path):
        with pytest.raises(HTTPException) as exc_info:
            _rename_profile(tmp_path, "ghost", "new_name")
        assert exc_info.value.status_code == 404

    def test_write_creates_directory(self, tmp_path):
        nested = tmp_path / "sub" / "dir"
        profile = ProfileData(type="t", name="deep")
        _write_profile(nested, profile)
        assert (nested / "deep.json").exists()

    def test_roundtrip_preserves_cameras(self, tmp_path):
        cameras = {"cam_0": {"type": "opencv", "index": 0}, "cam_1": {"type": "realsense", "serial": "123"}}
        profile = ProfileData(type="t", name="cam_test", cameras=cameras)
        _write_profile(tmp_path, profile)

        data = _read_profile(tmp_path, "cam_test")
        assert data["cameras"] == cameras


# ============================================================================
# _collect_all_port_assignments
# ============================================================================


@dataclasses.dataclass
class _FakeRobotConfig:
    port: str = "/dev/ttyACM0"
    baudrate: int = 115200
    cameras: dict = dataclasses.field(default_factory=dict)


@dataclasses.dataclass
class _FakeTeleopConfig:
    port: str = "/dev/ttyACM1"
    cameras: dict = dataclasses.field(default_factory=dict)


@dataclasses.dataclass
class _NoPortConfig:
    baudrate: int = 115200
    cameras: dict = dataclasses.field(default_factory=dict)


class TestCollectAllPortAssignments:
    """Tests for port assignment extraction from saved profiles."""

    def _setup_dirs(self, tmp_path):
        """Create robot and teleop profile dirs under tmp_path."""
        robot_dir = tmp_path / "robots"
        teleop_dir = tmp_path / "teleops"
        robot_dir.mkdir()
        teleop_dir.mkdir()
        return robot_dir, teleop_dir

    def test_extracts_port_from_robot_profile(self, tmp_path):
        robot_dir, teleop_dir = self._setup_dirs(tmp_path)
        profile = {"name": "arm1", "type": "fake_robot", "fields": {"port": "/dev/ttyACM0"}}
        (robot_dir / "arm1.json").write_text(json.dumps(profile))

        mock_robot_base = MagicMock()
        mock_robot_base.get_known_choices.return_value = {"fake_robot": _FakeRobotConfig}
        mock_teleop_base = MagicMock()
        mock_teleop_base.get_known_choices.return_value = {}

        with (
            patch("lerobot.gui.api.robot.ROBOT_PROFILES_DIR", robot_dir),
            patch("lerobot.gui.api.robot.TELEOP_PROFILES_DIR", teleop_dir),
            patch("lerobot.gui.api.robot.RobotConfig", mock_robot_base, create=True),
            patch("lerobot.gui.api.robot.TeleoperatorConfig", mock_teleop_base, create=True),
            patch("lerobot.gui.api.robot._ensure_configs_loaded"),
        ):
            # Need to patch the local imports inside the function
            with patch.dict(
                "sys.modules",
                {
                    "lerobot.robots.config": MagicMock(RobotConfig=mock_robot_base),
                    "lerobot.teleoperators.config": MagicMock(TeleoperatorConfig=mock_teleop_base),
                },
            ):
                assignments = _collect_all_port_assignments()

        assert len(assignments) == 1
        assert assignments[0]["port"] == "/dev/ttyACM0"
        assert assignments[0]["profile_name"] == "arm1"
        assert assignments[0]["profile_kind"] == "robot"
        assert assignments[0]["field_name"] == "port"

    def test_extracts_ports_from_both_kinds(self, tmp_path):
        robot_dir, teleop_dir = self._setup_dirs(tmp_path)
        (robot_dir / "arm.json").write_text(
            json.dumps({"name": "arm", "type": "fake_robot", "fields": {"port": "/dev/ttyACM0"}})
        )
        (teleop_dir / "leader.json").write_text(
            json.dumps({"name": "leader", "type": "fake_teleop", "fields": {"port": "/dev/ttyACM1"}})
        )

        mock_robot_base = MagicMock()
        mock_robot_base.get_known_choices.return_value = {"fake_robot": _FakeRobotConfig}
        mock_teleop_base = MagicMock()
        mock_teleop_base.get_known_choices.return_value = {"fake_teleop": _FakeTeleopConfig}

        with (
            patch("lerobot.gui.api.robot.ROBOT_PROFILES_DIR", robot_dir),
            patch("lerobot.gui.api.robot.TELEOP_PROFILES_DIR", teleop_dir),
            patch("lerobot.gui.api.robot._ensure_configs_loaded"),
            patch.dict(
                "sys.modules",
                {
                    "lerobot.robots.config": MagicMock(RobotConfig=mock_robot_base),
                    "lerobot.teleoperators.config": MagicMock(TeleoperatorConfig=mock_teleop_base),
                },
            ),
        ):
            assignments = _collect_all_port_assignments()

        assert len(assignments) == 2
        ports = {a["port"] for a in assignments}
        assert ports == {"/dev/ttyACM0", "/dev/ttyACM1"}

    def test_skips_unknown_type(self, tmp_path):
        robot_dir, teleop_dir = self._setup_dirs(tmp_path)
        (robot_dir / "mystery.json").write_text(
            json.dumps({"name": "mystery", "type": "unknown_type", "fields": {"port": "/dev/ttyACM0"}})
        )

        mock_robot_base = MagicMock()
        mock_robot_base.get_known_choices.return_value = {"fake_robot": _FakeRobotConfig}
        mock_teleop_base = MagicMock()
        mock_teleop_base.get_known_choices.return_value = {}

        with (
            patch("lerobot.gui.api.robot.ROBOT_PROFILES_DIR", robot_dir),
            patch("lerobot.gui.api.robot.TELEOP_PROFILES_DIR", teleop_dir),
            patch("lerobot.gui.api.robot._ensure_configs_loaded"),
            patch.dict(
                "sys.modules",
                {
                    "lerobot.robots.config": MagicMock(RobotConfig=mock_robot_base),
                    "lerobot.teleoperators.config": MagicMock(TeleoperatorConfig=mock_teleop_base),
                },
            ),
        ):
            assignments = _collect_all_port_assignments()

        assert assignments == []

    def test_skips_empty_port_value(self, tmp_path):
        robot_dir, teleop_dir = self._setup_dirs(tmp_path)
        (robot_dir / "noport.json").write_text(
            json.dumps({"name": "noport", "type": "fake_robot", "fields": {"port": ""}})
        )

        mock_robot_base = MagicMock()
        mock_robot_base.get_known_choices.return_value = {"fake_robot": _FakeRobotConfig}
        mock_teleop_base = MagicMock()
        mock_teleop_base.get_known_choices.return_value = {}

        with (
            patch("lerobot.gui.api.robot.ROBOT_PROFILES_DIR", robot_dir),
            patch("lerobot.gui.api.robot.TELEOP_PROFILES_DIR", teleop_dir),
            patch("lerobot.gui.api.robot._ensure_configs_loaded"),
            patch.dict(
                "sys.modules",
                {
                    "lerobot.robots.config": MagicMock(RobotConfig=mock_robot_base),
                    "lerobot.teleoperators.config": MagicMock(TeleoperatorConfig=mock_teleop_base),
                },
            ),
        ):
            assignments = _collect_all_port_assignments()

        assert assignments == []

    def test_skips_non_port_fields(self, tmp_path):
        robot_dir, teleop_dir = self._setup_dirs(tmp_path)
        (robot_dir / "noport.json").write_text(
            json.dumps({"name": "noport", "type": "no_port", "fields": {"baudrate": 115200}})
        )

        mock_robot_base = MagicMock()
        mock_robot_base.get_known_choices.return_value = {"no_port": _NoPortConfig}
        mock_teleop_base = MagicMock()
        mock_teleop_base.get_known_choices.return_value = {}

        with (
            patch("lerobot.gui.api.robot.ROBOT_PROFILES_DIR", robot_dir),
            patch("lerobot.gui.api.robot.TELEOP_PROFILES_DIR", teleop_dir),
            patch("lerobot.gui.api.robot._ensure_configs_loaded"),
            patch.dict(
                "sys.modules",
                {
                    "lerobot.robots.config": MagicMock(RobotConfig=mock_robot_base),
                    "lerobot.teleoperators.config": MagicMock(TeleoperatorConfig=mock_teleop_base),
                },
            ),
        ):
            assignments = _collect_all_port_assignments()

        assert assignments == []

    def test_skips_corrupt_json(self, tmp_path):
        robot_dir, teleop_dir = self._setup_dirs(tmp_path)
        (robot_dir / "corrupt.json").write_text("not json {{{")

        mock_robot_base = MagicMock()
        mock_robot_base.get_known_choices.return_value = {"fake_robot": _FakeRobotConfig}
        mock_teleop_base = MagicMock()
        mock_teleop_base.get_known_choices.return_value = {}

        with (
            patch("lerobot.gui.api.robot.ROBOT_PROFILES_DIR", robot_dir),
            patch("lerobot.gui.api.robot.TELEOP_PROFILES_DIR", teleop_dir),
            patch("lerobot.gui.api.robot._ensure_configs_loaded"),
            patch.dict(
                "sys.modules",
                {
                    "lerobot.robots.config": MagicMock(RobotConfig=mock_robot_base),
                    "lerobot.teleoperators.config": MagicMock(TeleoperatorConfig=mock_teleop_base),
                },
            ),
        ):
            assignments = _collect_all_port_assignments()

        assert assignments == []


class TestPortConflictSuppression:
    """The port-conflict UI lives in robot.js and there's no JS test runner, so this is a
    static guard over the source.

    Two profiles that map the same physical port to the same field are a shared setup —
    one device wired to one role, referenced by both (e.g. "white" and "white_pred" sharing
    arms). renderPortList must treat that as agreement, not flag it as "in use by" the other
    profile. This guard fails if the suppression is dropped in a future refactor.
    """

    def _render_port_list_source(self) -> str:
        import lerobot.gui as gui_pkg

        robot_js = Path(gui_pkg.__file__).parent / "static" / "robot.js"
        src = robot_js.read_text()
        start = src.index("function renderPortList")
        end = src.index("\nfunction ", start + 1)
        return src[start:end]

    def test_render_port_list_suppresses_shared_setup(self):
        body = self._render_port_list_source()
        assert "curPortToField" in body, "shared-setup suppression removed from renderPortList"
        assert "curPortToField[a.port] === a.field_name" in body

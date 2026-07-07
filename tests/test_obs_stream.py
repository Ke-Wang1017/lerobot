"""Tests for the robot observation stream (shared memory)."""

import os
from dataclasses import dataclass
from unittest.mock import patch

import numpy as np
import pytest

from lerobot.robots.config import RobotConfig
from lerobot.robots.obs_stream import (
    ENV_VAR,
    SHM_PREFIX,
    ObservationStream,
    ObservationStreamReader,
    cleanup_stale_streams,
)
from lerobot.robots.robot import Robot

# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def simple_features():
    obs = {"j1.pos": float, "j2.pos": float, "cam": (240, 320, 3)}
    act = {"j1.pos": float, "j2.pos": float}
    return obs, act


@pytest.fixture
def stream(simple_features):
    obs_ft, act_ft = simple_features
    s = ObservationStream(obs_ft, act_ft)
    yield s
    s.cleanup()


@pytest.fixture
def obs_dict():
    return {
        "j1.pos": 10.5,
        "j2.pos": -3.2,
        "cam": np.random.randint(0, 255, (240, 320, 3), dtype=np.uint8),
    }


@pytest.fixture
def action_dict():
    return {"j1.pos": 11.0, "j2.pos": -2.5}


# ============================================================================
# Writer / Reader round-trip
# ============================================================================


class TestRoundTrip:
    def test_scalar_obs(self, stream, obs_dict):
        stream.write_obs(obs_dict)
        reader = ObservationStreamReader()
        result = reader.read_obs()
        assert result is not None
        obs, ts = result
        assert abs(obs["j1.pos"] - 10.5) < 1e-4
        assert abs(obs["j2.pos"] - (-3.2)) < 1e-4
        assert ts > 0
        reader.close()

    def test_action(self, stream, obs_dict, action_dict):
        stream.write_obs(obs_dict)  # meta block needs data
        stream.write_action(action_dict)
        reader = ObservationStreamReader()
        result = reader.read_action()
        assert result is not None
        act, ts = result
        assert abs(act["j1.pos"] - 11.0) < 1e-4
        assert abs(act["j2.pos"] - (-2.5)) < 1e-4
        reader.close()

    def test_image(self, stream, obs_dict):
        stream.write_obs(obs_dict)
        reader = ObservationStreamReader()
        result = reader.read_image("cam")
        assert result is not None
        img, ts = result
        assert img.shape == (240, 320, 3)
        assert img.dtype == np.uint8
        assert np.array_equal(img, obs_dict["cam"])
        reader.close()

    def test_missing_camera_returns_none(self, stream, obs_dict):
        stream.write_obs(obs_dict)
        reader = ObservationStreamReader()
        assert reader.read_image("nonexistent") is None
        reader.close()

    def test_no_data_returns_none(self, stream):
        """Reader attached but no write_obs called yet."""
        reader = ObservationStreamReader()
        assert reader.read_obs() is None
        assert reader.read_action() is None
        assert reader.read_image("cam") is None
        reader.close()


# ============================================================================
# Sequence counter
# ============================================================================


class TestSequenceCounter:
    def test_seq_increments(self, stream, obs_dict):
        reader = ObservationStreamReader()
        assert reader.image_seq("cam") == 0

        stream.write_obs(obs_dict)
        assert reader.image_seq("cam") == 1

        stream.write_obs(obs_dict)
        assert reader.image_seq("cam") == 2
        reader.close()

    def test_seq_unchanged_without_write(self, stream, obs_dict):
        stream.write_obs(obs_dict)
        reader = ObservationStreamReader()
        seq1 = reader.image_seq("cam")
        seq2 = reader.image_seq("cam")
        assert seq1 == seq2
        reader.close()


# ============================================================================
# Layout metadata
# ============================================================================


class TestMetadata:
    def test_reader_sees_layout(self, stream, obs_dict):
        stream.write_obs(obs_dict)
        reader = ObservationStreamReader()
        assert sorted(reader.obs_scalar_keys) == ["j1.pos", "j2.pos"]
        assert sorted(reader.action_keys) == ["j1.pos", "j2.pos"]
        assert "cam" in reader.image_keys
        assert reader.image_keys["cam"] == [240, 320, 3]
        reader.close()


# ============================================================================
# __init_subclass__ wrapping with mock robots
# ============================================================================


@dataclass
class _MockConfig(RobotConfig):
    pass


class _SimpleRobot(Robot):
    config_class = _MockConfig
    name = "test_simple"

    def __init__(self):
        self._connected = False

    @property
    def observation_features(self):
        return {"j.pos": float, "cam": (120, 160, 3)}

    @property
    def action_features(self):
        return {"j.pos": float}

    @property
    def is_connected(self):
        return self._connected

    @property
    def is_calibrated(self):
        return True

    def connect(self, calibrate=True):
        self._connected = True

    def disconnect(self):
        self._connected = False

    def calibrate(self):
        pass

    def configure(self):
        pass

    def get_observation(self):
        return {"j.pos": 1.0, "cam": np.zeros((120, 160, 3), dtype=np.uint8)}

    def send_action(self, action):
        return action


class _SubRobot(Robot):
    """Sub-robot for composite testing."""

    config_class = _MockConfig
    name = "test_sub"

    def __init__(self, prefix):
        self._connected = False
        self._prefix = prefix

    @property
    def observation_features(self):
        return {f"{self._prefix}_j.pos": float}

    @property
    def action_features(self):
        return self.observation_features

    @property
    def is_connected(self):
        return self._connected

    @property
    def is_calibrated(self):
        return True

    def connect(self, calibrate=True):
        self._connected = True

    def disconnect(self):
        self._connected = False

    def calibrate(self):
        pass

    def configure(self):
        pass

    def get_observation(self):
        return {f"{self._prefix}_j.pos": 5.0}

    def send_action(self, action):
        return action


class _CompositeRobot(Robot):
    """Composite robot wrapping two sub-robots."""

    config_class = _MockConfig
    name = "test_composite"

    def __init__(self):
        self.left = _SubRobot("left")
        self.right = _SubRobot("right")

    @property
    def observation_features(self):
        return {**self.left.observation_features, **self.right.observation_features}

    @property
    def action_features(self):
        return {**self.left.action_features, **self.right.action_features}

    @property
    def is_connected(self):
        return self.left.is_connected and self.right.is_connected

    @property
    def is_calibrated(self):
        return True

    def connect(self, calibrate=True):
        self.left.connect()
        self.right.connect()

    def disconnect(self):
        self.left.disconnect()
        self.right.disconnect()

    def calibrate(self):
        pass

    def configure(self):
        pass

    def get_observation(self):
        obs = {}
        obs.update(self.left.get_observation())
        obs.update(self.right.get_observation())
        return obs

    def send_action(self, action):
        self.left.send_action({k: v for k, v in action.items() if "left" in k})
        self.right.send_action({k: v for k, v in action.items() if "right" in k})
        return action


class TestInitSubclassWrapping:
    """Test that Robot.__init_subclass__ wrapping works correctly."""

    def test_methods_are_wrapped(self):
        """connect/send_action/disconnect should be closures.

        NOTE: get_observation is NOT wrapped — obs stream writes are handled
        by ObservationStreamWriterStep in the processor pipeline.
        """
        for method_name in ("connect", "disconnect", "send_action"):
            fn = _SimpleRobot.__dict__[method_name]
            assert fn.__closure__ is not None, f"{method_name} should be wrapped"
        # get_observation should NOT be wrapped
        assert _SimpleRobot.__dict__["get_observation"].__closure__ is None

    def test_no_stream_without_env_var(self):
        import lerobot.robots.obs_stream as mod

        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop(ENV_VAR, None)
            robot = _SimpleRobot()
            robot.connect()
            assert mod._active_stream is None
            robot.disconnect()

    def test_stream_created_with_env_var(self):
        import lerobot.robots.obs_stream as mod

        with patch.dict(os.environ, {ENV_VAR: "1"}):
            robot = _SimpleRobot()
            robot.connect()
            assert mod._active_stream is not None
            assert mod._active_robot_id == id(robot)

            # Obs writes are done by the processor step, not get_observation.
            # Manually write to verify stream is functional.
            obs = robot.get_observation()
            mod._active_stream.write_obs(obs)
            reader = ObservationStreamReader()
            result = reader.read_obs()
            assert result is not None
            assert abs(result[0]["j.pos"] - 1.0) < 1e-4
            reader.close()

            robot.disconnect()
            assert mod._active_stream is None

    def test_composite_only_outermost_publishes(self):
        import lerobot.robots.obs_stream as mod

        with patch.dict(os.environ, {ENV_VAR: "1"}):
            robot = _CompositeRobot()
            robot.connect()

            # Only the composite should own the stream
            assert mod._active_robot_id == id(robot)
            assert mod._active_robot_id != id(robot.left)
            assert mod._active_robot_id != id(robot.right)

            # Manually write composite obs to the stream
            obs = robot.get_observation()
            mod._active_stream.write_obs(obs)
            reader = ObservationStreamReader()
            result = reader.read_obs()
            assert result is not None
            assert "left_j.pos" in result[0]
            assert "right_j.pos" in result[0]
            reader.close()

            robot.disconnect()
            assert mod._active_stream is None


# ============================================================================
# Multiple cameras / various feature shapes
# ============================================================================


class TestMultipleCameras:
    def test_four_cameras(self):
        obs_ft = {f"j{i}.pos": float for i in range(14)}
        for cam in ["front", "top", "left_wrist", "right_wrist"]:
            obs_ft[cam] = (720, 1280, 3)
        act_ft = {f"j{i}.pos": float for i in range(14)}

        stream = ObservationStream(obs_ft, act_ft)
        obs = {f"j{i}.pos": float(i) for i in range(14)}
        for cam in ["front", "top", "left_wrist", "right_wrist"]:
            obs[cam] = np.zeros((720, 1280, 3), dtype=np.uint8)
        stream.write_obs(obs)

        reader = ObservationStreamReader()
        assert len(reader.image_keys) == 4
        for cam in ["front", "top", "left_wrist", "right_wrist"]:
            result = reader.read_image(cam)
            assert result is not None
            assert result[0].shape == (720, 1280, 3)
        reader.close()
        stream.cleanup()

    def test_no_cameras(self):
        """Robots with only scalar features (no cameras)."""
        stream = ObservationStream({"j.pos": float}, {"j.pos": float})
        stream.write_obs({"j.pos": 42.0})

        reader = ObservationStreamReader()
        assert len(reader.image_keys) == 0
        result = reader.read_obs()
        assert result is not None
        assert abs(result[0]["j.pos"] - 42.0) < 1e-4
        reader.close()
        stream.cleanup()


# ============================================================================
# Stale stream detection (crash → restart)
# ============================================================================


class TestStaleStreamDetection:
    """When a teleop session crashes and restarts, the GUI reader must detect
    that shared memory was recreated (unlinked + new allocation) and re-attach
    to the fresh segments rather than reading stale data from the old mapping.
    """

    def test_reader_reads_stale_after_recreate_without_reattach(self, simple_features):
        """Demonstrate the problem: reader on old mapping reads stale data."""
        obs_ft, act_ft = simple_features
        stream1 = ObservationStream(obs_ft, act_ft)
        stream1.write_obs({"j1.pos": 1.0, "j2.pos": 2.0, "cam": np.zeros((240, 320, 3), dtype=np.uint8)})

        reader = ObservationStreamReader()
        result = reader.read_obs()
        assert result is not None
        assert abs(result[0]["j1.pos"] - 1.0) < 1e-4

        # Simulate crash + restart: old stream cleaned up, new one created
        stream1.cleanup()
        stream2 = ObservationStream(obs_ft, act_ft)
        stream2.write_obs({"j1.pos": 99.0, "j2.pos": 88.0, "cam": np.zeros((240, 320, 3), dtype=np.uint8)})

        # Old reader still mapped to unlinked segments — reads stale data
        result = reader.read_obs()
        assert result is not None
        assert abs(result[0]["j1.pos"] - 1.0) < 1e-4  # Still the old value!

        reader.close()
        stream2.cleanup()

    def test_inode_changes_on_recreate(self, simple_features):
        """The /dev/shm inode changes when segments are recreated, which is
        what the GUI reader uses to detect staleness."""
        obs_ft, act_ft = simple_features
        stream1 = ObservationStream(obs_ft, act_ft)
        stream1.write_obs({"j1.pos": 1.0, "j2.pos": 2.0, "cam": np.zeros((240, 320, 3), dtype=np.uint8)})

        ino1 = os.stat("/dev/shm/lerobot_obs_meta").st_ino

        stream1.cleanup()
        stream2 = ObservationStream(obs_ft, act_ft)

        ino2 = os.stat("/dev/shm/lerobot_obs_meta").st_ino
        assert ino1 != ino2, "Inode should change after unlink + recreate"

        stream2.cleanup()

    def test_new_reader_after_recreate_sees_fresh_data(self, simple_features):
        """A fresh reader attached after recreate reads the new data."""
        obs_ft, act_ft = simple_features
        stream1 = ObservationStream(obs_ft, act_ft)
        stream1.write_obs({"j1.pos": 1.0, "j2.pos": 2.0, "cam": np.zeros((240, 320, 3), dtype=np.uint8)})

        reader1 = ObservationStreamReader()
        result = reader1.read_obs()
        assert abs(result[0]["j1.pos"] - 1.0) < 1e-4
        reader1.close()

        # Recreate stream (simulates new teleop session)
        stream1.cleanup()
        stream2 = ObservationStream(obs_ft, act_ft)
        stream2.write_obs({"j1.pos": 99.0, "j2.pos": 88.0, "cam": np.zeros((240, 320, 3), dtype=np.uint8)})

        # New reader sees new data
        reader2 = ObservationStreamReader()
        result = reader2.read_obs()
        assert abs(result[0]["j1.pos"] - 99.0) < 1e-4
        reader2.close()
        stream2.cleanup()

    def test_new_stream_header_zeroed_before_first_write(self, simple_features):
        """After recreate but before first write, read returns None (not stale)."""
        obs_ft, act_ft = simple_features
        stream1 = ObservationStream(obs_ft, act_ft)
        stream1.write_obs({"j1.pos": 1.0, "j2.pos": 2.0, "cam": np.zeros((240, 320, 3), dtype=np.uint8)})
        stream1.cleanup()

        stream2 = ObservationStream(obs_ft, act_ft)
        # No write yet — header is zeroed
        reader = ObservationStreamReader()
        assert reader.read_obs() is None
        assert reader.read_image("cam") is None
        reader.close()
        stream2.cleanup()


# ============================================================================
# cleanup_stale_streams() — used by GUI lifecycle hooks
# ============================================================================


def _touch(path):
    path.write_bytes(b"")


class TestCleanupStaleStreams:
    """Sweep used by GUI startup/shutdown to remove shm leaked by a writer
    that died via SIGKILL/SIGABRT/segfault before its disconnect()
    cleanup ran. Tested against tmp_path so it doesn't touch /dev/shm."""

    def test_empty_dir_returns_zero(self, tmp_path):
        assert cleanup_stale_streams(tmp_path) == 0

    def test_nonexistent_dir_returns_zero(self, tmp_path):
        # Don't crash if the shm path doesn't exist (macOS, weird containers).
        missing = tmp_path / "does-not-exist"
        assert cleanup_stale_streams(missing) == 0

    def test_removes_only_matching_files(self, tmp_path):
        # Files matching SHM_PREFIX → removed.
        for suffix in ("meta", "obs", "act", "img_top", "img_left_wrist"):
            _touch(tmp_path / f"{SHM_PREFIX}{suffix}")
        # Files not matching → preserved (other tools' shm, user data).
        unrelated = [
            tmp_path / "some_other_app",
            tmp_path / "lerobot_other_thing",  # different prefix
            tmp_path / "lerobotobs_typo",  # missing underscore
        ]
        for p in unrelated:
            _touch(p)

        n = cleanup_stale_streams(tmp_path)

        assert n == 5
        assert not list(tmp_path.glob(f"{SHM_PREFIX}*"))
        for p in unrelated:
            assert p.exists(), f"unrelated file {p.name} was wrongly removed"

    def test_idempotent(self, tmp_path):
        _touch(tmp_path / f"{SHM_PREFIX}meta")
        assert cleanup_stale_streams(tmp_path) == 1
        assert cleanup_stale_streams(tmp_path) == 0

    def test_unremovable_file_logs_and_continues(self, tmp_path, caplog):
        # A read-only dir prevents unlink. Sweep should log a warning and
        # continue rather than abort the whole pass.
        sub = tmp_path / "ro"
        sub.mkdir()
        _touch(sub / f"{SHM_PREFIX}stuck")
        sub.chmod(0o500)
        try:
            n = cleanup_stale_streams(sub)
        finally:
            sub.chmod(0o700)  # restore so pytest can clean up

        assert n == 0
        assert (sub / f"{SHM_PREFIX}stuck").exists()
        assert any("could not remove" in r.message for r in caplog.records)

    def test_count_reflects_only_actually_removed(self, tmp_path):
        # Mix of removable + unremovable to verify the count stays accurate.
        ok = tmp_path / f"{SHM_PREFIX}removable"
        _touch(ok)
        # Plant a path that looks like a match but is a directory — unlink()
        # refuses, so it's not "removed" and shouldn't be counted.
        wedge = tmp_path / f"{SHM_PREFIX}directory_not_file"
        wedge.mkdir()

        n = cleanup_stale_streams(tmp_path)

        assert n == 1
        assert not ok.exists()
        assert wedge.exists()
        os.rmdir(wedge)  # clean up the planted dir for pytest

    def test_default_dir_is_dev_shm(self, simple_features):
        """Smoke test against the real /dev/shm: create a real stream, sweep,
        verify the segments are gone. This is the path the GUI lifecycle
        hooks actually use."""
        obs_ft, act_ft = simple_features
        stream = ObservationStream(obs_ft, act_ft)
        # At this point /dev/shm/lerobot_obs_* exists.
        n = cleanup_stale_streams()  # default arg = /dev/shm
        # Stream creates 4 blocks (meta, obs, act, cam=1 image) → 4 files.
        assert n >= 4
        # The stream object holds stale handles now; close them quietly so
        # cleanup() doesn't raise.
        try:
            stream.cleanup()
        except Exception:
            pass

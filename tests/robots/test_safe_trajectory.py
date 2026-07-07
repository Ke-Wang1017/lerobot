"""Tests for the safe-trajectory recording and replay module."""

import threading
import time
from unittest.mock import MagicMock

import pytest

from lerobot.robots.safe_trajectory import (
    SCHEMA_VERSION,
    TrajectoryRecorder,
    replay_trajectory,
    validate_trajectory,
)


def _make_mock_robot(
    initial_positions: dict[str, float],
    action_keys: list[str] | None = None,
):
    """Mock Robot with get_observation/action_features/send_action."""
    robot = MagicMock()
    robot.get_observation.return_value = {**initial_positions, "cam_front": "img"}
    if action_keys is None:
        action_keys = list(initial_positions.keys())
    robot.action_features = dict.fromkeys(action_keys, float)
    robot.sent_actions = []
    robot.send_action.side_effect = lambda a: robot.sent_actions.append(dict(a))
    return robot


class TestTrajectoryRecorder:
    def test_basic_record_returns_frames(self):
        robot = _make_mock_robot({"shoulder.pos": 1.0, "elbow.pos": 2.0})
        rec = TrajectoryRecorder(robot, fps=100)
        rec.start()
        time.sleep(0.15)  # ~15 frames at 100 Hz
        traj = rec.stop()
        assert traj["schema_version"] == SCHEMA_VERSION
        assert traj["fps"] == 100
        assert traj["joints"] == ["elbow.pos", "shoulder.pos"]  # sorted
        assert len(traj["timestamps"]) >= 5
        assert len(traj["positions"]) == len(traj["timestamps"])
        # All timestamps strictly increasing
        ts = traj["timestamps"]
        for a, b in zip(ts, ts[1:], strict=False):
            assert b > a

    def test_excludes_non_action_keys(self):
        robot = _make_mock_robot({"a.pos": 1.0, "b.pos": 2.0})
        # Add stray observation keys that aren't in action_features
        robot.get_observation.return_value = {
            "a.pos": 1.0,
            "b.pos": 2.0,
            "cam_front": "img",
            "extra": 999,
        }
        rec = TrajectoryRecorder(robot, fps=100)
        rec.start()
        time.sleep(0.05)
        traj = rec.stop()
        assert traj["joints"] == ["a.pos", "b.pos"]
        for row in traj["positions"]:
            assert len(row) == 2

    def test_cancel_does_not_raise(self):
        robot = _make_mock_robot({"a.pos": 1.0})
        rec = TrajectoryRecorder(robot, fps=50)
        rec.start()
        time.sleep(0.02)
        rec.cancel()  # should join cleanly

    def test_invalid_fps_rejected(self):
        robot = _make_mock_robot({"a.pos": 1.0})
        with pytest.raises(ValueError):
            TrajectoryRecorder(robot, fps=0)
        with pytest.raises(ValueError):
            TrajectoryRecorder(robot, fps=-5)

    def test_double_start_rejected(self):
        robot = _make_mock_robot({"a.pos": 1.0})
        rec = TrajectoryRecorder(robot, fps=50)
        rec.start()
        try:
            with pytest.raises(RuntimeError):
                rec.start()
        finally:
            rec.cancel()

    def test_stopped_robot_thread_is_dead(self):
        robot = _make_mock_robot({"a.pos": 1.0})
        rec = TrajectoryRecorder(robot, fps=100)
        rec.start()
        time.sleep(0.05)
        assert rec.is_running
        rec.stop()
        assert not rec.is_running


class TestValidateTrajectory:
    def _good(self):
        return {
            "schema_version": SCHEMA_VERSION,
            "fps": 30,
            "joints": ["a", "b"],
            "timestamps": [0.0, 0.033],
            "positions": [[1.0, 2.0], [1.1, 2.1]],
        }

    def test_accepts_well_formed(self):
        validate_trajectory(self._good())

    def test_rejects_missing_keys(self):
        for key in ("fps", "joints", "timestamps", "positions", "schema_version"):
            traj = self._good()
            del traj[key]
            with pytest.raises(ValueError, match="missing required keys"):
                validate_trajectory(traj)

    def test_rejects_bad_schema_version(self):
        traj = self._good()
        traj["schema_version"] = 99
        with pytest.raises(ValueError, match="schema_version"):
            validate_trajectory(traj)

    def test_rejects_length_mismatch(self):
        traj = self._good()
        traj["positions"] = [[1.0, 2.0]]  # only 1 row, but 2 timestamps
        with pytest.raises(ValueError, match="rows"):
            validate_trajectory(traj)

    def test_rejects_joint_count_mismatch(self):
        traj = self._good()
        traj["positions"] = [[1.0], [1.1]]  # 1 value/row but 2 joints
        with pytest.raises(ValueError, match="values"):
            validate_trajectory(traj)


class TestReplayTrajectory:
    def test_basic_replay_sends_each_frame(self):
        robot = _make_mock_robot({"a.pos": 0.0, "b.pos": 0.0})
        traj = {
            "schema_version": SCHEMA_VERSION,
            "fps": 100,
            "joints": ["a.pos", "b.pos"],
            "timestamps": [0.0, 0.01, 0.02],
            "positions": [[1.0, 2.0], [1.1, 2.1], [1.2, 2.2]],
        }
        # Skip the ramp so the test stays fast and the assertion is exact.
        replay_trajectory(robot, traj, ramp_to_start_s=0.0)
        # Three replay frames sent (no ramp).
        assert len(robot.sent_actions) == 3
        assert robot.sent_actions[0] == {"a.pos": 1.0, "b.pos": 2.0}
        assert robot.sent_actions[-1] == {"a.pos": 1.2, "b.pos": 2.2}

    def test_ramp_then_replay(self):
        robot = _make_mock_robot({"a.pos": 0.0})
        traj = {
            "schema_version": SCHEMA_VERSION,
            "fps": 100,
            "joints": ["a.pos"],
            "timestamps": [0.0, 0.01],
            "positions": [[10.0], [11.0]],
        }
        replay_trajectory(robot, traj, ramp_to_start_s=0.1)
        # ramp at 50 Hz × 0.1s = 5 ramp steps + 2 replay frames = 7
        assert len(robot.sent_actions) >= 7
        # First ramp action shouldn't teleport — final value should be 10.0 (start of traj)
        ramp_last = robot.sent_actions[-3]
        assert ramp_last["a.pos"] == pytest.approx(10.0, abs=1e-9)

    def test_rejects_missing_joint_in_observation_during_ramp(self):
        # Robot's action_features advertises a.pos and b.pos, but its
        # get_observation only returns a.pos — a bus-read miss for b.pos.
        # The ramp must fail loudly rather than silently collapse to an
        # instant jump on b.pos (start_pose missing → fallback to target).
        robot = _make_mock_robot({"a.pos": 0.0, "b.pos": 0.0})
        robot.get_observation.return_value = {"a.pos": 0.0, "cam_front": "img"}
        traj = {
            "schema_version": SCHEMA_VERSION,
            "fps": 30,
            "joints": ["a.pos", "b.pos"],
            "timestamps": [0.0],
            "positions": [[10.0, 20.0]],
        }
        with pytest.raises(RuntimeError, match="Cannot ramp.*missing joints"):
            replay_trajectory(robot, traj, ramp_to_start_s=0.1)

    def test_rejects_unknown_joints(self):
        robot = _make_mock_robot({"a.pos": 0.0})
        traj = {
            "schema_version": SCHEMA_VERSION,
            "fps": 30,
            "joints": ["a.pos", "ghost.pos"],
            "timestamps": [0.0],
            "positions": [[1.0, 2.0]],
        }
        with pytest.raises(ValueError, match="not in robot.action_features"):
            replay_trajectory(robot, traj, ramp_to_start_s=0.0)

    def test_empty_trajectory_noop(self):
        robot = _make_mock_robot({"a.pos": 0.0})
        traj = {
            "schema_version": SCHEMA_VERSION,
            "fps": 30,
            "joints": ["a.pos"],
            "timestamps": [],
            "positions": [],
        }
        replay_trajectory(robot, traj, ramp_to_start_s=0.0)
        assert robot.sent_actions == []

    def test_concurrent_recorder_does_not_corrupt(self):
        """Two recorders on independent mock robots shouldn't share state."""
        r1 = _make_mock_robot({"a.pos": 1.0})
        r2 = _make_mock_robot({"a.pos": 5.0})
        rec1 = TrajectoryRecorder(r1, fps=50)
        rec2 = TrajectoryRecorder(r2, fps=50)
        rec1.start()
        rec2.start()
        time.sleep(0.05)
        t1 = rec1.stop()
        t2 = rec2.stop()
        # Each recorder sees only its own robot's positions
        assert all(row == [1.0] for row in t1["positions"])
        assert all(row == [5.0] for row in t2["positions"])


def test_thread_count_drops_after_stop():
    """Sanity check: the sampler thread is daemon and joins cleanly."""
    before = threading.active_count()
    robot = _make_mock_robot({"a.pos": 0.0})
    rec = TrajectoryRecorder(robot, fps=100)
    rec.start()
    time.sleep(0.02)
    rec.stop()
    # Give the OS a moment to reclaim the thread
    time.sleep(0.02)
    after = threading.active_count()
    assert after == before

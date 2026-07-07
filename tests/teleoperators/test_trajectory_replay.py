"""Tests for the trajectory_replay teleoperator.

The chunk-aware bits (``get_action_with_horizon`` returning an
``ActionChunk``) replaced the previous in-teleop ``lookahead_s`` +
``simulate_chunk_size`` features. Lookahead now lives in the consumer
(``SO107FollowerPredictive``) — the teleop just exposes intent.
"""

import json
from pathlib import Path
from unittest.mock import patch

import pytest

from lerobot.robots.safe_trajectory import SCHEMA_VERSION
from lerobot.teleoperators.trajectory_replay import (
    TrajectoryReplayTeleop,
    TrajectoryReplayTeleopConfig,
)
from lerobot.types import ActionChunk


def _write_trajectory(tmp_path: Path, frames: list[tuple[float, list[float]]], joints: list[str]) -> Path:
    """Write a minimal valid trajectory JSON and return its path."""
    timestamps = [t for t, _ in frames]
    positions = [p for _, p in frames]
    traj = {
        "schema_version": SCHEMA_VERSION,
        "robot_type": "test",
        "fps": 30,
        "joints": joints,
        "timestamps": timestamps,
        "positions": positions,
    }
    path = tmp_path / "test.trajectory.json"
    path.write_text(json.dumps(traj))
    return path


def _make_teleop(
    tmp_path: Path,
    frames=None,
    joints=None,
    chunk_window_s: float = 0.5,
) -> TrajectoryReplayTeleop:
    if frames is None:
        frames = [(0.0, [1.0, 2.0]), (0.1, [1.1, 2.1]), (0.2, [1.2, 2.2])]
    if joints is None:
        joints = ["a.pos", "b.pos"]
    path = _write_trajectory(tmp_path, frames, joints)
    cfg = TrajectoryReplayTeleopConfig(
        id="test",
        trajectory_path=str(path),
        chunk_window_s=chunk_window_s,
    )
    return TrajectoryReplayTeleop(cfg)


class TestConnect:
    def test_loads_valid_trajectory(self, tmp_path):
        t = _make_teleop(tmp_path)
        t.connect()
        assert t.is_connected
        assert t.frame_count == 3
        assert t.duration_s == pytest.approx(0.2)
        assert t.action_features == {"a.pos": float, "b.pos": float}

    def test_missing_path_rejected(self, tmp_path):
        cfg = TrajectoryReplayTeleopConfig(id="test", trajectory_path="")
        t = TrajectoryReplayTeleop(cfg)
        with pytest.raises(ValueError, match="trajectory_path"):
            t.connect()

    def test_missing_file_rejected(self, tmp_path):
        cfg = TrajectoryReplayTeleopConfig(id="test", trajectory_path=str(tmp_path / "nope.json"))
        t = TrajectoryReplayTeleop(cfg)
        with pytest.raises(FileNotFoundError):
            t.connect()

    def test_empty_trajectory_rejected(self, tmp_path):
        t = _make_teleop(tmp_path, frames=[])
        with pytest.raises(ValueError, match="empty"):
            t.connect()

    def test_malformed_trajectory_rejected(self, tmp_path):
        path = tmp_path / "bad.trajectory.json"
        path.write_text(
            json.dumps({"schema_version": 999, "fps": 30, "joints": [], "timestamps": [], "positions": []})
        )
        cfg = TrajectoryReplayTeleopConfig(id="test", trajectory_path=str(path))
        t = TrajectoryReplayTeleop(cfg)
        with pytest.raises(ValueError, match="schema_version"):
            t.connect()


class TestGetAction:
    def test_first_frame_at_t0(self, tmp_path):
        t = _make_teleop(tmp_path)
        t.connect()
        with patch(
            "lerobot.teleoperators.trajectory_replay.teleop_trajectory_replay.time.perf_counter",
            return_value=100.0,
        ):
            a = t.get_action()
        assert a == {"a.pos": 1.0, "b.pos": 2.0}
        assert not t.is_exhausted

    def test_frame_interpolated_between_recorded_samples(self, tmp_path):
        """Action is linearly interpolated between bracketing trajectory frames."""
        t = _make_teleop(tmp_path)
        t.connect()
        with patch(
            "lerobot.teleoperators.trajectory_replay.teleop_trajectory_replay.time.perf_counter"
        ) as mock_t:
            mock_t.return_value = 100.0
            t.get_action()  # primes _start_t = 100.0
            mock_t.return_value = 100.05
            a = t.get_action()
            assert a["a.pos"] == pytest.approx(1.05)
            assert a["b.pos"] == pytest.approx(2.05)
            mock_t.return_value = 100.11
            a = t.get_action()
            assert a["a.pos"] == pytest.approx(1.11)
            assert a["b.pos"] == pytest.approx(2.11)
            mock_t.return_value = 100.15
            a = t.get_action()
            assert a["a.pos"] == pytest.approx(1.15)
            assert a["b.pos"] == pytest.approx(2.15)

    def test_exhausted_after_last_frame(self, tmp_path):
        t = _make_teleop(tmp_path)
        t.connect()
        with patch(
            "lerobot.teleoperators.trajectory_replay.teleop_trajectory_replay.time.perf_counter"
        ) as mock_t:
            mock_t.return_value = 100.0
            t.get_action()
            mock_t.return_value = 100.25  # past final timestamp 0.2
            a = t.get_action()
        assert t.is_exhausted
        # When exhausted, returns the last recorded frame (clamps).
        assert a == {"a.pos": 1.2, "b.pos": 2.2}

    def test_call_before_connect_raises(self, tmp_path):
        t = _make_teleop(tmp_path)
        with pytest.raises(RuntimeError, match="connect"):
            t.get_action()


class TestGetActionWithHorizon:
    """``get_action_with_horizon`` exposes the chunk-aware path."""

    def test_returns_action_chunk_starting_at_now(self, tmp_path):
        # 1 second of trajectory at 0.1 s spacing, value = 1 + t.
        frames = [(i * 0.1, [1.0 + i * 0.1]) for i in range(10)]
        t = _make_teleop(tmp_path, frames=frames, joints=["a.pos"], chunk_window_s=0.3)
        t.connect()
        with patch(
            "lerobot.teleoperators.trajectory_replay.teleop_trajectory_replay.time.perf_counter",
            return_value=100.0,
        ):
            chunk = t.get_action_with_horizon()
        assert isinstance(chunk, ActionChunk)
        assert chunk.fps == 30.0  # from trajectory's recorded fps
        # window_s=0.3 @ 30 fps → 0.3*30 + 1 = 10 frames
        assert len(chunk.frames) == 10
        # frame[0] should equal get_action() at the same wall-time tick
        assert chunk.frames[0]["a.pos"] == pytest.approx(1.0)

    def test_frame_zero_matches_get_action(self, tmp_path):
        """The chunk's frame 0 is the contract: it MUST equal get_action()
        at the same wall-clock instant."""
        t = _make_teleop(tmp_path)
        t.connect()
        with patch(
            "lerobot.teleoperators.trajectory_replay.teleop_trajectory_replay.time.perf_counter"
        ) as mock_t:
            mock_t.return_value = 100.0
            t.get_action()  # primes _start_t
            mock_t.return_value = 100.05
            single = t.get_action()
            chunk = t.get_action_with_horizon()
        assert chunk.frames[0] == single

    def test_chunk_frames_are_at_fps_cadence(self, tmp_path):
        """Frame k is intent at now + k / fps, regardless of recorded
        timestamp spacing — the consumer interpolates by index."""
        frames = [(i * 0.1, [1.0 + i * 0.1]) for i in range(10)]
        t = _make_teleop(tmp_path, frames=frames, joints=["a.pos"], chunk_window_s=0.2)
        t.connect()
        with patch(
            "lerobot.teleoperators.trajectory_replay.teleop_trajectory_replay.time.perf_counter",
            return_value=100.0,
        ):
            chunk = t.get_action_with_horizon()
        # fps=30 → 1/fps ≈ 0.0333 s spacing
        # frame[0] at t=0 → 1.0
        # frame[1] at t=1/30 ≈ 0.0333 → interp between (0, 1.0) and (0.1, 1.1) at α=1/3
        # = 1.0 + 0.1 * (1/3) ≈ 1.0333
        assert chunk.frames[1]["a.pos"] == pytest.approx(1.0 + 0.1 * (1.0 / 3.0))

    def test_horizon_clamps_at_trajectory_end(self, tmp_path):
        """Past the trajectory's end, the chunk holds at the last frame."""
        frames = [(0.0, [1.0]), (0.1, [1.1])]
        t = _make_teleop(tmp_path, frames=frames, joints=["a.pos"], chunk_window_s=0.5)
        t.connect()
        with patch(
            "lerobot.teleoperators.trajectory_replay.teleop_trajectory_replay.time.perf_counter",
            return_value=100.0,
        ):
            chunk = t.get_action_with_horizon()
        # Last frame value is 1.1; later frames should clamp to it (not
        # extrapolate past the end — that's the consumer's job).
        assert chunk.frames[-1]["a.pos"] == pytest.approx(1.1)


class TestStartPose:
    def test_returns_first_frame_as_dict(self, tmp_path):
        t = _make_teleop(tmp_path)
        t.connect()
        assert t.start_pose == {"a.pos": 1.0, "b.pos": 2.0}

    def test_raises_if_not_connected(self, tmp_path):
        t = _make_teleop(tmp_path)
        with pytest.raises(RuntimeError, match="connect"):
            _ = t.start_pose


class TestDisconnect:
    def test_clears_state(self, tmp_path):
        t = _make_teleop(tmp_path)
        t.connect()
        t.get_action()
        t.disconnect()
        assert not t.is_connected
        assert not t.is_exhausted
        # Reconnecting works (re-loads the file, re-arms).
        t.connect()
        assert t.is_connected

    def test_reconnect_resets_exhausted(self, tmp_path):
        t = _make_teleop(tmp_path)
        t.connect()
        with patch(
            "lerobot.teleoperators.trajectory_replay.teleop_trajectory_replay.time.perf_counter"
        ) as mock_t:
            mock_t.return_value = 100.0
            t.get_action()
            mock_t.return_value = 100.5
            t.get_action()
            assert t.is_exhausted
        t.disconnect()
        t.connect()
        assert not t.is_exhausted

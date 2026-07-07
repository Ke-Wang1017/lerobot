"""Tests for the rest position recording and playback module."""

from unittest.mock import MagicMock, patch

import pytest

from lerobot.robots.rest_position import move_to_rest_position, record_rest_position

# ============================================================================
# Helpers
# ============================================================================


def _make_mock_robot(motor_positions: dict[str, float], action_keys: list[str] | None = None):
    """Create a mock Robot that returns the given positions from get_observation().

    Args:
        motor_positions: {key: value} for motor readings.
        action_keys: Keys that action_features reports. Defaults to motor_positions keys.
    """
    robot = MagicMock()
    # Observation includes motor positions + a camera key
    obs = {**motor_positions, "cam_front": "image_data"}
    robot.get_observation.return_value = obs

    if action_keys is None:
        action_keys = list(motor_positions.keys())
    robot.action_features = dict.fromkeys(action_keys, float)

    # Track send_action calls
    robot.sent_actions = []
    robot.send_action.side_effect = lambda a: robot.sent_actions.append(dict(a))

    return robot


# ============================================================================
# record_rest_position
# ============================================================================


class TestRecordRestPosition:
    """Tests for snapshotting current joint positions."""

    def test_basic_recording(self):
        robot = _make_mock_robot(
            {
                "shoulder_pan.pos": 10.0,
                "shoulder_lift.pos": -25.3,
                "gripper.pos": 50.0,
            }
        )

        result = record_rest_position(robot)

        assert result == {
            "shoulder_pan.pos": 10.0,
            "shoulder_lift.pos": -25.3,
            "gripper.pos": 50.0,
        }

    def test_filters_to_action_features_only(self):
        """Camera keys and non-action keys should be excluded."""
        robot = _make_mock_robot(
            motor_positions={"a.pos": 1.0, "b.pos": 2.0, "extra.vel": 3.0},
            action_keys=["a.pos", "b.pos"],
        )
        # Also add non-motor data to observation
        robot.get_observation.return_value.update({"cam_front": "img", "extra.vel": 3.0})

        result = record_rest_position(robot)

        assert "a.pos" in result
        assert "b.pos" in result
        assert "extra.vel" not in result
        assert "cam_front" not in result

    def test_converts_to_float(self):
        robot = _make_mock_robot({"a.pos": 42})  # int, not float
        result = record_rest_position(robot)
        assert isinstance(result["a.pos"], float)

    def test_missing_key_in_observation_warns(self):
        """If action_features has a key not in observation, it should be skipped with warning."""
        robot = _make_mock_robot({"a.pos": 1.0}, action_keys=["a.pos", "missing.pos"])
        # "missing.pos" won't be in the observation

        result = record_rest_position(robot)

        assert "a.pos" in result
        assert "missing.pos" not in result

    def test_no_matching_keys_raises(self):
        """If no action keys are found in observation, should raise."""
        robot = _make_mock_robot({}, action_keys=["a.pos"])
        robot.get_observation.return_value = {"cam": "img"}

        with pytest.raises(RuntimeError, match="No motor positions found"):
            record_rest_position(robot)

    def test_bimanual_prefixed_keys(self):
        """Bimanual robots have left_/right_ prefixed keys — should work transparently."""
        robot = _make_mock_robot(
            {
                "left_shoulder_pan.pos": 5.0,
                "left_gripper.pos": 30.0,
                "right_shoulder_pan.pos": -5.0,
                "right_gripper.pos": 70.0,
            }
        )

        result = record_rest_position(robot)

        assert len(result) == 4
        assert result["left_shoulder_pan.pos"] == 5.0
        assert result["right_gripper.pos"] == 70.0


# ============================================================================
# move_to_rest_position
# ============================================================================


class TestMoveToRestPosition:
    """Tests for smooth interpolation to rest position."""

    def test_basic_interpolation(self):
        robot = _make_mock_robot({"a.pos": 0.0, "b.pos": 100.0})
        target = {"a.pos": 10.0, "b.pos": 0.0}

        with patch("lerobot.robots.rest_position.precise_sleep"):
            move_to_rest_position(robot, target, duration_s=1.0, steps_per_second=10)

        # Should have sent exactly 10 actions
        assert len(robot.sent_actions) == 10

        # First step: alpha = 1/10 = 0.1
        first = robot.sent_actions[0]
        assert abs(first["a.pos"] - 1.0) < 1e-6  # 0.0 * 0.9 + 10.0 * 0.1
        assert abs(first["b.pos"] - 90.0) < 1e-6  # 100.0 * 0.9 + 0.0 * 0.1

        # Last step: alpha = 1.0 → exactly at target
        last = robot.sent_actions[-1]
        assert abs(last["a.pos"] - 10.0) < 1e-6
        assert abs(last["b.pos"] - 0.0) < 1e-6

    def test_step_count(self):
        robot = _make_mock_robot({"a.pos": 0.0})

        with patch("lerobot.robots.rest_position.precise_sleep"):
            move_to_rest_position(robot, {"a.pos": 10.0}, duration_s=3.0, steps_per_second=50)

        assert len(robot.sent_actions) == 150  # 3 * 50

    def test_minimum_one_step(self):
        """Even with very small duration, at least one step should be sent."""
        robot = _make_mock_robot({"a.pos": 0.0})

        with patch("lerobot.robots.rest_position.precise_sleep"):
            move_to_rest_position(robot, {"a.pos": 10.0}, duration_s=0.001, steps_per_second=1)

        assert len(robot.sent_actions) >= 1

    def test_already_at_target(self):
        """If already at target, should still run interpolation (all steps send target)."""
        robot = _make_mock_robot({"a.pos": 5.0})

        with patch("lerobot.robots.rest_position.precise_sleep"):
            move_to_rest_position(robot, {"a.pos": 5.0}, duration_s=0.1, steps_per_second=10)

        assert len(robot.sent_actions) >= 1
        for action in robot.sent_actions:
            assert abs(action["a.pos"] - 5.0) < 1e-6

    def test_empty_rest_position_raises(self):
        robot = _make_mock_robot({"a.pos": 0.0})

        with pytest.raises(ValueError, match="empty"):
            move_to_rest_position(robot, {})

    def test_missing_current_position_uses_target(self):
        """If a key in rest_position can't be read from observation, assume at target."""
        robot = _make_mock_robot({"a.pos": 0.0}, action_keys=["a.pos"])
        # rest_position has a key not in observation
        target = {"a.pos": 10.0, "unknown.pos": 5.0}
        robot.get_observation.return_value = {"a.pos": 0.0}

        with patch("lerobot.robots.rest_position.precise_sleep"):
            move_to_rest_position(robot, target, duration_s=0.1, steps_per_second=10)

        # All actions should include both keys
        for action in robot.sent_actions:
            assert "a.pos" in action
            # unknown.pos should be at target (interpolating from target to target = target)
            assert abs(action["unknown.pos"] - 5.0) < 1e-6

    def test_calls_precise_sleep(self):
        """Verify that precise_sleep is called for timing control."""
        robot = _make_mock_robot({"a.pos": 0.0})

        with patch("lerobot.robots.rest_position.precise_sleep") as mock_sleep:
            move_to_rest_position(robot, {"a.pos": 10.0}, duration_s=1.0, steps_per_second=10)

        assert mock_sleep.call_count == 10
        # Sleep duration should be approximately 0.1s (minus elapsed)
        for call in mock_sleep.call_args_list:
            assert call[0][0] >= 0  # non-negative sleep

    def test_interpolation_is_linear(self):
        """Verify that intermediate steps follow linear interpolation."""
        robot = _make_mock_robot({"a.pos": 0.0})
        target = {"a.pos": 100.0}

        with patch("lerobot.robots.rest_position.precise_sleep"):
            move_to_rest_position(robot, target, duration_s=1.0, steps_per_second=5)

        # 5 steps: alpha = 0.2, 0.4, 0.6, 0.8, 1.0
        expected = [20.0, 40.0, 60.0, 80.0, 100.0]
        actual = [a["a.pos"] for a in robot.sent_actions]
        for exp, act in zip(expected, actual, strict=False):
            assert abs(exp - act) < 1e-6

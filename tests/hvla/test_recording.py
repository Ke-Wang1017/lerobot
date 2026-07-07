"""Tests for inference episode recording to LeRobotDataset.

Verifies that _create_recording_dataset and _add_frame_to_dataset
produce correct dataset structure without requiring GPU or robot hardware.
"""

import shutil
from unittest.mock import MagicMock

import numpy as np
import pytest


@pytest.fixture
def mock_robot():
    """Mock robot with observation_features and action_features like BiSO107."""
    robot = MagicMock()
    robot.__class__.__name__ = "BiSO107Follower"
    robot.robot_type = "biso107"

    joint_names = [
        "left_shoulder_pan.pos",
        "left_shoulder_lift.pos",
        "left_elbow_flex.pos",
        "left_forearm_roll.pos",
        "left_wrist_flex.pos",
        "left_wrist_roll.pos",
        "left_gripper.pos",
        "right_shoulder_pan.pos",
        "right_shoulder_lift.pos",
        "right_elbow_flex.pos",
        "right_forearm_roll.pos",
        "right_wrist_flex.pos",
        "right_wrist_roll.pos",
        "right_gripper.pos",
    ]
    # action_features: joint_name -> float
    robot.action_features = dict.fromkeys(joint_names, float)
    # observation_features: joints + cameras (tuple = image shape)
    robot.observation_features = {
        **dict.fromkeys(joint_names, float),
        "front": (480, 640, 3),
        "top": (480, 640, 3),
    }
    return robot


@pytest.fixture
def tmp_dataset_dir(tmp_path):
    """Provide a temp directory and clean up after."""
    yield tmp_path
    # LeRobotDataset creates in HF_LEROBOT_HOME by default;
    # we override via root param in the test.


class TestCreateRecordingDataset:
    """Verify dataset creation from robot features."""

    def test_features_match_robot(self, mock_robot, tmp_path):
        """Dataset features should reflect robot's joints and cameras."""
        from lerobot.policies.hvla.s1_process import _create_recording_dataset

        dataset = _create_recording_dataset(
            repo_id="test_user/test_dataset",
            fps=30,
            robot=mock_robot,
            task="test task",
        )

        try:
            # Check features exist
            assert "observation.state" in dataset.features
            assert "action" in dataset.features
            assert "observation.images.front" in dataset.features
            assert "observation.images.top" in dataset.features

            # Check shapes
            assert dataset.features["observation.state"]["shape"] == (14,)
            assert dataset.features["action"]["shape"] == (14,)
            assert dataset.features["observation.images.front"]["dtype"] == "video"

            # Check joint names preserved
            assert len(dataset.features["observation.state"]["names"]) == 14
            assert "left_gripper.pos" in dataset.features["observation.state"]["names"]
        finally:
            dataset.finalize()
            # Clean up
            if dataset.root.exists():
                shutil.rmtree(dataset.root)

    def test_single_arm_robot(self, tmp_path):
        """Should work with a 7-DOF single-arm robot too."""
        from lerobot.policies.hvla.s1_process import _create_recording_dataset

        robot = MagicMock()
        robot.__class__.__name__ = "SO100Follower"
        robot.robot_type = "so100"
        joint_names = [f"joint_{i}.pos" for i in range(7)]
        robot.action_features = dict.fromkeys(joint_names, float)
        robot.observation_features = {
            **dict.fromkeys(joint_names, float),
            "camera": (224, 224, 3),
        }

        dataset = _create_recording_dataset("test_user/single_arm", 30, robot, "test")
        try:
            assert dataset.features["observation.state"]["shape"] == (7,)
            assert dataset.features["action"]["shape"] == (7,)
            assert "observation.images.camera" in dataset.features
        finally:
            dataset.finalize()
            if dataset.root.exists():
                shutil.rmtree(dataset.root)


class TestAddFrameToDataset:
    """Verify frame building from obs dict."""

    def test_frame_structure(self):
        """_add_frame_to_dataset should call dataset.add_frame with correct keys."""
        from lerobot.policies.hvla.s1_process import _add_frame_to_dataset

        dataset = MagicMock()
        joint_names = ["j0.pos", "j1.pos", "j2.pos"]
        obs = {
            "j0.pos": 10.0,
            "j1.pos": 20.0,
            "j2.pos": 30.0,
            "camera": np.zeros((480, 640, 3), dtype=np.uint8),
        }
        action = np.array([1.0, 2.0, 3.0])

        _add_frame_to_dataset(dataset, obs, action, joint_names, "pick up cube")

        dataset.add_frame.assert_called_once()
        frame = dataset.add_frame.call_args[0][0]

        assert frame["task"] == "pick up cube"
        np.testing.assert_array_equal(frame["observation.state"], [10.0, 20.0, 30.0])
        np.testing.assert_array_equal(frame["action"], [1.0, 2.0, 3.0])
        assert "observation.images.camera" in frame
        assert frame["observation.images.camera"].shape == (480, 640, 3)

    def test_missing_joint_defaults_zero(self):
        """Missing joint in obs should default to 0."""
        from lerobot.policies.hvla.s1_process import _add_frame_to_dataset

        dataset = MagicMock()
        obs = {"j0.pos": 5.0}  # j1 missing
        action = np.array([1.0, 2.0])

        _add_frame_to_dataset(dataset, obs, action, ["j0.pos", "j1.pos"], "task")

        frame = dataset.add_frame.call_args[0][0]
        np.testing.assert_array_equal(frame["observation.state"], [5.0, 0.0])

    def test_multiple_cameras(self):
        """All camera images should be included."""
        from lerobot.policies.hvla.s1_process import _add_frame_to_dataset

        dataset = MagicMock()
        obs = {
            "j0.pos": 1.0,
            "front": np.zeros((480, 640, 3), dtype=np.uint8),
            "top": np.ones((480, 640, 3), dtype=np.uint8),
            "wrist": np.zeros((224, 224, 3), dtype=np.uint8),
        }

        _add_frame_to_dataset(dataset, obs, np.array([1.0]), ["j0.pos"], "task")

        frame = dataset.add_frame.call_args[0][0]
        assert "observation.images.front" in frame
        assert "observation.images.top" in frame
        assert "observation.images.wrist" in frame

    def test_action_dtype_float32(self):
        """Action should be cast to float32."""
        from lerobot.policies.hvla.s1_process import _add_frame_to_dataset

        dataset = MagicMock()
        action = np.array([1.0, 2.0], dtype=np.float64)  # wrong dtype

        _add_frame_to_dataset(dataset, {"j.pos": 0.0}, action, ["j.pos"], "task")

        frame = dataset.add_frame.call_args[0][0]
        assert frame["action"].dtype == np.float32

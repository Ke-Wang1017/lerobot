"""Regression tests for camera preview FD lifecycle.

The GUI was leaking `/dev/video*` file descriptors when camera initialisation
failed partway through `_detect_and_open_cameras()`. The per-camera body now
uses an ownership-transfer try/finally so any error between `connect()` and
successful registration triggers `disconnect()`.
"""

from unittest.mock import MagicMock, patch

import pytest

from lerobot.gui.api import robot as robot_module


@pytest.fixture(autouse=True)
def _reset_preview_state():
    """Ensure module-level preview lists are clean around each test."""
    robot_module._preview_cameras.clear()
    robot_module._preview_camera_info.clear()
    yield
    robot_module._preview_cameras.clear()
    robot_module._preview_camera_info.clear()


def _make_mock_camera_class(connect_side_effects):
    """Create a mock camera class whose instances follow the given connect()
    side effects in order. Each instance's `disconnect()` is a MagicMock.

    Returns (MockClass, list_of_instances) so callers can assert on per-camera
    behaviour after the call.
    """
    instances: list[MagicMock] = []
    side_effect_iter = iter(connect_side_effects)

    def factory(_config):
        inst = MagicMock()
        try:
            effect = next(side_effect_iter)
        except StopIteration:
            effect = None
        if effect is None:
            inst.connect = MagicMock(return_value=None)
        else:
            inst.connect = MagicMock(side_effect=effect)
        instances.append(inst)
        return inst

    cls = MagicMock(side_effect=factory)
    return cls, instances


def test_opencv_connect_failure_invokes_disconnect():
    """If OpenCV camera connect() raises, the partial handle is disconnected.

    Mirrors the originally-reported FD leak: connect() opens the V4L2 node,
    then warmup raises, and the old code abandoned the partially-opened
    handle. With the fix, the finally clause must call disconnect().
    """
    mock_opencv_cls, opencv_instances = _make_mock_camera_class([RuntimeError("warmup failed")])
    # Use integer ID to bypass the `/dev/video*` sysfs RealSense check.
    mock_opencv_cls.find_cameras = MagicMock(return_value=[{"id": 0, "name": "test-cam"}])

    mock_realsense_stub = MagicMock()
    mock_realsense_stub.find_cameras = MagicMock(return_value=[])

    with (
        patch(
            "lerobot.cameras.opencv.camera_opencv.OpenCVCamera",
            mock_opencv_cls,
        ),
        patch(
            "lerobot.cameras.opencv.configuration_opencv.OpenCVCameraConfig",
            MagicMock(),
        ),
        patch(
            "lerobot.cameras.realsense.camera_realsense.RealSenseCamera",
            mock_realsense_stub,
        ),
        patch(
            "lerobot.cameras.realsense.configuration_realsense.RealSenseCameraConfig",
            MagicMock(),
        ),
    ):
        robot_module._detect_and_open_cameras()

    assert len(opencv_instances) == 1
    opencv_instances[0].disconnect.assert_called_once()
    # The leaked camera must NOT remain in the preview list.
    assert robot_module._preview_cameras == []
    assert robot_module._preview_camera_info == []


def test_realsense_connect_failure_invokes_disconnect():
    """Same invariant for the RealSense branch."""
    mock_opencv_stub = MagicMock()
    mock_opencv_stub.find_cameras = MagicMock(return_value=[])

    mock_realsense_cls, rs_instances = _make_mock_camera_class([RuntimeError("librealsense init failed")])
    mock_realsense_cls.find_cameras = MagicMock(return_value=[{"id": "123456789", "name": "D435"}])

    with (
        patch(
            "lerobot.cameras.realsense.camera_realsense.RealSenseCamera",
            mock_realsense_cls,
        ),
        patch(
            "lerobot.cameras.realsense.configuration_realsense.RealSenseCameraConfig",
            MagicMock(),
        ),
        patch(
            "lerobot.cameras.opencv.camera_opencv.OpenCVCamera",
            mock_opencv_stub,
        ),
        patch(
            "lerobot.cameras.opencv.configuration_opencv.OpenCVCameraConfig",
            MagicMock(),
        ),
    ):
        robot_module._detect_and_open_cameras()

    assert len(rs_instances) == 1
    rs_instances[0].disconnect.assert_called_once()
    assert robot_module._preview_cameras == []


def test_successful_connect_does_not_disconnect():
    """The happy path must NOT spuriously disconnect (ownership transferred).

    Without the `camera = None` reset after successful registration, a buggy
    fix could call disconnect() on every camera.
    """
    mock_opencv_cls, opencv_instances = _make_mock_camera_class([None, None])
    # Use integer IDs to bypass the `/dev/video*` sysfs RealSense check.
    mock_opencv_cls.find_cameras = MagicMock(
        return_value=[
            {"id": 0, "name": "cam-a"},
            {"id": 1, "name": "cam-b"},
        ]
    )

    mock_realsense_stub = MagicMock()
    mock_realsense_stub.find_cameras = MagicMock(return_value=[])

    with (
        patch(
            "lerobot.cameras.opencv.camera_opencv.OpenCVCamera",
            mock_opencv_cls,
        ),
        patch(
            "lerobot.cameras.opencv.configuration_opencv.OpenCVCameraConfig",
            MagicMock(),
        ),
        patch(
            "lerobot.cameras.realsense.camera_realsense.RealSenseCamera",
            mock_realsense_stub,
        ),
        patch(
            "lerobot.cameras.realsense.configuration_realsense.RealSenseCameraConfig",
            MagicMock(),
        ),
    ):
        robot_module._detect_and_open_cameras()

    assert len(opencv_instances) == 2
    for inst in opencv_instances:
        inst.disconnect.assert_not_called()
    assert len(robot_module._preview_cameras) == 2


def test_cleanup_in_process_resources_releases_everything():
    """Server-shutdown hook releases preview cameras + recording robots.

    Without this cleanup the OS still reclaims FDs on process death, but
    motor torque / serial port handoff is never run cleanly, leaving the
    hardware in whatever state the user last touched. The helper must:
    - disconnect every preview camera and clear the lists
    - stop any safe-trajectory recorder and disconnect its robot
    - disconnect the rest-position recording robot
    """
    from lerobot.gui.api.robot import cleanup_in_process_resources

    cam_a = MagicMock()
    cam_b = MagicMock()
    robot_module._preview_cameras.extend([cam_a, cam_b])
    robot_module._preview_camera_info.extend([{"id": "a"}, {"id": "b"}])

    rest_robot = MagicMock()
    robot_module._rest_recording_robot = rest_robot

    traj_robot = MagicMock()
    traj_recorder = MagicMock()
    robot_module._trajectory_recording_robot = traj_robot
    robot_module._trajectory_recorder = traj_recorder

    try:
        cleanup_in_process_resources()
    finally:
        robot_module._rest_recording_robot = None
        robot_module._trajectory_recording_robot = None
        robot_module._trajectory_recorder = None

    cam_a.disconnect.assert_called_once()
    cam_b.disconnect.assert_called_once()
    rest_robot.disconnect.assert_called_once()
    traj_robot.disconnect.assert_called_once()
    traj_recorder.stop.assert_called_once()
    assert robot_module._preview_cameras == []
    assert robot_module._preview_camera_info == []
    assert robot_module._rest_recording_robot is None
    assert robot_module._trajectory_recording_robot is None
    assert robot_module._trajectory_recorder is None


def test_cleanup_in_process_resources_survives_disconnect_errors():
    """A camera or robot raising during disconnect must not break the rest
    of the sweep — `cleanup_in_process_resources` runs on shutdown so each
    step has to be best-effort.
    """
    from lerobot.gui.api.robot import cleanup_in_process_resources

    bad_cam = MagicMock()
    bad_cam.disconnect.side_effect = RuntimeError("camera HW gone")
    robot_module._preview_cameras.append(bad_cam)
    robot_module._preview_camera_info.append({"id": "bad"})

    bad_robot = MagicMock()
    bad_robot.disconnect.side_effect = RuntimeError("motor bus locked")
    robot_module._rest_recording_robot = bad_robot

    try:
        cleanup_in_process_resources()  # must not raise
    finally:
        robot_module._rest_recording_robot = None

    bad_cam.disconnect.assert_called_once()
    bad_robot.disconnect.assert_called_once()
    assert robot_module._preview_cameras == []
    assert robot_module._rest_recording_robot is None


def test_partial_failure_does_not_leak_other_cameras():
    """One camera failing mid-iteration must not affect the others."""
    mock_opencv_cls, opencv_instances = _make_mock_camera_class([None, RuntimeError("cam 2 failed"), None])
    # Use integer IDs to bypass the `/dev/video*` sysfs RealSense check.
    mock_opencv_cls.find_cameras = MagicMock(
        return_value=[
            {"id": 0, "name": "ok-1"},
            {"id": 1, "name": "bad"},
            {"id": 2, "name": "ok-2"},
        ]
    )

    mock_realsense_stub = MagicMock()
    mock_realsense_stub.find_cameras = MagicMock(return_value=[])

    with (
        patch(
            "lerobot.cameras.opencv.camera_opencv.OpenCVCamera",
            mock_opencv_cls,
        ),
        patch(
            "lerobot.cameras.opencv.configuration_opencv.OpenCVCameraConfig",
            MagicMock(),
        ),
        patch(
            "lerobot.cameras.realsense.camera_realsense.RealSenseCamera",
            mock_realsense_stub,
        ),
        patch(
            "lerobot.cameras.realsense.configuration_realsense.RealSenseCameraConfig",
            MagicMock(),
        ),
    ):
        robot_module._detect_and_open_cameras()

    assert len(opencv_instances) == 3
    # The two successful cameras are registered and NOT disconnected.
    opencv_instances[0].disconnect.assert_not_called()
    opencv_instances[2].disconnect.assert_not_called()
    # The failing camera IS disconnected.
    opencv_instances[1].disconnect.assert_called_once()
    assert len(robot_module._preview_cameras) == 2

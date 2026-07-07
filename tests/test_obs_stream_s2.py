"""Tests for ObservationStreamWriterStep's S2 SharedImageBuffer integration.

Covers:
  - No-op when env var not set
  - Lazy attach when S2 shared memory exists
  - Retry when S2 not started yet
  - Recovery after S2 unload (shared memory unlinked)
  - Reconnect after S2 reload
"""

import numpy as np
import pytest


@pytest.fixture(autouse=True)
def _s2_env(monkeypatch):
    """Set the env vars for obs stream + S2 buffer."""
    monkeypatch.setenv("LEROBOT_OBS_STREAM", "1")
    monkeypatch.setenv("LEROBOT_S2_IMAGE_BUFFER", "1")


@pytest.fixture
def dummy_obs():
    """Minimal observation matching SO107 bimanual layout."""
    obs = {
        "front": np.zeros((720, 1280, 3), dtype=np.uint8),
        "top": np.zeros((720, 1280, 3), dtype=np.uint8),
        "left_wrist": np.zeros((720, 1280, 3), dtype=np.uint8),
        "right_wrist": np.zeros((720, 1280, 3), dtype=np.uint8),
    }
    for i in range(14):
        obs[f"joint_{i}.pos"] = float(i)
    return obs


def _create_s2_shared_memory():
    """Simulate S2 standalone creating SharedImageBuffer."""
    from lerobot.policies.hvla.ipc import DEFAULT_S2_IMAGE_KEYS, SharedImageBuffer

    buf = SharedImageBuffer(
        camera_keys=DEFAULT_S2_IMAGE_KEYS,
        height=720,
        width=1280,
        create=True,
        state_dim=32,
    )
    return buf


def _cleanup_s2_shared_memory(buf):
    """Simulate S2 standalone exiting."""
    buf.cleanup()


class TestS2BufferDisabled:
    def test_noop_without_env_var(self, monkeypatch, dummy_obs):
        """Without LEROBOT_S2_IMAGE_BUFFER, no S2 buffer is created."""
        monkeypatch.delenv("LEROBOT_S2_IMAGE_BUFFER", raising=False)
        from lerobot.robots.obs_stream import ObservationStreamWriterStep

        step = ObservationStreamWriterStep()
        assert not step._s2_enabled
        result = step.observation(dummy_obs)
        assert result is dummy_obs
        assert step._s2_buffer is None


class TestS2BufferAttach:
    def test_attach_when_s2_exists(self, dummy_obs):
        """Writer attaches to S2's shared memory when it exists."""
        s2_buf = _create_s2_shared_memory()
        try:
            from lerobot.robots.obs_stream import ObservationStreamWriterStep

            step = ObservationStreamWriterStep()
            step.observation(dummy_obs)
            assert step._s2_buffer is not None
        finally:
            _cleanup_s2_shared_memory(s2_buf)

    def test_retry_when_s2_not_started(self, dummy_obs):
        """Writer retries silently when S2 shared memory doesn't exist yet."""
        from lerobot.robots.obs_stream import ObservationStreamWriterStep

        step = ObservationStreamWriterStep()

        # First call — S2 not started, buffer stays None
        step.observation(dummy_obs)
        assert step._s2_buffer is None
        assert step._s2_joint_names is not None  # init happened

        # S2 starts
        s2_buf = _create_s2_shared_memory()
        try:
            # Next call — should attach
            step.observation(dummy_obs)
            assert step._s2_buffer is not None
        finally:
            _cleanup_s2_shared_memory(s2_buf)

    def test_write_data_reaches_s2(self, dummy_obs):
        """Data written by the step is readable from S2's buffer."""
        s2_buf = _create_s2_shared_memory()
        try:
            from lerobot.robots.obs_stream import ObservationStreamWriterStep

            step = ObservationStreamWriterStep()

            # Write a distinctive image
            dummy_obs["front"][0, 0, 0] = 42
            step.observation(dummy_obs)

            # Read from S2's side
            result = s2_buf.read_images()
            assert result is not None
            assert result["base_0_rgb"][0, 0, 0] == 42  # front → base_0_rgb
        finally:
            _cleanup_s2_shared_memory(s2_buf)


class TestS2BufferUnloadReload:
    def test_recovery_after_unload(self, dummy_obs):
        """After S2 unloads (unlinks shm), writer resets and stops crashing."""
        s2_buf = _create_s2_shared_memory()
        from lerobot.robots.obs_stream import ObservationStreamWriterStep

        step = ObservationStreamWriterStep()

        # Attach and write successfully
        step.observation(dummy_obs)
        assert step._s2_buffer is not None

        # S2 unloads — shared memory destroyed
        _cleanup_s2_shared_memory(s2_buf)

        # Next write should fail gracefully and reset buffer
        step.observation(dummy_obs)
        assert step._s2_buffer is None  # reset after error

    def test_reconnect_after_reload(self, dummy_obs):
        """After S2 reloads (new shm), writer re-attaches."""
        # First S2 session
        s2_buf = _create_s2_shared_memory()
        from lerobot.robots.obs_stream import ObservationStreamWriterStep

        step = ObservationStreamWriterStep()
        step.observation(dummy_obs)
        assert step._s2_buffer is not None

        # S2 unloads
        _cleanup_s2_shared_memory(s2_buf)
        step.observation(dummy_obs)  # resets buffer
        assert step._s2_buffer is None

        # S2 reloads — new shared memory
        s2_buf2 = _create_s2_shared_memory()
        try:
            step.observation(dummy_obs)
            assert step._s2_buffer is not None  # re-attached

            # Write works
            dummy_obs["front"][0, 0, 0] = 99
            step.observation(dummy_obs)
            result = s2_buf2.read_images()
            assert result is not None
            assert result["base_0_rgb"][0, 0, 0] == 99
        finally:
            _cleanup_s2_shared_memory(s2_buf2)

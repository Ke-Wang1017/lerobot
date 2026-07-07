"""Observable robot state via shared memory (zero-copy).

Activated by setting LEROBOT_OBS_STREAM=1.  Robot subclasses are
automatically wrapped (via __init_subclass__ in robot.py) so that
connect/disconnect/send_action publish to shared memory.

Observation writes are done via ObservationStreamWriterStep — a processor
step appended to the end of robot_observation_processor pipeline, so the
stream receives post-processed observations (with overlays etc.).

Any process can read the latest state by constructing an ObservationStreamReader.

Layout (all created dynamically based on observation_features/action_features):
  - meta block:  JSON descriptor (feature names, image dims) — written once
  - obs block:   float32[N_obs]  — scalar observation values
  - act block:   float32[N_act]  — scalar action values (last sent)
  - img blocks:  uint8[H*W*C]   — one per camera

Torn-read protection via sequence counters (same pattern as policies.hvla.ipc).
"""

from __future__ import annotations

import contextlib
import functools
import json
import logging
import multiprocessing.shared_memory as _shm
import os
import struct
import time
from multiprocessing import resource_tracker

import numpy as np

logger = logging.getLogger(__name__)

ENV_VAR = "LEROBOT_OBS_STREAM"
SHM_PREFIX = "lerobot_obs_"

# Linux POSIX shared memory backing dir. /dev/shm/<name> is what
# multiprocessing.shared_memory.SharedMemory uses on Linux.
_SHM_DIR = "/dev/shm"  # nosec B108  # well-known POSIX shm path


def cleanup_stale_streams(shm_dir: str | os.PathLike[str] = _SHM_DIR) -> int:
    """Remove ``<shm_dir>/lerobot_obs_*`` segments left by a crashed writer.

    A normal ``ObservationStream.cleanup()`` (called from the robot's
    ``disconnect()``) unlinks the segments — but if the writer process
    dies via SIGKILL, SIGABRT, segfault, etc. the cleanup never runs and
    the segments persist in ``/dev/shm`` until a reboot.

    A stale segment is *worse than nothing* for the GUI's reader path:
    ``ObservationStreamReader()`` happily attaches to whatever exists and
    serves frozen data, making the GUI think teleop is alive when it
    isn't. Sweeping at process boundaries (GUI startup / shutdown) closes
    that hole.

    Safe to call only when no observation stream writer is active in
    *any* process. The GUI calls this on startup (before any subprocess
    runs) and on shutdown (after killing the active subprocess).

    Args:
      shm_dir: directory to scan. Defaults to the system POSIX shm path
        (``/dev/shm`` on Linux); tests pass a tmp_path.

    Returns the number of segments removed.
    """
    if not os.path.isdir(shm_dir):
        return 0
    removed = 0
    for name in os.listdir(shm_dir):
        if not name.startswith(SHM_PREFIX):
            continue
        path = os.path.join(shm_dir, name)
        try:
            # safe-destruct: shm we created (matches our SHM_PREFIX), only swept when no writer is active
            os.unlink(path)
            removed += 1
        except OSError as e:
            logger.warning("could not remove stale shm %s: %s", path, e)
    return removed


# Header per block: seq_write(i64) + seq_done(i64) + timestamp(f64) = 24 bytes
_HDR = struct.Struct("<qqd")
_HDR_SIZE = _HDR.size

# ---------------------------------------------------------------------------
# Module-level singleton — one active stream per process
# ---------------------------------------------------------------------------
_active_stream: ObservationStream | None = None
_active_robot_id: int | None = None  # id() of the Robot that owns the stream
_connect_depth: int = 0  # nesting depth of wrapped connect() calls


# ============================================================================
# Low-level shared memory block
# ============================================================================


class _Block:
    """Named shared-memory block with sequence-counter torn-read protection."""

    def __init__(self, name: str, data_size: int, create: bool):
        self._owner = create
        self._data_size = data_size
        total = _HDR_SIZE + data_size

        # Suppress resource_tracker register AND unregister for ALL
        # SharedMemory calls in this constructor.  We manage lifecycle
        # explicitly (cleanup/unlink), so the tracker is not needed and
        # causes KeyError spam on Python 3.10 (old.unlink() triggers
        # unregister for a name that was never registered in this process).
        _orig_register = resource_tracker.register
        _orig_unregister = resource_tracker.unregister
        resource_tracker.register = lambda *a, **kw: None
        resource_tracker.unregister = lambda *a, **kw: None
        try:
            if create:
                # Clean up stale shm from a crashed prior run
                try:
                    old = _shm.SharedMemory(name=name)
                    old.close()
                    # safe-destruct: stale shm cleanup we created
                    old.unlink()
                except FileNotFoundError:
                    pass
                self._shm = _shm.SharedMemory(name=name, create=True, size=total)
                self._shm.buf[:_HDR_SIZE] = _HDR.pack(0, 0, 0.0)
            else:
                self._shm = _shm.SharedMemory(name=name)
                total = self._shm.size
                data_size = total - _HDR_SIZE
        finally:
            resource_tracker.register = _orig_register
            resource_tracker.unregister = _orig_unregister

        self.name = self._shm.name
        self._total = total
        self._data_size = data_size

    def write_bytes(self, data: bytes | memoryview) -> None:
        buf = self._shm.buf
        seq = struct.unpack_from("<q", buf, 0)[0] + 1
        struct.pack_into("<q", buf, 0, seq)
        buf[_HDR_SIZE : _HDR_SIZE + len(data)] = data
        struct.pack_into("<qd", buf, 8, seq, time.time())

    def write_array(self, arr: np.ndarray) -> None:
        self.write_bytes(arr.tobytes())

    @property
    def seq(self) -> int:
        """Current completed-write sequence number (cheap header read)."""
        return struct.unpack_from("<q", self._shm.buf, 8)[0]  # seq_done

    def read(self) -> tuple[bytes, float] | None:
        """Read with torn-read detection.  Returns (data, timestamp) or None."""
        buf = self._shm.buf
        for _ in range(3):
            s1, d1, ts = _HDR.unpack_from(buf, 0)
            if d1 == 0:
                return None
            data = bytes(buf[_HDR_SIZE : self._total])
            s2, d2, _ = _HDR.unpack_from(buf, 0)
            if s1 == d1 == s2 == d2:
                return data, ts
        return data, ts  # best-effort

    def cleanup(self) -> None:
        with contextlib.suppress(Exception):
            self._shm.close()
        if self._owner:
            # Suppress unregister inside unlink() — we never registered,
            # so unregister would send a message that causes KeyError in
            # the resource tracker daemon.
            _orig = resource_tracker.unregister
            resource_tracker.unregister = lambda *a, **kw: None
            try:
                # safe-destruct: shm we created, cleanup on close
                self._shm.unlink()
            except Exception:
                pass
            finally:
                resource_tracker.unregister = _orig


# ============================================================================
# Writer (robot side)
# ============================================================================


class ObservationStream:
    """Publishes robot observations and actions to shared memory."""

    def __init__(self, obs_features: dict, action_features: dict):
        self.obs_scalar_keys: list[str] = sorted(
            k for k, v in obs_features.items() if not isinstance(v, tuple)
        )
        self.action_keys: list[str] = sorted(action_features.keys())
        self.image_keys: dict[str, tuple[int, ...]] = dict(
            sorted((k, v) for k, v in obs_features.items() if isinstance(v, tuple))
        )

        # Meta block — JSON descriptor, written once
        meta_json = json.dumps(
            {
                "obs_scalar_keys": self.obs_scalar_keys,
                "action_keys": self.action_keys,
                "image_keys": {k: list(v) for k, v in self.image_keys.items()},
            }
        ).encode()
        self._meta = _Block(f"{SHM_PREFIX}meta", len(meta_json), create=True)
        self._meta.write_bytes(meta_json)

        # Obs scalar block
        self._obs_buf = np.zeros(len(self.obs_scalar_keys), dtype=np.float32)
        self._obs_block = _Block(f"{SHM_PREFIX}obs", self._obs_buf.nbytes, create=True)

        # Action scalar block
        self._act_buf = np.zeros(len(self.action_keys), dtype=np.float32)
        self._act_block = _Block(f"{SHM_PREFIX}act", self._act_buf.nbytes, create=True)

        # One image block per camera
        self._img_blocks: dict[str, _Block] = {}
        for cam_key, dims in self.image_keys.items():
            h, w = dims[0], dims[1]
            c = dims[2] if len(dims) > 2 else 3
            safe = cam_key.replace("/", "_")
            self._img_blocks[cam_key] = _Block(f"{SHM_PREFIX}img_{safe}", h * w * c, create=True)

        logger.info(
            "ObservationStream started: %d obs scalars, %d action scalars, %d cameras",
            len(self.obs_scalar_keys),
            len(self.action_keys),
            len(self.image_keys),
        )

    def write_obs(self, obs: dict) -> None:
        for i, key in enumerate(self.obs_scalar_keys):
            val = obs.get(key)
            self._obs_buf[i] = float(val) if val is not None else 0.0
        self._obs_block.write_array(self._obs_buf)

        for cam_key, block in self._img_blocks.items():
            img = obs.get(cam_key)
            if img is not None:
                block.write_array(np.ascontiguousarray(img, dtype=np.uint8))

    def write_action(self, action: dict) -> None:
        for i, key in enumerate(self.action_keys):
            val = action.get(key)
            self._act_buf[i] = float(val) if val is not None else 0.0
        self._act_block.write_array(self._act_buf)

    def cleanup(self) -> None:
        self._meta.cleanup()
        self._obs_block.cleanup()
        self._act_block.cleanup()
        for block in self._img_blocks.values():
            block.cleanup()
        logger.info("ObservationStream cleaned up")


# ============================================================================
# Reader (GUI / external process side)
# ============================================================================


class ObservationStreamReader:
    """Attaches to an existing observation stream (read-only, any process)."""

    def __init__(self):
        self._meta = _Block(f"{SHM_PREFIX}meta", 0, create=False)
        result = self._meta.read()
        if result is None:
            raise RuntimeError("ObservationStream has no data yet")
        meta = json.loads(result[0].rstrip(b"\x00"))

        self.obs_scalar_keys: list[str] = meta["obs_scalar_keys"]
        self.action_keys: list[str] = meta["action_keys"]
        self.image_keys: dict[str, list[int]] = meta["image_keys"]

        self._obs_block = _Block(f"{SHM_PREFIX}obs", len(self.obs_scalar_keys) * 4, create=False)
        self._act_block = _Block(f"{SHM_PREFIX}act", len(self.action_keys) * 4, create=False)
        self._img_blocks: dict[str, _Block] = {}
        for cam_key, dims in self.image_keys.items():
            safe = cam_key.replace("/", "_")
            h, w = dims[0], dims[1]
            c = dims[2] if len(dims) > 2 else 3
            self._img_blocks[cam_key] = _Block(f"{SHM_PREFIX}img_{safe}", h * w * c, create=False)

    def read_obs(self) -> tuple[dict[str, float], float] | None:
        """Returns (obs_dict, timestamp) or None if no data."""
        result = self._obs_block.read()
        if result is None:
            return None
        data, ts = result
        arr = np.frombuffer(data, dtype=np.float32)
        return {k: float(arr[i]) for i, k in enumerate(self.obs_scalar_keys)}, ts

    def read_action(self) -> tuple[dict[str, float], float] | None:
        """Returns (action_dict, timestamp) or None if no data."""
        result = self._act_block.read()
        if result is None:
            return None
        data, ts = result
        arr = np.frombuffer(data, dtype=np.float32)
        return {k: float(arr[i]) for i, k in enumerate(self.action_keys)}, ts

    def image_seq(self, cam_key: str) -> int:
        """Return the sequence number for a camera's image block (cheap)."""
        block = self._img_blocks.get(cam_key)
        return block.seq if block is not None else 0

    def read_image(self, cam_key: str) -> tuple[np.ndarray, float] | None:
        """Returns (HWC uint8 array, timestamp) or None."""
        block = self._img_blocks.get(cam_key)
        if block is None:
            return None
        result = block.read()
        if result is None:
            return None
        data, ts = result
        dims = self.image_keys[cam_key]
        return np.frombuffer(data, dtype=np.uint8).reshape(dims), ts

    def close(self) -> None:
        self._meta.cleanup()
        self._obs_block.cleanup()
        self._act_block.cleanup()
        for block in self._img_blocks.values():
            block.cleanup()


# ============================================================================
# Observation processor step (writes processed obs to stream)
# ============================================================================


class ObservationStreamWriterStep:
    """Processor step that writes observations to shared memory stream(s).

    Append this as the LAST step in robot_observation_processor so the stream
    receives post-processed observations (with overlays etc.), not raw.

    Automatically also writes to HVLA's SharedImageBuffer if the
    LEROBOT_S2_IMAGE_BUFFER env var is set (for debug S2 model alongside teleop).

    Follows the ObservationProcessorStep interface (observation() method)
    but does not subclass it to avoid importing the processor module at
    module level (which would create a circular dependency).
    """

    def __init__(self):
        self._s2_buffer = None
        self._s2_joint_names = None
        # Lazily create S2 shared image buffer if env var is set
        if os.environ.get("LEROBOT_S2_IMAGE_BUFFER"):
            self._s2_enabled = True
        else:
            self._s2_enabled = False

    def _ensure_s2_buffer(self, obs: dict):
        """Attach to S2 shared memory. Lazy init + retry until S2 process creates it."""
        if self._s2_buffer is not None:
            return

        # One-time: discover joint names and image resolution from first observation
        if self._s2_joint_names is None:
            try:
                from lerobot.policies.hvla.ipc import DEFAULT_S2_CAM_KEY_MAP

                self._s2_cam_key_map = DEFAULT_S2_CAM_KEY_MAP
                self._s2_joint_names = [
                    k for k, v in obs.items() if isinstance(v, (int, float)) and not k.startswith("_")
                ]
                for cam_name in self._s2_cam_key_map:
                    img = obs.get(cam_name)
                    if img is not None:
                        self._s2_height, self._s2_width = img.shape[0], img.shape[1]
                        break
                else:
                    self._s2_height, self._s2_width = 720, 1280
            except Exception:
                self._s2_enabled = False
                return

        # Try attach (cheap existence check first)
        shm_name = "hvla_img_" + next(iter(self._s2_cam_key_map.values()))
        if _shm_exists(shm_name):
            self._try_attach_s2()

    def _try_attach_s2(self):
        """Try to attach to S2's SharedImageBuffer. No-op if not found."""
        try:
            from lerobot.policies.hvla.ipc import SharedImageBuffer

            s2_image_keys = tuple(self._s2_cam_key_map.values())
            self._s2_buffer = SharedImageBuffer(
                camera_keys=s2_image_keys,
                height=self._s2_height,
                width=self._s2_width,
                create=False,
                state_dim=max(len(self._s2_joint_names), 32),
            )
            logger.info(
                "Attached to S2 SharedImageBuffer (%dx%d, %d joints)",
                self._s2_width,
                self._s2_height,
                len(self._s2_joint_names),
            )
        except FileNotFoundError:
            logger.debug("S2 SharedImageBuffer not found yet, will retry")

    def observation(self, observation: dict) -> dict:
        """Write observation to stream(s) and pass through unchanged."""
        if _active_stream is not None:
            with contextlib.suppress(Exception):
                _active_stream.write_obs(observation)
        if self._s2_enabled:
            self._ensure_s2_buffer(observation)
            if self._s2_buffer is not None:
                # Check if S2 shared memory is still alive (S2 may have been unloaded)
                shm_name = "hvla_img_" + next(iter(self._s2_cam_key_map.values()))
                if not _shm_exists(shm_name):
                    self._s2_buffer = None  # S2 unloaded — will re-attach on next call
                else:
                    try:
                        self._s2_buffer.write_images(
                            observation,
                            self._s2_cam_key_map,
                            self._s2_joint_names,
                        )
                    except Exception:
                        self._s2_buffer = None
        return observation

    def __call__(self, transition):
        """Support being called as a pipeline step (EnvTransition interface)."""
        observation = transition.get("observation") if isinstance(transition, dict) else transition
        if isinstance(observation, dict):
            self.observation(observation)
        return transition

    def transform_features(self, features):
        """Pass-through — this step doesn't modify features."""
        return features


def _shm_exists(name: str) -> bool:
    """Check if a named shared memory segment exists via /dev/shm (no syscall overhead)."""
    return os.path.exists(f"/dev/shm/{name}")  # nosec B108  /dev/shm is the POSIX shared-memory FS, not a temp dir.


def make_obs_stream_writer_step() -> ObservationStreamWriterStep | None:
    """Create an ObservationStreamWriterStep if the obs stream is enabled.

    Returns None if LEROBOT_OBS_STREAM is not set — callers should skip appending.
    """
    if not os.environ.get(ENV_VAR):
        return None
    return ObservationStreamWriterStep()


# ============================================================================
# __init_subclass__ wrapping helpers
# ============================================================================


def _maybe_start_stream(robot) -> None:
    global _active_stream, _active_robot_id
    if _connect_depth > 0:
        return  # nested connect (sub-robot inside composite) — skip
    if not os.environ.get(ENV_VAR):
        return
    if _active_stream is not None:
        _active_stream.cleanup()
        _active_stream = None
        _active_robot_id = None
    try:
        _active_stream = ObservationStream(robot.observation_features, robot.action_features)
        _active_robot_id = id(robot)
    except Exception:
        logger.warning("Failed to create ObservationStream", exc_info=True)


def _maybe_stop_stream(robot) -> None:
    global _active_stream, _active_robot_id
    if _active_stream is not None and _active_robot_id == id(robot):
        _active_stream.cleanup()
        _active_stream = None
        _active_robot_id = None


def _maybe_write_action(robot, action: dict) -> None:
    if _active_stream is not None and _active_robot_id == id(robot):
        try:
            _active_stream.write_action(action)
        except Exception:
            logger.debug("ObservationStream.write_action failed", exc_info=True)


def wrap_robot_cls(cls) -> None:
    """Wrap connect/disconnect/get_observation/send_action on a Robot subclass.

    Called automatically by Robot.__init_subclass__().
    Uses _connect_depth to ensure only the outermost robot in a composite
    creates the observation stream.
    """
    if "connect" in cls.__dict__:
        _orig_connect = cls.__dict__["connect"]

        @functools.wraps(_orig_connect)
        def _connect(self, *args, **kwargs):
            global _connect_depth
            _connect_depth += 1
            try:
                result = _orig_connect(self, *args, **kwargs)
            finally:
                _connect_depth -= 1
            _maybe_start_stream(self)
            return result

        cls.connect = _connect

    if "disconnect" in cls.__dict__:
        _orig_disconnect = cls.__dict__["disconnect"]

        @functools.wraps(_orig_disconnect)
        def _disconnect(self, *args, **kwargs):
            result = _orig_disconnect(self, *args, **kwargs)
            _maybe_stop_stream(self)
            return result

        cls.disconnect = _disconnect

    # NOTE: get_observation is NOT wrapped — obs stream writes are handled by
    # ObservationStreamWriterStep in the processor pipeline, so the stream
    # receives post-processed observations (with overlays etc.).

    if "send_action" in cls.__dict__:
        _orig_send_act = cls.__dict__["send_action"]

        @functools.wraps(_orig_send_act)
        def _send_action(self, *args, **kwargs):
            action = _orig_send_act(self, *args, **kwargs)
            _maybe_write_action(self, action)
            return action

        cls.send_action = _send_action

"""Safe trajectory recording and open-loop replay for any LeRobot robot.

A "safe trajectory" is a human-recorded joint-space motion the robot can
later replay on its own, open-loop, without a leader arm or a policy.
The user hand-guides the (torque-off) follower through the motion; we
sample joint positions at a fixed rate and persist the time series.
Replay sends the recorded positions back as goals at the recorded fps,
relying on the human-recorded velocities as the natural safety envelope.

Pure logic — no disk I/O.  Callers (GUI backend, future CLI) own
persistence.
"""

import logging
import threading
import time
from typing import Any

from lerobot.robots.robot import Robot
from lerobot.utils.robot_utils import precise_sleep

logger = logging.getLogger(__name__)


SCHEMA_VERSION = 1


class TrajectoryRecorder:
    """Streams motor positions from a connected, torque-off robot.

    Preconditions:
    - Robot must already be connected.
    - Robot's torque must already be disabled (so the user can move it).

    Lifecycle: ``start()`` once, ``stop()`` or ``cancel()`` once. Not
    re-startable.
    """

    def __init__(self, robot: Robot, fps: int = 30) -> None:
        if fps <= 0:
            raise ValueError(f"fps must be positive, got {fps}")
        self.robot = robot
        self.fps = fps
        self._joints: list[str] = sorted(robot.action_features.keys())
        self._timestamps: list[float] = []
        self._positions: list[list[float]] = []
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._error: BaseException | None = None

    @property
    def is_running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    @property
    def frame_count(self) -> int:
        return len(self._timestamps)

    def start(self) -> None:
        if self._thread is not None:
            raise RuntimeError("TrajectoryRecorder already started")
        self._thread = threading.Thread(target=self._run, name="SafeTrajectoryRecorder", daemon=True)
        self._thread.start()

    def _run(self) -> None:
        period = 1.0 / self.fps
        start_t = time.perf_counter()
        next_t = start_t
        try:
            while not self._stop_event.is_set():
                obs = self.robot.get_observation()
                frame = [float(obs[j]) for j in self._joints if j in obs]
                if len(frame) != len(self._joints):
                    # Skip incomplete frames silently — most likely a
                    # transient bus read miss.
                    logger.debug(
                        "Skipping incomplete frame (got %d/%d joints)", len(frame), len(self._joints)
                    )
                else:
                    self._timestamps.append(time.perf_counter() - start_t)
                    self._positions.append(frame)
                next_t += period
                wait = next_t - time.perf_counter()
                if wait > 0:
                    precise_sleep(wait)
                else:
                    # We're falling behind — reset the target so we don't
                    # spin trying to catch up.
                    next_t = time.perf_counter()
        except BaseException as e:  # pragma: no cover — defensive
            self._error = e
            logger.exception("TrajectoryRecorder loop crashed")

    def stop(self, timeout_s: float = 2.0) -> dict[str, Any]:
        """Stop the recording thread and return the captured trajectory."""
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=timeout_s)
            if self._thread.is_alive():
                logger.warning("TrajectoryRecorder thread did not stop within %.1fs", timeout_s)
        if self._error is not None:
            raise self._error
        return {
            "schema_version": SCHEMA_VERSION,
            "robot_type": getattr(self.robot, "name", type(self.robot).__name__),
            "fps": self.fps,
            "joints": self._joints,
            "timestamps": list(self._timestamps),
            "positions": list(self._positions),
        }

    def cancel(self, timeout_s: float = 2.0) -> None:
        """Stop the thread and discard the captured trajectory."""
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=timeout_s)


def validate_trajectory(trajectory: dict[str, Any]) -> None:
    """Raise ``ValueError`` if the trajectory dict is structurally invalid."""
    required = {"schema_version", "fps", "joints", "timestamps", "positions"}
    missing = required - set(trajectory.keys())
    if missing:
        raise ValueError(f"Trajectory missing required keys: {sorted(missing)}")
    if trajectory["schema_version"] != SCHEMA_VERSION:
        raise ValueError(
            f"Unsupported trajectory schema_version={trajectory['schema_version']} "
            f"(this code handles version {SCHEMA_VERSION})"
        )
    n_joints = len(trajectory["joints"])
    n_frames = len(trajectory["timestamps"])
    if len(trajectory["positions"]) != n_frames:
        raise ValueError(f"positions has {len(trajectory['positions'])} rows but timestamps has {n_frames}")
    for i, row in enumerate(trajectory["positions"]):
        if len(row) != n_joints:
            raise ValueError(f"positions[{i}] has {len(row)} values but joints has {n_joints}")


def replay_trajectory(
    robot: Robot,
    trajectory: dict[str, Any],
    ramp_to_start_s: float = 2.0,
) -> None:
    """Replay a recorded trajectory open-loop on the robot.

    Preconditions:
    - Robot must be connected. Torque must be enabled (the default after
      ``connect()`` on most followers).
    - ``trajectory["joints"]`` must be a subset of the robot's
      ``action_features``. Extra/missing joints raise ``ValueError``.

    The robot is first smoothly ramped from its current pose to the
    trajectory's first frame over ``ramp_to_start_s`` seconds (linear
    interpolation at 50 Hz), then the recorded frames are played back
    at the recorded fps. No closed-loop tracking, no velocity clamps —
    the recorded human motion is the safety envelope.
    """
    validate_trajectory(trajectory)

    joints: list[str] = trajectory["joints"]
    timestamps: list[float] = trajectory["timestamps"]
    positions: list[list[float]] = trajectory["positions"]

    if not timestamps:
        logger.warning("Trajectory is empty, nothing to replay")
        return

    unknown = [j for j in joints if j not in robot.action_features]
    if unknown:
        raise ValueError(
            f"Trajectory joints not in robot.action_features: {unknown}. "
            f"Robot has: {sorted(robot.action_features.keys())}"
        )

    # Smooth ramp from current pose to the trajectory's first frame.
    if ramp_to_start_s > 0:
        obs = robot.get_observation()
        missing = [j for j in joints if j not in obs]
        if missing:
            # Falling back to target_pose for missing joints would silently
            # collapse the ramp to an instant jump on those joints — exactly
            # what the ramp is supposed to prevent. Fail loudly instead so
            # the operator hears it before the motors move.
            raise RuntimeError(
                f"Cannot ramp: robot observation missing joints {missing} "
                f"(present: {sorted(obs.keys())}). Check motor connectivity."
            )
        start_pose = {j: float(obs[j]) for j in joints}
        target_pose = {j: positions[0][i] for i, j in enumerate(joints)}
        ramp_hz = 50
        num_steps = max(1, int(ramp_to_start_s * ramp_hz))
        dt = ramp_to_start_s / num_steps
        logger.info("Ramping to trajectory start over %.1fs", ramp_to_start_s)
        for step in range(1, num_steps + 1):
            t0 = time.perf_counter()
            alpha = step / num_steps
            action = {j: start_pose[j] * (1.0 - alpha) + target_pose[j] * alpha for j in joints}
            robot.send_action(action)
            precise_sleep(max(dt - (time.perf_counter() - t0), 0.0))

    # Now replay the trajectory itself at its recorded cadence.
    logger.info(
        "Replaying trajectory: %d frames over %.1fs at %d fps",
        len(timestamps),
        timestamps[-1],
        trajectory["fps"],
    )
    replay_start = time.perf_counter()
    for ts, frame in zip(timestamps, positions, strict=True):
        target_t = replay_start + ts
        wait = target_t - time.perf_counter()
        if wait > 0:
            precise_sleep(wait)
        action = {j: frame[i] for i, j in enumerate(joints)}
        robot.send_action(action)

    logger.info("Replay complete")

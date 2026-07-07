#!/usr/bin/env python

# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Teleoperator implementation that replays a saved trajectory file.

The whole control loop runs unchanged — only the leader oracle is swapped
for a file reader. After the last frame, ``is_exhausted`` flips to True so
the calling loop can exit cleanly (one trajectory = one episode).

Lookahead lives in the consuming robot, not here. The teleop exposes
two methods:

  * ``get_action()`` — single intent at "now", interpolated between
    bracketing trajectory frames. Back-compat path: any loop driver
    that doesn't recognise ``ActionChunk`` keeps working unchanged.
  * ``get_action_with_horizon()`` — a sliding window of upcoming
    intent samples as an ``ActionChunk`` starting at "now". Chunk-aware
    robots (e.g. ``SO107FollowerPredictive``) use this for exact-lookup
    motor-τ compensation at ``now + L``.
"""

import json
import logging
import time
from pathlib import Path

from lerobot.robots.safe_trajectory import validate_trajectory
from lerobot.types import ActionChunk, RobotAction

from ..teleoperator import Teleoperator
from .configuration_trajectory_replay import TrajectoryReplayTeleopConfig

logger = logging.getLogger(__name__)


class TrajectoryReplayTeleop(Teleoperator):
    """Stand-in "leader" that emits frames from a recorded trajectory.

    Frame selection is by **elapsed wall time since first** ``get_action()``,
    not iteration count: if the calling loop overruns we skip frames, if
    the loop underruns we repeat a frame. Either way the trajectory always
    finishes in the same wall time as it was recorded.

    Exhaustion (``is_exhausted == True``) is reached when elapsed time
    exceeds the last recorded timestamp. The calling loop is expected to
    check ``is_exhausted`` and exit.
    """

    config_class = TrajectoryReplayTeleopConfig
    name = "trajectory_replay"

    def __init__(self, config: TrajectoryReplayTeleopConfig) -> None:
        super().__init__(config)
        self.config = config
        self._trajectory: dict | None = None
        self._timestamps: list[float] = []
        self._positions: list[list[float]] = []
        self._joints: list[str] = []
        self._fps: float = 0.0
        self._start_t: float | None = None
        self._exhausted: bool = False

    @property
    def action_features(self) -> dict[str, type]:
        return dict.fromkeys(self._joints, float)

    @property
    def feedback_features(self) -> dict[str, type]:
        return {}

    @property
    def is_connected(self) -> bool:
        return self._trajectory is not None

    @property
    def is_calibrated(self) -> bool:
        return True

    @property
    def is_exhausted(self) -> bool:
        """True once the recorded duration has elapsed since first
        ``get_action()``. The loop driver should break when this flips."""
        return self._exhausted

    @property
    def duration_s(self) -> float:
        return self._timestamps[-1] if self._timestamps else 0.0

    @property
    def frame_count(self) -> int:
        return len(self._timestamps)

    @property
    def start_pose(self) -> dict[str, float]:
        """First-frame joint positions, suitable for ``move_to_rest_position``.

        Recording's invariant is that frame 0 IS the rest_position of the
        robot — recording always starts after a move-to-rest. Callers can
        use this for an automated reset between episodes (e.g. multi-
        episode dataset recording) without needing the profile's
        rest_position dict separately.

        Raises ``RuntimeError`` if called before ``connect()``.
        """
        if self._trajectory is None:
            raise RuntimeError("start_pose accessed before connect()")
        return {j: float(p) for j, p in zip(self._joints, self._positions[0], strict=True)}

    def connect(self, calibrate: bool = True) -> None:
        if not self.config.trajectory_path:
            raise ValueError("trajectory_path is empty; pass --teleop.trajectory_path=...")
        path = Path(self.config.trajectory_path).expanduser()
        if not path.exists():
            raise FileNotFoundError(f"Trajectory file not found: {path}")
        traj = json.loads(path.read_text())
        validate_trajectory(traj)
        if not traj["timestamps"]:
            raise ValueError(f"Trajectory at {path} is empty")
        fps = float(traj.get("fps", 0))
        if fps <= 0:
            raise ValueError(f"Trajectory at {path} has non-positive fps={fps}")
        self._trajectory = traj
        self._timestamps = list(traj["timestamps"])
        self._positions = list(traj["positions"])
        self._joints = list(traj["joints"])
        self._fps = fps
        self._start_t = None
        self._exhausted = False
        logger.info(
            "TrajectoryReplayTeleop loaded %s: %d frames over %.1fs (recorded at %.1f fps)",
            path,
            len(self._timestamps),
            self.duration_s,
            fps,
        )

    def calibrate(self) -> None:
        pass

    def configure(self) -> None:
        pass

    def get_action(self) -> RobotAction:
        """Return the single intent for the current wall-clock tick.

        Back-compat path: any loop driver that doesn't call
        ``get_action_with_horizon`` keeps working as before. The predictive
        robot will fall back to its velocity-extrapolation path in that
        case — slightly less accurate than exact-lookup but still correct.
        """
        elapsed = self._elapsed_now()
        if elapsed >= self._timestamps[-1]:
            self._exhausted = True
            frame = list(self._positions[-1])
        else:
            frame = self._interp_at(elapsed)
        return {j: float(p) for j, p in zip(self._joints, frame, strict=True)}

    def get_action_with_horizon(self) -> ActionChunk:
        """Return a chunk of upcoming intent samples starting at "now".

        ``frames[0]`` matches ``get_action()`` at the same tick (same
        ``_elapsed_now`` snapshot). The remainder is generated at
        ``self._fps`` cadence by interpolating the trajectory at
        ``elapsed + k / fps``.

        Past the trajectory's end the chunk holds at the rest pose; the
        consumer is expected to honour ``is_exhausted`` and stop the loop.
        """
        elapsed = self._elapsed_now()
        if elapsed >= self._timestamps[-1]:
            self._exhausted = True

        fps = self._fps
        # +1 so the integer count covers exactly the requested window:
        # window_s = 0.5 s @ 30 fps → 15 inter-frame spans + 1 = 16 frames.
        # max(2, ...) so we always have at least two frames for the
        # consumer's chunk-tail extrapolation if it needs it.
        n_frames = max(2, int(self.config.chunk_window_s * fps) + 1)
        frames = tuple(
            {j: float(p) for j, p in zip(self._joints, self._interp_at(elapsed + k / fps), strict=True)}
            for k in range(n_frames)
        )
        return ActionChunk(fps=fps, frames=frames)

    # ── Internals ───────────────────────────────────────────────────────

    def _elapsed_now(self) -> float:
        """Wall-time elapsed since the first ``get_action*`` call.

        Anchoring on first-call (rather than ``connect()``) lets the
        loop driver do setup work between connect and the first tick
        without burning trajectory time. Both ``get_action`` and
        ``get_action_with_horizon`` go through this, so they agree on
        the anchor regardless of which one is called first.
        """
        if self._trajectory is None:
            raise RuntimeError("TrajectoryReplayTeleop accessed before connect()")
        now = time.perf_counter()
        if self._start_t is None:
            self._start_t = now
        return now - self._start_t

    def _interp_at(self, query_t: float) -> list[float]:
        """Linear interpolation at ``query_t``. Clamps to the endpoints."""
        n = len(self._timestamps)
        if query_t >= self._timestamps[-1]:
            return list(self._positions[-1])
        if query_t <= self._timestamps[0]:
            return list(self._positions[0])
        idx = self._locate_frame(query_t)
        if idx + 1 >= n:
            return list(self._positions[idx])
        t0 = self._timestamps[idx]
        t1 = self._timestamps[idx + 1]
        dt = t1 - t0
        alpha = (query_t - t0) / dt if dt > 1e-9 else 0.0
        p0 = self._positions[idx]
        p1 = self._positions[idx + 1]
        return [a + (b - a) * alpha for a, b in zip(p0, p1, strict=True)]

    def _locate_frame(self, elapsed: float) -> int:
        """Index of the latest frame whose timestamp <= elapsed.

        Maintains a forward-walking cursor so the common case is O(1)
        amortised. Resets if the query goes backwards (e.g. in tests or
        when a new chunk extrapolates past the end and we then re-query
        an earlier elapsed within the same tick).
        """
        cursor = getattr(self, "_cursor", 0)
        n = len(self._timestamps)
        if cursor >= n or self._timestamps[cursor] > elapsed:
            cursor = 0
        while cursor + 1 < n and self._timestamps[cursor + 1] <= elapsed:
            cursor += 1
        self._cursor = cursor
        return cursor

    def send_feedback(self, feedback: dict) -> None:
        # No physical leader — feedback is a no-op.
        pass

    def disconnect(self) -> None:
        self._trajectory = None
        self._timestamps = []
        self._positions = []
        self._joints = []
        self._fps = 0.0
        self._start_t = None
        self._exhausted = False
        if hasattr(self, "_cursor"):
            delattr(self, "_cursor")

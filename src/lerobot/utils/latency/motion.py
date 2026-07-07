"""Per-tick motion logger: intent + state per iteration → .npz sidecar.

The latency framework captures per-iteration *stage timings*. This module
captures the *motion data itself* — per-tick leader intent (or policy
output) and follower state — into a small .npz file alongside the
latency snapshot.

Intended use: enabled when the user wants to compare controller knobs
(``corrector_alpha``, ``lookahead_ms``, ...) and needs to compute
jitter / lag from a real teleop or record session — not just stage
timings. Kept lightweight: just two numpy arrays appended per tick,
dumped on close.

Output path: ``<output_dir>/motion_<YYYYMMDD_HHMMSS>.npz``. A new
timestamped file per run so back-to-back runs don't clobber each other
(unlike the latency snapshot, which is overwritten on each run by
design).
"""

from __future__ import annotations

import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


class MotionLogger:
    """Append-only per-tick intent + state logger; dumps to .npz on close.

    Preconditions:
      - ``output_dir`` is writable; created on first write.
      - ``tick()`` is called from the loop thread once per iteration.

    Postconditions:
      - On ``close()``, a file
        ``<output_dir>/motion_<YYYYMMDD_HHMMSS>.npz`` exists with arrays
        ``t``, ``intent``, ``state``, ``joint_names``.
      - If ``close()`` is never called (e.g. crash), no file is written —
        partial logs are not flushed to avoid corrupted output.

    Memory: ~16 floats per joint × 2 arrays per tick. For 14 joints at
    30 Hz over 60 s, ~50 KB in memory. At 200 Hz over 5 min, ~3 MB.
    Negligible relative to the snapshot writer's own overhead.
    """

    def __init__(self, output_dir: str | Path) -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.path = self.output_dir / f"motion_{ts}.npz"
        self._records: list[tuple[float, np.ndarray, np.ndarray]] = []
        self._joint_names: list[str] | None = None
        self._t0 = time.perf_counter()

    def tick(self, intent: dict[str, Any], state: dict[str, Any]) -> None:
        """Append one iteration's intent + state.

        ``intent`` and ``state`` are flat dicts ``{"motor_name.pos": float}``.
        Only the intersection of keys ending with ``.pos`` is logged — non-
        position values (images, camera tensors) are silently skipped.
        The joint name order is fixed from the FIRST call and used for all
        subsequent records; any keys added later are dropped.
        """
        if self._joint_names is None:
            self._joint_names = sorted(
                k for k in set(intent) & set(state) if isinstance(k, str) and k.endswith(".pos")
            )
            if not self._joint_names:
                logger.warning(
                    "MotionLogger.tick: no .pos keys in common between intent and state; nothing to log"
                )
                return
        try:
            intent_arr = np.fromiter(
                (float(intent[j]) for j in self._joint_names), dtype=np.float64, count=len(self._joint_names)
            )
            state_arr = np.fromiter(
                (float(state[j]) for j in self._joint_names), dtype=np.float64, count=len(self._joint_names)
            )
        except (KeyError, TypeError, ValueError):
            # Missing key or non-float value — skip this tick rather than
            # corrupt the trace. Happens during e.g. brief reconnect blips.
            return
        self._records.append((time.perf_counter() - self._t0, intent_arr, state_arr))

    def close(self) -> None:
        if not self._records or self._joint_names is None:
            logger.info("MotionLogger.close: no records, not writing %s", self.path)
            return
        try:
            np.savez(
                self.path,
                t=np.asarray([r[0] for r in self._records]),
                intent=np.stack([r[1] for r in self._records]),
                state=np.stack([r[2] for r in self._records]),
                joint_names=np.asarray(self._joint_names),
            )
            logger.info("MotionLogger wrote %s (%d ticks)", self.path, len(self._records))
        except OSError:
            logger.exception("MotionLogger: failed to write %s", self.path)

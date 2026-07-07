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

"""Mixin that adds a background-thread bus poller to any leader teleop.

Use::

    class MyLeaderHighRate(HighRateLeaderMixin, MyLeader):
        config_class = MyLeaderHighRateConfig
        name = "my_leader_highrate"

The mixin order matters: ``HighRateLeaderMixin`` must come BEFORE the
base leader class in the MRO so its method overrides take precedence
and chain into the base via ``super()``.

The base leader is expected to:
  * set ``self.bus`` to a ``SerialMotorsBus`` (any flavour that exposes
    ``sync_read("Present_Position")``) in its ``__init__``.
  * implement a ``get_action()`` that does a blocking bus read returning
    ``{f"{motor}.pos": value}`` — the mixin's override falls back to
    this on cache-cold get_action calls before the read thread warms up.

The config is expected to include :class:`HighRateLeaderConfig` fields
(``read_rate_hz``).
"""

from __future__ import annotations

import logging
import threading
import time

from lerobot.motors.locked_bus import LockedBus
from lerobot.utils.robot_utils import precise_sleep

logger = logging.getLogger(__name__)


class HighRateLeaderMixin:
    """Inheritance mixin: layers a background bus-read thread on a leader.

    Responsibilities:
      * Wrap ``self.bus`` in :class:`LockedBus` so the background read
        thread + any main-thread bus access (disable_torque, configure,
        soft-land per-motor writes) serialize automatically.
      * Run a background thread at ``config.read_rate_hz`` that
        ``sync_read``s ``Present_Position`` and stashes the pose in a
        guarded cache.
      * Override ``get_action()`` to return the cached pose (or fall
        through to the base's blocking read if the cache isn't warm
        yet).
      * Stop the thread in ``disconnect()`` BEFORE the bus closes (the
        thread would otherwise race with bus.disconnect()).

    Cache-state invariants:
      * ``self._cached_pose is None`` ⟺ the read thread hasn't yet
        produced its first sample, or the leader has just been
        disconnected (cleared on disconnect to avoid stale-data leak
        across reconnects).
      * Cache-hit path is lock-free in the steady state — the read
        thread atomically reassigns ``self._cached_pose`` to a new
        ``dict``; we copy under ``_cache_lock`` so the caller can
        mutate without poisoning the next consumer.
    """

    bus: LockedBus

    def __init__(self, config) -> None:  # noqa: D401 — mixin __init__
        super().__init__(config)
        assert hasattr(self, "bus") and self.bus is not None, (
            f"{type(self).__name__}: base class __init__ did not set self.bus before "
            f"HighRateLeaderMixin.__init__"
        )
        assert not isinstance(self.bus, LockedBus), (
            f"{type(self).__name__}: self.bus is already wrapped in LockedBus. "
            f"The mixin owns this wrapping — base classes must not pre-wrap."
        )
        self.bus = LockedBus(self.bus)
        # Cache state — see class docstring for invariants.
        self._cache_lock = threading.Lock()
        self._cached_pose: dict[str, float] | None = None
        self._read_stop = threading.Event()
        self._read_thread: threading.Thread | None = None

    # ── Lifecycle ─────────────────────────────────────────────────────────

    def connect(self, calibrate: bool = True) -> None:
        """Run base connect() then start the background read thread."""
        super().connect(calibrate)
        self._start_read_thread()
        logger.info(
            "%s: background read thread started at %.0f Hz",
            self,
            self.config.read_rate_hz,
        )

    def disconnect(self) -> None:
        """Stop the read thread BEFORE chaining to base disconnect.

        Otherwise the thread's next ``sync_read`` races with
        ``bus.disconnect()`` inside the base and raises.

        Also clears the pose cache so a subsequent reconnect can't
        return a stale pose from the previous session before the new
        read thread has produced its first sample. With the cache
        cleared, ``get_action()`` falls through to the parent's
        blocking ``sync_read`` until the new thread warms up.
        """
        self._read_stop.set()
        if self._read_thread is not None and self._read_thread.is_alive():
            self._read_thread.join(timeout=2.0)
            if self._read_thread.is_alive():
                logger.warning("%s: read thread did not stop within 2s", self)
        self._read_thread = None
        with self._cache_lock:
            self._cached_pose = None
        super().disconnect()

    # ── Caller-facing methods ─────────────────────────────────────────────

    def get_action(self) -> dict[str, float]:
        """Return the most recent cached leader pose.

        Steady-state path is lock-free (the cache reference is swapped
        atomically by the read thread). The very first call before the
        cache is warm — i.e. before the read thread has produced its
        first sample — falls through to the parent's blocking
        ``sync_read`` so callers never see a None / empty dict.
        """
        with self._cache_lock:
            pose = self._cached_pose
        if pose is not None:
            # Return a copy so the caller can mutate without poisoning
            # the cache for the next consumer.
            return dict(pose)
        # Cache not warmed yet. Block once on a direct read so the
        # caller has something to work with.
        return super().get_action()

    # ── Background read loop ──────────────────────────────────────────────

    def _start_read_thread(self) -> None:
        """Idempotent thread launch — safe to call from any reconnect path."""
        if self._read_thread is not None and self._read_thread.is_alive():
            return
        self._read_stop.clear()
        self._read_thread = threading.Thread(
            target=self._read_loop,
            name=f"{self}_leader_read",
            daemon=True,
        )
        self._read_thread.start()

    def _read_loop(self) -> None:
        period = 1.0 / self.config.read_rate_hz
        next_tick = time.perf_counter()
        while not self._read_stop.is_set():
            try:
                pose = self.bus.sync_read("Present_Position")
                pose = {f"{motor}.pos": float(val) for motor, val in pose.items()}
                with self._cache_lock:
                    self._cached_pose = pose
            except Exception:
                # A transient bus blip shouldn't kill the thread —
                # the cache stays at the last good value, the next
                # iteration tries again. Log so we hear about it.
                logger.exception("%s: leader read failed", self)
            next_tick += period
            sleep_for = next_tick - time.perf_counter()
            if sleep_for > 0:
                precise_sleep(sleep_for)
            else:
                # Fell behind (bus retry, sleep jitter) — reset target
                # so we don't spin trying to catch up over many ticks.
                next_tick = time.perf_counter()

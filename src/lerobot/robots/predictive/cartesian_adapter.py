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

"""Adapter that exposes a bimanual Cartesian teleop as two per-arm joint teleops.

The predictive follower's per-arm controllers each poll a sub-teleop at
200 Hz, expecting an unprefixed ``{<motor>.pos: float}`` dict — the
contract a joint-space leader (``SO107LeaderHighRate``) satisfies. A
bimanual Cartesian teleop (Quest VR) instead emits one combined
EE-delta dict with ``left_target_x/y/z/wx/wy/wz, left_gripper_pos`` +
right keys. Without a shim, the per-arm controller would either crash
on missing motor keys or treat ``left_target_x`` as a motor position.

:class:`BimanualCartesianIKAdapter` wraps the Cartesian teleop and an
already-built bimanual IK transform (the same one
:func:`make_bimanual_ik_transform` returns), runs the IK on a background
thread at WebXR rate (90 Hz default), and exposes ``.left_arm`` /
``.right_arm`` sub-teleops whose ``get_action()`` returns the per-arm
joint dict from the cached output. Both per-arm controllers poll the
cache lock-free; IK does not re-run per poll.

The adapter does **not** install a transform on the Cartesian teleop; it
just *reads* the teleop's native EE-delta output. The wrapping robot's
``attach_teleop`` is the one that decides what to install on the teleop
(typically a thin "return the adapter's cached joint dict" transform so
the script-side ``teleop.get_action()`` keeps returning a recordable
joint dict consistent with the plain-follower path).

Lifetime is owned by the wrapping robot's ``attach_teleop`` / detach:
``start()`` on attach, ``stop()`` on detach. The thread is daemonic so a
forgotten stop never blocks process exit.
"""

from __future__ import annotations

import logging
import threading
import time
from collections.abc import Callable
from typing import Any

logger = logging.getLogger(__name__)


class BimanualCartesianIKAdapter:
    """Make a bimanual Cartesian teleop look like two per-arm joint teleops.

    The IK is run in a background thread; both per-arm sub-teleops read
    the cached result through a short cache-lock copy. Coalesces: new
    Cartesian frames flowing in faster than ``rate_hz`` are absorbed by
    only using the latest on each tick.
    """

    def __init__(
        self,
        teleop: Any,
        transform: Callable[[dict], dict],
        rate_hz: float = 90.0,
    ) -> None:
        """Construct an adapter.

        Args:
            teleop: A bimanual Cartesian teleoperator that ``get_action()``
                returns a dict with the ``left_/right_`` EE-delta keys.
                The teleop must NOT have an ``action_transform`` installed
                — the adapter reads the raw EE-delta output and runs the
                transform itself.
            transform: The bimanual IK transform (e.g. the callable from
                :func:`make_bimanual_ik_transform`). Takes the teleop's
                EE-delta dict, returns a prefixed joint dict
                (``{left_<m>.pos: float, right_<m>.pos: float}``).
            rate_hz: IK tick rate. Default 90 Hz — Quest's native WebXR
                frame rate. Higher rates pointlessly re-run IK on
                unchanged input; lower rates limit how fresh the cache
                is when the per-arm 200 Hz controllers poll.
        """
        self._teleop = teleop
        self._transform = transform
        self._rate_hz = float(rate_hz)

        self._cached: dict[str, float] | None = None
        self._cache_lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._n_ticks = 0
        self._n_skipped = 0

        # Sub-teleops the per-arm predictive controllers will poll. Only
        # ``get_action()`` is needed; both sub-teleops just filter the
        # parent's cached prefixed dict to one arm's keys.
        self.left_arm = _PerArmSubTeleop(self, "left_")
        self.right_arm = _PerArmSubTeleop(self, "right_")

    # ── Lifecycle ─────────────────────────────────────────────────────────

    @property
    def is_running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def start(self) -> None:
        """Start the background IK thread. Idempotent."""
        if self.is_running:
            return
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._loop,
            name="bimanual_cart_ik_adapter",
            daemon=True,
        )
        self._thread.start()
        logger.info(
            "BimanualCartesianIKAdapter started: wrapping %r at %.0f Hz",
            type(self._teleop).__name__,
            self._rate_hz,
        )

    def stop(self) -> None:
        """Stop the IK thread and clear the cache. Idempotent."""
        self._stop.set()
        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=2.0)
            if self._thread.is_alive():
                logger.warning("BimanualCartesianIKAdapter: thread did not stop within 2 s")
        self._thread = None
        with self._cache_lock:
            self._cached = None
        logger.info(
            "BimanualCartesianIKAdapter stopped (ran %d IK ticks, skipped %d)",
            self._n_ticks,
            self._n_skipped,
        )

    # ── Cache access ──────────────────────────────────────────────────────

    def get_full_joint_action(self) -> dict[str, float] | None:
        """Return the latest merged prefixed joint dict, or None if not yet warm.

        Useful as the source for an ``action_transform`` installed on the
        Cartesian teleop so the script-side ``teleop.get_action()`` keeps
        returning a recordable joint dict (the same dict the per-arm
        controllers are pulling).
        """
        with self._cache_lock:
            return dict(self._cached) if self._cached is not None else None

    @property
    def hold_per_arm(self) -> tuple[bool, bool]:
        """Per-arm IK-hold state from the wrapped transform.

        Surfaced so a teleop that polls its installed action_transform
        for ``hold_per_arm`` (Quest VR's haptic-rumble path) sees through
        the adapter to the underlying ``BimanualSO107IKTransform`` —
        without this the rumble dies at the adapter wrapper. ``(False,
        False)`` if the wrapped transform doesn't expose it (e.g. a
        generic bimanual transform).
        """
        return getattr(self._transform, "hold_per_arm", (False, False))

    def _filtered(self, prefix: str) -> dict[str, float]:
        """Return one arm's joint dict (unprefixed) from the cache, or {}."""
        with self._cache_lock:
            full = self._cached
        if full is None:
            return {}
        n = len(prefix)
        return {k[n:]: float(v) for k, v in full.items() if k.startswith(prefix) and k.endswith(".pos")}

    # ── Background loop ───────────────────────────────────────────────────

    def _loop(self) -> None:
        period = 1.0 / self._rate_hz
        next_tick = time.perf_counter()
        while not self._stop.is_set():
            try:
                self._tick()
            except Exception:
                logger.exception("BimanualCartesianIKAdapter tick failed")
            next_tick += period
            remaining = next_tick - time.perf_counter()
            if remaining > 0:
                # Use the stop event's wait() so .stop() unblocks promptly.
                self._stop.wait(timeout=remaining)
            else:
                # IK fell behind — reset target so we don't spin to catch up.
                next_tick = time.perf_counter()

    def _tick(self) -> None:
        # Always read the *raw* Cartesian-delta action. The wrapping robot
        # typically installs an ``action_transform`` on the teleop that
        # returns this adapter's cached joint dict (for recording
        # consistency with the plain-follower path); going through
        # ``get_action()`` here would feed those joints back into the
        # adapter's transform call below — a silent feedback loop. The
        # teleop is required to expose ``get_action_raw()``; this is part
        # of the Cartesian-teleop contract.
        cartesian = self._teleop.get_action_raw()
        if not cartesian:
            self._n_skipped += 1
            return
        joint = self._transform(cartesian)
        with self._cache_lock:
            self._cached = joint
        self._n_ticks += 1


class _PerArmSubTeleop:
    """Per-arm view exposing ``get_action()`` on the parent adapter's cache."""

    def __init__(self, parent: BimanualCartesianIKAdapter, prefix: str) -> None:
        self._parent = parent
        self._prefix = prefix

    def get_action(self) -> dict[str, float]:
        return self._parent._filtered(self._prefix)

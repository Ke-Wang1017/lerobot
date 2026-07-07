"""Thread-safe proxy around a motor bus.

The upstream :class:`SerialMotorsBus` has no internal locking — it
assumes one owner thread. Most of the codebase is fine with that:
plain followers and leaders have a single owner. But two classes in
this branch are explicitly multi-threaded:

  * :class:`SO107LeaderHighRate` — background reader at 200 Hz +
    main-thread torque toggles / configure / disconnect.
  * :class:`SO107FollowerPredictive` — controller writer thread at
    200 Hz + main-thread :meth:`get_observation` reads + soft-land
    register writes.

In both, two threads end up calling into the same Feetech half-duplex
serial port concurrently. The SDK's ``port.is_using`` check-and-set
isn't atomic and has *no wait semantics* — concurrent callers either
clobber each other's TX or bail with ``[TxRxResult] Port is in use!``
(see ``scservo_sdk.protocol_packet_handler.txPacket``).

Rather than modify upstream ``motors_bus.py`` (which causes merge
conflicts on every rebase), we wrap the bus at construction time in
:class:`LockedBus`. The proxy holds a ``threading.RLock`` and
serializes every I/O method through it. Single-threaded consumers do
nothing different — they keep using the unwrapped bus and pay no
overhead. Multi-threaded owners do ``self.bus = LockedBus(self.bus)``
right after construction and inherit thread safety for free.

Why a proxy instead of subclassing: the bus instance is built by the
parent class' ``__init__`` in most cases (e.g. ``SO107Leader`` builds
the bus, then ``SO107LeaderHighRate.__init__`` wraps it). A subclass
would require duplicating the construction. The proxy lets us swap in
the wrapper at any point post-construction.

Why ``RLock`` over ``Lock``: the public API methods on the bus
sometimes call other public API methods (e.g. ``enable_torque``
iterates over motors calling ``write`` per motor — both paths go
through the proxy). With ``RLock``, recursive acquisition from the
same thread is fine. With ``Lock``, it would deadlock.

Overhead: ~62 ns per uncontended ``with self._lock`` on a modern
x86. Compared to a Feetech serial round-trip (~1 ms), invisible.
Under contention the loser waits one round-trip — exactly what we
want vs the current "fail with COMM_PORT_BUSY" behaviour.
"""

from __future__ import annotations

import threading
from typing import Any


class LockedBus:
    """Forwards every attribute to the wrapped bus; serializes I/O."""

    # Slots-based to make typos at our own attrs fail loudly rather than
    # silently shadow underlying-bus attrs. ``_bus`` and ``_lock`` are
    # the only state the proxy carries; everything else is delegated.
    __slots__ = ("_bus", "_lock")

    def __init__(self, bus: Any) -> None:
        # Use object.__setattr__ to bypass __setattr__ delegation below.
        object.__setattr__(self, "_bus", bus)
        object.__setattr__(self, "_lock", threading.RLock())

    # ── Delegation for non-I/O attributes ────────────────────────────
    # __getattr__ is called only when normal lookup fails — so the
    # methods we override below take precedence, and everything else
    # (``motors``, ``is_connected``, ``port_handler``, calibration
    # helpers, etc.) passes through transparently.

    def __getattr__(self, name: str) -> Any:
        return getattr(self._bus, name)

    def __setattr__(self, name: str, value: Any) -> None:
        # Proxy state goes via object.__setattr__ in ``__init__``. Any
        # other assignment is delegated to the underlying bus — keeps
        # ``bus.attr = value`` semantics identical to the unwrapped
        # version (no surprise where the value ended up).
        if name in ("_bus", "_lock"):
            object.__setattr__(self, name, value)
        else:
            setattr(self._bus, name, value)

    def __repr__(self) -> str:
        return f"LockedBus({self._bus!r})"

    # ── Locked I/O methods ───────────────────────────────────────────

    def read(self, *args: Any, **kwargs: Any) -> Any:
        with self._lock:
            return self._bus.read(*args, **kwargs)

    def write(self, *args: Any, **kwargs: Any) -> Any:
        with self._lock:
            return self._bus.write(*args, **kwargs)

    def sync_read(self, *args: Any, **kwargs: Any) -> Any:
        with self._lock:
            return self._bus.sync_read(*args, **kwargs)

    def sync_write(self, *args: Any, **kwargs: Any) -> Any:
        with self._lock:
            return self._bus.sync_write(*args, **kwargs)

    def enable_torque(self, *args: Any, **kwargs: Any) -> Any:
        with self._lock:
            return self._bus.enable_torque(*args, **kwargs)

    def disable_torque(self, *args: Any, **kwargs: Any) -> Any:
        with self._lock:
            return self._bus.disable_torque(*args, **kwargs)

    def disconnect(self, *args: Any, **kwargs: Any) -> Any:
        # disconnect calls bus.disable_torque() internally (via the
        # default ``disable_torque=True`` arg), which would re-enter
        # this lock. RLock handles that correctly.
        with self._lock:
            return self._bus.disconnect(*args, **kwargs)

"""Tests for the :class:`LockedBus` proxy.

The proxy is structurally trivial — every public bus method is
``with self._lock: return self._bus.method(*args, **kwargs)``. These
tests pin the two things that ARE non-trivial:

  1. Forwarding correctness — every attribute read / write / method
     call falls through to the wrapped bus. Anything that doesn't is a
     silent attribute-shadowing bug that would produce confusing
     "AttributeError on bus" failures downstream.
  2. Locking does NOT deadlock under realistic single-thread re-entry
     (``disconnect`` → internal ``disable_torque``) or under simple
     concurrent multi-thread access. The proxy uses ``RLock`` precisely
     so the re-entry case is safe; the concurrent case is what
     motivated the proxy in the first place.

Not testing: actual mutual-exclusion guarantees, because the contract
is just "serialize I/O". A regression in that would surface as
``COMM_PORT_BUSY`` on hardware, which is the failure mode the proxy
was introduced to fix.
"""

from __future__ import annotations

import threading
import time
from unittest.mock import MagicMock

from lerobot.motors.locked_bus import LockedBus

# =============================================================================
# Forwarding
# =============================================================================


class _DummyBus:
    """Minimal bus stand-in. Just enough attributes + methods to verify
    every proxy method round-trips correctly."""

    def __init__(self):
        self.motors = {"shoulder_pan": object(), "gripper": object()}
        self.is_connected = False
        self.is_calibrated = True
        self.calls: list[tuple[str, tuple, dict]] = []

    def _record(self, name: str, args: tuple, kwargs: dict) -> str:
        self.calls.append((name, args, kwargs))
        return f"{name}-return"

    def read(self, *args, **kwargs):
        return self._record("read", args, kwargs)

    def write(self, *args, **kwargs):
        return self._record("write", args, kwargs)

    def sync_read(self, *args, **kwargs):
        return self._record("sync_read", args, kwargs)

    def sync_write(self, *args, **kwargs):
        return self._record("sync_write", args, kwargs)

    def enable_torque(self, *args, **kwargs):
        return self._record("enable_torque", args, kwargs)

    def disable_torque(self, *args, **kwargs):
        return self._record("disable_torque", args, kwargs)

    def disconnect(self, *args, **kwargs):
        return self._record("disconnect", args, kwargs)

    def some_other_method(self):
        # Not in the locked-set; should pass through via __getattr__.
        return "passthrough"


def test_attribute_read_falls_through():
    """Non-method attributes (motors, is_connected, ...) must read straight
    from the wrapped bus."""
    bus = _DummyBus()
    proxy = LockedBus(bus)
    assert proxy.motors is bus.motors
    assert proxy.is_connected is False
    assert proxy.is_calibrated is True


def test_attribute_write_falls_through_to_bus():
    """Setting an attribute on the proxy must mutate the wrapped bus,
    NOT silently land on the proxy itself."""
    bus = _DummyBus()
    proxy = LockedBus(bus)
    proxy.is_connected = True
    assert bus.is_connected is True
    # The proxy itself has __slots__ and shouldn't carry the new attribute.
    assert not hasattr(proxy.__class__, "is_connected")


def test_locked_method_returns_underlying_value_and_records_call():
    """The locked methods must round-trip args/kwargs verbatim and return
    whatever the wrapped bus returns."""
    bus = _DummyBus()
    proxy = LockedBus(bus)
    result = proxy.sync_write("Goal_Position", {"a": 1.0})
    assert result == "sync_write-return"
    assert bus.calls == [("sync_write", ("Goal_Position", {"a": 1.0}), {})]


def test_all_locked_methods_are_forwarded():
    """Sweep every locked method to confirm none was dropped by a typo."""
    bus = _DummyBus()
    proxy = LockedBus(bus)
    for name in ("read", "write", "sync_read", "sync_write", "enable_torque", "disable_torque", "disconnect"):
        bus.calls.clear()
        getattr(proxy, name)("arg", k=1)
        assert len(bus.calls) == 1, f"{name} did not reach the wrapped bus"
        recorded_name, args, kwargs = bus.calls[0]
        assert recorded_name == name
        assert args == ("arg",)
        assert kwargs == {"k": 1}


def test_non_io_method_falls_through_via_getattr():
    """``__getattr__`` forwards anything we didn't explicitly override.
    Without this, adding a new bus method upstream would silently fail
    when called through the proxy."""
    bus = _DummyBus()
    proxy = LockedBus(bus)
    assert proxy.some_other_method() == "passthrough"


def test_repr_mentions_wrapped_bus():
    """Debug logs format the proxy via str/repr; the wrapped bus's repr
    must surface so log lines aren't useless."""
    bus = MagicMock(name="DummyBus")
    bus.__repr__ = lambda _self: "<MockBus #42>"
    assert repr(LockedBus(bus)) == "LockedBus(<MockBus #42>)"


def test_proxy_state_does_not_leak_to_wrapped_bus():
    """The proxy's own attributes (``_bus``, ``_lock``) must NOT show up
    on the wrapped bus via the __setattr__ delegation. They're handled
    via ``object.__setattr__`` in ``__init__``."""
    bus = _DummyBus()
    LockedBus(bus)
    assert not hasattr(bus, "_bus")
    assert not hasattr(bus, "_lock")


# =============================================================================
# Locking — RLock allows same-thread reentry, no deadlock from realistic use
# =============================================================================


def test_rlock_allows_reentry_from_same_thread():
    """RLock-via-proxy must let the same thread re-enter — the canonical
    case is ``disconnect()`` calling ``disable_torque()`` internally."""

    class _ReentrantBus(_DummyBus):
        def disconnect(self, *args, **kwargs):
            # Re-enter through the proxy. Lock acquired twice in same
            # thread — RLock is OK with this; plain Lock would deadlock.
            self._proxy.disable_torque()
            return super().disconnect(*args, **kwargs)

    bus = _ReentrantBus()
    proxy = LockedBus(bus)
    bus._proxy = proxy  # back-ref so the inner call can re-enter
    # If RLock were Lock this would hang. We don't even need a timeout
    # — pytest's default test timeout will catch it; on success we
    # return immediately.
    proxy.disconnect()
    # Verify both methods reached the bus, in order.
    assert [c[0] for c in bus.calls] == ["disable_torque", "disconnect"]


def test_concurrent_sync_writes_complete_without_deadlock():
    """Two threads hammering the proxy concurrently must all complete.
    This is the failure mode (``COMM_PORT_BUSY``) the proxy was
    introduced to prevent — verify the mutual-exclusion path doesn't
    self-deadlock."""

    class _SlowBus(_DummyBus):
        # Imitate a small serial round-trip so the threads actually
        # interleave; without this, the first thread might finish
        # before the second even starts and the test passes trivially.
        def sync_write(self, *args, **kwargs):
            time.sleep(0.002)
            return super().sync_write(*args, **kwargs)

    bus = _SlowBus()
    proxy = LockedBus(bus)
    n_per_thread = 20
    errors: list[BaseException] = []

    def hammer():
        try:
            for _ in range(n_per_thread):
                proxy.sync_write("Goal_Position", {"a": 1.0})
        except BaseException as e:  # noqa: BLE001
            errors.append(e)

    t1 = threading.Thread(target=hammer)
    t2 = threading.Thread(target=hammer)
    t1.start()
    t2.start()
    # Generous timeout: each call sleeps 2 ms, 40 total calls serialized
    # = ~80 ms wall clock. 5 s is many orders of magnitude over.
    t1.join(timeout=5.0)
    t2.join(timeout=5.0)
    assert not t1.is_alive() and not t2.is_alive(), "thread did not complete (deadlock?)"
    assert not errors, f"thread raised: {errors}"
    assert len(bus.calls) == 2 * n_per_thread


def test_proxy_does_not_introduce_lock_when_attribute_only_read():
    """Non-method attribute reads via ``__getattr__`` must NOT take the
    lock — that would serialize reads of ``.motors`` / ``.is_connected``
    against I/O, defeating the point of those being plain fields. Test:
    while one thread holds the lock inside a slow I/O call, another
    thread reading ``proxy.motors`` should return immediately.
    """

    class _SlowBus(_DummyBus):
        def sync_read(self, *args, **kwargs):
            time.sleep(0.1)
            return super().sync_read(*args, **kwargs)

    bus = _SlowBus()
    proxy = LockedBus(bus)
    motors_read_completed = threading.Event()

    def hold_lock_via_io():
        proxy.sync_read("Present_Position")

    def read_attribute():
        # Should NOT block on the I/O thread's lock.
        _ = proxy.motors
        motors_read_completed.set()

    t_io = threading.Thread(target=hold_lock_via_io)
    t_attr = threading.Thread(target=read_attribute)
    t_io.start()
    # Race: start the I/O thread first, then the attribute read.
    time.sleep(0.01)
    t_attr.start()
    # Attribute read should complete well before the I/O does.
    assert motors_read_completed.wait(timeout=0.05), (
        "attribute read blocked on I/O lock (LockedBus is over-serializing)"
    )
    t_io.join(timeout=1.0)
    t_attr.join(timeout=0.1)

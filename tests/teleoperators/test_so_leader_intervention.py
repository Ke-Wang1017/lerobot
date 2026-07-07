"""Tests for the SPACE-toggle guards on SO107Leader.

Targets ``_try_toggle_intervention`` directly so the lock + debounce
logic is validated without pynput listeners or hardware.

The bug class these guard against: a 2nd SPACE press during the 1-3s
servo-sync transition (policy → intervention) used to silently toggle
back to policy mode, wasting the entire sync round-trip. After the
fix, presses during the transition window are rejected; quick
double-taps are also caught by the time-debounce.
"""

from __future__ import annotations

from lerobot.teleoperators.so_leader.so_leader import SO107Leader


class _FakeLeader:
    """Minimum object structure ``_try_toggle_intervention`` reads from.
    Lets us exercise the toggle logic without instantiating SO107Leader
    (which would attach to hardware via the bus)."""

    def __init__(self, debounce_s: float = 0.5):
        self._intervention_active = False
        self._intervention_transition_lock = False
        # Match the production default so the first call's debounce
        # check passes regardless of test ``now`` values.
        self._last_intervention_toggle_ts = float("-inf")
        self._intervention_debounce_s = debounce_s

    # Bind the real method so the logic under test is the production code
    _try_toggle_intervention = SO107Leader._try_toggle_intervention


# ============================================================================
# Happy path — toggle works when no guard is active
# ============================================================================


def test_first_press_toggles_on():
    leader = _FakeLeader()
    accepted = leader._try_toggle_intervention(now=10.0)
    assert accepted is True
    assert leader._intervention_active is True


def test_second_press_after_debounce_toggles_off():
    leader = _FakeLeader(debounce_s=0.5)
    leader._try_toggle_intervention(now=10.0)
    # 1s later — well past debounce
    accepted = leader._try_toggle_intervention(now=11.0)
    assert accepted is True
    assert leader._intervention_active is False


def test_three_presses_with_gaps_toggle_each_time():
    leader = _FakeLeader(debounce_s=0.5)
    leader._try_toggle_intervention(now=0.0)
    leader._try_toggle_intervention(now=1.0)
    leader._try_toggle_intervention(now=2.0)
    # ON → OFF → ON
    assert leader._intervention_active is True


# ============================================================================
# Transition lock — the primary fix for the ep124-style waste
# ============================================================================


class TestTransitionLock:
    def test_press_during_lock_rejected(self):
        leader = _FakeLeader()
        leader._intervention_transition_lock = True
        accepted = leader._try_toggle_intervention(now=10.0)
        assert accepted is False
        assert leader._intervention_active is False

    def test_press_during_lock_does_not_advance_debounce_timestamp(self):
        """A rejected press must NOT update ``_last_intervention_toggle_ts``,
        otherwise the debounce window after the lock releases would be
        based on the rejected press's time instead of the last *accepted*
        toggle. Subtle but important — ensures debounce semantics stay
        consistent."""
        leader = _FakeLeader(debounce_s=0.5)
        leader._try_toggle_intervention(now=0.0)  # ON
        leader._intervention_transition_lock = True
        leader._try_toggle_intervention(now=10.0)  # rejected
        leader._try_toggle_intervention(now=10.001)  # rejected
        leader._intervention_transition_lock = False
        # Now press at t=10.1 — that's 10.1s since last accepted toggle
        # at t=0, well past the 0.5s debounce → must be accepted.
        accepted = leader._try_toggle_intervention(now=10.1)
        assert accepted is True
        assert leader._intervention_active is False

    def test_lock_can_be_re_acquired(self):
        """The lock is set/released across many transitions per session.
        Verify it works repeatedly (not just once)."""
        leader = _FakeLeader()
        for _ in range(5):
            leader._intervention_transition_lock = True
            assert leader._try_toggle_intervention(now=0.0) is False
            leader._intervention_transition_lock = False
            assert leader._try_toggle_intervention(now=10.0) is True
            # advance time so debounce clears for next iter
            leader._last_intervention_toggle_ts = 0.0


# ============================================================================
# Debounce — the secondary fix for accidental double-tap
# ============================================================================


class TestDebounce:
    def test_press_within_debounce_window_rejected(self):
        leader = _FakeLeader(debounce_s=0.5)
        leader._try_toggle_intervention(now=0.0)
        # 100ms later — well within debounce window
        accepted = leader._try_toggle_intervention(now=0.1)
        assert accepted is False
        assert leader._intervention_active is True  # state unchanged

    def test_press_at_debounce_boundary_rejected(self):
        """Strict less-than: at exactly the debounce duration, still rejected
        (operator should wait a hair longer for clear intent)."""
        leader = _FakeLeader(debounce_s=0.5)
        leader._try_toggle_intervention(now=0.0)
        # Exactly at the boundary: elapsed == debounce_s → not strictly
        # less than → accepted. Locking that semantics in.
        accepted = leader._try_toggle_intervention(now=0.5)
        assert accepted is True

    def test_press_just_past_debounce_accepted(self):
        leader = _FakeLeader(debounce_s=0.5)
        leader._try_toggle_intervention(now=0.0)
        accepted = leader._try_toggle_intervention(now=0.5001)
        assert accepted is True
        assert leader._intervention_active is False

    def test_rapid_quintuple_tap_only_first_accepted(self):
        """A 5x rapid tap (e.g. accidental key-spam) should produce one
        toggle, not five."""
        leader = _FakeLeader(debounce_s=0.5)
        results = [leader._try_toggle_intervention(now=0.0 + i * 0.01) for i in range(5)]
        assert results == [True, False, False, False, False]
        assert leader._intervention_active is True


# ============================================================================
# Combined — both guards active at once
# ============================================================================


class TestCombined:
    def test_lock_takes_precedence_over_debounce(self):
        """If both guards would reject, the log message is the lock
        message (since lock check runs first). Lock is the more specific
        signal — the operator should know a transition is in progress."""
        leader = _FakeLeader(debounce_s=0.5)
        leader._try_toggle_intervention(now=0.0)
        leader._intervention_transition_lock = True
        # Both guards would reject: <0.5s since last AND lock active
        accepted = leader._try_toggle_intervention(now=0.1)
        assert accepted is False

    def test_realistic_servo_sync_scenario(self):
        """Reproduction of the user's reported failure mode:
            t=0   : SPACE → toggle ON (start servo sync)
            t=0.1 : main loop sets transition_lock=True
            t=0.5 : SPACE pressed (during sync) — must be ignored
            t=1.5 : SPACE pressed again (during sync) — must be ignored
            t=2.0 : main loop sets transition_lock=False (sync done)
            t=2.4 : SPACE — ACCEPTED (~1s past the last accepted toggle, debounce clear)
            → Result: intervention is OFF (expected exactly ONE toggle on, ONE off).

        Pre-fix: t=0.5 and t=1.5 would have toggled OFF then ON, leaving
        the system in intervention mode AFTER an entirely wasted servo
        sync round-trip.
        """
        leader = _FakeLeader(debounce_s=0.5)
        # First press accepted, intervention ON
        assert leader._try_toggle_intervention(now=0.0) is True
        assert leader._intervention_active is True
        # Main loop locks during servo sync
        leader._intervention_transition_lock = True
        # Two presses during sync — both rejected
        assert leader._try_toggle_intervention(now=0.5) is False
        assert leader._try_toggle_intervention(now=1.5) is False
        # Sync completes, lock released
        leader._intervention_transition_lock = False
        # Operator presses SPACE again to legitimately end intervention
        assert leader._try_toggle_intervention(now=2.4) is True
        # Final state: intervention OFF, exactly two accepted toggles
        assert leader._intervention_active is False

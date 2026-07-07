"""Tests for ``EpisodeLifecycle`` — the per-episode state object that
replaces the prior racy ``reward_triggered`` / ``abort_triggered`` /
``ignore_triggered`` flags on ``rlt_state``.

The structural fix here is the one-shot ``consume_terminal_for_storage``
contract: even if the inference thread fires N times during the post-
signal window, exactly one of those calls gets the terminal kind, the
rest get ``None`` and write intermediate transitions. Real bug
observed at ep124: 17 ``done=True`` transitions written instead of 1.
The threading test below locks that fix in.
"""

from __future__ import annotations

import threading
import time

import pytest

from lerobot.policies.hvla.rlt.episode import EpisodeLifecycle, TerminalKind

# ============================================================================
# Initial state and lifecycle transitions
# ============================================================================


class TestInitialState:
    def test_fresh_lifecycle_is_inactive(self):
        lc = EpisodeLifecycle()
        assert lc.active is False
        assert lc.peek_terminal() is None
        assert lc.is_ignored() is False
        assert lc.had_intervention is False

    def test_signal_outside_episode_is_silent_no_op(self):
        """signal_terminal / signal_ignore are operator-driven (called
        from the OS keyboard listener thread). The operator's keypress
        timing is async — they may press R/LEFT/DOWN before the first
        episode begins, between episodes during reset, or after the
        last episode ends. None of those are bugs.

        Strict asserting here was a regression: AssertionError raised
        in the listener thread gets swallowed by pynput, making
        hotkeys appear "broken" with no log line. Lenient drop is
        the right behavior — keypress has no effect, no exception,
        no log noise above DEBUG."""
        lc = EpisodeLifecycle()
        # Should NOT raise. State stays clean.
        lc.signal_terminal(TerminalKind.SUCCESS)
        lc.signal_ignore()
        lc.signal_terminal(TerminalKind.ABORT)
        assert lc.peek_terminal() is None
        assert lc.is_ignored() is False

    def test_consume_outside_episode_returns_none(self):
        """Consume from inference thread must NOT raise — the inference
        thread can fire before begin() races in (e.g. very early during
        episode setup). It should just return None."""
        lc = EpisodeLifecycle()
        assert lc.consume_terminal_for_storage() is None

    def test_end_episode_without_begin_asserts(self):
        lc = EpisodeLifecycle()
        with pytest.raises(AssertionError, match="not active"):
            lc.end_episode()


class TestBegin:
    def test_begin_activates(self):
        lc = EpisodeLifecycle()
        lc.begin(buffer_size=42)
        assert lc.active is True
        assert lc.buffer_size_at_start == 42
        assert lc.had_intervention is False
        assert lc.is_ignored() is False
        assert lc.peek_terminal() is None

    def test_begin_resets_intervention_flag(self):
        """Stale intervention flag must NOT leak into the next episode."""
        lc = EpisodeLifecycle()
        lc.begin(0)
        lc.mark_intervention()
        lc.end_episode()
        lc.begin(10)
        assert lc.had_intervention is False

    def test_begin_resets_buffer_size(self):
        lc = EpisodeLifecycle()
        lc.begin(0)
        lc.end_episode()
        lc.begin(99)
        assert lc.buffer_size_at_start == 99

    def test_begin_while_active_asserts(self):
        """Calling begin() on an already-active lifecycle would silently
        overwrite an in-progress episode's state. Hard assert."""
        lc = EpisodeLifecycle()
        lc.begin(0)
        with pytest.raises(AssertionError, match="still active"):
            lc.begin(0)

    def test_begin_with_negative_buffer_size_asserts(self):
        """Defensive: catches caller bugs in computing buffer length."""
        lc = EpisodeLifecycle()
        with pytest.raises(AssertionError, match="buffer_size must be >= 0"):
            lc.begin(-1)

    def test_begin_with_zero_buffer_size_ok(self):
        """Edge of the valid range."""
        lc = EpisodeLifecycle()
        lc.begin(0)
        assert lc.buffer_size_at_start == 0

    def test_begin_after_unconsumed_terminal_asserts(self):
        """The data-loss guard: previous episode signaled but inference
        never wrote the terminal AND the episode wasn't ignored. The
        operator's intent was lost — assert loud rather than silently
        clear it."""
        lc = EpisodeLifecycle()
        lc.begin(0)
        lc.signal_terminal(TerminalKind.ABORT)
        lc.end_episode()
        with pytest.raises(AssertionError, match="never consumed"):
            lc.begin(0)

    def test_begin_after_consumed_terminal_ok(self):
        """Normal happy path: signal, consume, end, begin next episode."""
        lc = EpisodeLifecycle()
        lc.begin(0)
        lc.signal_terminal(TerminalKind.SUCCESS)
        assert lc.consume_terminal_for_storage() == TerminalKind.SUCCESS
        lc.end_episode()
        lc.begin(0)  # must not raise
        assert lc.active is True

    def test_begin_after_ignored_episode_ok(self):
        """If operator ignored the episode (DOWN), the terminal signal
        — if any — becomes irrelevant. The buffer is rolled back, so
        a non-consumed terminal is fine."""
        lc = EpisodeLifecycle()
        lc.begin(0)
        lc.signal_terminal(TerminalKind.ABORT)
        lc.signal_ignore()
        # Operator hit ABORT then changed mind with IGNORE; inference
        # may or may not have consumed. Either way, begin must not raise.
        lc.end_episode()
        lc.begin(0)  # must not raise

    def test_begin_after_ignore_only_ok(self):
        """Pure ignore — no terminal signaled. begin() must not raise."""
        lc = EpisodeLifecycle()
        lc.begin(0)
        lc.signal_ignore()
        lc.end_episode()
        lc.begin(0)  # must not raise


class TestEndEpisode:
    def test_end_deactivates(self):
        lc = EpisodeLifecycle()
        lc.begin(0)
        lc.end_episode()
        assert lc.active is False

    def test_signal_after_end_is_silent_no_op(self):
        """Same lenient behavior in the post-end window: operator may
        press keys after we've ended an episode but before the next
        begin(). Silent drop, no AssertionError that would get
        swallowed by the listener thread."""
        lc = EpisodeLifecycle()
        lc.begin(0)
        lc.end_episode()
        # Must not raise
        lc.signal_terminal(TerminalKind.SUCCESS)
        lc.signal_ignore()
        # State unchanged from post-end (terminal still None, not set)
        assert lc.peek_terminal() is None
        assert lc.is_ignored() is False

    def test_consume_after_end_returns_none(self):
        """Inference thread firing during the post-end window must not
        crash and must not write a terminal."""
        lc = EpisodeLifecycle()
        lc.begin(0)
        lc.signal_terminal(TerminalKind.SUCCESS)
        lc.end_episode()
        assert lc.consume_terminal_for_storage() is None

    def test_end_twice_asserts(self):
        lc = EpisodeLifecycle()
        lc.begin(0)
        lc.end_episode()
        with pytest.raises(AssertionError, match="not active"):
            lc.end_episode()


# ============================================================================
# Terminal signal + consumption — the main bug-fix surface
# ============================================================================


class TestTerminalSignalConsume:
    def test_signal_then_consume_returns_kind(self):
        lc = EpisodeLifecycle()
        lc.begin(0)
        lc.signal_terminal(TerminalKind.SUCCESS)
        assert lc.consume_terminal_for_storage() == TerminalKind.SUCCESS

    def test_consume_without_signal_returns_none(self):
        lc = EpisodeLifecycle()
        lc.begin(0)
        assert lc.consume_terminal_for_storage() is None

    def test_consume_is_one_shot_per_episode(self):
        """THE bug-fix invariant. After the first consume, all subsequent
        consume calls in the same episode return None — even though the
        underlying terminal kind is still set. This is what prevents the
        multi-write at ep124."""
        lc = EpisodeLifecycle()
        lc.begin(0)
        lc.signal_terminal(TerminalKind.ABORT)
        assert lc.consume_terminal_for_storage() == TerminalKind.ABORT
        # 16 follow-up calls — all None
        for _ in range(16):
            assert lc.consume_terminal_for_storage() is None
        # And peek still shows the kind (for main-thread logging)
        assert lc.peek_terminal() == TerminalKind.ABORT

    def test_first_signal_wins(self):
        """Operator hammering the terminal key, or hitting different
        terminals by accident, must not change the recorded outcome."""
        lc = EpisodeLifecycle()
        lc.begin(0)
        lc.signal_terminal(TerminalKind.SUCCESS)
        lc.signal_terminal(TerminalKind.ABORT)
        lc.signal_terminal(TerminalKind.SUCCESS)
        assert lc.peek_terminal() == TerminalKind.SUCCESS
        assert lc.consume_terminal_for_storage() == TerminalKind.SUCCESS

    def test_no_assertion_error_when_called_from_async_listener_path(self):
        """Regression test for the 2026-04-29 hotkey outage. Pynput's
        keyboard listener thread caught an AssertionError raised by
        signal_terminal when called outside an active episode (operator
        pressed LEFT/R/DOWN during the reset phase after one episode
        ended and before the next began). Pynput logged the exception
        and silently continued, making hotkeys appear "broken" with
        no operator-visible feedback.

        Lock in: calling signal_terminal / signal_ignore from a
        listener-style context with no active episode must NEVER
        raise. Silent no-op is the contract."""
        lc = EpisodeLifecycle()
        # Operator presses keys before the first begin() has fired
        # (e.g. process started, listener attached, but main loop
        # hasn't reached the first episode-start yet).
        lc.signal_terminal(TerminalKind.ABORT)
        lc.signal_terminal(TerminalKind.SUCCESS)
        lc.signal_ignore()
        # Begin first episode after the keypresses landed
        lc.begin(0)
        # State is clean — the inactive-window keypresses didn't
        # leave residue that would corrupt this episode
        assert lc.peek_terminal() is None
        assert lc.is_ignored() is False

        # Now end the episode and press more keys during the
        # reset window before the next begin()
        lc.signal_terminal(TerminalKind.SUCCESS)  # legit episode-end
        assert lc.consume_terminal_for_storage() == TerminalKind.SUCCESS
        lc.end_episode()
        # Operator presses LEFT here (the original bug scenario)
        lc.signal_terminal(TerminalKind.ABORT)
        lc.signal_terminal(TerminalKind.ABORT)
        lc.signal_ignore()
        # Next episode begins cleanly
        lc.begin(0)
        assert lc.peek_terminal() is None

    def test_repeat_same_kind_is_silent(self, caplog):
        """Hammering the same terminal key (or held key, or sticky
        listener) is benign — no log noise."""
        lc = EpisodeLifecycle()
        lc.begin(0)
        with caplog.at_level("WARNING", logger="lerobot.policies.hvla.rlt.episode"):
            lc.signal_terminal(TerminalKind.SUCCESS)
            lc.signal_terminal(TerminalKind.SUCCESS)
            lc.signal_terminal(TerminalKind.SUCCESS)
        assert not any(r.levelname == "WARNING" for r in caplog.records), (
            "Repeated SAME-kind signals must not warn — operator hammering "
            "is benign and shouldn't generate log noise."
        )

    def test_conflicting_kind_emits_warning(self, caplog):
        """Different terminal kind after the first is the suspicious
        case: slider-focus bug, accidental adjacent key, listener
        regression. First-wins still holds, but train.log must surface
        the conflict so the operator can investigate."""
        lc = EpisodeLifecycle()
        lc.begin(0)
        lc.signal_terminal(TerminalKind.SUCCESS)
        with caplog.at_level("WARNING", logger="lerobot.policies.hvla.rlt.episode"):
            lc.signal_terminal(TerminalKind.ABORT)
        warnings = [r for r in caplog.records if r.levelname == "WARNING"]
        assert len(warnings) == 1, f"Expected exactly 1 conflict warning, got {len(warnings)}"
        # Message references both kinds + the diagnostic context
        msg = warnings[0].message
        assert "SUCCESS" in msg and "ABORT" in msg
        assert "first-wins" in msg.lower() or "first wins" in msg.lower()
        # Recorded outcome unchanged
        assert lc.peek_terminal() == TerminalKind.SUCCESS

    def test_conflicting_kind_warning_fires_only_once_per_subsequent(self, caplog):
        """If the operator presses the wrong key 3 times, we want 3
        warnings — each one might encode different diagnostic info
        (e.g. timestamps that show a key-spam pattern)."""
        lc = EpisodeLifecycle()
        lc.begin(0)
        lc.signal_terminal(TerminalKind.SUCCESS)
        with caplog.at_level("WARNING", logger="lerobot.policies.hvla.rlt.episode"):
            lc.signal_terminal(TerminalKind.ABORT)
            lc.signal_terminal(TerminalKind.ABORT)
            lc.signal_terminal(TerminalKind.ABORT)
        warnings = [r for r in caplog.records if r.levelname == "WARNING"]
        assert len(warnings) == 3

    def test_peek_does_not_consume(self):
        lc = EpisodeLifecycle()
        lc.begin(0)
        lc.signal_terminal(TerminalKind.SUCCESS)
        # Peek 100 times
        for _ in range(100):
            assert lc.peek_terminal() == TerminalKind.SUCCESS
        # Consume still works
        assert lc.consume_terminal_for_storage() == TerminalKind.SUCCESS
        # Peek still returns the kind even after consume
        assert lc.peek_terminal() == TerminalKind.SUCCESS


class TestIgnoreSignal:
    def test_ignore_independent_of_terminal(self):
        """Ignore + terminal can both be signaled. Main-thread
        bookkeeping decides which wins (currently: ignore overrides)."""
        lc = EpisodeLifecycle()
        lc.begin(0)
        lc.signal_ignore()
        lc.signal_terminal(TerminalKind.ABORT)
        assert lc.is_ignored() is True
        assert lc.peek_terminal() == TerminalKind.ABORT

    def test_ignore_idempotent(self):
        lc = EpisodeLifecycle()
        lc.begin(0)
        lc.signal_ignore()
        lc.signal_ignore()
        lc.signal_ignore()
        assert lc.is_ignored() is True


class TestIntervention:
    def test_intervention_default_false(self):
        lc = EpisodeLifecycle()
        lc.begin(0)
        assert lc.had_intervention is False

    def test_mark_intervention_persists_within_episode(self):
        lc = EpisodeLifecycle()
        lc.begin(0)
        lc.mark_intervention()
        assert lc.had_intervention is True
        # Subsequent signals don't clear it
        lc.signal_terminal(TerminalKind.SUCCESS)
        lc.consume_terminal_for_storage()
        assert lc.had_intervention is True

    def test_mark_intervention_idempotent(self):
        lc = EpisodeLifecycle()
        lc.begin(0)
        lc.mark_intervention()
        lc.mark_intervention()
        assert lc.had_intervention is True

    def test_mark_intervention_outside_episode_asserts(self):
        """Calling mark_intervention without begin() would let the flag
        leak forward into the next episode and misclassify it as
        non-autonomous. Hard assert."""
        lc = EpisodeLifecycle()
        with pytest.raises(AssertionError, match="not active"):
            lc.mark_intervention()
        # After end_episode, also asserts
        lc.begin(0)
        lc.end_episode()
        with pytest.raises(AssertionError, match="not active"):
            lc.mark_intervention()


class TestInternalInvariants:
    """Defense-in-depth: even if internal state is corrupted by a bug
    outside this class's public API, the invariant assert catches it."""

    def test_consume_invariant_consumed_implies_terminal_set(self):
        """If a future bug poked _terminal_consumed=True without setting
        _terminal, consume_terminal_for_storage must fail loud rather
        than silently swallow the corruption."""
        lc = EpisodeLifecycle()
        lc.begin(0)
        lc._terminal_consumed = True  # simulate corruption
        lc._terminal = None
        with pytest.raises(AssertionError, match="invariant violated"):
            lc.consume_terminal_for_storage()


# ============================================================================
# Threading — the actual fingerprint of the original bug
# ============================================================================


class TestThreadSafety:
    """The original bug was a race: main thread sets a flag, inference
    thread reads it on every cycle until main thread resets it many ms
    later. These tests simulate that pattern and verify exactly-one-
    consume behavior under concurrent access."""

    def test_concurrent_consumers_only_one_wins(self):
        """N inference-thread reads racing against a single signal: at
        most one read returns the terminal kind, the others return None.

        This is the structural property that prevents the 17-aborts bug.
        Without the lock + ``_terminal_consumed`` flag, multiple threads
        could see ``_terminal != None`` before any of them set
        ``_terminal_consumed = True`` and all return the kind."""
        lc = EpisodeLifecycle()
        lc.begin(0)
        lc.signal_terminal(TerminalKind.ABORT)

        N_THREADS = 32
        results: list[TerminalKind | None] = [None] * N_THREADS
        barrier = threading.Barrier(N_THREADS)

        def consume(idx: int):
            barrier.wait()  # release all together for max contention
            results[idx] = lc.consume_terminal_for_storage()

        threads = [threading.Thread(target=consume, args=(i,)) for i in range(N_THREADS)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        wins = [r for r in results if r is not None]
        assert len(wins) == 1, (
            f"Expected exactly 1 consume to win, got {len(wins)}: {wins}. "
            f"This is the multi-terminal race we're guarding against."
        )
        assert wins[0] == TerminalKind.ABORT

    def test_signal_during_consume_storm(self):
        """The original bug pattern: signal fires once, the inference
        thread fires repeatedly during the cleanup window. Even if the
        signal lands after a few consume calls, the FIRST consume after
        the signal wins; all subsequent consumes in the same episode
        return None.

        Locks in: a 5-second post-signal cleanup window with ~20
        inference cycles writes EXACTLY 1 terminal, not 20.
        """
        lc = EpisodeLifecycle()
        lc.begin(0)

        consumed: list[TerminalKind | None] = []
        stop = threading.Event()

        def inference_thread():
            while not stop.is_set():
                consumed.append(lc.consume_terminal_for_storage())
                time.sleep(0.001)  # ~1ms inter-cycle

        t = threading.Thread(target=inference_thread)
        t.start()
        time.sleep(0.01)  # let inference fire ~10 times pre-signal

        # Operator presses ABORT (in main thread)
        lc.signal_terminal(TerminalKind.ABORT)

        time.sleep(0.05)  # let inference fire ~50 more times post-signal
        stop.set()
        t.join()

        wins = [r for r in consumed if r is not None]
        assert len(wins) == 1, (
            f"Across the simulated 50ms post-signal window, expected "
            f"exactly 1 inference call to consume the terminal, got "
            f"{len(wins)}: {wins}. This is the EXACT bug at ep124."
        )
        assert wins[0] == TerminalKind.ABORT

    def test_concurrent_main_signal_and_inference_consume(self):
        """Both threads writing concurrently: main thread signals,
        inference thread consumes. No data corruption."""
        lc = EpisodeLifecycle()
        lc.begin(0)

        consumed: list[TerminalKind | None] = []
        signal_done = threading.Event()
        stop = threading.Event()

        def inference():
            while not stop.is_set():
                consumed.append(lc.consume_terminal_for_storage())

        def main():
            time.sleep(0.005)
            lc.signal_terminal(TerminalKind.SUCCESS)
            signal_done.set()

        ti = threading.Thread(target=inference)
        tm = threading.Thread(target=main)
        ti.start()
        tm.start()
        signal_done.wait(timeout=1.0)
        time.sleep(0.02)
        stop.set()
        ti.join()
        tm.join()

        wins = [r for r in consumed if r is not None]
        assert len(wins) == 1
        assert wins[0] == TerminalKind.SUCCESS

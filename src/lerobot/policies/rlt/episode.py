"""Episode-level lifecycle state for RLT online RL.

Owns the per-episode mutable state that the main thread sets via the
operator's keyboard handlers (R / LEFT / DOWN / SPACE) and the
inference thread reads when deciding what reward / done flags to
attach to each transition.

Replaces the prior dict-based flags (``reward_triggered`` /
``abort_triggered`` / ``ignore_triggered``) which had a structural
race: the inference thread re-read them on every cycle while the
main thread didn't reset them until the *end* of post-episode
bookkeeping. Any inference cycle that fired in the window between
the operator's keypress and the flag reset wrote another transition
flagged ``done=True``. Real bug observed at ep124: 17 ``done=True``
transitions written instead of 1.

The fix here is structural, not just a flag-reset reorder: terminal
consumption is one-shot per episode, enforced by an internal
``_terminal_consumed`` flag inside this class. Even if the inference
thread fires N times during the post-signal window, only the first
``consume_terminal_for_storage()`` call returns the terminal kind;
all subsequent calls return ``None`` and the inference thread writes
``done=False, reward=0``.

See also ``tests/hvla/test_rlt_episode.py`` for the invariants this
class enforces and the threading test that locks in the race-free
behavior.
"""

from __future__ import annotations

import logging
import threading
from enum import Enum, auto


class TerminalKind(Enum):
    """A terminal that produces a buffer transition with ``done=True``.

    IGNORE is intentionally NOT a TerminalKind — ignored episodes are
    rolled back at the buffer level (truncate to ``buffer_size_at_start``)
    and produce no ``done=True`` transition. ``signal_ignore()`` is a
    separate API.
    """

    SUCCESS = auto()  # operator pressed R; reward = +1.0
    ABORT = auto()  # operator pressed LEFT; reward = config.abort_reward


logger = logging.getLogger(__name__)


class EpisodeLifecycle:
    """Per-episode operator-event tracking with one-shot terminal consumption.

    Thread-safe: ``signal_*`` calls come from the main thread (keyboard
    handlers); ``consume_terminal_for_storage`` is called from the
    inference thread; ``peek_*`` and ``end_episode`` are main-thread.
    A single per-instance lock serializes all access.

    Lifecycle:
        lifecycle = EpisodeLifecycle()
        lifecycle.begin(buffer_size=42)         # at episode start
        lifecycle.signal_terminal(SUCCESS)       # operator presses R
        ...
        kind = lifecycle.consume_terminal_for_storage()   # inference thread
        # → returns SUCCESS once, then None forever for this episode
        lifecycle.peek_terminal()                # main thread, no consume
        lifecycle.end_episode()                  # at episode end

    Asserts on contract violations:
        * ``signal_*`` outside an active episode (forgot ``begin()``)
        * ``end_episode()`` called twice
        * ``begin()`` called when previous episode signaled a terminal
          but it was never consumed and the episode wasn't ignored
          (indicates the inference thread never had a chance to write
          the terminal — e.g. the operator pressed the key faster than
          one inference cycle. Real edge case; the assert is loud so
          we can decide whether to soften it later if it fires in
          practice).
    """

    def __init__(self):
        self._lock = threading.Lock()
        self._terminal: TerminalKind | None = None
        self._terminal_consumed: bool = False
        self._ignore: bool = False
        self._had_intervention: bool = False
        self._buffer_size_at_start: int = 0
        self._active: bool = False

    # ---------------------------------------------------------------------
    # Lifecycle transitions (main thread)
    # ---------------------------------------------------------------------

    def begin(self, buffer_size: int) -> None:
        """Reset state for a new episode.

        Asserts:
            * Lifecycle is NOT currently active. begin() while a previous
              episode is still active would silently overwrite live state
              and lose track of an in-progress episode. Caller must
              ``end_episode()`` first.
            * ``buffer_size >= 0`` (defensive — negatives indicate a
              caller bug computing the buffer length).
            * The previous episode was properly closed: any signaled
              terminal must have been consumed (= written to the buffer)
              or the episode must have been ignored (= rolled back). A
              terminal that was signaled but neither consumed nor
              ignored means the inference thread missed it — the data
              point is lost. Loud assert lets us notice and decide.
        """
        with self._lock:
            assert not self._active, (
                "EpisodeLifecycle.begin: lifecycle is still active from "
                "the previous episode. Call end_episode() first. Calling "
                "begin() on an active lifecycle would silently overwrite "
                "in-progress state."
            )
            assert buffer_size >= 0, f"EpisodeLifecycle.begin: buffer_size must be >= 0, got {buffer_size}."
            if self._terminal is not None and not self._terminal_consumed and not self._ignore:
                raise AssertionError(
                    f"EpisodeLifecycle.begin: previous episode signaled "
                    f"terminal {self._terminal.name} but it was never "
                    f"consumed by the inference thread and the episode "
                    f"was not ignored. The signal was lost — no "
                    f"transition in the buffer reflects it. Did the "
                    f"operator press the key before any inference fired?"
                )
            self._terminal = None
            self._terminal_consumed = False
            self._ignore = False
            self._had_intervention = False
            self._buffer_size_at_start = int(buffer_size)
            self._active = True

    def end_episode(self) -> None:
        """Mark the episode inactive. Called after main-thread bookkeeping
        finishes. After this point ``signal_*`` and
        ``consume_terminal_for_storage`` will return without effect /
        return None."""
        with self._lock:
            assert self._active, (
                "EpisodeLifecycle.end_episode: called when not active. "
                "Either begin() was never called or end_episode() ran twice."
            )
            self._active = False
            # Do NOT clear _terminal / _ignore here — peek/is_ignored may
            # still be read from the bookkeeping path. begin() does the
            # full reset.

    # ---------------------------------------------------------------------
    # Operator signals (main thread)
    # ---------------------------------------------------------------------

    def signal_terminal(self, kind: TerminalKind) -> None:
        """Operator pressed R (SUCCESS) or LEFT (ABORT). First signal
        wins; subsequent signals never change the recorded outcome.

        Operator-driven (called from the OS keyboard listener thread),
        so timing is asynchronous: the operator may press keys before
        the first episode begins, between episodes during reset, or
        after end_episode. None of those are bugs — they're normal
        operator-key timing. The handler silently no-ops when the
        lifecycle is inactive, leaving the keypress with no effect
        but also no exception (which would otherwise be swallowed
        inside the listener thread, making hotkeys appear "broken").

        When active:
          * Repeat of the SAME kind is silently ignored (operator
            hammering / held key / sticky listener — all benign).
          * Conflicting kind (e.g. SUCCESS already signaled, then
            ABORT arrives) emits a WARNING. Diagnostic surface for
            slider-focus key duplication, accidental adjacent-key
            presses, future listener regressions. First-wins outcome
            stays unchanged; the warning makes the conflict visible
            in train.log so you can investigate.
        """
        with self._lock:
            if not self._active:
                logger.debug(
                    "EpisodeLifecycle.signal_terminal(%s): no active episode — keypress dropped silently.",
                    kind.name,
                )
                return
            if self._terminal is None:
                self._terminal = kind
                logger.debug("EpisodeLifecycle: terminal %s signaled", kind.name)
            elif self._terminal != kind:
                logger.warning(
                    "EpisodeLifecycle: %s signaled but %s was already "
                    "recorded — keeping first-wins (%s). This indicates "
                    "either operator key confusion or an upstream listener "
                    "bug (e.g. UI element eating an arrow key). Episode "
                    "outcome is unaffected.",
                    kind.name,
                    self._terminal.name,
                    self._terminal.name,
                )
            # else: same kind repeated — silent no-op (benign)

    def signal_ignore(self) -> None:
        """Operator pressed DOWN. Idempotent. Independent of
        signal_terminal — both can be set; the main-thread bookkeeping
        path lets ignore win (truncate, don't write terminal).

        Operator-driven (same lifecycle as signal_terminal): silently
        no-ops when no episode is active. Keypress timing is async and
        not a bug if it lands outside an episode."""
        with self._lock:
            if not self._active:
                logger.debug("EpisodeLifecycle.signal_ignore: no active episode — keypress dropped silently.")
                return
            self._ignore = True

    def mark_intervention(self) -> None:
        """Called the first time SPACE flips intervention on in this
        episode. Idempotent — once flagged, the episode is recorded as
        non-autonomous regardless of how many SPACE toggles follow.

        Asserts active: marking intervention outside an episode is a
        caller bug (the flag would persist into the next episode and
        misclassify it as non-autonomous).
        """
        with self._lock:
            assert self._active, (
                "EpisodeLifecycle.mark_intervention: called when not active. Was begin() called?"
            )
            self._had_intervention = True

    # ---------------------------------------------------------------------
    # Read-out — main thread (no consumption)
    # ---------------------------------------------------------------------

    def peek_terminal(self) -> TerminalKind | None:
        """Main-thread bookkeeping: which terminal kind was signaled?
        Does NOT consume. Returns None if no terminal was signaled."""
        with self._lock:
            return self._terminal

    def is_ignored(self) -> bool:
        with self._lock:
            return self._ignore

    @property
    def had_intervention(self) -> bool:
        with self._lock:
            return self._had_intervention

    @property
    def buffer_size_at_start(self) -> int:
        with self._lock:
            return self._buffer_size_at_start

    @property
    def active(self) -> bool:
        with self._lock:
            return self._active

    # ---------------------------------------------------------------------
    # Read-out — inference thread (one-shot consumption)
    # ---------------------------------------------------------------------

    def consume_terminal_for_storage(self) -> TerminalKind | None:
        """Inference-thread side of the contract. Returns the signaled
        terminal kind ONCE per episode. All subsequent calls during the
        same episode return None even if ``signal_terminal`` is still
        set on the underlying state.

        This is the structural fix for the multi-terminal bug. The
        inference thread can fire N times during the post-signal /
        pre-end_episode window — only the first call gets the kind,
        the rest get None and store ``done=False, reward=0``.

        Returns None if:
          * No terminal was signaled this episode
          * The terminal was already consumed
          * Episode is not active (between end_episode and the next begin)

        Internal invariant asserted: ``_terminal_consumed`` may only be
        True if ``_terminal`` is also set. If consumed=True and
        terminal=None, something corrupted the lifecycle state outside
        of this class's API — fail loud rather than silently masking it.
        """
        with self._lock:
            assert self._terminal is not None or not self._terminal_consumed, (
                "EpisodeLifecycle invariant violated: _terminal_consumed=True "
                "but _terminal=None. Internal state was corrupted outside "
                "the public API."
            )
            if not self._active:
                return None
            if self._terminal is None or self._terminal_consumed:
                return None
            self._terminal_consumed = True
            return self._terminal

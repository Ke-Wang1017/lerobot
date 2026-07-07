"""InterventionRecorder — owns the human-action → replay-buffer pipeline
during intervention episodes.

A small focused class so that (a) the main loop stays uncluttered, and
(b) we have a single call site at which to assert the ``z_rl``
precondition. A regression in the upstream inference-thread plumbing
fails loudly on the first intervention frame instead of silently
dropping data.

Behavior (Paper Alg 1 lines 9, 11, 12 during intervention):
    * Each frame: human action is appended to an internal C-frame buffer.
    * At the start of each C-frame window: z_rl + state are snapshotted.
    * When a window completes: write a transition to the replay buffer
      pairing the previous window with the current (s, a, r=0, s').
      action == ref == human_chunk (so BC penalty is zero on human data).

The assert in ``on_frame`` catches the specific failure mode where the
upstream inference thread stops exposing a fresh ``z_rl`` during
intervention. See InferenceThread for the producer side."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import torch
from torch import Tensor

logger = logging.getLogger(__name__)


class InterventionRecorder:
    """Records human-driven intervention frames into the RLT replay buffer.

    Lifecycle (caller's responsibility):
        1. ``reset()`` at intervention start.
        2. ``on_frame(...)`` once per frame while intervention is active.
        3. ``log_summary()`` at intervention end (optional, watchdog).
        4. ``reset()`` at intervention end and at episode end so the
           next intervention starts clean.

    Out-of-order calls or missing resets violate the recorder's
    preconditions and will trip its internal asserts.
    """

    def __init__(
        self,
        replay: Any,  # ReplayBuffer — typed Any to avoid circular import
        policy: Any,  # S1 policy (for normalization stats)
        device: torch.device,
        chunk_length: int,
        joint_names: list[str],
    ):
        assert chunk_length > 0, f"chunk_length must be positive, got {chunk_length}"
        assert len(joint_names) > 0, "joint_names must not be empty"
        self._replay = replay
        self._policy = policy
        self._device = device
        self._C = chunk_length
        self._joint_names = joint_names
        self._frame_count: int = 0
        self._chunks_stored: int = 0
        self._chunk_buf: list[Tensor] = []
        self._snap_z_rl: Tensor | None = None
        self._snap_state: Tensor | None = None
        self._prev_chunk: dict[str, Tensor] | None = None

    def reset(self) -> None:
        """Clear all intervention state.

        Postcondition:
            ``frames_observed == 0`` and ``chunks_stored == 0``.
            Internal buffers (``_chunk_buf``, ``_snap_*``, ``_prev_chunk``)
            are cleared, so the next ``on_frame`` starts a fresh window
            and cannot pair against stale state from the previous
            intervention.

        Call at intervention start, intervention end, and episode end."""
        self._frame_count = 0
        self._chunks_stored = 0
        self._chunk_buf = []
        self._snap_z_rl = None
        self._snap_state = None
        self._prev_chunk = None

    @property
    def frames_observed(self) -> int:
        return self._frame_count

    @property
    def chunks_stored(self) -> int:
        return self._chunks_stored

    def log_summary(self) -> None:
        """Log a watchdog comparing ``frames_observed`` to ``chunks_stored``.

        Precondition:
            Should be called at intervention end, before ``reset()``,
            so the counters reflect the just-finished intervention.

        Postcondition:
            One log line emitted (INFO on match, WARNING on mismatch).
            Internal state is unchanged — does not call ``reset()``.

        Expected count is ``max(0, frames // C - 1)``: every C frames
        produces a complete chunk, but the first chunk in an
        intervention seeds the prev-pair and writes nothing, so
        ``chunks_stored`` is one less than the number of complete
        C-frame windows. A mismatch means the human-action pipeline
        dropped data silently."""
        frames = self._frame_count
        stored = self._chunks_stored
        expected = max(0, frames // self._C - 1)
        if stored != expected:
            logger.warning(
                "RLT: intervention ended — %d frames, %d chunks stored, "
                "expected %d. Some intervention data was dropped!",
                frames,
                stored,
                expected,
            )
        else:
            logger.info(
                "RLT: intervention ended — %d frames, %d chunks stored (expected %d)",
                frames,
                stored,
                expected,
            )

    def on_frame(
        self,
        human_action_np: np.ndarray,
        current_z_rl: Tensor | None,
        current_obs: dict,
    ) -> None:
        """Record one intervention frame.

        Preconditions:
            * ``reset()`` has been called at the start of this
              intervention (so prev-pair state isn't stale).
            * ``current_z_rl is not None`` (asserted) — the upstream
              inference thread must be exposing a fresh z_rl.
            * ``current_obs`` contains every joint listed in
              ``joint_names``.

        Postconditions:
            * ``frames_observed`` is incremented by 1.
            * On every C-th call (frame ``C-1`` of each window): a
              completed chunk is flushed. The very first flush only
              seeds the prev-pair (no replay write); every subsequent
              flush writes one transition to the replay buffer and
              increments ``chunks_stored`` by 1.

        Args:
            human_action_np: [A] raw (unnormalized) human action.
            current_z_rl: latest z_rl from the inference thread.
            current_obs: robot observation dict for state extraction.
        """
        # Precondition: z_rl must be fresh. If the inference thread isn't
        # exposing one, chunks would be silently dropped — every human
        # demonstration frame lost.
        assert current_z_rl is not None, (
            "InterventionRecorder.on_frame got current_z_rl=None. The "
            "inference thread is not exposing a fresh z_rl while "
            "rlt_active=False. Check InferenceThread / how the main loop "
            "sources z_rl during intervention. Silently dropping chunks "
            "here loses all human demonstration data."
        )

        # Snapshot z_rl + state at frame 0 of each C-step window.
        if self._frame_count % self._C == 0:
            self._snap_z_rl = current_z_rl
            self._snap_state = self._extract_state(current_obs)
            self._chunk_buf = []

        # The previous flush must have emptied the buffer when it filled
        # to exactly C; if we're about to append into an already-full
        # buffer something has gone wrong upstream (off-by-one in flush
        # or someone re-entered on_frame with a stale state).
        assert len(self._chunk_buf) < self._C, (
            f"chunk_buf has {len(self._chunk_buf)} frames already "
            f"(>= chunk_length={self._C}); previous flush did not run"
        )

        self._chunk_buf.append(self._normalize_action(human_action_np))
        self._frame_count += 1

        # Flush completed C-frame chunk.
        if len(self._chunk_buf) == self._C:
            self._flush_chunk()

    def _extract_state(self, obs: dict) -> Tensor:
        state_np = np.array(
            [float(obs[j]) for j in self._joint_names],
            dtype=np.float32,
        )
        state_t = torch.from_numpy(state_np).to(self._device)
        if self._policy._state_mean is not None:
            state_t = (state_t - self._policy._state_mean.to(self._device)) / self._policy._state_std.to(
                self._device
            )
        return state_t

    def _normalize_action(self, action_np: np.ndarray) -> Tensor:
        a_t = torch.from_numpy(action_np).float()
        if self._policy._action_mean is not None:
            a_t = (a_t - self._policy._action_mean) / self._policy._action_std
        return a_t

    def _flush_chunk(self) -> None:
        # Two invariants the caller must have established:
        #   1. Buffer is filled to exactly C frames.
        #   2. The window-start snapshot was taken (set at frame % C == 0
        #      in on_frame). Without it, _prev_chunk would carry None
        #      tensors into a replay write — corrupting the buffer.
        assert len(self._chunk_buf) == self._C, (
            f"flush called with {len(self._chunk_buf)} frames buffered, expected exactly {self._C}"
        )
        assert self._snap_z_rl is not None and self._snap_state is not None, (
            "flush called without a window-start snapshot — the snapshot "
            "branch in on_frame did not run when it should have"
        )

        human_chunk = torch.stack(self._chunk_buf)  # [C, A]

        # Pair the previous completed chunk with the current one as
        # (s, a, r=0, s') — the intermediate transition. The first chunk
        # in an intervention seeds ``_prev_chunk`` and writes nothing;
        # subsequent flushes produce a transition.
        if self._prev_chunk is not None:
            self._replay.add(
                z_rl=self._prev_chunk["z_rl"],
                state=self._prev_chunk["state"],
                action_chunk=self._prev_chunk["action"],
                ref_chunk=self._prev_chunk["ref"],
                reward=0.0,  # intermediate, not terminal
                next_z_rl=self._snap_z_rl,
                next_state=self._snap_state,
                next_ref_chunk=human_chunk,
                done=False,
            )
            self._chunks_stored += 1

        # Current chunk becomes the new "prev". action == ref == human
        # (Paper Alg 1 line 11) so the BC penalty on replay is zero on
        # human data.
        self._prev_chunk = {
            "z_rl": self._snap_z_rl,
            "state": self._snap_state,
            "action": human_chunk,
            "ref": human_chunk,
        }
        self._chunk_buf = []

    def flush_terminal(
        self,
        reward: float,
        current_z_rl: Tensor | None,
        current_obs: dict,
    ) -> bool:
        """Write a final terminal transition to close out a rescued episode.

        The inference thread is paused during intervention, so it cannot
        record the terminal transition itself when the operator presses
        R or LEFT-ARROW while still teleop-driving. Without this method
        the +1 (or abort) reward never enters the replay buffer and the
        critic learns rescue successes look like timeouts.

        Anchors the terminal at the most recently completed C-frame
        human chunk: stores ``(s_prev, a_prev, r, s_current, done=True)``
        with action == ref == ``_prev_chunk["action"]`` (Paper Alg 1
        line 11 — BC penalty stays zero on human data). Bootstrap then
        propagates the terminal reward backward through the prior r=0
        intervention transitions and the actor's failed lead-in, exactly
        as TD3 expects.

        Preconditions:
            * ``current_z_rl is not None`` if a transition will be
              written (asserted; same failure mode as ``on_frame``).
            * ``current_obs`` contains every joint listed in
              ``joint_names``.

        Postconditions:
            * If ``_prev_chunk`` is set: one transition is appended to
              the replay buffer with ``done=True`` and the given
              reward, and this returns ``True``. ``_chunks_stored`` is
              not incremented — the caller is responsible for any
              accounting (so the regular-pairing watchdog formula in
              ``log_summary`` stays correct).
            * If no complete C-frame window has been flushed yet
              (intervention shorter than C frames): logs a WARNING and
              returns ``False`` without writing.

        Args:
            reward: terminal reward (+1 for SUCCESS, ``cfg.abort_reward``
                for ABORT).
            current_z_rl: latest z_rl from the inference thread.
            current_obs: robot observation dict for state extraction.

        Returns:
            True if a terminal transition was written, False if skipped.
        """
        if self._prev_chunk is None:
            logger.warning(
                "RLT: intervention ended with terminal r=%+.2f but no "
                "complete C-frame chunk yet (%d frames observed); "
                "reward signal lost.",
                reward,
                self._frame_count,
            )
            return False
        assert current_z_rl is not None, (
            "InterventionRecorder.flush_terminal got current_z_rl=None. "
            "The inference thread is not exposing a fresh z_rl while "
            "intervention is active. Without it, the terminal transition "
            "cannot be stored and ASSISTED success is silently lost."
        )
        next_state = self._extract_state(current_obs)
        self._replay.add(
            z_rl=self._prev_chunk["z_rl"],
            state=self._prev_chunk["state"],
            action_chunk=self._prev_chunk["action"],
            ref_chunk=self._prev_chunk["ref"],
            reward=float(reward),
            next_z_rl=current_z_rl,
            next_state=next_state,
            next_ref_chunk=self._prev_chunk["ref"],
            done=True,
        )
        return True

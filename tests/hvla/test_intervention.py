"""Tests for ``InterventionRecorder`` — the human-action → replay-buffer
pipeline used during intervention episodes.

Covers logic that's specific to the recorder itself, not the bug it
helps reproduce. Bug repro is the runtime assert + on-robot run.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from lerobot.policies.hvla.rlt.intervention import InterventionRecorder
from lerobot.policies.hvla.rlt.replay_buffer import ReplayBuffer

C = 5  # chunk_length used in tests
A = 14  # action_dim
STATE_DIM = 14  # joint dims
Z_DIM = 10  # rl_token_dim


@pytest.fixture
def device():
    return torch.device("cpu")


@pytest.fixture
def joint_names():
    """Match the joint count to STATE_DIM. Names just need to be unique
    keys in the obs dict."""
    return [f"j{i}.pos" for i in range(STATE_DIM)]


class _MockPolicy:
    """Minimal policy with normalization stats. None means skip
    normalization (mirrors the real policy when stats aren't loaded)."""

    def __init__(self, normalize=False):
        if normalize:
            # Distinct mean/std so we can detect normalization being applied.
            self._state_mean = torch.full((STATE_DIM,), 0.5)
            self._state_std = torch.full((STATE_DIM,), 2.0)
            self._action_mean = torch.full((A,), 1.0)
            self._action_std = torch.full((A,), 4.0)
        else:
            self._state_mean = None
            self._state_std = None
            self._action_mean = None
            self._action_std = None


@pytest.fixture
def replay(device):
    return ReplayBuffer(
        capacity=100,
        rl_token_dim=Z_DIM,
        state_dim=STATE_DIM,
        action_dim=A,
        chunk_length=C,
        device=device,
    )


def _make_recorder(replay, device, joint_names, normalize=False):
    return InterventionRecorder(
        replay=replay,
        policy=_MockPolicy(normalize=normalize),
        device=device,
        chunk_length=C,
        joint_names=joint_names,
    )


def _obs(joint_names, fill=0.0):
    """Build a minimal obs dict the recorder can read state from."""
    return dict.fromkeys(joint_names, fill)


def _human_action(val=0.0):
    return np.full(A, val, dtype=np.float32)


def _z_rl(val=0.0):
    return torch.full((Z_DIM,), val)


# ============================================================================
# Initial / reset state
# ============================================================================


class TestState:
    def test_init_counters_are_zero(self, replay, device, joint_names):
        rec = _make_recorder(replay, device, joint_names)
        assert rec.frames_observed == 0
        assert rec.chunks_stored == 0

    def test_reset_returns_counters_to_zero(self, replay, device, joint_names):
        rec = _make_recorder(replay, device, joint_names)
        # Drive enough frames to force one chunk store
        for i in range(2 * C):
            rec.on_frame(_human_action(i), _z_rl(), _obs(joint_names))
        assert rec.frames_observed == 2 * C
        assert rec.chunks_stored == 1
        rec.reset()
        assert rec.frames_observed == 0
        assert rec.chunks_stored == 0

    def test_reset_clears_prev_chunk_so_next_intervention_does_not_pair_across(
        self,
        replay,
        device,
        joint_names,
    ):
        """If reset doesn't clear ``_prev_chunk``, the first flush of the
        NEXT intervention would pair against stale state from the LAST
        intervention — wrong transition, contamination."""
        rec = _make_recorder(replay, device, joint_names)
        # First intervention: 2C frames → 1 chunk stored
        for i in range(2 * C):
            rec.on_frame(_human_action(i), _z_rl(0.1), _obs(joint_names, fill=0.1))
        before = len(replay)
        assert before == 1

        rec.reset()

        # Second intervention: 2C frames again → expect EXACTLY 1 more
        # write (first seeds prev, second flushes). If reset failed, the
        # very first flush would pair against the prior intervention's
        # state and we'd see 2.
        for i in range(2 * C):
            rec.on_frame(_human_action(i), _z_rl(0.2), _obs(joint_names, fill=0.2))
        after = len(replay)
        assert after - before == 1


# ============================================================================
# Chunk lifecycle (the core write logic)
# ============================================================================


class TestChunkLifecycle:
    """The recorder's contract: every C frames produces a "chunk." The
    FIRST chunk seeds ``_prev_chunk`` and writes nothing — it has no
    "previous" to pair with. Each subsequent chunk pairs with the prior
    and writes one transition. So with N frames where N = k·C, the
    expected store count is max(0, k-1)."""

    @pytest.mark.parametrize(
        "frames,expected_stored",
        [
            (0, 0),
            (C - 1, 0),  # partial first chunk
            (C, 0),  # first chunk completes, seeds prev
            (C + 1, 0),  # partial second chunk
            (2 * C, 1),  # second flush stores (prev,now)
            (2 * C + 1, 1),  # partial third chunk
            (3 * C, 2),  # third flush stores
            (5 * C, 4),  # k=5 → k-1 = 4
        ],
    )
    def test_chunks_stored_counts_match_expectation(
        self,
        replay,
        device,
        joint_names,
        frames,
        expected_stored,
    ):
        rec = _make_recorder(replay, device, joint_names)
        for i in range(frames):
            rec.on_frame(_human_action(i), _z_rl(), _obs(joint_names))
        assert rec.frames_observed == frames
        assert rec.chunks_stored == expected_stored
        assert len(replay) == expected_stored

    def test_first_chunk_is_seeded_not_stored(
        self,
        replay,
        device,
        joint_names,
    ):
        """Specifically verify the seed-not-store behavior so future
        refactors don't silently change it. If someone makes the first
        chunk ALSO write to replay, they'd get a transition with no
        meaningful "prev" — that's a bug we want to lock against."""
        rec = _make_recorder(replay, device, joint_names)
        for i in range(C):  # exactly one chunk's worth
            rec.on_frame(_human_action(i), _z_rl(), _obs(joint_names))
        assert rec.chunks_stored == 0
        assert len(replay) == 0

    def test_flush_at_exact_chunk_boundary(
        self,
        replay,
        device,
        joint_names,
    ):
        """The flush must trigger at len(buf)==C, not at C-1 or C+1.
        Off-by-one here would mis-time the snapshot vs. the action."""
        rec = _make_recorder(replay, device, joint_names)
        # First C-1 frames: no flush yet
        for _i in range(C - 1):
            rec.on_frame(_human_action(), _z_rl(), _obs(joint_names))
        assert rec.chunks_stored == 0
        # Cth frame: chunk completes (seeds prev, no write yet)
        rec.on_frame(_human_action(), _z_rl(), _obs(joint_names))
        assert rec.chunks_stored == 0
        # 2Cth frame: SECOND chunk completes, this one writes
        for _ in range(C):
            rec.on_frame(_human_action(), _z_rl(), _obs(joint_names))
        assert rec.chunks_stored == 1


# ============================================================================
# Snapshot semantics (the part that matches Paper Alg 1)
# ============================================================================


class TestSnapshotSemantics:
    def test_snapshot_is_taken_at_frame_zero_of_window(
        self,
        replay,
        device,
        joint_names,
    ):
        """z_rl + state for a chunk are sampled at the FIRST frame of its
        C-window, not the last. This matches the paper: the transition
        records (state_at_window_start, action_chunk, reward, next_state)."""
        rec = _make_recorder(replay, device, joint_names)
        # Window 0: frames 0..C-1, z_rl=0.1, state-fill=0.1
        for _ in range(C):
            rec.on_frame(_human_action(), _z_rl(0.1), _obs(joint_names, fill=0.1))
        # Window 1: frames C..2C-1, z_rl=0.2, state-fill=0.2
        # → flush at end of window 1 stores (window-0-state, window-0-action,
        #   r=0, window-1-state). Window 0's z_rl was 0.1, window 1's is 0.2.
        for _ in range(C):
            rec.on_frame(_human_action(), _z_rl(0.2), _obs(joint_names, fill=0.2))

        assert len(replay) == 1
        # Sample directly from the buffer's CPU-side storage to inspect.
        z_rl_stored = replay._z_rl[0]
        next_z_rl_stored = replay._next_z_rl[0]
        state_stored = replay._state[0]
        next_state_stored = replay._next_state[0]
        assert torch.allclose(z_rl_stored, torch.full_like(z_rl_stored, 0.1))
        assert torch.allclose(next_z_rl_stored, torch.full_like(next_z_rl_stored, 0.2))
        assert torch.allclose(state_stored, torch.full_like(state_stored, 0.1))
        assert torch.allclose(next_state_stored, torch.full_like(next_state_stored, 0.2))

    def test_action_equals_ref_for_human_data(
        self,
        replay,
        device,
        joint_names,
    ):
        """Per Paper Alg 1 line 11: ã ← a_human. The replay must record
        action == ref so the BC penalty term β·||a-ref||² is zero on
        human-demonstration transitions."""
        rec = _make_recorder(replay, device, joint_names)
        for i in range(2 * C):
            rec.on_frame(_human_action(i), _z_rl(), _obs(joint_names))
        action_stored = replay._action[0]
        ref_stored = replay._ref[0]
        assert torch.equal(action_stored, ref_stored)

    def test_action_chunk_preserves_frame_order(
        self,
        replay,
        device,
        joint_names,
    ):
        """Frame N's action must end up at position N within its C-window
        in the stored chunk. Out-of-order would mean the actor would be
        trained against a permuted action sequence."""
        rec = _make_recorder(replay, device, joint_names)
        # Window 0: frame i has action filled with value i (0..C-1)
        for i in range(C):
            rec.on_frame(_human_action(i), _z_rl(), _obs(joint_names))
        # Window 1: also fills its frames with i (C..2C-1)
        for i in range(C, 2 * C):
            rec.on_frame(_human_action(i), _z_rl(), _obs(joint_names))
        # Stored transition is (window-0, window-1)
        # Action chunk is flattened [C*A]; first A entries = frame 0, next A = frame 1, etc.
        action_stored = replay._action[0].reshape(C, A)
        for frame_idx in range(C):
            expected = torch.full((A,), float(frame_idx))
            assert torch.allclose(action_stored[frame_idx], expected), (
                f"frame {frame_idx} ended up out of order"
            )

    def test_reward_and_done_are_intermediate(
        self,
        replay,
        device,
        joint_names,
    ):
        """Intervention chunks are intermediate transitions: r=0, done=False.
        Terminal reward only fires when the operator presses 'r'."""
        rec = _make_recorder(replay, device, joint_names)
        for _ in range(2 * C):
            rec.on_frame(_human_action(), _z_rl(), _obs(joint_names))
        assert replay._reward[0, 0].item() == 0.0
        assert replay._done[0, 0].item() == 0.0


# ============================================================================
# Normalization (using the policy's stats)
# ============================================================================


class TestNormalization:
    def test_state_normalization_applied(
        self,
        replay,
        device,
        joint_names,
    ):
        """When the policy has state_mean/std, raw state values get
        normalized before storage. With mean=0.5, std=2.0 and raw=2.5
        we expect (2.5-0.5)/2.0 = 1.0."""
        rec = _make_recorder(replay, device, joint_names, normalize=True)
        for _ in range(2 * C):
            rec.on_frame(_human_action(), _z_rl(), _obs(joint_names, fill=2.5))
        assert torch.allclose(replay._state[0], torch.full((STATE_DIM,), 1.0))

    def test_action_normalization_applied(
        self,
        replay,
        device,
        joint_names,
    ):
        """Same for actions: mean=1.0, std=4.0, raw=5.0 → (5-1)/4 = 1.0."""
        rec = _make_recorder(replay, device, joint_names, normalize=True)
        for _ in range(2 * C):
            rec.on_frame(_human_action(5.0), _z_rl(), _obs(joint_names))
        action_stored = replay._action[0].reshape(C, A)
        assert torch.allclose(action_stored, torch.full((C, A), 1.0))

    def test_no_normalization_when_stats_are_none(
        self,
        replay,
        device,
        joint_names,
    ):
        """The default policy has no stats; raw values pass through."""
        rec = _make_recorder(replay, device, joint_names, normalize=False)
        for _ in range(2 * C):
            rec.on_frame(_human_action(7.0), _z_rl(0.3), _obs(joint_names, fill=4.0))
        # State stored is raw 4.0 (no shift/scale)
        assert torch.allclose(replay._state[0], torch.full((STATE_DIM,), 4.0))
        # Action stored is raw 7.0
        action_stored = replay._action[0].reshape(C, A)
        assert torch.allclose(action_stored, torch.full((C, A), 7.0))


# ============================================================================
# log_summary (watchdog)
# ============================================================================


class TestLogSummary:
    def test_summary_logs_info_when_match(
        self,
        replay,
        device,
        joint_names,
        caplog,
    ):
        rec = _make_recorder(replay, device, joint_names)
        for _ in range(3 * C):  # 3 windows → expected = 2 stored
            rec.on_frame(_human_action(), _z_rl(), _obs(joint_names))
        assert rec.chunks_stored == 2
        with caplog.at_level("INFO", logger="lerobot.policies.hvla.rlt.intervention"):
            rec.log_summary()
        assert any("(expected 2)" in r.message for r in caplog.records), (
            "happy-path log should restate the expected count for confirmation"
        )

    def test_summary_logs_warning_when_drops_detected(
        self,
        replay,
        device,
        joint_names,
        caplog,
        monkeypatch,
    ):
        """Force the recorder into a state where chunks_stored < expected
        (simulating the bug's silent-drop scenario) and verify WARN fires."""
        rec = _make_recorder(replay, device, joint_names)
        # Drive frames forward without writes by directly poking the count
        # — simulates "frames seen but storage path was broken".
        rec._frame_count = 3 * C  # expected = 2
        rec._chunks_stored = 0  # but nothing was stored
        with caplog.at_level("WARNING", logger="lerobot.policies.hvla.rlt.intervention"):
            rec.log_summary()
        assert any("Some intervention data was dropped" in r.message for r in caplog.records), (
            "drop-detection warning must fire on mismatch"
        )


# ============================================================================
# flush_terminal (intervention-success / intervention-abort path)
# ============================================================================


class TestFlushTerminal:
    """Pre-fix, episodes where the operator pressed 'r' or LEFT-ARROW
    during intervention dropped the terminal reward — the inference
    thread was paused so no done=True transition was written. The
    critic only ever saw rescue episodes as r=0 timeouts.
    flush_terminal closes that gap by writing the missing terminal
    through the recorder."""

    def test_writes_terminal_with_positive_reward(
        self,
        replay,
        device,
        joint_names,
    ):
        rec = _make_recorder(replay, device, joint_names)
        # Drive 2 complete windows so _prev_chunk is populated
        for _ in range(2 * C):
            rec.on_frame(_human_action(), _z_rl(0.4), _obs(joint_names, fill=0.4))
        before = len(replay)

        wrote = rec.flush_terminal(
            reward=1.0,
            current_z_rl=_z_rl(0.7),
            current_obs=_obs(joint_names, fill=0.7),
        )
        assert wrote is True
        assert len(replay) == before + 1
        # The terminal transition must be at the most recent slot:
        last = before  # buffer wraps but cap=100 won't here
        assert replay._reward[last, 0].item() == 1.0
        assert replay._done[last, 0].item() == 1.0
        # next_state should reflect the post-rescue snapshot, not the prev.
        assert torch.allclose(replay._next_state[last], torch.full((STATE_DIM,), 0.7))
        assert torch.allclose(replay._next_z_rl[last], torch.full((Z_DIM,), 0.7))

    def test_writes_terminal_with_negative_reward_for_abort(
        self,
        replay,
        device,
        joint_names,
    ):
        rec = _make_recorder(replay, device, joint_names)
        for _ in range(2 * C):
            rec.on_frame(_human_action(), _z_rl(), _obs(joint_names))
        wrote = rec.flush_terminal(
            reward=-1.0,
            current_z_rl=_z_rl(),
            current_obs=_obs(joint_names),
        )
        assert wrote is True
        last = len(replay) - 1
        assert replay._reward[last, 0].item() == -1.0
        assert replay._done[last, 0].item() == 1.0

    def test_action_equals_ref_on_terminal_chunk(
        self,
        replay,
        device,
        joint_names,
    ):
        """Same Paper Alg 1 line 11 contract as intermediate intervention
        transitions: BC penalty must be zero on the terminal too."""
        rec = _make_recorder(replay, device, joint_names)
        for _ in range(2 * C):
            rec.on_frame(_human_action(2.0), _z_rl(), _obs(joint_names))
        rec.flush_terminal(reward=1.0, current_z_rl=_z_rl(), current_obs=_obs(joint_names))
        last = len(replay) - 1
        assert torch.equal(replay._action[last], replay._ref[last])

    def test_no_op_when_no_complete_chunk_yet(
        self,
        replay,
        device,
        joint_names,
        caplog,
    ):
        """Intervention shorter than C frames: no _prev_chunk to anchor
        the terminal to. Returns False and logs a WARNING — better than
        writing a corrupt transition with stale state."""
        rec = _make_recorder(replay, device, joint_names)
        for _ in range(C - 1):  # less than one full window
            rec.on_frame(_human_action(), _z_rl(), _obs(joint_names))
        before = len(replay)
        with caplog.at_level("WARNING", logger="lerobot.policies.hvla.rlt.intervention"):
            wrote = rec.flush_terminal(
                reward=1.0,
                current_z_rl=_z_rl(),
                current_obs=_obs(joint_names),
            )
        assert wrote is False
        assert len(replay) == before
        assert any("no complete C-frame chunk yet" in r.message for r in caplog.records)

    def test_does_not_double_increment_chunks_stored(
        self,
        replay,
        device,
        joint_names,
    ):
        """The terminal flush is accounted for separately by the caller's
        ``total_transitions += 1`` — it must NOT bump chunks_stored, or
        log_summary's expected formula (frames//C - 1) would be wrong."""
        rec = _make_recorder(replay, device, joint_names)
        for _ in range(2 * C):
            rec.on_frame(_human_action(), _z_rl(), _obs(joint_names))
        before_stored = rec.chunks_stored
        rec.flush_terminal(reward=1.0, current_z_rl=_z_rl(), current_obs=_obs(joint_names))
        assert rec.chunks_stored == before_stored

    def test_terminal_anchors_at_last_completed_window(
        self,
        replay,
        device,
        joint_names,
    ):
        """The (s, a) of the terminal transition is taken from the most
        recently FLUSHED window — i.e. the snapshot at frame 0 of that
        window and the action chunk recorded over those C frames. The
        next-state is the moment of operator R-press."""
        rec = _make_recorder(replay, device, joint_names)
        # Window 0 (the one that becomes _prev_chunk): z_rl=0.1 state=0.1 action=2
        for _ in range(C):
            rec.on_frame(_human_action(2.0), _z_rl(0.1), _obs(joint_names, fill=0.1))
        # Window 1 in progress (incomplete): z_rl=0.2 state=0.2 action=3
        for _ in range(C // 2):
            rec.on_frame(_human_action(3.0), _z_rl(0.2), _obs(joint_names, fill=0.2))
        rec.flush_terminal(reward=1.0, current_z_rl=_z_rl(0.9), current_obs=_obs(joint_names, fill=0.9))
        last = len(replay) - 1
        # "from" side = window 0 (snapshot + its action)
        assert torch.allclose(replay._z_rl[last], torch.full((Z_DIM,), 0.1))
        assert torch.allclose(replay._state[last], torch.full((STATE_DIM,), 0.1))
        action = replay._action[last].reshape(C, A)
        assert torch.allclose(action, torch.full((C, A), 2.0))
        # "to" side = the operator-R moment
        assert torch.allclose(replay._next_z_rl[last], torch.full((Z_DIM,), 0.9))
        assert torch.allclose(replay._next_state[last], torch.full((STATE_DIM,), 0.9))

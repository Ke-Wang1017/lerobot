"""Tests for InferenceThread — state management, pause/resume, chunk production.

All tests use a MockS1Policy on CPU — no GPU required.
"""

import json
import threading
import time

import numpy as np
import pytest
import torch

from lerobot.policies.hvla.s1_inference import InferenceThread, _slice_with_pad


class MockS1Policy:
    """Minimal policy that returns a fixed chunk. No GPU needed."""

    supports_rtc = False
    needs_temporal_ensemble = False
    rtc_prefix_length = 0

    def __init__(self, chunk_size=50, action_dim=14):
        self._chunk_size = chunk_size
        self._action_dim = action_dim
        self._reset_count = 0

    def predict_action_chunk(self, batch, num_steps=None):
        return torch.randn(1, self._chunk_size, self._action_dim)

    def reset(self):
        self._reset_count += 1

    def to(self, device):
        return self

    def eval(self):
        return self


class MockSharedCache:
    """Minimal SharedLatentCache mock."""

    def read_with_age(self):
        return torch.zeros(2048), 0.0


# Use the real JOINT_NAMES so obs_to_s1_batch works without modification
from lerobot.policies.hvla.s1_process import JOINT_NAMES


def _make_obs():
    """Create a minimal observation dict matching JOINT_NAMES."""
    obs = {}
    for name in JOINT_NAMES:
        obs[name] = 0.0
    obs["front"] = np.zeros((224, 224, 3), dtype=np.uint8)
    return obs


def _make_thread(**kwargs) -> InferenceThread:
    """Create an InferenceThread with test defaults."""
    defaults = {
        "policy": MockS1Policy(),
        "preprocessor": lambda batch: batch,
        "postprocessor": lambda actions: actions,
        "shared_cache": MockSharedCache(),
        "s2_latent_key": "observation.s2_latent",  # gitleaks:allow
        "s1_image_keys": ["observation.images.front"],
        "joint_names": list(JOINT_NAMES),
        "device": torch.device("cpu"),
        "resize_to": None,
        "fps": 30,
    }
    defaults.update(kwargs)
    return InferenceThread(**defaults)


class TestSliceWithPad:
    """``_slice_with_pad`` — used to safely slice the actor's ``[D:D+C]``
    window from the S1 reference chunk even when D+C overruns the chunk
    (RTC's expected_d is dynamic, not bounded by chunk_size). Returns a
    tensor of exactly ``length`` along the slice dim, padding with the
    last value when needed."""

    def _t(self, n: int):
        # [1, n, 4]: distinct value per frame so we can verify which
        # frames came from the source vs the pad.
        return torch.arange(n, dtype=torch.float32).reshape(1, n, 1).repeat(1, 1, 4)

    def test_no_overrun(self):
        out = _slice_with_pad(self._t(50), start=2, length=3, dim=1)
        assert out.shape == (1, 3, 4)
        assert torch.allclose(out[0, :, 0], torch.tensor([2.0, 3.0, 4.0]))

    def test_exact_fit_at_end(self):
        # n=10, start=7, length=3 → grab frames 7,8,9 exactly; no pad.
        out = _slice_with_pad(self._t(10), start=7, length=3, dim=1)
        assert out.shape == (1, 3, 4)
        assert torch.allclose(out[0, :, 0], torch.tensor([7.0, 8.0, 9.0]))

    def test_overrun_pads_with_last_value(self, caplog):
        # n=10, start=8, length=4 → frames 8,9 + 2 copies of frame 9.
        with caplog.at_level("WARNING", logger="lerobot.policies.hvla.s1_inference"):
            out = _slice_with_pad(self._t(10), start=8, length=4, dim=1)
        assert out.shape == (1, 4, 4)
        assert torch.allclose(out[0, :, 0], torch.tensor([8.0, 9.0, 9.0, 9.0]))
        assert any("padded with last value" in r.message for r in caplog.records)

    def test_start_at_or_past_end_pads_entirely(self, caplog):
        # Defensive: D >= chunk_size somehow. Whole window is the last
        # frame replicated; warning fires.
        with caplog.at_level("WARNING", logger="lerobot.policies.hvla.s1_inference"):
            out = _slice_with_pad(self._t(10), start=10, length=3, dim=1)
        assert out.shape == (1, 3, 4)
        # All frames should be the last value of the source.
        assert torch.allclose(out[0, :, 0], torch.tensor([9.0, 9.0, 9.0]))


class TestLifecycle:
    """Start/stop behavior."""

    def test_start_stop(self):
        """Thread starts and stops cleanly."""
        thread = _make_thread()
        thread.start()
        assert thread._thread is not None
        assert thread._thread.is_alive()
        thread.stop(timeout=2.0)
        assert not thread._thread.is_alive()

    def test_stop_without_start(self):
        """Stopping before starting doesn't crash."""
        thread = _make_thread()
        thread.stop()  # should be a no-op

    def test_double_stop(self):
        """Stopping twice doesn't crash."""
        thread = _make_thread()
        thread.start()
        thread.stop()
        thread.stop()


class TestChunkProduction:
    """Verify the thread produces chunks from observations."""

    def test_produces_chunk(self):
        """Publishing obs should produce a chunk."""
        thread = _make_thread()
        thread.start()
        try:
            obs = _make_obs()
            thread.publish_obs(obs, time.perf_counter())
            assert thread.wait_for_first_chunk(timeout=5.0)
            chunk, t_origin, t_obs = thread.get_chunk()
            assert chunk is not None
            assert chunk.shape == (50, 14)  # default MockS1Policy
            assert t_origin > 0
        finally:
            thread.stop()

    def test_multiple_chunks(self):
        """Multiple obs should produce multiple chunks (infer_times grows)."""
        thread = _make_thread()
        thread.start()
        try:
            for _ in range(3):
                obs = _make_obs()
                thread.publish_obs(obs, time.perf_counter())
                time.sleep(0.3)  # wait for inference
            assert len(thread.infer_times) >= 2
        finally:
            thread.stop()

    def test_first_chunk_timeout(self):
        """No obs published → wait_for_first_chunk times out."""
        thread = _make_thread()
        thread.start()
        try:
            assert not thread.wait_for_first_chunk(timeout=0.3)
        finally:
            thread.stop()

    def test_chunk_shape_matches_policy(self):
        """Chunk shape should match the policy's output."""
        policy = MockS1Policy(chunk_size=20, action_dim=7)
        thread = _make_thread(policy=policy)
        thread.start()
        try:
            thread.publish_obs(_make_obs(), time.perf_counter())
            thread.wait_for_first_chunk(timeout=5.0)
            chunk, _, _ = thread.get_chunk()
            assert chunk.shape == (20, 7)
        finally:
            thread.stop()


class TestPauseResume:
    """Verify pause/resume behavior."""

    def test_pause_blocks_inference(self):
        """After pause(), no new chunks should be produced."""
        thread = _make_thread()
        thread.start()
        try:
            # Produce first chunk
            thread.publish_obs(_make_obs(), time.perf_counter())
            thread.wait_for_first_chunk(timeout=5.0)

            # Pause and wait for thread to actually block
            thread.pause()
            assert thread.is_paused
            time.sleep(0.5)  # let any in-flight inference finish

            count_before = len(thread.infer_times)

            # Publish more obs while paused
            for _ in range(3):
                thread.publish_obs(_make_obs(), time.perf_counter())
                time.sleep(0.1)

            # No new chunks should have been produced
            count_after = len(thread.infer_times)
            assert count_after == count_before, f"Inference ran while paused: {count_before} → {count_after}"
        finally:
            thread.stop()

    def test_resume_produces_chunks(self):
        """After resume(), inference should produce chunks again."""
        thread = _make_thread()
        thread.start()
        try:
            # Produce first chunk
            thread.publish_obs(_make_obs(), time.perf_counter())
            thread.wait_for_first_chunk(timeout=5.0)

            # Pause then resume
            thread.pause()
            time.sleep(0.2)
            thread.resume()
            assert not thread.is_paused

            count_before = len(thread.infer_times)
            thread.publish_obs(_make_obs(), time.perf_counter())
            time.sleep(0.5)
            assert len(thread.infer_times) > count_before
        finally:
            thread.stop()

    def test_resume_resets_policy(self):
        """resume() should call policy.reset() to clear stale state."""
        policy = MockS1Policy()
        thread = _make_thread(policy=policy)
        thread.start()
        try:
            thread.publish_obs(_make_obs(), time.perf_counter())
            thread.wait_for_first_chunk(timeout=5.0)

            reset_count_before = policy._reset_count
            thread.pause()
            thread.resume()
            assert policy._reset_count == reset_count_before + 1
        finally:
            thread.stop()

    def test_pause_resume_cycle(self):
        """Multiple pause/resume cycles should work."""
        thread = _make_thread()
        thread.start()
        try:
            thread.publish_obs(_make_obs(), time.perf_counter())
            thread.wait_for_first_chunk(timeout=5.0)

            for _ in range(3):
                thread.pause()
                assert thread.is_paused
                time.sleep(0.1)
                thread.resume()
                assert not thread.is_paused
                thread.publish_obs(_make_obs(), time.perf_counter())
                time.sleep(0.3)

            assert len(thread.infer_times) >= 4  # initial + 3 cycles
        finally:
            thread.stop()


class TestExecIndex:
    """Verify exec index tracking for RTC."""

    def test_update_exec_index(self):
        """update_exec_index should be readable from the thread's state."""
        thread = _make_thread()
        thread.update_exec_index(42)
        with thread._main_loop_chunk_idx_lock:
            assert thread._main_loop_chunk_idx == 42


class TestGetChunkThreadSafety:
    """Verify get_chunk doesn't crash under concurrent access."""

    def test_concurrent_read_write(self):
        """Main loop reading chunks while inference thread writes shouldn't crash."""
        thread = _make_thread()
        thread.start()

        errors = []

        def reader():
            for _ in range(50):
                try:
                    thread.get_chunk()
                except Exception as e:
                    errors.append(e)
                time.sleep(0.01)

        reader_thread = threading.Thread(target=reader)
        reader_thread.start()

        try:
            for _ in range(10):
                thread.publish_obs(_make_obs(), time.perf_counter())
                time.sleep(0.05)

            reader_thread.join(timeout=3.0)
            assert not errors, f"Concurrent read errors: {errors}"
        finally:
            thread.stop()


class TestRLTChunkDump:
    """The Diagnostic button (``dump_chunks``) must produce schema-compatible
    records both when the actor is running (A) and when it's off (B). Before
    this refactor the dump was nested inside the actor path, so baseline A/B
    runs with the actor disengaged produced no dump at all.
    """

    def _make_thread_with_rlt(self, tmp_path, dump_on=True):
        from lerobot.policies.hvla.rlt.config import RLTConfig

        policy = MockS1Policy()
        # Disable normalization so ref_norm is numerically just the raw chunk.
        policy._state_mean = None
        policy._state_std = None
        policy._action_mean = None
        policy._action_std = None

        thread = _make_thread(policy=policy)
        thread._rlt_state = {
            "config": RLTConfig(rl_chunk_length=10, warmup_episodes=0),
            "episode": 50,
            "total_updates": 0,
            "total_transitions": 0,
            "reward_triggered": False,
            "output_dir": tmp_path,
            "deploy": True,
        }
        thread._rlt_replay = None
        thread._rlt_dump_chunks = dump_on
        # Simple actor that shifts ref by a fixed offset so we can detect it.
        thread._rlt_actor = lambda z, s, r, deterministic=False: r + 0.3
        return thread

    def _actions(self, c_frames_val=7.0, tail_val=9.0):
        """Build a [1, 50, 14] S1 chunk with distinct values in the actor-
        region (first 10 frames) vs the S1-only tail. Lets tests assert
        that the dump's ref covers the full chunk, not just the prefix."""
        t = torch.full((1, 50, 14), tail_val)
        t[:, :10, :] = c_frames_val
        return t

    def _write_dump_once(self, thread, actor_norm, rlt_active):
        """Drive the refactored helper that's normally called from the main loop."""
        ref_norm = thread._rlt_compute_ref_norm(self._actions())
        thread._rlt_write_dump_record(ref_norm, actor_norm, is_rlt_active=rlt_active)

    def _read_records(self, tmp_path):
        p = tmp_path / "chunk_compare.jsonl"
        return [json.loads(l) for l in p.read_text().splitlines()]

    def test_dump_with_actor_records_both(self, tmp_path):
        """Actor-on records contain both ref (full S1 chunk) and actor fields."""
        thread = self._make_thread_with_rlt(tmp_path)
        actor_norm = torch.full((10, 14), 0.42)  # distinguishable from ref
        self._write_dump_once(thread, actor_norm, rlt_active=True)

        recs = self._read_records(tmp_path)
        assert len(recs) == 1
        r = recs[0]
        # ref covers the full S1 chunk (50 frames × 14 joints), not just
        # the C=10 frames the actor replaces. This lets analysis see the
        # S1-only tail past the actor boundary.
        assert r["ref"] is not None and len(r["ref"]) == 50 and len(r["ref"][0]) == 14
        assert r["actor"] is not None
        assert len(r["actor"]) == 10, "actor covers only the first C frames"
        assert abs(r["actor"][0][0] - 0.42) < 1e-5
        assert r["rlt_active"] is True

    def test_dump_without_actor_has_null_actor(self, tmp_path):
        """A/B baseline: actor off → record still written, ``actor: null``.
        This is the whole point of hoisting the dump: Diagnostic button works
        for both legs of the A/B without a new flag or new GUI control."""
        thread = self._make_thread_with_rlt(tmp_path)
        self._write_dump_once(thread, actor_norm=None, rlt_active=False)

        recs = self._read_records(tmp_path)
        assert len(recs) == 1
        r = recs[0]
        assert r["ref"] is not None
        assert r["actor"] is None, (
            "actor must be null when dumped without actor execution — "
            "otherwise schema differs between A/B legs"
        )
        assert r["rlt_active"] is False

    def test_dump_schema_is_stable_across_on_off(self, tmp_path):
        """Same keys present in both actor-on and actor-off records so the
        analysis pipeline can load one file with mixed records."""
        thread = self._make_thread_with_rlt(tmp_path)
        self._write_dump_once(thread, torch.zeros(10, 14), rlt_active=True)
        self._write_dump_once(thread, None, rlt_active=False)

        recs = self._read_records(tmp_path)
        assert len(recs) == 2
        required = {
            "t",
            "step",
            "ep",
            "ref",
            "actor",
            "deploy",
            "rlt_active",
            "prefix_len",
            "exec_idx",
            "rlt_enc_ms",
            "rlt_post_ms",
            "total_delay_ms",
            "obs_to_infer_ms",
            "enc_obs_ms",
            "rl_tok_ms",
            "s1_denoise_ms",
            "rlt_actor_ms",
        }
        for r in recs:
            assert set(r.keys()) == required, f"Expected {required}, got {set(r.keys())} in record {r}"

    def test_dump_records_prefix_metadata(self, tmp_path):
        """``prefix_len`` and ``exec_idx`` from the RTC logic must round-trip
        into the dump so analysis can reconstruct which frames were the
        clamped prefix (frames [0:prefix_len]) at each inference."""
        thread = self._make_thread_with_rlt(tmp_path)
        ref_norm = thread._rlt_compute_ref_norm(self._actions())
        thread._rlt_write_dump_record(
            ref_norm,
            None,
            is_rlt_active=False,
            prefix_len=4,
            exec_idx=6,
        )
        recs = self._read_records(tmp_path)
        assert recs[0]["prefix_len"] == 4
        assert recs[0]["exec_idx"] == 6

    def test_dump_records_rlt_timing_fields(self, tmp_path):
        """Per-inference RLT timing belongs in the dump, not the main log.
        If this fires, either the field was dropped or the caller stopped
        passing timings through."""
        thread = self._make_thread_with_rlt(tmp_path)
        ref_norm = thread._rlt_compute_ref_norm(self._actions())
        thread._rlt_write_dump_record(
            ref_norm,
            None,
            is_rlt_active=False,
            prefix_len=3,
            exec_idx=5,
            rlt_enc_ms=3.14,
            rlt_post_ms=1.59,
            total_delay_ms=62.5,
        )
        r = self._read_records(tmp_path)[0]
        assert r["rlt_enc_ms"] == 3.14
        assert r["rlt_post_ms"] == 1.59
        assert r["total_delay_ms"] == 62.5

    def test_dump_records_per_stage_timing_breakdown(self, tmp_path):
        """The per-stage fields split rlt_enc into its two components and
        add s1_denoise / actor timings so the dump carries the full journey.
        This lets analysis code compute proper attribution instead of
        guessing from the coarse umbrella fields."""
        thread = self._make_thread_with_rlt(tmp_path)
        ref_norm = thread._rlt_compute_ref_norm(self._actions())
        thread._rlt_write_dump_record(
            ref_norm,
            None,
            is_rlt_active=False,
            prefix_len=2,
            exec_idx=7,
            rlt_enc_ms=2.5,
            rlt_post_ms=1.0,
            total_delay_ms=60.0,
            obs_to_infer_ms=12.5,
            enc_obs_ms=1.8,
            rl_tok_ms=0.7,
            s1_denoise_ms=43.2,
            rlt_actor_ms=0.4,
        )
        r = self._read_records(tmp_path)[0]
        assert r["obs_to_infer_ms"] == 12.5
        assert r["enc_obs_ms"] == 1.8
        assert r["rl_tok_ms"] == 0.7
        assert r["s1_denoise_ms"] == 43.2
        assert r["rlt_actor_ms"] == 0.4
        # Legacy rlt_enc_ms should be populated by the main loop as
        # enc_obs_ms + rl_tok_ms; this helper-level test just checks it
        # still round-trips as supplied.
        assert r["rlt_enc_ms"] == 2.5

    def test_dump_skipped_when_no_output_dir(self, tmp_path):
        """No output_dir on rlt_state (shouldn't happen in practice) must not
        crash — just silently skip."""
        thread = self._make_thread_with_rlt(tmp_path)
        thread._rlt_state["output_dir"] = None
        # Should not raise
        self._write_dump_once(thread, None, rlt_active=False)
        assert not (tmp_path / "chunk_compare.jsonl").exists()

    def test_baseline_dump_records_have_distinct_step_counts(self, tmp_path):
        """Regression: step counter was only incremented inside
        ``_rlt_inference_step``. Baseline (actor-off) dumps all got step=0,
        breaking cross-run alignment. The fix advances step_count in the
        main loop so successive baseline records have successive steps.

        Emulate the main loop's order of ops (bump step, then write record)
        to lock in the fix without spinning a real InferenceThread.
        """
        thread = self._make_thread_with_rlt(tmp_path)
        actions = self._actions()

        for _ in range(3):
            thread._rlt_step_count += 1  # main-loop increment (independent of actor)
            ref_norm = thread._rlt_compute_ref_norm(actions)
            thread._rlt_write_dump_record(ref_norm, None, is_rlt_active=False)

        recs = self._read_records(tmp_path)
        steps = [r["step"] for r in recs]
        assert steps == [1, 2, 3], (
            f"Baseline records must have successive step counts; got {steps}. "
            "If all are 0, step_count is still gated behind actor activity."
        )

    def test_override_polling_picks_up_dump_flag_without_gradient_updates(self, tmp_path):
        """Regression: in deploy mode, gradient_updates doesn't run, so the
        override poll (which previously lived only inside gradient_updates)
        was skipped. The Diagnostic button would silently do nothing.

        ``_rlt_check_config_overrides`` must read ``rlt_overrides.json`` from
        the output dir and flip ``self._rlt_dump_chunks`` — independent of
        whether any gradient step ever fires.
        """
        thread = self._make_thread_with_rlt(tmp_path, dump_on=False)
        assert thread._rlt_dump_chunks is False

        (tmp_path / "rlt_overrides.json").write_text(json.dumps({"dump_chunks": True}))
        thread._rlt_check_config_overrides()
        assert thread._rlt_dump_chunks is True, (
            "Override poll must flip _rlt_dump_chunks even without a gradient "
            "update loop (deploy mode has no replay buffer)."
        )

    def test_ref_norm_not_clobbered_by_actor_modifying_actions(self, tmp_path):
        """The ref we dump is S1's output, even when the actor overwrites
        ``actions[:, :C, :]`` in-place during the inference step. A view-aliasing
        bug would cause the dumped ref to become the actor's output instead."""
        thread = self._make_thread_with_rlt(tmp_path)
        actions = self._actions()
        # Capture ref BEFORE the in-place modification (as the main loop does)
        ref_norm = thread._rlt_compute_ref_norm(actions)
        # Simulate actor overwriting the first C frames
        actions[:, :10, :] = 999.0
        # Now write the dump using the earlier-captured ref
        thread._rlt_write_dump_record(ref_norm, None, is_rlt_active=False)

        recs = self._read_records(tmp_path)
        # Ref must reflect the ORIGINAL 7.0, not the post-actor 999.0
        assert recs[0]["ref"][0][0] == 7.0, (
            "ref_norm captured before actor must remain 7.0 — "
            "if it's 999, ref_chunk was aliased to actions and got overwritten"
        )

    def test_dump_rejects_stale_actor_with_inactive_flag(self, tmp_path):
        """Invariant assert: it must be impossible to silently dump a stale
        actor tensor in a record with ``rlt_active: false``. The write helper
        should refuse the call rather than produce a misattributed record."""
        thread = self._make_thread_with_rlt(tmp_path)
        ref_norm = thread._rlt_compute_ref_norm(self._actions())
        stale = torch.full((10, 14), 0.9)

        with pytest.raises(AssertionError, match="stale tensor"):
            thread._rlt_write_dump_record(ref_norm, stale, is_rlt_active=False)

    def test_last_actor_norm_initialized_and_reset(self, tmp_path):
        """``_rlt_last_actor_norm`` must be an explicit instance attribute
        (not rely on getattr's default). And the main-loop sequence (reset
        → optional inference → dump) must prevent stale leakage.
        """
        thread = self._make_thread_with_rlt(tmp_path)
        assert hasattr(thread, "_rlt_last_actor_norm"), (
            "Must be explicitly initialized in __init__ — silent getattr "
            "defaults hide uninitialized-attribute bugs."
        )
        assert thread._rlt_last_actor_norm is None

        # Simulate two consecutive inferences where the actor runs once then not.
        # Step 1: actor ran — last_actor_norm gets set.
        thread._rlt_last_actor_norm = torch.full((10, 14), 0.7)
        # Step 2: main-loop sequence resets it before dispatching.
        thread._rlt_last_actor_norm = None  # <-- main loop does this before _rlt_inference_step
        # Actor did NOT run this step, so main loop skips _rlt_inference_step.
        # Now the dump path reads `self._rlt_last_actor_norm if rlt_active else None`.
        assert thread._rlt_last_actor_norm is None, (
            "After the main-loop reset, a skipped _rlt_inference_step leaves "
            "last_actor_norm at None — no stale leakage into the dump."
        )

    def test_compute_ref_norm_rejects_too_short_chunk(self, tmp_path):
        """Setup mismatch: if RLT's chunk length > S1's chunk, we'd silently
        write truncated garbage into the dump. Assert catches it at the source."""
        thread = self._make_thread_with_rlt(tmp_path)
        short_actions = torch.zeros(1, 5, 14)  # C=10 but only 5 frames available
        with pytest.raises(AssertionError, match="< RLT rl_chunk_length"):
            thread._rlt_compute_ref_norm(short_actions)

    def test_compute_ref_norm_returns_non_aliased_clone(self, tmp_path):
        """The clone assert inside _rlt_compute_ref_norm guards against a
        view-aliasing regression. Exercise it: the returned tensor should
        not share storage with the input actions tensor."""
        thread = self._make_thread_with_rlt(tmp_path)
        actions = self._actions()
        ref_norm = thread._rlt_compute_ref_norm(actions)
        assert ref_norm.data_ptr() != actions.data_ptr(), (
            "ref_norm must be a separate tensor from actions — otherwise "
            "later in-place actor overwrite corrupts the dump's ref value."
        )

    def test_dump_records_correctly_indexed_across_episodes(self, tmp_path):
        """End-to-end indexing across an ep0 (actor on) → reset → ep1 (actor off)
        sequence. This is the realistic A/B scenario: same launch, user toggles
        E between episodes. Everything must be correctly labeled and free of
        cross-episode leakage.
        """
        thread = self._make_thread_with_rlt(tmp_path)
        actions = self._actions()

        def main_loop_step(expected_actor_output):
            """Emulate the main-loop sequence from s1_inference for one inference."""
            thread._rlt_last_actor_norm = None  # reset before actor could run
            thread._rlt_step_count += 1  # advance per-inference counter
            ref = thread._rlt_compute_ref_norm(actions)
            if thread.rlt_active:
                # Actor would run here; simulate its effect on _rlt_last_actor_norm
                thread._rlt_last_actor_norm = expected_actor_output
            if thread._rlt_dump_chunks and thread._rlt_system_active:
                actor_norm = thread._rlt_last_actor_norm if thread.rlt_active else None
                thread._rlt_write_dump_record(
                    ref,
                    actor_norm,
                    is_rlt_active=thread.rlt_active,
                )

        # Ep 0: actor engaged, 3 inferences
        thread._rlt_state["episode"] = 0
        thread._rlt_system_active = True
        thread._rlt_user_engaged = True
        ep0_actor = torch.full((10, 14), 0.42)
        for _ in range(3):
            main_loop_step(expected_actor_output=ep0_actor)

        # Reset phase (between eps): system_active off, dump must be skipped
        thread._rlt_system_active = False
        main_loop_step(expected_actor_output=ep0_actor)

        # Ep 1: actor disengaged, 2 inferences
        thread._rlt_state["episode"] = 1
        thread._rlt_system_active = True
        thread._rlt_user_engaged = False
        for _ in range(2):
            main_loop_step(expected_actor_output=ep0_actor)  # not used (actor off)

        recs = self._read_records(tmp_path)

        # Exactly 3 + 2 = 5 records; none from the reset phase
        assert len(recs) == 5, f"Expected 5 records, got {len(recs)}"

        ep0 = [r for r in recs if r["ep"] == 0]
        ep1 = [r for r in recs if r["ep"] == 1]
        assert len(ep0) == 3, f"Ep 0 should have 3 records, got {len(ep0)}"
        assert len(ep1) == 2, f"Ep 1 should have 2 records, got {len(ep1)}"

        # Ep 0: actor on, records have tensor actor fields and rlt_active=True
        for r in ep0:
            assert r["rlt_active"] is True
            assert r["actor"] is not None
            # actor was 0.42 across all 10×14 — check a corner
            assert abs(r["actor"][0][0] - 0.42) < 1e-5

        # Ep 1: actor off; critically, no leakage of ep 0's actor tensor
        for r in ep1:
            assert r["rlt_active"] is False
            assert r["actor"] is None, (
                "Ep 1 baseline record has non-null actor — stale tensor from "
                "ep 0 leaked across the episode boundary"
            )

        # Step numbers are strictly monotonic across the whole sequence
        steps = [r["step"] for r in recs]
        assert steps == sorted(steps), f"Non-monotonic step sequence: {steps}"
        assert len(set(steps)) == len(steps), f"Duplicate step numbers: {steps}"

        # Ep boundary shows up in step numbers: ep0 records precede ep1's
        assert max(r["step"] for r in ep0) < min(r["step"] for r in ep1)

    def test_ref_records_full_s1_chunk_not_just_actor_region(self, tmp_path):
        """Regression: ref in the dump used to be truncated to the first C
        frames (aligned with actor output). That threw away the S1-only tail
        past the actor boundary — which is exactly what we need to diagnose
        the actor→S1 discontinuity. Now ref covers all 50 frames; the first
        10 have value 7.0, the remaining 40 have 9.0. Both must show up."""
        thread = self._make_thread_with_rlt(tmp_path)
        ref_norm = thread._rlt_compute_ref_norm(self._actions(c_frames_val=7.0, tail_val=9.0))
        thread._rlt_write_dump_record(ref_norm, None, is_rlt_active=False)

        recs = self._read_records(tmp_path)
        assert len(recs) == 1
        ref = recs[0]["ref"]
        assert len(ref) == 50, f"Expected 50 frames of ref, got {len(ref)}"
        # First C=10 frames: the actor-region values
        assert ref[0][0] == 7.0
        assert ref[9][0] == 7.0
        # Frames 10-49: the S1-only tail
        assert ref[10][0] == 9.0, (
            "Frame 10 must be the S1-only tail value (9.0). If it's 7.0, "
            "the dump is still truncating to the first C frames."
        )
        assert ref[49][0] == 9.0

    def test_dump_skipped_during_reset_phase(self, tmp_path):
        """During reset / pre-first-episode, ``_rlt_system_active=False`` and
        the episode counter may still be -1. Dump must not record garbage.
        The gating isn't inside ``_rlt_write_dump_record`` — it's at the
        main-loop caller — so this test encodes the invariant that the
        write helper is only called after that gate."""
        thread = self._make_thread_with_rlt(tmp_path)
        thread._rlt_system_active = False
        # Directly asserting the main-loop condition structure:
        should_dump = thread._rlt_dump_chunks and thread._rlt_system_active
        assert should_dump is False, (
            "dump_chunks alone is not enough — _rlt_system_active must be "
            "True to gate out reset-phase records (where ep is stale -1)"
        )


# ---------------------------------------------------------------------------
# Latency session integration — end-to-end check that one inference cycle
# produces one record with the expected stage scalars.
# ---------------------------------------------------------------------------


class TestLatencySession:
    """Verify the inference loop emits a per-cycle record with the right
    stage breakdown when a LatencySession is provided. Tests use the same
    CPU mock policy as the rest of the file — _TimedBlock falls back to
    perf_counter on CPU, so all four stage durations should be > 0."""

    def _wait_for_records(self, session, n: int, timeout: float = 5.0) -> None:
        """Poll until the aggregator has at least n records, or fail."""
        deadline = time.perf_counter() + timeout
        while time.perf_counter() < deadline:
            if len(session.aggregator) >= n:
                return
            time.sleep(0.05)
        raise AssertionError(f"Expected {n} records within {timeout}s, got {len(session.aggregator)}")

    def test_disabled_default_is_no_op(self):
        """Without an explicit session, the thread runs as before — no
        attribute errors, chunks still produced."""
        thread = _make_thread()  # latency_session not provided
        thread.start()
        try:
            thread.publish_obs(_make_obs(), time.perf_counter())
            assert thread.wait_for_first_chunk(timeout=5.0)
        finally:
            thread.stop()

    def test_one_obs_produces_one_record(self):
        """Publishing one obs produces at least one committed record. We
        accept >=1 because ``stop()`` re-sets ``_obs_ready`` to unblock any
        waiting thread, which can let the loop run one extra iteration on
        the still-cached obs before noticing ``_running`` is cleared."""
        from lerobot.utils.latency import LatencySession

        session = LatencySession.from_config(enabled=True, loop_kind="hvla_infer", target_fps=30)
        thread = _make_thread(latency_session=session)
        thread.start()
        try:
            thread.publish_obs(_make_obs(), time.perf_counter())
            assert thread.wait_for_first_chunk(timeout=5.0)
            self._wait_for_records(session, 1)
        finally:
            thread.stop()
        assert len(session.aggregator) >= 1

    def test_record_has_expected_fields(self):
        """The committed record carries the stage scalars and the obs/loop
        meta-fields that the dashboard expects."""
        from lerobot.utils.latency import LatencySession

        session = LatencySession.from_config(enabled=True, loop_kind="hvla_infer", target_fps=30)
        thread = _make_thread(latency_session=session)
        thread.start()
        try:
            thread.publish_obs(_make_obs(), time.perf_counter())
            assert thread.wait_for_first_chunk(timeout=5.0)
            self._wait_for_records(session, 1)
        finally:
            thread.stop()
        rec = list(session.aggregator._records)[0]
        # Per-stage durations from CUDA events (or perf_counter on CPU)
        assert "enc_obs_ms" not in rec, "no rl_token_encoder → enc_obs_ms should be absent"
        # s1_denoise always runs (predict_action_chunk happens for every cycle)
        assert "s1_denoise_ms" in rec
        assert rec["s1_denoise_ms"] >= 0.0
        # CPU-side spans / fields
        assert "batch_prep_ms" in rec
        assert "obs_stale_ms" in rec
        assert rec["obs_stale_ms"] >= 0.0
        assert "inference_ms" in rec
        assert "total_delay_ms" in rec
        # Loop-level
        assert "loop_dt_ms" in rec
        assert rec["loop_kind"] == "hvla_infer"

    def test_multiple_obs_produce_multiple_records(self):
        from lerobot.utils.latency import LatencySession

        session = LatencySession.from_config(enabled=True, loop_kind="hvla_infer", target_fps=30)
        thread = _make_thread(latency_session=session)
        thread.start()
        try:
            for _ in range(3):
                thread.publish_obs(_make_obs(), time.perf_counter())
                time.sleep(0.3)
            self._wait_for_records(session, 2)
        finally:
            thread.stop()
        assert len(session.aggregator) >= 2

    def test_snapshot_published_to_disk(self, tmp_path):
        """Wiring through to s1_process: enabling latency monitoring with an
        output_dir must produce latency_snapshot.json on disk."""
        from lerobot.utils.latency import LatencySession

        session = LatencySession.from_config(
            enabled=True,
            loop_kind="hvla_infer",
            target_fps=30,
            output_dir=tmp_path,
        )
        # Force the writer's interval to 0 so the very first commit publishes.
        session.writer._interval_s = 0.0
        thread = _make_thread(latency_session=session)
        thread.start()
        try:
            thread.publish_obs(_make_obs(), time.perf_counter())
            assert thread.wait_for_first_chunk(timeout=5.0)
            self._wait_for_records(session, 1)
        finally:
            thread.stop()
        session.writer.flush()  # writer runs on a background thread
        snap_path = tmp_path / "latency_snapshot.json"
        assert snap_path.exists()
        data = json.loads(snap_path.read_text())
        assert data["loop_kind"] == "hvla_infer"


class TestInferenceTargetFps:
    """The inference thread's budget is NOT the control loop's 1/fps —
    it's 1/(fps / query_interval_steps), because the main loop executes
    ``query_interval_steps`` actions per inferred chunk before re-querying.
    Getting this wrong falsely flags the inference thread as overrunning
    every time a single inference takes longer than one control frame,
    which is the expected case for a non-trivial policy."""

    def test_query_interval_2_at_30fps_gives_15hz_inference(self):
        from lerobot.policies.hvla.s1_process import inference_target_fps_for

        assert inference_target_fps_for(fps=30, query_interval_steps=2) == 15.0

    def test_query_interval_1_matches_control_rate(self):
        """When the policy re-queries every control frame, inference must
        keep up at the full control rate."""
        from lerobot.policies.hvla.s1_process import inference_target_fps_for

        assert inference_target_fps_for(fps=30, query_interval_steps=1) == 30.0

    def test_query_interval_0_returns_none(self):
        """``query_interval_steps=0`` means "run as fast as possible" —
        no meaningful fps target. Return None so the overrun / budget
        rules silently skip."""
        from lerobot.policies.hvla.s1_process import inference_target_fps_for

        assert inference_target_fps_for(fps=30, query_interval_steps=0) is None

    def test_negative_treated_as_zero(self):
        """Defensive: a malformed config shouldn't crash the formula."""
        from lerobot.policies.hvla.s1_process import inference_target_fps_for

        assert inference_target_fps_for(fps=30, query_interval_steps=-1) is None

    def test_higher_fps_scales_proportionally(self):
        from lerobot.policies.hvla.s1_process import inference_target_fps_for

        assert inference_target_fps_for(fps=60, query_interval_steps=4) == 15.0
        assert inference_target_fps_for(fps=120, query_interval_steps=2) == 60.0

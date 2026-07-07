"""Background inference thread for S1 policy.

Encapsulates the pipelined inference loop: receives observations from the main
control loop, runs model inference on GPU, and publishes action chunks.

Supports pause/resume for human intervention: when paused, the thread stops
consuming observations and producing chunks, freeing the GPU.
"""

import logging
import threading
import time

import numpy as np
import torch

from lerobot.policies.hvla.rlt.episode import TerminalKind
from lerobot.utils.latency import LatencySession

logger = logging.getLogger(__name__)


def _slice_with_pad(
    tensor: torch.Tensor,
    start: int,
    length: int,
    dim: int = 1,
) -> torch.Tensor:
    """Take a length-window slice along ``dim``, padding with the last
    value if ``start + length`` overruns the tensor (constant-extrapolation,
    similar to RTC's hold-last-value pattern). Always returns a tensor of
    size ``length`` along ``dim``.

    Logs a WARNING when padding triggers — not an assert because the
    overrun depends on runtime values (e.g. RTC's ``expected_d`` from
    rolling latency) and isn't a programmer error. The caller can keep
    going on a padded slice; the only cost is some redundant frames at
    the tail end of the actor's window.
    """
    n = tensor.shape[dim]
    if start + length <= n:
        return tensor.narrow(dim, start, length)
    actual = max(0, n - start)
    pad_n = length - actual
    pieces = []
    if actual > 0:
        pieces.append(tensor.narrow(dim, start, actual))
    if pad_n > 0:
        last = tensor.narrow(dim, n - 1, 1)
        repeat_shape = [1] * tensor.ndim
        repeat_shape[dim] = pad_n
        pieces.append(last.repeat(repeat_shape))
    out = torch.cat(pieces, dim=dim) if len(pieces) > 1 else pieces[0]
    logger.warning(
        "slice [start=%d, len=%d] overruns dim %d size %d by %d frames; "
        "padded with last value (RTC-style hold-last)",
        start,
        length,
        dim,
        n,
        pad_n,
    )
    return out


class _TimedBlock:
    """Measures the duration of a block of code.

    Uses ``torch.cuda.Event`` on CUDA devices to capture real GPU execution
    time (CUDA ops are async; CPU-side wall clock only captures dispatch).
    Falls back to ``time.perf_counter`` on CPU-only devices where there's
    no async gap.

    Intended use:

        enc_timer = _TimedBlock(device)
        with enc_timer:
            # ... GPU / CPU work ...
        # Later, AFTER the CUDA stream has been synchronized:
        ms = enc_timer.read_ms()  # for CUDA events this requires the sync

    The read_ms() must be called after a torch.cuda.synchronize() in the
    CUDA case — otherwise the events aren't done and elapsed_time would
    block or return stale values.
    """

    __slots__ = ("_use_events", "_ev_start", "_ev_end", "_cpu_start", "_cached_ms")

    def __init__(self, device: "torch.device"):
        self._use_events = device.type == "cuda"
        self._ev_start = None
        self._ev_end = None
        self._cpu_start: float | None = None
        self._cached_ms: float = 0.0

    def __enter__(self) -> "_TimedBlock":
        if self._use_events:
            self._ev_start = torch.cuda.Event(enable_timing=True)
            self._ev_end = torch.cuda.Event(enable_timing=True)
            self._ev_start.record()
        else:
            self._cpu_start = time.perf_counter()
        return self

    def __exit__(self, *_exc) -> None:
        if self._use_events:
            self._ev_end.record()
        else:
            self._cached_ms = (time.perf_counter() - self._cpu_start) * 1000

    def read_ms(self) -> float:
        """Return elapsed ms. CUDA variant reads after the end-of-inference
        sync has drained the stream; CPU variant is already final."""
        if self._use_events and self._ev_start is not None and self._ev_end is not None:
            return self._ev_start.elapsed_time(self._ev_end)
        return self._cached_ms


class InferenceThread:
    """Background GPU inference with pause/resume support.

    The main loop publishes observations via publish_obs().
    The inference thread consumes them, runs the model, and publishes chunks.
    The main loop reads chunks via get_chunk().

    For intervention: pause() blocks the inference loop (GPU idle),
    resume() resets the policy and unblocks.
    """

    def __init__(
        self,
        policy,
        preprocessor,
        postprocessor,
        shared_cache,
        s2_latent_key: str,
        s1_image_keys: list[str],
        joint_names: list[str],
        device: torch.device,
        resize_to: tuple[int, int] | None,
        fps: int,
        num_denoise_steps: int | None = None,
        query_interval_steps: int = 0,
        grip_drop_save_dir: str | None = None,
        # RLT parameters (all None when RLT is disabled)
        rl_token_encoder=None,
        rlt_actor=None,
        rlt_agent=None,
        rlt_state: dict | None = None,
        rlt_replay=None,
        # Latency monitoring — pass a LatencySession to surface per-stage
        # GPU timings (enc_obs, rl_tok, s1_denoise, actor) into the same
        # aggregator/snapshot infrastructure used by teleop/record. The
        # default disabled session is a no-op (~50 ns per iteration).
        latency_session: LatencySession | None = None,
    ):
        self._policy = policy
        self._preprocessor = preprocessor
        self._postprocessor = postprocessor
        self._shared_cache = shared_cache
        self._s2_latent_key = s2_latent_key
        self._s1_image_keys = s1_image_keys
        self._joint_names = joint_names
        self._device = device
        self._resize_to = resize_to
        self._fps = fps
        self._num_denoise_steps = num_denoise_steps
        self._query_interval_s = query_interval_steps / fps if query_interval_steps > 0 else 0.0
        self._grip_drop_save_dir = grip_drop_save_dir

        # RLT components
        self._rl_token_encoder = rl_token_encoder
        self._rlt_actor = rlt_actor
        self._rlt_agent = rlt_agent  # TD3Agent (for gradient updates)
        self._rlt_state = rlt_state
        self._rlt_replay = rlt_replay
        # Dual flags for RL gating:
        #   _rlt_user_engaged: operator toggle (E key). Persists across episodes.
        #   _rlt_system_active: system gate (paused during intervention/reset/pre-episode).
        # Actor runs only when both are True (via rlt_active property).
        self._rlt_user_engaged = True
        self._rlt_system_active = False
        self._rlt_prev: dict | None = None  # previous transition for replay buffer
        # Latest z_rl from the most recent inference cycle. Updated
        # unconditionally every inference — must stay fresh even when
        # rlt_active=False, because the main thread's intervention
        # chunker reads this to snapshot window-start z_rl while the
        # actor is paused.
        self._rlt_latest_z_rl: torch.Tensor | None = None
        self._rlt_step_count: int = 0
        self._rlt_last_update_time: float = 0.0  # wall-clock of last gradient call (for rate)
        self._rlt_dump_chunks: bool = False  # diagnostic: dump ref/actor chunks to jsonl
        self._rlt_override_mtime: float = 0.0  # mtime of last-read rlt_overrides.json
        # Last actor output, used by the dump path. Reset every inference so
        # a stale tensor from a previous step can never leak into a record
        # (would produce a wrong 'actor' field if rlt_active flipped off).
        self._rlt_last_actor_norm: torch.Tensor | None = None
        # Rolling per-inference timing of the RLT portion (token encoder +
        # actor + dump). Separated from S1's time so we can verify RLT's
        # contribution to inference latency — which feeds RTC's expected_d.
        self._rlt_enc_ms_hist: list[float] = []
        self._rlt_post_ms_hist: list[float] = []

        # RTC config
        self._supports_rtc = getattr(policy, "supports_rtc", False)
        self._rtc_max_delay = getattr(policy, "rtc_prefix_length", 5) if self._supports_rtc else 0

        # Chunk state (written by inference thread, read by main loop)
        self._chunk_data: np.ndarray | None = None
        self._chunk_t_obs: float = 0.0
        self._chunk_t_origin: float = 0.0
        self._chunk_prefix_len: int = 0
        self._chunk_lock = threading.Lock()
        self._chunk_ready = threading.Event()

        # Obs buffer (written by main loop, read by inference thread)
        self._obs_data: dict | None = None
        self._obs_time: float = 0.0
        self._obs_lock = threading.Lock()
        self._obs_ready = threading.Event()

        # RTC: main loop reports which chunk index it's executing
        self._main_loop_chunk_idx: int = 0
        self._main_loop_chunk_idx_lock = threading.Lock()

        # Lifecycle
        self._running = threading.Event()
        self._running.set()
        self._paused = threading.Event()
        self._paused.set()  # starts unpaused

        # Timing
        self.infer_times: list[float] = []
        self.inference_delays: list[float] = []

        self._latency_session = latency_session if latency_session is not None else LatencySession.disabled()

        self._thread: threading.Thread | None = None

    # --- RLT methods ---

    @property
    def rlt_active(self) -> bool:
        """Actor runs only when both user (E key) and system (not intervening) say yes."""
        return self._rlt_user_engaged and self._rlt_system_active

    def _rlt_inference_step(
        self, batch, actions, z_rl, obs, ref_norm=None, actor_timer=None, expected_d: int = 0
    ):
        """Run RLT actor (if past warmup) and store transition.
        Called inside inference loop — one call per S1 chunk.
        Only runs when rlt_active is True (user engaged + system active).

        ``ref_norm`` may be supplied by the caller (main inference loop uses
        the same normalized ref for both actor input and diagnostic dump);
        when None, it's computed here.

        Paper Algorithm 1:
        - Line 7: S1 produces ref chunk (already done before this call)
        - Line 8: Form RL state x = (z_rl, s^p)
        - Line 9: Actor produces action (or VLA ref during warmup)
        - Line 12: Store transition in replay buffer
        """
        cfg = self._rlt_state["config"]
        C = cfg.rl_chunk_length
        is_warmup = cfg.is_warmup(self._rlt_state["episode"])

        # Normalize state
        state_t = batch.get("observation.state")
        if state_t is None:
            state_np = np.array(
                [float(obs.get(j, 0)) for j in self._joint_names],
                dtype=np.float32,
            )
            state_t = torch.from_numpy(state_np).unsqueeze(0).to(self._device)
        state_norm = state_t.float()
        if self._policy._state_mean is not None:
            state_norm = (
                state_t.float() - self._policy._state_mean.to(self._device)
            ) / self._policy._state_std.to(self._device)

        if ref_norm is None:
            ref_norm = self._rlt_compute_ref_norm(actions)
        # ref_norm covers the full S1 chunk (all 50 frames) so the dump
        # can record the S1-only tail past the actor boundary. Actor
        # input, BC comparison and replay-buffer storage use the C-frame
        # window starting at ``expected_d`` — the frames immediately
        # AFTER the RTC prefix. The RTC prefix [0:expected_d] holds
        # previous-chunk tail actions that the robot won't actually
        # execute, so refining them is wasted compute and confounds
        # training signal.
        #
        # ``_slice_with_pad`` rather than direct indexing because
        # expected_d is a runtime value (rolling latency); if it ever
        # grows enough that D+C overruns the chunk, hold-last-value
        # padding keeps the actor running with a warning instead of
        # crashing. With current config (D ≤ 6, C=10, chunk_size=50)
        # this never triggers in practice.
        D = max(0, int(expected_d))
        actor_window_end = D + C
        actor_ref = _slice_with_pad(ref_norm, D, C, dim=1)

        # Actor forward (Paper Alg 1 line 9)
        is_deploy = self._rlt_state.get("deploy", False)
        if is_warmup and not is_deploy:
            # Warmup: execute VLA reference, actor not called
            actor_norm = actor_ref
        else:
            # Actor training (``_rlt_gradient_updates``) runs in fp32 with no
            # autocast; inference reaches this point from inside S1's bf16
            # autocast block. Disable autocast locally and cast inputs to
            # fp32 so the actor sees the same dtype it was trained on —
            # otherwise bf16 rounding introduces a per-joint bias ~1e-2 that
            # accumulates into the actor's learned "bias direction".
            # ``actor_timer`` lets the caller (main inference loop) measure
            # the actor forward in isolation for the per-stage dump fields.
            _noop_cm = _TimedBlock(self._device) if actor_timer is None else actor_timer
            with torch.autocast(device_type=self._device.type, enabled=False), _noop_cm:
                actor_norm = self._rlt_actor(
                    z_rl.float(),
                    state_norm.float(),
                    actor_ref.float(),
                    deterministic=is_deploy,  # no exploration noise in deploy mode
                )
            # Denormalize and replace the C actions starting at the
            # post-prefix slice. With expected_d=0 (first chunk of an
            # episode, or RTC disabled) this reduces to the legacy
            # actions[:, :C, :] write.
            if self._policy._action_mean is not None:
                actor_denorm = actor_norm * self._policy._action_std.to(
                    self._device
                ) + self._policy._action_mean.to(self._device)
            else:
                actor_denorm = actor_norm
            # Write [D:D+C] back into the action chunk, clamped to the
            # chunk's actual size. Mirror of the read-side _slice_with_pad:
            # if the actor's window extends past the action chunk's tail,
            # discard those frames since the robot wouldn't execute them.
            write_end = min(actor_window_end, actions.shape[1])
            write_len = write_end - D
            if write_len > 0:
                actions[:, D:write_end, :] = actor_denorm[:, :write_len, :]

        # Expose actor output for the caller's dump helper, which also runs
        # when the actor did NOT run (rlt_active False) so A/B baseline
        # records share a schema. None marks "no meaningful actor output".
        self._rlt_last_actor_norm = (
            actor_norm.squeeze(0).detach() if not (is_warmup and not is_deploy) else None
        )

        # Store transition in replay buffer (Paper Alg 1 line 12)
        # Guard: rlt_active may have been cleared by main thread during this call
        # (race between main thread setting flag and inference thread mid-method).
        # If cleared, skip storage but don't crash — the actor already modified
        # the actions tensor which is fine (it'll be overwritten next inference).
        if not self.rlt_active:
            logger.warning("RLT: rlt_active cleared mid-inference, skipping transition storage")
            self._rlt_prev = None
            return

        prev = self._rlt_prev
        if prev is not None and self._rlt_replay is not None:
            # ONE-SHOT terminal consumption per episode (see
            # rlt/episode.py and the ep124 17-aborts incident).
            # consume_terminal_for_storage returns the terminal kind
            # ONCE per episode — even if we fire N times during the
            # post-keypress / pre-end_episode window, only the first
            # call gets the kind, the rest get None and we write
            # done=False, reward=0. Structural fix: the bug becomes
            # impossible regardless of timing.
            terminal = self._rlt_state["lifecycle"].consume_terminal_for_storage()
            done = terminal is not None
            if terminal == TerminalKind.ABORT:
                reward = float(self._rlt_state["config"].abort_reward)
            elif terminal == TerminalKind.SUCCESS:
                reward = 1.0
            else:
                reward = 0.0
            # Final precondition before writing: rlt_active was checked at the
            # top of this method, but the main thread can flip
            # _rlt_system_active between that guard and this point. If that
            # races inside this narrow window, we'd write a transition whose
            # "next" state is technically post-intervention-start. Assert so
            # any future occurrence is loud — if this ever fires, the race
            # is real and we need to add a lock around the toggle.
            assert self.rlt_active, (
                "RL transition store entered with rlt_active=False — main "
                "thread flipped _rlt_system_active mid-iteration. Race "
                "between intervention transition and replay.add(). Need a "
                "lock around _rlt_system_active updates."
            )
            self._rlt_replay.add(
                z_rl=prev["z_rl"],
                state=prev["state"],
                action_chunk=prev["action"],
                ref_chunk=prev["ref"],
                reward=reward,
                next_z_rl=z_rl.squeeze(0).float().detach(),
                next_state=state_norm.squeeze(0).detach(),
                next_ref_chunk=actor_ref.squeeze(0).detach(),
                done=done,
            )
            self._rlt_state["total_transitions"] += 1

        self._rlt_prev = {
            "z_rl": z_rl.squeeze(0).float().detach(),
            "state": state_norm.squeeze(0).detach(),
            "action": actor_norm.squeeze(0).detach(),
            "ref": actor_ref.squeeze(0).detach(),
        }

        # Metrics + logging. Note: step_count is advanced in the main loop
        # (so baseline / actor-off dumps also get a unique step per inference).
        actor_delta = (actor_norm - actor_ref).abs().mean().item()

        from lerobot.policies.hvla.rlt.metrics import get_metrics, save_metrics_to_file

        is_deploy = self._rlt_state.get("deploy", False)
        mode = "DEPLOY" if is_deploy else ("WARMUP" if is_warmup else "RL")
        get_metrics().record_inference(
            step=self._rlt_step_count,
            delta=actor_delta,
            buffer_size=len(self._rlt_replay) if self._rlt_replay else 0,
            total_updates=self._rlt_state["total_updates"],
            mode=mode,
        )
        if self._rlt_step_count % 100 == 0:
            logger.info(
                "RLT step %d [%s] | delta=%.3f | buf=%d | updates=%d",
                self._rlt_step_count,
                mode,
                actor_delta,
                len(self._rlt_replay) if self._rlt_replay else 0,
                self._rlt_state["total_updates"],
            )
            save_metrics_to_file()

    def _rlt_compute_ref_norm(self, actions):
        """Normalize the full S1 action chunk into actor input space.

        Returns the complete chunk (e.g. 50 frames) — callers slice the first
        ``cfg.rl_chunk_length`` for the actor input and for replay-buffer
        storage; the dump path records the full chunk so analysis can see
        the actor-modified prefix alongside the S1-only tail.

        Computed from a clone so later in-place modification of
        ``actions[:, :C, :]`` (when actor executes) cannot corrupt the
        dump's ref value.
        """
        cfg = self._rlt_state["config"]
        C = cfg.rl_chunk_length
        # The actor needs at least C frames to operate on. If S1 emits
        # fewer, downstream slices truncate silently and per-frame
        # indexing between actor output and ref desyncs.
        assert actions.shape[1] >= C, (
            f"S1 chunk length {actions.shape[1]} < RLT rl_chunk_length {C}; "
            "ref would be truncated and per-frame indexing desyncs from actor"
        )
        ref = actions.clone().float()
        if self._policy._action_mean is not None:
            ref = (ref - self._policy._action_mean.to(self._device)) / self._policy._action_std.to(
                self._device
            )
        # Clone must decouple ref from actions; otherwise the actor's
        # later in-place write to ``actions[:, D:D+C, :]`` would overwrite
        # the dump's ref in place — silently misattributing actor output
        # as "S1 reference" in every record.
        assert ref.data_ptr() != actions.data_ptr(), (
            "ref aliases actions — in-place actor overwrite would corrupt the dump's ref value"
        )
        return ref

    def _rlt_write_dump_record(
        self,
        ref_norm,
        actor_norm,
        is_rlt_active: bool,
        prefix_len: int = 0,
        exec_idx: int | None = None,
        rlt_enc_ms: float | None = None,
        rlt_post_ms: float | None = None,
        total_delay_ms: float | None = None,
        obs_to_infer_ms: float | None = None,
        enc_obs_ms: float | None = None,
        rl_tok_ms: float | None = None,
        s1_denoise_ms: float | None = None,
        rlt_actor_ms: float | None = None,
    ):
        """Append one chunk_compare.jsonl record.

        Runs whenever RLT is loaded and ``dump_chunks`` is on — regardless
        of whether the actor executed this step. When the actor didn't run
        (user disengaged via E / Start-engaged off), ``actor_norm`` is None
        and the record stores ``"actor": null``. This lets A/B baseline runs
        (actor off) produce schema-compatible records alongside actor-on runs.

        ``prefix_len`` is the RTC prefix size D this inference received.
        The first D frames of ``ref`` are the clamped prefix (already-
        committed past actions); the rest are S1's denoised continuation.
        The actor refines frames ``[D:D+C]`` of the chunk — i.e. the
        ``actor`` field in the record corresponds to ``ref[D:D+C]``, NOT
        ``ref[0:C]``. Analysis scripts that compare actor[i] vs ref[i]
        should use ``ref[prefix_len + i]``.
        ``exec_idx`` is the chunk index the robot had reached when this
        inference started — useful for reconstructing which frames
        actually executed.

        ``rlt_enc_ms`` / ``rlt_post_ms`` / ``total_delay_ms`` are documented
        at the record construction site: total_delay is GPU-sync'd end-to-end
        and accurate; the enc/post splits are CPU dispatch times only (async
        GPU work, lower bound).
        """
        assert ref_norm is not None, "_rlt_write_dump_record called without ref_norm"
        # Misattribution tripwire: if ``actor_norm`` is a tensor while
        # ``is_rlt_active`` says the actor didn't run, the record would claim
        # "actor=X" in a baseline leg — exactly the staleness bug we've been
        # chasing. Callers must zero out actor_norm themselves.
        assert is_rlt_active or actor_norm is None, (
            "actor_norm must be None when is_rlt_active=False; otherwise "
            "a stale tensor from a previous step misattributes a baseline record"
        )
        if self._rlt_state is None:
            return
        output_dir = self._rlt_state.get("output_dir")
        if output_dir is None:
            return
        try:
            import json as _json
            import time as _t

            record = {
                "t": _t.time(),
                "step": self._rlt_step_count,
                "ep": self._rlt_state.get("episode", 0),
                "ref": ref_norm.squeeze(0).detach().cpu().tolist()
                if ref_norm.dim() > 2
                else ref_norm.detach().cpu().tolist(),
                "actor": actor_norm.detach().cpu().tolist() if actor_norm is not None else None,
                "deploy": self._rlt_state.get("deploy", False),
                "rlt_active": bool(is_rlt_active),
                "prefix_len": int(prefix_len),
                "exec_idx": int(exec_idx) if exec_idx is not None else None,
                # Per-inference timing — the full journey in one record.
                #
                # ``total_delay_ms`` = obs-to-chunk-ready end-to-end latency,
                # measured AFTER torch.cuda.synchronize(). Feeds RTC's
                # expected_d. This is the ground truth.
                #
                # Per-stage timings (all GPU execution via torch.cuda.Event,
                # no extra syncs on hot path; CPU fallback uses perf_counter):
                #   * obs_to_infer_ms — queue + transfer before inference starts
                #   * enc_obs_ms      — DINOv2 feature extraction (encode_observations)
                #   * rl_tok_ms       — RL token encoder forward only
                #   * s1_denoise_ms   — predict_action_chunk (flow matching + RTC prefix)
                #   * rlt_actor_ms    — actor MLP forward only (0 when actor didn't run)
                #   * rlt_post_ms     — umbrella for post-S1 RLT block
                #                       (ref_norm + actor + transition + dump prep)
                #   * rlt_enc_ms      — legacy sum (enc_obs_ms + rl_tok_ms); kept for
                #                       continuity with earlier analysis code.
                "rlt_enc_ms": round(rlt_enc_ms, 3) if rlt_enc_ms is not None else None,
                "rlt_post_ms": round(rlt_post_ms, 3) if rlt_post_ms is not None else None,
                "total_delay_ms": round(total_delay_ms, 1) if total_delay_ms is not None else None,
                "obs_to_infer_ms": round(obs_to_infer_ms, 2) if obs_to_infer_ms is not None else None,
                "enc_obs_ms": round(enc_obs_ms, 3) if enc_obs_ms is not None else None,
                "rl_tok_ms": round(rl_tok_ms, 3) if rl_tok_ms is not None else None,
                "s1_denoise_ms": round(s1_denoise_ms, 3) if s1_denoise_ms is not None else None,
                "rlt_actor_ms": round(rlt_actor_ms, 3) if rlt_actor_ms is not None else None,
            }
            with open(output_dir / "chunk_compare.jsonl", "a") as f:
                f.write(_json.dumps(record) + "\n")
        except Exception as e:
            logger.warning("Failed to dump chunk comparison: %s", e)

    def _rlt_check_config_overrides(self):
        """Check for GUI config overrides (beta, sigma sliders).
        Called each gradient step — cost is one stat() syscall (~1us)."""
        import json as _json

        override_path = self._rlt_state["output_dir"] / "rlt_overrides.json"
        try:
            mtime = override_path.stat().st_mtime
        except OSError:
            return
        last = getattr(self, "_rlt_override_mtime", 0)
        if mtime <= last:
            return
        self._rlt_override_mtime = mtime
        try:
            with open(override_path) as f:
                overrides = _json.load(f)
            # Back-compat: accept legacy "actor_sigma" as a synonym for
            # "exploration_sigma". Older rlt_overrides.json files on disk
            # (and older GUI builds) still use the old name.
            if "actor_sigma" in overrides and "exploration_sigma" not in overrides:
                overrides["exploration_sigma"] = overrides["actor_sigma"]
            cfg = self._rlt_state["config"]
            for key in ("beta", "exploration_sigma", "target_sigma", "target_noise_clip"):
                if key in overrides:
                    old = getattr(cfg, key)
                    new = float(overrides[key])
                    if old != new:
                        setattr(cfg, key, new)
                        logger.info("RLT config override: %s %.4f → %.4f", key, old, new)
            if "dump_chunks" in overrides:
                new = bool(overrides["dump_chunks"])
                old = getattr(self, "_rlt_dump_chunks", False)
                if old != new:
                    self._rlt_dump_chunks = new
                    logger.info("RLT diagnostic dump: %s", "ON" if new else "OFF")
        except Exception as e:
            logger.warning("RLT config override read failed: %s", e)

    def _rlt_gradient_updates(self):
        """Run UTD gradient steps during query interval idle time.
        Paper: UTD=5, two critic updates per actor update.
        Paper Alg 1 lines 13-18: runs every step including warmup.
        """
        import time as _time

        t0 = _time.perf_counter()

        # Compute wall-clock update rate = UTD / elapsed since last call.
        # First call or long gap: rate = 0 (treat as non-training time).
        elapsed_since_last = t0 - self._rlt_last_update_time if self._rlt_last_update_time > 0 else 0.0
        self._rlt_last_update_time = t0

        self._rlt_check_config_overrides()
        cfg = self._rlt_state["config"]
        C = cfg.rl_chunk_length
        A_flat = self._rlt_replay._action.shape[1]
        A = A_flat // C

        critic_sum, actor_sum = 0.0, 0.0
        grad_norm_sum, grad_norm_max = 0.0, 0.0
        q_term_sum, bc_term_sum = 0.0, 0.0
        n_critic, n_actor = 0, 0

        for _ in range(cfg.utd_ratio):
            # 2 critic updates (Paper Appendix B)
            for _ in range(2):
                b = self._rlt_replay.sample(256)
                cl, gn = self._rlt_agent.update_critic(
                    b["z_rl"],
                    b["state"],
                    b["action"].reshape(-1, C, A),
                    b["ref"].reshape(-1, C, A),
                    b["reward"],
                    b["next_z_rl"],
                    b["next_state"],
                    b["next_ref"].reshape(-1, C, A),
                    b["done"],
                )
                critic_sum += cl
                grad_norm_sum += gn
                grad_norm_max = max(grad_norm_max, gn)
                n_critic += 1
            # 1 actor update (Paper Alg 1 line 17)
            b = self._rlt_replay.sample(256)
            al, q_term, bc_term = self._rlt_agent.update_actor(
                b["z_rl"],
                b["state"],
                b["ref"].reshape(-1, C, A),
            )
            actor_sum += al
            q_term_sum += q_term
            bc_term_sum += bc_term
            n_actor += 1
            self._rlt_state["total_updates"] += 1

        elapsed = (_time.perf_counter() - t0) * 1000
        avg_c = critic_sum / n_critic if n_critic else 0
        avg_gn = grad_norm_sum / n_critic if n_critic else 0
        avg_a = actor_sum / n_actor if n_actor else 0
        avg_q_term = q_term_sum / n_actor if n_actor else 0
        avg_bc_term = bc_term_sum / n_actor if n_actor else 0

        # Log Q values for monitoring (Paper's primary metric for critic quality)
        with torch.no_grad():
            b = self._rlt_replay.sample(min(256, len(self._rlt_replay)))
            qs = self._rlt_agent.critic.min_q(
                b["z_rl"],
                b["state"],
                b["action"].reshape(-1, C, A),
            )
            q_mean = qs.mean().item()
            q_min = qs.min().item()
            q_max = qs.max().item()

        if self._rlt_state["total_updates"] % 10 < cfg.utd_ratio:
            # q_term = -Q.mean (sign depends on critic; negative when Q>0 as expected).
            # bc_term = β * ||a-ref||² (always ≥ 0, pulls actor toward S1 ref).
            # gn_max: max pre-clip grad norm across this batch of critic steps.
            # Persistently near cfg.critic_grad_clip = clip triggering often;
            # a sudden spike = the defense is catching a bad update.
            logger.info(
                "RLT grad | critic=%.4f gn=%.2f/%.2f actor=%.4f (q=%.4f bc=%.4f) | Q: mean=%.4f min=%.4f max=%.4f | updates=%d | %.0fms",
                avg_c,
                avg_gn,
                grad_norm_max,
                avg_a,
                avg_q_term,
                avg_bc_term,
                q_mean,
                q_min,
                q_max,
                self._rlt_state["total_updates"],
                elapsed,
            )

        # Update metrics with Q values
        from lerobot.policies.hvla.rlt.metrics import get_metrics

        # Update rate: actor updates since last call / elapsed time.
        # 0 on first call or after a long pause (intervention/reset).
        update_rate = cfg.utd_ratio / elapsed_since_last if elapsed_since_last > 0 else 0.0
        get_metrics().record_grad_update(
            total_updates=self._rlt_state["total_updates"],
            mode="TRAIN",
            critic_loss=avg_c,
            critic_grad_norm=grad_norm_max,
            actor_loss=avg_a,
            q_mean=q_mean,
            q_min=q_min,
            q_max=q_max,
            actor_q_term=avg_q_term,
            actor_bc_term=avg_bc_term,
            update_rate=update_rate,
        )

    def start(self) -> None:
        """Start the background inference thread."""
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self, timeout: float = 3.0) -> None:
        """Signal thread to stop and join."""
        self._running.clear()
        self._paused.set()  # unblock if paused
        self._obs_ready.set()  # unblock if waiting for obs
        if self._thread is not None:
            self._thread.join(timeout=timeout)

    def pause(self) -> None:
        """Pause inference (for intervention). Thread blocks until resume()."""
        self._paused.clear()
        logger.info("S1 inference: paused")

    def resume(self) -> None:
        """Resume inference after intervention. Resets policy state."""
        self._policy.reset()
        # Clear stale obs so we get a fresh one
        self._obs_ready.clear()
        self._paused.set()
        logger.info("S1 inference: resumed")

    @property
    def is_paused(self) -> bool:
        return not self._paused.is_set()

    def publish_obs(self, obs: dict, t_now: float) -> None:
        """Main loop publishes observation for the inference thread."""
        with self._obs_lock:
            self._obs_data = obs
            self._obs_time = t_now
        self._obs_ready.set()

    def get_chunk(self) -> tuple[np.ndarray | None, float, float]:
        """Read latest (chunk, t_origin, t_obs). Thread-safe."""
        with self._chunk_lock:
            return self._chunk_data, self._chunk_t_origin, self._chunk_t_obs

    def update_exec_index(self, idx: int) -> None:
        """Main loop reports which chunk index it just executed (for RTC prefix)."""
        with self._main_loop_chunk_idx_lock:
            self._main_loop_chunk_idx = idx

    def wait_for_first_chunk(self, timeout: float = 60.0) -> bool:
        """Block until first chunk is ready. Returns False on timeout."""
        return self._chunk_ready.wait(timeout=timeout)

    # --- Internal ---

    def _loop(self) -> None:
        """Background loop: wait for obs → prep batch → infer → publish chunk."""
        from lerobot.policies.hvla.s1.protocol import ACTION_PREFIX_KEY
        from lerobot.policies.hvla.s1_process import _save_infer_drop, obs_to_s1_batch

        while self._running.is_set():
            # Pause gate: block here during intervention
            if not self._paused.wait(timeout=0.5):
                continue

            if not self._running.is_set():
                break

            # Wait for fresh obs from main loop
            if not self._obs_ready.wait(timeout=0.5):
                continue
            self._obs_ready.clear()

            with self._obs_lock:
                obs = self._obs_data
                t_obs = self._obs_time

            if obs is None:
                continue

            # Latency monitoring: start iteration after we have a valid obs.
            # On the happy path, end_iter() runs after chunk publish below.
            # If anything raises in between, the next start_iter() resets the
            # tracer cleanly — failed iterations simply don't appear in the
            # aggregator (which is the right behaviour for latency stats).
            self._latency_session.start_iter()
            self._latency_session.set_field("obs_stale_ms", (time.perf_counter() - t_obs) * 1000.0)

            # Prepare batch (CPU resize + GPU transfer)
            with self._latency_session.span("batch_prep"):
                batch = obs_to_s1_batch(
                    obs,
                    self._s1_image_keys,
                    self._shared_cache,
                    self._s2_latent_key,
                    self._device,
                    resize_to=self._resize_to,
                )
                batch = self._preprocessor(batch)

            # RTC prefix
            current_prefix_len = 0
            exec_idx = None
            expected_d = 0
            if self._supports_rtc:
                with self._chunk_lock:
                    old_chunk = self._chunk_data
                with self._main_loop_chunk_idx_lock:
                    exec_idx = self._main_loop_chunk_idx
                    t_exec_idx = time.perf_counter()

                if old_chunk is not None and exec_idx < len(old_chunk):
                    if self.inference_delays:
                        expected_d = round(np.mean(self.inference_delays[-10:]) * self._fps)
                    else:
                        expected_d = 3
                    expected_d = max(1, min(expected_d, self._rtc_max_delay, len(old_chunk) - exec_idx))

                    prefix = old_chunk[exec_idx : exec_idx + expected_d]
                    batch[ACTION_PREFIX_KEY] = torch.from_numpy(prefix).unsqueeze(0).to(self._device)
                    current_prefix_len = prefix.shape[0]

            # Inference — per-stage timers for dump-level journey tracking.
            # Each block is wrapped; the main log stays quiet (one RTC diag
            # every 50 steps). Readings happen after the end-of-inference
            # sync, so CUDA events resolve correctly.
            t_infer_start = time.perf_counter()
            enc_obs_timer = _TimedBlock(self._device)
            rl_tok_timer = _TimedBlock(self._device)
            s1_denoise_timer = _TimedBlock(self._device)
            actor_timer = _TimedBlock(self._device)
            # Legacy umbrella timer — kept for pre-split consumers, covers
            # the same range as the post-S1 RLT block (ref_norm + actor +
            # dump prep).
            rlt_post_timer = _TimedBlock(self._device)
            with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                # RLT: extract z_rl from S1 context tokens. Context is
                # computed ONCE and reused — we also pass it to
                # predict_action_chunk below so S1's internal
                # encode_observations call is skipped (otherwise the same
                # DINOv2 forward runs twice).
                z_rl_out = None
                cached_context = None
                if self._rl_token_encoder is not None:
                    with enc_obs_timer:
                        # CRITICAL: use the policy's shared prep helper so the
                        # state is normalized (z-scored) the same way it was
                        # during RL-token-encoder training. Calling
                        # encode_observations on un-normalized state produces
                        # OOD z_rl — a latent bug that hurt RLT learning for
                        # weeks before the parity tests caught it.
                        rlt_batch = self._policy.prepare_batch_for_encode_observations(batch)
                        cached_context = self._policy.model.encode_observations(rlt_batch)
                    with rl_tok_timer:
                        z_rl_out = self._rl_token_encoder(cached_context.float()).detach()

                # S1 flow matching decoder → reference chunk. Pass the
                # pre-computed context (if any) so the internal
                # encode_observations call is skipped — saves the ~11ms
                # DINOv2 forward. The kwarg is only added when we have
                # a context to share, so policy implementations without
                # the ``context`` parameter (mocks, older variants) aren't
                # forced to accept it.
                with s1_denoise_timer:
                    predict_kwargs = {"num_steps": self._num_denoise_steps}
                    if cached_context is not None:
                        predict_kwargs["context"] = cached_context
                    actions = self._policy.predict_action_chunk(batch, **predict_kwargs)
                    actions = self._postprocessor(actions)

                # RLT: compute ref_norm, optionally run actor, optionally dump.
                # Dump is decoupled from actor activity so A/B baseline runs
                # (actor off) produce schema-compatible records — the
                # Diagnostic button just works for both modes.
                ref_norm_snapshot = None
                _dump_pending = False
                _dump_actor = None
                if self._rlt_state is not None and z_rl_out is not None:
                    # Publish the freshest z_rl unconditionally, before the
                    # rlt_active gate. Main-thread consumers (intervention
                    # recorder) need this to stay current even while the
                    # actor is paused.
                    self._rlt_latest_z_rl = z_rl_out.squeeze(0).float().detach()
                    with rlt_post_timer:
                        # Refresh live overrides (beta / sigma / dump_chunks)
                        # here, not only inside gradient_updates. Gradient updates
                        # don't fire in deploy mode, so without this call the
                        # Diagnostic button would be ignored during deploy runs.
                        self._rlt_check_config_overrides()

                        # Reset the per-step actor snapshot BEFORE dispatching so
                        # that if _rlt_inference_step doesn't run (or returns
                        # without assigning), the dump path cannot see a stale
                        # tensor from a previous inference.
                        self._rlt_last_actor_norm = None
                        self._rlt_step_count += 1
                        ref_norm_snapshot = self._rlt_compute_ref_norm(actions)
                        if self.rlt_active:
                            self._rlt_inference_step(
                                batch,
                                actions,
                                z_rl_out,
                                obs,
                                ref_norm=ref_norm_snapshot,
                                actor_timer=actor_timer,
                                expected_d=expected_d,
                            )
                        # Dump only during an active episode — system_active is
                        # cleared during reset / pre-first-episode, where the
                        # episode counter is stale (-1 before first ep) and the
                        # state is meaningless for analysis.
                        # Snapshot the dump args now; the actual write happens
                        # after t_infer_end so the record can carry real timing.
                        _dump_pending = self._rlt_dump_chunks and self._rlt_system_active
                        _dump_actor = self._rlt_last_actor_norm if self.rlt_active else None

            # Force GPU to finish before stamping t_infer_end. Without this,
            # t_infer_end captures CPU dispatch time while GPU is still
            # running — which systematically understates latency and leads
            # to an expected_d that's too short for the next RTC prefix.
            # The sync cost is ~0 here because the chunk tensor is needed
            # immediately below (it would block at .cpu() anyway). This same
            # sync also makes the CUDA events readable on the next line.
            if self._device.type == "cuda":
                torch.cuda.synchronize()

            # Now that the stream has drained, read the event-based GPU
            # timings (accurate, unlike the CPU-dispatch perf_counter path).
            enc_obs_ms = enc_obs_timer.read_ms()
            rl_tok_ms = rl_tok_timer.read_ms()
            s1_denoise_ms = s1_denoise_timer.read_ms()
            rlt_actor_ms = actor_timer.read_ms()  # 0 if actor didn't run
            rlt_post_ms = rlt_post_timer.read_ms()
            # Legacy umbrella field for the dump's "enc" stage — retained
            # for continuity with earlier record schema, now redundant with
            # the two-way split into enc_obs_ms + rl_tok_ms.
            rlt_enc_ms = enc_obs_ms + rl_tok_ms

            t_infer_end = time.perf_counter()
            infer_ms = (t_infer_end - t_infer_start) * 1000
            self.infer_times.append(infer_ms)
            total_delay = t_infer_end - t_obs
            self.inference_delays.append(total_delay)
            self._rlt_enc_ms_hist.append(rlt_enc_ms)
            self._rlt_post_ms_hist.append(rlt_post_ms)

            # Stamp the GPU stages onto the latency tracer. Stages serialize
            # on the CUDA stream, so laying them out sequentially from
            # t_infer_start gives a Gantt that matches their execution order;
            # the cumulative durations match (t_infer_end - t_infer_start)
            # modulo dispatch overhead.
            _t = t_infer_start
            for _name, _dur_ms in (
                ("enc_obs", enc_obs_ms),
                ("rl_tok", rl_tok_ms),
                ("s1_denoise", s1_denoise_ms),
                ("actor", rlt_actor_ms),
            ):
                if _dur_ms > 0.0:
                    _end = _t + _dur_ms / 1000.0
                    self._latency_session.add_span(_name, _t, _end)
                    _t = _end
            self._latency_session.set_field("inference_ms", infer_ms)
            self._latency_session.set_field("total_delay_ms", total_delay * 1000.0)
            if len(self._rlt_enc_ms_hist) > 200:
                self._rlt_enc_ms_hist = self._rlt_enc_ms_hist[-200:]
                self._rlt_post_ms_hist = self._rlt_post_ms_hist[-200:]

            # Dump record now that the post-sync timing is known. Writing
            # the record here (rather than inside the autocast block) lets
            # us include ``total_delay_ms`` — the same quantity that feeds
            # RTC's ``expected_d``. Per-inference timing therefore lives in
            # the dump file, not in the main log.
            if self._rlt_state is not None and z_rl_out is not None and _dump_pending:
                obs_to_infer_ms = (t_infer_start - t_obs) * 1000
                self._rlt_write_dump_record(
                    ref_norm_snapshot,
                    _dump_actor,
                    is_rlt_active=self.rlt_active,
                    prefix_len=current_prefix_len,
                    exec_idx=exec_idx,
                    rlt_enc_ms=rlt_enc_ms,
                    rlt_post_ms=rlt_post_ms,
                    total_delay_ms=total_delay * 1000,
                    obs_to_infer_ms=obs_to_infer_ms,
                    enc_obs_ms=enc_obs_ms,
                    rl_tok_ms=rl_tok_ms,
                    s1_denoise_ms=s1_denoise_ms,
                    rlt_actor_ms=rlt_actor_ms,
                )

            chunk_np = actions.cpu().numpy()[0]

            # RTC diagnostics (periodic)
            if self._supports_rtc and (len(self.infer_times) <= 5 or len(self.infer_times) % 50 == 0):
                actual_d_frames = round((t_infer_end - t_infer_start) * self._fps)
                obs_to_infer_ms = (t_infer_start - t_obs) * 1000
                prefix_drift = None
                if current_prefix_len > 0:
                    inner_model = self._policy.model if hasattr(self._policy, "model") else self._policy
                    prefix_drift = getattr(inner_model, "_last_prefix_drift", None)
                logger.info(
                    "S1 RTC diag | expected_d=%d actual_d=%d prefix_len=%d "
                    "| obs→infer=%.0fms infer=%.0fms total=%.0fms "
                    "| prefix_drift=%s exec_idx=%s",
                    expected_d,
                    actual_d_frames,
                    current_prefix_len,
                    obs_to_infer_ms,
                    infer_ms,
                    total_delay * 1000,
                    f"{prefix_drift:.4f}" if prefix_drift is not None else "N/A",
                    exec_idx,
                )

            # RTC alignment diagnostic
            with self._chunk_lock:
                old_chunk_for_align = self._chunk_data

            if self._supports_rtc and old_chunk_for_align is not None and exec_idx is not None:
                scan_center = exec_idx
                best_offset = 0
                best_err = float("inf")
                errors = []
                for offset in range(
                    max(0, scan_center - 3), min(len(old_chunk_for_align) - 1, scan_center + 8)
                ):
                    n = min(8, len(chunk_np), len(old_chunk_for_align) - offset)
                    if n <= 0:
                        continue
                    err = np.mean(np.abs(chunk_np[:n] - old_chunk_for_align[offset : offset + n]))
                    errors.append(f"{offset}={err:.1f}")
                    if err < best_err:
                        best_err = err
                        best_offset = offset
                if len(errors) > 0 and (len(self.infer_times) <= 5 or len(self.infer_times) % 50 == 0):
                    logger.info(
                        "S1 RTC align | exec_idx=%d best=%d (err=%.2f) | %s",
                        exec_idx,
                        best_offset,
                        best_err,
                        " ".join(errors),
                    )

            # Grip drop diagnostics
            if self._grip_drop_save_dir:
                _save_infer_drop(chunk_np, obs, len(self.infer_times), self._grip_drop_save_dir)
                infer_count = len(self.infer_times)
                if infer_count % 50 == 0:
                    import os

                    ctrl_dir = os.path.join(self._grip_drop_save_dir, f"control_{infer_count}")
                    os.makedirs(ctrl_dir, exist_ok=True)
                    state_arr = np.array([float(obs.get(j, 0)) for j in self._joint_names])
                    np.save(os.path.join(ctrl_dir, "state.npy"), state_arr)
                    np.save(os.path.join(ctrl_dir, "chunk.npy"), chunk_np)

            # Publish chunk
            with self._chunk_lock:
                self._chunk_data = chunk_np
                self._chunk_t_obs = t_obs
                if self._supports_rtc and exec_idx is not None and "t_exec_idx" in dir():
                    self._chunk_t_origin = t_exec_idx
                else:
                    self._chunk_t_origin = t_obs
                self._chunk_prefix_len = current_prefix_len
                self._chunk_ready.set()

            # Commit the iteration to the latency aggregator now — gradient
            # updates and the query-interval sleep below are post-publish
            # housekeeping that shouldn't inflate loop_dt_ms.
            self._latency_session.end_iter()

            # Query interval: use idle time for RLT gradient updates
            if self._query_interval_s > 0 and self._running.is_set():
                if (
                    self._rlt_state is not None
                    and self._rlt_replay is not None
                    and len(self._rlt_replay) >= 256
                ):
                    self._rlt_gradient_updates()
                    # Sleep remaining time if updates were faster than query interval
                    elapsed_since_infer = time.perf_counter() - t_infer_end
                    remaining = self._query_interval_s - elapsed_since_infer
                    if remaining > 0:
                        time.sleep(remaining)
                else:
                    time.sleep(self._query_interval_s)

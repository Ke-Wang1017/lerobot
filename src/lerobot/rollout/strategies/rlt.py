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

"""RLT rollout strategy: online RL fine-tuning with human intervention.

RLT ("RL Token") trains a lightweight TD3 actor-critic on top of a *frozen*
base policy, online, on the robot. It shares the human-in-the-loop mechanism
with DAgger (see :class:`InterventionStrategy`) — autonomous execution, teleop
takeover, smooth handover — but instead of only recording a supervised dataset,
it compresses the policy's encoded observation into an RL token, refines the
policy's action chunk with the actor, stores transitions in a replay buffer, and
runs gradient updates.

The RL machinery lives in :mod:`lerobot.policies.rlt` and reaches the base
policy through an :class:`~lerobot.policies.rlt.policy_adapter.RLTPolicyAdapter`,
so this strategy is policy-agnostic. The adapter is resolved from the concrete
policy on the :class:`RolloutContext` at setup time.

Structure (Phase 2 of the RLT/DAgger unification):

* :meth:`setup` — build the RL machinery (token encoder, TD3 agent, replay
  buffer, intervention recorder, metrics, episode lifecycle) and, if a
  checkpoint is given, resume actor/critic/optimizer/replay/metrics. Ported from
  ``hvla/s1_process.py:run_s1`` (the ``rlt_mode`` block).
* the training **hooks** (:meth:`on_transition`, :meth:`background_step`,
  :meth:`on_intervention_frame`, :meth:`on_terminal`, :meth:`on_episode_begin`,
  :meth:`on_episode_commit`, :meth:`on_episode_discard`) — the RL numerics,
  ported from ``hvla/s1_inference.py:InferenceThread`` (``_rlt_inference_step`` /
  ``_rlt_gradient_updates``) and the episode-end bookkeeping in ``s1_process``.
  These operate on already-computed ``z_rl`` / ``state`` / ``action`` / ``ref``
  tensors, so they are policy-agnostic and unit-testable on CPU.
* :meth:`run` — the control loop wiring the shared intervention state machine to
  those hooks. It drives the base policy through the adapter (encode_context →
  z_rl, reference chunk, actor refinement) and executes the refined chunk. The
  chunk-execution + action-space plumbing is the on-robot-validation surface;
  ``tests/hvla/test_rlt_parity.py`` guards the numerics.
"""

from __future__ import annotations

import logging
import os
import time
from contextlib import nullcontext
from pathlib import Path
from threading import Event
from typing import TYPE_CHECKING, Any

import numpy as np
import torch

from lerobot.common.control_utils import (
    follower_smooth_move_to,
    teleop_smooth_move_to,
    teleop_supports_feedback,
)
from lerobot.policies.utils import prepare_observation_for_inference
from lerobot.utils.constants import OBS_STR
from lerobot.utils.feature_utils import build_dataset_frame
from lerobot.utils.robot_utils import precise_sleep

from .intervention import InterventionPhase, InterventionStrategy

if TYPE_CHECKING:
    from ..configs import RLTStrategyConfig
    from ..context import RolloutContext

logger = logging.getLogger(__name__)


def _atomic_torch_save(obj, path) -> None:
    """``torch.save`` with crash-safe atomicity (write tmp + ``os.replace``).

    Either the new file is fully present or the previous one is untouched —
    never a half-written checkpoint. Mirrors ``hvla/s1_process._atomic_torch_save``.
    """
    target = str(path)
    tmp = target + ".tmp"
    torch.save(obj, tmp)
    os.replace(tmp, target)


def resolve_rlt_adapter(policy):
    """Return an ``RLTPolicyAdapter`` for the given base policy.

    Duck-typed dispatch: HVLA S1 (flow-matching) exposes
    ``prepare_batch_for_encode_observations`` and an inner ``model`` with
    ``encode_observations`` — the only adapter available today. pi0 / pi05 /
    multi_task_dit adapters arrive in Phase 4 (each needs a new
    ``encode_observations`` hook on the policy first), and will register here.
    """
    has_s1_context = hasattr(policy, "prepare_batch_for_encode_observations") and hasattr(
        getattr(policy, "model", None), "encode_observations"
    )
    if has_s1_context:
        from lerobot.policies.hvla.rlt_adapter import HVLAS1Adapter

        return HVLAS1Adapter(policy)

    raise NotImplementedError(
        f"No RLT adapter for policy type {type(policy).__name__!r}. Only HVLA S1 is "
        "supported today; pi0 / pi05 / multi_task_dit adapters land in Phase 4 "
        "(each requires an `encode_observations` hook on the policy). Register the "
        "new adapter in `resolve_rlt_adapter`."
    )


class RLTStrategy(InterventionStrategy):
    """Online RL fine-tuning (TD3 actor-critic) with human intervention.

    Inherits the intervention state machine, input-device wiring, and smooth
    teleop handover from :class:`InterventionStrategy`, and overrides the
    training hooks to store transitions and run gradient updates. The frozen
    base policy is reached through an ``RLTPolicyAdapter`` resolved at setup.
    """

    config: RLTStrategyConfig

    def __init__(self, config: RLTStrategyConfig):
        # Base sets up self._listener / self._pedal_thread / self._events.
        super().__init__(config)
        self._device: torch.device | None = None
        self._adapter = None  # RLTPolicyAdapter

        # RL components (built in setup()).
        self._rlt_config = None  # policies.rlt.config.RLTConfig
        self._token_encoder = None  # RLTokenEncoder (frozen)
        self._agent = None  # TD3Agent
        self._replay = None  # TransactionalReplayBuffer (None in deploy)
        self._recorder = None  # InterventionRecorder (None in deploy)
        self._rlt_state: dict[str, Any] | None = None
        self._resolved_token_ckpt: str | None = None

        # Per-transition SARS carry (the "prev" half of the one-step-delayed
        # transition; mirrors InferenceThread._rlt_prev).
        self._rlt_prev: dict[str, torch.Tensor] | None = None
        # Freshest z_rl exposed to the intervention recorder (mirrors
        # InferenceThread._rlt_latest_z_rl).
        self._rlt_latest_z_rl: torch.Tensor | None = None
        # E-key actor engage toggle (persists across episodes) and the
        # system gate (cleared during intervention / reset).
        self._rlt_user_engaged: bool = True
        self._rlt_system_active: bool = False
        self._rlt_step_count: int = 0
        # Wall-clock of the last gradient-update batch (for the update-rate
        # metric); 0.0 means "no batch yet".
        self._rlt_last_update_time: float = 0.0
        # Latest processed observation + episode duration, set by the control
        # loop and read by the episode-commit hook.
        self._last_obs_processed: dict | None = None
        self._episode_duration_s_actual: float = 0.0

        # Control-loop state.
        # Terminal hotkey (R/LEFT/DOWN) sets this to end the current episode.
        self._episode_end = Event()
        # The current autonomous action chunk (raw action space, [C, A]) and the
        # frame index into it. Recomputed every C frames (D=0 synchronous mode).
        self._chunk_raw: torch.Tensor | None = None
        self._chunk_idx: int = 0
        # Last raw action dict sent to the robot (for the smooth teleop handover).
        self._last_action: dict | None = None

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------

    def setup(self, ctx: RolloutContext) -> None:
        """Initialise adapter, RL machinery, and the input device.

        RLT does **not** start the rollout inference engine: it drives the base
        policy directly through the adapter (so it can read ``z_rl`` and refine
        the chunk), and an RTC engine's background thread would double-run the
        policy. The engine on the context is left untouched.
        """
        self._device = torch.device(ctx.runtime.cfg.device or "cpu")

        # The frozen base policy is on the context; wrap it in its adapter so
        # the RL machinery stays policy-agnostic.
        self._adapter = resolve_rlt_adapter(ctx.policy.policy)

        self._build_rl_machinery(ctx)
        self._setup_rlt_input_device()

        logger.info(
            "RLT strategy ready (adapter=%s, deploy=%s, output_dir=%s, C=%d, "
            "buffer=%d, episode=%d)",
            type(self._adapter).__name__,
            self.config.deploy,
            self.config.output_dir,
            self.config.rl_chunk_length,
            len(self._replay) if self._replay is not None else 0,
            self._rlt_state["episode"],
        )

    def _build_rl_machinery(self, ctx: RolloutContext) -> None:
        """Construct (and optionally resume) the RLT RL components.

        Faithful port of ``hvla/s1_process.py:run_s1`` lines 822-1083, with the
        S1-specific ``policy.config.*`` reads replaced by the policy-agnostic
        :class:`RLTPolicyAdapter` metadata.
        """
        from lerobot.policies.rlt.actor_critic import TD3Agent
        from lerobot.policies.rlt.config import RLTConfig
        from lerobot.policies.rlt.episode import EpisodeLifecycle
        from lerobot.policies.rlt.metrics import set_metrics_path
        from lerobot.policies.rlt.replay_buffer import TransactionalReplayBuffer
        from lerobot.policies.rlt.token import RLTokenEncoder, load_rlt_token_config

        cfg = self.config
        device = self._device
        adapter = self._adapter

        rlt_config = RLTConfig(
            rl_token_dim=adapter.context_dim,
            rl_chunk_length=cfg.rl_chunk_length,
            shared_noise_per_chunk=cfg.shared_noise_per_chunk,
        )

        resolved_token_ckpt = self._resolve_token_checkpoint(cfg, device)

        # Apply the trained checkpoint's architecture manifest BEFORE
        # instantiating the encoder — state_dict load fails otherwise when the
        # checkpoint was trained with a non-default arch (e.g. widened d=2048).
        ckpt_path = Path(resolved_token_ckpt)
        ckpt_dir = ckpt_path if ckpt_path.is_dir() else ckpt_path.parent
        rlt_config = load_rlt_token_config(ckpt_dir, base=rlt_config)

        self._token_encoder = RLTokenEncoder(rlt_config).to(device)
        enc_file = ckpt_path / "encoder.pt" if ckpt_path.is_dir() else ckpt_path
        self._token_encoder.load_state_dict(
            torch.load(str(enc_file), weights_only=True, map_location=device)
        )
        logger.info(
            "RLT: Loaded RL token encoder from %s (enc_layers=%d dec_layers=%d)",
            enc_file,
            rlt_config.token_encoder_layers,
            rlt_config.token_decoder_layers,
        )
        self._token_encoder.eval()
        for p in self._token_encoder.parameters():
            p.requires_grad = False

        action_dim = adapter.action_dim
        state_dim = adapter.state_dim
        self._agent = TD3Agent(rlt_config, state_dim, action_dim, device)

        if cfg.deploy:
            assert cfg.rlt_checkpoint, "deploy=True requires --strategy.rlt_checkpoint (which actor to deploy?)"
            self._replay = None
        else:
            self._replay = TransactionalReplayBuffer(
                rlt_config.replay_capacity,
                rlt_config.rl_token_dim,
                state_dim,
                action_dim,
                cfg.rl_chunk_length,
                device,
            )

        self._rlt_config = rlt_config
        self._rlt_state = {
            "config": rlt_config,
            # 0-indexed episode counter. -1 means "no episode has started yet";
            # incremented at the top of each episode so the first runs as ep0.
            "episode": -1,
            "total_updates": 0,
            "total_transitions": 0,
            "successes": [],
            "lifecycle": EpisodeLifecycle(),
            "output_dir": Path(cfg.output_dir),
            "deploy": cfg.deploy,
        }
        self._rlt_state["output_dir"].mkdir(parents=True, exist_ok=True)
        self._resolved_token_ckpt = resolved_token_ckpt

        set_metrics_path(str(self._rlt_state["output_dir"] / "metrics.json"))

        # Joint names for the intervention recorder's state extraction come
        # from the dataset's ordered action keys (the robot's ``.pos`` motors).
        joint_names = list(ctx.data.ordered_action_keys)
        if not self.config.deploy and self._replay is not None:
            from lerobot.policies.rlt.intervention import InterventionRecorder

            self._recorder = InterventionRecorder(
                replay=self._replay,
                adapter=adapter,
                device=device,
                chunk_length=cfg.rl_chunk_length,
                joint_names=joint_names,
            )
        else:
            self._recorder = None

        self._resume_from_checkpoint(cfg, device)

    def _resolve_token_checkpoint(self, cfg, device) -> str:
        """Resolve the RL-token-encoder checkpoint path (or raise).

        Precedence: (1) explicit ``rl_token_checkpoint``; (2) auto-discover from
        ``training_state.pt['rlt_token_checkpoint']`` inside ``rlt_checkpoint``;
        (3) hard error — refuse to build a random encoder (the actor's input dim
        depends on the encoder's ``rl_token_dim``, so a missing token checkpoint
        would build the actor at the default dim and crash on state_dict load).
        """
        resolved = cfg.rl_token_checkpoint
        if not resolved and cfg.rlt_checkpoint:
            ts_search_dir = Path(cfg.rlt_checkpoint)
            if (
                not (ts_search_dir / "training_state.pt").exists()
                and (ts_search_dir / "latest" / "training_state.pt").exists()
            ):
                ts_search_dir = ts_search_dir / "latest"
            ts_path = ts_search_dir / "training_state.pt"
            if ts_path.exists():
                try:
                    ts = torch.load(str(ts_path), weights_only=False, map_location=device)
                    resolved = ts.get("rlt_token_checkpoint")
                    if resolved:
                        logger.info("RLT: Auto-discovered token encoder from checkpoint: %s", resolved)
                except Exception as e:
                    logger.warning("RLT: Failed to read token path from training state: %s", e)
        if not resolved:
            raise RuntimeError(
                "RLT requires an RL Token Encoder checkpoint. Pass "
                "--strategy.rl_token_checkpoint=<path-to-encoder-dir> "
                "(Phase 1 output, e.g. outputs/rlt_token_v4_4layer_d2048/checkpoint-10000). "
                "Without it the encoder would be random and the actor would be built at "
                "the default rl_token_dim, crashing on state_dict load."
            )
        return resolved

    def _resume_from_checkpoint(self, cfg, device) -> None:
        """Resume actor/critic/optimizer/replay/metrics from ``rlt_checkpoint``.

        Faithful port of ``hvla/s1_process.py:run_s1`` lines 954-1072.
        """
        agent = self._agent
        rlt_state = self._rlt_state

        load_dir = None
        if cfg.rlt_checkpoint:
            load_dir = Path(cfg.rlt_checkpoint)
            if not (load_dir / "actor.pt").exists() and (load_dir / "latest" / "actor.pt").exists():
                load_dir = load_dir / "latest"
        else:
            latest_dir = rlt_state["output_dir"] / "latest"
            if (latest_dir / "actor.pt").exists():
                raise RuntimeError(
                    f"RLT output_dir already contains a trained checkpoint at {latest_dir}. "
                    f"Refusing to start a fresh session here — that would overwrite prior "
                    f"training. To resume, pass --strategy.rlt_checkpoint={rlt_state['output_dir']}. "
                    f"To start a new run, pass a different --strategy.output_dir."
                )

        if not (load_dir and (load_dir / "actor.pt").exists()):
            logger.info(
                "RLT: online RL enabled — C=%d, UTD=%d, beta=%.2f, expl_sigma=%.3f, "
                "target_sigma=%.3f, shared_noise_per_chunk=%s",
                cfg.rl_chunk_length,
                self._rlt_config.utd_ratio,
                self._rlt_config.beta,
                self._rlt_config.exploration_sigma,
                self._rlt_config.target_sigma,
                self._rlt_config.shared_noise_per_chunk,
            )
            return

        logger.info("RLT: Loading checkpoint from %s", load_dir)
        agent.actor.load_state_dict(
            torch.load(str(load_dir / "actor.pt"), weights_only=True, map_location=device)
        )
        if not cfg.deploy:
            if (load_dir / "critic.pt").exists():
                agent.critic.load_state_dict(
                    torch.load(str(load_dir / "critic.pt"), weights_only=True, map_location=device)
                )
            else:
                logger.warning("RLT: critic.pt not found, using random init")
            if (load_dir / "critic_target.pt").exists():
                agent.critic_target.load_state_dict(
                    torch.load(str(load_dir / "critic_target.pt"), weights_only=True, map_location=device)
                )
            else:
                logger.warning("RLT: critic_target.pt not found, using random init")
            if (load_dir / "training_state.pt").exists():
                ts = torch.load(str(load_dir / "training_state.pt"), weights_only=True, map_location=device)
                agent.actor_opt.load_state_dict(ts["actor_opt"])
                agent.critic_opt.load_state_dict(ts["critic_opt"])
                rlt_state["episode"] = ts["episode"]
                rlt_state["total_transitions"] = ts["total_transitions"]
                rlt_state["total_updates"] = ts["total_updates"]
                rlt_state["successes"] = ts["successes"]
                logger.info(
                    "RLT: Loaded training state (ep=%d, updates=%d)",
                    rlt_state["episode"],
                    rlt_state["total_updates"],
                )
            else:
                logger.warning("RLT: training_state.pt not found, starting from ep=0")
            if self._replay is not None and (load_dir / "replay_buffer.pt").exists():
                self._replay.load(str(load_dir / "replay_buffer.pt"))
                logger.info("RLT: Loaded replay buffer (%d transitions)", len(self._replay))
            else:
                logger.warning(
                    "RLT: replay buffer not loaded (replay=%s, file exists=%s)",
                    self._replay is not None,
                    (load_dir / "replay_buffer.pt").exists(),
                )

        self._restore_metrics()
        logger.info(
            "RLT: %s mode — loaded from %s (ep=%d, updates=%d)",
            "DEPLOY" if cfg.deploy else "TRAIN",
            load_dir,
            rlt_state["episode"],
            rlt_state["total_updates"],
        )

    def _restore_metrics(self) -> None:
        """Restore the metrics JSON (with legacy-flat-format promotion)."""
        import json

        from lerobot.policies.rlt.metrics import get_metrics

        metrics_path = str(self._rlt_state["output_dir"] / "metrics.json")
        if not os.path.exists(metrics_path):
            return
        try:
            with open(metrics_path) as f:
                saved = json.load(f)
            series = saved.get("series", {})
            if "episodes" not in series and "episode_successes" in series:
                series = {
                    **series,
                    "episodes": {
                        "successes": series.get("episode_successes", []),
                        "autonomous": series.get("episode_autonomous", []),
                        "timestamps": series.get("episode_timestamps", []),
                        "lengths_s": series.get("episode_lengths_s", []),
                    },
                }
                logger.warning(
                    "RLT: legacy metrics format — restored episode history; per-step "
                    "training series (Q values, critic loss, actor delta) start fresh.",
                )
                saved = {**saved, "series": series}
            get_metrics().restore(saved)
        except Exception as e:
            logger.warning("RLT: Failed to restore metrics: %s", e)

    # ------------------------------------------------------------------
    # Training hooks (override InterventionStrategy no-ops).
    #
    # These carry the RL numerics ported from InferenceThread. They operate on
    # already-computed tensors (z_rl / state / action / ref), so they are
    # policy-agnostic and unit-testable on CPU without hardware.
    # ------------------------------------------------------------------

    @property
    def _rlt_active(self) -> bool:
        """Actor runs only when the operator has engaged it AND the system gate
        is open (cleared during intervention / reset). Mirrors
        ``InferenceThread.rlt_active``."""
        return self._rlt_user_engaged and self._rlt_system_active

    def on_transition(
        self,
        z_rl: torch.Tensor,
        state_norm: torch.Tensor,
        actor_norm: torch.Tensor,
        actor_ref: torch.Tensor,
    ) -> None:
        """Store the one-step-delayed SARS transition for an autonomous chunk.

        Faithful port of ``InferenceThread._rlt_inference_step`` lines 351-410:
        the current step's ``z_rl`` / ``state`` become the ``next_*`` of the
        previously buffered ``_rlt_prev``; ``done`` / ``reward`` come from a
        one-shot ``consume_terminal_for_storage`` (exactly one ``done=True`` per
        episode). All tensors are expected batch-free (``[D]`` / ``[C, A]``).
        """
        from lerobot.policies.rlt.episode import TerminalKind

        if not self._rlt_active:
            self._rlt_prev = None
            return

        prev = self._rlt_prev
        if prev is not None and self._replay is not None:
            terminal = self._rlt_state["lifecycle"].consume_terminal_for_storage()
            done = terminal is not None
            if terminal == TerminalKind.ABORT:
                reward = float(self._rlt_state["config"].abort_reward)
            elif terminal == TerminalKind.SUCCESS:
                reward = 1.0
            else:
                reward = 0.0
            self._replay.add(
                z_rl=prev["z_rl"],
                state=prev["state"],
                action_chunk=prev["action"],
                ref_chunk=prev["ref"],
                reward=reward,
                next_z_rl=z_rl.detach(),
                next_state=state_norm.detach(),
                next_ref_chunk=actor_ref.detach(),
                done=done,
            )
            self._rlt_state["total_transitions"] += 1

        self._rlt_prev = {
            "z_rl": z_rl.detach(),
            "state": state_norm.detach(),
            "action": actor_norm.detach(),
            "ref": actor_ref.detach(),
        }

    def on_intervention_frame(self, human_action_np, obs_processed: dict) -> None:
        """Record one human-teleop frame into the RLT replay buffer.

        Delegates to :class:`InterventionRecorder`, sourcing the freshest
        ``z_rl`` exposed by the autonomous path. Mirrors the intervention-frame
        call site in ``s1_process.py`` (~line 1806).
        """
        if self._recorder is None:
            return
        self._recorder.on_frame(
            human_action_np=human_action_np,
            current_z_rl=self._rlt_latest_z_rl,
            current_obs=obs_processed,
        )

    def on_terminal(self, obs_processed: dict) -> None:
        """Flush a terminal transition for an intervention-rescued episode.

        When the operator marks success/abort while still teleop-driving, the
        autonomous path never runs, so this closes out the reward. Port of
        ``s1_process._rlt_flush_intervention_terminal`` (lines 176-233).
        """
        from lerobot.policies.rlt.episode import TerminalKind

        if self._recorder is None or self._recorder.frames_observed == 0:
            return
        lifecycle = self._rlt_state["lifecycle"]
        terminal = lifecycle.consume_terminal_for_storage()
        if terminal is None:
            return
        terminal_reward = (
            float(self._rlt_state["config"].abort_reward) if terminal == TerminalKind.ABORT else 1.0
        )
        if not self._recorder.flush_terminal(
            reward=terminal_reward,
            current_z_rl=self._rlt_latest_z_rl,
            current_obs=obs_processed,
        ):
            return
        self._rlt_state["total_transitions"] += 1
        last = self._replay.peek_last_pending()
        assert last is not None, "flush_terminal returned True but the buffer has no pending writes"
        assert bool(last["done"]) is True and abs(float(last["reward"]) - terminal_reward) < 1e-6, (
            "flush_terminal returned True but the last pending transition does not reflect the terminal "
            f"(expected done=True, reward={terminal_reward:+.3f})"
        )
        logger.info("RLT: intervention-terminal r=%+.2f flushed (%s)", terminal_reward, terminal.name)

    def background_step(self) -> None:
        """Run one batch of UTD gradient updates when the buffer is warm.

        Port of ``InferenceThread._rlt_gradient_updates`` (lines 613-728):
        ``utd_ratio`` × (2 critic + 1 actor) steps at batch 256, gated on
        ``len(replay) >= 256``. Called in the control loop's idle time.
        """
        if self.config.deploy or self._replay is None or len(self._replay) < 256:
            return
        self._run_gradient_updates()

    def _run_gradient_updates(self) -> None:
        import time as _time

        from lerobot.policies.rlt.metrics import get_metrics

        cfg = self._rlt_state["config"]
        C = cfg.rl_chunk_length
        A_flat = self._replay._action.shape[1]
        A = A_flat // C

        t0 = _time.perf_counter()
        elapsed_since_last = t0 - self._rlt_last_update_time if self._rlt_last_update_time > 0 else 0.0
        self._rlt_last_update_time = t0

        critic_sum = actor_sum = grad_norm_sum = grad_norm_max = q_term_sum = bc_term_sum = 0.0
        n_critic = n_actor = 0

        for _ in range(cfg.utd_ratio):
            for _ in range(2):
                b = self._replay.sample(256)
                cl, gn = self._agent.update_critic(
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
            b = self._replay.sample(256)
            al, q_term, bc_term = self._agent.update_actor(
                b["z_rl"],
                b["state"],
                b["ref"].reshape(-1, C, A),
            )
            actor_sum += al
            q_term_sum += q_term
            bc_term_sum += bc_term
            n_actor += 1
            self._rlt_state["total_updates"] += 1

        avg_c = critic_sum / n_critic if n_critic else 0
        avg_a = actor_sum / n_actor if n_actor else 0
        avg_q_term = q_term_sum / n_actor if n_actor else 0
        avg_bc_term = bc_term_sum / n_actor if n_actor else 0

        with torch.no_grad():
            b = self._replay.sample(min(256, len(self._replay)))
            qs = self._agent.critic.min_q(b["z_rl"], b["state"], b["action"].reshape(-1, C, A))
            q_mean, q_min, q_max = qs.mean().item(), qs.min().item(), qs.max().item()

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

    def on_episode_begin(self, ctx: RolloutContext) -> None:
        """Advance the episode counter and (re)open the collection gate."""
        rlt_state = self._rlt_state
        rlt_state["episode"] += 1
        self._rlt_system_active = True
        self._rlt_user_engaged = self.config.start_engaged
        self._rlt_prev = None
        rlt_state["lifecycle"].begin(buffer_size=len(self._replay) if self._replay is not None else 0)
        in_warmup = rlt_state["config"].is_warmup(rlt_state["episode"])
        logger.info(
            "RLT: episode %d start — actor %s%s",
            rlt_state["episode"],
            "engaged" if self.config.start_engaged else "disengaged (press E to engage)",
            " [warmup]" if in_warmup else "",
        )

    def on_episode_commit(self, ctx: RolloutContext) -> None:
        """Episode-end bookkeeping: terminal flush, metrics, buffer commit, save.

        Port of the ``else`` (non-ignored) branch of ``s1_process`` lines
        2128-2250.
        """
        from lerobot.policies.rlt.episode import TerminalKind
        from lerobot.policies.rlt.metrics import get_metrics, save_metrics_to_file

        rlt_state = self._rlt_state
        lifecycle = rlt_state["lifecycle"]
        self._rlt_system_active = False

        # Close out the terminal transition (done=True). Two paths, mirroring
        # s1_process: an intervention-rescued episode flushes through the
        # recorder; a pure-autonomous episode ended by a terminal hotkey flushes
        # the pending (autonomous) transition. Unlike the source's continuous
        # inference thread — which consumed the terminal on its next cycle — the
        # synchronous loop broke immediately, so we consume it here. Either way
        # the terminal MUST be consumed or the next ``lifecycle.begin`` asserts.
        if self._recorder is not None and self._recorder.frames_observed > 0:
            rlt_state["total_transitions"] += self._recorder.chunks_stored
            self._recorder.log_summary()
            self.on_terminal(self._last_obs_processed or {})
        else:
            self._flush_autonomous_terminal(ctx)
        if self._recorder is not None:
            self._recorder.reset()
        self._rlt_prev = None

        terminal = lifecycle.peek_terminal()
        success = terminal == TerminalKind.SUCCESS
        had_intervention = lifecycle.had_intervention
        ep_duration = self._episode_duration_s_actual

        rlt_state["successes"].append(success)
        get_metrics().record_episode(
            rlt_state["episode"],
            success,
            autonomous=not had_intervention,
            duration_s=ep_duration,
        )
        save_metrics_to_file()

        if self._replay is not None:
            self._replay.commit()
        lifecycle.end_episode()

        self._save_checkpoint()

    def on_episode_discard(self, ctx: RolloutContext) -> None:
        """IGNORE path (operator pressed DOWN): drop staged transitions & roll back.

        Port of the ``lifecycle.is_ignored()`` branch of ``s1_process`` lines
        2100-2124.
        """
        rlt_state = self._rlt_state
        self._rlt_system_active = False
        self._rlt_prev = None
        if self._recorder is not None:
            self._recorder.reset()
        dropped = self._replay.discard() if self._replay is not None else 0
        rlt_state["total_transitions"] -= dropped
        ep_ignored = rlt_state["episode"]
        rlt_state["episode"] -= 1
        rlt_state["lifecycle"].end_episode()
        logger.info(
            "RLT ep%d: IGNORED — discarded %d transitions; counter rolled back to %d",
            ep_ignored,
            dropped,
            rlt_state["episode"],
        )

    def _flush_autonomous_terminal(self, ctx: RolloutContext) -> None:
        """Write the terminal (``done=True``) transition for a pure-autonomous episode.

        The synchronous loop breaks the instant a terminal hotkey fires, so the
        pending autonomous transition (``_rlt_prev``) is closed out here by
        re-encoding the current observation as its ``next_*`` state — the same
        pairing the continuous inference thread produced in ``s1_process``. A
        no-op when there is no pending transition, no terminal signalled, or the
        terminal was already consumed by an in-loop ``on_transition`` (the race
        the source tolerated); consuming here also satisfies the next
        ``lifecycle.begin`` terminal-consumed assert.
        """
        from lerobot.policies.rlt.episode import TerminalKind

        lifecycle = self._rlt_state["lifecycle"]
        prev = self._rlt_prev
        if prev is None or self._replay is None or lifecycle.peek_terminal() is None:
            return
        terminal = lifecycle.consume_terminal_for_storage()
        if terminal is None:
            # Already consumed in-loop (a late-arriving inference tick) — the
            # done=True transition is already in the buffer.
            return
        reward = float(self._rlt_state["config"].abort_reward) if terminal == TerminalKind.ABORT else 1.0

        robot_type = getattr(
            ctx.hardware.robot_wrapper.inner, "robot_type", getattr(ctx.hardware.robot_wrapper.inner, "name", "")
        )
        task_str = ctx.runtime.cfg.dataset.single_task if ctx.runtime.cfg.dataset else ctx.runtime.cfg.task
        obs = ctx.hardware.robot_wrapper.get_observation()
        obs_processed = ctx.processors.robot_observation_processor(obs)
        batch = self._build_policy_batch(ctx, obs_processed, task_str, robot_type)
        autocast_ctx = (
            torch.autocast(device_type=self._device.type, dtype=torch.bfloat16)
            if self._device.type == "cuda"
            else nullcontext()
        )
        with torch.no_grad(), autocast_ctx:
            context = self._adapter.encode_context(batch)
            next_z_rl = self._token_encoder(context.float()).detach().squeeze(0).float()
        next_state = self._adapter.normalize_state(batch["observation.state"].float()).squeeze(0)

        self._replay.add(
            z_rl=prev["z_rl"],
            state=prev["state"],
            action_chunk=prev["action"],
            ref_chunk=prev["ref"],
            reward=reward,
            next_z_rl=next_z_rl,
            next_state=next_state,
            next_ref_chunk=prev["ref"],
            done=True,
        )
        self._rlt_state["total_transitions"] += 1
        logger.info("RLT: autonomous-terminal r=%+.2f flushed (%s)", reward, terminal.name)

    # ------------------------------------------------------------------
    # Checkpoint save (shared by episode-commit and teardown)
    # ------------------------------------------------------------------

    def _save_checkpoint(self, snapshot_every_10: bool = True) -> None:
        """Atomically save actor/critic/target/optimizer/replay to ``latest/``.

        Port of ``s1_process`` lines 2202-2250. Training mode only.
        """
        if self._rlt_state.get("deploy"):
            return
        try:
            save_dir = self._rlt_state["output_dir"] / "latest"
            save_dir.mkdir(parents=True, exist_ok=True)
            _atomic_torch_save(self._agent.actor.state_dict(), save_dir / "actor.pt")
            _atomic_torch_save(self._agent.critic.state_dict(), save_dir / "critic.pt")
            _atomic_torch_save(self._agent.critic_target.state_dict(), save_dir / "critic_target.pt")
            _atomic_torch_save(
                {
                    "actor_opt": self._agent.actor_opt.state_dict(),
                    "critic_opt": self._agent.critic_opt.state_dict(),
                    "episode": self._rlt_state["episode"],
                    "total_transitions": self._rlt_state["total_transitions"],
                    "total_updates": self._rlt_state["total_updates"],
                    "successes": self._rlt_state["successes"],
                    "rlt_token_checkpoint": self._resolved_token_ckpt,
                },
                save_dir / "training_state.pt",
            )
            if self._replay is not None:
                self._replay.save(str(save_dir / "replay_buffer.pt"))
            logger.info(
                "RLT: Saved checkpoint → %s (ep=%d, buf=%d, updates=%d)",
                save_dir,
                self._rlt_state["episode"],
                len(self._replay) if self._replay is not None else 0,
                self._rlt_state["total_updates"],
            )
            ep = self._rlt_state["episode"]
            if snapshot_every_10 and ep > 0 and (ep + 1) % 10 == 0:
                import shutil

                snap_dir = self._rlt_state["output_dir"] / f"ep_{ep + 1}"
                try:
                    if snap_dir.exists():
                        # safe-destruct: RLT checkpoint snapshot we just wrote — refresh
                        shutil.rmtree(snap_dir)
                    shutil.copytree(save_dir, snap_dir)
                    logger.info("RLT: Snapshot → %s (permanent, for rollback)", snap_dir)
                except Exception as e:
                    logger.error("RLT: Snapshot failed: %s", e)
        except Exception as e:
            logger.error("RLT: Failed to save checkpoint: %s", e)

    # ------------------------------------------------------------------
    # Input device (RLT keyboard with terminal hotkeys)
    # ------------------------------------------------------------------

    def _setup_rlt_input_device(self) -> None:
        """Start the input listener, extending the intervention controls with
        RLT's terminal-outcome hotkeys.

        Keyboard controls (mirrors ``s1_process.py`` bindings):
            pause_resume (space)  AUTONOMOUS <-> PAUSED
            correction   (tab)    PAUSED <-> CORRECTING (human teleop)
            r                     SUCCESS  — reward +1, ends the episode
            left                  ABORT    — reward ``abort_reward``, ends episode
            down                  IGNORE   — discard episode, roll back counter
            e                     toggle the RL actor engage/disengage
            esc                   stop the session

        Foot-pedal input keeps only the intervention toggles (a pedal can't
        express success/abort/ignore); RLT is keyboard-first.
        """
        if self.config.input_device != "keyboard":
            self._setup_input_device(self.config.input_device, self.config.keyboard, self.config.pedal)
            logger.warning(
                "RLT with input_device=pedal: terminal hotkeys (success/abort/ignore) are "
                "keyboard-only; the pedal drives the intervention toggle only."
            )
            return

        from lerobot.policies.rlt.episode import TerminalKind
        from lerobot.utils.keyboard_input import create_key_listener

        events = self._events
        kb = self.config.keyboard
        lifecycle = self._rlt_state["lifecycle"]
        key_to_event = {kb.pause_resume: "pause_resume", kb.correction: "correction"}

        def dispatch(name: str) -> None:
            if name == "esc":
                logger.info("RLT: stop requested")
                events.stop_recording.set()
                return
            if name in key_to_event:
                events.request_transition(key_to_event[name])
                return
            if name == "r":
                lifecycle.signal_terminal(TerminalKind.SUCCESS)
                self._episode_end.set()
                logger.info("RLT: SUCCESS — reward +1, ending episode")
            elif name == "left":
                lifecycle.signal_terminal(TerminalKind.ABORT)
                self._episode_end.set()
                logger.info("RLT: ABORT — reward %.2f, ending episode", self._rlt_config.abort_reward)
            elif name == "down":
                lifecycle.signal_ignore()
                self._episode_end.set()
                logger.info("RLT: IGNORE — episode will be discarded")
            elif name == "e":
                self._rlt_user_engaged = not self._rlt_user_engaged
                logger.info("RLT: RL actor %s (E key)", "ENGAGED" if self._rlt_user_engaged else "DISENGAGED")

        self._listener = create_key_listener(
            dispatch,
            controls_help=(
                f"pause_resume='{kb.pause_resume}', correction='{kb.correction}', "
                "r=success, left=abort, down=ignore, e=toggle-actor, esc=stop"
            ),
        )

    # ------------------------------------------------------------------
    # Run
    # ------------------------------------------------------------------

    def run(self, ctx: RolloutContext) -> None:
        """Run the RLT online loop (autonomous + intervention + training).

        Outer loop = episodes (delimited by the terminal hotkeys); inner loop =
        the shared intervention state machine. In AUTONOMOUS the policy drives
        through the adapter with actor refinement and transitions are stored; in
        CORRECTING the human teleoperates and frames are recorded; gradient
        updates run in idle time via :meth:`background_step`.
        """
        cfg = ctx.runtime.cfg
        events = self._events
        control_interval = 1.0 / cfg.fps

        events.reset()
        session_start = time.perf_counter()

        while not events.stop_recording.is_set() and not ctx.runtime.shutdown_event.is_set():
            if cfg.duration > 0 and (time.perf_counter() - session_start) >= cfg.duration:
                logger.info("RLT: duration limit reached (%.0fs)", cfg.duration)
                break
            self._run_episode(ctx, control_interval)

        logger.info("RLT: session loop ended")

    def _run_episode(self, ctx: RolloutContext, control_interval: float) -> None:
        """Run a single RLT episode until a terminal hotkey / stop / shutdown."""
        cfg = ctx.runtime.cfg
        robot = ctx.hardware.robot_wrapper
        teleop = ctx.hardware.teleop
        events = self._events
        task_str = cfg.dataset.single_task if cfg.dataset else cfg.task
        robot_type = getattr(robot.inner, "robot_type", getattr(robot.inner, "name", ""))
        ordered_keys = ctx.data.ordered_action_keys

        self.on_episode_begin(ctx)
        self._episode_end.clear()
        self._chunk_raw = None
        self._chunk_idx = 0
        self._last_action = None
        ctx.policy.policy.reset()
        self._adapter.reset()
        ep_start = time.perf_counter()

        while (
            not self._episode_end.is_set()
            and not events.stop_recording.is_set()
            and not ctx.runtime.shutdown_event.is_set()
        ):
            loop_start = time.perf_counter()

            transition = events.consume_transition()
            if transition is not None:
                self._apply_rlt_transition(transition[0], transition[1], ctx)

            phase = events.phase
            obs = robot.get_observation()
            obs_processed = ctx.processors.robot_observation_processor(obs)
            self._last_obs_processed = obs_processed

            if phase == InterventionPhase.CORRECTING and teleop is not None:
                teleop_action = teleop.get_action()
                processed_teleop = ctx.processors.teleop_action_processor((teleop_action, obs))
                robot_action = ctx.processors.robot_action_processor((processed_teleop, obs))
                robot.send_action(robot_action)
                self._last_action = robot_action
                human_np = np.array(
                    [float(processed_teleop[k]) for k in ordered_keys], dtype=np.float32
                )
                self.on_intervention_frame(human_np, obs_processed)
            elif phase == InterventionPhase.PAUSED:
                if self._last_action is not None:
                    robot.send_action(self._last_action)
            else:  # AUTONOMOUS
                self._rlt_autonomous_tick(ctx, obs, obs_processed, task_str, robot_type, ordered_keys)
                self.background_step()

            dt = time.perf_counter() - loop_start
            if (sleep_t := control_interval - dt) > 0:
                precise_sleep(sleep_t)

        self._episode_duration_s_actual = time.perf_counter() - ep_start
        if self._rlt_state["lifecycle"].is_ignored():
            self.on_episode_discard(ctx)
        else:
            self.on_episode_commit(ctx)

    def _rlt_autonomous_tick(self, ctx, obs, obs_processed, task_str, robot_type, ordered_keys) -> None:
        """Execute one autonomous control frame, recomputing the chunk every C frames."""
        C = self.config.rl_chunk_length
        if self._chunk_raw is None or self._chunk_idx >= C:
            self._rlt_infer_chunk(ctx, obs_processed, task_str, robot_type)
            self._chunk_idx = 0

        action_raw = self._chunk_raw[self._chunk_idx]  # [A]
        action_dict = {k: float(action_raw[i].item()) for i, k in enumerate(ordered_keys)}
        processed = ctx.processors.robot_action_processor((action_dict, obs))
        ctx.hardware.robot_wrapper.send_action(processed)
        self._last_action = processed
        self._chunk_idx += 1

    def _build_policy_batch(self, ctx, obs_processed, task_str, robot_type) -> dict:
        """Build the raw (un-normalized) policy batch the adapter expects.

        Mirrors ``obs_to_s1_batch`` + the flow-matching identity preprocessor:
        tensors on device, images in ``[0, 1]`` CHW, a batch dim, and ``task``.
        The adapter owns any policy-specific normalization internally.
        """
        obs_frame = build_dataset_frame(ctx.data.dataset_features, obs_processed, prefix=OBS_STR)
        batch = prepare_observation_for_inference(obs_frame, self._device, task_str, robot_type)
        batch["task"] = [task_str]
        return batch

    def _rlt_infer_chunk(self, ctx, obs_processed, task_str, robot_type) -> None:
        """One policy inference + actor refinement; fills ``self._chunk_raw``.

        Faithful port of ``InferenceThread._loop`` (lines 863-942) +
        ``_rlt_inference_step`` (lines 247-436) for the synchronous ``D=0`` case:
        encode_context → z_rl, reference chunk, actor refinement (fp32,
        autocast-off), transition storage. The executed chunk is kept in raw
        action space so :meth:`_rlt_autonomous_tick` can send frames directly.
        """
        from lerobot.policies.rlt.metrics import get_metrics, save_metrics_to_file

        cfg = self._rlt_state["config"]
        C = cfg.rl_chunk_length
        device = self._device
        adapter = self._adapter

        batch = self._build_policy_batch(ctx, obs_processed, task_str, robot_type)

        autocast_ctx = (
            torch.autocast(device_type=device.type, dtype=torch.bfloat16)
            if device.type == "cuda"
            else nullcontext()
        )
        with torch.no_grad(), autocast_ctx:
            context = adapter.encode_context(batch)
            z_rl = self._token_encoder(context.float()).detach()  # [1, D]
            ref_raw = adapter.predict_reference_chunk(batch, context=context)  # [1, T, A] raw

        # State in actor-input space (z-scored), full fp32.
        state_t = batch["observation.state"].float()  # [1, S]
        state_norm = adapter.normalize_state(state_t)
        ref_norm = adapter.normalize_action(ref_raw.float())  # [1, T, A]
        actor_ref = ref_norm[:, 0:C, :]  # D=0: the actor refines the first C frames

        # Publish the freshest z_rl for the intervention recorder (unconditional,
        # before the engage gate — the recorder needs it while the actor is paused).
        self._rlt_latest_z_rl = z_rl.squeeze(0).float().detach()
        self._rlt_step_count += 1

        is_deploy = self._rlt_state.get("deploy", False)
        is_warmup = cfg.is_warmup(self._rlt_state["episode"])

        if self._rlt_active:
            if is_warmup and not is_deploy:
                # Warmup: execute the S1 reference; the actor is not called but
                # the transition is still recorded (action == ref).
                actor_norm = actor_ref
            else:
                # Actor trains in fp32 with no autocast — match that dtype here or
                # bf16 rounding adds a per-joint ~1e-2 bias. (test_rlt_parity guard.)
                with torch.autocast(device_type=device.type, enabled=False):
                    actor_norm = self._agent.actor(
                        z_rl.float(),
                        state_norm.float(),
                        actor_ref.float(),
                        deterministic=is_deploy,
                    )
            self.on_transition(
                z_rl.squeeze(0).float(),
                state_norm.squeeze(0),
                actor_norm.squeeze(0),
                actor_ref.squeeze(0),
            )
            executed_raw = adapter.denormalize_action(actor_norm.squeeze(0))  # [C, A]
            actor_delta = (actor_norm - actor_ref).abs().mean().item()
            mode = "DEPLOY" if is_deploy else ("WARMUP" if is_warmup else "RL")
        else:
            # Actor disengaged (E key): execute the raw S1 reference, store nothing.
            executed_raw = ref_raw.squeeze(0)[0:C]
            self._rlt_prev = None
            actor_delta = 0.0
            mode = "POLICY"

        self._chunk_raw = executed_raw.detach().to("cpu")

        get_metrics().record_inference(
            step=self._rlt_step_count,
            delta=actor_delta,
            buffer_size=len(self._replay) if self._replay is not None else 0,
            total_updates=self._rlt_state["total_updates"],
            mode=mode,
        )
        if self._rlt_step_count % 100 == 0:
            logger.info(
                "RLT step %d [%s] | delta=%.3f | buf=%d | updates=%d",
                self._rlt_step_count,
                mode,
                actor_delta,
                len(self._replay) if self._replay is not None else 0,
                self._rlt_state["total_updates"],
            )
            save_metrics_to_file()

    def _apply_rlt_transition(self, old: InterventionPhase, new: InterventionPhase, ctx) -> None:
        """RLT variant of the intervention handover: smooth teleop takeover plus
        the RLT system-gate / recorder bookkeeping.

        Combines the teleop handover of :meth:`InterventionStrategy._apply_transition`
        (RLT does not use the rollout inference engine, so the engine's
        pause/reset/resume calls are replaced by the RLT collection gate) with
        the intervention side-effects from ``s1_process.py`` (lines 1633-1882).
        """
        teleop = ctx.hardware.teleop
        robot = ctx.hardware.robot_wrapper
        logger.info("RLT phase: %s -> %s", old.value, new.value)

        if old == InterventionPhase.AUTONOMOUS and new == InterventionPhase.PAUSED:
            # Stop the actor + collection; the robot holds its last commanded pose.
            self._rlt_system_active = False
            self._rlt_prev = None
            self._chunk_raw = None
            if teleop_supports_feedback(teleop) and self._last_action is not None:
                logger.info("RLT: smooth handover — moving leader to follower position")
                teleop_smooth_move_to(teleop, self._last_action)

        elif old == InterventionPhase.PAUSED and new == InterventionPhase.CORRECTING:
            self._rlt_state["lifecycle"].mark_intervention()
            if self._recorder is not None:
                self._recorder.reset()
            if not teleop_supports_feedback(teleop) and self._last_action is not None:
                obs = robot.get_observation()
                teleop_action = teleop.get_action()
                processed = ctx.processors.teleop_action_processor((teleop_action, obs))
                target = ctx.processors.robot_action_processor((processed, obs))
                follower_smooth_move_to(robot, self._last_action, target)
            if teleop_supports_feedback(teleop):
                teleop.disable_torque()

        elif old == InterventionPhase.CORRECTING and new == InterventionPhase.PAUSED:
            if self._recorder is not None and self._recorder.frames_observed > 0:
                self._rlt_state["total_transitions"] += self._recorder.chunks_stored
                self._recorder.log_summary()
                self._recorder.reset()
            if teleop_supports_feedback(teleop):
                teleop.enable_torque()

        elif new == InterventionPhase.AUTONOMOUS:
            # Resume autonomous collection; re-infer from a fresh chunk.
            self._rlt_system_active = True
            self._rlt_prev = None
            self._chunk_raw = None
            ctx.policy.policy.reset()
            self._adapter.reset()
            if teleop_supports_feedback(teleop):
                teleop.disable_torque()

    def teardown(self, ctx: RolloutContext) -> None:
        """Stop listeners, save the final checkpoint, and disconnect hardware."""
        from lerobot.policies.rlt.metrics import save_metrics_to_file

        logger.info("Stopping RLT session")
        self._teardown_input_device()

        if not self.config.deploy and self._agent is not None:
            self._save_checkpoint(snapshot_every_10=False)
            save_metrics_to_file()
            logger.info("RLT: Final checkpoint + metrics saved")

        self._teardown_hardware(
            ctx.hardware,
            return_to_initial_position=ctx.runtime.cfg.return_to_initial_position,
        )
        logger.info("RLT strategy teardown complete")

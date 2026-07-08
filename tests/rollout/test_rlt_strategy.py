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

"""CPU tests for the RLT rollout strategy (Phase 2, step 3).

Covers the machinery construction (``_build_rl_machinery``), checkpoint
resume, and the RL training hooks (``on_transition`` / ``background_step`` /
``on_terminal`` / episode lifecycle). These operate on already-computed
tensors so they run on CPU with no hardware. The autonomous chunk-execution
control loop (``run``) is the on-robot-validation surface and is not exercised
here — ``tests/hvla/test_rlt_parity.py`` guards its numerics.
"""

from __future__ import annotations

from threading import Event
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from lerobot.policies.rlt.config import RLTConfig
from lerobot.policies.rlt.episode import TerminalKind
from lerobot.policies.rlt.policy_adapter import RLTPolicyAdapter
from lerobot.policies.rlt.token import RLTokenEncoder, save_rlt_token_config
from lerobot.rollout.configs import RLTStrategyConfig
from lerobot.rollout.strategies.rlt import RLTStrategy
from tests.utils import require_cuda

# Small, self-consistent dims so the whole stack fits on CPU in milliseconds.
CTX_DIM = 16
STATE_DIM = 6
ACTION_DIM = 4
CHUNK = 3


class _StubAdapter(RLTPolicyAdapter):
    """Minimal RLTPolicyAdapter with identity normalization for tests.

    ``encode_context`` / ``predict_reference_chunk`` return correctly-shaped
    tensors on ``device`` so the same stub drives both the CPU orchestration
    tests and the GPU numeric tests.
    """

    def __init__(self, device: str = "cpu"):
        self._device = torch.device(device)

    @property
    def context_dim(self) -> int:
        return CTX_DIM

    @property
    def action_dim(self) -> int:
        return ACTION_DIM

    @property
    def state_dim(self) -> int:
        return STATE_DIM

    @property
    def chunk_size(self) -> int:
        return CHUNK

    def encode_context(self, batch: dict) -> torch.Tensor:
        return torch.zeros(1, 2, CTX_DIM, device=self._device)

    def predict_reference_chunk(self, batch: dict, context=None, **kwargs) -> torch.Tensor:
        return torch.zeros(1, CHUNK, ACTION_DIM, device=self._device)

    def normalize_state(self, state: torch.Tensor) -> torch.Tensor:
        return state

    def normalize_action(self, action: torch.Tensor) -> torch.Tensor:
        return action

    def denormalize_action(self, action: torch.Tensor) -> torch.Tensor:
        return action


def _make_token_checkpoint(dir_path) -> str:
    """Write a symmetric (context_dim == rl_token_dim) token encoder + manifest."""
    cfg = RLTConfig(rl_token_dim=CTX_DIM, context_dim=CTX_DIM)
    enc = RLTokenEncoder(cfg)
    torch.save(enc.state_dict(), dir_path / "encoder.pt")
    save_rlt_token_config(dir_path, cfg)
    return str(dir_path)


def _make_config(tmp_path, **overrides) -> RLTStrategyConfig:
    token_dir = tmp_path / "token"
    token_dir.mkdir(exist_ok=True)
    _make_token_checkpoint(token_dir)
    kwargs = {
        "rl_token_checkpoint": str(token_dir),
        "output_dir": str(tmp_path / "rlt_out"),
        "rl_chunk_length": CHUNK,
        "deploy": False,
    }
    kwargs.update(overrides)
    return RLTStrategyConfig(**kwargs)


def _make_ctx(action_keys):
    """A RolloutContext stub exposing only ``data.ordered_action_keys``."""
    return SimpleNamespace(data=SimpleNamespace(ordered_action_keys=list(action_keys)))


def _build_strategy(config, device: str = "cpu") -> RLTStrategy:
    """Construct an RLTStrategy and its machinery without touching hardware."""
    strat = RLTStrategy(config)
    strat._device = torch.device(device)
    strat._adapter = _StubAdapter(device)
    joint_names = [f"j{i}.pos" for i in range(STATE_DIM)]
    strat._build_rl_machinery(_make_ctx(joint_names))
    return strat


# ---------------------------------------------------------------------------
# Machinery construction
# ---------------------------------------------------------------------------


def test_build_machinery_constructs_components(tmp_path):
    strat = _build_strategy(_make_config(tmp_path))
    assert strat._agent is not None
    assert strat._token_encoder is not None
    assert strat._replay is not None
    assert strat._recorder is not None
    # Fresh run: episode counter at -1 (incremented to 0 at first episode start).
    assert strat._rlt_state["episode"] == -1
    assert strat._rlt_state["total_updates"] == 0
    # Token encoder is frozen.
    assert all(not p.requires_grad for p in strat._token_encoder.parameters())
    # Actor input dim matches the loaded token dim + adapter dims.
    expected_in = CTX_DIM + STATE_DIM + CHUNK * ACTION_DIM
    assert strat._agent.actor.mlp[0].in_features == expected_in


def test_missing_token_checkpoint_raises(tmp_path):
    config = _make_config(tmp_path, rl_token_checkpoint=None, rlt_checkpoint=None)
    strat = RLTStrategy(config)
    strat._device = torch.device("cpu")
    strat._adapter = _StubAdapter()
    with pytest.raises(RuntimeError, match="RL Token Encoder checkpoint"):
        strat._build_rl_machinery(_make_ctx([f"j{i}.pos" for i in range(STATE_DIM)]))


def test_deploy_mode_has_no_replay_or_recorder(tmp_path):
    # deploy=True requires an rlt_checkpoint; point it at a run dir we first
    # populate by saving from a training strategy.
    train = _build_strategy(_make_config(tmp_path))
    train._rlt_state["episode"] = 0
    train._save_checkpoint(snapshot_every_10=False)

    config = _make_config(
        tmp_path,
        deploy=True,
        rlt_checkpoint=str(tmp_path / "rlt_out"),
        output_dir=str(tmp_path / "deploy_out"),
    )
    strat = _build_strategy(config)
    assert strat._replay is None
    assert strat._recorder is None
    assert strat._agent is not None


# ---------------------------------------------------------------------------
# Checkpoint save + resume
# ---------------------------------------------------------------------------


def test_save_then_resume_restores_state(tmp_path):
    strat = _build_strategy(_make_config(tmp_path))
    strat._rlt_state["episode"] = 4
    strat._rlt_state["total_updates"] = 123
    strat._rlt_state["total_transitions"] = 45
    strat._rlt_state["successes"] = [True, False, True]
    strat._save_checkpoint(snapshot_every_10=False)

    # New strategy resuming from the same output dir.
    resumed = RLTStrategy(
        _make_config(tmp_path, rlt_checkpoint=str(tmp_path / "rlt_out"))
    )
    resumed._device = torch.device("cpu")
    resumed._adapter = _StubAdapter()
    resumed._build_rl_machinery(_make_ctx([f"j{i}.pos" for i in range(STATE_DIM)]))

    assert resumed._rlt_state["episode"] == 4
    assert resumed._rlt_state["total_updates"] == 123
    assert resumed._rlt_state["total_transitions"] == 45
    assert resumed._rlt_state["successes"] == [True, False, True]


def test_fresh_run_into_populated_output_dir_raises(tmp_path):
    strat = _build_strategy(_make_config(tmp_path))
    strat._rlt_state["episode"] = 0
    strat._save_checkpoint(snapshot_every_10=False)

    # A second fresh launch (no rlt_checkpoint) into the same output_dir must
    # refuse to clobber the existing latest/.
    config = _make_config(tmp_path, output_dir=str(tmp_path / "rlt_out"))
    strat2 = RLTStrategy(config)
    strat2._device = torch.device("cpu")
    strat2._adapter = _StubAdapter()
    with pytest.raises(RuntimeError, match="already contains a trained checkpoint"):
        strat2._build_rl_machinery(_make_ctx([f"j{i}.pos" for i in range(STATE_DIM)]))


# ---------------------------------------------------------------------------
# Training hooks
# ---------------------------------------------------------------------------


def _z(device="cpu"):
    return torch.randn(CTX_DIM, device=device)


def _s(device="cpu"):
    return torch.randn(STATE_DIM, device=device)


def _chunk(device="cpu"):
    return torch.randn(CHUNK, ACTION_DIM, device=device)


def test_on_transition_is_one_step_delayed(tmp_path):
    strat = _build_strategy(_make_config(tmp_path))
    strat._rlt_user_engaged = True
    strat._rlt_system_active = True
    strat._rlt_state["lifecycle"].begin(buffer_size=0)

    # First call only seeds _rlt_prev (no write).
    strat.on_transition(_z(), _s(), _chunk(), _chunk())
    assert strat._replay.pending_size == 0
    assert strat._rlt_prev is not None

    # Second call writes the (prev -> current) transition.
    strat.on_transition(_z(), _s(), _chunk(), _chunk())
    assert strat._replay.pending_size == 1
    last = strat._replay.peek_last_pending()
    assert bool(last["done"]) is False
    assert last["reward"] == 0.0


def test_on_transition_terminal_reward_is_one_shot(tmp_path):
    strat = _build_strategy(_make_config(tmp_path))
    strat._rlt_user_engaged = True
    strat._rlt_system_active = True
    lifecycle = strat._rlt_state["lifecycle"]
    lifecycle.begin(buffer_size=0)

    strat.on_transition(_z(), _s(), _chunk(), _chunk())  # seed
    lifecycle.signal_terminal(TerminalKind.SUCCESS)
    strat.on_transition(_z(), _s(), _chunk(), _chunk())  # consumes SUCCESS
    strat.on_transition(_z(), _s(), _chunk(), _chunk())  # must NOT re-fire terminal

    pending = strat._replay._pending
    dones = [bool(p["done"]) for p in pending]
    assert dones.count(True) == 1, f"expected exactly one done=True, got {dones}"
    term_reward = next(p["reward"] for p in pending if p["done"])
    assert term_reward == 1.0


def test_on_transition_skipped_when_actor_disengaged(tmp_path):
    strat = _build_strategy(_make_config(tmp_path))
    strat._rlt_user_engaged = False  # E-key disengaged
    strat._rlt_system_active = True
    strat._rlt_state["lifecycle"].begin(buffer_size=0)
    strat.on_transition(_z(), _s(), _chunk(), _chunk())
    strat.on_transition(_z(), _s(), _chunk(), _chunk())
    assert strat._replay.pending_size == 0
    assert strat._rlt_prev is None


def test_background_step_runs_updates_when_warm(tmp_path):
    strat = _build_strategy(_make_config(tmp_path))
    cfg = strat._rlt_state["config"]
    # Fill committed buffer past the 256 warmup gate.
    for _ in range(300):
        strat._replay.add(
            z_rl=_z(), state=_s(), action_chunk=_chunk(), ref_chunk=_chunk(),
            reward=0.0, next_z_rl=_z(), next_state=_s(), next_ref_chunk=_chunk(), done=False,
        )
    strat._replay.commit()
    assert len(strat._replay) >= 256

    before = strat._rlt_state["total_updates"]
    strat.background_step()
    # utd_ratio actor updates per call.
    assert strat._rlt_state["total_updates"] == before + cfg.utd_ratio


def test_background_step_noop_when_cold(tmp_path):
    strat = _build_strategy(_make_config(tmp_path))
    strat.background_step()
    assert strat._rlt_state["total_updates"] == 0


def test_episode_discard_rolls_back(tmp_path):
    strat = _build_strategy(_make_config(tmp_path))
    strat.on_episode_begin(ctx=None)
    assert strat._rlt_state["episode"] == 0
    strat._rlt_user_engaged = True
    strat._rlt_system_active = True
    # Stage a couple of transitions this episode.
    strat.on_transition(_z(), _s(), _chunk(), _chunk())
    strat.on_transition(_z(), _s(), _chunk(), _chunk())
    assert strat._replay.pending_size == 1

    strat.on_episode_discard(ctx=None)
    # Counter rolled back and staged writes dropped.
    assert strat._rlt_state["episode"] == -1
    assert strat._replay.pending_size == 0
    assert len(strat._replay) == 0


def test_intervention_frame_records_human_chunk(tmp_path):
    strat = _build_strategy(_make_config(tmp_path))
    strat.on_episode_begin(ctx=None)
    # Simulate a fresh z_rl available from the autonomous path.
    strat._rlt_latest_z_rl = _z()
    strat._recorder.reset()
    obs = {f"j{i}.pos": float(i) for i in range(STATE_DIM)}
    # Feed enough frames to complete two C-frame windows (first seeds, second writes).
    for _ in range(2 * CHUNK):
        strat.on_intervention_frame(np.zeros(ACTION_DIM, dtype=np.float32), obs)
    assert strat._recorder.frames_observed == 2 * CHUNK
    assert strat._recorder.chunks_stored == 1


# ---------------------------------------------------------------------------
# Control loop (mock robot / no teleop) — orchestration only, not on-robot numerics
# ---------------------------------------------------------------------------


class _MockRobot:
    """Minimal robot_wrapper stub. Fires a terminal after ``terminal_at`` obs."""

    def __init__(self, strat, terminal_at, terminal_kind):
        self.inner = SimpleNamespace(robot_type="mock", name="mock")
        self._strat = strat
        self._terminal_at = terminal_at
        self._terminal_kind = terminal_kind
        self._n = 0
        self.sent: list = []

    def get_observation(self):
        self._n += 1
        if self._n == self._terminal_at:
            if self._terminal_kind == "ignore":
                self._strat._rlt_state["lifecycle"].signal_ignore()
            else:
                self._strat._rlt_state["lifecycle"].signal_terminal(self._terminal_kind)
            self._strat._episode_end.set()
        return {f"j{i}.pos": 0.0 for i in range(ACTION_DIM)}

    def send_action(self, action):
        self.sent.append(action)


def _make_run_ctx(strat, terminal_at, terminal_kind):
    robot = _MockRobot(strat, terminal_at, terminal_kind)
    proc = SimpleNamespace(
        robot_observation_processor=lambda o: o,
        robot_action_processor=lambda t: t[0],
        teleop_action_processor=lambda t: t[0],
    )
    ctx = SimpleNamespace(
        runtime=SimpleNamespace(
            cfg=SimpleNamespace(fps=500.0, duration=0.0, dataset=None, task="pick"),
            shutdown_event=Event(),
        ),
        hardware=SimpleNamespace(robot_wrapper=robot, teleop=None),
        processors=proc,
        data=SimpleNamespace(
            ordered_action_keys=[f"j{i}.pos" for i in range(ACTION_DIM)],
            dataset_features={},
        ),
        policy=SimpleNamespace(policy=SimpleNamespace(reset=lambda: None)),
    )
    # Bypass the real policy-batch builder (needs valid dataset_features).
    strat._build_policy_batch = lambda *a, **k: {"observation.state": torch.zeros(1, STATE_DIM)}
    return ctx, robot


def test_run_episode_autonomous_commits_terminal(tmp_path):
    strat = _build_strategy(_make_config(tmp_path))
    ctx, robot = _make_run_ctx(strat, terminal_at=9, terminal_kind=TerminalKind.SUCCESS)

    strat._run_episode(ctx, control_interval=1 / 500.0)

    # Autonomous frames were dispatched to the robot.
    assert len(robot.sent) > 0
    # Transitions were committed and exactly one is a terminal (done=True).
    assert len(strat._replay) >= 1
    committed_dones = strat._replay._done[: len(strat._replay)].sum().item()
    assert committed_dones == 1
    assert strat._rlt_state["episode"] == 0
    assert strat._rlt_state["successes"] == [True]
    # Terminal was consumed, so the next episode can begin without asserting.
    strat.on_episode_begin(ctx)
    assert strat._rlt_state["episode"] == 1


def test_run_episode_ignore_discards(tmp_path):
    strat = _build_strategy(_make_config(tmp_path))
    ctx, robot = _make_run_ctx(strat, terminal_at=9, terminal_kind="ignore")

    strat._run_episode(ctx, control_interval=1 / 500.0)

    # IGNORE drops all staged transitions and rolls the counter back.
    assert len(strat._replay) == 0
    assert strat._rlt_state["episode"] == -1


# ---------------------------------------------------------------------------
# GPU numeric validation — the RL machinery (TD3 updates, actor/critic forward,
# replay sampling, gradient flow) running on cuda. Guards device-placement bugs
# that the CPU tests can't catch.
# ---------------------------------------------------------------------------


@require_cuda
def test_gpu_machinery_lives_on_cuda(tmp_path):
    strat = _build_strategy(_make_config(tmp_path), device="cuda")
    assert next(strat._agent.actor.parameters()).is_cuda
    assert next(strat._agent.critic.parameters()).is_cuda
    assert next(strat._token_encoder.parameters()).is_cuda


@require_cuda
def test_gpu_gradient_updates_are_finite(tmp_path):
    strat = _build_strategy(_make_config(tmp_path), device="cuda")
    cfg = strat._rlt_state["config"]
    for _ in range(300):
        strat._replay.add(
            z_rl=_z("cuda"), state=_s("cuda"), action_chunk=_chunk("cuda"), ref_chunk=_chunk("cuda"),
            reward=0.0, next_z_rl=_z("cuda"), next_state=_s("cuda"), next_ref_chunk=_chunk("cuda"), done=False,
        )
    strat._replay.commit()

    before = strat._rlt_state["total_updates"]
    strat.background_step()
    assert strat._rlt_state["total_updates"] == before + cfg.utd_ratio

    # Critic Q on cuda must be finite (no NaN/Inf from a device-mismatch bug).
    sample = strat._replay.sample(64)
    q = strat._agent.critic.min_q(
        sample["z_rl"], sample["state"], sample["action"].reshape(-1, CHUNK, ACTION_DIM)
    )
    assert q.is_cuda
    assert torch.isfinite(q).all()


@require_cuda
def test_gpu_infer_chunk_produces_executable_frames(tmp_path):
    """Drive _rlt_infer_chunk on cuda with a stub adapter and assert the actor
    refines the chunk into a CPU tensor ready for frame-by-frame execution."""
    strat = _build_strategy(_make_config(tmp_path), device="cuda")
    strat.on_episode_begin(ctx=None)  # engage the actor + open the gate
    strat._build_policy_batch = lambda *a, **k: {
        "observation.state": torch.zeros(1, STATE_DIM, device="cuda")
    }

    # Warm past warmup so the actor (not the raw ref) drives.
    strat._rlt_state["episode"] = strat._rlt_config.warmup_episodes + 1
    strat._rlt_infer_chunk(ctx=None, obs_processed={}, task_str="t", robot_type="stub")

    assert strat._chunk_raw is not None
    assert strat._chunk_raw.shape == (CHUNK, ACTION_DIM)
    assert strat._chunk_raw.device.type == "cpu"  # kept on CPU for the send loop
    assert torch.isfinite(strat._chunk_raw).all()
    # z_rl was published for the intervention recorder.
    assert strat._rlt_latest_z_rl is not None and strat._rlt_latest_z_rl.numel() == CTX_DIM

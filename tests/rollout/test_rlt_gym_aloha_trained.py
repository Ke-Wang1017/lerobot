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

"""End-to-end RLT numeric + training pipeline against gym-aloha.

Where ``test_rlt_gym_aloha.py`` drives the control loop with a *stub* adapter
(zeros), this file uses a small **real** policy (``_TinyAlohaPolicy``) whose
``encode_observations`` / ``predict_action_chunk`` are genuine neural forward
passes, wrapped in a real :class:`RLTPolicyAdapter` with real (non-identity)
normalization stats. That exercises the full RLT numeric pipeline end to end:

  * **Phase 1** — the policy-agnostic ``train_rl_token_encoder`` learns to
    reconstruct the policy's context tokens; we assert the trained encoder/
    decoder reconstruct a held-out context materially better than random init.
  * **Phase 2** — ``RLTStrategy`` runs online against the gym-aloha sim using
    that trained token encoder and the real adapter: real z_rl, real reference
    chunks, real actor/critic gradient updates, staying finite and stable.

Limitation: ``_TinyAlohaPolicy`` has random (untrained) weights — no trained
ALOHA S1 exists in-repo — so this validates the numeric + training *mechanics*,
not task competence. Solving the task is genuine on-robot / trained-checkpoint
validation.
"""

from __future__ import annotations

import os
from threading import Event
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import torch.nn as nn

from lerobot.policies.rlt.config import RLTConfig
from lerobot.policies.rlt.episode import TerminalKind
from lerobot.policies.rlt.policy_adapter import RLTPolicyAdapter
from lerobot.policies.rlt.token import (
    RLTokenDecoder,
    RLTokenEncoder,
    load_rlt_token_config,
    rl_token_reconstruction_loss,
)
from lerobot.policies.rlt.train_token import train_rl_token_encoder
from lerobot.rollout.configs import RLTStrategyConfig
from lerobot.rollout.strategies.rlt import RLTStrategy

N_JOINTS = 14
CTX_DIM = 32
N_TOKENS = 4
CHUNK = 4
JOINTS = [f"joint_{i}.pos" for i in range(N_JOINTS)]


# ---------------------------------------------------------------------------
# A small REAL policy: genuine forward passes for context + reference chunk.
# ---------------------------------------------------------------------------


class _TinyAlohaPolicy(nn.Module):
    """Minimal real policy exposing the surface the RLT adapter needs.

    ``encode_observations`` maps proprioceptive state to a structured token
    sequence (a real, low-rank manifold the token autoencoder can learn to
    reconstruct); ``predict_action_chunk`` maps state to a raw action chunk.
    """

    def __init__(self):
        super().__init__()
        self.ctx_head = nn.Sequential(
            nn.Linear(N_JOINTS, 64), nn.Tanh(), nn.Linear(64, N_TOKENS * CTX_DIM)
        )
        self.act_head = nn.Sequential(
            nn.Linear(N_JOINTS, 64), nn.Tanh(), nn.Linear(64, CHUNK * N_JOINTS)
        )

    def encode_observations(self, state: torch.Tensor) -> torch.Tensor:
        b = state.shape[0]
        return self.ctx_head(state).reshape(b, N_TOKENS, CTX_DIM)

    def predict_action_chunk(self, state: torch.Tensor) -> torch.Tensor:
        b = state.shape[0]
        # Bounded raw actions in roughly [-1, 1] (gym-aloha action range).
        return torch.tanh(self.act_head(state)).reshape(b, CHUNK, N_JOINTS)


class _TinyAlohaAdapter(RLTPolicyAdapter):
    """Real adapter over ``_TinyAlohaPolicy`` with real normalization stats."""

    def __init__(self, policy: _TinyAlohaPolicy, device: torch.device):
        self._policy = policy
        self._device = device
        # Fixed, non-trivial per-joint stats so normalize/denormalize are real.
        self._state_mean = torch.linspace(-0.5, 0.5, N_JOINTS, device=device)
        self._state_std = torch.linspace(0.5, 1.5, N_JOINTS, device=device)
        self._action_mean = torch.zeros(N_JOINTS, device=device)
        self._action_std = torch.full((N_JOINTS,), 0.5, device=device)

    @property
    def context_dim(self) -> int:
        return CTX_DIM

    @property
    def action_dim(self) -> int:
        return N_JOINTS

    @property
    def state_dim(self) -> int:
        return N_JOINTS

    @property
    def chunk_size(self) -> int:
        return CHUNK

    def encode_context(self, batch: dict) -> torch.Tensor:
        state = batch["observation.state"].to(self._device).float()
        return self._policy.encode_observations(state)

    def predict_reference_chunk(self, batch: dict, context=None, **kwargs) -> torch.Tensor:
        state = batch["observation.state"].to(self._device).float()
        return self._policy.predict_action_chunk(state)

    def normalize_state(self, state: torch.Tensor) -> torch.Tensor:
        return (state - self._state_mean) / self._state_std

    def normalize_action(self, action: torch.Tensor) -> torch.Tensor:
        return (action - self._action_mean) / self._action_std

    def denormalize_action(self, action: torch.Tensor) -> torch.Tensor:
        return action * self._action_std + self._action_mean


# ---------------------------------------------------------------------------
# Gym→hardware bridge (shared shape with test_rlt_gym_aloha)
# ---------------------------------------------------------------------------


class _GymAlohaRobot:
    def __init__(self, env, strat, on_obs=None):
        self.env = env
        self.inner = SimpleNamespace(robot_type="gym_aloha", name="gym_aloha")
        self._strat = strat
        self._on_obs = on_obs
        self.n_obs = 0
        self.n_steps = 0
        obs, _ = env.reset(seed=0)
        self._obs = obs

    def get_observation(self):
        self.n_obs += 1
        if self._on_obs is not None:
            self._on_obs(self._strat, self.n_obs)
        agent_pos = self._obs["agent_pos"]
        return {JOINTS[i]: float(agent_pos[i]) for i in range(N_JOINTS)}

    def send_action(self, action_dict):
        action = np.clip(np.array([action_dict[k] for k in JOINTS], dtype=np.float32), -1.0, 1.0)
        self._obs, _r, _term, _trunc, _info = self.env.step(action)
        self.n_steps += 1


@pytest.fixture(scope="module")
def aloha_env():
    os.environ.setdefault("MUJOCO_GL", "egl")
    gym = pytest.importorskip("gymnasium")
    pytest.importorskip("gym_aloha")
    env = gym.make("gym_aloha/AlohaInsertion-v0", obs_type="pixels_agent_pos")
    try:
        yield env
    finally:
        env.close()


def _device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---------------------------------------------------------------------------
# Phase 1 — token-encoder training measurably reduces reconstruction error.
# ---------------------------------------------------------------------------


def _collect_contexts(policy: _TinyAlohaPolicy, device: torch.device, n: int = 256) -> torch.Tensor:
    """Encode a pool of varied states into context tokens ``[n, N_TOKENS, CTX]``."""
    torch.manual_seed(0)
    states = torch.empty(n, N_JOINTS, device=device).uniform_(-1.0, 1.0)
    with torch.no_grad():
        return policy.encode_observations(states).float()


def _phase1_config() -> RLTConfig:
    cfg = RLTConfig(rl_token_dim=CTX_DIM, context_dim=CTX_DIM)
    cfg.token_train_steps = 400
    cfg.token_encoder_layers = 2
    cfg.token_decoder_layers = 2
    cfg.token_ffn_dim = 128
    cfg.token_dropout = 0.0
    return cfg


def test_phase1_token_training_reduces_reconstruction(tmp_path):
    device = _device()
    policy = _TinyAlohaPolicy().to(device).eval()
    pool = _collect_contexts(policy, device)

    def context_provider():
        def gen():
            g = torch.Generator(device="cpu").manual_seed(0)
            while True:
                idx = torch.randint(0, pool.shape[0], (8,), generator=g)
                yield pool[idx]

        return gen()

    cfg = _phase1_config()
    out = tmp_path / "token_train"
    out.mkdir(parents=True, exist_ok=True)
    train_rl_token_encoder(
        cfg, context_provider, output_dir=out, device=device, save_freq=cfg.token_train_steps
    )
    ckpt = out / f"checkpoint-{cfg.token_train_steps}"
    assert (ckpt / "encoder.pt").exists() and (ckpt / "decoder.pt").exists()

    # Held-out context: trained encoder/decoder should reconstruct it far better
    # than freshly-initialized ones.
    held_out = _collect_contexts(_TinyAlohaPolicy().to(device).eval(), device, n=32)

    trained_cfg = load_rlt_token_config(ckpt, base=_phase1_config())
    enc, dec = RLTokenEncoder(trained_cfg).to(device), RLTokenDecoder(trained_cfg).to(device)
    enc.load_state_dict(torch.load(ckpt / "encoder.pt", weights_only=True, map_location=device))
    dec.load_state_dict(torch.load(ckpt / "decoder.pt", weights_only=True, map_location=device))
    enc.eval()
    dec.eval()

    torch.manual_seed(1)
    fresh_enc, fresh_dec = RLTokenEncoder(trained_cfg).to(device), RLTokenDecoder(trained_cfg).to(device)
    fresh_enc.eval()
    fresh_dec.eval()

    with torch.no_grad():
        trained_loss = rl_token_reconstruction_loss(enc, dec, held_out).item()
        fresh_loss = rl_token_reconstruction_loss(fresh_enc, fresh_dec, held_out).item()

    assert np.isfinite(trained_loss)
    assert trained_loss < 0.5 * fresh_loss, (
        f"token training did not learn: trained={trained_loss:.4f} vs fresh={fresh_loss:.4f}"
    )


# ---------------------------------------------------------------------------
# Phase 2 — online RLT loop consuming a trained encoder + the real adapter.
# ---------------------------------------------------------------------------


def _train_token_checkpoint(tmp_path, policy, device) -> str:
    pool = _collect_contexts(policy, device)

    def context_provider():
        def gen():
            g = torch.Generator(device="cpu").manual_seed(0)
            while True:
                idx = torch.randint(0, pool.shape[0], (8,), generator=g)
                yield pool[idx]

        return gen()

    cfg = _phase1_config()
    cfg.token_train_steps = 200
    out = tmp_path / "token_train"
    out.mkdir(parents=True, exist_ok=True)
    train_rl_token_encoder(
        cfg, context_provider, output_dir=out, device=device, save_freq=cfg.token_train_steps
    )
    return str(out / f"checkpoint-{cfg.token_train_steps}")


def _make_ctx(strat, robot):
    proc = SimpleNamespace(
        robot_observation_processor=lambda o: o,
        robot_action_processor=lambda t: t[0],
        teleop_action_processor=lambda t: t[0],
    )
    ctx = SimpleNamespace(
        runtime=SimpleNamespace(
            cfg=SimpleNamespace(fps=1000.0, duration=0.0, dataset=None, task="insertion"),
            shutdown_event=Event(),
        ),
        hardware=SimpleNamespace(robot_wrapper=robot, teleop=None),
        processors=proc,
        data=SimpleNamespace(ordered_action_keys=list(JOINTS), dataset_features={}),
        policy=SimpleNamespace(policy=SimpleNamespace(reset=lambda: None)),
    )
    strat._build_policy_batch = lambda _ctx, obs_p, _task, _rt: {
        "observation.state": torch.tensor(
            [[obs_p[JOINTS[i]] for i in range(N_JOINTS)]], dtype=torch.float32, device=strat._device
        )
    }
    return ctx


def test_end_to_end_trained_pipeline(aloha_env, tmp_path):
    device = _device()
    policy = _TinyAlohaPolicy().to(device).eval()
    token_ckpt = _train_token_checkpoint(tmp_path, policy, device)

    config = RLTStrategyConfig(
        rl_token_checkpoint=token_ckpt,
        output_dir=str(tmp_path / "rlt_out"),
        rl_chunk_length=CHUNK,
        deploy=False,
    )
    strat = RLTStrategy(config)
    strat._device = device
    strat._adapter = _TinyAlohaAdapter(policy, device)
    strat._build_rl_machinery(SimpleNamespace(data=SimpleNamespace(ordered_action_keys=list(JOINTS))))

    # Warm the buffer so online gradient updates run during the loop.
    for _ in range(300):
        strat._replay.add(
            z_rl=torch.randn(CTX_DIM, device=device),
            state=torch.randn(N_JOINTS, device=device),
            action_chunk=torch.randn(CHUNK, N_JOINTS, device=device),
            ref_chunk=torch.randn(CHUNK, N_JOINTS, device=device),
            reward=float(np.random.default_rng(0).uniform(-1, 1)),
            next_z_rl=torch.randn(CTX_DIM, device=device),
            next_state=torch.randn(N_JOINTS, device=device),
            next_ref_chunk=torch.randn(CHUNK, N_JOINTS, device=device),
            done=False,
        )
    strat._replay.commit()

    def on_obs(s, n):
        if n == 14:
            s._rlt_state["lifecycle"].signal_terminal(TerminalKind.SUCCESS)
            s._episode_end.set()

    robot = _GymAlohaRobot(aloha_env, strat, on_obs)
    strat._run_episode(_make_ctx(strat, robot), control_interval=1e-3)

    # Real forwards flowed through the whole pipeline: env stepped, transitions
    # were committed with a terminal, and online gradient updates fired while
    # keeping Q finite.
    assert robot.n_steps > 0
    assert len(strat._replay) >= 1
    assert int(strat._replay._done[: len(strat._replay)].sum().item()) == 1
    assert strat._rlt_state["total_updates"] >= strat._rlt_config.utd_ratio

    sample = strat._replay.sample(64)
    q = strat._agent.critic.min_q(
        sample["z_rl"], sample["state"], sample["action"].reshape(-1, CHUNK, N_JOINTS)
    )
    assert torch.isfinite(q).all()
    # z_rl exposed to the recorder came from the REAL token encoder (non-zero).
    assert strat._rlt_latest_z_rl is not None
    assert strat._rlt_latest_z_rl.abs().sum().item() > 0

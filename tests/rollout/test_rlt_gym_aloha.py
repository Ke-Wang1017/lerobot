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

"""RLT control-loop integration tests against the gym-aloha MuJoCo simulation.

The rollout framework drives the hardware ``Robot`` / ``Teleoperator`` interface
(``get_observation`` / ``send_action`` / ``get_action``); gym-aloha is a
Gymnasium env (``reset`` / ``step``). :class:`_GymAlohaRobot` bridges the two so
``RLTStrategy``'s *real* control loop runs against real physics: proprioceptive
state → policy batch → z_rl → actor refinement → executed action → ``env.step``
→ transition storage. A scripted ``on_obs`` callback plays the operator
(terminal hotkeys, intervention toggles), and :class:`_GymAlohaTeleop` supplies
human correction actions.

Scope / limitation: this validates the **loop mechanics and RL plumbing** end to
end against a stepping simulator across the loop's branches (autonomous success/
abort, ignore, human intervention, deploy, gradient updates, multi-episode). The
base policy is a *stub* adapter — a trained ALOHA S1 + RL-token encoder do not
exist in this repo — so it does not validate real-policy task numerics, which
remains genuine on-robot validation. The zero-initialised actor emits ~0
actions, so the arm holds; the point is that the whole loop steps the simulator
without error and records well-formed transitions.
"""

from __future__ import annotations

import os
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

# ALOHA is a 14-DoF bimanual arm; gym-aloha's action + agent_pos are both 14-dim.
N_JOINTS = 14
CTX_DIM = 32
CHUNK = 4
JOINTS = [f"joint_{i}.pos" for i in range(N_JOINTS)]


# ---------------------------------------------------------------------------
# gym-aloha env fixture (made once per module; reset per test)
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Stub policy + Gym→hardware bridges
# ---------------------------------------------------------------------------


class _AlohaStubAdapter(RLTPolicyAdapter):
    """Stub base policy sized for ALOHA (identity normalization)."""

    def __init__(self, device: str = "cpu"):
        self._device = torch.device(device)

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
        return torch.zeros(1, 4, CTX_DIM, device=self._device)

    def predict_reference_chunk(self, batch: dict, context=None, **kwargs) -> torch.Tensor:
        return torch.zeros(1, CHUNK, N_JOINTS, device=self._device)

    def normalize_state(self, state):
        return state

    def normalize_action(self, action):
        return action

    def denormalize_action(self, action):
        return action


class _GymAlohaRobot:
    """Adapts a gym-aloha env to ``get_observation`` / ``send_action``.

    ``on_obs(strat, n_obs)`` plays the operator: it may signal terminals or
    request intervention transitions at chosen ticks.
    """

    def __init__(self, env, strat: RLTStrategy, on_obs=None):
        self.env = env
        self.inner = SimpleNamespace(robot_type="gym_aloha", name="gym_aloha")
        self._strat = strat
        self._on_obs = on_obs
        self.n_obs = 0
        self.n_steps = 0
        obs, _ = env.reset(seed=0)
        self._obs = obs

    def get_observation(self) -> dict:
        self.n_obs += 1
        if self._on_obs is not None:
            self._on_obs(self._strat, self.n_obs)
        agent_pos = self._obs["agent_pos"]
        return {JOINTS[i]: float(agent_pos[i]) for i in range(N_JOINTS)}

    def send_action(self, action_dict: dict) -> None:
        action = np.array([action_dict[k] for k in JOINTS], dtype=np.float32)
        action = np.clip(action, -1.0, 1.0)
        self._obs, _r, _term, _trunc, _info = self.env.step(action)
        self.n_steps += 1


class _GymAlohaTeleop:
    """Non-actuated teleop leader: supplies human correction actions.

    ``feedback_features`` is empty so the strategy treats it as non-actuated
    (no torque calls); ``get_action`` returns small joint targets that the
    intervention recorder captures.
    """

    feedback_features: dict = {}

    def get_action(self) -> dict:
        return dict.fromkeys(JOINTS, 0.01)


# ---------------------------------------------------------------------------
# Builders
# ---------------------------------------------------------------------------


def _make_strategy(tmp_path, device: str, *, deploy: bool = False, rlt_checkpoint: str | None = None):
    token_dir = tmp_path / "token"
    if not token_dir.exists():
        token_dir.mkdir()
        token_cfg = RLTConfig(rl_token_dim=CTX_DIM, context_dim=CTX_DIM)
        torch.save(RLTokenEncoder(token_cfg).state_dict(), token_dir / "encoder.pt")
        save_rlt_token_config(token_dir, token_cfg)

    config = RLTStrategyConfig(
        rl_token_checkpoint=str(token_dir),
        output_dir=str(tmp_path / ("deploy_out" if deploy else "rlt_out")),
        rl_chunk_length=CHUNK,
        deploy=deploy,
        rlt_checkpoint=rlt_checkpoint,
    )
    strat = RLTStrategy(config)
    strat._device = torch.device(device)
    strat._adapter = _AlohaStubAdapter(device)
    strat._build_rl_machinery(SimpleNamespace(data=SimpleNamespace(ordered_action_keys=list(JOINTS))))
    return strat


def _make_ctx(strat: RLTStrategy, robot: _GymAlohaRobot, teleop=None):
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
        hardware=SimpleNamespace(robot_wrapper=robot, teleop=teleop),
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


def _device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def _committed_dones(strat) -> int:
    n = len(strat._replay)
    return int(strat._replay._done[:n].sum().item()) if n else 0


# ---------------------------------------------------------------------------
# Autonomous terminals
# ---------------------------------------------------------------------------


def test_autonomous_success_commits_terminal(aloha_env, tmp_path):
    strat = _make_strategy(tmp_path, _device())

    def on_obs(s, n):
        if n == 13:
            s._rlt_state["lifecycle"].signal_terminal(TerminalKind.SUCCESS)
            s._episode_end.set()

    robot = _GymAlohaRobot(aloha_env, strat, on_obs)
    strat._run_episode(_make_ctx(strat, robot), control_interval=1e-3)

    assert robot.n_steps > 0
    assert len(strat._replay) >= 1
    assert _committed_dones(strat) == 1
    assert strat._rlt_state["successes"] == [True]
    # Terminal consumed → next episode can begin.
    strat.on_episode_begin(ctx=None)
    assert strat._rlt_state["episode"] == 1


def test_autonomous_abort_records_negative_terminal(aloha_env, tmp_path):
    strat = _make_strategy(tmp_path, _device())

    def on_obs(s, n):
        if n == 13:
            s._rlt_state["lifecycle"].signal_terminal(TerminalKind.ABORT)
            s._episode_end.set()

    robot = _GymAlohaRobot(aloha_env, strat, on_obs)
    strat._run_episode(_make_ctx(strat, robot), control_interval=1e-3)

    assert _committed_dones(strat) == 1
    assert strat._rlt_state["successes"] == [False]
    # The terminal reward is the abort reward.
    n = len(strat._replay)
    dones = strat._replay._done[:n].squeeze(-1).bool()
    term_reward = strat._replay._reward[:n].squeeze(-1)[dones]
    assert torch.allclose(term_reward, torch.tensor(strat._rlt_config.abort_reward))


def test_ignore_discards_episode(aloha_env, tmp_path):
    strat = _make_strategy(tmp_path, _device())

    def on_obs(s, n):
        if n == 13:
            s._rlt_state["lifecycle"].signal_ignore()
            s._episode_end.set()

    robot = _GymAlohaRobot(aloha_env, strat, on_obs)
    strat._run_episode(_make_ctx(strat, robot), control_interval=1e-3)

    assert len(strat._replay) == 0
    assert strat._rlt_state["episode"] == -1


# ---------------------------------------------------------------------------
# Human intervention
# ---------------------------------------------------------------------------


def test_intervention_records_human_frames(aloha_env, tmp_path, monkeypatch):
    # Skip the 2 s smooth-handover interpolation so the test stays fast; we are
    # validating recording, not the handover motion.
    monkeypatch.setattr("lerobot.rollout.strategies.rlt.follower_smooth_move_to", lambda *a, **k: None)
    monkeypatch.setattr("lerobot.rollout.strategies.rlt.teleop_smooth_move_to", lambda *a, **k: None)

    strat = _make_strategy(tmp_path, _device())

    def on_obs(s, n):
        # A few autonomous frames prime z_rl, then take over, correct, hand back,
        # and finally mark success.
        if n == 3:
            s._events.request_transition("pause_resume")  # AUTONOMOUS -> PAUSED
        elif n == 4:
            s._events.request_transition("correction")  # PAUSED -> CORRECTING
        elif n == 18:
            s._events.request_transition("correction")  # CORRECTING -> PAUSED
        elif n == 19:
            s._events.request_transition("pause_resume")  # PAUSED -> AUTONOMOUS
        elif n == 24:
            s._rlt_state["lifecycle"].signal_terminal(TerminalKind.SUCCESS)
            s._episode_end.set()

    robot = _GymAlohaRobot(aloha_env, strat, on_obs)
    teleop = _GymAlohaTeleop()
    strat._run_episode(_make_ctx(strat, robot, teleop), control_interval=1e-3)

    from lerobot.policies.rlt.metrics import get_metrics

    # Transitions were recorded, including human correction chunks.
    assert len(strat._replay) >= 1
    assert strat._rlt_state["total_transitions"] >= 1
    assert strat._rlt_state["successes"] == [True]
    # The CORRECTING branch was genuinely entered: the episode is registered as
    # non-autonomous (mark_intervention fired on PAUSED -> CORRECTING).
    assert get_metrics().episodes.autonomous[-1] is False


# ---------------------------------------------------------------------------
# Deploy mode
# ---------------------------------------------------------------------------


def test_deploy_mode_runs_without_replay(aloha_env, tmp_path):
    # First produce a checkpoint to deploy from.
    trainer = _make_strategy(tmp_path, _device())
    trainer._rlt_state["episode"] = 0
    trainer._save_checkpoint(snapshot_every_10=False)

    strat = _make_strategy(
        tmp_path, _device(), deploy=True, rlt_checkpoint=str(tmp_path / "rlt_out")
    )
    assert strat._replay is None
    assert strat._recorder is None

    def on_obs(s, n):
        if n == 10:
            s._rlt_state["lifecycle"].signal_terminal(TerminalKind.SUCCESS)
            s._episode_end.set()

    robot = _GymAlohaRobot(aloha_env, strat, on_obs)
    # Deploy runs the actor deterministically; the loop must step the sim cleanly.
    strat._run_episode(_make_ctx(strat, robot), control_interval=1e-3)
    assert robot.n_steps > 0
    assert strat._rlt_state["successes"] == [True]


# ---------------------------------------------------------------------------
# In-loop gradient updates
# ---------------------------------------------------------------------------


def test_inloop_gradient_updates_fire(aloha_env, tmp_path):
    strat = _make_strategy(tmp_path, _device())
    dev = strat._device
    # Warm the committed buffer past the 256-transition gate so background_step
    # runs on every autonomous tick.
    for _ in range(300):
        strat._replay.add(
            z_rl=torch.randn(CTX_DIM, device=dev),
            state=torch.randn(N_JOINTS, device=dev),
            action_chunk=torch.randn(CHUNK, N_JOINTS, device=dev),
            ref_chunk=torch.randn(CHUNK, N_JOINTS, device=dev),
            reward=0.0,
            next_z_rl=torch.randn(CTX_DIM, device=dev),
            next_state=torch.randn(N_JOINTS, device=dev),
            next_ref_chunk=torch.randn(CHUNK, N_JOINTS, device=dev),
            done=False,
        )
    strat._replay.commit()

    def on_obs(s, n):
        if n == 13:
            s._rlt_state["lifecycle"].signal_terminal(TerminalKind.SUCCESS)
            s._episode_end.set()

    robot = _GymAlohaRobot(aloha_env, strat, on_obs)
    strat._run_episode(_make_ctx(strat, robot), control_interval=1e-3)

    # Gradient updates fired during the loop and left Q finite.
    assert strat._rlt_state["total_updates"] >= strat._rlt_config.utd_ratio
    sample = strat._replay.sample(64)
    q = strat._agent.critic.min_q(
        sample["z_rl"], sample["state"], sample["action"].reshape(-1, CHUNK, N_JOINTS)
    )
    assert torch.isfinite(q).all()


# ---------------------------------------------------------------------------
# Multi-episode run()
# ---------------------------------------------------------------------------


def test_full_run_multiple_episodes(aloha_env, tmp_path):
    strat = _make_strategy(tmp_path, _device())

    def on_obs(s, n):
        # End each episode after 8 of its frames (episode_end is cleared at the
        # start of each episode); stop the session after the second.
        if s._rlt_state["lifecycle"].active and (n % 8 == 0):
            s._rlt_state["lifecycle"].signal_terminal(TerminalKind.SUCCESS)
            s._episode_end.set()
        if n >= 17:
            s._events.stop_recording.set()

    robot = _GymAlohaRobot(aloha_env, strat, on_obs)
    strat.run(_make_ctx(strat, robot))

    # At least two episodes completed and were recorded.
    assert len(strat._rlt_state["successes"]) >= 2
    assert strat._rlt_state["episode"] >= 1

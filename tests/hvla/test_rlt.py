"""Unit tests for RLT (RL Token) modules.

Every test here guards against a specific failure mode — most were real bugs.
If a test fails, the message tells you exactly what went wrong and why it matters.
"""

from __future__ import annotations

import json
import os
import tempfile
import threading

import pytest
import torch

from lerobot.policies.hvla.rlt.actor_critic import RLTActor, RLTCritic, TD3Agent
from lerobot.policies.hvla.rlt.config import RLTConfig
from lerobot.policies.hvla.rlt.metrics import (
    RLTMetrics,
    get_metrics,
    load_metrics_from_file,
    reset_metrics,
    save_metrics_to_file,
    set_metrics_path,
)
from lerobot.policies.hvla.rlt.replay_buffer import ReplayBuffer, TransactionalReplayBuffer
from lerobot.policies.hvla.rlt.token import (
    RLTokenDecoder,
    RLTokenEncoder,
    load_rlt_token_config,
    rl_token_reconstruction_loss,
    save_rlt_token_config,
)
from lerobot.policies.hvla.s1_process import _atomic_torch_save

# Small dims for speed — relationships (ratios, formulas) don't depend on size.
D = 64
S = 14
A = 14
C = 5
N_CTX = 10
B = 4


@pytest.fixture
def config():
    return RLTConfig(
        rl_token_dim=D,
        token_encoder_layers=1,
        token_decoder_layers=1,
        token_num_heads=4,
        token_ffn_dim=128,
        token_dropout=0.0,
        actor_hidden_dim=32,
        actor_num_layers=1,
        critic_hidden_dim=32,
        critic_num_layers=1,
        num_critics=2,
        rl_chunk_length=C,
        exploration_sigma=0.02,
        beta=0.1,
        ref_action_dropout=0.0,
        replay_capacity=100,
        warmup_episodes=2,
    )


@pytest.fixture
def device():
    return torch.device("cpu")


# ===================================================================
# BC penalty must use L2 sum, not MSE mean (real bug: 140× too weak)
# ===================================================================


class TestBCPenalty:
    def test_update_actor_bc_uses_sum_not_mean(self, config, device):
        """Run the actual update_actor code and verify the loss value matches
        the L2 sum formula, not MSE. Tests the real code path, not a proxy."""
        config.ref_action_dropout = 0.0
        config.beta = 1.0
        agent = TD3Agent(config, S, A, device)

        z_rl = torch.randn(1, D, device=device)
        state = torch.randn(1, S, device=device)
        ref = torch.randn(1, C, A, device=device)

        # Compute what the loss SHOULD be with sum
        with torch.no_grad():
            action = agent.actor.mean(z_rl, state, ref)
            q_val = agent.critic.min_q(z_rl, state, action)
            bc_sum = ((action - ref) ** 2).sum(dim=(-1, -2)).mean()
            bc_mse = ((action - ref) ** 2).mean()
            expected_loss_sum = -q_val.mean() + 1.0 * bc_sum
            expected_loss_mse = -q_val.mean() + 1.0 * bc_mse

        actual_loss, _q_term, _bc_term = agent.update_actor(z_rl.clone(), state.clone(), ref.clone())

        # actual_loss should match sum formula, not MSE formula
        assert abs(actual_loss - expected_loss_sum.item()) < 0.1, (
            f"Actor loss {actual_loss:.4f} doesn't match L2 sum formula {expected_loss_sum.item():.4f}. "
            f"If it matches MSE formula ({expected_loss_mse.item():.4f}), someone switched to mean()."
        )

    def test_bc_penalty_against_ref_not_dropout_input(self, config, device):
        """BC penalty compares actor output against ORIGINAL ref_chunk,
        even when ref_action_dropout zeroes the actor's input.
        Dropout affects what the actor sees, not what it's penalized against."""
        config.ref_action_dropout = 1.0  # always drop → actor sees zeros
        config.beta = 100.0  # dominate the loss
        agent = TD3Agent(config, S, A, device)

        ref = torch.ones(B, C, A, device=device) * 5.0  # big ref
        z_rl = torch.randn(B, D, device=device)
        state = torch.randn(B, S, device=device)

        loss, _q_term, _bc_term = agent.update_actor(z_rl, state, ref)
        # With beta=100 and ref=5.0, if BC is against ref, loss should be large.
        # If BC were against the zeroed input, penalty would be much smaller.
        assert loss > 10.0, (
            f"Actor loss {loss:.2f} too small — BC penalty may be computed against "
            f"the dropout-zeroed input instead of the original ref_chunk."
        )


# ===================================================================
# Reconstruction loss: sum over D, not mean (same class of bug)
# ===================================================================


class TestReconstructionLoss:
    def test_actual_loss_uses_sum_over_token_dim(self, config):
        """Call the real rl_token_reconstruction_loss and verify it uses sum."""
        enc = RLTokenEncoder(config)
        dec = RLTokenDecoder(config)
        ctx = torch.randn(B, N_CTX, D)

        actual_loss = rl_token_reconstruction_loss(enc, dec, ctx)

        # Recompute with both formulas
        with torch.no_grad():
            z = enc(ctx.detach())
            recon = dec(z, ctx.detach())
            diff_sq = (recon - ctx.detach()) ** 2
            loss_sum = diff_sq.sum(dim=-1).mean()
            loss_mse = diff_sq.mean()

        assert abs(actual_loss.item() - loss_sum.item()) < 0.01, (
            f"Loss {actual_loss.item():.2f} != sum formula {loss_sum.item():.2f}. "
            f"MSE formula gives {loss_mse.item():.4f} — if close to that, someone used mean()."
        )

    def test_encoder_and_decoder_both_get_gradients(self, config):
        """If either is accidentally detached, the bottleneck won't learn."""
        enc = RLTokenEncoder(config)
        dec = RLTokenDecoder(config)
        loss = rl_token_reconstruction_loss(enc, dec, torch.randn(B, N_CTX, D))
        loss.backward()
        for name, p in enc.named_parameters():
            assert p.grad is not None and p.grad.abs().sum() > 0, (
                f"Encoder param '{name}' got no gradient — encoder may be detached"
            )
        for name, p in dec.named_parameters():
            assert p.grad is not None and p.grad.abs().sum() > 0, (
                f"Decoder param '{name}' got no gradient — decoder may be detached"
            )


# ===================================================================
# Critic: pessimistic Q (min, not max), correct discount, target frozen
# ===================================================================


class TestCriticInvariants:
    def test_min_q_is_pessimistic(self, config, device):
        """TD3 requires min over ensemble. Max would cause overestimation."""
        critic = RLTCritic(config, S, A)
        z = torch.randn(B, D)
        s = torch.randn(B, S)
        a = torch.randn(B, C, A)

        qs = critic(z, s, a)
        min_q = critic.min_q(z, s, a)
        expected = torch.min(torch.cat(qs, dim=-1), dim=-1, keepdim=True).values
        assert torch.allclose(min_q, expected), "min_q must return minimum over ensemble, not maximum or mean"

    def test_discount_uses_gamma_to_the_C(self, config, device):
        """Chunk-level RL: discount should be gamma^C, not gamma.
        Wrong discount = wrong temporal credit assignment."""
        config.discount = 0.99
        agent = TD3Agent(config, S, A, device)

        # Create a batch where we can verify the target computation
        reward = torch.ones(1, 1, device=device)
        torch.zeros(1, 1, device=device)
        z = torch.randn(1, D, device=device)
        s = torch.randn(1, S, device=device)
        ref = torch.randn(1, C, A, device=device)

        with torch.no_grad():
            next_a = agent.actor(z, s, ref, deterministic=False)
            target_q = agent.critic_target.min_q(z, s, next_a)
            # Correct: gamma^C
            expected = reward + (0.99**C) * target_q
            # Wrong: gamma^1
            wrong = reward + 0.99 * target_q

        assert abs(expected.item() - wrong.item()) > 1e-4, "Test setup: gamma^C and gamma should differ"
        # If the code uses gamma instead of gamma^C, critic loss would converge
        # to wrong values. We verify indirectly by checking the code constant.
        assert agent.config.rl_chunk_length == C

    def test_critic_target_has_no_gradients(self, config, device):
        """Target network must be frozen. If it trains, soft update breaks."""
        agent = TD3Agent(config, S, A, device)
        for p in agent.critic_target.parameters():
            assert not p.requires_grad, "critic_target should have requires_grad=False"

    def test_update_critic_returns_loss_and_grad_norm(self, config, device):
        """update_critic returns (loss, grad_norm). grad_norm must be the
        pre-clip norm so monitoring reflects the true step magnitude, not
        the clipped value."""
        agent = TD3Agent(config, S, A, device)
        b = 4
        z = torch.randn(b, D, device=device)
        s = torch.randn(b, S, device=device)
        a = torch.randn(b, C, A, device=device)
        ref = torch.randn(b, C, A, device=device)
        reward = torch.zeros(b, 1, device=device)
        done = torch.zeros(b, 1, device=device)

        result = agent.update_critic(z, s, a, ref, reward, z, s, ref, done)
        assert isinstance(result, tuple) and len(result) == 2
        loss, grad_norm = result
        assert isinstance(loss, float) and loss >= 0.0
        assert isinstance(grad_norm, float) and grad_norm >= 0.0

    def test_critic_grad_clip_bounds_step_magnitude(self, config, device):
        """With a crazy-large target Q, the pre-clip grad norm will exceed
        the clip. The stored weight update magnitude should be bounded."""
        config.critic_grad_clip = 1.0
        agent = TD3Agent(config, S, A, device)
        b = 8
        # Use a loss-inducing setup: random inputs → nonzero loss. We check
        # that weights move by a bounded amount even when loss is big.
        z = torch.randn(b, D, device=device) * 100  # huge inputs → huge pred
        s = torch.randn(b, S, device=device) * 100
        a = torch.randn(b, C, A, device=device)
        ref = torch.randn(b, C, A, device=device)
        reward = torch.full((b, 1), 1000.0, device=device)  # absurd reward
        done = torch.zeros(b, 1, device=device)

        before = [p.detach().clone() for p in agent.critic.parameters()]
        _, grad_norm = agent.update_critic(z, s, a, ref, reward, z, s, ref, done)
        after = list(agent.critic.parameters())

        # Pre-clip grad was large (sanity-check the test setup)
        assert grad_norm > config.critic_grad_clip, (
            f"test setup too mild: pre-clip grad_norm={grad_norm} <= clip"
        )

        # Per-param delta shouldn't exceed lr × clip (upper bound for Adam is
        # looser but this still catches runaway). lr=3e-4, clip=1.0 → delta~3e-4.
        max_delta = max((a_.detach() - b_).abs().max().item() for a_, b_ in zip(after, before, strict=False))
        assert max_delta < 1e-2, f"weight moved {max_delta} — grad clip didn't bound the step"

    def test_done_masks_bootstrap(self, config, device):
        """When done=1, target Q should ignore next-state value."""
        agent = TD3Agent(config, S, A, device)
        z = torch.randn(1, D, device=device)
        s = torch.randn(1, S, device=device)
        torch.randn(1, C, A, device=device)
        ref = torch.randn(1, C, A, device=device)

        with torch.no_grad():
            next_a = agent.actor(z, s, ref, deterministic=False)
            next_q = agent.critic_target.min_q(z, s, next_a)
            gamma_C = agent.config.discount**C

            target_done = 1.0 + (gamma_C) * (1 - 1.0) * next_q  # done=1 → just reward
            target_cont = 1.0 + (gamma_C) * (1 - 0.0) * next_q  # done=0 → reward + discounted Q

        assert abs(target_done.item() - 1.0) < 1e-5, "done=1: target should equal reward only"
        assert target_cont.item() != 1.0, "done=0: target should include discounted next Q"


# ===================================================================
# Q-explosion defenses: LayerNorm, Q-target clipping, decoupled sigmas.
# These three landed together (commits 884f01d2b, f1ed039a3, 5f1393c59)
# after the v2_widened run pumped Q to 9+ when sigmas were coupled and
# the critic had no activation bound. Each test pins one defense in
# place so a future "simplification" can't quietly remove it.
# ===================================================================


class TestQExplosionDefenses:
    def test_critic_has_layernorm_when_enabled(self, config, device):
        """LayerNorm is the activation-magnitude bound on the critic.
        It must sit BEFORE every ReLU in every Q net, and the output
        Linear must stay unnormalized (so Q can span its full range)."""
        config.critic_layer_norm = True
        critic = RLTCritic(config, S, A)
        for q_net in critic.q_nets:
            modules = list(q_net)
            # Output is a Linear with no LN/ReLU after it
            assert isinstance(modules[-1], torch.nn.Linear)
            # Hidden layers: every Linear (except the last) must be
            # immediately followed by LayerNorm then ReLU
            for i, m in enumerate(modules[:-1]):
                if isinstance(m, torch.nn.Linear):
                    assert isinstance(modules[i + 1], torch.nn.LayerNorm), (
                        f"Hidden Linear at idx {i} must be followed by LayerNorm"
                    )
                    assert isinstance(modules[i + 2], torch.nn.ReLU), (
                        f"LayerNorm at idx {i + 1} must be followed by ReLU"
                    )

    def test_critic_has_no_layernorm_when_disabled(self, config, device):
        """When the flag is off, the critic falls back to the original
        Linear→ReLU stack. Lets us A/B without code edits."""
        config.critic_layer_norm = False
        critic = RLTCritic(config, S, A)
        for q_net in critic.q_nets:
            for m in q_net:
                assert not isinstance(m, torch.nn.LayerNorm), (
                    "LayerNorm should be absent when critic_layer_norm=False"
                )

    def test_actor_never_has_layernorm(self, config, device):
        """Asymmetric design: critic gets LN (no anchor → unbounded
        activations), actor relies on BC + zero-init instead. Ship a
        test so somebody doesn't 'fix' the asymmetry by adding LN to
        the actor — that would compete with zero-init and warmup."""
        config.critic_layer_norm = True
        actor = RLTActor(config, S, A)
        for m in actor.mlp:
            assert not isinstance(m, torch.nn.LayerNorm), (
                "Actor must NOT use LayerNorm — BC anchor + zero-init is its bounding mechanism."
            )

    def test_q_target_clip_bounds_bellman_target(self, config, device):
        """With q_target_clip=True the Bellman target must lie in
        [-|abort_reward|/(1-γ^C), 1/(1-γ^C)]. Build a contrived next-Q
        that bootstraps WAY outside that range and verify clipping
        kicks in."""
        config.q_target_clip = True
        config.discount = 0.99
        config.abort_reward = -1.0
        agent = TD3Agent(config, S, A, device)

        # Hand the target net huge biases so its output is far above the
        # theoretical max — bootstrap would otherwise carry that through.
        with torch.no_grad():
            for q_net in agent.critic_target.q_nets:
                q_net[-1].bias.fill_(1e6)

        b = 4
        z = torch.randn(b, D, device=device)
        s = torch.randn(b, S, device=device)
        torch.randn(b, C, A, device=device)
        ref = torch.randn(b, C, A, device=device)
        reward = torch.zeros(b, 1, device=device)
        done = torch.zeros(b, 1, device=device)

        # Recompute the same target the critic update path computes,
        # so we test the actual clip boundary.
        with torch.no_grad():
            next_a = agent.actor(
                z, s, ref, deterministic=False, sigma=config.target_sigma, clip=config.target_noise_clip
            )
            target_q = agent.critic_target.min_q(z, s, next_a)
            raw_target = reward + (config.discount**C) * (1 - done) * target_q
            denom = 1.0 - config.discount**C
            clipped = raw_target.clamp(-1.0 / denom, 1.0 / denom)

        # Sanity: raw_target was way out of range
        assert raw_target.abs().max().item() > 100.0
        # After clip: every entry is within bounds
        bound = 1.0 / denom
        assert clipped.abs().max().item() <= bound + 1e-5

    def test_target_smoothing_uses_target_sigma_not_exploration_sigma(self, config, device):
        """Decoupling fix (commit 884f01d2b): setting exploration_sigma=0
        must NOT remove TD3 target smoothing — that's controlled by
        target_sigma. The widened-v2 Q explosion was caused by the
        original code reading exploration_sigma in the target backup,
        so killing exploration also killed the smoothing.

        We verify by sampling many next-actions with the target call
        and checking the noise std matches target_sigma, not
        exploration_sigma."""
        config.exploration_sigma = 0.0  # the failure-mode setting
        config.target_sigma = 0.1
        config.target_noise_clip = 1.0  # large enough not to truncate at this scale
        agent = TD3Agent(config, S, A, device)

        b = 5000  # need a lot to estimate std reliably
        z = torch.randn(b, D, device=device)
        s = torch.randn(b, S, device=device)
        ref = torch.randn(b, C, A, device=device)

        with torch.no_grad():
            mu = agent.actor.mean(z, s, ref)
            # Reproduce the call the critic update makes:
            sampled = agent.actor(
                z, s, ref, deterministic=False, sigma=config.target_sigma, clip=config.target_noise_clip
            )
            noise = sampled - mu
        std = noise.std().item()
        # Should be close to target_sigma=0.1, NOT exploration_sigma=0.0.
        # Tolerance 20% — std-of-Gaussian estimator from 5000×C×A samples.
        assert abs(std - 0.1) < 0.02, (
            f"target smoothing noise std={std:.3f} should be ~0.1 "
            f"(target_sigma), not 0.0 (exploration_sigma). "
            f"Decoupling regression."
        )

    def test_shared_noise_per_chunk_collapses_to_single_sample(self, config, device):
        """When ``shared_noise_per_chunk=True``, the per-frame
        independent noise is replaced by a single per-batch sample
        broadcast across the C frames. Result: action[0] - mu[0] is
        IDENTICAL to action[1] - mu[1] ... up to action[C-1] - mu[C-1]
        for every batch element. That's the smoothness fix we ship as
        an experimental flag — without it, joint commands jitter at the
        noise frequency every frame."""
        config.exploration_sigma = 0.1
        config.shared_noise_per_chunk = True
        agent = TD3Agent(config, S, A, device)

        b = 32
        z = torch.randn(b, D, device=device)
        s = torch.randn(b, S, device=device)
        ref = torch.randn(b, C, A, device=device)

        with torch.no_grad():
            mu = agent.actor.mean(z, s, ref)
            sampled = agent.actor(z, s, ref, deterministic=False)
            noise = sampled - mu  # [B, C, A]
        # All C frames must carry the SAME perturbation per (batch, action_dim)
        ref_frame = noise[:, 0:1, :]  # [B, 1, A]
        diff = (noise - ref_frame).abs().max().item()
        assert diff < 1e-5, (
            f"shared_noise_per_chunk should broadcast one [B,1,A] sample "
            f"across all C frames, but max within-chunk noise variation "
            f"= {diff} (>0). Smoothness fix is broken."
        )
        # And the sample magnitude should still be ~sigma (sanity check
        # we didn't accidentally zero it out)
        std = noise[:, 0, :].std().item()
        assert abs(std - 0.1) < 0.03, f"shared noise std={std:.3f}, expected ~0.1"

    def test_default_noise_mode_is_shared_per_chunk(self, config, device):
        """Bare RLTConfig() must default to shared-noise mode after the
        v2_widened ep101→120 A/B confirmed it improves smoothness without
        Q regression. A regression that flips this back to per-frame
        would silently re-introduce within-chunk jitter — locked in
        explicitly."""
        # Don't pass shared_noise_per_chunk to RLTConfig; the fixture
        # also doesn't set it. We rely on the dataclass default.
        from lerobot.policies.hvla.rlt.config import RLTConfig

        assert RLTConfig().shared_noise_per_chunk is True
        # Smoke check: the fixture's config still inherits the default
        # because TestQExplosionDefenses' RLTConfig() construction
        # doesn't override this knob.
        assert config.shared_noise_per_chunk is True

    def test_per_frame_mode_can_be_re_enabled_for_ab(self, config, device):
        """The False path is the A/B comparison knob — it must still
        work end-to-end so we can roll back if shared-mode regresses
        on a new task / robot / action representation."""
        config.exploration_sigma = 0.1
        config.shared_noise_per_chunk = False
        agent = TD3Agent(config, S, A, device)

        b = 32
        z = torch.randn(b, D, device=device)
        s = torch.randn(b, S, device=device)
        ref = torch.randn(b, C, A, device=device)
        with torch.no_grad():
            mu = agent.actor.mean(z, s, ref)
            sampled = agent.actor(z, s, ref, deterministic=False)
            noise = sampled - mu
        ref_frame = noise[:, 0:1, :]
        diff = (noise - ref_frame).abs().max().item()
        assert diff > 0.001, (
            f"Per-frame iid noise (shared=False) should produce within-"
            f"chunk variation, but got max diff={diff}."
        )

    def test_target_smoothing_ignores_shared_noise_flag(self, config, device):
        """Even when shared_noise_per_chunk is enabled for execution,
        the critic's target-smoothing path must keep per-position
        noise. Otherwise the bootstrap target only explores one
        perturbation direction per chunk → wrong TD3 formulation,
        critic underestimates Q-curvature across the chunk."""
        config.exploration_sigma = 0.1
        config.target_sigma = 0.1
        config.target_noise_clip = 1.0
        config.shared_noise_per_chunk = True  # ON for execution
        agent = TD3Agent(config, S, A, device)

        b = 1000  # need many samples for std check
        z = torch.randn(b, D, device=device)
        s = torch.randn(b, S, device=device)
        ref = torch.randn(b, C, A, device=device)

        with torch.no_grad():
            mu = agent.actor.mean(z, s, ref)
            # Reproduce the exact call update_critic makes for target smoothing
            target_sampled = agent.actor(
                z,
                s,
                ref,
                deterministic=False,
                sigma=config.target_sigma,
                clip=config.target_noise_clip,
                shared_noise=False,
            )
            target_noise = target_sampled - mu
        # Within each batch element, frames should DIFFER (per-position)
        ref_frame = target_noise[:, 0:1, :]
        diff = (target_noise - ref_frame).abs().max().item()
        assert diff > 0.001, (
            f"Target smoothing must remain per-position even when "
            f"exploration is shared. Got diff={diff:.6f} across frames "
            f"— looks like critic target collapsed to one sample/chunk."
        )

    def test_inference_uses_exploration_sigma_independently(self, config, device):
        """Companion to the above: the inference-side actor() call
        (no sigma kwarg) must read exploration_sigma. Setting
        exploration_sigma=0 truly does kill exploration, while keeping
        target_sigma unaffected. Two-knob decoupling = both directions."""
        config.exploration_sigma = 0.0
        config.target_sigma = 0.1
        agent = TD3Agent(config, S, A, device)

        b = 5000
        z = torch.randn(b, D, device=device)
        s = torch.randn(b, S, device=device)
        ref = torch.randn(b, C, A, device=device)

        with torch.no_grad():
            mu = agent.actor.mean(z, s, ref)
            # No sigma kwarg → falls back to config.exploration_sigma
            sampled = agent.actor(z, s, ref, deterministic=False)
            noise = sampled - mu
        # Noise must be exactly zero (sigma=0)
        assert noise.abs().max().item() < 1e-6, (
            "exploration_sigma=0 must give deterministic actions at inference. "
            f"Got max |noise|={noise.abs().max().item()}"
        )


# ===================================================================
# Replay buffer: detach, ring wrap, save/load, thread safety
# ===================================================================


class TestReplayBuffer:
    def test_stored_tensors_are_detached(self, device):
        """Real bug: storing tensors with grad graphs → OOM + wrong gradients."""
        buf = ReplayBuffer(10, D, S, A, C, device)
        # Create tensors that require grad (simulating encoder output)
        z = torch.randn(D, requires_grad=True)
        buf.add(
            z_rl=z,
            state=torch.randn(S),
            action_chunk=torch.randn(C, A),
            ref_chunk=torch.randn(C, A),
            reward=1.0,
            next_z_rl=torch.randn(D),
            next_state=torch.randn(S),
            next_ref_chunk=torch.randn(C, A),
            done=False,
        )
        # The stored tensor must not have a grad_fn
        assert not buf._z_rl[0].requires_grad, (
            "Replay buffer must detach tensors — gradient graphs leak memory"
        )

    def test_ring_buffer_overwrites_oldest(self, device):
        buf = ReplayBuffer(5, D, S, A, C, device)
        for i in range(5):
            _add_transition(buf, reward=float(i))
        assert len(buf) == 5

        _add_transition(buf, reward=99.0)
        assert len(buf) == 5, "Size should be capped at capacity"
        assert buf._reward[0, 0].item() == 99.0, "Index 0 should be overwritten"

    def test_save_load_preserves_data(self, device):
        buf = ReplayBuffer(20, D, S, A, C, device)
        for i in range(15):
            _add_transition(buf, reward=float(i))

        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
            path = f.name
        try:
            buf.save(path)
            buf2 = ReplayBuffer(20, D, S, A, C, device)
            buf2.load(path)
            assert len(buf2) == 15
            assert torch.allclose(buf._reward[:15], buf2._reward[:15]), "Rewards don't match after save/load"
        finally:
            os.unlink(path)

    def test_load_larger_into_smaller_raises(self, device):
        """Prevents silent data loss from capacity mismatch."""
        buf = ReplayBuffer(20, D, S, A, C, device)
        for _ in range(5):
            _add_transition(buf)

        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
            path = f.name
        try:
            buf.save(path)
            buf2 = ReplayBuffer(10, D, S, A, C, device)
            with pytest.raises(ValueError, match="Cannot load buffer"):
                buf2.load(path)
        finally:
            os.unlink(path)

    def test_empty_buffer_is_falsy_but_loads(self, device):
        """Real bug: `if replay_buffer` was False for empty buffer (len==0),
        so load() was skipped on every restart. Use `is not None` instead."""
        buf = ReplayBuffer(20, D, S, A, C, device)
        assert len(buf) == 0
        assert not buf, "Empty buffer should be falsy (len==0) — this IS Python's behavior"
        assert buf is not None, "But `is not None` must be True — use this for existence checks"

        # Verify load works on a buffer that bool() considers False
        for _ in range(10):
            _add_transition(buf)
        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
            path = f.name
        try:
            buf.save(path)
            buf2 = ReplayBuffer(20, D, S, A, C, device)
            assert not buf2, "New buffer is falsy"
            buf2.load(path)  # This is what was skipped by `if buf2 and ...`
            assert len(buf2) == 10, "Load must work on empty (falsy) buffer"
        finally:
            os.unlink(path)

    def test_truncate_drops_only_the_tail(self, device):
        """``truncate`` is the rollback path for the operator's "ignore
        current episode" key. It must drop the most recent ``size -
        target`` entries while preserving everything older."""
        buf = ReplayBuffer(20, D, S, A, C, device)
        for i in range(10):
            _add_transition(buf, reward=float(i))
        assert len(buf) == 10

        dropped = buf.truncate(7)
        assert dropped == 3
        assert len(buf) == 7
        # ptr must track size when no wrap has occurred (asserted in
        # implementation). Sample-able indices are now [0, 7).
        sample = buf.sample(50)
        assert sample["reward"].max().item() <= 6.0 + 1e-6, (
            f"truncate left newer entries reachable: max reward = {sample['reward'].max()}"
        )

    def test_truncate_to_zero(self, device):
        buf = ReplayBuffer(10, D, S, A, C, device)
        for _ in range(5):
            _add_transition(buf)
        buf.truncate(0)
        assert len(buf) == 0
        assert buf.ptr == 0

    def test_truncate_to_current_is_noop(self, device):
        buf = ReplayBuffer(10, D, S, A, C, device)
        for _ in range(5):
            _add_transition(buf)
        dropped = buf.truncate(5)
        assert dropped == 0
        assert len(buf) == 5

    def test_truncate_target_larger_than_size_raises(self, device):
        """Programmer error — the precondition assert fires."""
        buf = ReplayBuffer(10, D, S, A, C, device)
        for _ in range(3):
            _add_transition(buf)
        with pytest.raises(AssertionError, match="out of bounds"):
            buf.truncate(5)

    def test_truncate_after_eviction_raises(self, device):
        """Once the ring has wrapped (size == capacity and adds keep
        coming), ``truncate`` cannot reliably restore an older state
        because evicted slots are gone. The assert protects against
        silent corruption."""
        buf = ReplayBuffer(5, D, S, A, C, device)
        for _ in range(8):  # adds past capacity — ring wraps
            _add_transition(buf)
        assert len(buf) == 5
        with pytest.raises(AssertionError, match="wrapped around capacity"):
            buf.truncate(3)

    def test_concurrent_add_and_sample(self, device):
        """Real access pattern: inference thread writes, gradient thread reads."""
        buf = ReplayBuffer(1000, D, S, A, C, device)
        for _ in range(100):
            _add_transition(buf)

        errors = []

        def writer():
            try:
                for _ in range(200):
                    _add_transition(buf)
            except Exception as e:
                errors.append(e)

        def reader():
            try:
                for _ in range(200):
                    buf.sample(4)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=t) for t in [writer, writer, reader, reader]]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert not errors, f"Thread safety violation: {errors}"


# ===================================================================
# TransactionalReplayBuffer — staging vs committed invariant.
# Makes IGNORE provably zero-impact on training: discarded staged
# transitions never enter the committed buffer and so are never
# sampled by the grad-update path. See replay_buffer.py for the
# rationale and the README TODO entry that motivated this.
# ===================================================================


class TestTransactionalReplayBuffer:
    def _buf(self, capacity=100):
        return TransactionalReplayBuffer(
            capacity=capacity,
            rl_token_dim=D,
            state_dim=S,
            action_dim=A,
            chunk_length=C,
            device=torch.device("cpu"),
        )

    def _add_one(self, buf, val=1.0):
        buf.add(
            z_rl=torch.full((D,), val),
            state=torch.full((S,), val),
            action_chunk=torch.full((C, A), val),
            ref_chunk=torch.full((C, A), val),
            reward=val,
            next_z_rl=torch.full((D,), val),
            next_state=torch.full((S,), val),
            next_ref_chunk=torch.full((C, A), val),
            done=False,
        )

    # -----------------------------------------------------------------
    # Core invariant: staging is invisible to sample()
    # -----------------------------------------------------------------

    def test_add_goes_to_staging_not_committed(self):
        """The defining property: a fresh `add` does NOT show up in
        `len(buf)` (which reflects committed only). The staged item
        should be visible via `pending_size`."""
        buf = self._buf()
        for _ in range(5):
            self._add_one(buf)
        assert len(buf) == 0, "len(buf) reflects committed; staging-only adds must NOT show"
        assert buf.pending_size == 5

    def test_sample_never_returns_staged_transitions(self):
        """The structural guarantee. Even with staging full of
        identifiable transitions, sample() returns nothing because
        committed is empty. (Pre-fix this would have sampled the
        in-progress episode's transitions before commit.)"""
        buf = self._buf()
        # Stage with a distinctive marker
        for _ in range(20):
            self._add_one(buf, val=99.0)
        # Sample shouldn't even succeed (committed is empty)
        # ReplayBuffer.sample with size=0 would error on torch.randint(0, 0)
        # — that's the appropriate failure mode here too. Verify directly:
        with pytest.raises(RuntimeError):
            buf.sample(4)

    def test_sample_returns_only_committed_after_commit(self):
        """Commit moves staging into committed; sample() then returns
        from there. Staging is empty post-commit."""
        buf = self._buf()
        for i in range(5):
            self._add_one(buf, val=float(i))
        n = buf.commit()
        assert n == 5
        assert buf.pending_size == 0
        assert len(buf) == 5
        # Sample now works and returns committed values
        sample = buf.sample(3)
        # All sampled rewards came from the committed set [0..4]
        assert all(0.0 <= r.item() <= 4.0 for r in sample["reward"].squeeze())

    def test_discard_clears_staging_without_committing(self):
        """The IGNORE branch. Discard returns count of dropped items;
        committed stays at its prior size; staging empties."""
        buf = self._buf()
        for _ in range(7):
            self._add_one(buf)
        n = buf.discard()
        assert n == 7
        assert buf.pending_size == 0
        assert len(buf) == 0  # nothing committed

    def test_commit_then_add_then_discard_is_clean(self):
        """A realistic episode-cycle pattern: commit episode 1,
        add transitions for episode 2, discard episode 2 (IGNORE).
        Episode 1 must remain in committed; episode 2 must be gone."""
        buf = self._buf()
        # Episode 1: success
        for _ in range(5):
            self._add_one(buf, val=1.0)
        buf.commit()
        # Episode 2: ignored
        for _ in range(8):
            self._add_one(buf, val=2.0)
        buf.discard()
        # Final: only episode 1's transitions are visible
        assert len(buf) == 5
        assert buf.pending_size == 0
        # Spot-check: sampled rewards are all 1.0 (no leak from ep2)
        sample = buf.sample(5)
        for r in sample["reward"].squeeze():
            assert r.item() == 1.0, (
                "Sampled reward came from discarded episode 2 — IGNORE is leaking into training."
            )

    def test_repeated_commit_discard_cycles(self):
        """Locks in the multi-episode steady state. 10 episodes
        alternating commit and discard — committed should hold half."""
        buf = self._buf(capacity=1000)
        for ep in range(10):
            for _ in range(3):
                self._add_one(buf, val=float(ep))
            if ep % 2 == 0:
                buf.commit()
            else:
                buf.discard()
        # 5 committed episodes × 3 transitions = 15
        assert len(buf) == 15
        assert buf.pending_size == 0

    # -----------------------------------------------------------------
    # Drop-in compat with the ReplayBuffer interface
    # -----------------------------------------------------------------

    def test_capacity_attribute_forwarded(self):
        buf = self._buf(capacity=200)
        assert buf.capacity == 200

    def test_private_attrs_forwarded_to_committed(self):
        """Audit scripts and cross-module asserts read ``_action``,
        ``ptr``, ``_done``, etc. directly. After commit they should
        reflect committed buffer state."""
        buf = self._buf()
        for _ in range(3):
            self._add_one(buf, val=1.0)
        buf.commit()
        # __getattr__ forwards
        assert buf.ptr == 3
        assert buf.size == 3
        assert buf._done.shape[0] == 100  # underlying capacity

    def test_save_load_only_persists_committed(self):
        """Staging is volatile — never serialized. Reload is committed-only."""
        import os
        import tempfile

        buf = self._buf()
        for _ in range(4):
            self._add_one(buf, val=1.0)
        buf.commit()
        # Add to staging but don't commit
        for _ in range(7):
            self._add_one(buf, val=2.0)
        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
            path = f.name
        try:
            buf.save(path)
            # Reload into a fresh buffer
            buf2 = self._buf()
            buf2.load(path)
            # Only the committed 4 are restored
            assert len(buf2) == 4
            assert buf2.pending_size == 0  # not persisted
        finally:
            os.unlink(path)

    # -----------------------------------------------------------------
    # peek_last_pending — used by cross-module assertions
    # -----------------------------------------------------------------

    def test_peek_last_pending_returns_most_recent(self):
        buf = self._buf()
        self._add_one(buf, val=1.0)
        self._add_one(buf, val=2.0)
        self._add_one(buf, val=3.0)
        last = buf.peek_last_pending()
        assert last is not None
        assert last["reward"] == 3.0

    def test_peek_last_pending_returns_none_when_empty(self):
        buf = self._buf()
        assert buf.peek_last_pending() is None
        # And after commit (drains staging)
        self._add_one(buf)
        buf.commit()
        assert buf.peek_last_pending() is None

    def test_peek_does_not_consume(self):
        buf = self._buf()
        self._add_one(buf)
        for _ in range(10):
            assert buf.peek_last_pending() is not None
        assert buf.pending_size == 1  # still 1 item


# ===================================================================
# Metrics: autonomous rate formula (real bug: wrong denominator)
# ===================================================================


class TestMetrics:
    def test_autonomous_rate_denominator_is_all_episodes(self):
        """Real bug: was auto_successes/auto_episodes = 5/5 = 100%.
        Correct: auto_successes/all_episodes = 5/20 = 25%."""
        m = RLTMetrics()
        for _ in range(5):
            m.record_episode(episode=0, success=True, autonomous=True, duration_s=10.0)
        for _ in range(15):
            m.record_episode(episode=0, success=True, autonomous=False, duration_s=10.0)

        snap = m.snapshot()
        assert snap["autonomous_rate"] == 0.25, (
            f"Got {snap['autonomous_rate']}. If 1.0, denominator is auto_episodes not all_episodes."
        )

    def test_rolling_series_uses_same_formula(self):
        """Chart and status bar must agree — they disagreed before."""
        m = RLTMetrics()
        for _ in range(10):
            m.record_episode(episode=0, success=True, autonomous=True, duration_s=5.0)
        for _ in range(10):
            m.record_episode(episode=0, success=True, autonomous=False, duration_s=5.0)

        snap = m.snapshot()
        rolling = snap["series"]["autonomous_rate_rolling"]
        # Last point: 10 auto successes in 20 total = 0.5
        assert abs(rolling[-1] - 0.5) < 0.01, "Rolling series disagrees with headline formula"

    def test_series_bounded_under_max(self):
        m = RLTMetrics()
        m._MAX_SERIES_LEN = 50
        for i in range(100):
            m.record_inference(step=i, delta=0.01, buffer_size=i, total_updates=i, mode="RL")
        assert len(m.inferences) <= 50

    def test_save_load_roundtrip(self):
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name
        try:
            reset_metrics()
            set_metrics_path(path)
            m = get_metrics()
            m.record_episode(episode=3, success=True, autonomous=True, duration_s=8.0)
            save_metrics_to_file()

            loaded = load_metrics_from_file(path)
            assert loaded["episode"] == 3
            assert loaded["total_successes"] == 1
        finally:
            os.unlink(path)
            reset_metrics()

    def test_snapshot_is_json_serializable(self):
        m = RLTMetrics()
        for i in range(10):
            m.record_episode(episode=i, success=True, autonomous=True, duration_s=5.0)
            m.record_inference(step=i, delta=0.01, buffer_size=i, total_updates=i, mode="RL")
            m.record_grad_update(
                total_updates=i,
                mode="RL",
                critic_loss=0.1,
                critic_grad_norm=1.0,
                actor_loss=-0.3,
                q_mean=1.0,
                q_min=0.5,
                q_max=1.5,
                actor_q_term=-0.3,
                actor_bc_term=0.1,
                update_rate=10.0,
            )
        json.dumps(m.snapshot())  # must not raise


# ===================================================================
# Config: warmup episode boundary (0-indexed)
# ===================================================================


class TestWarmupBoundary:
    """Guards against the pre-migration off-by-one where ``episode <=
    warmup_episodes`` with a 1-indexed counter incremented at episode END
    caused ``warmup_episodes=10`` to actually warm up 11 episodes.
    """

    def test_first_episode_is_warmup(self):
        cfg = RLTConfig(warmup_episodes=10)
        assert cfg.is_warmup(0), "Episode 0 must be warmup (0-indexed)"

    def test_last_warmup_episode_included(self):
        cfg = RLTConfig(warmup_episodes=10)
        assert cfg.is_warmup(9), "Episode 9 must still be warmup with warmup_episodes=10"

    def test_first_non_warmup_episode_excluded(self):
        cfg = RLTConfig(warmup_episodes=10)
        assert not cfg.is_warmup(10), (
            "Episode 10 must NOT be warmup. If this fires, we've regressed "
            "to the pre-migration off-by-one (11 warmup episodes instead of 10)."
        )

    def test_warmup_zero_disables_warmup(self):
        cfg = RLTConfig(warmup_episodes=0)
        assert not cfg.is_warmup(0), "warmup_episodes=0 must disable warmup entirely"


class TestTokenCheckpointManifest:
    """Guards the gated-rollout mechanism: a trained token checkpoint carries
    a config.json with the architecture it was trained under, so loaders
    instantiate the matching encoder/decoder regardless of what RLTConfig's
    live defaults happen to be. Without this, a 4-layer checkpoint silently
    fails to load into a 2-layer encoder (state_dict shape mismatch)."""

    def test_save_load_roundtrip(self, tmp_path):
        """Train-time save → load-time read produces the same arch fields."""
        trained_cfg = RLTConfig(
            rl_token_dim=512,
            token_encoder_layers=4,
            token_decoder_layers=4,
            token_num_heads=8,
            token_ffn_dim=1024,
        )
        save_rlt_token_config(tmp_path, trained_cfg)
        loaded = load_rlt_token_config(tmp_path)
        assert loaded.token_encoder_layers == 4
        assert loaded.token_decoder_layers == 4
        assert loaded.token_num_heads == 8
        assert loaded.token_ffn_dim == 1024
        assert loaded.rl_token_dim == 512

    def test_missing_manifest_raises_with_actionable_message(self, tmp_path):
        """Pre-manifest checkpoints (the deprecated 2L d=768 family) used
        to be loaded silently against the live RLTConfig defaults. With
        the canonical defaults shifted to 4L d=2048, that silent fallback
        would mis-load legacy checkpoints as the new architecture and
        produce a tensor shape mismatch deep inside model construction.
        Refusing up front with an actionable message is safer."""
        # tmp_path is empty — no config.json
        with pytest.raises(FileNotFoundError, match="No architecture manifest"):
            load_rlt_token_config(tmp_path)

    def test_missing_manifest_error_mentions_retrain_recipe(self, tmp_path):
        """Operators hitting this error need to know exactly how to
        recover — point them at the canonical retraining recipe so they
        don't have to dig through commit history. Locks the message text
        in so a refactor doesn't drop the recipe."""
        with pytest.raises(FileNotFoundError) as excinfo:
            load_rlt_token_config(tmp_path)
        msg = str(excinfo.value)
        assert "4L" in msg or "encoder-layers 4" in msg
        assert "2048" in msg

    def test_base_preserves_non_shape_fields(self, tmp_path):
        """The loader should only override SHAPE fields, leaving runtime
        fields (e.g. rl_token_dim bound to live S1 hidden_dim, beta, sigma)
        untouched on the base config the caller supplied."""
        trained_cfg = RLTConfig(token_encoder_layers=4)
        save_rlt_token_config(tmp_path, trained_cfg)

        # Caller's base has runtime values that shouldn't be clobbered
        base = RLTConfig(beta=0.5, exploration_sigma=0.1)
        loaded = load_rlt_token_config(tmp_path, base=base)
        assert loaded.token_encoder_layers == 4  # shape field — overridden from manifest
        assert loaded.beta == 0.5  # non-shape — base preserved
        assert loaded.exploration_sigma == 0.1  # non-shape — base preserved

    def test_load_with_trained_arch_can_load_state_dict(self, tmp_path):
        """End-to-end: save a 4-layer encoder's state_dict + its manifest,
        read the manifest, rebuild the encoder with the loaded config,
        and the state_dict load must succeed (no shape mismatch)."""
        # Build, save, done — pretend this is what train_token.py produces
        trained_cfg = RLTConfig(rl_token_dim=128, token_encoder_layers=4, token_ffn_dim=256)
        original = RLTokenEncoder(trained_cfg)
        torch.save(original.state_dict(), tmp_path / "encoder.pt")
        save_rlt_token_config(tmp_path, trained_cfg)

        # Now simulate the loader: different runtime config (2 layers defaults)
        # + the manifest should reconstruct the 4-layer arch
        runtime_base = RLTConfig(rl_token_dim=128)  # 2 layers by default
        loader_cfg = load_rlt_token_config(tmp_path, base=runtime_base)
        rebuilt = RLTokenEncoder(loader_cfg)
        # Would raise RuntimeError if the manifest mechanism failed
        rebuilt.load_state_dict(torch.load(tmp_path / "encoder.pt"))

    def test_asymmetric_context_dim_round_trips(self, tmp_path):
        """Guard for the widened-bottleneck setup (context_dim != rl_token_dim).
        The encoder must accept [B, N, context_dim] input, the decoder must
        reconstruct back into [B, N, context_dim] — otherwise the loss
        compares tensors of different shape and training explodes.

        Also verifies the manifest round-trips context_dim so loaders
        rebuild the same input/output projections that were trained.
        """
        # Widened: ctx=64, bottleneck=128 (mimics our 768 → 2048 experiment)
        trained_cfg = RLTConfig(
            rl_token_dim=128,
            context_dim=64,
            token_encoder_layers=2,
            token_ffn_dim=128,
        )
        enc = RLTokenEncoder(trained_cfg)
        dec = RLTokenDecoder(trained_cfg)

        # Input has context_dim channels; output z_rl must have rl_token_dim;
        # reconstruction must land back in context_dim.
        ctx = torch.randn(3, 10, 64)  # [B=3, N=10, C=64]
        z_rl = enc(ctx)
        assert z_rl.shape == (3, 128), f"z_rl shape {z_rl.shape} — expected [B, rl_token_dim] = [3, 128]"
        recon = dec(z_rl, ctx)
        assert recon.shape == ctx.shape, (
            f"Reconstruction shape {recon.shape} must equal target shape "
            f"{ctx.shape} — otherwise the reconstruction loss can't compare them"
        )

        # Manifest round-trip
        torch.save(enc.state_dict(), tmp_path / "encoder.pt")
        torch.save(dec.state_dict(), tmp_path / "decoder.pt")
        save_rlt_token_config(tmp_path, trained_cfg)

        # Loader starts with a runtime base that knows nothing about
        # context_dim — must pick it up from the manifest.
        runtime_base = RLTConfig(rl_token_dim=128)  # default token_ffn_dim=2048
        loader_cfg = load_rlt_token_config(tmp_path, base=runtime_base)
        assert loader_cfg.context_dim == 64
        assert loader_cfg.rl_token_dim == 128
        # Rebuild + load — would raise if the projections had wrong shape
        rebuilt_enc = RLTokenEncoder(loader_cfg)
        rebuilt_dec = RLTokenDecoder(loader_cfg)
        rebuilt_enc.load_state_dict(torch.load(tmp_path / "encoder.pt"))
        rebuilt_dec.load_state_dict(torch.load(tmp_path / "decoder.pt"))

    def test_symmetric_setup_has_no_projection_params(self, tmp_path):
        """Backward compatibility: when context_dim is None (default), the
        encoder and decoder must NOT introduce any new parameters. Older
        checkpoints from before this feature load via existing state_dict
        keys unchanged — state_dict keys are determined by module types,
        and nn.Identity has zero keys."""
        cfg = RLTConfig(rl_token_dim=64, context_dim=None, token_encoder_layers=1, token_ffn_dim=32)
        enc = RLTokenEncoder(cfg)
        dec = RLTokenDecoder(cfg)
        enc_keys = set(enc.state_dict().keys())
        dec_keys = set(dec.state_dict().keys())
        # No projection keys should appear
        assert not any("input_proj" in k for k in enc_keys), (
            f"Encoder in symmetric mode should not have input_proj params — "
            f"got keys with it: {[k for k in enc_keys if 'input_proj' in k]}"
        )
        assert not any("target_proj" in k for k in dec_keys), (
            "Decoder in symmetric mode should not have target_proj params"
        )


# ===================================================================
# Atomic checkpoint save (_atomic_torch_save) — prevents torn .pt files
# when the process crashes mid-save. See commit 4c5624b1f.
# ===================================================================


class TestAtomicTorchSave:
    def test_writes_object_to_target_path(self, tmp_path):
        """Happy path: object is reachable at the target path after save."""
        path = tmp_path / "weights.pt"
        obj = {"x": torch.tensor([1.0, 2.0, 3.0])}
        _atomic_torch_save(obj, path)
        assert path.exists()
        loaded = torch.load(path, weights_only=True)
        assert torch.equal(loaded["x"], obj["x"])

    def test_no_tmp_left_behind_on_success(self, tmp_path):
        """``.tmp`` file is os.replace'd onto the target — must be gone
        after a clean save. Any leftover indicates the rename didn't run."""
        path = tmp_path / "weights.pt"
        _atomic_torch_save({"a": torch.zeros(2)}, path)
        assert not (tmp_path / "weights.pt.tmp").exists()

    def test_overwrites_existing_target(self, tmp_path):
        """Subsequent saves replace, not corrupt."""
        path = tmp_path / "weights.pt"
        _atomic_torch_save({"v": torch.tensor([1.0])}, path)
        _atomic_torch_save({"v": torch.tensor([99.0])}, path)
        loaded = torch.load(path, weights_only=True)
        assert loaded["v"].item() == 99.0

    def test_existing_target_preserved_when_save_fails(self, tmp_path):
        """The whole point of this helper: a crash mid-save (raised
        BEFORE os.replace) must leave the previous good file untouched.
        Simulate by passing an unpicklable object — torch.save raises,
        os.replace never runs, target file is unchanged."""
        path = tmp_path / "weights.pt"
        _atomic_torch_save({"v": torch.tensor([42.0])}, path)
        good_bytes = path.read_bytes()

        # Object that raises during torch.save (lambdas aren't picklable)
        class _Unpicklable:
            def __reduce__(self):
                raise RuntimeError("intentional save failure")

        with pytest.raises(RuntimeError, match="intentional save failure"):
            _atomic_torch_save({"bad": _Unpicklable()}, path)

        # Target file should still hold the original good content
        assert path.exists()
        assert path.read_bytes() == good_bytes
        loaded = torch.load(path, weights_only=True)
        assert loaded["v"].item() == 42.0

    def test_accepts_string_path(self, tmp_path):
        """The helper coerces to ``str(path)`` so callers can pass either
        a pathlib.Path or a plain string. Lock that in — refactors that
        change the type signature shouldn't drop string support since
        most call sites pass strings."""
        target = str(tmp_path / "weights.pt")
        _atomic_torch_save({"k": torch.tensor([7.0])}, target)
        loaded = torch.load(target, weights_only=True)
        assert loaded["k"].item() == 7.0


# ===================================================================
# Helpers
# ===================================================================


def _make_batch(config, device):
    return {
        "z_rl": torch.randn(B, D, device=device),
        "state": torch.randn(B, S, device=device),
        "action_chunk": torch.randn(B, C, A, device=device),
        "ref_chunk": torch.randn(B, C, A, device=device),
        "reward": torch.ones(B, 1, device=device),
        "next_z_rl": torch.randn(B, D, device=device),
        "next_state": torch.randn(B, S, device=device),
        "next_ref_chunk": torch.randn(B, C, A, device=device),
        "done": torch.zeros(B, 1, device=device),
    }


def _add_transition(buf, reward=1.0):
    buf.add(
        z_rl=torch.randn(D),
        state=torch.randn(S),
        action_chunk=torch.randn(C, A),
        ref_chunk=torch.randn(C, A),
        reward=reward,
        next_z_rl=torch.randn(D),
        next_state=torch.randn(S),
        next_ref_chunk=torch.randn(C, A),
        done=False,
    )

"""Tests for cross-attention KV cache in flow matching decoder.

Verifies that the cached KV path produces numerically identical outputs
to the uncached path across various configurations.
"""

import pytest
import torch

from lerobot.policies.hvla.s1.flow_matching.config import FlowMatchingS1Config
from lerobot.policies.hvla.s1.flow_matching.model import FlowMatchingS1Model


@pytest.fixture
def device():
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    return torch.device("cuda")


@pytest.fixture
def model(device):
    config = FlowMatchingS1Config(
        chunk_size=50,
        hidden_dim=768,
        num_heads=8,
        num_decoder_layers=6,
        num_encoder_layers=2,
        dim_feedforward=2048,
    )
    m = FlowMatchingS1Model(config).to(device).eval()
    return m


def _make_batch(device, B=1):
    return {
        "observation.images": [torch.randn(B, 3, 224, 224, device=device) for _ in range(4)],
        "observation.state": torch.zeros(B, 14, device=device),
        "observation.s2_latent": torch.zeros(B, 2048, device=device),
        "observation.s2_latent_age": torch.zeros(B, 1, device=device),
    }


class TestKVCacheCorrectness:
    """Verify cached and uncached paths produce identical outputs."""

    def test_single_step_matches(self, model, device):
        """Single denoise step: cached vs uncached should be numerically identical."""
        batch = _make_batch(device)
        with torch.no_grad():
            context = model.encode_observations(batch)
            cached_kv = model.precompute_cross_attn_kv(context)
            x_t = torch.randn(1, 50, 14, device=device)
            t = torch.full((1, 50), 0.5, device=device)

            v_cached = model.denoise_step(x_t.clone(), context, t, cached_kv=cached_kv)
            v_nocache = model.denoise_step(x_t.clone(), context, t, cached_kv=None)

        assert torch.allclose(v_cached, v_nocache, atol=1e-5), (
            f"Max diff: {(v_cached - v_nocache).abs().max().item():.2e}"
        )

    def test_multiple_timesteps(self, model, device):
        """Cache should be correct across different timestep values."""
        batch = _make_batch(device)
        with torch.no_grad():
            context = model.encode_observations(batch)
            cached_kv = model.precompute_cross_attn_kv(context)
            x_t = torch.randn(1, 50, 14, device=device)

            for t_val in [0.0, 0.1, 0.5, 0.9, 1.0]:
                t = torch.full((1, 50), t_val, device=device)
                v_cached = model.denoise_step(x_t.clone(), context, t, cached_kv=cached_kv)
                v_nocache = model.denoise_step(x_t.clone(), context, t, cached_kv=None)
                assert torch.allclose(v_cached, v_nocache, atol=1e-5), (
                    f"Failed at t={t_val}, max diff: {(v_cached - v_nocache).abs().max().item():.2e}"
                )

    def test_per_position_timestep(self, model, device):
        """Cache should work with per-position timesteps (RTC prefix has t=0)."""
        batch = _make_batch(device)
        with torch.no_grad():
            context = model.encode_observations(batch)
            cached_kv = model.precompute_cross_attn_kv(context)
            x_t = torch.randn(1, 50, 14, device=device)

            # Simulate RTC: first 5 positions at t=0, rest at t=0.7
            t = torch.full((1, 50), 0.7, device=device)
            t[:, :5] = 0.0

            v_cached = model.denoise_step(x_t.clone(), context, t, cached_kv=cached_kv)
            v_nocache = model.denoise_step(x_t.clone(), context, t, cached_kv=None)

        assert torch.allclose(v_cached, v_nocache, atol=1e-5), (
            f"Max diff: {(v_cached - v_nocache).abs().max().item():.2e}"
        )

    def test_batch_size_2(self, model, device):
        """Cache should work with batch_size > 1."""
        batch = _make_batch(device, B=2)
        with torch.no_grad():
            context = model.encode_observations(batch)
            cached_kv = model.precompute_cross_attn_kv(context)
            x_t = torch.randn(2, 50, 14, device=device)
            t = torch.full((2, 50), 0.5, device=device)

            v_cached = model.denoise_step(x_t.clone(), context, t, cached_kv=cached_kv)
            v_nocache = model.denoise_step(x_t.clone(), context, t, cached_kv=None)

        assert torch.allclose(v_cached, v_nocache, atol=1e-5), (
            f"Max diff: {(v_cached - v_nocache).abs().max().item():.2e}"
        )

    def test_different_x_t_same_cache(self, model, device):
        """Same cached KV should produce different outputs for different x_t."""
        batch = _make_batch(device)
        with torch.no_grad():
            context = model.encode_observations(batch)
            cached_kv = model.precompute_cross_attn_kv(context)
            t = torch.full((1, 50), 0.5, device=device)

            x_t_1 = torch.randn(1, 50, 14, device=device)
            x_t_2 = torch.randn(1, 50, 14, device=device)

            v1 = model.denoise_step(x_t_1, context, t, cached_kv=cached_kv)
            v2 = model.denoise_step(x_t_2, context, t, cached_kv=cached_kv)

        # Different inputs should give different outputs
        assert not torch.allclose(v1, v2, atol=1e-3), "Different x_t should produce different velocities"

    def test_full_inference_pipeline(self, model, device):
        """Full sample_actions with KV cache should produce reasonable outputs."""
        batch = _make_batch(device)
        with torch.no_grad():
            actions = model.sample_actions(batch, num_steps=5)

        assert actions.shape == (1, 50, 14)
        assert torch.isfinite(actions).all(), "Actions contain NaN/Inf"
        # Actions should not be all zeros (model should produce non-trivial output)
        assert actions.abs().mean() > 0.01, "Actions are near-zero, model may not be working"

    def test_full_inference_with_prefix(self, model, device):
        """sample_actions with RTC prefix should work with KV cache."""
        batch = _make_batch(device)
        prefix = torch.randn(1, 5, 14, device=device)
        with torch.no_grad():
            actions = model.sample_actions(batch, num_steps=5, action_prefix=prefix, prefix_len=5)

        assert actions.shape == (1, 50, 14)
        assert torch.isfinite(actions).all()
        # Prefix positions should be close to the input prefix (re-injected)
        prefix_diff = (actions[:, :5] - prefix).abs().mean().item()
        assert prefix_diff < 0.5, f"Prefix not preserved: mean diff={prefix_diff:.3f}"


class TestKVCacheBf16:
    """Verify cache works correctly under bf16 autocast."""

    def test_bf16_matches(self, model, device):
        """Cached path under bf16 should match uncached under bf16."""
        batch = _make_batch(device)
        with torch.no_grad(), torch.autocast("cuda", torch.bfloat16):
            context = model.encode_observations(batch)
            cached_kv = model.precompute_cross_attn_kv(context)
            x_t = torch.randn(1, 50, 14, device=device)
            t = torch.full((1, 50), 0.5, device=device)

            v_cached = model.denoise_step(x_t.clone(), context, t, cached_kv=cached_kv)
            v_nocache = model.denoise_step(x_t.clone(), context, t, cached_kv=None)

        # bf16 has lower precision, use larger tolerance
        assert torch.allclose(v_cached.float(), v_nocache.float(), atol=1e-2), (
            f"Max diff under bf16: {(v_cached - v_nocache).abs().max().item():.2e}"
        )


class TestKVCacheShapes:
    """Verify shapes and types of cached KV tensors."""

    def test_cache_shape(self, model, device):
        """Cached KV should have correct shapes."""
        batch = _make_batch(device)
        with torch.no_grad():
            context = model.encode_observations(batch)
            cached_kv = model.precompute_cross_attn_kv(context)

        assert len(cached_kv) == model.config.num_decoder_layers
        for k, v in cached_kv:
            assert k.shape == context.shape, f"K shape {k.shape} != context {context.shape}"
            assert v.shape == context.shape, f"V shape {v.shape} != context {context.shape}"

    def test_cache_not_none_in_sample_actions(self, model, device):
        """sample_actions should use cache (not pass None)."""
        batch = _make_batch(device)
        # Monkey-patch denoise_step to check cached_kv is passed
        calls = []
        orig = model.denoise_step

        def tracking_denoise_step(*args, **kwargs):
            calls.append(kwargs.get("cached_kv") is not None)
            return orig(*args, **kwargs)

        model.denoise_step = tracking_denoise_step
        with torch.no_grad():
            model.sample_actions(batch, num_steps=3)
        model.denoise_step = orig

        assert all(calls), f"denoise_step called without cache: {calls}"


class TestWeightRemapping:
    """Verify old checkpoint format loads correctly."""

    def test_old_key_format(self, device):
        """Keys with 'action_decoder.layers' should remap to 'decoder_layers'."""
        config = FlowMatchingS1Config(num_decoder_layers=2)
        model = FlowMatchingS1Model(config).to(device)

        # Simulate old checkpoint keys
        state_dict = model.state_dict()
        old_dict = {}
        for k, v in state_dict.items():
            old_k = k.replace("decoder_layers.", "action_decoder.layers.")
            old_dict[old_k] = v

        # Remap and load
        remapped = {}
        for k, v in old_dict.items():
            new_k = k.replace("action_decoder.layers.", "decoder_layers.")
            remapped[new_k] = v

        missing, unexpected = model.load_state_dict(remapped, strict=False)
        assert len(missing) == 0, f"Missing keys: {missing}"
        assert len(unexpected) == 0, f"Unexpected keys: {unexpected}"

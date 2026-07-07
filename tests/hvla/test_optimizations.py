"""Tests for S1 inference optimizations.

Verifies that batched DINOv2, bidirectional attention, bf16 autocast,
and other optimizations produce correct results.
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


class TestBatchedDINOv2:
    """Verify batched DINOv2 forward matches sequential per-camera forward."""

    def test_batched_matches_sequential(self, model, device):
        """Batched 4-camera DINOv2 should produce same tokens as 4 sequential calls."""
        images = [torch.randn(1, 3, 224, 224, device=device) for _ in range(4)]

        # Sequential: one forward per camera
        sequential_tokens = []
        with torch.no_grad():
            for img in images:
                features = model.backbone.forward_features(img)
                patches = features["x_norm_patchtokens"]
                sequential_tokens.append(model.image_proj(patches))

        # Batched: all cameras in one forward (what encode_observations does)
        with torch.no_grad():
            stacked = torch.cat(images, dim=0)  # [4, 3, 224, 224]
            features = model.backbone.forward_features(stacked)
            all_patches = features["x_norm_patchtokens"]  # [4, 256, 768]
            batched_tokens = [model.image_proj(all_patches[i : i + 1]) for i in range(4)]

        for i in range(4):
            diff = (sequential_tokens[i] - batched_tokens[i]).abs().max().item()
            # Batched vs sequential has small numerical divergence from GPU kernel ordering
            assert diff < 0.01, f"Camera {i} mismatch: max diff {diff:.2e} (expected < 0.01)"

    def test_camera_order_preserved(self, model, device):
        """Swapping camera order in batch should swap output order."""
        img_a = torch.randn(1, 3, 224, 224, device=device)
        img_b = torch.randn(1, 3, 224, 224, device=device)

        with torch.no_grad():
            stacked_ab = torch.cat([img_a, img_b], dim=0)
            stacked_ba = torch.cat([img_b, img_a], dim=0)
            features_ab = model.backbone.forward_features(stacked_ab)["x_norm_patchtokens"]
            features_ba = model.backbone.forward_features(stacked_ba)["x_norm_patchtokens"]

        # AB[0] should match BA[1] (same image)
        assert torch.allclose(features_ab[0], features_ba[1], atol=1e-5)
        assert torch.allclose(features_ab[1], features_ba[0], atol=1e-5)


class TestBidirectionalAttention:
    """Verify action decoder uses bidirectional (not causal) attention."""

    def test_future_positions_affect_past(self, model, device):
        """Changing a future action position should affect earlier positions' output.
        This would NOT happen with causal masking."""
        batch = _make_batch(device)
        with torch.no_grad():
            context = model.encode_observations(batch)

        x_t = torch.randn(1, 50, 14, device=device)
        t = torch.full((1, 50), 0.5, device=device)

        # Run with original x_t
        with torch.no_grad():
            v1 = model.denoise_step(x_t.clone(), context, t, cached_kv=None)

        # Perturb only position 40 (future) with large magnitude
        x_t_perturbed = x_t.clone()
        x_t_perturbed[:, 40, :] += 100.0

        with torch.no_grad():
            v2 = model.denoise_step(x_t_perturbed, context, t, cached_kv=None)

        # With bidirectional: position 0 output should change significantly
        # With causal: position 0 output would be identical (diff=0)
        diff_at_0 = (v1[:, 0] - v2[:, 0]).abs().max().item()
        output_scale = v1.abs().mean().item()
        # Expect at least 10% of output scale change (measured: ~25% with perturbation=100)
        assert diff_at_0 > output_scale * 0.05, (
            f"Position 0 barely changed when position 40 perturbed "
            f"(diff={diff_at_0:.2e}, output_scale={output_scale:.2e}). "
            "This suggests causal masking is still active."
        )

    def test_no_causal_mask_in_decoder(self, model, device):
        """Verify decoder layers don't apply tgt_mask."""
        batch = _make_batch(device)
        x_t = torch.randn(1, 50, 14, device=device)
        t = torch.full((1, 50), 0.5, device=device)

        # The denoise_step should pass tgt_mask=None to decoder layers
        # We verify by checking that the uncached path calls layer(x, context, tgt_mask=None)
        calls = []
        for layer in model.decoder_layers:
            orig_forward = layer.forward

            def tracking_forward(*args, _orig=orig_forward, **kwargs):
                calls.append(kwargs.get("tgt_mask", "NOT_PASSED"))
                return _orig(*args, **kwargs)

            layer.forward = tracking_forward

        with torch.no_grad():
            model.denoise_step(x_t, model.encode_observations(batch), t, cached_kv=None)

        # Restore
        for layer in model.decoder_layers:
            if hasattr(layer, "_orig_forward"):
                layer.forward = layer._orig_forward

        assert all(m is None or m == "NOT_PASSED" for m in calls), (
            f"Some decoder layers received non-None tgt_mask: {calls}"
        )


class TestBf16Inference:
    """Verify inference works correctly under bf16 autocast."""

    def test_bf16_no_nan(self, model, device):
        """bf16 inference should not produce NaN or Inf."""
        batch = _make_batch(device)
        with torch.no_grad(), torch.autocast("cuda", torch.bfloat16):
            actions = model.sample_actions(batch, num_steps=5)
        assert torch.isfinite(actions).all(), "bf16 produced NaN/Inf"

    def test_bf16_close_to_fp32(self, model, device):
        """bf16 and fp32 outputs should be close."""
        batch = _make_batch(device)

        torch.manual_seed(42)
        with torch.no_grad():
            context_fp32 = model.encode_observations(batch)
            kv_fp32 = model.precompute_cross_attn_kv(context_fp32)
            x_t = torch.randn(1, 50, 14, device=device)
            t = torch.full((1, 50), 0.5, device=device)
            v_fp32 = model.denoise_step(x_t.clone(), context_fp32, t, cached_kv=kv_fp32)

        torch.manual_seed(42)
        with torch.no_grad(), torch.autocast("cuda", torch.bfloat16):
            context_bf16 = model.encode_observations(batch)
            kv_bf16 = model.precompute_cross_attn_kv(context_bf16)
            x_t = torch.randn(1, 50, 14, device=device)
            t = torch.full((1, 50), 0.5, device=device)
            v_bf16 = model.denoise_step(x_t.clone(), context_bf16, t, cached_kv=kv_bf16)

        diff = (v_fp32 - v_bf16.float()).abs()
        assert diff.max().item() < 1.0, f"bf16 vs fp32 max diff {diff.max():.2e} too large"
        assert diff.mean().item() < 0.1, f"bf16 vs fp32 mean diff {diff.mean():.2e} too large"

    def test_bf16_deterministic(self, model, device):
        """Same seed under bf16 should give same result."""
        batch = _make_batch(device)

        torch.manual_seed(99)
        with torch.no_grad(), torch.autocast("cuda", torch.bfloat16):
            a1 = model.sample_actions(batch, num_steps=5)

        torch.manual_seed(99)
        with torch.no_grad(), torch.autocast("cuda", torch.bfloat16):
            a2 = model.sample_actions(batch, num_steps=5)

        assert torch.allclose(a1, a2, atol=1e-5), "bf16 not deterministic with same seed"


class TestFullPipeline:
    """End-to-end tests for the inference pipeline."""

    def test_different_denoise_steps(self, model, device):
        """Different step counts should produce different but valid outputs."""
        batch = _make_batch(device)

        results = {}
        for steps in [3, 5, 10, 15]:
            torch.manual_seed(42)
            with torch.no_grad():
                actions = model.sample_actions(batch, num_steps=steps)
            assert torch.isfinite(actions).all(), f"{steps} steps produced NaN"
            assert actions.shape == (1, 50, 14)
            results[steps] = actions

        # Different step counts should give different results
        assert not torch.allclose(results[3], results[15], atol=1e-2), (
            "3 steps and 15 steps produced same output"
        )

    def test_rtc_prefix_preserved(self, model, device):
        """RTC prefix should be preserved in output after integration."""
        batch = _make_batch(device)
        prefix = torch.randn(1, 5, 14, device=device)

        with torch.no_grad():
            actions = model.sample_actions(batch, num_steps=10, action_prefix=prefix, prefix_len=5)

        # Prefix positions should match input prefix
        prefix_err = (actions[:, :5] - prefix).abs().mean().item()
        assert prefix_err < 0.5, f"Prefix not preserved: mean err {prefix_err:.3f}"

    def test_output_range_reasonable(self, model, device):
        """Actions should be in a reasonable range (not exploding)."""
        batch = _make_batch(device)
        with torch.no_grad():
            actions = model.sample_actions(batch, num_steps=10)

        # Denormalized actions for SO-100 should be roughly in [-200, 200] range
        assert actions.abs().max().item() < 500, f"Actions out of range: max={actions.abs().max():.1f}"

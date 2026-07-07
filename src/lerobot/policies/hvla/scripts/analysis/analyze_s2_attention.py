"""Diagnose how much S1 actually attends to the S2 latent.

Three analyses:
  1. Ablation: real S2 vs. zero vs. random → action divergence
  2. Cross-attention weights: how much each action token attends to the S2 position
  3. Encoder self-attention: how much image/state tokens attend to S2 inside obs_encoder

Usage:
    python src/lerobot/policies/hvla/scripts/analysis/analyze_s2_attention.py \
        --s1-checkpoint outputs/flow_s1_hvla_v7/checkpoint-50000 \
        --s2-latents-dir outputs/s2_latents_fast7998 \
        --device cuda
"""

import argparse
import logging
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


def load_s1_policy(checkpoint_dir: str, device: str):
    from lerobot.policies.hvla.s1.flow_matching.config import FlowMatchingS1Config
    from lerobot.policies.hvla.s1.flow_matching.model import FlowMatchingS1Policy

    ckpt_dir = Path(checkpoint_dir)
    safetensors_path = ckpt_dir / "model.safetensors"
    if not safetensors_path.exists():
        # Maybe it's a file path directly
        safetensors_path = Path(checkpoint_dir)
        ckpt_dir = safetensors_path.parent

    config = FlowMatchingS1Config()
    policy = FlowMatchingS1Policy.from_pretrained(str(safetensors_path), config)
    policy.to(device)
    policy.eval()
    return policy


def load_sample_batch(latents_dir: str, device: str, n_samples: int = 8):
    """Load a few real S2 latents and create a dummy observation batch."""
    latents_dir = Path(latents_dir)

    # Load latents — accept direct .npy file or directory
    if latents_dir.is_file() and latents_dir.suffix == ".npy":
        all_latents = np.load(str(latents_dir))
        logger.info("Loaded latents from %s: shape %s", latents_dir, all_latents.shape)
    elif latents_dir.is_dir():
        latent_files = sorted(latents_dir.glob("*.npy"))
        if not latent_files:
            raise FileNotFoundError(f"No latent files found in {latents_dir}")
        all_latents = np.load(str(latent_files[0]))
        logger.info("Loaded latents from %s: shape %s", latent_files[0], all_latents.shape)
    else:
        raise FileNotFoundError(f"Not a valid latent file or directory: {latents_dir}")

    # Pick random samples
    indices = np.random.choice(len(all_latents), size=min(n_samples, len(all_latents)), replace=False)
    latents = torch.from_numpy(all_latents[indices]).float().to(device)

    B = latents.shape[0]

    # Create dummy observations (random images + state)
    # This is fine for measuring RELATIVE attention — we just need valid tensor shapes
    batch = {
        "observation.state": torch.randn(B, 14, device=device),
        "observation.s2_latent": latents,
    }

    # DINOv2 expects [B, 3, 224, 224] images
    for cam in [
        "observation.images.front",
        "observation.images.left_wrist",
        "observation.images.right_wrist",
        "observation.images.top",
    ]:
        batch[cam] = torch.randn(B, 3, 224, 224, device=device)

    return batch


# ─── Analysis 1: Ablation ──────────────────────────────────────────────


def ablation_study(policy, batch, device):
    """Compare actions: real S2 vs zero S2 vs random S2."""
    logger.info("\n" + "=" * 60)
    logger.info("ANALYSIS 1: S2 Latent Ablation")
    logger.info("=" * 60)

    model = policy.model
    real_latent = batch["observation.s2_latent"].clone()
    B = real_latent.shape[0]

    conditions = {
        "real": real_latent,
        "zero": torch.zeros_like(real_latent),
        "random": torch.randn_like(real_latent),
        "negative": -real_latent,
        "scaled_2x": real_latent * 2.0,
        "scaled_0.5x": real_latent * 0.5,
    }

    results = {}
    for name, latent in conditions.items():
        b = dict(batch)
        b["observation.s2_latent"] = latent
        with torch.no_grad():
            actions = model.sample_actions(b)  # [B, 50, 14]
        results[name] = actions
        logger.info(
            "  %12s: action mean=%.4f, std=%.4f, range=[%.4f, %.4f]",
            name,
            actions.mean().item(),
            actions.std().item(),
            actions.min().item(),
            actions.max().item(),
        )

    # Compute divergences relative to real
    real_actions = results["real"]
    logger.info("\nDivergence from real S2 latent:")
    logger.info("  %-12s  %10s  %10s  %10s  %10s", "condition", "L2_mean", "L2_max", "cos_sim", "rel_change%")
    for name, actions in results.items():
        if name == "real":
            continue
        diff = actions - real_actions
        l2_per_step = diff.norm(dim=-1)  # [B, 50]
        l2_mean = l2_per_step.mean().item()
        l2_max = l2_per_step.max().item()

        # Cosine similarity of flattened action sequences
        cos_sim = (
            F.cosine_similarity(actions.reshape(B, -1), real_actions.reshape(B, -1), dim=-1).mean().item()
        )

        # Relative change
        real_norm = real_actions.norm(dim=-1).mean().item()
        rel_change = l2_mean / max(real_norm, 1e-8) * 100

        logger.info("  %-12s  %10.4f  %10.4f  %10.4f  %10.1f%%", name, l2_mean, l2_max, cos_sim, rel_change)

    # Per-joint divergence (zero vs real) — shows which joints are most affected
    diff_zero = results["zero"] - real_actions  # [B, 50, 14]
    per_joint = diff_zero.abs().mean(dim=(0, 1))  # [14]
    joint_labels = [
        "L1",
        "L2",
        "L3",
        "L4",
        "L5",
        "L6",
        "L_grip",
        "R1",
        "R2",
        "R3",
        "R4",
        "R5",
        "R6",
        "R_grip",
    ]
    logger.info("\nPer-joint divergence (zero vs real):")
    for _j, (label, val) in enumerate(zip(joint_labels, per_joint, strict=False)):
        bar = "#" * int(val.item() * 50 / max(per_joint.max().item(), 1e-8))
        logger.info("  %7s: %.4f  %s", label, val.item(), bar)

    return results


# ─── Analysis 2: Cross-attention weights ────────────────────────────────


def cross_attention_analysis(policy, batch, device):
    """Extract cross-attention weights to see S2 latent contribution."""
    logger.info("\n" + "=" * 60)
    logger.info("ANALYSIS 2: Cross-Attention Weights on S2 Token")
    logger.info("=" * 60)

    model = policy.model

    # Step 1: encode context
    with torch.no_grad():
        context = model.encode_observations(batch)

    B, N_ctx, D = context.shape
    logger.info("Context shape: [%d, %d, %d]", B, N_ctx, D)
    logger.info("S2 token position: %d (last)", N_ctx - 1)

    nhead = model.config.num_heads
    head_dim = D // nhead

    # Step 2: precompute K,V
    with torch.no_grad():
        cached_kv = model.precompute_cross_attn_kv(context)

    # Step 3: run one denoise step and manually compute attention weights
    T = model.config.chunk_size
    x_t = torch.randn(B, T, model.config.action_dim, device=device)
    per_pos_t = torch.full((B, T), 1.0, device=device)  # t=1 (full noise)

    with torch.no_grad():
        # Replicate the denoise_step forward but capture cross-attention weights
        t_emb = _sinusoidal_embedding_external(per_pos_t, D, device)
        action_emb = model.action_in_proj(x_t)
        action_time = torch.cat([action_emb, t_emb], dim=-1)
        action_time = model.action_time_mlp_in(action_time)
        action_time = F.silu(action_time)
        action_time = model.action_time_mlp_out(action_time)
        pos_ids = torch.arange(T, device=device)
        action_time = action_time + model.action_pos_embed(pos_ids).unsqueeze(0)

        x = action_time
        layer_weights = []

        for i, layer in enumerate(model.decoder_layers):
            ck, cv = cached_kv[i]
            mha = layer.multihead_attn
            d = mha.embed_dim

            # Self-attention
            x = layer.norm1(x + layer.dropout1(layer.self_attn(x, x, x, need_weights=False)[0]))

            # Cross-attention — compute weights manually
            q = F.linear(
                x, mha.in_proj_weight[:d], mha.in_proj_bias[:d] if mha.in_proj_bias is not None else None
            )
            q = q.reshape(B, T, nhead, head_dim).transpose(1, 2)  # [B, nhead, T, head_dim]
            k = ck.reshape(B, N_ctx, nhead, head_dim).transpose(1, 2)  # [B, nhead, N_ctx, head_dim]
            v = cv.reshape(B, N_ctx, nhead, head_dim).transpose(1, 2)

            # Compute attention scores
            scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(head_dim)
            attn_weights = F.softmax(scores, dim=-1)  # [B, nhead, T, N_ctx]

            layer_weights.append(attn_weights)

            # Continue forward
            attn_out = torch.matmul(attn_weights, v)
            attn_out = attn_out.transpose(1, 2).reshape(B, T, D)
            attn_out = mha.out_proj(attn_out)
            x = layer.norm2(x + layer.dropout2(attn_out))
            x = layer.norm3(
                x + layer.dropout3(layer.linear2(layer.dropout(layer.activation(layer.linear1(x)))))
            )

    # Analyze attention on S2 token (last position) vs. images vs. state
    # Context layout: [image_patches(~1024), state(1), s2(1)]
    n_image_tokens = N_ctx - 2  # everything except state and S2
    s2_pos = N_ctx - 1
    state_pos = N_ctx - 2

    logger.info("\nPer-layer cross-attention weight on S2 token (mean across heads & action positions):")
    logger.info("  %-8s  %10s  %10s  %10s  %10s", "Layer", "S2_weight", "State_wt", "Image_wt", "S2_rank")

    for i, w in enumerate(layer_weights):
        # w: [B, nhead, T, N_ctx]
        # Average across batch and action positions
        w_mean = w.mean(dim=(0, 2))  # [nhead, N_ctx]

        s2_weight = w_mean[:, s2_pos].mean().item()  # mean across heads
        state_weight = w_mean[:, state_pos].mean().item()
        image_weight = w_mean[:, :n_image_tokens].mean().item()  # per-image-token average

        # Rank of S2 among all context tokens (1 = highest)
        w_per_token = w.mean(dim=(0, 1, 2))  # [N_ctx]
        rank = (w_per_token > w_per_token[s2_pos]).sum().item() + 1

        logger.info(
            "  Layer %d   %10.6f  %10.6f  %10.6f  %10d / %d",
            i,
            s2_weight,
            state_weight,
            image_weight,
            rank,
            N_ctx,
        )

    # Per-head breakdown for first and last layer
    for layer_idx in [0, len(layer_weights) - 1]:
        w = layer_weights[layer_idx]
        w_mean = w.mean(dim=(0, 2))  # [nhead, N_ctx]
        logger.info("\n  Layer %d per-head S2 weight:", layer_idx)
        for h in range(nhead):
            s2_w = w_mean[h, s2_pos].item()
            bar = "#" * int(s2_w * 1000)  # scale up since weights are small
            logger.info("    Head %d: %.6f  %s", h, s2_w, bar)

    # Attention over time (action positions 0..49)
    w_last = layer_weights[-1]  # last decoder layer
    s2_over_time = w_last[:, :, :, s2_pos].mean(dim=(0, 1))  # [T]
    logger.info("\n  S2 attention across action positions (last layer, avg over heads):")
    for t in range(0, T, 5):
        bar = "#" * int(s2_over_time[t].item() * 1000)
        logger.info("    t=%2d: %.6f  %s", t, s2_over_time[t].item(), bar)


# ─── Analysis 3: Encoder self-attention ─────────────────────────────────


def encoder_attention_analysis(policy, batch, device):
    """Analyze how S2 latent mixes with other tokens in the obs_encoder."""
    logger.info("\n" + "=" * 60)
    logger.info("ANALYSIS 3: Obs Encoder Self-Attention (S2 ↔ other tokens)")
    logger.info("=" * 60)

    model = policy.model
    D = model.config.hidden_dim
    nhead = model.config.num_heads
    head_dim = D // nhead

    # Build the pre-encoder token sequence manually
    with torch.no_grad():
        tokens = []
        images = [batch[k] for k in model.config.image_features]
        B = images[0].shape[0]
        stacked = torch.cat(images, dim=0)
        with torch.no_grad():
            features = model.backbone.forward_features(stacked)
            all_patches = features["x_norm_patchtokens"]
        N_cams = len(images)
        per_cam = all_patches.reshape(N_cams, B, all_patches.shape[1], all_patches.shape[2])
        for i in range(N_cams):
            tokens.append(model.image_proj(per_cam[i]))
        state_token = model.state_proj(batch["observation.state"]).unsqueeze(1)
        tokens.append(state_token)
        s2_token = model.s2_proj(batch["observation.s2_latent"]).unsqueeze(1)
        tokens.append(s2_token)

        pre_encoder = torch.cat(tokens, dim=1)  # [B, N_ctx, D]
        N_ctx = pre_encoder.shape[1]
        s2_pos = N_ctx - 1

    logger.info("Pre-encoder shape: [%d, %d, %d], S2 at position %d", B, N_ctx, D, s2_pos)

    # Run through encoder layers one by one, extracting self-attention
    x = pre_encoder
    for layer_idx, enc_layer in enumerate(model.obs_encoder.layers):
        with torch.no_grad():
            # Manually compute self-attention weights
            sa = enc_layer.self_attn
            d = sa.embed_dim
            w = sa.in_proj_weight
            b = sa.in_proj_bias

            q = F.linear(x, w[:d], b[:d] if b is not None else None)
            k = F.linear(x, w[d : 2 * d], b[d : 2 * d] if b is not None else None)
            v = F.linear(x, w[2 * d : 3 * d], b[2 * d : 3 * d] if b is not None else None)

            q = q.reshape(B, N_ctx, nhead, head_dim).transpose(1, 2)
            k = k.reshape(B, N_ctx, nhead, head_dim).transpose(1, 2)
            v_r = v.reshape(B, N_ctx, nhead, head_dim).transpose(1, 2)

            scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(head_dim)
            attn = F.softmax(scores, dim=-1)  # [B, nhead, N_ctx, N_ctx]

            # How much do OTHER tokens attend to S2?
            # attn[..., s2_pos] = weight that each token gives to S2
            s2_received = attn[:, :, :, s2_pos].mean(dim=(0, 1))  # [N_ctx]
            # Exclude S2's self-attention
            other_to_s2 = s2_received[:s2_pos].mean().item()  # avg attention FROM non-S2 TO S2

            # How much does S2 attend to itself vs. others?
            s2_to_s2 = attn[:, :, s2_pos, s2_pos].mean().item()
            s2_to_others = attn[:, :, s2_pos, :s2_pos].mean(dim=-1).mean().item()

            logger.info("  Encoder layer %d:", layer_idx)
            logger.info("    other→S2: %.6f  (how much image/state tokens attend to S2)", other_to_s2)
            logger.info("    S2→S2:    %.6f  (self-attention)", s2_to_s2)
            logger.info("    S2→other: %.6f  (S2 attending to image/state, per token)", s2_to_others)

            # Complete the layer forward for next iteration
            attn_out = torch.matmul(attn, v_r)
            attn_out = attn_out.transpose(1, 2).reshape(B, N_ctx, D)
            attn_out = sa.out_proj(attn_out)
            x = enc_layer.norm1(x + enc_layer.dropout1(attn_out))
            x = enc_layer.norm2(
                x
                + enc_layer.dropout2(
                    enc_layer.linear2(enc_layer.dropout(enc_layer.activation(enc_layer.linear1(x))))
                )
            )

    # After encoder: how different is context with vs. without S2?
    logger.info("\n  Context token norm change (with S2 vs. without):")
    with torch.no_grad():
        # Re-run without S2
        b_no_s2 = dict(batch)
        b_no_s2["observation.s2_latent"] = torch.zeros_like(batch["observation.s2_latent"])
        context_with = model.encode_observations(batch)
        context_without = model.encode_observations(b_no_s2)

        diff = (context_with - context_without).norm(dim=-1).mean(dim=0)  # [N_ctx]
        for pos_name, pos_range in [
            ("images", slice(0, N_ctx - 2)),
            ("state", slice(N_ctx - 2, N_ctx - 1)),
            ("S2", slice(N_ctx - 1, N_ctx)),
        ]:
            d_val = diff[pos_range].mean().item()
            logger.info("    %s tokens: Δnorm = %.4f", pos_name, d_val)


def _sinusoidal_embedding_external(timesteps, dim, device, min_period=4e-3, max_period=4.0):
    """Same as model's _sinusoidal_embedding but standalone."""
    half = dim // 2
    freqs = torch.exp(torch.linspace(math.log(min_period), math.log(max_period), half, device=device))
    args = timesteps[..., None] * freqs
    return torch.cat([torch.cos(args), torch.sin(args)], dim=-1)


def main():
    parser = argparse.ArgumentParser(description="Diagnose S1 attention to S2 latent")
    parser.add_argument("--s1-checkpoint", required=True, help="S1 checkpoint directory")
    parser.add_argument(
        "--s2-latents-dir", required=True, help="Directory with pre-extracted S2 latents (.npy)"
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--n-samples", type=int, default=8, help="Number of samples to analyze")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    logger.info("Loading S1 policy from %s...", args.s1_checkpoint)
    policy = load_s1_policy(args.s1_checkpoint, args.device)
    logger.info("Loading S2 latents from %s...", args.s2_latents_dir)
    batch = load_sample_batch(args.s2_latents_dir, args.device, args.n_samples)

    ablation_study(policy, batch, args.device)
    cross_attention_analysis(policy, batch, args.device)
    encoder_attention_analysis(policy, batch, args.device)

    logger.info("\n" + "=" * 60)
    logger.info("DONE")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()

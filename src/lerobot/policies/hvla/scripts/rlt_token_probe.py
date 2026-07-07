#!/usr/bin/env python
"""Offline reconstruction-error probe for the RLT token encoder/decoder.

Purpose: the state-normalization bug (fixed in commit 480efb41c) meant
the online RL actor/critic trained on contexts built from raw joint
values. The RL token encoder itself was trained on *normalized* contexts
— so the encoder should still be usable once the inference path is
fixed, IF its reconstruction error on correctly-normalized data is low
enough.

This probe loads the trained encoder/decoder, runs them over a few
batches of a dataset (same preprocessing FlowMatchingDataset applies at
training time), and reports relative reconstruction error:

    relative_rmse = RMSE / context_std

<10% relative RMSE → Phase-1 token encoder is good, no retrain needed.
>10% → retrain Phase 1 from scratch before starting fresh Phase 2.

Usage:
    python src/lerobot/policies/hvla/scripts/rlt_token_probe.py \\
        --s1-checkpoint outputs/flow_s1_hvla_v6/checkpoint-40000 \\
        --rl-token-checkpoint outputs/rlt_token/latest/rl_token.pt \\
        --dataset thewisp/cylinder_ring_assembly \\
        --batches 20 --batch-size 16

Doesn't require a robot, doesn't modify any checkpoint. Pure measurement.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from lerobot.policies.hvla.rlt.token import (
    RLTokenDecoder,
    RLTokenEncoder,
    load_rlt_token_config,
    rl_token_reconstruction_loss,
)

logger = logging.getLogger(__name__)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument(
        "--s1-checkpoint", required=True, type=Path, help="Path to S1 checkpoint dir (with pretrained_model/)"
    )
    p.add_argument(
        "--rl-token-checkpoint",
        required=True,
        type=Path,
        help="Path to a train_token.py checkpoint directory (contains encoder.pt + decoder.pt)",
    )
    p.add_argument(
        "--dataset", required=True, help="HF dataset repo id (e.g. thewisp/cylinder_ring_assembly)"
    )
    p.add_argument("--batches", type=int, default=20, help="Number of batches to probe (default 20)")
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--resize-images", default="224x224")
    p.add_argument("--device", default="cuda")
    args = p.parse_args()

    # ``force=True``: downstream imports (safetensors, LeRobotDataset) call
    # ``logging.basicConfig`` as a side effect, which silently wins the
    # first-configured race. Without force=True our INFO messages never
    # reach stdout.
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", force=True)
    logger.info("starting")

    device = torch.device(args.device)
    h, w = (int(x) for x in args.resize_images.split("x"))
    resize_to = (h, w)

    logger.info("loading S1 policy...")
    from lerobot.policies.hvla.s1.flow_matching.model import FlowMatchingS1Policy

    s1 = FlowMatchingS1Policy.from_pretrained(args.s1_checkpoint)
    s1.eval().to(device)
    for p_ in s1.parameters():
        p_.requires_grad_(False)
    logger.info("S1 loaded (image_features=%s)", list(s1.config.image_features.keys()))

    # Load dataset (same preprocessing FlowMatchingDataset uses)
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from lerobot.policies.hvla.s1.flow_matching.train import FlowMatchingDataset

    logger.info("opening LeRobotDataset...")
    lerobot_dataset = LeRobotDataset(args.dataset)
    s1_config = s1.config
    logger.info("dataset episodes=%d", lerobot_dataset.num_episodes)
    dataset = FlowMatchingDataset(
        lerobot_dataset,
        s2_latents=None,  # probe without S2 — adjust if your S1 uses S2
        chunk_size=s1_config.chunk_size,
        max_delay_seconds=0.0,
        resize_to=resize_to,
        image_keys=list(s1_config.image_features.keys()),
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True,
    )

    # Load encoder + decoder from a train_token.py checkpoint dir (each is a
    # separate state_dict file). Read the architecture manifest if present
    # so we instantiate layers/heads/ffn matching the trained checkpoint
    # (supports the gated 4-layer variant). Legacy checkpoints without a
    # config.json fall back to RLTConfig defaults (2 layers).
    ckpt_dir = args.rl_token_checkpoint
    rlt_config = load_rlt_token_config(ckpt_dir)
    logger.info(
        "RLT arch: encoder_layers=%d decoder_layers=%d heads=%d ffn=%d dim=%d",
        rlt_config.token_encoder_layers,
        rlt_config.token_decoder_layers,
        rlt_config.token_num_heads,
        rlt_config.token_ffn_dim,
        rlt_config.rl_token_dim,
    )
    encoder = RLTokenEncoder(rlt_config).to(device).eval()
    decoder = RLTokenDecoder(rlt_config).to(device).eval()
    encoder.load_state_dict(torch.load(ckpt_dir / "encoder.pt", map_location=device))
    decoder.load_state_dict(torch.load(ckpt_dir / "decoder.pt", map_location=device))

    # Run probe
    rmses = []
    stds = []
    losses = []
    with torch.no_grad(), torch.autocast(device_type=device.type, dtype=torch.bfloat16):
        for i, batch in enumerate(loader):
            if i >= args.batches:
                break
            batch = {k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in batch.items()}
            if s1_config.image_features:
                batch["observation.images"] = [batch[k] for k in s1_config.image_features]
            context = s1.model.encode_observations(batch).float()
            z_rl = encoder(context)
            recon = decoder(z_rl, context)
            rmse = ((recon - context) ** 2).mean().sqrt().item()
            std = context.std().item()
            loss = rl_token_reconstruction_loss(encoder, decoder, context).item()
            rmses.append(rmse)
            stds.append(std)
            losses.append(loss)
            logger.info(
                "batch %d: rmse=%.4f std=%.4f rel=%.1f%% loss=%.4f", i, rmse, std, 100 * rmse / std, loss
            )

    rmse = np.mean(rmses)
    std = np.mean(stds)
    rel = 100 * rmse / std
    logger.info("")
    logger.info("=== Summary ===")
    logger.info("  mean RMSE:           %.4f", rmse)
    logger.info("  mean context std:    %.4f", std)
    logger.info("  relative error:      %.1f%%", rel)
    logger.info("  mean paper-loss:     %.4f", np.mean(losses))
    logger.info("")
    if rel < 10:
        logger.info(
            "VERDICT: Phase-1 encoder is good (< 10%% rel error). "
            "No retrain needed. Proceed directly to fresh Phase 2 with "
            "the fixed inference pipeline."
        )
    elif rel < 20:
        logger.info(
            "VERDICT: Phase-1 encoder is borderline (%.1f%%). Proceed "
            "to Phase 2 but expect slower convergence; consider retraining "
            "Phase 1 for longer if Phase 2 stalls.",
            rel,
        )
    else:
        logger.info(
            "VERDICT: Phase-1 encoder is too lossy (%.1f%% rel error). "
            "Retrain Phase 1 from scratch on the fixed pipeline before "
            "starting fresh Phase 2.",
            rel,
        )


if __name__ == "__main__":
    main()

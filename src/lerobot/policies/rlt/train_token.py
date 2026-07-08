"""Phase 1: Offline RL token encoder-decoder training on demo data.

Trains the RL token to compress S1's observation encoder output into a
fixed-dim vector (default 768; pair ``--rl-token-dim 2048`` with the
4-layer encoder for the paper-style widened bottleneck) that retains
enough information to reconstruct the original context tokens. The
frozen S1 provides the context; only the encoder-decoder parameters
(phi) are trained.

Current canonical checkpoint: ``outputs/rlt_token_v4_4layer_d2048``
(4-layer encoder/decoder, d=2048, 24.9% reconstruction relative
error). See ``src/lerobot/policies/hvla/scripts/rlt_arch_experiment.sh``
for the architecture comparison that produced it.

Usage:
    python -m lerobot.policies.rlt.train_token \
        --s1-checkpoint outputs/flow_s1_no_s2_v1/checkpoints/last/pretrained_model/model.safetensors \
        --dataset-repo-id thewisp/cylinder_ring_assembly \
        --output-dir outputs/rlt_token_v5 \
        --encoder-layers 4 --decoder-layers 4 --rl-token-dim 2048 \
        --steps 10000
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from collections.abc import Callable, Iterator
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from lerobot.policies.rlt.config import RLTConfig
from lerobot.policies.rlt.token import (
    RLTokenDecoder,
    RLTokenEncoder,
    rl_token_reconstruction_loss,
    save_rlt_token_config,
)

logger = logging.getLogger(__name__)


def train_rl_token_encoder(
    rlt_config: RLTConfig,
    context_provider: Callable[[], Iterator[torch.Tensor]],
    *,
    output_dir: Path,
    device: torch.device,
    save_freq: int,
) -> None:
    """Policy-agnostic Phase-1 training loop.

    Trains the RL-token encoder/decoder to reconstruct a policy's context
    tokens. Knows nothing about which policy produced the context — the
    caller supplies a ``context_provider`` factory that returns an iterator
    yielding fp32 context tensors ``[B, N, D]`` already on ``device`` (D must
    equal ``rlt_config.context_dim or rlt_config.rl_token_dim``). This is the
    seam that lets pi0/pi05/multi_task_dit reuse the exact same loop; each
    supplies its own provider. Lives here (no policy imports) so it can move
    to ``policies/rlt`` unchanged.
    """
    encoder = RLTokenEncoder(rlt_config).to(device)
    decoder = RLTokenDecoder(rlt_config).to(device)
    enc_params = sum(p.numel() for p in encoder.parameters())
    dec_params = sum(p.numel() for p in decoder.parameters())
    logger.info("Encoder params: %.1fM | Decoder params: %.1fM", enc_params / 1e6, dec_params / 1e6)

    optimizer = torch.optim.Adam(
        list(encoder.parameters()) + list(decoder.parameters()),
        lr=rlt_config.token_lr,
    )

    logger.info(
        "Training RL token for %d steps",
        rlt_config.token_train_steps,
    )
    step = 0
    t0 = time.time()
    running_loss = 0.0
    contexts = context_provider()

    while step < rlt_config.token_train_steps:
        try:
            context = next(contexts)
        except StopIteration:
            contexts = context_provider()
            context = next(contexts)

        # Train encoder-decoder (context is already fp32 on device).
        loss = rl_token_reconstruction_loss(encoder, decoder, context)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        step += 1

        if step % 100 == 0:
            avg_loss = running_loss / 100
            elapsed = time.time() - t0
            steps_per_sec = step / elapsed
            logger.info(
                "step %d/%d | loss=%.6f | %.1f steps/s",
                step,
                rlt_config.token_train_steps,
                avg_loss,
                steps_per_sec,
            )
            running_loss = 0.0

        if step % save_freq == 0 or step == rlt_config.token_train_steps:
            save_dir = output_dir / f"checkpoint-{step}"
            save_dir.mkdir(exist_ok=True)
            torch.save(encoder.state_dict(), save_dir / "encoder.pt")
            torch.save(decoder.state_dict(), save_dir / "decoder.pt")
            # Manifest: shape-defining RLTConfig fields so a loader can
            # rebuild the same architecture without guessing (e.g. when
            # we train with --encoder-layers 4 the loader knows to
            # instantiate 4 layers, not the default 2).
            save_rlt_token_config(save_dir, rlt_config)
            torch.save(
                {"optimizer": optimizer.state_dict(), "step": step},
                save_dir / "optimizer.pt",
            )
            logger.info("Saved checkpoint at step %d → %s", step, save_dir)

    elapsed = time.time() - t0
    logger.info("Training complete: %d steps in %.1fs (%.1f steps/s)", step, elapsed, step / elapsed)


def train(args):
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        force=True,
    )
    logging.getLogger().handlers[0].stream = sys.stderr

    device = torch.device(args.device)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    file_handler = logging.FileHandler(output_dir / "train_token.log", mode="a")
    file_handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
    logging.getLogger().addHandler(file_handler)
    logger.info("Command: %s", " ".join(sys.argv))

    torch.set_float32_matmul_precision("high")

    # HVLA-S1-specific setup: load the frozen policy, build the RLT config
    # from its dims, and build a context provider over S1's dataset pipeline.
    # Everything below is the S1 wiring that stays in the hvla package when
    # ``train_rl_token_encoder`` (the policy-agnostic loop) moves to
    # policies/rlt. pi0/pi05/multi_task_dit will grow their own equivalents.
    from lerobot.policies.hvla.s1.flow_matching.model import FlowMatchingS1Policy

    logger.info("Loading frozen S1 from %s", args.s1_checkpoint)
    s1_policy = FlowMatchingS1Policy.from_pretrained(args.s1_checkpoint)
    s1_policy.to(device)
    s1_policy.eval()
    for p in s1_policy.parameters():
        p.requires_grad = False
    s1_config = s1_policy.config
    logger.info("S1 hidden_dim=%d, chunk_size=%d", s1_config.hidden_dim, s1_config.chunk_size)

    # --- RLT config ---
    # Default: bottleneck dim = S1 hidden_dim (symmetric setup).
    rlt_config = RLTConfig(rl_token_dim=s1_config.hidden_dim)
    if args.steps:
        rlt_config.token_train_steps = args.steps
    if args.lr:
        rlt_config.token_lr = args.lr
    # Architecture overrides: gated rollout so we can train a 4-layer
    # variant alongside the existing 2-layer checkpoints. The values
    # land in config.json next to each checkpoint so loaders rebuild
    # the same arch (see token.save_rlt_token_config).
    if args.encoder_layers is not None:
        rlt_config.token_encoder_layers = args.encoder_layers
    if args.decoder_layers is not None:
        rlt_config.token_decoder_layers = args.decoder_layers
    if args.rl_token_dim is not None and args.rl_token_dim != s1_config.hidden_dim:
        # Widen the bottleneck past S1's hidden_dim. The encoder inserts
        # an input projection (context_dim = s1_hidden → rl_token_dim);
        # the decoder a symmetric output projection. Memory scales with
        # rl_token_dim², so consider dropping batch_size accordingly.
        rlt_config.context_dim = s1_config.hidden_dim
        rlt_config.rl_token_dim = args.rl_token_dim

    # --- Load dataset (reuse S1's dataset pipeline) ---
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from lerobot.policies.hvla.s1.flow_matching.train import FlowMatchingDataset

    logger.info("Loading dataset: %s", args.dataset_repo_id)
    lerobot_dataset = LeRobotDataset(args.dataset_repo_id)

    s2_latents = None
    if args.s2_latent_path:
        s2_latents = np.load(args.s2_latent_path)
        logger.info("S2 latents: %s", s2_latents.shape)

    resize_to = None
    if args.resize_images:
        h, w = (int(x) for x in args.resize_images.split("x"))
        resize_to = (h, w)

    dataset = FlowMatchingDataset(
        lerobot_dataset,
        s2_latents=s2_latents,
        chunk_size=s1_config.chunk_size,
        max_delay_seconds=0.0,  # no delay aug for token training
        resize_to=resize_to,
        image_keys=list(s1_config.image_features.keys()),
    )
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
    )

    def s1_context_provider() -> Iterator[torch.Tensor]:
        """Yield fp32 S1 context tokens [B, N_ctx, D] for each batch.

        NOTE(rlt-generic): this calls ``model.encode_observations`` directly
        (with only the image-key arrangement), NOT the policy's
        ``prepare_batch_for_encode_observations`` that the *inference* path
        uses (which additionally z-scores state). Preserved verbatim here to
        keep Phase-1 numerics identical during the refactor; reconciling this
        with ``HVLAS1Adapter.encode_context`` is tracked for the adapter-parity
        pass so training- and inference-time z_rl match exactly.
        """
        for batch in dataloader:
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                if s1_config.image_features:
                    batch["observation.images"] = [batch[key] for key in s1_config.image_features]
                context = s1_policy.model.encode_observations(batch)  # [B, N_ctx, D]
                yield context.float()  # cast back to fp32 for encoder-decoder

    logger.info("Training RL token (batch=%d)", args.batch_size)
    train_rl_token_encoder(
        rlt_config,
        s1_context_provider,
        output_dir=output_dir,
        device=device,
        save_freq=args.save_freq,
    )


def main():
    parser = argparse.ArgumentParser(description="Train RL token encoder-decoder (Phase 1)")
    parser.add_argument("--s1-checkpoint", required=True, help="Path to frozen S1 checkpoint")
    parser.add_argument("--dataset-repo-id", required=True, help="LeRobot dataset repo ID")
    parser.add_argument("--s2-latent-path", default=None, help="S2 latents .npy (optional)")
    parser.add_argument("--output-dir", required=True, help="Output directory")
    parser.add_argument("--steps", type=int, default=5000)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--save-freq", type=int, default=1000)
    parser.add_argument("--resize-images", default="224x224")
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", default="cuda")
    # Architecture gating — default None means "keep RLTConfig defaults"
    # (currently 2+2 layers). Pass --encoder-layers 4 --decoder-layers 4
    # to train the larger variant. Saved to each checkpoint's config.json
    # so inference / probe loaders rebuild the matching architecture.
    parser.add_argument(
        "--encoder-layers", type=int, default=None, help="Override RLTConfig.token_encoder_layers"
    )
    parser.add_argument(
        "--decoder-layers", type=int, default=None, help="Override RLTConfig.token_decoder_layers"
    )
    parser.add_argument(
        "--rl-token-dim",
        type=int,
        default=None,
        help="Widen bottleneck past S1 hidden_dim (adds "
        "input/output projections in encoder/decoder). "
        "Default = S1 hidden_dim (symmetric).",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = main()
    train(args)

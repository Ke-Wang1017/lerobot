#!/usr/bin/env python
"""Training script for ACT with VLM conditioning (dual-system S1 policy).

Wraps a LeRobot dataset to inject precomputed S2 latents from Pi0.5,
then trains the ACTWithVLM policy.

Usage:
    python scripts/train_act_vlm.py \
        --dataset-repo-id thewisp/cylinder_ring_assembly \
        --s2-latent-path ~/.cache/huggingface/lerobot/thewisp/cylinder_ring_assembly/s2_latents.npy \
        --output-dir outputs/act_vlm \
        --steps 100000 \
        --batch-size 32
"""

import argparse
import logging
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from lerobot.configs.types import FeatureType
from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
from lerobot.datasets.utils import dataset_to_policy_features
from lerobot.policies.act_vlm.configuration_act_vlm import ACTWithVLMConfig
from lerobot.policies.act_vlm.modeling_act_vlm import S2_AGE_KEY, S2_LATENT_KEY, ACTWithVLMPolicy
from lerobot.policies.factory import make_pre_post_processors

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", force=True)
logger = logging.getLogger("train_act_vlm")
logger.setLevel(logging.INFO)


def _t():
    return time.perf_counter()


class LeRobotDatasetWithLatents(Dataset):
    """Wraps a LeRobotDataset to inject precomputed S2 latents.

    The latents .npy file must have shape [N_frames, latent_dim] where N_frames
    matches the dataset length exactly.

    Supports delay augmentation: randomly shifts the latent index backward by
    k frames (within the same episode) to simulate S2 staleness during inference.
    The age (k / fps) is injected as S2_AGE_KEY for the age embedding.
    """

    def __init__(
        self,
        dataset: LeRobotDataset,
        latent_path: str,
        max_delay_seconds: float = 0.0,
        fps: int = 30,
    ):
        """
        Args:
            dataset: Base LeRobot dataset.
            latent_path: Path to .npy file with shape [N_frames, latent_dim].
            max_delay_seconds: Maximum simulated S2 delay in seconds. 0 = no augmentation.
            fps: Dataset recording FPS (for converting delay to frames).
        """
        self.dataset = dataset
        self.fps = fps
        self.max_delay_frames = int(max_delay_seconds * fps)
        latent_path = os.path.expanduser(latent_path)

        if not os.path.exists(latent_path):
            raise FileNotFoundError(f"S2 latent file not found: {latent_path}")

        t0 = _t()
        self.latents = np.load(latent_path)
        load_ms = (_t() - t0) * 1000
        logger.info(
            f"[INIT] np.load(latents): shape={self.latents.shape}, "
            f"dtype={self.latents.dtype}, size={self.latents.nbytes / 1e9:.2f}GB, "
            f"time={load_ms:.0f}ms"
        )

        if self.latents.shape[0] != len(dataset):
            raise ValueError(
                f"S2 latent count ({self.latents.shape[0]}) != dataset frames ({len(dataset)}). "
                f"Re-run extract_s2_latents.py to generate latents for all frames."
            )

        # Build episode start index lookup for delay clipping.
        # episode_starts[i] = first frame index of the episode containing frame i.
        self._episode_starts = self._build_episode_starts()

        if self.max_delay_frames > 0:
            logger.info(
                f"[INIT] Delay augmentation enabled: max_delay={max_delay_seconds}s "
                f"= {self.max_delay_frames} frames at {fps}fps"
            )

        self._getitem_calls = 0

    def _build_episode_starts(self) -> np.ndarray:
        """Build array mapping frame_idx → episode start frame."""
        n = len(self.dataset)
        ep_starts = np.zeros(n, dtype=np.int64)

        # Load the hf_dataset to read episode_index per frame
        self.dataset._ensure_hf_dataset_loaded()
        ep_indices = self.dataset.hf_dataset["episode_index"]

        current_ep = -1
        current_start = 0
        for i, ep_idx in enumerate(ep_indices):
            if isinstance(ep_idx, torch.Tensor):
                ep_idx = ep_idx.item()
            if ep_idx != current_ep:
                current_ep = ep_idx
                current_start = i
            ep_starts[i] = current_start

        return ep_starts

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        t0 = _t()
        item = self.dataset[idx]
        inner_ms = (_t() - t0) * 1000

        # Delay augmentation: shift latent index backward within episode
        if self.max_delay_frames > 0 and self.training_mode:
            k = np.random.randint(0, self.max_delay_frames + 1)
            ep_start = self._episode_starts[idx]
            delayed_idx = max(idx - k, ep_start)
            age_seconds = k / self.fps
        else:
            delayed_idx = idx
            age_seconds = 0.0

        item[S2_LATENT_KEY] = torch.from_numpy(self.latents[delayed_idx]).float()
        item[S2_AGE_KEY] = torch.tensor([age_seconds], dtype=torch.float32)

        total_ms = (_t() - t0) * 1000

        self._getitem_calls += 1
        if self._getitem_calls <= 5 or self._getitem_calls % 1000 == 0:
            shapes = {k: tuple(v.shape) if hasattr(v, "shape") else type(v).__name__ for k, v in item.items()}
            delay_info = (
                f" delay={idx - delayed_idx}frames age={age_seconds:.2f}s"
                if self.max_delay_frames > 0
                else ""
            )
            logger.info(
                f"[GETITEM #{self._getitem_calls}] idx={idx}{delay_info} "
                f"inner={inner_ms:.1f}ms total={total_ms:.1f}ms | shapes={shapes}"
            )

        return item

    @property
    def training_mode(self):
        """Check if we should apply augmentation (only during training)."""
        return True  # DataLoader workers always in training mode

    # Expose attributes that the training loop may need
    @property
    def meta(self):
        return self.dataset.meta

    @property
    def num_frames(self):
        return self.dataset.num_frames

    @property
    def num_episodes(self):
        return self.dataset.num_episodes

    @property
    def episodes(self):
        return self.dataset.episodes


def make_delta_timestamps(delta_indices: list[int] | None, fps: int) -> list[float]:
    if delta_indices is None:
        return [0]
    return [i / fps for i in delta_indices]


def main():
    parser = argparse.ArgumentParser(description="Train ACTWithVLM (dual-system S1 policy)")

    # Dataset
    parser.add_argument("--dataset-repo-id", required=True, help="LeRobot dataset repo_id or local path")
    parser.add_argument("--dataset-root", default=None, help="Local root override for dataset")
    parser.add_argument("--s2-latent-path", required=True, help="Path to precomputed S2 latents .npy file")
    parser.add_argument("--episodes", default=None, help="Comma-separated episode indices (default: all)")
    parser.add_argument(
        "--max-delay",
        type=float,
        default=0.5,
        help="Max S2 delay augmentation in seconds (0=disabled, default=0.5)",
    )
    parser.add_argument("--dataset-fps", type=int, default=30, help="Dataset recording FPS")

    # Model
    parser.add_argument("--s2-latent-dim", type=int, default=2048, help="S2 latent dimension")
    parser.add_argument("--use-dino-backbone", action="store_true", help="Use DINOv2-S instead of ResNet18")
    parser.add_argument(
        "--freeze-vision-backbone", action="store_true", help="Freeze vision backbone during training"
    )
    parser.add_argument(
        "--resize-images", default=None, help="Resize images to HxW before forward (e.g. '224x224')"
    )
    parser.add_argument("--use-vae", action="store_true", default=True, help="Use VAE (default: true)")
    parser.add_argument("--no-vae", dest="use_vae", action="store_false")
    parser.add_argument("--chunk-size", type=int, default=100, help="Action chunk size")
    parser.add_argument("--dim-model", type=int, default=512, help="Transformer hidden dim")
    parser.add_argument("--n-action-steps", type=int, default=100, help="Action steps per inference")

    # Training
    parser.add_argument("--output-dir", default="outputs/act_vlm", help="Output directory")
    parser.add_argument("--steps", type=int, default=100_000, help="Training steps")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-5, help="Learning rate")
    parser.add_argument("--lr-backbone", type=float, default=1e-5, help="Backbone learning rate")
    parser.add_argument("--grad-clip-norm", type=float, default=10.0, help="Gradient clip norm")
    parser.add_argument("--num-workers", type=int, default=4, help="DataLoader workers")
    parser.add_argument("--save-freq", type=int, default=20_000, help="Checkpoint save frequency")
    parser.add_argument("--log-freq", type=int, default=100, help="Log frequency")
    parser.add_argument("--device", default="cuda", help="Device")

    # Resume
    parser.add_argument("--resume", default=None, help="Path to checkpoint to resume from")

    args = parser.parse_args()

    # Re-apply logging config after all imports (lerobot may have overridden it)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", force=True)
    logging.getLogger().setLevel(logging.INFO)
    print("[MAIN] Script started, logging configured.", flush=True)
    logger.info("[MAIN] Logger working.")

    device = torch.device(args.device)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 60)
    logger.info("[INIT START] Beginning initialization sequence")
    logger.info(f"  dataset_repo_id: {args.dataset_repo_id}")
    logger.info(f"  s2_latent_path: {args.s2_latent_path}")
    logger.info(f"  batch_size: {args.batch_size}, num_workers: {args.num_workers}")
    logger.info(f"  device: {device}")
    logger.info("=" * 60)

    # --- Load dataset metadata ---
    t0 = _t()
    logger.info("[INIT 1/7] Loading dataset metadata...")
    dataset_metadata = LeRobotDatasetMetadata(args.dataset_repo_id, root=args.dataset_root)
    features = dataset_to_policy_features(dataset_metadata.features)
    logger.info(f"[INIT 1/7] DONE in {(_t() - t0) * 1000:.0f}ms | fps={dataset_metadata.fps}")

    output_features = {key: ft for key, ft in features.items() if ft.type is FeatureType.ACTION}
    input_features = {key: ft for key, ft in features.items() if key not in output_features}

    # Note: S2 latent is NOT added to input_features — it passes through the
    # normalizer untouched (the normalizer only processes features it knows about).
    logger.info(f"  Input features: {list(input_features.keys())}")
    logger.info(f"  Output features: {list(output_features.keys())}")

    # --- Create config ---
    # Parse optional resize target
    resize_to = None
    if args.resize_images:
        h, w = args.resize_images.lower().split("x")
        resize_to = (int(h), int(w))
        logger.info(f"  Image resize: {resize_to}")

    cfg = ACTWithVLMConfig(
        input_features=input_features,
        output_features=output_features,
        s2_latent_dim=args.s2_latent_dim,
        use_dino_backbone=args.use_dino_backbone,
        freeze_vision_backbone=args.freeze_vision_backbone,
        use_vae=args.use_vae,
        chunk_size=args.chunk_size,
        n_action_steps=args.n_action_steps,
        dim_model=args.dim_model,
        optimizer_lr=args.lr,
        optimizer_lr_backbone=args.lr_backbone,
    )

    # --- Create policy ---
    t0 = _t()
    logger.info("[INIT 2/7] Creating policy...")
    if args.resume:
        logger.info(f"  Resuming from checkpoint: {args.resume}")
        policy = ACTWithVLMPolicy.from_pretrained(pretrained_name_or_path=args.resume)
    else:
        policy = ACTWithVLMPolicy(cfg)
    logger.info(f"[INIT 2/7] Policy created in {(_t() - t0) * 1000:.0f}ms")

    t0 = _t()
    logger.info(f"[INIT 3/7] Moving policy to {device}...")
    policy.train()
    policy.to(device)
    if device.type == "cuda":
        torch.cuda.synchronize()
    logger.info(f"[INIT 3/7] .to(device) done in {(_t() - t0) * 1000:.0f}ms")

    num_params = sum(p.numel() for p in policy.parameters())
    num_trainable = sum(p.numel() for p in policy.parameters() if p.requires_grad)
    logger.info(f"  Parameters: {num_trainable:,} trainable / {num_params:,} total")

    # --- Processors ---
    t0 = _t()
    logger.info("[INIT 4/7] Creating pre/post processors...")
    preprocessor, postprocessor = make_pre_post_processors(cfg, dataset_stats=dataset_metadata.stats)
    logger.info(f"[INIT 4/7] Processors created in {(_t() - t0) * 1000:.0f}ms")

    # --- Dataset ---
    delta_timestamps = {
        "action": make_delta_timestamps(cfg.action_delta_indices, dataset_metadata.fps),
    }
    delta_timestamps |= {
        k: make_delta_timestamps(cfg.observation_delta_indices, dataset_metadata.fps)
        for k in cfg.image_features
    }

    episodes = None
    if args.episodes:
        episodes = [int(e.strip()) for e in args.episodes.split(",")]

    t0 = _t()
    logger.info("[INIT 5/7] Building LeRobotDataset (video index)...")
    dataset = LeRobotDataset(
        args.dataset_repo_id,
        root=args.dataset_root,
        episodes=episodes,
        delta_timestamps=delta_timestamps,
        video_backend="pyav",
    )
    logger.info(
        f"[INIT 5/7] LeRobotDataset ready in {(_t() - t0) * 1000:.0f}ms | "
        f"{dataset.num_frames} frames, {dataset.num_episodes} episodes"
    )

    # Wrap with S2 latents (np.load happens inside constructor)
    t0 = _t()
    logger.info("[INIT 6/7] Wrapping dataset with S2 latents...")
    dataset_with_latents = LeRobotDatasetWithLatents(
        dataset,
        args.s2_latent_path,
        max_delay_seconds=args.max_delay,
        fps=args.dataset_fps,
    )
    logger.info(
        f"[INIT 6/7] Wrap done in {(_t() - t0) * 1000:.0f}ms | {dataset_with_latents.num_frames} frames"
    )

    # --- Dataloader ---
    t0 = _t()
    logger.info(f"[INIT 7/7] Creating DataLoader (num_workers={args.num_workers})...")
    dataloader = torch.utils.data.DataLoader(
        dataset_with_latents,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        drop_last=True,
        prefetch_factor=2 if args.num_workers > 0 else None,
    )
    logger.info(
        f"[INIT 7/7] DataLoader created in {(_t() - t0) * 1000:.0f}ms | {len(dataloader)} batches/epoch"
    )

    # --- Optimizer ---
    optimizer = torch.optim.AdamW(
        policy.get_optim_params(),
        lr=args.lr,
        weight_decay=cfg.optimizer_weight_decay,
    )

    logger.info("=" * 60)
    logger.info("[INIT COMPLETE] All init done. Fetching first batch...")
    logger.info("=" * 60)

    # --- Training loop ---
    step = 0
    done = False
    epoch = 0

    # Timing accumulators
    t_data_total = 0.0
    t_pre_total = 0.0
    t_fwd_total = 0.0
    t_bwd_total = 0.0
    t_opt_total = 0.0

    while not done:
        epoch += 1
        logger.info(f"[EPOCH {epoch}] Starting epoch, iterating dataloader ({len(dataloader)} batches)...")

        iter_start = _t()
        data_iter = iter(dataloader)

        # Time first batch separately — workers spawn here
        t_first = _t()
        logger.info(f"[EPOCH {epoch}] Waiting for first batch (workers starting)...")
        batch = next(data_iter)
        first_batch_ms = (_t() - t_first) * 1000
        logger.info(
            f"[EPOCH {epoch}] First batch arrived in {first_batch_ms:.0f}ms | "
            f"keys={list(batch.keys())} | "
            f"batch_size={next(iter(batch.values())).shape[0] if hasattr(next(iter(batch.values())), 'shape') else '?'}"
        )

        # Report image shapes in first batch
        for k, v in batch.items():
            if hasattr(v, "shape"):
                logger.info(f"  {k}: {tuple(v.shape)} {v.dtype}")

        # Process first batch
        first = True
        while True:
            loop_start = _t()

            # --- Preprocess + optional resize ---
            # Note: preprocessor's DeviceProcessorStep already moves batch to device.
            t0 = _t()
            batch = preprocessor(batch)
            if resize_to is not None:
                for key in list(batch.keys()):
                    v = batch[key]
                    if isinstance(v, torch.Tensor) and v.dim() == 4 and v.shape[1] == 3:
                        batch[key] = F.interpolate(v, size=resize_to, mode="bilinear", align_corners=False)
            t_pre = (_t() - t0) * 1000

            # --- Forward ---
            t0 = _t()
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
                loss, loss_dict = policy.forward(batch)
            if device.type == "cuda":
                torch.cuda.synchronize()
            t_fwd = (_t() - t0) * 1000

            # --- Backward ---
            t0 = _t()
            loss.backward()
            if device.type == "cuda":
                torch.cuda.synchronize()
            t_bwd = (_t() - t0) * 1000

            # --- Optimizer ---
            t0 = _t()
            if args.grad_clip_norm > 0:
                torch.nn.utils.clip_grad_norm_(policy.parameters(), args.grad_clip_norm)
            optimizer.step()
            optimizer.zero_grad()
            t_opt = (_t() - t0) * 1000

            step += 1
            step_ms = (_t() - loop_start) * 1000

            # Accumulate
            t_pre_total += t_pre
            t_fwd_total += t_fwd
            t_bwd_total += t_bwd
            t_opt_total += t_opt

            if first or step % args.log_freq == 0:
                first = False
                loss_str = " | ".join(f"{k}: {v:.4f}" for k, v in loss_dict.items())
                logger.info(
                    f"step {step}/{args.steps} | loss: {loss.item():.4f} | {loss_str} | "
                    f"total={step_ms:.0f}ms "
                    f"[pre={t_pre:.0f}ms fwd={t_fwd:.0f}ms bwd={t_bwd:.0f}ms opt={t_opt:.0f}ms]"
                )

            if step % args.save_freq == 0:
                ckpt_dir = output_dir / f"checkpoint-{step}"
                policy.save_pretrained(ckpt_dir)
                preprocessor.save_pretrained(ckpt_dir)
                postprocessor.save_pretrained(ckpt_dir)
                logger.info(f"Saved checkpoint: {ckpt_dir}")

            if step >= args.steps:
                done = True
                break

            # --- Load next batch ---
            t0 = _t()
            try:
                batch = next(data_iter)
            except StopIteration:
                break
            t_data = (_t() - t0) * 1000
            t_data_total += t_data

            # Log data loading time if suspiciously slow
            if t_data > 500:
                logger.warning(f"[SLOW DATA] step {step + 1}: data load took {t_data:.0f}ms")
            elif step % args.log_freq == 0:
                logger.info(f"  data load: {t_data:.0f}ms")

        epoch_ms = (_t() - iter_start) * 1000
        n = max(step, 1)
        logger.info(
            f"[EPOCH {epoch}] Done in {epoch_ms / 1000:.1f}s | "
            f"avg per step: pre={t_pre_total / n:.0f}ms fwd={t_fwd_total / n:.0f}ms "
            f"bwd={t_bwd_total / n:.0f}ms opt={t_opt_total / n:.0f}ms data={t_data_total / max(n - 1, 1):.0f}ms"
        )

        if done:
            break

    # --- Final save ---
    policy.save_pretrained(output_dir)
    preprocessor.save_pretrained(output_dir)
    postprocessor.save_pretrained(output_dir)
    logger.info(f"Training complete. Model saved to {output_dir}")


if __name__ == "__main__":
    main()

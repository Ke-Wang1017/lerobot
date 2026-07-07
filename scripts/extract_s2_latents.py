#!/usr/bin/env python
"""Batch extraction of S2 latents from Pi0.5 (PyTorch) for dual-system VLA training.

Calls the PyTorch model directly — no WebSocket server needed — and batches
multiple frames per GPU forward pass for maximum throughput.

Usage:
    python scripts/extract_s2_latents.py \
        --checkpoint-path /path/to/pi05_checkpoint \
        --dataset-path /path/to/lerobot/dataset \
        --high-level-prompt "do the task" \
        --output-path s2_latents.npy \
        --image-keys observation.images.front,observation.images.wrist_left,observation.images.wrist_right \
        --batch-size 8

Output .npy has shape [N_frames, 2048], index-aligned with the dataset.
"""

import argparse
import os
import sys
import time

import numpy as np
import torch

# Ensure openpi_subtask is importable
_OPENPI_SRC = os.path.expanduser("~/Documents/openpi_subtask/src")
_OPENPI_ROOT = os.path.expanduser("~/Documents/openpi_subtask")
for _p in (_OPENPI_SRC, _OPENPI_ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from lerobot.datasets.lerobot_dataset import LeRobotDataset  # noqa: E402

# Map LeRobot image keys → Pi0.5 model keys
IMAGE_KEY_MAP = {
    "observation.images.front": "base_0_rgb",
    "observation.images.top": "base_1_rgb",
    "observation.images.wrist_left": "left_wrist_0_rgb",
    "observation.images.left_wrist": "left_wrist_0_rgb",
    "observation.images.wrist_right": "right_wrist_0_rgb",
    "observation.images.right_wrist": "right_wrist_0_rgb",
}


def lerobot_img_to_uint8_hwc(img_tensor) -> np.ndarray:
    """Convert LeRobot float32 CHW [0,1] tensor to uint8 HWC numpy array."""
    arr = img_tensor.numpy()
    if arr.ndim == 3 and arr.shape[0] in (1, 3):
        arr = arr.transpose(1, 2, 0)
    return (arr * 255).clip(0, 255).astype(np.uint8)


def load_engine(checkpoint_path: str, device: str, model_image_keys: list[str]):
    """Instantiate and initialize PyTorchPi05Inference directly (no server)."""
    from scripts.async_pi05.pytorch_pi05_inference import PyTorchPi05Inference

    engine = PyTorchPi05Inference(
        checkpoint_path=checkpoint_path,
        device=device,
        image_keys=model_image_keys,
    )
    engine._initialize_blocking()
    print(f"Model loaded on {device}")
    return engine


def prepare_batch(engine, frames: list[dict], lerobot_image_keys: list[str], high_level_prompt: str):
    """Build a batched SimpleObservation by catting B individual B=1 observations."""
    from scripts.async_pi05.pytorch_pi05_inference import SimpleObservation

    obs_list = []
    for item in frames:
        images_np = {}
        for lr_key in lerobot_image_keys:
            model_key = IMAGE_KEY_MAP.get(lr_key, lr_key.split(".")[-1])
            if lr_key in item:
                images_np[model_key] = lerobot_img_to_uint8_hwc(item[lr_key])

        state_tensor = item.get("observation.state")
        state = state_tensor.numpy() if state_tensor is not None else None

        obs = engine._prepare_observation(images_np, high_level_prompt, "", state)
        obs_list.append(obs)

    if len(obs_list) == 1:
        return obs_list[0]

    return SimpleObservation(
        images={k: torch.cat([o.images[k] for o in obs_list], dim=0) for k in obs_list[0].images},
        image_masks={
            k: torch.cat([o.image_masks[k] for o in obs_list], dim=0) for k in obs_list[0].image_masks
        },
        state=torch.cat([o.state for o in obs_list], dim=0),
        tokenized_prompt=torch.cat([o.tokenized_prompt for o in obs_list], dim=0),
        tokenized_prompt_mask=torch.cat([o.tokenized_prompt_mask for o in obs_list], dim=0),
        token_ar_mask=torch.cat([o.token_ar_mask for o in obs_list], dim=0),
        token_loss_mask=torch.cat([o.token_loss_mask for o in obs_list], dim=0),
    )


def extract_latents(
    checkpoint_path: str,
    dataset_path: str,
    high_level_prompt: str,
    output_path: str,
    lerobot_image_keys: list[str],
    device: str,
    batch_size: int,
):
    model_image_keys = [IMAGE_KEY_MAP.get(k, k.split(".")[-1]) for k in lerobot_image_keys]
    engine = load_engine(checkpoint_path, device, model_image_keys)

    print(f"Loading dataset from {dataset_path}...")
    dataset = LeRobotDataset(dataset_path, video_backend="pyav")
    n_frames = len(dataset)
    print(f"Dataset: {n_frames} frames")

    # Resume support
    output_path = os.path.expanduser(output_path)
    latents = []
    resume_from = 0
    if os.path.exists(output_path):
        existing = np.load(output_path)
        if existing.shape[0] >= n_frames:
            print(f"Already complete ({n_frames} frames). Nothing to do.")
            return existing
        resume_from = existing.shape[0]
        latents = list(existing)
        print(f"Resuming from frame {resume_from}/{n_frames}")

    timings = []
    idx = resume_from
    while idx < n_frames:
        batch_end = min(idx + batch_size, n_frames)
        frames = [dataset[i] for i in range(idx, batch_end)]
        actual_batch = len(frames)

        t0 = time.time()
        batched_obs = prepare_batch(engine, frames, lerobot_image_keys, high_level_prompt)
        with torch.no_grad():
            latent_batch = engine.model.extract_prefix_latent(
                engine.device, batched_obs, image_keys=engine.image_keys
            )  # [B, 2048]
        latent_np = latent_batch.float().cpu().numpy()

        elapsed_ms = (time.time() - t0) * 1000
        ms_per_frame = elapsed_ms / actual_batch
        timings.append(ms_per_frame)

        for b in range(actual_batch):
            latents.append(latent_np[b])
        idx = batch_end

        if idx % 100 == 0 or idx == n_frames:
            avg_ms = np.mean(timings[-20:])
            eta_s = avg_ms * (n_frames - idx) / 1000
            print(f"  [{idx}/{n_frames}] {avg_ms:.1f}ms/frame | ETA: {eta_s / 60:.1f}min")

        if idx % 5000 == 0:
            partial = np.stack(latents, axis=0)
            np.save(output_path, partial)
            print(f"  Checkpoint: {partial.shape}")

    latents_array = np.stack(latents, axis=0)
    np.save(output_path, latents_array)
    print(f"Saved {latents_array.shape} → {output_path}")
    print(f"Mean: {np.mean(timings):.1f}ms/frame  (batch_size={batch_size})")
    return latents_array


def main():
    parser = argparse.ArgumentParser(description="Extract S2 latents directly from PyTorch Pi0.5")
    parser.add_argument("--checkpoint-path", required=True, help="Pi0.5 PyTorch checkpoint directory")
    parser.add_argument("--dataset-path", required=True, help="LeRobot dataset path or repo_id")
    parser.add_argument("--high-level-prompt", default="do the task")
    parser.add_argument("--output-path", default="s2_latents.npy")
    parser.add_argument(
        "--image-keys",
        default="observation.images.front,observation.images.wrist_left,observation.images.wrist_right",
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Frames per GPU forward pass. Increase if VRAM allows (16+ on 24GB).",
    )
    args = parser.parse_args()

    extract_latents(
        checkpoint_path=args.checkpoint_path,
        dataset_path=args.dataset_path,
        high_level_prompt=args.high_level_prompt,
        output_path=args.output_path,
        lerobot_image_keys=[k.strip() for k in args.image_keys.split(",")],
        device=args.device,
        batch_size=args.batch_size,
    )


if __name__ == "__main__":
    main()

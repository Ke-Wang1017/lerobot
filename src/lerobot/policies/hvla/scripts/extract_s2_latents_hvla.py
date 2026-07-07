#!/usr/bin/env python
"""Batch extraction of S2 latents using the HVLA S2 VLM model (no OpenPI dependency).

Calls lerobot.policies.hvla.s2.S2VLMModel directly — no WebSocket server,
no OpenPI imports.

Usage:
    python src/lerobot/policies/hvla/scripts/extract_s2_latents_hvla.py \
        --checkpoint /path/to/pi05_checkpoint/model.safetensors \
        --dataset thewisp/cylinder_ring_assembly \
        --prompt "assemble cylinder into ring" \
        --output s2_latents.npy \
        --batch-size 8

Output .npy has shape [N_frames, 2048], index-aligned with the dataset.
"""

import argparse
import os
import time

import numpy as np
import torch

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies.hvla.s2.model import S2VLMModel
from lerobot.policies.hvla.s2.preprocessing import preprocess_images
from lerobot.policies.hvla.s2.tokenizer import PaligemmaTokenizer

# Map LeRobot image keys → S2 camera keys
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


def extract_latents(
    checkpoint: str,
    dataset_path: str,
    prompt: str,
    output_path: str,
    lerobot_image_keys: list[str],
    device: str,
    batch_size: int,
):
    # Load S2 VLM model
    print(f"Loading S2 VLM from {checkpoint}...")
    model = S2VLMModel.from_pretrained(checkpoint)
    model.to(device).eval()
    print(f"S2 VLM loaded on {device} ({sum(p.numel() for p in model.parameters()) / 1e6:.0f}M params)")

    # Load tokenizer
    tokenizer = PaligemmaTokenizer(max_len=256)

    # Pre-tokenize prompt (same for all frames)
    token_ids, token_mask = tokenizer.tokenize_prompt(prompt)

    # S2 camera keys
    s2_image_keys = tuple(IMAGE_KEY_MAP.get(k, k.split(".")[-1]) for k in lerobot_image_keys)

    # Load dataset
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

        # Prepare images: convert each frame's images to S2 format and batch
        batch_images = {key: [] for key in s2_image_keys}
        for item in frames:
            for lr_key, s2_key in zip(lerobot_image_keys, s2_image_keys, strict=False):
                if lr_key in item:
                    batch_images[s2_key].append(lerobot_img_to_uint8_hwc(item[lr_key]))

        # Preprocess and stack batch
        all_img_tensors = []
        all_img_masks = []
        for cam_key in s2_image_keys:
            cam_batch = batch_images[cam_key]
            if cam_batch:
                # Preprocess each image individually, then cat
                tensors = []
                masks = []
                for img_np in cam_batch:
                    t_list, m_list = preprocess_images(
                        {cam_key: img_np},
                        image_keys=(cam_key,),
                        device=device,
                    )
                    tensors.append(t_list[0])
                    masks.append(m_list[0])
                all_img_tensors.append(torch.cat(tensors, dim=0))  # [B, C, H, W]
                all_img_masks.append(torch.cat(masks, dim=0))  # [B]

        # Batch language tokens
        lang_tokens = torch.from_numpy(token_ids).unsqueeze(0).long().to(device)
        lang_masks = torch.from_numpy(token_mask).unsqueeze(0).bool().to(device)
        lang_tokens = lang_tokens.expand(actual_batch, -1)
        lang_masks = lang_masks.expand(actual_batch, -1)

        with torch.no_grad():
            latent_batch = model.extract_prefix_latent(
                all_img_tensors,
                all_img_masks,
                lang_tokens,
                lang_masks,
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
    parser = argparse.ArgumentParser(description="Extract S2 latents using HVLA S2 VLM (no OpenPI)")
    parser.add_argument("--checkpoint", required=True, help="Pi0.5 safetensors checkpoint path")
    parser.add_argument("--dataset", required=True, help="LeRobot dataset path or repo_id")
    parser.add_argument("--prompt", default="do the task", help="High-level task prompt")
    parser.add_argument("--output", default="s2_latents.npy", help="Output .npy path")
    parser.add_argument(
        "--image-keys",
        default="observation.images.front,observation.images.top,observation.images.wrist_left,observation.images.wrist_right",
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args()

    extract_latents(
        checkpoint=args.checkpoint,
        dataset_path=args.dataset,
        prompt=args.prompt,
        output_path=args.output,
        lerobot_image_keys=[k.strip() for k in args.image_keys.split(",")],
        device=args.device,
        batch_size=args.batch_size,
    )


if __name__ == "__main__":
    main()

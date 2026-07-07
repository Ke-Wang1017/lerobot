#!/usr/bin/env python
"""Benchmark S1 training speed across backbone/resolution configs.

Configs tested:
  1. ResNet18, native 720x1280 (baseline)
  2. ResNet18, 224x224
  3. ResNet18, 480x480
  4. DINOv2-S (frozen), 224x224
  5. DINOv2-S (unfrozen), 224x224

Usage:
    python scripts/benchmark_training_speed.py \
        --dataset-repo-id thewisp/cylinder_ring_assembly \
        --s2-latent-path ~/.cache/huggingface/lerobot/thewisp/cylinder_ring_assembly/s2_latents.npy \
        --warmup 5 --steps 20
"""

import argparse
import logging
import os
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from lerobot.configs.types import FeatureType
from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
from lerobot.datasets.utils import dataset_to_policy_features
from lerobot.policies.act_vlm.configuration_act_vlm import ACTWithVLMConfig
from lerobot.policies.act_vlm.modeling_act_vlm import S2_LATENT_KEY, ACTWithVLMPolicy
from lerobot.policies.factory import make_pre_post_processors

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger("benchmark")


def _t():
    return time.perf_counter()


def _sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize()


class LeRobotDatasetWithLatents(Dataset):
    def __init__(self, dataset, latent_path):
        self.dataset = dataset
        latent_path = os.path.expanduser(latent_path)
        self.latents = np.load(latent_path)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        item[S2_LATENT_KEY] = torch.from_numpy(self.latents[idx]).float()
        return item

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


def make_delta_timestamps(delta_indices, fps):
    if delta_indices is None:
        return [0]
    return [i / fps for i in delta_indices]


def run_benchmark(args, label, resize_to=None, use_dino=False, freeze_dino=True):
    print(f"\n{'=' * 60}")
    print(f"  Benchmarking: {label}")
    print(f"{'=' * 60}")

    device = torch.device(args.device)

    dataset_metadata = LeRobotDatasetMetadata(args.dataset_repo_id, root=args.dataset_root)
    features = dataset_to_policy_features(dataset_metadata.features)
    output_features = {k: v for k, v in features.items() if v.type is FeatureType.ACTION}
    input_features = {k: v for k, v in features.items() if k not in output_features}

    cfg = ACTWithVLMConfig(
        input_features=input_features,
        output_features=output_features,
        s2_latent_dim=2048,
        use_dino_backbone=use_dino,
        freeze_vision_backbone=freeze_dino if use_dino else False,
    )

    policy = ACTWithVLMPolicy(cfg)
    policy.train()
    if use_dino and freeze_dino:
        # Keep DINOv2 in eval mode (batchnorm etc.) when frozen
        policy.model.dino_backbone.eval()
    policy.to(device)

    n_params = sum(p.numel() for p in policy.parameters())
    n_trainable = sum(p.numel() for p in policy.parameters() if p.requires_grad)
    print(f"  params: {n_trainable / 1e6:.1f}M trainable / {n_params / 1e6:.1f}M total")

    preprocessor, _ = make_pre_post_processors(cfg, dataset_stats=dataset_metadata.stats)

    delta_timestamps = {"action": make_delta_timestamps(cfg.action_delta_indices, dataset_metadata.fps)}
    delta_timestamps |= {
        k: make_delta_timestamps(cfg.observation_delta_indices, dataset_metadata.fps)
        for k in cfg.image_features
    }

    dataset = LeRobotDataset(
        args.dataset_repo_id,
        root=args.dataset_root,
        delta_timestamps=delta_timestamps,
        video_backend="pyav",
    )
    dataset_with_latents = LeRobotDatasetWithLatents(dataset, args.s2_latent_path)

    dataloader = torch.utils.data.DataLoader(
        dataset_with_latents,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        drop_last=True,
        prefetch_factor=2 if args.num_workers > 0 else None,
    )

    optimizer = torch.optim.AdamW(policy.get_optim_params(), lr=1e-5)
    data_iter = iter(dataloader)

    t_data_list, t_pre_list, t_fwd_list, t_bwd_list, t_wall_list = [], [], [], [], []

    for i in range(args.warmup + args.steps):
        wall_start = _t()

        # --- Data ---
        t0 = _t()
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(dataloader)
            batch = next(data_iter)
        _sync(device)
        t_data = (_t() - t0) * 1000

        # --- Preprocess + optional resize ---
        t0 = _t()
        batch = preprocessor(batch)
        if resize_to is not None:
            for key in list(batch.keys()):
                v = batch[key]
                if isinstance(v, torch.Tensor) and v.dim() == 4 and v.shape[1] == 3:
                    batch[key] = F.interpolate(
                        v.to(device), size=resize_to, mode="bilinear", align_corners=False
                    )
                elif isinstance(v, torch.Tensor):
                    batch[key] = v.to(device)
        else:
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
        _sync(device)
        t_pre = (_t() - t0) * 1000

        # --- Forward (bf16 autocast to match actual training) ---
        t0 = _t()
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
            loss, _ = policy.forward(batch)
        _sync(device)
        t_fwd = (_t() - t0) * 1000

        # --- Backward ---
        t0 = _t()
        loss.backward()
        _sync(device)
        t_bwd = (_t() - t0) * 1000

        optimizer.step()
        optimizer.zero_grad()

        _sync(device)
        t_wall = (_t() - wall_start) * 1000

        if i >= args.warmup:
            t_data_list.append(t_data)
            t_pre_list.append(t_pre)
            t_fwd_list.append(t_fwd)
            t_bwd_list.append(t_bwd)
            t_wall_list.append(t_wall)

        status = "WARMUP" if i < args.warmup else f"step {i - args.warmup + 1}/{args.steps}"
        print(
            f"  [{status}] wall={t_wall:.0f}ms | data={t_data:.0f}ms pre={t_pre:.0f}ms fwd={t_fwd:.0f}ms bwd={t_bwd:.0f}ms | loss={loss.item():.4f}"
        )

    def avg(lst):
        return sum(lst) / len(lst) if lst else 0

    # Token count
    if use_dino and resize_to:
        tokens_per_cam = (resize_to[0] // 14) * (resize_to[1] // 14)
        token_info = f"DINOv2 patches: {tokens_per_cam}/cam × {len(cfg.image_features)} = {tokens_per_cam * len(cfg.image_features)} total"
    elif resize_to:
        tokens_per_cam = (resize_to[0] // 32) * (resize_to[1] // 32)
        token_info = f"ResNet tokens: {tokens_per_cam}/cam × {len(cfg.image_features)} = {tokens_per_cam * len(cfg.image_features)} total"
    else:
        tokens_per_cam = (720 // 32) * (1280 // 32)
        token_info = f"ResNet tokens: {tokens_per_cam}/cam × {len(cfg.image_features)} = {tokens_per_cam * len(cfg.image_features)} total"

    sps = args.batch_size * 1000 / avg(t_wall_list) if avg(t_wall_list) > 0 else 0
    print(f"\n  --- {label} SUMMARY (avg over {args.steps} steps, batch_size={args.batch_size}) ---")
    print(f"  {token_info}")
    print(f"  wall-clock: {avg(t_wall_list):.1f}ms/step  →  {sps:.1f} samples/sec")
    print(f"  data      : {avg(t_data_list):.1f}ms  ({100 * avg(t_data_list) / avg(t_wall_list):.0f}%)")
    print(f"  pre+resize: {avg(t_pre_list):.1f}ms  ({100 * avg(t_pre_list) / avg(t_wall_list):.0f}%)")
    print(f"  fwd       : {avg(t_fwd_list):.1f}ms  ({100 * avg(t_fwd_list) / avg(t_wall_list):.0f}%)")
    print(f"  bwd       : {avg(t_bwd_list):.1f}ms  ({100 * avg(t_bwd_list) / avg(t_wall_list):.0f}%)")
    print(f"  compute   : {avg(t_fwd_list) + avg(t_bwd_list):.1f}ms  (fwd+bwd only)")

    return {
        "label": label,
        "wall": avg(t_wall_list),
        "data": avg(t_data_list),
        "pre": avg(t_pre_list),
        "fwd": avg(t_fwd_list),
        "bwd": avg(t_bwd_list),
        "sps": sps,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-repo-id", required=True)
    parser.add_argument("--dataset-root", default=None)
    parser.add_argument("--s2-latent-path", required=True)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--skip-baseline", action="store_true", help="Skip slow 720x1280 baseline")
    args = parser.parse_args()

    results = []

    if not args.skip_baseline:
        results.append(run_benchmark(args, "ResNet18 native 720x1280", resize_to=None))

    results.append(run_benchmark(args, "ResNet18 224x224", resize_to=(224, 224)))
    results.append(run_benchmark(args, "ResNet18 480x480", resize_to=(480, 480)))
    results.append(
        run_benchmark(args, "DINOv2-S frozen 224x224", resize_to=(224, 224), use_dino=True, freeze_dino=True)
    )
    results.append(
        run_benchmark(
            args, "DINOv2-S unfrozen 224x224", resize_to=(224, 224), use_dino=True, freeze_dino=False
        )
    )
    # results.append(run_benchmark(args, "DINOv2-S unfrozen 448x448", resize_to=(448, 448), use_dino=True, freeze_dino=False))

    print(f"\n{'=' * 60}")
    print(f"  COMPARISON (batch_size={args.batch_size}, num_workers={args.num_workers})")
    print(f"{'=' * 60}")
    baseline_wall = results[0]["wall"]
    for r in results:
        speedup = baseline_wall / r["wall"] if r["wall"] > 0 else 0
        compute = r["fwd"] + r["bwd"]
        print(
            f"  {r['label']:30s}  wall={r['wall']:5.0f}ms  ({speedup:.1f}x)  "
            f"data={r['data']:.0f}ms  compute={compute:.0f}ms  "
            f"→ {r['sps']:.1f} samples/sec"
        )


if __name__ == "__main__":
    main()

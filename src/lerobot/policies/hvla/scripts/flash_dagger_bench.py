"""Micro-benchmark: per-step wall time at various LoRA ranks, plus full FT.

Removes data-loader overhead by pre-fetching one batch, moving it to GPU,
then running N forward+backward passes on that same batch. The numbers
reflect pure GPU compute + Python/optimizer overhead — no video decode.

Configs measured:
  • LoRA rank ∈ {4, 8, 16, 32, 64, 128}
  • decoder-only FT (freeze backbone + encoder, train decoder fully)
  • full FT (everything trainable)

Reports per-config:
  • median step time (ms)
  • trainable params (M)
  • optimizer state (MB, AdamW fp32)
  • peak GPU memory (MB)
  • weight-delta size for sync (MB, bf16)
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path

import numpy as np
import torch

from lerobot.policies.hvla.scripts.flash_dagger_lora import apply_lora_to_decoder
from lerobot.policies.hvla.scripts.flash_dagger_phase_a_rank import (
    compute_per_sample_loss,
)
from lerobot.policies.hvla.scripts.flash_dagger_phase_b import (
    build_dataset,
    make_collate_fn,
    split_episode,
)

logger = logging.getLogger(__name__)


def freeze_backbone_and_encoder(policy):
    """Freeze DINOv2 + obs_encoder + projections. Decoder + action heads trainable."""
    inner = policy.model
    if inner.backbone is not None:
        for p in inner.backbone.parameters():
            p.requires_grad = False
    for p in inner.obs_encoder.parameters():
        p.requires_grad = False
    if hasattr(inner, "image_proj"):
        for p in inner.image_proj.parameters():
            p.requires_grad = False
    if hasattr(inner, "state_proj"):
        for p in inner.state_proj.parameters():
            p.requires_grad = False
    if hasattr(inner, "s2_proj"):
        for p in inner.s2_proj.parameters():
            p.requires_grad = False


def benchmark_one(policy, config, batch_gpu, n_steps: int, lr: float) -> dict:
    """Run N forward+backward steps on a fixed batch; return median step time + memory stats."""
    device = next(policy.parameters()).device
    trainable = [p for p in policy.parameters() if p.requires_grad]
    n_train = sum(p.numel() for p in trainable)
    if n_train == 0:
        return {"trainable_M": 0, "step_ms_median": float("nan")}

    opt = torch.optim.AdamW(trainable, lr=lr, weight_decay=0.0)

    torch.cuda.reset_peak_memory_stats(device)
    torch.cuda.synchronize(device)

    times: list[float] = []
    for _ in range(n_steps + 5):  # 5 warmup
        t0 = time.perf_counter()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            losses = compute_per_sample_loss(policy, batch_gpu, config)
            loss = losses.mean()
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable, 1.0)
        opt.step()
        torch.cuda.synchronize(device)
        times.append((time.perf_counter() - t0) * 1000)

    peak_mb = torch.cuda.max_memory_allocated(device) / (1024 * 1024)
    measured = times[5:]  # drop warmup

    # Optimizer-state size: AdamW = 2× param memory in fp32
    optim_mb = (n_train * 2 * 4) / (1024 * 1024)
    # Weight-delta size: trainable params × 2 bytes (bf16) for sync
    delta_mb = (n_train * 2) / (1024 * 1024)

    return {
        "trainable_M": n_train / 1e6,
        "step_ms_median": float(np.median(measured)),
        "step_ms_p10": float(np.percentile(measured, 10)),
        "step_ms_p90": float(np.percentile(measured, 90)),
        "peak_mem_MB": peak_mb,
        "optim_state_MB": optim_mb,
        "delta_MB_bf16": delta_mb,
    }


def build_policy_with_config(state_dict_path, config, device, mode: str, rank: int):
    """Create a fresh policy and apply the requested training mode."""
    import safetensors.torch as sft

    from lerobot.policies.hvla.s1.flow_matching.model import FlowMatchingS1Policy

    policy = FlowMatchingS1Policy(config).to(device)
    state_dict = sft.load_file(str(state_dict_path))
    policy.load_state_dict(state_dict, strict=False)

    if mode == "lora":
        apply_lora_to_decoder(policy, rank=rank, alpha=2 * rank)
        policy.to(device)
    elif mode == "decoder_only":
        freeze_backbone_and_encoder(policy)
    elif mode == "full":
        # All params trainable (default after load)
        for p in policy.parameters():
            p.requires_grad = True
    else:
        raise ValueError(f"unknown mode {mode}")

    policy.train()
    return policy


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--eval-repo-id", required=True)
    parser.add_argument("--output-csv", default="outputs/flash_dagger/bench/rank_compare.csv")
    parser.add_argument("--ranks", type=int, nargs="+", default=[4, 8, 16, 32, 64, 128])
    parser.add_argument("--also-decoder-only", action="store_true", default=True)
    parser.add_argument("--also-full-ft", action="store_true", default=True)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--episode-for-batch", type=int, default=247)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", force=True)

    output_csv = Path(args.output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device)
    torch.set_float32_matmul_precision("high")

    ckpt_dir = Path(args.checkpoint) / "pretrained_model"
    with open(ckpt_dir / "config.json") as f:
        ckpt_cfg = json.load(f)

    from lerobot.policies.hvla.s1.flow_matching.config import FlowMatchingS1Config

    config = FlowMatchingS1Config(
        chunk_size=ckpt_cfg["chunk_size"],
        hidden_dim=ckpt_cfg["hidden_dim"],
        num_decoder_layers=ckpt_cfg["num_decoder_layers"],
        num_inference_steps=ckpt_cfg["num_inference_steps"],
        rtc_max_delay=ckpt_cfg["rtc_max_delay"],
        rtc_drop_prob=ckpt_cfg["rtc_drop_prob"],
    )
    config.image_features = ckpt_cfg["image_features"]

    norm_stats = torch.load(ckpt_dir / "norm_stats.pt", map_location="cpu", weights_only=True)

    logger.info("Loading eval dataset (only to grab one batch)")
    eval_ld, eval_ds, _, eval_starts, eval_ends = build_dataset(args.eval_repo_id, config, norm_stats)

    train_idx, _ = split_episode(eval_starts, eval_ends, args.episode_for_batch, val_pct=0.2, seed=0)

    # Build one batch by sampling indices, collating, moving to GPU
    import random as _random

    _random.seed(0)
    chunk = _random.sample(train_idx, args.batch_size)
    samples = [eval_ds[i] for i in chunk]
    collate = make_collate_fn(config)
    batch = collate(samples)
    batch_gpu = {
        k: v.to(device, non_blocking=True) if isinstance(v, torch.Tensor) else v for k, v in batch.items()
    }
    logger.info("Pre-cached batch on GPU (batch_size=%d)", args.batch_size)

    state_dict_path = ckpt_dir / "model.safetensors"

    rows: list[dict] = []
    configs: list[tuple[str, str, int]] = []
    for r in args.ranks:
        configs.append((f"lora_r{r}", "lora", r))
    if args.also_decoder_only:
        configs.append(("decoder_only_ft", "decoder_only", 0))
    if args.also_full_ft:
        configs.append(("full_ft", "full", 0))

    for name, mode, r in configs:
        logger.info("=== %s ===", name)
        policy = build_policy_with_config(state_dict_path, config, device, mode, r)
        try:
            stats = benchmark_one(policy, config, batch_gpu, args.steps, args.lr)
        except torch.cuda.OutOfMemoryError as e:
            logger.warning("  OOM: %s", e)
            stats = {
                "trainable_M": 0,
                "step_ms_median": float("nan"),
                "peak_mem_MB": float("nan"),
                "optim_state_MB": float("nan"),
                "delta_MB_bf16": float("nan"),
            }
        stats["config"] = name
        stats["mode"] = mode
        stats["rank"] = r
        logger.info(
            "  trainable=%.2fM | step=%.1fms (p10=%.1f, p90=%.1f) | peak_mem=%.0fMB | optim=%.0fMB | delta=%.1fMB",
            stats["trainable_M"],
            stats["step_ms_median"],
            stats.get("step_ms_p10", 0),
            stats.get("step_ms_p90", 0),
            stats["peak_mem_MB"],
            stats["optim_state_MB"],
            stats["delta_MB_bf16"],
        )
        rows.append(stats)

        del policy
        torch.cuda.empty_cache()

    # Write CSV
    keys = [
        "config",
        "mode",
        "rank",
        "trainable_M",
        "step_ms_median",
        "step_ms_p10",
        "step_ms_p90",
        "peak_mem_MB",
        "optim_state_MB",
        "delta_MB_bf16",
    ]
    with open(output_csv, "w") as f:
        f.write(",".join(keys) + "\n")
        for r in rows:
            cells = []
            for k in keys:
                v = r.get(k, "")
                if isinstance(v, float):
                    cells.append(f"{v:.4f}" if not np.isnan(v) else "")
                else:
                    cells.append(str(v))
            f.write(",".join(cells) + "\n")
    logger.info("Saved → %s", output_csv)

    logger.info("=== Summary ===")
    logger.info("config           | trainable | step_ms | peak_mem | optim | delta")
    for r in rows:
        logger.info(
            "  %-15s | %7.2fM | %6.1f  | %5.0fMB  | %4.0fMB | %.1fMB",
            r["config"],
            r["trainable_M"],
            r["step_ms_median"],
            r["peak_mem_MB"],
            r["optim_state_MB"],
            r["delta_MB_bf16"],
        )


if __name__ == "__main__":
    main()

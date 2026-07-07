"""Phase D: capacity test with three-way batch mix.

Same sequential setup as Phase C, but each batch is composed of three pools:
  • OLD (original training set replay) — protects broad task distribution
  • FLASHED (union of all previously-flashed eval episodes) — protects past
    corrections
  • NEW (current eval episode) — lands the new correction

Phase C showed that 10% old-data replay alone fails to protect previously-
flashed eval episodes (their loss settles ~2× the peak fit). Adding flashed-
rehearsal to each batch is the natural fix; this phase tests the ratio.

Usage:
    python -m lerobot.policies.hvla.scripts.flash_dagger_phase_d_rehearsal \\
        --checkpoint outputs/flow_s1_no_s2_merged_raw/checkpoints/checkpoint-50000 \\
        --eval-repo-id eval/eval_cylinder_ring_assembly_apr_24 \\
        --train-repo-id thewisp/cylinder_ring_assembly_merged_raw \\
        --output-dir outputs/flash_dagger/phase_d \\
        --episodes 247 76 174 235 245 141 147 276 309 177 \\
        --steps 60 --old-pct 0.10 --flashed-pct 0.25
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from lerobot.policies.hvla.scripts.flash_dagger_lora import apply_lora_to_decoder
from lerobot.policies.hvla.scripts.flash_dagger_phase_a_rank import (
    compute_per_sample_loss,
)
from lerobot.policies.hvla.scripts.flash_dagger_phase_b import (
    build_dataset,
    eval_loss_on_indices,
    make_collate_fn,
    split_episode,
    write_csv,
)

logger = logging.getLogger(__name__)


class ThreeWayMixDataset(torch.utils.data.Dataset):
    """Mixes three pools per batch: old / flashed / new.

    Each __getitem__ rolls a categorical with probabilities (old_pct,
    flashed_pct, 1 - old_pct - flashed_pct):
      • old: sample from `old_indices` in `old_ds` (training-set replay pool)
      • flashed: sample uniformly from `flashed_indices_per_ep` in `flashed_ds`
        (each previously-flashed eval episode contributes equally)
      • new: sample from `new_indices` in `new_ds` (current eval episode train
        portion)

    If `flashed_indices_per_ep` is empty (iter 1), all flashed-pct mass folds
    into new-pct.
    """

    def __init__(
        self,
        old_ds,
        old_indices,
        flashed_ds,
        flashed_indices_per_ep: list[list[int]],
        new_ds,
        new_indices,
        old_pct: float,
        flashed_pct: float,
        length: int,
    ):
        self.old_ds = old_ds
        self.old_idx = list(old_indices)
        self.flashed_ds = flashed_ds
        self.flashed_per_ep = [list(ep) for ep in flashed_indices_per_ep]
        self.new_ds = new_ds
        self.new_idx = list(new_indices)
        self.length = length

        if not self.flashed_per_ep:
            # No flashed pool yet — collapse flashed_pct into new
            old_pct = old_pct
            flashed_pct = 0.0
        self.old_pct = old_pct
        self.flashed_pct = flashed_pct
        # New is the rest

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        r = random.random()
        if r < self.old_pct:
            i = self.old_idx[random.randrange(len(self.old_idx))]
            return self.old_ds[i]
        if r < self.old_pct + self.flashed_pct:
            # Pick an episode uniformly, then a frame uniformly within it
            ep_pool = self.flashed_per_ep[random.randrange(len(self.flashed_per_ep))]
            i = ep_pool[random.randrange(len(ep_pool))]
            return self.flashed_ds[i]
        # Otherwise: new
        i = self.new_idx[random.randrange(len(self.new_idx))]
        return self.new_ds[i]


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--eval-repo-id", required=True)
    parser.add_argument("--train-repo-id", required=True)
    parser.add_argument("--output-dir", default="outputs/flash_dagger/phase_d")
    parser.add_argument(
        "--episodes",
        type=int,
        nargs="+",
        default=[247, 76, 174, 235, 245, 141, 147, 276, 309, 177],
    )
    parser.add_argument("--old-pct", type=float, default=0.10)
    parser.add_argument("--flashed-pct", type=float, default=0.25)
    parser.add_argument("--rank", type=int, default=16)
    parser.add_argument("--alpha", type=float, default=32.0)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--steps", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--eval-passes", type=int, default=1)
    parser.add_argument("--forget-val-size", type=int, default=500)
    parser.add_argument("--replay-pool-size", type=int, default=5000)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", force=True)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.set_float32_matmul_precision("high")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device)
    ckpt_dir = Path(args.checkpoint) / "pretrained_model"
    if not ckpt_dir.is_dir():
        raise FileNotFoundError(f"Expected pretrained_model/ under {args.checkpoint}")

    with open(ckpt_dir / "config.json") as f:
        ckpt_cfg = json.load(f)

    import safetensors.torch as sft

    from lerobot.policies.hvla.s1.flow_matching.config import FlowMatchingS1Config
    from lerobot.policies.hvla.s1.flow_matching.model import FlowMatchingS1Policy

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

    logger.info("Loading eval dataset")
    eval_ld, eval_ds, _, eval_starts, eval_ends = build_dataset(args.eval_repo_id, config, norm_stats)
    logger.info("Eval: %d frames", len(eval_ds))

    logger.info("Loading train dataset")
    train_ld, train_ds, _, _, _ = build_dataset(args.train_repo_id, config, norm_stats)
    logger.info("Train: %d frames", len(train_ds))

    rng = random.Random(args.seed)
    forget_val_indices = sorted(rng.sample(range(len(train_ds)), args.forget_val_size))
    replay_pool = rng.sample(range(len(train_ds)), min(args.replay_pool_size, len(train_ds)))

    splits: dict[int, tuple[list[int], list[int]]] = {}
    for ep in args.episodes:
        tr, va = split_episode(eval_starts, eval_ends, ep, val_pct=0.2, seed=args.seed)
        splits[ep] = (tr, va)

    state_dict = sft.load_file(str(ckpt_dir / "model.safetensors"))

    logger.info("Building model + LoRA (rank=%d, alpha=%.0f)", args.rank, args.alpha)
    policy = FlowMatchingS1Policy(config).to(device)
    policy.load_state_dict(state_dict, strict=False)
    n_lora, n_total = apply_lora_to_decoder(policy, rank=args.rank, alpha=args.alpha)
    policy.to(device)
    policy.train()
    logger.info("LoRA params: %.2fM / %.1fM (%.2f%%)", n_lora / 1e6, n_total / 1e6, 100 * n_lora / n_total)
    logger.info(
        "Mix ratios: old=%.2f flashed=%.2f new=%.2f",
        args.old_pct,
        args.flashed_pct,
        1 - args.old_pct - args.flashed_pct,
    )

    logger.info("=== Baselines ===")
    policy.eval()
    baselines: dict[int | str, float] = {}
    for ep in args.episodes:
        _, va = splits[ep]
        baselines[ep] = eval_loss_on_indices(
            policy, eval_ds, va, config, args.batch_size, device, passes=args.eval_passes
        )
        logger.info("  ep %d baseline = %.4f", ep, baselines[ep])
    baselines["forget"] = eval_loss_on_indices(
        policy, train_ds, forget_val_indices, config, args.batch_size, device, passes=args.eval_passes
    )
    logger.info("  forget baseline = %.4f", baselines["forget"])
    policy.train()

    rows: list[dict] = []
    flashed_train_pools: list[list[int]] = []  # grows as iterations advance

    for i, ep in enumerate(args.episodes, start=1):
        tr, va = splits[ep]
        logger.info(
            "--- Iteration %d/%d: ep=%d (%d train) | flashed-pool=%d episodes ---",
            i,
            len(args.episodes),
            ep,
            len(tr),
            len(flashed_train_pools),
        )

        mix_ds = ThreeWayMixDataset(
            old_ds=train_ds,
            old_indices=replay_pool,
            flashed_ds=eval_ds,
            flashed_indices_per_ep=flashed_train_pools,
            new_ds=eval_ds,
            new_indices=tr,
            old_pct=args.old_pct,
            flashed_pct=args.flashed_pct,
            length=args.steps * args.batch_size,
        )
        loader = DataLoader(
            mix_ds,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=True,
            drop_last=True,
            collate_fn=make_collate_fn(config),
        )

        opt = torch.optim.AdamW(
            [p for p in policy.parameters() if p.requires_grad],
            lr=args.lr,
            weight_decay=0.0,
        )
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt,
            T_max=args.steps,
            eta_min=args.lr * 0.05,
        )

        t_start = time.time()
        policy.train()
        running_loss, running_n = 0.0, 0
        for step, batch in enumerate(loader, start=1):
            batch = {
                k: v.to(device, non_blocking=True) if isinstance(v, torch.Tensor) else v
                for k, v in batch.items()
            }
            with torch.autocast("cuda", dtype=torch.bfloat16):
                losses = compute_per_sample_loss(policy, batch, config)
                loss = losses.mean()
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_([p for p in policy.parameters() if p.requires_grad], 1.0)
            opt.step()
            sched.step()
            running_loss += loss.item()
            running_n += 1
            if step >= args.steps:
                break

        train_elapsed = time.time() - t_start
        logger.info(
            "  trained %d steps in %.1fs (avg train_loss=%.4f)",
            args.steps,
            train_elapsed,
            running_loss / max(running_n, 1),
        )

        t_eval = time.time()
        per_ep_losses: dict[int, float] = {}
        for ep_seen in args.episodes[:i]:
            _, va_seen = splits[ep_seen]
            per_ep_losses[ep_seen] = eval_loss_on_indices(
                policy,
                eval_ds,
                va_seen,
                config,
                args.batch_size,
                device,
                passes=args.eval_passes,
            )
        forget_loss = eval_loss_on_indices(
            policy,
            train_ds,
            forget_val_indices,
            config,
            args.batch_size,
            device,
            passes=args.eval_passes,
        )

        diag = per_ep_losses[ep]
        olds = [per_ep_losses[e] for e in args.episodes[: i - 1]]
        avg_old = float(np.mean(olds)) if olds else 0.0
        worst_old = float(np.max(olds)) if olds else 0.0
        logger.info(
            "  iter=%d ep=%d | new_fit=%.4f (Δ%+.0f%%) | old_avg=%.4f | old_worst=%.4f | forget=%.4f (Δ%+.0f%%) | eval %.1fs",
            i,
            ep,
            diag,
            100 * (diag - baselines[ep]) / max(baselines[ep], 1e-6),
            avg_old,
            worst_old,
            forget_loss,
            100 * (forget_loss - baselines["forget"]) / max(baselines["forget"], 1e-6),
            time.time() - t_eval,
        )

        for ep_eval, loss_val in per_ep_losses.items():
            rows.append(
                {
                    "iter": i,
                    "trained_ep": ep,
                    "eval_ep": ep_eval,
                    "loss": loss_val,
                    "baseline": baselines[ep_eval],
                    "delta_pct": 100 * (loss_val - baselines[ep_eval]) / max(baselines[ep_eval], 1e-6),
                    "added_at_iter": args.episodes.index(ep_eval) + 1,
                    "age_iters": i - (args.episodes.index(ep_eval) + 1),
                }
            )
        rows.append(
            {
                "iter": i,
                "trained_ep": ep,
                "eval_ep": "forget",
                "loss": forget_loss,
                "baseline": baselines["forget"],
                "delta_pct": 100 * (forget_loss - baselines["forget"]) / max(baselines["forget"], 1e-6),
                "added_at_iter": 0,
                "age_iters": -1,
            }
        )

        # Add the just-trained episode's TRAIN portion to the flashed pool
        # (val portion stays held-out; it's only used for eval).
        flashed_train_pools.append(tr)

        del opt, sched, loader, mix_ds

    csv_path = (
        output_dir
        / f"capacity_curves_old{int(args.old_pct * 100):02d}_fl{int(args.flashed_pct * 100):02d}.csv"
    )
    write_csv(csv_path, rows)
    logger.info("Saved curves → %s", csv_path)

    logger.info(
        "=== Phase D summary (old=%.0f%% flashed=%.0f%% new=%.0f%%) ===",
        100 * args.old_pct,
        100 * args.flashed_pct,
        100 * (1 - args.old_pct - args.flashed_pct),
    )
    logger.info("iter | trained_ep | new_fit | old_avg | old_worst | forget")
    for i, ep in enumerate(args.episodes, start=1):
        iter_rows = [r for r in rows if r["iter"] == i]
        new = next(r["loss"] for r in iter_rows if r["eval_ep"] == ep)
        olds = [r["loss"] for r in iter_rows if isinstance(r["eval_ep"], int) and r["eval_ep"] != ep]
        forget = next(r["loss"] for r in iter_rows if r["eval_ep"] == "forget")
        logger.info(
            "  %2d | %4d | %.4f | %.4f | %.4f | %.4f",
            i,
            ep,
            new,
            float(np.mean(olds)) if olds else 0.0,
            float(np.max(olds)) if olds else 0.0,
            forget,
        )


if __name__ == "__main__":
    main()

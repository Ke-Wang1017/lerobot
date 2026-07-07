"""Phase C: capacity test — sequential LoRA flash across many episodes.

The same LoRA adapter is trained on episodes 1..N one at a time. Between
episodes the adapter is NOT reset and previously-seen episodes are NOT
rehearsed (only the original training set's replay buffer mixes in). After
each iteration we evaluate held-out loss on every episode seen so far plus
a fixed forget-validation set.

Two questions this answers:
  • At what point does the adapter saturate — fit on the new episode no
    longer reaches demo-noise floor?
  • At what point does it forget — held-out loss on earlier episodes climbs?

Episode selection: from Phase A's ranking, skip ranks 0–5 (likely demo
errors per visual inspection), require >= 150 frames, take the next 10.

Usage:
    python -m lerobot.policies.hvla.scripts.flash_dagger_phase_c_capacity \\
        --checkpoint outputs/flow_s1_no_s2_merged_raw/checkpoints/checkpoint-50000 \\
        --eval-repo-id eval/eval_cylinder_ring_assembly_apr_24 \\
        --train-repo-id thewisp/cylinder_ring_assembly_merged_raw \\
        --output-dir outputs/flash_dagger/phase_c \\
        --episodes 247 76 174 235 245 141 147 276 309 177 \\
        --steps 60 --replay-pct 0.10
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
    FlashMixDataset,
    build_dataset,
    eval_loss_on_indices,
    make_collate_fn,
    split_episode,
    write_csv,
)

logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--eval-repo-id", required=True)
    parser.add_argument("--train-repo-id", required=True)
    parser.add_argument("--output-dir", default="outputs/flash_dagger/phase_c")
    parser.add_argument(
        "--episodes",
        type=int,
        nargs="+",
        default=[247, 76, 174, 235, 245, 141, 147, 276, 309, 177],
        help="Ordered list of episodes to flash sequentially.",
    )
    parser.add_argument("--replay-pct", type=float, default=0.10)
    parser.add_argument("--rank", type=int, default=16)
    parser.add_argument("--alpha", type=float, default=32.0)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--steps", type=int, default=60, help="Training steps per episode iteration.")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--eval-passes", type=int, default=2)
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
    logger.info("Eval: %d episodes, %d frames", len(eval_starts), len(eval_ds))

    logger.info("Loading train dataset")
    train_ld, train_ds, _, _, _ = build_dataset(args.train_repo_id, config, norm_stats)
    logger.info("Train: %d frames", len(train_ds))

    rng = random.Random(args.seed)
    forget_val_indices = sorted(rng.sample(range(len(train_ds)), args.forget_val_size))
    replay_pool = rng.sample(range(len(train_ds)), min(args.replay_pool_size, len(train_ds)))

    # Pre-compute per-episode train/val splits
    splits: dict[int, tuple[list[int], list[int]]] = {}
    for ep in args.episodes:
        tr, va = split_episode(eval_starts, eval_ends, ep, val_pct=0.2, seed=args.seed)
        splits[ep] = (tr, va)
        logger.info("Episode %d: %d train / %d val frames", ep, len(tr), len(va))

    state_dict = sft.load_file(str(ckpt_dir / "model.safetensors"))

    # Build LoRA-adapted policy ONCE, reused across iterations
    logger.info("Building model + attaching LoRA (rank=%d, alpha=%.0f)", args.rank, args.alpha)
    policy = FlowMatchingS1Policy(config).to(device)
    policy.load_state_dict(state_dict, strict=False)
    n_lora, n_total = apply_lora_to_decoder(policy, rank=args.rank, alpha=args.alpha)
    policy.to(device)
    policy.train()
    logger.info("LoRA params: %.2fM / %.1fM (%.2f%%)", n_lora / 1e6, n_total / 1e6, 100 * n_lora / n_total)

    # Baseline (pre-flash) eval for ALL episodes + forget set, for reference
    logger.info("=== Baselines (pre-LoRA, all episodes) ===")
    policy.eval()
    baselines: dict[int | str, float] = {}
    t0 = time.time()
    for ep in args.episodes:
        _, va = splits[ep]
        baselines[ep] = eval_loss_on_indices(
            policy, eval_ds, va, config, args.batch_size, device, passes=args.eval_passes
        )
        logger.info("  ep %d baseline fit_val = %.4f", ep, baselines[ep])
    baselines["forget"] = eval_loss_on_indices(
        policy, train_ds, forget_val_indices, config, args.batch_size, device, passes=args.eval_passes
    )
    logger.info("  forget baseline = %.4f (%.1fs)", baselines["forget"], time.time() - t0)
    policy.train()

    # Sequential capacity test
    rows: list[dict] = []  # one row per (iteration, eval_target)
    for i, ep in enumerate(args.episodes, start=1):
        tr, va = splits[ep]
        logger.info(
            "--- Iteration %d/%d: training on episode %d (%d train frames) ---",
            i,
            len(args.episodes),
            ep,
            len(tr),
        )

        mix_ds = FlashMixDataset(
            fresh_ds=eval_ds,
            fresh_indices=tr,
            replay_ds=train_ds,
            replay_indices=replay_pool,
            replay_pct=args.replay_pct,
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

        # Reset optimizer + LR schedule (each correction is a "fresh" flash)
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
        avg_train_loss = running_loss / max(running_n, 1)
        logger.info(
            "  trained %d steps in %.1fs (avg train_loss=%.4f)", args.steps, train_elapsed, avg_train_loss
        )

        # Eval held-out fit_val for ALL episodes seen so far + forget set
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
        eval_elapsed = time.time() - t_eval

        # Log compact one-line summary
        diag = per_ep_losses[ep]  # just-trained episode (diagonal)
        worst_old = max((per_ep_losses[e] for e in args.episodes[: i - 1]), default=0.0)
        avg_old = np.mean([per_ep_losses[e] for e in args.episodes[: i - 1]]) if i > 1 else 0.0
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
            eval_elapsed,
        )

        # Persist row per (iteration, episode_evaluated)
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

        del opt, sched, loader, mix_ds

    csv_path = output_dir / "capacity_curves.csv"
    write_csv(csv_path, rows)
    logger.info("Saved curves → %s", csv_path)

    # Compact summary
    logger.info("=== Capacity-test summary ===")
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
            np.mean(olds) if olds else 0.0,
            np.max(olds) if olds else 0.0,
            forget,
        )


if __name__ == "__main__":
    main()

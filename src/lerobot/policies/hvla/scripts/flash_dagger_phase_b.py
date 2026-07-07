"""Phase B of the offline flash-DAgger experiment: LoRA flash sweep.

For each (episode, replay_pct) config:
  1. Load pretrained checkpoint, attach a fresh LoRA adapter on the decoder.
  2. Train N steps on a mixed dataloader: 90-replay_pct% fresh-episode samples
     (80% of the chosen eval episode) + replay_pct% sampled from the original
     training set.
  3. Every K steps, evaluate three losses:
        - fit_val:   held-out 20% of the chosen episode (did the correction land)
        - fit_train: the 80% used for training (sanity — should drop fastest)
        - forget:    fixed 500-frame sample from training set (forgetting tripwire)
  4. Save loss curves to CSV.

Episode picks are hand-curated from Phase A inspection — the loss-based ranking
mixes "model genuinely struggles" cases with "demonstration is a disaster"
cases, and only the former should be trained on.

Usage:
    python -m lerobot.policies.hvla.scripts.flash_dagger_phase_b \\
        --checkpoint outputs/flow_s1_no_s2_merged_raw/checkpoints/checkpoint-50000 \\
        --eval-repo-id eval/eval_cylinder_ring_assembly_apr_24 \\
        --train-repo-id thewisp/cylinder_ring_assembly_merged_raw \\
        --output-dir outputs/flash_dagger/phase_b \\
        --episodes 247 174 299 \\
        --replay-pcts 0.0 0.05 0.10 0.25
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
    override_norm_stats,
    patch_episode_data_index,
)

logger = logging.getLogger(__name__)


def build_dataset(repo_id, config, ckpt_norm, image_size=(224, 224), video_backend="pyav"):
    """Load LeRobotDataset, wrap with FlowMatchingDataset, override norm stats.

    Defaults to pyav video backend. torchcodec is faster sequentially but
    less robust under random access from multiple DataLoader workers — Phase B
    samples randomly within episodes, which can hit torchcodec decode errors
    on some frames.
    """
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from lerobot.policies.hvla.s1.flow_matching.train import FlowMatchingDataset

    ld = LeRobotDataset(repo_id, video_backend=video_backend)
    n_episodes, ep_starts, ep_ends = patch_episode_data_index(ld)
    ds = FlowMatchingDataset(
        ld,
        s2_latents=None,
        chunk_size=config.chunk_size,
        max_delay_seconds=0.0,
        resize_to=image_size,
        image_keys=list(config.image_features.keys()),
    )
    override_norm_stats(ds, ckpt_norm)
    return ld, ds, n_episodes, ep_starts, ep_ends


class FlashMixDataset(torch.utils.data.Dataset):
    """Mixes fresh-episode samples and replay samples at a given ratio.

    Each __getitem__ rolls a Bernoulli with p=replay_pct: heads → sample from
    the replay pool, tails → sample from the fresh subset. Length is set to
    `n_steps * batch_size` so DataLoader yields exactly the right number of
    batches when shuffle=False, drop_last=True.
    """

    def __init__(
        self,
        fresh_ds,
        fresh_indices,
        replay_ds,
        replay_indices,
        replay_pct: float,
        length: int,
    ):
        self.fresh_ds = fresh_ds
        self.fresh_idx = list(fresh_indices)
        self.replay_ds = replay_ds
        self.replay_idx = list(replay_indices)
        self.replay_pct = replay_pct
        self.length = length

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        # random module is reseeded per worker by torch.utils.data.DataLoader,
        # so workers produce independent draws.
        if self.replay_pct > 0 and random.random() < self.replay_pct:
            ridx = self.replay_idx[random.randrange(len(self.replay_idx))]
            return self.replay_ds[ridx]
        fidx = self.fresh_idx[random.randrange(len(self.fresh_idx))]
        return self.fresh_ds[fidx]


def split_episode(ep_starts, ep_ends, episode_idx, val_pct=0.2, seed=42):
    """Random 80/20 split of an episode's frame indices."""
    s = int(ep_starts[episode_idx])
    e = int(ep_ends[episode_idx])
    indices = list(range(s, e))
    random.Random(seed).shuffle(indices)
    n_val = max(1, int(len(indices) * val_pct))
    val = sorted(indices[:n_val])
    train = sorted(indices[n_val:])
    return train, val


def _wanted_keys(config):
    """Keys the model actually consumes — used to filter mixed-schema samples."""
    keys = {"action", "observation.state", "action_is_pad"}
    keys.update(config.image_features.keys())
    return keys


def make_collate_fn(config):
    """Collate that retains only model-relevant keys.

    Eval and train datasets may have slightly different schemas (e.g. one has
    `subtask`, the other has `task_index`). Default torch collate requires
    matching keys across all samples in a batch, which breaks the FlashMixDataset.
    """
    wanted = _wanted_keys(config)

    def _collate(samples):
        out = {}
        common = set(samples[0].keys())
        for s in samples[1:]:
            common &= set(s.keys())
        for k in wanted & common:
            v0 = samples[0][k]
            if isinstance(v0, torch.Tensor):
                out[k] = torch.stack([s[k] for s in samples])
            else:
                out[k] = [s[k] for s in samples]
        return out

    return _collate


def collate(dataset, indices, config=None):
    """Manual collate for eval (no DataLoader workers).

    If `config` is provided, filters to model-relevant keys only — matches
    the train collate behavior so eval and train see identical batch shapes.
    """
    samples = [dataset[i] for i in indices]
    wanted = _wanted_keys(config) if config is not None else set(samples[0].keys())
    out = {}
    for k in samples[0]:
        if k not in wanted:
            continue
        if isinstance(samples[0][k], torch.Tensor):
            out[k] = torch.stack([s[k] for s in samples])
        else:
            out[k] = [s[k] for s in samples]
    return out


@torch.no_grad()
def eval_loss_on_indices(policy, dataset, indices, config, batch_size, device, passes=2):
    """Mean per-sample loss on given indices, averaged over `passes` stochastic forwards."""
    was_training = policy.training
    policy.eval()
    total = 0.0
    n = 0
    for _ in range(passes):
        for i in range(0, len(indices), batch_size):
            chunk = indices[i : i + batch_size]
            batch = collate(dataset, chunk, config=config)
            batch = {
                k: v.to(device, non_blocking=True) if isinstance(v, torch.Tensor) else v
                for k, v in batch.items()
            }
            with torch.autocast("cuda", dtype=torch.bfloat16):
                losses = compute_per_sample_loss(policy, batch, config)
            total += losses.float().sum().item()
            n += losses.numel()
    if was_training:
        policy.train()
    return total / max(n, 1)


def write_csv(path: Path, rows: list[dict]):
    if not rows:
        return
    keys = list(rows[0].keys())
    with open(path, "w") as f:
        f.write(",".join(keys) + "\n")
        for r in rows:
            cells = []
            for k in keys:
                v = r.get(k, "")
                if isinstance(v, float):
                    cells.append(f"{v:.6f}")
                else:
                    cells.append(str(v))
            f.write(",".join(cells) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--eval-repo-id", required=True)
    parser.add_argument("--train-repo-id", required=True)
    parser.add_argument("--output-dir", default="outputs/flash_dagger/phase_b")
    parser.add_argument("--episodes", type=int, nargs="+", default=[247, 174, 299])
    parser.add_argument("--replay-pcts", type=float, nargs="+", default=[0.0, 0.05, 0.10, 0.25])
    parser.add_argument("--rank", type=int, default=16)
    parser.add_argument("--alpha", type=float, default=32.0)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--eval-freq", type=int, default=10)
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

    logger.info("Loading eval dataset (fresh-demo source)")
    eval_ld, eval_ds, _, eval_starts, eval_ends = build_dataset(args.eval_repo_id, config, norm_stats)
    logger.info("Eval: %d episodes, %d frames", len(eval_starts), len(eval_ds))

    logger.info("Loading train dataset (replay + forget-val source)")
    train_ld, train_ds, _, _, _ = build_dataset(args.train_repo_id, config, norm_stats)
    logger.info("Train: %d frames", len(train_ds))

    # Fixed random samples from training set, shared across all configs
    rng = random.Random(args.seed)
    forget_val_indices = sorted(rng.sample(range(len(train_ds)), args.forget_val_size))
    replay_pool = rng.sample(range(len(train_ds)), min(args.replay_pool_size, len(train_ds)))
    logger.info("Forget-val: %d frames | Replay pool: %d frames", len(forget_val_indices), len(replay_pool))

    state_dict_path = ckpt_dir / "model.safetensors"
    state_dict = sft.load_file(str(state_dict_path))

    summary_rows: list[dict] = []
    baselines: dict[int, dict] = {}  # ep_idx → baseline metrics

    for ep_idx in args.episodes:
        train_idx, val_idx = split_episode(eval_starts, eval_ends, ep_idx, val_pct=0.2, seed=args.seed)
        logger.info(
            "=== Episode %d: %d fit-train / %d fit-val frames ===", ep_idx, len(train_idx), len(val_idx)
        )

        # Baseline: original checkpoint (no LoRA) on this episode + forget set
        logger.info("Computing baseline losses (no LoRA)...")
        baseline_policy = FlowMatchingS1Policy(config).to(device)
        baseline_policy.load_state_dict(state_dict, strict=False)
        baseline_policy.eval()

        t0 = time.time()
        b_fit_val = eval_loss_on_indices(
            baseline_policy, eval_ds, val_idx, config, args.batch_size, device, passes=args.eval_passes
        )
        b_fit_train = eval_loss_on_indices(
            baseline_policy, eval_ds, train_idx, config, args.batch_size, device, passes=args.eval_passes
        )
        b_forget = eval_loss_on_indices(
            baseline_policy,
            train_ds,
            forget_val_indices,
            config,
            args.batch_size,
            device,
            passes=args.eval_passes,
        )
        logger.info(
            "Baseline ep=%d: fit_val=%.4f fit_train=%.4f forget=%.4f (%.1fs)",
            ep_idx,
            b_fit_val,
            b_fit_train,
            b_forget,
            time.time() - t0,
        )
        baselines[ep_idx] = {"fit_val": b_fit_val, "fit_train": b_fit_train, "forget": b_forget}

        del baseline_policy
        torch.cuda.empty_cache()

        for replay_pct in args.replay_pcts:
            cfg_name = f"ep{ep_idx}_rp{int(round(replay_pct * 100)):03d}"
            logger.info(
                "--- Config %s (replay=%.2f, rank=%d, lr=%.0e, steps=%d) ---",
                cfg_name,
                replay_pct,
                args.rank,
                args.lr,
                args.steps,
            )

            # Fresh policy + LoRA
            policy = FlowMatchingS1Policy(config).to(device)
            policy.load_state_dict(state_dict, strict=False)
            n_lora, n_total = apply_lora_to_decoder(policy, rank=args.rank, alpha=args.alpha)
            policy.to(device)
            policy.train()
            logger.info(
                "LoRA params: %.2fM trainable / %.1fM total (%.2f%%)",
                n_lora / 1e6,
                n_total / 1e6,
                100 * n_lora / n_total,
            )

            mix_ds = FlashMixDataset(
                fresh_ds=eval_ds,
                fresh_indices=train_idx,
                replay_ds=train_ds,
                replay_indices=replay_pool,
                replay_pct=replay_pct,
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

            curve_rows: list[dict] = [
                {
                    "step": 0,
                    "train_loss": float("nan"),
                    "fit_val_loss": b_fit_val,
                    "fit_train_loss": b_fit_train,
                    "forget_val_loss": b_forget,
                    "lr": opt.param_groups[0]["lr"],
                }
            ]

            t_start = time.time()
            data_iter = iter(loader)
            running_train_loss = 0.0
            running_train_n = 0

            for step in range(1, args.steps + 1):
                batch = next(data_iter)
                batch = {
                    k: v.to(device, non_blocking=True) if isinstance(v, torch.Tensor) else v
                    for k, v in batch.items()
                }

                with torch.autocast("cuda", dtype=torch.bfloat16):
                    losses = compute_per_sample_loss(policy, batch, config)
                    loss = losses.mean()

                opt.zero_grad()
                loss.backward()
                trainable = [p for p in policy.parameters() if p.requires_grad]
                torch.nn.utils.clip_grad_norm_(trainable, 1.0)
                opt.step()
                sched.step()

                running_train_loss += loss.item()
                running_train_n += 1

                if step % args.eval_freq == 0 or step == args.steps:
                    fit_val = eval_loss_on_indices(
                        policy, eval_ds, val_idx, config, args.batch_size, device, passes=args.eval_passes
                    )
                    fit_train = eval_loss_on_indices(
                        policy, eval_ds, train_idx, config, args.batch_size, device, passes=args.eval_passes
                    )
                    forget = eval_loss_on_indices(
                        policy,
                        train_ds,
                        forget_val_indices,
                        config,
                        args.batch_size,
                        device,
                        passes=args.eval_passes,
                    )
                    cur_lr = opt.param_groups[0]["lr"]
                    elapsed = time.time() - t_start
                    avg_train = running_train_loss / max(running_train_n, 1)
                    logger.info(
                        "  step %d/%d | train=%.4f | fit_val=%.4f (Δ%+.1f%%) | "
                        "fit_train=%.4f | forget=%.4f (Δ%+.1f%%) | lr=%.0e | %.1fs",
                        step,
                        args.steps,
                        avg_train,
                        fit_val,
                        100 * (fit_val - b_fit_val) / max(b_fit_val, 1e-6),
                        fit_train,
                        forget,
                        100 * (forget - b_forget) / max(b_forget, 1e-6),
                        cur_lr,
                        elapsed,
                    )
                    curve_rows.append(
                        {
                            "step": step,
                            "train_loss": avg_train,
                            "fit_val_loss": fit_val,
                            "fit_train_loss": fit_train,
                            "forget_val_loss": forget,
                            "lr": cur_lr,
                        }
                    )
                    running_train_loss = 0.0
                    running_train_n = 0

            curve_csv = output_dir / f"{cfg_name}_curve.csv"
            write_csv(curve_csv, curve_rows)
            logger.info("Saved curve → %s", curve_csv)

            final = curve_rows[-1]
            summary_rows.append(
                {
                    "episode": ep_idx,
                    "replay_pct": replay_pct,
                    "rank": args.rank,
                    "lr": args.lr,
                    "steps": args.steps,
                    "baseline_fit_val": b_fit_val,
                    "baseline_forget": b_forget,
                    "final_train_loss": final["train_loss"],
                    "final_fit_val": final["fit_val_loss"],
                    "final_forget": final["forget_val_loss"],
                    "fit_drop_pct": 100 * (1 - final["fit_val_loss"] / max(b_fit_val, 1e-6)),
                    "forget_drift_pct": 100 * (final["forget_val_loss"] / max(b_forget, 1e-6) - 1),
                }
            )

            del policy, opt, sched, loader, mix_ds, data_iter
            torch.cuda.empty_cache()

    summary_csv = output_dir / "summary.csv"
    write_csv(summary_csv, summary_rows)
    logger.info("Saved summary → %s", summary_csv)

    logger.info("=== Sweep summary ===")
    logger.info("ep   | replay% | fit_drop% | forget_drift% | final_fit_val | final_forget")
    for r in summary_rows:
        logger.info(
            "  %3d | %6.0f%% | %+8.1f%% | %+12.1f%% | %12.4f | %12.4f",
            r["episode"],
            r["replay_pct"] * 100,
            r["fit_drop_pct"],
            r["forget_drift_pct"],
            r["final_fit_val"],
            r["final_forget"],
        )


if __name__ == "__main__":
    main()

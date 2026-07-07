"""Phase A of the offline flash-DAgger experiment: rank eval episodes by BC loss.

Loads the pretrained Flow Matching S1 checkpoint and computes the flow-matching
loss on each episode of an eval dataset, averaging across multiple stochastic
forward passes (independent t and noise samples) to reduce variance. Saves a
CSV ranked by mean loss (worst → best) so Phase B can pick the hardest
episodes as "operator-would-intervene" cases.

Norm-stats note: the FlowMatchingDataset wrapper computes its own action/state
stats from whatever dataset it wraps. The model was trained with stats from the
training set, so we must override the eval dataset's stats with the checkpoint
ones — otherwise the loss reflects normalization mismatch, not policy quality.

Usage:
    python -m lerobot.policies.hvla.scripts.flash_dagger_phase_a_rank \\
        --checkpoint outputs/flow_s1_no_s2_merged_raw/checkpoints/checkpoint-50000 \\
        --eval-repo-id eval/eval_cylinder_ring_assembly_apr_24 \\
        --output-csv outputs/flash_dagger/eval_episode_loss.csv \\
        --epochs 3
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

logger = logging.getLogger(__name__)


def compute_per_sample_loss(policy, batch, config):
    """Same forward as policy.forward() but returns per-sample loss [B] instead of scalar.

    Mirrors FlowMatchingS1Policy.forward + FlowMatchingS1Model.forward exactly
    (image-key mapping → encode_observations → denoise_step → flow MSE with RTC
    prefix, loss mask, action padding). Kept inline rather than monkey-patched
    so the production forward stays untouched.
    """
    # Mirror Policy.forward: add OBS_IMAGES list (the inner Model expects this
    # rather than reading individual image keys).
    if config.image_features:
        batch = dict(batch)
        batch["observation.images"] = [batch[key] for key in config.image_features]

    actions = batch["action"]
    B, T, A = actions.shape
    device = actions.device

    context = policy.model.encode_observations(batch)

    t_beta = (
        torch.distributions.Beta(
            config.time_sampling_beta_alpha,
            config.time_sampling_beta_beta,
        )
        .sample((B,))
        .to(device)
    )
    t_flow = t_beta * (config.time_max - config.time_min) + config.time_min

    noise = torch.randn_like(actions)
    t_expand = t_flow[:, None, None]
    x_t = t_expand * noise + (1 - t_expand) * actions
    u_target = noise - actions

    per_pos_t = t_flow[:, None].expand(B, T).clone()
    loss_mask = torch.ones(B, T, 1, device=device)

    if config.rtc_max_delay > 0:
        max_d = min(config.rtc_max_delay, T - 1)
        delays = torch.randint(1, max_d + 1, (B,), device=device)
        drop_mask = torch.rand(B, device=device) < config.rtc_drop_prob
        delays = delays * (~drop_mask).long()
        for b in range(B):
            d = delays[b].item()
            if d > 0:
                x_t[b, :d] = actions[b, :d]
                per_pos_t[b, :d] = 0.0
                loss_mask[b, :d] = 0.0

    v_pred = policy.model.denoise_step(x_t, context, per_pos_t)

    if "action_is_pad" in batch:
        loss_mask = loss_mask * (~batch["action_is_pad"].unsqueeze(-1)).float()

    mse = F.mse_loss(v_pred.float(), u_target.float(), reduction="none")
    denom = loss_mask.sum(dim=(1, 2)).clamp(min=1.0)
    per_sample = (mse * loss_mask).sum(dim=(1, 2)) / denom / A
    return per_sample


def patch_episode_data_index(lerobot_dataset):
    """Populate `episode_data_index` from the v3.0 `episode_index` column.

    FlowMatchingDataset reads `episode_data_index["from"]/["to"]` to clip
    action chunks at episode boundaries. v3.0 LeRobotDatasets don't expose
    that attribute — boundaries live in the `episode_index` column instead.
    Monkey-patching keeps the train-side dataset code unchanged.
    """
    ep_index = np.asarray(lerobot_dataset.hf_dataset["episode_index"])
    assert (np.diff(ep_index) >= 0).all(), "episode_index must be monotonically non-decreasing"
    n_episodes = int(ep_index.max()) + 1
    eps = np.arange(n_episodes)
    starts = np.searchsorted(ep_index, eps, side="left")
    ends = np.searchsorted(ep_index, eps, side="right")
    lerobot_dataset.episode_data_index = {
        "from": torch.from_numpy(starts),
        "to": torch.from_numpy(ends),
    }
    return n_episodes, starts, ends


def override_norm_stats(dataset, ckpt_norm):
    """Renormalize preloaded actions/states using checkpoint stats."""
    raw_actions = dataset._all_actions * dataset.action_std + dataset.action_mean
    dataset._all_actions = (raw_actions - ckpt_norm["action_mean"]) / ckpt_norm["action_std"]
    dataset.action_mean = ckpt_norm["action_mean"]
    dataset.action_std = ckpt_norm["action_std"]

    if dataset._all_states is not None and "state_mean" in ckpt_norm:
        raw_states = dataset._all_states * dataset.state_std + dataset.state_mean
        dataset._all_states = (raw_states - ckpt_norm["state_mean"]) / ckpt_norm["state_std"]
        dataset.state_mean = ckpt_norm["state_mean"]
        dataset.state_std = ckpt_norm["state_std"]


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--checkpoint", required=True, help="Path to checkpoint dir (must contain pretrained_model/)"
    )
    parser.add_argument(
        "--eval-repo-id",
        required=True,
        help="LeRobot repo_id of eval dataset (e.g. eval/eval_cylinder_ring_assembly_apr_24)",
    )
    parser.add_argument("--output-csv", default="eval_episode_loss.csv")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--epochs",
        type=int,
        default=3,
        help="Stochastic averaging passes (more = lower variance, linear cost)",
    )
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", force=True)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.set_float32_matmul_precision("high")

    device = torch.device(args.device)
    ckpt_dir = Path(args.checkpoint) / "pretrained_model"
    if not ckpt_dir.is_dir():
        raise FileNotFoundError(f"Expected pretrained_model/ under {args.checkpoint}")

    # Load checkpoint config
    with open(ckpt_dir / "config.json") as f:
        ckpt_cfg = json.load(f)
    logger.info(
        "Checkpoint config: chunk=%d, hidden=%d, dec_layers=%d, rtc_max_delay=%d",
        ckpt_cfg["chunk_size"],
        ckpt_cfg["hidden_dim"],
        ckpt_cfg["num_decoder_layers"],
        ckpt_cfg["rtc_max_delay"],
    )

    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from lerobot.policies.hvla.s1.flow_matching.config import FlowMatchingS1Config
    from lerobot.policies.hvla.s1.flow_matching.model import FlowMatchingS1Policy
    from lerobot.policies.hvla.s1.flow_matching.train import FlowMatchingDataset

    config = FlowMatchingS1Config(
        chunk_size=ckpt_cfg["chunk_size"],
        hidden_dim=ckpt_cfg["hidden_dim"],
        num_decoder_layers=ckpt_cfg["num_decoder_layers"],
        num_inference_steps=ckpt_cfg["num_inference_steps"],
        rtc_max_delay=ckpt_cfg["rtc_max_delay"],
        rtc_drop_prob=ckpt_cfg["rtc_drop_prob"],
    )
    config.image_features = ckpt_cfg["image_features"]

    # Eval dataset
    logger.info("Loading eval dataset: %s", args.eval_repo_id)
    eval_dataset = LeRobotDataset(args.eval_repo_id)
    n_episodes, ep_starts, ep_ends = patch_episode_data_index(eval_dataset)
    n_frames = len(eval_dataset)
    logger.info("Eval dataset: %d episodes, %d frames", n_episodes, n_frames)

    dataset = FlowMatchingDataset(
        eval_dataset,
        s2_latents=None,
        chunk_size=config.chunk_size,
        max_delay_seconds=0.0,
        resize_to=(224, 224),
        image_keys=list(config.image_features.keys()),
    )

    norm_stats = torch.load(ckpt_dir / "norm_stats.pt", map_location="cpu", weights_only=True)
    override_norm_stats(dataset, norm_stats)
    logger.info("Norm stats overridden from checkpoint")

    # Episode lookup: sample_idx → episode_idx
    ep_lookup = np.zeros(n_frames, dtype=np.int64)
    for ep_idx in range(n_episodes):
        ep_lookup[ep_starts[ep_idx] : ep_ends[ep_idx]] = ep_idx

    # Build model and load weights
    logger.info("Building model (DINOv2 may download on first run)...")
    policy = FlowMatchingS1Policy(config).to(device)
    total_params = sum(p.numel() for p in policy.parameters())
    logger.info("Model params: %.1fM", total_params / 1e6)

    import safetensors.torch as sft

    state_dict = sft.load_file(str(ckpt_dir / "model.safetensors"))
    missing, unexpected = policy.load_state_dict(state_dict, strict=False)
    if missing:
        logger.warning("Missing keys when loading: %d (e.g. %s)", len(missing), missing[:3])
    if unexpected:
        logger.warning("Unexpected keys when loading: %d (e.g. %s)", len(unexpected), unexpected[:3])
    policy.eval()

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
    )

    ep_loss_sum = np.zeros(n_episodes, dtype=np.float64)
    ep_loss_count = np.zeros(n_episodes, dtype=np.int64)

    use_amp = device.type == "cuda"

    for epoch in range(args.epochs):
        logger.info(
            "Epoch %d/%d — accumulating per-episode losses (%d batches)", epoch + 1, args.epochs, len(loader)
        )
        t_start = time.time()
        with torch.no_grad():
            for batch_i, batch in enumerate(loader):
                start = batch_i * args.batch_size
                actual_b = batch["action"].shape[0]
                indices = np.arange(start, start + actual_b)

                batch = {
                    k: v.to(device, non_blocking=True) if isinstance(v, torch.Tensor) else v
                    for k, v in batch.items()
                }

                with torch.autocast("cuda", dtype=torch.bfloat16, enabled=use_amp):
                    per_sample = compute_per_sample_loss(policy, batch, config)

                per_sample = per_sample.float().cpu().numpy()
                eps = ep_lookup[indices]
                # Vectorized scatter-add
                np.add.at(ep_loss_sum, eps, per_sample)
                np.add.at(ep_loss_count, eps, 1)

                if (batch_i + 1) % 50 == 0 or (batch_i + 1) == len(loader):
                    rate = (batch_i + 1) * args.batch_size / max(time.time() - t_start, 1e-6)
                    logger.info("  batch %d/%d (%.0f frames/s)", batch_i + 1, len(loader), rate)
        logger.info("Epoch %d done in %.1fs", epoch + 1, time.time() - t_start)

    ep_mean_loss = ep_loss_sum / ep_loss_count.clip(min=1)

    # Save sorted CSV (worst → best)
    output_path = Path(args.output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    order = np.argsort(-ep_mean_loss)
    with open(output_path, "w") as f:
        f.write("rank,episode_idx,mean_loss,n_frames,n_samples\n")
        for rank, ep_idx in enumerate(order):
            n_frames_ep = ep_ends[ep_idx] - ep_starts[ep_idx]
            f.write(f"{rank},{ep_idx},{ep_mean_loss[ep_idx]:.6f},{n_frames_ep},{ep_loss_count[ep_idx]}\n")
    logger.info("Saved ranked losses → %s", output_path)

    # Summary
    logger.info(
        "Loss distribution: min=%.4f, median=%.4f, max=%.4f, max/median=%.2fx",
        ep_mean_loss.min(),
        np.median(ep_mean_loss),
        ep_mean_loss.max(),
        ep_mean_loss.max() / max(np.median(ep_mean_loss), 1e-6),
    )
    logger.info("Top 5 hardest episodes (candidates for flash-DAgger):")
    for rank, ep_idx in enumerate(order[:5]):
        n_frames_ep = ep_ends[ep_idx] - ep_starts[ep_idx]
        logger.info(
            "  rank=%d ep=%d loss=%.4f frames=%d (~%.1fs @ 30Hz)",
            rank,
            ep_idx,
            ep_mean_loss[ep_idx],
            n_frames_ep,
            n_frames_ep / 30.0,
        )
    logger.info("Bottom 3 easiest episodes (sanity check):")
    for ep_idx in order[-3:]:
        logger.info("  ep=%d loss=%.4f", ep_idx, ep_mean_loss[ep_idx])


if __name__ == "__main__":
    main()

#!/usr/bin/env python
"""Analyze grip drops: find similar training frames, run model on both, identify differences.

Usage:
    # Step 1: Run inference with --save-grip-drops /tmp/grip_drops to collect drops
    # Step 2: Run this script
    python scripts/analyze_grip_drops.py \
        --drop-dir /tmp/grip_drops \
        --dataset thewisp/cylinder_ring_assembly \
        --checkpoint outputs/flow_s1_hvla_v6/checkpoint-40000/model.safetensors \
        --num-nearest 20 \
        --num-repeats 20
"""

import argparse
import logging
import os

import numpy as np
import torch

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger(__name__)


def load_drop_states(drop_dir: str) -> list[dict]:
    """Load all grip_drop_* and infer_drop_* states from the drop directory."""
    drops = []
    for entry in sorted(os.listdir(drop_dir)):
        if not (entry.startswith("grip_drop_") or entry.startswith("infer_drop_")):
            continue
        path = os.path.join(drop_dir, entry)
        state_path = os.path.join(path, "state.npy")
        chunk_path = os.path.join(path, "chunk.npy")
        if not os.path.exists(state_path):
            continue
        drops.append(
            {
                "name": entry,
                "state": np.load(state_path),
                "chunk": np.load(chunk_path) if os.path.exists(chunk_path) else None,
                "path": path,
            }
        )
    return drops


def find_nearest_training_frames(
    drop_state: np.ndarray,
    all_states: np.ndarray,
    all_episodes: np.ndarray,
    num_nearest: int = 20,
    max_per_episode: int = 3,
) -> list[dict]:
    """Find nearest training frames by Euclidean distance on robot state."""
    dists = np.linalg.norm(all_states - drop_state[None, :], axis=1)
    sorted_indices = np.argsort(dists)

    results = []
    ep_counts = {}
    for idx in sorted_indices:
        idx = int(idx)
        ep = int(all_episodes[idx])
        if ep_counts.get(ep, 0) >= max_per_episode:
            continue
        ep_counts[ep] = ep_counts.get(ep, 0) + 1
        results.append(
            {
                "frame_idx": idx,
                "episode": ep,
                "distance": dists[idx],
                "state": all_states[idx],
            }
        )
        if len(results) >= num_nearest:
            break
    return results


def build_batch_from_dataset(dataset, frame_idx: int, image_keys: list[str], device: torch.device) -> dict:
    """Build a model-ready batch from a training dataset frame."""
    sample = dataset[frame_idx]
    batch = {}
    for key in image_keys:
        img = sample[key]  # [C, H, W] float
        if img.shape[1] != 224 or img.shape[2] != 224:
            import torchvision.transforms.functional as TF

            img = TF.resize(img, [224, 224], interpolation=TF.InterpolationMode.BILINEAR, antialias=True)
        batch[key] = img.unsqueeze(0).to(device)
    batch["observation.state"] = sample["observation.state"].unsqueeze(0).to(device)
    return batch


def build_batch_from_drop(
    drop: dict, image_keys: list[str], device: torch.device, resize_to: tuple = (224, 224)
) -> dict:
    """Build a model-ready batch from a saved grip drop observation."""
    import cv2

    batch = {}
    # Load images from saved JPGs
    for key in image_keys:
        cam_name = key.split(".")[-1]
        # Try different naming patterns
        for pattern in [
            f"{cam_name}.jpg",
            f"observation_images_{cam_name}.jpg",
            f"{cam_name.replace('.', '_')}.jpg",
        ]:
            img_path = os.path.join(drop["path"], pattern)
            if os.path.exists(img_path):
                img = cv2.imread(img_path)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = cv2.resize(img, (resize_to[1], resize_to[0]))
                img_t = torch.from_numpy(img).permute(2, 0, 1).float().div_(255.0)
                batch[key] = img_t.unsqueeze(0).to(device)
                break
    batch["observation.state"] = torch.from_numpy(drop["state"]).float().unsqueeze(0).to(device)
    return batch


def run_model_repeated(policy, batch, s2_latent, num_repeats: int, device: torch.device) -> dict:
    """Run model multiple times and analyze gripper predictions."""
    from lerobot.policies.hvla.s1.protocol import S2_LATENT_KEY

    # Inject S2 latent
    batch[S2_LATENT_KEY] = s2_latent.unsqueeze(0).to(device)
    batch["observation.s2_latent_age"] = torch.zeros(1, 1, device=device)

    # Normalize state if policy has norm stats
    inner = policy.model if hasattr(policy, "model") else policy
    if hasattr(inner, "_state_mean") and inner._state_mean is not None:
        batch["observation.state"] = (
            batch["observation.state"] - inner._state_mean.to(device)
        ) / inner._state_std.to(device)

    gripper_r_chunks = []  # index 13
    gripper_l_chunks = []  # index 6
    all_chunks = []

    for _ in range(num_repeats):
        with torch.no_grad():
            actions = policy.predict_action_chunk(batch)  # [1, chunk_size, 14]
        chunk = actions.cpu().numpy()[0]
        all_chunks.append(chunk)
        gripper_r_chunks.append(chunk[:20, 13])
        gripper_l_chunks.append(chunk[:20, 6])

    policy.reset()

    gripper_r = np.array(gripper_r_chunks)  # [num_repeats, 20]
    gripper_l = np.array(gripper_l_chunks)

    # Detect "drop" = gripper jumps > 10 in first 10 positions
    r_drops = 0
    l_drops = 0
    for i in range(num_repeats):
        r_diff = np.max(np.abs(np.diff(gripper_r[i, :10])))
        l_diff = np.max(np.abs(np.diff(gripper_l[i, :10])))
        if r_diff > 10:
            r_drops += 1
        if l_diff > 10:
            l_drops += 1

    return {
        "r_drop_rate": r_drops / num_repeats,
        "l_drop_rate": l_drops / num_repeats,
        "r_mean_trajectory": gripper_r.mean(axis=0),
        "r_std_trajectory": gripper_r.std(axis=0),
        "l_mean_trajectory": gripper_l.mean(axis=0),
        "l_std_trajectory": gripper_l.std(axis=0),
        "all_chunks": np.array(all_chunks),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--drop-dir", default="/tmp/grip_drops")
    parser.add_argument("--dataset", default="thewisp/cylinder_ring_assembly")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--num-nearest", type=int, default=20)
    parser.add_argument("--num-repeats", type=int, default=20)
    parser.add_argument("--max-drops", type=int, default=3, help="Max drop states to analyze")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    device = torch.device(args.device)

    # Load drops
    drops = load_drop_states(args.drop_dir)
    logger.info("Found %d drop states in %s", len(drops), args.drop_dir)
    if not drops:
        logger.error("No drop states found. Run inference with --save-grip-drops first.")
        return

    # Load dataset
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    dataset = LeRobotDataset(args.dataset)
    logger.info("Dataset: %d frames, %d episodes", len(dataset), dataset.meta.total_episodes)

    # Pre-load all states from parquet (fast — no video decoding)
    logger.info("Loading all states from parquet...")
    import glob

    import pyarrow.parquet as pq

    parquet_files = sorted(glob.glob(str(dataset.root / "data" / "**" / "*.parquet"), recursive=True))
    all_rows = []
    for pf in parquet_files:
        table = pq.read_table(pf, columns=["observation.state", "episode_index"])
        all_rows.append(table)
    import pyarrow as pa

    merged = pa.concat_tables(all_rows)
    # observation.state is stored as list<float>, convert to 2D numpy
    all_states = np.array(merged.column("observation.state").to_pylist(), dtype=np.float32)
    all_episodes = merged.column("episode_index").to_numpy()
    logger.info("States loaded: shape %s (from parquet)", all_states.shape)

    # Build episode index
    episode_starts, episode_ends = {}, {}
    for i, ep in enumerate(all_episodes):
        ep = int(ep)
        if ep not in episode_starts:
            episode_starts[ep] = i
        episode_ends[ep] = i + 1

    # Load policy
    from lerobot.policies.hvla.s1.flow_matching import FlowMatchingS1Config, FlowMatchingS1Policy

    config = FlowMatchingS1Config()
    policy = FlowMatchingS1Policy.from_pretrained(args.checkpoint, config=config)
    policy.to(device).eval()
    image_keys = list(config.image_features.keys())

    # Load S2 latent (use zeros — same as what the model sees when S2 hasn't updated)
    s2_latent = torch.zeros(2048)

    # Analyze each drop
    for _drop_idx, drop in enumerate(drops[: args.max_drops]):
        logger.info("\n" + "=" * 80)
        logger.info("ANALYZING DROP: %s", drop["name"])
        logger.info("State: %s", " ".join(f"{v:.1f}" for v in drop["state"]))
        if drop["chunk"] is not None:
            logger.info(
                "Chunk R gripper[0:15]: %s", " ".join(f"{drop['chunk'][i, 13]:.0f}" for i in range(15))
            )

        # Find nearest training frames
        nearest = find_nearest_training_frames(
            drop["state"],
            all_states,
            all_episodes,
            num_nearest=args.num_nearest,
        )

        logger.info("\nNearest training frames:")
        for nn in nearest[:5]:
            logger.info(
                "  frame=%d ep=%d dist=%.2f state=%s",
                nn["frame_idx"],
                nn["episode"],
                nn["distance"],
                " ".join(f"{v:.1f}" for v in nn["state"]),
            )

        # Run model on drop observation
        drop_batch = build_batch_from_drop(drop, image_keys, device)
        if len(drop_batch) < len(image_keys) + 1:  # images + state
            logger.warning("Could not load all images for drop %s, skipping", drop["name"])
            continue

        logger.info("\nRunning model on DROP observation (%d repeats)...", args.num_repeats)
        drop_results = run_model_repeated(policy, drop_batch, s2_latent, args.num_repeats, device)
        logger.info(
            "  R gripper drop rate: %.0f%% (%d/%d)",
            drop_results["r_drop_rate"] * 100,
            int(drop_results["r_drop_rate"] * args.num_repeats),
            args.num_repeats,
        )
        logger.info(
            "  R gripper mean[0:10]: %s", " ".join(f"{v:.1f}" for v in drop_results["r_mean_trajectory"][:10])
        )

        # Run model on nearest training frames
        logger.info(
            "\nRunning model on TRAINING observations (%d nearest × %d repeats)...",
            min(5, len(nearest)),
            args.num_repeats,
        )
        for nn in nearest[:5]:
            train_batch = build_batch_from_dataset(dataset, nn["frame_idx"], image_keys, device)
            train_results = run_model_repeated(policy, train_batch, s2_latent, args.num_repeats, device)
            logger.info(
                "  frame=%d ep=%d dist=%.2f | R drop=%.0f%% | R mean[0:10]: %s",
                nn["frame_idx"],
                nn["episode"],
                nn["distance"],
                train_results["r_drop_rate"] * 100,
                " ".join(f"{v:.1f}" for v in train_results["r_mean_trajectory"][:10]),
            )

        # --- Camera swap ablation ---
        # Find the best training frame (lowest drop rate, closest distance)
        # and swap cameras one by one to isolate which visual difference causes drops
        best_nn = None
        best_nn_drop_rate = 1.0
        for nn in nearest[:5]:
            train_batch = build_batch_from_dataset(dataset, nn["frame_idx"], image_keys, device)
            result = run_model_repeated(policy, train_batch, s2_latent, args.num_repeats, device)
            if result["r_drop_rate"] < best_nn_drop_rate:
                best_nn_drop_rate = result["r_drop_rate"]
                best_nn = nn
                best_nn_batch = train_batch

        if best_nn is not None and best_nn_drop_rate < 0.1 and drop_results["r_drop_rate"] > 0.3:
            logger.info("\n--- CAMERA SWAP ABLATION for %s ---", drop["name"])
            logger.info(
                "Baseline: DROP obs = %.0f%% drops | TRAIN frame=%d = %.0f%% drops",
                drop_results["r_drop_rate"] * 100,
                best_nn["frame_idx"],
                best_nn_drop_rate * 100,
            )

            # For each camera: take training batch, replace ONE camera with inference image
            for cam_key in image_keys:
                if cam_key not in drop_batch or cam_key not in best_nn_batch:
                    continue
                # Start from training batch (0% drops), swap in ONE inference camera
                hybrid_batch = {
                    k: v.clone() if isinstance(v, torch.Tensor) else v for k, v in best_nn_batch.items()
                }
                hybrid_batch[cam_key] = drop_batch[cam_key].clone()
                hybrid_result = run_model_repeated(policy, hybrid_batch, s2_latent, args.num_repeats, device)
                cam_name = cam_key.split(".")[-1]
                logger.info("  Swap %s → infer: R drop=%.0f%%", cam_name, hybrid_result["r_drop_rate"] * 100)

            # Also test: take inference batch, replace ONE camera with training image
            logger.info("  --- Reverse (infer base, swap in training cameras) ---")
            for cam_key in image_keys:
                if cam_key not in drop_batch or cam_key not in best_nn_batch:
                    continue
                hybrid_batch = {
                    k: v.clone() if isinstance(v, torch.Tensor) else v for k, v in drop_batch.items()
                }
                hybrid_batch[cam_key] = best_nn_batch[cam_key].clone()
                hybrid_result = run_model_repeated(policy, hybrid_batch, s2_latent, args.num_repeats, device)
                cam_name = cam_key.split(".")[-1]
                logger.info("  Swap %s → train: R drop=%.0f%%", cam_name, hybrid_result["r_drop_rate"] * 100)

            # Test swapping ALL cameras at once (should match training drop rate)
            all_train_batch = {
                k: v.clone() if isinstance(v, torch.Tensor) else v for k, v in best_nn_batch.items()
            }
            # Keep inference state, use all training images
            all_train_batch["observation.state"] = drop_batch["observation.state"].clone()
            all_train_result = run_model_repeated(
                policy, all_train_batch, s2_latent, args.num_repeats, device
            )
            logger.info(
                "  All cameras → train (infer state): R drop=%.0f%%", all_train_result["r_drop_rate"] * 100
            )
        else:
            logger.info(
                "\n--- SKIP ablation: no clean training baseline found (best=%.0f%% drops) ---",
                best_nn_drop_rate * 100,
            )

        logger.info("\n--- SUMMARY for %s ---", drop["name"])
        logger.info("Drop observation R drop rate: %.0f%%", drop_results["r_drop_rate"] * 100)

    logger.info("\n" + "=" * 80)
    logger.info("DONE. Analyzed %d drops.", min(len(drops), args.max_drops))


if __name__ == "__main__":
    main()

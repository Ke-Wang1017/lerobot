"""Evaluate S2 VLM subtask prediction accuracy on the training set.

Runs S2 with subtask_only=True on each frame, AR-decodes the subtask text,
and compares to ground truth subtask labels from the dataset.

Usage:
    python src/lerobot/policies/hvla/scripts/analysis/eval_s2_subtask.py \
        --checkpoint ~/.cache/lerobot/converted/soarm-pi05-fast-7998-pytorch/model.safetensors \
        --dataset-dir /home/feit/.cache/huggingface/lerobot/thewisp/cylinder_ring_assembly \
        --task "assemble cylinder into ring" \
        --device cuda \
        --stride 10
"""

import argparse
import logging
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


def load_s2_model(checkpoint: str, device: str):
    from lerobot.policies.hvla.s2.config import S2VLMConfig
    from lerobot.policies.hvla.s2.model import S2VLMModel

    config = S2VLMConfig()
    model = S2VLMModel.from_pretrained(checkpoint, config)
    model.to(device).eval()
    logger.info(
        "S2 VLM loaded on %s (%dM params)", device, sum(p.numel() for p in model.parameters()) // 1_000_000
    )
    return model, config


def load_dataset_metadata(dataset_dir: str):
    """Load subtask labels and episode boundaries from parquet."""
    data_path = Path(dataset_dir) / "data" / "chunk-000" / "file-000.parquet"
    df = pq.read_table(str(data_path)).to_pandas()
    return df


def main():
    parser = argparse.ArgumentParser(description="Evaluate S2 subtask prediction accuracy")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--dataset-dir", required=True)
    parser.add_argument("--task", default="assemble cylinder into ring")
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--stride", type=int, default=10, help="Evaluate every N-th frame (1=all, 10=10%% sample)"
    )
    parser.add_argument("--max-frames", type=int, default=0, help="Max frames to evaluate (0=all)")
    parser.add_argument(
        "--random-sample", action="store_true", help="Randomly sample frames instead of fixed stride"
    )
    parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducible sampling")
    parser.add_argument("--image-keys", default="base_0_rgb,left_wrist_0_rgb,right_wrist_0_rgb,base_1_rgb")
    args = parser.parse_args()

    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from lerobot.policies.hvla.s2.preprocessing import preprocess_images
    from lerobot.policies.hvla.s2.tokenizer import PaligemmaTokenizer

    # Load model
    model, config = load_s2_model(args.checkpoint, args.device)
    tokenizer = PaligemmaTokenizer(max_len=config.max_token_len)
    image_keys = tuple(args.image_keys.split(","))

    # Pre-tokenize prompt (subtask_only=True for AR subtask decoding)
    token_ids, token_mask = tokenizer.tokenize_prompt(
        args.task,
        low_prompt="",
        state=None,
        subtask_only=True,
    )
    lang_tokens = torch.from_numpy(token_ids).unsqueeze(0).long().to(args.device)
    lang_masks = torch.from_numpy(token_mask).unsqueeze(0).bool().to(args.device)

    # Load dataset
    logger.info("Loading dataset from %s...", args.dataset_dir)
    dataset = LeRobotDataset(args.dataset_dir, video_backend="pyav")
    df = load_dataset_metadata(args.dataset_dir)
    n_frames = len(dataset)
    logger.info("Dataset: %d frames, %d episodes", n_frames, df["episode_index"].nunique())

    # Subtask distribution
    subtask_counts = Counter(df["subtask"].tolist())
    logger.info("Ground truth subtask distribution:")
    for st, count in subtask_counts.most_common():
        logger.info('  "%s": %d frames (%.1f%%)', st, count, 100 * count / n_frames)

    # LeRobot image key → S2 camera key mapping
    IMAGE_KEY_MAP = {
        "observation.images.front": "base_0_rgb",
        "observation.images.top": "base_1_rgb",
        "observation.images.wrist_left": "left_wrist_0_rgb",
        "observation.images.left_wrist": "left_wrist_0_rgb",
        "observation.images.wrist_right": "right_wrist_0_rgb",
        "observation.images.right_wrist": "right_wrist_0_rgb",
    }
    lerobot_image_keys = [k for k, v in IMAGE_KEY_MAP.items() if v in image_keys]

    # Evaluation loop — random sample or fixed stride
    if args.random_sample:
        import random

        if args.seed is not None:
            random.seed(args.seed)
        n_sample = args.max_frames if args.max_frames > 0 else n_frames // args.stride
        indices = sorted(random.sample(range(n_frames), min(n_sample, n_frames)))
        logger.info("\nEvaluating %d randomly sampled frames (seed=%s)...", len(indices), args.seed)
    else:
        indices = list(range(0, n_frames, args.stride))
        if args.max_frames > 0:
            indices = indices[: args.max_frames]
        logger.info("\nEvaluating %d frames (stride=%d)...", len(indices), args.stride)

    results = []  # (idx, gt_subtask, pred_subtask, match)
    per_subtask = defaultdict(lambda: {"correct": 0, "total": 0, "preds": Counter()})
    timings = []

    for eval_i, idx in enumerate(indices):
        item = dataset[idx]
        gt_subtask = df.iloc[idx]["subtask"]

        # Prepare images
        images = {}
        for lr_key in lerobot_image_keys:
            if lr_key in item:
                img = item[lr_key].numpy()
                if img.ndim == 3 and img.shape[0] in (1, 3):
                    img = img.transpose(1, 2, 0)
                img = (img * 255).clip(0, 255).astype(np.uint8)
                s2_key = IMAGE_KEY_MAP[lr_key]
                images[s2_key] = img

        image_tensors, image_masks = preprocess_images(
            images,
            image_keys=image_keys,
            resolution=config.image_resolution,
            device=args.device,
        )

        t0 = time.perf_counter()
        with torch.no_grad():
            _, subtask_tokens, _, _token_log_probs = model.extract_prefix_latent_and_subtask(
                image_tensors,
                image_masks,
                lang_tokens,
                lang_masks,
                temperature=config.subtask_temperature,
            )
        elapsed_ms = (time.perf_counter() - t0) * 1000
        timings.append(elapsed_ms)

        # Decode subtask text
        pred_subtask = ""
        if subtask_tokens:
            try:
                pred_subtask = tokenizer.detokenize(np.array(subtask_tokens))
            except Exception:
                pred_subtask = f"<{len(subtask_tokens)} tokens>"

        # Clean prediction: take text before semicolon, strip whitespace and punctuation
        pred_clean = pred_subtask.split(";")[0].strip().lower().rstrip(".,!?")
        gt_clean = gt_subtask.strip().lower().rstrip(".,!?")

        # Match: check if GT subtask is contained in prediction (or exact match)
        exact_match = pred_clean == gt_clean
        contains_match = gt_clean in pred_clean or pred_clean in gt_clean

        results.append((idx, gt_subtask, pred_subtask, exact_match, contains_match))
        per_subtask[gt_subtask]["total"] += 1
        if exact_match:
            per_subtask[gt_subtask]["correct"] += 1
        per_subtask[gt_subtask]["preds"][pred_clean] += 1

        if (eval_i + 1) % 50 == 0 or eval_i == 0:
            avg_ms = np.mean(timings[-20:])
            correct = sum(1 for r in results if r[3])
            total = len(results)
            eta_min = avg_ms * (len(indices) - eval_i) / 1000 / 60
            logger.info(
                '  [%d/%d] acc=%.1f%% (%d/%d) | %.0fms/frame | ETA: %.1fmin | last: "%s" → "%s" %s',
                eval_i + 1,
                len(indices),
                100 * correct / total,
                correct,
                total,
                avg_ms,
                eta_min,
                gt_subtask,
                pred_clean,
                "✓" if exact_match else "✗",
            )

    # Final results
    total_exact = sum(1 for r in results if r[3])
    total_contains = sum(1 for r in results if r[4])
    total = len(results)

    logger.info("\n" + "=" * 70)
    logger.info("S2 SUBTASK PREDICTION RESULTS")
    logger.info("=" * 70)
    logger.info("Checkpoint: %s", args.checkpoint)
    logger.info("Frames evaluated: %d / %d (stride=%d)", total, n_frames, args.stride)
    logger.info("Avg inference: %.0fms/frame", np.mean(timings))
    logger.info("")
    logger.info("Overall exact accuracy:    %d / %d = %.1f%%", total_exact, total, 100 * total_exact / total)
    logger.info(
        "Overall contains accuracy: %d / %d = %.1f%%", total_contains, total, 100 * total_contains / total
    )

    logger.info("\nPer-subtask breakdown:")
    logger.info("  %-30s  %8s  %8s  %s", "Ground Truth", "Accuracy", "N", "Top Predictions")
    for gt_st in sorted(per_subtask.keys()):
        info = per_subtask[gt_st]
        acc = 100 * info["correct"] / info["total"] if info["total"] > 0 else 0
        top_preds = info["preds"].most_common(3)
        top_str = " | ".join(f'"{p}" ({c})' for p, c in top_preds)
        logger.info("  %-30s  %7.1f%%  %8d  %s", gt_st, acc, info["total"], top_str)

    # Confusion matrix
    logger.info("\nPrediction distribution (all):")
    all_preds = Counter(r[2].split(";")[0].strip().lower() for r in results)
    for pred, count in all_preds.most_common(10):
        logger.info('  "%s": %d (%.1f%%)', pred, count, 100 * count / total)


if __name__ == "__main__":
    main()

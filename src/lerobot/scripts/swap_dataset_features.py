#!/usr/bin/env python

# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Swap two features in a LeRobot dataset.

This script is useful for fixing datasets where features like camera views
were accidentally swapped during data collection.

Usage Examples:

Swap camera features (modifies in place with backup):
    python -m lerobot.scripts.swap_dataset_features \
        --repo-id thewisp/pick_place_white_pawn_jan_10 \
        --root /home/user/.cache/huggingface/lerobot/thewisp/pick_place_white_pawn_jan_10 \
        --feature1 observation.images.left_wrist \
        --feature2 observation.images.right_wrist

Swap and save to new dataset:
    python -m lerobot.scripts.swap_dataset_features \
        --repo-id lerobot/my_dataset \
        --feature1 observation.images.camera1 \
        --feature2 observation.images.camera2 \
        --output-dir /path/to/output \
        --new-repo-id lerobot/my_dataset_fixed

Swap from local dataset:
    python -m lerobot.scripts.swap_dataset_features \
        --repo-id my_dataset \
        --root /path/to/local/dataset \
        --feature1 observation.images.left_wrist \
        --feature2 observation.images.right_wrist
"""

import argparse
import logging
from pathlib import Path

from lerobot.datasets.dataset_tools import swap_features
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.utils.utils import init_logging


def main():
    parser = argparse.ArgumentParser(
        description="Swap two features in a LeRobot dataset",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--repo-id",
        type=str,
        required=True,
        help="Repository ID of the dataset to modify",
    )
    parser.add_argument(
        "--root",
        type=str,
        default=None,
        help="Root directory where the dataset is stored (for local datasets)",
    )
    parser.add_argument(
        "--feature1",
        type=str,
        required=True,
        help="Name of first feature to swap (e.g., observation.images.left_wrist)",
    )
    parser.add_argument(
        "--feature2",
        type=str,
        required=True,
        help="Name of second feature to swap (e.g., observation.images.right_wrist)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory for the modified dataset. If not specified, modifies in place with backup.",
    )
    parser.add_argument(
        "--new-repo-id",
        type=str,
        default=None,
        help="Repository ID for the output dataset. If not specified, uses original repo_id.",
    )

    args = parser.parse_args()

    init_logging()

    # Load dataset
    logging.info(f"Loading dataset: {args.repo_id}")
    dataset = LeRobotDataset(args.repo_id, root=args.root)

    logging.info(f"Dataset has {dataset.meta.total_episodes} episodes, {dataset.meta.total_frames} frames")
    logging.info(f"Features to swap: {args.feature1} <-> {args.feature2}")

    # Swap features
    output_dir = Path(args.output_dir) if args.output_dir else None

    # Use swap_features
    new_dataset = swap_features(
        dataset=dataset,
        feature1=args.feature1,
        feature2=args.feature2,
        output_dir=output_dir,
        repo_id=args.new_repo_id,
    )
    logging.info("\n" + "=" * 70)
    logging.info("✓ Feature swap completed successfully!")
    logging.info("=" * 70)
    logging.info(f"Output dataset: {new_dataset.repo_id}")
    logging.info(f"Location: {new_dataset.root}")
    logging.info(f"Episodes: {new_dataset.meta.total_episodes}")
    logging.info(f"Frames: {new_dataset.meta.total_frames}")
    logging.info("\nVerify the changes with:")
    logging.info(
        f"  lerobot-dataset-viz --repo-id {new_dataset.repo_id} --root {new_dataset.root} --mode local"
    )


if __name__ == "__main__":
    main()

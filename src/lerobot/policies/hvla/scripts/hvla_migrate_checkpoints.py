#!/usr/bin/env python
"""Migrate legacy HVLA checkpoints to standard LeRobot format.

Legacy format:
    outputs/flow_s1_hvla_v7/
        checkpoint-10000/
            model.safetensors
            norm_stats.pt
            optimizer.pt

Standard format (after migration):
    outputs/flow_s1_hvla_v7/
        checkpoints/
            checkpoint-10000/
                pretrained_model/
                    model.safetensors
                    norm_stats.pt
                    config.json
                    train_config.json
                training_state/
                    optimizer.pt
                    training_step.json
            last -> checkpoint-50000

Usage:
    python src/lerobot/policies/hvla/scripts/hvla_migrate_checkpoints.py outputs/flow_s1_hvla_v7
    python src/lerobot/policies/hvla/scripts/hvla_migrate_checkpoints.py outputs/flow_s1_hvla_v7 --dry-run
"""

import argparse
import json
import shutil
from pathlib import Path


def migrate_run(run_dir: Path, dry_run: bool = False):
    """Migrate a single HVLA training run directory."""
    # Find legacy checkpoint dirs (checkpoint-N with model.safetensors, no pretrained_model/)
    legacy_ckpts = []
    for child in sorted(run_dir.iterdir()):
        if not child.is_dir() or not child.name.startswith("checkpoint-"):
            continue
        if (child / "model.safetensors").exists() and not (child / "pretrained_model").exists():
            legacy_ckpts.append(child)

    if not legacy_ckpts:
        # Also check under checkpoints/ subdirectory
        ckpts_dir = run_dir / "checkpoints"
        if ckpts_dir.is_dir():
            for child in sorted(ckpts_dir.iterdir()):
                if not child.is_dir() or not child.name.startswith("checkpoint-"):
                    continue
                if (child / "model.safetensors").exists() and not (child / "pretrained_model").exists():
                    legacy_ckpts.append(child)

    if not legacy_ckpts:
        print(f"No legacy checkpoints found in {run_dir}")
        return

    print(f"Found {len(legacy_ckpts)} legacy checkpoint(s) in {run_dir}")

    # Determine if we need to create checkpoints/ wrapper
    needs_wrapper = not (run_dir / "checkpoints").is_dir()
    if needs_wrapper:
        ckpts_dir = run_dir / "checkpoints"
        print(f"  Will create {ckpts_dir}/")
        if not dry_run:
            ckpts_dir.mkdir(exist_ok=True)

    for ckpt in legacy_ckpts:
        print(f"\n  Migrating {ckpt.name}:")

        # Target location (may need to move into checkpoints/)
        target_dir = run_dir / "checkpoints" / ckpt.name if needs_wrapper else ckpt

        pretrained_dir = target_dir / "pretrained_model"
        training_state_dir = target_dir / "training_state"

        # Extract step from dir name
        try:
            step = int(ckpt.name.split("-")[-1])
        except ValueError:
            step = None

        if needs_wrapper and target_dir != ckpt:
            print(f"    Move {ckpt} → {target_dir}")
            if not dry_run:
                # safe-destruct: explicit migration script
                shutil.move(str(ckpt), str(target_dir))

        print(f"    Create {pretrained_dir}/")
        if not dry_run:
            pretrained_dir.mkdir(exist_ok=True)

        # Move model.safetensors and norm_stats.pt into pretrained_model/
        for fname in ["model.safetensors", "norm_stats.pt"]:
            src = target_dir / fname
            dst = pretrained_dir / fname
            if src.exists():
                print(f"    Move {fname} → pretrained_model/{fname}")
                if not dry_run:
                    # safe-destruct: explicit migration script
                    shutil.move(str(src), str(dst))

        # Create config.json
        config = {
            "type": "hvla_flow_s1",
            "action_dim": 14,
            "state_dim": 14,
            "chunk_size": 50,
            "hidden_dim": 768,
            "num_heads": 8,
            "num_encoder_layers": 4,
            "num_decoder_layers": 6,
            "s2_latent_dim": 2048,
            "num_inference_steps": 10,
            "rtc_max_delay": 6,
            "rtc_drop_prob": 0.2,
            "image_features": {
                "observation.images.front": 224,
                "observation.images.left_wrist": 224,
                "observation.images.right_wrist": 224,
                "observation.images.top": 224,
            },
            "dino_model": "dinov2_vits14",
        }
        config_path = pretrained_dir / "config.json"
        print("    Write config.json")
        if not dry_run:
            config_path.write_text(json.dumps(config, indent=2))

        # Create training_state/
        print(f"    Create {training_state_dir}/")
        if not dry_run:
            training_state_dir.mkdir(exist_ok=True)

        # Move optimizer.pt into training_state/
        opt_src = target_dir / "optimizer.pt"
        opt_dst = training_state_dir / "optimizer.pt"
        if opt_src.exists():
            print("    Move optimizer.pt → training_state/optimizer.pt")
            if not dry_run:
                # safe-destruct: explicit migration script
                shutil.move(str(opt_src), str(opt_dst))

        # Create training_step.json
        if step is not None:
            step_path = training_state_dir / "training_step.json"
            print(f"    Write training_step.json (step={step})")
            if not dry_run:
                step_path.write_text(json.dumps({"step": step}))

    # Create 'last' symlink pointing to highest checkpoint
    if not dry_run:
        ckpts_dir = run_dir / "checkpoints" if needs_wrapper else run_dir
        ckpt_dirs = sorted(
            [d for d in ckpts_dir.iterdir() if d.is_dir() and d.name.startswith("checkpoint-")],
            key=lambda d: int(d.name.split("-")[-1]) if d.name.split("-")[-1].isdigit() else 0,
        )
        if ckpt_dirs:
            last_link = ckpts_dir / "last"
            if last_link.exists() or last_link.is_symlink():
                # safe-destruct: explicit migration script: symlink update
                last_link.unlink()
            last_link.symlink_to(ckpt_dirs[-1].name)
            print(f"\n  Created symlink: last → {ckpt_dirs[-1].name}")

    print(f"\nMigration {'would be ' if dry_run else ''}complete for {run_dir}")


def main():
    parser = argparse.ArgumentParser(description="Migrate legacy HVLA checkpoints to standard format")
    parser.add_argument("run_dir", help="Path to HVLA training run directory")
    parser.add_argument(
        "--dry-run", action="store_true", help="Show what would be done without making changes"
    )
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    if not run_dir.is_dir():
        print(f"Error: {run_dir} is not a directory")
        return

    migrate_run(run_dir, dry_run=args.dry_run)


if __name__ == "__main__":
    main()

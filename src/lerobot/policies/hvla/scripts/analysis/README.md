# HVLA Analysis Scripts

Tools for debugging inference failures (gripper drops, visual mismatch, dataset corruption).

## analyze_grip_drops.py

Diagnoses why the robot drops items during inference. Compares inference observations against training data to identify visual mismatches.

### Workflow

**Step 1: Collect drop data during inference**

```bash
python -m lerobot.policies.hvla.launch \
    --s1-type flow \
    --s1-checkpoint outputs/flow_s1_hvla_v6/checkpoint-40000/model.safetensors \
    --task "assemble cylinder into ring" \
    --save-grip-drops /tmp/grip_drops_fresh
```

This saves robot state, action chunk, and camera images whenever a gripper jump (>10 degrees) is detected.

**Step 2: Analyze drops**

```bash
python src/lerobot/policies/hvla/scripts/analysis/analyze_grip_drops.py \
    --drop-dir /tmp/grip_drops_fresh \
    --checkpoint outputs/flow_s1_hvla_v6/checkpoint-40000/model.safetensors \
    --num-nearest 20 \
    --num-repeats 20 \
    --max-drops 3
```

### What it does

1. **Loads saved drop states** (robot state + images from `--save-grip-drops`)
2. **Finds nearest training frames** by Euclidean distance on robot state (from parquet, instant)
3. **Runs the model repeatedly** (default 20x) on both the drop observation and each training frame to measure gripper drop probability
4. **Camera swap ablation**: For drops with clear train/infer divergence (train=0% drops, infer>30% drops), systematically swaps each camera one-by-one between training and inference observations to identify which camera causes the drop

### Interpreting results

- If training frames at similar states have **0% drop rate** but inference has **40%+**, the visual difference is causing it
- The **camera swap ablation** shows which camera is responsible:
  - "Swap right_wrist -> train: R drop=0%" means replacing inference right_wrist with training right_wrist fixes the drop
  - This identifies the specific camera whose visual mismatch drives the failure
- If **training frames also have high drop rates** (>30%), the model learned dropping behavior from the dataset — check for corrupted/mislabeled training episodes

### Key findings (2026-03-21)

Camera swap ablation revealed the **right_wrist camera** is the primary driver of gripper drops. Further investigation found that **~32% of training episodes have swapped left/right wrist camera labels**, causing the model to learn contradictory visual-action associations.

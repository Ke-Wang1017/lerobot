# Model Tab Design

Design for the Model tab: browsing, inspecting, training, and deploying models.

---

## What We Know

### Local checkpoint structure (`outputs/` directory)

```
outputs/
  act_training_pickup_socket_feb_20/
    checkpoints/
      020000/
        pretrained_model/
          config.json              # policy architecture config
          model.safetensors        # weights (206MB for ACT, 7.3GB for pi05)
          train_config.json        # full training config
          policy_preprocessor.json
          policy_postprocessor.json
          *.safetensors            # normalizer states
        training_state/
          training_step.json       # {"step": 20000}
          optimizer_state.safetensors
          scheduler_state.json
          rng_state.safetensors
      last -> 020000               # symlink to latest
```

`train_config.json` contains: dataset repo_id, policy type + architecture params,
optimizer/scheduler config, wandb run_id, batch_size, total steps, save_freq, etc.

### HF Hub models

Downloaded to `~/.cache/huggingface/`. Policy loading via `PreTrainedPolicy.from_pretrained()`
accepts both local paths and HF repo IDs.

### Policy types

ACT, diffusion, pi0, pi0_fast, pi05, vqbet, tdmpc, sac, smolvla, groot, xvla, wall_x,
reward_classifier. Registry in `policies/factory.py`.

### Training commands

```bash
# Fresh training
lerobot-train \
    --dataset.repo_id=thewisp/pickup_socket_merged_head \
    --policy.type=act \
    --output_dir=outputs/act_training_mar_01 \
    --batch_size=4 --steps=100000

# Resume
lerobot-train \
    --config_path=outputs/.../pretrained_model/train_config.json \
    --resume=true
```

### WandB

`WandBConfig` fields: enable, project, entity, run_id, mode. The `run_id` in `train_config.json`
links back to the wandb run.

---

## Design

### Layout: Source Folders + Model List + Detail Panel

```
+------------------+------------------------------------------------+
| Source Folders    |  Model Detail                                  |
|                  |                                                 |
| [+ Add folder]   |  act_training_pickup_socket_feb_20             |
|                  |                                                 |
| ~/Documents/     |  [Overview] [Config] [Checkpoints] [Logs]      |
|   lerobot/       |                                                 |
|   outputs/       |  Policy: ACT                                   |
|   > act_train..  |  Dataset: thewisp/pickup_socket_merged_head    |
|   > pi05_trai..  |  Steps: 20000 / 100000 (20%)                  |
|     > 002500     |  Batch size: 4                                 |
|     > 005000     |  Model size: 206 MB                            |
|     > last       |  WandB: lerobot/iqm4fb0y                       |
|                  |                                                 |
| ~/.cache/        |  [Train] [Resume] [Use in Run tab]             |
|   huggingface/   |                                                 |
+------------------+------------------------------------------------+
```

### Source folders

- Default: `outputs/` (cwd), `~/.cache/huggingface/lerobot/` (HF cache)
- User can add custom folders
- Scan for `checkpoints/*/pretrained_model/config.json`
- HF cache: different scan pattern (config.json + model.safetensors pairs)
- Persist to `~/.config/lerobot/settings.json`

### Model tree

- Top level: training runs (output directories)
- Expandable: individual checkpoints (002500, 005000, last)
- Policy type badge, training status (completed / in-progress / failed)
- Running training shows spinner + live step count

### Detail panel subtabs

**Overview** — policy type, dataset, training progress, key hyperparams, model size, WandB link,
created date. For checkpoint: step number, file sizes, resume capability.

**Config** — formatted, collapsible key-value tree of `train_config.json`, grouped by section.
Read-only.

**Checkpoints** — table with step, model size, has training state, date. Actions: "Use in Run tab",
"Resume training", "Open folder".

**Metrics** — three options in order of feasibility:

1. Embed WandB iframe (simplest, requires browser login)
2. Parse wandb local logs for offline chart rendering
3. Parse stdout log lines for live training

### Actions

**"Use in Run tab"** — sets checkpoint as policy for Run tab's policy workflow.

**"Train"** — form with dataset selector, policy type dropdown, base checkpoint, output dir,
hyperparams (batch_size, steps, lr, save_freq), advanced (gradient_checkpointing, dtype, peft,
wandb). Launches `lerobot-train` as subprocess.

**"Resume"** — one-click: `--config_path=.../train_config.json --resume=true`, optional step override.

---

## Backend API

```
GET  /api/models/sources          # list source folders
POST /api/models/sources          # add source folder
DELETE /api/models/sources        # remove source folder
GET  /api/models/list             # scan sources, return model tree
GET  /api/models/{path}/config    # train_config.json contents
GET  /api/models/{path}/info      # computed metadata (sizes, step, status)
POST /api/models/train            # launch training subprocess
POST /api/models/train/stop       # stop training subprocess
GET  /api/models/train/status     # training job status
GET  /api/models/train/output     # SSE stream of training stdout
```

Training subprocess management reuses the Run tab pattern (one active subprocess, SSE output
streaming). Initially one training job at a time.

### Model scanning

1. Scan each source for `checkpoints/` subdirectories
2. Read `pretrained_model/config.json` and `train_config.json`
3. Extract: policy type, dataset, steps, wandb info
4. Check `training_state/` for resume capability
5. Cache results in memory, re-scan on request or after training completes

---

## Open Questions

- Training + Run tab: share subprocess slot or independent? (GPU vs robot-bound — could coexist)
- Multi-GPU training? (Low priority — most users single GPU)
- Cloud training launch from GUI? (Low priority)

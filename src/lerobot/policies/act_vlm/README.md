# ACT with VLM Conditioning (Dual-System S1 Policy)

A temporally hierarchical dual-system VLA inspired by [Figure's Helix](https://www.figure.ai/news/helix) and [OpenHelix](https://arxiv.org/abs/2505.03912).

- **S2 (slow brain)**: Pi0.5 at ~1Hz — extracts a scene-understanding latent from 4 cameras via SigLIP + Gemma LLM prefix encoding → mean-pooled `[2048]` vector
- **S1 (fast brain)**: ACTWithVLM at ~30-140Hz — reactive visuomotor control conditioned on cached S2 latent

## Architecture

```
S2 (~1Hz, async)                    S1 (~66Hz, sync)
┌─────────────────┐                 ┌──────────────────────────────┐
│  4 cameras      │                 │  2 cameras → ResNet18/DINOv2 │
│  → SigLIP       │                 │  state → state token         │
│  → Gemma LLM    │  s2_latent      │  s2_latent → MLP → S2 token  │
│  → mean pool    │──[2048]────────►│  VAE latent → latent token   │
│  (prefix only,  │  (cached)       │                              │
│   no AR decode) │                 │  [S2, latent, state, imgs]   │
└─────────────────┘                 │  → Transformer encoder       │
                                    │  → Transformer decoder       │
                                    │  → Action head → chunk       │
                                    └──────────────────────────────┘
```

The S2 token is injected as the **first encoder token**, before the VAE latent — the same pattern ACT already uses for its latent variable. The transformer's self-attention naturally integrates it with state and image tokens.

## Files

| File                       | Description                                                                |
| -------------------------- | -------------------------------------------------------------------------- |
| `configuration_act_vlm.py` | Config extending ACTConfig with `s2_latent_dim`, `use_dino_backbone`, etc. |
| `modeling_act_vlm.py`      | Model with S2 projector MLP + optional DINOv2-S backbone                   |
| `processor_act_vlm.py`     | Pre/post processor (same as ACT; S2 latent passes through unnormalized)    |

Related files outside this directory:

| File                              | Description                                                    |
| --------------------------------- | -------------------------------------------------------------- |
| `scripts/extract_s2_latents.py`   | Batch extraction of S2 latents from Pi0.5 server               |
| `dual_system_infer.py`            | Async S2 worker + sync S1 control loop for real-time inference |
| `src/lerobot/policies/factory.py` | Factory registration for `act_vlm` policy type                 |

Server-side (openpi_subtask repo):

| File                                                | Description                                             |
| --------------------------------------------------- | ------------------------------------------------------- |
| `openpi/models/pi05.py`                             | `extract_prefix_latent()` — prefix encoding + mean pool |
| `scripts/async_pi05/async_pi05_inference.py`        | `extract_latent()` async method                         |
| `scripts/async_pi05/async_pi05_websocket_server.py` | `"mode": "extract_latent"` endpoint                     |

## Usage

### 1. Extract S2 latents (offline)

Start the Pi0.5 WebSocket server, then:

```bash
python scripts/extract_s2_latents.py \
    --dataset-path /path/to/lerobot/dataset \
    --server-uri ws://localhost:8765 \
    --high-level-prompt "pick up the cup" \
    --output-path s2_latents.npy
```

Produces `s2_latents.npy` with shape `[N_frames, 2048]`.

### 2. Train S1

Wrap the LeRobot dataset to inject precomputed latents:

```python
from torch.utils.data import Dataset

class LeRobotDatasetWithLatents(Dataset):
    def __init__(self, lerobot_dataset, latent_path):
        self.dataset = lerobot_dataset
        self.latents = np.load(latent_path)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        item["observation.s2_latent"] = torch.from_numpy(self.latents[idx]).float()
        return item
```

Train with policy type `act_vlm`. Key config options:

```yaml
policy:
  type: act_vlm
  s2_latent_dim: 2048 # Must match Pi0.5 output
  s2_projector_hidden_dim: 1024
  use_dino_backbone: false # true for DINOv2-S (22M, ~5ms/img)
  use_vae: true
  chunk_size: 100
  dim_model: 512
```

### 3. Dual-system inference (online)

```python
from dual_system_infer import DualSystemController
import asyncio

controller = DualSystemController(
    s1_checkpoint="/path/to/act_vlm_checkpoint",
    s2_server_uri="ws://localhost:8765",
    high_level_prompt="pick up the cup",
)

asyncio.run(controller.run(robot, get_observation_fn, duration_s=60))
```

The controller manages:

- **S2 worker**: async WebSocket client querying Pi0.5 at ~1Hz
- **S1 loop**: synchronous ACTWithVLM forward pass at ~30-140Hz
- **Thread-safe cache**: S2 latent shared between threads without blocking S1
- **Graceful degradation**: if S2 disconnects, S1 continues with last cached latent (or zeros if never received)

## Config reference

| Parameter                 | Default         | Description                             |
| ------------------------- | --------------- | --------------------------------------- |
| `s2_latent_dim`           | 2048            | Dimension of S2 latent from Pi0.5       |
| `s2_projector_hidden_dim` | 1024            | Hidden dim of the 2-layer MLP projector |
| `use_dino_backbone`       | false           | Replace ResNet18 with DINOv2-S          |
| `dino_model`              | `dinov2_vits14` | DINOv2 variant (vits14 = 22M params)    |
| `dino_output_dim`         | 384             | DINOv2-S output dimension               |
| `freeze_vision_backbone`  | true            | Freeze DINOv2 weights during training   |

All other ACT config parameters (chunk_size, dim_model, n_heads, use_vae, etc.) are inherited and work as before.

## TODO

- ~~**Batch extraction**: `extract_s2_latents.py` currently sends one frame at a time (batch_size=1)~~ ✅ Done — direct PyTorch batched inference, ~41ms/frame at B=8
- ~~**Image resize to 480×640**~~ ✅ Done — GPU-side `F.interpolate` to 224×224, DINOv2 unfrozen selected as best config
- **Pre-decode video frames to JPEG**: Data loading is the bottleneck (~175ms/step vs 195ms compute). pyav random-access video decode is ~100ms/frame due to keyframe seeking. Pre-decoding to 224×224 JPEGs would drop per-frame load to ~3-5ms, cutting total step time from ~370ms to ~210ms (~1.75× speedup). Estimated disk cost: ~15GB for 186k×4 frames at 224×224 JPEG q=95.

## Design decisions

- **Token concatenation** over cross-attention or FiLM: matches ACT's existing latent injection pattern, no architectural surgery needed
- **Mean pooling** over last-N tokens: simpler, works well per OpenHelix findings; can upgrade to multi-token if needed
- **Precomputed latents**: avoids needing Pi0.5 during training; pin the checkpoint and re-extract if weights change
- **Optional DINOv2**: for when ResNet18 features aren't enough; frozen by default to keep training fast

## Commands

Run server

```
cd ~/Documents/openpi_subtask && XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run python -u scripts/async_pi05/async_pi05_websocket_server.py   --config soarm_pi05_flow_lora   --checkpoint ~/.cache/openpi/checkpoints/soarm-pi05-state-11997   --gpu-id 0 --port 8765

# new flow:

cd ~/Documents/openpi_subtask && .venv/bin/python scripts/async_pi05/pytorch_pi05_server.py \
  --checkpoint ~/.cache/lerobot/converted/soarm-pi05-state-11997-pytorch \
  --norm-stats ~/.cache/openpi/checkpoints/soarm-pi05-state-11997/assets/thewisp/cylinder_ring_assembly/norm_stats.json \
  --port 8765 \
  --device cuda:0

```

Extract latents from the trained pi05

```
cd ~/Documents/openpi_subtask && source .venv/bin/activate
# prompt should change to (.venv)

python ~/Documents/lerobot/scripts/extract_s2_latents.py \
  --checkpoint-path ~/.cache/lerobot/converted/soarm-pi05-state-11997-pytorch \
  --dataset-path thewisp/cylinder_ring_assembly \
  --high-level-prompt "assemble cylinder into ring" \
  --output-path ~/.cache/huggingface/lerobot/thewisp/cylinder_ring_assembly/s2_latents_pt_11997.npy \
  --image-keys observation.images.front,observation.images.top,observation.images.left_wrist,observation.images.right_wrist \
  --batch-size 8
```

Training

```
python scripts/train_act_vlm.py \
  --dataset-repo-id thewisp/cylinder_ring_assembly \
  --s2-latent-path ~/.cache/huggingface/lerobot/thewisp/cylinder_ring_assembly/s2_latents_pt_11997.npy \
  --output-dir outputs/act_vlm_cylinder_ring_v4 \
  --steps 100000 \
  --batch-size 16 \
  --save-freq 20000 \
  --use-dino-backbone \
  --resize-images 224x224 \
  --num-workers 8
```

Inference

```
python dual_system_infer.py \
    --s1-checkpoint outputs/act_vlm_cylinder_ring_v4/checkpoint-80000 \
    --task "assemble cylinder into ring" \
    --s2-host localhost --s2-port 8765 \
    --resize-images 224x224 \
    --temporal-ensemble-coeff 0.01
```

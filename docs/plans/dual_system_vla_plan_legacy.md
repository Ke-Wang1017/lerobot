# Plan: Dual-System VLA (Helix-Inspired) for SOARM Robot

## Context

We have Pi0.5 running at ~840ms/query (~1.2Hz) as a monolithic VLA. Profiling showed the bottleneck is prefix encoding (4 images through SigLIP + Gemma LLM = ~600ms) — irreducible without model changes. We also have ACT trained on single tasks for the same robot.

**Goal**: Build a temporally hierarchical dual-system where:

- **S2**: Pi0.5 at ~1Hz extracts a scene-understanding latent (reuse existing server)
- **S1**: DINOv2-S + modified ACT at ~30-140Hz for reactive control, conditioned on cached S2 latent

Inspired by Figure's Helix and OpenHelix's empirical findings on dual-system VLA design.

---

## Phase 1: S2 Latent Extraction

**Goal**: Expose Pi0.5's intermediate prefix encoding as a `[2048]` latent vector.

### 1A. Add `extract_prefix_latent()` to Pi0.5

**File**: `~/Documents/openpi_subtask/src/openpi/models/pi05.py`

In `sample_low_level_task()` (line 377), after the LLM forward pass:

```python
(prefix_out, _), kv_cache = self.PaliGemma.llm(...)
# prefix_out shape: [B, seq_len, 2048]
```

Add new method that runs only up to this point (skip AR decoding), then mean-pools over valid tokens → `[B, 2048]`.

### 1B. Add latent mode to inference engine

**File**: `~/Documents/openpi_subtask/scripts/async_pi05/async_pi05_inference.py`

Add `extract_latent()` method that prepares observation (same as `infer()`), calls `model.extract_prefix_latent()`, returns numpy `[2048]`.

### 1C. Add server endpoint

**File**: `~/Documents/openpi_subtask/scripts/async_pi05/async_pi05_websocket_server.py`

When request contains `"mode": "extract_latent"`, skip action generation, return:

```json
{"status": "success", "s2_latent": [2048 floats], "timing": {"prefix_ms": ...}}
```

### 1D. Batch extraction script

**New file**: `~/Documents/lerobot/scripts/extract_s2_latents.py`

- Load LeRobot dataset, iterate all frames
- Send each frame to server with `"mode": "extract_latent"`
- Save all latents as `s2_latents.npy` shape `[N_frames, 2048]`

### Verification

- Single frame → server returns 2048-dim vector, no NaN
- Extraction time ~600ms (same as prefix encoding, no AR overhead)
- Batch 100 frames, confirm consistent shapes

---

## Phase 2: ACTWithVLM Policy

**Goal**: New LeRobot policy extending ACT with S2 conditioning + optional DINOv2 backbone.

### 2A. Configuration

**New file**: `src/lerobot/policies/act_vlm/configuration_act_vlm.py`

```python
@PreTrainedConfig.register_subclass("act_vlm")
class ACTWithVLMConfig(ACTConfig):
    s2_latent_dim: int = 2048
    s2_projector_hidden_dim: int = 1024
    use_dino_backbone: bool = False
    dino_model: str = "dinov2_vits14"    # 22M params
    dino_output_dim: int = 384
    freeze_vision_backbone: bool = True
```

### 2B. Model

**New file**: `src/lerobot/policies/act_vlm/modeling_act_vlm.py`

Key changes from ACT (`src/lerobot/policies/act/modeling_act.py`):

**S2 projector** (in `__init__`):

```python
self.s2_projector = nn.Sequential(
    nn.Linear(config.s2_latent_dim, config.s2_projector_hidden_dim),
    nn.GELU(),
    nn.Linear(config.s2_projector_hidden_dim, config.dim_model),
)
```

**Injection point** — line 459 of ACT, where encoder tokens are assembled:

```python
# Original: encoder_in_tokens = [self.encoder_latent_input_proj(latent_sample)]
# New: prepend S2 token before the VAE latent token
encoder_in_tokens = [
    self.s2_projector(batch["s2_latent"]),   # S2 conditioning [B, 512]
    self.encoder_latent_input_proj(latent_sample),  # VAE latent [B, 512]
]
# ... then state token, image tokens as before
```

The S2 token enters the encoder sequence alongside state and image tokens. The transformer's self-attention naturally integrates it. This is the same pattern ACT already uses for its VAE latent.

**Optional DINOv2 backbone** (replaces ResNet18):

```python
# In __init__:
self.backbone = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14')
self.backbone.eval()
self.dino_proj = nn.Linear(384, config.dim_model)

# In forward, replace ResNet feature extraction (lines 472-484):
with torch.no_grad():
    patch_tokens = self.backbone.forward_features(img)["x_norm_patchtokens"]
    # shape: [B, 256, 384] for 224×224 input
cam_features = self.dino_proj(patch_tokens)  # [B, 256, 512]
```

### 2C. Processor

**New file**: `src/lerobot/policies/act_vlm/processor_act_vlm.py`

Same as ACT processor, plus handles `s2_latent` feature (pass-through normalization).

### 2D. Register policy

**File**: `src/lerobot/policies/factory.py`

Add `act_vlm` entries in `get_policy_class()`, `make_policy_config()`, and `make_pre_post_processors()`.

### Verification

- Instantiate policy, forward with dummy batch including `s2_latent: torch.randn(1, 2048)` → output shape `(1, chunk_size, 14)`
- Trainable params: ~15-20M (ACT transformer + projectors, DINOv2 frozen)
- Forward pass: <10ms on GPU

---

## Phase 3: Training

### 3A. Dataset wrapper

```python
class LeRobotDatasetWithLatents(Dataset):
    def __init__(self, lerobot_dataset, latent_path):
        self.dataset = lerobot_dataset
        self.latents = np.load(latent_path)  # [N_frames, 2048]
    def __getitem__(self, idx):
        item = self.dataset[idx]
        item["s2_latent"] = torch.from_numpy(self.latents[idx]).float()
        return item
```

### 3B. Training stages

| Stage                    | What trains            | Data                       | Compute            |
| ------------------------ | ---------------------- | -------------------------- | ------------------ |
| **Pre-align** (optional) | S2 projector MLP only  | latents + actions          | Minutes, 1 GPU     |
| **Main training**        | Projector + ACT + CVAE | latents + images + actions | Hours, same as ACT |

- Freeze DINOv2-S backbone
- S2 latents are precomputed (from Phase 1D), loaded from disk
- Loss: L1 on actions + KL on CVAE latent (same as ACT)
- LR: 1e-4 (higher than ACT default 1e-5 since training with frozen backbone)

### 3C. Ablation

- Train with S2 latent zeroed out → confirm S2 conditioning helps
- Compare against vanilla ACT on same data

### Verification

- Training loss decreases on 100-episode subset
- L1 loss after 10k steps ≤ vanilla ACT on same data
- Ablation shows S2 conditioning matters

---

## Phase 4: Dual-System Inference

**New file**: `~/Documents/lerobot/dual_system_infer.py`

```
Async S2 coroutine (~1Hz):
    4 cameras → Pi0.5 server (mode: extract_latent) → s2_latent [2048]
    Update shared cache (thread-safe)

Main S1 loop (~30Hz+):
    2 cameras (front + wrist) → DINOv2-S (5ms × 2) → visual tokens
    14-joint state → state token
    cached s2_latent → projector → S2 conditioning token
    ACT forward (~2ms) → action chunk
    Execute N steps open-loop, then re-query S1
```

**Timing budget**:

- DINOv2-S: 2 × 5ms = 10ms
- ACT forward: ~2ms
- Overhead: ~3ms
- **Total S1: ~15ms → ~66Hz** (well above 30Hz target)

**Graceful degradation**: If S2 disconnects, S1 continues with last cached latent.

### Verification

- S1 latency <15ms per step
- S2 updates arrive async without blocking S1
- Robot moves smoothly; S2 latent updates don't cause jerks
- Escape → soft landing

---

## File Summary

| File                                                               | Action                         | Phase |
| ------------------------------------------------------------------ | ------------------------------ | ----- |
| `openpi_subtask/src/openpi/models/pi05.py`                         | Add `extract_prefix_latent()`  | 1A    |
| `openpi_subtask/scripts/async_pi05/async_pi05_inference.py`        | Add `extract_latent()`         | 1B    |
| `openpi_subtask/scripts/async_pi05/async_pi05_websocket_server.py` | Add `"mode": "extract_latent"` | 1C    |
| `lerobot/scripts/extract_s2_latents.py`                            | **New** — batch extraction     | 1D    |
| `lerobot/src/lerobot/policies/act_vlm/__init__.py`                 | **New**                        | 2     |
| `lerobot/src/lerobot/policies/act_vlm/configuration_act_vlm.py`    | **New** — config               | 2A    |
| `lerobot/src/lerobot/policies/act_vlm/modeling_act_vlm.py`         | **New** — model                | 2B    |
| `lerobot/src/lerobot/policies/act_vlm/processor_act_vlm.py`        | **New** — processor            | 2C    |
| `lerobot/src/lerobot/policies/factory.py`                          | Register act_vlm               | 2D    |
| `lerobot/dual_system_infer.py`                                     | **New** — inference loop       | 4     |

## Dependencies

Phases 1 and 2 are independent (OpenPI vs LeRobot codebases) — can develop in parallel.
Phase 3 requires Phase 1D (latents) + Phase 2 (model).
Phase 4 requires Phase 1C (server endpoint) + Phase 3 (trained model).

## Risks & Mitigations

1. **S2 latent too lossy**: If mean-pooled `[2048]` isn't enough, take last 8 tokens → inject as 8 encoder tokens instead of 1
2. **DINOv2 token count**: 256/camera may be heavy for ACT; can use CLS token (1/camera) or spatial pooling (16/camera)
3. **Training data mismatch**: Pin Pi0.5 checkpoint used for extraction; re-extract if weights change
4. **S1 too slow**: Drop to 1 camera or DINOv2 CLS-only if >15ms

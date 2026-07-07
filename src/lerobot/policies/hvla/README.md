# Hierarchical VLA (HVLA)

A dual-system VLA for bimanual robot control. S2 (VLM) provides scene understanding, S1 (action policy) generates actions conditioned on S2's latent.

Inspired by [Helix](https://www.figure.ai/news/helix), [OpenHelix](https://arxiv.org/abs/2505.03912), and [Dual Process VLA](https://arxiv.org/abs/2410.15549).

```
S2 Process (~4-15Hz)                shared memory           S1 Process (~22-30Hz)
┌──────────────────────┐           ┌───────────┐          ┌───────────────────────────┐
│ 4 cameras → SigLIP   │           │ [2048]    │          │ 2-4 cameras → DINOv2/ResNet│
│ task text → Gemma 2B │──latent──→│ + age(s)  │←──read──│ state → state token        │
│ mean-pool → [2048]   │           └───────────┘          │ latent + age → S2 token    │
│ (VLM-only, no action │                                  │ → ACT encoder/decoder      │
│  expert)             │                                  │ → action chunk (14-DOF)    │
└──────────────────────┘                                  └───────────────────────────┘
```

## Quick Start

### 0. Convert S2 checkpoint (JAX → PyTorch, one-time)

S2 uses a [Pi0.5](https://huggingface.co/KeWangRobotics/soarm-pi05-state-11997) checkpoint converted from [OpenPI](https://github.com/Ke-Wang1017/openpi_subtask)'s JAX format to PyTorch safetensors:

```bash
# Download JAX checkpoint
huggingface-cli download KeWangRobotics/soarm-pi05-state-11997 \
    --local-dir ~/.cache/openpi/checkpoints/soarm-pi05-state-11997

# Convert (requires openpi_subtask repo: https://github.com/Ke-Wang1017/openpi_subtask)
cd ~/Documents/openpi_subtask && .venv/bin/python scripts/run_conversion.py
```

Output: `~/.cache/lerobot/converted/soarm-pi05-state-11997-pytorch/model.safetensors`

Available checkpoints (public):

- [`KeWangRobotics/soarm-pi05-state-11997`](https://huggingface.co/KeWangRobotics/soarm-pi05-state-11997) — Pi0.5 base (state-based)
- [`KeWangRobotics/soarm-pi05-fast-7998`](https://huggingface.co/KeWangRobotics/soarm-pi05-fast-7998) — Pi0.5 with FAST tokens

> **S2 is optional.** S1 can run without S2 using `--zero-s2`. Skip steps 0-1 if you don't need VLM conditioning.

### 1. Extract S2 latents (offline, one-time)

No OpenPI dependency — uses the HVLA S2 model directly:

```bash
python src/lerobot/policies/hvla/scripts/extract_s2_latents_hvla.py \
    --checkpoint ~/.cache/lerobot/converted/soarm-pi05-state-11997-pytorch/model.safetensors \
    --dataset thewisp/cylinder_ring_assembly \
    --prompt "assemble cylinder into ring" \
    --output ~/.cache/huggingface/lerobot/thewisp/cylinder_ring_assembly/s2_latents.npy \
    --image-keys observation.images.front,observation.images.top,observation.images.left_wrist,observation.images.right_wrist \
    --batch-size 8
```

Produces `s2_latents.npy` shape `[N_frames, 2048]`. ~81ms/frame on GPU.

### 2a. Train S1 — ACT (default, CVAE)

```bash
python scripts/train_act_vlm.py \
    --dataset-repo-id thewisp/cylinder_ring_assembly \
    --s2-latent-path ~/.cache/huggingface/lerobot/thewisp/cylinder_ring_assembly/s2_latents_pt_11997.npy \
    --output-dir outputs/act_vlm_hvla \
    --steps 100000 \
    --batch-size 16 \
    --save-freq 20000 \
    --use-dino-backbone \
    --resize-images 224x224 \
    --num-workers 8 \
    --max-delay 0.15
```

`--max-delay 0.15` enables delay augmentation: each training sample randomly shifts the S2 latent backward by 0-150ms (0-5 frames at 30fps), within episode boundaries. The age (seconds of staleness) is injected as a learned embedding — same scalar used at inference.

### 2b. Train S1 — Flow Matching with Training-Time RTC

```bash
# With S2 conditioning:
python -u -m lerobot.policies.hvla.s1.flow_matching.train \
    --dataset-repo-id thewisp/cylinder_ring_assembly \
    --s2-latent-path ~/.cache/huggingface/lerobot/thewisp/cylinder_ring_assembly/s2_latents_pt_11997.npy \
    --output-dir outputs/flow_s1_hvla_v6 \
    --steps 50000 \
    --batch-size 64 \
    --save-freq 10000 \
    --num-workers 16 \
    --resize-images 224x224

# Without S2 (images + state only):
python -u -m lerobot.policies.hvla.s1.flow_matching.train \
    --dataset-repo-id thewisp/cylinder_ring_assembly \
    --output-dir outputs/flow_s1_no_s2_v1 \
    --steps 50000 \
    --batch-size 64 \
    --save-freq 10000 \
    --num-workers 16 \
    --resize-images 224x224
```

Training logs are automatically saved to `{output_dir}/train.log`.

Flow matching S1 implements [Training-Time Action Conditioning for Efficient Real-Time Chunking](https://arxiv.org/abs/2512.05964) (Mees et al., 2025): simulated inference delay during training, with ground-truth action prefix inpainting. No architecture changes vs standard flow matching — just masking. At inference, actually-executed actions replace the prefix positions at each denoising step.

- 97M trainable params (DINOv2 ViT-S finetuned + 75M decoder)
- LR 2.5e-5 with cosine decay to 2.5e-6 (matching Pi0)
- bf16 autocast + TF32 matmul
- Bidirectional action decoder attention (Pi0-style, all positions attend to all others)
- 15 denoising steps (Euler integration), rtc_max_delay=6
- No S2 latent delay augmentation or age embedding (matching old ACT setup that worked)
- v6: curated dataset + bidirectional attention (v5 used causal)

### 3a. Inference — ACT (default)

```bash
python -m lerobot.policies.hvla.launch \
    --s1-checkpoint outputs/act_vlm_hvla/checkpoint-100000 \
    --s2-checkpoint ~/.cache/lerobot/converted/soarm-pi05-state-11997-pytorch/model.safetensors \
    --task "assemble cylinder into ring" \
    --resize-images 224x224 \
    --temporal-ensemble-coeff 0.07
```

### 3b. Inference — Flow Matching with RTC

```bash
python -m lerobot.policies.hvla.launch \
    --s1-type flow \
    --s1-checkpoint outputs/flow_s1_hvla_v6/checkpoint-40000/model.safetensors \
    --s2-checkpoint ~/.cache/lerobot/converted/soarm-pi05-state-11997-pytorch/model.safetensors \
    --task "assemble cylinder into ring" \
    --resize-images 224x224
```

No `--temporal-ensemble-coeff` needed — RTC provides chunk continuity natively. Key flags:

- `--denoise-steps N`: override denoising steps (default 10, min usable ~5, max quality ~15)
- `--s1-query-interval N`: re-query every N actions (default 2, ~67ms). Lower = fresher chunks, higher = more of each chunk executed
- `--no-compile-s1`: disable torch.compile if needed (on by default, ~2ms savings)
- `--save-grip-drops DIR`: save observations when gripper drops detected (debugging)
- `--record-dataset REPO_ID`: record inference episode to LeRobotDataset (e.g. `user/hvla_ep1`). Saves obs+actions every frame, commits on shutdown. Compatible with GUI, Rerun, and further training.
- `--teleop-config PATH`: path to teleop profile JSON for intervention / inverse follow. When set, the leader arm mirrors the robot during policy mode; press SPACE to toggle human control.
- `--intervention-dataset REPO_ID`: record intervention fragments to a separate dataset (e.g. `user/hvla_interventions`). Each human takeover segment becomes a separate episode.

S2 auto-discovery: the launcher first tries to attach to an existing S2 process via shared memory. If found, S1 starts instantly (no 45s cold start). If not found, spawns S2 from `--s2-checkpoint`.

### 3c. Persistent S2 (recommended for iteration)

Start S2 once in a separate terminal — it stays hot between S1 restarts:

```bash
# Terminal 1: start S2 (one-time, stays running)
python -m lerobot.policies.hvla.s2_standalone \
    --checkpoint ~/.cache/lerobot/converted/soarm-pi05-state-11997-pytorch/model.safetensors \
    --task "assemble cylinder into ring"
```

```bash
# Terminal 2: start/restart S1 freely (connects to S2 instantly)
python -m lerobot.policies.hvla.launch \
    --s1-type flow \
    --s1-checkpoint outputs/flow_s1_hvla_v6/checkpoint-40000/model.safetensors \
    --task "assemble cylinder into ring" \
    --resize-images 224x224
```

S1 auto-discovers the running S2 via well-known shared memory names. No `--s2-checkpoint` needed when S2 is already running.

## Module Structure

```
src/lerobot/policies/hvla/
├── s2/
│   ├── config.py           S2 VLM config (PaliGemma dims, LoRA settings)
│   ├── model.py            VLM-only model: extract_prefix_latent, AR subtask decode
│   ├── preprocessing.py    Camera image resize + normalize to [-1,1]
│   └── tokenizer.py        SentencePiece tokenizer for PaliGemma prompt format
├── s1/
│   ├── protocol.py        S1Policy interface (supports_rtc, needs_temporal_ensemble)
│   ├── __init__.py         Load helpers: load_act_policy / load_flow_matching_policy
│   └── flow_matching/
│       ├── config.py       FlowMatchingS1Config
│       ├── model.py        Flow matching decoder with RTC prefix conditioning
│       └── train.py        Training script with delay augmentation + RTC prefix
├── ipc.py                  SharedLatentCache: CPU shared memory between processes
├── s2_process.py           S2 process entry point (extraction loop)
├── s1_process.py           S1 process entry point (robot control loop)
├── launch.py               Spawns S2 process + runs S1
└── README.md               This file
```

## Design Decisions

### S2 is VLM-only

S2 uses PaliGemma (SigLIP + Gemma 2B) for scene understanding. The Pi0.5 action expert is NOT loaded — S1 generates all actions. This saves ~40% GPU memory and decouples the action policy from the VLM. S2 loads from any Pi0.5 safetensors checkpoint — action expert keys are filtered out at load time.

### Latent interface: 2048-dim, single token

S2 → S1 communication is a single [2048] float32 vector (PaliGemma's hidden dim), mean-pooled from the VLM's prefix features. S1 projects it down via MLP: 2048→1024→512.

| System           | Latent dim     | Projection               |
| ---------------- | -------------- | ------------------------ |
| OpenHelix        | 512            | Linear from 4096         |
| Dual Process VLA | 4096           | MLP                      |
| RoboDual         | 256 + 8 tokens | Perceiver                |
| **Ours**         | **2048**       | MLP: 2048→1024→512 in S1 |

### Action decoder: bidirectional attention (v6)

Pi0's action expert uses bidirectional self-attention so all chunk positions "negotiate" and commit to one mode. Our v5 (causal) showed within-chunk gripper oscillation. v6 switches to bidirectional (`tgt_mask=None`), matching Pi0. Requires retraining — not a drop-in change.

### S1 training: pre-extract S2 latents

S2 latents are pre-extracted offline rather than computed during S1 training:

- S2 inference (50ms/frame) on the same GPU as S1 training causes contention
- DataLoader workers can't efficiently share a GPU model
- Extraction is one-time: ~2.5h for 186k frames, produces ~1.5GB .npy file
- If S2 is retrained, latents must be re-extracted before S1 retraining

### Delay augmentation: tried and dropped (v5+)

v1-v4 shifted the S2 latent backward by k frames (k ~ Uniform(0, 15) at 30fps = 0-500ms) to simulate S2 staleness, with an age embedding MLP so S1 knows how stale the latent is. In practice, this caused S1 to learn to ignore the latent entirely. v5+ trains with aligned latents (k=0, no age embedding), matching the original ACT setup that worked. The age embedding MLP still exists in the model but is disabled (`use_s2_age_embedding=False`).

### Shared memory IPC

`SharedLatentCache` uses `torch.Tensor.share_memory_()` backed by `/dev/shm`. Each process manages its own CUDA context — no GPU contention from IPC. The [2048] float32 latent (8KB) transfers in <0.1ms.

## Performance

### S1 Flow Matching (v6, 10 denoise steps, compiled)

| Component               | Latency | Notes                                      |
| ----------------------- | ------- | ------------------------------------------ |
| S2 latent extraction    | ~81ms   | VLM-only, no action expert                 |
| S1 obs prep (4 cameras) | ~6ms    | cv2.resize on CPU, then GPU transfer       |
| S1 inference (10 steps) | 22-42ms | bf16 + KV cache + batched DINOv2 + compile |
| IPC round-trip          | <0.1ms  | CPU shared memory                          |
| Robot send_action       | <1ms    | Feetech serial, non-blocking               |

Typical S1 loop interval: ~34ms (~29Hz). Inference spikes to ~59ms under S2 GPU contention.

### Optimization stack (cumulative)

| Optimization                   | Inference time   | Savings |
| ------------------------------ | ---------------- | ------- |
| Baseline (fp32, 15 steps)      | ~95ms            | —       |
| + TF32 matmul                  | ~54ms            | 41ms    |
| + bf16 autocast                | ~25ms            | 29ms    |
| + Cross-attention KV cache     | ~14ms (isolated) | 11ms    |
| + Batched DINOv2 (4 cameras)   | marginal         | ~1ms    |
| + torch.compile (denoise_step) | ~2ms saved       | ~2ms    |
| Real-world (10 steps, w/ S2)   | 22-42ms          | —       |

## TODO

### High priority

- [x] **[CRITICAL] RLT state normalization mismatch** — fixed in 480efb41c via `FlowMatchingS1Policy.prepare_batch_for_encode_observations`. Parity test `test_policy_prepare_batch_normalizes_state` locks in the invariant. Old actor/critic checkpoint renamed to `latest.buggy_pre_fixes_20260423/` for safekeeping; fresh Phase-2 training required.
- [x] **[RLT] Image preprocessing parity** — fixed in 78da63e8d. `obs_to_s1_batch` now uses `TF.resize(bilinear, antialias=True)` matching `FlowMatchingDataset`. Policy-owned preprocessing refactor still pending (ACT-VLM has a different training pipeline — xfail-tagged).
- [x] **[RLT] Actor forward dtype** — fixed in ccbdf7dee. Actor call wrapped in `torch.autocast(enabled=False)` with explicit fp32 cast. Parity test `test_actor_receives_fp32_inputs_at_inference` verifies.
- [x] **[RLT] Duplicate `encode_observations`** — eliminated in b1f8f2e57 via `context` kwarg on `predict_action_chunk` + `sample_actions`. Saves ~11 ms per inference.
- [x] **[RLT] Guardrail tests** — added in d9fcd9f1d (`tests/hvla/test_rlt_parity.py`). Covers state, image, actor-dtype; xfail marker for ACT-VLM until policy-owned preprocessing refactor.
- [x] **[RLT] Per-inference-stage timing** — added in 25627fb90. Dump now carries `obs_to_infer_ms`, `enc_obs_ms`, `rl_tok_ms`, `s1_denoise_ms`, `rlt_actor_ms`, `rlt_post_ms`, `total_delay_ms` via CUDA events (no hot-path sync).
- [x] **[RLT] Offline Phase-1 encoder probe + retrain** — `src/lerobot/policies/hvla/scripts/rlt_token_probe.py` (a7bbb33ea) + `src/lerobot/policies/hvla/scripts/rlt_arch_experiment.sh` daemon (3552ceb31). Curve: 2L d=768 = 36.4% → 4L d=768 = 29.8% → **4L d=2048 = 24.9%**. Each halving of the capacity squeeze bought ~5 pp. Both d=768 and d=2048 curves flatten in the last 2k steps → hitting information-theoretic floor with frozen S1. Reaching the paper's implied < 10% would require joint S1 fine-tune during Phase 1 (Alg. 1 L3) — deferred until Phase-2 evidence says we need it.
- [x] **[HIGH][RLT-metrics] `episode_lengths_s` (and `episode_timestamps`) missing from snapshot** — `record_episode()` was appending durations to the in-memory list, but the snapshot dict at the bottom of `RLTMetrics.snapshot()` never copied them into `series`, so consumers reading `metrics.json` saw the keys absent (and analysis scripts using `dict.get(key, [])` got an empty list silently). Fixed by adding both keys to the series output. GUI dashboard, throughput-derived charts, and resume-restore can now read durations.
- [x] **[HIGH][RLT-metrics] Misleading window naming in summary fields** — went with option (b): emit both rolling-20 and all-time versions side-by-side. Existing fields (`success_rate` / `autonomous_rate` / `intervention_rate`) keep their rolling-20 semantics for GUI back-compat; new `*_alltime` companions added so post-hoc analysis and any "how's this run going overall" caller can pick the window deliberately. GUI dashboard separately may want to render both — left to a follow-up.
- [x] **[RLT] Measured on-robot inference delay with the 4L d=2048 encoder.** Analyzed `outputs/rlt_online_v2_widened/chunk_compare.jsonl` (3997 RLT-active records over 70 eps). Per-stage timings (mean / median / p95 / max in ms): `rl_tok_ms` 3.7 / 2.8 / 11.3 / 35.3 (the widened encoder), `enc_obs_ms` 12.2 / 10.0 / 24.4 / 80.9, `s1_denoise_ms` 39.8 / 38.2 / 58.7 / 149.8 (largest contributor), `rlt_actor_ms` 0.6 / 0.3 / 1.3 / 52.3, `obs_to_infer_ms` 36.9 / 32.9 / 59.9 / 513.5 (mostly inter-frame wait, not compute). End-to-end `total_delay_ms` 95.9 / 91.3 / 131.5 / 569.5. RTC budget at 30Hz with `rtc_max_delay=6` is ~200ms; only 1.4% of inferences exceed it (vs 33.8% above 100ms). The widened encoder cost is negligible — d=2048 ships fine on real hardware.
- [x] **[RLT] Actor slice `[D:D+C]` vs `[0:C]`** — implemented. `_rlt_inference_step` now takes `expected_d` (sourced from RTC's existing computation in `_loop`); the actor reads/writes `[D:D+C]` instead of `[0:C]`. Frames `[0:D]` are the RTC prefix that won't actually be executed by the robot, so refining them was wasted ~20% of the actor's output. Edge cases: `D=0` first-chunk-of-episode preserves prior behavior; warmup is identity at the new slice; `D+C ≤ chunk_size` asserted just before the slice. No replay buffer migration needed (entries keep `[C, A]` shape; old `[0:C]` slices roll out of the 200k ring naturally).
- [ ] **[RLT] Phase 2 retrain on robot** — start fresh with `rlt_output_dir=outputs/rlt_online_v2_widened` and `rl_token_checkpoint=outputs/rlt_token_v4_4layer_d2048/checkpoint-10000`. Launch via GUI Run tab → RLT Online RL → "+ New training...". Baseline first with S1-only (no RLT) for autonomous success rate reference. Watch dashboard: Q-value spread should widen, actor delta should grow after warmup, autonomous rate should exceed S1 baseline after ~100 eps. Old buggy checkpoint preserved at `outputs/rlt_online/latest.buggy_pre_fixes_20260423/` — delete when confident.
- [x] **[HIGH][RLT] Transactional replay buffer for IGNORE invariant.** Implemented as `TransactionalReplayBuffer` in `rlt/replay_buffer.py`. Wraps the existing `ReplayBuffer` (now the committed buffer) with database-style transaction semantics: `add()` writes are pending until `commit()` (move them into committed) or `discard()` (drop them). `sample()` reads only from committed. Used at episode granularity — each episode is one transaction; success/abort/fail call `commit()`, IGNORE calls `discard()`. Locked in as a structural invariant by `TestTransactionalReplayBuffer::test_sample_never_returns_staged_transitions` and `test_commit_then_add_then_discard_is_clean`. Wired into `s1_process.py`: IGNORE branch calls `discard()` instead of the old `truncate(ep_start_size)`; non-IGNORE branch calls `commit()` after metrics save and before `lifecycle.end_episode()`. `_rlt_flush_intervention_terminal` assert updated to inspect pending writes via `peek_last_pending()`. 12 new tests; 464 total pass. Cost as predicted: episode N's transitions become sampleable in episode N+1 — fine, the BC anchor + slow target update absorb the ~250 grad-step lag.
- [ ] **Policy-owned image preprocessing** (follow-up to the image-resize fix) — move preprocessing into each S1 policy type so `obs_to_s1_batch` delegates. ACT-VLM needs `F.interpolate(bilinear)` no-antialias; flow_matching needs `TF.resize(bilinear, antialias)`. Removes the "silently diverges with new policy type" bug class permanently.
- [x] **[RLT] Atomic checkpoint saves** — every `.pt` save in `s1_process.py` (actor / critic / critic_target / training_state) now goes through `_atomic_torch_save` (writes to `*.tmp` then `os.replace` to the final path). Same pattern applied to `ReplayBuffer.save()`. Either the new file is fully present or the previous one is untouched — never half-written. Closes the torn-checkpoint window where a crash mid-save could leave actor and critic from different gradient steps. (Each file is independently atomic; if you want dir-level atomicity across all files of a checkpoint, that would require a separate `latest.new/` + dir-rename — overkill given files are independent.)
- [x] **[HVLA] stderr missing from subprocess run log** — installed `sys.excepthook` and `threading.excepthook` in `run_s1` right after the `FileHandler` is wired up. Uncaught exceptions in main thread or any background thread now flow through `logger.error(..., exc_info=...)` so their tracebacks land in the persistent `outputs/hvla_runs/run_*.log` alongside everything else. KeyboardInterrupt and SystemExit are passed through to default handling.
- [ ] **S1 inference thread latency** — obs→infer gap is 8-51ms (typical ~14ms, spikes to 51ms) due to `threading.Event.wait()` OS scheduler jitter. Options: (1) busy-wait with CPU yield, (2) double-buffered obs with atomic swap, (3) redesign so inference thread owns obs capture.
- [ ] **S1 GPU priority over S2** — S1 inference spikes from 22ms to 59ms under S2 contention. Options: (1) CUDA MPS for proper time-slicing, (2) separate GPUs, (3) Triton kernel optimization.
- [ ] S2 LoRA finetuning — S2 cannot produce subtask transitions (stuck on "pick up the cylinder"). Plan: LoRA rank 32 on both Gemma 2B (q/k/v/o_proj, 18 layers) + SigLIP (QKV+out, 27 layers) ≈ 24M trainable params. Two losses: subtask cross-entropy (weight=10) + FAST token cross-entropy (weight=1). Needs per-frame `high_level_task` + `low_level_subtask` annotations.
- [ ] S2 latent normalization — z-score normalize S2 latents before feeding to S1 (44× cosine gap improvement validated). Requires S1 retraining.
- [ ] Generic SharedMemoryStore to replace per-type IPC classes
- [ ] Pre-decode and cache training images to avoid repeated video decode (main training bottleneck)

### Medium priority

- [ ] **Image resize: switch to BICUBIC to match DINOv2 original pretraining** (currently BILINEAR + antialias everywhere in our pipeline). DINOv2 was pretrained on LVD-142M with PIL-style bicubic, so feeding it bilinear-resized images leaves ~1% feature quality on the table. Low priority: requires full S1 retrain (backbone has fine-tuned to bilinear). The inference-vs-training consistency bug is unrelated and must be fixed regardless (see CRITICAL entries above).
- [ ] **Training DataLoader optimization** — GPU drops to 10% for ~2s every ~7s (video decode stall). Candidates: `persistent_workers=True` + `prefetch_factor=3` in DataLoader, pre-decode images to memmap. Needs benchmarking to measure actual improvement.
- [ ] `torch.compile` for S1 training (mode=default, skip DINOv2 — needs investigation)
- [ ] Separate backbone LR (lower for DINOv2 pretrained weights, matching ACT's approach)
- [ ] wandb integration for training monitoring
- [ ] Inference-time RTC guidance (LeRobot's `RTCProcessor`) on top of training-time RTC for extra smoothness
- [ ] Corrupt JPEG data — "premature end of data segment" from OpenCV cameras (not seen recently, may be resolved)

### Done

- [x] Intervention / inverse follow (`--teleop-config`, `--intervention-dataset`)
- [x] Inference episode recording to LeRobotDataset (`--record-dataset`)
- [x] Robot-agnostic control loop (joint names derived from robot, no hardcoded SO107)
- [x] Extracted helpers: `load_robot()`, `load_s1_policy()`, `create_recording_dataset()`
- [x] Bidirectional action attention (Pi0-style, `tgt_mask=None`) — v6
- [x] Cross-attention KV cache (pre-compute K,V from static context)
- [x] Batched DINOv2 (4 cameras in one forward pass)
- [x] bf16 autocast + TF32 matmul
- [x] torch.compile denoise_step
- [x] Image resize on CPU (cv2.resize, 0.7ms/4 images)
- [x] Persistent S2 process (s2_standalone)
- [x] Observation processor steps (DepthEdgeOverlayProcessorStep)
- [x] Grip drop diagnostics with inference-time obs saving
- [x] Soft landing with hold-position during torque ramp-down

### Future

- [ ] Co-training S1 + S2 (currently sequential: extract latents → train S1)
- [ ] Adaptive action horizon (short chunks for precision, long for transit)
- [ ] Switch S1 to using Pi0-style action expert architecture (proven at scale, matches Ψ₀)
- [ ] ONNX/TensorRT export for S1 (potential 2× inference speedup)

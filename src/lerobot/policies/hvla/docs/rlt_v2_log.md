# RLT v2 Running Log

## 2026-04-03: Initial implementation

### Branch setup

- Created `feature/rlt-v2` from `0592ce70` (pre-RLT, has intervention support)
- Copied rlt/ core modules from v1 with fixes:
  - Replay buffer: added threading.Lock on all public methods
  - Token reconstruction loss: changed from MSE (mean) to L2 sum (paper Eq. 2)
  - Actor BC penalty: L2 sum (paper Eq. 5) — carried from v1 fix
  - Config: sigma=0.01 (joint angles need lower noise than delta EE)

### RL Token v2 Training (Phase 1)

- S1 checkpoint: `outputs/flow_s1_no_s2_v1/checkpoints/last/pretrained_model`
- Dataset: `thewisp/cylinder_ring_assembly` (199K frames)
- Steps: 10,000, batch=64, lr=1e-4
- Loss: 451 → 114 (sum loss, not comparable to v1's 0.207 MSE loss)
- Per-element relative error: 60.4% (v1 was 66.7% on same test data)
- Compression ratio: 1025 tokens × 768 dim → 1 × 768 = 1025× compression
- Paper uses same ratio (~1025 × 2048 → 1 × 2048) and trains 2000-10000 steps
- Conclusion: bottleneck is inherently lossy at this ratio. Proceed to RL and see if it works.
- Checkpoint: `outputs/rlt_token_v2/checkpoint-10000`

### Integration (v2 architecture)

- Episode = RL boundary. Operator prepares scene during reset.
- InferenceThread: S1 encoder → z_rl → actor → refined chunk → replay buffer → inline gradient updates
- Actor NOT called during intervention. Human chunks accumulated by main thread.
- R key = success (+1). Collection paused during intervention/reset.
- Q values logged directly (mean/min/max per batch)
- GUI dashboard: success rate, Q values, actor delta, critic loss

### Commits on feature/rlt-v2:

1. `97e49406` — Core modules with infra fixes
2. `dd3e3a35` — Updated design doc
3. `a897f9cb` — Integration (s1_inference, s1_process, launch, GUI)

## 2026-04-03: Baseline measurement

### Setup

- S1 checkpoint: `flow_s1_no_s2_v1`
- Task: cylinder-ring assembly, critical phase only (operator teleops to pre-critical)
- Episode timeout: 60s
- No RLT, no intervention
- 20 episodes

### Results

| Metric                  | Value                                       |
| ----------------------- | ------------------------------------------- |
| Autonomous success rate | **65%** (13/20)                             |
| Throughput              | **10.4 successes / 10 min**                 |
| Median success time     | ~24s                                        |
| Failed episodes         | all timed out at 60s                        |
| Log file                | `outputs/hvla_runs/run_20260403_155719.log` |

## 2026-04-03: RLT v2 Training Session

### Training progression

- Ep 1-10: Warmup (S1 passthrough), 80% success
- Ep 11-90 (beta=1.0, sigma=0.01→0.02): Actor delta flat at 0.02, no visible learning. BC penalty dominated Q gradient.
- Ep 91+ (beta=0.1): Immediate behavior change — faster successes, more exploration, some wrong directions.
- Ep 179: 70% autonomous, mean 15.2s (vs baseline 24s). First clear RL improvement.
- Ep 256: 70% autonomous, mean 17.4s. Stable.
- Ep ~280: Q value range temporarily collapsed (buffer 85% full, old successes overwritten). Recovered. Expanded buffer 50K→200K.
- Ep 289: 60% success rate (dip from cold cases + low beta)

### Key findings

- beta=0.1 unlocked learning but destabilized cold (unseen) states — actor follows random Q noise without BC anchor
- sigma=0.02 adds visible jitter but unclear if it helps vs just adding noise on joint angles
- Speed improvement confirmed: 6.5-17s vs 24s baseline on successful episodes
- Buffer capacity matters — 50K too small, old trajectories overwritten causing Q collapse
- Consistent reset positions (paper's "small randomization") would help critic coverage

### TODO

1. **Adaptive beta**: High beta for cold/unseen states, low beta for well-covered states. At minimum expose beta as a GUI slider for live tuning during training.
2. **Deployment mode**: Run trained RL actor alongside base VLA without training loop. Inference-only mode that loads actor checkpoint and applies to S1 chunks.
3. ~~**Sigma research**~~ — answered on the v2_widened run: setting `exploration_sigma=0` on robot caused a cluster of catastrophic episodes (7 ignores in 5 attempts) — the actor commits 100% to its deterministic mean and any systematic actor drift goes unmasked. Keep `exploration_sigma≈0.02` (≈0.5° per joint). Decoupled `target_sigma=0.1` for TD3 target smoothing landed in 884f01d2b so this knob is independent.
4. **More episodes**: Paper runs 400-1000. v2_widened currently at 70+ eps with rolling-20 success climbing 50% → 75%. Keep running.
5. ~~**Tests**~~ — 90 unit tests now in `tests/hvla/test_rlt.py` + `test_rlt_metrics.py` + `test_intervention.py`. Coverage:
   - BC penalty, reconstruction loss, critic invariants (min-Q, gamma^C, target frozen, grad-clip)
   - Q-explosion defenses (LayerNorm presence/absence, Q-target clipping, decoupled sigmas — `TestQExplosionDefenses`)
   - Replay buffer (detach, ring wrap, save/load, **truncate** for the DOWN-arrow ignore key, thread safety)
   - Three-group metrics with within-group invariants + atomic round-trip
   - InterventionRecorder full lifecycle including `flush_terminal` for ASSISTED-success terminal +1
   - Atomic checkpoint save (`_atomic_torch_save` — `TestAtomicTorchSave`)
   - Token checkpoint manifest, warmup boundary, parity guards.

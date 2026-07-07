# RLT: RL Token for Online Fine-Tuning

Online RL fine-tuning of a frozen [HVLA S1](../README.md) policy using a
lightweight actor-critic. Based on
[Xu et al., "RL Token: Bootstrapping Online RL with Vision-Language-Action Models"](https://pi.website/research/rlt).
See [RLT design doc](../docs/rlt_design.md) for architecture decisions and
[experiment log](../docs/rlt_v2_log.md) for training results.

![RLT Dashboard](dashboard.png)

> **Model support**: RLT currently only supports [HVLA S1](../README.md) (flow
> matching). The [RL token encoder](token.py) is trained on S1's internal
> representations (1025x768 context tokens). The core RL components
> ([TD3 actor-critic](actor_critic.py), [replay buffer](replay_buffer.py),
> [metrics](metrics.py)) are model-agnostic, but the integration layer
> ([s1_inference.py](../s1_inference.py), [s1_process.py](../s1_process.py))
> is S1-specific.

## How It Works

1. **[RL Token Encoder](token.py)** compresses S1's observation context (1025
   tokens) into a single 768-dim bottleneck vector `z_rl`
2. **[Actor MLP](actor_critic.py)** takes `z_rl` + proprioceptive state + S1's
   reference chunk and outputs a refined action chunk
3. **[TD3 Critic](actor_critic.py)** evaluates action quality; actor follows
   critic's Q gradients
4. **BC regularizer** `beta * ||a - a_ref||^2` keeps the actor near S1's output

S1 is frozen throughout. Only the actor and critic train (~2M params total).
All [hyperparameters](config.py) are in `RLTConfig`.

## Quick Start

### Phase 1: Train RL Token Encoder (offline)

Compresses S1's internal representations into a bottleneck. Run once per S1
checkpoint. See [`train_token.py`](train_token.py) for full options.

```bash
python -m lerobot.policies.hvla.rlt.train_token \
    --s1-checkpoint outputs/flow_s1_no_s2_v1/checkpoints/last/pretrained_model \
    --dataset-repo-id thewisp/cylinder_ring_assembly \
    --output-dir outputs/rlt_token \
    --steps 10000 \
    --batch-size 64
```

Output: `outputs/rlt_token/checkpoint-10000/encoder.pt`

### Phase 2: Online RL Training (on-robot)

#### Option A: GUI (recommended)

1. Open the [LeRobot GUI](../../../gui/README.md), go to the **Run** tab
2. Select your S1 checkpoint and task
3. In the **RLT** dropdown, select **New** or an existing checkpoint
4. Set the **RL Token** checkpoint path (from Phase 1)
5. Click **Start**
6. The **RL Training** tab shows live metrics

#### Option B: Command line

Uses [`launch.py`](../launch.py) with RLT flags:

```bash
python -m lerobot.policies.hvla.launch \
    --s1-checkpoint outputs/flow_s1_no_s2_v1/checkpoints/last/pretrained_model \
    --s1-type flow \
    --task "assemble cylinder into ring" \
    --robot-config robot.json \
    --zero-s2 \
    --rlt-mode \
    --rl-token-checkpoint outputs/rlt_token/checkpoint-10000 \
    --rlt-output-dir outputs/rlt_online \
    --num-episodes 99 \
    --episode-time-s 30 \
    --reset-time-s 999 \
    --teleop-config teleop.json
```

### Resuming Training

Pass the run directory as `--rlt-checkpoint`:

```bash
    --rlt-checkpoint outputs/rlt_online \
    --rlt-output-dir outputs/rlt_online
```

Or select the existing checkpoint from the GUI dropdown. The actor, critic,
[replay buffer](replay_buffer.py), and episode counter are all restored.

### Deploy Mode (inference only)

Run a trained actor without training — no critic, no replay buffer, no gradient
updates, no exploration noise.

**GUI**: Select checkpoint, set mode to **Deploy**.

**CLI**: Add `--rlt-deploy`:

```bash
    --rlt-checkpoint outputs/rlt_online \
    --rlt-deploy
```

## Controls During Training

Intervention requires a [teleop config](../launch.py) (`--teleop-config`).

| Key             | Action                                            |
| --------------- | ------------------------------------------------- |
| **R**           | Mark success (+1 reward, ends episode)            |
| **SPACE**       | Toggle intervention (human takes over / releases) |
| **Right arrow** | End episode (failure), advance to next            |
| **Left arrow**  | End episode, re-record                            |
| **ESC**         | Stop                                              |

## Live Tuning

The **RL Training** panel in the [GUI](../../../gui/README.md) has two sliders:

- **beta (BC weight)**: How strongly the actor is pulled toward S1's reference.
  High beta (0.5-1.0) = stay close to S1. Low beta (0.05-0.1) = let the actor
  deviate based on Q gradients. Start high, lower as the critic matures.
- **sigma (exploration)**: Gaussian noise added to actor output during training.
  Set to 0 if jitter is a problem (human intervention provides exploration).

Changes take effect within ~200ms (next gradient step). Overrides are stored in
`rlt_overrides.json` in the output directory and picked up by the
[inference thread](../s1_inference.py).

## Training Tips

- **Start with the critical phase**: prepare the scene to just before the hard
  part. The actor learns faster on a focused subtask.
- **Intervene when it fails badly**: intervention data teaches the critic what
  good states look like. Don't intervene on minor imperfections.
- **Press R on success**: sparse reward is the only training signal. Every R
  press creates a positive reward transition.
- **Watch the Q values**: if Q mean is near zero and not rising, the critic
  hasn't learned yet. Keep collecting episodes.
- **Watch actor delta**: `delta=0.0` means the actor isn't deviating from S1.
  `delta>0.1` means it's found its own behavior. High delta + low success =
  actor is drifting (raise beta).

## Output Structure

```
outputs/rlt_online/
    train.log            # Training log
    metrics.json         # Dashboard metrics (read by GUI)
    rlt_overrides.json   # Live slider values (beta, sigma)
    latest/              # Latest checkpoint
        actor.pt
        critic.pt
        critic_target.pt
        training_state.pt
        replay_buffer.pt   # (~1.5 GB for 200K capacity)
```

## Architecture

```
Frozen S1 encoder ──→ context [1025, 768]
                          │
                    RL Token Encoder (2-layer transformer)
                          │
                      z_rl [768]
                          │
                 ┌────────┴────────┐
                 │                 │
           Actor MLP          TD3 Critic
         (z_rl + state        (z_rl + state
          + ref_chunk)         + action_chunk)
                 │                 │
          refined chunk      Q value [scalar]
           [C, 14]
```

## Key Hyperparameters

Defined in [`RLTConfig`](config.py). Tunable live via GUI sliders where noted.

| Parameter           | Default | Description                                                      |
| ------------------- | ------- | ---------------------------------------------------------------- |
| `beta`              | 0.1     | BC regularizer weight (live tunable)                             |
| `exploration_sigma` | 0.02    | Noise std added to executed action — joint jitter (live tunable) |
| `target_sigma`      | 0.02    | Noise std for TD3 target policy smoothing (live tunable)         |
| `target_noise_clip` | 0.5     | Symmetric clip on target smoothing noise (TD3)                   |
| `rl_chunk_length`   | 10      | Action chunk length C                                            |
| `utd_ratio`         | 5       | Gradient updates per transition                                  |
| `discount`          | 0.99    | Gamma for TD target                                              |
| `tau`               | 0.005   | Soft target update rate                                          |
| `critic_grad_clip`  | 10.0    | Global-norm clip on critic gradient                              |
| `replay_capacity`   | 200,000 | Max replay buffer transitions                                    |
| `warmup_episodes`   | 10      | Episodes before actor is used                                    |

## Code Map

| File                                       | Purpose                                                             |
| ------------------------------------------ | ------------------------------------------------------------------- |
| [`config.py`](config.py)                   | All hyperparameters (`RLTConfig` dataclass)                         |
| [`token.py`](token.py)                     | RL token encoder/decoder (Phase 1)                                  |
| [`train_token.py`](train_token.py)         | Phase 1 training script                                             |
| [`actor_critic.py`](actor_critic.py)       | Actor, Critic, TD3Agent (Phase 2)                                   |
| [`replay_buffer.py`](replay_buffer.py)     | Thread-safe ring buffer                                             |
| [`metrics.py`](metrics.py)                 | Training metrics + GUI dashboard data                               |
| [`../s1_inference.py`](../s1_inference.py) | Inference thread: actor forward, gradient updates, config overrides |
| [`../s1_process.py`](../s1_process.py)     | Main loop: episode management, intervention, checkpointing          |
| [`../launch.py`](../launch.py)             | CLI entry point with RLT flags                                      |

## Tests

```bash
python -m pytest tests/hvla/test_rlt.py -v
```

19 tests covering: BC penalty formula, reconstruction loss, critic invariants,
replay buffer correctness, metrics computation.

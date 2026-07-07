# Plan: RLT v2 — Correct Architecture Based on Paper

## Paper Understanding (verified against Algorithm 1 + Section VI)

### How RLT actually works during training:

1. **Human operator prepares the scene** to the critical phase start (via teleop or any other means — this is outside the RL episode)
2. **Episode starts** → RL actor runs, transitions stored, gradient updates inline
3. **Human can intervene** during episode — overwrites actor output
4. **Operator signals success/failure** — episode ends
5. Reset, repeat

Preparation (step 1) is NOT part of the episode. No transitions stored. The episode boundary = the RL boundary.

### Reward signal:

- Sparse binary: +1 at episode end (success) or 0 (timeout/failure)
- _Paper Section III_: "r_T = 1 for success"
- Applied to the last transition of the episode

### Replay buffer contents:

- Warmup: VLA reference actions (executed instead of actor during early episodes)
- RL actor actions (after warmup)
- Human intervention actions during episodes
- NOT human preparation actions (those happen outside the episode)

### Human intervention role:

- **Demonstrations**: shows the critic what good actions look like from difficult states
- **Corrections**: prevents damage when RL fails, completes the task
- Stored as action=ref=a_human (BC penalty `β||a_human - a_human||² = 0`)
- **How the critic learns from intervention data**: intervention episodes that end with R (+1 reward) teach the critic that the states the human passed through have high Q value. Failed episodes (timeout, reward=0) without intervention teach the critic those states have low Q. The critic learns which states are promising vs hopeless — regardless of who was controlling. The actor then follows the critic's Q gradients to reach high-Q states.

### Training curriculum:

Paper Section VI: "For full-task training, we start by having RL focus on the critical phase with small randomization, and then move on to the full-task setting."

- Stage 1: Episodes start right before the hard part (critical phase only)
- Stage 2: Full episodes (base VLA → RL switch)

## What was wrong with our v1 implementation:

1. **Training curriculum**: episodes included easy phases (pick up, transport) diluting the critical-phase signal. Fix: operator prepares scene to critical phase, entire episode is RL.
2. **Between-episode training** → batched updates (450 in 3s burst) caused actor drift. Fix: inline updates during query interval (paper's async approach).
3. **sigma=0.1 for joint angles** — paper uses delta EE, different scale. Fix: sigma=0.01 works but may be too low for exploration. Need to tune based on metrics.
4. **No Q value logging** — can't tell if critic is learning. Fix: log Q values directly.
5. **Replay buffer pollution** — transitions stored during servo sync, reset, and intervention with wrong data. Fix: collection gating + correct intervention chunk storage.
6. **RL token quality** — 46% reconstruction error, encoder undertrained with wrong loss function.
7. **Integration architecture** — actor in main thread, stale/duplicate transitions, collecting flag races. Fix: clean rewrite in inference thread.

## Plan: Reset and rebuild correctly

### Step 0: Infra and bug fixes (commit first, before any RLT)

- Servo sync non-fatal: timeout logs warning, continues into intervention mode (human takes over). Does not crash or exit.
- ReplayBuffer threading lock (add, sample, save, load)
- Any other unrelated GUI or intervention fixes from current branch

### Step 1: Branch management

- Create new branch `feature/rlt-v2` from `0592ce70` (pre-RLT, has intervention + GUI fixes)
- Apply Step 0 infra fixes first
- Copy rlt/ core modules from current branch: token.py, config.py, actor_critic.py, replay_buffer.py, metrics.py, train_token.py, **init**.py
- Copy design doc, tests
- Save current s1_inference.py and s1_process.py RLT snippets in `docs/rlt_v1_backup/` for reference
- Rewrite s1_inference.py and s1_process.py integration from scratch
- Keep `feature/leader-inverse-follow` intact as backup

### Cleanup (after v2 is verified working on robot):

- Reset `feature/leader-inverse-follow` to `0592ce70` (remove v1 RLT commits)
- Merge `feature/rlt-v2` into `feature/leader-inverse-follow`
- Delete `feature/rlt-v2` branch

### Step 2: Correct integration architecture

**Paper references for each decision:**

**Simplified design: critical-phase-only episodes (Paper Stage 1)**

- _Paper Section VI_: "episodes begin after being reset to a partially completed task state right before the critical phase"
- No base VLA phase. No base→RL switch needed.
- Operator teleops robot to pre-critical position during reset (using existing leader arm)
- Presses start → RL actor runs entire episode (just the critical phase)
- S1 still runs every inference to produce z_rl + ref_chunk for the actor

Two modes during an episode:

- **RL mode** (default, entire episode): Actor refines S1 chunks. Transitions stored. Gradient updates run inline during query interval.
  - S1 runs every inference → produces z_rl (via RL token encoder) AND ref_chunk (via flow matching decoder). Actor input is `(z_rl, state, ref_chunk[:C])`. Actor DEPENDS on S1 — it refines S1's output, does not run independently (though ref_chunk is zeroed 50% of training batches via dropout).
  - _Paper Algorithm 1 line 9_: `a ~ π_θ(· | x, ã)` when not warmup/intervention
  - _Paper Algorithm 1 lines 13-18_: gradient updates run every step (async, using replay buffer)
  - _Paper Appendix B_: UTD=5, two critic updates per actor update

- **INTERVENTION mode** (SPACE during episode): Human controls via leader arm. C-frame chunks accumulated and stored.
  - _Paper Algorithm 1 line 9_: `a ← a^human` if intervention — actor is NOT called
  - _Paper Algorithm 1 line 7_: S1 still runs (produces z_rl for state + ref_chunk). But ref_chunk is discarded — line 11 overwrites reference with a_human before storage.
  - _Paper Algorithm 1 line 11_: `ã ← a^human` — reference also set to human action
  - _Paper Section V_: "the intervention replaces the VLA reference in the replay buffer"
  - Stored as action=ref=human_chunk → BC penalty `β||a_human - a_human||² = 0`
  - Why BC=0 for human data: critic still trains on these transitions (learns Q values for good states/actions), but actor loss becomes purely `-Q(s,a)` without BC anchor. The actor learns what's good from the critic, not by copying the human directly.
  - _Paper Algorithm 1 lines 13-18_: gradient updates CONTINUE during intervention (off-policy, uses replay buffer data). Both critic and actor train.
  - **Actor does NOT execute during intervention** — only S1 runs (for z_rl). Human's action is the output. Main thread accumulates human actions into C-frame chunks and writes to replay buffer directly.

- **R key**: Success (+1 reward), ends episode.
  - _Paper Section III_: "r_T = 1 for success... sparse binary reward"
  - _Paper Section VI_: "receiving a terminal signal from the human operator"

- **Episode timeout** (e.g. 60s): Failure (reward=0), episode ends.

Key binding: configurable in code (user has game controller remapping).

**Episode flow:**

1. Reset phase: torque off, operator teleops to pre-critical position via leader arm
2. Press start (right arrow) → torque on, RL runs
3. Actor refines S1 chunks, transitions stored, gradient updates inline
4. Human can intervene (SPACE) → human actions stored as chunks
5. R = success (+1) or timeout = failure (0) → episode ends
6. Between-episode: save checkpoint, log metrics → back to step 1

### Step 3: Replay buffer with lock

- Add threading.Lock to ReplayBuffer (add, sample, save, load)
  - _Rationale_: inference thread writes during RL, main thread writes during intervention — concurrent access
- Main thread writes during intervention, inference thread writes during RL

### Step 4: Metrics

- **Log Q values directly** — critic loss near 0 doesn't tell us if Q values are meaningful
- **Throughput**: successes per 10 minutes
  - _Paper Section VI_: "we also report throughput, the number of successful task completions per 10 minute interval"
- **Episode length** (frames in episode only) — should decrease if RL is learning speed
  - _Paper Fig 9_: "RLT significantly improves the speed... median 86 steps vs teleop 146"
- GUI dashboard: lower chart height, add throughput to status bar, add Q value chart (min/max/mean per batch), log all to metrics.json for cross-process visibility

### Step 5: Human intervention chunks

During INTERVENTION within episode:

- Main thread accumulates human [14] actions every frame
- Every C frames: snapshot z_rl + state from frame 0, store full [C,14] chunk
  - z_rl from inference thread (still running during intervention)
  - State from current obs at frame 0 of C-step window
- action = ref = human_chunk (BC penalty = 0 for human data)
  - _Paper Algorithm 1 line 11_: `ã ← a^human`
  - _Paper Section V_: "each transition stored in B includes the executed action and the corresponding reference"
- Partial chunks at end of intervention discarded (replay buffer is unordered, no continuity needed)
- Normalization: same as S1 actions: `(action - action_mean) / action_std`

### Step 6: Warmup handling

- _Paper Section V, "Warmup"_: "we pre-fill the replay buffer B by rolling out the VLA reference policy for N_warm environment steps"
- During warmup, RL mode executes VLA reference instead of actor
  - _Algorithm 1 line 9_: `if t < N_warm → ã` (VLA reference)
- Both actor and critic train from step 0 (including during warmup)
  - _Algorithm 1 lines 13-18_: gradient update loop is inside the main for loop, no warmup gate
  - _Paper_: "begin learning shortly after the warmup phase" — this refers to when the actor starts EXECUTING (taking over from VLA reference), not when training starts. Training starts immediately.

### Step 7: Training curriculum

- _Paper Section VI_: "For full-task training, we start by having RL focus on the critical phase with small randomization, and then move on to the full-task setting"
- Start with critical-phase-only episodes: operator teleops to pre-critical position during reset
- Later, operator can stop preparing the scene → episode includes the easy phase. **No code change needed** — same code, different operator behavior. The RL actor won't malfunction on easy-phase states because BC regularizer keeps output ≈ S1 reference for any states not in the replay buffer (Q gradient ≈ 0 for unseen states, BC dominates).
- The paper's "two-stage" curriculum is an optimization for training efficiency, not a technical requirement. Stage 2 makes the actor robust to realistic approach variations, not prevent malfunction.
- _Paper_: "We report the policy performance after gathering about 5 hours of data" → expect ~600 episodes at 30Hz/30s before seeing clear improvement

## Audit: rlt/ core modules vs paper

| Component                | Match? | Issue                                                                                                                                                                               |
| ------------------------ | ------ | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| RL Token Encoder (Eq. 1) | ✓      | Matches: readout embed, encoder transformer, take last position                                                                                                                     |
| RL Token Decoder (Eq. 2) | ⚠️     | Reconstruction loss uses `F.mse_loss` (mean). Paper Eq. 2 sums over token positions. Same MSE-vs-sum class as the BC penalty bug. May want to fix for v2 token retraining.          |
| Actor (Eq. 4)            | ✓      | Direct output, Gaussian with fixed σ, ref as input. Zero-init not in paper but compatible.                                                                                          |
| Ref action dropout       | ✓      | 50% per-sample mask on actor input, BC compares to unmasked ref. Matches paper.                                                                                                     |
| Critic (Eq. 3)           | ⚠️     | C-step return simplified: `reward + γ^C * Q(s',a')`. Paper sums discounted rewards over C steps. With sparse reward this is nearly equivalent (off by γ^{C-1} ≈ 0.914 at terminal). |
| BC penalty (Eq. 5)       | ✓      | L2 sum over dims, mean over batch. Fixed.                                                                                                                                           |
| TD3 target smoothing     | ⚠️     | Uses actor's sigma for target noise. Standard TD3 uses separate clipped noise. Minor.                                                                                               |
| Replay buffer            | ✓      | Correct fields. Needs lock (planned).                                                                                                                                               |

**Critical finding: RL token quality is poor.**
Context tokens have std≈1.0. Reconstruction RMSE=0.455 → 46% relative error.
The encoder is losing ~half the information. Likely causes:

- Only trained 3000 steps (loss still decreasing)
- Reconstruction loss uses MSE (mean) not sum — weaker gradient signal
- Need to retrain with fixed loss and more steps

**Action items for v2:**

- **MUST** fix token reconstruction loss to sum and retrain encoder (longer)
- Add TD3 target policy smoothing with clipped noise (optional improvement)
- The C-step return simplification is acceptable for sparse reward
- Verify reconstruction relative error < 10% before proceeding to Phase 2

## Files to create/modify (on new branch):

1. `src/lerobot/policies/hvla/rlt/replay_buffer.py` — add lock
2. `src/lerobot/policies/hvla/rlt/actor_critic.py` — copy with BC L2 sum fix, zero-init
3. `src/lerobot/policies/hvla/rlt/config.py` — copy with sigma=0.01, warmup=10
4. `src/lerobot/policies/hvla/rlt/metrics.py` — copy + add Q values, throughput
5. `src/lerobot/policies/hvla/s1_inference.py` — NEW integration: actor + transitions only in RL mode
6. `src/lerobot/policies/hvla/s1_process.py` — NEW integration: BASE/RL/INTERVENTION mode, intervention chunks, key bindings
7. `src/lerobot/policies/hvla/launch.py` — RLT flags
8. `src/lerobot/gui/api/run.py` — RLT API + throughput endpoint
9. `src/lerobot/gui/static/run.js` — dashboard + throughput + Q values
10. `src/lerobot/gui/static/style.css` — chart height
11. `src/lerobot/gui/static/index.html` — RL tab
12. `tests/hvla/test_rlt.py` — updated tests

## Success Criteria Per Stage

### Phase 1: RL Token Training

**Goal:** Compress S1 context tokens into z_rl with minimal information loss.

- **Metric:** Reconstruction relative error = RMSE / context_std
- **Target:** < 10% relative error (currently 46% — unacceptable)
- **How to verify:** After training, run encoder on held-out data, compute RMSE / std
- **Actions if target not met:** Train longer, increase encoder capacity, check loss function

### Phase 2: Warmup (first N episodes on robot)

**Goal:** Fill replay buffer with VLA data, actor learns to copy VLA via BC.

- **Metric:** Actor delta (mean |actor_output - ref|) in normalized space
- **Target:** Delta < 0.02 (≈ 0.5° per joint). Currently gets to ~0.09.
- **How to verify:** Log delta every 100 steps. Should converge during warmup.
- **Secondary:** Critic loss should be low (accurately predicting Q≈0 for zero-reward data)
- **Actions if target not met:** Check BC penalty scale, train longer, verify token quality

### Phase 3: RL Training (after warmup, on robot)

**Goal:** Actor learns refinements that improve task success/speed.

**Step 0: Establish baseline (BEFORE any RL)**
Run 20 episodes with base S1 only (no RLT), same task setup (critical phase only).
Record: autonomous success rate, throughput (successes per 10 min active time).
This is the ground truth. Q values and actor delta have no baseline (they don't exist without RL).

**Metrics to track (with units):**

- **Throughput:** Autonomous successes per 10 minutes of active robot time (excluding reset phases). Unit: count/10min. Compare to baseline.
- **Autonomous success rate:** Rolling 20 episodes, counting ONLY episodes where RL completed without intervention.
- **Q values:** Range [0, 1]. Q=1 means "success on next C-step." Q=γ^100≈0.37 means "100 steps away." Q≈0 means "no path to success."
  - Log: mean, min, max Q across each training batch.
  - **What to look for:** spread (max - min) growing over time = critic learning to distinguish states.
- **Actor delta:** Mean |actor - ref| in normalized space (×24° for degrees).
  - Increasing delta + increasing throughput = useful refinements ✓
  - Increasing delta + decreasing throughput = diverging ✗
  - Flat delta near 0 = not exploring

**How to verify learning is happening:**

1. Q value spread growing over episodes (critic differentiating states)
2. Throughput exceeding baseline
3. Autonomous success rate exceeding baseline

**Actions if metrics don't improve after 100+ episodes:**

- Q values flat near 0: not enough reward signal → need more episodes or denser reward
- Delta flat near 0: exploration too low → increase sigma or decrease beta
- Delta grows but throughput drops: actor diverging → increase beta, check critic

**Pre-check before Phase 3 (can verify offline):**

- Token quality: reconstruction relative error < 10% (currently 46%, must retrain)

### Infrastructure Verification

- All existing tests pass
- New tests: lock safety, intervention chunks, mode switching assertions
- Manual: run on robot with configurable key workflow
- Manual: verify transitions only stored during episode (check train.log)
- Manual: verify Q values are logged and visible in dashboard
- Manual: verify throughput metric
- Log + assert at every mode transition:
  - Episode start: replay buffer collection ON, actor enabled
  - Intervention ON: collection OFF (inference thread), actor NOT running, main thread starts chunk accumulation
  - Intervention OFF: collection ON (inference thread), actor resumes, main thread stops accumulation
  - Episode end: collection OFF, no transitions during reset
  - Reset → next episode start: collection ON
  - Assert: no replay buffer writes outside episodes
  - Assert: no replay buffer writes by inference thread during intervention (only main thread writes human chunks)
  - Assert: actor does not execute during intervention

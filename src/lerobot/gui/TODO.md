# GUI TODO

## Data Tab

- [High] **Warning/error panel**: dataset verification errors and warnings are currently buried in server log text. Add a visible warning panel (banner or sidebar) that surfaces verification results when a dataset is opened — errors as red, warnings as yellow. Users must not miss data integrity issues.
- [High] **Open local dataset by path**: opening a copied/renamed local dataset fails because `LeRobotDataset.__init__` tries to reach HuggingFace Hub when the folder name doesn't match a cached `owner/name` repo_id. Spaces in folder names also rejected. Need to bypass Hub entirely for local-only datasets.
- [ ] Parquet data display (action/state charts in timeline) — superseded by Feature Editing (see below); action/state co-display alongside cameras tracked as a follow-up there
- [ ] Monitor local dataset changes — auto-refresh UI when new episodes recorded while GUI is open
- [ ] Duplicate episode
- [ ] Copy/move episodes between datasets
- [ ] Reorder episodes
- [ ] Create new dataset from UI
- [ ] Drag-drop dataset opening
- [ ] Undo/redo — explicitly punted from Feature Editing V1 (per-chip removal in edits-bar covers most "oops" cases). Real undo across Saves needs pre-edit value capture or a Git-like history.

### Feature Editing (per-frame view + edit)

See [docs/feature_editing.md](docs/feature_editing.md) for the full design.

V1: schema-driven, drag-to-select-range + Inspector typed editing. Lays the foundation for RECAP-style labeling (`reward`, `success`, `subtask`).

V1 phases (A1–B6) and the schema-add layer (see [docs/add_feature.md](docs/add_feature.md)) are done as of 2026-05-06. Phase C1 (resize handles) and follow-ups remain.

- [x] Phase A1: schema in `DatasetInfo` (extend the existing dataset-open response with full features dict — dtype, shape, names)
- [x] Phase A2: per-frame feature values endpoint + Inspector dataset-summary empty state (schema-driven renderer registry)
- [x] Phase A3: episode feature-series endpoint + timeline rows (line / band / stripe per dtype)
- [x] Phase B1: vertical-slice selection + click-to-seek-and-select (click inside trim → playhead + selection `[N, N+1)`; drag inside trim → seek + range; playhead-thumb drag scrubs without re-selecting; click outside trim → no-op; Esc clears)
- [x] Phase B2: Inspector edit widgets (checkbox / slider+number / dropdown for `subtask_index` / text / row-of-N inputs / "Edit as JSON…"); auto-staging on change with `● pending` indicator
- [x] Phase B3: stage `feature_set` edits in `PendingEdit` pipeline; group-by-`(feature, episode)` rendering with expand/collapse
- [x] Phase B4: "Show pending edits" toggle (current vs post-Save data overlay on timeline)
- [x] Phase B5: validation + safety rails (block edits on `DEFAULT_FEATURES` / `action` / `observation.*` / image / video; >10k frames confirmation)
- [x] Phase B6: **new `set_feature_values()` API** in `dataset_tools.py` (peer to `modify_features`) — in-place parquet rewrite of value cells, stats recomputation, `subtasks.parquet` updates, `finalize()`; GUI's `_apply_feature_set_edits()` translates staged edits into one call
- [x] **Schema-add layer (2026-05-06)**: `dataset_tools.add_features_inplace()` + GUI banner offering to add MUST-have `reward`/`success` defaults + generic "+ Add feature" dialog for custom columns + `success` tri-state widget + declared-`per_episode`-wins-over-inference + orphan-`.tmp` sweep on dataset open. Single-tab UX only — multi-tab cross-update is a follow-up.
- [ ] Phase C1: two mouse-draggable resize handles (vertical Inspector ↔ main, horizontal cameras ↔ timeline)

Follow-ups (post-V1, listed in design doc):

- [ ] **WS broadcast for `dataset.schema_changed`** — multi-tab cross-update for the schema-add path. Today's frontend updates `window.datasets[id]` from the POST response only (single-tab); other tabs/windows looking at the same dataset don't auto-refresh. Needs a connection registry in `AppState` and a per-dataset `broadcast_to_dataset(id, msg)` helper. Skipped during the schema-add build because no broadcast infra existed.
- [ ] **`add_features_inplace` UX polish** — confirmation modal for >100k frames (per spec); pre-canned "Common features" preset list in the Add Feature dialog (success/reward/subtask/quality_score) so common cases are one click; tri-state colored band rendering on the timeline for pinned per-episode int8 features (today: line plot fallback).
- [P1/P2] **Stats viewer** — surface `meta/episodes/*.parquet` `stats/*` columns: per-feature min/max/mean/std/quantiles, per-episode and per-dataset, read-only
- [ ] In-place segment editing: boundary drag, double-click rename, click-empty-and-type, split / merge
- [ ] Row context menus (Rename, Delete, Split here, Merge with next)
- [ ] Multi-range selection
- [ ] Loop-on-selection during playback
- [ ] Curve editor for continuous features (reward, value, advantage)
- [ ] **Multi-dim numeric vector visualization** for `action` / `observation.*` — current state: up to 32 components rendered as overlaid lines with a 16-color palette (post-PR3), but for a 14-DOF action the lines cluster into a dense band and the components aren't individually readable. Decided direction: **expandable per-dim rows** — chevron toggle on the row label; collapsed (default) shows the overlaid view, expanded shows a stack of mini-rows (~16px each), one per component, with `ft.names[i]` labels (or `[i]` fallback). State persisted per-dataset-session via `featureRowState.expanded` (already declared, not yet wired).
- [ ] Schema mutations: add / remove / rename features via `modify_features`
- [ ] URDF / 3D trajectory views
  - [x] ~~**EE-trajectory line as an always-live alternative.**~~ Done — landed as a 5 mm cyan tube along the FK'd EE positions of multi-frame sources. Surfaces immediately in dataset/replay mode (the next `horizon` rows of `action` are always available); also surfaces in any live mode that publishes chunks.
  - [ ] **Chunk-source playback in the URDF viewer.** Now that the trajectory tube ships the path-in-space dimension, the remaining work is the time dimension: step through the captured chunk's frames at the producer's `fps`. Needs (a) producer-side pause/snapshot so a captured chunk doesn't fight live motion, (b) scrubber UI in the URDF tile.
  - [ ] **Chunk-publishing for live policy modes.** Today an ACT/diffusion/HVLA policy writes one action per step through `robot.send_action`; the trajectory tube only lights up when the source returns N > 1 frames. To light it up live, the policy needs to publish its chunk on a small shared-memory channel; the renderer reads it unchanged.
  - [ ] **Multi-source overlay UI.** Today the URDF tile has a single "show action" toggle. When meta.sources lists a second overlay candidate (e.g. "prediction" from a debug model), the toggle needs to become a multi-pick (or dropdown). The backend API (`?source=` parameterised, meta lists names) is already shaped for this.
  - [ ] **Layered correctness viz: solid = real, ghost = desired-now, tube = desired-future.** Today the URDF viz can't distinguish IK accuracy from control accuracy. "The arm is off by 2 cm" gets blamed on IK by default when it's usually the motor PID's steady-state lag under gravity load (e.g. shoulder_lift sag on SO-107). Three layers, each a different error source:
    1. **Solid body** — FK of `Present_Position`. The actual physical state.
    2. **Translucent ghost arm** — FK of `Goal_Position`. Where the controller wants the arm to be RIGHT NOW. The solid-vs-ghost gap is per-joint tracking error (sag, friction, load); when tracking is clean the ghost collapses onto the solid and disappears.
    3. **Cyan EE-trajectory tube** (already shipped) — FK of the next N frames of `action`. Where the controller wants the arm to be SOON.

    The chain reads as "this used to be true → should be true now → should be true soon," and the tube starts at the ghost EE so the geometry is visually continuous. Couple with a per-arm strip in the latency panel that decomposes the error sources numerically: `IK: 0.2 mm / 0.1°  ·  tracking: 12 mm / 3.5° (sag: -8 mm Z, mostly shoulder_lift)` so the user knows which layer owns the deviation.

    Decisions before implementation: (a) show ghost when the clutch is **disengaged** (helpful — "we'll resume tracking from here") or hide it (avoids confusion since Goal_Position is held)? (b) ghost color — desaturated arm color vs single neutral shade — must not compete with the cyan tube; (c) the "sag" callout — useful when the gap exceeds N° but easy to over-clutter, probably gated behind a toggle; (d) live (informs ongoing teleop, doubles per-frame URDF work + needs a `Present_Position` subscription) vs dataset-playback-only (cheap, no hot-loop cost) vs both. Recommended first pass: replay-only ghost arm, no callout, single neutral color — minimum surface to validate the mental model before paying the live-rendering cost.

- [Mid] **Compact per-episode-feature storage** (tech debt): per-episode features are currently marked via `info.json` but stored by replicating the same value across every frame in `data/*.parquet` — wasteful for what is logically one value per episode. Find a more compact representation (e.g. dedicated columns in `meta/episodes/*.parquet`, or sparse encoding in `data/*.parquet`) while staying backward-compatible with readers that expect the per-frame layout.
- [Mid] **Remove implicit "episode-wide if uniform" detection** (tech debt, depends on the compact storage above): today a feature is recognized as episode-wide if all its frame values within an episode are identical. That heuristic is fragile (one stray edit makes it look per-frame again) and only exists because we lacked an explicit format. Once compact storage is in place, drop the heuristic in favor of the explicit `episode_features` declaration.
- [ ] Episode-list shortcuts: tri-state `success` checkbox column with bulk-set; inline-editable `task` per episode
- [ ] First-class `subtask_index` writer at recording time (`add_frame(subtask=...)` analog to `add_frame(task=...)`); today only the read hook + offline annotation Space exist

### Per-Episode Quality Badges (Latency)

`lerobot_record` now writes `meta/episodes_health.jsonl` (one line per
episode) and `meta/recording_health.json` (session summary) into the
dataset's meta dir. Each line carries `healthy`, `issues`, `overrun_ratio`,
`loop_dt_ms` percentiles, and per-camera staleness. The data panel
doesn't yet read this — wire it up:

- [ ] **Render per-episode quality badge** in the episode list (yellow/red dot next to bad episodes; tooltip shows the issue messages from `episodes_health.jsonl`).
- [ ] **Dataset-level health banner** when `recording_health.json` reports `healthy: false` — surfaces "this run had X% overrun, Y issues" on dataset open, so users can decide to keep / discard / fix before training.
- [ ] **Filter / bulk-select bad episodes** for deletion via the existing data-panel deletion flow.
- [ ] **Backend endpoint** `GET /api/datasets/{id}/health` returning the parsed contents of both files (or empty when the files aren't present — older recordings).
- [ ] Schema docs in [latency_monitoring.md](../docs/latency_monitoring.md) — already covers the file format under "Persistent per-episode quality metadata".

Verdict thresholds live in `src/lerobot/utils/latency/recording_health.py::DEFAULT_VERDICT_THRESHOLDS` — surface them somewhere the user can tweak per-dataset if needed (probably not in V1).

### Control-Lag Analyzer

Cross-correlation of `action[:, j]` vs `observation.state[:, j]` per joint gives the effective control lag for free, from any existing dataset — no extra recording, no extra hardware. Prototype lives at `/tmp/analyze_control_lag.py`; ran clean on three SO-107 datasets (199K / 179K / 137K frames each) and converged on a uniform 3–4 frame baseline (~100–133 ms at 30 Hz). The fact that the lag is the same across all 14 joints — not a gradient from heavy shoulders to light gripper — points at the sampling rate + default PID gains as the systemic floor rather than per-joint mechanics.

Worth landing as a proper tool:

- [ ] **Promote the analyzer** out of `/tmp` to `src/lerobot/utils/analyzers/control_lag.py` with a `lerobot-analyze-control-lag <dataset_dir>` CLI entry (or as a function callable from the data panel backend).
- [ ] **Integrate into `meta/recording_health.json`**: append a `control_lag` block with `{joint_name: {lag_frames, lag_ms, corr, n_moving}}` so the verdict has a measured number rather than just "did the loop hit its budget." Computed once at end of recording from the freshly-written parquet, no live overhead.
- [ ] **Data panel visualisation**: a small per-joint bar chart on the dataset summary, color-coded against expected baseline (~4 frames at 30 Hz for hobby servos). Outliers (e.g. one joint at 7+ frames) flag specific hardware issues — backlash, retries, loose mount.
- [ ] **Gripper handling**: gripper position has nonlinear contact dynamics so its lag correlation is ~0.94 vs ~0.99 for other joints. Either filter to no-contact segments before correlating, or flag the gripper number as low-confidence in the UI.
- [ ] **Cross-dataset comparison view**: same set of joints across N datasets to spot drift (servo wear, firmware change, mount loosening). Lives naturally next to the per-episode-quality badges work above — same data-panel concept.

Caveat: this measures _tracking lag_ (action-to-state convergence during continuous motion), which is a strict superset of the pure command-to-actuation dead time discussed in the latency design doc. The rig-based C2A calibration (IMU + MCU master clock) is still the only way to isolate the few-ms servo dead time itself — but for "is the control loop tracking?" / "did this hardware regress?", cross-correlation is enough and free.

### Dataset Merge

See [docs/data_tab.md](docs/data_tab.md) for full design.

- [ ] Merge dialog: select target (new dataset), select source datasets from opened datasets
- [ ] Episode selection: checkbox per episode in each source (start with whole-dataset, add per-episode later)
- [ ] Pre-merge validation panel (FPS, robot_type, features — green/red checks)
- [ ] Post-merge integrity verification (video timestamps, parquet row counts, contiguous indices)
- [ ] Backend: `/api/edits/merge` endpoint wrapping `merge_datasets()` from `dataset_tools.py`
- [ ] Test chained merges (A+B->C, C+D->E) — existing regression test covers meta file mapping, but need end-to-end chain test

### HuggingFace Hub Sync

Background Transfers tray + subprocess-worker pipeline landed in PR #15
(merged 2026-05-26). Design: [docs/hub_transfers.md](docs/hub_transfers.md).

- [x] `GET /api/hub/auth-status` — check login state
- [x] `POST /api/datasets/{id}/hub/download` — pull from Hub (with completeness check + confirm_force override)
- [x] `POST /api/datasets/{id}/hub/upload` — push to Hub via PR (atomic merge to main)
- [x] Frontend: auth indicator in header, Hub Upload / Hub Download per-dataset entry points
- [x] **Background Transfers tray** — pill in top-right, popover with one card per job, real progress, Cancel / Retry / Discard / Hide actions. Multi-tab consistent (server holds state), workers survive server restart, multi-dataset parallel.
- [ ] **Stale-PR sweep** ([open question in design doc](docs/hub_transfers.md#open-questions)) — failed uploads leave draft PRs; surface them on Upload-modal-open so the user sees stale attempts.
- [ ] **Re-enable `super_squash_history`** ([open question](docs/hub_transfers.md#open-questions)) — currently disabled; main accumulates N commits per upload (cosmetic only — atomicity and throughput unaffected). Need correct HF API usage or post-merge squash-on-main.
- [ ] **Retry budget UX** — third-strike retries should surface differently ("Failed 3× — check connection") rather than repeating the latest error.
- [ ] **`POST /api/hub/login`** — currently delegated to `huggingface-cli login` (out-of-band terminal flow). Add an in-GUI login form only if the standalone GUI runtime (no terminal) is supported.

## Model Tab

See [docs/model_tab.md](docs/model_tab.md) for the (older) browse/inspect design and [training/DESIGN.md](training/DESIGN.md) for the cloud-GPU training pipeline + deployment design (the active workstream as of 2026-06).

- [ ] Phase 1: Browse & Inspect — source scanning, list/info/config endpoints, source tree, detail panel
- [ ] Phase 2: Training — subprocess launch/stop/status, training form, terminal output, resume
- [ ] Phase 3: Run Tab Integration — "Use in Run tab" passes checkpoint path to policy workflow
- [ ] Phase 4: Metrics & Polish — WandB embed, training curves, model comparison, HF Hub model sync

### Cloud-GPU training pipeline

Active workstream tracked in [`training/DESIGN.md`](training/DESIGN.md). Phased status (Phase 0 + 1 done; Phases 2–6 pending — spawn step, API endpoints, host-profile UI, Model-tab frontend, auto-push-to-Hub, recipes). The bash CLI surface (`scripts/training/{install_prereqs,setup_host,run_training}.sh`) is the working prototype today; Python worker scaffold at `src/lerobot/gui/training_{jobs,worker}.py` is staged for the GUI integration.

## Model Debugger (Single-Frame Inference)

- [High] **Single-frame model debugger**: Run any model (S1, S2, Pi0, ACT, etc.) on a single dataset frame and inspect outputs. Goal: understand what a model "sees" at a specific moment, compare outputs across frames to diagnose bugs (gripper drops, wrong subtask predictions, action divergence).

  **Core features:**
  - Select a frame from any opened dataset (episode + frame index, or scrub to it in playback)
  - Select a model checkpoint (from Models tab)
  - Run inference -> show outputs:
    - **VLM (S2)**: decoded subtask text, latent vector norm/stats, token probabilities
    - **Action policy (S1, ACT, Pi0)**: action chunk as trajectory plot (per-joint over horizon), key stats (max delta, gripper values)
    - **Any model**: raw tensor shapes, norms, timing
  - Show input images (all cameras) alongside outputs

  **Compare mode:**
  - Pin frame A's output, select frame B -> side-by-side diff
  - Highlight which outputs changed and by how much
  - Use case: "frame 100 gives smooth actions, frame 105 gives a gripper drop — what changed?"

  **Episode sweep:**
  - Run model on every N-th frame of an episode, plot outputs over time
  - Visualize subtask transitions, latent norm trajectory, action variance
  - Overlay ground-truth subtask labels from dataset

  **Integration points:**
  - Data tab: right-click frame -> "Debug with model"
  - Models tab: "Test on frame" button on checkpoint detail
  - Saved dumps (`/tmp/hvla_drops/`): load images + state from dump directory

  **Live teleop probe mode:**
  - Teleop the robot freely while model(s) predict in real-time — predictions are displayed but NOT executed
  - Use cases:
    - **Subtask discovery**: teleop through a task, watch S2 subtask label change live
    - **Action preview**: freeze at a pose, see the predicted action chunk trajectory
    - **Confidence mapping**: teleop slowly through workspace, display model uncertainty
    - **Multi-model comparison**: run two checkpoints side-by-side on same live observation
    - **Data collection guidance**: while teleoping, see what the current model predicts
  - Implementation: teleop process writes obs to shared memory, separate model process reads and predicts, GUI displays predictions

  **Backend:**
  - `/api/debug/run-frame` endpoint: accepts model path + dataset + frame index (or image paths)
  - `/api/debug/live-probe` endpoint: start/stop live prediction alongside teleop
  - Lazy model loading with caching (don't reload for consecutive frames)
  - Returns structured JSON (predictions, stats, timing)

## Run Tab

- [Low] Text output freezes after a while — teleoperate uses ANSI cursor-up in piped stdout
- [Low] Rerun web viewer has ~200ms visual lag (Rerun 0.26 limitation)
- [Low] Replay FPS setting doesn't seem to affect playback speed — remove if not useful
- [Low] **SIGKILL during HVLA shutdown leaks semaphores**. When stopping S1 inference (or any HVLA run) via the GUI's stop button, the 5s SIGTERM→SIGKILL grace window in `gui/api/run.py` sometimes isn't enough — visible as `resource_tracker: There appear to be N leaked semaphore objects` and `Process exited with code -9`. Cosmetic only (kernel reclaims on process death) but noisy. **Don't extend the timeout blindly** — 15s feels like an eternity from the GUI when the user wants to relaunch. First instrument the shutdown: log per-phase elapsed times for soft-land, teleop disconnect, robot disconnect (+ per-camera if accessible), shm cleanup. Get measured numbers from a few runs, then decide whether the fix is "kill the slow phase", "parallelize cameras", "release shm first", or just "the timeout is fine, the warning is harmless".
- [High] **Foolproof shutdown-cleanup registry.** Today the server's shutdown hook is a hand-maintained list of cleanup calls (`_stop_debug_process`, `_terminate_active_process`, `cleanup_stale_streams`, `cleanup_in_process_resources`, `shutdown_prefetch_executor`). Adding a new module-level resource — preview cameras, recording robots, prefetch executor, anything yet to come — means remembering to wire it into `server.py::shutdown_event`. Easy to forget, and the omission is silent (the OS reclaims FDs on process death so cleanup never visibly fails). Shape options:
  1. **Registry pattern.** Add `lerobot.gui.shutdown.register(callback)`; each module that opens a long-lived resource calls it at import / first-use. `shutdown_event` iterates the registry. Symmetric, discoverable; the resource owner is co-located with its cleanup. Easy to add.
  2. **AsyncExitStack on the FastAPI lifespan.** Migrate `@app.on_event("startup"/"shutdown")` to the modern `lifespan` async-context-manager API and have each module yield its `AsyncExitStack.callback(cleanup_fn)`. More idiomatic FastAPI, gets us off the deprecated `on_event` decorator (already a `DeprecationWarning` in the test output).
  3. **Resource ownership in `AppState`.** Every long-lived resource becomes an attribute of `AppState`; `AppState.close()` cascades cleanup. Most invasive but makes the lifecycle most explicit.

  Option 2 is the natural endpoint (we'll have to migrate off `on_event` eventually anyway). Option 1 is the quick win. Either way, the goal is: a contributor adding a new background thread / FD-owning resource doesn't need to know about `server.py`.

  **Orphan-subprocess failure mode (observed 2026-05-13).** A specific incarnation of this bug that bit hard: the GUI launches `lerobot-teleoperate` / `lerobot-record` / `lerobot-replay` subprocesses via `asyncio.create_subprocess_exec(..., start_new_session=True)`. The `start_new_session=True` is intentional — it detaches the child's process group so a Ctrl+C in the user's GUI terminal doesn't kill the workload mid-recording — but it also means **any abnormal GUI exit leaves the subprocess running**. Concrete case found: a `lerobot-teleoperate` subprocess was still alive almost 2 hours after the GUI server process that spawned it had exited, holding all four ttyACM ports and the cameras, blocking any subsequent launch. Only the graceful `shutdown_event` path calls `_terminate_active_process`; SIGKILL on the GUI (or any path that skips FastAPI lifespan) is silent and the subprocess survives indefinitely. Required to fix:
  - Whatever cleanup-registry mechanism we land must wrap the active subprocess so a parent-death detector (Linux: `prctl(PR_SET_PDEATHSIG)`, set in the child before exec) can SIGTERM it when the GUI's process tree dies — independent of the lifespan hook firing.
  - On GUI startup, scan for prior-launch zombies (e.g. by writing the subprocess PID into `~/.cache/huggingface/lerobot/gui/active_process.json` at launch and reading + killing on next startup if still alive and the heartbeat is stale).
  - The shutdown registry's iteration should be best-effort under exception, so a hung cleanup callback can't block the others.

- [High] **RLT metrics pipeline is wasteful end-to-end**. Two compounding issues: (1) Training subprocess rewrites the entire `metrics.json` (~150KB for 5000 points × 8 series) on every save, even when only a few new points were appended — atomic write via `.tmp` + `os.replace` on each episode end and every 100 inference steps. (2) GUI polls `/api/run/rlt-metrics` every 2s and gets the same full snapshot back, of which ~99.6% is unchanged from the previous poll. Fine on localhost, wasteful otherwise. Options: append-only JSONL or shared memory on the write side; SSE push of only new points or cursor-based polling (`?since_step=N`) on the read side. Frontend maintains a local buffer.
- [Mid] **RLT dashboard chart smoothing**. Per-inference series like `actor_deltas` have per-sample noise comparable to the actual trend (e.g. δ raw σ ≈ 0.015 per sample vs a real β-driven shift of 0.018 — z=17.8 over 500 samples but invisible in any single-sample view). The dashboard plots raw values, so genuine learning signals get hidden. Add a smoothing toggle / moving-average overlay (window picker: raw / 50 / 200 / 500), or always show both the raw line (light) and a smoothed line (bold). Same applies to `q_values_*`, `critic_losses`, `actor_q_term`, `actor_bc_term` — all have per-grad-step noise. Cheap if the smoothing happens in JS over the buffered series the dashboard already pulls.
- [Mid] **Backend-driven Launch validation schema**. The Launch button's required-field rules currently live in JS-side `_WORKFLOW_VALIDATORS` and `_POLICY_VALIDATORS` registries in `gui/static/run.js`. Each policy_type's Pydantic/Draccus config already declares its required fields in Python — the JS registry has to be updated by hand whenever those declarations change, and silent drift between the two is invisible until a user clicks Launch and gets a backend 400. Replace with a `GET /api/policy-schemas/{policy_type}` (and similar for workflows) returning e.g. `{required: ["task"], required_when: {"rlt_token_checkpoint": "rlt_mode"}}`; the frontend consumes that verbatim, so adding a new policy means only one Python edit. Until then, any change to a policy's required fields must be reflected in both places.
- [Mid] **Audit `torch.load()` for `weights_only=True`** (bandit B614, currently in global skips). Call sites: `src/lerobot/policies/act_vlm/modeling_act_vlm.py`, `src/lerobot/policies/hvla/s1/flow_matching/model.py`, `src/lerobot/policies/hvla/s1_process.py`. Since PyTorch 2.6 the default flipped to `weights_only=True`; our checkpoints predate that and contain non-tensor pickled metadata, so wholesale flip would break loaders. Plan: per-site, switch to `weights_only=True` and migrate any non-tensor state to a sidecar JSON / safetensors. Remove B614 from `pyproject.toml` `[tool.bandit].skips` once the audit completes.
- [Mid] **Predictive follower amplifies Quest VR hand jitter — feel A/B.**
  Real-arm Quest-VR session 2026-05-30 (right shoulder_lift freshly
  swapped, calibration preserved): the **plain** `bi_so107_follower`
  logged 67 IK-warning events (66 on the right arm) but felt OK to
  drive. The **predictive** `bi_so107_follower_predictive` with VR
  logged 0 IK warnings — strictly cleaner targets — yet felt harder to
  control subjectively, with the arm appearing to overshoot small hand
  motions. Hypothesis: the predictive controller's 80 ms lookahead
  multiplied by the small high-frequency jitter that the VR hand
  carries (controller pose is a noisy 6-DoF estimate, not a smoothed
  leader-arm encoder) produces velocity-corrected setpoints that
  amplify rather than dampen hand tremor. Plain-follower's leader-pose
  is a low-pass-filtered motor encoder so the equivalent jitter never
  reaches the IK. To pursue: (1) record matched VR sessions on both
  followers + replay through both with a tremor-injected leader stream
  to confirm the predictive path amplifies frequencies the encoder
  path attenuates; (2) if confirmed, add a VR-specific
  `velocity_lowpass_hz` (or disable corrector_alpha) just on the
  `BimanualCartesianIKAdapter` -> predictive path, leaving leader-arm
  teleop unaffected. The 66/1 right-vs-left IK warning split on plain
  also needs explanation — could be the swapped motor's tighter range
  limits clipping IK output, or VR-hand resting orientation that
  drives the right arm closer to a singularity than the left. Worth
  splitting the two questions before tuning.

### Recording Loop Performance

**Measured baseline** — 2-episode bi_so107_follower record run with trajectory_replay, 4 cameras at 1280×720 @ 30 fps, instrumented build with `attach_pipeline_step_spans` + temporary per-encoder probes (2026-05-11, 729 iterations, snapshot in `outputs/record_instr/latency_snapshot.json`):

| stage           |    p50 ms |    p95 ms | p95/p50 | attribution (measured)                               |
| --------------- | --------: | --------: | ------: | ---------------------------------------------------- |
| loop_dt         |     23.76 |     27.98 |    1.2x | overrun 0.69%                                        |
| get_observation |      2.31 |      8.72 |    3.8x | motor reads + cached cam reads                       |
| process_obs     | **11.54** | **19.86** |    1.7x | **99.8% DepthEdgeOverlayProcessorStep**              |
| process_action  |      0.02 |      0.02 |       — | file-backed leader                                   |
| action_send     |      0.25 |      0.31 |    1.2x | sync_write to follower bus                           |
| dataset_write   |  **4.56** | **11.91** |    2.6x | **4 × per-camera `push_frame` (∑ medians ≈ 4.4 ms)** |

Per-camera `OurStreamingVideoEncoder.push_frame()` (inside `dataset_write`):

| camera                               | p50 ms | p95 ms | p95/p50 |
| ------------------------------------ | -----: | -----: | ------: |
| front (1280×720 OpenCV)              |   0.81 |   6.36 |    7.8x |
| left_wrist (1280×720 OpenCV)         |   1.15 |   7.70 |    6.7x |
| right_wrist (1280×720 OpenCV)        |   1.15 |   6.80 |    5.9x |
| top (1280×720 RealSense color+depth) |   1.26 |   8.16 |    6.5x |

**What the data invalidated about the earlier static-analysis pass:**

- ❌ **"DepthEdge ≈ 7 ms of 17.7 ms process_obs, with 8-10 ms unexplained gap."** Wrong. DepthEdge IS process_obs — 99.8% of it (11.52 / 11.54). No gap. Optimizing depth-edge captures the whole stage win, not a partial one.
- ❌ **"obs_stream writer adds ~0.84 ms per frame."** Doesn't fire from CLI launches — the step is gated on `LEROBOT_OBS_STREAM=1`, only set by the GUI runner. Only relevant under GUI-launched recordings.
- ❌ **"dataset_write p95 spike = `feed_frame()` timeout on the batch streaming encoder ([video_utils.py:828](../../datasets/video_utils.py))."** Wrong path. Default config (`streaming_encoding=False`, `record_images=True`, `batch_encoding_size=1`) goes through the per-camera `OurStreamingVideoEncoder.push_frame()` ([dataset_writer.py:278-280](../../datasets/dataset_writer.py)), not the batch `_streaming_encoder.feed_frame()`. The supposedly-non-blocking `push_frame` is bursting 6-8x p95/p50 anyway — different root cause.
- ✅ **"Off-loop `dataset.add_frame()` removes 4.5 / 12 ms from the critical path."** Confirmed.

**Actionable optimizations, ordered by measured impact ÷ effort:**

- [x] **Cut `DepthEdgeOverlayProcessorStep` cost.** Done in two steps (2026-05-11):
  1. **Inside the algorithm:** cProfile found 70% of the cost was `np.percentile(valid_gradients, 95)` on a ~900k-sample array (full sort, O(N log N)). Replaced with `np.partition([idx_lo, idx_hi])` + the same linear-interpolation formula np.percentile uses (O(N), numerically equivalent). Also swapped `np.sqrt(gx**2 + gy**2)` for `cv2.magnitude(gx, gy)`. Output bit-identical against a golden frame; in-loop savings ~2 ms p50 (run-to-run noise dominated the headline number).
  2. **Move it off the control loop entirely:** Added an optional `post_grab_processor` attribute on `RealSenseCamera` invoked inside `_read_loop` after color+depth are ready. The grab thread consumes depth and caches the overlay-RGB; the control thread reads `latest_color_frame` like a normal camera. `BiSO107Follower.__init__` installs `DepthEdgeOverlayProcessorStep` on each RealSense camera with `use_depth=True`. Empty `get_observation_processor_steps()`. Same algorithm, same bit-identical output, but the control loop never pays the cost.
     Measured: process_obs p50 9.8 → **0.01 ms** (entire stage gone from the loop). Loop p50 12.7 → **2.6 ms** (-79% this step, -89% from the pre-optimization baseline of 24 ms). Bonus: `top` camera staleness 11 → 6.8 ms p50 because the loop now iterates ~5× faster than the camera produces frames, so the consumer always grabs the freshest possible cached frame.
- [x] **Off-loop the CPU work hidden inside `OurStreamingVideoEncoder.push_frame()`.** Done (2026-05-11). The function claimed "Never blocks the caller" but called `_reservoir_sample()` synchronously, which allocated a 345 KB float64 array per call per camera (for 4 reductions on the running stats). Moved the entire reservoir + running-stats work into `_encoding_loop()` (which is already a background thread). Switched the reservoir's `random.randint` to a per-encoder `random.Random()` so concurrent encoder threads don't fight over the global random state. Stats output verified mathematically identical via a synthetic-frames correctness test. Measured on a real recording: `dataset_write` p50 4.56 ms → 0.07 ms (-98%); p95 11.91 ms → 0.09 ms (-99%); push_frame standalone benchmark 1.1 ms → 0.5 µs (-2000×). Bonus: `get_observation` p95 also dropped from 8.72 ms → 2.51 ms because the eliminated float64 allocations were the GIL contention source for camera cache reads. Headline: loop_dt p50 17.88 ms → 12.68 ms (the off-loop save alone was -5.2 ms, half from `dataset_write` and half from reduced GIL contention).
- [x] **Investigate per-camera `push_frame` 6-8x p95/p50 ratios.** Resolved by moving `_reservoir_sample` off the caller thread (2026-05-11). Post-fix push_frame p50 dropped from 1.1 ms to 0.5 µs — the ratio is now irrelevant because both percentiles are sub-microsecond.
- [Low] **`get_observation` p95 stays ~3.8× p50 even after the loop is otherwise gutted.** Measured 2026-05-11 post-refactor: p50 2.28 ms, p95 8.73 ms. Per-span attribution rules out the obvious suspects — `motor_read_*` p95/p50 ratio is ~1.2× (stable, hardware-bound) and the `camera_read_*` spans are all <0.1 ms p95 (the lock-protected cache read is effectively free). The remaining ~6 ms tail must be in the **gaps between spans** in [bi_so107_follower.py:get_observation](../../robots/bi_so107_follower/bi_so107_follower.py) — Python dict updates between motor*read_right exit and the camera loop, the for-loop dispatch itself, or GIL handoffs while the grab thread is mid-frame on depth-edge processing (~10 ms CPU work per RealSense frame, cv2 releases the GIL but Python wrapper code doesn't). Diagnostic: add a `latency_session.add_span("obs_gap*<n>", t0, t1)` between the existing spans to localise where the gap lives. Acting on it is only worth it when pushing past 60 Hz (16.7 ms budget) — at 30 Hz the p95 fits comfortably and a real fix likely involves moving more work to the grab thread or replacing the dict-merge bookkeeping.
- [Low] **`AsyncImageWriter` non-blocking + drop-frame fallback.** [datasets/image_writer.py:174-180](../../datasets/image_writer.py) — `queue.put()` defaults to blocking with infinite timeout; switch to `put(block=False)` with logging on `queue.Full`. Not active in our per-camera config (image writer bypassed for video keys), but defensive in case a future config path reaches it.

**Already-measured non-issues:**

- `motor_read_left/right` are stable (~1.2-1.4x p95/p50). No work to do here.
- `process_action`, `action_send` together cost 0.27 ms p50. Not worth touching.
- `IdentityProcessorStep` in the default pipeline costs 0 ms — confirmed no-op.

### Latency Panel UX

- [High] **Residual teleop jitter at fast speed**: the 1 Hz writer-thread refactor cut the loop-side spike from 22 ms to ~1 ms, but the user still feels small jitter under fast movement. Suspected GIL contention: the background writer thread holds the GIL during snapshot computation (transient aggregator construction, dict building, JSON dump), and the loop thread waits for the GIL on each iteration that overlaps the writer's work. To verify: time the loop thread's iteration during the second the writer runs vs during a quiet second; difference > 0 confirms GIL hit. Mitigations to try in order: (a) reduce writer-thread compute (skip `iterations` / `aggregate_iteration` when no GUI is polling? compute fewer percentiles?), (b) drop snapshot rate to 0.5 Hz, (c) move the writer to a subprocess (`multiprocessing.Process`) which doesn't share the GIL — heavyweight but the only way to truly isolate Python work from the loop.
- [Mid] **Surface action-to-state latency**: the panel shows camera staleness, loop timing, and model/inference stages (Gantt) but not the action-to-state latency — the physical lag from issuing a `Goal_Position` command to the motor actually reaching it. `action_send_ms` is bus-TX only ("not motor motion"), and `latency_monitoring.md` deliberately scoped tracking dynamics out of "latency". The predictive controller already measures this for free: its adaptive lookahead `L` is derived from a ~2 s cross-correlation of intent vs observed state, so for `*_follower_predictive` robots the number exists at runtime and just needs plumbing into the latency snapshot + a panel card. For non-predictive robots there is no live xcorr — either compute a rolling per-joint action-vs-state xcorr in the latency aggregator, or leave the card predictive-only with an "n/a" state. Offline counterpart is the "Control-Lag Analyzer" section above (dataset xcorr); this is the live-panel version.

### Safe Trajectory Probe

Goal: a tiny, pre-baked, motion-only "probe trajectory" that any robot can execute on demand to produce a quality report — joint-space, conservative speed limits, stays inside a known safe envelope, no Cartesian collision math, no human teleop required.

Use cases:

- **Real action_send_ms measurement** under bus load — currently 0 in our dry-run tests because send_action is no-op'd.
- **Tracking lag measurement** without needing an operator at the leader arm; replaces the cross-correlation analyzer for hardware-side characterisation.
- **Reproducible hardware benchmarks** — same trajectory across different robots / different days, compare per-joint lag, peak loop_dt, encoder behaviour under predictable motion.
- **Cross-camera sync verification** when the arm IS moving (camera staleness against a known-moving target).
- **Per-joint motor health checks** — does each motor respond? backlash, thermal envelope.

Design constraints:

- [ ] Joint-space waypoints, not Cartesian — sidesteps IK / collision concerns.
- [ ] Per-robot **safe envelope** config: per-joint min/max angles, max velocity, max acceleration. Stored next to the robot profile.
- [ ] Smooth profile (cubic spline / trapezoidal velocity) — no step inputs, no near-limit positions.
- [ ] Hand-recorded once with the leader arm — the operator records the safe envelope trajectory, the file becomes the reference. Stored alongside the robot config (e.g. `safe_probe.parquet`).
- [ ] **Not** the same as Replay (which is for dataset playback). This lives in `src/lerobot/scripts/lerobot_probe.py` and emits a structured report.

Output: `<robot_id>_probe_report.json` with per-joint tracking lag, peak loop_dt, overrun ratio, per-camera staleness during motion, per-joint backlash estimate (max(state - action) reversing direction), and a pass/fail verdict against the robot's expected baseline. Drop-in replacement for the `meta/recording_health.json` format so the data panel can render it the same way.

Initially: build for `bi_so107_follower` (the active hardware), generalise later. Build only when there's a concrete use case forcing the hand — for now this TODO captures the design.

### Dataset Debugging Overlay

Live overlay during teleop/record showing how the current state compares to the dataset — helps the user identify gaps in data coverage and fill them efficiently.

- [ ] **Live coverage indicator**: compare current observation against the dataset (either a selected reference dataset or the one currently being recorded). Show as an overlay badge (like the S2 subtask overlay) indicating how "novel" the current state is relative to existing data.
- [ ] **Growing coverage feedback**: as the user records more episodes, the feedback should reflect that more cases are covered — "you've seen states like this N times" or a heatmap-style confidence.
- [ ] **Define similarity metric**: what does "similar" mean? Options to explore:
  - Joint state L2 distance (cheap, ignores visual context)
  - Image embedding distance (e.g. DINO/CLIP features, captures visual similarity)
  - S2 latent distance (if debug model is loaded — reuses existing infrastructure)
  - Hybrid: state distance + image embedding distance
- [ ] **Nearest-neighbor lookup**: build an index (e.g. FAISS) over dataset observations, query with current obs each frame. Display distance + closest episode/frame reference.
- [ ] **TODO: hardcoded vs generic**: start with a simple metric (joint state L2 or S2 latent distance), add a comment that this will be generalized later (same pattern as model debugger overlay).

## Robot Tab

- [Low] UX consistency pass: ensure consistent button coloring/hierarchy across views
- [Low] ~1s latency when first opening the Robot tab while loading profiles
- [ ] **Surface `/api/robot/recover` from runtime error states (deeper integration).** A dedicated "Recover" button is shipped next to the rest-position buttons (`renderRestPositionSection` in `robot.js`), and `recoverRobot()` chains into `/move-to-rest-position` when the report is clean and a rest position is on file (with doubled `duration_s` if anything was recovered). What's missing: rendering an inline "Try recovery" link next to the error toast when teleop / record / replay / move-to-rest fails because of a wedged motor chain (e.g. `FeetechMotorsBus motor check failed ... Missing motor IDs`). Today these errors are printed-only (`logger.exception` + raw string in the response body), so users have to read the server log to know what's wrong, and there's no path from "I see an error" to "click to attempt recovery". Prereq: consolidate error handling — promote the relevant `RuntimeError` / `ConnectionError` from the bus and robot layers into a small set of typed exceptions (e.g. `MotorChainWedged`, `PortBusy`, `MissingMotor`) that the FastAPI layer maps to structured error bodies (`{"error_code": "...", "detail": "...", "remediation_hint": "recover"}`). The frontend can then render the appropriate hint generically instead of string-matching error messages.

### Guided Calibration Wizard

LeRobot's stock calibration is rudimentary — `robot.calibrate()` is blocking
`input()` / `print()` prompts, no visual reference for the "middle of range"
homing pose, no verification. Replace it with a GUI-driven, visually-guided
wizard. Designed as three phases; ship phase 1 first as its own PR.

- [ ] **Phase 1 — motor calibration, any robot, no URDF.** (1) Homing: show a
      mid-range reference pose, the user matches the physical arm to it, capture.
      (2) Ranges: GUI-driven range-of-motion recording. Reuse the motor-bus
      primitives directly (`set_half_turn_homings`, `record_ranges_of_motion`,
      `MotorCalibration`, `write_calibration`) — NOT the `calibrate()` wrapper,
      which is CLI-`input()`-bound. Emit the same calibration files so the tool
      complements `lerobot-calibrate`. No camera pane — the human is at the robot.
- [ ] **Phase 2 — URDF verification + live layer-2 tuning.** For robots with a
      vendored `*_description` URDF: verify the URDF against the live observed
      pose, and let the user tune the per-joint `(sign, offset_deg)` layer-2
      alignment interactively instead of hand-editing `joint_alignment.py`. Only
      CAD-exported URDFs (e.g. SO-107) need this; a calibration-aligned URDF
      (SO-101) is identity. Depends on the URDF state visualization.
- [ ] **Phase 3 — kinematic calibration (research).** Beyond joint zero/range:
      link-length / mounting-offset estimation. Scope TBD.

### Motor Diagnostics (per-arm health panel)

When a motor on a real arm misbehaves the user currently has to drop into
ad-hoc Python (e.g. `scripts/diag_motor_2.py`, `scripts/scan_motor_variants.py`)
to figure out _why_. Surface the same information in the Robot tab as a
per-arm diagnostic panel so the diagnosis flow is "click → see structured
report" instead of "open a terminal, write a probe, paste output to a
collaborator." Captured after a real session where a dying right-arm motor
took several rounds of CLI work to isolate.

**Working CLI today:** the diagnostic + migration flows already exist as
`scripts/motor_tools.py` on the `feat/motor-tools` branch
(`inventory` / `health` / `read` / `write` / `set-id` subcommands) plus
`scripts/MOTOR_MIGRATION.md` for the end-to-end swap-preserving
procedure. The work below is to lift those flows into the GUI; the CLI
is the spec.

- [ ] **Per-motor health probe.** For each motor on the selected robot
      profile: ping, then read `Torque_Enable`, `Goal_Position`,
      `Present_Position`, `Present_Current`, `Moving`, `Operating_Mode`,
      `Lock`, `Min_Position_Limit`, `Max_Position_Limit`, `Max_Torque_Limit`,
      `Torque_Limit`, `P_Coefficient`, plus a tiny commanded-move test
      (both directions) with a watch loop. Diagnostic verdict ladder:
      _bus failure_ (ping fails) → _MCU rejects writes_ (Torque*Enable
      doesn't latch) → \_driver dead* (Goal/Present diverge but Current=0)
      → _mechanical jam_ (Current spikes, Position doesn't change) →
      _misconfigured_ (Operating_Mode / Lock / P=0). Sounds-like-buttons
      next to each motor row.
- [ ] **Motor registry / per-arm inventory.** Read
      `Firmware_Major/Minor_Version`, `Max_Voltage_Limit`, `Present_Voltage`,
      `Model_Number` for every motor on every connected port and surface
      them in a per-arm table. Caveats discovered while building the CLI
      version: STS3215 variants (7.4V vs 12V; 1:147, 1:191, 1:345 gear
      ratios) all share `Model_Number=777` and identical firmware
      (3.10). **`Max_Voltage_Limit` is a writable EEPROM field**, so a
      heuristic that calls 8.0V "7.4V variant" and 12.0V "12V variant"
      breaks the moment anyone writes that register (real example:
      shoulder*lift motors on both leaders of `white` read 8.0V while
      the rest of the same arm read 12.0V — manual rewrites, not actual
      variant difference). **Gear ratio isn't readable from firmware at
      all** — the MCU only sees the encoder. So the registry is honest
      diagnostic info but **must not pretend to identify variant or gear
      ratio**; ground truth is the printed label on the motor housing.
      A useful UI gives the user a place to \_record* the variant they
      see on the label, persisting alongside the calibration so future
      sessions remember which motor is where.
- [ ] **Swap-preserving recalibration helper.** When a single motor needs
      replacement, the standard `calibrate()` flow throws away the rest
      of the arm's calibration and re-measures _everything_ via human
      eyeballing the canonical pose (±2-5° per joint of drift versus old
      datasets). A surgical tool would let the user (1) before pulling
      the dying motor, capture `raw_old` and `homing_offset_old` at any
      held pose; (2) pause for the swap; (3) re-read `raw_new` at the
      same physical pose; (4) compute `homing_offset_new = raw_new -
(raw_old - homing_offset_old)` and update only that one motor's
      entry in the calibration JSON. Preserves the rest of the arm's
      old calibration exactly, so existing datasets, FK, and IK frames
      stay valid without recalibrating six other joints. Critical detail:
      the captured-pose step needs a visual / mechanical reference the
      user can reproduce post-swap (mechanical end-stop, marked tape, or
      lining the broken arm against the working twin) — surface those
      options in the wizard. **Also write `Min_Position_Limit` /
      `Max_Position_Limit` to the new motor's EEPROM**, not just
      `Homing_Offset` — a fresh motor harvested from a different role
      carries stale limit registers that silently clip `Goal_Position`
      to a narrow window unrelated to your arm. The CLI counterpart
      (`scripts/motor_tools.py write … --min-position-limit=… \
      --max-position-limit=…`) hit this exactly: first attempt drove to
      809 ticks then froze at 958 because the new motor had been a
      gripper on its previous arm with max=952.

## Architecture

- [**Critical**] **Reactive UI state management**: the current imperative DOM manipulation (innerHTML + manual toggle/refresh calls scattered across functions) is fundamentally broken. Field state (disabled, visible, selected) must be called at every possible code path that reveals the element, leading to endless monkey-patching. Migrate to a reactive pattern where UI state is derived from data (React, Preact, or even a minimal reactive store + render loop). This blocks every new UI feature.
- [High] **Coherent GUI-wide persistence policy.** Persistence is currently ad-hoc, per-feature: opened datasets live in `opened.json`; model sources in `model_sources.json`; robot/teleop profiles under `~/.config/lerobot/`; per-feature view state in scattered LocalStorage keys; **launch settings (Run tab, Model Debugger, RLT mode, episode counts, robot/teleop profile choice, etc.) aren't persisted at all and reset to defaults every session**. The result is two distinct failure modes for users:
  1. **Silent default surprise.** A toggle that controls destructive behaviour (e.g. RLT `--rlt-deploy` vs train mode) reverts to its default between sessions. The user reasonably assumes their last choice is still in effect, launches, and only realises mid-run (or later, reading logs) that the wrong mode ran. Real example: 2026-05-14, an HVLA + RLT rollout intended to be DEPLOY mode launched in TRAIN mode because `--rlt-deploy` wasn't toggled in the GUI — gradient updates ran for an entire evaluation session, contaminating the checkpoint.
  2. **Stale-choice surprise.** A persistent dropdown (robot profile, model checkpoint path) retains a previous selection from a different workflow; user launches without noticing. Reported once at least.

  These are symptoms of a missing GUI-wide policy. Worth thinking through, not just patching per tab.

  Concrete questions to answer in a short design doc before implementation:
  - **What to persist?** Per-tab session state (current selections, expanded/collapsed view), per-workflow last-used parameters (last train args, last record args, last debug-model checkpoint), or both? Different lifetimes — session state is per-browser; workflow state is per-user.
  - **Where to persist?** Server-side JSON in `~/.config/lerobot/gui/` (survives browser changes, can be backed up, single source of truth) vs LocalStorage (zero round-trip, per-browser). Probably server for workflow state, LocalStorage only for UI affordances (which panel was expanded, scroll position).
  - **Reset semantics.** Some launches must NOT inherit prior state (training run on dataset A → next launch shouldn't auto-pick A; user might be deliberately switching tasks). Need an explicit "use last config" vs "fresh defaults" affordance, and the default should be the safer of the two.
  - **Visibility / freshness.** When the GUI restores 2-week-old launch flags, the user should be told. Show a freshness banner ("Last used 14 days ago — review settings before launching") or surface a diff against a documented default.
  - **Audit trail.** Currently the only record of what was launched is the GUI server log. Make the resolved CLI / config visible in the UI before submission — both as a confirmation step and as a record. The Launch button modal could show the constructed command in a copyable block.
  - **Safe defaults for destructive toggles.** Things like RLT train mode, dataset deletion, push-to-Hub: defaults should fail closed (require explicit opt-in each session), regardless of persistence.

  Likely deliverable: a `GuiPersistenceStore` server-side abstraction (single JSON file per concern, atomic write, schema-versioned for migrations) + a frontend wrapper that hydrates form fields on tab mount + a "Launch summary" modal that surfaces resolved settings before submission. Touches every tab — but the consolidation is the win.

  Scaling concern: as platform features grow (HVLA, RLT, intervention, eval suites, dataset tools), the surface of "settings that could surprise the user if not persisted (or persisted incorrectly)" grows linearly. Doing this once at the policy level is cheaper than retrofitting per-tab.

- [Mid] **Config grouping / categories** (to be designed): the number of configurable settings keeps growing (Run tab workflow params, Robot tab profile fields, RLT toggles, debug-model knobs, etc.) and they're currently rendered as one flat form per tab. Need a grouping/category abstraction so related fields cluster visually — collapsible sections, tabs-within-tab, "Advanced" reveal, or schema-driven groups declared alongside the Pydantic/Draccus config. Design open: where the grouping metadata lives (Python config decorators? frontend JSON?) and how collapsed state persists (ties to the [Persistence policy](#) item above).
- [Mid] Cross-tab data synchronization — replace point-to-point refresh calls with pub/sub event bus
- [High] Extract frontend to separate files (then optionally migrate to React/Vite)
- [Mid] FastAPI dependency injection for AppState — replace global `_app_state` with `Depends(get_app_state)`
- [Mid] Consolidate module-level caches (`_episode_start_indices`, `_dataset_info_mtime`) into AppState
- [High] **Unified intervention contract — any teleop in any flow.** Three call sites assume a human-intervention surface on the teleop: [`s1_process.py` (HVLA + RLT)](../policies/hvla/s1_process.py), [`lerobot_record.py` (regular record with policy)](../scripts/lerobot_record.py), and [`hil_processor.py` (HIL serl gym)](../processor/hil_processor.py). The "contract" today is **five methods that every teleop re-implements independently**:
  - `intervention_enabled: bool` field in the config (HVLA forces it `True` at launch — a teleop config that doesn't declare the field crashes at draccus decode time)
  - `get_teleop_events() → {IS_INTERVENTION: bool, TERMINATE_EPISODE: bool, SUCCESS: bool, RERECORD_EPISODE: bool}`
  - `reset_intervention()` at episode boundary
  - `disable_torque()` / `enable_torque()` for the policy↔teleop swap (only meaningful for physical leaders; non-physical teleops like Quest VR have nothing to do here, but the call sites guard with `hasattr` so this is more annoyance than blocker)
  - `set_intervention_transition_lock(bool)` to gate user toggles while the swap is mid-flight (servo sync, torque enable, etc.)

  Every leader teleop (`so_leader`, `bi_so107_leader`, the two highrate variants, `keyboard`, `gamepad`) re-implements the toggle plumbing — each spins up its own `pynput` SPACE listener, owns its own `_intervention_active` / `_intervention_transition_lock` / `_intervention_debounce_s` state, duplicates the `_try_toggle_intervention` decision function. There's no shared mixin, wrapper, or protocol. A new teleop (Quest VR landed 2026-05-31 with **zero** intervention surface) doesn't plug in — it can't be used as the human source for HVLA RLT, the record-loop's intervention path, or HIL gym, until five methods + one config field are added by hand.

  There's a second, deeper gap: **`teleop.get_action()` is assumed to return joint commands**. A Cartesian teleop (Quest VR) returns Cartesian EE deltas until a Cartesian-IK robot wires it up via `robot.attach_teleop(teleop)`. HVLA / record never call `attach_teleop`, so even if Quest VR sprouted the five intervention methods tomorrow, the joint dict the intervention path needs wouldn't materialise. Today the joint-emitting teleop is implicit in the leader's hardware; for any non-leader teleop the call site has to opt in.

  Proposed shape (design before code):
  - **Shared toggle abstraction** — extract the SPACE-listener + `_intervention_active` + debounce + transition-lock state into a small `InterventionState` class (or `InterventionMixin`) the teleop composes in. The **toggle source is pluggable**: keyboard SPACE for leaders, gamepad button for `gamepad`, **Quest controller face button B/Y at WebXR index 5** for VR (haptic-feedback path already exists to signal "intervention engaged" without the operator looking outside the headset).
  - **Joint-emission entry point** — HVLA / record / HIL-gym call `robot.attach_teleop(teleop)` after `teleop.connect()`. Robots that don't need it (leaders that already emit joints) keep today's no-op `attach_teleop`. Robots with Cartesian-IK paths (`BiSO107Follower`, `BiSO107FollowerPredictive`, `VirtualBiSO107Follower`) install the IK transform transparently. Same machinery `lerobot_teleoperate` already uses for the Cartesian path; intervention paths just have to call it.
  - **Torque ops are optional** — keep the `hasattr` guards in the call sites, but document explicitly that `enable_torque`/`disable_torque` are leader-only. Non-physical teleops (VR) implement neither.

  Until this lands, every new teleop pays the duplicated-plumbing tax, and the immediate Quest VR ask "use VR for HVLA intervention" requires per-teleop hacks rather than a clean opt-in.

  Open design questions before any code:
  - Mixin vs wrapper class — mixin is less invasive, wrapper composes more cleanly and survives multiple inheritance. Wrapper probably wins.
  - Where does the toggle binding live — config field (per-teleop, declared in the config dataclass) vs hardcoded in each teleop class? Config field is more flexible but inflates each config; per-class with a class-level default is probably the right shape.
  - Bimanual leaders attach the intervention state to one arm only (the left arm owns the SPACE listener). Quest VR is intrinsically bimanual — what does intervention mean when only one Quest controller is engaged? Likely: any-arm-engaged = intervention is active. Document the per-teleop semantics.
  - Bidirectional torque sync on leaders (servo to follower at intervention release) is leader-specific. For Quest VR the equivalent is "re-anchor the engage snapshot at intervention start" (which the existing reset-button path already does — same primitive). Both can be expressed as a single `on_intervention_transition(direction)` hook the teleop implements.

- [High] **Decouple data loaders from UI state**: today, three loaders gate data fetching on UI flags:
  - `model.js::loadModelSources` only scans sources where `s.expanded` is true
  - `app.js::loadSources` (datasets) — same shape
  - `robot.js`'s `robotTabInitialized` / `teleopTabInitialized` one-shot guards force consumers like `run.js:23-35` to inline-fetch profiles as a workaround

  Symptom: cross-tab consumers (e.g. Run -> Policy dropdown) get stuck on "Loading models..." when no Models-tab source is expanded. Each new consumer keeps inventing "ensure-loaded" workarounds because the data layer leaks UI state.

  Refactor: each owner module exposes an idempotent `ensureXxxLoaded()` that fetches and caches without consulting UI state. Tab init and other tabs both call the same function — the data layer never references `expanded` / `tabInitialized` flags. The `expanded` flag remains in `~/.config/lerobot/model_sources.json` purely as UI persistence (governs render shape, never data fetches).

  Sites:
  - `model.js`: introduce `ensureAllModelsLoaded()`. `modelTabInit` calls it. Drop the `expanded`-gated scan inside `loadModelSources`. `run.js::_ensureModelDataLoaded` becomes a one-liner.
  - `app.js`: introduce `ensureAllSourceDatasetsLoaded()`, same pattern.
  - `robot.js`: `ensureRobotProfilesLoaded()` / `ensureTeleopProfilesLoaded()`. Remove the inline fetches in `run.js:23-35`.

  Invariant after refactor: `s.expanded` references appear only in render/toggle code. Periodic check:

  ```bash
  grep -nE "if.*\.expanded" src/lerobot/gui/static/*.js | grep -vE "render|toggle|html"
  # Should match nothing.
  ```

## Bug Reporting

The "Report bug" button in the tab-bar (top right) stores a report **on the GUI server** under `~/.cache/lerobot/bug_reports/<timestamp>_<slug>/` with `report.json` + `screenshot.png`. Backend: `gui/api/bug_reports.py`. Frontend: `gui/static/bug_report.js`. Nothing is sent off the server.

- [Mid] **Upload to GitHub issue**: optional "Upload" button on a saved report. Use `gh issue create --repo <owner>/<repo> --title ... --body-file report.json --attach screenshot.png` (piggy-backs on `gh auth login` configured on the GUI server). If `gh` is missing or unauthenticated, the button is disabled with a tooltip. Add `GET /api/bug_reports/{id}/gh-status` to surface readiness, and `POST /api/bug_reports/{id}/upload-gh` to do the upload. Server-side save stays the default; upload is opt-in per report.
- [Low] **Vendor html2canvas**: today the client lazy-loads html2canvas from jsdelivr. Air-gapped / restricted-network setups need a vendored copy at `gui/static/vendor/html2canvas.min.js` with the loader preferring the local path. ~150 KB.
- [Low] **Recent reports panel**: small "View past reports" section that lists what `GET /api/bug_reports` returns. On localhost the server path is also the user's path, so a "copy path" affordance is enough; once hosted, the panel should offer "download report" (zip the directory) instead, since the server filesystem won't be the user's.
- [Low] **Camera/video tiles in screenshots**: html2canvas renders `<video>` and `<canvas>` elements blank or as poster frames. For reports about a live camera glitch this drops the most useful pixels. Workaround: also snapshot each visible `<video>` / `<canvas>` via its own `.captureStream()` / `getContext('2d').getImageData()` and stitch them into the report directory as separate PNGs.
- [Low] **Per-user scoping when hosted**: today every report lands in one shared directory keyed by timestamp+slug. Fine for a single-operator localhost setup; on a hosted deployment we'd want `<user>/<timestamp>_<slug>/` (or a small SQLite index) so triage can filter by submitter and one user can't see another's reports. Postpone until there's a real auth story.

## UX

- [Mid] Cross-reference navigation: clickable links from dataset/model/robot references to their tab (generic utility, not one-off per instance)
- [Mid] **Demote the "AI setup" top-bar link into a settings/gear (⚙) menu.** Currently it sits next to Transfers and Report bug — both session-frequent — while AI setup is a per-device, one-off ceremony. Visual prominence mismatches usage. A small ⚙ icon opening a dropdown ("AI setup", and future admin links like token rotation / system info / log viewer / the WS-auth toggle if that audit followup lands) de-emphasizes one-offs without hiding them, and matches conventions users already know (VSCode, Slack, GitHub). No new tab required.
- [Mid] **Unify notification UX.** The GUI currently has two parallel notification surfaces: the existing toast bar (`showToast()` in app.js, top-right, used for "dataset opened", "save failed", etc.) and the browser's OS-level Notification API (used by the MCP bridge tool `notify_user` for backgrounded-tab pings). They look and feel different and don't share routing. Slack-style pattern is the target: in-tab events → toast; background-tab events → OS notification with the same content. A small notification dispatcher would let both `showToast` callers and `notify_user` share one surface and one escalation policy. Until this lands, `notify_user` keeps the OS-notification path (it's the only thing that refocuses a background tab) but isn't surfaced in the e2e demo because it's not visually integrated with the rest of the GUI.

## AI-native (MCP)

- [High] **Bridge WS subscriber auth.** Today `/api/bridge/ws` accepts any LAN connection — fine under LAN-trust + single-operator, but a wildcard subscriber can tail every AI command the operator's AI fires (every `navigate_to`, `notify_user`, `highlight_in_viewer`). Two design choices to make: (a) require a bearer on WS upgrade (and give GUI tabs a way to obtain one — currently they have none), or (b) default tabs to a per-tab UUID target so the wildcard subscription becomes an explicit operator action. Plan doc's "Open decisions" item 2 captures the choice. Audit caught this pre-merge and accepted the risk for the LAN-trust + single-operator posture this PR ships under.
- [High] **Surface MCP comments in the GUI.** The MCP `tag_episode` tool writes to a sidecar SQLite (`_mcp_annotations.sqlite`) that's keyed by `(repo_id, episode_id, key)`. Today only AI tools read them back via `get_episode_tags`; humans see nothing. A small Annotations panel in the Data tab (next to Inspector, or as an Inspector tab) would close decision #7 in the plan doc — "comments are eventually surfaced in the GUI too." Spec: per-episode list of `(key, value, set_at)` rows, clickable to filter / highlight, with a manual "Add comment" affordance so it's not AI-only. Plan doc captures the intent.
- [Mid] **Filter UI for the dataset tree → re-register `set_filter`.** The MCP `set_filter` tool was dropped from v1 because no GUI viewer has a real filter input. When the data sidebar grows a search/criteria field (likely with the Annotations panel above), re-register `set_filter` in `mcp/bridge_tools.py` and re-wire the `lerobot-bridge:filter` consumer in `bridge_consumers.js` (the listener was stubbed out, comment left as a marker). The wire shape (`"filter"` in `SUPPORTED_COMMAND_TYPES`) is already stable.
- [Mid] **Bridge tool surface — edit + operate tiers.** This PR ships the read + comment + bridge surface (~12 tools). The plan doc's "Tools (design surface)" table describes the full ~52-tool target. Next chunks, one PR per domain: Dataset edits (`propose_*` + `apply_pending_edits` mirroring the GUI's PendingEdit pipeline), Hub (auth-status / repo-info / `start_upload` / job-progress), Models (sources + debug-model load/unload), Robots (profile reads first, then writes, then motor ops), Run (status reads + `stop_current_run` first, then `start_*`). Each domain needs a shared-business-logic refactor between its FastAPI route and the MCP wrapper — don't auto-bind.
- [Low] **Surface `list_tagged_episodes` MCP tool.** The underlying storage method (`AnnotationStore.list_tagged_episodes`) already exists; only the MCP wrapper at `mcp/server.py` is missing. ~3 lines + a test. Bundle with whichever larger PR touches the dataset-read surface next.
- [Mid] **Add `get_feature_series` MCP tool.** Per-frame time series for a chosen feature (e.g. action / state / reward) across an episode. The GUI's `/api/datasets/{repo_id}/episodes/{episode_idx}/feature-series` route already does this; the MCP version needs the same business logic refactored out so both surfaces share it (NOT another HTTP self-call from MCP — that's the in-process anti-pattern). Estimate is larger (~30 lines + extraction) than the `list_tagged_episodes` wrapper above.
- [Low] **Retro transcript log for the original PR #20 tools.** The dataset-edit PR (#22) established the "transcript log alongside screenshots" standard documented in `mcp/README.md`. The PR #20 tools (list_datasets, get_dataset_info, list_episodes, get_episode_summary, get_frame, get_episode_tags, tag_episode, delete_episode_tag, lerobot_whoami, navigate_to, notify_user, highlight_in_viewer) have `demo_e2e.md` (happy-path 9-call demo) but no structured error-case transcript. Capture an e2e transcript analogous to `dataset_edit_transcript.md`, or fold the missing error coverage into the existing demo. Not blocking, but closes the standard-uniformity gap.
- [Low] **Consider splitting `navigate_to` into typed sub-tools.** Today `navigate_to(view: str, params: dict)` takes a view-dependent dict whose shape depends on `view`. Documented in the docstring but the AI has to read carefully. Alternatives: `navigate_to_episode(repo_id, episode_id)`, `navigate_to_dataset(repo_id)`, `navigate_to_tab(tab)`. Trade-off: more tools to maintain, but each has an unambiguous schema. Audit considered; left as-is for v1 — revisit if AI agents trip over the dict-params shape in real use.

## Agentic frontend observation

Goal: an AI agent on the operator's machine can both _drive_ the lerobot GUI (existing bridge MCP) and _observe_ it (screenshots, recordings, DOM state) via natural language, without per-task bespoke Python scripts. Pattern is Bash + skill markdown + small single-purpose CLIs — no new MCP server. The agent orchestrates; the CLIs each do one thing.

- [High] **`scripts/gui/capture_window.py` — capture the operator's open browser tab.** The blocking gap. Today `screenshot_gui.py` spawns its own Chrome at a fixed position; for interactive flows the operator already has the GUI open. New util: `find_lerobot_window()` returns `{x, y, w, h}` via `xdotool search --name lerobot` (Linux) / `osascript` (macOS) / `EnumWindows` (Windows), then `ffmpeg x11grab` against that geometry, write to `~/.lerobot/captures/<ts>.png`. ~80 lines. Without this nothing else in this section works.
- [High] **Skill: `.claude/skills/capture-lerobot-frontend.md`.** Tells the AI "to see the GUI, run `scripts/gui/capture_window.py` and Read the PNG it prints." Tiny markdown. Pairs with the script above; the skill IS the agentic orchestration layer (no MCP needed for synchronous capture — Bash + Read tool already cover it).
- [Mid] **`scripts/gui/record_window.py` — timed video capture.** Same `find_lerobot_window` underneath; ffmpeg x11grab with `-t <duration>` writes MP4 to `~/.lerobot/captures/<job_id>.mp4`. PID-file lifecycle for `--start` / `--stop <job_id>` so an agent can record across other tool calls.
- [Mid] **`scripts/gui/gui_compose.py` — post-process toolkit.** Side-by-side composer, crop-to-window, extract-frame-at-time. Each is a 5-line ffmpeg/Pillow wrapper, but canonical CLIs mean skill markdown reads "for before/after, call `gui_compose.py side_by_side a.png b.png out.png`" instead of teaching ffmpeg flags every conversation.
- [Mid] **Bridge symbolic-observation extension.** Adds `get_frontend_state()`, `get_console_logs()`, `query_dom(selector)` to the existing bridge — browser tab answers via injected JS over the WS. No new infrastructure; same channel that already carries `navigate_to`. Closes "did the bridge command actually render?" without needing pixel capture. Pairs with the bridge WS subscriber auth work above.

Explicitly out of scope (ruled out in the design discussion before merge of PR #20): bundling general screen-capture / clipboard / OS-input as MCP tools; shipping a browser extension; persistent screen-share permission via `getDisplayMedia`. The OS-level capture path (ffmpeg x11grab, screencapture, BitBlt) inherits the operator's existing user permissions and needs zero per-session consent on Linux X11, a one-time settings toggle on macOS.

## Dataset Tools

- [Mid] Consolidate `_keep_episodes_from_video_by_time` (time-based) with `_keep_episodes_from_video_with_av` (frame-based, upstream) in `dataset_tools.py`. Migrate trim callers to frame indices.
- [Mid] Consolidate streaming video encoders: our `video_encoder.py` vs upstream's `video_utils.py`. Upstream's is more mature (HW encoders, frame dropping). Consider migrating.
- [High] **Move per-frame stats work off the loop thread** in `StreamingVideoEncoder.push_frame`. Measured cost: ~4 ms p50 / 5 ms p95 per `dataset.add_frame()` call (4 cameras × 720p, synthetic random images on a fast machine) — see `/tmp/encoder_benchmark.py`. That's ~12% of a 30 Hz budget consumed by work that does NOT need to happen synchronously: `_reservoir_sample` (downsample copy + running per-channel mean/std/min/max accumulation) and HWC/CHW + dtype normalisation all run on the caller's thread inside `push_frame`. The actual encoding already runs in the background thread; the stats work is the leftover synchronous tax. Real cameras compress better than random noise, so production numbers may be lower, but the synchronous portion is the same. Risk surface: at higher control rates (60 Hz halves the budget to 16 ms) or on weaker CPUs, this 4 ms becomes a real fraction of the iteration time and can push the loop into overrun under load. Plan: (a) push the raw frame onto the queue first (cheap), (b) let the encoder thread do the reservoir sampling + running stats off-loop. The queue is already unbounded so backpressure isn't a concern; only the memory of in-flight frames grows during a backlog, which is acceptable. Sites: `src/lerobot/datasets/video_encoder.py::push_frame` and `::_reservoir_sample`.
- [High] **Duplicate detection within dataset**: detect near-duplicate episodes during dataset opening and before merging. Prevents wasted training compute on redundant data. Could use joint state trajectory similarity or image embedding distance.
- [Mid] **Subtask labeling in GUI** — **superseded by [docs/feature_editing.md](docs/feature_editing.md)**. V1 of Feature Editing delivers exactly this: drag-select a frame range on the subtask row → type the label in the Inspector → Apply.
- [Mid] **Subtask format**: conform subtask column to LeRobot 3.0 format + OpenPI changes. Currently uses raw string column; may need task_index remapping.

## HVLA / Policy Evaluation

- [**Critical / Safety**] **State-to-first-action gap at inference start.** At t=0 of a policy rollout, the robot sits at some state `s₀` and the policy emits `chunk[0..H]`. If `chunk[0]` is far from `s₀` — e.g. the policy was conditioned on an observation where the gripper was already in pre-grasp pose, but the operator left the arm in a random rest position — the motor goal is set to a position several joints' degrees away from where the arm currently is, and the motor moves at full available speed to close the gap. **This is a real safety hazard**: the arm can crash into the workspace, the operator, or other arms before anyone reacts. Fix shapes, roughly cheap → safe:
  1. **Pre-position step before policy runs.** Reuse the existing `move_to_rest_position` flow / safe-trajectory ramp: drive the robot from `s₀` to `chunk[0]` over `ramp_duration_s` (default 2-3 s, scale with joint-space distance). Only then engage the policy loop. Mirrors what `lerobot_record.py` already does for the auto-reset phase between episodes — same code path, just driven by the first policy action instead of `start_pose`. Smallest blast radius.
  2. **Velocity-limited safe-step controller in `send_action`.** Cap the per-step `Goal_Position` delta to `max_velocity_deg_per_step` regardless of what the policy asks for. The motor can't be commanded to teleport. Protects against bad policy outputs across the entire episode, not just the first frame.
  3. **Pre-flight check.** Refuse to start the rollout if `||chunk[0] − s₀||₂ > safety_threshold`. Show the operator a "robot is too far from policy's expected starting pose; reset arm first" message. Belongs in the GUI before `/api/run/start_hvla` flips to "running".
     Highest priority: ship #1 (reuses existing infrastructure, immediate). Track #2 as the durable safety net since it protects against more than just startup. #3 is the user-facing complement. Design tracked separately before any code lands.
- [High] **Train S1 with intervention data**: merge `thewisp/intervention_cylinder_ring_assembly` corrections into training dataset and retrain S1 to learn from human corrections (DAgger-style).
- [Mid] **Optional success dataset**: when running HVLA with intervention, optionally save episodes where the user advanced without ever intervening to a separate "success" dataset. Requires proper `clear_episode_buffer` lifecycle and video encoder management — refactor `s1_process.py` recording logic first.
- [Mid] **Refactor s1_process.py recording**: the episode lifecycle (dataset creation, encoder start/stop/discard, intervention buffer management) has grown organically. Align with `lerobot_record.py`'s structure before adding more features.

## Testing

- [ ] API datasets tests (`tests/gui/test_api_datasets.py`) — TestClient + mock dataset fixtures
- [ ] API edits tests (`tests/gui/test_api_edits.py`)
- [Low] Playback WebSocket tests (`tests/gui/test_api_playback.py`)

## Hardware

- [Low] Use stable Linux device paths (`/dev/serial/by-id/`, `/dev/v4l/by-id/`, `/dev/v4l/by-path/`) instead of volatile `/dev/ttyACM*` and `/dev/video*` in profiles. GUI Robot tab could auto-detect and offer these in port selection.

## Python 3.12+ Compatibility

- [Mid] Remove Python < 3.12 workarounds once we drop 3.10/3.11 support. Upstream lerobot now requires 3.12+. Our fork pins 3.10 compatibility via:
  - `datasets/utils.py`: `class Backtrackable(Generic[T])` -> native `class Backtrackable[T]:`
  - `motors/motors_bus.py`: `NameOrID = Union[str, int]` -> native `type NameOrID = str | int`
  - `utils/io_utils.py`: module-level `T = TypeVar(...)` -> native `def foo[T: Bound](...)`
  - `processor/pipeline.py`: `Generic[TInput, TOutput]` -> native type params
  - `policies/pretrained.py`: conditional `Unpack` import -> direct `from typing import Unpack`
  - Multiple modeling files: `from typing_extensions import Unpack` -> `from typing import Unpack`

## Workflow

To work on this TODO autonomously:

1. Read the TODO.md
2. Find the next incomplete task and implement it
3. Commit your changes
4. Update TODO with what you did
   ONLY DO ONE TASK AT A TIME.

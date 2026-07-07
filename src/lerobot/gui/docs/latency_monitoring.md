# Latency Monitoring Design

End-to-end latency observability across **every real-time loop in lerobot**: teleop, dataset recording, and policy inference (sync and async). The operator/researcher gets live numbers in the GUI (per-camera corner badges + a bottom dashboard); optional per-iteration JSONL gives offline analysis when wanted; one-shot calibration against an external rig (V2) grounds the software-measured numbers in physical reality.

V1 lands the infrastructure on the teleop loop because it's the simplest closed loop with all the right ingredients (cameras, motors, fixed-rate iterations). The same `LatencyTracer` / `LatencyAggregator` / `JSONLWriter` then plug into `lerobot-record` and the inference paths without re-inventing the format. **One schema, one aggregator, one GUI surface** across all loops.

---

## Goals

- **Live monitoring is the primary deliverable**, in _any_ real-time loop in the system. Offline analysis is secondary and opt-in.
- **Loop-agnostic infrastructure**: the tracer/logger/aggregator don't know whether they're inside teleop, record, or inference — only the instrumentation sites differ.
- **Layered model + a single end-to-end number per loop**. Per-stage breakdowns are how we localize problems; the E2E number is how we cross-check that the breakdown is complete. If `sum(stages) ≪ E2E`, we're missing a stage. (See [End-to-end latency](#end-to-end-latency-the-cross-check).)
- **Honest reporting**: the GUI overlay tags numbers as _measured_ vs _calibrated_. A cell labelled "total camera latency" must trace back to either a clock or a physical reference, never a guess.
- **Cross-loop comparability**: numbers from a teleop session and a policy-rollout session are directly comparable for shared stages (camera staleness, motor read, action send). Policy-specific stages (model forward, action-chunk dispatch) live in their own columns and are tagged with a `loop_kind` field.

### Overhead budget — what we actually mean

Two distinct concerns we keep separate:

|                         | Concern                                               | Target                                                                                                | Lever if exceeded                                    |
| ----------------------- | ----------------------------------------------------- | ----------------------------------------------------------------------------------------------------- | ---------------------------------------------------- |
| **Capture overhead**    | wall-clock cost added to the hot path by the tracer   | **< 100 µs / iteration** at the worst case (≈ 0.3% of a 30 Hz budget).                                | uniform sampling (1-in-N), or coarsen the stage list |
| **Reporting freshness** | how stale the numbers shown in the overlay/stderr are | **100 ms – 1 s is fine** (and even desirable — the human can't read updates faster than ~4 Hz anyway) | tighten the aggregator's flush interval if needed    |

These are independent. Capture overhead is on the producer; reporting freshness is on the consumer side of the queue. The GUI showing 1-second-old stats has zero impact on the teleop loop's iteration time.

**Measured cost on a 3+ GHz x86 box (Python 3.12)** — see `TestOverhead` in [test_latency_session.py](../../../../tests/utils/test_latency_session.py):

| Path                                                         | Per-iteration cost | % of 30 Hz budget |
| ------------------------------------------------------------ | ------------------ | ----------------- |
| `LatencySession.disabled()` (no-op overrides)                | ~0.7 µs            | 0.002%            |
| Enabled, empty iteration (`start_iter`/`end_iter` only)      | ~0.9 µs            | 0.003%            |
| Enabled, realistic shape (4 spans + 3 cameras + `set_field`) | ~11 µs             | 0.033%            |

Thresholds in `TestOverhead` are set ~6× above measured cost: tight enough to catch a regression, loose enough to tolerate slow CI hardware. The 1 Hz snapshot write and 1 Hz stderr summary are amortized — both gated by `time.time()` comparisons that cost ~100 ns when not yet due.

If `--latency-log` causes any measurable overrun in profiling, we add `--latency-sample` (e.g. 0.1 = 1-in-10) before optimizing the tracer further. Honest > clever.

## Non-goals (V1)

- Tuning servo PID or characterizing tracking dynamics (steady-state lag, overshoot). Those are _dynamics_, not latency.
- Operator/human-loop latency (visual processing, reaction time). Out of scope.
- **External calibration rig.** Measuring sensor exposure, USB transport, motor dead time via MCU + LED + IMU is deferred to V2. V1 ships only software-measurable numbers, with the un-measurable layers explicitly labelled in the latency stack as "H — calibration deferred to V2".
- Distributed-clock infrastructure (PTP, hardware-synced cameras). Even when calibration lands in V2, the MCU-master-clock pattern is sufficient for ms resolution.
- Replacing the existing `FPSTracker` in [async_inference/helpers.py](../../async_inference/helpers.py#L238-L263) — we _consume_ its output and unify it under the new schema, but don't duplicate it.

---

## Loops covered

| Loop                     | Entry point                                                         | What runs                                                            | Iteration rate                 | Notes                                                                                                         |
| ------------------------ | ------------------------------------------------------------------- | -------------------------------------------------------------------- | ------------------------------ | ------------------------------------------------------------------------------------------------------------- |
| **Teleop** (V1 target)   | [lerobot_teleoperate.py](../../scripts/lerobot_teleoperate.py)      | leader read → process → follower write                               | target FPS via `precise_sleep` | Closed loop; both leader and follower observable.                                                             |
| **Record** (V2)          | [lerobot_record.py](../../scripts/lerobot_record.py)                | observation → optional policy → action → write to dataset            | target FPS                     | Same as teleop or inference, plus dataset-write cost (a new stage).                                           |
| **Sync inference** (V2)  | record/eval loop with `--policy.path=...`                           | obs → `policy.select_action` → action                                | target FPS                     | Adds `inference_ms` (model forward) as a stage.                                                               |
| **Async inference** (V3) | [policy_server.py](../../async_inference/policy_server.py) + client | client batches obs → server forward → action chunk → client dispatch | server-side bursty             | Already has timestamps on the wire; we adopt them. Adds `network_ms`, `queue_ms`, `chunk_dispatch_ms` stages. |

The `loop_kind` field in every JSONL record names which one wrote it, so post-hoc analysis can filter and compare.

---

## Prior art in lerobot — what we reuse, complement, or replace

There is no centralized latency system today, but four independent timing utilities exist. The new design **builds on them rather than replacing them**.

| Existing                          | Where                                                                                                                          | What it does                                                                                                                                                    | Plan                                                                                                                                                                                                                                                |
| --------------------------------- | ------------------------------------------------------------------------------------------------------------------------------ | --------------------------------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **`TimerManager`**                | [utils/utils.py:308-406](../../utils/utils.py#L308-L406)                                                                       | Context manager + rolling history (`last`, `avg`, `percentile`, `fps_*`). Used in `benchmarks/video/`.                                                          | **Reuse as the building block** for `LatencyTracer.span()`. Almost exactly the API we need; no reason to write a parallel class.                                                                                                                    |
| **`_TimedBlock`**                 | [policies/hvla/s1_inference.py:76-115](../../policies/hvla/s1_inference.py#L76-L115)                                           | GPU-aware bracket: uses `torch.cuda.Event(enable_timing=True)` on CUDA, falls back to `perf_counter` on CPU. `read_ms()` requires a prior `cuda.synchronize()`. | **Promote out of HVLA** to `utils/latency/` for V2 inference instrumentation. CPU `perf_counter` around a CUDA forward pass measures launch latency, not forward time. Without this, `infer_forward_ms` is wrong by 10–50× depending on the policy. |
| **`LatencyTracker`**              | [policies/rtc/latency_tracker.py](../../policies/rtc/latency_tracker.py)                                                       | Sliding-window `deque` with `max()`, `percentile()`, `p95()`. RTC-specific consumer.                                                                            | **Reuse as the percentile primitive** inside `LatencyAggregator`. Tested ([tests/policies/rtc/test_latency_tracker.py](../../../tests/policies/rtc/test_latency_tracker.py)); battle-proven in RTC.                                                 |
| **`FPSTracker`**                  | [async_inference/helpers.py:237-262](../../async_inference/helpers.py#L237-L262)                                               | Dataclass; observation-rate over running time, used in `policy_server.py:192`.                                                                                  | **Feed, don't duplicate.** Already named in non-goals. The aggregator can derive the same number; FPSTracker keeps its existing role in async-inference logs.                                                                                       |
| **`chunk_compare.jsonl`**         | written from [s1_inference.py:481-564](../../policies/hvla/s1_inference.py#L481-L564)                                          | Per-inference JSONL with HVLA-specific stage breakdown.                                                                                                         | **Schema parent.** Our `latency.jsonl` adopts its conventions (`*_ms`, `t`, `step`). HVLA-specific stages stay in a sidecar JSONL keyed by `step` for join. (See [Open questions](#open-questions).)                                                |
| **`RLTMetrics` / `metrics.json`** | [policies/hvla/rlt/metrics.py](../../policies/hvla/rlt/metrics.py), read by GUI at [api/run.py:697](../../gui/api/run.py#L697) | Multi-rate metric groups (per-episode / per-inference / per-grad-update), thread-safe, JSON snapshot. GUI already consumes it.                                  | **Pattern reference for the GUI integration.** The new aggregator's GUI endpoint mirrors `get_rlt_metrics` shape — same JSON contract, different content.                                                                                           |
| Raw `perf_counter` scattered      | `lerobot_record.py`, `lerobot_teleoperate.py`, `policy_server.py`, `robot_client.py`, `gym_manipulator.py`, etc.               | Per-loop `dt_s` calculations, mostly logged-and-discarded.                                                                                                      | **Replace site-by-site** as we instrument each loop. The existing one-line stderr logs become aggregator output; numbers don't get lost.                                                                                                            |

### What's genuinely new

- **A loop-level `LatencyTracer` that owns one record per iteration** — the existing utilities are point timers; nothing in the repo today produces a per-iteration record with all stages keyed together.
- **Cross-loop schema** with `loop_kind` — we have HVLA-specific JSONL and RLT-specific JSON, but nothing that lets you compare teleop motor-read p95 to record motor-read p95 in the same query.
- **End-to-end + residual cross-check** — no existing utility tracks "did the breakdown account for everything."
- **Aggregator with bounded memory + GUI overlay** — the closest existing thing is RLT's `metrics.json` rewritten on disk, which won't scale to 60 Hz teleop.

### Implication for `infer_forward_ms` (V2)

Quietly wrapping `policy.select_action` in `perf_counter` brackets will return wrong numbers for any GPU policy. The HVLA team already solved this with `_TimedBlock`. **V2 inference instrumentation must use `_TimedBlock` (or equivalent), and call `cuda.synchronize()` before reading.** This is documented at the instrumentation site, not buried in implementation. The architecture section's `optional_span(tracer, "infer_forward")` resolves to a `_TimedBlock` when the underlying device is CUDA and a `TimerManager` otherwise.

---

## Latency stack — what each layer is, and how it's seen

Three signal paths intersect in one teleop iteration: **vision** (camera → policy/operator), **proprioception** (encoder → policy), and **action** (goal → motion). Each layer is tagged **S** (measurable in software), **H** (needs hardware reference), or **M** (mixed).

### Vision path

| #   | Stage                                    | Typical          | Tag | Notes                                                                                           |
| --- | ---------------------------------------- | ---------------- | --- | ----------------------------------------------------------------------------------------------- |
| V1  | Sensor exposure (midpoint)               | 4–30 ms          | H   | Driven by exposure setting; rolling shutter adds row-dependent skew.                            |
| V2  | Sensor readout + ISP                     | 5–15 ms          | H   | RealSense exposes `actual_ts` metadata; UVC does not.                                           |
| V3  | USB / MIPI transport                     | 1–10 ms          | H   |                                                                                                 |
| V4  | V4L2 / kernel buffer queue               | 0–N×period       | H   | Depends on driver buffering policy.                                                             |
| V5  | `cap.read()` returns to grab thread      | 0.5–2 ms         | S   | Wrap with `perf_counter`.                                                                       |
| V6  | Grab thread sets `latest_timestamp`      | <100 µs          | S   | Already happening: [camera_opencv.py:447-451](../../cameras/opencv/camera_opencv.py#L447-L451). |
| V7  | `read_latest()` consumes from cache      | 0–1 frame period | S   | **Staleness** = `now − latest_timestamp`.                                                       |
| V8  | Processor pipeline                       | <1 ms            | S   |                                                                                                 |
| V9  | Display / WS encode (operator path only) | 5–30 ms          | M   | Monitor refresh ≤ 16.7 ms at 60 Hz is H.                                                        |

### Proprioception path

| #   | Stage                                  | Typical        | Tag | Notes                                                     |
| --- | -------------------------------------- | -------------- | --- | --------------------------------------------------------- |
| P1  | Encoder ADC sample on servo MCU        | <1 ms          | H   | Bound by servo internal control period.                   |
| P2  | Servo register update                  | <1 ms          | H   |                                                           |
| P3  | `sync_read` packet TX over Feetech bus | 1–8 ms / chain | S   | Wrap [`bus.sync_read`](../../motors/motors_bus.py#L1124). |
| P4  | Servo TX response, host RX parse       | included in P3 | S   | Retries (default 5×) compound on failure.                 |

### Action path

| #   | Stage                                          | Typical            | Tag | Notes                                                                                  |
| --- | ---------------------------------------------- | ------------------ | --- | -------------------------------------------------------------------------------------- |
| A1  | `sync_write(Goal_Position)` packet TX          | 0.5–2 ms           | S   | Fire-and-forget on Feetech. Wrap [`bus.sync_write`](../../motors/motors_bus.py#L1217). |
| A2  | Servo MCU updates target register              | <1 ms (next cycle) | H   |                                                                                        |
| A3  | Onboard PID computes new PWM                   | servo cycle, ~5 ms | H   | STS3215 ≈ 5 ms; Dynamixel X ≈ 1–2 ms.                                                  |
| A4  | Motor torque builds, overcomes static friction | 5–30 ms            | H   | The "dead time" before any motion.                                                     |
| A5  | Joint accelerates toward goal                  | continuous         | —   | _Dynamics, not latency._ Out of scope.                                                 |
| A6  | Joint reaches goal                             | depends on Δ       | —   | _Trajectory, not latency._ Out of scope.                                               |

### What "command-to-actuation" means here

`sync_write` returns when the bus TX completes (A1). The user-relevant number is **emit-to-first-motion dead time** = A1 + A2 + A3 + A4. A1 is S; A2–A4 are H and require the calibration rig (accelerometer or external encoder). Cross-correlation of `goal_position` vs `present_position` is **rejected** as a latency estimator: PID + inertia bias it upward by tens of ms. We isolate dead time via onset-of-motion detection on the IMU stream.

### Inference path (record with policy / sync-inference / async-inference)

| #   | Stage                                                        | Typical        | Tag | Notes                                                                                                     |
| --- | ------------------------------------------------------------ | -------------- | --- | --------------------------------------------------------------------------------------------------------- |
| I1  | Observation pre-processing (resize, normalize, tensor build) | 0.5–5 ms       | S   | Wrap `policy_processor` pipeline.                                                                         |
| I2  | Host→GPU transfer                                            | 0.2–2 ms       | S   | Bracket explicitly; can dominate at high res.                                                             |
| I3  | Model forward                                                | 5–100 ms       | S   | `policy.select_action` body. Includes any internal denoise loops (e.g. flow-matching steps for π05/HVLA). |
| I4  | Action post-processing (denormalize, slice chunk)            | <1 ms          | S   |                                                                                                           |
| I5  | **Async only**: client→server network TX                     | 1–20 ms        | S   | Already measured in [policy_server.py:198](../../async_inference/policy_server.py#L198).                  |
| I6  | **Async only**: server queue wait                            | 0–chunk-period | S   | When server is saturated.                                                                                 |
| I7  | **Async only**: action-chunk dispatch jitter                 | 0–chunk-period | S   | Time from chunk arrival to first action consumed.                                                         |

I3 is the headline number for policy work and the one that varies most between checkpoints. Keep it in its own column; never bundle it into a "total inference" until reporting.

**GPU timing caveat**: I3 must be measured with CUDA events (`torch.cuda.Event(enable_timing=True)` + `synchronize` before reading), not raw `perf_counter`. CPU brackets around an async CUDA op time only the launch, not the kernel. We reuse [`_TimedBlock`](../../policies/hvla/s1_inference.py#L76-L115) for this — see [Prior art](#prior-art-in-lerobot--what-we-reuse-complement-or-replace).

---

## End-to-end latency — the cross-check

Per-stage breakdowns answer "where does the time go." A single E2E number answers "is the breakdown complete." We always emit both, and a sanity-check column for their difference.

### Definition for V1 (teleop)

`e2e_obs_to_action_ms` = `t_action_sent − t_input_oldest`

where:

- `t_action_sent` is `perf_counter` immediately after `bus.sync_write` returns.
- `t_input_oldest` is the **oldest** of the inputs the action was based on:
  - `min(cam.latest_timestamp for cam in cameras)` — the staleness of the freshest-decision-relevant input is bounded by the _most stale_ camera, not the freshest one.
  - `t_motor_read_done` — when proprioception was last refreshed.
  - For policy-in-the-loop: also clamp to `t_obs_handed_to_policy` — the policy can't see anything fresher than what it received.

Taking the _oldest_ is the operator-/policy-relevant number: it tells us the worst-case age of the world model the action committed to.

### Why this and not "from start of `get_observation`"

The alternative — "from the call to `read_latest()` to action TX" — collapses to `loop_dt_ms` minus the `precise_sleep`. It tells us what we already know. The interesting fact is that `cam.latest_timestamp` can be 30 ms older than the moment we read it, and that 30 ms is invisible to a `loop_dt_ms`-only view. `e2e_obs_to_action_ms` exposes it.

### The cross-check column

```
sum_of_stages_ms = (max_cam_stale_ms) + motor_read_ms + process_obs_ms
                 + infer_total_ms (if present)
                 + process_action_ms + action_send_ms

residual_ms = e2e_obs_to_action_ms − sum_of_stages_ms
```

`residual_ms` should be near zero. Persistent positive residual means there's a stage we're not measuring (queue waits, GIL contention, GC pauses); we add a span for it. Persistent negative residual means stages overlap (e.g. parallel work) — surprising in our current serial loops, would be worth investigating.

The aggregator surfaces `residual_p95_ms` as a top-line health metric. If it's > 1 ms, something in the loop is unattributed.

### Per-loop variants

| Loop                      | E2E definition                           | Notes                                                 |
| ------------------------- | ---------------------------------------- | ----------------------------------------------------- |
| Teleop                    | obs-oldest → action-sent                 | as above                                              |
| Record (no policy)        | obs-oldest → frame-written               | adds `dataset_write_ms` to sum                        |
| Record (with sync policy) | obs-oldest → action-sent                 | inference inside the iter                             |
| Async inference           | `obs_age_at_action_ms` — already defined | client-side: obs capture → action dispatched to robot |

The E2E definition shifts per loop, but the cross-check (sum-vs-E2E) generalizes.

---

## Live monitoring — what we capture every iteration

Per iteration, the tracer assembles one record and ingests it into the aggregator (always) and into the JSONL writer queue (if Tier 2 is on). The schema below is what shows up in both surfaces. Fields are grouped by which loops emit them; `loop_kind` in the record names the writer.

**Always present** (every loop):

- `loop_kind` ∈ {`teleop`, `record`, `sync_infer`, `async_infer_client`, `async_infer_server`}
- `t` — wall-clock, `step` — int, `ep` — int (or null)
- `loop_dt_ms` — total iteration time
- `e2e_obs_to_action_ms` — end-to-end latency (see [End-to-end latency](#end-to-end-latency-the-cross-check))
- `residual_ms` — `e2e − sum(stages)`, sanity-check column
- `overrun` — `loop_dt_ms > 1000/fps` (null when no fixed target)
- `outlier` — bool, present only when `true` (records flagged for "always-keep" under sampling)

**Sensing stages, software-measured (S)** — emitted by any loop that reads observations:

- `motor_read_ms` — wraps `_sync_read_with_motor_fallback("Present_Position")`
- `cam_<key>_stale_ms` — `t_consume − cam.latest_timestamp` per camera
- `cam_<key>_period_ms` — delta between successive `latest_timestamp` updates (rolling, computed in tracer)
- `process_obs_ms` — observation processor pipeline duration

**Action stages (S)** — emitted by any loop that sends actions:

- `action_send_ms` — wraps `bus.sync_write("Goal_Position")`
- `process_action_ms` — action processor pipeline duration

**Inference stages (S)** — emitted when a policy is in the loop:

- `infer_preproc_ms` (I1), `infer_h2d_ms` (I2), `infer_forward_ms` (I3), `infer_postproc_ms` (I4)
- `infer_total_ms` — sum of I1–I4

**Async-inference extras (S)** — emitted only by async loops:

- `net_tx_ms` (I5), `server_queue_ms` (I6), `chunk_dispatch_ms` (I7)
- `obs_age_at_action_ms` — `t_action_dispatch − t_obs_capture`. Single-number summary of total reactive latency.

**Record-loop extras (S)** — emitted only by `lerobot-record`:

- `dataset_write_ms` — frame buffer append + (occasional) parquet flush
- `video_encode_ms` — when video encoding is in the hot path

**Calibration-augmented (V2, deferred)** — appended by the logger, not the tracer, when `calibration.yaml` is present:

- `cam_<key>_total_est_ms` = `stale_ms + abs_latency_p50_ms`
- `c2a_total_est_ms` = `action_send_ms + dead_time_p50_ms`

These calibrated columns are explicitly tagged downstream so the GUI never blends them with measured-only fields. **Not present in V1** — V1 ships only software-measurable numbers and is honest about it.

**Optional, when running with the calibration rig attached (V2)**:

- `c2a_dead_time_ms` — direct IMU onset measurement, replaces the calibrated estimate
- `cam_<key>_abs_latency_ms` — direct LED-detection measurement

---

## Calibration — external sources of truth (V2)

Software covers V5–V8, P3, A1. Everything else needs an external clock and physical sensors. One MCU rig characterizes the rest. **Deferred to V2** — V1 ships with these layers labelled "H" in the latency stack and absent from JSONL/overlay.

### Apparatus

```
        ┌──────── ESP32 / Teensy ────────┐
        │   master clock, USB serial     │
        │   (round-trip sync, ~50 µs)    │
        └──┬───────────┬───────────┬─────┘
           │           │           │
        LED in      Photodiode    IMU on
        camera FOV  at end-eff.   end-effector
                                  (≥1 kHz, 3-axis)
```

Single MCU acts as master clock. Host periodically requests `t?`; MCU replies with its monotonic time; host caches the offset (halve RTT). Re-sync every few seconds; sub-ms accuracy on USB-serial is plenty.

### Procedures

**(a) Absolute camera latency** (V1–V7). MCU pulses the LED at logged `t_emit`. Camera grab thread runs a brightness-ROI detector; logs `t_seen` from `latest_timestamp`. `latency = t_seen − t_emit`. Run 200 pulses; histogram p50/p95. Decompose: pulse on/off both edges — the off→on edge includes exposure, the on→off edge does not, so `(rise + fall)/2 ≈` transport-only and `rise − transport ≈` exposure midpoint.

**(b) Motor dead time** (A1–A4). Send `Goal_Position` step at host-logged `t_cmd`; IMU streams at ≥1 kHz; first sample with `|a| > threshold` is `t_motion`. `dead_time = t_motion − t_cmd`. Sweep step sizes; small steps isolate dead time from PID dynamics. Per joint.

**(c) End-to-end vision-action loop** (V1 → A4). LED in camera FOV; policy/control loop hard-coded `if bright then Δ+ else Δ−`. MCU toggles LED, IMU records motion. `loop_latency = t_motion − t_LED_on`. This is the single number that matters for closed-loop policy stability.

(a) and (b) are the platform-characterization runs. (c) is the integration test after any hardware change.

### Calibration storage

```yaml
# calibration/cameras.yaml
top:
  abs_latency_ms_p50: 28.4
  abs_latency_ms_p95: 41.2
  measured_at: 2026-05-07
  conditions: { resolution: "640x480", fps: 30, exposure: "auto" }
wrist:
  abs_latency_ms_p50: 38.1
  ...

# calibration/motors.yaml
shoulder_pan:
  dead_time_ms_p50: 8.2
  dead_time_ms_p95: 14.0
  measured_at: 2026-05-07
  conditions: { baud: 1_000_000, return_delay: 0 }
```

Loaded at teleop start. Mismatch detection: if the camera negotiates a different resolution/FPS than the calibration's `conditions`, log a loud warning and _omit_ the calibrated total from the JSONL (don't lie).

### Recalibration triggers

- **Camera**: change cam, hub, USB cable, kernel, exposure mode, resolution, FPS. Auto-exposure is sneaky — in low light, exposure can balloon to 50 ms and dominate the budget.
- **Motor**: swap servo, change baud, change `Return_Delay_Time`, change end-effector load.
- **End-to-end**: after any of the above. Re-run (c).

---

## Architecture

Default mode: **no thread, no queue.** `tracer.commit()` is an inline call that does a deque append on the aggregator (~5 µs). A daemon writer thread is added _only_ when JSONL is enabled (Tier 2), and exists solely to absorb disk-write tail risk — not for the aggregator path.

```
Default (Tier 1 only) — no thread, no queue:
  Hot path
    │ tracer.span("motor_read"): bus.sync_read(...)
    │ tracer.span("action_send"): bus.sync_write(...)
    │ tracer.cam_consume(cam_key, latest_ts)   per camera
    │ tracer.commit(record)
    │   └─► aggregator.ingest(record)   inline, deque append (~5 µs)
    ▼
  Aggregator (queryable, lock-free reads)
    │ per-stage LatencyTracker (rolling deque, last 3–60 s)
    │ p50/p95/p99/max computed lazily on read
    ▼
  Live sinks (poll the aggregator at their own cadence)
    ├─ stderr summary       (1 Hz)
    └─ GUI overlay          (1–4 Hz, per-camera + dashboard)

When --latency-log adds Tier 2:
  tracer.commit(record)
    ├─► aggregator.ingest(record)       inline, as above
    └─► queue.put_nowait(record)        ~1 µs
                │
                ▼
              Writer thread (daemon, 10 Hz drain)
                drain queue → json.dumps batch → write to latency.jsonl
                rotate at --latency-log-max-mb; preserve outliers
```

**Why a thread for disk only**: a `write()` to a buffered file is normally µs, but the kernel can choose to flush dirty pages mid-call and stall for ms. On a 16.7 ms loop budget, that's an overrun. Aggregator ingest has no such tail risk; it stays inline. JSONL `json.dumps + write` could in principle stay inline too with a large block buffer, but the cost-of-being-wrong is loop overruns the user can't easily diagnose, and the cost-of-being-right (one daemon thread, ~50 lines) is trivial. Worth the asymmetry.

If queue ever overflows (writer thread stalled by something pathological), drop and increment a `dropped_records` counter on the aggregator; never block the producer. Surface `dropped_records` in the GUI overlay so the operator sees when this happens.

The aggregator is the **product**; JSONL is the audit trail when you want it.

### Components

- **`src/lerobot/utils/latency/tracer.py`** — `LatencyTracer`. Per-iteration record builder. `span()` context manager (delegates to `TimerManager` for CPU spans, `_TimedBlock` for GPU spans), `cam_consume()` helper. Zero-cost when disabled (`tracer is None` short-circuits at instrumentation sites).
- **`src/lerobot/utils/latency/jsonl_writer.py`** — `JSONLWriter`. Owns the queue, daemon writer thread, rotation, and JSONL file handles. Only constructed when `--latency-log` is set.
- **`src/lerobot/utils/latency/aggregator.py`** — `LatencyAggregator`. Rolling deque (wrapping the existing [`LatencyTracker`](../../policies/rtc/latency_tracker.py) primitive per stage). Read by stderr summary loop and GUI overlay.
- **`src/lerobot/utils/latency/timed_block.py`** — `_TimedBlock` promoted out of HVLA. GPU-aware bracket; CUDA events when device is CUDA, else `perf_counter`. (V2.)
- **`src/lerobot/utils/latency/calibration.py`** — load/save YAML; compute `*_total_est_ms` columns; emit warnings on config mismatch. (V2.)
- **`src/lerobot/utils/latency/analyzer.py`** — offline analysis. Mirrors [scripts/rlt_perf_audit.py](../../policies/hvla/scripts/rlt_perf_audit.py).
- **`scripts/latency/calibrate_camera.py`**, **`calibrate_motors.py`**, **`e2e_loop_test.py`** — one-shot calibration runs. All consume the same MCU protocol. (V2.)
- **`firmware/latency_mcu/`** — ESP32 sketch: master clock + LED driver + IMU streamer. (V2.)

### Storage of live data

The aggregator backs every live query — current numbers, time-series, histograms, overrun ratios — from a single data structure: a **bounded cyclic buffer** (`collections.deque(maxlen=N)`) of full records.

**Sizing**: `maxlen = 4096`. At 60 Hz that's ~68 s of history; at 30 Hz, ~136 s. Memory: 4096 × ~250 B ≈ 1 MB worst case. Bounded by record count, not time — so a slower loop sees a longer window automatically.

**Eviction**: pure FIFO. New record appends, oldest is evicted. No copies, no compaction.

**Query patterns**, all derived from this single deque:

| Query                             | Implementation                                             | Cost   |
| --------------------------------- | ---------------------------------------------------------- | ------ |
| Current p50/p95 of stage X        | `np.percentile([r[X] for r in deque if X in r], [50, 95])` | ~20 µs |
| Latency-over-time of X (last N s) | filter by `r["t"] > now − N`; project `(t, value)`         | ~50 µs |
| Histogram of X                    | bin the same projection                                    | ~50 µs |
| Overrun ratio in last N s         | `mean(r["overrun"] for r in window)`                       | ~10 µs |
| `dropped_records` counter         | not in the deque; lives as a scalar on the aggregator      | O(1)   |

Queries run on the consumer side at 1–4 Hz, so even pessimistic costs are negligible. No incremental percentile maintenance needed in V1; if profiling shows this dominates, we add per-stage `LatencyTracker` instances as a cache. Not before.

**The deque is unrelated to the JSONL queue.** When `--latency-log` is on, `tracer.commit()` does both an inline `aggregator.ingest(record)` AND a `queue.put_nowait(record)` for the writer thread. The two paths are independent: the queue exists solely to absorb disk-write tail risk; the deque exists for live queries. When `--latency-log` is off, no queue, no thread; the deque is the only consumer.

**What if I want a longer history than 68 s?** For V1, set a larger `maxlen` and pay the memory (e.g. `maxlen=110_000` for ~30 min at 60 Hz costs ~28 MB). For sessions longer than that, use Tier 2 (JSONL) and the offline analyzer; the live aggregator is not where multi-hour history belongs.

### Instrumentation sites

The tracer threads through `Robot.get_observation` and `Robot.send_action` as an optional kwarg (default `None`, zero cost when off):

```python
# src/lerobot/robots/so100_follower/so_follower.py
def get_observation(self, tracer: LatencyTracer | None = None) -> RobotObservation:
    with optional_span(tracer, "motor_read"):
        obs_dict = self._sync_read_with_motor_fallback("Present_Position")
    for cam_key, cam in self.cameras.items():
        frame, ts = cam.read_latest_with_timestamp()  # new method, returns both
        if tracer:
            tracer.cam_consume(cam_key, ts)
        obs_dict[cam_key] = frame
    return obs_dict
```

```python
# src/lerobot/scripts/lerobot_teleoperate.py — teleop_loop()
aggregator = LatencyAggregator() if cfg.latency_monitor else None
tracer = LatencyTracer(aggregator) if aggregator else None
writer = JSONLWriter(cfg.latency_log_path) if cfg.latency_log else None  # spawns daemon
while True:
    if tracer:
        tracer.start()
    obs = robot.get_observation(tracer=tracer)
    ...
    robot.send_action(action, tracer=tracer)
    if tracer:
        record = tracer.commit()           # inline aggregator update
        if writer:
            writer.put(record)             # only when JSONL is on
```

No call site changes when latency monitoring is off. Aggregator is the always-on cheap path; the writer thread is opt-in.

---

## Storage tiers

Live monitoring is the goal; persistence is opt-in. Three tiers, each independently togglable:

### Tier 1 — `LatencyAggregator` (always on when monitoring is enabled, in-memory)

The primary source of live numbers. Holds the last N seconds of records (configurable; default 3 s for overlay use, expandable to 60 s for "show me the last minute" diagnostics). Computes p50/p95/p99/max and overrun ratio lazily on read from per-stage `LatencyTracker` instances. **Bounded memory**: at 60 Hz × 60 s × ~250 B = ~900 KB worst case. **No threads, no disk traffic, no queue.** `tracer.commit()` calls `aggregator.ingest()` inline. Powers the GUI overlay and stderr summary.

When the user asks "is teleop currently slow?" — this is the only thing they need. No JSONL required, no thread spawned.

### Tier 2 — Rotated `latency.jsonl` (opt-in, default off)

Enabled with `--latency-log`. Spawns one daemon writer thread; aggregator stays inline. One record per iteration goes to both (inline ingest + queued write).

- **Rotation**: each file capped at `--latency-log-max-mb` (default 100 MB ≈ 25–35 min at 60 Hz × ~250 B/record). After cap, rotate to `latency.<n>.jsonl`. Keep the last `--latency-log-keep` files (default 5). Older are deleted.
- **Outlier preservation**: records with `loop_dt_ms > 1.5 × target_period` (or any flagged stage breach) are tagged `"outlier": true` and **always written**, even when sampling is on. The tail is what matters; we don't sample it away.
- **Uniform sampling** (off by default): `--latency-sample 0.1` writes 1-in-10 typical records (outliers still always written). For multi-day runs.
- **Bounded disk usage**: with defaults, max 500 MB on disk per session. Old sessions don't accumulate (they live under their `<run_id>/` and are cleaned with the run dir).

The default isn't "log forever." It's "log opt-in, rotated, sampled when long-running."

### Tier 3 — Snapshot-on-demand (V2)

Button in the GUI overlay: "Save next 30 s." Forces full-rate JSONL capture for a window, regardless of sampling. Useful for "I just saw a hiccup, capture it." Implemented as a temporary override of the sampling rate. Not in V1.

### Decision matrix

| Use case                                       | Tiers needed                         |
| ---------------------------------------------- | ------------------------------------ |
| Operator wants live overlay                    | 1                                    |
| 1-hour teleop session, want post-hoc histogram | 1 + 2 (defaults fine)                |
| Multi-day continuous recording                 | 1 + 2 (with `--latency-sample 0.05`) |
| "Capture the next anomaly"                     | 1 + 3 (V2)                           |
| Regression CI on a benchmark replay            | 1 + 2 (full, no sampling)            |

---

## JSONL record format

Schema mirrors [chunk_compare.jsonl](../../policies/hvla/s1_inference.py#L561-L562) conventions: `t` is wall-clock seconds, `step` is integer, all latencies are `*_ms` floats. Missing stages are simply absent from the record (sparse-by-omission); consumers must tolerate missing keys.

**Teleop iteration** (V1; no calibration columns since V1 is software-only):

```json
{
  "loop_kind": "teleop",
  "t": 1714932103.421,
  "step": 4823,
  "ep": 12,
  "loop_dt_ms": 19.4,
  "e2e_obs_to_action_ms": 49.1,
  "residual_ms": 0.6,
  "motor_read_ms": 6.2,
  "action_send_ms": 1.1,
  "process_obs_ms": 0.4,
  "process_action_ms": 0.3,
  "cam_top_stale_ms": 28.3,
  "cam_top_period_ms": 33.4,
  "cam_wrist_stale_ms": 41.7,
  "cam_wrist_period_ms": 33.5,
  "overrun": false
}
```

**Record loop with policy** (V2):

```json
{
  "loop_kind": "record",
  "t": 1714932200.118,
  "step": 871,
  "ep": 4,
  "loop_dt_ms": 33.1,
  "e2e_obs_to_action_ms": 53.4,
  "residual_ms": 0.2,
  "motor_read_ms": 6.4,
  "cam_top_stale_ms": 24.1,
  "cam_top_period_ms": 33.3,
  "infer_preproc_ms": 1.8,
  "infer_h2d_ms": 0.7,
  "infer_forward_ms": 18.2,
  "infer_postproc_ms": 0.4,
  "infer_total_ms": 21.1,
  "action_send_ms": 1.0,
  "dataset_write_ms": 0.6,
  "overrun": false
}
```

**Async inference, client side** (V3):

```json
{
  "loop_kind": "async_infer_client",
  "t": 1714932250.502,
  "step": 12044,
  "loop_dt_ms": 33.3,
  "e2e_obs_to_action_ms": 41.6,
  "residual_ms": 0.4,
  "cam_top_stale_ms": 22.1,
  "motor_read_ms": 5.9,
  "action_send_ms": 1.0,
  "net_tx_ms": 8.4,
  "server_queue_ms": 2.1,
  "chunk_dispatch_ms": 0.3
}
```

**Not in the schema (intentional)**: anything inferred from cross-correlation, anything from a calibration run that's not currently loaded, anything we can't trace to a specific stage. The schema is auditable.

**File layout**: `outputs/<loop_kind>/<run_id>/latency.jsonl` where `<run_id>` is `YYYY-MM-DD_HHMMSS`. Calibration files copied alongside (`outputs/<loop_kind>/<run_id>/calibration/`) so the trace is self-describing.

**Overhead bound** (Tier 1 only, default mode): `aggregator.ingest()` is a deque append per stage, ~5 µs total at 15 keys. **Tier 2 adds**: one `queue.put_nowait` (~1 µs); `json.dumps` runs on the writer thread, not the hot path. Total hot-path cost in Tier 2 is still ~6–10 µs. If the queue overflows, drop records and increment `dropped_records` on the aggregator; never block the producer.

---

## GUI overlay

Two surfaces, both reading from the same `LatencyAggregator`. Inspired by the RLT GUI's pattern ([api/run.py:697](../../gui/api/run.py#L697) → frontend rendering): a JSON snapshot endpoint, polled by the frontend, distributed across the existing layout rather than parked in one floating panel.

### 1. Per-camera corner overlay

Tiny semi-transparent badge in a corner of each camera view (top-right or bottom-right; consistent across cameras). Camera-specific only — the operator looks at a stream and immediately sees its health.

```
┌──────────────────────────┐
│ camera frame             │
│              ┌─────────┐ │
│              │  29.4 Hz│ │
│              │ stale 28│ │
│              └─────────┘ │
└──────────────────────────┘
```

Two numbers per camera:

- **Effective FPS** (`1000 / cam_<key>_period_ms` p50). Color-coded: green when within 10% of target, amber when 10–25% off, red when >25% off.
- **Stale ms** (`cam_<key>_stale_ms` p50). Color-coded against a per-camera budget (default: 1.5 × frame period).

V2 adds an optional second line for the calibrated total (`stale + abs_latency_p50`) when calibration is loaded, with a small distinguishing mark (🔵 measured · 🟡 calibrated, or any tagging pattern the existing GUI prefers).

### 2. Bottom dashboard (next to the output area)

Fixed-position strip below the camera grid, in the same band as the existing output/log area. System-wide health, not per-camera. Mirrors the RLT metrics dashboard in shape: each cell is **number + sparkline**, where the sparkline is a tiny ~80×20 px line of the last 30 s of that stage.

```
┌─ Loop Health ──────────────────────────────────────────────────────────────────────┐
│ loop      17.8 / 24.1 ms  ▁▂▁▂▂▃▂▁▁▂  │ e2e        45.2 / 58.0 ms  ▁▁▂▂▃▂▁▁▁▁    │
│ motor rd   6.0 ms         ▁▁▁▁▂▁▁▁▁▁  │ action snd  1.1 ms         ▁▁▁▁▁▁▁▁▁▁    │
│ residual p95 0.6 ms       ▁▁▁▁▁▁▁▁▁▁  │ overrun     2 / last 60    .....X.....   │
│ infer fwd  ─                          │ dropped recs 0              │ jsonl  off  │
└────────────────────────────────────────────────────────────────────────────────────┘
```

Numbers shown:

- **`loop`** p50/p95 — the iteration budget number.
- **`e2e`** p50/p95 — the cross-check number; the one operators will end up watching most.
- **`overrun`** count over the rolling window. Sparkline shows where in time the overruns happened.
- **`motor rd`**, **`action send`** p50.
- **`residual p95`** — silent until > ~1 ms, then loud (red). Tells us our breakdown is leaking time.
- **`infer fwd`** — present only when a policy is in the loop (V2). Else dash.
- **`dropped recs`** — increments when the JSONL queue overflows. Should be 0; if not, investigate.
- **`jsonl`** — `off` / `on` / `on (sampled 1:10)`. Cheap reminder of current logging mode.

Sparklines come from the same in-memory deque as the percentiles — no JSONL involvement, no extra storage. The dashboard polls one snapshot endpoint and the frontend renders both the numbers and the sparklines from the same payload.

### Click-to-expand: chart + histogram

Clicking any cell opens a panel with two views of that stage:

- **Latency-over-time line chart**, last ~60 s, drawn from the deque.
- **Histogram** of the same window, log-y so the tail is readable.

Together these answer "when did this start" and "is the p95 from a long tail or a bimodal." Both backed by the same deque; no JSONL involvement.

### What we are not building (in V1)

- **Per-iteration timeline / Gantt breakdown** ("iteration 4823 took 80 ms — what dominated?"). Useful, but to do it right we'd need per-stage _start_ timestamps on every record (currently we store only durations) — that doubles record size and adds tracer complexity for a use case that's better served post-hoc by the offline analyzer reading JSONL. Defer.
- **Flamegraph.** Our stages are flat — there's no nested call structure for a flamegraph to expose. Wrong shape; skip.

### Reporting cadence

Both surfaces poll at 1–4 Hz. The aggregator's snapshot endpoint is cheap (~50 µs to compute percentiles + project sparklines); the cost lives entirely on the consumer side and never touches the teleop loop. See [Overhead budget](#overhead-budget--what-we-actually-mean).

---

## Implementation phases

**V1 — teleop, software-only, live-first**

1. **`LatencyTracer` + `LatencyAggregator` (in-memory, no thread).** Built on `TimerManager` and `LatencyTracker` (see [Prior art](#prior-art-in-lerobot--what-we-reuse-complement-or-replace)). Self-test with a synthetic loop. Verify capture overhead < 1 ms / iter at 60 Hz; in practice we expect ~5–10 µs.
2. **Instrument `Robot.get_observation` / `send_action`** for software-measured stages (V5–V8, P3, A1) + camera staleness/period. Start with `SOFollower`; API is generic so other robots adopt it freely.
3. **Wire into teleop loop** ([lerobot_teleoperate.py](../../scripts/lerobot_teleoperate.py)). `loop_kind = "teleop"`. Compute `e2e_obs_to_action_ms` and `residual_ms`.
4. **Stderr 1 Hz summary** consumer of the aggregator. Smallest useful UX; works headless.
   5a. **GUI per-camera corner overlay**. Reads from aggregator via the same JSON-snapshot pattern as RLT metrics. Per-camera FPS + staleness, color-coded.
   5b. **GUI bottom dashboard**. Loop / e2e / motor / action / residual / overrun. Same snapshot endpoint.
5. **Optional `--latency-log` flag** (Tier 2). Adds the daemon JSONL writer thread + rotation. Off by default; aggregator stays inline either way.

**V2 — record, sync inference, external calibration** 7. **Wire into `lerobot-record`** ([lerobot_record.py](../../scripts/lerobot_record.py)). `loop_kind = "record"`. Add `dataset_write_ms` instrumentation around the frame buffer append. 8. **Inference instrumentation hooks**. Add `tracer.span("infer_*")` brackets inside `policy.select_action` (or at the call site). Generic across policy types — reuse HVLA's per-stage timing where present, route through the same tracer. 9. **Snapshot-on-demand (Tier 3)**. GUI button: "Save next 30 s of full-rate JSONL." Useful for capturing transient hiccups. 10. **Camera calibration rig** (a). MCU + LED + photodiode. `calibrate_camera.py` produces `cameras.yaml`. Loader adds `cam_<key>_total_est_ms` columns and tags GUI overlay. 11. **Motor calibration rig** (b). IMU streamer + `calibrate_motors.py` produces `motors.yaml`. Adds `c2a_total_est_ms`. 12. **End-to-end loop test** (c). One-shot script returning a single number for the full vision-action loop.

**V3 — async inference** 13. **Adopt async-inference timestamps**. The wire protocol already carries per-observation timestamps ([policy_server.py:198](../../async_inference/policy_server.py#L198)); route them through the tracer rather than logging in two places. 14. **Client + server sides emit JSONL independently**. `loop_kind = "async_infer_client"` / `"async_infer_server"`. Joinable post-hoc via `step` + `t`. 15. **Async `e2e_obs_to_action_ms`** captures the cumulative effect of net + queue + dispatch.

**Continuous** 16. **Offline analyzer**. Mirrors [rlt_perf_audit.py](../../policies/hvla/scripts/rlt_perf_audit.py). Histograms, p95, budget violations, per-stage breakdown. Filters by `loop_kind`. Cross-loop comparison (e.g., teleop vs record motor-read p95 — should match if the hardware is unchanged).

**V1 phases 1–5b are the minimum useful product**: live per-camera overlay + bottom dashboard + stderr summary + per-stage breakdown + E2E sanity check. Phase 6 (JSONL) only matters if someone wants offline analysis. Calibration is V2; we ship V1 honestly labelled as "software-measured only."

---

## Post-V1 additions

### `LatencySession` — the per-loop helper

The first time `lerobot-record` was instrumented, it became clear that every loop wires the same six lines: build aggregator, build tracer, build snapshot writer, throttle the stderr summary, commit on every iteration, swallow errors so the next iteration starts clean. `LatencySession` ([utils/latency/session.py](../../utils/latency/session.py)) bundles all of that:

```python
session = LatencySession.from_config(
    enabled=cfg.latency_monitor,
    loop_kind="teleop",
    target_fps=float(cfg.fps),
    output_dir=cfg.latency_output_dir if cfg.latency_monitor else None,
)
# In the loop body:
with session.iteration():
    with session.span("get_observation"):
        obs = robot.get_observation()
    session.cam_consume_all(getattr(robot, "cameras", None))
    with session.span("process_obs"):
        ...
    with session.span("action_send"):
        robot.send_action(...)
```

`LatencySession.disabled()` returns a no-op session whose methods all short-circuit; call sites have no `if tracer is None` branches. For loops with branchy `continue` paths that can't fit in a single `with iteration():` block (e.g. `record_loop`), use the explicit `start_iter()` / `end_iter()` pair — `continue`-skipped iterations are simply discarded and the next `start_iter()` resets cleanly.

### HVLA inference profiling

The `InferenceThread` ([policies/hvla/s1_inference.py](../../policies/hvla/s1_inference.py)) takes an optional `latency_session` and stamps the four GPU stages — `enc_obs`, `rl_tok`, `s1_denoise`, `actor` — sequentially from `t_infer_start`. CUDA-event durations from the existing `_TimedBlock` timers are read after the end-of-inference `cuda.synchronize()`, then converted to wall-clock spans. The Gantt visual shows them in execution order; cumulative durations match `(t_infer_end - t_infer_start)` modulo dispatch overhead. The `obs_stale_ms` field carries the obs→infer-start delay, and `inference_ms` / `total_delay_ms` are the umbrella scalars.

Enable via `--latency-monitor` on `lerobot.policies.hvla.launch`, or unconditionally when launched from the GUI. Snapshot lands in `outputs/hvla_runs/latency_snapshot.json` so it doesn't collide with the teleop snapshot.

### Multi-source GUI endpoint

Multiple loops can run concurrently (e.g. teleop + HVLA inference, when running HVLA with a teleop intervention path). The dashboard has to display them side-by-side, so the snapshot endpoint takes a `source` query param:

- `GET /api/run/latency-metrics?source=teleop` — teleop snapshot (default; back-compat for existing callers)
- `GET /api/run/latency-metrics?source=hvla_infer` — HVLA inference snapshot
- `GET /api/run/latency-sources` — lists all known sources with `{key, loop_kind, fresh, age_s}`. Snapshots older than 5 s are reported with `fresh=False` so the dashboard can grey them out instead of dropping them.

`LATENCY_SOURCES` in [api/run.py](../../gui/api/run.py) is the registry; adding a new loop kind means appending to the dict and (if the script's CLI doesn't yet expose `--latency-output-dir`) plumbing it through.

### Persistent per-episode quality metadata (record loop)

The live `LoopHealthDetector` fires WARN logs during recording but the
operator may not be looking. After the recording finishes, the dataset
on disk should answer two questions on its own:

- _"Did this run record cleanly overall?"_ → `meta/recording_health.json`
- _"Which episodes were bad?"_ → `meta/episodes_health.jsonl` (one JSON
  line per saved episode)

Both files live inside the dataset directory (`<dataset.root>/meta/`) so
they travel with the data — sync the dataset to the Hub, the quality
verdict goes with it. The data panel in the GUI reads these files to
render red/yellow badges next to bad episodes without having to
re-derive anything.

Schemas:

```json
// meta/episodes_health.jsonl  (one line per save_episode)
{
  "episode_index": 5,
  "n_records": 600,
  "duration_s": 20.0,
  "target_period_ms": 33.33,
  "overrun_ratio": 0.12,
  "loop_dt_ms": {"p50": 31.0, "p95": 42.0, "p99": 55.0, "max": 67.0},
  "cameras": {"front": {"stale_p50": 14, "stale_p95": 22, "period_p50": 33.3, "fps_effective": 30.0}, ...},
  "healthy": false,
  "issues": [{"rule": "overrun_high", "message": "12.0% of frames over the 33.3 ms budget (threshold 5%)."}]
}
```

```json
// meta/recording_health.json  (one file per recording session)
{
  "loop_kind": "record",
  "n_records": 18000,
  "duration_s": 600.5,
  "target_period_ms": 33.33,
  "overrun_ratio": 0.03,
  ...
  "healthy": true,
  "issues": []
}
```

The per-episode verdict thresholds are intentionally stricter than the
live `LoopHealthDetector`'s (5% vs 10% overrun) because the post-hoc
check covers the whole episode while the live one watches a rolling
window. An episode with one bad second out of sixty may not have fired
the live warning (the window cleared) but is still arguably unhealthy
for training — so the persistent file flags it for review. See
`DEFAULT_VERDICT_THRESHOLDS` in [recording_health.py](../../utils/latency/recording_health.py).

Tagging is required for per-episode breakdown: `record_loop` calls
`latency_session.set_field("ep", episode_index)` at the top of every
iteration. Reset and setup phases pass `episode_index=None`, so those
records carry no `ep` tag and are excluded from any specific episode's
verdict.

### Thread-safety contract

The latency package contains **zero locks**. Thread safety is achieved by partitioning state per-thread rather than synchronising shared state. Two principles keep this working; both are easy to break by accident, so they're written down here:

**1. One session per thread. Sessions never cross thread boundaries.**

Each `LatencySession` owns its own `LatencyTracer`, `LatencyAggregator`, `LatencySnapshotWriter`, and `LoopHealthDetector`. The owning thread is whichever one calls `start_iter` / `end_iter` / `span` etc. Multi-thread loops (the canonical example is HVLA's control thread + inference thread) construct **two distinct sessions** with two distinct aggregators and two distinct snapshot files — never one session shared across both threads.

Concretely in HVLA: `s1_process.run_s1` constructs `main_session` (used only by the control thread) and `inference_session` (passed into `InferenceThread`, used only by the daemon thread inside `_loop`). Each thread does `from_config(...)` ⇒ gets a fresh `LatencyAggregator` ⇒ owns it for the rest of its life.

> ⚠️ **Don't do this**: constructing two `LatencySession`s that share one aggregator (e.g. by passing the same `LatencyAggregator` instance to two `LatencySession.__init__` calls). The public API (`from_config`) prevents this — but the constructor is permissive. If a future refactor exposes shared aggregators, the deque's append/iteration race becomes real and we'd need locking.

**2. `current_span` is per-thread by construction.**

The thread-local that `start_iter` writes (`_thread_session.session = self`) and `end_iter` clears uses Python's `threading.local()`. Each thread sees its own attribute storage — the inference thread cannot accidentally observe the control thread's "current session," and vice versa. No coordination needed.

> ⚠️ **Don't do this**: calling `current_span("...")` from a coroutine that yielded mid-iteration. Coroutines share an OS thread but switch context arbitrarily, so the "current session" registered at `start_iter` could be the wrong one by the time the coroutine resumes. We don't use asyncio/gevent inside any loop body today; if that ever changes, switch to `contextvars` instead of `threading.local`.

**3. Cross-process IPC uses atomic `os.replace`.**

The snapshot file is the _only_ place where the loop process and the GUI process exchange data. `LatencySnapshotWriter._write` writes to `latency_snapshot.json.tmp` then atomically renames over `latency_snapshot.json`. POSIX guarantees the GUI reader sees either the complete old file or the complete new file, never a partial — and the GUI also catches and swallows any read exception as belt-and-suspenders. So even though producer and consumer are on different processes (and therefore different threads), they never coordinate explicitly.

> ⚠️ **Don't do this**: writing to the snapshot file directly without going through `LatencySnapshotWriter._write`. Skipping the tmp+rename dance reintroduces the partial-read race.

**Why partition over lock?**

The hot path runs at 30–60 Hz with sub-µs budget for monitoring overhead. Even an uncontended `threading.Lock` acquire is ~1 µs; a contended one is much worse. By making sessions thread-private, the hot path is pure attribute access (`time.perf_counter()`, dict writes, deque append). Measured cost: ~11 µs/iter at the realistic shape, including the detector throttle ticks — 0.03% of a 30 Hz budget. See `TestOverhead` in [test_latency_session.py](../../../../tests/utils/test_latency_session.py).

**What this means for new loops:**

When adding `LatencySession` to a new loop, the rules are:

- Call `LatencySession.from_config(...)` once, in the thread that will own the loop. Do not construct manually.
- Drive `start_iter` / `end_iter` from that same thread. Do not hand the session off.
- For multi-thread setups (background inference, async writer), each thread that needs latency data gets its own `from_config(...)` call, its own `loop_kind`, and (for multi-thread processes) a distinct `track` field on the snapshot envelope.
- Read snapshots from the GUI process via `/api/run/latency-metrics?source=...` — never read the file directly from a different thread of the producer process.

---

## Open questions

- **GUI transport**: the per-camera corner overlay and bottom dashboard both want a JSON snapshot of the aggregator at 1–4 Hz. Three options: (a) extend `ObservationStream` shared-mem with a `latency` block (matches the existing live-frame plumbing); (b) add a polled HTTP endpoint following the RLT `get_rlt_metrics` pattern at [api/run.py:697](../../gui/api/run.py#L697) (simplest, already a known pattern); (c) WebSocket push (lowest latency, most code). Default leaning: **(b)** — RLT already proves the pattern works, both surfaces poll the same endpoint, no new transport machinery.
- **Per-camera grab thread instrumentation**: should the grab thread itself log `cap.read()` durations, or is consume-side staleness enough? Logging in the grab thread tells us whether a slow camera is the cause of high staleness — useful — but adds another timing site to maintain. Default leaning: defer; staleness + period jitter together usually localize the problem.
- **Onset-of-motion estimator without rig**: a passive analyzer that detects leader-rest→motion transitions and follower-rest→motion transitions during normal teleop, computes the lag distribution. Cheap; no hardware. Useful as a fallback when the IMU rig isn't attached. Worth adding to the analyzer (phase 9)? Default leaning: yes, but as a _cross-check_, not a substitute for the rig.
- **Sync vs async inference unification**: async-inference already has its own logging discipline ([policy_server.py](../../async_inference/policy_server.py)). The plan is to _route_ its existing timestamps through `LatencyLogger` rather than keep two parallel pipelines. Risk: if the protocol evolves, we maintain two writers. Default leaning: tolerate it — the JSONL is the contract; how data gets there is fungible.
- **Per-policy infer-stage breakdown**: HVLA decomposes inference into `enc_obs_ms`, `rl_tok_ms`, `s1_denoise_ms`, etc. ([s1_inference.py:561-562](../../policies/hvla/s1_inference.py#L561-L562)). Should those policy-specific stages live in `latency.jsonl` directly, or in a sidecar `infer.jsonl`? Default leaning: keep `infer_forward_ms` as the unified stage in the main JSONL; let policies emit their own sidecar with the same `step` for join.
- **Record loop "inference inside or outside the iter"**: when `lerobot-record` runs with a policy, `infer_forward_ms` is inside the iteration's `loop_dt_ms`. When it runs in async mode, it's not. The schema handles this naturally (sparse keys), but the audit script needs to know not to compare across modes. Add `loop_kind` filtering to the analyzer from day one.
- **Doc location**: this lives under `gui/docs/` to match `feature_editing.md`, but the feature is broader than the GUI. Move to `src/lerobot/utils/latency/docs/` once that directory is created? Default leaning: yes, when V1 phase 1 lands.

---

## Glossary

- **Latency**: dead time between cause and observable effect. A scalar; measurable.
- **Dynamics / tracking lag**: time-varying error during continuous motion. A trajectory; not what we measure.
- **Staleness**: `now − latest_timestamp` at the moment of consumption. The age of the data the consumer sees.
- **Period jitter**: variance in the interval between successive frames at the source. Diagnoses dropped/late frames upstream of consumption.
- **S / H / M tags**: software-measurable / needs-hardware-reference / mixed.
- **Measured vs calibrated**: live numbers from the running system / numbers derived from a one-shot calibration run, applied as a constant.

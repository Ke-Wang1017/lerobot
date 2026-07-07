# Quest VR teleop — TODO

Follow-ups not blocking the initial landing. Captured here so they don't
get lost.

## Unimanual robots: which Quest controller drives the one arm

Quest is hardware-bimanual (one immersive session always streams both
controllers' poses), and we lean into that: a single config type,
`quest_vr`, emits `left_*` / `right_*` action keys and pairs naturally
with a bimanual robot whose `attach_teleop` splits the action dict
per-arm and installs IK on each half.

A single-arm robot (e.g. a unimanual SO-101 / SO-107 follower) wants the
opposite: drop one controller's keys and feed the other to its single
Cartesian-IK controller. The question is **which controller**, and
**where the choice lives**.

Sketches to consider when this actually has a consumer:

- **Config field on the single-arm follower's `attach_teleop`.** The
  follower decides which prefix to consume (`left_target_x` vs
  `right_target_x`), strips the prefix, and feeds the unprefixed dict
  into its `CartesianIKController`. Keeps the teleop unaware of arm
  count; keeps the mapping where the arm lives.
- **Config field on `QuestVRTeleopConfig`** (e.g. `controller_hand:
Literal["left", "right", "both"]`). The teleop filters its emitted
  keys to just the chosen half (unprefixing in the `"left" | "right"`
  case). Single-arm robots get a clean unprefixed dict. Bimanual robots
  use `"both"` (the current default behavior).
- **Wrapper teleop class.** Thin `UniHandQuestVRTeleop` that wraps the
  bimanual instance and forwards a deprefixed action. More code, but
  the bimanual path stays untouched.

No single-arm SO-\* follower currently has a Cartesian `attach_teleop`
branch, so this is genuinely deferred work — the decision is cleanest
when there's an actual single-arm consumer to fit it to. Until then,
the Quest VR + bimanual SO-107 path is the only configuration that
exists.

Related — the action-feature shape that a follower's `attach_teleop`
matches against currently checks for `left_target_x` AND `right_target_x`
([cartesian_ik.py: `is_so107_bimanual_cartesian_teleop`](cartesian_ik.py)).
A unimanual variant will need a sibling detector or a broader contract.

## Tracking-lost / re-acquired haptic

The page pulses on clutch edges and IK-hold transitions; the only
event left from the original design sketch is a controller-tracking
dropout. Tricky UX:

- The lost controller can't pulse (its actuator is gone with its
  pose). The remaining one would have to signal on the lost
  controller's behalf — e.g. a two-pulse pattern on the right
  controller meaning "your LEFT just dropped out." Pattern isn't
  obvious to the operator without prior training.
- Re-acquired is easier: the regained controller fires its own
  pulse. Trivial extension if we add the lost-signal path.

Worth designing before shipping. The plumbing for it (server→page WS
message on transition, per-handedness page-side handler) already
exists from the IK-hold path — adding a `{type: "tracking_lost",
hand: "left"}` message is mostly an extension of the existing
`ws.onmessage` branches, ~15 lines.

## Jitter diagnostics: server-side WS metrics + Quest-side perf reporting

On 2026-06-03 a real-hardware run on bi_so107 showed visible operator
jitter ramping up over a 5 m 37 s session. The server-side teleop loop
held 30 Hz throughout (p50 33.4 ms, p90 34.7 ms over the whole run),
yet the operator perceived it as 8 fps-ish toward the end. None of our
existing metrics can tell us whether the jitter source was:

- the Quest browser thermal-throttling its WebXR `requestAnimationFrame`
  loop (sustained-XR + camera-passthrough on Quest 3 commonly throttles
  to ~12–15 Hz after a few minutes),
- the camera-to-headset WebRTC stream stuttering, or
- the IK / motor stack falling behind a fast-moving target (state-vs-
  intent lag p95 grew ~25 → ~68 across active control windows in that
  run, so it's _also_ on the table).

Two cheap additions would distinguish these. Both should land together
since either alone leaves the diagnosis ambiguous:

**P0 — server-side, ~30 lines in `server.py`:**

- Track WS pose-message arrival times in a 1024-deep ring per
  controller. Surface `quest_pose_period_ms` as a new latency-monitor
  stage (`p50 / p95 / p99`). If Quest throttles from 72 → 12 Hz, p50
  jumps from ~14 ms to ~80 ms — direct, unambiguous signal.
- Include a monotonic `seq` field in the JS side's pose message;
  server logs `seq - last_seq`. Gaps > 1 distinguish a _dropped_
  message from a _delayed_ one.
- Add a `clutch_engaged` boolean to the motion logger's per-tick
  record. Today, "intent didn't change for 400 ticks" is ambiguous
  between "Quest stopped streaming" and "operator released clutch".

**P1 — Quest-side, ~20 lines in the page JS:**

- Sample `requestAnimationFrame` cadence in the WebXR session loop.
  Every 1 s, POST `{type: "perf", raf_p50_ms, raf_p95_ms, sent_at}`
  back over the existing WS. The Quest itself knows its render rate;
  we just don't ask. Logged server-side alongside the WS-arrival
  histogram, this is the direct thermal-throttle indicator.

**P2 — display-side, deferred unless P0+P1 prove insufficient:**

- WebRTC `RTCPeerConnection.getStats()` on the Quest browser side —
  `framesPerSecond`, `framesDropped`, `jitterBufferDelay` per camera
  stream. Catches stutter in the camera-back-to-headset loopback
  independently of pose-input rate. Useful if WS arrival looks clean
  but the operator still reports jitter.
- Server-issued WS ping every 1 s; Quest echoes immediately; server
  logs RTT. Catches network-induced jitter.

Land P0+P1 together; rerun the same 5-minute hardware session; the
resulting `latency_snapshot.json` will conclusively name the bottleneck.

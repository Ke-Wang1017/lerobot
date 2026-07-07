#!/usr/bin/env python

# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Latency monitoring for real-time loops (teleop, record, inference).

See ``src/lerobot/gui/docs/latency_monitoring.md`` for the design.
"""

from contextlib import nullcontext
from typing import Any

from lerobot.utils.latency.aggregator import LatencyAggregator
from lerobot.utils.latency.snapshot import LatencySnapshotWriter
from lerobot.utils.latency.tracer import LatencyTracer

# Note: LatencySession imported lazily below to avoid a circular import
# (session.py imports format_latency_summary from this module).


def maybe_span(tracer: LatencyTracer | None, name: str):
    """Context manager around a tracer span; nullcontext when ``tracer`` is None.

    Lets call sites use the same ``with maybe_span(tracer, "name"):``
    pattern whether monitoring is enabled or not — the cost when off is
    a nullcontext enter/exit, ~50 ns.
    """
    return tracer.span(name) if tracer is not None else nullcontext()


def attach_pipeline_step_spans(pipeline: Any, latency_session: Any, prefix: str = "obs") -> None:
    """Attach per-step timing hooks to a ``RobotProcessorPipeline``.

    For each step in ``pipeline.steps``, record a child span under the
    active iteration named ``f"{prefix}_{step.__class__.__name__}"``.
    Lets us attribute the umbrella ``process_obs`` cost back to specific
    processor steps (depth-edge overlay, obs-stream writer, etc.) without
    modifying the pipeline class.

    No-op when ``latency_session`` is disabled — hooks aren't attached at
    all, so there's zero per-step overhead in the default off case.
    """
    if not getattr(latency_session, "enabled", False):
        return
    import time as _time

    starts: list[float] = []

    def _before(idx: int, transition: Any) -> None:
        starts.append(_time.perf_counter())

    def _after(idx: int, transition: Any) -> None:
        if not starts:
            return  # defensive — shouldn't happen unless hooks desync
        start = starts.pop()
        step_name = type(pipeline.steps[idx]).__name__
        latency_session.add_span(f"{prefix}_{step_name}", start, _time.perf_counter())

    pipeline.before_step_hooks.append(_before)
    pipeline.after_step_hooks.append(_after)


# Stages we know how to render in the stderr digest. Order is the
# preferred display order; only stages actually present in a snapshot
# are emitted, so this works for teleop, record, and HVLA inference
# without per-loop branching.
_DIGEST_STAGES: tuple[tuple[str, str], ...] = (
    # Teleop / record stages
    ("get_observation_ms", "obs"),
    ("process_obs_ms", "p_obs"),
    ("process_action_ms", "p_act"),
    ("inference_ms", "infer"),
    ("action_send_ms", "send"),
    ("dataset_write_ms", "dswr"),
    # HVLA inference-thread stages (none of these collide with the teleop
    # set; if a loop_kind doesn't emit them they're silently skipped).
    ("batch_prep_ms", "prep"),
    ("enc_obs_ms", "enc"),
    ("rl_tok_ms", "rl_tok"),
    ("s1_denoise_ms", "denoise"),
    ("actor_ms", "actor"),
)


def format_latency_summary(snap: dict[str, Any]) -> str:
    """One-line digest of an aggregator snapshot for stderr at 1 Hz.

    Generic across loop kinds: prints loop p50/p95, then each known stage
    that's present in the snapshot, then per-camera staleness, then
    overrun%. Stages absent from the snapshot are skipped silently.
    """
    stages = snap.get("stages", {})

    parts: list[str] = []
    loop = stages.get("loop_dt_ms")
    if loop:
        parts.append(f"loop {loop.get('p50', 0):.1f}/{loop.get('p95', 0):.1f}ms")
    for key, label in _DIGEST_STAGES:
        s = stages.get(key)
        if s:
            parts.append(f"{label} {s.get('p50', 0):.1f}ms")
    cam_keys = sorted(k for k in stages if k.startswith("cam_") and k.endswith("_stale_ms"))
    for k in cam_keys:
        cam_name = k[len("cam_") : -len("_stale_ms")]
        parts.append(f"{cam_name} stale {stages[k].get('p50', 0):.0f}ms")
    parts.append(f"overrun {snap.get('overrun_ratio', 0) * 100:.0f}%")
    return " · ".join(parts)


def __getattr__(name: str):
    """Lazy attribute resolution to break the session.py ↔ __init__.py circular import."""
    if name == "LatencySession":
        from lerobot.utils.latency.session import LatencySession

        return LatencySession
    if name == "current_span":
        from lerobot.utils.latency.session import current_span

        return current_span
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "LatencyAggregator",
    "LatencySession",
    "LatencySnapshotWriter",
    "LatencyTracer",
    "attach_pipeline_step_spans",
    "current_span",
    "format_latency_summary",
    "maybe_span",
]

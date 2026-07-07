# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""Parse a real ``lerobot-train`` stdout stream into progress + metrics.

This is the single source of training signal for every backend (local
subprocess, SSH, future HF Jobs): the orchestrator runs these pure
functions over whatever log it can read (``stderr.log`` for local/SSH,
``fetch_job_logs`` for cloud) and writes the results to ``progress.json``
(position) and ``metrics.jsonl`` (training-signal series). Nothing here
touches the filesystem or the network — it's line-in, struct-out.

Two distinct concerns, two parsers (see ``DESIGN.md`` § Polling):

* **Progress** — position only (step / total / ETA), from the tqdm bar
  lerobot prints ~1/s. Latest-wins.
* **Metrics** — the training-signal line lerobot prints every ``log_freq``
  steps (``... step:N loss:X grdn:Y lr:Z ...``). Auto-captured: *every*
  numeric ``key:value`` becomes a field, so new / policy-specific metrics
  need no code change. ``step`` is the series x-axis; curation (which keys
  to chart by default, axis scaling) happens at display time, not here.

The tqdm + metric-line shapes are also what HF's LeLab parses; the regexes
below are adapted from its ``parse_metrics_into`` (Apache-2.0).
"""

from __future__ import annotations

import re
from dataclasses import dataclass

# tqdm progress bar, e.g.
#   "Training:   1%|▏         | 125/10000 [02:02<2:36:10,  1.05step/s]"
# Groups: current step, total steps, ETA (the time after '<').
_TQDM_RE = re.compile(r"Training:\s*\d+%[^|]*\|[^|]*\|\s*(\d+)\s*/\s*(\d+)\s*\[(?:[\d:]+)<([\d:?]+)")

# A numeric ``key:value`` token. Suffix set matches lerobot's
# ``format_big_number`` (K/M/B/T/Q — B is billion, e.g. ``step:10K``). The two
# guards reject noise from the logging prefix + tqdm bar that real output glues
# onto the metric line: ``(?<!\.)`` drops ``file.py:611``; ``(?=\s|$)`` drops
# ``Training: 39%`` and tqdm ``00:17`` times.
_KV_RE = re.compile(
    r"(?<!\.)\b([A-Za-z_][A-Za-z0-9_]*)\s*:\s*(-?\d[\d,]*\.?\d*(?:[eE][+-]?\d+)?)([KMBTQ])?(?=\s|$)"
)

_MAGNITUDE = {"K": 1e3, "M": 1e6, "B": 1e9, "T": 1e12, "Q": 1e15}

# The metric line is identified by carrying a loss — avoids treating arbitrary
# "word:number" lines (timestamps, URLs) as training samples.
_METRIC_GATE = "loss:"


@dataclass(frozen=True)
class ProgressSample:
    """One position reading. ``total_steps``/``eta_seconds`` are None until the
    tqdm bar has printed them (it omits ETA on the very first ticks)."""

    step: int
    total_steps: int | None
    eta_seconds: float | None


def parse_progress(line: str) -> ProgressSample | None:
    """Return a :class:`ProgressSample` if ``line`` is a tqdm progress bar, else None.

    Pre: ``line`` is a single stdout line (trailing newline ok).
    Post: on match, ``step >= 0``; ``total_steps`` is None or ``> 0``;
    ``eta_seconds`` is None or ``>= 0``. Never raises on malformed input.
    """
    m = _TQDM_RE.search(line)
    if m is None:
        return None
    step = int(m.group(1))
    total = int(m.group(2))
    eta = _parse_duration(m.group(3))
    return ProgressSample(
        step=step,
        total_steps=total if total > 0 else None,
        eta_seconds=eta,
    )


def parse_metric_sample(line: str) -> dict[str, float] | None:
    """Auto-capture every numeric ``key:value`` from a lerobot metric line.

    Returns a flat ``{key: float}`` bag (e.g. ``{"step": 1000.0, "loss":
    0.043, "lr": 1e-05, "grdn": 1.2}``) or None if ``line`` isn't a metric
    line. The bag is deliberately uncurated — keep every numeric field so a
    new metric is chartable without code changes; the UI decides what to show.

    Pre: ``line`` is a single stdout line.
    Post: on a non-None return the bag is non-empty and contains ``"loss"``;
    all values are finite floats. Never raises on malformed input.
    """
    if _METRIC_GATE not in line:
        return None
    bag: dict[str, float] = {}
    for key, num, suffix in _KV_RE.findall(line):
        val = _to_float(num, suffix)
        if val is not None:
            bag[key] = val
    # Gate guarantees a 'loss:' token, but it may have been non-numeric
    # (e.g. 'loss:nan' as a literal) — only return a sample we can chart.
    if "loss" not in bag:
        return None
    return bag


def _to_float(num: str, suffix: str) -> float | None:
    try:
        val = float(num.replace(",", ""))
    except ValueError:
        return None
    if suffix:
        val *= _MAGNITUDE[suffix]
    return val


def _parse_duration(s: str) -> float | None:
    """tqdm ``MM:SS`` / ``HH:MM:SS`` → seconds. None for the ``?`` placeholder
    tqdm prints before it can estimate."""
    if "?" in s:
        return None
    parts = s.split(":")
    try:
        nums = [int(p) for p in parts]
    except ValueError:
        return None
    if len(parts) == 2:
        return nums[0] * 60 + nums[1]
    if len(parts) == 3:
        return nums[0] * 3600 + nums[1] * 60 + nums[2]
    return None

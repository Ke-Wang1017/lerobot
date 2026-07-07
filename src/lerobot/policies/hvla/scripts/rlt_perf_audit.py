"""Per-stage inference timing audit for an RLT-online run.

Reads ``chunk_compare.jsonl`` from a run dir and prints the
distribution of every per-inference timing field (mean / median /
p90 / p95 / max) plus the fraction of inferences that exceed a
configurable RTC budget.

Use after running on robot to confirm the stack still fits the
real-time budget — especially when changing the RL token encoder
size or the S1 backbone.

Usage:
    python src/lerobot/policies/hvla/scripts/rlt_perf_audit.py outputs/rlt_online_v2_widened
    python src/lerobot/policies/hvla/scripts/rlt_perf_audit.py outputs/rlt_online_v2_widened --budget-ms 200

Prints to stdout; no side effects.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import statistics
from collections import defaultdict
from pathlib import Path

FIELDS = (
    "total_delay_ms",
    "obs_to_infer_ms",
    "enc_obs_ms",
    "rl_tok_ms",
    "s1_denoise_ms",
    "rlt_actor_ms",
    "rlt_post_ms",
)


def _percentile(xs: list[float], p: float) -> float:
    if not xs:
        return float("nan")
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(round(p * (len(xs) - 1))))]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path, help="Directory containing chunk_compare.jsonl")
    parser.add_argument(
        "--budget-ms", type=float, default=200.0, help="RTC budget in ms (default 200 = 6 frames @ 30Hz)"
    )
    parser.add_argument(
        "--per-episode", action="store_true", help="Also print per-episode median/p95 of total_delay_ms"
    )
    args = parser.parse_args()

    path = args.run_dir / "chunk_compare.jsonl"
    if not path.exists():
        print(f"error: {path} not found")
        return 1

    rows: list[dict] = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            with contextlib.suppress(json.JSONDecodeError):
                rows.append(json.loads(line))

    rlt_rows = [r for r in rows if r.get("rlt_active")]
    print(f"records (rlt_active=True): {len(rlt_rows)} of {len(rows)} total")
    if not rlt_rows:
        print("no rlt-active records — nothing to audit")
        return 0

    print(f"\n{'field':<18s} {'mean':>8s} {'median':>8s} {'p90':>8s} {'p95':>8s} {'max':>8s}  unit=ms")
    print("-" * 70)
    for field in FIELDS:
        vals = [r[field] for r in rlt_rows if r.get(field) is not None]
        if not vals:
            continue
        print(
            f"{field:<18s} {statistics.mean(vals):>8.2f} "
            f"{statistics.median(vals):>8.2f} "
            f"{_percentile(vals, 0.90):>8.2f} "
            f"{_percentile(vals, 0.95):>8.2f} "
            f"{max(vals):>8.2f}"
        )

    total = [r["total_delay_ms"] for r in rlt_rows if r.get("total_delay_ms") is not None]
    if total:
        n_above_half = sum(1 for t in total if t > args.budget_ms * 0.5)
        n_above = sum(1 for t in total if t > args.budget_ms)
        print(f"\nRTC budget = {args.budget_ms:.0f} ms:")
        print(
            f"  total > {args.budget_ms * 0.5:.0f} ms: "
            f"{n_above_half}/{len(total)} = "
            f"{100 * n_above_half / len(total):.1f}%"
        )
        print(
            f"  total > {args.budget_ms:.0f} ms: {n_above}/{len(total)} = {100 * n_above / len(total):.1f}%"
        )

    if args.per_episode:
        by_ep: dict[int, list[float]] = defaultdict(list)
        for r in rlt_rows:
            t = r.get("total_delay_ms")
            if t is not None:
                by_ep[r["ep"]].append(t)
        print("\nPer-episode total_delay_ms:")
        for ep in sorted(by_ep):
            vs = by_ep[ep]
            print(
                f"  ep{ep:3d}: median={statistics.median(vs):5.1f}ms  "
                f"p95={_percentile(vs, 0.95):5.1f}ms  n={len(vs)}"
            )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

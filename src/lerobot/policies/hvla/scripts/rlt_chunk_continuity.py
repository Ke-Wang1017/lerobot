"""Audit chunk_compare.jsonl for RTC + RLT slice correctness.

Two checks, intentionally separate.

Check 1 — RTC continuity (ref ↔ ref, RLT-agnostic).
    For each consecutive pair of inferences (i, i+1) within the same
    episode, S1's reproduced prefix should match S1's pure output at the
    overlap point in the previous chunk. Concretely:

        ref_{i+1}[0:D]  ≈  ref_i[exec_idx_{i+1} : exec_idx_{i+1} + D]

    Both sides come from the dump's ``ref`` field — no actor, no C, no
    splicing. Distance interpretation:
      • ≈ 0 in the typical case where the actor didn't write at the
        prefix-source region of the previous chunk (or no actor).
      • ≈ actor's deflection magnitude at frames the prefix overlaps
        with the actor's write region. Not an RTC bug — just records
        what the actor changed in the prefix the next inference saw.
      • Large across the board → real RTC bug (S1 not honoring its
        prefix conditioning).

Check 2 — RLT slice + prefix sourcing (RLT-specific).
    Where does the next inference's prefix region fall relative to the
    previous chunk's actor window? Buckets transitions into:
      • in actor window — desired
      • pre actor window — actor never executed
      • straddles boundary — prefix crosses actor/S1 stitch
      • past actor window — actor finished, prefix from S1 tail
    Assumes actor wrote at [D:D+C] (current code; old dumps where actor
    wrote at [0:C] need ``--legacy-actor-at-zero``).

Usage:
    python src/lerobot/policies/hvla/scripts/rlt_chunk_continuity.py outputs/rlt_online_v2_widened/chunk_compare.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def frame_distance(a: list[float], b: list[float]) -> float:
    """Per-joint L2 between two single-frame action vectors. Same scale
    as the normalized-action space used elsewhere in RLT."""
    return sum((x - y) ** 2 for x, y in zip(a, b, strict=False)) ** 0.5


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("dump", type=Path, help="path to chunk_compare.jsonl")
    p.add_argument(
        "--rtc-fail-threshold",
        type=float,
        default=0.5,
        help="per-frame distance above which the RTC continuity is flagged (default 0.5 in normalized space)",
    )
    p.add_argument("--max-bad", type=int, default=20, help="how many bad records to print before truncating")
    p.add_argument(
        "--legacy-actor-at-zero",
        action="store_true",
        help="Check 2 assumes actor wrote at [D:D+C] (current "
        "code). Pass this for dumps from before that "
        "change, where actor wrote at [0:C].",
    )
    args = p.parse_args()

    if not args.dump.exists():
        print(f"ERROR: {args.dump} does not exist", file=sys.stderr)
        return 1

    with args.dump.open() as f:
        records = [json.loads(line) for line in f if line.strip()]
    print(f"loaded {len(records)} dump records")
    if len(records) < 2:
        print("need at least 2 records to check continuity")
        return 1

    # ─── Check 1: ref ↔ ref RTC continuity ───────────────────────────────
    rtc_distances: list[float] = []
    rtc_bad: list[dict] = []
    skipped_rtc = 0

    # ─── Check 2: prefix sourcing buckets ────────────────────────────────
    bucket_in_window = 0
    bucket_pre_window = 0
    bucket_straddles = 0
    bucket_past_window = 0
    cross_episode = 0

    for i in range(len(records) - 1):
        rec_i, rec_j = records[i], records[i + 1]

        if rec_i.get("ep") != rec_j.get("ep"):
            cross_episode += 1
            continue

        D_j = rec_j["prefix_len"]
        exec_idx_j = rec_j.get("exec_idx")
        if D_j == 0 or exec_idx_j is None:
            skipped_rtc += 1
            continue

        ref_i = rec_i["ref"]
        ref_j = rec_j["ref"]

        if exec_idx_j + D_j > len(ref_i):
            print(
                f"WARNING: step {rec_j.get('step')} has exec_idx {exec_idx_j} "
                f"+ D {D_j} > chunk_size {len(ref_i)}; skipping"
            )
            continue

        # Check 1: pure ref-to-ref distance at the prefix overlap.
        prefix_source = ref_i[exec_idx_j : exec_idx_j + D_j]
        prefix_actual = ref_j[:D_j]
        per_frame = [frame_distance(a, b) for a, b in zip(prefix_source, prefix_actual, strict=False)]
        max_d = max(per_frame) if per_frame else 0.0
        rtc_distances.append(max_d)
        if max_d > args.rtc_fail_threshold:
            rtc_bad.append(
                {
                    "step_j": rec_j.get("step"),
                    "ep": rec_j.get("ep"),
                    "max_dist": round(max_d, 3),
                    "exec_idx": exec_idx_j,
                    "D_j": D_j,
                }
            )

        # Check 2: actor window bucketing.
        D_i = rec_i["prefix_len"]
        actor_present = rec_i.get("actor") is not None
        C_i = len(rec_i["actor"]) if actor_present else 0
        if not actor_present or C_i == 0:
            continue

        if args.legacy_actor_at_zero:
            win_start, win_end = 0, C_i
        else:
            win_start, win_end = D_i, D_i + C_i
        prefix_end = exec_idx_j + D_j

        if exec_idx_j < win_start:
            bucket_pre_window += 1
        elif prefix_end > win_end:
            if exec_idx_j >= win_end:
                bucket_past_window += 1
            else:
                bucket_straddles += 1
        else:
            bucket_in_window += 1

    # ─── Report ───────────────────────────────────────────────────────────
    print()
    print("=== Check 1: RTC ref↔ref continuity ===")
    print(f"sample size: {len(rtc_distances)}  (skipped {skipped_rtc} D=0 / cross-episode {cross_episode})")
    if rtc_distances:
        rtc_distances.sort()
        n = len(rtc_distances)
        p50 = rtc_distances[n // 2]
        p95 = rtc_distances[int(n * 0.95)]
        p99 = rtc_distances[int(n * 0.99)]
        max_d = rtc_distances[-1]
        print(f"max-frame prefix distance — p50={p50:.4f}  p95={p95:.4f}  p99={p99:.4f}  max={max_d:.4f}")
        print(
            f"records exceeding threshold ({args.rtc_fail_threshold}): "
            f"{len(rtc_bad)} / {len(rtc_distances)} "
            f"({len(rtc_bad) / len(rtc_distances) * 100:.1f}%)"
        )
        if rtc_bad:
            print("first few flagged transitions:")
            print(
                "  (large value here = either S1 RTC bug, OR actor "
                "wrote in the prefix region of the previous chunk)"
            )
            for r in rtc_bad[: args.max_bad]:
                print(
                    f"  ep{r['ep']} step{r['step_j']}: "
                    f"max_dist={r['max_dist']}  exec_idx={r['exec_idx']}  "
                    f"D={r['D_j']}"
                )

    print()
    print(
        f"=== Check 2: prefix sourcing relative to actor window "
        f"({'actor at [0:C]' if args.legacy_actor_at_zero else 'actor at [D:D+C]'}) ==="
    )
    total = bucket_in_window + bucket_pre_window + bucket_straddles + bucket_past_window
    if total == 0:
        print("no actor windows to evaluate")
    else:

        def pct(n: int) -> str:
            return f"{n / total * 100:.1f}%"

        print(f"sample size: {total}")
        print(f"  in actor window     :  {bucket_in_window}  ({pct(bucket_in_window)})  ← desired")
        print(
            f"  pre actor window    :  {bucket_pre_window}  "
            f"({pct(bucket_pre_window)})  ← actor discarded, never executed"
        )
        print(
            f"  straddles boundary  :  {bucket_straddles}  "
            f"({pct(bucket_straddles)})  ← prefix crosses actor/S1 stitch"
        )
        print(
            f"  past actor window   :  {bucket_past_window}  "
            f"({pct(bucket_past_window)})  ← actor finished; "
            f"prefix entirely from S1 tail"
        )

    return 0


if __name__ == "__main__":
    sys.exit(main())

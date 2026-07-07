#!/usr/bin/env python
"""Per-joint control-lag analyzer for recorded trajectory_replay datasets.

For each joint:
- ``lag_frames``: argmax_k Pearson-corr(action[t], state[t+k]) over k in [0, max_lag].
  This is the time offset that best aligns the commanded position with the
  measured response. Cross-correlation summary of the whole control loop's
  tracking lag (structural + bus + motor PID + dynamics).
- ``plateau_std``: standard deviation of ``observation.state`` over frames
  where ``action`` is held constant. Detects shakiness — any oscillation
  contributes to variance regardless of frequency (Parseval), so this works
  even when the oscillation is faster than the 30 Hz sample rate (the energy
  aliases into the 0-15 Hz band but the total variance is preserved).

Usage:
    python scripts/analyze_control_lag.py <dataset_name_or_path>

Output: one row per joint with lag in frames + ms, peak correlation, and
plateau-σ. A run without any plateaus (e.g. continuous-motion recording)
reports plateau_std as NaN.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq


def find_dataset(name_or_path: str) -> Path:
    """Resolve a name (under ``$HF_LEROBOT_HOME``) or an absolute/relative path."""
    p = Path(name_or_path).expanduser()
    if p.is_absolute() and p.exists():
        return p
    cache_root = Path.home() / ".cache" / "huggingface" / "lerobot"
    candidate = cache_root / name_or_path
    if candidate.exists():
        return candidate
    if p.exists():
        return p
    raise FileNotFoundError(f"dataset '{name_or_path}' not found (looked under {cache_root}/ and as a path)")


def load_action_state(dataset_path: Path) -> tuple[np.ndarray, np.ndarray, list[str], int]:
    """Load action + state arrays and the joint-name list.

    Returns ``(action, state, joint_names, fps)`` where action and state are
    ``(num_frames, num_joints)`` float64.
    """
    info_path = dataset_path / "meta" / "info.json"
    info = json.loads(info_path.read_text())
    fps = int(info.get("fps", 30))
    joint_names = info["features"]["action"]["names"]

    parquets = sorted((dataset_path / "data").rglob("*.parquet"))
    if not parquets:
        raise FileNotFoundError(f"no parquet files under {dataset_path}/data")

    action_chunks: list[np.ndarray] = []
    state_chunks: list[np.ndarray] = []
    for p in parquets:
        t = pq.read_table(p, columns=["action", "observation.state"])
        action_chunks.append(np.array(t.column("action").to_pylist(), dtype=np.float64))
        state_chunks.append(np.array(t.column("observation.state").to_pylist(), dtype=np.float64))

    action = np.concatenate(action_chunks, axis=0)
    state = np.concatenate(state_chunks, axis=0)
    return action, state, joint_names, fps


def cross_correlation_lag(action: np.ndarray, state: np.ndarray, max_lag: int) -> tuple[int, float]:
    """Find the lag k in [0, max_lag] that maximises Pearson correlation
    between ``action[:-k]`` and ``state[k:]`` (state lags action by k frames).
    Returns ``(k, corr)``. For a flat signal returns ``(0, 0.0)``.
    """
    best_k, best_corr = 0, -np.inf
    for k in range(0, max_lag + 1):
        a = action[: len(action) - k] if k > 0 else action
        s = state[k:] if k > 0 else state
        a_c = a - a.mean()
        s_c = s - s.mean()
        a_norm = float(np.linalg.norm(a_c))
        s_norm = float(np.linalg.norm(s_c))
        if a_norm < 1e-12 or s_norm < 1e-12:
            continue
        corr = float(np.dot(a_c, s_c) / (a_norm * s_norm))
        if corr > best_corr:
            best_corr = corr
            best_k = k
    return best_k, best_corr if np.isfinite(best_corr) else 0.0


def find_plateaus(action: np.ndarray, min_length: int = 10, tolerance: float = 1e-3) -> list[tuple[int, int]]:
    """Find runs of consecutive frames where ``action[t]`` is constant within
    ``tolerance`` (in action units — degrees-or-radians, same scale as the
    parquet). A "plateau" needs at least ``min_length`` frames.

    Returns list of ``(start, end_exclusive)`` index pairs.
    """
    n = len(action)
    plateaus: list[tuple[int, int]] = []
    i = 0
    while i < n:
        j = i + 1
        anchor = action[i]
        while j < n and abs(action[j] - anchor) < tolerance:
            j += 1
        if j - i >= min_length:
            plateaus.append((i, j))
        i = j
    return plateaus


def plateau_std(state: np.ndarray, plateaus: list[tuple[int, int]], skip_head: int = 5) -> float:
    """Average ``std(state)`` across plateaus. Skip the first ``skip_head``
    frames of each plateau so settling transients (right after motion stops)
    don't count toward the shakiness measurement.
    """
    stds: list[float] = []
    for start, end in plateaus:
        a = start + skip_head
        if end - a >= 5:
            stds.append(float(state[a:end].std()))
    return float(np.mean(stds)) if stds else float("nan")


def main(dataset_name_or_path: str, max_lag: int) -> int:
    dataset = find_dataset(dataset_name_or_path)
    action, state, joints, fps = load_action_state(dataset)
    print(f"dataset: {dataset}")
    print(f"frames:  {len(action)}   joints: {len(joints)}   fps: {fps}")
    print()
    print(f"{'joint':<26}{'lag_fr':>8}{'lag_ms':>9}{'corr':>8}{'σ_state':>10}{'σ_action':>10}")
    print("-" * 71)

    avg_lag_ms = []
    avg_sigma = []
    for j, name in enumerate(joints):
        a = action[:, j]
        s = state[:, j]
        k, corr = cross_correlation_lag(a, s, max_lag)
        plats = find_plateaus(a, min_length=10, tolerance=1e-3)
        sig_state = plateau_std(s, plats)
        sig_action = plateau_std(a, plats)  # sanity: should be ~0 by construction
        lag_ms = k * (1000.0 / fps)
        avg_lag_ms.append(lag_ms)
        if np.isfinite(sig_state):
            avg_sigma.append(sig_state)
        sig_state_s = f"{sig_state:>9.4f}" if np.isfinite(sig_state) else "      n/a"
        sig_action_s = f"{sig_action:>9.4f}" if np.isfinite(sig_action) else "      n/a"
        print(f"{name:<26}{k:>8d}{lag_ms:>9.1f}{corr:>8.3f}{sig_state_s}{sig_action_s}")

    print()
    print(
        f"average lag: {np.mean(avg_lag_ms):.1f} ms ({np.mean(avg_lag_ms) * fps / 1000:.2f} frames at {fps} Hz)"
    )
    if avg_sigma:
        print(f"average plateau σ_state: {np.mean(avg_sigma):.4f}")
    else:
        print("no plateaus detected in action signal (continuous motion?)")
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("dataset", help="Dataset name (under $HF_LEROBOT_HOME) or path to its root dir")
    parser.add_argument(
        "--max-lag",
        type=int,
        default=20,
        help="Search lag in [0, max_lag] frames (default: 20 = ~660 ms at 30 Hz)",
    )
    args = parser.parse_args()
    sys.exit(main(args.dataset, args.max_lag))

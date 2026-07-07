"""Build a test dataset with editable features for manually exercising the
feature-editing GUI.

Copies an existing local dataset and bolts on a configurable subset of
features:

* ``reward`` — float32[1], per-frame sinusoid around -1.0 (RECAP-style step
  penalty with visual variation). Slider+number widget.
* ``success`` — bool[1], per-episode broadcast (alternating true/false
  across episodes). Checkbox widget; edits coerce to whole-episode.
* ``subtask_index`` — int64[1], 3-segment progression per episode plus a
  ``meta/subtasks.parquet`` lookup with starter strings. Frontend stages
  strings; backend resolves to indices at Save.
* ``quality`` — int64[1], per-frame rating 1–5 (π0.7-style annotator
  score; the values vary within each episode, oscillating 5→1→5 with
  ~60-frame period, so sub-range edits exercise the declared-bounds
  slider). Slider scaled to [1, 5] via declared bounds.
* ``control_mode`` — int64[1] with names=["ee","joint"], per-episode
  categorical alternating ee/joint across episodes. Demonstrates the
  ``<select>`` widget; lives only in the Inspector under
  "Per-episode" since the timeline row of a uniform feature wastes
  space.

Default builds all five features. Pass ``--features a,b,c`` to pick a
subset.

Usage:
    python scripts/gui/make_test_feature_editing_dataset.py \\
        --src ~/.cache/huggingface/lerobot/thewisp/intervention_cylinder_ring_assembly \\
        --dst /tmp/feature-editing-test-dataset

    # Just the π0.7 features:
    python scripts/gui/make_test_feature_editing_dataset.py \\
        --src ~/.cache/huggingface/lerobot/thewisp/intervention_cylinder_ring_assembly \\
        --dst /tmp/pi07-features-test-dataset \\
        --features quality,control_mode

After it runs, open the dst path in the GUI and try:

1. Drag-select on the ``reward`` row → slider+number widget. Type a
   value or drag the slider. Confirm the chip appears in the edits bar.
2. Drag-select on the ``success`` row → checkbox widget. The card
   shows "per-episode" — the band should expand to fill the episode
   when you stage. Toggle and confirm a chip appears whose range is
   the full episode regardless of what you dragged.
3. Stage two overlapping edits on ``reward`` for the same episode →
   the second one should pop the overlap-confirmation modal.
4. Save → reload → values persist.

For ``quality`` / ``control_mode``: after the first Save the
``observed_min``/``observed_max`` should populate (e.g. ``quality
[1.0 … 5.0]`` in the card header) and the slider scale to that range.

For subtask editing, type a new string (e.g. "inspect knot"). After
Save, confirm it appears in ``meta/subtasks.parquet`` with a fresh
integer index.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

# ── Feature builders ────────────────────────────────────────────────────────
#
# Each builder returns a dict with:
#   - schema: the info.json features entry (dtype, shape, optionally names/min/max)
#   - fill:   (df, episodes_seen) -> column data of length len(df)
#   - extras: optional list of (relpath, write_fn) for sidecar files like
#             meta/subtasks.parquet
#
# ``episodes_seen`` is a per-builder scratch dict shared across shards so the
# builder can pin a per-episode value the first time it sees an episode and
# reuse it on subsequent shards.


def _build_reward(df: pd.DataFrame, _seen: dict) -> np.ndarray:
    frame_idx = df["frame_index"].astype(int).values
    return (-1.0 + 0.2 * np.sin(0.1 * frame_idx.astype(np.float32))).astype(np.float32)


def _build_success(df: pd.DataFrame, seen: dict) -> np.ndarray:
    """Per-episode bool: alternates true/false across episodes."""
    eps = df["episode_index"].astype(int).values
    out = np.zeros(len(df), dtype=bool)
    for i, ep in enumerate(eps):
        if ep not in seen:
            seen[ep] = bool(ep % 2)
        out[i] = seen[ep]
    return out


def _build_subtask_index(df: pd.DataFrame, _seen: dict) -> np.ndarray:
    """3-segment progression by frame position (crude bucket)."""
    fi = df["frame_index"].astype(int).values
    return np.where(fi < 30, 0, np.where(fi < 70, 1, 2)).astype(np.int64)


def _build_quality(df: pd.DataFrame, _seen: dict) -> np.ndarray:
    """Per-frame int 1-5 — π0.7-style quality rating.

    π0.7 annotators mark quality per frame so they can flag individual
    fumbles within an otherwise-good demonstration. We mirror that here
    with an oscillation that dips from 5 down to 1 and back, period
    ~60 frames — gives a visually interesting per-frame curve and lets
    the per-frame slider edit a sub-range without coercion.
    """
    fi = df["frame_index"].astype(int).values
    val = 3.0 + 2.0 * np.cos(2.0 * np.pi * fi / 60.0)
    return np.clip(np.round(val), 1, 5).astype(np.int64)


def _build_control_mode(df: pd.DataFrame, seen: dict) -> np.ndarray:
    """Per-episode categorical int (0=ee, 1=joint), alternating across episodes."""
    eps = df["episode_index"].astype(int).values
    out = np.zeros(len(df), dtype=np.int64)
    for i, ep in enumerate(eps):
        if ep not in seen:
            seen[ep] = int(ep % 2)
        out[i] = seen[ep]
    return out


# Each entry: name → (info.json schema, builder, optional extras factory).
# The schema entries demonstrate four patterns:
#   - reward: plain float scalar
#   - success: plain bool scalar
#   - subtask_index: int scalar with sidecar lookup table (string-decoded by GUI)
#   - quality: int scalar with declared min/max bounds (validated)
#   - control_mode: int scalar with declared names (categorical)
_FEATURE_BUILDERS = {
    "reward": {
        "schema": {"dtype": "float32", "shape": [1], "names": None},
        "fill": _build_reward,
    },
    "success": {
        # ``per_episode: true`` is the format flag declaring intent: every
        # frame of an episode carries the same success value. The detector
        # honors this without scanning the data, and ``add_frame`` rejects
        # writes that violate the invariant.
        "schema": {"dtype": "bool", "shape": [1], "names": None, "per_episode": True},
        "fill": _build_success,
    },
    "subtask_index": {
        "schema": {"dtype": "int64", "shape": [1], "names": None},
        "fill": _build_subtask_index,
        "extras": [
            (
                "meta/subtasks.parquet",
                lambda: pd.DataFrame(
                    {"subtask_index": [0, 1, 2]},
                    index=pd.Index(["approach", "grasp", "release"], name="subtask"),
                ),
            ),
        ],
    },
    "quality": {
        # min/max are *declared bounds* — backend's validate_feature_dtype_and_shape
        # rejects values outside [1, 5] at add_frame and stage time. The GUI
        # also uses these to scale the slider so the range stays at [1, 5]
        # regardless of episode contents.
        "schema": {"dtype": "int64", "shape": [1], "names": None, "min": 1, "max": 5},
        "fill": _build_quality,
    },
    "control_mode": {
        # ``names`` makes this a categorical: stored as int index, displayed
        # by label. ``per_episode: true`` declares it episode-wide so the GUI
        # doesn't need to scan the parquet to figure that out (each recording
        # is in one control mode for its entire duration).
        "schema": {
            "dtype": "int64",
            "shape": [1],
            "names": ["ee", "joint"],
            "per_episode": True,
        },
        "fill": _build_control_mode,
    },
}

ALL_FEATURES = list(_FEATURE_BUILDERS.keys())


def add_editable_features(dst: Path, features_to_add: list[str]) -> None:
    """Mutate the dataset at ``dst`` in place: add the named features."""
    info_path = dst / "meta" / "info.json"
    info = json.loads(info_path.read_text())

    # Schema additions ──────────────────────────────────────────────────
    for name in features_to_add:
        spec = _FEATURE_BUILDERS[name]["schema"]
        if name not in info["features"]:
            info["features"][name] = spec
            print(f"  + declared {name} in info.json  ({spec})")
        else:
            print(f"  ~ {name} already exists in info.json — leaving as-is")

    info_path.write_text(json.dumps(info, indent=2))

    # Data shards ───────────────────────────────────────────────────────
    data_dir = dst / "data"
    shards = sorted(data_dir.rglob("*.parquet"))
    print(f"  data shards: {len(shards)}")

    # Per-builder scratch dicts (so per-episode values stay consistent
    # across shards).
    scratch: dict[str, dict] = {n: {} for n in features_to_add}

    for shard in shards:
        df = pd.read_parquet(shard)
        for name in features_to_add:
            df[name] = _FEATURE_BUILDERS[name]["fill"](df, scratch[name])
        df.to_parquet(shard, compression="snappy", index=False)
        print(f"  ✓ {shard.relative_to(dst)} ({len(df)} rows)")

    # Sidecar extras (e.g. meta/subtasks.parquet) ───────────────────────
    for name in features_to_add:
        for relpath, factory in _FEATURE_BUILDERS[name].get("extras", []):
            target = dst / relpath
            target.parent.mkdir(parents=True, exist_ok=True)
            df = factory()
            df.to_parquet(target)
            print(f"  + {relpath}  → {list(df.index)}")

    # Episodes parquet stats columns ────────────────────────────────────
    # Best-effort: re-add stats columns for the new features so the GUI's
    # verification path doesn't warn. Skip if the file doesn't already
    # track stats for any feature.
    eps_dir = dst / "meta" / "episodes"
    eps_shards = sorted(eps_dir.rglob("*.parquet"))
    for shard in eps_shards:
        eps_df = pd.read_parquet(shard)
        sample_stat = next(
            (c for c in eps_df.columns if c.startswith("stats/") and c.endswith("/min")),
            None,
        )
        if sample_stat is None:
            print(f"  · {shard.name}: no existing stats columns — skipping stats fill")
            continue

        for feature in features_to_add:
            if f"stats/{feature}/min" in eps_df.columns:
                continue  # already populated
            placeholder = [[0.0]] * len(eps_df)
            for stat in ("min", "max", "mean", "std", "q01", "q10", "q50", "q90", "q99"):
                eps_df[f"stats/{feature}/{stat}"] = placeholder
            eps_df[f"stats/{feature}/count"] = [[1]] * len(eps_df)

        eps_df.to_parquet(shard)
        print(f"  ✓ {shard.relative_to(dst)} (stats columns added)")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--src",
        required=True,
        type=Path,
        help="Source dataset root (must be a local v3 LeRobot dataset).",
    )
    parser.add_argument(
        "--dst",
        required=True,
        type=Path,
        help="Destination directory. Will be created; refuses to overwrite an existing non-empty dir.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="If set, wipe the destination first.",
    )
    parser.add_argument(
        "--features",
        type=str,
        default=",".join(ALL_FEATURES),
        help=(
            f"Comma-separated subset of features to add. Available: {','.join(ALL_FEATURES)}. Default: all."
        ),
    )
    args = parser.parse_args()

    requested = [s.strip() for s in args.features.split(",") if s.strip()]
    unknown = [n for n in requested if n not in _FEATURE_BUILDERS]
    if unknown:
        raise SystemExit(f"Unknown feature(s): {unknown}. Available: {ALL_FEATURES}")

    src: Path = args.src.expanduser().resolve()
    dst: Path = args.dst.expanduser().resolve()

    if not (src / "meta" / "info.json").exists():
        raise SystemExit(f"src {src} doesn't look like a LeRobot dataset (no meta/info.json)")
    if dst.exists() and any(dst.iterdir()):
        if not args.force:
            raise SystemExit(f"dst {dst} exists and is non-empty (pass --force to wipe)")
        # safe-destruct: user passed --force; this is a test fixture path under their control.
        shutil.rmtree(dst)

    print(f"Copying {src} → {dst}")
    shutil.copytree(src, dst)
    print(f"  ✓ copied {sum(1 for _ in dst.rglob('*'))} files")

    print(f"Adding editable features: {requested}…")
    add_editable_features(dst, requested)

    print()
    print(f"✓ Test dataset ready at {dst}")
    print("Open it in the GUI's Data tab and try:")
    print("  1. Drag-select on the `reward` row → slider+number widget")
    print("  2. Drag-select on the `success` row → checkbox; band should expand to whole episode")
    print("  3. Stage two overlapping reward edits → triggers the overlap-confirmation modal")
    print("  4. Save → reload → values persist")


if __name__ == "__main__":
    main()

# Add Feature Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement in-place schema-add for LeRobot datasets, surface `reward` (per-frame) and `success` (per-episode tri-state) as MUST-have default features via a banner, and expose a generic "+ Add feature" dialog for other custom columns.

**Architecture:** A new `dataset_tools.add_features_inplace()` rewrites parquet shards in place (videos untouched), preserving an explicit `per_episode` hint in `info.json`. Two GUI endpoints (`POST /api/datasets/{id}/features` and `.../features/defaults`) wrap it with cache invalidation, a pending-edits guard, and a `dataset.schema_changed` WS push. Frontend adds a banner, a dialog, and a tri-state success widget.

**Tech Stack:** Python 3.12+, pyarrow (parquet), FastAPI (WS + REST), pytest, draccus configs, vanilla JS frontend, `uv run` for execution.

**Spec:** [add_feature.md](add_feature.md)

---

## File Structure

**LeRobot core (data layer):**

- Modify: `src/lerobot/datasets/dataset_tools.py` — new public `add_features_inplace()`, helpers, `_sweep_orphan_tmp_shards()`.
- Modify: `src/lerobot/datasets/dataset_metadata.py` — preserve `per_episode` hint when reading/writing `info.json`.
- Modify (light): `src/lerobot/datasets/feature_utils.py` — accept `per_episode` key in feature spec without erroring.

**GUI backend:**

- Modify: `src/lerobot/gui/api/datasets.py` — two new POST endpoints, surface declared `per_episode` in `FeatureSchema`, call orphan-`.tmp` sweep in dataset open path.
- Modify: `src/lerobot/gui/api/playback.py` — emit `dataset.schema_changed` WS event.
- Modify: `src/lerobot/gui/state.py` — `pending_feature_set_edits_for_dataset()` helper for the guard.

**GUI frontend:**

- Modify: `src/lerobot/gui/static/feature_editing.js` — banner, success widget renderer, schema-changed handler, "+ Add feature" button hook.
- Create: `src/lerobot/gui/static/add_feature_dialog.js` — dialog modal logic.
- Modify: `src/lerobot/gui/static/index.html` — load new JS, banner DOM slot, dialog DOM template.
- Modify: `src/lerobot/gui/static/style.css` — banner, dialog, segment-control styles.

**Tests:**

- Modify: `tests/datasets/test_dataset_tools.py` — add `add_features_inplace` tests inline (matches existing convention).
- Modify: `tests/datasets/test_dataset_metadata.py` — `per_episode` round-trip.
- Create: `tests/gui/test_feature_add_endpoints.py` — endpoint coverage.
- Modify: `tests/gui/test_state.py` — pending-edit guard helper.

---

## Task 1: `per_episode` hint round-trips through `info.json`

**Files:**

- Modify: `src/lerobot/datasets/dataset_metadata.py` (preserve key when copying features)
- Modify: `src/lerobot/datasets/feature_utils.py` (accept the key without erroring)
- Test: `tests/datasets/test_dataset_metadata.py`

- [ ] **Step 1: Read existing patterns**

```bash
grep -n "per_episode\|features\[" src/lerobot/datasets/dataset_metadata.py | head -30
grep -n "required_keys\|feature_info" src/lerobot/datasets/feature_utils.py | head -20
```

Read `LeRobotDatasetMetadata.create()` and `_load_info()` (or equivalent) so the test points to the right code paths.

- [ ] **Step 2: Write the failing test**

Append to `tests/datasets/test_dataset_metadata.py`:

```python
def test_per_episode_hint_round_trips(tmp_path):
    """Adding a feature with per_episode=True preserves the hint in info.json."""
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    features = {
        "reward": {"dtype": "float32", "shape": [1], "names": None},
        "success": {"dtype": "int8", "shape": [1], "names": None, "per_episode": True},
    }
    ds = LeRobotDataset.create(
        repo_id="test/per_episode_roundtrip",
        fps=10,
        features=features,
        root=tmp_path / "ds",
        robot_type="dummy",
        use_videos=False,
    )

    # Reload from disk and check the hint survived.
    ds2 = LeRobotDataset(repo_id="test/per_episode_roundtrip", root=tmp_path / "ds")
    assert ds2.meta.features["success"].get("per_episode") is True
    assert ds2.meta.features["reward"].get("per_episode", False) is False
```

- [ ] **Step 3: Run the test and confirm it fails**

```bash
uv run pytest tests/datasets/test_dataset_metadata.py::test_per_episode_hint_round_trips -svv
```

Expected: FAIL — either `per_episode` is dropped, or feature validation rejects the unknown key.

- [ ] **Step 4: Make the test pass**

In `src/lerobot/datasets/dataset_metadata.py` find the feature-write path (look near line 634 where `DEFAULT_FEATURES` is merged in). Ensure the `per_episode` key, if present in input, is carried through into the persisted `info.json` features dict. If serialization filters keys, add `per_episode` to the allowed set (or remove the filter — the persisted dict should preserve any extra keys for forward compatibility).

In `src/lerobot/datasets/feature_utils.py` near the `validate_feature_dtype_and_shape` function (line 221), confirm validation does not reject features with extra keys like `per_episode`. If it does, allow extra keys silently.

- [ ] **Step 5: Run the test and confirm it passes**

```bash
uv run pytest tests/datasets/test_dataset_metadata.py::test_per_episode_hint_round_trips -svv
```

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add src/lerobot/datasets/dataset_metadata.py src/lerobot/datasets/feature_utils.py tests/datasets/test_dataset_metadata.py
git commit -m "feat(datasets): preserve per_episode hint in info.json features dict"
```

---

## Task 2: `add_features_inplace()` — basic per-frame add (reward)

**Files:**

- Modify: `src/lerobot/datasets/dataset_tools.py` (new public function)
- Test: `tests/datasets/test_dataset_tools.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/datasets/test_dataset_tools.py`. Reuse the existing fixture pattern from the file (search the file for `tmp_path` and `LeRobotDataset.create` to find a similar pattern):

```python
def test_add_features_inplace_per_frame_reward(tmp_path):
    """Adding a per-frame `reward` rewrites parquet shards in place; videos untouched."""
    import numpy as np
    import pyarrow.parquet as pq
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from lerobot.datasets.dataset_tools import add_features_inplace

    # Minimal dataset: 2 episodes, 5 frames each, no images.
    features = {
        "observation.state": {"dtype": "float32", "shape": [2], "names": None},
        "action": {"dtype": "float32", "shape": [2], "names": None},
    }
    ds = LeRobotDataset.create(
        repo_id="test/add_inplace_reward",
        fps=10,
        features=features,
        root=tmp_path / "ds",
        robot_type="dummy",
        use_videos=False,
    )
    for _ in range(2):
        for _ in range(5):
            ds.add_frame({"observation.state": np.zeros(2, dtype=np.float32),
                          "action": np.zeros(2, dtype=np.float32)}, task="t")
        ds.save_episode()
    ds.finalize()

    # Snapshot data shard mtimes & video dir contents (should be empty here).
    data_files_before = sorted((tmp_path / "ds" / "data").rglob("*.parquet"))
    assert data_files_before, "test setup expected at least one parquet shard"

    add_features_inplace(
        ds,
        features={"reward": (0.0, {"dtype": "float32", "shape": [1], "names": None})},
    )

    # info.json now has reward.
    ds2 = LeRobotDataset(repo_id="test/add_inplace_reward", root=tmp_path / "ds")
    assert "reward" in ds2.meta.features
    assert ds2.meta.features["reward"]["dtype"] == "float32"

    # Every parquet shard now has a `reward` column with all 0.0.
    for f in (tmp_path / "ds" / "data").rglob("*.parquet"):
        table = pq.read_table(f)
        assert "reward" in table.column_names, f"reward missing from {f}"
        col = table.column("reward").to_pylist()
        assert all(v == 0.0 for v in col), f"non-zero values in {f}: {col[:5]}"

    # No .tmp orphans.
    assert not list((tmp_path / "ds" / "data").rglob("*.tmp")), "orphan .tmp files left over"
```

- [ ] **Step 2: Run test, confirm failure**

```bash
uv run pytest tests/datasets/test_dataset_tools.py::test_add_features_inplace_per_frame_reward -svv
```

Expected: FAIL — `add_features_inplace` doesn't exist yet.

- [ ] **Step 3: Implement `add_features_inplace()`**

In `src/lerobot/datasets/dataset_tools.py`, add after the existing `set_feature_values()` block (look for "set_feature_values — in-place per-frame value editing" around line 948 to find the right neighborhood):

```python
def add_features_inplace(
    dataset: "LeRobotDataset",
    features: dict[str, tuple],
    *,
    recompute_stats: bool = True,
) -> None:
    """Add features to an existing dataset in place.

    Each entry in `features` maps a feature name to (fill_value, feature_info).
    `fill_value` is a scalar broadcast across every frame, OR a numpy array of
    length == total_frames if per-frame values are pre-computed.
    `feature_info` follows the schema used by `modify_features` and may include
    an optional `per_episode: true` hint.

    Videos are NOT touched. Only parquet shards under
    `data/chunk-*/file-*.parquet` and `meta/info.json` are rewritten.
    Per-episode stats for new columns are recomputed when
    `recompute_stats=True`. Calls `dataset.finalize()` before returning.
    """
    import json
    import shutil
    import numpy as np
    import pyarrow as pa
    import pyarrow.parquet as pq
    from lerobot.utils.constants import DEFAULT_FEATURES

    # ── 1. Validate ────────────────────────────────────────────────────
    if not features:
        raise ValueError("features dict is empty")
    required_keys = {"dtype", "shape"}
    for name, (_, info) in features.items():
        if name in dataset.meta.features:
            raise ValueError(f"Feature '{name}' already exists in dataset")
        if name in DEFAULT_FEATURES:
            raise ValueError(f"Feature '{name}' is a reserved DEFAULT_FEATURE")
        if not required_keys.issubset(info.keys()):
            raise ValueError(f"feature_info for '{name}' must include keys: {required_keys}")
        if not isinstance(info["shape"], (list, tuple)) or not all(isinstance(d, int) and d > 0 for d in info["shape"]):
            raise ValueError(f"feature_info['shape'] for '{name}' must be a list of positive ints")
        if "per_episode" in info and not isinstance(info["per_episode"], bool):
            raise ValueError(f"feature_info['per_episode'] for '{name}' must be a bool")

    root = Path(dataset.root)
    info_path = root / "meta" / "info.json"

    # ── 2. Build new info.json in memory ───────────────────────────────
    with info_path.open("r") as f:
        info_dict = json.load(f)
    for name, (_, finfo) in features.items():
        info_dict["features"][name] = dict(finfo)
    info_dict["total_frames"] = info_dict.get("total_frames", 0)  # unchanged but normalize key
    # codebase_version unchanged — additive op.

    # ── 3. Rewrite parquet shards via .tmp + rename ────────────────────
    data_dir = root / "data"
    parquet_files = sorted(data_dir.rglob("*.parquet"))
    if not parquet_files:
        raise RuntimeError(f"No parquet files found under {data_dir}")

    tmp_targets: list[tuple[Path, Path]] = []  # (tmp, real)
    try:
        for shard in parquet_files:
            table = pq.read_table(shard)
            n_rows = table.num_rows
            new_cols = {}
            for name, (fill, finfo) in features.items():
                if isinstance(fill, np.ndarray):
                    raise NotImplementedError("per-frame fill arrays not yet implemented; pass a scalar")
                shape = tuple(finfo["shape"])
                dtype = finfo["dtype"]
                np_dtype = np.dtype(dtype)
                if shape == (1,):
                    arr = np.full(n_rows, fill, dtype=np_dtype)
                else:
                    arr = np.broadcast_to(np.full(shape, fill, dtype=np_dtype), (n_rows, *shape)).copy()
                    arr = arr.reshape(n_rows, -1)  # parquet stores fixed-size lists
                new_cols[name] = pa.array(arr.tolist() if shape != (1,) else arr.tolist())
            for name, col in new_cols.items():
                table = table.append_column(name, col)
            tmp = shard.with_suffix(shard.suffix + ".tmp")
            pq.write_table(table, tmp)
            tmp_targets.append((tmp, shard))
        for tmp, real in tmp_targets:
            tmp.replace(real)
        tmp_targets.clear()
    finally:
        for tmp, _ in tmp_targets:
            try:
                tmp.unlink(missing_ok=True)
            except Exception:
                pass

    # ── 4. Atomically rewrite info.json ────────────────────────────────
    info_tmp = info_path.with_suffix(info_path.suffix + ".tmp")
    with info_tmp.open("w") as f:
        json.dump(info_dict, f, indent=2)
    info_tmp.replace(info_path)

    # ── 5. Recompute stats for new columns ─────────────────────────────
    if recompute_stats:
        # Reload metadata so stats helpers see the new schema.
        dataset.meta = type(dataset.meta)(repo_id=dataset.repo_id, root=dataset.root)
        for ep_idx in dataset.meta.episodes_stats.keys():
            _recompute_episode_stats_from_data(dataset.root, ep_idx, dataset.meta.features)

    dataset.finalize()
```

Note the imports added at function scope are intentional (keeps top-of-file imports unchanged).

- [ ] **Step 4: Run test, confirm pass**

```bash
uv run pytest tests/datasets/test_dataset_tools.py::test_add_features_inplace_per_frame_reward -svv
```

If the dataset reload step fails, adapt to the actual `dataset.meta` reload pattern in this file (search `dataset.meta = ` in the file for examples).

- [ ] **Step 5: Commit**

```bash
git add src/lerobot/datasets/dataset_tools.py tests/datasets/test_dataset_tools.py
git commit -m "feat(datasets): add_features_inplace — per-frame schema-add without video re-copy"
```

---

## Task 3: `add_features_inplace()` — per-episode `success` with hint preservation

**Files:**

- Test: `tests/datasets/test_dataset_tools.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/datasets/test_dataset_tools.py`:

```python
def test_add_features_inplace_per_episode_success(tmp_path):
    """Per-episode int8 feature: per_episode hint preserved, fill applied."""
    import numpy as np
    import pyarrow.parquet as pq
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from lerobot.datasets.dataset_tools import add_features_inplace

    features = {
        "observation.state": {"dtype": "float32", "shape": [2], "names": None},
        "action": {"dtype": "float32", "shape": [2], "names": None},
    }
    ds = LeRobotDataset.create(
        repo_id="test/add_inplace_success",
        fps=10,
        features=features,
        root=tmp_path / "ds",
        robot_type="dummy",
        use_videos=False,
    )
    for _ in range(3):
        for _ in range(4):
            ds.add_frame({"observation.state": np.zeros(2, dtype=np.float32),
                          "action": np.zeros(2, dtype=np.float32)}, task="t")
        ds.save_episode()
    ds.finalize()

    add_features_inplace(
        ds,
        features={
            "success": (0, {"dtype": "int8", "shape": [1], "names": None, "per_episode": True}),
        },
    )

    ds2 = LeRobotDataset(repo_id="test/add_inplace_success", root=tmp_path / "ds")
    assert ds2.meta.features["success"]["per_episode"] is True
    assert ds2.meta.features["success"]["dtype"] == "int8"

    for f in (tmp_path / "ds" / "data").rglob("*.parquet"):
        table = pq.read_table(f)
        assert "success" in table.column_names
        assert all(v == 0 for v in table.column("success").to_pylist())
```

- [ ] **Step 2: Run test, expect pass on first run** (Task 1 + Task 2 should make this work)

```bash
uv run pytest tests/datasets/test_dataset_tools.py::test_add_features_inplace_per_episode_success -svv
```

If FAIL: most likely `per_episode` is being dropped during info-dict copy. Fix in Task 1's code path.

- [ ] **Step 3: Commit**

```bash
git add tests/datasets/test_dataset_tools.py
git commit -m "test(datasets): per-episode int8 feature add preserves per_episode hint"
```

---

## Task 4: Validation rejections

**Files:**

- Test: `tests/datasets/test_dataset_tools.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/datasets/test_dataset_tools.py`:

```python
import pytest

def _build_minimal_ds(tmp_path):
    import numpy as np
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    features = {
        "observation.state": {"dtype": "float32", "shape": [2], "names": None},
        "action": {"dtype": "float32", "shape": [2], "names": None},
    }
    ds = LeRobotDataset.create(
        repo_id="test/add_inplace_validation",
        fps=10,
        features=features,
        root=tmp_path / "ds",
        robot_type="dummy",
        use_videos=False,
    )
    for _ in range(2):
        for _ in range(3):
            ds.add_frame({"observation.state": np.zeros(2, dtype=np.float32),
                          "action": np.zeros(2, dtype=np.float32)}, task="t")
        ds.save_episode()
    ds.finalize()
    return ds


@pytest.mark.parametrize("name,info,fragment", [
    ("action", {"dtype": "float32", "shape": [1], "names": None}, "already exists"),
    ("timestamp", {"dtype": "float32", "shape": [1], "names": None}, "DEFAULT_FEATURE"),
    ("ok_name", {"dtype": "float32"}, "must include keys"),               # missing shape
    ("ok_name", {"dtype": "float32", "shape": []}, "positive ints"),
    ("ok_name", {"dtype": "float32", "shape": [1], "per_episode": "yes"}, "must be a bool"),
])
def test_add_features_inplace_validation(tmp_path, name, info, fragment):
    from lerobot.datasets.dataset_tools import add_features_inplace
    ds = _build_minimal_ds(tmp_path)
    with pytest.raises(ValueError, match=fragment):
        add_features_inplace(ds, features={name: (0.0, info)})


def test_add_features_inplace_empty_dict_rejected(tmp_path):
    from lerobot.datasets.dataset_tools import add_features_inplace
    ds = _build_minimal_ds(tmp_path)
    with pytest.raises(ValueError, match="empty"):
        add_features_inplace(ds, features={})
```

- [ ] **Step 2: Run test, expect pass on first run** (Task 2 already implemented these checks)

```bash
uv run pytest tests/datasets/test_dataset_tools.py -svv -k "validation or empty_dict_rejected"
```

If any case FAILs, tighten the validation block in `add_features_inplace`.

- [ ] **Step 3: Commit**

```bash
git add tests/datasets/test_dataset_tools.py
git commit -m "test(datasets): add_features_inplace validation rejections"
```

---

## Task 5: Stats recomputation for new columns

**Files:**

- Test: `tests/datasets/test_dataset_tools.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/datasets/test_dataset_tools.py`:

```python
def test_add_features_inplace_recomputes_stats(tmp_path):
    """After add, stats columns for the new feature exist in episodes parquet."""
    import numpy as np
    import pyarrow.parquet as pq
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from lerobot.datasets.dataset_tools import add_features_inplace

    features = {
        "observation.state": {"dtype": "float32", "shape": [2], "names": None},
        "action": {"dtype": "float32", "shape": [2], "names": None},
    }
    ds = LeRobotDataset.create(
        repo_id="test/add_inplace_stats",
        fps=10,
        features=features,
        root=tmp_path / "ds",
        robot_type="dummy",
        use_videos=False,
    )
    for _ in range(2):
        for _ in range(4):
            ds.add_frame({"observation.state": np.zeros(2, dtype=np.float32),
                          "action": np.zeros(2, dtype=np.float32)}, task="t")
        ds.save_episode()
    ds.finalize()

    add_features_inplace(
        ds,
        features={"reward": (0.0, {"dtype": "float32", "shape": [1], "names": None})},
    )

    # Episodes parquet should now have stats columns for reward.
    eps_dir = tmp_path / "ds" / "meta" / "episodes"
    eps_files = list(eps_dir.rglob("*.parquet"))
    assert eps_files, "no episodes metadata files found"
    table = pq.read_table(eps_files[0])
    cols = set(table.column_names)
    # Stats columns follow the pattern stats/<feature>/{min,max,mean,std,...}
    reward_stat_cols = [c for c in cols if c.startswith("stats/reward/")]
    assert reward_stat_cols, f"no stats/reward/* columns in {cols}"
```

- [ ] **Step 2: Run test**

```bash
uv run pytest tests/datasets/test_dataset_tools.py::test_add_features_inplace_recomputes_stats -svv
```

- [ ] **Step 3: If FAIL, debug the stats path**

The recompute helper signature is `_recompute_episode_stats_from_data(root, episode_index, features)`. Verify:

1. `dataset.meta.episodes_stats` returns a mapping keyed by episode index. If not, find the right iterator (search the file for usages of `_recompute_episode_stats_from_data`).
2. The reload step (`dataset.meta = ...`) actually picks up the updated `info.json`. If the meta class doesn't take a fresh-load constructor, use the same pattern as elsewhere in `dataset_tools.py`.

Iterate until stats columns appear.

- [ ] **Step 4: Commit**

```bash
git add src/lerobot/datasets/dataset_tools.py tests/datasets/test_dataset_tools.py
git commit -m "feat(datasets): recompute per-episode stats after add_features_inplace"
```

---

## Task 6: Orphan `.tmp` cleanup helper

**Files:**

- Modify: `src/lerobot/datasets/dataset_tools.py`
- Test: `tests/datasets/test_dataset_tools.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/datasets/test_dataset_tools.py`:

```python
def test_sweep_orphan_tmp_shards(tmp_path):
    """Orphan .tmp files left from a crashed save are cleaned up."""
    from lerobot.datasets.dataset_tools import _sweep_orphan_tmp_shards

    # Build a fake data tree.
    data_dir = tmp_path / "ds" / "data" / "chunk-000"
    data_dir.mkdir(parents=True)
    (data_dir / "file-000.parquet").write_text("not real parquet")
    (data_dir / "file-000.parquet.tmp").write_text("orphan")
    (data_dir / "file-001.parquet.tmp").write_text("orphan")
    info_dir = tmp_path / "ds" / "meta"
    info_dir.mkdir(parents=True)
    (info_dir / "info.json").write_text("{}")
    (info_dir / "info.json.tmp").write_text("orphan")

    removed = _sweep_orphan_tmp_shards(tmp_path / "ds")
    assert removed == 3
    assert not list((tmp_path / "ds").rglob("*.tmp"))
    # Real files untouched.
    assert (data_dir / "file-000.parquet").exists()
    assert (info_dir / "info.json").exists()
```

- [ ] **Step 2: Run test, confirm failure**

```bash
uv run pytest tests/datasets/test_dataset_tools.py::test_sweep_orphan_tmp_shards -svv
```

- [ ] **Step 3: Implement the helper**

In `src/lerobot/datasets/dataset_tools.py`, near `add_features_inplace`:

```python
def _sweep_orphan_tmp_shards(dataset_root: Path) -> int:
    """Delete `.tmp` siblings under `data/` and `meta/` left by a crashed save.

    Returns count of files removed. Safe to call on dataset open.
    """
    root = Path(dataset_root)
    removed = 0
    for sub in ("data", "meta"):
        sub_dir = root / sub
        if not sub_dir.exists():
            continue
        for tmp in sub_dir.rglob("*.tmp"):
            try:
                tmp.unlink()
                removed += 1
            except OSError:
                pass
    return removed
```

- [ ] **Step 4: Run the test, confirm pass**

```bash
uv run pytest tests/datasets/test_dataset_tools.py::test_sweep_orphan_tmp_shards -svv
```

- [ ] **Step 5: Commit**

```bash
git add src/lerobot/datasets/dataset_tools.py tests/datasets/test_dataset_tools.py
git commit -m "feat(datasets): _sweep_orphan_tmp_shards for crash-recovery cleanup"
```

---

## Task 7: GUI endpoint `POST /api/datasets/{id}/features`

**Files:**

- Modify: `src/lerobot/gui/api/datasets.py`
- Modify: `src/lerobot/gui/api/models.py` (request body model)
- Test: `tests/gui/test_feature_add_endpoints.py` (new)

- [ ] **Step 1: Read existing endpoint patterns**

```bash
grep -n "@router.post\|@router\\.get\|class.*Request\|class.*Response" src/lerobot/gui/api/datasets.py | head -30
```

Skim one existing POST endpoint top-to-bottom to absorb error handling, locking, and response style. Mirror it.

- [ ] **Step 2: Write the failing test (new file)**

Create `tests/gui/test_feature_add_endpoints.py`. Look at the imports and fixture usage in `tests/gui/test_feature_endpoints.py` to find the existing FastAPI TestClient + dataset fixture pattern, then:

```python
"""Tests for POST /api/datasets/{id}/features and .../features/defaults."""
import numpy as np
import pytest
from fastapi.testclient import TestClient

# Reuse the existing dataset fixture from test_feature_endpoints.py
# (or its conftest) — check that file for import pattern, then mirror.

def test_post_features_adds_per_frame_column(client: TestClient, dataset_id: str):
    """POST /features adds a new column to the dataset schema."""
    resp = client.post(
        f"/api/datasets/{dataset_id}/features",
        json={
            "name": "custom_metric",
            "dtype": "float32",
            "shape": [1],
            "per_episode": False,
            "fill_value": 0.0,
        },
    )
    assert resp.status_code == 200, resp.text
    payload = resp.json()
    assert payload["added"] == ["custom_metric"]

    # Schema response now includes the new feature.
    info = client.get(f"/api/datasets/{dataset_id}").json()
    assert "custom_metric" in info["features_schema"]


def test_post_features_rejects_default_feature_name(client: TestClient, dataset_id: str):
    resp = client.post(
        f"/api/datasets/{dataset_id}/features",
        json={"name": "timestamp", "dtype": "float32", "shape": [1], "per_episode": False, "fill_value": 0.0},
    )
    assert resp.status_code == 400
    assert "default" in resp.json()["detail"].lower() or "reserved" in resp.json()["detail"].lower()


def test_post_features_rejects_existing_name(client: TestClient, dataset_id: str):
    resp = client.post(
        f"/api/datasets/{dataset_id}/features",
        json={"name": "action", "dtype": "float32", "shape": [1], "per_episode": False, "fill_value": 0.0},
    )
    assert resp.status_code == 400
    assert "already exists" in resp.json()["detail"]
```

- [ ] **Step 3: Run test, confirm failure**

```bash
uv run pytest tests/gui/test_feature_add_endpoints.py -svv
```

Expected: FAIL — endpoint doesn't exist; possibly fixture-name mismatch (adapt to local fixtures).

- [ ] **Step 4: Add request model**

In `src/lerobot/gui/api/models.py`, append:

```python
class AddFeatureRequest(BaseModel):
    """Body for POST /api/datasets/{id}/features."""
    name: str
    dtype: str
    shape: list[int] = [1]
    per_episode: bool = False
    fill_value: Any = 0  # auto-typed by add_features_inplace from dtype

class AddFeatureResponse(BaseModel):
    added: list[str]
```

(Add `from typing import Any` at top if not present.)

- [ ] **Step 5: Implement the endpoint**

In `src/lerobot/gui/api/datasets.py`, add:

```python
@router.post("/{dataset_id}/features", response_model=AddFeatureResponse)
def add_feature(dataset_id: str, body: AddFeatureRequest, _state: AppState = Depends(get_state)):
    """Add one new feature column to the dataset in place."""
    from lerobot.datasets.dataset_tools import add_features_inplace
    from lerobot.utils.constants import DEFAULT_FEATURES

    if body.name in DEFAULT_FEATURES:
        raise HTTPException(400, f"'{body.name}' is a reserved DEFAULT_FEATURE")

    with _state.get_lock(dataset_id):
        ds = _state.get_open_dataset(dataset_id)
        if ds is None:
            raise HTTPException(404, f"Dataset {dataset_id} not open")
        if body.name in ds.meta.features:
            raise HTTPException(400, f"Feature '{body.name}' already exists in dataset")

        info = {"dtype": body.dtype, "shape": list(body.shape), "names": None}
        if body.per_episode:
            info["per_episode"] = True

        try:
            add_features_inplace(ds, features={body.name: (body.fill_value, info)})
        except ValueError as e:
            raise HTTPException(400, str(e))

        # Cache invalidation.
        _per_episode_features_cache.pop(dataset_id, None)
        _episode_start_indices.pop(dataset_id, None)
        # frame_cache + DatasetInfo mtime invalidation:
        _invalidate_dataset_info_cache(dataset_id)  # use existing helper; if name differs, search file
        # (frame cache: see existing pattern in close_dataset / similar)

    return AddFeatureResponse(added=[body.name])
```

Match local helper names — search `src/lerobot/gui/api/datasets.py` for `pop(dataset_id, None)` and similar to see what's already there. The exact dependency/state injection pattern (`Depends(get_state)`) may differ — mirror an existing endpoint.

- [ ] **Step 6: Run test, confirm pass**

```bash
uv run pytest tests/gui/test_feature_add_endpoints.py -svv
```

Iterate on fixture names / state-injection style until green.

- [ ] **Step 7: Commit**

```bash
git add src/lerobot/gui/api/datasets.py src/lerobot/gui/api/models.py tests/gui/test_feature_add_endpoints.py
git commit -m "feat(gui): POST /api/datasets/{id}/features — in-place schema add"
```

---

## Task 8: GUI endpoint `POST /api/datasets/{id}/features/defaults`

**Files:**

- Modify: `src/lerobot/gui/api/datasets.py`
- Test: `tests/gui/test_feature_add_endpoints.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/gui/test_feature_add_endpoints.py`:

```python
def test_defaults_adds_missing_reward_and_success(client: TestClient, dataset_id: str):
    """When both are missing, POST /features/defaults adds both."""
    resp = client.post(f"/api/datasets/{dataset_id}/features/defaults")
    assert resp.status_code == 200, resp.text
    assert sorted(resp.json()["added"]) == ["reward", "success"]

    info = client.get(f"/api/datasets/{dataset_id}").json()
    assert "reward" in info["features_schema"]
    assert "success" in info["features_schema"]
    assert info["features_schema"]["success"].get("per_episode") is True
    assert info["features_schema"]["success"]["dtype"] == "int8"
    assert info["features_schema"]["reward"]["dtype"] == "float32"


def test_defaults_idempotent_when_present(client: TestClient, dataset_id: str):
    """Calling defaults twice doesn't error; second call adds nothing."""
    client.post(f"/api/datasets/{dataset_id}/features/defaults")
    resp2 = client.post(f"/api/datasets/{dataset_id}/features/defaults")
    assert resp2.status_code == 200
    assert resp2.json()["added"] == []
```

- [ ] **Step 2: Run, confirm fail**

```bash
uv run pytest tests/gui/test_feature_add_endpoints.py -svv -k "defaults"
```

- [ ] **Step 3: Implement**

In `src/lerobot/gui/api/datasets.py`:

```python
DEFAULT_FEATURE_SPECS = {
    "reward": {
        "fill_value": 0.0,
        "info": {"dtype": "float32", "shape": [1], "names": None},
    },
    "success": {
        "fill_value": 0,
        "info": {"dtype": "int8", "shape": [1], "names": None, "per_episode": True},
    },
}


@router.post("/{dataset_id}/features/defaults", response_model=AddFeatureResponse)
def add_default_features(dataset_id: str, _state: AppState = Depends(get_state)):
    """Add `reward` and/or `success` if either is missing from the schema."""
    from lerobot.datasets.dataset_tools import add_features_inplace

    with _state.get_lock(dataset_id):
        ds = _state.get_open_dataset(dataset_id)
        if ds is None:
            raise HTTPException(404, f"Dataset {dataset_id} not open")

        to_add = {
            name: (spec["fill_value"], spec["info"])
            for name, spec in DEFAULT_FEATURE_SPECS.items()
            if name not in ds.meta.features
        }
        if not to_add:
            return AddFeatureResponse(added=[])

        try:
            add_features_inplace(ds, features=to_add)
        except ValueError as e:
            raise HTTPException(400, str(e))

        _per_episode_features_cache.pop(dataset_id, None)
        _episode_start_indices.pop(dataset_id, None)
        _invalidate_dataset_info_cache(dataset_id)

    return AddFeatureResponse(added=sorted(to_add.keys()))
```

- [ ] **Step 4: Run, confirm pass**

```bash
uv run pytest tests/gui/test_feature_add_endpoints.py -svv -k "defaults"
```

- [ ] **Step 5: Commit**

```bash
git add src/lerobot/gui/api/datasets.py tests/gui/test_feature_add_endpoints.py
git commit -m "feat(gui): POST /features/defaults adds missing reward/success"
```

---

## Task 9: Pending-edits guard

**Files:**

- Modify: `src/lerobot/gui/state.py` (helper)
- Modify: `src/lerobot/gui/api/datasets.py` (use the helper in both endpoints)
- Test: `tests/gui/test_state.py`, `tests/gui/test_feature_add_endpoints.py`

- [ ] **Step 1: Write the state helper test**

Append to `tests/gui/test_state.py`:

```python
def test_pending_feature_set_edits_for_dataset():
    from lerobot.gui.state import AppState, PendingEdit

    state = AppState()
    state.add_edit(PendingEdit(
        edit_type="feature_set",
        dataset_id="ds1",
        episode_index=0,
        params={"feature": "reward", "frame_from": 0, "frame_to": 5, "value": 1.0},
    ))
    state.add_edit(PendingEdit(edit_type="trim", dataset_id="ds1", episode_index=0, params={}))
    assert len(state.pending_feature_set_edits_for_dataset("ds1")) == 1
    assert state.pending_feature_set_edits_for_dataset("ds2") == []
```

- [ ] **Step 2: Run, confirm fail**

```bash
uv run pytest tests/gui/test_state.py -svv -k "pending_feature_set"
```

- [ ] **Step 3: Implement helper in `src/lerobot/gui/state.py`**

Add method to `AppState`:

```python
def pending_feature_set_edits_for_dataset(self, dataset_id: str) -> list[PendingEdit]:
    return [
        e for e in self.pending_edits
        if e.dataset_id == dataset_id and e.edit_type == "feature_set"
    ]
```

- [ ] **Step 4: Write the endpoint guard test**

Append to `tests/gui/test_feature_add_endpoints.py`:

```python
def test_post_features_blocked_by_pending_feature_edits(client: TestClient, dataset_id: str, app_state):
    """Schema add is blocked when there are pending feature_set edits on the dataset."""
    from lerobot.gui.state import PendingEdit
    app_state.add_edit(PendingEdit(
        edit_type="feature_set",
        dataset_id=dataset_id,
        episode_index=0,
        params={"feature": "action", "frame_from": 0, "frame_to": 1, "value": [0.0, 0.0]},
    ))
    try:
        resp = client.post(
            f"/api/datasets/{dataset_id}/features",
            json={"name": "x", "dtype": "float32", "shape": [1], "per_episode": False, "fill_value": 0.0},
        )
        assert resp.status_code == 409
        assert "pending" in resp.json()["detail"].lower()
    finally:
        app_state.pending_edits.clear()
```

- [ ] **Step 5: Run, confirm fail (409 not yet wired)**

```bash
uv run pytest tests/gui/test_feature_add_endpoints.py -svv -k "blocked_by_pending"
```

- [ ] **Step 6: Add guard to both endpoints**

In `src/lerobot/gui/api/datasets.py`, top of both `add_feature` and `add_default_features`:

```python
pending = _state.pending_feature_set_edits_for_dataset(dataset_id)
if pending:
    raise HTTPException(409, f"{len(pending)} pending feature edits — Save or Discard them first")
```

- [ ] **Step 7: Run all add-endpoint tests, confirm pass**

```bash
uv run pytest tests/gui/test_feature_add_endpoints.py tests/gui/test_state.py -svv
```

- [ ] **Step 8: Commit**

```bash
git add src/lerobot/gui/state.py src/lerobot/gui/api/datasets.py tests/gui/test_state.py tests/gui/test_feature_add_endpoints.py
git commit -m "feat(gui): block schema add when pending feature_set edits exist"
```

---

## Task 10: `dataset.schema_changed` WS event

**Files:**

- Modify: `src/lerobot/gui/api/playback.py` (or wherever WS broadcasting lives)
- Modify: `src/lerobot/gui/api/datasets.py` (call broadcaster)
- Test: `tests/gui/test_feature_add_endpoints.py`

- [ ] **Step 1: Find the WS broadcasting helper**

```bash
grep -rn "broadcast\|websocket\|ws_send\|connected" src/lerobot/gui/api/playback.py src/lerobot/gui/server.py | head -20
```

Note the function used to send a message to all connected clients for a dataset. Common name: `_broadcast(dataset_id, message)` or similar.

- [ ] **Step 2: Write the failing test**

Append to `tests/gui/test_feature_add_endpoints.py`:

```python
def test_post_features_emits_schema_changed_event(client: TestClient, dataset_id: str):
    """A successful add publishes a schema_changed message on the dataset's WS channel."""
    from unittest.mock import patch

    with patch("lerobot.gui.api.datasets.broadcast_to_dataset") as mock_bcast:
        resp = client.post(
            f"/api/datasets/{dataset_id}/features",
            json={"name": "tcustom", "dtype": "float32", "shape": [1], "per_episode": False, "fill_value": 0.0},
        )
        assert resp.status_code == 200
        # At least one call mentioning schema_changed.
        events = [c.args[1] for c in mock_bcast.call_args_list if dataset_id in c.args]
        assert any(isinstance(e, dict) and e.get("type") == "dataset.schema_changed" for e in events), events
```

(Adjust `broadcast_to_dataset` to the actual module-qualified name found in step 1.)

- [ ] **Step 3: Run, confirm fail**

```bash
uv run pytest tests/gui/test_feature_add_endpoints.py -svv -k "schema_changed"
```

- [ ] **Step 4: Implement broadcast call**

In `src/lerobot/gui/api/datasets.py`, after the lock block in BOTH `add_feature` and `add_default_features`, add:

```python
from lerobot.gui.api.playback import broadcast_to_dataset  # or actual module
broadcast_to_dataset(dataset_id, {"type": "dataset.schema_changed"})
```

If no such broadcaster exists, add a thin one in `playback.py`:

```python
def broadcast_to_dataset(dataset_id: str, message: dict) -> None:
    """Send a JSON message to all WS clients subscribed to this dataset."""
    # Reuse the existing connection registry; iterate active connections
    # for `dataset_id` and call `await ws.send_json(message)` from a sync
    # wrapper using `asyncio.run_coroutine_threadsafe` against the FastAPI loop.
    ...  # mirror existing patterns; do not invent a new event loop
```

- [ ] **Step 5: Run, confirm pass**

```bash
uv run pytest tests/gui/test_feature_add_endpoints.py -svv -k "schema_changed"
```

- [ ] **Step 6: Commit**

```bash
git add src/lerobot/gui/api/datasets.py src/lerobot/gui/api/playback.py tests/gui/test_feature_add_endpoints.py
git commit -m "feat(gui): emit dataset.schema_changed WS event on schema add"
```

---

## Task 11: Surface declared `per_episode` in `FeatureSchema`

**Files:**

- Modify: `src/lerobot/gui/api/models.py` (verify `per_episode` field is populated from declared spec, not just inference)
- Modify: `src/lerobot/gui/api/datasets.py` (FeatureSchema construction)
- Test: `tests/gui/test_feature_endpoints.py`

- [ ] **Step 1: Read current FeatureSchema construction**

```bash
grep -n "FeatureSchema\|per_episode" src/lerobot/gui/api/models.py src/lerobot/gui/api/datasets.py | head -30
```

Find where `FeatureSchema(per_episode=...)` is built. Today it's set from inference (`_detect_per_episode_features`). Need: declared hint wins; inference is fallback.

- [ ] **Step 2: Write failing test**

Append to `tests/gui/test_feature_endpoints.py`:

```python
def test_feature_schema_uses_declared_per_episode_hint(client: TestClient, dataset_id: str):
    """A feature with per_episode=true in info.json reports per_episode=True in the schema."""
    # Add via the new endpoint to ensure a declared per_episode feature exists.
    resp = client.post(
        f"/api/datasets/{dataset_id}/features",
        json={"name": "pe_flag", "dtype": "bool", "shape": [1], "per_episode": True, "fill_value": False},
    )
    assert resp.status_code == 200, resp.text
    info = client.get(f"/api/datasets/{dataset_id}").json()
    assert info["features_schema"]["pe_flag"]["per_episode"] is True
```

- [ ] **Step 3: Run, confirm pass or fail**

```bash
uv run pytest tests/gui/test_feature_endpoints.py -svv -k "declared_per_episode"
```

- [ ] **Step 4: Fix `FeatureSchema` construction if needed**

In `src/lerobot/gui/api/datasets.py` where `FeatureSchema` is built, change to:

```python
declared = bool(feat_info.get("per_episode", False))
inferred = name in per_episode_set  # from _detect_per_episode_features
per_episode = declared or inferred
schema = FeatureSchema(..., per_episode=per_episode)
```

- [ ] **Step 5: Run all gui tests**

```bash
uv run pytest tests/gui -svv -k "feature"
```

- [ ] **Step 6: Commit**

```bash
git add src/lerobot/gui/api/datasets.py tests/gui/test_feature_endpoints.py
git commit -m "feat(gui): declared per_episode hint wins over inference in FeatureSchema"
```

---

## Task 12: Frontend banner for missing default features

**Files:**

- Modify: `src/lerobot/gui/static/feature_editing.js`
- Modify: `src/lerobot/gui/static/index.html` (banner DOM slot)
- Modify: `src/lerobot/gui/static/style.css` (banner styling)

(No JS unit tests in this codebase — manual smoke test required.)

- [ ] **Step 1: Add banner DOM slot**

In `src/lerobot/gui/static/index.html`, just inside the data-tab container, before `#feature-rows`:

```html
<div id="default-features-banner" class="default-features-banner" hidden>
  <span class="banner-icon">⚠️</span>
  <span class="banner-text">
    This dataset is missing default features:
    <strong id="banner-missing-list"></strong>.
  </span>
  <button id="banner-add-btn" class="btn btn-primary">
    Add missing features
  </button>
  <button id="banner-dismiss-btn" class="btn btn-secondary">Dismiss</button>
</div>
```

- [ ] **Step 2: Add banner styles**

Append to `src/lerobot/gui/static/style.css`:

```css
.default-features-banner {
  display: flex;
  align-items: center;
  gap: 12px;
  padding: 10px 14px;
  background: #fff8e1;
  border: 1px solid #f0c36d;
  border-radius: 6px;
  margin: 0 0 8px 0;
  font-size: 13px;
}
.default-features-banner .banner-icon {
  font-size: 16px;
}
.default-features-banner .banner-text {
  flex: 1;
}
.default-features-banner button {
  white-space: nowrap;
}
```

- [ ] **Step 3: Wire banner logic in `feature_editing.js`**

Add module-level state at top:

```javascript
const bannerDismissed = new Set(); // dataset_ids dismissed for this session
```

In `onDatasetOpened(datasetId)`, after the existing `_log` line, add:

```javascript
maybeShowDefaultsBanner(datasetId);
```

In `onDatasetClosed(datasetId)`, add:

```javascript
bannerDismissed.delete(datasetId);
hideDefaultsBanner();
```

Add the helpers (anywhere in the module IIFE, before the `}())`):

```javascript
function maybeShowDefaultsBanner(datasetId) {
  const ds = window.datasets && window.datasets[datasetId];
  const fs = (ds && ds.features_schema) || {};
  const missing = ["reward", "success"].filter((n) => !fs[n]);
  const banner = document.getElementById("default-features-banner");
  if (!banner) return;
  if (missing.length === 0 || bannerDismissed.has(datasetId)) {
    banner.hidden = true;
    return;
  }
  document.getElementById("banner-missing-list").textContent =
    missing.join(", ");
  banner.hidden = false;
  banner.dataset.datasetId = datasetId;
  document.getElementById("banner-add-btn").onclick = () =>
    addDefaultsFor(datasetId);
  document.getElementById("banner-dismiss-btn").onclick = () => {
    bannerDismissed.add(datasetId);
    banner.hidden = true;
  };
}

function hideDefaultsBanner() {
  const banner = document.getElementById("default-features-banner");
  if (banner) banner.hidden = true;
}

async function addDefaultsFor(datasetId) {
  const btn = document.getElementById("banner-add-btn");
  btn.disabled = true;
  btn.textContent = "Adding…";
  try {
    const r = await fetch(
      `/api/datasets/${encodeURIComponent(datasetId)}/features/defaults`,
      { method: "POST" },
    );
    if (!r.ok) {
      const detail = (await r.json().catch(() => ({}))).detail || r.statusText;
      alert(`Failed to add defaults: ${detail}`);
      return;
    }
    // schema_changed WS event will trigger schema reload + re-render.
    // Also force-refresh schema here in case WS is slow.
    if (typeof window.refreshDatasetInfo === "function") {
      await window.refreshDatasetInfo(datasetId);
    }
    hideDefaultsBanner();
  } finally {
    btn.disabled = false;
    btn.textContent = "Add missing features";
  }
}
```

- [ ] **Step 4: Bump cache-busting version in `index.html`**

```bash
grep -n "feature_editing.js?v=" src/lerobot/gui/static/index.html
```

Increment the `?v=NN` query string by one (e.g. `v=16` → `v=17`) so browsers reload the file.

- [ ] **Step 5: Manual smoke test**

```bash
uv run python -m lerobot.gui --port 8001
```

Open `http://127.0.0.1:8001`, open `KeWangRobotics/sim_pick` (lacks reward + success). Verify: banner appears with both names; click Dismiss → hides; reload page → reappears. Click Add → progress, then banner disappears, rows for `reward` and `success` show in the feature column.

- [ ] **Step 6: Commit**

```bash
git add src/lerobot/gui/static/feature_editing.js src/lerobot/gui/static/index.html src/lerobot/gui/static/style.css
git commit -m "feat(gui): banner offering to add missing reward/success defaults on dataset open"
```

---

## Task 13: "+ Add feature" button + dialog modal

**Files:**

- Create: `src/lerobot/gui/static/add_feature_dialog.js`
- Modify: `src/lerobot/gui/static/feature_editing.js` (mount the button)
- Modify: `src/lerobot/gui/static/index.html` (script tag, dialog DOM template)
- Modify: `src/lerobot/gui/static/style.css` (dialog styles)

- [ ] **Step 1: Add dialog DOM template to `index.html`**

Inside `<body>`, near other modals:

```html
<dialog id="add-feature-dialog" class="add-feature-dialog">
  <form method="dialog" id="add-feature-form">
    <h3>Add feature</h3>
    <label
      >Name <input name="name" required pattern="[a-zA-Z_][a-zA-Z0-9_.]*"
    /></label>
    <label
      >Dtype
      <select name="dtype">
        <option value="float32">float32</option>
        <option value="int64">int64</option>
        <option value="int8">int8</option>
        <option value="bool">bool</option>
        <option value="string">string</option>
      </select>
    </label>
    <label>Shape <input name="shape" value="[1]" /></label>
    <label
      ><input type="checkbox" name="per_episode" /> Per-episode (one value per
      episode)</label
    >
    <label>Initial fill value <input name="fill_value" value="0" /></label>
    <div class="dialog-error" id="add-feature-error" hidden></div>
    <menu>
      <button value="cancel">Cancel</button>
      <button id="add-feature-submit" value="default">Add</button>
    </menu>
  </form>
</dialog>
```

Add the script tag (next to the existing `feature_editing.js` line):

```html
<script src="/static/add_feature_dialog.js?v=1"></script>
```

- [ ] **Step 2: Add dialog styles**

Append to `src/lerobot/gui/static/style.css`:

```css
.add-feature-dialog {
  padding: 16px 20px;
  min-width: 360px;
  border-radius: 8px;
  border: 1px solid #ccc;
}
.add-feature-dialog form > label {
  display: block;
  margin: 8px 0;
  font-size: 13px;
}
.add-feature-dialog input[type="text"],
.add-feature-dialog input:not([type]),
.add-feature-dialog select {
  width: 100%;
  padding: 4px 6px;
}
.add-feature-dialog menu {
  display: flex;
  gap: 8px;
  justify-content: flex-end;
  padding: 0;
  margin: 12px 0 0 0;
}
.dialog-error {
  color: #b00;
  font-size: 12px;
  margin-top: 6px;
}
```

- [ ] **Step 3: Implement `add_feature_dialog.js`**

Create `src/lerobot/gui/static/add_feature_dialog.js`:

```javascript
// Generic "Add feature" dialog — for non-default custom features only.
(function () {
  "use strict";

  const dialog = () => document.getElementById("add-feature-dialog");
  const form = () => document.getElementById("add-feature-form");
  const errBox = () => document.getElementById("add-feature-error");

  const DTYPE_DEFAULTS = {
    float32: "0",
    int64: "0",
    int8: "0",
    bool: "false",
    string: "",
  };

  function resetForm() {
    const f = form();
    f.reset();
    f.shape.value = "[1]";
    f.fill_value.value = DTYPE_DEFAULTS.float32;
    errBox().hidden = true;
  }

  function autoUpdateFill() {
    const f = form();
    const dtype = f.dtype.value;
    const isPerEpisodeBool = f.per_episode.checked && dtype === "bool";
    f.fill_value.value = isPerEpisodeBool
      ? "true"
      : (DTYPE_DEFAULTS[dtype] ?? "0");
  }

  function parseFillValue(raw, dtype) {
    if (dtype === "bool") return raw.trim().toLowerCase() === "true";
    if (dtype === "string") return raw;
    if (dtype === "float32") return parseFloat(raw);
    return parseInt(raw, 10);
  }

  function parseShape(raw) {
    const trimmed = raw.trim();
    if (!trimmed.startsWith("[") || !trimmed.endsWith("]")) {
      throw new Error("Shape must be a JSON list, e.g. [1]");
    }
    const arr = JSON.parse(trimmed);
    if (
      !Array.isArray(arr) ||
      !arr.every((n) => Number.isInteger(n) && n > 0)
    ) {
      throw new Error("Shape must be a list of positive ints");
    }
    return arr;
  }

  async function submit(e) {
    e.preventDefault();
    errBox().hidden = true;
    const f = form();
    const datasetId = window.currentDataset;
    if (!datasetId) {
      errBox().textContent = "No dataset open";
      errBox().hidden = false;
      return;
    }
    const name = f.name.value.trim();
    if (name === "reward" || name === "success") {
      errBox().textContent = `'${name}' is a default feature — use the banner instead.`;
      errBox().hidden = false;
      return;
    }
    let shape, fillValue;
    try {
      shape = parseShape(f.shape.value);
      fillValue = parseFillValue(f.fill_value.value, f.dtype.value);
    } catch (err) {
      errBox().textContent = err.message;
      errBox().hidden = false;
      return;
    }
    const body = {
      name,
      dtype: f.dtype.value,
      shape,
      per_episode: f.per_episode.checked,
      fill_value: fillValue,
    };
    const r = await fetch(
      `/api/datasets/${encodeURIComponent(datasetId)}/features`,
      {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
      },
    );
    if (!r.ok) {
      const detail = (await r.json().catch(() => ({}))).detail || r.statusText;
      errBox().textContent = `Add failed: ${detail}`;
      errBox().hidden = false;
      return;
    }
    if (typeof window.refreshDatasetInfo === "function") {
      await window.refreshDatasetInfo(datasetId);
    }
    dialog().close();
  }

  function open() {
    resetForm();
    const d = dialog();
    if (!d) {
      console.warn("[add-feature] dialog DOM not present");
      return;
    }
    d.showModal();
  }

  document.addEventListener("DOMContentLoaded", () => {
    const f = form();
    if (!f) return;
    f.dtype.addEventListener("change", autoUpdateFill);
    f.per_episode.addEventListener("change", autoUpdateFill);
    f.addEventListener("submit", submit);
  });

  window.AddFeatureDialog = { open };
})();
```

- [ ] **Step 4: Mount the "+ Add feature" button**

In `feature_editing.js`, in the function that renders feature rows (search for `renderFeatureRows`), append a final row:

```javascript
const addBtnRow = document.createElement("div");
addBtnRow.className = "feature-row feature-row-addbtn";
addBtnRow.innerHTML = `<button class="feature-add-btn">+ Add feature</button>`;
addBtnRow.querySelector("button").onclick = () =>
  window.AddFeatureDialog && window.AddFeatureDialog.open();
container.appendChild(addBtnRow);
```

(Adjust selector / variable names — `container` is whatever the existing render uses; mirror the existing append pattern.)

- [ ] **Step 5: Manual smoke test**

```bash
uv run python -m lerobot.gui --port 8001
```

Open dataset → click "+ Add feature" → dialog opens → fill in name=`my_metric`, dtype=`float32` → Add → row appears with line plot. Try name=`reward` → in-dialog error message; submit blocked.

- [ ] **Step 6: Commit**

```bash
git add src/lerobot/gui/static/add_feature_dialog.js src/lerobot/gui/static/feature_editing.js src/lerobot/gui/static/index.html src/lerobot/gui/static/style.css
git commit -m "feat(gui): + Add feature dialog for generic custom column add"
```

---

## Task 14: Success widget renderer (tri-state segment control)

**Files:**

- Modify: `src/lerobot/gui/static/feature_editing.js`
- Modify: `src/lerobot/gui/static/style.css`

- [ ] **Step 1: Add segment-control styles**

Append to `src/lerobot/gui/static/style.css`:

```css
.success-segment {
  display: inline-flex;
  gap: 0;
  border: 1px solid #ccc;
  border-radius: 6px;
  overflow: hidden;
}
.success-segment button {
  border: 0;
  background: transparent;
  padding: 6px 14px;
  font-size: 13px;
  cursor: pointer;
}
.success-segment button + button {
  border-left: 1px solid #ccc;
}
.success-segment button.active {
  background: #e0e0e0;
  font-weight: 600;
}
.success-segment button.failure.active {
  background: #fbe0e0;
  color: #b00;
}
.success-segment button.success.active {
  background: #e0f5e0;
  color: #0a0;
}
.feature-row-band-success-pos {
  background: #cfe9cf;
}
.feature-row-band-success-neg {
  background: #f5cfcf;
}
.feature-row-band-success-zero {
  background: #e6e6e6;
}
```

- [ ] **Step 2: Register the success widget renderer**

In `feature_editing.js`, find the renderer registry (search for the pattern that maps `(dtype, shape)` → render function — likely a map of widget builders). Add a special case at the top of the dispatch:

```javascript
function renderInspectorWidget(name, schema, valuesInRange) {
  if (name === "success" && schema.dtype === "int8" && schema.per_episode) {
    return renderSuccessSegment(name, valuesInRange);
  }
  // ... existing dispatch by (dtype, shape) below
}

function renderSuccessSegment(name, valuesInRange) {
  const wrap = document.createElement("div");
  wrap.className = "success-segment";
  const states = [
    { value: -1, label: "✗ Failure", cls: "failure" },
    { value: 0, label: "— Unmarked", cls: "unmarked" },
    { value: 1, label: "✓ Success", cls: "success" },
  ];
  const uniformValue = uniformOrNull(valuesInRange);
  for (const s of states) {
    const b = document.createElement("button");
    b.type = "button";
    b.className = s.cls + (uniformValue === s.value ? " active" : "");
    b.textContent = s.label;
    b.onclick = () => stageFeatureSetForCurrentSelection(name, s.value);
    wrap.appendChild(b);
  }
  return wrap;
}

function uniformOrNull(arr) {
  if (!arr || !arr.length) return null;
  const first = arr[0];
  return arr.every((v) => v === first) ? first : null;
}
```

`stageFeatureSetForCurrentSelection(name, value)` — reuse the existing staging helper used by other widgets in this file. Search for `feature_set` / `POST .*/edits/feature-set` to find it; do not invent a new staging path.

- [ ] **Step 3: Update timeline-row rendering for success**

Find the per-row band rendering. For `success` (or any int8 per_episode feature), color cells per value: `+1` → `feature-row-band-success-pos`, `-1` → `feature-row-band-success-neg`, `0` → `feature-row-band-success-zero`. Mirror the existing bool band path; just add a third bucket.

- [ ] **Step 4: Bump cache-busting version**

In `index.html`, bump `feature_editing.js?v=NN` again.

- [ ] **Step 5: Manual smoke test**

```bash
uv run python -m lerobot.gui --port 8001
```

Open a dataset that has `success` (add via banner if not present). Drag-select on the success row → Inspector shows three-button segment → click Failure → row band turns red across the whole episode (per-episode coercion) → chip appears in edits-bar → Save → reload → value persists.

- [ ] **Step 6: Commit**

```bash
git add src/lerobot/gui/static/feature_editing.js src/lerobot/gui/static/style.css src/lerobot/gui/static/index.html
git commit -m "feat(gui): tri-state segment-control widget for success int8 per_episode"
```

---

## Task 15: `dataset.schema_changed` WS handler in JS

**Files:**

- Modify: `src/lerobot/gui/static/app.js` (or wherever the playback WS message dispatch lives)
- Modify: `src/lerobot/gui/static/feature_editing.js` (handler)

- [ ] **Step 1: Find the WS message dispatch**

```bash
grep -n "ws.onmessage\|websocket\|onmessage\|dispatch.*event" src/lerobot/gui/static/app.js src/lerobot/gui/static/feature_editing.js | head -20
```

Find the existing `onmessage` switch on `data.type`.

- [ ] **Step 2: Add handler case**

In the dispatcher, add:

```javascript
case "dataset.schema_changed":
    if (window.refreshDatasetInfo) await window.refreshDatasetInfo(window.currentDataset);
    if (window.FeatureEditing && window.FeatureEditing.onDatasetOpened) {
        window.FeatureEditing.onDatasetOpened(window.currentDataset);
    }
    break;
```

- [ ] **Step 3: Manual smoke test**

```bash
uv run python -m lerobot.gui --port 8001
```

Open dataset in Tab A. In Tab B, hit `POST /api/datasets/{id}/features/defaults` via curl. Tab A should reload schema and show the new rows automatically (no manual refresh).

```bash
curl -X POST http://127.0.0.1:8001/api/datasets/<id>/features/defaults
```

- [ ] **Step 4: Commit**

```bash
git add src/lerobot/gui/static/app.js src/lerobot/gui/static/feature_editing.js
git commit -m "feat(gui): handle dataset.schema_changed WS event — re-render rows"
```

---

## Task 16: Orphan `.tmp` cleanup on dataset open

**Files:**

- Modify: `src/lerobot/gui/api/datasets.py` (call `_sweep_orphan_tmp_shards` in dataset-open path)
- Test: `tests/gui/test_feature_add_endpoints.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/gui/test_feature_add_endpoints.py`:

```python
def test_dataset_open_sweeps_orphan_tmp_files(tmp_path, fresh_client):
    """Stale .tmp files left by a crashed save are removed on dataset open."""
    # fresh_client fixture: starts a fresh app + temp HF_LEROBOT_HOME under tmp_path.
    # Build a minimal dataset on disk first (mirror existing fixture pattern).
    # Drop a stale .tmp file into its data dir BEFORE opening via the API.
    # Open dataset via POST /api/datasets/{id}/open (or whatever the endpoint is).
    # Assert .tmp is gone afterwards.
    ...
```

If `fresh_client` / `tmp_path` fixture combinations don't already exist in `tests/gui`, look at `tests/gui/test_open_dataset_local_cache.py` for the pattern to copy.

- [ ] **Step 2: Run, confirm fail**

- [ ] **Step 3: Implement**

In `src/lerobot/gui/api/datasets.py`, find the dataset-open handler. After the dataset is loaded but before returning the response:

```python
from lerobot.datasets.dataset_tools import _sweep_orphan_tmp_shards
removed = _sweep_orphan_tmp_shards(ds.root)
if removed:
    logger.info(f"Cleaned {removed} orphan .tmp file(s) for {dataset_id}")
```

- [ ] **Step 4: Run, confirm pass**

- [ ] **Step 5: Commit**

```bash
git add src/lerobot/gui/api/datasets.py tests/gui/test_feature_add_endpoints.py
git commit -m "feat(gui): sweep orphan .tmp shards on dataset open (crash recovery)"
```

---

## Task 17: End-to-end manual verification (per spec)

- [ ] **Step 1: Start GUI**

```bash
uv run python -m lerobot.gui --port 8001
```

- [ ] **Step 2: Run each verification scenario from `add_feature.md`**

1. Open `KeWangRobotics/sim_pick` (lacks `reward` / `success`) → banner appears → click Add → progress modal → rows appear → drag-select → edit values → Save → reload → values persist, schema persists.
2. Open `+ Add feature` dialog → add a custom `int64[1]` per-frame feature → row appears with line plot → edit via slider → Save → reload → new feature persists.
3. Add a per-episode bool feature via dialog → confirm full-episode coercion → toggle for one episode → confirm timeline shows uniform band.
4. Try adding a feature named `reward` via the generic dialog → rejected with clear error pointing to the banner path.
5. Stage a `feature_set` edit, then attempt `+ Add feature` → rejected with "Save or discard pending edits first."

- [ ] **Step 3: Update TODO.md**

In `src/lerobot/gui/TODO.md` under "Feature Editing (per-frame view + edit)", mark the relevant Phase B items completed (or add a note that this work covers the schema-add follow-up).

- [ ] **Step 4: Final commit if anything was tweaked during smoke testing**

```bash
git add -A
git commit -m "chore(gui): smoke-test fixes after add-feature E2E"
```

(Skip if no changes.)

---

## Self-Review Checklist (run after writing the plan)

- [x] **Spec coverage:** every section of `add_feature.md` mapped to at least one task — defaults banner (T7+T8+T12), `add_features_inplace` (T2-T6), `per_episode` hint (T1+T11), endpoints (T7-T10), pending-edit guard (T9), WS event (T10+T15), success widget (T14), generic dialog (T13), orphan sweep (T6+T16), manual verification (T17).
- [x] **Placeholder scan:** every step has either real code, an exact command, or an explicit "find this in the file by greppping for X" instruction. The few `...` placeholders (e.g. broadcast plumbing in T10 step 4) point to "mirror existing pattern" with the grep command to find it — they're concrete, not "TODO".
- [x] **Type consistency:** `add_features_inplace(dataset, features={name: (fill, info)})` signature used the same way in T2-T8. `AddFeatureRequest`/`AddFeatureResponse` model names consistent across T7+T8. `_sweep_orphan_tmp_shards` signature consistent across T6+T16.

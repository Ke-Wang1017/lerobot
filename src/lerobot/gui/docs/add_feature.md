# Add Feature — default `reward` / `success` + generic schema-add dialog

Extends [feature_editing.md](feature_editing.md) with the schema-mutation path it explicitly punted: adding new per-frame and per-episode features to an existing dataset, in place, without re-encoding videos. Two surfaces:

1. **Default features (`reward`, `success`)** — treated as MUST-have columns, like the existing `DEFAULT_FEATURES` (`timestamp`, `frame_index`, `episode_index`, `index`, `task_index`). Surfaced via a one-time banner on dataset open if absent.
2. **Generic Add Feature dialog** — a "+ Add feature" button at the bottom of the feature-rows column that opens a small modal for any other custom feature.

Both surfaces hit the same new core API: `dataset_tools.add_features_inplace()`, peer to `modify_features` and `set_feature_values`.

---

## Why "in-place" matters

`modify_features(add_features=...)` exists today and works — but it forks the entire dataset, including a full `shutil.copy` of every video file ([dataset_tools.py:2002](../../datasets/dataset_tools.py#L2002)). For a typical recorded dataset (many GB of video), that's slow and doubles disk briefly. Adding columns is logically a parquet-only operation; videos are untouched. A new in-place function avoids the cost entirely and stays at the LeRobot data level so CLI / notebook callers benefit equally — same justification the design doc gave for putting `set_feature_values` in `dataset_tools.py` rather than burying it in the GUI.

---

## Default features

### `reward` (per-frame)

|                     |                                                                                          |
| ------------------- | ---------------------------------------------------------------------------------------- |
| Schema              | `{dtype: "float32", shape: [1], names: null}`                                            |
| Initial fill on add | `0.0`                                                                                    |
| Timeline row        | line plot, scaled to per-episode min/max                                                 |
| Inspector widget    | slider + number input (per [feature_editing.md](feature_editing.md) numeric scalar rule) |

### `success` (per-episode, tri-state)

|                     |                                                                                                                                                     |
| ------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------- |
| Schema              | `{dtype: "int8", shape: [1], names: null, per_episode: true}`                                                                                       |
| Initial fill on add | `0` (unmarked)                                                                                                                                      |
| Encoding            | `1` = success, `0` = unmarked, `-1` = failure                                                                                                       |
| Timeline row        | colored band — gray (`0`) / green (`+1`) / red (`-1`), full episode width                                                                           |
| Inspector widget    | three-button segment control: `[ ✗ Failure ]  [ — Unmarked ]  [ ✓ Success ]`. Edit auto-coerced to whole episode via existing per-episode handling. |

Both are **visible by default** in the feature column (already in the [feature_editing.md](feature_editing.md) heuristic) and **pinned at the top**.

### Banner UX (existing dataset missing one or both)

On dataset open, if `reward` or `success` is absent from `info.json`'s `features` dict, render a top-of-data-tab banner:

> ⚠️ This dataset is missing default features: **reward**, **success**.
> [ Add missing features ] [ Dismiss ]

- **Add** → `POST /api/datasets/{id}/features/defaults`. Modal shows progress (`Rewriting N parquet shards… ~10s`). On success: dataset metadata reloads, schema event pushed, banner disappears, rows appear.
- **Dismiss** → banner hides for the rest of the **browser session** only (in-memory `bannerDismissed` set keyed by dataset_id; cleared on `onDatasetClosed` and on page reload). Reappears on next open — we want the user to add them eventually.

### `per_episode` declaration

Today [`_detect_per_episode_features`](../api/datasets.py) infers per-episode features by checking that every episode has uniform values. Works but slow (full scan of `episode_index` + candidate columns) and ambiguous for freshly-added columns where every value is the seed default.

This work adds explicit `per_episode: true` to the feature spec dict in `info.json` for `success` (and any other per-episode feature added via the dialog with the checkbox set). Declared hint wins; inference remains as the fallback for legacy datasets. The hint is preserved through `LeRobotDatasetMetadata` and surfaces in the `FeatureSchema` API model (already has the field).

---

## Generic Add Feature dialog

A "**+ Add feature**" button at the bottom of the feature-rows column opens a modal:

| Field       | Type     | Notes                                                                                                                                                                                            |
| ----------- | -------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| Name        | text     | required, unique vs existing schema, must match `[a-zA-Z_][a-zA-Z0-9_.]*`, not in `DEFAULT_FEATURES`, not `reward` or `success` (those go through the banner)                                    |
| Dtype       | dropdown | `bool`, `int8`, `int64`, `float32`, `string` (V1 — vectors and images excluded)                                                                                                                  |
| Shape       | text     | default `[1]`; for V1 the editor only supports `[1]` and `[N≤8]` cleanly per [feature_editing.md](feature_editing.md) edit-affordances table                                                     |
| Per-episode | checkbox | declared hint; defaults to off                                                                                                                                                                   |
| Fill value  | text     | auto-defaulted from dtype (`0` for numerics, `False` for bool, `""` for string); editable. **If per-episode + bool, default flips to `True`** (matches the "successful demo by default" pattern) |

Submit → `POST /api/datasets/{id}/features` with `{name, dtype, shape, per_episode, fill_value}` → in-place rewrite → schema reload.

---

## Backend

### New core API: `dataset_tools.add_features_inplace()`

```python
def add_features_inplace(
    dataset: LeRobotDataset,
    features: dict[str, tuple[Any, dict]],
    *,
    recompute_stats: bool = True,
) -> None:
    """Add features to an existing dataset in place.

    Each entry in `features` maps a feature name to (fill_value, feature_info).
    `fill_value` is a scalar broadcast across every frame (or a numpy array of
    length == total_frames if per-frame values are pre-computed).
    `feature_info` follows the same schema as `modify_features`:
    `{"dtype": ..., "shape": ..., "names": ..., "per_episode": ...}` (the
    `per_episode` key is new and optional).

    Videos are NOT touched. Only parquet shards under `data/chunk-*/file-*.parquet`
    and `meta/info.json` are rewritten. Per-episode stats for new columns are
    recomputed via existing `compute_stats.py` helpers when `recompute_stats=True`.

    Calls `dataset.finalize()` before returning.
    """
```

Save sequence inside `add_features_inplace`:

1. **Validate.** For each new feature: name unique vs `dataset.meta.features`, name not in `DEFAULT_FEATURES`, dtype in supported set, shape is a list of positive ints, `per_episode` is bool. Raise `ValueError` on any failure before touching the disk.
2. **Build new `info.json` in memory.** Copy existing `features` dict, add new entries with `per_episode` preserved. The `codebase_version` field is unchanged — this is a content-additive operation within v3.0, not a format migration. (If implementation finds the v3 docs require otherwise, fix here.)
3. **For each parquet shard:** read with pyarrow → append constant-fill columns (one column per new feature, length = shard row count) → write to `<shard>.tmp` sibling.
4. **Rename all `.tmp` → real** in sequence (per-shard atomic via POSIX rename; cross-shard window has the same risk model as `set_feature_values` — orphan `.tmp` cleanup on next open).
5. **Atomically write `info.json`** via `.tmp` + rename, last.
6. **Recompute stats** for the new feature(s) only via `compute_stats.py`, scoped to the new columns across all episodes. Per-episode entries in `meta/episodes/*.parquet` get the `stats/<new_feature>/*` columns appended. (Existing `_recompute_episode_stats_from_data` operates on a full row set; if it can't be scoped to specific columns, factor out a `_recompute_stats_for_columns(...)` variant — see Critical files.)
7. **`dataset.finalize()`** as required by v3 docs.

Reuses [`_write_parquet`](../../datasets/dataset_tools.py), `_recompute_episode_stats_from_data` (or a scoped variant), [`compute_stats.py`](../../datasets/compute_stats.py), [`validate_feature_dtype_and_shape`](../../datasets/feature_utils.py).

### New GUI endpoints

In [`api/datasets.py`](../api/datasets.py):

- **`POST /api/datasets/{id}/features`** — body `{name, dtype, shape, per_episode, fill_value}`. Acquires `_app_state.get_lock(dataset_id)`. Calls `add_features_inplace(ds, {name: (fill_value, info)})`. Invalidates `_per_episode_features_cache[id]`, `_episode_start_indices[id]`, frame cache for the dataset, and the `DatasetInfo` mtime cache. Reloads `ds.meta`. Pushes a `dataset.schema_changed` event over the existing playback WS channel so frontend re-fetches `DatasetInfo` and re-renders the feature column.
- **`POST /api/datasets/{id}/features/defaults`** — convenience: adds whichever of `reward` / `success` are absent from the schema in a single `add_features_inplace` call. Used by the banner. Returns `{added: ["reward", "success"]}` or `{added: []}` if both already present.

### Concurrency

Same model as `set_feature_values`: in-process lock serializes against API requests on the same dataset. External readers (training jobs) must pause before schema changes — surfaced in the modal copy and in user docs.

### Safety rails

- Confirmation modal in the GUI when a single Add affects > 100k frames (rough wall-time threshold for "this will take more than a few seconds").
- Reject schema add if any pending `feature_set` edits reference the dataset — Save staged edits or Discard them first. Avoids races between schema mutation and value mutation on the same parquet shards.

---

## Frontend

In [`feature_editing.js`](../static/feature_editing.js) and a new [`add_feature_dialog.js`](../static/add_feature_dialog.js):

- **Banner component** at the top of the data tab. Visible iff `reward` or `success` is missing from `dataset.features_schema`. State lives in module-level `bannerDismissed` set keyed by dataset_id (cleared on `onDatasetClosed`).
- **"+ Add feature" button** at the bottom of `#feature-rows`. Opens the dialog modal.
- **Add Feature dialog** — vanilla HTML modal with the fields listed above. Auto-default fill_value updates when dtype or per_episode changes. Disable Submit until name + dtype valid. Submit calls the POST endpoint, shows progress, closes on success.
- **Success widget renderer** — register `(dtype="int8", shape=[1], per_episode=true, name="success")` in the renderer registry → three-button segment control. Shared `feature_set` staging path; value is the int (`-1` / `0` / `+1`).
- **Reward widget renderer** — already handled by the existing numeric scalar renderer (slider + number input).
- **Schema event handling** — listen for `dataset.schema_changed` on the playback WS, re-fetch `DatasetInfo`, re-render rows + Inspector. The same handler triggers when `add_features_inplace` finishes server-side.

---

## What this design does NOT change

- `modify_features` stays as is — still the right tool for forked schema changes (e.g. removal, rename, batch operations against a copy). `add_features_inplace` is purely additive.
- Existing `set_feature_values` flow is untouched. New default features become editable through the same staging / save pipeline immediately after the schema add commits.
- Video pipeline: not touched at all.

---

## Scope

**In:**

- New `add_features_inplace()` in `dataset_tools.py` with cross-shard atomic-rename pipeline + scoped stats recomputation.
- `per_episode` hint in feature spec dict (declared in `info.json`, propagated through `LeRobotDatasetMetadata` and `FeatureSchema`).
- Two GUI endpoints (`POST .../features`, `POST .../features/defaults`).
- Banner for missing `reward` / `success` on dataset open.
- "+ Add feature" button + modal dialog.
- Success tri-state segment-control renderer.
- Schema-changed WS event + frontend re-render.
- Reject schema add when pending value edits exist; >100k-frame confirmation.

**Out (Follow-ups):**

- Removing or renaming features from the GUI (covered by existing `modify_features` paths if surfaced later — separate work).
- Editing values of vector / image / video features.
- Newly-created datasets seeding `reward` / `success` automatically at creation time (a separate change to `LeRobotDataset.create()` defaults; this design only addresses retrofit on existing datasets).
- Automatic `failure` reason text alongside the `success` int (RECAP-style annotation; would be a separate per-episode `string` feature added later).
- Per-feature "delete column" button.
- Cross-file atomic Save / manifest protocol — same risk model and same deferral as `set_feature_values`.

---

## Critical files

**LeRobot core:**

- [src/lerobot/datasets/dataset_tools.py](../../datasets/dataset_tools.py) — new `add_features_inplace()`; reuses existing parquet helpers and `compute_stats.py`.
- [src/lerobot/datasets/feature_utils.py](../../datasets/feature_utils.py) — extend `validate_feature_dtype_and_shape` if needed for the `int8` + `per_episode` combo.
- [src/lerobot/datasets/dataset_metadata.py](../../datasets/dataset_metadata.py) — preserve the `per_episode` hint when round-tripping `info.json`.

**GUI backend:**

- [src/lerobot/gui/api/datasets.py](../api/datasets.py) — two new POST endpoints; `FeatureSchema.per_episode` already present, surface the declared value (not just inferred).
- [src/lerobot/gui/api/playback.py](../api/playback.py) — emit `dataset.schema_changed` over the WS.
- [src/lerobot/gui/state.py](../state.py) — pending-edits guard helper for the schema-add safety rail.

**GUI frontend:**

- [src/lerobot/gui/static/feature_editing.js](../static/feature_editing.js) — register success renderer; subscribe to `schema_changed`; render banner; render "+ Add feature" button.
- new `src/lerobot/gui/static/add_feature_dialog.js` — modal dialog logic.
- [src/lerobot/gui/static/index.html](../static/index.html) — load the new JS file; banner DOM slot.
- [src/lerobot/gui/static/style.css](../static/style.css) — banner, dialog, success segment control styles.

**Tests:**

- new `tests/datasets/test_add_features_inplace.py` — schema additions; per-episode hint round-trip; stats recomputation for new columns; orphan `.tmp` cleanup; rejection of name collisions / `DEFAULT_FEATURES` / bad dtypes.
- new `tests/gui/test_feature_add_endpoints.py` — POST `.../features` and POST `.../features/defaults`; pending-edit guard; cache invalidation.
- extend [tests/gui/test_state.py](../../../tests/gui/test_state.py) for pending-edit safety rail.

---

## Verification

`uv run pytest tests/gui tests/datasets -svv -k "add_feature"` after implementation.

Manual:

1. Open `KeWangRobotics/sim_pick` (lacks `reward` / `success`) → banner appears → click Add → progress modal → rows appear → drag-select → edit values → Save → reload → values persist, schema persists.
2. Open `+ Add feature` dialog → add a custom `int64[1]` per-frame feature → row appears with line plot → edit via slider → Save → reload → new feature persists.
3. Add a per-episode bool feature via dialog → confirm full-episode coercion → toggle for one episode → confirm timeline shows uniform band.
4. Try adding a feature named `reward` via the generic dialog → rejected with clear error pointing to the banner path.
5. Stage a `feature_set` edit, then attempt `+ Add feature` → rejected with "Save or discard pending edits first."

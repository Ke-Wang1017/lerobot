# Feature Editing Design

Per-frame feature view + edit in the Data tab. A schema-driven way to display and edit any per-frame feature (`reward`, `success`, `subtask`, …) without leaving the GUI. V1 is one primitive — drag-select a range, edit values in the Inspector — built so feature-specific tooling (curve editors, segment-brushes) layers on top later.

![Feature editing layout sketch](feature_editing_layout.svg)

---

## Context

Motivated by RECAP-style labeling ([π\*0.6](https://www.pi.website/blog/pistar06)): per-frame `reward`, dense per-frame `subtask` language, per-episode `success` — relabeled iteratively. The GUI has no view/edit for these today.

### LeRobot format primer

- **Schema is per-dataset, values are per-frame.** `info.json` declares one `features` dict; every frame in every episode shares it. Adding or removing a feature is a structural change: the data parquet shards (`data/chunk-*/file-*.parquet`) get rewritten with the new column set, and per-feature `stats/<feature>/*` columns in `meta/episodes/*.parquet` are added or dropped. Videos are untouched unless the feature being added/removed is itself `image` or `video`. Per-frame value edits never touch any of this — they overwrite cells in the existing data shards.
- **Adding a feature, properly.** Two paths: declare it in the `features` dict at `LeRobotDataset.create(...)` time, or call [`modify_features(add_features=...)`](../../datasets/dataset_tools.py) on an existing dataset. There's no in-place "just edit `info.json` and append" path — the shards must agree with the schema, and stats need to be computed.
- **`task` is fully first-class.** Auto-deduped at write ([dataset_metadata.py:373](../../datasets/dataset_metadata.py#L373)) — strings in `meta/tasks.parquet`, frames carry `task_index`.
- **`subtask` is officially supported on the read side.** Same dedup pattern (`meta/subtasks.parquet` + per-frame `subtask_index`, decoded by `dataset[i]`); see the [official subtask guide](https://huggingface.co/docs/lerobot/dataset_subtask). What's missing is an `add_frame(subtask=...)` writer; today subtasks are written via the [lerobot-annotate Space](https://huggingface.co/spaces/lerobot/annotate). V1 treats `subtask_index` as a normal per-frame `int64` feature and updates `meta/subtasks.parquet` on Save when new strings appear.
- **Per-frame features are extensible; per-episode features are not.** The per-frame `features` dict can hold any user-defined column. But `meta/episodes/*.parquet` has a hardcoded schema (episode_index, length, tasks list, file pointers, stats) — there's no `episode_features` slot in `info.json`. So per-episode logical fields like `success` live as per-frame `bool[1]` columns broadcast across the episode.

---

## What's preserved from the existing GUI

Unchanged: the global tab bar, the dataset-tree sidebar, the camera grid, the existing `controls-bar` (Play / speed dropdown / frame info / time info / Reset-trim / status), the timeline scrubber + playhead, the trim handles, and the bottom edits-bar (Discard / Save). This design slots **per-feature timeline rows** under the existing scrubber, adds the **right-hand Inspector**, and reshapes the edits-bar from flat chips into a grouped/expandable list with a "Show pending edits" toggle.

---

## Editing model

**Selection is a vertical slice — a frame range, not a feature.** Drag on any row produces `{ episodeIndex, frameFrom, frameTo }`; the band extends through all feature rows. Inspector then shows per-feature edit cards scoped to that range.

**Origin-row focus**: when the user drags on a specific row, the Inspector auto-scrolls that feature's card into view and gives it the focus ring. With ~10–15 cards in the column, this skips the hunt and lets the user start typing immediately.

In V1 the only on-row primitive is **drag to select a range**.

### Selection semantics

- **Half-open** `[from, to)` — matches LeRobot's `dataset_from_index`/`dataset_to_index` (where `to - from == length`). Single-frame = `[N, N+1)`. Empty `[N, N)` = no selection.
- **No multi-range selection.** Holding shift to select non-contiguous frame ranges (e.g. 120–129 ∪ 200–211) isn't supported in V1.
- **Boundary frames are selectable.** Half-open trim `[trim_from, trim_to)` includes `trim_from`, so the first selectable frame is `trim_from` and the last is `trim_to - 1`. Hit-testing must keep feature-row click zones full-width (including the columns under the trim handles), since trim handles live on the time axis row, not on feature rows — there's no overlap.
- **Trim ≠ selection.** Trim adjusts only the outer envelope `[trim_from, trim_to)` (V1 doesn't support internal cuts — that would force episode splits + metadata/hash rewrites). Trim handles live on the time axis; gestures don't conflict with feature-row drag.

### Click semantics — one rule, with an escape hatch

Clicking anywhere inside the trim envelope (on the time axis or any feature row) seeks the playhead AND produces a selection. Single rule for the entire editable area; no special "this row is for playback only" carve-out.

- **Click inside the trim envelope at frame N** → playhead → N, selection = `[N, N+1)`.
- **Drag inside the trim envelope from A to B** → while dragging, the playhead tracks the drag-end so the camera shows the frame at the current "to" edge. On release, selection = `[min(A,B), max(A,B) + 1)`.
- **Drag the existing playhead thumb** → scrub only; selection is unchanged. This is the explicit "I just want to navigate without re-selecting" gesture. Also covers the existing playback workflow (`Play` / frame-step buttons in the controls bar) which doesn't touch selection either.
- **Click outside the trim envelope** (the dimmed region) → no-op. The dim region behaves like a wall — you can't put a selection there because those frames won't ship after Save.
- **Esc** → clear selection.

The accepted cost: a misclick during inspection overwrites the selection ROI. Mitigations are the playhead-thumb drag, the existing Play / frame-step controls, and the fact that any staged edits are still recoverable per-chip in the edits bar — only the ROI is lost, not committed work.

### Boundary visualization — frame cells, not ticks

Each frame N occupies cell `[N, N+1)` on the timeline. Selection handles sit on cell _boundaries_, not on cells — same as DaVinci Resolve, Premiere, Audacity, Aegisub.

```
│ 119 │ 120 │ 121 │ 122 │ 123 │ 124 │ 125 │ 126 │ 127 │ 128 │ 129 │ 130 │
      ╞═══════════════════════════════════════════════════════════╡
      ↑                                                           ↑
      from = 120                                                  to = 130
```

Coverage = cells `from`..`to-1`, exactly `to - from` cells. Human label: "frames N…M (K frames)" with M = `to-1`. Click snaps to cell boundaries; drag uses `from = min(a,b)`, `to = max(a,b) + 1`.

All user-facing copy (chip text, confirmation dialogs, hover tooltips, history JSON labels) uses the inclusive form via a single shared formatting helper. Internal storage and APIs use the exclusive form. Pinned to one helper to prevent drift.

### Per-episode broadcast features

Some logical per-episode fields (`success` being the canonical example) live as per-frame `bool[1]` columns broadcast across the entire episode, because `meta/episodes/*.parquet` has no `episode_features` slot (see format primer). Editing a sub-range would silently break the broadcast invariant and leave half the episode disagreeing with the other half.

V1 detects these features by a `per_episode: true` hint — declared either at feature-creation time or inferred when every frame in an episode currently shares one value — and **coerces edits to the whole episode**. When the user drag-selects `[120, 130)` on a `per_episode` row, the Inspector card shows the range as `[0, episode_length)` with a small note ("`success` is per-episode — edit applies to the full episode"). The selection band on the row visually expands to fill the episode while the card is focused.

Inference fallback: if no hint is declared, treat any feature where `nunique(values_in_episode) == 1` for every episode in the dataset as per-episode for editing purposes. Cheap to compute from existing stats. A `display_hint`-style declaration in `info.json` is the proper fix and is already a follow-up.

### Overlapping staged edits

Two staged edits on the same `(feature, episode_index)` with overlapping frame ranges trigger a confirmation modal at stage time:

> **Overlapping edit**
> You already have a staged edit on `reward` covering frames 100…149. The new edit covers frames 120…180. Accepting will replace values in the overlap (120…149) with the new edit's value.
>
> [ Cancel ] &nbsp; [ Accept ]

Cancel discards the new edit. Accept resolves last-write-wins: the new edit is staged, and the older edit's range is **clipped** to the non-overlapping portion (here: 100…119). If the older edit is fully contained in the new one, it's removed entirely. Both edits remain visible as separate chips in the edits bar after clipping; the user can still discard either independently.

Resolution happens at stage time, not Save time, so the edits bar always shows a non-overlapping picture of what will be written. The Save pipeline can therefore assume no overlapping edits per `(feature, episode_index)` as an invariant.

---

## V1 edit affordances per type

| Feature                                           | Row display                               | Inspector widget                                                                                                                                                                                                                                                                                            |
| ------------------------------------------------- | ----------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `bool[1]`                                         | on/off band                               | **checkbox** (project convention — not toggle); indeterminate state when range has mixed values                                                                                                                                                                                                             |
| per-episode `bool[1]` (e.g. `success`)            | on/off band, full-episode                 | checkbox; edit coerced to full episode with note                                                                                                                                                                                                                                                            |
| numeric scalar (e.g. `float32[1]`, `int64[1]`)    | line                                      | **slider + number input** (slider range from per-feature stats min/max; number input wins for out-of-range; free-form input only when stats absent or degenerate)                                                                                                                                           |
| `string`                                          | colored stripe + text labels              | single-line input or textarea; uniform value across the selected range                                                                                                                                                                                                                                      |
| `subtask_index` (with `meta/subtasks.parquet`)    | colored stripe (decoded `subtask` string) | dropdown over existing subtask strings + "+ new subtask…" — staged edit stores the **string**, not an index. On Save: dedup against `meta/subtasks.parquet`, append any new strings, assign indices, then write `subtask_index` cells. Two concurrent Saves adding the same new string converge to one row. |
| numeric vector, small (e.g. `float32[N]`, N ≤ ~8) | mini multi-line                           | row of N inputs                                                                                                                                                                                                                                                                                             |
| numeric vector, large (e.g. `float32[N]`, N > ~8) | line plot                                 | "Edit as JSON…" textarea                                                                                                                                                                                                                                                                                    |
| `action`, `observation.*` (any dtype)             | hidden by default in V1; pin to show      | **read-only** — recorded data, multi-dim viz design deferred (see Follow-ups)                                                                                                                                                                                                                               |
| `image`, `video`, 2D+                             | not on timeline                           | **read-only**                                                                                                                                                                                                                                                                                               |
| `DEFAULT_FEATURES`                                | hidden by default                         | **read-only always**                                                                                                                                                                                                                                                                                        |

---

## Inspector behavior

Schema-driven by `(dtype, shape)`. Always visible when a dataset is open.

- **Empty state** (no selection): compact dataset summary.
- **In-selection state**: scrollable feature-card column scoped to `[from, to)`. Origin-row card auto-focused. Per card: name + dtype, summary of values in the range (min/max for scalars, unique values for strings, "uniform/mixed" for bools), edit widget or read-only badge.
- **Edits stage automatically** as the user changes a card. No per-card Apply. Text inputs stage on blur (or 300 ms debounce); checkboxes/sliders/dropdowns stage on commit. The card shows `● pending`; a chip appears in the edits bar; bottom **Save** is the single confirmation, **Discard** drops everything.

### Schema discoverability

Schema is read-only and surfaces in two always-on places once a dataset is open:

- **Inline on each timeline row** — name + dtype label at the row's left edge gives an at-a-glance overview.
- **Inspector cards on selection** — full per-feature detail (dtype, shape, range stats, edit widget).

There's no separate schema modal. The dataset row in the sidebar is not itself a selectable / inspectable target today, so a "right-click → Info…" gesture would target a thing with no other selection semantics. If "peek schema before opening" ever becomes a real need (e.g. browsing 50 datasets in a folder), the right answer is search/filter, not a schema viewer.

---

## Edit application pipeline

Staging extends `PendingEdit` ([state.py](../state.py)) with `edit_type: "feature_set"`, params `{ feature, episode_index, frame_from, frame_to, value }`. Persists to `.lerobot_gui_edits.json`. Discard / Save / per-chip removal work for free.

**Applying on Save needs a new LeRobot API** — value editing is a missing function at the dataset level. The available `dataset_tools` operations are all schema-level (`add_features`, `remove_feature`, `rename_feature`, `swap_features`) or episode-level (`delete_episodes`, `split_dataset`, `merge_*`); none overwrite values in existing columns. `modify_features` re-encodes the whole dataset — right tool for schema changes, wrong for value edits (slow, touches videos, forks the dataset on every Save). Sidecar JSON overlay is cheap and reversible but invisible to training, so a non-starter.

This work adds `set_feature_values()` to [`dataset_tools.py`](../../datasets/dataset_tools.py), as a peer to `modify_features`:

```python
def set_feature_values(
    dataset: LeRobotDataset,
    edits: list[dict],          # [{feature, from_index, to_index, value}, ...]
    *,
    in_place: bool = True,      # rewrite shards in place; if False, fork to output_dir
    output_dir: Path | None = None,
) -> None:
    """Overwrite values of EXISTING features for specific frame ranges.
    Schema is unchanged. Stats for affected episodes are recomputed.
    Calls dataset.finalize() before returning."""
```

It belongs at the LeRobot data level, not buried in the GUI: any tool (CLI, notebook, GUI) that needs to relabel a dataset benefits from one canonical entry point. The GUI's `_apply_feature_set_edits()` in [edits.py](../api/edits.py) just translates staged `PendingEdit`s into a `set_feature_values()` call. Reuses existing helpers (`_write_parquet`, `_recompute_episode_stats_from_data`, `_copy_episodes_metadata_and_stats`) and path templates (`DATA_DIR`, `DEFAULT_DATA_PATH` from [utils.py](../../datasets/utils.py)).

### Save sequence

A Save may touch multiple parquet shards (different chunks, possibly different episodes) and `meta/subtasks.parquet`. The sequence inside `set_feature_values()`:

1. **Resolve string-keyed feature edits to indices.** For `subtask_index` (and any future string-deduped feature), look up each staged string in `meta/subtasks.parquet`; append new strings and assign indices; rewrite the affected rows of `meta/subtasks.parquet`. Staged edits in-memory now reference resolved `int64` indices for the subsequent shard rewrite.
2. **Group remaining staged edits by target shard.** For each shard, compute the row updates locally. Staging guarantees no overlapping edits per `(feature, episode_index)`, so this is straightforward.
3. **Write all new shards to `.tmp` siblings.** Do all the expensive work — parquet rewriting — before any rename. A crash here leaves only orphan `.tmp` files, which a startup sweep cleans up.
4. **Rename each `.tmp` over its target in sequence.** Per-file atomic via POSIX rename. Cross-file is _not_ atomic — a crash mid-rename can leave a partially-applied Save — but the window is the duration of N renames, and all I/O before that point has already succeeded.
5. **Recompute stats scoped to the edit.** Only the edited feature(s), only the touched episodes — `meta/episodes/<...>.parquet` rows for those `episode_index` values, `stats/<feature>/*` columns for those features. Not the whole dataset, not all features. Reuses [compute_stats.py](../../datasets/compute_stats.py).
6. **Call `dataset.finalize()`** as required by the [v3 docs](https://huggingface.co/docs/lerobot/lerobot-dataset-v3#always-call-finalize-before-pushing). For a values-only edit this should be effectively a no-op; verify during implementation that finalize() doesn't trigger work scoped to schema changes (video re-encode, `info.json` rewrite).

Stronger cross-file atomicity is out of scope for V1 — a manifest-based "save in progress" marker with resume-on-open is a follow-up if the partial-Save window proves to be a real problem in practice.

### Concurrency assumption

The GUI server serializes operations against an opened dataset via an in-process lock (`_app_state.get_lock(dataset_id)`), so two API requests don't race. **External** concurrent readers — e.g. a training job in another process reading the same parquets — are NOT prevented; on Linux, renaming a parquet file out from under an mmap'd reader is benign, on Windows it fails outright; in either case a training job mid-Save sees torn state across shards. Surfaced in the Save confirmation dialog and in user docs: **pause training jobs before Saving feature edits.**

### Safety rails

- Validate every staged edit against schema (dtype + shape) via [`validate_feature_dtype_and_shape`](../../datasets/feature_utils.py).
- Block edits on read-only features (`DEFAULT_FEATURES`, `action`, `observation.*`, image/video).
- Confirmation dialog if a single Save touches > 10 000 frames.
- **No edit-edit dependencies in V1.** All editable V1 features are independent primitives. When derived features arrive (e.g. RECAP `reward = f(success, step_count)`), they'll be read-only with a "Recompute from inputs" button — dependencies declared at the feature-definition level, never between staged edits.

---

## Default visibility & scrolling

Heuristic without a format change:

- **Hidden by default**: `image`/`video`, `DEFAULT_FEATURES`, `task_index` (replaced by `task` string), and `action` / `observation.*` (multi-dim numeric vectors — read-only in V1, viz design deferred).
- **Visible by default**: everything else — `reward`, `success`, `subtask`, scalar custom features, short-string custom features.
- **Pin / hide** any feature individually; preference persisted in localStorage.

A `display_hint` in `info.json` for finer control is a follow-up.

---

## V1 design decisions

| Concern                        | Decision                                                                                                                                                                                                 | Reason                                                                                                                                                       |
| ------------------------------ | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| Selection model                | Vertical slice through all rows                                                                                                                                                                          | Frames are tuples; one mental model; faster multi-feature labeling                                                                                           |
| Per-episode broadcast features | Edits coerced to whole episode                                                                                                                                                                           | Sub-range edit silently breaks the broadcast invariant                                                                                                       |
| Trim semantics                 | Outer-bounds-only; handles on time axis                                                                                                                                                                  | Internal cuts force episode splits + metadata rewrites (out of scope)                                                                                        |
| Click semantics                | One rule: click inside the trim envelope seeks **and** selects `[N, N+1)`; drag inside the envelope seeks (tracks drag-end) and produces a range. Playhead-thumb drag scrubs without changing selection. | Simpler than splitting "playback territory" vs "editing territory"; one gesture, predictable; playhead-thumb is the escape hatch for navigate-without-select |
| Apply strategy                 | New `set_feature_values()` LeRobot API (peer to `modify_features`) that does an in-place parquet rewrite                                                                                                 | Value editing is a missing dataset-level function; reusable from CLI/notebooks too                                                                           |
| Cross-file atomicity           | Per-file atomic only; all `.tmp` writes precede any rename                                                                                                                                               | Stronger guarantees need a manifest-based protocol; window is small in practice                                                                              |
| Subtask index assignment       | Resolved at Save, not stage (frontend stages strings)                                                                                                                                                    | Avoids cross-session index collisions; concurrent new-string Saves converge                                                                                  |
| Overlapping edits              | Warn + accept = last-write-wins with clipping of prior edit                                                                                                                                              | Explicit user choice; edits bar always reflects what will be written                                                                                         |
| Undo across Saves              | Punted (V1 has per-chip removal + Discard / Save only)                                                                                                                                                   | Single-step "Revert last Save" via pre-edit snapshot is a clear follow-up; full undo is a separate, larger build                                             |
| Boundary semantics             | Half-open `[from, to)`, cells (not ticks)                                                                                                                                                                | Matches LeRobot's `dataset_from_index`/`dataset_to_index` and every pro video editor                                                                         |
| Edit application               | Direct staging on card change/blur; bottom Save commits                                                                                                                                                  | Matches Figma/Photoshop; bottom Save is the single confirmation                                                                                              |
| Layout customization           | Two mouse-draggable resize boundaries (vertical, horizontal); positions persist in localStorage                                                                                                          | Reuses existing `sidebar-resize-handle` pattern; everything else deferred                                                                                    |

---

## Implementation phases

### Phase A — Read-only foundation

- **A1.** Schema in `DatasetInfo`. Backend: extend `DatasetInfo` ([api/datasets.py](../api/datasets.py)) so the existing dataset-open response includes the full features dict (dtype, shape, names) — currently only feature names are returned. No new endpoint, no modal.
- **A2.** Per-frame feature values. Backend: `GET /api/datasets/{id}/episodes/{ep}/frames/{frame}/features` (skip `image`/`video`); optionally piggyback on the playback WS `seek` ([playback.py](../api/playback.py)). Frontend: schema-driven renderer registry by `(dtype, ndim)`.
- **A3.** Episode feature-series + timeline rows. Backend: `GET /api/datasets/{id}/episodes/{ep}/feature-series?features=…` → `{name: [...], ...}`; cache. Client requests visible/pinned features only, never `*`. Frontend: stacked rows; trim envelope tinted through all rows; outside-trim dimmed; pin/unpin per row.

### Phase B — Editing

- **B1.** Vertical-slice selection + click-to-seek-and-select. Drag inside the trim envelope → `{ episodeIndex, frameFrom, frameTo }`; vertical band across all feature rows; origin row remembered for Inspector focus; Esc clears. Click inside trim → playhead → N, selection = `[N, N+1)`. Drag inside trim → playhead tracks drag-end. Existing playhead-thumb drag remains scrub-only (selection unchanged) — the explicit navigate-without-select gesture. Click outside trim → no-op.
- **B2.** Inspector in-selection state. Scrollable card column; widgets per type (checkbox / slider+number / dropdown / text / row-of-N inputs / "Edit as JSON…" / read-only badge); `subtask_index` dropdown sourced from `meta/subtasks.parquet` with "+ new subtask…"; auto-staging with `● pending` indicator on each modified card. Per-episode broadcast features coerce range to full episode.
- **B3.** Stage `feature_set` edits. Extend `PendingEdit` ([state.py](../state.py)). New `POST /api/edits/feature-set`; existing `DELETE /api/edits/{idx}` works. Group-by-`(feature, episode)` rendering with expand / collapse.
- **B4.** Overlapping-edit detection + warning modal at stage time; clip-or-remove prior edit on accept.
- **B5.** "Show pending edits" toggle on the timeline.
- **B6.** Validation + safety rails (schema check, read-only block, > 10k-frame confirmation).
- **B7.** Apply pipeline — add `set_feature_values()` in [dataset_tools.py](../../datasets/dataset_tools.py) (see "Save sequence" above). The GUI's `_apply_feature_set_edits()` in [edits.py](../api/edits.py) translates staged `PendingEdit`s into a single `set_feature_values()` call, then reloads the dataset to refresh `meta.episodes` stats. Startup sweep cleans orphan `.tmp` files.

### Phase C — Layout

- **C1.** Two mouse-draggable resize boundaries (vertical + horizontal). Reuse `sidebar-resize-handle`. Persist in localStorage.

---

## V1 scope

**In:** schema in row labels + Inspector cards · always-visible Inspector · 1D feature rows with scroll + pin/unpin · trim handles on time axis with envelope through all rows · vertical-slice selection · click-inside-trim seeks + selects `[N, N+1)`; drag = range; playhead-thumb drag scrubs without re-selecting · per-episode broadcast feature detection with whole-episode coercion · auto-staging Inspector cards · range value edits for `bool` / numeric scalar / numeric vector / strings · `subtask_index` dropdown with strings staged and indices resolved at Save · overlapping-edit warning with last-write-wins clipping · grouped expandable edits bar · "Show pending edits" toggle · two resize handles · `action` / `observation.*` / image / video / `DEFAULT_FEATURES` are read-only · **new `set_feature_values()` LeRobot API** for in-place value edits (peer to `modify_features`).

**Out (Follow-ups):** in-place segment manipulation · row context menus · multi-range selection · loop-on-selection · undo / redo across Saves (incl. **single-step "Revert last Save"** via a pre-edit snapshot file at `.lerobot_gui_edits_history/<timestamp>.json` — designed but punted) · schema mutations (add/remove/rename) · derived/computed features · curve editor · **multi-dim numeric vector visualization** (`action`, `observation.*`) — design between stacked-rows / overlaid-curves / collapsed-expandable, deferred · URDF / 3D trajectory views · stats viewer (P1/P2) · first-class per-episode features (format extension) · `add_frame(subtask=...)` writer · `display_hint` in `info.json` · episode-list shortcuts (success column, inline `task` edit) · cross-file atomic Save (manifest + resume-on-open) · keyboard-only frame-range entry (alternative to drag-select).

### Post-V1 additions landed during implementation

The branch went beyond the original V1 surface — these are documented here so the next reader knows they're part of the shipped system, not deferred.

- **Declared bounds (`min` / `max`) on a feature** — optional fields in
  `info.json`'s features dict, validated by
  `validate_feature_numeric_bounds` in
  [feature_utils.py](../../datasets/feature_utils.py) at `add_frame`
  and at GUI stage time. Backward-compatible: features without these
  fields validate identically to before. Surfaced in the schema
  endpoint as `FeatureSchema.declared_min` / `declared_max`. Used
  by the Inspector slider's lo/hi precedence (declared > observed >
  series-fallback) and by the `[min … max]` chip rendered next to
  the feature name.
- **Categorical (int + names) widget** — `int` features with a
  non-empty `names` list (and scalar shape) now render as a
  dropdown over the labels in the Inspector and as a colored band
  per category on the timeline row. The on-disk value is the
  integer index `[0, len(names))`; the strings are display labels.
  `is_categorical_feature(ft)` in
  [feature_utils.py](../../datasets/feature_utils.py) is the canonical
  predicate (single source of truth for the
  scalar-int-with-names contract).
- **Dataset-wide observed `[min, max]`** — `FeatureSchema.observed_min`
  / `observed_max` populated from `meta/stats.json` (aggregated
  across episodes by `compute_stats.py`). The Inspector shows the
  observed range as a fallback chip when declared bounds aren't
  present, and the slider scale falls back to it when stats are
  available but no declared bounds.
- **Per-episode segmentation in the Inspector** — features detected
  as per-episode-broadcast (uniform within each episode, e.g.
  `success`, `control_mode`) no longer get full-width solid-color
  timeline rows. They live in their own "Episode N (M frames)"
  Inspector section _above_ the per-frame section. Per-frame cards
  remain editable when a selection exists; per-episode cards are
  always editable (selection-independent).
- **Stage-time bounds enforcement** — the GUI's
  `POST /api/edits/feature-set` runs the bounds checker inline, so
  out-of-range values get an immediate 400 with the user-submitted
  feature name in the error message. This is in addition to the
  source-of-truth gate in `set_feature_values` itself.
- **`StatsRecomputationError`** — `set_feature_values` raises this
  specific exception when per-episode stats recompute fails _after_
  data shards were written. The data is already on disk; only stats
  are stale. The GUI's apply pipeline catches it and reports
  `status: "partial"` with the failure surfaced. Without the
  specific exception type, stale stats (which corrupt the schema
  endpoint's `observed_min/max` chip and break normalization at
  training time) silently land in production.

The new `set_feature_values()` API now lives in its own module —
[src/lerobot/datasets/feature_value_edits.py](../../datasets/feature_value_edits.py)
— re-exported from `dataset_tools.py` for back-compat. Isolating it
keeps our changes off `dataset_tools.py`'s busy refactor surface.

---

## Critical files

**LeRobot core**:

- [src/lerobot/datasets/dataset_tools.py](../../datasets/dataset_tools.py) — new `set_feature_values()` (peer to `modify_features`); reuses existing `_write_parquet`, `_recompute_episode_stats_from_data`, `_copy_episodes_metadata_and_stats`.
- [src/lerobot/datasets/utils.py](../../datasets/utils.py) — `DATA_DIR`, `DEFAULT_DATA_PATH` path templates already exist.

**GUI backend** ([api/datasets.py](../api/datasets.py), [api/playback.py](../api/playback.py), [api/edits.py](../api/edits.py), [state.py](../state.py), [api/models.py](../api/models.py)) — extend `DatasetInfo` with full features dict; new feature-values + feature-series endpoints; new `feature_set` `PendingEdit` type; `_apply_feature_set_edits()` calls `set_feature_values()`. Startup sweep cleans orphan `.tmp` files.

**GUI frontend** ([static/index.html](../static/index.html), [static/app.js](../static/app.js), [static/style.css](../static/style.css)) — Inspector, feature rows, vertical-slice selection, click-to-seek wiring, resize handles, grouped edits bar, overlapping-edit modal, range-format helper used by every chip/dialog/tooltip.

**Reused helpers**: [feature_utils.py](../../datasets/feature_utils.py) `validate_feature_dtype_and_shape`; [compute_stats.py](../../datasets/compute_stats.py) for scoped stats recomputation on Save.

**Tests**:

- new `tests/datasets/test_set_feature_values.py` — value rewrites, stats recomputation, sidecar lookup updates (`subtasks.parquet`), in-place vs forked output, `finalize()` is called.
- extend [tests/gui/test_state.py](../../../../tests/gui/test_state.py) for `feature_set` `PendingEdit` serialization.
- new `tests/gui/test_feature_endpoints.py` — schema, per-frame features, feature-series.
- new `tests/gui/test_feature_edits.py` — staging, validation, safety rails, end-to-end Save flow.
- new `tests/gui/test_overlapping_edits.py` — overlap detection at stage time + clipping behavior.

---

## Verification

`uv run pytest tests/gui -svv` after each phase. Schema validation rejects type-mismatched edits. Range semantics: staging `[120, 130)` writes 10 frames; UI shows "frames 120…129". Overlapping-edit modal fires and clips correctly. Per-episode broadcast features coerce to full episode.

Manual: open a dataset with `reward` → drag-select on a row → edit cards (slider, text, checkbox) → confirm chips appear and timeline preview toggles correctly → Save → reload → values persist.

# PR #3 (`gui/add_reward`) — Review Follow-ups

Tracker for issues found during review of [PR #3](https://github.com/TheWisp/lerobot/pull/3) (`gui/add_reward`). Companion to the design doc [`add_feature.md`](add_feature.md). Each entry has a severity, repro, proposed fix, and test plan.

## Status legend

- 🔴 **blocker** — fix before merge
- 🟠 **important** — fix soon, before users hit it in anger
- 🟡 **nit** — follow-up, file as issue
- ✅ **fixed** — landed on main

## Fixed (post-merge fix-forward)

| ID      | Commit        | Notes                                                                                                                  |
| ------- | ------------- | ---------------------------------------------------------------------------------------------------------------------- |
| B1      | `6ae00a9c6`   | Missing imports for `add_features_inplace`. Mechanical.                                                                |
| Quality | `22a3f7e80`   | SIM108 + missing safe-destruct + ruff format leftovers.                                                                |
| I2      | `76d26c146`   | `_atomic_swap_files` helper makes add / remove / rename atomic. Real red → green TDD cycle (xfail tests drove it).     |
| B2      | `8b76de1da`   | Per-episode widgets stage edit without prior selection (frontend `_resolvedRangeFor`).                                 |
| I1      | (this commit) | Banner surfaces lossy-migration plan; explicit `confirm()` before triggering `next.success` migration.                 |
| I3      | (this commit) | Read-only inspector renders formatted vector view (`renderReadOnlyView`); replaces "read-only in V1" placeholder tag.  |
| I4      | (this commit) | Multi-dim viz cap bumped 8 → 32; covers ALOHA / SO-100 / humanoid DOF cleanly.                                         |
| I5      | (this commit) | Dialog centered via position/inset/margin auto; explicit confirm with frame/episode counts; dataset-wide warning copy. |

---

## 🔴 B1. Missing imports in `dataset_tools.py`

**Repro**: every test in `tests/datasets/test_dataset_tools.py::test_add_features_inplace_*` and `tests/gui/test_feature_add_endpoints.py` fails with `NameError: name '_shape_value_for_column' is not defined`. The `Any` symbol used at `dataset_tools.py:1426` is also not imported.

**Fix** (mechanical, two lines):

```python
# src/lerobot/datasets/dataset_tools.py around line 956
from typing import Any  # noqa: E402

from lerobot.datasets.feature_value_edits import (  # noqa: E402, F401
    FeatureValueEdit,
    StatsRecomputationError,
    _shape_value_for_column,   # ← add
    set_feature_values,
)
```

**Test**: existing tests already cover this — they all pass once the import lands. No new test needed.

**Process note**: contributor cannot have run their own new tests before pushing. Worth a comment on the PR.

---

## 🔴 B2. Per-episode feature edits silently no-op when no selection

**Repro**:

1. Open a dataset with a per-episode `success` feature.
2. Click ✓ Success on the tri-state widget (without first drag-selecting in the timeline).
3. Observe: button shows hover/click feedback, but no edit is staged. Save reveals nothing pending.

**Root cause**: [`feature_editing.js:914`](../static/feature_editing.js#L914):

```js
async function stageFeatureEdit(featureName, value) {
    if (!selection) return;   // ← silent no-op
```

The success-segment click handler at [line 891](../static/feature_editing.js#L891) calls `stageFeatureEdit` directly. For per-episode features the backend coerces `frame_from=0, frame_to=ep_length` regardless of what the frontend sends ([`edits.py:326-327`](../api/edits.py#L326)) — so the selection literally doesn't matter for these features. Same trap exists for any widget that should work without a drag-selection.

**Fix**: in the success-segment click handler (and any other per-episode widget), synthesize a whole-episode selection when `selection` is null:

```js
} else if (kind === "success-segment") {
    w.addEventListener("click", () => {
        if (w.disabled) return;
        const v = parseInt(w.getAttribute("data-value"), 10);
        if (!selection && window.currentEpisode != null) {
            const epLen = window.datasets[window.currentDataset]
                ?.episodes?.[window.currentEpisode]?.length;
            if (epLen) {
                selection = {
                    datasetId: window.currentDataset,
                    episodeIndex: window.currentEpisode,
                    frameFrom: 0, frameTo: epLen, originRow: featureName,
                };
            }
        }
        stageFeatureEdit(featureName, v);
    });
}
```

**Test plan**: lerobot has no JS test runner today, so test this at the API integration layer. Add to `tests/gui/test_feature_add_endpoints.py`:

```python
def test_per_episode_edit_with_full_episode_range_persists(
    app_with_state, opened_dataset
):
    """A per-episode feature edit submitted with [0, ep_length) range round-trips
    correctly through staging → apply. This is what the success-segment click
    handler MUST send when no drag-selection exists."""
    app, state = app_with_state
    dataset_id, ds = opened_dataset

    # Add success (per_episode=True, int8 tri-state).
    _post_json(app, f"/api/datasets/{dataset_id}/features/defaults", None)

    ep_length = int(ds.meta.episodes[0]["length"])
    resp = _post_json(app, "/api/edits/feature-set", {
        "dataset_id": dataset_id, "episode_index": 0,
        "feature": "success",
        "frame_from": 0, "frame_to": ep_length,
        "value": 1,  # ✓ Success
    })
    assert resp.status_code == 200, resp.text

    # Apply via the Save endpoint and confirm the value persisted on disk.
    apply = _post_json(app, "/api/edits/apply", {"dataset_id": dataset_id})
    assert apply.status_code == 200

    import pyarrow.parquet as pq
    for f in (ds.root / "data").rglob("*.parquet"):
        t = pq.read_table(f, columns=["episode_index", "success"])
        df = t.to_pandas()
        ep0 = df[df["episode_index"] == 0]
        assert (ep0["success"] == 1).all(), f"success not +1 for episode 0 in {f}"
```

This proves the apply path is correct. The frontend fix is then trivially verifiable by manual click.

---

## 🟠 I1. Silent destructive `next.success` migration

**Repro**:

1. Open a dataset that has `next.success` (per-frame bool, written by `lerobot-eval`).
2. Click "Add missing features" on the banner.
3. Observe: `next.success` is gone, replaced by per-episode `success` (-1/0/+1). The frame-level timing of the success transition is **lost forever**, with no confirmation prompt, no progress indicator, no "Cancel" affordance, and no mention in the banner text.

**Root cause**: [`add_default_features` at `datasets.py:1755`](../api/datasets.py#L1755) unconditionally calls `_migrate_next_success_inplace(dataset)` if `next.success` is present and bool. The banner's rename note only surfaces compatible renames (`next.reward → reward`); the lossy bool→tri-state transformation is invisible to the user.

**Why this isn't just a nit**: `next.success` carries the exact frame index where the rollout succeeded. The collapsed `success=+1` discards that. A user who labeled their rollouts on `main` for a downstream RL training script that consumed `next.success` will lose information they can't recover.

**Fix** (recommended): make the migration explicit. Three options, in order of effort:

1. **Banner text update** (lowest effort): if `next.success` is bool and present, show in the banner:

   > Will replace `next.success` (per-frame) with `success` (per-episode tri-state). Frame-level success timing will be lost.
   > And require a separate confirmation modal before proceeding. Still triggered by "Add missing features" but gated.

2. **Split the migration off into a separate endpoint** with its own confirm flag. The defaults endpoint refuses to run if `next.success` exists; user has to call `POST /api/datasets/{id}/features/migrate-next-success?confirm=true` explicitly.

3. **Preserve both columns**: rename `next.success` → `next.success.original` (or similar) before adding `success`. User keeps the frame-level data. Doubles a tiny amount of disk per episode but recoverable.

**Test plan**: add to `tests/gui/test_feature_add_endpoints.py`:

```python
def test_next_success_migration_preserves_per_episode_correctness(
    app_with_state, tmp_path, empty_lerobot_dataset_factory
):
    """Migration: per-frame bool next.success → per-episode int8 success.
    Episode with any True frame → +1; all-False episode → -1."""
    app, state = app_with_state
    features = {
        "action": {"dtype": "float32", "shape": (2,), "names": None},
        "observation.state": {"dtype": "float32", "shape": (2,), "names": None},
        "next.success": {"dtype": "bool", "shape": (1,), "names": None},
    }
    ds = empty_lerobot_dataset_factory(root=tmp_path / "ds", features=features)
    # Episode 0: success at frame 5 (any-True)
    for i in range(10):
        ds.add_frame({
            "action": np.zeros(2, dtype=np.float32),
            "observation.state": np.zeros(2, dtype=np.float32),
            "next.success": np.array([i == 5], dtype=bool),
            "task": "t",
        })
    ds.save_episode()
    # Episode 1: all False (failure)
    for i in range(10):
        ds.add_frame({
            "action": np.zeros(2, dtype=np.float32),
            "observation.state": np.zeros(2, dtype=np.float32),
            "next.success": np.array([False], dtype=bool),
            "task": "t",
        })
    ds.save_episode()
    ds.finalize()

    dataset_id = str(ds.root)
    state.datasets[dataset_id] = ds
    resp = _post_json(app, f"/api/datasets/{dataset_id}/features/defaults", None)
    assert resp.status_code == 200, resp.text
    payload = resp.json()
    assert "next.success→success" in payload["renamed"]

    # Verify on-disk: episode 0 → +1, episode 1 → -1, next.success column is gone.
    import pyarrow.parquet as pq
    for f in (ds.root / "data").rglob("*.parquet"):
        t = pq.read_table(f).to_pandas()
        assert "next.success" not in t.columns
        ep0 = t[t["episode_index"] == 0]
        ep1 = t[t["episode_index"] == 1]
        if len(ep0):
            assert (ep0["success"] == 1).all()
        if len(ep1):
            assert (ep1["success"] == -1).all()


def test_next_success_migration_handles_multi_shard_episode(
    app_with_state, tmp_path, empty_lerobot_dataset_factory
):
    """If an episode spans 2+ data shards and the True frame is in shard B
    while shard A is all False, migration must still produce success=+1.
    The current implementation has an early-exit `if per_episode_value.get(ep_idx_int) == 1: continue`
    that should handle this — test confirms."""
    # ... build a dataset that produces multiple shards per episode
    # (force shard rotation via ds._shard_threshold or similar) and assert
    # the cross-shard reduction is correct.
    pass


def test_next_success_migration_is_lossy_by_design(...):
    """Document the lossiness as a property: after migration, the original
    frame index of the success transition is unrecoverable. This isn't a bug
    to fix — it's a contract to surface in the UI."""
    # Assert that next.success is gone post-migration. This is the canary
    # for any future change that tries to make the migration reversible —
    # such a change MUST be a deliberate decision, not a regression.
    pass
```

---

## 🟠 I2. `pending_renames` cleanup is asymmetric (Pass 2 crash window)

**Repro**: difficult to trigger naturally. Inject `os.replace` failure mid-loop:

```python
def test_partial_rename_leaves_inconsistent_schema(monkeypatch, small_dataset_no_video):
    """If os.replace fails partway through Pass 2 of add_features_inplace,
    some shards have the new column and some don't. The current writer doesn't
    revert; sweep on next open removes orphan tmps but leaves mixed schema."""
    ds = small_dataset_no_video
    real_replace = os.replace
    n_calls = [0]
    def flaky(src, dst):
        n_calls[0] += 1
        # Let Pass 1 (writing tmps) succeed, fail mid-Pass 2.
        if n_calls[0] == 2 and str(src).endswith(".tmp"):
            raise OSError("simulated crash mid-rename")
        return real_replace(src, dst)
    monkeypatch.setattr("lerobot.datasets.dataset_tools.os.replace", flaky)

    with pytest.raises(OSError):
        add_features_inplace(
            ds, features={"r": (0.0, {"dtype": "float32", "shape": [1], "names": None})}
        )

    # Now the on-disk state: shard 0 has 'r', shard 1 still has tmp + old.
    # Sweep should clean tmps but the schema is split.
    import pyarrow.parquet as pq
    shards = sorted((ds.root / "data").rglob("*.parquet"))
    has_r = ["r" in pq.read_table(s).column_names for s in shards]
    # Currently expect mixed True/False; documenting the issue.
    assert any(has_r) and not all(has_r), \
        "If you see this assertion fail, the writer was made transactional — update the test."
```

**Severity**: not a blocker because the failure mode is rare in practice (`os.replace` doesn't normally fail partway through a sequence) and the partial state isn't corrupting (just inconsistent — readers ignore unknown columns or fail loudly). But the test pins the current behavior so any future "we made it transactional" claim has a regression guard.

**Fix options** (post-merge):

- **Two-phase commit lite**: write a `pending-rename.json` manifest before Pass 2 listing all `(tmp, final)` pairs. On dataset open, if the manifest exists and points to incomplete renames, finish them before serving the dataset. Combined with the existing tmp sweep, this closes the gap.
- **Schema verification on open**: after sweep, compare the parquet schemas across shards. If they don't match, surface a "this dataset was interrupted during a save — reapply migration?" prompt.

---

## 🟡 N1. `_compatible_for_rename` ignores `names`

`next.reward` with `names: ["scalar_reward"]` → renamed to `reward` (which has `names: null`). `rename_overrides` clobbers `names` to None — probably correct, but no test confirms intentionality. Add a test pinning the behavior either way.

---

## 🟠 I3. Action / observation.\* not inspectable (user-reported bug)

**Repro**: open a dataset, click on the `action` or `observation.state` row in the timeline. Inspector shows the row as read-only but provides no way to actually see the recorded values — the user has no way to know what value those frames carry.

**Why this is a bug, not a nit**: the bundled commit `9388f5475` made `action` and `observation.*` show on the timeline by default specifically to expose recorded data, but the inspector's read-only path doesn't deliver on that promise. Either show the values clearly, or revert to hiding the rows.

**Recommendation**: render a read-only formatted view (`obs[0]: 0.123  obs[1]: -0.456 …`) for read-only vector features in the inspector. Pure frontend.

**Doc**: the design doc for `feature_editing` says action/obs are hidden by default; this PR flipped the default but didn't update the doc. Update `feature_editing.md` to match.

---

## 🟠 I4. Multi-dim vector visualization caps at 8 dims (user-reported bug)

`renderTrackSvg` at [`feature_editing.js:1331`](../static/feature_editing.js#L1331):

```js
if (Array.isArray(series[0]) && series[0].length <= 8) {
  // Mini multi-line: overlay up to 8 series.
}
if (Array.isArray(series[0])) {
  // Large vector: just render the norm of each vector.
  return numericLineSvg(norms, length);
}
```

A 14-DOF action (e.g. ALOHA bimanual) falls through to single-line L2 norm — confusing. Quick fix: bump to 16. Real fix: stack mini-rows per dim with a "+N more" affordance.

The shape claim in the row label (`float32[14]`) is correct, but the visualization shows only one curve, so the user-facing reading is "the GUI is lying about the data".

---

## 🟠 I5. Add Feature dialog: irreversibility + missing confirmation (user-reported bug)

- Dialog renders top-left instead of centered. CSS fix: `position: fixed; inset: 0; margin: auto;` or use the standard `<dialog>` show modal centering pattern.
- Adding a feature is irreversible (no undo, no save/discard for schema mutations) — needs a confirmation modal: "Add column `foo` (float32[1]) with fill `0.0` to all N frames across M episodes? Cannot be undone via Discard."
- Dialog doesn't communicate "this is a dataset-wide operation". Add explanatory copy.

---

## 🟡 N5. Pending-edits guard scope

Currently filters `edit_type == "feature_set"` only. `delete` and `trim` edit types exist and also rewrite shards on apply.

**Fix**: replace `pending_feature_set_edits_for_dataset` with the existing `get_edits_for_dataset(dataset_id)` and refuse on any non-empty list. Conservative + simple. Document the invariant in `state.py`:

```python
# Schema mutations (add/remove/rename feature columns) refuse to run while ANY
# pending edits exist on the dataset. The schema rewrite is a full-shard
# operation that races with any other pending shard mutation.
```

If a pending edit type is added later that _doesn't_ mutate shards (a hypothetical UI-only marker), the doc forces the author to think about whether the guard still applies.

---

## 🟡 N6. `_DEFAULT_FEATURE_SPECS` lives in the GUI app layer

`reward` and `success` are positioned as MUST-have columns "like `DEFAULT_FEATURES`", but the constant only exists in `gui/api/datasets.py`. CLI users don't get the auto-default behavior. The frontend mirrors part of it (`DEFAULT_RENAME_FROM` in `feature_editing.js`).

**Fix**: hoist `_DEFAULT_FEATURE_SPECS` to `lerobot.utils.constants` next to `DEFAULT_FEATURES`. Frontend reads it from a `/api/defaults` endpoint or inlines it from a single source. Three places → one.

---

## 🟡 N7. `_add_new_feature_stats_to_episodes` performance

Builds DataFrames row-by-row from per-episode dicts then `pd.concat`. O(episodes × features) DataFrame constructions. Acknowledged in code comment ("same drop-and-rebuild pattern as `_recompute_episode_stats_from_data`") but on real-size datasets (>1000 episodes) this could be noticeably slow. Profile on a real dataset before optimizing.

---

## 🟡 N8. `add_features_inplace` doesn't acquire the dataset lock

Only the API endpoints (`add_dataset_feature`, `add_default_features`, `remove_dataset_feature`) wrap calls in `_app_state.get_lock(dataset_id)`. CLI/notebook usage of the primitive itself is unprotected. If a user runs `add_features_inplace` from a notebook against a dataset open in the GUI (or vice versa), the writes can interleave.

**Fix**: file-level locking via `flock` on `info.json` inside the primitive. Documented as best-effort. Same change should apply to `set_feature_values`.

---

## 🟡 N9. Banner-managed defaults set is incomplete (`subtask_index` missing)

The PR added `reward` and `success` to `_DEFAULT_FEATURE_SPECS` (banner offers to add them when missing). But `subtask_index` — the storage-side counterpart of the synthesized `subtask` string feature — is arguably **more** core to the V1 design than reward/success: the design doc calls RECAP-style subtask labeling the flagship use case, the GUI already special-cases `subtask` → `subtask_index` in the synthetic-name translation logic ([`edits.py:_resolve_synthetic_feature`](../api/edits.py)), and `lerobot/scripts/lerobot_eval.py` doesn't write `next.success` everywhere — but every recording flow eventually wants subtask annotation.

Asymmetry is user-reported: "I'm pretty sure subtask is a default feature (more default than success and reward, actually)?"

**Two design directions** (post-V1):

1. **Add `subtask_index` to `_DEFAULT_FEATURE_SPECS`** with `dtype=int64, shape=[1], per_episode=True` (matching the existing per-episode coercion behavior of `subtask`). Banner offers to add it when missing. Plus the lookup table at `meta/subtasks.parquet` that the synthesis already requires — banner has to provision that too. Bigger surface than reward/success because of the lookup.

2. **Drop the dedicated banner entirely** in favor of a **"Common features" preset list inside the "+ Add feature" dialog** (already noted as a TODO follow-up). Each preset (reward / success / subtask_index / quality_score) fills the dialog with the right dtype/shape/per_episode/fill defaults. Same one-click convenience, no parallel banner UI, the `isBannerManaged` predicate goes away. Special migration paths (`next.reward → reward`, `next.success → success`) become explicit "Migrate from existing column?" sub-flows.

(2) is cleaner architecturally; (1) is closer to the current PR's spirit. Either way, the current state — reward and success special-cased, subtask_index left out — is the asymmetry to fix.

---

## 🟡 N10. `features_schema` open-response contract pinned by integration test

User reported: opening `/tmp/feature-editing-test-dataset` (which has reward + success on disk) showed the banner falsely claiming both were missing. Backend was confirmed correct via direct `_dataset_info_from` and `GET /api/datasets` checks — the response carries reward + success in `features_schema`. Root cause: stale browser cache of `feature_editing.js` from before the `?v=` cache-buster bumps.

To prevent regression of the **backend contract**, three integration tests were added in `tests/gui/test_feature_add_endpoints.py::TestOpenResponseSchemaContract`:

1. `test_open_response_carries_reward_and_success_when_present` — explicit `reward` and `success` end up in `features_schema`, with correct dtype and `per_episode` flag.
2. `test_list_datasets_endpoint_includes_features_schema` — the `GET /api/datasets` list endpoint (used by page-reload restore) returns full `features_schema` for each open dataset.
3. `test_features_legacy_list_and_features_schema_agree` — the legacy `features: list[str]` and `features_schema: dict` agree on the set of feature names. If they ever drift, the frontend has two sources of truth.

These pin the response shape; if the banner false-positives in the future, these tests will narrow it to either the frontend, the cache, or a regression in `_dataset_info_from`.

---

## Summary table

| ID  | Severity | Title                                                           | Action                                          |
| --- | -------- | --------------------------------------------------------------- | ----------------------------------------------- |
| B1  | 🔴       | Missing imports in `dataset_tools.py`                           | Mechanical fix                                  |
| B2  | 🔴       | Per-episode edits silently no-op when no selection (user-bug)   | Frontend fix + API integration test             |
| I1  | 🟠       | Silent destructive `next.success` migration                     | Confirmation modal + `_migrate_*` test coverage |
| I2  | 🟠       | `pending_renames` cleanup asymmetric (Pass 2 crash window)      | Fault-injection test + post-merge fix           |
| I3  | 🟠       | Action/obs not inspectable in read-only inspector (user-bug)    | Render read-only formatted vector view          |
| I4  | 🟠       | Multi-dim vector visualization caps at 8 dims (user-bug)        | Bump cap or per-dim mini-rows                   |
| I5  | 🟠       | Add Feature dialog UX: not centered, no confirmation (user-bug) | CSS centering + confirmation modal + copy       |
| N1  | 🟡       | `_compatible_for_rename` ignores `names`                        | Test pinning behavior                           |
| N5  | 🟡       | Pending-edits guard scope (only feature_set blocked)            | Switch to `get_edits_for_dataset` + invariant   |
| N6  | 🟡       | `_DEFAULT_FEATURE_SPECS` should live in `lerobot.utils`         | Hoist constant                                  |
| N7  | 🟡       | `_add_new_feature_stats_to_episodes` slow on large datasets     | Profile then optimize                           |
| N8  | 🟡       | Primitive doesn't acquire dataset lock                          | Add `flock` on info.json                        |

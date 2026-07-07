# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""Tests for Run, RunPaths, RunRegistry, and the state machine."""

from __future__ import annotations

import json
import time
from pathlib import Path

import pytest

from lerobot.gui.training.runs import (
    TERMINAL_STATES,
    Run,
    RunPaths,
    RunRegistry,
    RunState,
    append_event,
    new_idempotency_key,
    new_run_id,
)

# ── Run dataclass + state machine ─────────────────────────────────────────────


def _make_run(**over) -> Run:
    base = {
        "run_id": new_run_id(),
        "host_id": "this-server",
        "recipe_name": "act-default",
        "dataset_id": "ds/example",
        "args": {"num_steps": 100},
        "state": RunState.PENDING,
        "created_at": time.time(),
    }
    base.update(over)
    return Run(**base)


def test_run_to_from_json_roundtrip() -> None:
    r = _make_run()
    re = Run.from_json(r.to_json())
    assert re.run_id == r.run_id
    assert re.state == RunState.PENDING
    assert re.args == {"num_steps": 100}


def test_run_from_json_coerces_legacy_int_session_id() -> None:
    """Regression (found by the 2026-06-12 GPU smoke): pre-SSH run.json
    files persisted session_id as the raw subprocess PID (int). After the
    Protocol widened session_id to str, one legacy record made every
    list_runs() call 500 on DTO validation. from_json coerces on load."""
    import json as _json

    r = _make_run()
    d = _json.loads(r.to_json())
    d["session_id"] = 1230680  # legacy int PID
    re = Run.from_json(_json.dumps(d))
    assert re.session_id == "1230680"


def test_run_advance_pending_to_running_sets_started_at() -> None:
    r = _make_run()
    assert r.started_at is None
    r.advance(RunState.RUNNING)
    assert r.state == RunState.RUNNING
    assert r.started_at is not None


def test_run_advance_to_terminal_sets_finished_at() -> None:
    r = _make_run()
    r.advance(RunState.RUNNING)
    r.advance(RunState.COMPLETING)
    assert r.finished_at is None
    r.advance(RunState.COMPLETED)
    assert r.finished_at is not None


def test_run_advance_illegal_transition_asserts() -> None:
    r = _make_run()
    r.advance(RunState.RUNNING)
    r.advance(RunState.COMPLETED)
    with pytest.raises(AssertionError, match="illegal state transition"):
        r.advance(RunState.RUNNING)  # COMPLETED → RUNNING not allowed


def test_run_advance_pending_to_failed_allowed() -> None:
    """Launch failure: PENDING can go straight to FAILED."""
    r = _make_run()
    r.advance(RunState.FAILED)
    assert r.state == RunState.FAILED


def test_terminal_states_set() -> None:
    assert {RunState.COMPLETED, RunState.FAILED, RunState.ABORTED} == TERMINAL_STATES


# ── RunPaths ───────────────────────────────────────────────────────────────────


def test_run_paths_for_run_with_dir(tmp_path: Path) -> None:
    p = RunPaths.for_run("abc123", runs_dir=tmp_path)
    assert p.root == tmp_path / "abc123"
    assert p.run_json == tmp_path / "abc123" / "run.json"
    assert p.progress_json == tmp_path / "abc123" / "progress.json"
    assert p.events_jsonl == tmp_path / "abc123" / "events.jsonl"
    assert p.checkpoints_jsonl == tmp_path / "abc123" / "checkpoints.jsonl"
    assert p.stderr_log == tmp_path / "abc123" / "stderr.log"
    assert p.checkpoints_dir == tmp_path / "abc123" / "checkpoints"


def test_run_paths_ensure_exists_creates(tmp_path: Path) -> None:
    """ensure_exists creates the run root but NOT checkpoints/ — pre-creating
    an empty checkpoints/ would confuse scanners that use its existence as
    the "is this a training run?" signal."""
    p = RunPaths.for_run("xyz", runs_dir=tmp_path)
    assert not p.root.exists()
    p.ensure_exists()
    assert p.root.is_dir()
    assert not p.checkpoints_dir.exists()


# ── RunRegistry ────────────────────────────────────────────────────────────────


def test_registry_save_load_roundtrip(tmp_path: Path) -> None:
    reg = RunRegistry(runs_dir=tmp_path)
    r = _make_run()
    reg.save(r)
    loaded = reg.load(r.run_id)
    assert loaded is not None
    assert loaded.run_id == r.run_id
    assert loaded.recipe_name == r.recipe_name


def test_registry_load_missing_returns_none(tmp_path: Path) -> None:
    reg = RunRegistry(runs_dir=tmp_path)
    assert reg.load("missing") is None


def test_registry_list_all_newest_first(tmp_path: Path) -> None:
    reg = RunRegistry(runs_dir=tmp_path)
    r1 = _make_run(created_at=1.0)
    r2 = _make_run(created_at=2.0)
    r3 = _make_run(created_at=3.0)
    reg.save(r1)
    reg.save(r2)
    reg.save(r3)
    runs = reg.list_all()
    assert [r.run_id for r in runs] == [r3.run_id, r2.run_id, r1.run_id]


def test_registry_list_all_skips_malformed(tmp_path: Path) -> None:
    reg = RunRegistry(runs_dir=tmp_path)
    r = _make_run()
    reg.save(r)
    # Drop a malformed run.json
    bad_dir = tmp_path / "broken"
    bad_dir.mkdir()
    (bad_dir / "run.json").write_text("{not-json")
    runs = reg.list_all()
    assert len(runs) == 1
    assert runs[0].run_id == r.run_id


def test_registry_list_all_empty_dir(tmp_path: Path) -> None:
    reg = RunRegistry(runs_dir=tmp_path / "does-not-exist")
    assert reg.list_all() == []


def test_registry_find_by_idempotency_key_matches(tmp_path: Path) -> None:
    reg = RunRegistry(runs_dir=tmp_path)
    key = new_idempotency_key()
    r = _make_run(idempotency_key=key)
    reg.save(r)
    found = reg.find_by_idempotency_key(key)
    assert found is not None
    assert found.run_id == r.run_id


def test_registry_find_by_idempotency_key_no_match(tmp_path: Path) -> None:
    reg = RunRegistry(runs_dir=tmp_path)
    r = _make_run(idempotency_key="key-a")
    reg.save(r)
    assert reg.find_by_idempotency_key("key-b") is None
    assert reg.find_by_idempotency_key("") is None


def test_registry_active_run_on_host(tmp_path: Path) -> None:
    reg = RunRegistry(runs_dir=tmp_path)
    # Finished run on the host — doesn't count
    r_done = _make_run(host_id="server-a")
    r_done.advance(RunState.RUNNING)
    r_done.advance(RunState.COMPLETED)
    reg.save(r_done)
    assert reg.active_run_on_host("server-a") is None
    # Active run — counts
    r_active = _make_run(host_id="server-a")
    r_active.advance(RunState.RUNNING)
    reg.save(r_active)
    found = reg.active_run_on_host("server-a")
    assert found is not None
    assert found.run_id == r_active.run_id


def test_registry_atomic_save(tmp_path: Path) -> None:
    """A save with a JSON-uncodable value would surface the error before
    leaving a partial file behind. Sanity: normal save creates the file."""
    reg = RunRegistry(runs_dir=tmp_path)
    r = _make_run()
    reg.save(r)
    assert (tmp_path / r.run_id / "run.json").exists()
    # No leftover .tmp files
    tmps = list((tmp_path / r.run_id).glob("*.tmp"))
    assert tmps == []


def test_new_run_id_unique_enough() -> None:
    ids = {new_run_id() for _ in range(1000)}
    assert len(ids) == 1000


# ── append_event ───────────────────────────────────────────────────────────────


def test_append_event_creates_file_and_appends(tmp_path: Path) -> None:
    p = tmp_path / "events.jsonl"
    append_event(p, "started", host_id="this-server")
    append_event(p, "completed_naturally", final_step=100, exit_code=0)
    lines = p.read_text().splitlines()
    assert len(lines) == 2
    e1 = json.loads(lines[0])
    e2 = json.loads(lines[1])
    assert e1["type"] == "started"
    assert e1["host_id"] == "this-server"
    assert "ts" in e1
    assert e2["type"] == "completed_naturally"
    assert e2["final_step"] == 100


def test_append_event_creates_parent_dir(tmp_path: Path) -> None:
    p = tmp_path / "deeply" / "nested" / "events.jsonl"
    append_event(p, "started")
    assert p.exists()

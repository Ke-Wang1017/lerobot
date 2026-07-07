# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""The Models tab should identify a run by its human name (the GUI run.json's
recipe_name), not the random run_id directory — round-5 "which output is which
model?". The run_id stays available so the dir is still locatable.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
sft = pytest.importorskip("safetensors.torch")

from lerobot.gui.api.models import (  # noqa: E402
    _dir_has_step_subdirs,
    _is_step_dir,
    _scan_source,
    _scan_training_run,
)


def _make_checkpoint(run_dir: Path, step: str = "000030") -> None:
    pretrained = run_dir / "output" / "checkpoints" / step / "pretrained_model"
    pretrained.mkdir(parents=True)
    sft.save_file({"w": torch.zeros(1)}, str(pretrained / "model.safetensors"))
    (pretrained / "config.json").write_text(json.dumps({"type": "act"}))


def test_scan_uses_run_name_from_run_json(tmp_path: Path) -> None:
    _make_checkpoint(tmp_path)
    (tmp_path / "run.json").write_text(json.dumps({"recipe_name": "my-cool-run", "created_at": 123.0}))

    result = _scan_training_run(tmp_path)
    assert result is not None
    assert result["name"] == "my-cool-run"  # human name, not the hash dir
    assert result["run_id"] == tmp_path.name  # id still available
    assert result["created_at"] == 123.0


def test_scan_falls_back_to_dir_name_without_run_json(tmp_path: Path) -> None:
    _make_checkpoint(tmp_path, step="000010")  # converted / non-GUI: no run.json

    result = _scan_training_run(tmp_path)
    assert result is not None
    assert result["name"] == tmp_path.name


def test_scan_source_keeps_run_name_not_dir_hash(tmp_path: Path) -> None:
    """Integration: _scan_source (what the Models-tab endpoint calls) must NOT
    overwrite the recipe_name with the run-id dir name. Round-7: _scan_training_run
    set the name but _scan_recursive clobbered it, so the tree showed hashes."""
    run_dir = tmp_path / "0ab7fbe199c7"  # random run-id dir name
    _make_checkpoint(run_dir)
    (run_dir / "run.json").write_text(json.dumps({"recipe_name": "round7-final"}))

    entries = _scan_source(str(tmp_path))
    assert len(entries) == 1
    assert entries[0]["name"] == "round7-final"  # the name, not the hash dir
    assert entries[0]["run_id"] == "0ab7fbe199c7"


def test_scan_source_falls_back_to_relpath_without_run_json(tmp_path: Path) -> None:
    run_dir = tmp_path / "abc123"
    _make_checkpoint(run_dir, step="000010")  # no run.json → disambiguating rel-path name

    entries = _scan_source(str(tmp_path))
    assert len(entries) == 1
    assert entries[0]["name"] == "abc123"


def test_gui_runs_source_tracks_orchestrator_runs_dir() -> None:
    """The auto-registered GUI-runs model source must follow the orchestrator's
    actual RUNS_DIR (honours LEROBOT_RUNS_DIR), not a hardcoded path — else a
    custom runs dir leaves trained models unscanned in the Models tab (round-6)."""
    from lerobot.gui.api import models
    from lerobot.gui.training.runs import RUNS_DIR

    assert str(RUNS_DIR) == models._GUI_RUNS_SOURCE


# --- Checkpoint-dir naming conventions. The scanner must recognize every layout a real trainer
#     writes, not only zero-padded numerics. Regression: HF / HVLA S1-standalone "checkpoint-NNNNN"
#     runs went silently undetected from 2026-06-07 (commit 64f0233) — the run dir was skipped with
#     no error because `"checkpoint-50000".lstrip("0").isdigit()` is False. ---


@pytest.mark.parametrize(
    "name, is_step",
    [
        ("000005", True),  # lerobot-train, zero-padded
        ("008000", True),
        ("50000", True),  # unpadded numeric
        ("000000", True),  # all-zeros parses cleanly (the old lstrip("0") dropped it)
        ("checkpoint-50000", True),  # HF Trainer / HVLA S1-standalone
        ("checkpoint-0", True),
        ("last", False),  # the rolling symlink is not a step dir
        ("checkpoint", False),
        ("checkpoint-", False),
        ("checkpoint-50000-backup", False),  # a backup copy, not a live step
        ("logs", False),
    ],
)
def test_is_step_dir(name: str, is_step: bool) -> None:
    assert _is_step_dir(name) is is_step


def _make_standard_checkpoint(run_dir: Path, step_dir: str) -> None:
    """Standard lerobot layout: <run_dir>/checkpoints/<step_dir>/pretrained_model/..."""
    pretrained = run_dir / "checkpoints" / step_dir / "pretrained_model"
    pretrained.mkdir(parents=True)
    sft.save_file({"w": torch.zeros(1)}, str(pretrained / "model.safetensors"))
    (pretrained / "config.json").write_text(json.dumps({"type": "hvla_flow_s1"}))


@pytest.mark.parametrize("step_dir", ["000050", "checkpoint-50000"])
def test_scan_detects_both_checkpoint_naming_conventions(tmp_path: Path, step_dir: str) -> None:
    """A run is detected whether its checkpoints are bare-numeric (lerobot-train) or checkpoint-N
    (HF Trainer / S1-standalone). The latter regressed in 64f0233; the real casualty was
    flow_s1_no_s2_v1, which vanished from the Models tab."""
    run_dir = tmp_path / "flow_s1_no_s2_v1"
    _make_standard_checkpoint(run_dir, step_dir)
    entries = _scan_source(str(tmp_path))
    assert [e["name"] for e in entries] == ["flow_s1_no_s2_v1"]


def test_dir_has_step_subdirs_rejects_placeholder(tmp_path: Path) -> None:
    """The filter's real purpose — empty / non-step checkpoints dirs stay rejected."""
    ckpts = tmp_path / "checkpoints"
    ckpts.mkdir()
    assert not _dir_has_step_subdirs(ckpts)  # empty
    (ckpts / "logs").mkdir()
    assert not _dir_has_step_subdirs(ckpts)  # only a non-step child
    assert not _dir_has_step_subdirs(tmp_path / "nonexistent")

# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""Tests for src/lerobot/gui/training/jobs.py.

Covers the types boundary between server + worker: TrainingJobConfig
roundtrip, paths derivation, PollScheduler backoff (the resilience-
design heart), error classification, HostProfile load/save, atomic
JSON write, events.jsonl appender, and TrainingJobState merge invariant.

Parallel of tests/gui/test_hub_jobs.py — same shape of tests for the
analogous training-side module.
"""

from __future__ import annotations

import dataclasses
import json
import time

import pytest

from lerobot.gui.training import jobs as tj

# ── TrainingJobConfig ───────────────────────────────────────────────────────


def _example_config(**overrides) -> tj.TrainingJobConfig:
    defaults = {
        "job_id": "test-job-1",
        "host_name": "runpod-h100",
        "dataset_id": "lerobot/pusht",
        "recipe_name": "act-default",
        "args": ["--policy.type=act", "--steps=10"],
        "image_ref": "ghcr.io/thewisp/lerobot-training:latest",
    }
    defaults.update(overrides)
    return tj.TrainingJobConfig(**defaults)


class TestTrainingJobConfig:
    def test_json_roundtrip_preserves_fields(self):
        cfg = _example_config()
        roundtripped = tj.TrainingJobConfig.from_json(cfg.to_json())
        assert roundtripped == cfg

    def test_immutable(self):
        cfg = _example_config()
        # dataclasses.FrozenInstanceError on a frozen=True dataclass
        with pytest.raises(dataclasses.FrozenInstanceError):
            cfg.job_id = "different"  # type: ignore[misc] — testing the frozen invariant

    def test_default_bind_local_is_false(self):
        cfg = _example_config()
        assert cfg.bind_local_src is False

    def test_default_jobs_dir(self):
        cfg = _example_config()
        assert cfg.jobs_dir == str(tj.JOBS_DIR)


# ── TrainingJobPaths ────────────────────────────────────────────────────────


class TestTrainingJobPaths:
    def test_for_job_derives_per_job_subdir(self, tmp_path):
        paths = tj.TrainingJobPaths.for_job("abc123", jobs_dir=tmp_path)
        assert paths.base == tmp_path / "abc123"
        assert paths.progress == tmp_path / "abc123" / "progress.json"
        assert paths.events == tmp_path / "abc123" / "events.jsonl"
        assert paths.log == tmp_path / "abc123" / "stderr.log"
        assert paths.pid == tmp_path / "abc123" / "worker.pid"

    def test_ensure_dir_creates_parent(self, tmp_path):
        paths = tj.TrainingJobPaths.for_job("xyz", jobs_dir=tmp_path)
        assert not paths.base.exists()
        paths.ensure_dir()
        assert paths.base.is_dir()


# ── PollScheduler (THE resilience-design heart) ─────────────────────────────


class TestPollScheduler:
    def test_first_success_returns_base_interval(self):
        s = tj.PollScheduler()
        assert s.schedule_next(success=True) == 5.0

    def test_success_resets_consecutive_failures(self):
        s = tj.PollScheduler()
        s.schedule_next(success=False)
        s.schedule_next(success=False)
        assert s.consecutive_failures == 2
        s.schedule_next(success=True)
        assert s.consecutive_failures == 0

    def test_exponential_backoff_timeline(self):
        """The documented timeline: 5, 10, 20, 40, 60 (cap), 60, ..."""
        s = tj.PollScheduler()
        delays = [s.schedule_next(success=False) for _ in range(7)]
        assert delays == [5.0, 10.0, 20.0, 40.0, 60.0, 60.0, 60.0]

    def test_max_attempts_returns_none(self):
        s = tj.PollScheduler()
        # 9 failures take us to consecutive_failures=9
        for _ in range(9):
            assert s.schedule_next(success=False) is not None
        # 10th failure: consecutive_failures becomes 10 == MAX_ATTEMPTS → None
        assert s.schedule_next(success=False) is None

    def test_permanent_failure_returns_none_immediately(self):
        """Auth failures shouldn't burn 10 retries."""
        s = tj.PollScheduler()
        assert s.schedule_next(success=False, permanent=True) is None
        # No backoff state was updated (we gave up before scheduling)
        assert s.consecutive_failures == 0

    def test_is_failing_property(self):
        s = tj.PollScheduler()
        assert not s.is_failing
        s.schedule_next(success=False)
        assert s.is_failing
        s.schedule_next(success=True)
        assert not s.is_failing


# ── classify_ssh_error ──────────────────────────────────────────────────────


class TestClassifySshError:
    def test_timeout_is_transient(self):
        assert tj.classify_ssh_error(TimeoutError("timed out")) == "transient"

    def test_connection_refused_is_transient(self):
        assert tj.classify_ssh_error(ConnectionRefusedError("refused")) == "transient"

    def test_connection_reset_is_transient(self):
        assert tj.classify_ssh_error(ConnectionResetError("reset")) == "transient"

    def test_socket_timeout_is_transient(self):
        assert tj.classify_ssh_error(TimeoutError("timed out")) == "transient"

    def test_ehostunreach_is_transient(self):
        e = OSError(113, "No route to host")
        assert tj.classify_ssh_error(e) == "transient"

    def test_auth_exception_by_class_name_is_permanent(self):
        """Paramiko exception names are matched without hard-importing paramiko.

        We deliberately name these mocks with the `Exception` suffix to
        mirror paramiko's actual class names — classify_ssh_error matches
        by class name (`type(exc).__name__`), so the name has to match.
        Ruff's N818 ("exception name should end in Error") doesn't apply
        here because we're explicitly mimicking external API names.
        """

        class AuthenticationException(Exception):  # noqa: N818 — mimics paramiko
            pass

        assert tj.classify_ssh_error(AuthenticationException("bad key")) == "permanent"

    def test_bad_host_key_is_permanent(self):
        class BadHostKeyException(Exception):  # noqa: N818 — mimics paramiko
            pass

        assert tj.classify_ssh_error(BadHostKeyException("changed")) == "permanent"

    def test_ssh_exception_with_host_key_message_is_permanent(self):
        class SSHException(Exception):  # noqa: N818 — mimics paramiko
            pass

        assert tj.classify_ssh_error(SSHException("host key verification failed")) == "permanent"

    def test_unknown_error_defaults_to_transient(self):
        """We prefer over-retrying transient errors to giving up too soon."""
        assert tj.classify_ssh_error(RuntimeError("mystery")) == "transient"


# ── HostProfile ─────────────────────────────────────────────────────────────


class TestHostProfile:
    def test_endpoint_format(self):
        hp = tj.HostProfile(name="x", ssh_user="feit", ssh_host="1.2.3.4", ssh_port=22)
        assert hp.ssh_endpoint == "feit@1.2.3.4:22"

    def test_default_port_22(self):
        hp = tj.HostProfile(name="x", ssh_user="feit", ssh_host="1.2.3.4")
        assert hp.ssh_port == 22

    def test_save_then_load_roundtrip(self, tmp_path):
        hp = tj.HostProfile(
            name="runpod-h100",
            ssh_user="root",
            ssh_host="213.45.10.22",
            ssh_port=51234,
            kind="temporary",
            workdir="/workspace/lerobot",
            image_ref="ghcr.io/thewisp/lerobot-training:0.4.0",
            persistent_volume="/workspace",
            capabilities={"gpu": "H100 80GB", "cuda": "12.4"},
        )
        path = hp.save(dir_=tmp_path)
        assert path == tmp_path / "runpod-h100.json"
        loaded = tj.HostProfile.load(path)
        assert loaded == hp

    def test_default_kind_temporary(self):
        hp = tj.HostProfile(name="x", ssh_user="u", ssh_host="h")
        assert hp.kind == "temporary"

    def test_display_name_defaults_to_name(self):
        hp = tj.HostProfile(name="lab", ssh_user="u", ssh_host="h")
        assert hp.display_name == "lab"

    def test_display_name_respected_when_provided(self):
        hp = tj.HostProfile(name="lab", ssh_user="u", ssh_host="h", display_name="Lab GPU")
        assert hp.display_name == "Lab GPU"

    def test_load_drops_unknown_keys(self, tmp_path):
        path = tmp_path / "x.json"
        path.write_text(
            '{"name":"x","ssh_user":"u","ssh_host":"h","ssh_port":22,'
            '"future_field":42,"another_unknown":"ignored"}'
        )
        loaded = tj.HostProfile.load(path)
        assert loaded.name == "x"
        assert loaded.ssh_host == "h"


class TestHostProfileLoadAll:
    def test_load_all_returns_all_profiles(self, tmp_path):
        for n in ("lab-a", "lab-b", "lab-c"):
            tj.HostProfile(name=n, ssh_user="u", ssh_host=f"{n}.lan").save(dir_=tmp_path)
        loaded = tj.HostProfile.load_all(tmp_path)
        assert sorted(p.name for p in loaded) == ["lab-a", "lab-b", "lab-c"]

    def test_load_all_ignores_non_json(self, tmp_path):
        tj.HostProfile(name="real", ssh_user="u", ssh_host="h").save(dir_=tmp_path)
        (tmp_path / "README.md").write_text("not a profile")
        (tmp_path / "host.json.bak").write_text("not a profile either")
        loaded = tj.HostProfile.load_all(tmp_path)
        assert [p.name for p in loaded] == ["real"]

    def test_load_all_handles_missing_dir(self, tmp_path):
        assert tj.HostProfile.load_all(tmp_path / "does-not-exist") == []

    def test_load_all_skips_unreadable_file(self, tmp_path):
        tj.HostProfile(name="good", ssh_user="u", ssh_host="h").save(dir_=tmp_path)
        (tmp_path / "broken.json").write_text("{not valid json")
        loaded = tj.HostProfile.load_all(tmp_path)
        assert [p.name for p in loaded] == ["good"]


class TestHostProfileDelete:
    def test_delete_removes_file(self, tmp_path):
        tj.HostProfile(name="lab", ssh_user="u", ssh_host="h").save(dir_=tmp_path)
        assert tj.HostProfile.delete("lab", dir_=tmp_path) is True
        assert not (tmp_path / "lab.json").exists()

    def test_delete_missing_is_idempotent(self, tmp_path):
        assert tj.HostProfile.delete("nope", dir_=tmp_path) is False


# ── atomic_write_json ───────────────────────────────────────────────────────


class TestAtomicWriteJson:
    def test_writes_file_then_replaces_atomically(self, tmp_path):
        target = tmp_path / "progress.json"
        tj.atomic_write_json(target, {"step": 100, "loss": 0.5})
        assert json.loads(target.read_text()) == {"step": 100, "loss": 0.5}

    def test_no_tmp_file_left_behind_on_success(self, tmp_path):
        target = tmp_path / "progress.json"
        tj.atomic_write_json(target, {"x": 1})
        # Only progress.json should exist — the .tmp got renamed away
        assert sorted(p.name for p in tmp_path.iterdir()) == ["progress.json"]

    def test_overwrites_existing(self, tmp_path):
        target = tmp_path / "progress.json"
        target.write_text(json.dumps({"old": True}))
        tj.atomic_write_json(target, {"new": True})
        assert json.loads(target.read_text()) == {"new": True}


# ── append_event ────────────────────────────────────────────────────────────


class TestAppendEvent:
    def test_appends_jsonl_line(self, tmp_path):
        events = tmp_path / "events.jsonl"
        tj.append_event(events, "connected", host="runpod")
        tj.append_event(events, "poll_failed", error_class="transient", attempt=1)

        lines = events.read_text().strip().split("\n")
        assert len(lines) == 2

        first = json.loads(lines[0])
        assert first["kind"] == "connected"
        assert first["host"] == "runpod"
        assert isinstance(first["ts"], float)

        second = json.loads(lines[1])
        assert second["kind"] == "poll_failed"
        assert second["error_class"] == "transient"
        assert second["attempt"] == 1

    def test_creates_parent_dir_if_missing(self, tmp_path):
        events = tmp_path / "subdir" / "events.jsonl"
        assert not events.parent.exists()
        tj.append_event(events, "started")
        assert events.exists()

    def test_appends_dont_overwrite(self, tmp_path):
        events = tmp_path / "events.jsonl"
        for i in range(5):
            tj.append_event(events, "poll_failed", attempt=i)
        lines = events.read_text().strip().split("\n")
        assert len(lines) == 5


# ── TrainingJobState ────────────────────────────────────────────────────────


class TestTrainingJobState:
    def test_initial_state(self):
        s = tj.make_training_job(host_name="h", dataset_id="d", recipe_name="r")
        assert s.status == "pending"
        assert s.connection_state == "initial"
        assert s.step == 0
        assert s.consecutive_poll_failures == 0
        assert len(s.job_id) == 32  # uuid4().hex

    def test_merge_progress_updates_fields(self):
        s = tj.make_training_job(host_name="h", dataset_id="d", recipe_name="r")
        s.merge_progress({"status": "running", "step": 42, "loss_recent": [0.1, 0.08]})
        assert s.status == "running"
        assert s.step == 42
        assert s.loss_recent == [0.1, 0.08]

    def test_terminal_state_is_monotonic(self):
        """Once complete/failed/cancelled, a stale read can't roll it back."""
        s = tj.make_training_job(host_name="h", dataset_id="d", recipe_name="r")
        s.status = "complete"
        s.merge_progress({"status": "running", "step": 999})
        assert s.status == "complete"  # didn't roll back
        assert s.step == 0  # the whole merge was a no-op

    def test_merge_ignores_none_values(self):
        s = tj.make_training_job(host_name="h", dataset_id="d", recipe_name="r")
        s.step = 100
        s.merge_progress({"step": None, "loss_recent": [0.5]})
        assert s.step == 100  # not overwritten by None
        assert s.loss_recent == [0.5]


# ── make_training_job ───────────────────────────────────────────────────────


class TestMakeTrainingJob:
    def test_each_call_returns_unique_job_id(self):
        a = tj.make_training_job(host_name="h", dataset_id="d", recipe_name="r")
        b = tj.make_training_job(host_name="h", dataset_id="d", recipe_name="r")
        assert a.job_id != b.job_id

    def test_started_at_is_recent(self):
        s = tj.make_training_job(host_name="h", dataset_id="d", recipe_name="r")
        assert abs(s.started_at - time.time()) < 1.0

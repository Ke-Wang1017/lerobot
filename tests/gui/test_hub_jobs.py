# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Unit tests for the Hub-transfer helper layer.

What this file covers:
  * JobConfig JSON round-trip
  * JobPaths derives consistent locations from a job_id
  * HubJobState shape + merge_progress invariants (won't un-terminalize)
  * PID-file identity: pid_file_payload, is_worker_alive against live + dead processes
  * Error classification across the HF exception hierarchy
  * Milestone extraction against recorded HF stderr samples
  * Upload-side file enumeration (respects ignore patterns, sorted)
  * check_upload_completeness logic (fresh repo, missing locally, incomplete)

What this file does NOT cover (separate test files):
  * Worker subprocess end-to-end — see test_hub_worker.py
  * Endpoint flow with FastAPI TestClient — see test_hub_endpoints.py
  * Real HF interaction — see test_hub_live.py (gated by ``@pytest.mark.hub_live``)
"""

from __future__ import annotations

import json
import os
import time

import pytest

from lerobot.gui import hub_jobs

# ── JobConfig ───────────────────────────────────────────────────────────────


class TestJobConfig:
    def _good(self, **overrides):
        defaults = {
            "job_id": "abc123",
            "dataset_id": "user/foo",
            "direction": "upload",
            "repo_id": "user/foo",
            "repo_type": "dataset",
            "local_path": "/tmp/foo",
            "jobs_dir": "/tmp/jobs",
        }
        defaults.update(overrides)
        return hub_jobs.JobConfig(**defaults)

    def test_json_roundtrip_preserves_all_fields(self):
        cfg = self._good(
            private=False,
            commit_message="hello",
            ignore_patterns=(".cache/", ".DS_Store"),
            reuse_pr_num=42,
        )
        round_tripped = hub_jobs.JobConfig.from_json(cfg.to_json())
        assert round_tripped == cfg

    def test_rejects_bad_direction(self):
        with pytest.raises(ValueError, match="bad direction"):
            self._good(direction="sideways")

    def test_rejects_bad_repo_type(self):
        with pytest.raises(ValueError, match="bad repo_type"):
            self._good(repo_type="bucket")

    def test_ignore_patterns_normalized_to_tuple(self):
        cfg = self._good(ignore_patterns=(".cache/",))
        # Round-trip through JSON to confirm the list-roundtrip turns it back to tuple.
        rt = hub_jobs.JobConfig.from_json(cfg.to_json())
        assert isinstance(rt.ignore_patterns, tuple)
        assert rt.ignore_patterns == (".cache/",)

    def test_ignore_patterns_none_stays_none(self):
        cfg = self._good(ignore_patterns=None)
        rt = hub_jobs.JobConfig.from_json(cfg.to_json())
        assert rt.ignore_patterns is None


# ── JobPaths ────────────────────────────────────────────────────────────────


class TestJobPaths:
    def test_paths_share_directory_and_job_id_prefix(self, tmp_path):
        paths = hub_jobs.JobPaths.for_job("abc123", tmp_path)
        assert paths.progress == tmp_path / "abc123.json"
        assert paths.log == tmp_path / "abc123.log"
        assert paths.pid == tmp_path / "abc123.pid"

    def test_paths_for_different_jobs_dont_collide(self, tmp_path):
        p1 = hub_jobs.JobPaths.for_job("aaa", tmp_path)
        p2 = hub_jobs.JobPaths.for_job("bbb", tmp_path)
        assert p1.progress != p2.progress
        assert p1.log != p2.log
        assert p1.pid != p2.pid


# ── HubJobState ─────────────────────────────────────────────────────────────


class TestHubJobState:
    def test_make_job_starts_pending(self):
        j = hub_jobs.make_job(dataset_id="ds", direction="upload", repo_id="u/r")
        assert j.status == "pending"
        assert j.started_at > 0
        assert j.finished_at is None
        assert j.pr_num is None
        assert j.error is None

    def test_initial_milestone_is_direction_aware(self):
        """Pre-worker-spawn milestone must say which direction the job is.

        A bare "starting" was historically rendered in the tray before the
        worker process attached, leading users to misread an upload as a
        download (see the Hub-transfers UX bug). The default now spells the
        direction out so the tray cannot ambiguate.
        """
        up = hub_jobs.make_job(dataset_id="ds", direction="upload", repo_id="u/r")
        down = hub_jobs.make_job(dataset_id="ds", direction="download", repo_id="u/r")
        assert up.milestone == "Starting upload"
        assert down.milestone == "Starting download"

    def test_unique_job_ids(self):
        ids = {
            hub_jobs.make_job(dataset_id="ds", direction="upload", repo_id="u/r").job_id for _ in range(20)
        }
        assert len(ids) == 20

    def test_merge_progress_updates_live_state(self):
        j = hub_jobs.make_job(dataset_id="ds", direction="upload", repo_id="u/r")
        j.status = "running"
        j.merge_progress(
            {
                "status": "running",
                "milestone": "Uploading files",
                "milestone_at": 12345.0,
                "files_total": 10,
                "files_done_estimate": 4,
                "pr_num": 7,
            }
        )
        assert j.milestone == "Uploading files"
        assert j.files_total == 10
        assert j.files_done_estimate == 4
        assert j.pr_num == 7

    def test_merge_progress_cannot_un_terminalize(self):
        """Once status is terminal, no merge from the worker drags it back.

        Protects against a confused worker writing a stale snapshot after
        we've already marked the job failed/cancelled server-side.
        """
        j = hub_jobs.make_job(dataset_id="ds", direction="upload", repo_id="u/r")
        j.status = "failed"
        j.error = "auth"
        j.merge_progress({"status": "running", "milestone": "Uploading"})
        assert j.status == "failed"
        # Other fields also don't move (the snapshot is ignored entirely).
        # The default milestone is direction-aware so a user looking at the
        # tray during the brief pre-spawn window sees "Starting upload" rather
        # than a context-free "starting" that historically read as ambiguous
        # (could be download — see the original Hub-transfers UX bug report).
        assert j.milestone == "Starting upload"

    def test_to_dict_omits_nothing_serialisable(self):
        """Sanity: every field in to_dict() is JSON-serialisable."""
        j = hub_jobs.make_job(dataset_id="ds", direction="upload", repo_id="u/r")
        json.dumps(j.to_dict())  # raises if any value is non-JSON


# ── PID-file identity ───────────────────────────────────────────────────────


class TestPidFileIdentity:
    def test_payload_contains_pid_and_started_at(self):
        payload = hub_jobs.pid_file_payload(os.getpid())
        assert payload["pid"] == os.getpid()
        assert "start_time" in payload
        assert "started_at" in payload
        assert payload["started_at"] > 0

    def test_alive_check_for_current_process_is_true(self):
        payload = hub_jobs.pid_file_payload(os.getpid())
        assert hub_jobs.is_worker_alive(payload) is True

    def test_alive_check_for_dead_pid_is_false(self):
        # Use PID 1 (init) but tamper the start_time so the identity check
        # fails. init's process exists but its start_time differs from any
        # we'd record, simulating PID reuse.
        payload = {"pid": 1, "start_time": -1.0, "started_at": 0.0}
        # On Linux, init's start_time is well-defined and != -1.0; the
        # check should reject. On non-Linux, start_time is None so we
        # degrade to "process exists" — which also returns True for init.
        # We accept either outcome on non-Linux; on Linux we expect False.
        result = hub_jobs.is_worker_alive(payload)
        if hub_jobs._process_start_time(1) is not None:
            assert result is False, "Linux should reject mismatched start_time"
        # Otherwise the result is platform-dependent; just verify no crash.

    def test_alive_check_for_truly_dead_pid_is_false(self):
        # Spawn a short-lived child, capture its pid, wait for it to die.
        import subprocess

        proc = subprocess.Popen(["true"])
        proc.wait()
        payload = {"pid": proc.pid, "start_time": None, "started_at": time.time()}
        # The pid is now dead. is_worker_alive should return False.
        assert hub_jobs.is_worker_alive(payload) is False

    def test_alive_check_handles_missing_pid_key(self):
        assert hub_jobs.is_worker_alive({}) is False
        assert hub_jobs.is_worker_alive({"pid": "not an int"}) is False

    def test_read_pid_file_returns_none_for_missing_path(self, tmp_path):
        assert hub_jobs.read_pid_file(tmp_path / "nope.pid") is None

    def test_read_pid_file_returns_none_for_malformed_json(self, tmp_path):
        path = tmp_path / "bad.pid"
        path.write_text("not json {")
        assert hub_jobs.read_pid_file(path) is None

    def test_read_pid_file_roundtrips(self, tmp_path):
        path = tmp_path / "ok.pid"
        payload = hub_jobs.pid_file_payload(os.getpid())
        path.write_text(json.dumps(payload))
        assert hub_jobs.read_pid_file(path) == payload


# ── Error classification ────────────────────────────────────────────────────


class TestErrorClassification:
    def _hf_http_error(self, status_code: int, message: str):
        """Construct an HfHubHTTPError with the given status, version-agnostic.

        Newer huggingface_hub (1.x) requires a real httpx.Response. We build
        a minimal one — only the status_code attribute matters for our
        classification.
        """
        import httpx
        from huggingface_hub.errors import HfHubHTTPError

        response = httpx.Response(status_code, request=httpx.Request("GET", "http://x"))
        return HfHubHTTPError(message, response=response)

    def test_classifies_auth_via_status_code(self):
        assert hub_jobs.classify_error(self._hf_http_error(401, "unauthorized")) == "auth"
        assert hub_jobs.classify_error(self._hf_http_error(403, "forbidden")) == "auth"

    def test_classifies_rate_limit(self):
        assert hub_jobs.classify_error(self._hf_http_error(429, "too many requests")) == "rate_limit"

    def test_classifies_5xx_as_network(self):
        assert hub_jobs.classify_error(self._hf_http_error(503, "service unavailable")) == "network"
        assert hub_jobs.classify_error(self._hf_http_error(502, "bad gateway")) == "network"

    def test_classifies_repository_not_found_as_auth(self):
        """Private-repo-no-access surfaces as RepositoryNotFoundError; treat as auth.

        Construct one with the required response kwarg (newer HF) or fall
        back to setting response post-init (older HF).
        """
        import httpx
        from huggingface_hub.errors import RepositoryNotFoundError

        response = httpx.Response(404, request=httpx.Request("GET", "http://x"))
        try:
            rnf = RepositoryNotFoundError("private_inaccessible", response=response)
        except TypeError:
            rnf = RepositoryNotFoundError("private_inaccessible")  # type: ignore[call-arg]
            rnf.response = response
        assert hub_jobs.classify_error(rnf) == "auth"

    def test_classifies_connection_error_as_network(self):
        err = ConnectionError("name resolution failed")
        assert hub_jobs.classify_error(err) == "network"

    def test_classifies_timeout_error_as_network(self):
        err = TimeoutError("read timed out")
        assert hub_jobs.classify_error(err) == "network"

    def test_classifies_text_substring_fallback_auth(self):
        # Plain Exception with auth-related message → "auth"
        assert hub_jobs.classify_error(Exception("401 Unauthorized")) == "auth"
        assert hub_jobs.classify_error(Exception("403 Forbidden")) == "auth"

    def test_classifies_unknown_as_other(self):
        assert hub_jobs.classify_error(Exception("something else")) == "other"
        assert hub_jobs.classify_error(ValueError("bad input")) == "other"


# ── Milestone extraction ────────────────────────────────────────────────────
#
# Pinned samples of HF/tqdm stderr. If HF's format shifts in a future
# version, update the samples + fixtures here; the parser falls back to
# unmatched and the rest of the system stays functional.


class TestMilestoneExtraction:
    """The parser is best-effort — these tests pin the patterns we recognise.

    The graceful-degradation contract is more important than format
    coverage: if the parser returns None on a line, the system uses the
    fallback ``"running"`` milestone. So a regression here is a UX
    degradation, not a correctness bug.
    """

    def test_upload_processing_files(self):
        from lerobot.gui.hub_worker import extract_milestone

        result = extract_milestone("Processing Files (3 / 47)", "upload")
        assert result == "Processing files 3 / 47"

    def test_upload_committing(self):
        from lerobot.gui.hub_worker import extract_milestone

        assert extract_milestone("Committing files (5 / 12)", "upload") == "Committing 5 / 12"

    def test_download_fetching(self):
        from lerobot.gui.hub_worker import extract_milestone

        result = extract_milestone("Fetching 8 files: 100%|########| 5/8 [00:02<00:00]", "download")
        # Pattern matches "Fetching {N} files" and "{k}/{N}"
        assert result == "Downloading 5 / 8 files"

    def test_download_bare_percentage(self):
        from lerobot.gui.hub_worker import extract_milestone

        # Generic tqdm percentage matches the fallback pattern.
        assert extract_milestone("42%|####     |  4/10", "download") == "Downloading 42%"

    def test_unmatched_returns_none(self):
        from lerobot.gui.hub_worker import extract_milestone

        assert extract_milestone("Some unrelated log line", "upload") is None
        assert extract_milestone("", "upload") is None
        assert extract_milestone("Some unrelated log line", "download") is None


# ── enumerate_upload_files ──────────────────────────────────────────────────


class TestEnumerateUploadFiles:
    def test_returns_every_regular_file(self, tmp_path):
        (tmp_path / "meta").mkdir()
        (tmp_path / "meta" / "info.json").write_text("{}")
        (tmp_path / "data.parquet").write_bytes(b"x")
        files = hub_jobs.enumerate_upload_files(tmp_path)
        rels = sorted(p.relative_to(tmp_path).as_posix() for p in files)
        assert rels == ["data.parquet", "meta/info.json"]

    def test_skips_default_ignores(self, tmp_path):
        (tmp_path / "data.parquet").write_bytes(b"x")
        (tmp_path / ".lerobot_gui_edits.json").write_text("{}")
        (tmp_path / ".cache").mkdir()
        (tmp_path / ".cache" / "stuff").write_bytes(b"y")
        (tmp_path / ".DS_Store").write_bytes(b"z")
        files = hub_jobs.enumerate_upload_files(tmp_path)
        rels = sorted(p.relative_to(tmp_path).as_posix() for p in files)
        assert rels == ["data.parquet"]

    def test_returns_sorted_order(self, tmp_path):
        for name in ["z.bin", "a.bin", "m.bin"]:
            (tmp_path / name).write_bytes(b"")
        files = hub_jobs.enumerate_upload_files(tmp_path)
        rels = [p.relative_to(tmp_path).as_posix() for p in files]
        assert rels == ["a.bin", "m.bin", "z.bin"]

    def test_asserts_on_missing_root(self, tmp_path):
        with pytest.raises(AssertionError):
            hub_jobs.enumerate_upload_files(tmp_path / "nope")


# ── check_upload_completeness ──────────────────────────────────────────────
#
# Defends against the download-fail-then-upload corruption scenario.


class _FakeApi:
    """Minimal HfApi mock for the completeness check."""

    def __init__(self, siblings=None, raise_not_found=False):
        self._siblings = siblings or []
        self._raise = raise_not_found

    def repo_info(self, repo_id, repo_type="dataset", files_metadata=False):
        if self._raise:
            import httpx
            from huggingface_hub.errors import RepositoryNotFoundError

            response = httpx.Response(404, request=httpx.Request("GET", "http://x"))
            try:
                err: Exception = RepositoryNotFoundError(f"{repo_id} not found", response=response)
            except TypeError:
                err = RepositoryNotFoundError(f"{repo_id} not found")  # type: ignore[call-arg]
            raise err

        class _Sib:
            def __init__(self, rfilename):
                self.rfilename = rfilename

        class _Info:
            siblings = [_Sib(s) for s in self._siblings]

        return _Info()


class TestCheckUploadCompleteness:
    def test_fresh_repo_returns_empty(self, tmp_path):
        api = _FakeApi(raise_not_found=True)
        out = hub_jobs.check_upload_completeness(tmp_path, "user/new_repo", api=api)
        assert out == {"missing_locally": [], "incomplete_locally": []}

    def test_all_files_present_returns_empty(self, tmp_path):
        (tmp_path / "a.bin").write_bytes(b"x")
        (tmp_path / "meta").mkdir()
        (tmp_path / "meta" / "info.json").write_text("{}")
        api = _FakeApi(siblings=["a.bin", "meta/info.json"])
        out = hub_jobs.check_upload_completeness(tmp_path, "user/repo", api=api)
        assert out["missing_locally"] == []
        assert out["incomplete_locally"] == []

    def test_detects_missing_locally(self, tmp_path):
        (tmp_path / "a.bin").write_bytes(b"x")
        # remote has a.bin AND b.bin; b.bin is missing locally
        api = _FakeApi(siblings=["a.bin", "b.bin"])
        out = hub_jobs.check_upload_completeness(tmp_path, "user/repo", api=api)
        assert out["missing_locally"] == ["b.bin"]
        assert out["incomplete_locally"] == []

    def test_detects_incomplete_marker(self, tmp_path):
        (tmp_path / "a.bin").write_bytes(b"x")
        # Simulate a half-finished download: HF leaves <name>.incomplete.
        (tmp_path / "a.bin.incomplete").write_bytes(b"partial")
        api = _FakeApi(siblings=["a.bin"])
        out = hub_jobs.check_upload_completeness(tmp_path, "user/repo", api=api)
        assert out["incomplete_locally"] == ["a.bin"]

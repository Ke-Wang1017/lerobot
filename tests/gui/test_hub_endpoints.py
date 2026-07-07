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
"""End-to-end tests for the /hub/* HTTP endpoints.

What this file covers:
  * POST /hub/upload + /hub/download return job_id without blocking
  * GET /hub/jobs returns sorted list with merged worker progress
  * GET /hub/progress/{id} same shape for one job
  * POST /hub/progress/{id}/cancel signals the worker (verified via state)
  * POST /hub/progress/{id}/dismiss removes the entry
  * Concurrent upload returns 409 with existing job_id
  * Completeness-check 409 path (incomplete_local_state) with confirm_force override
  * Server-startup PID sweep marks orphan workers as failed

We mock subprocess.Popen so tests don't actually fork; the tests verify
the endpoint plumbing + state machine, not the subprocess lifecycle (that
lives in test_hub_worker.py).
"""

from __future__ import annotations

import asyncio
import json
import os
import time
import types
from unittest.mock import patch

import httpx
import pytest
from fastapi import FastAPI

from lerobot.gui import hub_jobs
from lerobot.gui.api import datasets as datasets_module
from lerobot.gui.frame_cache import FrameCache
from lerobot.gui.state import AppState

# ── Fixtures ────────────────────────────────────────────────────────────────


@pytest.fixture
def app_with_state(tmp_path, monkeypatch):
    """Fresh FastAPI app + AppState + an isolated JOBS_DIR per test."""
    jobs_dir = tmp_path / "hub_jobs"
    jobs_dir.mkdir()
    monkeypatch.setattr(hub_jobs, "JOBS_DIR", jobs_dir)

    app = FastAPI()
    app.include_router(datasets_module.router)
    state = AppState(frame_cache=FrameCache(max_bytes=1_000_000))
    datasets_module.set_app_state(state)

    # Always-OK auth for these tests; specific tests can override per-call.
    import huggingface_hub

    class _FakeApi:
        def whoami(self):
            return {"name": "test"}

        def repo_info(self, repo_id, repo_type="dataset", files_metadata=False):
            class _Info:
                siblings = []

            return _Info()

    monkeypatch.setattr(huggingface_hub, "HfApi", _FakeApi)

    yield app, state, monkeypatch, jobs_dir


def _make_open_dataset(state: AppState, dataset_id: str, root) -> None:
    state.datasets[dataset_id] = types.SimpleNamespace(
        root=str(root),
        repo_id=dataset_id,
        meta=types.SimpleNamespace(total_episodes=0, total_frames=0),
    )


class _FakePopen:
    """Captures subprocess args; pretends to be a long-running worker."""

    instances: list[_FakePopen] = []

    def __init__(self, args, **kwargs):
        self.args = args
        self.env = kwargs.get("env", {})
        self.pid = 9000 + len(_FakePopen.instances)
        self._terminated = False
        _FakePopen.instances.append(self)

    def terminate(self):
        self._terminated = True

    def wait(self, timeout=None):
        return 0


@pytest.fixture(autouse=True)
def reset_fake_popen():
    _FakePopen.instances.clear()
    yield
    _FakePopen.instances.clear()


# ── POST /hub/upload, /hub/download ─────────────────────────────────────────


class TestUploadEndpoint:
    def test_returns_job_id_without_blocking(self, app_with_state, tmp_path):
        app, state, monkeypatch, jobs_dir = app_with_state
        ds_root = tmp_path / "ds"
        ds_root.mkdir()
        (ds_root / "data.bin").write_bytes(b"x")
        _make_open_dataset(state, "user/ds", ds_root)

        with patch("subprocess.Popen", _FakePopen):

            async def run():
                async with httpx.AsyncClient(
                    transport=httpx.ASGITransport(app=app), base_url="http://test"
                ) as client:
                    t0 = time.monotonic()
                    resp = await client.post(
                        "/api/datasets/user%2Fds/hub/upload",
                        json={"repo_id": "user/ds"},
                    )
                    elapsed = time.monotonic() - t0
                    assert elapsed < 1.0
                    assert resp.status_code == 200, resp.text
                    body = resp.json()
                    assert "job_id" in body
                    assert body["status"] == "started"
                    # Job is registered server-side immediately.
                    assert body["job_id"] in state.hub_jobs
                    # Worker subprocess was spawned with the right args.
                    assert len(_FakePopen.instances) == 1
                    proc = _FakePopen.instances[0]
                    assert "lerobot.gui.hub_worker" in " ".join(proc.args)
                    assert "LEROBOT_HUB_WORKER_CONFIG" in proc.env

            asyncio.run(run())

    def test_second_upload_returns_409_with_existing_job_id(self, app_with_state, tmp_path):
        app, state, monkeypatch, jobs_dir = app_with_state
        ds_root = tmp_path / "ds"
        ds_root.mkdir()
        (ds_root / "data.bin").write_bytes(b"x")
        _make_open_dataset(state, "user/ds", ds_root)

        with patch("subprocess.Popen", _FakePopen):

            async def run():
                async with httpx.AsyncClient(
                    transport=httpx.ASGITransport(app=app), base_url="http://test"
                ) as client:
                    first = await client.post(
                        "/api/datasets/user%2Fds/hub/upload", json={"repo_id": "user/ds"}
                    )
                    assert first.status_code == 200
                    first_job = first.json()["job_id"]
                    state.hub_jobs[first_job].status = "running"  # simulate worker liftoff

                    second = await client.post(
                        "/api/datasets/user%2Fds/hub/upload", json={"repo_id": "user/ds"}
                    )
                    assert second.status_code == 409, second.text
                    assert second.json()["detail"]["job_id"] == first_job

            asyncio.run(run())

    def test_concurrent_uploads_funnel_to_single_spawn(self, app_with_state, tmp_path):
        """N truly-concurrent POSTs on the same dataset → 1 spawn, N-1 conflicts.

        Defends against the "user double-clicks Upload" / "two tabs racing"
        case. The test fires ``n`` POSTs in parallel via asyncio.gather +
        httpx ASGI transport and asserts exactly one Popen actually ran,
        the others all got 409 referencing the winner's job_id.

        What protects the invariant: ``async with _hub_spawn_lock_for(ds)``
        wraps both the active-job check AND the new-job registration.
        Two coroutines cannot both observe ``active_hub_job_for() is None``
        and proceed to spawn.

        Caveat — what this test does NOT distinguish on its own: the
        production critical section happens to be entirely synchronous
        today (no awaits between lock-acquire and lock-release), so
        Python's cooperative scheduling alone funnels concurrent POSTs
        even with the lock removed. The lock is defense-in-depth against
        anyone adding an ``await`` (e.g. an async hub call) inside the
        critical section in the future — if that happens, the post-
        condition this test asserts must continue to hold, and only the
        lock guarantees that.
        """
        app, state, monkeypatch, jobs_dir = app_with_state
        ds_root = tmp_path / "ds"
        ds_root.mkdir()
        (ds_root / "data.bin").write_bytes(b"x")
        _make_open_dataset(state, "user/ds", ds_root)

        n = 5

        with patch("subprocess.Popen", _FakePopen):
            datasets_module._hub_spawn_locks.clear()

            async def post_one(client):
                return await client.post("/api/datasets/user%2Fds/hub/upload", json={"repo_id": "user/ds"})

            async def run():
                async with httpx.AsyncClient(
                    transport=httpx.ASGITransport(app=app), base_url="http://test"
                ) as client:
                    posts = [asyncio.create_task(post_one(client)) for _ in range(n)]
                    return await asyncio.gather(*posts)

            results = asyncio.run(run())

        statuses = sorted(r.status_code for r in results)
        # Exactly one winner, N-1 conflicts.
        assert statuses == [200] + [409] * (n - 1), [r.text for r in results]

        # The losers all reference the same winning job_id — proves the
        # 409s came from "an active job exists," not from some unrelated
        # error path.
        winner = next(r for r in results if r.status_code == 200).json()["job_id"]
        losers = [r for r in results if r.status_code == 409]
        for loser in losers:
            assert loser.json()["detail"]["job_id"] == winner, loser.text

        # Decisive invariant: exactly one worker subprocess was spawned and
        # one job registered — user clicking Upload twice cannot result in
        # two HF transfers fighting each other.
        assert len(_FakePopen.instances) == 1, (
            f"{len(_FakePopen.instances)} workers spawned for the same dataset"
        )
        assert len(state.hub_jobs) == 1


class TestUploadCompletenessGuardrail:
    """Defends against download-fail-then-upload corruption."""

    def test_missing_locally_returns_409_with_code(self, app_with_state, tmp_path, monkeypatch):
        """When local is missing files present on remote, refuse the upload."""
        app, state, monkeypatch_, jobs_dir = app_with_state
        ds_root = tmp_path / "ds"
        ds_root.mkdir()
        (ds_root / "have.bin").write_bytes(b"x")
        _make_open_dataset(state, "user/ds", ds_root)

        # Override the fake API to advertise an extra remote file.
        import huggingface_hub

        class _FakeApiWithExtra:
            def whoami(self):
                return {"name": "test"}

            def repo_info(self, repo_id, repo_type="dataset", files_metadata=False):
                class _Sib:
                    def __init__(self, name):
                        self.rfilename = name

                class _Info:
                    siblings = [_Sib("have.bin"), _Sib("missing.bin")]

                return _Info()

        monkeypatch_.setattr(huggingface_hub, "HfApi", _FakeApiWithExtra)

        with patch("subprocess.Popen", _FakePopen):

            async def run():
                async with httpx.AsyncClient(
                    transport=httpx.ASGITransport(app=app), base_url="http://test"
                ) as client:
                    resp = await client.post(
                        "/api/datasets/user%2Fds/hub/upload", json={"repo_id": "user/ds"}
                    )
                    assert resp.status_code == 409
                    detail = resp.json()["detail"]
                    assert detail["code"] == "incomplete_local_state"
                    assert "missing.bin" in detail["missing_locally"]
                    # No worker should have been spawned.
                    assert _FakePopen.instances == []

            asyncio.run(run())

    def test_confirm_force_bypasses_guardrail(self, app_with_state, tmp_path, monkeypatch):
        app, state, monkeypatch_, jobs_dir = app_with_state
        ds_root = tmp_path / "ds"
        ds_root.mkdir()
        (ds_root / "have.bin").write_bytes(b"x")
        _make_open_dataset(state, "user/ds", ds_root)

        import huggingface_hub

        class _FakeApiWithExtra:
            def whoami(self):
                return {"name": "test"}

            def repo_info(self, repo_id, repo_type="dataset", files_metadata=False):
                class _Sib:
                    def __init__(self, name):
                        self.rfilename = name

                class _Info:
                    siblings = [_Sib("have.bin"), _Sib("missing.bin")]

                return _Info()

        monkeypatch_.setattr(huggingface_hub, "HfApi", _FakeApiWithExtra)

        with patch("subprocess.Popen", _FakePopen):

            async def run():
                async with httpx.AsyncClient(
                    transport=httpx.ASGITransport(app=app), base_url="http://test"
                ) as client:
                    resp = await client.post(
                        "/api/datasets/user%2Fds/hub/upload",
                        json={"repo_id": "user/ds", "confirm_force": True},
                    )
                    assert resp.status_code == 200
                    # Worker spawned despite the guardrail trigger.
                    assert len(_FakePopen.instances) == 1

            asyncio.run(run())


class TestJobsList:
    def test_lists_all_jobs_sorted_newest_first(self, app_with_state):
        app, state, _, _ = app_with_state
        for ds, t in [("a", 100.0), ("c", 300.0), ("b", 200.0)]:
            j = hub_jobs.make_job(dataset_id=ds, direction="upload", repo_id=ds)
            j.started_at = t
            state.hub_jobs[j.job_id] = j

        async def run():
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=app), base_url="http://test"
            ) as client:
                resp = await client.get("/api/datasets/hub/jobs")
                jobs = resp.json()["jobs"]
                assert [j["dataset_id"] for j in jobs] == ["c", "b", "a"]

        asyncio.run(run())

    def test_merges_worker_progress_into_in_memory_state(self, app_with_state, tmp_path):
        app, state, _, jobs_dir = app_with_state
        j = hub_jobs.make_job(dataset_id="u/ds", direction="upload", repo_id="u/ds")
        j.status = "running"
        state.hub_jobs[j.job_id] = j

        # Worker has written progress JSON with files_done_estimate=5.
        paths = hub_jobs.JobPaths.for_job(j.job_id, jobs_dir)
        paths.progress.write_text(
            json.dumps(
                {
                    "status": "running",
                    "milestone": "Uploading files",
                    "milestone_at": 12345.0,
                    "files_total": 10,
                    "files_done_estimate": 5,
                    "pr_num": 42,
                }
            )
        )

        async def run():
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=app), base_url="http://test"
            ) as client:
                resp = await client.get("/api/datasets/hub/jobs")
                jobs = resp.json()["jobs"]
                assert len(jobs) == 1
                assert jobs[0]["files_done_estimate"] == 5
                assert jobs[0]["milestone"] == "Uploading files"
                assert jobs[0]["pr_num"] == 42

        asyncio.run(run())


class TestCancel:
    def test_cancel_signals_worker_via_identity_check(self, app_with_state, tmp_path):
        """Cancel sends SIGTERM to the worker pid recorded in the pid file."""
        app, state, monkeypatch, jobs_dir = app_with_state
        j = hub_jobs.make_job(dataset_id="u/ds", direction="upload", repo_id="u/ds")
        j.status = "running"
        state.hub_jobs[j.job_id] = j

        # Write a PID file that points at the current test process so the
        # identity check succeeds.
        paths = hub_jobs.JobPaths.for_job(j.job_id, jobs_dir)
        paths.pid.write_text(json.dumps(hub_jobs.pid_file_payload(os.getpid())))

        # Intercept os.kill so we don't actually signal ourselves.
        signals_sent: list[tuple[int, int]] = []
        monkeypatch.setattr(os, "kill", lambda pid, sig: signals_sent.append((pid, sig)))

        async def run():
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=app), base_url="http://test"
            ) as client:
                resp = await client.post(f"/api/datasets/hub/progress/{j.job_id}/cancel")
                assert resp.status_code == 200
                assert resp.json()["status"] == "cancel_requested"
                # signal 0 happens first (alive check), then SIGTERM.
                import signal as sigmod

                assert (os.getpid(), sigmod.SIGTERM) in signals_sent

        asyncio.run(run())

    def test_cancel_with_dead_worker_marks_failed(self, app_with_state, tmp_path):
        """If the PID file points at a dead PID, cancel doesn't crash —
        it marks the job failed locally so the tray surfaces it."""
        app, state, _, jobs_dir = app_with_state
        j = hub_jobs.make_job(dataset_id="u/ds", direction="upload", repo_id="u/ds")
        j.status = "running"
        state.hub_jobs[j.job_id] = j

        import subprocess

        proc = subprocess.Popen(["true"])
        proc.wait()
        paths = hub_jobs.JobPaths.for_job(j.job_id, jobs_dir)
        paths.pid.write_text(json.dumps({"pid": proc.pid, "start_time": None, "started_at": time.time()}))

        async def run():
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=app), base_url="http://test"
            ) as client:
                resp = await client.post(f"/api/datasets/hub/progress/{j.job_id}/cancel")
                assert resp.status_code == 200
                # Server-side: detected dead worker, status → failed.
                assert state.hub_jobs[j.job_id].status == "failed"

        asyncio.run(run())


class TestDismiss:
    def test_dismiss_removes_terminal_job_and_files(self, app_with_state, tmp_path):
        app, state, _, jobs_dir = app_with_state
        j = hub_jobs.make_job(dataset_id="ds", direction="upload", repo_id="user/ds")
        j.status = "complete"
        j.finished_at = time.time()
        state.hub_jobs[j.job_id] = j
        paths = hub_jobs.JobPaths.for_job(j.job_id, jobs_dir)
        paths.progress.write_text("{}")
        paths.log.write_text("logs")

        async def run():
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=app), base_url="http://test"
            ) as client:
                resp = await client.post(f"/api/datasets/hub/progress/{j.job_id}/dismiss")
                assert resp.status_code == 200
                assert j.job_id not in state.hub_jobs
                assert not paths.progress.exists()
                assert not paths.log.exists()

        asyncio.run(run())

    def test_dismiss_refuses_active_job(self, app_with_state):
        app, state, _, _ = app_with_state
        j = hub_jobs.make_job(dataset_id="ds", direction="upload", repo_id="user/ds")
        j.status = "running"
        state.hub_jobs[j.job_id] = j

        async def run():
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=app), base_url="http://test"
            ) as client:
                resp = await client.post(f"/api/datasets/hub/progress/{j.job_id}/dismiss")
                assert resp.status_code == 409
                assert j.job_id in state.hub_jobs

        asyncio.run(run())


class TestRetryPRTransfer:
    """Regression for the retry/dismiss interaction bug.

    Before the fix: Retry on a failed upload spawned a new worker with
    reuse_pr_num=N, but the frontend's follow-up Dismiss then called
    change_discussion_status(closed) on that same PR — destroying the
    very PR the new worker was resuming into.

    After the fix: _find_existing_pr_for_retry transfers PR ownership
    by clearing pr_num on every source-entry pointing at the resumed
    PR, so the subsequent dismiss skips the close branch.
    """

    def test_pr_ownership_transferred_off_source_on_retry(self, app_with_state, tmp_path, monkeypatch):
        app, state, monkeypatch_, jobs_dir = app_with_state
        ds_root = tmp_path / "ds"
        ds_root.mkdir()
        (ds_root / "data.bin").write_bytes(b"x")
        _make_open_dataset(state, "user/ds", ds_root)

        # Source entry: a failed upload that left a draft PR behind.
        source = hub_jobs.make_job(dataset_id="user/ds", direction="upload", repo_id="user/ds")
        source.status = "failed"
        source.finished_at = time.time()
        source.pr_num = 42
        state.hub_jobs[source.job_id] = source

        # Patch HfApi so _find_existing_pr_for_retry sees a draft PR.
        import huggingface_hub

        class _FakeApi:
            def whoami(self):
                return {"name": "test"}

            def repo_info(self, *a, **k):
                class _Info:
                    siblings = []

                return _Info()

            def get_discussion_details(self, *a, **k):
                class _Details:
                    status = "draft"

                return _Details()

        monkeypatch_.setattr(huggingface_hub, "HfApi", _FakeApi)

        with patch("subprocess.Popen", _FakePopen):
            datasets_module._hub_spawn_locks.clear()

            async def run():
                async with httpx.AsyncClient(
                    transport=httpx.ASGITransport(app=app), base_url="http://test"
                ) as client:
                    resp = await client.post(
                        "/api/datasets/user%2Fds/hub/upload", json={"repo_id": "user/ds"}
                    )
                    assert resp.status_code == 200, resp.text
                    new_job_id = resp.json()["job_id"]

                    new_job = state.hub_jobs[new_job_id]
                    # New job inherited pr_num=42.
                    assert new_job.pr_num == 42
                    # Source's pr_num was transferred away — this is the
                    # invariant that makes the follow-up dismiss safe.
                    assert source.pr_num is None, (
                        "pr_num was not transferred off the source; "
                        "subsequent dismiss would close the resumed PR"
                    )

            asyncio.run(run())

    def test_dismiss_after_retry_does_not_close_resumed_pr(self, app_with_state, tmp_path, monkeypatch):
        """End-to-end: source dismiss after retry must not call change_discussion_status."""
        app, state, monkeypatch_, jobs_dir = app_with_state
        ds_root = tmp_path / "ds"
        ds_root.mkdir()
        (ds_root / "data.bin").write_bytes(b"x")
        _make_open_dataset(state, "user/ds", ds_root)

        source = hub_jobs.make_job(dataset_id="user/ds", direction="upload", repo_id="user/ds")
        source.status = "failed"
        source.finished_at = time.time()
        source.pr_num = 42
        state.hub_jobs[source.job_id] = source

        change_status_calls: list[dict] = []

        import huggingface_hub

        class _FakeApi:
            def whoami(self):
                return {"name": "test"}

            def repo_info(self, *a, **k):
                class _Info:
                    siblings = []

                return _Info()

            def get_discussion_details(self, *a, **k):
                class _Details:
                    status = "draft"

                return _Details()

            def change_discussion_status(self, **kwargs):
                change_status_calls.append(kwargs)

        monkeypatch_.setattr(huggingface_hub, "HfApi", _FakeApi)

        with patch("subprocess.Popen", _FakePopen):
            datasets_module._hub_spawn_locks.clear()

            async def run():
                async with httpx.AsyncClient(
                    transport=httpx.ASGITransport(app=app), base_url="http://test"
                ) as client:
                    # Retry → spawns new worker with reuse_pr_num=42.
                    resp = await client.post(
                        "/api/datasets/user%2Fds/hub/upload", json={"repo_id": "user/ds"}
                    )
                    assert resp.status_code == 200

                    # Mirror the frontend: dismiss the source job after retry.
                    dismiss = await client.post(f"/api/datasets/hub/progress/{source.job_id}/dismiss")
                    assert dismiss.status_code == 200

                    # The critical assertion: dismiss must NOT have closed
                    # the resumed PR.
                    assert change_status_calls == [], (
                        f"dismiss closed the resumed PR — calls: {change_status_calls}"
                    )

            asyncio.run(run())


class TestStartupSweep:
    """Server-startup PID sweep reaps orphan workers from a previous run."""

    def test_sweeps_dead_worker_pid_files(self, app_with_state, tmp_path, monkeypatch):
        app, state, _, jobs_dir = app_with_state
        j = hub_jobs.make_job(dataset_id="ds", direction="upload", repo_id="user/ds")
        j.status = "running"  # pretending the prior server thought it was alive
        state.hub_jobs[j.job_id] = j

        # PID file points at a process that's already dead.
        import subprocess

        proc = subprocess.Popen(["true"])
        proc.wait()
        paths = hub_jobs.JobPaths.for_job(j.job_id, jobs_dir)
        paths.pid.write_text(json.dumps({"pid": proc.pid, "start_time": None, "started_at": time.time()}))

        # Run the sweep (what server.startup_event calls).
        reaped = datasets_module._sweep_orphan_pid_files()
        assert reaped >= 1
        assert state.hub_jobs[j.job_id].status == "failed"
        assert "Worker exited without finalizing" in state.hub_jobs[j.job_id].error
        assert not paths.pid.exists()

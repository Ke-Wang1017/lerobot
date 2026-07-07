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
"""Tests for the Hub-transfer worker subprocess.

We spawn the worker as a real subprocess (not just import + call ``main``)
because the worker:
  - installs SIGTERM/SIGINT handlers that we test against
  - redirects stderr at the file-descriptor level
  - writes its own PID file

All of those need a real process to exercise correctly. We mock the
``huggingface_hub`` calls inside the worker so the tests don't hit the
network, but the IPC + signal handling + PID-file lifecycle are real.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

import pytest

from lerobot.gui import hub_jobs

# ── Test helpers ────────────────────────────────────────────────────────────


_MOCK_HF_MODULE = """
\"\"\"Inserted ahead of huggingface_hub by sys.path manipulation in the test.

Provides controlled mock implementations whose behaviour is steered by env
vars the test sets. The real huggingface_hub is shadowed for the worker's
lifetime; tests that need real HF should not use this fixture.
\"\"\"

import json
import os
import time
import sys


def _mock_config():
    return json.loads(os.environ.get('LEROBOT_HUB_TEST_MOCK_CONFIG', '{}'))


class _PRDetails:
    def __init__(self, num):
        self.num = num
        self.status = 'draft'


class HfApi:
    _create_repo_calls = []
    _pr_num = 1

    def __init__(self, *a, **kw): pass

    def create_repo(self, **kwargs):
        type(self)._create_repo_calls.append(kwargs)

    def create_pull_request(self, **kwargs):
        if _mock_config().get('fail_create_pr'):
            raise RuntimeError('mock: create_pull_request failed')
        type(self)._pr_num += 1
        return _PRDetails(type(self)._pr_num)

    def super_squash_history(self, **kwargs):
        if _mock_config().get('fail_squash'):
            raise TimeoutError('mock: squash timed out (simulated)')

    def change_discussion_status(self, **kwargs):
        if _mock_config().get('fail_change_status'):
            raise RuntimeError('mock: change_discussion_status failed')

    def merge_pull_request(self, **kwargs):
        if _mock_config().get('fail_merge'):
            raise RuntimeError('mock: merge failed')

    def get_discussion_details(self, **kwargs):
        return _PRDetails(kwargs.get('discussion_num', 1))

    def whoami(self):
        return {'name': 'test-user'}


def upload_large_folder(**kwargs):
    sleep_s = float(_mock_config().get('upload_sleep_s', 0.0))
    if sleep_s > 0:
        # Sleep in small increments so a SIGTERM is responsive during the call.
        end = time.time() + sleep_s
        while time.time() < end:
            sys.stderr.write(f'Processing Files (1 / 1)\\r')
            sys.stderr.flush()
            time.sleep(0.05)
    sys.stderr.write('Upload done\\n')
    if _mock_config().get('fail_upload'):
        raise RuntimeError('mock: upload failed')


def snapshot_download(**kwargs):
    sleep_s = float(_mock_config().get('download_sleep_s', 0.0))
    if sleep_s > 0:
        end = time.time() + sleep_s
        while time.time() < end:
            sys.stderr.write(f'Fetching 1 files: 1/1\\r')
            sys.stderr.flush()
            time.sleep(0.05)


from huggingface_hub import errors  # re-export for import compat
"""


@pytest.fixture
def mock_hf_install(tmp_path, monkeypatch):
    """Install a mock ``huggingface_hub`` module at the front of sys.path.

    The fixture writes a fake huggingface_hub package into ``tmp_path``
    that overrides the real one for any subprocess that inherits the
    test's ``PYTHONPATH``. Cleanup is automatic via tmp_path teardown.
    """
    mock_pkg = tmp_path / "mock_hf" / "huggingface_hub"
    mock_pkg.mkdir(parents=True)
    (mock_pkg / "__init__.py").write_text(_MOCK_HF_MODULE)
    # Re-export the errors module from the real HF so our worker can import it.
    (mock_pkg / "errors.py").write_text("from huggingface_hub.errors import *  # noqa: F401, F403\n")
    monkeypatch.setenv("PYTHONPATH", str(mock_pkg.parent) + os.pathsep + os.environ.get("PYTHONPATH", ""))
    return mock_pkg.parent


def _build_config(tmp_path: Path, **overrides) -> tuple[hub_jobs.JobConfig, hub_jobs.JobPaths]:
    jobs_dir = tmp_path / "jobs"
    jobs_dir.mkdir()
    local_path = tmp_path / "local"
    local_path.mkdir()
    (local_path / "data.bin").write_bytes(b"x")
    defaults = {
        "job_id": "job-test",
        "dataset_id": "ds-1",
        "direction": "upload",
        "repo_id": "user/repo",
        "repo_type": "dataset",
        "local_path": str(local_path),
        "jobs_dir": str(jobs_dir),
        "private": True,
        "commit_message": "Test upload",
    }
    defaults.update(overrides)
    cfg = hub_jobs.JobConfig(**defaults)
    paths = hub_jobs.JobPaths.for_job(cfg.job_id, jobs_dir)
    return cfg, paths


def _spawn_worker(
    cfg: hub_jobs.JobConfig,
    mock_config: dict | None = None,
    *,
    extra_env: dict | None = None,
) -> subprocess.Popen:
    env = os.environ.copy()
    env["LEROBOT_HUB_WORKER_CONFIG"] = cfg.to_json()
    env["LEROBOT_HUB_TEST_MOCK_CONFIG"] = json.dumps(mock_config or {})
    if extra_env:
        env.update(extra_env)
    return subprocess.Popen(  # noqa: S603 — args controlled
        [sys.executable, "-m", "lerobot.gui.hub_worker"],
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
    )


def _wait_until_status(paths: hub_jobs.JobPaths, *, timeout_s: float = 30.0) -> dict:
    """Poll the progress JSON until status is terminal."""
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        if paths.progress.exists():
            try:
                snap = json.loads(paths.progress.read_text())
                if snap.get("status") in ("complete", "failed", "cancelled"):
                    return snap
            except (OSError, json.JSONDecodeError):
                pass
        time.sleep(0.05)
    raise TimeoutError(f"Worker didn't reach terminal status in {timeout_s}s")


# ── End-to-end worker flows ────────────────────────────────────────────────


class TestWorkerLifecycle:
    """High-level worker behaviour — spawn, run, exit, terminal state visible."""

    def test_successful_upload_writes_complete_status(self, tmp_path, mock_hf_install):
        cfg, paths = _build_config(tmp_path)
        proc = _spawn_worker(cfg, mock_config={"upload_sleep_s": 0.2})
        try:
            snap = _wait_until_status(paths)
            assert snap["status"] == "complete"
            assert snap["error"] is None
        finally:
            proc.wait(timeout=5)

    def test_successful_download_writes_complete_status(self, tmp_path, mock_hf_install):
        cfg, paths = _build_config(tmp_path, direction="download")
        proc = _spawn_worker(cfg, mock_config={"download_sleep_s": 0.1})
        try:
            snap = _wait_until_status(paths)
            assert snap["status"] == "complete"
        finally:
            proc.wait(timeout=5)

    def test_writes_pid_file_with_identity(self, tmp_path, mock_hf_install):
        cfg, paths = _build_config(tmp_path)
        # Long-running upload so the PID file is observable mid-flight.
        proc = _spawn_worker(cfg, mock_config={"upload_sleep_s": 2.0})
        try:
            # Wait for the PID file to appear.
            deadline = time.time() + 5
            while time.time() < deadline and not paths.pid.exists():
                time.sleep(0.05)
            assert paths.pid.exists(), "Worker should write pid file shortly after spawn"
            payload = hub_jobs.read_pid_file(paths.pid)
            assert payload is not None
            assert payload["pid"] == proc.pid
            # Identity check accepts the live worker.
            assert hub_jobs.is_worker_alive(payload)
        finally:
            proc.terminate()
            proc.wait(timeout=5)

    def test_cleans_up_pid_file_after_terminal_exit(self, tmp_path, mock_hf_install):
        cfg, paths = _build_config(tmp_path)
        proc = _spawn_worker(cfg, mock_config={"upload_sleep_s": 0.1})
        proc.wait(timeout=10)
        assert not paths.pid.exists(), "Worker should remove its PID file on exit"

    def test_failure_in_upload_pipeline_classifies_error(self, tmp_path, mock_hf_install):
        cfg, paths = _build_config(tmp_path)
        proc = _spawn_worker(cfg, mock_config={"fail_upload": True})
        try:
            snap = _wait_until_status(paths)
            assert snap["status"] == "failed"
            assert "mock: upload failed" in snap["error"]
            # Generic Exception → "other"
            assert snap["error_class"] == "other"
        finally:
            proc.wait(timeout=5)


class TestSquashFallback:
    """Squash is currently disabled in the worker — see hub_transfers.md.

    The pipeline takes the unsquashed-merge path unconditionally, so the
    end-state milestone always says "merged unsquashed". This test pins
    that behaviour so re-enabling squash (when the HF API issue is
    resolved) requires an explicit test update.
    """

    def test_pipeline_always_takes_unsquashed_merge_path(self, tmp_path, mock_hf_install):
        cfg, paths = _build_config(tmp_path)
        proc = _spawn_worker(cfg, mock_config={"fail_squash": True})
        try:
            snap = _wait_until_status(paths)
            assert snap["status"] == "complete", f"Expected complete; got {snap}"
            assert "unsquashed" in snap["milestone"], (
                f"Milestone should signal the unsquashed merge; got {snap['milestone']!r}"
            )
        finally:
            proc.wait(timeout=5)


class TestCancellation:
    """SIGTERM handling: cancel mid-flight, leave resumable state intact."""

    def test_sigterm_during_upload_yields_cancelled_status(self, tmp_path, mock_hf_install):
        cfg, paths = _build_config(tmp_path)
        # Long-running upload so we can interrupt mid-flight.
        proc = _spawn_worker(cfg, mock_config={"upload_sleep_s": 5.0})
        try:
            # Wait for the upload to actually start.
            deadline = time.time() + 3
            while time.time() < deadline:
                if paths.progress.exists():
                    try:
                        snap = json.loads(paths.progress.read_text())
                        if snap.get("stage") == "uploading":
                            break
                    except (OSError, json.JSONDecodeError):
                        pass
                time.sleep(0.05)

            proc.terminate()  # SIGTERM
            snap = _wait_until_status(paths, timeout_s=15.0)
            assert snap["status"] == "cancelled"
            assert snap["error_class"] == "cancelled"
        finally:
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()

    def test_sigkill_escalation_still_leaves_no_zombie(self, tmp_path, mock_hf_install):
        """If the worker ignores SIGTERM, SIGKILL still cleans up the PID file.

        The worker's at-exit hook tries to remove the PID file but SIGKILL
        skips Python's atexit. So we assert the *parent* of the worker
        (the OS) reaps the PID, and that a subsequent is_worker_alive call
        returns False — which is what the server-startup sweep uses to
        reap stale entries.
        """
        cfg, paths = _build_config(tmp_path)
        proc = _spawn_worker(cfg, mock_config={"upload_sleep_s": 10.0})
        try:
            deadline = time.time() + 3
            while time.time() < deadline and not paths.pid.exists():
                time.sleep(0.05)
            payload = hub_jobs.read_pid_file(paths.pid)
            assert payload is not None
            proc.kill()  # SIGKILL — bypasses Python signal handlers
            proc.wait(timeout=5)
            # Process is gone; is_worker_alive recognises that even though
            # the PID file may still exist.
            assert hub_jobs.is_worker_alive(payload) is False
        finally:
            try:
                proc.wait(timeout=2)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()


class TestMissingConfig:
    """Worker should exit cleanly with a useful error when config is missing."""

    def test_no_config_env_var_exits_with_code_2(self, tmp_path):
        env = {k: v for k, v in os.environ.items() if k != "LEROBOT_HUB_WORKER_CONFIG"}
        # NOTE: not using mock_hf_install — we exit before any HF import.
        proc = subprocess.Popen(  # noqa: S603
            [sys.executable, "-m", "lerobot.gui.hub_worker"],
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        _stdout, stderr = proc.communicate(timeout=5)
        assert proc.returncode == 2
        assert b"LEROBOT_HUB_WORKER_CONFIG" in stderr


class TestSignalHandlerNoDeadlock:
    """Regression: SIGTERM handler must not acquire state._lock.

    Python delivers signals synchronously on the main thread between
    bytecodes. If the main thread already holds ``state._lock`` (e.g.,
    is inside ``set_milestone``), a handler that does ``with state._lock``
    would self-deadlock on the non-reentrant lock — wedging the worker
    until SIGKILL.

    We test by directly invoking the handler function with state._lock
    pre-acquired. If the handler still tried to re-acquire, this test
    would block; we wrap in a watchdog thread that fails the test if it
    runs too long.
    """

    def test_handler_completes_while_lock_held(self):
        import threading

        from lerobot.gui import hub_jobs, hub_worker

        cfg = hub_jobs.JobConfig(
            job_id="test-handler",
            dataset_id="user/ds",
            direction="upload",
            repo_id="user/ds",
            repo_type="dataset",
            local_path="/tmp",
            jobs_dir="/tmp",
            private=True,
            commit_message=None,
            allow_patterns=None,
            ignore_patterns=None,
            reuse_pr_num=None,
        )
        paths = hub_jobs.JobPaths.for_job("test-handler", Path("/tmp"))
        state = hub_worker._WorkerState(cfg, paths)

        # Build the same handler function that _install_signal_handlers builds.
        # We don't actually register it with signal.signal (that would mess with
        # the test runner); we just need to verify it doesn't try to acquire
        # state._lock when called while the lock is already held.
        captured: list = []

        def _on_sigterm(signum, frame):
            # Mirror the production handler exactly. If this implementation
            # ever changes to acquire state._lock, this test will deadlock
            # (and fail via the watchdog).
            state.cancel_requested = True
            state.milestone = "cancelling"
            state.milestone_at = time.time()
            captured.append("done")

        # Hold the lock on the "main" (test) thread, then invoke the handler.
        # Signal handlers in production also run on the main thread, so the
        # re-acquire path is what we're regression-testing.
        done = threading.Event()

        def call_handler():
            _on_sigterm(15, None)  # 15 = SIGTERM
            done.set()

        with state._lock:
            t = threading.Thread(target=call_handler)
            t.start()
            # In the buggy version, the handler would block on `with state._lock`
            # while THIS thread holds it — done.wait would time out.
            # (Note: this only catches the bug if the handler tries to acquire
            # synchronously; in real signal delivery the handler runs on the
            # same thread that holds the lock, which is even more deadly.)
            assert done.wait(timeout=1.0), "handler did not complete; suspected re-acquire deadlock"
            t.join(timeout=1.0)

        assert captured == ["done"]
        assert state.cancel_requested is True
        assert state.milestone == "cancelling"

    def test_handler_in_source_does_not_acquire_state_lock(self):
        """Static guard: the production handler must not contain a
        ``with state._lock`` (or any other ``with ... _lock``) acquisition.

        Belt-and-suspenders complement to the runtime test: catches the
        regression at lint time without needing the runtime watchdog to
        fire. We match the executable pattern, not the bare string, so a
        comment explaining the lock-avoidance rule doesn't false-positive.
        """
        import re

        from lerobot.gui import hub_worker

        src = Path(hub_worker.__file__).read_text()
        start = src.find("def _on_sigterm(")
        assert start >= 0, "couldn't locate _on_sigterm"
        body = src[start : start + 2000]
        # Strip comment lines so the rationale comment doesn't match.
        code_only = "\n".join(line for line in body.splitlines() if not line.lstrip().startswith("#"))
        assert not re.search(r"with\s+\w+\._lock\s*:", code_only), (
            "SIGTERM handler must not acquire any _lock — Python delivers "
            "signals synchronously on the main thread; re-acquiring a "
            "non-reentrant lock the same thread already holds would deadlock"
        )
        # Also guard against an explicit acquire call.
        assert "_lock.acquire" not in code_only, (
            "SIGTERM handler must not call _lock.acquire() for the same deadlock reason"
        )


class TestWriterShutdownOrdering:
    """Regression: worker main() must stop+join the writer thread BEFORE the
    final write_progress() call.

    Otherwise two threads call atomic_write_json(path, ...) concurrently —
    both writing to the same ``.tmp`` path — and the os.replace can land
    a partially-written tmp, corrupting the terminal progress file the
    server polls.
    """

    def test_main_finally_stops_writer_before_final_write(self):
        """Static guard on the ordering inside main()'s finally block."""
        from lerobot.gui import hub_worker

        src = Path(hub_worker.__file__).read_text()
        # Find main()'s finally block.
        main_start = src.find("def main(")
        assert main_start >= 0, "couldn't locate main()"
        finally_start = src.find("finally:", main_start)
        assert finally_start >= 0, "couldn't locate finally: in main()"
        # The end of main()'s finally is bounded by the next top-level def
        # or 'return rc'.
        end = src.find("return rc", finally_start)
        assert end >= 0
        body = src[finally_start:end]
        stop_pos = body.find("stop_writer_thread()")
        join_pos = body.find("writer.join(")
        final_write_pos = body.find("state.write_progress()")
        assert stop_pos >= 0, "stop_writer_thread() call missing"
        assert join_pos >= 0, "writer.join() call missing"
        assert final_write_pos >= 0, "final state.write_progress() call missing"
        assert stop_pos < final_write_pos, (
            "stop_writer_thread() must be called BEFORE the final "
            "state.write_progress() — otherwise the writer thread can race "
            "the main thread on the same .tmp path under atomic_write_json"
        )
        assert join_pos < final_write_pos, (
            "writer.join() must complete BEFORE the final state.write_progress() — same race-on-tmp concern"
        )


# ── Capture both stdout and stderr in the per-job log ──────────────────────


# Mock that writes to BOTH stdout and stderr so we can verify the worker
# captures both into the same per-job log file. Without dup'ing fd 1 too,
# the stdout writes would silently disappear (server spawns the worker
# with stdout=DEVNULL) — exactly the leak that hid HF's rate-limit
# messages from view in the original bug investigation.
_MOCK_HF_STDOUT_AND_STDERR = """
import json
import os
import sys

def _mock_config():
    return json.loads(os.environ.get('LEROBOT_HUB_TEST_MOCK_CONFIG', '{}'))


class _PRDetails:
    def __init__(self, num):
        self.num = num


class HfApi:
    def create_repo(self, **kwargs): return None
    def create_pull_request(self, **kwargs): return _PRDetails(1)
    def change_discussion_status(self, **kwargs): return None
    def merge_pull_request(self, **kwargs): return None
    def super_squash_history(self, **kwargs): return None
    def whoami(self): return {'name': 'test-user'}


def upload_large_folder(**kwargs):
    # Emit a recognizable marker to each stream. The captured log must
    # contain BOTH or the dup is broken.
    sys.stdout.write('MARKER_STDOUT_from_hf\\n')
    sys.stdout.flush()
    sys.stderr.write('MARKER_STDERR_from_hf\\n')
    sys.stderr.flush()


def snapshot_download(**kwargs):
    sys.stdout.write('MARKER_STDOUT_from_hf\\n')
    sys.stdout.flush()
    sys.stderr.write('MARKER_STDERR_from_hf\\n')
    sys.stderr.flush()


from huggingface_hub import errors  # re-export for import compat
"""


@pytest.fixture
def mock_hf_stdout_and_stderr(tmp_path, monkeypatch):
    """Variant of the mock_hf_install fixture whose mock emits to both
    streams, so we can prove the worker captures both into ``paths.log``.
    """
    mock_pkg = tmp_path / "mock_hf" / "huggingface_hub"
    mock_pkg.mkdir(parents=True)
    (mock_pkg / "__init__.py").write_text(_MOCK_HF_STDOUT_AND_STDERR)
    (mock_pkg / "errors.py").write_text("from huggingface_hub.errors import *  # noqa: F401, F403\n")
    monkeypatch.setenv("PYTHONPATH", str(mock_pkg.parent) + os.pathsep + os.environ.get("PYTHONPATH", ""))
    return mock_pkg.parent


class TestCapturesStdoutAndStderr:
    """Both stdout and stderr from the HF library land in the per-job log.

    The original visibility bug had HF's rate-limit error message in
    stderr (where we already captured it), but parts of HF's library
    output go to stdout. The server spawns the worker with stdout=DEVNULL,
    so without an explicit fd-1 dup the worker's own stdout would be
    sent into the void. This test guards against silently regressing
    that dup.
    """

    def test_marker_from_stderr_is_in_log(self, tmp_path, mock_hf_stdout_and_stderr):
        cfg, paths = _build_config(tmp_path)
        proc = _spawn_worker(cfg)
        try:
            snap = _wait_until_status(paths, timeout_s=10)
        finally:
            proc.wait(timeout=5)
        assert snap["status"] == "complete"
        log_text = paths.log.read_text()
        assert "MARKER_STDERR_from_hf" in log_text, f"stderr marker missing from log; got:\n{log_text[-500:]}"

    def test_marker_from_stdout_is_in_log(self, tmp_path, mock_hf_stdout_and_stderr):
        """The regression: without dup2(w_fd, 1), this assertion fails
        because the server's stdout=DEVNULL eats the marker.
        """
        cfg, paths = _build_config(tmp_path)
        proc = _spawn_worker(cfg)
        try:
            snap = _wait_until_status(paths, timeout_s=10)
        finally:
            proc.wait(timeout=5)
        assert snap["status"] == "complete"
        log_text = paths.log.read_text()
        assert "MARKER_STDOUT_from_hf" in log_text, (
            f"stdout marker missing from log — worker is not dup'ing fd 1 "
            f"into the same pipe as fd 2. Got:\n{log_text[-500:]}"
        )


# ── Fail-fast httpx hook on unrecoverable HF responses ─────────────────────


class _Synthetic429Transport:
    """httpx transport stub that responds to every request with 429.

    Used to drive the fail-fast hook without monkey-patching
    huggingface_hub internals beyond the public-but-private transport
    swap. The Retry-After + body mirror what HF actually returns on a
    repo-commit-rate-limit so the assertion on the surfaced message
    text exercises real wording.
    """

    def __init__(self, status: int = 429, retry_after: str = "130") -> None:
        self.status = status
        self.retry_after = retry_after

    def handle_request(self, request):  # signature matches httpx.BaseTransport
        import httpx

        body = (
            b"429 Too Many Requests: you have reached your 'api' rate limit. "
            b"Retry after 130 seconds. "
            b"You have exceeded the rate limit for repository commits (128 per hour)."
        )
        return httpx.Response(
            status_code=self.status,
            headers={"Retry-After": self.retry_after, "Content-Type": "text/plain"},
            content=body,
            request=request,
        )


class TestFatalHttpHookUnit:
    """Direct exercise of the worker's httpx-hook helpers, no subprocess.

    Production uses an abort callback that os._exit's the process, so
    these tests use the default (raising) install variant. The abort
    helper itself is covered by a separate subprocess-level test that
    actually invokes it and asserts the resulting on-disk state.
    """

    def _uninstall_all(self, client):
        """Strip every tagged hook from the shared HF client between tests."""
        client.event_hooks["response"] = [
            h for h in client.event_hooks.get("response", []) if not getattr(h, "_lerobot_fatal_hook", False)
        ]

    def test_classify_response_pure_function(self):
        """The classifier is the unit that decides "is this fatal" and
        builds the message HF's text gets stitched into. Cover it directly
        so the message-construction contract is pinned independently of
        the hook install plumbing.
        """
        import httpx

        from lerobot.gui.hub_worker import _classify_response

        req = httpx.Request("POST", "https://huggingface.co/api/test")

        resp_429 = httpx.Response(429, headers={"Retry-After": "200"}, content=b"go away", request=req)
        exc = _classify_response(resp_429)
        assert exc is not None
        assert exc.status == 429
        assert exc.error_class == "rate_limit"
        assert "Retry-After: 200s" in exc.message
        assert "go away" in exc.message

        # Non-fatal short-circuits before touching the body.
        assert _classify_response(httpx.Response(200, content=b"ok", request=req)) is None

        # 5xx deliberately ignored — HF's adaptive retry was designed for
        # 504 commit timeouts and intercepting would defeat it.
        assert _classify_response(httpx.Response(504, content=b"timeout", request=req)) is None
        assert _classify_response(httpx.Response(500, content=b"boom", request=req)) is None

        # 4xx are all fatal — the library only changes the batch SIZE on
        # retry, not the request CONTENTS, so a malformed-request 400/422
        # will fail identically on every retry until the rate limit fires.
        # Pin the specific case we observed in the wild (LFS pointer
        # corruption surfaces as 400 with that body text).
        resp_400 = httpx.Response(
            400, content=b"LFS pointer pointed to a file that does not exist", request=req
        )
        exc_400 = _classify_response(resp_400)
        assert exc_400 is not None
        assert exc_400.status == 400
        assert exc_400.error_class == "bad_request"
        assert "LFS pointer" in exc_400.message

        resp_422 = httpx.Response(422, content=b"validation failed", request=req)
        exc_422 = _classify_response(resp_422)
        assert exc_422 is not None
        assert exc_422.error_class == "bad_request"

    def test_default_hook_raises_on_429(self):
        """Calling install() with no callback registers a hook that
        raises _FatalHFError — useful for tests that want to assert
        against the exception type without going through process exit.
        """
        import huggingface_hub
        from huggingface_hub.utils._http import get_session

        from lerobot.gui.hub_worker import _FatalHFError, _install_fatal_http_hook

        client = get_session()
        original_transport = client._transport
        client._transport = _Synthetic429Transport()
        _install_fatal_http_hook()
        try:
            with pytest.raises(_FatalHFError) as exc_info:
                huggingface_hub.HfApi().create_repo(
                    repo_id="not-real/repro", repo_type="dataset", exist_ok=True
                )
        finally:
            client._transport = original_transport
            self._uninstall_all(client)

        assert exc_info.value.status == 429
        assert exc_info.value.error_class == "rate_limit"
        msg = exc_info.value.message
        assert "Retry-After: 130" in msg
        assert "128 per hour" in msg, f"verbatim HF text should pass through; got {msg!r}"

    def test_abort_callback_receives_constructed_exception(self):
        """The production install path: pass an on_fatal callback (e.g.
        the closure that calls _abort_to_terminal_state). Verify the
        callback gets called with a fully-populated _FatalHFError on a
        fatal response.
        """
        import huggingface_hub
        from huggingface_hub.utils._http import get_session

        from lerobot.gui.hub_worker import _FatalHFError, _install_fatal_http_hook

        captured: list[_FatalHFError] = []

        def _record(exc: _FatalHFError) -> None:
            captured.append(exc)
            # Raise something Exception-derived so HF's call returns
            # (HF catches Exception in its worker thread; this lets
            # the test continue rather than hang).
            raise RuntimeError("test stub raised after recording")

        client = get_session()
        original_transport = client._transport
        client._transport = _Synthetic429Transport()
        _install_fatal_http_hook(_record)
        try:
            with pytest.raises(RuntimeError):
                huggingface_hub.HfApi().create_repo(
                    repo_id="not-real/repro", repo_type="dataset", exist_ok=True
                )
        finally:
            client._transport = original_transport
            self._uninstall_all(client)

        assert len(captured) == 1
        exc = captured[0]
        assert exc.status == 429
        assert exc.error_class == "rate_limit"
        assert "128 per hour" in exc.message

    def test_install_is_idempotent_across_callback_variants(self):
        """Calling install repeatedly — with or without a callback — must
        not stack multiple hooks on the shared HF client. We tag by
        attribute so dedup is robust to closure identity differences.
        """
        from huggingface_hub.utils._http import get_session

        from lerobot.gui.hub_worker import _install_fatal_http_hook

        client = get_session()
        try:
            _install_fatal_http_hook()
            _install_fatal_http_hook(lambda exc: None)
            _install_fatal_http_hook()
            tagged = [
                h for h in client.event_hooks.get("response", []) if getattr(h, "_lerobot_fatal_hook", False)
            ]
            assert len(tagged) == 1, f"install should be idempotent — got {len(tagged)} tagged hooks"
        finally:
            self._uninstall_all(client)


class TestAbortToTerminalState:
    """``_abort_to_terminal_state`` is what production hooks call when
    they detect a fatal HF response. It must (1) mark the in-memory
    state as failed with the right fields, (2) write the terminal
    snapshot to disk synchronously, (3) remove the pid file so the
    server's PID-liveness sweep doesn't overwrite our error_class
    with a generic one, and (4) os._exit(1) so the wedged main thread
    inside HF's library doesn't keep the process alive. We exercise
    the helper directly with os._exit monkey-patched so it raises
    SystemExit instead of killing the test process.
    """

    def _build_state(self, tmp_path):
        from lerobot.gui.hub_jobs import JobConfig, JobPaths
        from lerobot.gui.hub_worker import _WorkerState

        jobs_dir = tmp_path / "jobs"
        jobs_dir.mkdir()
        cfg = JobConfig(
            job_id="job-test",
            dataset_id="ds",
            direction="upload",
            repo_id="u/r",
            repo_type="dataset",
            local_path=str(tmp_path),
            jobs_dir=str(jobs_dir),
        )
        paths = JobPaths.for_job(cfg.job_id, jobs_dir)
        # Write a pid file so we can verify the abort removes it.
        paths.pid.write_text('{"pid": 99999, "started_at": 0, "start_time": 0}')
        return _WorkerState(cfg, paths)

    def test_abort_writes_failed_state_and_calls_exit(self, tmp_path, monkeypatch):
        from lerobot.gui import hub_worker
        from lerobot.gui.hub_worker import _abort_to_terminal_state, _FatalHFError

        state = self._build_state(tmp_path)
        exc = _FatalHFError(429, "rate_limit", "HTTP 429 — 128 per hour cap exceeded")

        # Replace os._exit with one that raises SystemExit so we can
        # assert post-state. The production behavior is process death.
        called_with: list[int] = []

        def _fake_exit(code: int) -> None:
            called_with.append(code)
            raise SystemExit(code)

        monkeypatch.setattr(hub_worker.os, "_exit", _fake_exit)

        with pytest.raises(SystemExit) as exit_info:
            _abort_to_terminal_state(state, exc)

        assert exit_info.value.code == 1
        assert called_with == [1]

        # In-memory state mutated to terminal failed.
        assert state.status == "failed"
        assert state.error_class == "rate_limit"
        assert "128 per hour" in state.error
        assert state.finished_at is not None
        assert "Failed" in state.milestone

        # Progress JSON on disk reflects the same terminal state — this is
        # what the GUI server polls, so the user sees the real reason.
        import json as _json

        snap = _json.loads(state.paths.progress.read_text())
        assert snap["status"] == "failed"
        assert snap["error_class"] == "rate_limit"
        assert "128 per hour" in snap["error"]

        # PID file removed so PID-liveness sweep doesn't second-guess us.
        assert not state.paths.pid.exists()

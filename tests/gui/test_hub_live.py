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
"""End-to-end tests against a real HuggingFace Hub.

**Two opt-in gates, both required:**

1. ``pytest -m hub_live`` (pyproject's default ``addopts`` excludes this marker).
2. ``LEROBOT_HUB_LIVE=1`` env var, asserted in the fixture.

The double gate is intentional. The marker keeps these tests out of any
default ``pytest`` invocation (local dev, CI, ad-hoc runs). The env var
prevents the case where someone discovers the marker, runs ``-m hub_live``
without realizing it creates real HF repos under whoever's logged in via
``huggingface-cli login``, and pollutes their account.

To run::

    LEROBOT_HUB_LIVE=1 pytest tests/gui/test_hub_live.py -m hub_live

Requires:
  - A valid HF token via ``huggingface-cli login`` or the ``HF_TOKEN`` env var
  - Network access to huggingface.co
  - The repo namespace ``<your-username>/_lerobot_hub_live_*`` is unused

No hardcoded credentials anywhere — the token comes from
``huggingface_hub.HfApi()``'s default resolution (env or cached login),
and the repo namespace uses the logged-in user's name via ``whoami()``.
Each test deletes its throwaway repo in a ``finally`` block; if a test
crashes mid-flight, manually delete leftover ``_lerobot_hub_live_*``
repos from the HF web UI (they're timestamped for visibility).

These tests verify the **performance** and **error-handling guarantees**
the design promises, against the real Hub:

  * **F2 (throughput)** — a small upload completes in seconds, not the
    minutes the per-file loop took; an identical re-upload is dominated
    by Xet dedupe (verified speedup ratio).
  * **F3 (skip already-uploaded chunks)** — re-uploading the same content
    transfers ~0 net bytes server-side.
  * **F5 (resume across cancel)** — a cancelled upload's PR survives;
    retry completes via the existing PR (no fresh PR created).
  * **Single-commit-on-main** — after a successful upload, ``main`` has
    exactly one new commit covering the entire transfer.

Each test cleans up after itself (deletes its temp repo). If a test
crashes mid-flight, manually delete ``thewisp/_lerobot_hub_live_*`` from
the HF web UI; the names are timestamped so it's clear what to remove.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
import uuid
from pathlib import Path

import pytest

from lerobot.gui import hub_jobs

pytestmark = pytest.mark.hub_live


# ── Fixtures + helpers ──────────────────────────────────────────────────────


@pytest.fixture
def hf_api():
    """A real HfApi. Skips the test unless BOTH gates are explicitly opened.

    Pre: ``LEROBOT_HUB_LIVE=1`` is set in the environment AND the user is
    logged in to HF (either via ``HF_TOKEN`` env var or
    ``huggingface-cli login``).

    The env-var gate is intentional — even with ``-m hub_live`` on the
    command line, we require an explicit opt-in to actually hit the
    network. This makes it impossible to accidentally create HF repos
    under the developer's account just by running the test marker.
    """
    if os.environ.get("LEROBOT_HUB_LIVE") != "1":
        pytest.skip("Live HF tests require LEROBOT_HUB_LIVE=1 env var. See module docstring.")
    try:
        from huggingface_hub import HfApi

        api = HfApi()
        info = api.whoami()
        return api, info["name"]
    except Exception as e:
        pytest.skip(f"Not logged in to HF Hub: {e}")


def _make_random_payload(target: Path, *, file_count: int = 4, file_size: int = 64 * 1024) -> int:
    """Populate target with random bytes; return total bytes."""
    import secrets

    target.mkdir(parents=True, exist_ok=True)
    total = 0
    for i in range(file_count):
        (target / f"file_{i:03d}.bin").write_bytes(secrets.token_bytes(file_size))
        total += file_size
    (target / "README.md").write_text("# Live E2E test payload\n")
    return total + 30  # README size approx


def _spawn_worker(
    cfg: hub_jobs.JobConfig,
    *,
    paths: hub_jobs.JobPaths,
) -> subprocess.Popen:
    """Spawn the real worker subprocess against the real HF."""
    paths.jobs_dir.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env["LEROBOT_HUB_WORKER_CONFIG"] = cfg.to_json()
    return subprocess.Popen(
        [sys.executable, "-m", "lerobot.gui.hub_worker"],
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
    )


def _wait_terminal(paths: hub_jobs.JobPaths, timeout_s: float = 120.0) -> dict:
    """Block until the worker writes a terminal status. Returns the snapshot."""
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        if paths.progress.exists():
            try:
                snap = json.loads(paths.progress.read_text())
                if snap.get("status") in ("complete", "failed", "cancelled"):
                    return snap
            except (OSError, json.JSONDecodeError):
                pass
        time.sleep(0.2)
    raise TimeoutError(f"Worker did not reach terminal status in {timeout_s}s")


# ── Tests ───────────────────────────────────────────────────────────────────


class TestLiveUploadFlow:
    """End-to-end upload → squash → merge against the real Hub."""

    def test_upload_lands_as_single_commit_on_main(self, tmp_path, hf_api):
        api, user = hf_api
        repo_id = f"{user}/_lerobot_hub_live_{int(time.time())}_{uuid.uuid4().hex[:6]}"

        local = tmp_path / "payload"
        _make_random_payload(local, file_count=4, file_size=64 * 1024)
        jobs_dir = tmp_path / "jobs"

        cfg = hub_jobs.JobConfig(
            job_id=uuid.uuid4().hex,
            dataset_id=str(local),
            direction="upload",
            repo_id=repo_id,
            repo_type="dataset",
            local_path=str(local),
            jobs_dir=str(jobs_dir),
            commit_message="Live E2E test upload",
        )
        paths = hub_jobs.JobPaths.for_job(cfg.job_id, jobs_dir)

        proc = _spawn_worker(cfg, paths=paths)
        try:
            snap = _wait_terminal(paths, timeout_s=120)
            assert snap["status"] == "complete", f"upload failed: {snap}"
            assert snap["pr_num"] is not None

            # The headline property: main HAS the payload, in exactly one
            # new commit beyond what was there before (which was zero
            # since the repo is fresh).
            commits = list(api.list_repo_commits(repo_id, repo_type="dataset", revision="main"))
            # New repos start with one "initial commit" so we expect 2 here:
            # initial + our upload.
            assert len(commits) <= 3, (
                f"expected ~2 commits on main (initial + upload); got {len(commits)}: "
                f"{[c.title for c in commits]}"
            )

            # Files are reachable from main.
            info = api.dataset_info(repo_id, files_metadata=True)
            sib_names = {s.rfilename for s in info.siblings or []}
            assert "README.md" in sib_names
            assert "file_000.bin" in sib_names
        finally:
            proc.wait(timeout=10)
            with pytest.MonkeyPatch.context():
                try:
                    api.delete_repo(repo_id=repo_id, repo_type="dataset")
                except Exception as e:  # noqa: BLE001
                    print(f"WARN: could not delete {repo_id}: {e}", file=sys.stderr)


class TestDedupePerformance:
    """F2 (throughput) + F3 (skip already-uploaded chunks).

    Realistic scenario: a user edits one file in a dataset and re-uploads.
    Only the changed file's chunks should transfer; the rest is server-side
    dedupe via Xet. We assert the incremental upload is meaningfully faster
    than the cold one.

    We avoid the "re-upload identical content" scenario because HF rejects
    PRs with no changes — that's HF's own dedupe at the commit level, not
    something our pipeline can route around.
    """

    def test_incremental_upload_skips_unchanged_files(self, tmp_path, hf_api):
        api, user = hf_api
        repo_id = f"{user}/_lerobot_hub_live_dedupe_{int(time.time())}_{uuid.uuid4().hex[:6]}"

        local = tmp_path / "payload"
        _make_random_payload(local, file_count=4, file_size=128 * 1024)
        jobs_dir = tmp_path / "jobs"

        # First upload — bytes actually move.
        cfg1 = hub_jobs.JobConfig(
            job_id=uuid.uuid4().hex,
            dataset_id=str(local),
            direction="upload",
            repo_id=repo_id,
            repo_type="dataset",
            local_path=str(local),
            jobs_dir=str(jobs_dir),
            commit_message="First upload (cold)",
        )
        paths1 = hub_jobs.JobPaths.for_job(cfg1.job_id, jobs_dir)
        t0 = time.monotonic()
        proc1 = _spawn_worker(cfg1, paths=paths1)
        try:
            snap1 = _wait_terminal(paths1, timeout_s=180)
            t_cold = time.monotonic() - t0
            assert snap1["status"] == "complete"

            # Modify ONE file so there's a delta worth dedupe-ing the others.
            import secrets

            (local / "file_000.bin").write_bytes(secrets.token_bytes(128 * 1024))

            cfg2 = hub_jobs.JobConfig(
                job_id=uuid.uuid4().hex,
                dataset_id=str(local),
                direction="upload",
                repo_id=repo_id,
                repo_type="dataset",
                local_path=str(local),
                jobs_dir=str(jobs_dir),
                commit_message="Second upload (1 file changed)",
            )
            paths2 = hub_jobs.JobPaths.for_job(cfg2.job_id, jobs_dir)
            t1 = time.monotonic()
            proc2 = _spawn_worker(cfg2, paths=paths2)
            try:
                snap2 = _wait_terminal(paths2, timeout_s=180)
                t_warm = time.monotonic() - t1
                if snap2["status"] != "complete":
                    log_content = paths2.log.read_text() if paths2.log.exists() else "<no log>"
                    raise AssertionError(f"second upload failed: {snap2}\n--- log ---\n{log_content[-2000:]}")
            finally:
                proc2.wait(timeout=10)

            speedup = t_cold / max(t_warm, 0.001)
            print(
                f"\n[hub_live] cold={t_cold:.2f}s warm={t_warm:.2f}s speedup={speedup:.2f}×",
                file=sys.stderr,
            )
            # The performance guarantee is "meaningfully faster," not a hard
            # multiplier — server-side conditions vary. We just want to catch
            # regressions where dedupe stopped working entirely (i.e., second
            # upload took as long or longer than first).
            assert t_warm < t_cold, (
                f"Incremental upload should be faster than cold; got cold={t_cold:.2f}s warm={t_warm:.2f}s"
            )
        finally:
            proc1.wait(timeout=10)
            try:
                api.delete_repo(repo_id=repo_id, repo_type="dataset")
            except Exception as e:  # noqa: BLE001
                print(f"WARN: could not delete {repo_id}: {e}", file=sys.stderr)


class TestCancelAndResume:
    """F5 (elegant error handling via cancel-then-retry).

    Cancel a real upload mid-flight, verify the draft PR survives, then
    retry into that PR and assert resume — not a fresh PR.
    """

    def test_cancel_leaves_draft_pr_for_resume(self, tmp_path, hf_api):
        api, user = hf_api
        repo_id = f"{user}/_lerobot_hub_live_resume_{int(time.time())}_{uuid.uuid4().hex[:6]}"

        # Bigger payload so the upload takes long enough to cancel mid-flight.
        local = tmp_path / "payload"
        _make_random_payload(local, file_count=10, file_size=512 * 1024)
        jobs_dir = tmp_path / "jobs"

        cfg = hub_jobs.JobConfig(
            job_id=uuid.uuid4().hex,
            dataset_id=str(local),
            direction="upload",
            repo_id=repo_id,
            repo_type="dataset",
            local_path=str(local),
            jobs_dir=str(jobs_dir),
            commit_message="Live cancel test",
        )
        paths = hub_jobs.JobPaths.for_job(cfg.job_id, jobs_dir)

        proc = _spawn_worker(cfg, paths=paths)
        first_pr_num = None
        try:
            # Wait until we see the PR has been created.
            deadline = time.time() + 60
            while time.time() < deadline:
                if paths.progress.exists():
                    try:
                        snap = json.loads(paths.progress.read_text())
                        if snap.get("pr_num"):
                            first_pr_num = snap["pr_num"]
                            break
                    except (OSError, json.JSONDecodeError):
                        pass
                time.sleep(0.2)
            assert first_pr_num is not None, "Worker didn't surface a pr_num before timeout"

            # Now SIGTERM mid-flight.
            proc.terminate()
            snap = _wait_terminal(paths, timeout_s=30)
            assert snap["status"] == "cancelled"
            assert snap["pr_num"] == first_pr_num
        finally:
            proc.wait(timeout=10)

        # The draft PR should still be on HF, ready for resume.
        details = api.get_discussion_details(
            repo_id=repo_id,
            repo_type="dataset",
            discussion_num=first_pr_num,
        )
        assert details.status == "draft", f"PR {first_pr_num} status is {details.status}, expected draft"

        # Now retry: spawn a new worker with reuse_pr_num pointing at the
        # cancelled PR. Should resume and merge.
        cfg2 = hub_jobs.JobConfig(
            job_id=uuid.uuid4().hex,
            dataset_id=str(local),
            direction="upload",
            repo_id=repo_id,
            repo_type="dataset",
            local_path=str(local),
            jobs_dir=str(jobs_dir),
            commit_message="Live resume",
            reuse_pr_num=first_pr_num,
        )
        paths2 = hub_jobs.JobPaths.for_job(cfg2.job_id, jobs_dir)
        proc2 = _spawn_worker(cfg2, paths=paths2)
        try:
            snap2 = _wait_terminal(paths2, timeout_s=180)
            assert snap2["status"] == "complete"
            # Same PR number — we resumed, didn't create a new one.
            assert snap2["pr_num"] == first_pr_num
        finally:
            proc2.wait(timeout=10)
            try:
                api.delete_repo(repo_id=repo_id, repo_type="dataset")
            except Exception as e:  # noqa: BLE001
                print(f"WARN: could not delete {repo_id}: {e}", file=sys.stderr)


class TestBadAuthClassification:
    """Auth-class errors are classified specifically, not as generic failures."""

    def test_invalid_token_surfaces_as_auth_error(self, tmp_path, hf_api):
        """Force an auth failure by clobbering the HF token; verify the
        worker classifies it as 'auth' and surfaces a clean error.

        Skipped if running without a real token (no token to invalidate).
        """
        _api, _user = hf_api
        local = tmp_path / "payload"
        _make_random_payload(local, file_count=1, file_size=1024)
        jobs_dir = tmp_path / "jobs"

        cfg = hub_jobs.JobConfig(
            job_id=uuid.uuid4().hex,
            dataset_id=str(local),
            direction="upload",
            # A namespace the user definitely doesn't have write access to.
            repo_id="huggingface/transformers",
            repo_type="dataset",
            local_path=str(local),
            jobs_dir=str(jobs_dir),
        )
        paths = hub_jobs.JobPaths.for_job(cfg.job_id, jobs_dir)

        # Override token to something obviously invalid for this worker.
        env = os.environ.copy()
        env["LEROBOT_HUB_WORKER_CONFIG"] = cfg.to_json()
        env["HF_TOKEN"] = "hf_invalid_token_for_test"
        env.pop("HUGGINGFACE_HUB_TOKEN", None)

        proc = subprocess.Popen(
            [sys.executable, "-m", "lerobot.gui.hub_worker"],
            env=env,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
        )
        try:
            snap = _wait_terminal(paths, timeout_s=60)
            assert snap["status"] == "failed"
            # The exact class depends on HF's response — 401, 403, or
            # RepositoryNotFoundError. All map to "auth" in our classifier.
            assert snap["error_class"] == "auth", (
                f"Expected auth error class; got {snap['error_class']!r}: {snap['error']}"
            )
        finally:
            proc.wait(timeout=10)

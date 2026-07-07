# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""Run orchestrator — starts, polls, stops training runs.

The orchestrator is the bridge between the API layer (which receives Start /
Stop / List from the frontend) and the lower-level pieces:

- :class:`HostRegistry` — picks which transport to use for a given host id
- :class:`TransportClient` — launches / reads from the training process
- :class:`RunRegistry` — persists run metadata + state machine

It does NOT own background polling. The API layer (FastAPI) is expected to
call :meth:`poll` periodically, which reads the structured files the worker
writes and updates the Run's state. This keeps the orchestrator stateless
beyond the registry, so resume across GUI server restarts is automatic.

DESIGN.md § Concurrency:
- Idempotency keys deflect double-Start clicks
- Per-host single-active-run lock prevents two trainings on the same host
- State machine prevents stale duplicate Start from re-running a finished run
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import logging
import re
import shutil
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from lerobot.gui.training.hosts import HostRegistry, TrainingHost
from lerobot.gui.training.jobs import atomic_write_json
from lerobot.gui.training.log_parse import ProgressSample, parse_metric_sample, parse_progress
from lerobot.gui.training.providers import get_provider
from lerobot.gui.training.providers.protocol import HostHandle
from lerobot.gui.training.recipes import (
    build_lerobot_train_command,
    is_fake_recipe,
    output_subdir_in_run,
    resolve_host_placeholders,
)
from lerobot.gui.training.runs import (
    TERMINAL_STATES,
    Run,
    RunPaths,
    RunRegistry,
    RunState,
    new_run_id,
)
from lerobot.gui.training.transport import (
    SshTransport,
    SubprocessClient,
    SubprocessTransport,
    TransportClient,
    make_client,
)

logger = logging.getLogger(__name__)

# How long to wait for a freshly-spawned ephemeral VM to accept SSH before
# giving up. Cloud-init (user + key) on a GPU VM is typically well under 2 min
# after the instance reports RUNNING; 5 min is a safe ceiling.
_EPHEMERAL_SSH_READY_TIMEOUT_S = 300.0

# How many times to (re-)fetch an artifact whose transfer failed or whose
# bytes don't match the remote sha256. A dropped scp can leave a partial file;
# we overwrite + re-verify rather than trust it.
_ARTIFACT_FETCH_RETRIES = 3


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _count_local_models(paths: RunPaths) -> int:
    """How many checkpoint model files are on the GUI server for this run.

    Counts both the docker layout (``output/checkpoints/<step>/pretrained_model/
    model.safetensors``) and the flat fake-runner layout. Used post-fetch at
    teardown: 0 for a completed run means the model never made it off the VM.
    """
    for base in (paths.root / "output" / "checkpoints", paths.root / "checkpoints"):
        if base.is_dir():
            return sum(
                1
                for d in base.iterdir()
                if (d / "pretrained_model" / "model.safetensors").is_file()
                or (d / "model.safetensors").is_file()
            )
    return 0


# ── Requests / responses ──────────────────────────────────────────────────────


@dataclass(frozen=True, kw_only=True)
class StartRequest:
    """What the API layer hands to :meth:`Orchestrator.start`."""

    host_id: str
    recipe_name: str
    dataset_id: str
    args: dict[str, Any] = field(default_factory=dict)
    idempotency_key: str | None = None


@dataclass(frozen=True)
class CheckpointEntry:
    """One line of ``checkpoints.jsonl`` — a completed checkpoint."""

    step: int
    path: str  # relative to the run dir
    sha256: str
    ts: float


@dataclass(frozen=True)
class RunSnapshot:
    """Polling result — current state of a run plus what's been observed.

    All fields readable by the API to render the UI without round-tripping
    to the orchestrator again.
    """

    run: Run
    progress: dict[str, Any] | None  # position snapshot (progress.json), or None if not parsed yet
    checkpoints: list[CheckpointEntry]  # all manifest entries
    stderr_tail: str  # last N bytes of stderr.log (configurable on poll)
    events: list[dict[str, Any]]  # all events.jsonl entries (oldest first)
    metrics: list[dict[str, float]]  # training-signal series (metrics.jsonl), one row per logged step


# ── Orchestrator ──────────────────────────────────────────────────────────────


# Number of bytes of stderr.log we surface back to the caller. Cheap to read;
# the full log is on disk for "open in files" later.
DEFAULT_STDERR_TAIL_BYTES = 16 * 1024


class HostBusyError(RuntimeError):
    """A start request targeted a host that already has an active run."""


class UnknownHostError(KeyError):
    """The requested host id isn't in the registry."""


class UnknownRunError(KeyError):
    """The requested run id doesn't exist."""


class RunNotTerminalError(RuntimeError):
    """A delete request hit a run that's still pending / running / completing.

    Refuses to delete an active run because:
      - the prep thread or worker subprocess is writing to files we'd
        rmtree out from under them
      - the per-host single-active-run lock would silently un-lock
      - the user's most likely intent was "stop and forget", not "kill
        and delete"; surfacing this as an error makes the two-step
        explicit (stop, then delete).
    """


class Orchestrator:
    """Owns start / poll / stop for training runs.

    Stateless beyond the two registries it composes: any state survives
    GUI server restart by reading from disk on the next call. No background
    threads — the API layer drives polling.
    """

    def __init__(
        self,
        host_registry: HostRegistry,
        run_registry: RunRegistry,
        *,
        make_client_fn: Callable[[Any], TransportClient] | None = None,
        provider_factory: Callable[..., Any] | None = None,
    ) -> None:
        self._hosts = host_registry
        self._runs = run_registry
        # Resolve a HostProvider by id for Ephemeral spawn/destroy. The Nebius
        # provider wires in the server-held service-account connection itself
        # (see :func:`get_provider`), so no per-run credential threading is
        # needed — background destroy authenticates with the same stored key.
        # Tests inject a fake provider (no SDK/credentials); production uses
        # the registry's :func:`get_provider`.
        self._provider_factory = provider_factory or get_provider
        # All host-state ops (file reads, dir listings, image docker calls,
        # checkpoint sha256s) go through the resolved TransportClient. Tests
        # inject a fake-transport factory; production uses :func:`make_client`
        # which picks SubprocessClient or SshClient based on the host's
        # transport type. See [DESIGN.md § HostProvider] + the
        # ``TransportClient`` Protocol in ``training/transport.py``.
        self._make_client_fn = make_client_fn or make_client
        # Background prep threads (image pull + worker spawn). Keyed by run_id;
        # daemon=True so they don't block process shutdown. We keep refs so
        # tests can join them; in production they're fire-and-forget.
        self._prep_threads: dict[str, threading.Thread] = {}
        # Per-run teardown locks. poll() runs on FastAPI's threadpool, so
        # several concurrent polls can enter _maybe_teardown_ephemeral at once;
        # without serialization one poll's pre-destroy artifact fetch races
        # another's provider.destroy() — the VM vanishes mid-scp (round-6: the
        # model never localized). One lock per run keeps cross-run teardowns
        # independent.
        self._teardown_locks: dict[str, threading.Lock] = {}
        self._teardown_locks_guard = threading.Lock()

    # ── Public API ────────────────────────────────────────────────────────────

    def start(self, req: StartRequest) -> Run:
        """Create + launch a new run.

        Deflects double-clicks via idempotency_key. Refuses if the target
        host already has an active run.
        """
        # 1. Idempotency: same key → return same run
        if req.idempotency_key:
            existing = self._runs.find_by_idempotency_key(req.idempotency_key)
            if existing is not None:
                return existing

        # 2. Host exists?
        host = self._hosts.get(req.host_id)
        if host is None:
            raise UnknownHostError(f"unknown host id: {req.host_id!r}")

        # 3. Per-host single-active-run lock
        busy = self._runs.active_run_on_host(req.host_id)
        if busy is not None:
            raise HostBusyError(
                f"host {req.host_id!r} is busy with run {busy.run_id!r} (state={busy.state.value})"
            )

        # 4. Create run, persist as PENDING
        run = Run(
            run_id=new_run_id(),
            host_id=req.host_id,
            recipe_name=req.recipe_name,
            dataset_id=req.dataset_id,
            args=dict(req.args),
            state=RunState.PENDING,
            created_at=time.time(),
            idempotency_key=req.idempotency_key,
        )
        self._runs.save(run)
        paths = RunPaths.for_run(run.run_id, self._runs.runs_dir)
        paths.ensure_exists()

        # 5. Prepare image + launch worker in a background thread.
        #
        # The HTTP request returns immediately with state=PENDING. The
        # background thread emits ``image_cache_hit`` / ``image_pull_started`` /
        # ``image_pulled`` / ``image_pull_failed`` events into events.jsonl
        # as it works, and the frontend's existing run-detail poller renders
        # them as visible status. On success it spawns the worker and
        # advances to RUNNING; on failure it advances to FAILED.
        #
        # Why a thread rather than a synchronous pull: first-time image
        # pulls measured ~13 min on this workstation's connection. Blocking
        # the POST that long times out the browser's fetch — and the user
        # sees nothing in the meantime since the run row only appears
        # after the response.
        t = threading.Thread(
            target=self._prepare_and_launch,
            args=(host, run.run_id, paths),
            daemon=True,
            name=f"prepare-{run.run_id[:8]}",
        )
        self._prep_threads[run.run_id] = t
        t.start()
        return run

    def poll(
        self,
        run_id: str,
        *,
        stderr_tail_bytes: int = DEFAULT_STDERR_TAIL_BYTES,
    ) -> RunSnapshot:
        """Read the worker's state files and reconcile with the state machine.

        Detects natural completion / abort / crash by reading the final
        ``events.jsonl`` entry (worker writes it before exit) cross-checked
        against the transport's ``is_alive``.

        Any ephemeral teardown driven by this poll authenticates with the
        server-held Nebius service-account key (resolved by the provider
        factory), so no per-request credential is needed.
        """
        run = self._runs.load(run_id)
        if run is None:
            raise UnknownRunError(f"unknown run id: {run_id!r}")

        paths = RunPaths.for_run(run.run_id, self._runs.runs_dir)
        host = self._hosts.get(run.host_id)
        client = self._client_for_host(host, paths, run)

        # Reconcile state with the worker, if it's still in a live state.
        # We only do the liveness probe when the host is known; otherwise
        # the run is treated as "we can read what we have, but we can't
        # check on it." Same semantic as before the refactor.
        if run.state in (RunState.RUNNING, RunState.COMPLETING) and host is not None:
            self._reconcile_state(run, paths, client)

        # Derive real position + training-signal from the host's stdout. This
        # is what populates the dashboard for real lerobot-train runs (which
        # print but never write progress.json). No-op when nothing parseable
        # has been logged yet.
        self._ingest_training_log(client, paths)

        progress = self._read_progress(client, paths.progress_json)
        checkpoints = self._read_manifest(client, paths.checkpoints_jsonl)
        metrics = self._read_metrics(paths.metrics_jsonl)

        # Completed-but-artifacts-elsewhere: a run that finished while the
        # GUI was down (or before the fetch feature existed) has a manifest
        # but no local model files. Guarded by a cheap local check so a
        # fully-localized run costs nothing per poll; only attempted while
        # the host is still registered.
        if (
            run.state == RunState.COMPLETED
            and host is not None
            and checkpoints
            and not (paths.root / checkpoints[-1].path).exists()
        ):
            self._fetch_run_artifacts(client, run, paths)
        stderr_tail = self._read_stderr_tail(client, paths.stderr_log, stderr_tail_bytes)
        events = self._read_events(client, paths.events_jsonl)

        # Ephemeral teardown LAST — after every remote read above, so the
        # final log/checkpoint pull happens while the VM is still alive
        # (artifact localization runs inside _reconcile_state on completion).
        # No-op unless this run is ephemeral, terminal, and not yet destroyed.
        self._maybe_teardown_ephemeral(run, paths)

        return RunSnapshot(
            run=run,
            progress=progress,
            checkpoints=checkpoints,
            stderr_tail=stderr_tail,
            events=events,
            metrics=metrics,
        )

    def stop(self, run_id: str) -> Run:
        """User-initiated stop — SIGTERM the worker, mark COMPLETING.

        The worker writes its final ``aborted_by_user`` event then exits;
        next ``poll()`` reconciles to ABORTED. Idempotent on terminal runs.

        Any ephemeral teardown driven by this stop authenticates with the
        server-held Nebius service-account key.
        """
        run = self._runs.load(run_id)
        if run is None:
            raise UnknownRunError(f"unknown run id: {run_id!r}")
        if run.state in TERMINAL_STATES:
            return run  # idempotent — already stopped
        if run.state == RunState.COMPLETING:
            return run  # idempotent — already stopping
        paths = RunPaths.for_run(run.run_id, self._runs.runs_dir)
        host = self._hosts.get(run.host_id)
        client = self._client_for_host(host, paths, run)
        if run.state == RunState.PENDING:
            # Prep thread is still running (image pull or pre-launch). No
            # worker to SIGTERM. Skip COMPLETING straight to ABORTED — the
            # prep thread will reload run state before launching and bail
            # if it sees a terminal state. (If it raced past that check
            # already, the spawned worker is --rm so it cleans up on exit.)
            run.advance(RunState.ABORTED)
            self._runs.save(run)
            self._emit_event(client, paths.events_jsonl, "aborted_by_user", final_step=0)
            # If the prep thread already spawned the VM, tear it down. (A
            # spawn racing in parallel is covered by the poll-time backstop.)
            self._maybe_teardown_ephemeral(run, paths)
            return run
        if host is None:
            # Host went away (deleted profile) — best-effort mark aborted.
            run.advance(RunState.ABORTED)
            self._runs.save(run)
            self._maybe_teardown_ephemeral(run, paths)
            return run
        if run.session_id is not None:
            client.stop(run.session_id, force=False)
        run.advance(RunState.COMPLETING)
        self._runs.save(run)
        self._emit_event(client, paths.events_jsonl, "stop_requested")
        return run

    def list_runs(self) -> list[Run]:
        """List all runs. Cheaply reconciles each non-terminal run from its
        ``events.jsonl`` so the list view shows up-to-date state even for
        runs the user hasn't clicked on (no transport calls — just a file
        read per non-terminal run).

        Full reconciliation including the process-liveness probe still lives
        in :meth:`poll` for the selected run.
        """
        runs = self._runs.list_all()
        for run in runs:
            if run.state in TERMINAL_STATES:
                continue
            paths = RunPaths.for_run(run.run_id, self._runs.runs_dir)
            if self._reconcile_from_events_only(run, paths):
                self._runs.save(run)
        return runs

    def delete_run(self, run_id: str) -> dict[str, Any]:
        """Drop a run from the training history.

        Removes the run's metadata files (run.json, events.jsonl,
        checkpoints.jsonl, stderr.log, progress.json) so the row stops
        showing up in the Training list. **Preserves
        ``output/checkpoints/`` intact** — dropping a run from history is
        NOT throwing away the model the run produced. The trained
        checkpoint continues to surface in the Models tab; disk cleanup
        of the model itself is a separate Models-tab action.

        If the run produced no checkpoints (failed pull, aborted before
        first save), the whole dir is removed — nothing to preserve.

        Returns ``{"run_id": str, "metadata_bytes_freed": int,
        "kept_model": bool}``.

        Refuses with :class:`RunNotTerminalError` if the run is still in a
        pending / running / completing state — stop it first. Refuses
        with :class:`UnknownRunError` if the run id is unknown.
        """
        run = self._runs.load(run_id)
        if run is None:
            raise UnknownRunError(f"unknown run id: {run_id!r}")
        if run.state not in TERMINAL_STATES:
            raise RunNotTerminalError(
                f"run {run_id!r} is in state {run.state.value!r}; stop it first, then delete"
            )
        paths = RunPaths.for_run(run.run_id, self._runs.runs_dir)
        bytes_freed, kept_model = _drop_run_metadata(paths)
        return {
            "run_id": run_id,
            "metadata_bytes_freed": bytes_freed,
            "kept_model": kept_model,
        }

    def clear_terminal_runs(self) -> dict[str, Any]:
        """Bulk-drop every terminal-state run from training history.

        Per-row semantics match :meth:`delete_run`: keeps checkpoints,
        removes only metadata. Returns ``{"deleted": [run_ids],
        "metadata_bytes_freed": int, "models_kept": int}`` —
        ``models_kept`` counts the rows whose ``output/checkpoints/``
        survived (and thus stay visible in the Models tab). Idempotent.
        """
        deleted: list[str] = []
        bytes_freed = 0
        models_kept = 0
        for run in self._runs.list_all():
            if run.state not in TERMINAL_STATES:
                continue
            paths = RunPaths.for_run(run.run_id, self._runs.runs_dir)
            b, kept = _drop_run_metadata(paths)
            bytes_freed += b
            if kept:
                models_kept += 1
            deleted.append(run.run_id)
        return {
            "deleted": deleted,
            "metadata_bytes_freed": bytes_freed,
            "models_kept": models_kept,
        }

    # ── Internals ─────────────────────────────────────────────────────────────

    def _client_for_host(
        self, host: TrainingHost | None, paths: RunPaths, run: Run | None = None
    ) -> TransportClient:
        """Resolve a :class:`TransportClient` for ops on a given host.

        Run-aware: an Ephemeral run carries its spawned VM's coordinates in
        ``run.ephemeral_handle`` (persisted), so once the VM exists every
        transport op routes to it over SSH — even after a GUI restart that
        lost the in-memory handle. Falls back to a local
        :class:`SubprocessClient` when the host is unknown (deleted profile)
        or an Ephemeral run hasn't spawned yet, preserving the pre-refactor
        "read what's local" behavior.
        """
        if run is not None and run.ephemeral_handle is not None and not run.ephemeral_destroyed:
            handle = _handle_from_dict(run.ephemeral_handle)
            transport = SshTransport(host=handle.ssh_host, port=handle.ssh_port, user=handle.ssh_user)
            return self._make_client_fn(transport)
        # Destroyed-ephemeral (or unknown host): the VM is gone — read from
        # whatever was localized rather than hang SSH on a dead IP.
        if host is not None and host.transport is not None:
            return self._make_client_fn(host.transport)
        return SubprocessClient(SubprocessTransport(workdir=paths.root))

    def _maybe_teardown_ephemeral(self, run: Run, paths: RunPaths) -> None:
        """Destroy the spawned VM once an Ephemeral run is terminal.

        Idempotent via ``run.ephemeral_destroyed``: a no-op for non-ephemeral
        runs, already-destroyed runs, and non-terminal runs. Called from the
        chokepoints where a run becomes terminal (poll reconcile, prep-failure,
        stop-from-pending) so teardown is prompt; the cloud-init poweroff is
        only the backstop for "GUI server never ran this again".
        """
        # Fast pre-check before taking the lock (the common no-op case).
        if run.state not in TERMINAL_STATES or run.ephemeral_handle is None or run.ephemeral_destroyed:
            return
        with self._teardown_lock_for(run.run_id):
            # Re-load under the lock: a concurrent poll may have already torn
            # this run down. Operating on the fresh record makes destroy run
            # exactly once and keeps the pre-destroy fetch from racing another
            # poll's destroy (which would yank the VM out from under the scp).
            run = self._runs.load(run.run_id) or run
            if run.ephemeral_handle is None or run.ephemeral_destroyed:
                return
            handle = _handle_from_dict(run.ephemeral_handle)
            provider = self._provider_factory(handle.provider)
            local = SubprocessClient(SubprocessTransport(workdir=paths.root))
            # Localize artifacts off the VM BEFORE destroying it — the VM's copy
            # dies with it, so a completed run must first pull a verified model
            # and rebuild the manifest LOCALLY (it was written via the SSH client
            # during the run, i.e. on the VM). Best-effort: never block teardown,
            # but DON'T fail silently — the per-file fetch is best-effort and a
            # flaky scp can leave the model on the doomed VM (round-5/6). Record
            # how many checkpoints actually made it local so a lost model is
            # visible after the VM (and its remote events.jsonl) are gone.
            if run.state == RunState.COMPLETED:
                try:
                    ssh = self._client_for_host(self._hosts.get(run.host_id), paths, run)
                    self._fetch_run_artifacts(ssh, run, paths)
                    self._sync_checkpoints_manifest(local, run, paths)
                except Exception:  # noqa: BLE001 — teardown must never wedge the run
                    logger.exception("pre-teardown artifact localization failed for run %s", run.run_id)
                n_local = _count_local_models(paths)
                self._emit_event(
                    local,
                    paths.events_jsonl,
                    "artifacts_localized" if n_local else "artifacts_fetch_failed",
                    count=n_local,
                )
            try:
                provider.destroy(handle)
                ok = provider.verify_destroyed(handle)
            except Exception as exc:  # noqa: BLE001 — teardown must never wedge the run
                logger.exception("ephemeral teardown failed for run %s", run.run_id)
                self._emit_event(local, paths.events_jsonl, "vm_destroy_failed", error=str(exc)[:200])
                return
            run.ephemeral_destroyed = True
            self._runs.save(run)
            self._emit_event(
                local,
                paths.events_jsonl,
                "vm_destroyed",
                resource_id=handle.provider_resource_id,
                verified=ok,
            )
            if not ok:
                logger.warning(
                    "ephemeral teardown: destroy issued but verify_destroyed=False for %s "
                    "(resource %s) — check the Nebius console",
                    run.run_id,
                    handle.provider_resource_id,
                )

    def _teardown_lock_for(self, run_id: str) -> threading.Lock:
        with self._teardown_locks_guard:
            return self._teardown_locks.setdefault(run_id, threading.Lock())

    @staticmethod
    def _emit_event(client: TransportClient, events_path: Path, type_: str, **fields: Any) -> None:
        """Append a JSON event line to the host's events.jsonl via the
        transport. Used by the orchestrator for its own event emits
        (``started`` / ``image_*`` / ``aborted_by_user`` / ``crashed`` /
        ``completed_naturally``). The worker still uses the direct-file
        :func:`runs.append_event` because it runs inside the container
        with local filesystem access.
        """
        line = json.dumps({"type": type_, "ts": time.time(), **fields})
        client.append_text(events_path, line + "\n")

    # ── Image prep + launch (background thread entry point) ───────────────────

    def _prepare_and_launch(self, host: TrainingHost, run_id: str, paths: RunPaths) -> None:
        """Pre-pull the image if needed, then launch the worker.

        Runs in a daemon thread spawned from :meth:`start`. On success,
        advances run to RUNNING and emits ``started`` event. On failure,
        advances to FAILED and emits ``image_pull_failed`` or ``crashed``
        as appropriate. All errors are caught here — never bubble up;
        the run state IS the error channel.
        """
        # Re-load — start() saved PENDING; we own the lifecycle now.
        run = self._runs.load(run_id)
        if run is None:
            logger.error("prepare-and-launch: run %s vanished", run_id)
            return
        # Ephemeral: provision the VM BEFORE anything else and persist its
        # handle, so even a crash right after spawn leaves a teardownable
        # record. Spawn before local image-prep is intentional — the recipe's
        # mount sources + host identity are resolved against the spawned host.
        if host.is_ephemeral and run.ephemeral_handle is None:
            try:
                handle = self._spawn_ephemeral(host, run, paths)
            except Exception as exc:
                logger.exception("prepare-and-launch: ephemeral spawn failed")
                run.error = f"spawn failed: {exc!r}"
                run.advance(RunState.FAILED)
                self._runs.save(run)
                local = SubprocessClient(SubprocessTransport(workdir=paths.root))
                self._emit_event(local, paths.events_jsonl, "spawn_failed", error=str(exc)[:300])
                return
            run.ephemeral_handle = _handle_to_dict(handle)
            self._runs.save(run)
            # The VM reports RUNNING before sshd/cloud-init are up. Wait for SSH
            # to answer before image-prep / host-identity, else the first remote
            # op races the boot and dies on a single attempt (the round-2 bug).
            local = SubprocessClient(SubprocessTransport(workdir=paths.root))
            try:
                self._client_for_host(host, paths, run).wait_until_ready(
                    timeout_s=_EPHEMERAL_SSH_READY_TIMEOUT_S
                )
            except Exception as exc:
                logger.exception("prepare-and-launch: ephemeral host never became SSH-ready")
                run.error = f"ssh not ready: {exc!r}"
                run.advance(RunState.FAILED)
                self._runs.save(run)
                self._emit_event(local, paths.events_jsonl, "ssh_not_ready", error=str(exc)[:300])
                self._maybe_teardown_ephemeral(run, paths)
                return
            self._emit_event(local, paths.events_jsonl, "ssh_ready", resource_id=handle.provider_resource_id)
        client = self._client_for_host(host, paths, run)
        # Ensure the host can actually run training (Docker + nvidia-toolkit +
        # docker-group membership). One idempotent step for every host type: a
        # fresh ephemeral VM gets provisioned; a manually-added host is a fast
        # no-op when already set up. Skipped for the test-only fake recipe,
        # which runs plain Python with no Docker.
        if not is_fake_recipe(run):
            local = SubprocessClient(SubprocessTransport(workdir=paths.root))
            try:
                client.ensure_prereqs()
            except Exception as exc:
                logger.exception("prepare-and-launch: host prereqs failed")
                run.error = f"host prereqs failed: {exc!r}"
                run.advance(RunState.FAILED)
                self._runs.save(run)
                self._emit_event(local, paths.events_jsonl, "prereqs_failed", error=str(exc)[:300])
                self._maybe_teardown_ephemeral(run, paths)
                return
            self._emit_event(local, paths.events_jsonl, "prereqs_ready", host_id=host.id)
        try:
            cmd = self._build_command(run, paths)
            image = _extract_image_from_docker_argv(cmd)
            if image is not None:
                self._ensure_image(client, image, paths)
        except _ImagePullError as exc:
            # Already emitted image_pull_failed; flip state to FAILED.
            run.error = f"image pull failed: {exc}"
            run.advance(RunState.FAILED)
            self._runs.save(run)
            self._maybe_teardown_ephemeral(run, paths)
            return
        except Exception as exc:
            logger.exception("prepare-and-launch: unexpected error before launch")
            run.error = f"prepare failed: {exc!r}"
            run.advance(RunState.FAILED)
            self._runs.save(run)
            self._emit_event(client, paths.events_jsonl, "crashed", error=str(exc), final_step=0)
            self._maybe_teardown_ephemeral(run, paths)
            return
        # Race check: did the user stop us between image-prep and launch?
        # (stop() on PENDING transitions directly to ABORTED.)
        run = self._runs.load(run_id)
        if run is None or run.state != RunState.PENDING:
            logger.info(
                "prepare-and-launch: run %s no longer PENDING (state=%s) — skipping launch",
                run_id,
                run.state.value if run else "?",
            )
            return
        try:
            session_id = self._launch_worker(host, run, paths)
        except Exception as exc:
            logger.exception("prepare-and-launch: worker launch failed")
            run.error = f"launch failed: {exc!r}"
            run.advance(RunState.FAILED)
            self._runs.save(run)
            self._emit_event(client, paths.events_jsonl, "crashed", error=str(exc), final_step=0)
            self._maybe_teardown_ephemeral(run, paths)
            return
        # Final race check: stop() can land between launch and advance.
        # If so, kill what we just spawned. --rm on docker run cleans up the
        # container even if we miss this; the kill is best-effort UX.
        run_after = self._runs.load(run_id)
        if run_after is None or run_after.state != RunState.PENDING:
            try:
                client.stop(session_id, force=True)
            except Exception:
                logger.exception("prepare-and-launch: post-launch kill failed (best-effort)")
            if run_after is not None:
                self._maybe_teardown_ephemeral(run_after, paths)
            return
        run_after.session_id = session_id
        run_after.advance(RunState.RUNNING)
        self._runs.save(run_after)
        self._emit_event(client, paths.events_jsonl, "started", session_id=session_id, host_id=host.id)

    def _ensure_image(self, client: TransportClient, image: str, paths: RunPaths) -> None:
        """Make sure ``image`` is present in the host's docker cache.

        Uses the transport client's image ops — ``image_inspect`` to check,
        ``image_pull`` to fetch, ``image_size`` for the post-pull size.
        For SubprocessClient those shell out to local ``docker``; for
        SshClient they run ``docker`` over SSH on the remote host. The
        orchestrator's view is identical either way.

        Emits one of:
          - ``image_cache_hit`` — image already local; no pull.
          - ``image_pull_started`` + ``image_pulled`` — pull succeeded; latter
            carries ``duration_s`` and (when available) ``size_bytes``.
          - ``image_pull_started`` + ``image_pull_failed`` — pull failed;
            raises :class:`_ImagePullError` so the caller can flip the
            run state to FAILED.

        Always emits AT LEAST ONE event so the frontend can render a
        deterministic "what's happening" status. Pre: ``paths.root`` exists.
        """
        if client.image_inspect(image):
            self._emit_event(client, paths.events_jsonl, "image_cache_hit", image=image)
            return
        self._emit_event(client, paths.events_jsonl, "image_pull_started", image=image)
        t0 = time.time()
        ok, err = client.image_pull(image)
        duration_s = time.time() - t0
        if not ok:
            self._emit_event(
                client,
                paths.events_jsonl,
                "image_pull_failed",
                image=image,
                duration_s=round(duration_s, 3),
                error=err[:500],
            )
            raise _ImagePullError(err[:200])
        # Best-effort size after pull (the docker manifest gives the
        # compressed size; the inspect gives the on-disk uncompressed size
        # — the latter is what most people mean by "image size").
        size_bytes = client.image_size(image)
        self._emit_event(
            client,
            paths.events_jsonl,
            "image_pulled",
            image=image,
            duration_s=round(duration_s, 3),
            size_bytes=size_bytes,
        )

    def _spawn_ephemeral(self, host: TrainingHost, run: Run, paths: RunPaths) -> HostHandle:
        """Provision the Ephemeral VM via its provider. Emits spawn events
        on the (local) events.jsonl so the UI can show "provisioning…"."""
        local = SubprocessClient(SubprocessTransport(workdir=paths.root))
        provider = self._provider_factory(host.provider_id)
        self._emit_event(local, paths.events_jsonl, "spawn_started", provider=host.provider_id)
        handle = provider.spawn(host.spawn_spec)
        self._emit_event(
            local,
            paths.events_jsonl,
            "vm_spawned",
            provider=host.provider_id,
            resource_id=handle.provider_resource_id,
            ssh_host=handle.ssh_host,
            expires_at_unix=handle.expires_at_unix,
        )
        return handle

    def _launch_worker(self, host: TrainingHost, run: Run, paths: RunPaths) -> int:
        """Build the worker command + invoke via the run's transport."""
        client = self._client_for_host(host, paths, run)
        command = self._build_command(run, paths)
        # Host-identity placeholders (--user uid:gid, $HOME-derived mount
        # sources) resolve against the LAUNCHING host, not the GUI server —
        # remote users are not reliably uid 1000 (first Nebius smoke: 1001).
        uid, gid, home = client.host_identity()
        command = resolve_host_placeholders(command, uid, gid, home)
        # Pre-create every bind-mount source as the transport's user. A
        # missing source is auto-created by dockerd AS ROOT, after which a
        # non-root container can never write into it (same smoke, bug #1).
        for src in _bind_mount_sources(command):
            client.ensure_dir(src)
        env = self._build_env(run, paths)
        # For subprocess transport, workdir is the run dir (worker writes here).
        # For SSH (future), the workdir param becomes the remote per-run dir
        # (e.g. /workspace/runs/<run_id>); SshClient will translate. For now,
        # paths.root is the right thing to pass in either case.
        return client.launch(command=command, env=env, workdir=paths.root, log_path=paths.stderr_log)

    def _build_command(self, run: Run, paths: RunPaths) -> list[str]:
        """Compose the worker command via the recipe builder.

        Returns the full argv: for real training, that's the
        ``docker run … training-image lerobot-train …`` argv; for the
        test-only fake recipe (``__recipe__=__fake__``), a
        ``python <test fake_runner.py> …`` argv.
        """
        cmd, _ = build_lerobot_train_command(run, paths)
        return cmd

    def _build_env(self, run: Run, paths: RunPaths) -> dict[str, str]:
        """Env vars passed to the worker subprocess.

        Recipe-builder env (e.g., HF_HUB_OFFLINE for future scratch-staging)
        plus a few orchestrator-side breadcrumbs the existing fake runner
        consults.
        """
        _, recipe_env = build_lerobot_train_command(run, paths)
        return {
            "LEROBOT_RUN_ID": run.run_id,
            "LEROBOT_RUN_DIR": str(paths.root),
            **recipe_env,
        }

    def _reconcile_from_events_only(self, run: Run, paths: RunPaths) -> bool:
        """Cheap reconciliation path used by :meth:`list_runs` to keep the
        sidebar fresh without holding state on the orchestrator side.

        Returns True iff the state actually changed (so the caller can save).

        Two layers:
          1. If a terminal event has already been written to ``events.jsonl``
             (by the worker for fake recipes, or by a prior full reconcile
             for real recipes), advance the run state to match.
          2. If we still think the run is RUNNING/PENDING and the worker
             process is gone, escalate to the full :meth:`_reconcile_state`
             so that the terminal event gets written from the process exit
             code + checkpoint artifacts. Without this, a real-recipe run
             that completed while the user wasn't looking stays marked
             RUNNING in the sidebar indefinitely — the orchestrator only
             learns about the exit when the user opens the run's detail.

        The ``is_alive`` probe is the same shape as the one in the full
        reconcile path, so the cost is one extra ``waitpid(WNOHANG)`` /
        ``kill -0`` per active run per list_runs call. Cheap.
        """
        before = run.state
        host = self._hosts.get(run.host_id)
        # Pass `run`: a live ephemeral run must route to its SSH client. Without
        # it, _client_for_host falls back to a SubprocessClient that int()s the
        # SSH-format session_id and crashes — 500-ing GET /runs for the whole
        # run (the GUI shows "No runs yet" while training is actually going).
        client = self._client_for_host(host, paths, run)
        terminal_event = self._read_terminal_event(client, paths.events_jsonl)
        if terminal_event == "completed_naturally" and run.state != RunState.COMPLETED:
            run.advance(RunState.COMPLETED)
        elif terminal_event == "aborted_by_user" and run.state != RunState.ABORTED:
            run.advance(RunState.ABORTED)
        elif terminal_event == "crashed" and run.state != RunState.FAILED:
            run.advance(RunState.FAILED)
        elif (
            run.state in (RunState.RUNNING, RunState.COMPLETING)
            and run.session_id is not None
            and not client.is_alive(run.session_id)
        ):
            # Process died without writing a terminal event. Don't let the
            # sidebar lie — escalate to the full reconcile, which writes the
            # terminal event from exit code + checkpoints. Skipping for
            # PENDING because the prep thread owns that lifecycle and a
            # false "not alive" during prep (PID not yet set) would race.
            self._reconcile_state(run, paths, client)
        return run.state != before

    def _reconcile_state(self, run: Run, paths: RunPaths, client: TransportClient) -> None:
        """Update ``run.state`` based on (a) what the worker wrote to
        events.jsonl and (b) whether the process is still alive.

        DESIGN.md § Health "Completion signal" — for the fake-training
        recipe, the worker writes the terminal event itself. For the real
        (docker) recipe, the orchestrator writes the terminal event when it
        detects the process has exited (since lerobot-train doesn't write
        our events.jsonl format).

        For both recipes, the orchestrator also incrementally appends to
        ``checkpoints.jsonl`` whenever it sees a new checkpoint directory
        on disk (via the bind-mounted output dir for docker mode, or in
        the worker's local dir for the fake mode).
        """
        # New checkpoints discovered on disk → appended to manifest. Cheap
        # filesystem scan, idempotent on re-poll.
        self._sync_checkpoints_manifest(client, run, paths)

        terminal_event = self._read_terminal_event(client, paths.events_jsonl)
        alive = client.is_alive(run.session_id) if run.session_id is not None else False

        if terminal_event == "completed_naturally":
            if run.state != RunState.COMPLETED:
                run.advance(RunState.COMPLETED)
                self._runs.save(run)
            # Idempotent (skips files already local) — also covers the
            # GUI-restarted-after-completion case where the terminal event
            # exists but artifacts were never localized.
            self._fetch_run_artifacts(client, run, paths)
        elif terminal_event == "aborted_by_user":
            if run.state != RunState.ABORTED:
                run.advance(RunState.ABORTED)
                self._runs.save(run)
        elif terminal_event == "crashed":
            if run.state != RunState.FAILED:
                run.advance(RunState.FAILED)
                self._runs.save(run)
        elif not alive:
            # Process gone, no terminal event. For the real-training (docker)
            # recipe this is the EXPECTED path — lerobot-train doesn't write
            # our events.jsonl. For the fake recipe, this is a crash.
            self._write_terminal_event_from_exit(client, run, paths)

    def _write_terminal_event_from_exit(self, client: TransportClient, run: Run, paths: RunPaths) -> None:
        """Process exited without writing a terminal event. Decide whether
        to call it ``completed_naturally`` or ``crashed`` based on exit
        code first (authoritative when available), then artifacts.

        Order of evidence:
          1. Stop intent (``state==COMPLETING``) → ``aborted_by_user``.
          2. Exit code from the transport, if known (i.e. the worker is
             still our subprocess and we have the Popen). Non-zero =>
             ``crashed`` with stderr tail as error; zero => ``completed``.
          3. Fallback (post-GUI-restart, no Popen): checkpoint heuristic.

        ``final_step`` is derived from the latest checkpoint (max step
        across discovered dirs) when present; otherwise progress.json.
        The previous code trusted progress.json alone, which the real
        lerobot-train never writes, so completed runs reported step 0.

        Aborted runs (state==COMPLETING after a Stop) → ``aborted_by_user``.
        """
        if run.state == RunState.COMPLETING:
            self._emit_event(client, paths.events_jsonl, "aborted_by_user")
            run.advance(RunState.ABORTED)
            self._runs.save(run)
            return

        ckpt_steps = [step for _, step in self._iter_checkpoint_dirs(client, run, paths)]
        ckpt_count = len(ckpt_steps)
        progress = self._read_progress(client, paths.progress_json) or {}
        progress_step = progress.get("step", 0) if isinstance(progress, dict) else 0
        # Latest checkpoint step beats progress.json: lerobot-train doesn't
        # write progress.json, so for real runs the only signal is the
        # checkpoint dirs on disk. Fake runner writes progress.json; use
        # whichever is higher (handles either).
        final_step = max([progress_step, *ckpt_steps]) if ckpt_steps else progress_step

        # Authoritative when we can get it: the inner docker container's
        # exit code propagates through `docker run` to Popen.returncode.
        # A crash-after-success (e.g., the HF 403 in push_model_to_hub
        # AFTER successful training + checkpoints written) returns non-zero
        # even though checkpoints exist — so don't gate this on ckpt_count.
        code = None
        if run.session_id is not None:
            code = client.exit_code(run.session_id)

        if code is not None and code != 0:
            stderr_tail = self._read_stderr_tail(client, paths.stderr_log, 4096)
            run.error = f"exit code {code}\n{stderr_tail}".strip()
            self._emit_event(
                client,
                paths.events_jsonl,
                "crashed",
                error=f"exit code {code}",
                final_step=final_step,
            )
            run.advance(RunState.FAILED)
            self._runs.save(run)
            return

        if code == 0 or ckpt_count > 0:
            # Clean exit, or unknown-exit-code-but-has-checkpoints fallback.
            self._emit_event(client, paths.events_jsonl, "completed_naturally", final_step=final_step)
            run.advance(RunState.COMPLETED)
            self._fetch_run_artifacts(client, run, paths)
        else:
            run.error = "process exited without writing a checkpoint"
            self._emit_event(client, paths.events_jsonl, "crashed", error=run.error, final_step=final_step)
            run.advance(RunState.FAILED)
        self._runs.save(run)

    def _fetch_run_artifacts(self, client: TransportClient, run: Run, paths: RunPaths) -> None:
        """Localize the run's checkpoint files onto the GUI server.

        On SSH hosts the checkpoints live on the remote; without this, a
        completed run shows a manifest but the Models tab has nothing to
        load (found by the first remote-GPU smoke). Fetches, per
        checkpoint dir, the model file + its siblings (config.json,
        train_config.json — the pretrained_model dir is flat). Skips
        files that already exist locally, so re-reconciles are cheap and
        the subprocess transport (same path, fetch_file no-ops) is
        unaffected. training_state is deliberately NOT fetched —
        remote-resume is a separate workstream.

        Per-file failures log and continue: a partial fetch beats a
        completed run with zero local artifacts. A whole-fetch failure
        (e.g. list_dir crash, layout drift) emits ``artifacts_fetch_failed``
        so the user isn't staring at a silently-checkpoint-less run.
        """
        try:
            self._fetch_run_artifacts_inner(client, run, paths)
        except Exception as e:
            logger.warning("artifact fetch failed for run %s: %s", run.run_id, e)
            with contextlib.suppress(Exception):
                self._emit_event(client, paths.events_jsonl, "artifacts_fetch_failed", error=str(e)[:200])

    def _fetch_run_artifacts_inner(self, client: TransportClient, run: Run, paths: RunPaths) -> None:
        fetched = 0
        for ckpt_dir, _step in self._iter_checkpoint_dirs(client, run, paths):
            # Same nested/flat discovery as _sync_checkpoints_manifest.
            src_dir: Path | None = None
            for child in client.list_dir(ckpt_dir):
                if child.name == "pretrained_model":
                    src_dir = child
                    break
                if child.name == "model.safetensors":
                    src_dir = ckpt_dir
                    break
            if src_dir is None:
                continue
            for src in client.list_dir(src_dir):
                if src.name == "training_state":
                    continue
                dst = paths.root / src.relative_to(paths.root) if src.is_relative_to(paths.root) else None
                if dst is None:
                    # Remote layout mirrors the local run dir by construction
                    # (same RunPaths on both sides); a path outside it means
                    # the layout drifted — log loudly rather than guess.
                    logger.warning("artifact outside run dir, not fetching: %s", src)
                    continue
                remote_sha = client.sha256_of(src)
                # Already hold a byte-identical copy → skip. Otherwise (missing
                # OR corrupt-partial from a dropped scp) (re-)fetch + verify.
                if dst.exists() and remote_sha is not None and _sha256_file(dst) == remote_sha:
                    continue
                if self._fetch_verified(client, src, dst, remote_sha):
                    fetched += 1
        if fetched:
            self._emit_event(client, paths.events_jsonl, "artifacts_fetched", count=fetched)

    def _fetch_verified(self, client: TransportClient, src: Path, dst: Path, remote_sha: str | None) -> bool:
        """Fetch ``src``→``dst``, retrying on transfer failure or sha mismatch.

        A dropped scp leaves a partial file; the old skip-if-exists kept it
        forever (silently-corrupt model, found in the round-5 smoke). Here we
        overwrite and re-verify against the remote sha256, and remove the file
        if we can't get a good copy so the run has no (rather than a broken)
        artifact. Returns True iff a verified copy now exists locally.
        """
        for attempt in range(_ARTIFACT_FETCH_RETRIES):
            try:
                client.fetch_file(src, dst)
            except Exception as e:
                logger.warning(
                    "fetch %s attempt %d/%d failed: %s", src, attempt + 1, _ARTIFACT_FETCH_RETRIES, e
                )
                continue
            if remote_sha is None:
                # Couldn't read the remote sha (e.g. file vanished after listing)
                # — keep the fetched bytes but say so; we can't byte-verify it.
                logger.warning("fetched %s but could not verify it (remote sha unavailable)", src)
                return True
            if _sha256_file(dst) == remote_sha:
                return True
            logger.warning(
                "fetch %s attempt %d/%d: sha mismatch, re-fetching", src, attempt + 1, _ARTIFACT_FETCH_RETRIES
            )
        logger.warning(
            "artifact NOT localized (failed/corrupt after %d tries): %s", _ARTIFACT_FETCH_RETRIES, src
        )
        with contextlib.suppress(Exception):
            if dst.exists():
                dst.unlink()  # safe-destruct: remove the corrupt/partial scp output
        return False

    def _sync_checkpoints_manifest(self, client: TransportClient, run: Run, paths: RunPaths) -> None:
        """Append newly-discovered checkpoint dirs to ``checkpoints.jsonl``.

        Idempotent — skips dirs already in the manifest. Called on every
        poll, so the manifest catches up incrementally during a long run.
        Routes every file op via the transport so the same code works for
        local (subprocess) and remote (ssh) hosts.
        """
        already_seen_steps = {e.step for e in self._read_manifest(client, paths.checkpoints_jsonl)}
        for ckpt_dir, step in self._iter_checkpoint_dirs(client, run, paths):
            if step in already_seen_steps:
                continue
            # Locate the model file in the dir. Real lerobot-train layout
            # is ``<ckpt>/pretrained_model/model.safetensors``; fake-runner
            # writes ``<ckpt>/model.safetensors`` directly. Try both via
            # transport-routed listing.
            model_file: Path | None = None
            for child in client.list_dir(ckpt_dir):
                if child.name == "pretrained_model":
                    # nested layout — model.safetensors lives one level deeper
                    for grandchild in client.list_dir(child):
                        if grandchild.name == "model.safetensors":
                            model_file = grandchild
                            break
                    if model_file:
                        break
                elif child.name == "model.safetensors":
                    model_file = child
                    break
            if model_file is None:
                continue
            digest = client.sha256_of(model_file)
            if digest is None:
                continue
            rel_path = str(model_file.relative_to(paths.root))
            line = json.dumps({"step": step, "path": rel_path, "sha256": digest, "ts": time.time()})
            client.append_text(paths.checkpoints_jsonl, line + "\n")
            already_seen_steps.add(step)

    @staticmethod
    def _iter_checkpoint_dirs(client: TransportClient, run: Run, paths: RunPaths):
        """Yield ``(checkpoint_dir, step_number)`` pairs found on disk.

        Looks in the right place for the recipe:

        - lerobot-train: ``paths.root / "output" / "checkpoints" / <NNNNNN>/``
        - HVLA flow_s1:  ``paths.root / "output" / "checkpoints" / checkpoint-<N>/``
        - Fake:          ``paths.root / "checkpoints" / <NNNNNN>/``
        """
        subdir = output_subdir_in_run(run)
        ckpts_base = paths.root / subdir / "checkpoints" if subdir else paths.root / "checkpoints"
        # Parse first, then sort by step number — sorting directories by
        # name puts ``checkpoint-10`` before ``checkpoint-5`` (alphabetical)
        # which would write the manifest out of step order.
        pairs: list[tuple[Path, int]] = []
        for child in client.list_dir(ckpts_base):
            # Parse step number from the dir name. Two layouts supported:
            #   "000005"        → lerobot-train (zero-padded)
            #   "checkpoint-5"  → HVLA flow_matching trainer
            m = re.fullmatch(r"(?:checkpoint-)?0*(\d+)", child.name)
            if not m:
                continue
            pairs.append((child, int(m.group(1))))
        pairs.sort(key=lambda p: p[1])
        yield from pairs

    @staticmethod
    def _read_progress(client: TransportClient, progress_path: Path) -> dict[str, Any] | None:
        text = client.read_text(progress_path)
        if text is None:
            return None
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            return None

    def _ingest_training_log(self, client: TransportClient, paths: RunPaths) -> None:
        """Parse the host's stdout into position (progress.json) + the
        training-signal series (metrics.jsonl) — the one source of real
        progress/metrics for every backend. The training container just
        prints; structure is derived here on each poll.

        Pre: ``client`` can read ``paths.stderr_log`` on the training host.
        Post: if the log carried a tqdm bar, progress.json reflects the latest
        position (its ``updated_at`` only advances when ``step`` advances, so a
        hung run reads stale); every metric line is in metrics.jsonl. Writes
        nothing it didn't parse — a backend that writes progress.json itself
        (the test fake-runner) is never clobbered. Never raises.

        v1 re-reads + re-parses the whole log each poll: idempotent,
        restart-safe, cheap locally. Incremental offset reads
        (``read_bytes_from_offset``) for large / SSH logs are a follow-up.
        """
        try:
            text = client.read_text(paths.stderr_log)
        except Exception as e:  # noqa: BLE001 — a read failure must not break poll()
            logger.debug("ingest: could not read stderr for %s: %s", paths.run_id, e)
            return
        if not text:
            return

        latest: ProgressSample | None = None
        samples: list[dict[str, float]] = []
        for raw_line in text.splitlines():
            # tqdm overwrites in place with \r within a single line; the last
            # \r-segment is the freshest bar state.
            for seg in raw_line.split("\r"):
                seg = seg.strip()
                if not seg:
                    continue
                # Both parsers run on every segment: real lerobot glues the
                # metric log onto the end of the tqdm bar line, so one segment
                # carries both — an early continue dropped every metric.
                p = parse_progress(seg)
                if p is not None:
                    latest = p
                m = parse_metric_sample(seg)
                if m is not None:
                    # The metric line's own step is coarse (format_big_number:
                    # 1156 → "1K"); use the precise step from the bar instead.
                    bar = p or latest
                    if bar is not None:
                        m["step"] = float(bar.step)
                    samples.append(m)

        if latest is not None:
            step = latest.step
            # A metric line's step can be fresher than the tqdm bar's.
            if samples and samples[-1].get("step", 0) > step:
                step = int(samples[-1]["step"])
            prev = self._read_progress(client, paths.progress_json) or {}
            advanced = step > int(prev.get("step", -1))
            atomic_write_json(
                paths.progress_json,
                {
                    "step": step,
                    "total_steps": latest.total_steps,
                    "eta_seconds": latest.eta_seconds,
                    # Freshness signal for liveness: only bump when training
                    # actually progressed, so a stalled run reads stale.
                    "updated_at": time.time() if advanced else prev.get("updated_at", time.time()),
                },
            )

        if samples:
            # Rewrite, not append: a full reparse is idempotent, so this can't
            # double-count rows across polls or a GUI restart.
            body = "".join(json.dumps(s) + "\n" for s in samples)
            tmp = paths.metrics_jsonl.parent / (paths.metrics_jsonl.name + ".tmp")
            tmp.write_text(body)
            tmp.replace(paths.metrics_jsonl)

    @staticmethod
    def _read_metrics(metrics_path: Path) -> list[dict[str, float]]:
        """Read the metrics series (local file the orchestrator owns). Skips
        malformed rows rather than failing the whole poll."""
        if not metrics_path.exists():
            return []
        out: list[dict[str, float]] = []
        for line in metrics_path.read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                continue
        return out

    @staticmethod
    def _read_manifest(client: TransportClient, manifest_path: Path) -> list[CheckpointEntry]:
        text = client.read_text(manifest_path)
        if text is None:
            return []
        entries: list[CheckpointEntry] = []
        for line in text.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                d = json.loads(line)
                entries.append(
                    CheckpointEntry(
                        step=int(d["step"]),
                        path=str(d["path"]),
                        sha256=str(d["sha256"]),
                        ts=float(d["ts"]),
                    )
                )
            except (json.JSONDecodeError, KeyError, ValueError) as e:
                # A malformed line means a half-written append or manifest
                # corruption — skipping keeps the run usable, but silence
                # would hide systematic corruption. One log line per bad line.
                logger.warning("skipping malformed manifest line in %s: %s", manifest_path, e)
                continue
        return entries

    @staticmethod
    def _read_stderr_tail(client: TransportClient, stderr_path: Path, n_bytes: int) -> str:
        if n_bytes <= 0:
            return ""
        return client.read_tail(stderr_path, n_bytes).decode("utf-8", errors="replace")

    @staticmethod
    def _read_events(client: TransportClient, events_path: Path) -> list[dict[str, Any]]:
        """Read events.jsonl entries in order. Routes via the transport so
        SSH hosts work without bind-mounting events.jsonl back to the GUI
        server."""
        text = client.read_text(events_path)
        if text is None:
            return []
        out: list[dict[str, Any]] = []
        for line in text.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                continue
        return out

    @staticmethod
    def _read_terminal_event(client: TransportClient, events_path: Path) -> str | None:
        """Scan events.jsonl for a terminal event type. Returns the type name
        or None. Terminal events: completed_naturally / aborted_by_user / crashed.
        """
        text = client.read_text(events_path)
        if text is None:
            return None
        terminal = {"completed_naturally", "aborted_by_user", "crashed"}
        for line in text.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                evt = json.loads(line)
            except json.JSONDecodeError:
                continue
            if evt.get("type") in terminal:
                return evt["type"]
        return None


# ── Module helpers ────────────────────────────────────────────────────────────


def _drop_run_metadata(paths: RunPaths) -> tuple[int, bool]:
    """Remove a run's metadata files (run.json, events.jsonl,
    checkpoints.jsonl, stderr.log, progress.json) but PRESERVE the
    ``output/checkpoints/`` subtree if it exists. That subtree is the
    trained model artefact; the Models tab continues to surface it after
    the run is dropped from history.

    If no model was ever written (no ``output/`` or it's empty of
    safetensors), the run dir is wholly removed — there's nothing to
    preserve.

    Returns ``(bytes_freed_metadata, kept_model)``. ``bytes_freed_metadata``
    counts only what we actually deleted (tens to hundreds of KB for the
    metadata case; full dir size for the no-model case). ``kept_model`` is
    True iff the model artefacts remain on disk.

    Idempotent: a second call on a run that's already been dropped (no
    metadata files present, only checkpoints) returns ``(0, True)``.
    """
    if not paths.root.exists():
        return 0, False
    metadata_files = [
        paths.run_json,
        paths.events_jsonl,
        paths.checkpoints_jsonl,
        paths.stderr_log,
        paths.progress_json,
    ]
    # Has-model probe — look for ANY .safetensors anywhere under
    # output/checkpoints/. If none, this is a no-model run (failed early
    # or aborted before the first save) and the whole dir is dead weight.
    ckpt_dir = (
        paths.checkpoints_dir
        if not (paths.root / "output").exists()
        else paths.root / "output" / "checkpoints"
    )
    has_model = ckpt_dir.is_dir() and any(ckpt_dir.rglob("*.safetensors"))

    if not has_model:
        # Nothing worth preserving. Tally + nuke. Note: callers gate this
        # behind a TERMINAL state check, so no worker is writing.
        total = 0
        for p in paths.root.rglob("*"):
            try:
                if p.is_file() and not p.is_symlink():
                    total += p.stat().st_size
            except OSError:
                continue
        # safe-destruct: orchestrator-owned <runs_dir>/<run_id>/ in terminal state with no model artefact
        shutil.rmtree(paths.root, ignore_errors=False)
        return total, False

    # Has a model → drop ONLY the metadata files. Checkpoints stay.
    freed = 0
    for f in metadata_files:
        if f.exists() and f.is_file():
            with contextlib.suppress(OSError):
                freed += f.stat().st_size
            with contextlib.suppress(OSError):
                # safe-destruct: orchestrator-owned metadata file in terminal-state run dir
                f.unlink()
    return freed, True


# ── Image preparation (pre-pull + cache check) ────────────────────────────────


class _ImagePullError(RuntimeError):
    """Raised internally when ``docker pull`` exits non-zero. Caught by
    :meth:`Orchestrator._prepare_and_launch` to flip the run to FAILED."""


def _handle_to_dict(handle: HostHandle) -> dict[str, Any]:
    """Serialize a HostHandle for persistence in run.json."""
    from dataclasses import asdict

    return asdict(handle)


def _handle_from_dict(d: dict[str, Any]) -> HostHandle:
    """Rebuild a HostHandle from its persisted dict."""
    return HostHandle(**d)


def _bind_mount_sources(cmd: list[str]) -> list[Path]:
    """Host-side source dirs of every ``-v src:dst`` pair in a docker argv.

    Empty for non-docker commands (fake recipe). Used to pre-create mount
    sources before launch — dockerd auto-creates missing sources as root,
    which a ``--user``-constrained container can then never write into.
    """
    if not cmd or cmd[0] != "docker":
        return []
    out: list[Path] = []
    for i, arg in enumerate(cmd):
        if arg == "-v" and i + 1 < len(cmd):
            src = cmd[i + 1].split(":", 1)[0]
            if src.startswith("/"):
                out.append(Path(src))
    return out


def _extract_image_from_docker_argv(cmd: list[str]) -> str | None:
    """Pick the image tag out of a ``docker run ...`` argv.

    Returns ``None`` if ``cmd`` isn't a docker invocation (e.g., the fake
    recipe's ``python <script> ...``). Recognising the docker recipe shape is
    cheap; full POSIX-style flag parsing would over-fit a structure we
    control.

    Convention from :func:`training_recipes.build_lerobot_train_command`:
    the image is the first positional after the last ``-v`` mount pair, and
    is followed by the entrypoint (``lerobot-train`` or
    ``python -u -m ...``). We just look for the first token after a
    sequence of recognised docker-flag pairs.
    """
    if not cmd or cmd[0] != "docker" or len(cmd) < 3 or cmd[1] != "run":
        return None
    i = 2
    # Skip flag pairs and standalone flags until we hit the image.
    # Recognised: --rm, --gpus all, --user UID:GID, -v X:Y, --network host, etc.
    while i < len(cmd):
        tok = cmd[i]
        if tok == "--rm":
            i += 1
            continue
        if tok in {
            "--gpus",
            "--user",
            "--network",
            "-v",
            "--volume",
            "-e",
            "--env",
            "--name",
            "--ipc",
            "--shm-size",
        }:
            i += 2
            continue
        if tok.startswith("-"):
            # Unknown flag — bail rather than guess pair vs. standalone.
            return None
        # First non-flag positional: this is the image.
        return tok
    return None

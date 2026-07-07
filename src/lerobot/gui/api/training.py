# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""Training API: orchestrator-backed run CRUD + hosts listing.

Endpoints, all under ``/api/training``:

- ``GET /hosts``                — list training hosts (workstation auto-detected)
- ``GET /runs``                 — list all runs (newest first)
- ``POST /runs``                — start a new run
- ``GET /runs/{run_id}``        — snapshot for one run (state, progress, checkpoints, log tail)
- ``POST /runs/{run_id}/stop``  — user-initiated stop

The router is constructed lazily — the orchestrator + host registry are
injected via a module-level state object so server.py can wire them at startup
with the right runs_dir. See :func:`init_state` and :func:`get_state`.
"""

from __future__ import annotations

import contextlib
import dataclasses
import importlib
import pkgutil
import typing
from dataclasses import asdict
from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from lerobot.gui.training.hosts import WORKSTATION_HOST_ID, HostRegistry, profile_to_training_host
from lerobot.gui.training.jobs import HOSTS_DIR, HostProfile
from lerobot.gui.training.orchestrator import (
    HostBusyError,
    Orchestrator,
    RunNotTerminalError,
    RunSnapshot,
    StartRequest,
    UnknownHostError,
    UnknownRunError,
)
from lerobot.gui.training.probe import probe_ssh
from lerobot.gui.training.recipes import HVLA_FLOW_S1_FIELD_TO_FLAG, HVLA_FLOW_S1_RECIPE
from lerobot.gui.training.runs import RUNS_DIR, RunRegistry

router = APIRouter(prefix="/api/training", tags=["training"])


# ── Lazy state ────────────────────────────────────────────────────────────────


# Held in module scope so server.py can call init_state() once at startup
# and the route handlers can read it. Keeps this module importable for tests
# without a full app context.
_state: dict[str, Any] = {"orch": None, "host_registry": None}


def init_state(orch: Orchestrator, host_registry: HostRegistry) -> None:
    """Wire the orchestrator + host registry the routes will use.

    Called once at GUI server startup. Tests can call it with a custom
    orchestrator + registry without spinning up the full app.
    """
    _state["orch"] = orch
    _state["host_registry"] = host_registry


def get_state() -> tuple[Orchestrator, HostRegistry]:
    """Return the wired-up orchestrator + host registry.

    Raises if :func:`init_state` was never called.
    """
    orch = _state["orch"]
    host_registry = _state["host_registry"]
    if orch is None or host_registry is None:
        raise RuntimeError("Training API state not initialized; call init_state() at server startup.")
    return orch, host_registry


def reset_state_for_testing() -> None:
    """Test helper: reset the module-level state between fixtures."""
    _state["orch"] = None
    _state["host_registry"] = None


def make_default_orchestrator(runs_dir: Path | None = None) -> Orchestrator:
    """Build a stock Orchestrator + auto-detected HostRegistry pair.

    Convenience for server startup. Pass ``runs_dir`` to override the default
    (``~/.cache/lerobot/runs``) — tests use a tmp dir.
    """
    runs_dir = runs_dir or RUNS_DIR
    host_registry = HostRegistry.auto(workdir=runs_dir, hosts_dir=HOSTS_DIR)
    run_registry = RunRegistry(runs_dir=runs_dir)
    return Orchestrator(host_registry=host_registry, run_registry=run_registry)


# ── Schemas ───────────────────────────────────────────────────────────────────


class HostInfo(BaseModel):
    id: str
    display_name: str
    transport_kind: str  # "subprocess" or "ssh"
    capabilities: dict[str, Any]


# Body for POST /hosts. The ``host`` field is the raw SSH spec the user
# typed — alias, ``user@host``, or ``user@host:port``. We parse out the
# components on the way in so the saved HostProfile has the same shape
# whether it was added via the dialog or hand-edited under
# ~/.config/lerobot/training_hosts/.
class HostProfileBody(BaseModel):
    name: str = Field(min_length=1, max_length=64, pattern=r"^[a-zA-Z0-9_-]+$")
    # SSH host string ("user@host[:port]" or ~/.ssh/config alias). Required
    # for a persistent SSH host; omitted for an Ephemeral cloud host.
    host: str | None = Field(default=None, max_length=256)
    display_name: str | None = None
    workdir: str = "/workspace/lerobot"
    image_ref: str | None = None  # falls back to HostProfile dataclass default
    # ── Ephemeral cloud host (set provider_id to select this path) ────────
    provider_id: str | None = None
    gpu: str = "L40S"
    gpu_count: int = Field(default=1, ge=1)
    disk_gib: int = Field(default=100, ge=1)
    preemptible: bool = True
    region_hint: str | None = None
    ttl_hours: int = Field(default=24, ge=1)


class HostProbeBody(BaseModel):
    host: str = Field(min_length=1, max_length=256)


class ProbeCheckDTO(BaseModel):
    name: str
    ok: bool
    detail: str


class ProbeResultDTO(BaseModel):
    ok: bool
    latency_ms: int
    checks: list[ProbeCheckDTO]
    error_class: str | None = None
    message: str | None = None


class StartRunBody(BaseModel):
    host_id: str
    recipe_name: str = Field(min_length=1)
    dataset_id: str = Field(min_length=1)
    args: dict[str, Any] = Field(default_factory=dict)
    idempotency_key: str | None = None


class RunDTO(BaseModel):
    run_id: str
    host_id: str
    recipe_name: str
    dataset_id: str
    args: dict[str, Any]
    state: str
    created_at: float
    started_at: float | None
    finished_at: float | None
    session_id: str | None
    error: str | None


class CheckpointDTO(BaseModel):
    step: int
    path: str
    sha256: str
    ts: float


class RunSnapshotDTO(BaseModel):
    run: RunDTO
    progress: dict[str, Any] | None
    checkpoints: list[CheckpointDTO]
    stderr_tail: str
    # Raw events.jsonl entries (oldest first). Frontend filters for
    # image-prep events to render the "Pulling image…" status banner.
    events: list[dict[str, Any]]
    # Training-signal series parsed from stdout (metrics.jsonl): one row per
    # logged step, each an auto-captured {key: value} bag (loss/lr/grdn/…).
    # The dashboard charts these; distinct from `progress` (position).
    metrics: list[dict[str, float]] = []


# ── Helpers ───────────────────────────────────────────────────────────────────


def _run_to_dto(run) -> RunDTO:
    d = asdict(run)
    d["state"] = run.state.value
    # Drop fields that aren't part of the API surface
    d.pop("idempotency_key", None)
    return RunDTO(**d)


def _snapshot_to_dto(snap: RunSnapshot) -> RunSnapshotDTO:
    return RunSnapshotDTO(
        run=_run_to_dto(snap.run),
        progress=snap.progress,
        checkpoints=[
            CheckpointDTO(step=c.step, path=c.path, sha256=c.sha256, ts=c.ts) for c in snap.checkpoints
        ],
        stderr_tail=snap.stderr_tail,
        events=list(snap.events),
        metrics=list(snap.metrics),
    )


def _transport_kind(transport: Any) -> str:
    if transport is None:
        return "ephemeral"  # transport-less host = provider-spawned VM
    cls = transport.__class__.__name__
    if cls == "SubprocessTransport":
        return "subprocess"
    if cls == "SshTransport":
        return "ssh"
    return cls


def _host_info(h: Any) -> HostInfo:
    """Build the HostInfo DTO from a TrainingHost, surfacing the spawn spec
    (provider + GPU) for Ephemeral hosts so the UI can show what it'll spawn."""
    caps = dict(h.capabilities)
    if getattr(h, "is_ephemeral", False):
        spec = h.spawn_spec
        caps = {
            **caps,
            "provider_id": h.provider_id,
            "gpu_name": spec.gpu,
            "gpu_count_detected": spec.gpu_count,
            "ttl_hours": spec.ttl_seconds // 3600,
        }
    return HostInfo(
        id=h.id,
        display_name=h.display_name,
        transport_kind=_transport_kind(h.transport),
        capabilities=caps,
    )


# ── Endpoints ─────────────────────────────────────────────────────────────────


@router.get("/hosts", response_model=list[HostInfo])
def list_hosts() -> list[HostInfo]:
    """List all training hosts. v1: workstation host if a GPU is auto-detected."""
    _, hosts = get_state()
    return [_host_info(h) for h in hosts.list_hosts()]


def _parse_host_spec(host_spec: str) -> tuple[str, str, int]:
    """Parse the user-typed host string into ``(user, host, port)``.

    Accepts:
      - ``alias`` or ``host``            → user="root", port=22
      - ``user@host``                    → user from prefix
      - ``host:port`` / ``user@host:port`` → port from suffix
    """
    raw = host_spec.strip()
    user = "root"
    port = 22
    if "@" in raw:
        user, raw = raw.split("@", 1)
    # Only treat ``:N`` as a port when there's exactly one colon and N is
    # numeric. Bare IPv6 ("2001:db8::1") has multiple colons — passing it
    # through unsplit beats silently misparsing "…:1" as host+port; ssh
    # itself accepts bare IPv6 as a destination.
    if raw.count(":") == 1:
        host_part, _, port_part = raw.rpartition(":")
        if port_part.isdigit():
            raw = host_part
            port = int(port_part)
    return user, raw, port


@router.post("/hosts", response_model=HostInfo, status_code=201)
def add_host(body: HostProfileBody) -> HostInfo:
    """Save a new SSH host and register it in the live registry.

    Returns 409 if a host with the same name already exists. Persistence
    is to ``~/.config/lerobot/training_hosts/<name>.json`` — the user can
    edit it by hand or remove via DELETE.
    """
    _, registry = get_state()
    if body.name == WORKSTATION_HOST_ID:
        raise HTTPException(
            status_code=400, detail=f"name {WORKSTATION_HOST_ID!r} is reserved for the workstation host"
        )
    if registry.get(body.name) is not None:
        raise HTTPException(status_code=409, detail=f"host {body.name!r} already exists")

    extra: dict[str, Any] = {}
    if body.image_ref:
        extra["image_ref"] = body.image_ref
    if body.provider_id is not None:
        # Ephemeral cloud host: no SSH endpoint yet — the VM is spawned on
        # first run. Persist the spawn spec instead.
        profile = HostProfile(
            name=body.name,
            kind="temporary",
            display_name=body.display_name or body.name,
            workdir=body.workdir,
            provider_id=body.provider_id,
            gpu=body.gpu,
            gpu_count=body.gpu_count,
            disk_gib=body.disk_gib,
            preemptible=body.preemptible,
            region_hint=body.region_hint,
            ttl_hours=body.ttl_hours,
            **extra,
        )
    else:
        if not body.host:
            raise HTTPException(status_code=422, detail="host is required for a persistent SSH host")
        user, host, port = _parse_host_spec(body.host)
        profile = HostProfile(
            name=body.name,
            ssh_user=user,
            ssh_host=host,
            ssh_port=port,
            kind="permanent",
            display_name=body.display_name or body.name,
            workdir=body.workdir,
            **extra,
        )
    # Register first, persist second: registry.add() is where the collision
    # assert lives, so a concurrent duplicate POST fails BEFORE the file is
    # written — no orphaned profile on disk for a GUI restart to resurrect.
    th = profile_to_training_host(profile)
    registry.add(th)
    profile.save(dir_=HOSTS_DIR)
    return _host_info(th)


@router.delete("/hosts/{host_id}", status_code=204)
def delete_host(host_id: str) -> None:
    """Remove a saved SSH host from the registry + disk.

    Refuses to delete the auto-detected workstation entry (400) and
    refuses while a run on this host is still active (409). Deliberately
    NOT idempotent: deleting a host that doesn't exist returns 404 so
    the caller can distinguish "already gone" from "never existed".
    """
    orch, registry = get_state()
    if host_id == WORKSTATION_HOST_ID:
        raise HTTPException(
            status_code=400, detail="the workstation host is auto-detected and cannot be removed"
        )
    if registry.get(host_id) is None:
        raise HTTPException(status_code=404, detail=f"unknown host id: {host_id!r}")
    busy = orch._runs.active_run_on_host(host_id)  # noqa: SLF001 — read-only orchestrator query
    if busy is not None:
        raise HTTPException(
            status_code=409,
            detail=f"host {host_id!r} has active run {busy.run_id!r} (state={busy.state.value}); stop it first",
        )
    registry.remove(host_id)
    HostProfile.delete(host_id, dir_=HOSTS_DIR)


@router.post("/hosts/probe", response_model=ProbeResultDTO)
async def probe_host(body: HostProbeBody) -> ProbeResultDTO:
    """Run a short SSH check against the host string. Stateless — does
    not save anything. Front-end calls this to power the "Test" button
    in the Add SSH host dialog before letting the user click Save."""
    result = await probe_ssh(body.host)
    return ProbeResultDTO(
        ok=result.ok,
        latency_ms=result.latency_ms,
        checks=[ProbeCheckDTO(name=c.name, ok=c.ok, detail=c.detail) for c in result.checks],
        error_class=result.error_class,
        message=result.message,
    )


@router.get("/runs", response_model=list[RunDTO])
def list_runs() -> list[RunDTO]:
    """List all runs, newest first."""
    orch, _ = get_state()
    return [_run_to_dto(r) for r in orch.list_runs()]


@router.post("/runs", response_model=RunDTO, status_code=201)
def start_run(body: StartRunBody) -> RunDTO:
    """Start a training run.

    Returns 409 if the target host already has an active run.
    Returns 404 if the host id isn't registered.

    Ephemeral (cloud) hosts authenticate via the server-held Nebius
    service-account connection (see the ``/nebius/connection`` endpoints);
    no per-request credential is passed.
    """
    orch, _ = get_state()
    try:
        run = orch.start(
            StartRequest(
                host_id=body.host_id,
                recipe_name=body.recipe_name,
                dataset_id=body.dataset_id,
                args=body.args,
                idempotency_key=body.idempotency_key,
            )
        )
    except UnknownHostError as e:
        raise HTTPException(status_code=404, detail=str(e)) from e
    except HostBusyError as e:
        raise HTTPException(status_code=409, detail=str(e)) from e
    return _run_to_dto(run)


@router.get("/runs/{run_id}", response_model=RunSnapshotDTO)
def get_run(run_id: str) -> RunSnapshotDTO:
    """Snapshot one run: state, progress.json, checkpoints manifest, stderr tail.

    A background ephemeral destroy driven by this poll authenticates with
    the server-held Nebius service-account key.
    """
    orch, _ = get_state()
    try:
        snap = orch.poll(run_id)
    except UnknownRunError as e:
        raise HTTPException(status_code=404, detail=str(e)) from e
    return _snapshot_to_dto(snap)


@router.post("/runs/{run_id}/stop", response_model=RunDTO)
def stop_run(run_id: str) -> RunDTO:
    """User-initiated stop. Idempotent on already-terminal runs."""
    orch, _ = get_state()
    try:
        run = orch.stop(run_id)
    except UnknownRunError as e:
        raise HTTPException(status_code=404, detail=str(e)) from e
    return _run_to_dto(run)


class DeleteRunResponse(BaseModel):
    run_id: str
    # Metadata-only delete; ``kept_model`` says whether the checkpoint
    # directory survived (= Models tab still has the artefact).
    metadata_bytes_freed: int
    kept_model: bool


class ClearTerminalResponse(BaseModel):
    deleted: list[str]
    metadata_bytes_freed: int
    models_kept: int


@router.delete("/runs/{run_id}", response_model=DeleteRunResponse)
def delete_run(run_id: str) -> DeleteRunResponse:
    """Drop a terminal-state run from training history.

    Metadata-only delete: removes the run row from the Training list but
    keeps ``output/checkpoints/`` so the trained model continues to show
    up in the Models tab. If the run produced no model (failed pull,
    aborted before first save), the whole dir is removed.

    Returns 409 if the run is still active (caller must Stop first).
    Returns 404 if the run id is unknown.
    """
    orch, _ = get_state()
    try:
        result = orch.delete_run(run_id)
    except UnknownRunError as e:
        raise HTTPException(status_code=404, detail=str(e)) from e
    except RunNotTerminalError as e:
        raise HTTPException(status_code=409, detail=str(e)) from e
    return DeleteRunResponse(**result)


@router.post("/runs/clear", response_model=ClearTerminalResponse)
def clear_terminal_runs() -> ClearTerminalResponse:
    """Drop every terminal-state run from training history in one shot.

    Metadata-only per row (see :func:`delete_run`). Active runs are
    skipped. Idempotent: a second call after the first returns
    ``deleted=[]``.
    """
    orch, _ = get_state()
    result = orch.clear_terminal_runs()
    return ClearTerminalResponse(**result)


# ── Nebius connection (server-held service-account credential) ────────────────
#
# One Nebius service-account key for the whole GUI server, configured once by
# whoever operates the deployment (a Nebius tenant admin creates the SA; the
# operator pastes its authorized-key JSON + project/subnet here). The key is
# server-held and therefore shared by anyone who can reach the GUI — same trust
# model as the ambient HF token / SSH key (documented in DESIGN.md). The
# private key is never returned; status only echoes the SA's own identifiers.


class NebiusConnectionDTO(BaseModel):
    configured: bool
    has_key: bool
    service_account_id: str | None = None
    key_id: str | None = None
    project_id: str | None = None
    subnet_id: str | None = None


class NebiusConnectionBody(BaseModel):
    # Two ways to supply the service-account key (both validated server-side):
    #  - CLI:     key_json — the full file from `auth-public-key generate`.
    #  - console: private_key (PEM) + key_id + service_account_id — the pieces
    #             the console gives you; assembled into the same JSON here.
    key_json: str | None = None
    private_key: str | None = None
    key_id: str | None = None
    service_account_id: str | None = None
    project_id: str = Field(min_length=1)
    subnet_id: str = Field(min_length=1)


def _connection_dto(status: Any) -> NebiusConnectionDTO:
    return NebiusConnectionDTO(
        configured=status.configured,
        has_key=status.has_key,
        service_account_id=status.service_account_id,
        key_id=status.key_id,
        project_id=status.project_id,
        subnet_id=status.subnet_id,
    )


@router.get("/nebius/connection", response_model=NebiusConnectionDTO)
def get_nebius_connection() -> NebiusConnectionDTO:
    """Non-secret status of the server-held Nebius connection.

    Never returns the private key; ``configured`` is True only when the key
    AND project/subnet are all present.
    """
    from lerobot.gui.training.nebius_credentials import NebiusConnectionStore

    return _connection_dto(NebiusConnectionStore().status())


@router.put("/nebius/connection", response_model=NebiusConnectionDTO)
def set_nebius_connection(body: NebiusConnectionBody) -> NebiusConnectionDTO:
    """Store (or replace) the Nebius service-account key + project/subnet.

    Returns 400 if the pasted key is malformed. The key is written ``0600``
    and used for every ephemeral spawn/teardown thereafter.
    """
    from lerobot.gui.training.nebius_credentials import (
        NebiusConnectionStore,
        NebiusCredentialError,
        assemble_authorized_key_json,
    )

    if body.key_json and body.key_json.strip():
        key_json = body.key_json
    elif body.private_key and body.key_id and body.service_account_id:
        key_json = assemble_authorized_key_json(
            private_key=body.private_key,
            key_id=body.key_id,
            service_account_id=body.service_account_id,
        )
    else:
        raise HTTPException(
            status_code=400,
            detail=(
                "Provide either the key file JSON (from the CLI), or the private key "
                "+ authorized key ID + service account ID (from the console)."
            ),
        )

    try:
        status = NebiusConnectionStore().set(
            key_json=key_json, project_id=body.project_id, subnet_id=body.subnet_id
        )
    except NebiusCredentialError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    return _connection_dto(status)


class ClearNebiusConnectionResponse(BaseModel):
    cleared: bool


@router.delete("/nebius/connection", response_model=ClearNebiusConnectionResponse)
def clear_nebius_connection() -> ClearNebiusConnectionResponse:
    """Remove the stored Nebius key + connection. Idempotent."""
    from lerobot.gui.training.nebius_credentials import NebiusConnectionStore

    return ClearNebiusConnectionResponse(cleared=NebiusConnectionStore().clear())


class NebiusDiscoverBody(BaseModel):
    # Same key shapes as the connection PUT; project_id is required to scope
    # the subnet listing. Not persisted — used only to query Nebius.
    key_json: str | None = None
    private_key: str | None = None
    key_id: str | None = None
    service_account_id: str | None = None
    project_id: str = Field(min_length=1)


class NebiusSubnetDTO(BaseModel):
    id: str
    name: str


@router.post("/nebius/discover/subnets", response_model=list[NebiusSubnetDTO])
def discover_nebius_subnets(body: NebiusDiscoverBody) -> list[NebiusSubnetDTO]:
    """List a project's VPC subnets from a not-yet-saved key, so the connection
    form can offer a picker instead of a hand-copied ID. The key is written to
    a ``0600`` temp file for the SDK and deleted immediately after; nothing is
    persisted. 400 on a bad key or if the service account can't list subnets
    (the form keeps a manual-entry fallback)."""
    import contextlib
    import os
    import tempfile

    from lerobot.gui.training.nebius_credentials import assemble_authorized_key_json
    from lerobot.gui.training.providers.nebius import (
        NebiusAuthError,
        NebiusConfigError,
        NebiusProvider,
    )

    if body.key_json and body.key_json.strip():
        key_json = body.key_json
    elif body.private_key and body.key_id and body.service_account_id:
        key_json = assemble_authorized_key_json(
            private_key=body.private_key, key_id=body.key_id, service_account_id=body.service_account_id
        )
    else:
        raise HTTPException(
            status_code=400, detail="Provide the key (file JSON, or private key + IDs) and a project ID."
        )

    fd, tmp = tempfile.mkstemp(prefix="nebius-discover-", suffix=".json")
    try:
        os.fchmod(fd, 0o600)
        with os.fdopen(fd, "w") as f:
            f.write(key_json)
        subnets = NebiusProvider(credentials_file=tmp, project_id=body.project_id).list_subnets(
            body.project_id
        )
    except (NebiusAuthError, NebiusConfigError) as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    except Exception as e:  # noqa: BLE001 — surface SDK / permission errors to the form
        raise HTTPException(status_code=400, detail=f"Couldn't list subnets: {e}") from e
    finally:
        with contextlib.suppress(FileNotFoundError):
            os.unlink(tmp)  # safe-destruct: our own mkstemp temp key file
    return [NebiusSubnetDTO(id=s["id"], name=s["name"]) for s in subnets]


# ── Policy catalog (GET /api/training/policies) ───────────────────────────────
#
# Dynamically discovers every PreTrainedConfig subclass via
# ``PreTrainedConfig.get_known_choices()`` and introspects its dataclass
# fields. Mirrors the same pattern used by api/robot.py's `_introspect_fields`
# + `/api/robot/schemas` — so what works for robot/teleop configs also
# works for policy configs without rebuilding from scratch.
#
# Manually-registered (non-draccus) recipes — currently just HVLA — are
# spliced in alongside the auto-discovered draccus policies so the
# frontend sees one uniform catalog.


_policies_loaded = False


def _ensure_policy_configs_loaded() -> None:
    """Import every submodule under ``lerobot.policies`` so each
    ``@PreTrainedConfig.register_subclass(...)`` decorator runs.

    Mirrors :func:`api.robot._ensure_configs_loaded` — auto-discovery via
    ``pkgutil.walk_packages`` so a new policy package picked up at runtime
    is exposed in the catalog without a code edit. Each import is wrapped
    in ``contextlib.suppress(Exception)`` so policies whose optional deps
    aren't installed (e.g. ``smolvla`` without ``transformers``) silently
    drop out instead of breaking discovery for the rest.
    """
    global _policies_loaded
    if _policies_loaded:
        return
    import lerobot.policies  # noqa: PLC0415

    for _importer, modname, _ispkg in pkgutil.walk_packages(
        lerobot.policies.__path__, prefix=lerobot.policies.__name__ + "."
    ):
        with contextlib.suppress(Exception):
            importlib.import_module(modname)
    _policies_loaded = True


# Field type classification for the frontend form renderer. Anything not
# in here is dropped from the schema — the form can't render a `list[str]`
# or a nested dataclass usefully, and falling through to a free-text input
# silently misleads the user. If you need a complex-typed field, hit
# /api/training/runs directly with the args dict.
_FORM_KIND_FROM_PY = {
    int: "int",
    float: "float",
    bool: "bool",
    str: "string",
}


def _classify_field(annotation: Any) -> tuple[str | None, list[str] | None]:
    """Map a type annotation to ``(form_kind, choices)`` for the frontend.

    Returns ``(None, None)`` if the field type isn't renderable in a form
    (lists, dicts, nested dataclasses, callables, ...). Handles
    ``Optional[X]`` by unwrapping. Handles ``Literal["a", "b"]`` by
    returning ``("select", ["a", "b"])``.
    """
    origin = typing.get_origin(annotation)
    args = typing.get_args(annotation)

    # Optional[X] → Union[X, None]: unwrap and recurse on X.
    if origin is typing.Union:
        non_none = [a for a in args if a is not type(None)]
        if len(non_none) == 1:
            return _classify_field(non_none[0])
        return None, None

    if origin is typing.Literal:
        return "select", [str(v) for v in args]

    if annotation in _FORM_KIND_FROM_PY:
        return _FORM_KIND_FROM_PY[annotation], None

    return None, None


# Fields the recipe builder force-overrides regardless of user input.
# Surfacing them in the form misleads (the user thinks they can pick a
# value, but the recipe drops it) and — worse for ``push_to_hub`` — the
# dataclass default of ``True`` for some policies (SmolVLA, etc.) used
# to leak through the recipe's "user wins" branch and trigger an HF 403
# post-training. Always exclude.
_RECIPE_FORCED_FIELDS = {"push_to_hub", "repo_id", "output_dir"}


def _introspect_policy_fields(cls: type) -> list[dict]:
    """Extract a frontend-renderable schema from a dataclass config class.

    Skips fields whose type isn't a scalar (int/float/bool/str) or a
    ``Literal[...]``. Skips fields without a default (we can't pre-fill
    the form and don't want to require them blindly). Skips fields the
    recipe builder force-overrides (see :data:`_RECIPE_FORCED_FIELDS`).
    Sorts: required-ish first, then alphabetical name.
    """
    try:
        type_hints = typing.get_type_hints(cls)
    except Exception:
        type_hints = {}

    out: list[dict] = []
    for f in dataclasses.fields(cls):
        if f.name in _RECIPE_FORCED_FIELDS:
            continue
        resolved = type_hints.get(f.name, f.type)
        form_kind, choices = _classify_field(resolved)
        if form_kind is None:
            continue  # not renderable in a form

        # Resolve default: literal default beats factory beats None.
        default: Any = None
        if f.default is not dataclasses.MISSING:
            default = f.default
        elif f.default_factory is not dataclasses.MISSING:  # type: ignore[misc]
            try:
                default = f.default_factory()  # type: ignore[misc]
            except Exception:
                default = None

        entry: dict[str, Any] = {
            "name": f.name,
            "label": _humanize_field_name(f.name),
            "type": form_kind,
            "default": default,
        }
        if choices is not None:
            entry["choices"] = choices
        description = f.metadata.get("description")
        if description:
            entry["description"] = description
        out.append(entry)

    out.sort(key=lambda e: (e["default"] is None and e["type"] != "bool", e["name"]))
    return out


def _humanize_field_name(name: str) -> str:
    """Render a snake_case field name as a Title Case label."""
    return name.replace("_", " ").strip().capitalize()


def _humanize_policy_name(type_name: str) -> str:
    """Fallback label for a registered policy with no curated display name."""
    return type_name.replace("_", " ").upper()


# Curated display labels for the policies the GUI exposes by default. Any
# auto-discovered policy without an entry here gets the humanized fallback
# from :func:`_humanize_policy_name`. The point is to keep the *catalog*
# fully dynamic while letting the label remain hand-tuned where it
# matters (so "act" → "ACT (Action Chunking Transformer)" instead of
# "ACT").
_POLICY_LABELS = {
    "act": "ACT (Action Chunking Transformer)",
    "diffusion": "Diffusion Policy",
    "smolvla": "SmolVLA",
    "sac": "SAC (Soft Actor-Critic)",
    "tdmpc": "TD-MPC",
    "vqbet": "VQ-BeT",
    "pi0": "Pi-0",
    "pi05": "Pi-0.5",
    "pi0_fast": "Pi-0 FAST",
    "multi_task_dit": "Multi-task DiT",
    "groot": "GR00T",
    "wall_x": "Wall-X",
    "xvla": "X-VLA",
    "act_vlm": "ACT-VLM",
    "sarm": "SARM",
    "reward_classifier": "Reward classifier",
}


# Manually-registered recipes — anything that doesn't go through
# ``lerobot-train``'s draccus CLI. HVLA's flow_matching trainer has its
# own argparse; the args-key convention is snake_case (no ``policy.``
# prefix). We surface a hand-curated field list here because HVLA's
# argparse isn't a dataclass — there's nothing to introspect dynamically.
# This is the one place hand-curation survives, and only because the
# underlying trainer hasn't yet been migrated to a dataclass-based config.
_NON_DRACCUS_RECIPES: list[dict[str, Any]] = [
    {
        "type_name": "hvla_flow_s1",
        "label": "HVLA Flow Matching S1 (no S2)",
        "recipe": HVLA_FLOW_S1_RECIPE,
        "arg_key_prefix": "",  # HVLA's keys are bare snake_case
        "fields": [
            {"name": "chunk_size", "label": "Chunk size", "type": "int", "default": 50},
            {"name": "num_inference_steps", "label": "Inference steps", "type": "int", "default": 15},
            {"name": "hidden_dim", "label": "Hidden dim", "type": "int", "default": 768},
            {"name": "num_decoder_layers", "label": "Decoder layers", "type": "int", "default": 6},
            {"name": "num_workers", "label": "Data workers", "type": "int", "default": 4},
        ],
        # Make explicit which form keys map to the trainer's CLI; the
        # frontend doesn't need to know but it's useful in tests + docs.
        "_known_flag_map": list(HVLA_FLOW_S1_FIELD_TO_FLAG),
    },
]


@router.get("/policies")
def list_policies() -> list[dict]:
    """Catalog of policies the GUI can launch.

    Auto-discovers everything registered with ``PreTrainedConfig`` (via
    ``@PreTrainedConfig.register_subclass(name)``) — adding a new policy
    upstream that follows the registration pattern surfaces in the GUI
    with no code edit here. Manually-registered non-draccus recipes
    (HVLA) are spliced in alongside.

    Each entry: ``{type_name, label, recipe, arg_key_prefix, fields}``
    where ``recipe`` is null for plain ``lerobot-train`` and a marker
    string (e.g. ``"hvla_flow_s1"``) for routed recipes, and
    ``arg_key_prefix`` tells the frontend whether to prepend
    ``"policy."`` to each field name when building the args dict.
    """
    _ensure_policy_configs_loaded()
    from lerobot.configs.policies import PreTrainedConfig  # noqa: PLC0415

    schemas: list[dict] = []
    for type_name, cls in sorted(PreTrainedConfig.get_known_choices().items()):
        fields = _introspect_policy_fields(cls)
        if not fields:
            # Policy registered but no renderable fields — skip rather
            # than emit a useless entry. Picked up automatically if the
            # config grows scalar fields later.
            continue
        schemas.append(
            {
                "type_name": type_name,
                "label": _POLICY_LABELS.get(type_name) or _humanize_policy_name(type_name),
                "recipe": None,  # default: lerobot-train via draccus
                "arg_key_prefix": "policy.",
                "fields": fields,
            }
        )

    # Splice in non-draccus recipes (HVLA, future custom trainers).
    for entry in _NON_DRACCUS_RECIPES:
        # Don't leak the introspection helper into the API; strip
        # internal-only keys before serialising.
        external = {k: v for k, v in entry.items() if not k.startswith("_")}
        schemas.append(external)
    return schemas

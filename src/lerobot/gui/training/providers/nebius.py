# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""Nebius Compute provider — Ephemeral VM lifecycle via the Nebius Python SDK.

Uses the official ``nebius`` SDK (``from nebius.sdk import SDK``;
``InstanceServiceClient``), an OPTIONAL GUI-server dependency
(``lerobot[nebius]``) — NOT part of the training image. The SDK is async;
this provider exposes the synchronous :class:`HostProvider` interface and
drives the SDK with ``asyncio.run`` from the orchestrator's prep thread.

The async ``InstanceServiceClient`` is INJECTABLE (``instance_service``)
so tests run against a fake service with no network/credentials — the
real SDK message construction is exercised in tests (it builds fine on
protobuf 6.x, which the GUI server already pins via wandb/grpcio-tools).

Grounding: the SKU table, image family, boot-disk + network-interface
shapes, and cloud-init come from the Terraform that provisioned the
GPU-smoke VM (2026-06-12) and were validated against the live SDK message
builders (2026-06-15). Account-scoped inputs — ``project_id`` (the
instance parent) and ``subnet_id`` — are required and read from
``NEBIUS_PROJECT_ID`` / ``NEBIUS_SUBNET_ID`` when not passed explicitly.

Auth: a Nebius **service-account** key, configured once on the GUI server
via the Nebius-connection form (:mod:`lerobot.gui.training.nebius_credentials`).
The provider is handed that key file's path (``credentials_file``) and
builds ``SDK(credentials_file_name=...)``. Service-account keys don't expire,
so unattended teardown past the 12 h personal-token ceiling works. When no
key file is given the SDK falls back to ambient credentials
(``NEBIUS_IAM_TOKEN`` env / ``~/.nebius`` CLI profile) — the single-user
workstation path. The key is server-held and therefore shared by anyone who
can reach the GUI, exactly like the existing HF token / SSH key (documented
limitation; see ``DESIGN.md`` § Authentication).

TTL caveat (see ``_HARD_TTL_NOTE``): the compute proto exposes no native
scheduled-delete, so the hard TTL is enforced two ways — a cloud-init
``poweroff`` timer on the VM itself (caps *compute* billing even if the
GUI server is gone) plus orchestrator-driven ``destroy`` on every terminal
run state (frees the *disk*). ``verify_destroyed`` is the backstop that
catches a leak. A true vendor-side scheduled-delete is a follow-up if the
SDK gains one.
"""

from __future__ import annotations

import asyncio
import getpass
import logging
import os
import time
import uuid
from collections.abc import Callable, Coroutine
from pathlib import Path
from typing import Any

from lerobot.gui.training.providers.protocol import (
    GpuKind,
    HostHandle,
    SpawnSpec,
)

logger = logging.getLogger(__name__)

# ── SKU mapping ─────────────────────────────────────────────────────────────


# Canonical GpuKind → Nebius platform string. Lookup is explicit — a GPU
# Nebius doesn't offer raises at lookup time, before spawn.
#
# PROVENANCE: only L40S is VERIFIED — its platform/preset come from the
# Terraform that actually provisioned the GPU-smoke VM (gpu-l40s-d,
# 1gpu-16vcpu-96gb). The others are best-effort and flagged UNVERIFIED;
# confirm against `nebius compute platform list` before relying on them.
NEBIUS_GPU_PLATFORMS: dict[GpuKind, str] = {
    "L40S": "gpu-l40s-d",  # VERIFIED (Terraform, 2026-06-12)
    "L4": "gpu-l4-a",  # UNVERIFIED
    "H100": "gpu-h100-sxm5",  # UNVERIFIED
    "H200": "gpu-h200-sxm5",  # UNVERIFIED
}


# Preset must match the GPU platform. v1 is single-GPU per host.
NEBIUS_DEFAULT_PRESET: dict[GpuKind, str] = {
    "L40S": "1gpu-16vcpu-96gb",  # VERIFIED (Terraform, 2026-06-12)
    "L4": "1gpu-8vcpu-32gb",  # UNVERIFIED
    "H100": "1gpu-16vcpu-200gb",  # UNVERIFIED
    "H200": "1gpu-16vcpu-200gb",  # UNVERIFIED
}


# Boot image family. VERIFIED from the Terraform spec.
NEBIUS_IMAGE_FAMILY = "ubuntu24.04-cuda13.0-serverless"

# Soft cap to surface a warning. The 1.28 TB Terraform web-form default is
# what bit us; warn well below that.
NEBIUS_DISK_WARN_GIB = 256

# Disk block size matching the verified Terraform spec.
NEBIUS_DISK_BLOCK_SIZE_BYTES = 4096

_HARD_TTL_NOTE = (
    "Nebius's compute API exposes no native scheduled-delete; the hard TTL "
    "is enforced by a cloud-init poweroff timer (caps compute) + "
    "orchestrator destroy on terminal (frees disk)."
)


# ── Provider ────────────────────────────────────────────────────────────────


class NebiusProvider:
    """Nebius Compute provider — Ephemeral VM lifecycle.

    Pre (default runner): ``lerobot[nebius]`` installed, an authenticated
    SDK credential source (``NEBIUS_IAM_TOKEN`` / SA key / CLI config),
    and ``project_id`` + ``subnet_id`` from the constructor or env.
    """

    id = "nebius"
    display_name = "Nebius (auto-managed)"

    def __init__(
        self,
        *,
        instance_service: Any | None = None,
        subnet_service: Any | None = None,
        sdk: Any | None = None,
        credentials_file: str | None = None,
        project_id: str | None = None,
        subnet_id: str | None = None,
        ssh_public_key: str | None = None,
        ssh_user: str | None = None,
        image_family: str = NEBIUS_IMAGE_FAMILY,
        poll_interval_s: float = 10.0,
        ready_timeout_s: float = 900.0,
        sleep: Callable[[float], None] = time.sleep,
        clock: Callable[[], float] = time.time,
    ) -> None:
        # instance_service: an injected async client (tests). When None it's
        # built lazily from the SDK on first use — so construction never
        # imports the optional SDK or touches credentials.
        self._instance_service = instance_service
        self._subnet_service = subnet_service
        self._sdk = sdk
        # Path to the server-held service-account key file (the one configured
        # via the Nebius-connection form). Passed straight to the SDK's
        # credentials_file_name; never read back to a client, never logged,
        # never sent to the pod. When None the SDK falls back to ambient
        # credentials (NEBIUS_IAM_TOKEN env / ~/.nebius/ CLI profile) — the
        # single-user workstation path.
        self._credentials_file = credentials_file
        self._project_id = project_id or os.environ.get("NEBIUS_PROJECT_ID")
        self._subnet_id = subnet_id or os.environ.get("NEBIUS_SUBNET_ID")
        self._ssh_public_key = ssh_public_key
        # Fix A (workaround for #11 — remote-workdir mapping): the spawned
        # VM's user defaults to the GUI server's OWN username, so the remote
        # $HOME matches the local run-dir path the orchestrator reuses as the
        # remote workdir + docker mount source. Without this, an ephemeral VM
        # whose user is e.g. "lerobot" can't create /home/<gui-user>/.cache/…
        # and every run fails at launch. Proper fix is the local↔remote path
        # split tracked in TODO.md ("Remote workdir mapping (#11)"); once that
        # lands, this can revert to a fixed "lerobot".
        self._ssh_user = ssh_user or getpass.getuser()
        self._image_family = image_family
        self._poll_interval_s = poll_interval_s
        self._ready_timeout_s = ready_timeout_s
        self._sleep = sleep
        self._clock = clock

    # ── Lifecycle (sync interface; async SDK underneath) ────────────────

    def spawn(self, spec: SpawnSpec) -> HostHandle:
        """Provision a VM with an inline boot disk + static public IP, wait
        until SSH is reachable, and return the handle.

        Pre: ``spec.ttl_seconds > 0``; the GpuKind is in the SKU tables;
        ``project_id`` + ``subnet_id`` are configured.
        Post: returned handle's ``ssh_host:ssh_port`` accepts SSH via the
        user's local key, and the VM carries a cloud-init poweroff timer at
        ``spawn + ttl_seconds`` (compute backstop; see ``_HARD_TTL_NOTE``).
        """
        if spec.ttl_seconds <= 0:
            raise ValueError(f"spec.ttl_seconds must be > 0, got {spec.ttl_seconds}")
        if not self._project_id:
            raise NebiusConfigError("project_id (or NEBIUS_PROJECT_ID) is required to spawn")
        if not self._subnet_id:
            raise NebiusConfigError("subnet_id (or NEBIUS_SUBNET_ID) is required to spawn")

        platform = self.platform_for(spec.gpu)
        preset = self.preset_for(spec.gpu)
        warning = self.disk_warning(spec.disk_gib)
        if warning:
            logger.warning("nebius spawn: %s", warning)

        name = _instance_name()
        pubkey = self._resolve_pubkey()
        user_data = build_cloud_init(self._ssh_user, pubkey, ttl_seconds=spec.ttl_seconds)
        spawned_at = self._clock()
        expires_at = int(spawned_at + spec.ttl_seconds)

        request = self._build_create_request(
            name=name, platform=platform, preset=preset, spec=spec, user_data=user_data
        )
        try:
            resource_id = self._run(self._acreate(request))
        except NebiusAuthError:
            raise
        except Exception as e:  # noqa: BLE001 — a bad/expired token fails at the RPC, not at SDK build
            if _looks_like_auth_error(e):
                raise NebiusAuthError(_AUTH_HINT) from e
            raise
        logger.info("nebius spawn: created instance %s (%s); %s", resource_id, name, _HARD_TTL_NOTE)

        deadline = spawned_at + self._ready_timeout_s
        return self._poll_until_ready(
            resource_id=resource_id, spec=spec, expires_at_unix=expires_at, deadline=deadline
        )

    def destroy(self, handle: HostHandle) -> None:
        """Idempotent delete of the VM (cascades the inline boot disk + IP).

        A delete of an already-gone instance is treated as success —
        destroy is the cleanup path and must never wedge.
        """
        try:
            self._run(self._adelete(handle.provider_resource_id))
        except Exception as e:  # noqa: BLE001 — classify not-found as success
            if _is_not_found(e):
                return
            raise

    def verify_destroyed(self, handle: HostHandle) -> bool:
        """True iff no billable resource for this handle remains.

        Re-gets the instance; not-found (or DELETING) is success. Any live
        state — or an error that prevents confirmation — reports False so
        the caller keeps looking rather than assuming clean.
        """
        try:
            state = self._run(self._aget_state(handle.provider_resource_id))
        except Exception as e:  # noqa: BLE001
            if _is_not_found(e):
                return True
            logger.warning("nebius verify_destroyed: get failed: %s", e)
            return False
        return state in {"DELETING", "DELETED", "UNSPECIFIED"}

    # ── Async SDK calls (overridable seams; tests inject the service) ────

    async def _acreate(self, request: Any) -> str:
        service = self._service()
        op = await service.create(request)
        await op.wait()
        if not getattr(op, "successful", lambda: True)():
            raise NebiusSpawnError(f"create operation failed: {getattr(op, 'status', None)}")
        return op.resource_id

    async def _aget(self, resource_id: str) -> Any:
        service = self._service()
        get_req = _get_request(resource_id)
        return await service.get(get_req)

    async def _aget_state(self, resource_id: str) -> str:
        instance = await self._aget(resource_id)
        return _state_name(instance)

    async def _adelete(self, resource_id: str) -> None:
        service = self._service()
        op = await service.delete(_delete_request(resource_id))
        await op.wait()

    # ── Internals ───────────────────────────────────────────────────────

    def _poll_until_ready(
        self, *, resource_id: str, spec: SpawnSpec, expires_at_unix: int, deadline: float
    ) -> HostHandle:
        last_state = "?"
        while True:
            try:
                instance = self._run(self._aget(resource_id))
                last_state = _state_name(instance)
                ip = _public_ip(instance)
                if last_state == "RUNNING" and ip:
                    return HostHandle(
                        provider="nebius",
                        provider_resource_id=resource_id,
                        ssh_host=ip,
                        ssh_port=22,
                        ssh_user=self._ssh_user,
                        region=_region(instance) or (spec.region_hint or "unknown"),
                        expires_at_unix=expires_at_unix,
                    )
            except Exception as e:  # noqa: BLE001 — transient get errors during boot are expected
                logger.debug("nebius poll: get(%s) failed (transient?): %s", resource_id, e)
            if self._clock() >= deadline:
                raise NebiusSpawnTimeoutError(
                    f"instance {resource_id} not reachable within "
                    f"{self._ready_timeout_s:.0f}s (last state={last_state!r})"
                )
            self._sleep(self._poll_interval_s)

    def _build_create_request(
        self, *, name: str, platform: str, preset: str, spec: SpawnSpec, user_data: str
    ) -> Any:
        sym = _sdk_symbols()
        return sym.CreateInstanceRequest(
            metadata=sym.ResourceMetadata(parent_id=self._project_id, name=name),
            spec=sym.InstanceSpec(
                resources=sym.ResourcesSpec(platform=platform, preset=preset),
                boot_disk=sym.AttachedDiskSpec(
                    attach_mode=sym.AttachedDiskSpec.AttachMode.READ_WRITE,
                    device_id="boot-disk",
                    managed_disk=sym.ManagedDisk(
                        name=f"{name}-boot",
                        spec=sym.DiskSpec(
                            size_gibibytes=spec.disk_gib,
                            block_size_bytes=NEBIUS_DISK_BLOCK_SIZE_BYTES,
                            type=sym.DiskSpec.DiskType.NETWORK_SSD,
                            source_image_family=sym.SourceImageFamily(image_family=self._image_family),
                        ),
                    ),
                ),
                network_interfaces=[
                    # name + ip_address are API-required. name is the in-VM OS
                    # interface name (truncated to 15 chars by Nebius); "eth0" is
                    # the conventional primary NIC. Empty IPAddress() auto-
                    # allocates the internal IPv4; public_ip_address is the
                    # external IP we SSH into.
                    sym.NetworkInterfaceSpec(
                        name="eth0",
                        subnet_id=self._subnet_id,
                        ip_address=sym.IPAddress(),
                        public_ip_address=sym.PublicIPAddress(static=True),
                    )
                ],
                cloud_init_user_data=user_data,
            ),
        )

    def _service(self) -> Any:
        if self._instance_service is not None:
            return self._instance_service
        sym = _sdk_symbols()
        if self._sdk is None:
            try:
                if self._credentials_file:
                    # Server-held service-account key (configured via the
                    # Nebius-connection form) → long-lived SDK.
                    self._sdk = sym.SDK(credentials_file_name=self._credentials_file)
                else:
                    # Ambient credentials (NEBIUS_IAM_TOKEN env / ~/.nebius/).
                    self._sdk = sym.SDK()
            except Exception as e:  # noqa: BLE001 — rewrap any cred-resolution failure
                raise NebiusAuthError(_AUTH_HINT) from e
        self._instance_service = sym.InstanceServiceClient(self._sdk)
        return self._instance_service

    @staticmethod
    def _run(coro: Coroutine[Any, Any, Any]) -> Any:
        # Provider methods are called from the orchestrator's prep daemon
        # thread, which has no running loop — asyncio.run is safe here.
        return asyncio.run(coro)

    def _subnet_svc(self) -> Any:
        if self._subnet_service is not None:
            return self._subnet_service
        self._service()  # ensures self._sdk is built (raises NebiusAuthError on bad creds)
        self._subnet_service = _sdk_symbols().SubnetServiceClient(self._sdk)
        return self._subnet_service

    def list_subnets(self, project_id: str) -> list[dict[str, str]]:
        """List the VPC subnets in a project as ``[{"id", "name"}]``.

        Lets the GUI offer a subnet picker so operators don't hand-copy IDs.
        Raises :class:`NebiusAuthError` on bad credentials; other API errors
        (e.g. the service account lacks ``vpc.subnets.list``) propagate so the
        caller can fall back to manual entry.

        Pre: ``project_id`` is non-empty. Post: list (possibly empty).
        """
        if not project_id or not project_id.strip():
            raise NebiusConfigError("project_id is required to list subnets")
        return self._run(self._alist_subnets(project_id.strip()))

    async def _alist_subnets(self, project_id: str) -> list[dict[str, str]]:
        sym = _sdk_symbols()
        resp = await self._subnet_svc().list(sym.ListSubnetsRequest(parent_id=project_id))
        out: list[dict[str, str]] = []
        for item in getattr(resp, "items", None) or []:
            md = item.metadata
            out.append({"id": md.id, "name": getattr(md, "name", "") or md.id})
        return out

    def _resolve_pubkey(self) -> str:
        if self._ssh_public_key:
            return self._ssh_public_key
        for name in ("id_ed25519.pub", "id_rsa.pub", "id_ecdsa.pub"):
            p = Path.home() / ".ssh" / name
            if p.exists():
                return p.read_text().strip()
        raise FileNotFoundError(
            "no SSH public key found under ~/.ssh — generate one with "
            "`ssh-keygen` so the spawned VM can authorize your login"
        )

    # ── Pure helpers (UI + tests) ────────────────────────────────────────

    @staticmethod
    def disk_warning(disk_gib: int) -> str | None:
        if disk_gib > NEBIUS_DISK_WARN_GIB:
            return (
                f"Boot disk is {disk_gib} GiB (warning at >{NEBIUS_DISK_WARN_GIB}). "
                f"Disks bill continuously until deleted, regardless of VM "
                f"state. Most training fits in 50–100 GiB."
            )
        return None

    @staticmethod
    def platform_for(gpu: GpuKind) -> str:
        if gpu not in NEBIUS_GPU_PLATFORMS:
            raise ValueError(
                f"Nebius has no platform mapping for GPU kind {gpu!r}. "
                f"Supported: {sorted(NEBIUS_GPU_PLATFORMS)}"
            )
        return NEBIUS_GPU_PLATFORMS[gpu]

    @staticmethod
    def preset_for(gpu: GpuKind) -> str:
        if gpu not in NEBIUS_DEFAULT_PRESET:
            raise ValueError(
                f"Nebius has no default preset for GPU kind {gpu!r}. "
                f"Supported: {sorted(NEBIUS_DEFAULT_PRESET)}"
            )
        return NEBIUS_DEFAULT_PRESET[gpu]


# ── Errors ──────────────────────────────────────────────────────────────────


class NebiusConfigError(RuntimeError):
    """Missing required configuration (project/subnet/credentials)."""


# Instruction shown verbatim to the user (run detail) when no usable Nebius
# credential is available — the "you're not logged in" path.
_AUTH_HINT = (
    "Nebius is not connected. A Nebius tenant admin must create a service "
    "account and paste its authorized-key JSON (plus project + subnet ids) "
    "in the GUI's 'Nebius connection' form. The key is stored on the GUI "
    "server and used for every spawn and teardown; service-account keys do "
    "not expire."
)


class NebiusAuthError(RuntimeError):
    """No usable Nebius credential (missing/expired token). Carries the
    user-facing instruction so the run's error tells them how to fix it."""


class NebiusSpawnError(RuntimeError):
    """Spawn failed or produced output we couldn't interpret."""


class NebiusSpawnTimeoutError(RuntimeError):
    """The VM did not become SSH-reachable within the readiness window."""


# ── SDK glue (lazy import; the SDK is an optional extra) ─────────────────────


class _SdkSymbols:
    """Lazily-imported handle to the SDK classes the provider needs.

    Imported on first use, not at module import, so the registry can import
    ``NebiusProvider`` on a GUI server that didn't install ``lerobot[nebius]``
    — only an actual spawn/destroy requires the SDK present.
    """

    __slots__ = (
        "SDK",
        "InstanceServiceClient",
        "CreateInstanceRequest",
        "GetInstanceRequest",
        "DeleteInstanceRequest",
        "InstanceSpec",
        "ResourcesSpec",
        "AttachedDiskSpec",
        "ManagedDisk",
        "DiskSpec",
        "SourceImageFamily",
        "NetworkInterfaceSpec",
        "IPAddress",
        "PublicIPAddress",
        "InstanceStatus",
        "ResourceMetadata",
        "SubnetServiceClient",
        "ListSubnetsRequest",
    )

    def __init__(self) -> None:
        try:
            from nebius.api.nebius.common.v1 import ResourceMetadata
            from nebius.api.nebius.compute.v1 import (
                AttachedDiskSpec,
                CreateInstanceRequest,
                DeleteInstanceRequest,
                DiskSpec,
                GetInstanceRequest,
                InstanceServiceClient,
                InstanceSpec,
                InstanceStatus,
                IPAddress,
                ManagedDisk,
                NetworkInterfaceSpec,
                PublicIPAddress,
                ResourcesSpec,
                SourceImageFamily,
            )
            from nebius.api.nebius.vpc.v1 import ListSubnetsRequest, SubnetServiceClient
            from nebius.sdk import SDK
        except ImportError as e:
            raise NebiusConfigError(
                "the Nebius SDK is not installed — `uv sync --extra nebius` "
                "(or `pip install 'nebius>=0.3,<0.4'`) on the GUI server"
            ) from e
        self.SDK = SDK
        self.InstanceServiceClient = InstanceServiceClient
        self.CreateInstanceRequest = CreateInstanceRequest
        self.GetInstanceRequest = GetInstanceRequest
        self.DeleteInstanceRequest = DeleteInstanceRequest
        self.InstanceSpec = InstanceSpec
        self.ResourcesSpec = ResourcesSpec
        self.AttachedDiskSpec = AttachedDiskSpec
        self.ManagedDisk = ManagedDisk
        self.DiskSpec = DiskSpec
        self.SourceImageFamily = SourceImageFamily
        self.NetworkInterfaceSpec = NetworkInterfaceSpec
        self.IPAddress = IPAddress
        self.PublicIPAddress = PublicIPAddress
        self.InstanceStatus = InstanceStatus
        self.ResourceMetadata = ResourceMetadata
        self.SubnetServiceClient = SubnetServiceClient
        self.ListSubnetsRequest = ListSubnetsRequest


_SYMS: _SdkSymbols | None = None


def _sdk_symbols() -> _SdkSymbols:
    global _SYMS
    if _SYMS is None:
        _SYMS = _SdkSymbols()
    return _SYMS


def _get_request(resource_id: str) -> Any:
    return _sdk_symbols().GetInstanceRequest(id=resource_id)


def _delete_request(resource_id: str) -> Any:
    return _sdk_symbols().DeleteInstanceRequest(id=resource_id)


def _state_name(instance: Any) -> str:
    """Instance.status.state enum → its NAME (e.g. 'RUNNING'). Robust to the
    enum being an IntEnum or already a string."""
    state = instance.status.state
    name = getattr(state, "name", None)
    return str(name if name is not None else state).upper()


def _public_ip(instance: Any) -> str | None:
    for nic in instance.status.network_interfaces or []:
        pub = getattr(nic, "public_ip_address", None)
        addr = getattr(pub, "address", None) if pub is not None else None
        if addr:
            # Nebius returns the address in CIDR form (e.g. "195.242.28.4/32");
            # ssh needs the bare host or it can't resolve it, so drop the prefix
            # (and any IPv6 brackets) → always a bare host.
            return str(addr).split("/", 1)[0].strip("[]")
    return None


def _region(instance: Any) -> str | None:
    meta = getattr(instance, "metadata", None)
    return getattr(meta, "parent_id", None) if meta is not None else None


def _is_not_found(exc: Exception) -> bool:
    blob = str(exc).lower()
    return "not found" in blob or "notfound" in blob or "does not exist" in blob


def _looks_like_auth_error(exc: Exception) -> bool:
    blob = (type(exc).__name__ + " " + str(exc)).lower()
    return any(
        k in blob
        for k in ("token", "unauthenticated", "unauthorized", "permission", "credential", "401", "403")
    )


# ── Pure helpers (no SDK) ────────────────────────────────────────────────────


def _instance_name() -> str:
    """Unique, identifiable instance name. ``lerobot-eph-`` prefix makes
    stray VMs obvious in the Nebius console (and greppable for cleanup)."""
    return f"lerobot-eph-{uuid.uuid4().hex[:10]}"


def build_cloud_init(ssh_user: str, ssh_public_key: str, *, ttl_seconds: int) -> str:
    """cloud-init user-data: create ``ssh_user`` with key-only login +
    NOPASSWD sudo, and arm a one-shot ``poweroff`` ``ttl_seconds`` from boot
    as the hard-TTL compute backstop (see ``_HARD_TTL_NOTE``).

    Pre: ``ssh_public_key`` is a single OpenSSH public-key line; ttl > 0.
    """
    assert ssh_public_key and not ssh_public_key.startswith("/"), (
        f"expected public-key contents, not a path: {ssh_public_key!r}"
    )
    assert ttl_seconds > 0, ttl_seconds
    # `shutdown -P +<minutes>` powers off (halts compute billing) even if the
    # GUI server is gone. Disk teardown is the orchestrator's destroy() job;
    # verify_destroyed catches any leak. Minimum 1 minute.
    ttl_minutes = max(1, ttl_seconds // 60)
    return (
        "#cloud-config\n"
        "users:\n"
        f"  - name: {ssh_user}\n"
        "    sudo: ALL=(ALL) NOPASSWD:ALL\n"
        # Join the image's pre-installed `docker` group so the SSH user can
        # reach /var/run/docker.sock (training runs via `docker run`). Set at
        # user-creation, not a later usermod: it must be in effect for the very
        # first login, since the persistent ControlMaster session would miss a
        # group added mid-session.
        "    groups: [docker]\n"
        "    shell: /bin/bash\n"
        "    ssh_authorized_keys:\n"
        f"      - {ssh_public_key}\n"
        "runcmd:\n"
        f"  - shutdown -P +{ttl_minutes}\n"
    )

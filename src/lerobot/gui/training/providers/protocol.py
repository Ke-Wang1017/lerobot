# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""HostProvider protocol — the abstraction over vendor-specific VM lifecycles.

The protocol is intentionally narrow:

  - spawn(spec)         → HostHandle             (creates VM, waits for SSH)
  - destroy(handle)     → None                   (destroys VM + disk + IP)
  - verify_destroyed(handle) → bool              (idempotency check)

What is deliberately NOT in the protocol:

  - stop() / start()  — Lambda has neither; including would force a
    fiction on vendors that don't support it. Ephemeral mode never
    stops; it destroys. Stopped-but-not-destroyed VMs cause the disk-
    billing-surprise the whole design is meant to prevent.
  - ssh_run() / ssh_tail() — vendor-agnostic. SshConnection + the polling
    loop in :mod:`lerobot.gui.training.worker` live above the protocol.
  - Auth flow — each provider reads its own credential source (e.g.
    `~/.nebius/`, RunPod env var, AWS profile). The GUI's API-key UI
    is a separate concern from the protocol.
  - Upload/download — checkpoints flow through HF Hub (see the design
    doc's "Recovery from preemption" section). Providers don't move files.
  - Preemption events — handled by :class:`PollScheduler` (3 fails →
    disconnected, 10 → lost contact). Providers don't emit events.
  - Cost estimation — no price tables, no dollar ceilings. Vendor prices
    drift and there's no catalog API to validate against; the UI links
    to the vendor's pricing page instead. Spend protection is mechanical:
    the spawn spec's hard TTL + the provider's disk-size warning.

Vendor-agnostic Persistent mode (BYO SSH) implements this protocol as a
degenerate provider: every method raises — the user provided the SSH
endpoint by hand and owns the VM.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Protocol, runtime_checkable

# Narrow union — extend as we add providers. Adding a new vendor here
# AND registering it in training_providers/__init__.py is the full plug-in
# checklist.
ProviderId = Literal["persistent", "nebius"]


# Canonical GPU kinds the GUI exposes. Vendor-specific SKUs map onto these
# in each provider implementation (e.g., Nebius's "gpu-l40s-a" → "L40S",
# RunPod's "NVIDIA L40S" → "L40S"). Keeps the UI dropdown sane.
GpuKind = Literal[
    "RTX_4090",
    "L4",
    "L40S",
    "A100_40",
    "A100_80",
    "H100",
    "H200",
    "B200",
]


@dataclass(frozen=True, kw_only=True)
class SpawnSpec:
    """What the user asked for. Vendor-neutral.

    Pre: ``ttl_seconds`` > 0.
    """

    gpu: GpuKind
    gpu_count: int = 1
    preemptible: bool = True  # Ephemeral defaults to spot/preemptible
    disk_gib: int = 100
    image: str  # docker ref the training container pulls
    region_hint: str | None = None  # provider may ignore; e.g. "us", "eu"
    # Hard kill — REQUIRED. The provider must enforce this server-side
    # (e.g., Nebius scheduled-delete async op; RunPod --terminate-after)
    # so a runaway VM dies even if our GUI server is gone.
    ttl_seconds: int


@dataclass(frozen=True, kw_only=True)
class HostHandle:
    """Live handle to a running host. The SSH path consumes this.

    All vendors converge on this shape — provider-specific identifiers
    are captured in ``provider_resource_id`` (opaque to the SSH layer).

    Pre: ``ssh_host:ssh_port`` reachable from the GUI server, and
    ``ssh_user`` can log in via the user's local SSH setup
    (``~/.ssh/config`` Host blocks, ssh-agent, default-path keys).
    Ephemeral providers that generated a per-pod key are responsible
    for surfacing it into the user's setup at spawn time (e.g. a
    ``Host`` block appended to ``~/.ssh/config`` keyed by
    ``provider_resource_id``) — the SSH layer never reads a key path
    directly. See DESIGN.md § Authentication.

    Post: ``destroy(handle)`` returns the system to a state where
    ``verify_destroyed(handle)`` returns True.
    """

    provider: ProviderId
    provider_resource_id: str  # vendor's VM/pod id; required for destroy()
    ssh_host: str
    ssh_port: int
    ssh_user: str = "root"
    # Persistent storage that survives this handle (None for fully-ephemeral).
    # Used so a follow-up Ephemeral spawn can reuse e.g. the HF cache.
    persistent_volume_id: str | None = None
    region: str
    # Provider's hard-TTL deadline as unix timestamp — never None. The GUI
    # uses this to display "auto-destroys at HH:MM" and as a sanity check.
    expires_at_unix: int


@runtime_checkable
class HostProvider(Protocol):
    """Vendor adapter. Implementations live one-per-file in this package.

    Conformance is checked by ``isinstance(obj, HostProvider)`` thanks
    to ``runtime_checkable``.
    """

    id: ProviderId
    display_name: str

    def spawn(self, spec: SpawnSpec) -> HostHandle:
        """Provision the VM, attach disk, allocate public IP, wait until
        SSH is reachable. The provider MUST configure server-side TTL so
        the VM dies even if we lose control of it.

        Post: returned handle's ssh_host:ssh_port accepts SSH via the
        user's local SSH setup.
        """
        ...

    def destroy(self, handle: HostHandle) -> None:
        """Idempotent. Destroys VM + boot disk + public IP allocation.

        Does NOT touch ``handle.persistent_volume_id`` — that's lifecycled
        separately by the named-volume API on each vendor.
        """
        ...

    def verify_destroyed(self, handle: HostHandle) -> bool:
        """Re-list vendor resources matching this handle.

        Returns True iff nothing identifying this handle remains billable.
        Automates the "always check Compute → Disks after teardown"
        hygiene rule.
        """
        ...

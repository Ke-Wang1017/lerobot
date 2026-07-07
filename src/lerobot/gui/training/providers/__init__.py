# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""Training host providers — pluggable VM-lifecycle backends.

Each provider implements :class:`HostProvider` for one vendor. The
GUI's application code (training_worker, API endpoints, frontend)
talks only to the protocol; vendor specifics are contained per-file.

Adding a new vendor: drop a new file (e.g. ``runpod.py``), register it
in :func:`get_provider`. No changes to application code.

For the full design see ``src/lerobot/gui/training/DESIGN.md`` § "HostProvider
protocol (Phase 2.5)".
"""

from __future__ import annotations

from lerobot.gui.training.providers.nebius import NebiusProvider
from lerobot.gui.training.providers.persistent import PersistentSshProvider
from lerobot.gui.training.providers.protocol import (
    HostHandle,
    HostProvider,
    ProviderId,
    SpawnSpec,
)

__all__ = [
    "HostHandle",
    "HostProvider",
    "ProviderId",
    "SpawnSpec",
    "NebiusProvider",
    "PersistentSshProvider",
    "get_provider",
    "list_providers",
]


# Registry. Keep narrow — adding a vendor adds one line.
_PROVIDERS: dict[str, type] = {
    "persistent": PersistentSshProvider,
    "nebius": NebiusProvider,
}


def get_provider(provider_id: str) -> HostProvider:
    """Resolve a vendor id to an instantiated HostProvider.

    For Nebius, wires in the server-held service-account connection
    (key file + project/subnet) from
    :class:`~lerobot.gui.training.nebius_credentials.NebiusConnectionStore`
    if one is configured; otherwise the provider falls back to ambient
    credentials (the single-user workstation path). The key is server-held
    and shared by anyone who can reach the GUI — same trust model as the
    existing HF token / SSH key (see ``DESIGN.md`` § Authentication).

    Pre: ``provider_id`` is one of the registered providers.
    Post: a fresh provider instance for one spawn/destroy cycle.
    """
    if provider_id not in _PROVIDERS:
        raise ValueError(f"unknown provider id: {provider_id!r}. Registered providers: {sorted(_PROVIDERS)}")
    cls = _PROVIDERS[provider_id]
    if cls is NebiusProvider:
        from lerobot.gui.training.nebius_credentials import NebiusConnectionStore

        status = NebiusConnectionStore().status()
        return cls(
            credentials_file=str(NebiusConnectionStore().key_path) if status.has_key else None,
            project_id=status.project_id,
            subnet_id=status.subnet_id,
        )
    return cls()


def list_providers() -> list[str]:
    """Return the registered provider ids, sorted for stable UI display."""
    return sorted(_PROVIDERS)

# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""Persistent SSH provider — the "BYO SSH endpoint" degenerate case.

User provisioned the VM themselves (on Nebius / RunPod / their lab box /
whatever) and pasted the SSH command into the GUI. We just connect.

Why is this a HostProvider at all, instead of a special case?

Because the rest of the training stack (training_worker, API endpoints,
frontend) talks only to HostProvider. Making the BYO-SSH path implement
the same protocol means there's one code path through the system, not
two. The lifecycle methods are deliberately fail-fast, not no-ops:

  - spawn() raises NotImplementedError — Persistent hosts are added by
    the user via the "Add training host" dialog, not spawned by the GUI.
    The GUI obtains a HostHandle by reading the saved HostProfile, NOT
    by calling spawn().
  - destroy() / verify_destroyed() raise NotImplementedError — the user
    owns the VM lifecycle; the GUI never touches it. Raising (instead of
    a silent no-op + vacuous True) means a caller that reaches for these
    on the wrong host type fails loudly rather than believing resources
    were freed. The C2 teardown sweep iterates GUI-spawned handles only,
    so no legitimate caller ever lands here.
"""

from __future__ import annotations

import time

from lerobot.gui.training.jobs import HostProfile
from lerobot.gui.training.providers.protocol import (
    HostHandle,
    SpawnSpec,
)


class PersistentSshProvider:
    """Wraps a user-provided SSH endpoint as a HostProvider.

    The GUI calls :meth:`handle_from_profile` to convert a saved
    :class:`HostProfile` (one JSON per host on disk) into a HostHandle
    that the SSH path can consume. spawn() is intentionally not
    implemented — there's no VM to create.
    """

    id = "persistent"
    display_name = "BYO SSH"

    def spawn(self, spec: SpawnSpec) -> HostHandle:
        raise NotImplementedError(
            "Persistent hosts are added by the user via the 'Add training host' "
            "dialog (BYO SSH endpoint). Use handle_from_profile() to construct "
            "a HostHandle from a saved HostProfile."
        )

    def destroy(self, handle: HostHandle) -> None:
        raise NotImplementedError(
            "Persistent hosts are user-owned; the GUI never destroys them. "
            "Remove the host via DELETE /api/training/hosts/<id> instead. "
            "If you reached this from a teardown sweep, the sweep should "
            "iterate GUI-spawned (ephemeral) handles only."
        )

    def verify_destroyed(self, handle: HostHandle) -> bool:
        raise NotImplementedError(
            "Persistent hosts are user-owned; there is nothing the GUI "
            "created to verify. A caller asking this question has the "
            "wrong host type."
        )

    # ── Persistent-mode-specific helper ───────────────────────────────────

    @staticmethod
    def handle_from_profile(profile: HostProfile) -> HostHandle:
        """Construct a HostHandle from a saved BYO-SSH HostProfile.

        Pre: ``profile`` is a Persistent host (has ssh_host, ssh_user
        populated; not Ephemeral/Nebius shape).
        Post: returned handle has expires_at_unix = +100 years (never
        expires, since the user owns the VM).
        """
        return HostHandle(
            provider="persistent",
            provider_resource_id=f"profile:{profile.name}",
            ssh_host=profile.ssh_host,
            ssh_port=profile.ssh_port,
            ssh_user=profile.ssh_user,
            persistent_volume_id=profile.persistent_volume,
            region="user-managed",
            # "Never expires" — the user controls the VM, not us. ~100 years
            # from now is more than enough; the UI just shouldn't show a
            # countdown for Persistent hosts.
            expires_at_unix=int(time.time()) + 100 * 365 * 24 * 3600,
        )

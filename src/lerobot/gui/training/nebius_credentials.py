# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""Server-held Nebius connection store — one service-account credential for
the whole GUI server.

Trust model (see ``DESIGN.md`` § Authentication): the GUI server has no
login of its own, so a server-held credential is usable by anyone who can
reach the server's port — exactly like the ambient HF token and SSH key
the GUI already relies on. Storing the Nebius service-account key here adds
no new trust boundary; it sits on the existing one. A Nebius **tenant
admin** creates the service account (only they can) and pastes its
authorized-key JSON once via the GUI; the operator never SSHes into the
server. The key is the only Nebius credential that doesn't expire, so
unattended teardown past the 12 h personal-token ceiling works.

Layout under ``~/.config/lerobot/nebius/`` (dir ``0700``):

* ``service_account.json`` (``0600``) — the pasted authorized-key JSON, in
  the exact shape the SDK's ``credentials_file_name`` consumes
  (``{"subject-credentials": {...}}``). Passed to ``SDK`` by path; never
  read back to a client, never logged, never sent to the training pod.
* ``connection.json`` (``0600``) — ``{"project_id", "subnet_id"}``. Not
  secret, but co-located so a single ``clear()`` removes everything.

The key is validated structurally on ``set`` (without importing the
optional SDK) so a malformed paste fails immediately, not at first spawn.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# Default location, parallel to ``HOSTS_DIR`` in ``jobs.py``.
NEBIUS_DIR = Path.home() / ".config" / "lerobot" / "nebius"
_KEY_FILENAME = "service_account.json"
_CONN_FILENAME = "connection.json"


class NebiusCredentialError(ValueError):
    """The pasted service-account key is malformed or missing required fields."""


@dataclass(frozen=True)
class NebiusConnectionStatus:
    """Non-secret view of the stored connection — safe to return to a client.

    ``configured`` is True only when BOTH the key and project/subnet are
    present (the minimum to spawn). ``service_account_id`` / ``key_id`` are
    the SA's own identifiers (``sub`` / ``kid``) — identifiers, not secrets —
    so the operator can confirm *which* account is wired up. The private key
    is never included.
    """

    configured: bool
    has_key: bool
    service_account_id: str | None
    key_id: str | None
    project_id: str | None
    subnet_id: str | None


def _validate_key_json(data: Any) -> tuple[str, str]:
    """Structurally validate a pasted authorized-key blob; return (sub, kid).

    Mirrors the SDK's ``credentials_file`` reader requirements without
    importing the optional ``nebius`` package: a ``subject-credentials``
    object carrying ``private-key`` (PEM), ``kid``, and matching ``iss``/
    ``sub``. Raises :class:`NebiusCredentialError` on any shape mismatch.
    """
    if not isinstance(data, dict):
        raise NebiusCredentialError("expected a JSON object (the service-account key)")
    subj = data.get("subject-credentials")
    if not isinstance(subj, dict):
        raise NebiusCredentialError(
            "missing 'subject-credentials' — paste the full authorized-key JSON "
            "from `nebius iam auth-public-key create` or the Nebius console"
        )
    missing = [f for f in ("private-key", "kid", "sub") if not subj.get(f)]
    if missing:
        raise NebiusCredentialError(f"service-account key missing field(s): {', '.join(missing)}")
    pk = subj["private-key"]
    if not isinstance(pk, str) or "PRIVATE KEY" not in pk:
        raise NebiusCredentialError("'private-key' is not a PEM private key")
    iss = subj.get("iss", subj["sub"])
    if iss != subj["sub"]:
        raise NebiusCredentialError(f"issuer must equal subject ('{iss}' != '{subj['sub']}')")
    return str(subj["sub"]), str(subj["kid"])


def assemble_authorized_key_json(*, private_key: str, key_id: str, service_account_id: str) -> str:
    """Build the SDK ``subject-credentials`` JSON from the pieces the Nebius
    console gives you — a locally-generated private key, the uploaded key's id,
    and the service-account id. The console path (unlike ``auth-public-key
    generate``) never hands you the combined file, so we assemble it here.

    Post: the result parses + validates via :func:`_validate_key_json`.
    """
    sa = service_account_id.strip()
    return json.dumps(
        {
            "subject-credentials": {
                "alg": "RS256",
                "private-key": private_key.strip(),
                "kid": key_id.strip(),
                "iss": sa,
                "sub": sa,
            }
        },
        indent=2,
    )


class NebiusConnectionStore:
    """File-backed store for the one server-held Nebius connection.

    Pre: the process can create/read files under ``dir_`` (defaults to
    ``~/.config/lerobot/nebius``). All writes are ``0600`` under a ``0700``
    directory.
    """

    def __init__(self, dir_: Path | None = None) -> None:
        # Read the module global at call time (not as a default arg) so tests
        # can redirect the whole store by monkeypatching ``NEBIUS_DIR``.
        self._dir = dir_ if dir_ is not None else NEBIUS_DIR

    @property
    def key_path(self) -> Path:
        """Absolute path to the SA key file — what ``SDK(credentials_file_name=...)``
        is given. May not exist yet; callers check :meth:`status` first."""
        return self._dir / _KEY_FILENAME

    def status(self) -> NebiusConnectionStatus:
        """Current connection state. Never raises; absence reads as not-configured.

        Post: ``configured`` ⇒ key file exists AND project_id/subnet_id set.
        """
        sub = kid = None
        has_key = self.key_path.exists()
        if has_key:
            try:
                data = json.loads(self.key_path.read_text())
                subj = data.get("subject-credentials", {})
                sub, kid = subj.get("sub"), subj.get("kid")
            except (OSError, ValueError, AttributeError):
                # A corrupt key on disk is treated as no key — the operator
                # re-pastes. Don't crash the status probe.
                has_key = False
        conn = self._read_connection()
        project_id, subnet_id = conn.get("project_id"), conn.get("subnet_id")
        return NebiusConnectionStatus(
            configured=bool(has_key and project_id and subnet_id),
            has_key=has_key,
            service_account_id=sub,
            key_id=kid,
            project_id=project_id,
            subnet_id=subnet_id,
        )

    def set(self, *, key_json: str, project_id: str, subnet_id: str) -> NebiusConnectionStatus:
        """Validate and persist the SA key + project/subnet. Replaces any prior.

        Pre: ``key_json`` is the authorized-key JSON; ``project_id`` and
        ``subnet_id`` are non-empty account-scoped ids.
        Post: key written ``0600``; :meth:`status` reports ``configured=True``.
        Raises :class:`NebiusCredentialError` on a malformed key or blank ids.
        """
        if not project_id or not project_id.strip():
            raise NebiusCredentialError("project_id is required")
        if not subnet_id or not subnet_id.strip():
            raise NebiusCredentialError("subnet_id is required")
        try:
            data = json.loads(key_json)
        except ValueError as e:
            raise NebiusCredentialError(f"service-account key is not valid JSON: {e}") from e
        _validate_key_json(data)  # raises on bad shape, before we touch disk

        self._dir.mkdir(parents=True, exist_ok=True)
        os.chmod(self._dir, 0o700)
        # Re-serialize (canonical form; strips any incidental whitespace) and
        # lock down before write via the open() mode.
        self._write_locked(self.key_path, json.dumps(data, indent=2))
        self._write_locked(
            self._dir / _CONN_FILENAME,
            json.dumps({"project_id": project_id.strip(), "subnet_id": subnet_id.strip()}, indent=2),
        )
        return self.status()

    def clear(self) -> bool:
        """Remove the stored key + connection. Idempotent.

        Post: no key/connection files remain. Returns True iff anything was
        removed (so the caller can report "nothing to clear").
        """
        removed = False
        for name in (_KEY_FILENAME, _CONN_FILENAME):
            p = self._dir / name
            if p.exists():
                p.unlink()  # safe-destruct: our own credential file under the lerobot config dir
                removed = True
        return removed

    def _read_connection(self) -> dict[str, Any]:
        p = self._dir / _CONN_FILENAME
        if not p.exists():
            return {}
        try:
            return json.loads(p.read_text())
        except (OSError, ValueError):
            return {}

    @staticmethod
    def _write_locked(path: Path, content: str) -> None:
        """Write ``content`` with ``0600`` perms from creation (no readable window)."""
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        try:
            os.write(fd, content.encode("utf-8"))
        finally:
            os.close(fd)
        os.chmod(path, 0o600)  # tighten if the file pre-existed with looser perms

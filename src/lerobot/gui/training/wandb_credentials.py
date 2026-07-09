# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""Weights & Biases connection for GUI-launched training runs.

Same trust model as the Nebius store next door (see ``DESIGN.md`` §
Authentication): the GUI server has no login of its own, so a server-held
credential is reachable by anyone who can reach the server's port — exactly
like the ambient HF token and SSH key the GUI already relies on.

Unlike the Nebius key, the W&B key **is** sent to the training worker: the
trainer process is what talks to wandb.ai. It travels as the ``WANDB_API_KEY``
environment variable (never on a command line — the docker recipe forwards it
by name with ``-e WANDB_API_KEY``, not by value).

**Resolution order** (:func:`resolve_api_key`), first hit wins:

1. ``stored`` — ``~/.config/lerobot/wandb/api_key`` (``0600``), pasted once via
   the GUI. Explicit beats ambient: the user typed this most recently.
2. ``env``    — ``$WANDB_API_KEY`` in the GUI server's own environment.
3. ``netrc``  — the ``api.wandb.ai`` entry ``wandb login`` writes to ``~/.netrc``.

So a user who has ever run ``wandb login`` on this machine is already connected
and never has to paste anything. This mirrors :func:`lerobot.jobs.hf.
resolve_wandb_api_key` (env → netrc), extended with the GUI's own store.
"""

from __future__ import annotations

import logging
import netrc
import os
import re
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)

# Default location, parallel to ``NEBIUS_DIR`` in ``nebius_credentials.py``.
WANDB_DIR = Path.home() / ".config" / "lerobot" / "wandb"
_KEY_FILENAME = "api_key"  # nosec B105 — filename, not a secret
_ENTITY_FILENAME = "entity"

# Where ``wandb login`` writes its credential, and the host we verify against.
WANDB_NETRC_MACHINE = "api.wandb.ai"
DEFAULT_WANDB_BASE_URL = "https://api.wandb.ai"

# Where the user gets a key. Surfaced in the GUI so nobody has to go hunting.
WANDB_AUTHORIZE_URL = "https://wandb.ai/authorize"

SOURCE_STORED = "stored"
SOURCE_ENV = "env"
SOURCE_NETRC = "netrc"

# Two shapes in the wild: the legacy 40-char hex key, and the newer
# ``wandb_v1_<base62>`` (~86 chars). Self-hosted deployments issue others still.
# So validate only what a paste can plausibly get wrong: empty, whitespace-
# bearing (a copied command line), or implausibly short.
_KEY_RE = re.compile(r"^[A-Za-z0-9_-]{20,256}$")

# Verification is a courtesy — it turns "key saved" into "connected as <you>".
# Kept short: a hung wandb.ai must not hang the form.
_VERIFY_TIMEOUT_S = 6.0


class WandbCredentialError(ValueError):
    """The pasted API key is malformed, or wandb.ai rejected it."""


@dataclass(frozen=True)
class WandbConnectionStatus:
    """Non-secret view of the connection — safe to return to a client.

    ``configured`` means "a run can enable tracking right now", from any of
    the three sources. ``masked_key`` shows only the last 4 characters, enough
    for the user to tell two keys apart. ``entity`` is the W&B username/team
    the key belongs to — known only when the key was pasted through the GUI
    and verification succeeded, since we never spend a network round-trip
    verifying an ambient key on a status read.
    """

    configured: bool
    source: str | None
    masked_key: str | None
    entity: str | None


def mask_key(key: str | None) -> str | None:
    """``"…7f3a"`` — enough to distinguish two keys, useless to steal."""
    if not key:
        return None
    return "…" + key[-4:] if len(key) > 4 else "…"


def validate_api_key(key: str) -> str:
    """Strip and structurally check a pasted key. Returns the clean key.

    Raises :class:`WandbCredentialError` on anything a paste gets wrong, before
    we touch disk or the network.
    """
    clean = (key or "").strip()
    if not clean:
        raise WandbCredentialError("API key is required")
    if not _KEY_RE.match(clean):
        raise WandbCredentialError(
            "that doesn't look like a W&B API key — copy the key straight from "
            f"{WANDB_AUTHORIZE_URL} (no spaces, no 'wandb login' prefix)"
        )
    return clean


def _netrc_key() -> str | None:
    """The key ``wandb login`` left in ``~/.netrc``, if any."""
    try:
        rc = netrc.netrc()
    except (FileNotFoundError, netrc.NetrcParseError, OSError):
        return None
    auth = rc.authenticators(WANDB_NETRC_MACHINE)
    if auth is None:
        return None
    _login, _account, password = auth
    return password or None


def resolve_api_key(store: WandbConnectionStore | None = None) -> tuple[str | None, str | None]:
    """Return ``(api_key, source)`` for the worker, or ``(None, None)``.

    Source is one of :data:`SOURCE_STORED` / :data:`SOURCE_ENV` /
    :data:`SOURCE_NETRC` — see the module docstring for the precedence.
    """
    store = store or WandbConnectionStore()
    stored = store.read_key()
    if stored:
        return stored, SOURCE_STORED
    env = os.environ.get("WANDB_API_KEY")
    if env and env.strip():
        return env.strip(), SOURCE_ENV
    from_netrc = _netrc_key()
    if from_netrc:
        return from_netrc, SOURCE_NETRC
    return None, None


def verify_api_key(key: str, *, timeout: float = _VERIFY_TIMEOUT_S) -> str | None:
    """Ask wandb.ai who this key belongs to. Returns the default entity name.

    Returns ``None`` when the key is valid-looking but the answer is unknown
    (wandb.ai unreachable, or a deployment whose GraphQL shape differs). Raises
    :class:`WandbCredentialError` only when the server actively rejects the key
    — an offline user must still be able to save one.
    """
    import httpx  # local import: the store is importable without the HTTP stack

    base = os.environ.get("WANDB_BASE_URL", DEFAULT_WANDB_BASE_URL).rstrip("/")
    try:
        resp = httpx.post(
            f"{base}/graphql",
            json={"query": "query Viewer { viewer { username entity } }"},
            auth=("api", key),
            timeout=timeout,
        )
    except httpx.HTTPError as e:
        logger.info("wandb key verification skipped (%s unreachable): %s", base, e)
        return None
    if resp.status_code in (401, 403):
        raise WandbCredentialError(f"wandb.ai rejected that API key — check it at {WANDB_AUTHORIZE_URL}")
    if resp.status_code != 200:
        logger.info("wandb key verification inconclusive: HTTP %s", resp.status_code)
        return None
    try:
        viewer = resp.json()["data"]["viewer"]
    except (ValueError, KeyError, TypeError):
        return None
    if not viewer:
        # A 200 with a null viewer is how wandb.ai reports a bad key on some
        # deployments — no 401, just nobody home.
        raise WandbCredentialError(f"wandb.ai rejected that API key — check it at {WANDB_AUTHORIZE_URL}")
    return viewer.get("entity") or viewer.get("username") or None


class WandbConnectionStore:
    """File-backed store for the one server-held W&B key.

    Pre: the process can create/read files under ``dir_`` (defaults to
    :data:`WANDB_DIR`). The key is written ``0600`` under a ``0700`` directory.
    """

    def __init__(self, dir_: Path | None = None) -> None:
        # Read the module global at call time (not as a default arg) so tests
        # can redirect the whole store by monkeypatching ``WANDB_DIR``.
        self._dir = dir_ if dir_ is not None else WANDB_DIR

    @property
    def key_path(self) -> Path:
        return self._dir / _KEY_FILENAME

    @property
    def entity_path(self) -> Path:
        return self._dir / _ENTITY_FILENAME

    def read_key(self) -> str | None:
        """The stored key, or None. Never raises; a corrupt file reads as absent."""
        try:
            key = self.key_path.read_text().strip()
        except (OSError, ValueError):
            return None
        return key or None

    def read_entity(self) -> str | None:
        try:
            return self.entity_path.read_text().strip() or None
        except (OSError, ValueError):
            return None

    def status(self) -> WandbConnectionStatus:
        """Current connection state across all three sources. Never raises."""
        key, source = resolve_api_key(self)
        return WandbConnectionStatus(
            configured=key is not None,
            source=source,
            masked_key=mask_key(key),
            entity=self.read_entity() if source == SOURCE_STORED else None,
        )

    def set(self, *, api_key: str, entity: str | None = None) -> WandbConnectionStatus:
        """Validate and persist the key (and the entity it verified as).

        Pre: ``api_key`` is the raw pasted key.
        Post: key written ``0600``; :meth:`status` reports ``source="stored"``.
        Raises :class:`WandbCredentialError` on a malformed key.
        """
        clean = validate_api_key(api_key)
        self._dir.mkdir(parents=True, exist_ok=True)
        os.chmod(self._dir, 0o700)
        self._write_locked(self.key_path, clean)
        if entity:
            self._write_locked(self.entity_path, entity)
        elif self.entity_path.exists():
            # A new key may belong to a different account; a stale entity label
            # would be a lie. Drop it rather than show the previous owner.
            self.entity_path.unlink()
        return self.status()

    def clear(self) -> bool:
        """Remove the stored key + entity. Idempotent.

        Returns True iff anything was removed. Note this does NOT disconnect a
        user whose key comes from ``$WANDB_API_KEY`` or ``~/.netrc`` — those
        aren't ours to delete, and :meth:`status` will keep reporting them.
        """
        removed = False
        for p in (self.key_path, self.entity_path):
            if p.exists():
                p.unlink()  # safe-destruct: our own credential file under the lerobot config dir
                removed = True
        return removed

    @staticmethod
    def _write_locked(path: Path, content: str) -> None:
        """Write ``content`` with ``0600`` perms from creation (no readable window)."""
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        try:
            os.write(fd, content.encode("utf-8"))
        finally:
            os.close(fd)
        os.chmod(path, 0o600)  # tighten if the file pre-existed with looser perms


# ── Run-args integration ──────────────────────────────────────────────────────
#
# The GUI form writes lerobot-train's own flag into Run.args (``wandb.enable``),
# rather than inventing a ``__wandb__`` meta marker: the recipe builder then
# needs no special case, since a user-supplied value already beats the
# soft-forced ``wandb.enable=false`` default.

WANDB_ENABLE_ARG = "wandb.enable"
WANDB_API_KEY_ENV = "WANDB_API_KEY"  # nosec B105 — env var name, not a secret


def wandb_enabled(args: dict) -> bool:
    """Whether this run's args ask for W&B tracking. Accepts the bool the API
    sends and the ``"true"`` string a hand-written run might carry."""
    v = args.get(WANDB_ENABLE_ARG)
    if isinstance(v, bool):
        return v
    return isinstance(v, str) and v.strip().lower() in ("true", "1", "yes")


def wandb_env_for_args(args: dict) -> dict[str, str]:
    """Secret env for the worker: ``{"WANDB_API_KEY": ...}``, or ``{}``.

    Empty when the run doesn't use W&B — an unrelated run must not carry the
    key into its container. Also empty when tracking is on but no key resolves;
    the API rejects that combination up front (see ``api/training.py``), so
    reaching here means a key vanished between submit and launch, and letting
    the trainer fail loudly on wandb's own auth error beats guessing.
    """
    if not wandb_enabled(args):
        return {}
    key, _source = resolve_api_key()
    return {WANDB_API_KEY_ENV: key} if key else {}

"""Bearer-token auth for the HTTP transport.

Three pieces:

* ``TokenStore`` — SQLite-backed CRUD over named, scoped, revocable tokens.
  Tokens are stored as SHA-256 hashes so a DB read does not surrender bearer
  values. The full bearer is returned only at issuance.

* ``LeRobotTokenVerifier`` — implements FastMCP's ``TokenVerifier`` protocol;
  hashes the presented bearer, looks it up, and returns an ``AccessToken``
  carrying the token's scope set.

* ``requires_scope(scope)`` — decorator for individual MCP tools. In HTTP
  mode it rejects tool calls whose token lacks the named scope. In stdio mode
  (no auth context) it's a no-op — stdio is single-process, single-user, and
  trusted by definition.

Scopes form a strict hierarchy: ``read`` ⊂ ``annotate`` ⊂ ``operate``. The
verifier just records what's in the token row; the decorator does the
membership check. Server-level middleware additionally enforces a global
"must have at least ``read``" floor so anonymous/unauthenticated requests
never reach a tool.
"""

from __future__ import annotations

import functools
import hashlib
import json
import os
import secrets
import sqlite3
import threading
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from mcp.server.auth.middleware.auth_context import get_access_token
from mcp.server.auth.provider import AccessToken, TokenVerifier

SCOPE_READ = "read"
SCOPE_COMMENT = "comment"  # sidecar writes — notes that don't touch canonical data
SCOPE_EDIT = "edit"  # canonical-state mutations on the host (datasets, profiles, Hub)
SCOPE_OPERATE = "operate"  # hardware-moving + long-running operations
ALL_SCOPES = (SCOPE_READ, SCOPE_COMMENT, SCOPE_EDIT, SCOPE_OPERATE)

# Legacy scope names — tokens issued before the 4-tier rename carry these in
# the SQLite store. Lookup normalizes on read so existing tokens keep working
# without forcing the user to re-issue. New tokens never get these names.
_LEGACY_SCOPE_ALIASES = {
    "annotate": SCOPE_COMMENT,
}

_TOKEN_PREFIX = "sk-lr-"  # nosec B105 — not a password, recognition prefix for issued bearers

# Re-bump last_used_at at most once per this many seconds, to keep per-request
# overhead to a single SELECT (and avoid SQLite write contention on bursty load).
_LAST_USED_DEBOUNCE_S = 60


def default_token_store_path() -> Path:
    """Single source of truth for where bearer tokens live.

    Honors ``LEROBOT_MCP_TOKEN_STORE``; falls back to ``$HF_LEROBOT_HOME/_mcp_tokens.sqlite``.
    The GUI's /ai_setup page, the mounted MCP server, and the standalone CLI
    must agree on this — otherwise a token issued from one surface is invisible
    to another and the user sees a silent 401.
    """
    override = os.environ.get("LEROBOT_MCP_TOKEN_STORE")
    if override:
        return Path(override)
    from lerobot.utils.constants import HF_LEROBOT_HOME

    return HF_LEROBOT_HOME / "_mcp_tokens.sqlite"


_SCHEMA = """
CREATE TABLE IF NOT EXISTS tokens (
    token_hash    TEXT PRIMARY KEY,
    name          TEXT NOT NULL UNIQUE,
    scopes_json   TEXT NOT NULL,
    created_at    TEXT NOT NULL,
    last_used_at  TEXT,
    revoked_at    TEXT
);
CREATE INDEX IF NOT EXISTS idx_tokens_active
    ON tokens(revoked_at) WHERE revoked_at IS NULL;
"""


def _now_iso() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def _hash(token: str) -> str:
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


def _generate_token() -> str:
    """Opaque high-entropy bearer prefixed for recognizability."""
    return _TOKEN_PREFIX + secrets.token_urlsafe(32)


def _validate_scopes(scopes: list[str]) -> list[str]:
    """Reject unknown scopes early; normalize to a sorted unique list.

    Legacy scope names (e.g. ``annotate``) are silently mapped to their
    new equivalents so callers issuing tokens with old code keep working
    during the transition.
    """
    canonical = [_LEGACY_SCOPE_ALIASES.get(s, s) for s in scopes]
    invalid = [s for s in canonical if s not in ALL_SCOPES]
    if invalid:
        raise ValueError(f"Unknown scope(s): {invalid}. Valid: {list(ALL_SCOPES)}")
    return sorted(set(canonical))


def _canonicalize_stored_scopes(scopes: list[str]) -> list[str]:
    """Map legacy scope names to current ones on read.

    Tokens issued before the 4-tier rename have ``annotate`` in their
    stored ``scopes_json``; this lets them keep working without forcing
    re-issuance. Returns a fresh list; never mutates the input.
    """
    return [_LEGACY_SCOPE_ALIASES.get(s, s) for s in scopes]


def _should_bump_last_used(prev_iso: str | None, now_iso: str) -> bool:
    """True iff the prior `last_used_at` is null or older than the debounce window.

    Parses ISO-8601 timestamps minted by ``_now_iso``. A malformed prior
    value is treated as 'old' so a recovery write still happens.
    """
    if prev_iso is None:
        return True
    try:
        prev = datetime.fromisoformat(prev_iso)
        now = datetime.fromisoformat(now_iso)
    except ValueError:
        return True
    return (now - prev).total_seconds() >= _LAST_USED_DEBOUNCE_S


class TokenStore:
    """SQLite-backed bearer-token store; thread-safe via per-thread connections.

    Precondition: ``db_path`` parent directory exists or can be created.
    Postcondition: the schema is installed; instance is safe to share across
    threads.
    """

    def __init__(self, db_path: Path):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._local = threading.local()
        # Install schema using the constructing thread's connection. PRAGMAs
        # are applied per-connection in _conn() instead of here, so worker
        # threads opening their own connection later also pick them up.
        self._conn().executescript(_SCHEMA)

    def _conn(self) -> sqlite3.Connection:
        conn = getattr(self._local, "conn", None)
        if conn is None:
            conn = sqlite3.connect(self.db_path, isolation_level=None)
            conn.row_factory = sqlite3.Row
            # WAL is persistent on the DB file once set; synchronous=NORMAL
            # is per-connection. Run both on every fresh per-thread conn so
            # secondary uvicorn workers don't silently fall back to FULL.
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA synchronous=NORMAL")
            self._local.conn = conn
        return conn

    def issue(self, name: str, scopes: list[str]) -> str:
        """Create a new token; return the cleartext bearer (shown once).

        Reissuing the name of a previously-revoked token is a normal rotation:
        the dead row is dropped first so ``UNIQUE(name)`` only ever guards
        *active* names (a revoked token keeps the name reserved otherwise).
        """
        assert name, "name must be non-empty"
        scopes = _validate_scopes(scopes)
        token = _generate_token()
        conn = self._conn()
        conn.execute("DELETE FROM tokens WHERE name = ? AND revoked_at IS NOT NULL", (name,))
        try:
            conn.execute(
                """
                INSERT INTO tokens(token_hash, name, scopes_json, created_at)
                VALUES (?, ?, ?, ?)
                """,
                (_hash(token), name, json.dumps(scopes), _now_iso()),
            )
        except sqlite3.IntegrityError as e:
            raise ValueError(
                f"Token name {name!r} is already in use by an active token — "
                f"revoke it first or pick another name"
            ) from e
        return token

    def revoke(self, name: str) -> bool:
        """Mark a token revoked by name. Returns False if no active token by that name."""
        cur = self._conn().execute(
            "UPDATE tokens SET revoked_at = ? WHERE name = ? AND revoked_at IS NULL",
            (_now_iso(), name),
        )
        return cur.rowcount > 0

    def list_tokens(self, include_revoked: bool = False) -> list[dict[str, Any]]:
        """All tokens (sans the bearer) ordered by created_at desc."""
        sql = "SELECT name, scopes_json, created_at, last_used_at, revoked_at FROM tokens"
        if not include_revoked:
            sql += " WHERE revoked_at IS NULL"
        sql += " ORDER BY created_at DESC"
        return [
            {
                "name": r["name"],
                "scopes": _canonicalize_stored_scopes(json.loads(r["scopes_json"])),
                "created_at": r["created_at"],
                "last_used_at": r["last_used_at"],
                "revoked_at": r["revoked_at"],
            }
            for r in self._conn().execute(sql).fetchall()
        ]

    def lookup(self, token: str) -> dict[str, Any] | None:
        """Return ``{name, scopes}`` if the token is active, else None.

        Side effect: bumps ``last_used_at`` to now on a successful lookup —
        but debounced (skipped if the prior bump was less than
        ``_LAST_USED_DEBOUNCE_S`` seconds ago) to avoid a SQLite write on
        every hot-path MCP request.
        """
        token_hash = _hash(token)
        row = (
            self._conn()
            .execute(
                "SELECT name, scopes_json, revoked_at, last_used_at FROM tokens WHERE token_hash = ?",
                (token_hash,),
            )
            .fetchone()
        )
        if row is None or row["revoked_at"] is not None:
            return None
        now_iso = _now_iso()
        if _should_bump_last_used(row["last_used_at"], now_iso):
            self._conn().execute(
                "UPDATE tokens SET last_used_at = ? WHERE token_hash = ?",
                (now_iso, token_hash),
            )
        return {
            "name": row["name"],
            "scopes": _canonicalize_stored_scopes(json.loads(row["scopes_json"])),
        }


class LeRobotTokenVerifier(TokenVerifier):
    """Adapts ``TokenStore`` to FastMCP's ``TokenVerifier`` protocol."""

    def __init__(self, store: TokenStore):
        self.store = store

    async def verify_token(self, token: str) -> AccessToken | None:
        info = self.store.lookup(token)
        if info is None:
            return None
        return AccessToken(
            token=token,
            client_id=info["name"],
            scopes=info["scopes"],
            expires_at=None,
        )


def requires_scope(scope: str):
    """Decorate a tool to require ``scope``. In stdio mode (no auth context) this is a no-op.

    Handles both sync and async tools — returns the same call shape the
    wrapped function returns, so ``functools.wraps`` + ``inspect.iscoroutinefunction``
    still report the tool's true nature to FastMCP.

    The check is intentionally outside the FastMCP middleware's global
    ``required_scopes``: that middleware enforces a single floor for the
    whole server; this decorator gives per-tool granularity above it.
    """
    assert scope in ALL_SCOPES, f"unknown scope {scope!r}"

    def _check_or_raise(fn_name: str) -> None:
        token = get_access_token()
        if token is not None and scope not in token.scopes:
            raise PermissionError(
                f"Tool {fn_name!r} requires scope '{scope}'; "
                f"caller {token.client_id!r} has scopes {token.scopes}"
            )

    def decorator(fn):
        import inspect

        if inspect.iscoroutinefunction(fn):

            @functools.wraps(fn)
            async def async_wrapper(*args, **kwargs):
                _check_or_raise(fn.__name__)
                return await fn(*args, **kwargs)

            # Surface the required scope on the wrapper so introspection
            # tools (lerobot_list_tools) can read it without re-parsing
            # the decorator chain.
            async_wrapper._required_scope = scope  # type: ignore[attr-defined]
            return async_wrapper

        @functools.wraps(fn)
        def sync_wrapper(*args, **kwargs):
            _check_or_raise(fn.__name__)
            return fn(*args, **kwargs)

        sync_wrapper._required_scope = scope  # type: ignore[attr-defined]
        return sync_wrapper

    return decorator

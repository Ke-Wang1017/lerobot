"""SQLite-backed sidecar for agent-written episode annotations.

Annotations live OUT-OF-BAND from the canonical ``LeRobotDataset``: writing a tag
never touches the dataset's parquet/video files. This is intentional — the
dataset is treated as immutable from the agent's perspective; only the sidecar
DB carries agent findings.

The DB is a single file shared across all datasets (one row per
``(repo_id, episode_id, key)``). Default location is
``$HF_LEROBOT_HOME / _mcp_annotations.sqlite`` but tests may pass any path.
"""

from __future__ import annotations

import json
import sqlite3
import threading
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

_SCHEMA = """
CREATE TABLE IF NOT EXISTS episode_tags (
    repo_id     TEXT    NOT NULL,
    episode_id  INTEGER NOT NULL,
    key         TEXT    NOT NULL,
    value_json  TEXT    NOT NULL,
    set_at      TEXT    NOT NULL,
    PRIMARY KEY (repo_id, episode_id, key)
);
CREATE INDEX IF NOT EXISTS idx_episode_tags_repo ON episode_tags(repo_id);
CREATE INDEX IF NOT EXISTS idx_episode_tags_key  ON episode_tags(repo_id, key);
"""


def _now_iso() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


class AnnotationStore:
    """Thread-safe SQLite wrapper for episode tags.

    Precondition: ``db_path`` parent directory exists OR can be created.
    Postcondition: the schema is installed; instance is safe to share across
    threads (each call acquires a short-held connection from a per-thread map).
    """

    def __init__(self, db_path: Path):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._local = threading.local()
        # Eagerly install schema on the main thread's connection so the file
        # exists immediately (callers may inspect it before writing).
        with self._conn() as conn:
            conn.executescript(_SCHEMA)
            conn.execute("PRAGMA journal_mode=WAL")  # better concurrent reads
            conn.execute("PRAGMA synchronous=NORMAL")

    def _conn(self) -> sqlite3.Connection:
        conn = getattr(self._local, "conn", None)
        if conn is None:
            conn = sqlite3.connect(self.db_path, isolation_level=None)
            conn.row_factory = sqlite3.Row
            self._local.conn = conn
        return conn

    def set_tag(self, repo_id: str, episode_id: int, key: str, value: Any) -> None:
        """Upsert one (repo_id, episode_id, key) → value annotation."""
        assert isinstance(episode_id, int) and episode_id >= 0, "episode_id must be non-negative int"
        assert key, "key must be non-empty"
        value_json = json.dumps(value)
        conn = self._conn()
        conn.execute(
            """
            INSERT INTO episode_tags(repo_id, episode_id, key, value_json, set_at)
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(repo_id, episode_id, key) DO UPDATE SET
                value_json = excluded.value_json,
                set_at     = excluded.set_at
            """,
            (repo_id, episode_id, key, value_json, _now_iso()),
        )

    def get_tags(self, repo_id: str, episode_id: int) -> dict[str, Any]:
        """All tags for one episode as a key→value dict (empty if none)."""
        conn = self._conn()
        rows = conn.execute(
            "SELECT key, value_json FROM episode_tags WHERE repo_id=? AND episode_id=?",
            (repo_id, episode_id),
        ).fetchall()
        return {r["key"]: json.loads(r["value_json"]) for r in rows}

    def delete_tag(self, repo_id: str, episode_id: int, key: str) -> bool:
        conn = self._conn()
        cur = conn.execute(
            "DELETE FROM episode_tags WHERE repo_id=? AND episode_id=? AND key=?",
            (repo_id, episode_id, key),
        )
        return cur.rowcount > 0

    def list_tagged_episodes(self, repo_id: str, key: str | None = None) -> list[dict[str, Any]]:
        """Episodes with any tag (or with a specific ``key`` set)."""
        conn = self._conn()
        if key is None:
            rows = conn.execute(
                "SELECT DISTINCT episode_id FROM episode_tags WHERE repo_id=? ORDER BY episode_id",
                (repo_id,),
            ).fetchall()
            return [{"episode_id": r["episode_id"]} for r in rows]
        rows = conn.execute(
            "SELECT episode_id, value_json, set_at FROM episode_tags "
            "WHERE repo_id=? AND key=? ORDER BY episode_id",
            (repo_id, key),
        ).fetchall()
        return [
            {
                "episode_id": r["episode_id"],
                "value": json.loads(r["value_json"]),
                "set_at": r["set_at"],
            }
            for r in rows
        ]

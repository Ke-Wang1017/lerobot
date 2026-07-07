# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Application state for the GUI server."""

from __future__ import annotations

import asyncio
import json
import logging
import os
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

if TYPE_CHECKING:
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from lerobot.gui.frame_cache import FrameCache
    from lerobot.gui.hub_jobs import HubJobState

logger = logging.getLogger(__name__)

# File names for persistence
EDITS_FILENAME = ".lerobot_gui_edits.json"
EDITS_FILE_VERSION = 1


@dataclass
class PendingEdit:
    """A pending edit operation that hasn't been applied yet.

    Edit types:
    * ``"delete"`` — virtual-delete an episode. ``params`` is empty.
    * ``"trim"`` — trim an episode to ``[start_frame, end_frame)``. ``params``
      has ``start_frame`` and ``end_frame``.
    * ``"feature_set"`` — overwrite a per-frame feature for a contiguous
      range. ``params`` has ``feature``, ``from_index``, ``to_index``,
      ``value``. Applied via ``set_feature_values``. ``episode_index`` is
      stored for grouping in the GUI; the actual range is in ``params``.
    """

    edit_type: Literal["delete", "trim", "feature_set"]
    dataset_id: str
    episode_index: int
    params: dict = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)


@dataclass
class AppState:
    """Global application state.

    Holds all opened datasets, frame cache, and pending edits.
    """

    frame_cache: FrameCache
    datasets: dict[str, LeRobotDataset] = field(default_factory=dict)
    pending_edits: list[PendingEdit] = field(default_factory=list)
    _dataset_locks: dict[str, asyncio.Lock] = field(default_factory=dict)
    # Active + recently-finished Hub transfers, keyed by job_id. The GUI's
    # top-bar Transfers tray polls /hub/jobs for the live list; completed
    # jobs stick around (bounded by hub_job_retention_s below) so a brief
    # tab close doesn't lose the "done — N MB pushed" status.
    hub_jobs: dict[str, HubJobState] = field(default_factory=dict)

    def add_edit(self, edit: PendingEdit) -> None:
        """Add a pending edit."""
        self.pending_edits.append(edit)

    def remove_edit(self, index: int) -> None:
        """Remove a pending edit by index."""
        if 0 <= index < len(self.pending_edits):
            self.pending_edits.pop(index)

    def clear_edits(self, dataset_id: str | None = None) -> None:
        """Clear pending edits, optionally filtered by dataset."""
        if dataset_id is None:
            self.pending_edits.clear()
        else:
            self.pending_edits = [e for e in self.pending_edits if e.dataset_id != dataset_id]

    def get_edits_for_dataset(self, dataset_id: str) -> list[PendingEdit]:
        """Get pending edits for a specific dataset."""
        return [e for e in self.pending_edits if e.dataset_id == dataset_id]

    def pending_feature_set_edits_for_dataset(self, dataset_id: str) -> list[PendingEdit]:
        """Pending ``feature_set`` edits scoped to one dataset.

        Used as the guard for schema mutations: the schema-add path refuses
        to run while value edits on the same dataset are queued, since
        cross-mutation races could leave parquet shards inconsistent.
        """
        return [e for e in self.pending_edits if e.dataset_id == dataset_id and e.edit_type == "feature_set"]

    def is_episode_deleted(self, dataset_id: str, episode_index: int) -> bool:
        """Check if an episode is marked for deletion."""
        return any(
            e.dataset_id == dataset_id and e.episode_index == episode_index and e.edit_type == "delete"
            for e in self.pending_edits
        )

    def get_episode_trim(self, dataset_id: str, episode_index: int) -> tuple[int, int] | None:
        """Get trim range for an episode if it exists."""
        for e in self.pending_edits:
            if e.dataset_id == dataset_id and e.episode_index == episode_index and e.edit_type == "trim":
                return (e.params.get("start_frame", 0), e.params.get("end_frame", 0))
        return None

    def get_lock(self, dataset_id: str) -> asyncio.Lock:
        """Get or create an asyncio.Lock for a dataset."""
        if dataset_id not in self._dataset_locks:
            self._dataset_locks[dataset_id] = asyncio.Lock()
        return self._dataset_locks[dataset_id]

    def is_locked(self, dataset_id: str) -> bool:
        """Check if a dataset is currently locked."""
        lock = self._dataset_locks.get(dataset_id)
        return lock is not None and lock.locked()

    def active_hub_job_for(self, dataset_id: str) -> HubJobState | None:
        """Return the in-flight Hub job for ``dataset_id``, if any.

        "In-flight" = ``status in {"pending", "running"}``. Completed,
        cancelled, and failed jobs stay in ``hub_jobs`` for the Transfers
        tray but don't block a new transfer on the same dataset.
        """
        for job in self.hub_jobs.values():
            if job.dataset_id == dataset_id and job.status in ("pending", "running"):
                return job
        return None

    def gc_finished_hub_jobs(self, *, max_age_s: float = 1800.0) -> int:
        """Drop completed / cancelled / failed jobs older than ``max_age_s``.

        Active jobs (pending / running) are never collected; their state
        is still being mutated by the executor thread. Returns the number
        of jobs dropped — caller can log this if non-zero.

        Called opportunistically from the list endpoint so the registry
        doesn't grow without bound across a long GUI session.
        """
        import time as _time

        now = _time.time()
        terminal = {"complete", "cancelled", "failed"}
        to_drop = [
            jid
            for jid, j in self.hub_jobs.items()
            if j.status in terminal and j.finished_at is not None and (now - j.finished_at) > max_age_s
        ]
        for jid in to_drop:
            del self.hub_jobs[jid]
        return len(to_drop)

    def discard_lock(self, dataset_id: str) -> None:
        """Drop the asyncio.Lock for a dataset that's no longer open.

        Without this, every distinct ``dataset_id`` ever seen in a session
        accumulates an entry in ``_dataset_locks`` even after the dataset
        is closed. Safe to call when no lock exists (no-op).
        """
        self._dataset_locks.pop(dataset_id, None)


# ============================================================================
# Persistence Functions
# ============================================================================


def _edit_to_dict(edit: PendingEdit) -> dict[str, Any]:
    """Convert a PendingEdit to a JSON-serializable dict."""
    return {
        "edit_type": edit.edit_type,
        "episode_index": edit.episode_index,
        "params": edit.params,
        "created_at": edit.created_at.isoformat(),
    }


def _dict_to_edit(d: dict[str, Any], dataset_id: str) -> PendingEdit:
    """Convert a dict back to a PendingEdit."""
    return PendingEdit(
        edit_type=d["edit_type"],
        dataset_id=dataset_id,
        episode_index=d["episode_index"],
        params=d.get("params", {}),
        created_at=datetime.fromisoformat(d["created_at"]) if "created_at" in d else datetime.now(),
    )


def save_edits_to_file(root: Path, edits: list[PendingEdit]) -> None:
    """Save pending edits to the dataset's edits file.

    Args:
        root: Dataset root directory.
        edits: List of pending edits to save.
    """
    edits_file = root / EDITS_FILENAME

    if not edits:
        # No edits - remove file if it exists
        if edits_file.exists():
            # safe-destruct: our edits metadata file
            edits_file.unlink()
            logger.info(f"Removed empty edits file: {edits_file}")
        return

    data = {
        "version": EDITS_FILE_VERSION,
        "edits": [_edit_to_dict(e) for e in edits],
    }

    # Atomic write: stage to a sibling .tmp and rename. A power-loss /
    # process-kill between `write_text` chunks would otherwise leave the
    # user with a half-written JSON the next session refuses to parse —
    # which means silently-lost pending edits even though the user saw
    # the "Saved …" toast.
    tmp_file = edits_file.with_suffix(edits_file.suffix + ".tmp")
    tmp_file.write_text(json.dumps(data, indent=2))
    os.replace(tmp_file, edits_file)
    logger.info(f"Saved {len(edits)} edits to {edits_file}")


def load_edits_from_file(root: Path, dataset_id: str) -> list[PendingEdit]:
    """Load pending edits from the dataset's edits file.

    Args:
        root: Dataset root directory.
        dataset_id: Dataset ID to associate with loaded edits.

    Returns:
        List of pending edits, empty if file doesn't exist.
    """
    edits_file = root / EDITS_FILENAME

    if not edits_file.exists():
        return []

    try:
        data = json.loads(edits_file.read_text())
        version = data.get("version", 1)

        if version > EDITS_FILE_VERSION:
            logger.warning(f"Edits file version {version} is newer than supported {EDITS_FILE_VERSION}")

        edits = [_dict_to_edit(d, dataset_id) for d in data.get("edits", [])]
        logger.info(f"Loaded {len(edits)} edits from {edits_file}")
        return edits

    except (json.JSONDecodeError, KeyError) as e:
        logger.error(f"Failed to load edits from {edits_file}: {e}")
        return []


def clear_edits_file(root: Path) -> None:
    """Remove the edits file after edits are applied.

    Args:
        root: Dataset root directory.
    """
    edits_file = root / EDITS_FILENAME
    if edits_file.exists():
        # safe-destruct: our edits metadata file
        edits_file.unlink()
        logger.info(f"Cleared edits file: {edits_file}")

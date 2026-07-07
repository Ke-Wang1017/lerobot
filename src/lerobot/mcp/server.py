"""FastMCP server exposing LeRobot datasets to AI agents over stdio.

Tools (v1):
  list_datasets               — discover datasets under $HF_LEROBOT_HOME
  get_dataset_info            — schema, feature shapes, length stats
  list_episodes               — paginated episode listing with summary fields
  get_episode_summary         — length, tasks, terminal-state hints
  get_frame                   — one image at (episode, frame_idx, camera)
  tag_episode                 — write a sidecar annotation
  get_episode_tags            — read sidecar annotations

Design intent: every tool takes logical IDs (``repo_id``, ``episode_id``),
never file paths. The server resolves to disk internally so that the same
surface works locally today and over a network later.
"""

from __future__ import annotations

import io
import logging
from collections import OrderedDict
from pathlib import Path
from typing import Any

import torch
from mcp.server.fastmcp import FastMCP, Image
from mcp.server.fastmcp.server import AuthSettings
from PIL import Image as PILImage

from lerobot.datasets.dataset_metadata import LeRobotDatasetMetadata
from lerobot.datasets.video_utils import decode_video_frames
from lerobot.utils.constants import HF_LEROBOT_HOME

from .auth import (
    SCOPE_COMMENT,
    SCOPE_EDIT,
    SCOPE_OPERATE,
    SCOPE_READ,
    LeRobotTokenVerifier,
    TokenStore,
    requires_scope,
)
from .bridge_tools import register_bridge_tools
from .store import AnnotationStore

logger = logging.getLogger("lerobot.mcp")

_META_CACHE_MAX = 32
_DEFAULT_DB_PATH = HF_LEROBOT_HOME / "_mcp_annotations.sqlite"


class ServerState:
    """Holds per-process state: dataset root, metadata cache, annotation store.

    A single instance is bound to the FastMCP app at startup. ``app_state``
    is a getter (zero-arg callable) returning the GUI's ``AppState`` when
    MCP is mounted inside the GUI process (unified deployment); edit-tier
    tools that write to the in-memory ``PendingEdit`` queue use it as the
    shared surface. Returns ``None`` in standalone ``lerobot-mcp serve``
    mode and during the GUI startup window before the AppState is
    constructed — edit-tier tools raise a descriptive error there.

    A getter rather than a value because in the unified mode, ``_mount_mcp``
    runs in ``run_server`` *before* the FastAPI ``startup_event`` allocates
    the AppState. Resolving lazily lets MCP attach early and pick up the
    live AppState on the first tool call.
    """

    def __init__(
        self,
        dataset_root: Path | None = None,
        db_path: Path | None = None,
        app_state: Any = None,
    ):
        self.dataset_root = Path(dataset_root) if dataset_root is not None else HF_LEROBOT_HOME
        self.store = AnnotationStore(db_path if db_path is not None else _DEFAULT_DB_PATH)
        self._meta_cache: OrderedDict[str, LeRobotDatasetMetadata] = OrderedDict()
        # Accept either a callable (preferred — handles startup ordering) or a
        # bare AppState value (test convenience). Normalised to a getter.
        if app_state is None:
            self._app_state_getter = lambda: None
        elif callable(app_state):
            self._app_state_getter = app_state
        else:
            self._app_state_getter = lambda: app_state

    @property
    def app_state(self):
        return self._app_state_getter()

    def discover_datasets(self) -> list[tuple[str, Path]]:
        """Return ``[(repo_id, local_root), ...]`` for datasets under ``dataset_root``."""
        out: list[tuple[str, Path]] = []
        root = self.dataset_root
        if not root.is_dir():
            return out
        for org_dir in sorted(root.iterdir()):
            if not org_dir.is_dir():
                continue
            for ds_dir in sorted(org_dir.iterdir()):
                if not ds_dir.is_dir():
                    continue
                if (ds_dir / "meta" / "info.json").is_file():
                    out.append((f"{org_dir.name}/{ds_dir.name}", ds_dir))
        return out

    def resolve(self, repo_id: str) -> Path:
        root = self.dataset_root / repo_id
        if not (root / "meta" / "info.json").is_file():
            raise ValueError(f"Dataset not found locally: {repo_id!r} (looked in {root})")
        return root

    def get_meta(self, repo_id: str) -> LeRobotDatasetMetadata:
        """Cached metadata loader; LRU-evicts past ``_META_CACHE_MAX``."""
        if repo_id in self._meta_cache:
            self._meta_cache.move_to_end(repo_id)
            return self._meta_cache[repo_id]
        root = self.resolve(repo_id)
        meta = LeRobotDatasetMetadata(repo_id=repo_id, root=root)
        self._meta_cache[repo_id] = meta
        while len(self._meta_cache) > _META_CACHE_MAX:
            self._meta_cache.popitem(last=False)
        return meta


# Bound at server build time; tools close over this via the global ``state``.
state: ServerState | None = None


def _state() -> ServerState:
    assert state is not None, "ServerState not initialized (call build_server first)"
    return state


def _episode_length(meta: LeRobotDatasetMetadata, ep_id: int) -> int:
    """Frame count of an episode, derived from the episodes metadata table."""
    row = meta.episodes[ep_id]
    if "length" in row and row["length"] is not None:
        return int(row["length"])
    return int(row["dataset_to_index"] - row["dataset_from_index"])


def _normalize_frame_idx(frame_idx: int, ep_length: int) -> int:
    """Resolve negative indices (``-1`` → last frame). Bounds-checked."""
    if frame_idx < 0:
        frame_idx = ep_length + frame_idx
    if not (0 <= frame_idx < ep_length):
        raise ValueError(f"frame_idx {frame_idx} out of range [0, {ep_length})")
    return frame_idx


def _format_http_error(exc: Any) -> str:
    """Render a FastAPI ``HTTPException`` as a clean message for an MCP error.

    ``HTTPException.detail`` is often a plain string but can be a dict
    (the conflict tools surface structured detail). For MCP, a single
    one-line message reads better than a Python-repr of a dict — we
    flatten ``{"message": "..."}`` if present, else ``str()`` the detail.
    """
    detail = getattr(exc, "detail", None)
    if isinstance(detail, dict):
        if "message" in detail:
            return str(detail["message"])
        return str(detail)
    return str(detail if detail is not None else exc)


def _tensor_to_image(frame: torch.Tensor) -> Image:
    """Encode a single decoded frame (C,H,W float[0,1] or uint8) as a PNG Image."""
    if frame.ndim == 4:
        # decode_video_frames returns (T, C, H, W); take the single frame
        frame = frame[0]
    assert frame.ndim == 3, f"unexpected frame shape {tuple(frame.shape)}"
    if frame.dtype != torch.uint8:
        frame = (frame.clamp(0.0, 1.0) * 255.0).to(torch.uint8)
    # (C, H, W) → (H, W, C)
    arr = frame.permute(1, 2, 0).cpu().numpy()
    pil = PILImage.fromarray(arr)
    buf = io.BytesIO()
    pil.save(buf, format="PNG", optimize=False)
    return Image(data=buf.getvalue(), format="png")


# ── Build & register tools ─────────────────────────────────────────────────


def build_server(
    dataset_root: Path | None = None,
    db_path: Path | None = None,
    token_store: TokenStore | None = None,
    host: str = "127.0.0.1",
    port: int = 7861,
    streamable_http_path: str | None = None,
    app_state: Any = None,
) -> FastMCP:
    """Construct a FastMCP server with the tool surface registered.

    Args:
        dataset_root: Override the dataset discovery root. Defaults to
            ``$HF_LEROBOT_HOME``.
        db_path: Override the annotation sidecar location. Defaults to
            ``$HF_LEROBOT_HOME/_mcp_annotations.sqlite``.
        token_store: If provided, the server runs with bearer-token auth
            (required for HTTP transport). The verifier checks every
            request and rejects with 401 if the bearer is missing or
            unknown, and 403 if it lacks the floor ``read`` scope. When
            ``None`` (stdio mode), no auth is enforced.
        host, port: Bind address used by HTTP transports. Ignored by stdio.
        streamable_http_path: Path the streamable-http transport listens on
            inside the FastMCP ASGI app. Default ``None`` keeps FastMCP's
            built-in ``/mcp``. Set to ``"/"`` when mounting the app under a
            FastAPI route (e.g., the GUI mounts at ``/mcp`` and wants the
            inner transport at its own root so the final URL stays
            ``<host>/mcp``, not ``<host>/mcp/mcp``).
    """
    global state
    state = ServerState(dataset_root=dataset_root, db_path=db_path, app_state=app_state)

    fastmcp_kwargs: dict[str, Any] = {"host": host, "port": port}
    if streamable_http_path is not None:
        fastmcp_kwargs["streamable_http_path"] = streamable_http_path
    if token_store is not None:
        base_url = f"http://{host}:{port}"
        fastmcp_kwargs["token_verifier"] = LeRobotTokenVerifier(token_store)
        fastmcp_kwargs["auth"] = AuthSettings(
            issuer_url=base_url,  # we issue our own tokens out-of-band
            resource_server_url=base_url,
            required_scopes=[SCOPE_READ],  # floor; per-tool decorators enforce annotate/operate
        )

    mcp = FastMCP("lerobot", **fastmcp_kwargs)

    @mcp.tool()
    @requires_scope(SCOPE_READ)
    def list_datasets() -> dict[str, Any]:
        """List datasets discoverable under the LeRobot dataset cache.

        Returns each dataset's ``repo_id``, episode count, fps, and camera keys
        (cheap — reads only ``meta/info.json``).
        """
        s = _state()
        items: list[dict[str, Any]] = []
        for repo_id, root in s.discover_datasets():
            try:
                info_path = root / "meta" / "info.json"
                import json as _json

                info = _json.loads(info_path.read_text())
                features = info.get("features", {})
                cameras = [k for k, ft in features.items() if ft.get("dtype") in ("video", "image")]
                items.append(
                    {
                        "repo_id": repo_id,
                        "n_episodes": int(info.get("total_episodes", 0)),
                        "n_frames": int(info.get("total_frames", 0)),
                        "fps": int(info.get("fps", 0)),
                        "robot_type": info.get("robot_type"),
                        "cameras": cameras,
                    }
                )
            except Exception as e:  # noqa: BLE001 — surface to agent
                items.append({"repo_id": repo_id, "error": f"{type(e).__name__}: {e}"})
        return {"datasets": items, "root": str(s.dataset_root)}

    @mcp.tool()
    @requires_scope(SCOPE_READ)
    def get_dataset_info(repo_id: str) -> dict[str, Any]:
        """Detailed schema and stats for one dataset.

        Args:
            repo_id: e.g. ``'thewisp/cylinder_ring_assembly'``.
        """
        meta = _state().get_meta(repo_id)
        features = {
            k: {"dtype": ft.get("dtype"), "shape": list(ft.get("shape", [])), "names": ft.get("names")}
            for k, ft in meta.features.items()
        }
        ep_lengths = [_episode_length(meta, i) for i in range(min(meta.total_episodes, 256))]
        length_stats = (
            {
                "min": int(min(ep_lengths)),
                "max": int(max(ep_lengths)),
                "mean": float(sum(ep_lengths) / len(ep_lengths)),
                "sampled": len(ep_lengths),
            }
            if ep_lengths
            else {}
        )
        return {
            "repo_id": repo_id,
            "robot_type": meta.robot_type,
            "fps": meta.fps,
            "total_episodes": meta.total_episodes,
            "total_frames": meta.total_frames,
            "total_tasks": meta.total_tasks,
            "cameras": meta.camera_keys,
            "image_keys": meta.image_keys,
            "video_keys": meta.video_keys,
            "features": features,
            "episode_length_stats": length_stats,
        }

    @mcp.tool()
    @requires_scope(SCOPE_READ)
    def list_episodes(repo_id: str, limit: int = 50, offset: int = 0) -> dict[str, Any]:
        """List episodes with per-episode summary fields (paginated).

        Each entry includes ``episode_id``, ``length``, ``tasks``, and any
        sidecar tags previously written via ``tag_episode``.

        Args:
            repo_id: Dataset id.
            limit: Max episodes to return (1..500). Default 50.
            offset: Starting episode index.
        """
        if not 1 <= limit <= 500:
            raise ValueError(f"limit must be in [1, 500]; got {limit}")
        if offset < 0:
            raise ValueError(f"offset must be >= 0; got {offset}")
        meta = _state().get_meta(repo_id)
        total = meta.total_episodes
        end = min(offset + limit, total)
        episodes: list[dict[str, Any]] = []
        for i in range(offset, end):
            row = meta.episodes[i]
            ep_summary: dict[str, Any] = {
                "episode_id": i,
                "length": _episode_length(meta, i),
            }
            if "tasks" in row:
                ep_summary["tasks"] = row["tasks"]
            ep_summary["tags"] = _state().store.get_tags(repo_id, i)
            episodes.append(ep_summary)
        return {
            "repo_id": repo_id,
            "total": total,
            "offset": offset,
            "returned": len(episodes),
            "episodes": episodes,
        }

    @mcp.tool()
    @requires_scope(SCOPE_READ)
    def get_episode_summary(repo_id: str, episode_id: int) -> dict[str, Any]:
        """Per-episode detail: length, tasks, terminal action/state, tags.

        Cheap relative to ``get_frame`` — reads only the episode metadata row
        plus the sidecar.
        """
        meta = _state().get_meta(repo_id)
        if not (0 <= episode_id < meta.total_episodes):
            raise ValueError(f"episode_id {episode_id} out of range [0, {meta.total_episodes})")
        row = meta.episodes[episode_id]
        length = _episode_length(meta, episode_id)
        # Surface anything available without doing heavy parquet reads.
        keys_of_interest = {"tasks", "dataset_from_index", "dataset_to_index"}
        extras = {k: row[k] for k in keys_of_interest if k in row}
        return {
            "repo_id": repo_id,
            "episode_id": episode_id,
            "length": length,
            "duration_s": length / meta.fps if meta.fps else None,
            "cameras": meta.camera_keys,
            "tags": _state().store.get_tags(repo_id, episode_id),
            **extras,
        }

    @mcp.tool()
    @requires_scope(SCOPE_READ)
    def get_frame(repo_id: str, episode_id: int, frame_idx: int, camera: str) -> Image:
        """One image from a video stream of a single episode.

        Args:
            repo_id: Dataset id.
            episode_id: Zero-based episode index.
            frame_idx: Zero-based frame index within the episode. Negative
                values count from the end (``-1`` = last frame).
            camera: One of the dataset's camera keys
                (see ``get_dataset_info(...).cameras``).

        Returns:
            A PNG image of that frame.
        """
        meta = _state().get_meta(repo_id)
        if not (0 <= episode_id < meta.total_episodes):
            raise ValueError(f"episode_id {episode_id} out of range [0, {meta.total_episodes})")
        if camera not in meta.camera_keys:
            raise ValueError(f"unknown camera {camera!r}; valid: {meta.camera_keys}")
        length = _episode_length(meta, episode_id)
        frame_idx = _normalize_frame_idx(frame_idx, length)

        if camera in meta.video_keys:
            rel_path = meta.get_video_file_path(episode_id, camera)
            video_path = meta.root / rel_path
            ts = frame_idx / meta.fps
            tol = 1.0 / max(meta.fps, 1)
            frames = decode_video_frames(video_path, [ts], tolerance_s=tol, return_uint8=True)
            return _tensor_to_image(frames)
        # image-feature path: not common in current datasets, defer to parquet
        raise NotImplementedError(
            f"Camera {camera!r} is an image-feature, not a video-feature. "
            "Image-feature frame access is not yet implemented in v1."
        )

    @mcp.tool()
    @requires_scope(SCOPE_COMMENT)
    def tag_episode(repo_id: str, episode_id: int, key: str, value: Any) -> dict[str, Any]:
        """Write a sidecar comment on an episode. Does not modify the canonical dataset.

        Comments live in a sidecar SQLite shared across all AI tools on this
        host and persist across sessions — leave a note today and the next AI
        session can read it. Comments are eventually surfaced in the GUI too.

        Last-write-wins on the same ``(episode_id, key)``. The response
        carries ``overwrote`` (bool) and ``previous_value`` so the agent
        can tell whether it just clobbered an existing comment.

        Args:
            repo_id: Dataset id.
            episode_id: Episode to comment on.
            key: Comment name (e.g. ``'outcome'``).
            value: JSON-serializable value (string, number, bool, list, dict).

        Returns the full tag dict for the episode plus an explicit
        ``overwrote`` / ``previous_value`` so the actual effect is visible.
        """
        s = _state()
        meta = s.get_meta(repo_id)  # also validates dataset exists
        if not (0 <= episode_id < meta.total_episodes):
            raise ValueError(f"episode_id {episode_id} out of range [0, {meta.total_episodes})")
        existing = s.store.get_tags(repo_id, episode_id)
        had_key = key in existing
        previous_value = existing.get(key) if had_key else None
        s.store.set_tag(repo_id, episode_id, key, value)
        return {
            "repo_id": repo_id,
            "episode_id": episode_id,
            "key": key,
            "overwrote": had_key,
            "previous_value": previous_value,
            "tags": s.store.get_tags(repo_id, episode_id),
        }

    @mcp.tool()
    @requires_scope(SCOPE_COMMENT)
    def delete_episode_tag(repo_id: str, episode_id: int, key: str) -> dict[str, Any]:
        """Delete one sidecar comment from an episode.

        Returns ``{deleted: bool, tags: <remaining>}`` — ``deleted=False`` if
        the comment didn't exist (idempotent no-op, no error).
        """
        s = _state()
        meta = s.get_meta(repo_id)
        if not (0 <= episode_id < meta.total_episodes):
            raise ValueError(f"episode_id {episode_id} out of range [0, {meta.total_episodes})")
        deleted = s.store.delete_tag(repo_id, episode_id, key)
        return {
            "repo_id": repo_id,
            "episode_id": episode_id,
            "deleted": deleted,
            "tags": s.store.get_tags(repo_id, episode_id),
        }

    @mcp.tool()
    @requires_scope(SCOPE_READ)
    def get_episode_tags(repo_id: str, episode_id: int) -> dict[str, Any]:
        """Read all sidecar comments for an episode."""
        s = _state()
        meta = s.get_meta(repo_id)
        if not (0 <= episode_id < meta.total_episodes):
            raise ValueError(f"episode_id {episode_id} out of range [0, {meta.total_episodes})")
        return {"repo_id": repo_id, "episode_id": episode_id, "tags": s.store.get_tags(repo_id, episode_id)}

    @mcp.tool()
    @requires_scope(SCOPE_READ)
    def list_tagged_episodes(
        repo_id: str,
        key: str | None = None,
        value: Any = None,
    ) -> dict[str, Any]:
        """Reverse lookup over sidecar tags.

        Inverse of ``tag_episode`` / ``get_episode_tags``. Given a dataset,
        returns all episodes that carry any tag (or a specific
        ``key`` / ``key=value`` combination). Cheap — reads only the
        sidecar SQLite, no parquet or video decode.

        Workflow: tag a batch of episodes by analysis result
        (``outcome=success``, ``gripper_miss=true``), then later get
        the list of matching episodes back to act on (delete, trim,
        merge into a curated dataset, ...).

        Args:
            repo_id: Dataset id.
            key: If given, only return episodes that have this tag set.
                If ``None``, return every episode that has ANY tag.
            value: If both ``key`` AND ``value`` are given, additionally
                filter to episodes whose ``key`` matches ``value``
                exactly. Ignored when ``key`` is None.

        Returns ``{"repo_id", "key", "value", "episodes": [...],
        "total": N}``. Each episode entry is
        ``{"episode_id"}`` when ``key`` is None, or
        ``{"episode_id", "value", "set_at"}`` when ``key`` is given.
        """
        s = _state()
        # Validate the dataset exists so a typo doesn't silently return
        # an empty list — same shape as the other sidecar tools.
        s.get_meta(repo_id)
        rows = s.store.list_tagged_episodes(repo_id, key=key)
        if key is not None and value is not None:
            rows = [r for r in rows if r.get("value") == value]
        return {
            "repo_id": repo_id,
            "key": key,
            "value": value,
            "episodes": rows,
            "total": len(rows),
        }

    @mcp.tool()
    @requires_scope(SCOPE_READ)
    async def get_feature_series(
        repo_id: str,
        episode_id: int,
        features: list[str] | None = None,
    ) -> dict[str, Any]:
        """Per-frame time series of one or more features for one episode.

        Returns the same data the GUI's timeline plots in the Inspector
        (Run tab → episode → feature row). Powers programmatic episode
        analysis without round-tripping ``get_frame`` per index.

        Image / video features are rejected — fetch those via
        ``get_frame`` instead.

        Args:
            repo_id: Dataset id (must be opened in the GUI's AppState
                — this tool reads parquet via the dataset object).
            episode_id: Episode within the dataset.
            features: List of feature names to return. Omit / pass
                ``None`` to get every per-frame feature except
                image / video. Synthetic decoded views (``'task'``,
                ``'subtask'``) are supported — they return the
                lookup-decoded string per frame, not the index.

        Returns ``{"repo_id", "episode_index", "length", "series":
        {name: [v0, v1, ..., v_{N-1}]}}``. Vector-valued features come
        back as lists of lists (one inner list per frame).

        Raises if the dataset or episode doesn't exist, or if a
        requested feature is unknown or image/video.
        """
        from fastapi import HTTPException as _HTTPException

        from lerobot.gui.api.datasets import get_episode_feature_series

        app_state = _require_app_state()
        if repo_id not in app_state.datasets:
            raise ValueError(f"Dataset not opened in GUI: {repo_id}")
        # The handler accepts a comma-separated string; convert from list.
        features_arg = ",".join(features) if features else ""
        try:
            result = await get_episode_feature_series(repo_id, episode_id, features_arg)
        except _HTTPException as e:
            raise ValueError(_format_http_error(e)) from e
        return {"repo_id": repo_id, **result}

    # ── Edit-tier: dataset pending-edit pipeline ───────────────────────────
    # These tools mirror the GUI's PendingEdit model. The AI proposes
    # edits (delete / trim / set-feature), the GUI's existing UI shows
    # them inline (badge in the header + range highlights in the
    # Inspector), and apply_pending_edits writes them through to disk in
    # one transaction. The same /api/edits routes the operator drives
    # the GUI with are reading and writing the same queue — proposals
    # via MCP and via the GUI are interchangeable.
    #
    # Requires unified deployment (`app_state` is set). Standalone
    # ``lerobot-mcp serve`` raises a descriptive error.

    def _require_app_state():
        s = _state()
        if s.app_state is None:
            raise ValueError(
                "Edit-tier tools require unified GUI deployment "
                "(open the dataset in the LeRobot GUI; MCP must be mounted in the same process)."
            )
        return s.app_state

    @mcp.tool()
    @requires_scope(SCOPE_READ)
    def list_pending_edits(repo_id: str | None = None) -> dict[str, Any]:
        """List currently staged (unsaved) dataset edits.

        Edits are staged into a per-process queue that the GUI also reads.
        The same queue is used whether the proposal came from the GUI's
        UI or from another MCP call. ``apply_pending_edits`` flushes them
        to disk; ``discard_pending_edits`` drops them.

        Args:
            repo_id: If given, only edits for this dataset. Otherwise all.

        Returns ``{"edits": [...], "total": N}``. Each edit carries its
        ``index`` (stable while the queue isn't mutated), ``edit_type``
        (``delete`` / ``trim`` / ``feature_set``), the target episode,
        and the type-specific ``params``.
        """
        from lerobot.gui.api._edits_core import list_pending

        return list_pending(_require_app_state(), repo_id)

    @mcp.tool()
    @requires_scope(SCOPE_EDIT)
    def propose_delete_episode(repo_id: str, episode_id: int) -> dict[str, Any]:
        """Stage the deletion of one episode from a dataset.

        Doesn't touch disk — adds to the pending-edit queue. The episode
        stays visible in the GUI's tree with a deletion badge until the
        operator (or you, via ``apply_pending_edits``) commits.

        Args:
            repo_id: Dataset id.
            episode_id: Episode to delete.

        Returns the staged-edit confirmation. Raises if the episode is
        already marked for deletion or out of range. Idempotent only on
        the failure side — proposing the same delete twice is an error.
        """
        from lerobot.gui.api._edits_core import propose_delete

        return propose_delete(_require_app_state(), repo_id, episode_id)

    @mcp.tool()
    @requires_scope(SCOPE_EDIT)
    def propose_trim_episode(
        repo_id: str,
        episode_id: int,
        keep_from: int,
        keep_to: int,
    ) -> dict[str, Any]:
        """Stage a trim of one episode — keep frames ``[keep_from, keep_to)``, drop the rest.

        Param names anchor the semantics: you specify the window to
        **keep**, not the window to remove. Frames outside the kept
        window — ``[0, keep_from)`` on the head and ``[keep_to,
        episode_length)`` on the tail — are dropped at apply time. The
        episode does NOT split: one episode in, one (shorter) episode
        out, with length ``keep_to - keep_from``.

        There is no "remove a middle range" operation. If you wanted
        that, you'd have to delete the episode and re-record, or split
        manually offline; this tool can't produce two episodes from one.

        Replaces any prior trim on the same episode (last-write-wins).
        Passing the full range (``keep_from=0``, ``keep_to=episode_length``)
        clears an existing trim — useful for undoing a stage.

        Args:
            repo_id: Dataset id.
            episode_id: Episode to trim.
            keep_from: First frame to keep (inclusive). Set to 0 to only chop the tail.
            keep_to: First frame to drop (exclusive). Set to ``episode_length`` to only chop the head.

        Returns the staged-edit confirmation with an explicit "kept M of
        N frames, dropped (N − M)" message so the actual outcome is
        visible regardless of how the call was phrased.
        """
        from lerobot.gui.api._edits_core import propose_trim

        result = propose_trim(_require_app_state(), repo_id, episode_id, keep_from, keep_to)
        # _edits_core keeps the historical start_frame/end_frame names
        # for FastAPI compat. Translate to keep_from/keep_to for the MCP
        # response so the schema and the field names line up.
        result["keep_from"] = result.pop("start_frame")
        result["keep_to"] = result.pop("end_frame")
        return result

    @mcp.tool()
    @requires_scope(SCOPE_EDIT)
    def propose_set_feature(
        repo_id: str,
        episode_id: int,
        feature: str,
        frame_from: int,
        frame_to: int,
        value: Any,
        confirm_large: bool = False,
        confirm_overlap: bool = False,
    ) -> dict[str, Any]:
        """Stage a per-frame feature-value edit over the range ``[frame_from, frame_to)``.

        For tagging episode outcomes, labelling subtasks, correcting per-frame
        signals like ``reward`` or ``success``. The edit is staged; nothing
        is written to disk until ``apply_pending_edits``. Per-episode
        features (e.g. ``success``) are coerced to the full episode range
        automatically.

        Two retry-able conflicts surface via ``{"status": "conflict", "detail": {...}}``:
        - ``code=large_edit_confirmation_required`` — touches > 10k frames;
          retry with ``confirm_large=True``.
        - ``code=overlapping_edit`` — overlaps prior staged edit(s);
          retry with ``confirm_overlap=True`` to clip them (last-write-wins).

        Other validation failures (unknown feature, out-of-range, bounds
        violation) raise the tool with an error message.

        Args:
            repo_id: Dataset id.
            episode_id: Target episode.
            feature: Feature name (e.g. ``'reward'``, ``'success'``, ``'subtask'``).
                Read-only features (`action`, `observation.*`, video / image
                features) are rejected.
            frame_from: Inclusive lower frame (in episode-local indices).
            frame_to: Exclusive upper frame.
            value: JSON-serializable value. Categorical features accept
                string or int representations; declared min/max bounds
                are enforced.
            confirm_large: Acknowledge an edit touching > 10k frames.
            confirm_overlap: Clip prior overlapping edits on retry.

        Returns the staged-edit confirmation, possibly including
        ``coerced_range`` when the range was widened (per-episode feature).
        """
        from lerobot.gui.api._edits_core import EditConflictError, propose_feature_set

        try:
            return propose_feature_set(
                _require_app_state(),
                repo_id,
                episode_id,
                feature,
                frame_from,
                frame_to,
                value,
                confirm_large=confirm_large,
                confirm_overlap=confirm_overlap,
            )
        except EditConflictError as e:
            # Overlap / large-edit conflicts surface as structured data so
            # the AI can read ``detail.code`` and retry with the right
            # confirm_* flag. Validation errors (unknown feature, bad range,
            # bounds violation) still propagate as tool errors.
            return {"status": "conflict", "detail": e.detail}

    @mcp.tool()
    @requires_scope(SCOPE_EDIT)
    def discard_pending_edits(repo_id: str | None = None) -> dict[str, Any]:
        """Drop pending edits without applying them.

        Use to abandon a staging session. Does not touch on-disk data —
        only the in-memory queue and the per-dataset edit-state file are
        cleared.

        Args:
            repo_id: If given, scope the discard to one dataset.
                Otherwise discards every dataset's queue.

        Returns ``{"status": "ok", "discarded": N}``.
        """
        from lerobot.gui.api._edits_core import discard_pending

        return discard_pending(_require_app_state(), repo_id)

    @mcp.tool()
    @requires_scope(SCOPE_EDIT)
    async def apply_pending_edits(repo_id: str) -> dict[str, Any]:
        """Apply all staged edits for a dataset to disk in one transaction.

        Acquires the dataset lock, writes feature-set edits first (cell
        rewrites in existing rows), then trims (episode-bound changes),
        then deletes (row removal). Re-aggregates stats once at the end.
        Reloads the in-memory dataset object to reflect the new
        ``total_episodes`` / ``total_frames``. Clears the pending-edit
        queue + on-disk state file on success.

        This is the destructive step. Reversibility for dataset-on-disk
        comes from a clean Hub push beforehand or a filesystem-level
        snapshot — there's no built-in undo at the MCP layer.

        Args:
            repo_id: Dataset id.

        Returns ``{"status": "ok"|"partial", "applied": N, "errors": [...],
        "warnings": [...]}``. ``partial`` means data writes landed but
        stats recompute or verification surfaced issues — the user's
        edits ARE on disk.
        """
        app_state = _require_app_state()
        if repo_id not in app_state.datasets:
            raise ValueError(f"Dataset not opened in GUI: {repo_id}")

        # The lock is shared with the GUI's apply route — concurrent
        # apply attempts from MCP and the GUI's "Save" button serialize
        # cleanly without either having to know the other exists.
        lock = app_state.get_lock(repo_id)
        if lock.locked():
            raise ValueError(f"Dataset {repo_id} is busy (operation in progress)")
        async with lock:
            from lerobot.gui.api.edits import _apply_edits_locked

            return await _apply_edits_locked(repo_id)

    # ── Read-tier: Hugging Face Hub introspection ──────────────────────────
    # Read-only Hub surface. Tells the AI whether the user is logged in,
    # what a remote repo looks like, and how any in-flight transfers
    # (kicked off via the GUI's "Push to Hub" or `hub_start_upload` once
    # it lands) are progressing. No uploads / downloads here — those are
    # edit-tier and ship in a follow-up PR.
    #
    # ``hub_auth_status`` and ``hub_repo_info`` work even in standalone
    # ``lerobot-mcp serve`` mode (they don't need AppState). The two
    # job-tracking tools require unified deployment because the job
    # registry lives on the GUI's AppState.

    @mcp.tool()
    @requires_scope(SCOPE_READ)
    def hub_auth_status() -> dict[str, Any]:
        """Whether the host process has a working HF Hub login.

        Probes via ``HfApi().whoami()`` on every call (cheap; HF caches
        the token resolution). Any failure path — no token, expired
        token, network down — collapses to ``logged_in=False``. The
        agent should treat all of these uniformly: ask the operator to
        ``huggingface-cli login`` or set ``HF_TOKEN`` on the host.

        Returns ``{"logged_in": bool, "username": str | None}``.
        """
        from lerobot.gui.api._hub_core import get_auth_status

        return get_auth_status()

    @mcp.tool()
    @requires_scope(SCOPE_READ)
    def hub_repo_info(repo_id: str) -> dict[str, Any]:
        """Look up a dataset repo on the Hub.

        Reads basic repo metadata (visibility, last commit, file count,
        total size) plus best-effort ``meta/info.json`` enrichment for
        episode + frame counts. Useful before proposing a sync — the
        agent can compare local vs remote sizes / episode counts and
        warn the user before kicking off a big transfer.

        Args:
            repo_id: e.g. ``'thewisp/cylinder_ring_assembly'``.

        Returns ``{"exists": bool, ...}``. When ``exists=False`` (repo
        missing, private with no access, network down) only ``repo_id``
        and ``error`` are filled; the AI should branch on ``"exists"``
        rather than parse error text.
        """
        from lerobot.gui.api._hub_core import get_repo_info

        return get_repo_info(repo_id)

    @mcp.tool()
    @requires_scope(SCOPE_READ)
    async def hub_diff_local_vs_remote(
        dataset_id: str,
        hub_repo_id: str | None = None,
    ) -> dict[str, Any]:
        """Compare a local dataset against its Hub counterpart by file size + presence.

        Fast — no downloads, just an ``HfApi.dataset_info`` call to fetch
        the remote sibling list, then a local filesystem walk. Useful as
        a pre-flight before ``hub_start_upload`` / ``hub_start_download``:

        - ``modified``: files present on both sides but with different sizes
        - ``local_only``: files only on the local copy
        - ``remote_only``: files only on the Hub side
        - ``in_sync``: ``True`` when all three lists are empty

        Excludes ``.cache/`` and ``.lerobot*`` prefixes (transient state
        that shouldn't be synced).

        Args:
            dataset_id: Local dataset id (must be opened in the GUI).
            hub_repo_id: Override the remote repo name. Defaults to
                ``dataset.repo_id`` (the local id usually IS the
                desired remote name).

        Returns ``{"status": "ok", "in_sync": bool, "unchanged": int,
        "modified": [...], "local_only": [...], "remote_only": [...]}``,
        or ``{"status": "error", "message": "..."}`` when the remote
        repo can't be reached (missing, private without access, network
        down). The AI branches on ``status`` rather than parsing error
        text.
        """
        from fastapi import HTTPException as _HTTPException

        from lerobot.gui.api.datasets import hub_diff

        app_state = _require_app_state()
        if dataset_id not in app_state.datasets:
            raise ValueError(f"Dataset not opened in GUI: {dataset_id}")
        try:
            return await hub_diff(dataset_id, hub_repo_id)
        except _HTTPException as e:
            raise ValueError(_format_http_error(e)) from e

    @mcp.tool()
    @requires_scope(SCOPE_READ)
    def hub_list_jobs() -> dict[str, Any]:
        """List all Hub transfers the GUI is tracking, newest-first.

        Same source the GUI's Transfers tray reads — pending / running
        / complete / cancelled / failed jobs from all the operator's
        sessions on this host. Terminal jobs older than 30 minutes are
        opportunistically GC'd on this call.

        Returns ``{"jobs": [...], "total": N, "active": N_active}``.
        ``active`` is jobs in ``pending`` or ``running`` — the
        outcome-transparent count so the agent doesn't have to filter
        the array.

        Requires unified GUI deployment (job registry lives on AppState).
        """
        from lerobot.gui.api._hub_core import list_hub_jobs

        return list_hub_jobs(_require_app_state())

    # ── Read-tier: Run observation ─────────────────────────────────────────
    # Read-only view of the currently-active subprocess the GUI manages
    # (teleop / record / replay / training). Lets the AI babysit a long
    # job — poll status, tail output, watch latency/training metrics —
    # without being able to start or stop one. The mutating tools
    # (start_*, stop_current_run) ship as operate-tier in a follow-up
    # after explicit operator sign-off.

    @mcp.tool()
    @requires_scope(SCOPE_READ)
    def get_run_status() -> dict[str, Any]:
        """Current state of the GUI-managed subprocess.

        Three-way response — the AI branches on ``running``:

        - ``{"running": False, "command": None}`` — no subprocess at all.
        - ``{"running": False, "command": "<kind>", "returncode": int}``
          — a subprocess existed but has since exited.
        - ``{"running": True, "command": "<kind>", "pid": int}`` —
          subprocess is currently active.

        ``command`` is one of ``'teleoperate'``, ``'record'``,
        ``'replay'``, ``'hvla'`` — the same value the GUI's Run tab shows.
        """
        from lerobot.gui.api._run_core import get_run_status as _impl

        return _impl()

    @mcp.tool()
    @requires_scope(SCOPE_READ)
    def get_run_output(last_n: int = 200) -> dict[str, Any]:
        """Tail of the active subprocess's captured stdout/stderr.

        Snapshot — not streaming. For live tailing the GUI's
        ``/api/run/output`` SSE channel is the right surface; this MCP
        tool is for "poll, look at the last 200 lines, decide what to
        do."

        Args:
            last_n: Max lines to return (0..2000; buffer cap on the GUI
                side is 2000, so larger values silently cap there).

        Returns ``{"lines": [...], "total_buffered": N, "truncated": bool}``.
        ``truncated=true`` when the buffer held more lines than were
        returned.
        """
        from lerobot.gui.api._run_core import get_run_output_tail

        return get_run_output_tail(last_n=last_n)

    @mcp.tool()
    @requires_scope(SCOPE_READ)
    def get_latency_metrics(source: str = "teleop") -> dict[str, Any]:
        """Latest latency snapshot for the requested loop source.

        Each subprocess (teleop / record / HVLA inference) atomically
        replaces its ``latency_snapshot.json`` once per second; this
        tool reads the matching file. Returns the empty-snapshot stub
        when no snapshot exists yet (fresh session, source not running,
        or unknown source) so the AI can branch on ``n_records`` instead
        of error handling.

        Args:
            source: One of ``'teleop'``, ``'record'``, ``'hvla'``.
                Unknown sources return the empty stub.

        Returns ``{"n_records", "dropped_records", "overrun_ratio",
        "stages", "series"}``. ``stages`` carries per-stage timing stats
        (mean/p95/max); ``series`` is the windowed time-series the
        dashboard plots.
        """
        from lerobot.gui.api._run_core import get_latency_metrics as _impl

        return _impl(source)

    @mcp.tool()
    @requires_scope(SCOPE_READ)
    def get_rlt_metrics() -> dict[str, Any]:
        """RLT training metrics for the currently-active RLT run.

        Reads the ``metrics.json`` for the active RLT session. Returns
        the empty-metrics stub when no RLT run is active, so the agent
        branches on ``mode == "IDLE"`` instead of error-handling.

        Returns ``{"episode", "step_count", "buffer_size", "total_updates",
        "mode", "success_rate", "total_successes", "total_episodes",
        "series"}`` — ``mode == "IDLE"`` means no live RLT session.
        """
        from lerobot.gui.api._run_core import get_rlt_metrics as _impl

        return _impl()

    # ── Training orchestrator (the GUI's Model/Run training pipeline) ──────
    # A SEPARATE domain from get_run_status above (that's the single local GUI
    # subprocess). These wrap the run orchestrator, incl. ephemeral cloud
    # (Nebius) runs. start/stop are operate-tier: a cloud start spawns a
    # BILLABLE VM (torn down when the run ends).

    def _training_orch():
        """Run orchestrator, or None if the GUI training API isn't initialized
        (e.g. stdio MCP with no GUI server)."""
        try:
            from lerobot.gui.api.training import get_state

            orch, _ = get_state()
            return orch
        except (RuntimeError, ImportError):
            # RuntimeError: GUI training state not initialized (no GUI server).
            # ImportError: GUI/MCP extras not installed. Either way → unavailable.
            return None

    _training_unavailable = {
        "error": "training_unavailable",
        "detail": "the training orchestrator isn't running — start it with `lerobot-gui`",
    }

    @mcp.tool()
    @requires_scope(SCOPE_READ)
    def training_list_runs() -> dict[str, Any]:
        """List training runs (newest first).

        Each entry carries run_id, name (recipe_name), host_id, dataset_id,
        state (pending / running / completing / completed / failed / aborted),
        and timestamps. Use ``training_get_run`` for the full per-run snapshot.
        Distinct from ``get_run_status`` (the single local teleop/record/replay
        subprocess) — these are the orchestrator's training runs.
        """
        orch = _training_orch()
        if orch is None:
            return dict(_training_unavailable)
        from lerobot.gui.api.training import _run_to_dto

        return {"runs": [_run_to_dto(r).model_dump() for r in orch.list_runs()]}

    @mcp.tool()
    @requires_scope(SCOPE_READ)
    def training_get_run(run_id: str) -> dict[str, Any]:
        """Full snapshot of one training run.

        Returns ``run`` (state + timestamps), ``progress`` (step/total),
        ``metrics`` (per-logged-step loss/lr/…), ``checkpoints`` (step, path,
        sha256), ``events`` (the lifecycle timeline: spawn_started → vm_spawned
        → ssh_ready → prereqs_ready → started → artifacts_localized →
        vm_destroyed) and ``stderr_tail``. This is how you poll a run and
        confirm metrics streamed, the checkpoint was localized, and (for cloud
        runs) the VM was verifiably destroyed.

        Note: polling a terminal ephemeral run drives its cloud-VM teardown +
        checkpoint localization (identical to the GUI's own status poll) — a
        read, but with that one lifecycle side effect.

        Returns ``{"error": "unknown_run", ...}`` for an unknown run_id.
        """
        orch = _training_orch()
        if orch is None:
            return dict(_training_unavailable)
        from lerobot.gui.api.training import _snapshot_to_dto
        from lerobot.gui.training.orchestrator import UnknownRunError

        try:
            snap = orch.poll(run_id)
        except UnknownRunError:
            return {"error": "unknown_run", "run_id": run_id}
        return _snapshot_to_dto(snap).model_dump()

    @mcp.tool()
    @requires_scope(SCOPE_READ)
    def training_list_hosts() -> dict[str, Any]:
        """List configured training hosts.

        Each entry carries id, display_name, transport_kind (subprocess / ssh /
        ephemeral) and capabilities (provider, GPU, TTL). Feed a host ``id`` to
        ``training_start_run``.
        """
        orch = _training_orch()
        if orch is None:
            return dict(_training_unavailable)
        from lerobot.gui.api.training import list_hosts

        return {"hosts": [h.model_dump() for h in list_hosts()]}

    @mcp.tool()
    @requires_scope(SCOPE_OPERATE)
    def training_start_run(
        host_id: str,
        dataset_id: str,
        recipe_name: str,
        args: dict[str, Any] | None = None,
        idempotency_key: str | None = None,
    ) -> dict[str, Any]:
        """Start a training run on ``host_id``. OPERATE scope.

        On an ephemeral cloud host this spawns a BILLABLE VM that runs until the
        job finishes (or you call ``training_stop_run``) and is then torn down.

        ``dataset_id`` is the dataset to train on (e.g. ``lerobot/pusht``); it's
        forwarded into ``args`` as the recipe's dataset flag for you
        (``dataset.repo_id`` for the default lerobot-train recipe), so the
        common case needs only ``dataset_id``. ``recipe_name`` is the run's
        display label. ``args`` are extra lerobot-train flags as a flat dotted
        dict (e.g. ``{"policy.type": "act", "steps": 5000, "batch_size": 8}``);
        anything you set there wins over the auto-fill. Use
        ``training_list_hosts`` for valid host ids. Returns the created run —
        poll it with ``training_get_run``.

        Returns ``{"error": "unknown_host" | "host_busy", ...}`` on rejection.
        """
        orch = _training_orch()
        if orch is None:
            return dict(_training_unavailable)
        from lerobot.gui.api.training import _run_to_dto
        from lerobot.gui.training.orchestrator import HostBusyError, StartRequest, UnknownHostError

        # Mirror dataset_id into the recipe's dataset flag (the GUI form does
        # the same) so callers needn't know the dotted-key convention: default
        # lerobot-train wants dataset.repo_id; non-draccus recipes (HVLA, via
        # __recipe__) want bare dataset_repo_id. setdefault → caller wins.
        args = dict(args or {})
        if "__recipe__" in args:
            args.setdefault("dataset_repo_id", dataset_id)
        else:
            args.setdefault("dataset.repo_id", dataset_id)

        try:
            run = orch.start(
                StartRequest(
                    host_id=host_id,
                    recipe_name=recipe_name,
                    dataset_id=dataset_id,
                    args=args,
                    idempotency_key=idempotency_key,
                )
            )
        except UnknownHostError:
            return {"error": "unknown_host", "host_id": host_id}
        except HostBusyError as e:
            return {"error": "host_busy", "detail": str(e)}
        return _run_to_dto(run).model_dump()

    @mcp.tool()
    @requires_scope(SCOPE_OPERATE)
    def training_stop_run(run_id: str) -> dict[str, Any]:
        """Stop a training run. OPERATE scope.

        Idempotent on already-terminal runs. On an ephemeral host this also
        triggers teardown of the cloud VM. Returns the run's post-stop state.

        Returns ``{"error": "unknown_run", ...}`` for an unknown run_id.
        """
        orch = _training_orch()
        if orch is None:
            return dict(_training_unavailable)
        from lerobot.gui.api.training import _run_to_dto
        from lerobot.gui.training.orchestrator import UnknownRunError

        try:
            run = orch.stop(run_id)
        except UnknownRunError:
            return {"error": "unknown_run", "run_id": run_id}
        return _run_to_dto(run).model_dump()

    # ── Read-tier: Robots — profile and port introspection ─────────────────
    # **Strictly read-only.** No motor connections, no port-opening, no
    # camera streams. Profile files are JSON under
    # ``~/.config/lerobot/{robots,teleops}/``. The port-scan path uses
    # ``pyserial.list_ports.comports()`` which queries the kernel device
    # tree (sysfs / udev on Linux, registry on Windows, IOKit on macOS)
    # without opening any file descriptor on the device.
    #
    # The hardware-touching surface (`identify-arm`, `move-to-rest-position`,
    # trajectory record/replay, camera detection) is `operate`-tier and
    # ships in a separate PR after explicit operator sign-off.
    #
    # The helpers below import the GUI's existing private helpers rather
    # than re-implement. The helpers in question (`_list_profiles`,
    # `_read_profile`, `_collect_all_port_assignments`, `scan_ports`)
    # have no AppState coupling and have an established test suite; the
    # cross-private-import is documented here so future maintainers know
    # why the MCP tool body reaches into ``lerobot.gui.api.robot``.

    @mcp.tool()
    @requires_scope(SCOPE_READ)
    async def list_robot_profiles() -> dict[str, Any]:
        """List saved robot profiles.

        Profiles live under ``~/.config/lerobot/robots/*.json`` — operator
        configurations the GUI's Robot tab writes when a user sets up an
        arm or fleet. Each entry is the profile's display name + its
        config type (e.g. ``'bi_so107_follower'``); the full config is
        retrieved via ``get_robot_profile``.

        Returns ``{"profiles": [{"name", "type"}, ...], "total": N}``.
        """
        from lerobot.gui.api.robot import ROBOT_PROFILES_DIR, _list_profiles

        profiles = _list_profiles(ROBOT_PROFILES_DIR)
        return {"profiles": profiles, "total": len(profiles)}

    @mcp.tool()
    @requires_scope(SCOPE_READ)
    async def get_robot_profile(name: str) -> dict[str, Any]:
        """Full saved robot profile by name.

        Returns the on-disk JSON: ``{name, type, fields: {port,
        baudrate, ...}, cameras: {...}, rest_position: {...}}``. The
        ``fields`` dict contains the type-specific config for the
        underlying ``RobotConfig`` subclass.

        Args:
            name: Profile name as listed by ``list_robot_profiles``.

        Raises if no profile by that name exists.
        """
        from fastapi import HTTPException as _HTTPException

        from lerobot.gui.api.robot import ROBOT_PROFILES_DIR, _read_profile

        try:
            return _read_profile(ROBOT_PROFILES_DIR, name)
        except _HTTPException as e:
            # Helper raises HTTPException for compat with FastAPI; translate
            # 404 → ValueError so the AI sees a clean tool-error message.
            raise ValueError(f"Robot profile not found: {name!r}") from e

    @mcp.tool()
    @requires_scope(SCOPE_READ)
    async def list_teleop_profiles() -> dict[str, Any]:
        """List saved teleoperator profiles.

        Profiles live under ``~/.config/lerobot/teleops/*.json`` — operator
        configurations for teleop devices (leader arms, Quest 3 VR, etc.).
        Each entry is name + type; full config via ``get_teleop_profile``.

        Returns ``{"profiles": [{"name", "type"}, ...], "total": N}``.
        """
        from lerobot.gui.api.robot import TELEOP_PROFILES_DIR, _list_profiles

        profiles = _list_profiles(TELEOP_PROFILES_DIR)
        return {"profiles": profiles, "total": len(profiles)}

    @mcp.tool()
    @requires_scope(SCOPE_READ)
    async def get_teleop_profile(name: str) -> dict[str, Any]:
        """Full saved teleop profile by name.

        Returns the on-disk JSON. Raises if no profile by that name exists.
        """
        from fastapi import HTTPException as _HTTPException

        from lerobot.gui.api.robot import TELEOP_PROFILES_DIR, _read_profile

        try:
            return _read_profile(TELEOP_PROFILES_DIR, name)
        except _HTTPException as e:
            raise ValueError(f"Teleop profile not found: {name!r}") from e

    @mcp.tool()
    @requires_scope(SCOPE_READ)
    async def list_ports() -> dict[str, Any]:
        """Enumerate USB serial ports (kernel-level, no device opened).

        Uses ``pyserial.list_ports.comports()`` which queries the kernel
        device tree — does NOT open or signal any device. Filters to USB
        serial adapters (ttyACM*, ttyUSB* on Linux) and skips legacy /
        virtual / console TTYs. Each entry includes device path,
        description, manufacturer, and USB vid:pid when available.

        Returns ``{"ports": [{"path", "name", "manufacturer", "vid_pid"}, ...],
        "total": N}``.
        """
        from lerobot.gui.api.robot import scan_ports

        ports = await scan_ports()
        return {"ports": ports, "total": len(ports)}

    @mcp.tool()
    @requires_scope(SCOPE_READ)
    async def get_all_port_assignments() -> dict[str, Any]:
        """Cross-reference port paths against saved profile configs.

        For every saved profile, finds the port-typed fields in its
        config schema and lists their current values. Useful for spotting
        port collisions ("two profiles think the same /dev/ttyACM0 is
        their motor bus") or remembering which port a given profile is
        bound to.

        Returns ``{"assignments": [{"port", "profile_name",
        "profile_kind", "field_name"}, ...], "total": N}``. ``profile_kind``
        is ``'robot'`` or ``'teleop'``.
        """
        from lerobot.gui.api.robot import _collect_all_port_assignments

        assignments = _collect_all_port_assignments()
        return {"assignments": assignments, "total": len(assignments)}

    # ── Edit-tier: Dataset merge ───────────────────────────────────────────
    # Merge all episodes of one opened dataset into another. The source
    # is read-only; the target's parquet + videos grow in place. The
    # GUI's existing "Merge into…" action calls the same shared core.

    @mcp.tool()
    @requires_scope(SCOPE_READ)
    async def validate_dataset_merge(source_repo_id: str, target_repo_id: str) -> dict[str, Any]:
        """Check whether two opened datasets are schema-compatible for merge.

        Pure read — no disk writes, no locks taken. Lets the AI sanity
        check before proposing the actual merge. Compares fps,
        robot_type, and feature schemas (per-feature dtype / shape /
        names).

        Args:
            source_repo_id: Dataset to copy episodes FROM.
            target_repo_id: Dataset to merge episodes INTO (will grow).

        Returns ``{"compatible": bool, "mismatches": [...]}``.
        ``mismatches`` is empty when compatible; otherwise each entry
        names the conflicting ``field`` (``fps`` / ``robot_type`` /
        ``features``) plus the source vs. target values.

        Raises if either dataset isn't opened in the GUI.
        """
        from lerobot.gui.api._edits_core import check_merge_compat

        return check_merge_compat(_require_app_state(), source_repo_id, target_repo_id)

    @mcp.tool()
    @requires_scope(SCOPE_EDIT)
    async def merge_into_dataset(
        source_repo_id: str,
        target_repo_id: str,
        force: bool = False,
    ) -> dict[str, Any]:
        """Merge all episodes from ``source_repo_id`` INTO ``target_repo_id``.

        Disk-mutating: copies the source's parquet rows and video files
        into the target's directory tree and grows the target's
        metadata. Source is untouched on disk. Takes both dataset
        locks for the duration — concurrent edits / saves on either
        side serialize cleanly.

        Outcome transparency: the response carries before/after counts
        on the target so the caller can see exactly how much grew.

        Args:
            source_repo_id: Dataset to copy episodes FROM (read-only).
            target_repo_id: Dataset to merge episodes INTO (grows).
            force: If True, skip schema validation in
                ``dataset_tools.merge_into`` — only set this when you
                know the schemas are compatible despite the validator's
                complaints (e.g., a known-safe shape mismatch that
                ``validate_dataset_merge`` already reported). Default
                False.

        Returns:
            ``{"status", "source_id", "target_id",
            "source_episodes_merged", "source_frames_merged",
            "target_episodes_before", "target_episodes_after",
            "target_frames_before", "target_frames_after",
            "force_used"}``.

        Raises if either dataset isn't opened, if ``source_repo_id ==
        target_repo_id``, if either is busy, or if the schemas mismatch
        and ``force=False``.
        """
        from lerobot.gui.api._edits_core import merge_dataset_into

        return await merge_dataset_into(
            _require_app_state(),
            source_repo_id,
            target_repo_id,
            force=force,
        )

    # ── Edit-tier: Hub — start upload / download / cancel job ──────────────
    # The actual transfers run in spawned worker subprocesses; these
    # tools kick off the worker, the AI polls progress via hub_job_progress
    # (read-tier) or hub_list_jobs. Wrappers around the existing FastAPI
    # handlers; same cross-private-import pattern as the Robots edit
    # tools — the underlying handlers (`hub_upload`, `hub_download`,
    # `hub_progress_cancel`) are well-tested and AppState-coupled, so we
    # call them directly rather than refactor further.

    @mcp.tool()
    @requires_scope(SCOPE_EDIT)
    async def hub_start_upload(
        dataset_id: str,
        hub_repo_id: str | None = None,
        confirm_force: bool = False,
    ) -> dict[str, Any]:
        """Start uploading a local dataset to Hugging Face Hub.

        Spawns a background worker that pushes through the full PR
        pipeline (create_pull_request → upload_large_folder →
        super_squash_history → merge_pull_request). Returns
        ``{"job_id", "status": "started"}`` immediately; poll
        ``hub_job_progress`` for completion.

        **Incomplete-local-state guard.** Before spawning the worker,
        ``hub_jobs.check_upload_completeness`` compares the local files
        against the remote repo's siblings. If anything is present on
        the remote but missing locally — or there's an HF
        ``.incomplete`` resume marker — the call returns
        ``{"status": "conflict", "detail": {"code":
        "incomplete_local_state", "missing_locally": [...],
        "incomplete_locally": [...]}}`` instead of starting the upload.
        The check is mechanical (file-presence comparison), but the
        intent is to catch the half-downloaded-then-immediately-
        re-uploaded footgun, where the local copy is a worse-than-
        remote partial. The case never fires on a fresh upload (no
        remote repo to compare against).

        Recommended agent flow on this conflict:

        1. Surface ``missing_locally`` / ``incomplete_locally`` to the
           operator — they're real file paths.
        2. Default recovery: call ``hub_start_download`` to finish the
           partial copy, then retry the upload (no force flag needed).
        3. Only set ``confirm_force=True`` if the operator explicitly
           opts in — e.g., they intentionally stripped large files
           locally and want Hub to mirror the new (smaller) state.

        Args:
            dataset_id: Local dataset to upload (must be opened in the
                GUI's AppState).
            hub_repo_id: Override the remote repo name. Defaults to
                ``dataset.repo_id`` (the local id is usually the
                desired remote name too).
            confirm_force: Bypass the completeness check. See the
                recommended flow above — almost never the right first
                response to the conflict.

        Returns ``{"job_id", "status": "started"}`` on success, or
        ``{"status": "conflict", "detail": {...}}`` for the
        ``incomplete_local_state`` conflict and for active-transfer
        collisions on the same dataset.

        Raises on missing dataset, not-logged-in, or other
        non-recoverable errors.
        """
        from fastapi import HTTPException as _HTTPException

        from lerobot.gui.api.datasets import HubUploadRequest, hub_upload

        request = HubUploadRequest(repo_id=hub_repo_id, confirm_force=confirm_force)
        try:
            return await hub_upload(dataset_id, request)
        except _HTTPException as e:
            # Surface the recoverable `incomplete_local_state` conflict
            # as a structured dict so the AI can read detail.code and
            # retry with confirm_force=True. Other 409s (active job in
            # flight) also surface as conflicts.
            if e.status_code == 409 and isinstance(e.detail, dict):
                return {"status": "conflict", "detail": e.detail}
            raise ValueError(_format_http_error(e)) from e

    @mcp.tool()
    @requires_scope(SCOPE_EDIT)
    async def hub_start_download(
        dataset_id: str,
        hub_repo_id: str | None = None,
    ) -> dict[str, Any]:
        """Start downloading a dataset from Hugging Face Hub into the local copy.

        Uses ``snapshot_download`` which writes directly into
        ``dataset.root``; HF's etag-skip + ``.incomplete``-resume
        primitives handle resumability transparently — re-running an
        interrupted download picks up where it left off.

        Args:
            dataset_id: Local dataset id whose root the download lands
                in (must be opened in the GUI's AppState).
            hub_repo_id: Override the remote repo name. Defaults to
                ``dataset.repo_id``.

        Returns ``{"job_id", "status": "started"}``; poll
        ``hub_job_progress`` for completion. Raises on missing dataset,
        not-logged-in, or active-transfer collision.
        """
        from fastapi import HTTPException as _HTTPException

        from lerobot.gui.api.datasets import HubDownloadRequest, hub_download

        request = HubDownloadRequest(repo_id=hub_repo_id)
        try:
            return await hub_download(dataset_id, request)
        except _HTTPException as e:
            if e.status_code == 409 and isinstance(e.detail, dict):
                return {"status": "conflict", "detail": e.detail}
            raise ValueError(_format_http_error(e)) from e

    @mcp.tool()
    @requires_scope(SCOPE_EDIT)
    async def hub_cancel_job(job_id: str) -> dict[str, Any]:
        """Cancel an in-flight Hub transfer by sending SIGTERM to its worker.

        Idempotent — calling on an already-terminal job is a no-op
        (status was synthesized to ``failed`` if the worker was gone).

        Args:
            job_id: From ``hub_list_jobs`` or the response of
                ``hub_start_upload`` / ``hub_start_download``.

        Returns ``{"status": "cancel_requested", "job_id"}``. The actual
        cancellation is asynchronous — poll ``hub_job_progress`` to see
        the worker exit and the job's status flip to ``cancelled``.

        Raises if the job_id is unknown.
        """
        from fastapi import HTTPException as _HTTPException

        from lerobot.gui.api.datasets import hub_progress_cancel

        try:
            return await hub_progress_cancel(job_id)
        except _HTTPException as e:
            raise ValueError(_format_http_error(e)) from e

    # ── Edit-tier: Robots — profile CRUD + port assignment ─────────────────
    # File CRUD on the operator's ~/.config/lerobot/{robots,teleops}/*.json
    # profile files. Strictly file operations: no motor connections, no
    # camera handles opened, no serial ports touched. The hardware-using
    # helpers (identify_arm, detect_cameras_into_profile) are operate-tier
    # in a separate PR.
    #
    # Same cross-private-import to gui.api.robot as the read tools — the
    # underlying helpers (`_write_profile`, `_delete_profile`,
    # `_rename_profile`, `ProfileData`) are well-tested and AppState-free.

    def _build_profile_data(
        name: str,
        profile_type: str,
        fields: dict[str, Any] | None,
        cameras: dict[str, Any] | None,
        rest_position: dict[str, float] | None,
    ):
        """Hand-build a ProfileData pydantic model from MCP arg dicts."""
        from lerobot.gui.api.robot import ProfileData

        return ProfileData(
            name=name,
            type=profile_type,
            fields=fields or {},
            cameras=cameras or {},
            rest_position=rest_position or {},
        )

    def _create_profile(
        profiles_dir,
        name: str,
        profile_type: str,
        fields: dict[str, Any] | None,
        cameras: dict[str, Any] | None,
        rest_position: dict[str, float] | None,
        kind: str,
    ) -> dict[str, Any]:
        from lerobot.gui.api.robot import _write_profile

        path = profiles_dir / f"{name}.json"
        if path.exists():
            raise ValueError(
                f"{kind.capitalize()} profile {name!r} already exists. "
                f"Use update_{kind}_profile to overwrite, or pick a different name."
            )
        data = _build_profile_data(name, profile_type, fields, cameras, rest_position)
        _write_profile(profiles_dir, data)
        return {
            "status": "ok",
            "created": True,
            "name": name,
            "type": profile_type,
            "path": str(path),
        }

    def _update_profile(
        profiles_dir,
        name: str,
        profile_type: str,
        fields: dict[str, Any] | None,
        cameras: dict[str, Any] | None,
        rest_position: dict[str, float] | None,
        kind: str,
    ) -> dict[str, Any]:
        import json as _json

        from lerobot.gui.api.robot import _write_profile

        path = profiles_dir / f"{name}.json"
        previous: dict[str, Any] | None = None
        existed = path.exists()
        if existed:
            try:
                previous = _json.loads(path.read_text())
            except Exception:  # noqa: BLE001
                previous = None
        data = _build_profile_data(name, profile_type, fields, cameras, rest_position)
        _write_profile(profiles_dir, data)
        return {
            "status": "ok",
            "name": name,
            "type": profile_type,
            "overwrote": existed,
            "previous_type": previous.get("type") if previous else None,
            "previous_fields_count": len(previous.get("fields", {})) if previous else 0,
        }

    def _delete_profile_tool(profiles_dir, name: str, kind: str) -> dict[str, Any]:
        from fastapi import HTTPException as _HTTPException

        from lerobot.gui.api.robot import _delete_profile, _read_profile

        try:
            removed = _read_profile(profiles_dir, name)
        except _HTTPException as e:
            raise ValueError(f"{kind.capitalize()} profile not found: {name!r}") from e
        # Echo back the on-disk shape so the agent (and any human reading
        # the chat transcript) can see what was lost. Avoids a silent
        # "deleted" without context.
        try:
            _delete_profile(profiles_dir, name)
        except _HTTPException as e:
            raise ValueError(str(e.detail)) from e
        return {
            "status": "ok",
            "deleted": True,
            "name": name,
            "removed_type": removed.get("type"),
            "removed_fields_count": len(removed.get("fields", {})),
        }

    def _rename_profile_tool(profiles_dir, old_name: str, new_name: str, kind: str) -> dict[str, Any]:
        from fastapi import HTTPException as _HTTPException

        from lerobot.gui.api.robot import _rename_profile

        try:
            _rename_profile(profiles_dir, old_name, new_name)
        except _HTTPException as e:
            if e.status_code == 404:
                raise ValueError(f"{kind.capitalize()} profile not found: {old_name!r}") from e
            if e.status_code == 409:
                raise ValueError(
                    f"{kind.capitalize()} profile {new_name!r} already exists "
                    "— pick a different name or delete the existing one first."
                ) from e
            raise ValueError(str(e.detail)) from e
        return {"status": "ok", "renamed": True, "from": old_name, "to": new_name}

    @mcp.tool()
    @requires_scope(SCOPE_EDIT)
    async def create_robot_profile(
        name: str,
        type: str,
        fields: dict[str, Any] | None = None,
        cameras: dict[str, Any] | None = None,
        rest_position: dict[str, float] | None = None,
    ) -> dict[str, Any]:
        """Create a new robot profile (writes a JSON file to disk).

        Fails if a profile with this name already exists — use
        ``update_robot_profile`` to overwrite. Profile schema follows
        the GUI's Robot tab (``RobotConfig`` subclass type + its
        type-specific fields + cameras + rest_position).

        Args:
            name: Unique profile name (filename will be ``<name>.json``).
            type: ``RobotConfig`` choice — e.g. ``'bi_so107_follower'``,
                ``'virtual_bi_so107'``. ``list_robot_profiles`` shows
                examples already saved; full schema via the GUI.
            fields: Type-specific config (e.g. ``{"port": "/dev/ttyACM0",
                "baudrate": 1000000}`` for a Feetech-backed arm).
            cameras: Camera config dict keyed by name.
            rest_position: Per-joint rest target.

        Returns ``{"status", "created": True, "name", "type", "path"}``.
        """
        from lerobot.gui.api.robot import ROBOT_PROFILES_DIR

        return _create_profile(ROBOT_PROFILES_DIR, name, type, fields, cameras, rest_position, "robot")

    @mcp.tool()
    @requires_scope(SCOPE_EDIT)
    async def update_robot_profile(
        name: str,
        type: str,
        fields: dict[str, Any] | None = None,
        cameras: dict[str, Any] | None = None,
        rest_position: dict[str, float] | None = None,
    ) -> dict[str, Any]:
        """Overwrite a robot profile (full-replace, not patch).

        Last-write-wins on the same name. The response carries
        ``overwrote: bool`` and ``previous_type`` / ``previous_fields_count``
        so the agent can see whether it just clobbered an existing
        profile and what shape it used to have.

        Args:
            name: Profile name (must exist for the response's
                ``overwrote`` to be ``True``; passing a missing name
                effectively creates it).
            type, fields, cameras, rest_position: Same as
                ``create_robot_profile``.

        Returns the staged-edit confirmation with overwrite metadata.
        """
        from lerobot.gui.api.robot import ROBOT_PROFILES_DIR

        return _update_profile(ROBOT_PROFILES_DIR, name, type, fields, cameras, rest_position, "robot")

    @mcp.tool()
    @requires_scope(SCOPE_EDIT)
    async def rename_robot_profile(old_name: str, new_name: str) -> dict[str, Any]:
        """Rename a robot profile.

        Raises if ``old_name`` doesn't exist or ``new_name`` already does.
        The profile's internal ``name`` field is updated too — both the
        filename and the JSON contents reflect the new name.

        Returns ``{"status": "ok", "renamed": True, "from": old, "to": new}``.
        """
        from lerobot.gui.api.robot import ROBOT_PROFILES_DIR

        return _rename_profile_tool(ROBOT_PROFILES_DIR, old_name, new_name, "robot")

    @mcp.tool()
    @requires_scope(SCOPE_EDIT)
    async def delete_robot_profile(name: str) -> dict[str, Any]:
        """Delete a robot profile (removes the JSON file from disk).

        The response echoes back ``removed_type`` and
        ``removed_fields_count`` so a transcript reader can tell what
        was lost — "robot profile gone" with no context is unhelpful
        for an audit log.

        Raises if no profile by that name exists.
        """
        from lerobot.gui.api.robot import ROBOT_PROFILES_DIR

        return _delete_profile_tool(ROBOT_PROFILES_DIR, name, "robot")

    @mcp.tool()
    @requires_scope(SCOPE_EDIT)
    async def create_teleop_profile(
        name: str,
        type: str,
        fields: dict[str, Any] | None = None,
        cameras: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Create a new teleop profile (writes a JSON file to disk).

        Same shape as ``create_robot_profile`` but for teleoperator
        configs (leader arms, Quest 3 VR, scripted EE, …).

        Returns ``{"status", "created": True, "name", "type", "path"}``.
        """
        from lerobot.gui.api.robot import TELEOP_PROFILES_DIR

        return _create_profile(TELEOP_PROFILES_DIR, name, type, fields, cameras, None, "teleop")

    @mcp.tool()
    @requires_scope(SCOPE_EDIT)
    async def update_teleop_profile(
        name: str,
        type: str,
        fields: dict[str, Any] | None = None,
        cameras: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Overwrite a teleop profile (full-replace).

        Response carries ``overwrote`` and ``previous_type`` /
        ``previous_fields_count`` for outcome transparency.
        """
        from lerobot.gui.api.robot import TELEOP_PROFILES_DIR

        return _update_profile(TELEOP_PROFILES_DIR, name, type, fields, cameras, None, "teleop")

    @mcp.tool()
    @requires_scope(SCOPE_EDIT)
    async def rename_teleop_profile(old_name: str, new_name: str) -> dict[str, Any]:
        """Rename a teleop profile."""
        from lerobot.gui.api.robot import TELEOP_PROFILES_DIR

        return _rename_profile_tool(TELEOP_PROFILES_DIR, old_name, new_name, "teleop")

    @mcp.tool()
    @requires_scope(SCOPE_EDIT)
    async def delete_teleop_profile(name: str) -> dict[str, Any]:
        """Delete a teleop profile."""
        from lerobot.gui.api.robot import TELEOP_PROFILES_DIR

        return _delete_profile_tool(TELEOP_PROFILES_DIR, name, "teleop")

    @mcp.tool()
    @requires_scope(SCOPE_EDIT)
    async def assign_port_to_arm(
        profile_name: str,
        port: str,
        field_name: str = "port",
        profile_kind: str = "robot",
    ) -> dict[str, Any]:
        """Set a port field on a saved profile to a specific device path.

        Convenience over ``update_robot_profile`` / ``update_teleop_profile``
        for the common case of "I just discovered which port is which arm,
        write it to the profile." Reads the existing profile, swaps the
        chosen field's value, writes it back.

        Outcome transparency: response carries ``previous_port`` so the AI
        can see whether the assignment changed anything or was a no-op.

        Args:
            profile_name: Name of the saved profile.
            port: Device path (e.g. ``'/dev/ttyACM0'``).
            field_name: Which port field to set (default ``'port'``).
                Some bi-arm profiles have ``port_left`` / ``port_right``.
            profile_kind: ``'robot'`` (default) or ``'teleop'``.

        Returns ``{"status", "name", "field", "previous_port",
        "new_port", "changed"}``.
        """
        from fastapi import HTTPException as _HTTPException

        from lerobot.gui.api.robot import (
            ROBOT_PROFILES_DIR,
            TELEOP_PROFILES_DIR,
            _read_profile,
            _write_profile,
        )

        if profile_kind not in ("robot", "teleop"):
            raise ValueError(f"profile_kind must be 'robot' or 'teleop'; got {profile_kind!r}")
        profiles_dir = ROBOT_PROFILES_DIR if profile_kind == "robot" else TELEOP_PROFILES_DIR
        try:
            current = _read_profile(profiles_dir, profile_name)
        except _HTTPException as e:
            raise ValueError(f"{profile_kind.capitalize()} profile not found: {profile_name!r}") from e

        previous_port = current.get("fields", {}).get(field_name)
        current.setdefault("fields", {})[field_name] = port
        # Build a ProfileData from the updated dict, then write through
        # the same atomic-rename path the GUI uses.
        data = _build_profile_data(
            profile_name,
            current.get("type", "unknown"),
            current.get("fields"),
            current.get("cameras"),
            current.get("rest_position"),
        )
        _write_profile(profiles_dir, data)
        return {
            "status": "ok",
            "name": profile_name,
            "field": field_name,
            "previous_port": previous_port,
            "new_port": port,
            "changed": previous_port != port,
        }

    @mcp.tool()
    @requires_scope(SCOPE_READ)
    def hub_job_progress(job_id: str) -> dict[str, Any]:
        """Snapshot of one Hub job — for polling a long-running transfer.

        For active jobs the snapshot is refreshed from the worker's
        progress JSON before being returned, so an agent that just
        kicked off a transfer can poll this without staleness.

        Args:
            job_id: From ``hub_list_jobs`` or from the response of the
                (future) ``hub_start_upload`` / ``hub_start_download``
                tool.

        Raises if ``job_id`` is unknown.
        """
        from lerobot.gui.api._hub_core import HubJobNotFoundError, get_job_progress

        try:
            return get_job_progress(_require_app_state(), job_id)
        except HubJobNotFoundError as e:
            raise ValueError(str(e)) from e

    # ── Read-tier: introspection ───────────────────────────────────────────
    # Self-description of the live MCP surface — what tools exist + what
    # scope the current caller has. Useful for AI agents driving the
    # server through raw streamable-http (which doesn't surface the tool
    # list as elegantly as the Claude Code / Codex IDE plugins do), and
    # for any agent that wants to verify its own privileges before
    # planning a multi-step workflow.

    @mcp.tool()
    @requires_scope(SCOPE_READ)
    def lerobot_list_tools() -> dict[str, Any]:
        """List every MCP tool the server exposes, with scope + description.

        Pure read — no state changes. The list is the live FastMCP
        registry, so it reflects whatever was attached during
        ``build_server`` (data tools + bridge tools + the introspection
        pair).

        Returns ``{"tools": [{"name", "description", "scope"}, ...],
        "total": N}``. ``scope`` is the floor required by
        ``@requires_scope`` on each tool (``read`` / ``comment`` /
        ``edit`` / ``operate``); fall-back to ``read`` for tools that
        skip the decorator (none today, but the field reads cleanly).
        ``description`` is the one-line summary from the docstring.
        """
        tools = []
        # FastMCP stores the registry in a private dict keyed by tool name.
        # The supported public read is ``mcp._tool_manager.list_tools()`` but
        # that's also internal. ``mcp.list_tools()`` is async, which forces
        # us into a sync-over-async dance — easier to introspect the
        # registry directly. Keys are stable in practice.
        registry = getattr(mcp, "_tool_manager", None)
        tool_dict = getattr(registry, "_tools", {}) if registry is not None else {}
        for name, tool in tool_dict.items():
            description = (tool.description or "").strip().split("\n")[0]
            scope = getattr(tool.fn, "_required_scope", SCOPE_READ)
            tools.append({"name": name, "description": description, "scope": scope})
        tools.sort(key=lambda t: (t["scope"], t["name"]))
        return {"tools": tools, "total": len(tools)}

    @mcp.tool()
    @requires_scope(SCOPE_READ)
    def lerobot_whoami() -> dict[str, Any]:
        """Report the calling client's identity + scopes on this MCP.

        Pure read. Use this to verify what the calling client can do
        before planning a workflow — e.g., decide whether to propose
        an ``edit``-tier dataset cleanup or fall back to a
        ``read``-only analysis.

        Returns ``{"client_id": str | None, "scopes": [...],
        "all_scopes": [...], "logged_in": bool}`` — ``scopes`` is what
        THIS token carries; ``all_scopes`` is the full hierarchy for
        reference; ``client_id`` is the token's display name from
        ``/ai_setup`` (None in stdio mode).

        When the server runs without auth (stdio mode), ``scopes``
        contains every tier and ``logged_in`` is False — there's no
        authentication context to narrow it down.
        """
        from .auth import ALL_SCOPES

        client_id: str | None = None
        logged_in = False
        try:
            from mcp.server.auth.middleware.auth_context import get_access_token

            token = get_access_token()
            if token is not None:
                scopes = list(token.scopes)
                client_id = token.client_id
                logged_in = True
            else:
                scopes = list(ALL_SCOPES)
        except Exception:  # noqa: BLE001 — stdio mode has no auth context
            scopes = list(ALL_SCOPES)
        return {
            "client_id": client_id,
            "scopes": scopes,
            "all_scopes": list(ALL_SCOPES),
            "logged_in": logged_in,
        }

    # Bridge tools (navigate_to / notify_user / highlight_in_viewer / set_filter)
    # are attached unconditionally; whether they actually deliver depends on
    # whether the GUI dispatch URL was configured via configure_bridge().
    register_bridge_tools(mcp)

    return mcp

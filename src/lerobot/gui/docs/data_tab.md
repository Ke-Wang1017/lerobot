# Data Tab Design

Design doc for the dataset visualization and editing tab.

---

## Core Features

### Dataset Management

- Load one or more LeRobot datasets
- Connect to HuggingFace Hub (authentication required)
- Open local datasets (drag and drop support)
- Create new datasets from the UI
- Tree view showing multiple datasets and their episodes

### Episode Visualization

- Main panel displays all episode features:
  - Camera views (multiple simultaneous video streams)
  - Actions (plotted as time series)
  - States (plotted as time series)
- Horizontal timeline with timestamps
- Play/pause controls, click-to-seek, frame-by-frame navigation

### Episode Editing

- Mark episodes as deleted (with confirmation)
- Reorder episodes (swap)
- Duplicate episodes
- Copy/move episodes between datasets (enables natural merging and splitting)
- Timeline trimming (iOS-style drag handles)

### Data Model

- Source of truth: locally cached dataset files
- In-memory working copy for edits
- Edits are local until explicit save
- UI navigation doesn't lose unsaved changes
- Global save operation (and later: undo/redo)

---

## Local vs Remote Dataset Philosophy

### Core Principle: Local-First Editing

```
+-----------------------------------------------------------+
|                    HuggingFace Hub                         |
|                   (remote storage)                         |
+-----------------------------------------------------------+
              ^ Upload                    v Download
              | (explicit,                | (explicit,
              |  overwrites remote)       |  overwrites local)
+-----------------------------------------------------------+
|                    Local Dataset                           |
|            (source of truth for editing)                   |
|                                                            |
|   - All edits happen here                                  |
|   - Reload = re-read from local disk only                  |
|   - User always knows the local path                       |
+-----------------------------------------------------------+
```

### Why No Sync/Merge with Hub?

1. **No version control** - LeRobot datasets don't have commit history or checksums
2. **No conflict resolution** - We can't reliably detect what changed or merge edits
3. **Explicit is better** - User decides direction: "replace local" or "replace remote"
4. **Simplicity** - No complex sync state to track or debug

### Dataset Opening Modes

| Mode              | Input              | Behavior                                  |
| ----------------- | ------------------ | ----------------------------------------- |
| **Open Local**    | `/path/to/dataset` | Work directly on local files              |
| **Open from Hub** | `user/repo_id`     | Download to local path, then work locally |

### Opening an incomplete or inconsistent local cache

Opening a local path never silently downloads from the Hub. When a dataset
can't be opened as-is, the open pre-check classifies _why_ and the dialog states
it faithfully — it never offers a download that wouldn't help.

| Failure                                                                                           | What the dialog does                                                   |
| ------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------- |
| **Metadata inconsistent** — `info.json`'s episode count disagrees with the episode-metadata table | States the mismatch; no download (it isn't a missing-files problem)    |
| **Missing files, repo not on the Hub**                                                            | Lists what's missing; _Download & Open_ is disabled (nothing to fetch) |
| **Missing files, repo on the Hub**                                                                | Lists what's missing; _Download & Open_ fetches them, then opens       |

![Metadata inconsistent — info.json and the episode table disagree, so no download is offered](images/open_dataset_metadata_inconsistent.png)

![Missing files, repo not on the Hub — Download & Open is disabled](images/open_dataset_missing_files_not_on_hub.png)

![Missing files, repo on the Hub — Download & Open is enabled](images/open_dataset_missing_files_on_hub.png)

### Reload Semantics

**"Reload" always means re-read from local disk** — never fetches from Hub. Used after applying edits or when an external tool modifies the dataset.

### HuggingFace Hub Operations

Both upload and download are **destructive** and require user confirmation:

- **Download**: "Replace local dataset with version from Hub? Local changes will be lost."
- **Upload**: "Push local dataset to Hub? Remote version will be overwritten."

---

## Technical Analysis: LeRobot Dataset Format

### Dataset Structure

```
dataset_root/
+-- meta/
|   +-- info.json              # fps, features, codec info
|   +-- tasks.parquet          # task definitions
|   +-- episodes/chunk-*/      # episode metadata (parquet)
+-- data/chunk-*/              # frame data (parquet, ~100MB chunks)
+-- videos/{camera}/chunk-*/   # video files (mp4, ~200MB chunks)
```

### Performance Characteristics

| Component              | Load Time          | Notes                             |
| ---------------------- | ------------------ | --------------------------------- |
| Metadata (info.json)   | <10ms              | Loaded once at init               |
| Episode info (parquet) | ~50ms              | Memory-mapped, fast lookups       |
| Frame data (parquet)   | ~1ms/frame         | Memory-mapped, snappy compression |
| **Video decode**       | **50-500ms/frame** | **THE BOTTLENECK**                |

### Video Decoding Bottleneck

- Each frame request triggers `decode_video_frames()` synchronously
- Video containers (MP4) require seeking to keyframes (every 1-2s) and decoding forward
- torchcodec: 50-100ms (preferred, GPU support); pyav: 200-500ms (fallback)

---

## Architecture

### Python Backend (FastAPI) + Web Frontend

**Rationale:**

1. Must use LeRobot Python code — no rewriting dataset logic
2. Video decoding stays in Python — torchcodec/pyav are Python libs
3. Web UI is best for timeline/video UX — rich ecosystem
4. Can wrap in Tauri later for native feel

```
+-------------------------------------------------------------+
|                        Web Frontend                          |
|  +----------+  +----------+  +----------+  +--------------+ |
|  | TreeView |  | Timeline |  | VideoGrid|  | Action Charts| |
|  |(datasets)|  |(scrubber)|  |(cameras) |  | (plotly/d3)  | |
+-------------------------------------------------------------+
                              | WebSocket + REST
                              v
+-------------------------------------------------------------+
|                    Python Backend (FastAPI)                   |
|  +----------------------------------------------------------+|
|  |  Frame Cache (LRU) — by (dataset, episode, frame_idx)    ||
|  +----------------------------------------------------------+|
|  |  Prefetch Worker (Thread) — decodes next N frames         ||
|  +----------------------------------------------------------+|
|  |  Edit State Manager — in-memory pending edits             ||
|  +----------------------------------------------------------+|
|  |  LeRobot Integration Layer — dataset instances            ||
|  +----------------------------------------------------------+|
+-------------------------------------------------------------+
```

### Key Optimizations

1. **Frame Cache with LRU Eviction** — ~500 frames (~1GB for 720p RGB), keyed by (dataset, episode, frame)
2. **Background Prefetching** — triggered on playback or seek, decodes ahead in worker thread
3. **JPEG Encoding for Transfer** — ~50-100KB vs 2.7MB raw per frame
4. **Thumbnail Strip** — 1 thumbnail/second for timeline scrubbing preview

---

## API Design

### REST Endpoints

```
GET  /api/datasets                     # List opened datasets
POST /api/datasets                     # Open dataset (local path or HF repo_id)
DELETE /api/datasets/{id}              # Close dataset

GET  /api/datasets/{id}/episodes       # List episodes with metadata
GET  /api/datasets/{id}/episodes/{ep}  # Episode details
GET  /api/datasets/{id}/episodes/{ep}/thumbnails  # Timeline strip

GET  /api/datasets/{id}/episodes/{ep}/frame/{idx}  # Single frame (JPEG)
GET  /api/datasets/{id}/episodes/{ep}/data         # Parquet data

POST /api/edits/trim                   # Queue trim edit
POST /api/edits/delete                 # Queue delete edit
POST /api/edits/reorder                # Queue reorder edit
POST /api/edits/merge                  # Queue merge operation
GET  /api/edits/pending                # List pending edits
POST /api/edits/save                   # Apply all pending edits to disk
POST /api/edits/discard                # Discard pending edits

# Hub operations
GET  /api/hub/auth-status
POST /api/hub/login
POST /api/datasets/{id}/hub/download
POST /api/datasets/{id}/hub/upload
```

### WebSocket for Playback

```
WS /ws/playback/{dataset_id}/{episode_idx}

Client -> Server:
  { "action": "play", "from_frame": 0 }
  { "action": "pause" }
  { "action": "seek", "frame": 150 }
  { "action": "set_speed", "fps": 30 }

Server -> Client:
  { "type": "frame", "frame_idx": 0, "timestamp": 0.0,
    "cameras": { "front": "<base64 jpeg>", "wrist": "<base64 jpeg>" } }
  { "type": "buffering", "progress": 0.5 }
```

---

## Edit State Model

```python
@dataclass
class PendingEdit:
    edit_type: Literal["trim", "delete", "reorder", "duplicate", "move", "merge"]
    dataset_id: str
    params: dict
    created_at: datetime

@dataclass
class EditState:
    pending_edits: list[PendingEdit]
    def get_virtual_episodes(self, dataset_id) -> list[VirtualEpisode]: ...
    def apply_all(self): ...
```

---

## Dataset Merge Design

### Motivation

Sometimes the original dataset has flaws. Rather than extending it directly, the workflow is:
record a new dataset, then selectively merge good episodes into the original. This gives
flexibility to cherry-pick data.

### Existing Backend

`lerobot-edit-dataset --operation.type merge` calls `merge_datasets()` in `dataset_tools.py`,
which delegates to `aggregate_datasets()` in `aggregate.py`. The pipeline:

1. `validate_all_metadata()` — checks FPS, robot_type, features match
2. `aggregate_videos()` — `shutil.copy` or ffmpeg concat (NO re-encoding, preserves HF hashes)
3. `aggregate_data()` — merge parquet files with remapped indices
4. `aggregate_metadata()` — remap episode metadata with `src_to_dst` mapping
5. `finalize_aggregation()` — write tasks, info, stats

**Key property:** Videos are never recompressed. They are copied or concatenated via ffmpeg stream
copy (remux only). This preserves HuggingFace upload hashes for unchanged files.

### Known Bug (Fixed, Verified)

When merging datasets with multiple meta/episodes files, indices were not remapped correctly.
Fixed with proper `meta_src_to_dst` mapping in `aggregate_metadata()`.

- Single merge regression test: `tests/datasets/test_merge_regression.py`
- Chained merge tests (A+B->C, C+D->E and triple chain A+B->C->E->G, with video and file
  rotation): `tests/datasets/test_chained_merge.py` — all pass.

### GUI UX: Merge Dialog

A "Merge" button on the Data tab opens a dialog:

1. Select target: always a new dataset (never mutate in-place)
2. Select source datasets from opened datasets (checkboxes)
3. Optionally select specific episodes from each source (checkbox list)
4. Show validation summary (FPS, robot_type, features — green/red checks)
5. Preview result: episode count, total frames, estimated size
6. Execute merge

Episode ordering: append source episodes in order (A's episodes, then B's). No drag-to-reorder
(would force video recompression).

### Integrity Safeguards

1. **Pre-merge validation panel** — green/red checks for FPS, robot_type, features
2. **Always merge to new dataset** — never mutate existing datasets
3. **Post-merge integrity check** — verify:
   - Every episode's video timestamps are seekable
   - Every episode's data parquet row count matches `length`
   - Episode indices are contiguous
   - No orphan video/data files
4. **Diff summary before commit** — "N episodes from A, M from B -> new dataset C"

### Episode Selection (Partial Merge)

For selecting specific episodes (not whole datasets): the user picks episodes via checkboxes,
backend creates a filtered view (similar to virtual edits), and passes only selected episodes
to the merge pipeline.

---

## Future Features

### 3D Robot Visualization (URDF-based)

- Load robot URDF, render synchronized with timeline
- Display current joint positions from `observation.state`
- Overlay commanded actions from `action`
- Use case: verify camera-to-robot time alignment, debug teleop latency

### Optional Tauri Wrapper

Wrap web UI in Tauri for native desktop feel without changing code.

---

## Open Questions

1. **Multiple camera sync:** Independent scrubbing per camera, or always sync?
2. **Large datasets:** 1000+ episodes need virtual scrolling in tree view
3. **Video re-encoding on trim:** Show progress bar? Allow cancel?
4. **URDF loading:** Stored in dataset metadata or loaded separately?

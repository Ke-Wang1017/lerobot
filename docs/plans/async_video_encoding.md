# Plan: Streaming Video Encoding During Recording

## Problem Statement

After each episode, `save_episode()` blocks the main thread for the entire duration of
video encoding (PNG read-back + PyAV encode). This pause — often several seconds — delays
the start of the next episode. The goal is to eliminate this blocking time.

## Current Architecture and Its Waste

### The Recording Flow Today

```
record_loop() captures frames at 30fps
    ↓
add_frame() → for BOTH image and video keys:
    AsyncImageWriter writes PNG to disk (async)
    episode_buffer stores PNG file path
    ↓
Episode ends → save_episode()
    ↓
_wait_image_writer()          — block until all PNGs are flushed
    ↓
compute_episode_stats()       — reads sampled PNGs BACK from disk for stats
    ↓
_save_episode_data()          — writes parquet (fast, video keys excluded)
    ↓
[BLOCKING] _save_episode_video() per camera:
    _encode_temporary_episode_video()
        → reads ALL PNGs back from disk
        → encodes to temp .mp4 via PyAV (the actual bottleneck)
    _save_episode_video()
        → concatenates temp .mp4 onto existing video file (fast, no re-encode)
        → or moves temp .mp4 as new file if size limit reached
    ↓
meta.save_episode()           — writes metadata (needs video timestamps/indices)
    ↓
clear_episode_buffer()        — deletes the PNGs
```

### Why PNGs Exist (And Why They Shouldn't for Video)

In `add_frame()` ([lerobot_dataset.py:1168-1176](src/lerobot/datasets/lerobot_dataset.py#L1168-L1176)),
both `"image"` and `"video"` dtype keys take the same code path — write a PNG via
`AsyncImageWriter`. The only difference is `compress_level=1` (fast) for video vs `6` for image.

For **image-mode datasets**, PNGs are the final storage format. They belong on disk.

For **video-mode keys**, the PNGs are pure waste:

1. Written to disk during recording (I/O cost)
2. Sampled and read back for stats computation ([compute_stats.py:511-512](src/lerobot/datasets/compute_stats.py#L511-L512))
3. ALL read back for video encoding ([video_utils.py:378-385](src/lerobot/datasets/video_utils.py#L378-L385))
4. Deleted after encoding

The entire PNG round-trip is unnecessary overhead for video keys. No downstream consumer
needs them — video keys are excluded from parquet entirely
([utils.py:588-589](src/lerobot/datasets/utils.py#L588-L589): `if ft["dtype"] == "video": continue`).
The PNG paths stored in `episode_buffer` for video keys are dead data.

### How Episode Videos Are Appended

A detail critical to any async design: episodes don't get independent video files.
`_save_episode_video()` ([lerobot_dataset.py:1435-1514](src/lerobot/datasets/lerobot_dataset.py#L1435-L1514))
concatenates each episode's temp video onto the existing video file:

1. **First episode** (or size limit reached): temp video is `shutil.move`'d as a new file
2. **Subsequent episodes**: `concatenate_video_files([existing.mp4, temp.mp4], existing.mp4)`
   using ffmpeg's concat demuxer — packet remux, no re-encoding
   ([video_utils.py:400-480](src/lerobot/datasets/video_utils.py#L400-L480))

This creates a **sequential dependency per camera**: you can't concatenate episode N+1 until
episode N's concatenation is complete, because the output file is the input for the next concat.
The metadata (chunk_index, file_index, from_timestamp, to_timestamp) also depends on the
cumulative state after each concatenation.

---

## Why Episode-Level Async Encoding Is the Wrong Approach

A previous version of this plan proposed deferring encoding to a background queue after
`save_episode()` returns. This has fundamental problems:

### Data Integrity / Crash Safety

The current synchronous design has a valuable property: **after `save_episode()` returns,
the dataset is fully consistent** — parquet, video files, and metadata all agree. If the
robot throws an exception (a common occurrence), everything already saved is intact.

With an episode-level async queue of depth N:

- Parquet and metadata are written immediately for episodes 1..N
- But video files only contain data up to episode 0
- A crash leaves the dataset **silently corrupt**: metadata references video timestamps
  that don't exist in the actual video file
- Recovery is non-trivial: can't just delete orphaned episodes because indices would have
  gaps, and the concatenated video format makes it hard to tell where episodes start

### Concatenation Ordering

Even if encoding runs in parallel, concatenation must be serial per camera (shared mutable
output file). The async queue would need a separate ordered concatenation stage, adding
significant complexity for unclear benefit.

### The Real Bottleneck Is Misidentified

The bottleneck isn't that encoding happens synchronously at episode boundaries — it's that
ALL encoding work is deferred to episode boundaries. The recording time itself is wasted
(from an encoding perspective). The solution isn't to defer work further, but to spread
it across the recording window.

---

## Proposed Approach: Frame-Level Streaming Encoding

### Core Idea

For video keys, **bypass PNG generation entirely** and stream frames directly to a
background video encoder during recording. By the time the episode ends — especially
after the reset phase — the temp video is already complete or nearly complete.

### New Flow

```
record_loop() captures frames at 30fps
    ↓
add_frame():
    image keys → AsyncImageWriter (unchanged)
    video keys → StreamingVideoEncoder.push_frame()
        → frame goes onto a queue
        → background thread encodes it incrementally via PyAV
        → encoder also keeps a reservoir sample of frames in memory
    ↓
Robot reset phase (several seconds, sometimes tens of seconds)
    → encoder finishes remaining frames during this dead time
    ↓
save_episode():
    encoder.finish()              — flush encoder, return temp video path
                                    (likely instant — encoding already done)
    write sampled frames to       — write ~300 sampled frames to temp PNG dir
      temp dir for stats            (trivial I/O vs writing ALL frames)
    compute_episode_stats()       — called UNCHANGED, reads the sampled PNGs
    _save_episode_data()          — parquet (unchanged, video keys still excluded)
    _save_episode_video()         — concatenate temp video onto existing file (unchanged)
    meta.save_episode()           — metadata (unchanged)
    encoder.start_new_episode()   — reset encoder for next episode
    clean up temp stats PNGs
```

### Why This Is Better in Every Dimension

| Concern                  | Episode-level async (old plan)           | Frame-level streaming (this plan)                                      |
| ------------------------ | ---------------------------------------- | ---------------------------------------------------------------------- |
| Crash safety             | Lose N queued episodes, corrupt metadata | Lose at most current episode (same as today)                           |
| Metadata consistency     | Broken for queued episodes               | Always consistent — sync save_episode()                                |
| Queue buildup            | Real risk with short episodes            | Impossible — queue bounded by episode length, lag absorbed at finish() |
| PNG I/O during recording | Writes ALL frames as PNGs                | Zero PNGs during recording for video keys                              |
| Reset time utilization   | Wasted                                   | Encoder finishes trailing frames                                       |
| Complexity               | New queue, recovery, reconciliation      | Replaces AsyncImageWriter for video keys                               |
| Downstream compatibility | Needs metadata split/deferral            | compute_episode_stats unchanged                                        |

---

## Detailed Design

### StreamingVideoEncoder Class

Location: new file `src/lerobot/datasets/video_encoder.py`

One instance per camera key. Mirrors the `AsyncImageWriter` pattern (queue + background thread).

```python
class StreamingVideoEncoder:
    """Encodes video frames incrementally during recording.

    Replaces the PNG-then-encode pipeline for video keys. Frames are pushed
    from the recording loop and encoded in a background thread. A reservoir
    sample of frames is kept in memory for stats computation.
    """

    def __init__(
        self,
        fps: int,
        vcodec: str = "libsvtav1",
        pix_fmt: str = "yuv420p",
        g: int = 2,
        crf: int = 30,
        preset: int = 12,
    ):
        self.fps = fps
        self.vcodec = vcodec
        self.pix_fmt = pix_fmt
        self.codec_options = {...}  # g, crf, preset etc.

        self._frame_queue: queue.Queue = queue.Queue()  # unbounded — see "Backpressure" section
        self._thread: threading.Thread | None = None
        self._episode_active = False

        # Reservoir sample for stats (kept in memory)
        self._sampled_frames: list[np.ndarray] = []
        self._frame_count: int = 0

    def start_episode(self, tmp_video_path: Path):
        """Begin encoding a new episode to a temporary video file."""
        self._episode_active = True
        self._frame_count = 0
        self._sampled_frames = []
        self._thread = threading.Thread(
            target=self._encoding_loop, args=(tmp_video_path,)
        )
        self._thread.start()

    def push_frame(self, image: np.ndarray):
        """Called from add_frame() for video keys. Never blocks."""
        self._maybe_sample(image)
        self._frame_queue.put(image)
        self._frame_count += 1

    def finish(self) -> Path:
        """Signal end of episode, wait for encoder to flush, return temp video path.
        Typically near-instant if reset time > encoding lag."""
        self._frame_queue.put(None)  # sentinel
        self._thread.join()
        self._episode_active = False
        return self._tmp_video_path

    def discard(self):
        """Abort current episode (for re-record). Kills encoder, deletes temp file."""
        if self._episode_active:
            self._frame_queue.put(_DISCARD_SENTINEL)
            self._thread.join()
            self._tmp_video_path.unlink(missing_ok=True)
            self._episode_active = False

    def write_sampled_frames_to_disk(self, output_dir: Path) -> list[str]:
        """Write reservoir-sampled frames as PNGs for compute_episode_stats().

        This writes only ~100-300 frames (not all frames), keeping I/O minimal.
        Returns the list of PNG paths, compatible with the interface that
        compute_episode_stats() expects via sample_images().
        """
        output_dir.mkdir(parents=True, exist_ok=True)
        paths = []
        for i, frame in enumerate(self._sampled_frames):
            path = output_dir / f"frame-{i:06d}.png"
            PIL.Image.fromarray(frame).save(path, compress_level=1)
            paths.append(str(path))
        return paths

    def _encoding_loop(self, video_path: Path):
        """Background thread: dequeue frames, encode via PyAV."""
        self._tmp_video_path = video_path
        with av.open(str(video_path), "w") as output:
            stream = output.add_stream(self.vcodec, self.fps, options=self.codec_options)
            # width/height set from first frame

            while True:
                frame = self._frame_queue.get()
                if frame is None:       # end of episode
                    packet = stream.encode()  # flush
                    if packet:
                        output.mux(packet)
                    break
                if frame is _DISCARD_SENTINEL:
                    break               # abort without flushing

                av_frame = av.VideoFrame.from_ndarray(frame, format="rgb24")
                packet = stream.encode(av_frame)
                if packet:
                    output.mux(packet)

    def _maybe_sample(self, image: np.ndarray):
        """Reservoir sampling. Keeps a representative subset of frames in memory.

        Uses reservoir sampling (Algorithm R) since we don't know the total
        frame count upfront. Target sample size matches what compute_stats
        uses (~100-300 frames depending on episode length).

        Frames are stored at original resolution. The downstream
        auto_downsample_height_width() in compute_stats handles downsampling.
        """
        ...

    def stop(self):
        """Graceful shutdown."""
        if self._episode_active:
            self.discard()
```

### Stats Computation: No Changes to compute_episode_stats

`compute_episode_stats()` ([compute_stats.py:477](src/lerobot/datasets/compute_stats.py#L477))
expects `episode_data[key]` to be a list of file paths for image/video keys. It calls
`sample_images(data)` which reads a sampled subset from those paths.

**We do not modify `compute_episode_stats` at all.** Instead:

1. During recording, the streaming encoder maintains a reservoir sample of ~300 frames
   in memory (negligible memory: ~300 frames at 640x480x3 = ~276MB worst case, and
   these are downsampled in practice by compute_stats anyway).

2. At `save_episode()` time, before calling `compute_episode_stats()`, we write just the
   sampled frames to a temp directory as PNGs and populate the episode_buffer's video key
   entries with those paths.

3. `compute_episode_stats()` runs unchanged — it sees a list of paths, calls
   `sample_images()`, gets stats. It doesn't know or care that the paths point to a
   pre-sampled subset rather than all frames.

4. The temp directory is cleaned up after stats are computed.

This approach is robust against upstream changes to `compute_episode_stats`: as long as
it continues to accept a list of file paths (its current interface), our code works. If
the upstream function changes its interface, we'd need to adapt regardless.

**I/O comparison**: current approach writes ALL frames as PNGs during recording, then
reads a sample back. New approach writes ~300 PNGs at save time only. For a 10-second
episode at 30fps (300 frames), this is ~300 PNGs → ~300 PNGs (similar). For a 60-second
episode (1800 frames), this is 1800 PNGs → ~300 PNGs (6x reduction). The savings grow
with episode length, and the PNGs are written during the `save_episode()` call rather
than competing with recording I/O.

### Integration: Minimal Changes to LeRobotDataset

**Design principle**: all streaming encoder logic lives in new helper methods on
`LeRobotDataset`. Changes to existing methods are single-line insertions/guards that
call into those helpers. This minimizes merge conflict surface with upstream.

#### New helper methods (added to LeRobotDataset, no upstream conflict)

```python
def _init_video_encoders(self):
    """Called from create() and __init__ when resuming. Initializes per-camera encoders."""
    self.video_encoders = {}
    if not self.meta.video_keys:
        return
    for video_key in self.meta.video_keys:
        self.video_encoders[video_key] = StreamingVideoEncoder(
            fps=self.fps, vcodec=self.vcodec
        )

def _start_video_encoders(self):
    """Start encoders for a new episode. Called at start of recording loop."""
    for video_key, encoder in self.video_encoders.items():
        encoder.start_episode(self._make_video_tmp_path(video_key))

def _stop_video_encoders(self):
    """Graceful shutdown. Parallel to stop_image_writer()."""
    for encoder in self.video_encoders.values():
        encoder.stop()
    self.video_encoders = {}

def _finish_video_encoders(self) -> dict[str, Path]:
    """Finish all encoders, return {video_key: temp_video_path}."""
    return {key: enc.finish() for key, enc in self.video_encoders.items()}

def _discard_and_restart_video_encoders(self):
    """Discard current episode and restart encoders. Used by re-record."""
    for video_key, encoder in self.video_encoders.items():
        encoder.discard()
        encoder.start_episode(self._make_video_tmp_path(video_key))

def _write_video_stats_samples(self, episode_buffer: dict, episode_index: int) -> list[Path]:
    """Write reservoir-sampled frames as PNGs for compute_episode_stats().
    Populates episode_buffer video key entries with PNG paths.
    Returns list of temp dirs to clean up."""
    tmp_dirs = []
    for video_key, encoder in self.video_encoders.items():
        stats_dir = self.root / "tmp_stats" / f"ep_{episode_index}" / video_key
        paths = encoder.write_sampled_frames_to_disk(stats_dir)
        episode_buffer[video_key] = paths
        tmp_dirs.append(stats_dir)
    return tmp_dirs

def _make_video_tmp_path(self, video_key: str) -> Path:
    """Generate a temp path for the streaming encoder output."""
    ep_idx = self.episode_buffer["episode_index"]
    return self.root / "tmp_videos" / f"ep_{ep_idx}_{video_key}.mp4"
```

#### Change 1: create() — one line added

[lerobot_dataset.py:1622](src/lerobot/datasets/lerobot_dataset.py#L1622)
— add after `obj._writer_closed_for_reading = False`:

```python
        obj._writer_closed_for_reading = False
+       obj._init_video_encoders()
        return obj
```

#### Change 2: add_frame() — guard around existing block

[lerobot_dataset.py:1168-1176](src/lerobot/datasets/lerobot_dataset.py#L1168-L1176)
— add an `elif` branch, existing code unchanged:

```python
            if self.features[key]["dtype"] in ["image", "video"]:
+               if self.features[key]["dtype"] == "video" and self.video_encoders:
+                   self.video_encoders[key].push_frame(frame[key])
+                   continue
                img_path = self._get_image_file_path(
                    episode_index=self.episode_buffer["episode_index"], image_key=key, frame_index=frame_index
                )
                # ... rest unchanged
```

#### Change 3: save_episode() — insert before/after existing blocks

[lerobot_dataset.py:1227-1229](src/lerobot/datasets/lerobot_dataset.py#L1227-L1229)
— insert before `_wait_image_writer()`:

```python
+       # Finish streaming video encoders (likely instant after reset time)
+       video_temp_paths = self._finish_video_encoders() if self.video_encoders else {}
+       stats_tmp_dirs = self._write_video_stats_samples(episode_buffer, episode_index) if video_temp_paths else []

        # Wait for image writer to end, so that episode stats over images can be computed
        self._wait_image_writer()
        ep_stats = compute_episode_stats(episode_buffer, self.features)

+       # Clean up stats temp PNGs
+       for d in stats_tmp_dirs:
+           shutil.rmtree(d, ignore_errors=True)
```

[lerobot_dataset.py:1235-1270](src/lerobot/datasets/lerobot_dataset.py#L1235-L1270)
— guard the existing encoding block:

```python
-       if has_video_keys and not use_batched_encoding:
+       if has_video_keys and not use_batched_encoding and not video_temp_paths:
            # ... existing ProcessPoolExecutor encoding (only runs if streaming is off)
```

[lerobot_dataset.py:1263-1270](src/lerobot/datasets/lerobot_dataset.py#L1263-L1270)
— after the existing encoding block, add the streaming path:

```python
+       if video_temp_paths:
+           for video_key in self.meta.video_keys:
+               ep_metadata.update(
+                   self._save_episode_video(video_key, episode_index,
+                                            temp_path=video_temp_paths[video_key])
+               )
```

[lerobot_dataset.py:1286](src/lerobot/datasets/lerobot_dataset.py#L1286)
— at end of save_episode(), restart encoders:

```python
+       if self.video_encoders:
+           self._start_video_encoders()
```

#### Change 4: clear_episode_buffer() — one line added

[lerobot_dataset.py:1516](src/lerobot/datasets/lerobot_dataset.py#L1516)
— add at top of method:

```python
    def clear_episode_buffer(self, delete_images: bool = True):
+       if self.video_encoders:
+           self._discard_and_restart_video_encoders()
        # Clean up image files for the current episode buffer
        if delete_images:
            # ... rest unchanged
```

#### Summary of Upstream Diff

| Location                      | Change                                  | Lines touched |
| ----------------------------- | --------------------------------------- | ------------- |
| `create()`                    | +1 line: `obj._init_video_encoders()`   | 1             |
| `add_frame()`                 | +3 lines: guard + push_frame + continue | 3             |
| `save_episode()`              | +~12 lines: finish/stats/guard/restart  | 12            |
| `clear_episode_buffer()`      | +2 lines: guard + discard call          | 2             |
| **Total in existing methods** |                                         | **~18 lines** |

All logic lives in the new helper methods + `StreamingVideoEncoder` class. The existing
methods gain small guards that check `if self.video_encoders:` and delegate. If upstream
changes the internals of `save_episode()`, the guards are unlikely to conflict because
they're inserted between existing blocks, not modifying them.

---

## Can Encoding Keep Up with Real-Time?

At 30fps, each frame has a 33ms budget. With libsvtav1 at preset 12 (fastest):

- Single-frame encode is typically well under 33ms on modern CPUs
- The background thread has the full frame interval to encode each frame
- If occasional frames take longer, the bounded queue (maxsize=60, ~2s) absorbs bursts
- The reset phase between episodes (often 5-30 seconds) provides additional catch-up time
- Even in the worst case, `finish()` only waits for trailing frames, not the whole episode

### Why the Queue Is Unbounded (No Backpressure)

The frame queue is deliberately unbounded. Backpressure (blocking `push_frame()` when the
queue is full) would slow down the recording loop, causing inconsistent frame timing or
missed camera polls — corrupting the very data we're trying to capture. The recording loop
must run at a rock-steady rate; it is not an acceptable place to absorb lag.

Instead, if the encoder falls behind, frames accumulate in memory:

- The queue can never grow beyond one episode's worth of frames (episodes are finite)
- The realistic cost is only the **lag**, not total frames, since the encoder consumes
  continuously. If the encoder is 10% slower than real-time over a 60-second episode,
  that's ~180 frames backed up (~160MB at 640x480, ~1GB at 1080p)
- The encoder catches up during the reset phase between episodes
- Any remaining lag is absorbed at `finish()` time, inside `save_episode()` — not during
  recording

If the encoder is consistently too slow (lag grows every episode), this indicates the
system can't handle real-time encoding at the configured settings. The right response is
to log a warning suggesting a faster preset or lower resolution, not to silently degrade
recording quality.

---

## Re-record Compatibility

Re-record calls `clear_episode_buffer()` → `encoder.discard()`:

- Background encoding thread receives discard sentinel and exits
- Temp video file is deleted
- Encoder resets and starts a fresh episode
- No PNGs to clean up for video keys
- Same safety guarantees as today

---

## Crash Safety

The transactional property of `save_episode()` is **preserved**:

- Encoding happens during recording and is finished (flushed) before any metadata is written
- `_save_episode_video()` (concatenation) runs synchronously as before
- `meta.save_episode()` runs after concatenation as before
- If the robot crashes during recording: lose current episode only (same as today)
- If the robot crashes during `save_episode()`: same as today (partially written episode)
- No queue of unencoded episodes, no orphaned metadata

---

## Batched Encoding Compatibility

The existing `batch_encoding_size > 1` path defers encoding across multiple episodes.
This is fundamentally incompatible with the streaming approach (streaming encodes during
recording, not after).

Options:

1. **Remove batched encoding** — streaming is strictly better
2. **Keep as fallback** — if `batch_encoding_size > 1`, fall back to the old PNG path

Recommend option 1 (remove), since streaming eliminates the problem batched encoding
was trying to solve.

---

## Implementation Steps

### Step 1: StreamingVideoEncoder class

Create `src/lerobot/datasets/video_encoder.py` with the class described above.
Key: get the background thread encoding loop right, with proper flush and discard handling.

### Step 2: Reservoir sampling + write_sampled_frames_to_disk

Implement `_maybe_sample()` using reservoir sampling.
Implement `write_sampled_frames_to_disk()` to write sampled frames as PNGs.
Validate that `compute_episode_stats()` produces equivalent results when given the
sampled subset vs the full frame set (the function already samples internally, so
giving it a pre-sampled set should produce statistically identical results).

### Step 3: Add helper methods to LeRobotDataset

Add `_init_video_encoders`, `_start_video_encoders`, `_stop_video_encoders`,
`_finish_video_encoders`, `_discard_and_restart_video_encoders`,
`_write_video_stats_samples`, `_make_video_tmp_path`. These are new methods —
zero conflict with upstream.

### Step 4: Insert minimal guards into existing methods

Add the ~18 lines of guards/calls described in the integration section:
`create()` (+1), `add_frame()` (+3), `save_episode()` (+12),
`clear_episode_buffer()` (+2). Existing logic is not modified, only guarded.

---

## Testing

### Unit Tests

- StreamingVideoEncoder: push N frames, finish, verify valid .mp4 with correct frame count
- StreamingVideoEncoder: push frames, discard, verify temp file deleted and encoder reusable
- Reservoir sampling: verify sample size is reasonable for various episode lengths
- write_sampled_frames_to_disk: verify PNGs are valid and paths work with sample_images()
- Lag monitoring: warn when encoder falls behind, verify catch-up during reset

### Integration Tests

- Record multiple episodes with streaming encoding, verify concatenated video matches
  frame-by-frame against old PNG-based encoding (bit-exact or within codec tolerance)
- Stats comparison: verify streaming-path stats match PNG-path stats within tolerance
- Re-record mid-session: verify discarded episode leaves no artifacts
- Mixed dataset: image keys use PNGs, video keys use streaming, both work correctly

### Edge Cases

- Very short episodes (< 1 second, < 30 frames)
- First frame: encoder must handle width/height discovery from first frame
- Keyboard interrupt during encoding
- Camera disconnection mid-episode (push_frame stops, encoder gets partial episode)

---

## Risks and Mitigations

| Risk                                   | Impact                                                | Mitigation                                                                                   |
| -------------------------------------- | ----------------------------------------------------- | -------------------------------------------------------------------------------------------- |
| Encoding can't keep up at 30fps        | Memory grows during recording, finish() blocks longer | Unbounded queue absorbs lag; reset time provides catch-up; warn if lag grows across episodes |
| PyAV threading issues                  | Crash or corruption                                   | Encoder runs in own thread with isolated av context; no shared state                         |
| GIL contention                         | Encoding thread starved                               | PyAV encode/mux are C calls that release GIL; queue ops are brief                            |
| Reservoir sample not representative    | Slightly different stats                              | Sample size matches upstream's sample_images(); validated in tests                           |
| Upstream compute_episode_stats changes | Our sampled PNGs may not suffice                      | Interface is stable (list of paths); if it changes we adapt regardless                       |

### GIL Note

The main concern with a threading approach is GIL contention. However, the heavy
operations — PyAV's `encode()` and `mux()` — are C-level calls that release the GIL.
`av.VideoFrame.from_ndarray()` also primarily operates in C. The GIL is only held briefly
for Python-level queue operations. If GIL contention proves problematic in practice,
the encoder could be moved to a subprocess (similar to how `AsyncImageWriter` supports
`num_processes > 0`), at the cost of frame serialization overhead.

---

## Future Extension: Live Streaming to GUI

### Current State

The recording loop has two consumers today, both called inline at each frame:

```
robot.get_observation()
    ↓
obs_processed = robot_observation_processor(obs)
    ↓
observation_frame = build_dataset_frame(...)     ← dict with camera arrays + state
    ↓
├── dataset.add_frame(frame)                     ← writes to disk (PNGs/encoder)
└── log_rerun_data(observation=obs_processed)    ← sends to rerun viewer
```

The GUI (`src/lerobot/gui/`) is currently **post-hoc only** — it reads completed datasets
from disk. It has no live recording integration. However, it already has unused WebSocket
infrastructure at `/ws/playback/{dataset_id}` that streams base64 JPEG frames with
play/pause/seek commands. The server side is fully implemented; the frontend just doesn't
connect to it.

### How Streaming Encoding Enables GUI Live Preview

With the streaming encoder, every video frame already passes through
`StreamingVideoEncoder.push_frame()` in a structured, per-camera-key manner.
This creates a natural fan-out point for a third consumer alongside encoding and
rerun.

However, the cleanest approach is to **not couple the GUI stream to the encoder**.
The encoder's job is encoding — adding GUI broadcasting to it would violate single
responsibility and complicate the threading model. Instead, the recording loop should
feed the GUI the same way it feeds rerun: as a parallel consumer.

### Proposed Architecture

```
record_loop() at each frame:
    ├── dataset.add_frame(frame)                 ← encoder gets video frames
    ├── log_rerun_data(obs, action)              ← rerun (existing)
    └── recording_broadcaster.push(obs)          ← GUI live stream (new)
```

**`RecordingBroadcaster`** — lightweight component that:

- Receives the observation dict (including camera arrays) each frame
- Converts camera frames to JPEG (like rerun does with `compress_images`)
- Publishes to connected WebSocket clients via the existing GUI server
- If no clients are connected, does nothing (zero overhead)
- Runs JPEG compression in a background thread to avoid slowing the recording loop

**GUI server changes**:

- New endpoint: `/ws/live` (distinct from `/ws/playback/{dataset_id}`)
- Receives frames from `RecordingBroadcaster` via an in-process queue or shared buffer
- Streams to connected browser clients in the same format as the existing
  playback WebSocket (`{"type": "frame", "data": base64_jpeg}`)
- The frontend connects to `/ws/live` when a recording session is active

**Why this is separate from the encoder**:

- Different data format: GUI needs JPEG for display, encoder needs raw arrays for video
- Different timing: GUI can drop frames if the client is slow, encoder cannot
- Different lifecycle: GUI stream is optional, encoder is mandatory
- Different threading: GUI JPEG compression is I/O-bound, encoder is CPU-bound

### Integration with Streaming Encoder

The encoder and GUI broadcaster share no state. They both consume frames from the
recording loop independently:

- `add_frame()` → video keys → `StreamingVideoEncoder.push_frame()` (raw numpy)
- `log_gui_data()` → `RecordingBroadcaster.push()` (observation dict → JPEG)

The recording loop calls both without blocking. The broadcaster is "fire and forget" —
if JPEG compression or WebSocket sending falls behind, it drops frames rather than
building a queue (unlike the encoder, which must encode every frame).

### Minimal Recording Script Change

```python
# In record_loop(), alongside existing rerun call:
        if display_data:
            log_rerun_data(observation=obs_processed, action=action_values, ...)

+       if gui_broadcaster is not None:
+           gui_broadcaster.push(obs_processed)
```

One line added. The broadcaster is initialized in `record()` if the GUI server is
running, otherwise `None`.

### Implementation Scope

This is a **separate feature** from the streaming encoder. The encoder change is
self-contained and should be implemented first. GUI live streaming can be added
afterward, reusing the WebSocket infrastructure already in `api/playback.py` as a
reference implementation.

---

## Summary

**Approach**: Stream frames directly to a background PyAV encoder during recording,
bypassing PNG generation entirely for video keys.

**Key insight**: The encoding work should be spread across the recording window
(frame-level pipelining), not deferred to episode boundaries (episode-level queuing).
The reset phase between episodes provides natural catch-up time.

**Benefits**:

1. Eliminates PNG write + read overhead for video keys during recording
2. Encoding is (nearly) free — absorbed into recording time + reset time
3. Preserves crash safety — no queue of unencoded episodes
4. Preserves metadata consistency — save_episode() remains synchronous and transactional
5. No changes to compute_episode_stats — sampled PNGs written at save time for compatibility

**Key changes**:

1. New `StreamingVideoEncoder` class (mirrors `AsyncImageWriter` pattern)
2. `add_frame()` branches: image keys → PNG writer, video keys → streaming encoder
3. `save_episode()` calls `encoder.finish()` instead of encoding from PNGs
4. Sampled frames written as PNGs at save time for stats (not during recording)
5. `clear_episode_buffer()` calls `encoder.discard()` for re-record

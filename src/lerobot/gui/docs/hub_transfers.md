# Hub Transfers Design

Background upload and download of LeRobot datasets to and from the HuggingFace Hub, surfaced through the GUI's Transfers tray. Optimized for **wall-clock throughput** on real workloads, which range from a few-MB sandbox dataset to a multi-hundred-GB long-horizon multi-camera recording session.

The shape: each transfer runs in its own subprocess; the GUI server holds the source-of-truth job registry that every connected frontend reads; the actual byte transfer uses HuggingFace's optimized helpers (`upload_large_folder` for uploads, `snapshot_download` for downloads) so we inherit Xet content-addressed dedupe, per-file resumability, etag skip, and parallel HTTP — none of which are worth re-implementing.

Every upload — regardless of size — runs through HF's pull-request mechanism: we create a PR (which auto-creates a `refs/pr/<num>` branch on the repo), `upload_large_folder` writes to that PR branch, and the PR merges into `main`. This is what gives us **unconditional atomicity** on the consumer-facing side: `main` either reflects the entire upload or none of it, never a partial state — even for multi-hundred-GB uploads spanning hours.

> **Implementation status — `super_squash_history`:** the design originally specified squashing the PR branch to a single commit before merge, for clean main-history hygiene. In practice `super_squash_history` rewrites the PR branch in a way that doesn't always preserve fast-forward-ability to main — observed on second-upload-to-the-same-repo as a `BadRequestError: There are merge conflicts`. Until the right HF API usage is understood, the worker takes the design's documented squash-failure-fallback path unconditionally: skip the squash, merge the unsquashed PR branch (a fast-forward when no one else has pushed to main concurrently). Main ends up with N commits per upload — atomicity is preserved at the merge moment, only commit-history hygiene degrades. Tracked in [Open questions](#open-questions).

---

## Goals

### Functional

- **F1 — Background, multi-tab consistent, close-tab-safe.** A transfer keeps running even after the user closes the browser tab or opens the GUI on a different device. The GUI server is the source of truth for job state; every frontend reads the same `/api/datasets/hub/jobs` and sees identical state. Killing the GUI server **does** terminate in-flight subprocesses — that's the orphan-subprocess problem tracked separately, out of scope here.
- **F2 — Throughput is the primary cost metric.** Wall-clock time dominates user pain on long uploads. Every design decision below trades against this metric; UX polish is secondary.
- **F3 — Skip already-uploaded files and chunks.** Three dedupe layers, all server-driven, all native to `upload_large_folder`, all branch-independent — they work identically when uploading to a PR branch as they would to `main`. See [Upload skip behavior](#upload-skip-behavior) below. Re-uploading an identical dataset transfers ~0 bytes (verified: 8.84× speedup on the small experimental upload). Re-uploading after editing one chunk of one file transfers ~one chunk.
- **F4 — Skip already-downloaded files.** Per-file etag matching in `snapshot_download` / `hf_hub_download`. Re-downloading a synced dataset is near-instant.
- **F5 — Elegant error handling.** Lean on HF's built-in per-file retries first; surface terminal failures clearly; expose a Retry button that exploits HF's resume primitives so retrying a failed transfer is cheap. Cancelled and failed uploads leave the PR branch behind for resume; `main` is never affected.
- **F6 — Unconditional atomicity on the consumer-facing side.** Every upload — small or huge — lands on `main` as a single squashed commit through the PR mechanism. There is no "partial state" failure mode visible to downstream consumers fetching from `main`. A pending PR branch may exist during the upload; that branch is the staging area, intentionally distinct from the consumable `main`.

### Non-functional

- **N1 — UX is acceptable but secondary.** Coarse milestones from HF's text output are fine; we don't need byte-precise bars. The tray must show "running" / "complete" / "failed" / "cancelled" honestly; perfection isn't required.
- **N2 — Cancelable.** Background transfers are killable from the GUI via `SIGTERM` to the worker subprocess. A cancelled job is restartable without error — HF's resume primitives (the PR branch + `.cache/.huggingface/`) make the retry pick up where it stopped. The atomic merge step has its own retry behavior: see [Safety guardrails](#safety-guardrails).
- **N3 — Crash-isolated.** A failed HF library call, an OOM during a multi-GB download, a stuck commit RPC — none of these can take the GUI server down with them.

## Non-goals

- **Byte-level progress bars.** HF 1.x exposes no programmatic callback on `upload_large_folder` — only stderr / stdout text. We do not build a parallel byte-tracking layer (custom `ProgressFile`-wrapped `create_commit`, monkey-patched `XetProgressReporter`, etc.) because doing so costs throughput, complexity, and a coupling to private HF internals. The text milestone output from `upload_large_folder` is the level of detail we operate at.
- **A pause button distinct from cancel.** The resume primitives that make cancel safe also make "pause then resume" work as "cancel then retry." A separate `SIGSTOP`-style pause is implementable but introduces fragility (HF's TCP connections may time out during suspension, signal handlers race) for no UX gain — see [Cancellation and pause](#cancellation-and-pause).
- **Two upload modes (atomic vs resumable).** An earlier version of this design proposed a 5 GB threshold switching between `upload_folder` and `upload_large_folder`. Verified on a throwaway repo that `upload_large_folder` going through a PR branch gives unconditional atomicity at all sizes with no measurable bandwidth penalty (the 1-2 second startup is invisible against any real upload, and Xet dedupe is unaffected by branch targeting). Two modes would have been complexity without a benefit. Single pipeline only.
- **Cross-process orphan-subprocess management.** Killing the GUI server should cleanly terminate in-flight transfers via `PR_SET_PDEATHSIG` in the worker. This belongs to the broader orphan-subprocess infrastructure already on [gui/TODO.md](../TODO.md) (Run-tab section) and applies equally to `lerobot-teleoperate` and `lerobot-record`. We use the registry that work lands, not duplicate it here.

---

## Architecture

Three pieces:

1. **`HubJobState`** on the GUI server — one entry per in-flight or recently-finished transfer. The single source of truth that every frontend polls via `GET /api/datasets/hub/jobs`. Lives in `AppState.hub_jobs`, keyed by `job_id`, garbage-collected 30 minutes after the transfer reaches a terminal state.
2. **Worker subprocess** per transfer — spawned with `start_new_session=True`, communicates with the GUI server via a per-job progress JSON file on disk (rewritten ~1 Hz) and a per-job log file (append-only, captures the HF library's stderr for debug + milestone parsing). The worker owns the full upload pipeline (PR creation → upload → squash → merge → cleanup) or the download. SIGTERM by the GUI server is the cancel mechanism.
3. **JSONL analytics** — one line appended to `~/.config/lerobot/gui/hub_transfers.jsonl` per terminal outcome (complete / failed / cancelled). Distinct from the IPC progress file — see [Analytics](#analytics) for the schema and [Worker IPC](#worker-ipc) for the per-job file layout.

```
┌──────────────┐  fetch /hub/jobs                ┌──────────────────┐
│  Frontend A  │ ───────────────────────────────▶│                  │
└──────────────┘                                 │                  │  spawn(SIGTERM)
                                                 │  GUI server      │ ────────────▶ ┌────────┐
┌──────────────┐  fetch /hub/jobs                │  (FastAPI)       │               │ worker │
│  Frontend B  │ ───────────────────────────────▶│                  │ ◀──────────── │ subproc│
└──────────────┘                                 │  AppState        │  reads JSON   └────────┘
                                                 │  .hub_jobs[*]    │  progress         │
                                                 └──────────────────┘                   │
                                                                                  ┌─────▼─────────────────────┐
                                                                                  │ huggingface_hub            │
                                                                                  │   create_pull_request      │
                                                                                  │   upload_large_folder      │
                                                                                  │   super_squash_history     │
                                                                                  │   merge_pull_request       │
                                                                                  │   snapshot_download        │
                                                                                  └───────────────────────────┘
```

### Why subprocess

Two properties only the process boundary buys cleanly:

- **Cancel without thread-pool wrestling.** `snapshot_download(max_workers=8)` spins an internal thread pool we cannot interrupt cleanly from inside the same process. `os.killpg(pid, SIGTERM)` kills the whole tree in one call.
- **Crash isolation.** HF's library can raise unexpected exceptions, hit OOM during very large transfers, or get stuck in a commit RPC. Inside the GUI server's event loop those failures cascade; behind a process boundary they're just a job that flipped to `failed` with a captured error string.

The cost is ~1 second of subprocess startup per transfer (Python init + HF imports). For any realistic dataset this is invisible against actual transfer time; on a one-off push of a tiny dataset it's the dominant cost, which we accept.

### Why HuggingFace's helpers, not a hand-rolled per-file loop

A prior iteration of this work (closed) used a `for file in files: upload_file(...)` / `hf_hub_download(...)` loop to get clean cancel-between-files and a custom progress bar. Measured cost (random bytes, no Xet dedupe, fresh HF cache):

| Direction | Dataset                  | per-file loop | HF helper                     | Slowdown |
| --------- | ------------------------ | ------------- | ----------------------------- | -------- |
| Download  | pusht (8 files / 7.5 MB) | 5.18 s        | 1.25 s (`snapshot_download`)  | **4.1×** |
| Download  | svla (28 files / 83 MB)  | 12.48 s       | 10.69 s (`snapshot_download`) | **1.2×** |
| Upload    | 8 files × 1 MB (8 MB)    | 18.25 s       | 6.01 s (`upload_folder`)      | **3.0×** |
| Upload    | 28 files × 3 MB (84 MB)  | 90.78 s       | 42.58 s (`upload_folder`)     | **2.1×** |

The slowdown shrinks as bytes-transferred dominates over per-file protocol overhead — but it never disappears, and on smaller datasets it is felt. Against F2 this is disqualifying. The per-file loop also forfeits Xet's cross-file chunk dedupe and lacks the internal retry behavior the helpers ship with — so it loses on F3 and F5 too.

The helpers are the right primitives. The remaining design question is which helper, when.

---

## Upload pipeline

One pipeline. Every upload — small or huge — goes through the same five steps. The user does not pick a mode; there is no mode to pick.

| Step | HF call                                                                  | What it does                                                                                                                                                     |
| ---- | ------------------------------------------------------------------------ | ---------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 1    | `HfApi.create_pull_request(repo_id, title=...)`                          | Creates a draft PR. Server auto-creates a `refs/pr/<num>` branch initially pointing at `main`.                                                                   |
| 2    | `HfApi.upload_large_folder(folder_path, revision="refs/pr/<num>")`       | Hashes files, pre-uploads chunks via Xet/LFS, commits files to the PR branch in batches. Resumable via `<folder>/.cache/.huggingface/`. **`main` is untouched.** |
| 3    | `HfApi.super_squash_history(branch="refs/pr/<num>", commit_message=...)` | Collapses the N batch commits on the PR branch into one.                                                                                                         |
| 4    | `HfApi.change_discussion_status(discussion_num, new_status="open")`      | Exits draft status.                                                                                                                                              |
| 5    | `HfApi.merge_pull_request(discussion_num)`                               | Fast-forwards `main` to the squashed PR branch tip. **Single new commit on `main`.**                                                                             |

Steps 1, 3, 4, 5 are server-side metadata operations totaling well under a second. Step 2 is the bulk transfer — by far the dominant cost.

**Verified end-to-end on a throwaway repo** (5 files / ~1.3 MB):

- Initial main HEAD `1d73a027`. After PR creation + 11.31 s upload to the PR branch, main is still `1d73a027`. Invariant: main untouched during upload.
- After `super_squash_history`, PR branch had 3 commits collapsed to 1.
- After `merge_pull_request`, main has +1 commit `"Upload dataset (test) (#1)"` and all uploaded files visible.
- A second identical-content upload to a fresh PR branch took **1.28 s** vs the first's 11.31 s — **8.84× speedup**, confirming Xet dedupe works through PR-branch targeting.

Downloads have a single, separate mode — see [Download mode](#download-mode) below.

### Cleanup and failure handling for the HF pull-request branch

Throughout this doc, "PR" refers to a **HuggingFace pull request** created via `HfApi.create_pull_request` — not the GitHub PR the developer is reviewing this design from. HF PRs are first-class objects on the dataset repo, with their own `refs/pr/<num>` branch and discussion thread.

The HF PR branch is the staging area for one upload. Its lifecycle is owned by the worker:

- **On successful merge**: HF's PR machinery cleans up `refs/pr/<num>` automatically. Nothing for us to do.
- **On worker SIGTERM / failure mid-upload**: the PR is left in draft state, `refs/pr/<num>` retains whatever commits made it. The user can **Retry** from the GUI — the retry worker re-uploads to the same PR branch (Xet dedupe makes it cheap) and resumes from `<folder>/.cache/.huggingface/`. Or the user can **Discard** the failed job, in which case the worker closes the discussion (`change_discussion_status("closed")`) as cleanup. (Discard vs Retry vs Hide is spelled out in [Tray card actions](#tray-card-actions-and-their-confirmation-prompts).)
- **On GUI server crash**: PR branches outlive the GUI process; the discussion list on the repo is the persistent record. On next GUI startup, a sweep reconciles the `hub_transfers.jsonl` analytics with open draft PRs and offers Retry or Discard on each.
- **Worst-case "dead" PRs**: if neither the GUI nor the user ever cleans up, the PR branch remains as a draft on the repo. Visible in HF's web UI; not visible to anyone fetching from `main`. A periodic sweep (e.g., > 30 days draft, no associated active job) garbage-collects.

### Upload skip behavior

F3 — "skip already-uploaded files and chunks" — is satisfied for free because HuggingFace classifies each file into one of three storage backends server-side and applies the appropriate dedupe layer. We do not implement skip logic ourselves; we rely on the server's pre-upload negotiation. Critically, all three dedupe layers operate on the repo's content-addressed storage, which is shared across branches — so uploading to a `refs/pr/<num>` branch sees exactly the same dedupe as uploading to `main`.

| Storage backend             | Picks up                                                                                                                        | Granularity of skip                                                                          | Mechanism                                                                                                                                                                                    |
| --------------------------- | ------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Xet** (content-addressed) | Most large binary files in modern datasets repos: parquet, mp4, large arrays. Chosen by the server based on repo configuration. | **Chunk-level.** Content-addressed; the server stores chunks keyed by their hash, repo-wide. | Client splits files into chunks, sends `(chunk_hash, size)` tuples to the server; server replies which chunks it already has. Client uploads only the missing ones.                          |
| **LFS**                     | Files configured for Git-LFS (legacy on most newer dataset repos, but still possible).                                          | **Whole-file.** SHA-256 of the full file, repo-wide.                                         | Client sends the file's SHA-256 in a pre-upload request; server returns "have it, skip" or "need it, upload here." No mid-file dedupe — a 1-byte change forces a full re-upload of the file. |
| **Regular**                 | Small text files: `README.md`, `meta/info.json`, `.gitattributes`.                                                              | **Whole-file**, but transparent because the files are tiny.                                  | Standard Git blob storage, content-addressed.                                                                                                                                                |

`upload_large_folder` invokes this negotiation as a pre-flight step (`_fetch_upload_modes` in `huggingface_hub._commit_api`, called for every batch of files before any bytes are transferred). The server's response includes a `should_ignore` flag per file when the content is fully known, and a list of needed Xet chunks for files where only some chunks are new. The negotiation does not consult the branch — the content store is repo-wide — so dedupe through a PR branch is byte-identical to dedupe through `main`.

Concrete behavior on three real scenarios:

1. **Re-uploading identical content** (e.g., user pushes the same dataset they already pushed last week). Every Xet chunk hash matches; every LFS file's SHA-256 matches; every regular blob's hash matches. Server says "I have all of this." Total bytes transferred: **0** (plus negotiation chatter on the order of a few KB). Verified on the experimental upload: 1.28 s vs 11.31 s for cold first upload.
2. **Re-uploading after editing one parquet's contents.** Xet rehashes the parquet locally, finds most chunks identical to the existing ones, asks the server about the rest. Server returns the list of changed-chunk hashes. Client uploads only those. Total bytes transferred ≈ size of the edited region, not size of the whole parquet.
3. **Re-uploading after replacing a video file.** Xet chunks differ entirely; server has none of them. Client uploads the full new file. This is the worst-case for chunk dedupe and is exactly the right behavior — the file genuinely changed.

`upload_large_folder` additionally caches the local hashing work in `<folder>/.cache/.huggingface/`. A resumed upload (worker retry after SIGTERM or crash) skips not only the bytes the server already has but also the local hashing work. A retry of an aborted upload picks up at the next unhashed-or-unuploaded file.

The takeaway: we get file-level skip for free on all file types, chunk-level skip for free on Xet-classified files (which covers the bulk of LeRobot dataset bytes — parquet + mp4), and the work of building or maintaining that machinery is on HuggingFace's side, not ours. The PR-branch indirection does not weaken any of this.

---

## Download mode

One mode: `snapshot_download(repo_id, repo_type, local_dir=dataset.root, max_workers=8)` directly into the dataset root. No staging directory, no atomic swap.

The temptation to write into `/tmp/<job_id>/` and atomically rename into `dataset.root` is real — "atomic local state on download" sounds responsible. It is the wrong choice here for three independent reasons:

1. **Loses the etag-skip optimization (F4).** HF's etag skip compares against files in `local_dir`. A fresh temp directory has no etag matches, so a "refresh" download has to re-fetch every file end-to-end, even if most are unchanged. Re-downloading an 83 GB dataset because four parquet files changed is the wrong trade.
2. **Clobbers local-only files.** A user who added an episode locally that's not on remote loses it when the temp directory is renamed over `dataset.root`. `snapshot_download` directly into `dataset.root` preserves unknown files.
3. **Adds no real atomicity.** HF already gives us per-file resumability via `.incomplete` cache files + `Range:` requests. A mid-file network drop resumes from byte offset; a missing file gets re-fetched. From the user's perspective, "click Download, network failed at 60%, click Download again, it finishes in seconds" — the partial state is self-healing, not a corruption.

The interaction with our open-dataset precheck closes the last gap. A download interrupted mid-run leaves `dataset.root` partially populated; if the user closes and reopens the dataset before retrying, the precheck returns HTTP 409 with `code=incomplete_local_cache` and the GUI prompts to resume. That mechanism already exists.

`max_workers=8` is HF's default; we don't tune it. The download throughput is bandwidth-bound for any realistic dataset; more workers don't help once the pipe is full.

### Per-file atomicity on overwrite

A fresh file lands atomically in the obvious way (it didn't exist; now it does). The interesting case is **overwriting an existing local file with a newer remote version** — what happens if a process is currently reading the old version, or the download is interrupted between bytes?

HF's writer uses a `<filename>.incomplete` staging file and `os.rename` to its final name only on successful completion + hash verification. POSIX `rename` over an existing file is atomic at the filesystem level: an open file descriptor on the old inode keeps reading the old content; new `open()` calls see the new file. So:

- **Mid-stream interruption.** No partial bytes ever reach `dataset.root/<filename>` itself. The `.incomplete` file holds the partial bytes; on retry, HF's `Range:` request resumes from offset N and the rename happens only at the end.
- **In-flight reader of the old version.** The GUI's video playback or parquet reader holds the file open via an fd. The rename swaps the directory entry but the fd stays bound to the old inode. The reader finishes reading the old content; the next `open()` after rename sees the new content. **No truncation race, no half-overwritten bytes.**
- **Hash mismatch on download.** HF re-downloads the file. The `.incomplete` from the failed attempt is discarded; only a successful hash-verified file gets renamed in. We never see a wrong-content-but-right-size file in `dataset.root`.
- **Windows.** `os.rename` over an existing in-use file fails (e.g., a parquet that the GUI's playback or a video player is actively reading). The worker catches `PermissionError` from this case specifically, surfaces it in the tray as "File is in use by another process — close any open viewers and click Retry," and leaves the partial download intact so retry-after-close resumes from byte offset. LeRobot's primary deployment target is Linux + macOS, so the experience on Windows is "best effort with clear failure mode" rather than fully transparent; closing reader processes is a one-time user action, not a recurring chore.

In short: download overwrites are per-file atomic on POSIX, with no truncation visible to concurrent readers. The guarantee only spans single files — the directory as a whole is not atomic, but the open-precheck (above) is what catches partial-directory states.

---

## Worker IPC

The worker is an internal Python entry point (`python -m lerobot.gui.hub_worker`) that the server spawns and never the user. Config — direction, repo_id, root, job_id, paths — is passed via a single JSON blob (env var or stdin), so no user-facing CLI surface to maintain. Args-vs-stdin-vs-env is implementation detail.

Two files per job, three lifetimes total in the system:

| File                                          | Per-job            | Purpose                                                                                                          | Write pattern                                                         | Lifetime                                     |
| --------------------------------------------- | ------------------ | ---------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------- | -------------------------------------------- |
| `~/.cache/lerobot/gui/hub_jobs/<job_id>.json` | ✓                  | **Progress IPC.** Current state of one transfer; read by the server on each `/hub/jobs` poll.                    | Rewritten atomically (`.tmp` + rename) at ~1 Hz. **Not append-only.** | Deleted on terminal state + 30 min retention |
| `~/.cache/lerobot/gui/hub_jobs/<job_id>.log`  | ✓                  | **Debug + milestone extraction.** Captures HF library stderr verbatim.                                           | Append-only.                                                          | Same as above                                |
| `~/.config/lerobot/gui/hub_transfers.jsonl`   | ✗ (shared, global) | **Analytics.** One line per terminal outcome — see [Analytics](#analytics). Distinct from the per-job IPC files. | Append-only. **One line per job, NOT per progress tick.**             | Persistent across sessions                   |

Don't confuse the per-job progress JSON (transient, IPC, current state) with the global analytics JSONL (persistent, historical log, terminal outcomes only). They live in different directories and serve different purposes.

**Why a filesystem-polled JSON for IPC, not a Unix socket or a stdout pipe.** Three reasons, in order of importance: (1) **debuggability** — `cat ~/.cache/lerobot/gui/hub_jobs/<id>.json` is the lowest-friction way to see what a worker thinks is happening, with no special tooling; (2) **simplicity** — atomic `.tmp` + rename is a 4-line implementation that handles every concurrency case, no socket lifecycle or pipe-buffering edge cases; (3) **survives server restarts** — a worker writes to disk regardless of whether the GUI server is reading at that moment, and the new server picks up the latest state on the next poll. The cost is ~1 KB/s disk write per active job, dominated by the inode-update overhead of `os.rename`; on local storage this is invisible against any actual work the worker is doing. The cache directory must be local — network-mounted `~/.cache` would amplify the I/O cost and is documented as unsupported for this purpose.

### Per-job progress JSON shape

```json
{
  "status": "running",
  "stage": "uploading",
  "started_at": 1779700000.123,
  "milestone": "Pre-uploading LFS files (12 / 47)",
  "milestone_at": 1779700087.456,
  "files_total": 47,
  "files_done_estimate": 12,
  "bytes_total": 7686801234,
  "bytes_done_estimate": null
}
```

The `_estimate` suffix is honest: the GUI displays whatever the worker can extract from HF's text output. For `upload_large_folder`, the `print_report` text gives file counts as it progresses; for `snapshot_download`, parallel tqdm bars give per-file progress that the worker aggregates into a single "X of N files" number. We do not claim byte precision (see N1).

On the server side, `GET /api/datasets/hub/jobs` reads each running job's progress file once per request and merges the contents into the `HubJobState` it returns. The file is the IPC; `HubJobState` is the projection clients see.

### Per-job stderr log

The worker redirects HF's stderr to `<job_id>.log` under the same directory. Two purposes:

- **Debug.** Full HF output preserved for post-mortem on a failed transfer.
- **Milestone extraction.** Lines matching certain patterns (e.g., `Fetching N files`, `Pre-uploading file ...`, `Committing ...`) drive `milestone` updates in the progress JSON. Best-effort parsing; if HF's format shifts, milestones degrade to `"running"` and the rest of the system stays functional.

### Robustness of stderr-based progress

Parsing tqdm output from HF's helpers is intentionally a soft dependency. We design assuming it can break across HF versions and keep the system functional when it does:

- **Pinned HF version**: `huggingface_hub` is pinned to a specific minor in `uv.lock`. A `huggingface_hub` upgrade is a deliberate user action, not a silent supply-chain shift.
- **Read by byte, not line.** tqdm uses `\r` (carriage return) to overwrite progress on the same terminal line, not `\n`. The worker reads its subprocess's stderr stream byte-by-byte (or with `\r`-aware line splitting), not via the default line-buffered text reader. A naive `for line in proc.stderr` misses every progress tick.
- **Unit tests with recorded samples.** A handful of pinned stderr captures from real `upload_large_folder` / `snapshot_download` runs live as test fixtures; the milestone parser is exercised against them, and any regression in the recognized patterns fails fast. When HF's format does shift, updating the fixtures is the breakage signal.
- **E2E test on a tiny real upload.** A `pytest` marker (`@pytest.mark.hub_live`) runs an actual upload of a few-MB random payload to a throwaway repo, asserts that the milestone parser produces at least the "uploading", "committing", "complete" transitions, and cleans up. Gated behind opt-in (network + auth required) so it doesn't run in default CI but does run when we touch this code.
- **Graceful degradation.** If parsing fails entirely on a given line, the milestone falls back to `"running, <elapsed>s"`. The job still completes correctly; the user just sees a less informative status. Correctness is independent of parser quality.
- **Cross-platform.** tqdm output is the same on Linux, macOS, and Windows in terms of byte content. The `\r`-overwrite handling, however, depends on disabling line-buffering at the `subprocess.Popen` level — `bufsize=0` and `text=False`. The parser then handles encoding to UTF-8 itself. (Windows console quirks around `\r\n` are handled by treating `\r` as the progress-update delimiter regardless of OS.)

### Cancel

`Transfers.cancel(job_id)` on the GUI calls `POST /hub/progress/{job_id}/cancel`. The server `os.killpg(worker.pid, SIGTERM)`s the subprocess group. The cancel path is designed under the assumption that **the worker may be buggy or stuck and must not be trusted to do the right thing**. If the worker is well-behaved, we lose at most some progress; if it's misbehaved, we never compromise correctness.

Specific patterns:

- **PID-reuse safety**: the server verifies `(pid, process_start_time)` matches what it recorded at spawn before sending SIGTERM. On Linux, `/proc/<pid>/stat` field 22 is the start_time in jiffies since boot; on macOS, `psutil.Process(pid).create_time()`. If the recorded tuple doesn't match, the PID has been reused and we **do not kill** — we just mark the job failed locally. (The original worker is presumed dead anyway.)
- **Escalation**: SIGTERM → 5-second grace → SIGKILL the process group. SIGKILL cannot be ignored.
- **Server is the source of truth for terminal status**, not the worker. After SIGTERM the server `waitpid`s the worker; only on confirmed exit does the server write `status: cancelled` to the progress JSON. The worker's own SIGTERM handler is best-effort — if it manages to write `cancelled` first, great; if not, the server's write wins on the next `waitpid` return.
- **Idempotent**: repeated cancel requests on a job that's already terminal or already mid-cancel are no-ops. Returns the current status without re-issuing signals.
- **PID file ownership**: at spawn time the worker writes `<job_id>.pid` containing `{pid, start_time, started_at}`. Any code that wants to act on a job (cancel, status read, server-restart sweep) consults this file. A stale `.pid` whose recorded process doesn't exist or doesn't match start_time is treated as a dead-worker indicator.
- **Server-restart sweep**: on GUI server startup, every `.pid` file in `~/.cache/lerobot/gui/hub_jobs/` is checked. If the PID is alive and start_time matches, the job is adopted and the progress JSON keeps being read. If it's dead, the job is marked `failed` ("worker exited without finalizing") and the `.pid` cleaned up.
- **Worst-case (worker stuck in uninterruptible syscall, SIGKILL'd but still consuming an fd somewhere)**: this is rare but possible. The OS reclaims the resources eventually; our state stays consistent because the server confirmed exit via `waitpid` and updated the progress JSON accordingly. The user sees "cancelled," can retry, and the new worker spawns cleanly (the old PID is gone from the process table after kernel cleanup).

After cancel:

- For the upload pipeline: the worker leaves the draft PR + `refs/pr/<num>` and the local `<folder>/.cache/.huggingface/` untouched. Retry reuses both, preserves Xet pre-upload state, picks up at the last completed file. The PR is closed only when the user explicitly **Discards** the cancelled job (see [Cancellation and pause](#cancellation-and-pause)).
- For downloads: partial files + HF's `.incomplete` markers stay; retry resumes via `Range:` requests.

### Spawn

A worker is spawned in response to user action (Upload, Download, Retry). Multiple safety patterns prevent races:

- **Per-dataset spawn lock**: the server holds an `asyncio.Lock` keyed by `dataset_id` over the spawn step. Rapid Upload clicks queue rather than racing.
- **Pre-spawn registry check**: before constructing a new worker, the server checks `_app_state.active_hub_job_for(dataset_id)`. If a non-terminal job exists, the endpoint returns 409 with the existing `job_id`. The frontend treats 409 by attaching to the existing job instead of creating a new one. (This invariant is enforced by the same lock from the previous bullet — read and write of the registry happen under the lock.)
- **Cross-process invariant via PID files**: spawn writes the `<job_id>.pid` file atomically (`.tmp` + rename) before exec'ing the worker. If two server processes (e.g., a stale one and a freshly started one) both try to spawn for the same job, the second's PID-file write either races and one wins, or — better — the second's server-restart sweep finds the first's PID file and adopts the existing worker instead of spawning a duplicate.
- **Retry semantics**: Retry is just a fresh spawn with the same args. The retry path inspects `hub_transfers.jsonl` for the most-recent terminal entry of `(dataset_id, repo_id)` and, if an HF PR number is recorded with `status: draft`, reuses it. Otherwise creates a new PR. This is what makes Retry cheap (Xet sees the pre-uploaded chunks already on the server-side content store).
- **Idempotent at the user level**: if the user clicks Retry twice in rapid succession on the same job, the second click is absorbed by the spawn lock + registry check; only one new worker comes up.

Best-practice references the design follows: PID-file + identity-tuple verification ([rsyslog's pid-file design notes](https://www.rsyslog.com/doc/), [systemd PIDFile= semantics](https://www.freedesktop.org/software/systemd/man/systemd.exec.html)), `waitpid`-confirmed termination (POSIX), and the "server is the source of truth, worker is opportunistic" invariant common to long-running task queues (Celery, RQ, Sidekiq all follow this pattern for resilience against misbehaving workers).

### Resume across sessions

Three layers of persistence make a transfer resumable across user actions of increasing severity:

1. **Tab close** (browser closes, GUI server keeps running). No-op for the worker; the in-memory `AppState.hub_jobs` registry holds the state, the worker subprocess keeps running, the next browser tab opens and reads `/api/datasets/hub/jobs` to see exactly what the previous tab saw. F1 covers this.
2. **GUI server restart / app reopen** (server process killed and restarted, worker may or may not have survived per orphan-subprocess infrastructure). On startup, the server scans `~/.cache/lerobot/gui/hub_jobs/*.pid` files. For each:
   - Verify the recorded `(pid, start_time)` matches a live process. If yes → adopt the worker; the job re-enters the registry with its progress JSON as the source of truth.
   - If no → mark the job `failed` in the registry with reason `worker exited without finalizing`, append a terminal entry to `hub_transfers.jsonl`, clean up the `.pid` file.
   - For each entry now in `failed` state from a sweep, the tray shows it as a recently-failed card. If the failed entry has an HF `pr_num` recorded, **Retry** reuses that PR.
3. **Retry of a terminal job** (failed / cancelled, possibly across server restarts or days later). The Retry path looks up `(dataset_id, repo_id)` in `hub_transfers.jsonl`:
   - If the most recent terminal entry has `pr_num=N` and the PR is still in draft state on HF (queried via `get_discussion_details(N)`) → reuse PR N. The worker spawns with `revision=refs/pr/N`, `upload_large_folder` reads its local `<folder>/.cache/.huggingface/`, sees what's already pre-uploaded, and continues from there.
   - If the PR is closed or merged → create a fresh PR. Xet dedupe still makes the byte transfer cheap; only the negotiation overhead is paid again.
   - If no terminal entry exists for this (dataset, repo) → fresh PR. Standard path.

This means **a user who starts a 6-hour upload, closes their laptop overnight, comes back, opens the GUI fresh, and clicks Retry on the failed tray entry resumes from approximately where the upload stopped** — the local `.cache/.huggingface/` survives shutdown, the HF PR survives indefinitely, the (pid, start_time) sweep handles the worker-died case, and `hub_transfers.jsonl` ties them together by `pr_num`. The PR-clutter risk Gemini raised is bounded by the "reuse existing PR" path in step 3 above; the cleanup sweep (in [Open questions](#open-questions)) handles the long-tail of PRs that were never retried.

---

## Error model

HF's helpers retry internally on transient 5xx and connection errors with exponential backoff. The worker treats their final exception as terminal: it writes `status: failed, error: <stringified exception>` and exits.

Three error classes get **specific handling** because the generic "show the error string" message doesn't tell the user what to do:

| Error class                                                                                                          | Detection                                                                           | Tray message                                                                                             | Suggested user action                                                                                     |
| -------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------- |
| **Auth failure** (`HfHubHTTPError` with 401/403, or `RepositoryNotFoundError` with `private_inaccessible` indicator) | Worker catches before re-raising; tags `error_class="auth"` in progress JSON        | "Authentication failed. Your HF token may be expired, revoked, or lacks write permission for this repo." | Run `huggingface-cli login` (or refresh token in the GUI's auth panel when shipped) and click **Retry**.  |
| **Rate-limited** (`429 Too Many Requests`, surfaces after HF's internal retries are exhausted)                       | Same pattern; `error_class="rate_limit"`                                            | "The HF Hub is rate-limiting this request. Wait a few minutes and retry."                                | Wait; click **Retry**. The cost of the wasted upload-so-far is bounded by Xet dedupe on the next attempt. |
| **Network-class** (`ConnectionError`, `ReadTimeoutError`, `HfHubHTTPError` with 5xx)                                 | HF's helpers already retry these internally; what reaches us is "tried and gave up" | "Connection error: `<verbatim message>`"                                                                 | Click **Retry**. Resume is cheap via HF's primitives.                                                     |

For long-running uploads (multi-hour, multi-TB), auth-expiry is the failure mode most likely to bite: a token that was valid at job start might rotate during a 6-hour upload. The handling above means the user sees a clean "token expired" message rather than a generic 500 stack trace, and Retry-after-refresh-token resumes from where the upload stopped.

What the GUI does with terminal failures:

| Surface             | Behavior                                                                                                                                                                |
| ------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Transfers tray card | Red-bordered, shows the error string verbatim. Per-card **Retry** and **Discard** buttons (see [Tray card actions](#tray-card-actions-and-their-confirmation-prompts)). |
| Toast               | One-shot "<Direction> failed: <error>" with 8-second visibility.                                                                                                        |
| Analytics line      | JSONL entry with `outcome: "failed"`, `error`, `duration_s`, all counters as last known.                                                                                |

**Retry** is a fresh worker spawn. For uploads, the worker checks `hub_jobs.jsonl` to see if a draft PR exists for this dataset+repo combination — if so, it reuses that PR's `refs/pr/<num>` so resume picks up the server-committed state, the local `.cache/.huggingface/`, and the Xet pre-upload cache simultaneously. For downloads, the `.incomplete` cache + etag skip handle resume without any client-side bookkeeping. We don't build retry logic on top — HF's resume primitives are the retry mechanism; we just re-invoke them.

### What "elegant" means specifically (E1–E4)

- **E1.** The error string in the tray is the HF exception message, not "Failed". Users debug from the GUI.
- **E2.** A Retry button on every failed card spawns a fresh worker with identical args; resume is automatic via HF's primitives.
- **E3.** Ambiguous-result uploads (network drop after merge_pull_request RPC sent but before client got the 200) are resolved by Retry: it inspects the PR's state via `get_discussion_details`, observes whether the PR is already merged, and either (a) starts fresh if it's merged or (b) re-attempts the merge if it's not. Xet dedupe handles any re-transmission cheaply. No user investigation required.
- **E4.** Every terminal outcome writes a JSONL line with enough detail to retry from CLI if the GUI is unavailable: `repo_id`, `root`, `pr_num` (if any), last `milestone`. A debug command can read the JSONL and replay.

### Cancellation and pause

We deliberately do not expose a "pause" button distinct from cancel. The reason: in our design, cancel and pause have the same implementation — kill the worker, leave the resume state alone. A subsequent Retry picks up where the cancelled job left off via HF's resume primitives. Distinguishing "paused" from "cancelled" in the UI without a behavioral difference would add a state with no semantics.

A theoretical alternative is OS-level suspension: `SIGSTOP` the worker on pause, `SIGCONT` on resume. This avoids re-establishing connections + re-negotiating Xet chunks, so theoretically faster than cancel-then-resume. In practice it breaks: HF's TCP connections idle-time-out on the server side during long suspension (minutes), then `SIGCONT` resumes into a dead socket that fails on next syscall; and the signal handlers for upload_large_folder's worker threads have race conditions with `SIGSTOP`. Not worth the fragility for a UX nicety the cancel-then-retry path already covers.

If pause becomes a real ask later, the right shape is a separate worker pool with checkpoint serialization that doesn't rely on OS signals. Out of scope here.

### Tray card actions and their confirmation prompts

Every tray card has at most one or two action buttons depending on its state. They are visually distinct (× alone is not enough — there are two meanings of × in this design, and we don't conflate them):

| Card state                       | Available actions                   | Confirmation prompt                                                                                                                                                                              |
| -------------------------------- | ----------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| Active upload, mid-transfer      | **Cancel** (✕)                      | "Cancel upload? Already-uploaded chunks stay on the server. The pending HF PR remains in draft so Retry can resume."                                                                             |
| Active upload, no bytes sent yet | **Cancel** (✕)                      | None — nothing was sent.                                                                                                                                                                         |
| Active download                  | **Cancel** (✕)                      | None — `Range:`-resume on retry is free.                                                                                                                                                         |
| Cancelled or failed              | **Retry**, **Discard** (trash icon) | Discard prompt: "Discard upload? The pending HF PR will be closed and partially uploaded data will be cleaned up. Resume will no longer be possible. Use Retry to resume from where it stopped." |
| Complete                         | **Hide** (✕)                        | None — the upload is on `main`; this just removes the tray entry from view.                                                                                                                      |

**Cancel** (✕ on active) stops the worker; **Discard** (trash icon on terminal) cleans up the orphan staging PR on HF; **Hide** (✕ on complete) is a UI-only no-op. Three distinct verbs, three distinct icons. This addresses the past confusion around "dismiss" — that word is replaced by either **Discard** (destructive cleanup of remote state) or **Hide** (purely visual) depending on whether there's anything left to clean up.

---

## Safety guardrails

Two corruption scenarios worth pre-empting:

> **Upload-fail → bad download.** _User uploads dataset to repo R, upload fails halfway, user doesn't notice. Later the user (or another user) downloads from R, overwriting the local copy with a half-uploaded state. Work is lost._
>
> **Download-fail → bad upload.** _User downloads dataset from repo R, download fails halfway, user doesn't notice. User then makes local edits and uploads. The upload pushes a dataset missing whatever the download didn't finish fetching — so the remote loses files that were there before the partial-download._

Under the PR-based pipeline neither **can happen** for any failure mode short of HF itself losing data. The defense against the first lives in the atomic PR merge; the defense against the second lives in a completeness check at upload time + HF's immutable history. Each protection layer is explicit below.

### What HuggingFace gives us

HF dataset repos are git-backed. Concretely:

- **Full git history.** Every commit is immutable and inspectable; nothing is silently overwritten. `HfApi.list_repo_commits(repo_id, repo_type="dataset")` returns the full history. Users can also browse it in the HF web UI's "Commits" tab.
- **Content integrity at upload time.** LFS files are verified against their declared SHA-256 by the server; Xet chunks are content-addressed (the chunk hash _is_ its identity, so a corrupted chunk simply doesn't match anything). A file that arrives intact stays intact.
- **Revertibility.** Any committed state is a candidate for `revert` via the API or web UI. There is no concept of "lost commit" in HF.
- **Atomic commits.** A `create_commit` either succeeds (the new commit is the branch's new HEAD) or fails (HEAD unchanged). There is no half-commit.
- **PR branches are first-class.** `refs/pr/<num>` exists, has history, is queryable; it's just not on `main`.

### What each failure mode leaves behind

| Failure                                                                                                    | State on `main`                                      | State on `refs/pr/<num>`                                                                                                                         | State locally                                                                                  | Downstream consumer fetching from `main` sees                       |
| ---------------------------------------------------------------------------------------------------------- | ---------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------ | ---------------------------------------------------------------------------------------------- | ------------------------------------------------------------------- |
| **PR creation fails**                                                                                      | Untouched                                            | N/A (never created)                                                                                                                              | Nothing changed                                                                                | Same `main` as before the attempt                                   |
| **`upload_large_folder` to PR branch fails mid-flight**                                                    | Untouched                                            | Whatever commits made it (recoverable on Retry via `.cache/.huggingface/`)                                                                       | Local files untouched                                                                          | Same `main` as before                                               |
| **`super_squash_history` fails** (most likely cause: server-side timeout on a multi-hundred-commit branch) | Untouched                                            | All upload commits intact (unsquashed). Worker falls back to merging without squashing — see [Squash failure fallback](#squash-failure-fallback) | Same                                                                                           | Same `main` as before until the fallback merge completes            |
| **`change_discussion_status` to "open" fails**                                                             | Untouched                                            | Squashed state intact                                                                                                                            | Same                                                                                           | Same `main` as before                                               |
| **`merge_pull_request` fails**                                                                             | Untouched                                            | Squashed commit ready to merge                                                                                                                   | Same                                                                                           | Same `main` as before                                               |
| **Network drop between `merge_pull_request` request sent and 200 received (ambiguous)**                    | _Maybe_ updated server-side; client doesn't know yet | Either pre-merge (if server didn't process) or post-merge (HF auto-cleans `refs/pr/<num>` on successful merge)                                   | Same                                                                                           | _Either_ the old `main` _or_ the new `main` — never a partial state |
| **GUI server crash mid-upload**                                                                            | Untouched                                            | Whatever made it before crash                                                                                                                    | Worker subprocess dies (via PDEATHSIG when that infrastructure lands; or remains orphan today) | Same `main` as before                                               |

**The cross-cutting invariant: `main` only ever changes via `merge_pull_request`, and `merge_pull_request` is atomic — it either updates main to the (squashed-or-not) PR tip in one go, or doesn't.** Every other operation (PR creation, upload, squash, status change) touches only the PR branch.

### Squash failure fallback

`super_squash_history` is a single HTTP call to HF that the server processes synchronously. For a multi-TB upload spanning hundreds of `upload_large_folder` batch commits, server-side squash time can exceed HF's request timeout — surfacing as `504 Gateway Timeout` or a generic `HfHubHTTPError`.

The worker handles this by **falling back to merge-without-squash**:

1. After upload completes, call `super_squash_history(branch=refs/pr/<num>)` with a short timeout (e.g., 60 s).
2. If it succeeds → PR has one commit, merge as usual. Done.
3. If it fails (timeout, 5xx, any error) → log the squash failure to `hub_transfers.jsonl` (`squashed=false`), skip the squash entirely, and proceed to merge. The unsquashed PR branch merges into main as a fast-forward, so atomicity is preserved at the merge moment — `main` jumps from its old HEAD to the PR branch tip in one ref update. The cost is N commits visible on main's history instead of 1; the dataset is fully consistent.
4. The tray's completion toast says "Upload complete (N commits on main; squash failed and was skipped)" so the user knows.

This trades commit-history hygiene for not-getting-stuck. A user who cares can manually call `super_squash_history` on `main` later via the HF web UI or API, and we expose this as an explicit Retry-Squash action in the tray for the latest entry if they ask for it.

The fast-forward merge of an unsquashed PR is well-defined HF behavior: since the PR branch was created off `main` and only added commits on top, `main` → PR tip is a strict ancestor relationship and merge is a ref update, not a 3-way merge. No conflict possible.

### How a downstream consumer (or future-you) detects an interrupted upload

- **Read `main`**: it's either the prior-state or the post-merge-state. Never partial. So a `snapshot_download` from `main` is always safe.
- **List recent PRs / discussions**: `HfApi.get_repo_discussions(repo_id, repo_type="dataset")`. Draft PRs from this codebase are visibly named ("Upload dataset (test)") and have associated branches that can be inspected. A user investigating "did my upload finish?" can look here.
- **Web UI**: HF shows pending PRs in the "Community" / "Discussions" tab. The PR's "Files changed" view shows the staged state for inspection.
- **Local GUI**: `hub_transfers.jsonl` records every terminal outcome with the PR number. If the user (or future-them on the same machine) wants to know "did my last push to this repo succeed?", grepping the JSONL is the source of truth. The GUI can surface this proactively — see Local safety surfacing below.

### Download-fail → bad upload — the upload-side guardrail

The mirror-image scenario is uploading from a corrupted local state. Concrete failure path:

1. User downloads `repo/A` to `dataset.root`. Network drops mid-flight at 80%; the GUI tray shows "cancelled" but the user misses it.
2. User makes some local edits — adds an episode, fixes a label — without ever re-opening the dataset (so the open-precheck doesn't fire).
3. User uploads. Without a guardrail, this pushes a dataset missing the 20% the download didn't finish, **overwriting the good remote with a worse one**.

Defenses, layered:

- **Upload-time completeness check.** Before constructing the upload operations, the worker queries `HfApi.dataset_info(repo_id).siblings` (when the repo already exists on the Hub) and diffs against the local file set. Any file present on the remote but missing or `.incomplete` locally is a red flag: the worker pauses, the tray surfaces "Your local copy is missing files that exist on the remote — this looks like an interrupted download. Re-download or confirm to upload anyway." User confirms or aborts.
- **HF's immutable history is the safety net.** Even if a bad upload lands (user confirmed past the warning, or the warning was off), HF retains the prior commit. Reverting via `HfApi.create_commit` with the old tree, or via the web UI, restores the good state. No data is ever truly lost — it's just one commit deeper in history.
- **The open-precheck for re-opens.** A user who closes and re-opens the dataset between the failed download and the upload attempt is caught by `_check_local_dataset_complete`'s 409 response and forced to acknowledge the incomplete state. The upload-time check is the same logic, run at a different lifecycle point, to catch the "never re-opened" path.

For models (when this design extends to `repo_type="model"`): the access pattern is read-mostly, so download-fail-upload is rarer. But the same guardrail applies — upload-time completeness check against the remote's siblings. The framework's symmetry holds across repo types.

### What we don't promise

- **Cross-machine awareness.** If you upload from machine A and the upload fails draft on the repo, then download from machine B, B does not _automatically_ know that A had a pending failed upload. But B downloading from `main` still gets a consistent state (just possibly missing files A intended to add). Inspecting the repo's PRs on the web UI reveals A's draft.
- **Per-file conflict resolution between concurrent draft PRs.** If A's draft PR adds files, then B uploads a different set of files without inspecting A's draft, B's PR overwrites A's draft on merge. Both PRs are visible in the discussions list; we don't proactively warn about overlap. This is acceptable behavior — the use case (large datasets, no exclusive write locks anywhere) doesn't have a clean "merge" semantics; last-merged-wins is the natural and expected outcome, matching how git itself behaves under similar conditions.

### Local safety surfacing

The GUI proactively protects the user against the failure-mode-they-didn't-notice case in both directions:

- **Before any download to `dataset.root`**, the GUI checks `hub_transfers.jsonl` for the most recent terminal outcome of any upload from this machine to the same repo. If the most recent outcome is `failed` or `cancelled`, the download confirmation prompt includes: _"Your last upload to this repo on this machine ended in `<status>` state and may not be complete. Check the repo's open PRs before overwriting your local copy."_
- **Before any upload from `dataset.root`**, the worker runs the upload-time completeness check above. The same `hub_transfers.jsonl` lookup also catches the case where the user's most recent terminal outcome was a failed download — the prompt becomes: _"Your last download from this repo on this machine ended in `<status>` state. Your local copy may be missing files. Re-download or proceed?"_
- **The open-dataset precheck** already catches a half-downloaded local cache (HTTP 409 `incomplete_local_cache`) on open. It does not need adjustment for this design.
- **The Transfers tray** retains terminal entries for 30 minutes specifically so a user who walks away returns to a clearly-visible "your transfer failed" indicator with the error string + Retry button. Not silently hidden.

The combined effect: a failure that goes unnoticed on `main` is structurally impossible (atomic merge), a failure that goes unnoticed locally is opt-in (terminal tray entries don't auto-dismiss), and the symmetric upload-fail-download / download-fail-upload pair is caught by completeness checks at both directions' entry points.

---

## UX surface

The visible surface is the Transfers tray in the tab bar — already on this branch as scaffolding. The pieces:

- **Indicator pill** in the tab bar, hidden when no jobs exist, cyan + pulsing when at least one job is active, neutral when only terminal entries remain.
- **Popover** anchored under the pill, listing one card per active or recently-terminal job. Each card shows direction, clickable repo link, status, milestone text from the worker, and the per-card action buttons described in [Tray card actions](#tray-card-actions-and-their-confirmation-prompts) (Cancel for active jobs, Retry + Discard for terminal failed/cancelled, Hide for terminal complete).
- **Hub Upload modal** stays simple: repo_id input + remote/local info + the Upload button. No mode dropdown — since there's only one upload pipeline, there's nothing to select.
- **Per-card link to the PR** during the upload (while the worker has a PR open): clicking the repo link in the tray jumps to the PR on the HF web UI, so a user wanting to inspect the in-flight state can do so. For terminal cards the link goes to the merged commit (success) or to the closed PR (cancelled/failed).

No new top-level UI; this is an evolution of the existing tray, not a new surface.

---

## Analytics

One JSONL file at `~/.config/lerobot/gui/hub_transfers.jsonl`, one line appended **per terminal outcome** (not per progress tick — that's a different file, see [Worker IPC](#worker-ipc)). Persistent across sessions; never pruned automatically.

Schema:

```json
{
  "ts": 1779700123.456,
  "direction": "upload",
  "repo_id": "thewisp/cylinder_ring_assembly",
  "dataset_id": "/home/feit/.cache/.../cylinder_ring_assembly",

  "started_at": 1779696523.111,
  "finished_at": 1779700123.456,
  "duration_s": 3600.345,

  "files_total": 412,
  "files_done": 412,
  "bytes_total": 47800000000,
  "bytes_transferred_estimate": 31200000000,

  "pr_num": 17,
  "pr_commits_pre_squash": 27,

  "outcome": "complete",
  "error": null
}
```

What we mine it for:

- **Failure clustering.** Group `error` strings by direction. Cluster what's actually breaking.
- **Throughput-over-time.** Plot `bytes_total / duration_s` over weeks. A sudden drop in throughput is a network or HF-side regression we can localize.
- **Xet dedupe ratio.** `bytes_transferred_estimate / bytes_total` over time. If this drops, something changed in how we ignore-pattern or how Xet sees our data.
- **Cancel rate.** Fraction of jobs with `outcome=cancelled`, segmented by `direction` and time-to-cancel. Tells us whether users are aborting because something looks stuck.
- **Local safety lookup.** Before any download, check whether this `(machine, repo_id)` has a recent failed/cancelled upload — used by the [Local safety surfacing](#local-safety-surfacing) confirmation prompt.

The format is plain JSONL, greppable from a shell, parseable with one-line Python. No tooling is required for the file to be useful; a Stats view in the GUI can come later if a question warrants it.

---

## Extending to model repos

The current endpoints live under `/api/datasets/.../hub/...` and the spawn paths default `repo_type="dataset"` because that's this module's scope. The underlying machinery — `JobConfig`, `HubJobState`, the worker subprocess, the IPC files, the tray UI, the cancel/retry/discard flow — is **already `repo_type`-agnostic**. Six HF calls in the worker pipeline all read `cfg.repo_type`; the retry-PR lookup reads `job.repo_type`; the cancel/dismiss paths read `job.repo_type`.

Adding model support means a small parallel layer, not a rewrite. The shape:

- **New endpoints** under `/api/models/{model_id}/hub/upload` (and `…/hub/download`) that mirror the dataset ones but call `make_job(..., repo_type="model")` and `_spawn_hub_worker(...)` exactly as the dataset endpoints do. Probably ~80 lines total — most is parameter wiring + the auto-open path adapted for model objects on `_app_state`.
- **Shared progress/cancel/dismiss endpoints**: the existing `/api/datasets/hub/jobs`, `/api/datasets/hub/progress/{job_id}/cancel`, etc. are already generic — they look the job up by `job_id`, not by repo type. Either reuse them as-is, or (cleaner) move them to `/api/hub/...` so the URL doesn't lie about repo-type scope.
- **Worker code**: no changes. The worker only reads `cfg.repo_type` and passes it through.
- **Frontend tray**: no changes. The cards already render `j.repo_type` from the job state and link to the right repo namespace; existing CSS / templates already handle "this card might be a model upload."
- **Tests**: add a `repo_type="model"` variant of the live E2E test once model uploads are in scope. The unit tests already exercise the `Literal["dataset", "model"]` boundary in `JobConfig.__post_init__`.

The one wrinkle: completeness check (`check_upload_completeness`) currently calls `api.repo_info(repo_id, repo_type=repo_type, files_metadata=True)` which works for both datasets and models. But what counts as "incomplete locally" for a model differs — there's no `.parquet` index, files are usually a flat set of `*.safetensors` + `config.json` + tokenizer. The siblings-diff approach works regardless; the user-facing copy in the warning prompt may want to say "model files" rather than "dataset files," which is a one-line UI tweak.

When the time comes, the implementation order is: (1) add `/api/models/{id}/hub/upload` and `/download` endpoints calling the existing spawn helpers, (2) decide whether to keep job-tray endpoints under `/api/datasets/hub/...` or refactor to `/api/hub/...`, (3) tweak the completeness-check warning text to be model-aware, (4) add a live E2E test with a small model repo.

---

## Open questions

- **Re-enable `super_squash_history`.** Currently disabled because it causes `BadRequestError: There are merge conflicts` on the subsequent merge for second-upload-to-the-same-repo. Need to either (a) find the correct HF API usage that preserves the fast-forward relationship, (b) follow squash with a separate "rebase onto current main" call, or (c) accept multi-commit main history as the permanent design (and remove the squash machinery entirely). Live E2E test `tests/gui/test_hub_live.py::TestDedupePerformance` reproduces the failure when squash is re-enabled.
- **Per-worker concurrency limit.** Two parallel transfers on different datasets currently run two subprocesses competing for the same uplink/downlink. At realistic bandwidths this is fine; at gigabit-plus with many small files in parallel transfers, we may want a global N=2 cap to avoid head-of-line stalls. Defer until a measurement says it matters.
- **Stale-PR sweep cadence.** Failed uploads leave draft PRs. We'll sweep them eventually — open question is whether the sweep runs on GUI startup, on every Upload-modal-open (so the user sees stale PRs from past failed attempts), or on a periodic timer. Likely all three with appropriate filters.
- **Cross-machine PR conflict warning.** If two machines independently push to the same repo, both create draft PRs and the second silently overrides the first. Adding a "Heads up: there's a recent draft PR on this repo from another session" warning is straightforward (`get_repo_discussions` filter on `status=draft`) but not in scope here.
- **Mode parity for downloads.** Today downloads use one mode (`snapshot_download` into `dataset.root`). If a future workload needs explicit checkpointing beyond HF's per-file `.incomplete`, we'd add a checkpointed-download mode. Not currently needed.
- **Retry budget surfacing.** A retry that fails should probably surface differently from a first-attempt failure (e.g. "Failed 3× in a row — check connection" instead of just the latest error). Not designed; small future addition.

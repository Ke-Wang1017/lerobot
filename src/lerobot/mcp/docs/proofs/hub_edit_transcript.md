## Hub edit-tier MCP — transcript

Captured live against the unified GUI's MCP at `127.0.0.1:8000/mcp/`. The transcript clones the throwaway dataset `thewisp/test_leader_follower_do_not_use` locally to `_mcp_upload_proof_68ad7c/dataset`, uploads it to a throwaway HF Hub repo `thewisp/_mcp_upload_proof_68ad7c`, polls until the worker reaches a terminal state, then DELETES both the Hub repo and the local clone. The operator's original throwaway dataset is **never** modified.

## hub_auth_status — pre-flight

_Intent: Any responsible agent calls this before kicking off an upload. If logged_in=false the upload would 401._

**→** `hub_auth_status()`

**←**

```json
{
  "logged_in": true,
  "username": "thewisp"
}
```

## hub_start_upload — kick off the real upload

_Intent: Spawn the worker subprocess that creates the repo, opens a PR, pushes the dataset, and merges. Returns a job_id immediately; we poll for completion._

**→** `hub_start_upload(dataset_id='_mcp_upload_proof_68ad7c/dataset', hub_repo_id='thewisp/_mcp_upload_proof_68ad7c')`

**←**

```json
{
  "job_id": "6482558346104a6a93a9f894d9b21610",
  "status": "started"
}
```

## Polling hub_job_progress until terminal

poll: status='running' milestone='Processing files 0 / 0' files=0/0

poll: status='complete' milestone='Upload complete (merged unsquashed)' files=0/0

## hub_job_progress — final snapshot (complete)

**→** `hub_job_progress(job_id='6482558346104a6a93a9f894d9b21610')`

**←**

```json
{
  "job_id": "6482558346104a6a93a9f894d9b21610",
  "dataset_id": "_mcp_upload_proof_68ad7c/dataset",
  "direction": "upload",
  "repo_id": "thewisp/_mcp_upload_proof_68ad7c",
  "repo_type": "dataset",
  "status": "complete",
  "started_at": 1780916903.9940555,
  "finished_at": 1780916919.3947847,
  "stage": "done",
  "milestone": "Upload complete (merged unsquashed)",
  "milestone_at": 1780916919.3944097,
  "files_total": 0,
  "files_done_estimate": 0,
  "bytes_total": 0,
  "bytes_done_estimate": 0,
  "current_file": null,
  "error": null,
  "error_class": null,
  "pr_num": 1,
  "pr_url": "https://huggingface.co/datasets/thewisp/_mcp_upload_proof_68ad7c/discussions/1"
}
```

Final status: 'complete'

## Error path — unknown job_id

_Intent: Confirm the cancel path's error shape — same clean tool-error pattern the read-tier hub_job_progress uses._

**→** `hub_cancel_job(job_id='does-not-exist')`

**← error** — `Error executing tool hub_cancel_job: Job not found: does-not-exist`

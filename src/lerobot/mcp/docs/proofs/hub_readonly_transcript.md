## Hub read-only MCP — transcript

Captured live against the unified GUI's MCP at `127.0.0.1:8000/mcp/`. All calls are read-only; nothing on the operator's host or HF account is mutated.

## hub_auth_status — host auth probe

_Intent: Ask whether the operator's host has a working HF Hub login. Used as a pre-flight before any upload/download tool._

**→** `hub_auth_status()`

**←**

```json
{
  "logged_in": true,
  "username": "thewisp"
}
```

## hub_repo_info — remote repo lookup

_Intent: Probe a public dataset known to exist: `lerobot/aloha_sim_insertion_human`. The agent typically calls this before proposing a sync so it can warn the operator about the transfer size._

**→** `hub_repo_info(repo_id='lerobot/aloha_sim_insertion_human')`

**←**

```json
{
  "exists": true,
  "repo_id": "lerobot/aloha_sim_insertion_human",
  "private": false,
  "last_modified": "2026-03-05 16:47:58+00:00",
  "downloads": 34919,
  "files": 11,
  "total_size_mb": 91.3,
  "sha": "a3565744bd67",
  "total_episodes": 50,
  "total_frames": 25000,
  "fps": 50
}
```

_Intent: Probe a deliberately-missing repo: `nope/this_does_not_exist_for_sure_2026`. The response collapses every failure mode (404 / 401 / network) to `exists: false` so the agent can branch on the boolean instead of parsing error strings._

**→** `hub_repo_info(repo_id='nope/this_does_not_exist_for_sure_2026')`

**←**

```json
{
  "exists": false,
  "repo_id": "nope/this_does_not_exist_for_sure_2026",
  "error": "RepositoryNotFoundError: 404 Client Error. (Request ID: Root=1-6a21a435-51dae4fd009a582f19adf898;ac862ed1-f278-44df-99a6-da93ec33bbcc)\n\nRepository Not Found for url: https://huggingface.co/api/datasets/nope/this_does_not_exist_for_sure_2026?blobs=true.\nPlease make sure you specified the correct `repo_id` and `repo_type`.\nIf you are trying to access a private or gated repo, make sure you are authenticated and your token has the required permissions.\nFor more details, see https://huggingface.co/docs/huggingface_hub/authentication"
}
```

## hub_list_jobs — Hub transfer registry

_Intent: List all known Hub jobs (active + recent terminals). Same source the GUI's Transfers tray polls. The `active` count is the outcome-transparent summary._

**→** `hub_list_jobs()`

**←**

```json
{
  "jobs": [],
  "total": 0,
  "active": 0
}
```

## hub_job_progress — single-job snapshot

_Intent: No live jobs in the registry today, so we probe the error path instead. An unknown `job_id` raises so the AI can distinguish 'job not yet started / already GC'd' from 'job finished with status X'._

**→** `hub_job_progress(job_id='does-not-exist')`

**← error** — `Error executing tool hub_job_progress: 'Hub job not found: does-not-exist'`

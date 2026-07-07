## Read-tier completers MCP — transcript

Captured live against the unified GUI's MCP at `127.0.0.1:8000/mcp/`. The token carries `read` + `comment` scopes; the only mutations are two sidecar tags written and then deleted by this script. The operator's dataset files are untouched.

## lerobot_list_tools — self-describe the surface

_Intent: Useful for AI clients that don't surface the tool list cleanly (raw streamable-http, scripted agents). Returns name + one-line description + scope for every registered tool._

**→** `lerobot_list_tools()`

**←**

```json
{
  "tools": [
    {
      "name": "delete_episode_tag",
      "description": "Delete one sidecar comment from an episode.",
      "scope": "comment"
    },
    {
      "name": "tag_episode",
      "description": "Write a sidecar comment on an episode. Does not modify the canonical dataset.",
      "scope": "comment"
    },
    {
      "name": "apply_pending_edits",
      "description": "Apply all staged edits for a dataset to disk in one transaction.",
      "scope": "edit"
    },
    {
      "name": "assign_port_to_arm",
      "description": "Set a port field on a saved profile to a specific device path.",
      "scope": "edit"
    },
    {
      "name": "create_robot_profile",
      "description": "Create a new robot profile (writes a JSON file to disk).",
      "scope": "edit"
    },
    {
      "name": "create_teleop_profile",
      "description": "Create a new teleop profile (writes a JSON file to disk).",
      "scope": "edit"
    },
    {
      "name": "delete_robot_profile",
      "description": "Delete a robot profile (removes the JSON file from disk).",
      "scope": "edit"
    },
    {
      "name": "delete_teleop_profile",
      "description": "Delete a teleop profile.",
      "scope": "edit"
    },
    {
      "name": "discard_pending_edits",
      "description": "Drop pending edits without applying them.",
      "scope": "edit"
    },
    {
      "name": "hub_cancel_job",
      "description": "Cancel an in-flight Hub transfer by sending SIGTERM to its worker.",
      "scope": "edit"
    },
    {
      "name": "hub_start_download",
      "description": "Start downloading a dataset from Hugging Face Hub into the local copy.",
      "scope": "edit"
    },
    {
      "name": "hub_start_upload",
      "description": "Start uploading a local dataset to Hugging Face Hub.",
      "scope": "edit"
    },
    {
      "name": "merge_into_dataset",
      "description": "Merge all episodes from ``source_repo_id`` INTO ``target_repo_id``.",
      "scope": "edit"
    },
    {
      "name": "propose_delete_episode",
      "description": "Stage the deletion of one episode from a dataset.",
      "scope": "edit"
    },
    {
      "name": "propose_set_feature",
      "description": "Stage a per-frame feature-value edit over the range ``[frame_from, frame_to)``.",
      "scope": "edit"
    },
    {
      "name": "propose_trim_episode",
      "description": "Stage a trim of one episode \u2014 keep frames ``[keep_from, keep_to)``, drop the rest.",
      "scope": "edit"
    },
    {
      "name": "rename_robot_profile",
      "description": "Rename a robot profile.",
      "scope": "edit"
    },
    {
      "name": "rename_teleop_profile",
      "description": "Rename a teleop profile.",
      "scope": "edit"
    },
    {
      "name": "update_robot_profile",
      "description": "Overwrite a robot profile (full-replace, not patch).",
      "scope": "edit"
    },
    {
      "name": "update_teleop_profile",
      "description": "Overwrite a teleop profile (full-replace).",
      "scope": "edit"
    },
    {
      "name": "get_all_port_assignments",
      "description": "Cross-reference port paths against saved profile configs.",
      "scope": "read"
    },
    {
      "name": "get_dataset_info",
      "description": "Detailed schema and stats for one dataset.",
      "scope": "read"
    },
    {
      "name": "get_episode_summary",
      "description": "Per-episode detail: length, tasks, terminal action/state, tags.",
      "scope": "read"
    },
    {
      "name": "get_episode_tags",
      "description": "Read all sidecar comments for an episode.",
      "scope": "read"
    },
    {
      "name": "get_feature_series",
      "description": "Per-frame time series of one or more features for one episode.",
      "scope": "read"
    },
    {
      "name": "get_frame",
      "description": "One image from a video stream of a single episode.",
      "scope": "read"
    },
    {
      "name": "get_latency_metrics",
      "description": "Latest latency snapshot for the requested loop source.",
      "scope": "read"
    },
    {
      "name": "get_rlt_metrics",
      "description": "RLT training metrics for the currently-active RLT run.",
      "scope": "read"
    },
    {
      "name": "get_robot_profile",
      "description": "Full saved robot profile by name.",
      "scope": "read"
    },
    {
      "name": "get_run_output",
      "description": "Tail of the active subprocess's captured stdout/stderr.",
      "scope": "read"
    },
    {
      "name": "get_run_status",
      "description": "Current state of the GUI-managed subprocess.",
      "scope": "read"
    },
    {
      "name": "get_teleop_profile",
      "description": "Full saved teleop profile by name.",
      "scope": "read"
    },
    {
      "name": "highlight_in_viewer",
      "description": "Mark a set of episodes as highlighted in the GUI's list/timeline.",
      "scope": "read"
    },
    {
      "name": "hub_auth_status",
      "description": "Whether the host process has a working HF Hub login.",
      "scope": "read"
    },
    {
      "name": "hub_diff_local_vs_remote",
      "description": "Compare a local dataset against its Hub counterpart by file size + presence.",
      "scope": "read"
    },
    {
      "name": "hub_job_progress",
      "description": "Snapshot of one Hub job \u2014 for polling a long-running transfer.",
      "scope": "read"
    },
    {
      "name": "hub_list_jobs",
      "description": "List all Hub transfers the GUI is tracking, newest-first.",
      "scope": "read"
    },
    {
      "name": "hub_repo_info",
      "description": "Look up a dataset repo on the Hub.",
      "scope": "read"
    },
    {
      "name": "lerobot_list_tools",
      "description": "List every MCP tool the server exposes, with scope + description.",
      "scope": "read"
    },
    {
      "name": "lerobot_whoami",
      "description": "Report the calling client's identity + scopes on this MCP.",
      "scope": "read"
    },
    {
      "name": "list_datasets",
      "description": "List datasets discoverable under the LeRobot dataset cache.",
      "scope": "read"
    },
    {
      "name": "list_episodes",
      "description": "List episodes with per-episode summary fields (paginated).",
      "scope": "read"
    },
    {
      "name": "list_pending_edits",
      "description": "List currently staged (unsaved) dataset edits.",
      "scope": "read"
    },
    {
      "name": "list_ports",
      "description": "Enumerate USB serial ports (kernel-level, no device opened).",
      "scope": "read"
    },
    {
      "name": "list_robot_profiles",
      "description": "List saved robot profiles.",
      "scope": "read"
    },
    {
      "name": "list_tagged_episodes",
      "description": "Reverse lookup over sidecar tags.",
      "scope": "read"
    },
    {
      "name": "list_teleop_profiles",
      "description": "List saved teleoperator profiles.",
      "scope": "read"
    },
    {
      "name": "navigate_to",
      "description": "Navigate the user's open GUI tab to a specific view.",
      "scope": "read"
    },
    {
      "name": "notify_user",
      "description": "Surface a notification in the user's GUI tab.",
      "scope": "read"
    },
    {
      "name": "validate_dataset_merge",
      "description": "Check whether two opened datasets are schema-compatible for merge.",
      "scope": "read"
    }
  ],
  "total": 50
}
```

_(Surface size: 50 tools; by scope: {'comment': 2, 'edit': 18, 'read': 30})_

## lerobot_whoami — verify caller privileges

_Intent: Pre-flight before proposing an edit-tier or operate-tier action. The agent can fail-fast when the token doesn't carry the needed scope._

**→** `lerobot_whoami()`

**←**

```json
{
  "client_id": "proofs_read_completers-221163a8",
  "scopes": ["comment", "read"],
  "all_scopes": ["read", "comment", "edit", "operate"],
  "logged_in": true
}
```

## list_tagged_episodes — reverse-lookup over tags

_Intent: First confirm there are no tags with our proof key yet (it's randomly suffixed, so almost certainly true)._

**→** `list_tagged_episodes(repo_id='thewisp/test_leader_follower_do_not_use', key='_mcp_proof_tag_24d2e7')`

**←**

```json
{
  "repo_id": "thewisp/test_leader_follower_do_not_use",
  "key": "_mcp_proof_tag_24d2e7",
  "value": null,
  "episodes": [],
  "total": 0
}
```

_Intent: Tag episode 0 with our proof key — value 'good'. The tag lives in the operator's sidecar SQLite; we'll delete it before exiting._

**→** `tag_episode(repo_id='thewisp/test_leader_follower_do_not_use', episode_id=0, key='_mcp_proof_tag_24d2e7', value='good')`

**←**

```json
{
  "repo_id": "thewisp/test_leader_follower_do_not_use",
  "episode_id": 0,
  "key": "_mcp_proof_tag_24d2e7",
  "overwrote": false,
  "previous_value": null,
  "tags": {
    "_mcp_proof_tag_24d2e7": "good",
    "mcp_test": {
      "written_by": "Claude via MCP",
      "purpose": "demonstrate UPSERT",
      "outcome": "rewritten"
    }
  }
}
```

_Intent: Now list_tagged_episodes by key — episode 0 should show up with value=good and a set_at timestamp._

**→** `list_tagged_episodes(repo_id='thewisp/test_leader_follower_do_not_use', key='_mcp_proof_tag_24d2e7')`

**←**

```json
{
  "repo_id": "thewisp/test_leader_follower_do_not_use",
  "key": "_mcp_proof_tag_24d2e7",
  "value": null,
  "episodes": [
    {
      "episode_id": 0,
      "value": "good",
      "set_at": "2026-06-08T17:25:39+00:00"
    }
  ],
  "total": 1
}
```

_Intent: Filter further by key + value. Same single result, demonstrating the value filter narrows correctly._

**→** `list_tagged_episodes(repo_id='thewisp/test_leader_follower_do_not_use', key='_mcp_proof_tag_24d2e7', value='good')`

**←**

```json
{
  "repo_id": "thewisp/test_leader_follower_do_not_use",
  "key": "_mcp_proof_tag_24d2e7",
  "value": "good",
  "episodes": [
    {
      "episode_id": 0,
      "value": "good",
      "set_at": "2026-06-08T17:25:39+00:00"
    }
  ],
  "total": 1
}
```

_Intent: Cleanup — drop the proof tag so the operator's sidecar is back to its prior state._

**→** `delete_episode_tag(repo_id='thewisp/test_leader_follower_do_not_use', episode_id=0, key='_mcp_proof_tag_24d2e7')`

**←**

```json
{
  "repo_id": "thewisp/test_leader_follower_do_not_use",
  "episode_id": 0,
  "deleted": true,
  "tags": {
    "mcp_test": {
      "written_by": "Claude via MCP",
      "purpose": "demonstrate UPSERT",
      "outcome": "rewritten"
    }
  }
}
```

## get_feature_series — per-frame trajectory

_Intent: Default mode: omit `features` to pull every per-frame non-image feature. Transcript only logs the shape (series keys + length) — the full payload is hundreds of frames × multiple features._

**→** `get_feature_series(repo_id='thewisp/test_leader_follower_do_not_use', episode_id=0)`

**←** _(summarised — full payload too long for transcript)_

```json
{
  "repo_id": "thewisp/test_leader_follower_do_not_use",
  "episode_index": 0,
  "length": 770,
  "series_keys": [
    "action",
    "episode_index",
    "frame_index",
    "index",
    "observation.state",
    "task",
    "task_index",
    "timestamp"
  ],
  "series_count": 8
}
```

## hub_diff_local_vs_remote — sync check

_Intent: Compare the local copy of the throwaway dataset against its Hub mirror. Same source the GUI's 'Compare to remote' button uses._

**→** `hub_diff_local_vs_remote(dataset_id='thewisp/test_leader_follower_do_not_use')`

**←**

```json
{
  "status": "ok",
  "in_sync": false,
  "unchanged": 9,
  "modified": [],
  "local_only": [],
  "remote_only": [".gitattributes", "README.md"]
}
```

## Error path — unknown dataset

_Intent: Same clean tool-error pattern as the read-only Hub tools._

**→** `get_feature_series(repo_id='nope/does_not_exist', episode_id=0)`

**← error** — `Error executing tool get_feature_series: Dataset not opened in GUI: nope/does_not_exist`

## Dataset Edit MCP — transcript

Captured live against the unified GUI at `127.0.0.1:8000` against the throwaway dataset `thewisp/test_leader_follower_do_not_use`. The dataset is left untouched on disk — only the in-memory PendingEdit queue is mutated, and every block ends with a `discard_pending_edits` or the queue is otherwise drained.

## Happy paths

_Intent: Stage a delete on episode 0_

**→** `propose_delete_episode(repo_id='thewisp/test_leader_follower_do_not_use', episode_id=0)`

**←**

```json
{
  "status": "ok",
  "message": "Episode 0 marked for deletion",
  "dataset_id": "thewisp/test_leader_follower_do_not_use",
  "episode_index": 0
}
```

_Intent: Look at the queue from the AI's side_

**→** `list_pending_edits(repo_id='thewisp/test_leader_follower_do_not_use')`

**←**

```json
{
  "edits": [
    {
      "index": 0,
      "edit_type": "delete",
      "dataset_id": "thewisp/test_leader_follower_do_not_use",
      "episode_index": 0,
      "params": {},
      "created_at": "2026-06-04T15:13:24.372258"
    }
  ],
  "total": 1
}
```

_Intent: Trim episode 0 to a 6-frame window — KEEP frames [2, 8)_

**→** `propose_trim_episode(repo_id='thewisp/test_leader_follower_do_not_use', episode_id=0, keep_from=2, keep_to=8)`

**←**

```json
{
  "status": "ok",
  "message": "Episode 0: keep frames [2, 8) (6 of 770 frames; dropping 764)",
  "dataset_id": "thewisp/test_leader_follower_do_not_use",
  "episode_index": 0,
  "kept_frames": 6,
  "dropped_frames": 764,
  "episode_length_before": 770,
  "keep_from": 2,
  "keep_to": 8
}
```

_Intent: Cancel everything_

**→** `discard_pending_edits(repo_id='thewisp/test_leader_follower_do_not_use')`

**←**

```json
{
  "status": "ok",
  "message": "Discarded 2 pending edits",
  "discarded": 2
}
```

## Error cases

_Intent: Out of range — only 1 episode exists, ask for ep 999_

**→** `propose_delete_episode(repo_id='thewisp/test_leader_follower_do_not_use', episode_id=999)`

**← error** — `Error executing tool propose_delete_episode: Invalid episode index 999 for dataset thewisp/test_leader_follower_do_not_use (total_episodes=1)`

_Intent: Unknown dataset — typo or unloaded_

**→** `propose_delete_episode(repo_id='nope/does_not_exist', episode_id=0)`

**← error** — `Error executing tool propose_delete_episode: 'Dataset not found: nope/does_not_exist'`

_Intent: Duplicate delete — stage once, then again_

**→** `propose_delete_episode(repo_id='thewisp/test_leader_follower_do_not_use', episode_id=0)`

**←**

```json
{
  "status": "ok",
  "message": "Episode 0 marked for deletion",
  "dataset_id": "thewisp/test_leader_follower_do_not_use",
  "episode_index": 0
}
```

**→** `propose_delete_episode(repo_id='thewisp/test_leader_follower_do_not_use', episode_id=0)`

**← error** — `Error executing tool propose_delete_episode: Episode 0 is already marked for deletion in thewisp/test_leader_follower_do_not_use`

**→** `discard_pending_edits(repo_id='thewisp/test_leader_follower_do_not_use')`

**←**

```json
{
  "status": "ok",
  "message": "Discarded 1 pending edits",
  "discarded": 1
}
```

_Intent: Invalid trim — keep window has zero size (keep_from == keep_to)_

**→** `propose_trim_episode(repo_id='thewisp/test_leader_follower_do_not_use', episode_id=0, keep_from=5, keep_to=5)`

**← error** — `Error executing tool propose_trim_episode: Invalid trim range [5, 5): the keep window must have positive length`

_Intent: If the AI confused 'trim' as 'remove this range', it would emit this call expecting frames 3-7 to be removed. The tool keeps [3, 7) (one contiguous window) and the response message states the actual outcome so the AI can self-correct._

**→** `propose_trim_episode(repo_id='thewisp/test_leader_follower_do_not_use', episode_id=0, keep_from=3, keep_to=7)`

**←**

```json
{
  "status": "ok",
  "message": "Episode 0: keep frames [3, 7) (4 of 770 frames; dropping 766)",
  "dataset_id": "thewisp/test_leader_follower_do_not_use",
  "episode_index": 0,
  "kept_frames": 4,
  "dropped_frames": 766,
  "episode_length_before": 770,
  "keep_from": 3,
  "keep_to": 7
}
```

**→** `discard_pending_edits(repo_id='thewisp/test_leader_follower_do_not_use')`

**←**

```json
{
  "status": "ok",
  "message": "Discarded 1 pending edits",
  "discarded": 1
}
```

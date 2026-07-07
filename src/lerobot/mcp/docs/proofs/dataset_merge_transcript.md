## Dataset merge MCP — transcript

Captured live against the unified GUI's MCP at `127.0.0.1:8000/mcp/`. The transcript clones a throwaway dataset (`thewisp/test_leader_follower_do_not_use`) twice to `_mcp_merge_fd0929/source` and `_mcp_merge_fd0929/target` under `$HF_LEROBOT_HOME`, opens both in the GUI, runs the validate + merge tool pair, then deletes both clones. The operator's original throwaway dataset is **never** modified.

## validate_dataset_merge — compat check

_Intent: Always run this before proposing a real merge. Same schema (cloned from the same template) → empty mismatches list._

**→** `validate_dataset_merge(source_repo_id='_mcp_merge_fd0929/source', target_repo_id='_mcp_merge_fd0929/target')`

**←**

```json
{
  "compatible": true,
  "mismatches": []
}
```

## get_dataset_info — target BEFORE merge

_Intent: Snapshot the target's episode/frame counts for before/after comparison._

**→** `get_dataset_info(repo_id='_mcp_merge_fd0929/target')`

**←**

```json
{
  "repo_id": "_mcp_merge_fd0929/target",
  "robot_type": "bi_so107_follower",
  "fps": 30,
  "total_episodes": 1,
  "total_frames": 770,
  "total_tasks": 1,
  "cameras": [
    "observation.images.front",
    "observation.images.left_wrist",
    "observation.images.right_wrist",
    "observation.images.top"
  ],
  "image_keys": [],
  "video_keys": [
    "observation.images.front",
    "observation.images.left_wrist",
    "observation.images.right_wrist",
    "observation.images.top"
  ],
  "features": {
    "action": {
      "dtype": "float32",
      "shape": [14],
      "names": [
        "left_shoulder_pan.pos",
        "left_shoulder_lift.pos",
        "left_elbow_flex.pos",
        "left_forearm_roll.pos",
        "left_wrist_flex.pos",
        "left_wrist_roll.pos",
        "left_gripper.pos",
        "right_shoulder_pan.pos",
        "right_shoulder_lift.pos",
        "right_elbow_flex.pos",
        "right_forearm_roll.pos",
        "right_wrist_flex.pos",
        "right_wrist_roll.pos",
        "right_gripper.pos"
      ]
    },
    "observation.state": {
      "dtype": "float32",
      "shape": [14],
      "names": [
        "left_shoulder_pan.pos",
        "left_shoulder_lift.pos",
        "left_elbow_flex.pos",
        "left_forearm_roll.pos",
        "left_wrist_flex.pos",
        "left_wrist_roll.pos",
        "left_gripper.pos",
        "right_shoulder_pan.pos",
        "right_shoulder_lift.pos",
        "right_elbow_flex.pos",
        "right_forearm_roll.pos",
        "right_wrist_flex.pos",
        "right_wrist_roll.pos",
        "right_gripper.pos"
      ]
    },
    "observation.images.front": {
      "dtype": "video",
      "shape": [720, 1280, 3],
      "names": ["height", "width", "channels"]
    },
    "observation.images.left_wrist": {
      "dtype": "video",
      "shape": [720, 1280, 3],
      "names": ["height", "width", "channels"]
    },
    "observation.images.right_wrist": {
      "dtype": "video",
      "shape": [720, 1280, 3],
      "names": ["height", "width", "channels"]
    },
    "observation.images.top": {
      "dtype": "video",
      "shape": [720, 1280, 3],
      "names": ["height", "width", "channels"]
    },
    "timestamp": {
      "dtype": "float32",
      "shape": [1],
      "names": null
    },
    "frame_index": {
      "dtype": "int64",
      "shape": [1],
      "names": null
    },
    "episode_index": {
      "dtype": "int64",
      "shape": [1],
      "names": null
    },
    "index": {
      "dtype": "int64",
      "shape": [1],
      "names": null
    },
    "task_index": {
      "dtype": "int64",
      "shape": [1],
      "names": null
    }
  },
  "episode_length_stats": {
    "min": 770,
    "max": 770,
    "mean": 770.0,
    "sampled": 1
  }
}
```

## merge_into_dataset — the destructive write

_Intent: Copy the source's episodes into the target on disk. Response carries before/after counts so the AI can see exactly how much the target grew without a separate follow-up read._

**→** `merge_into_dataset(source_repo_id='_mcp_merge_fd0929/source', target_repo_id='_mcp_merge_fd0929/target')`

**←**

```json
{
  "status": "ok",
  "source_id": "_mcp_merge_fd0929/source",
  "target_id": "_mcp_merge_fd0929/target",
  "source_episodes_merged": 1,
  "source_frames_merged": 770,
  "target_episodes_before": 1,
  "target_episodes_after": 2,
  "target_frames_before": 770,
  "target_frames_after": 1540,
  "force_used": false
}
```

## get_dataset_info — target AFTER merge

_Intent: Confirm the on-disk metadata reflects the merge (total_episodes / total_frames doubled because we merged a clone of the same dataset into itself)._

**→** `get_dataset_info(repo_id='_mcp_merge_fd0929/target')`

**←**

```json
{
  "repo_id": "_mcp_merge_fd0929/target",
  "robot_type": "bi_so107_follower",
  "fps": 30,
  "total_episodes": 1,
  "total_frames": 770,
  "total_tasks": 1,
  "cameras": [
    "observation.images.front",
    "observation.images.left_wrist",
    "observation.images.right_wrist",
    "observation.images.top"
  ],
  "image_keys": [],
  "video_keys": [
    "observation.images.front",
    "observation.images.left_wrist",
    "observation.images.right_wrist",
    "observation.images.top"
  ],
  "features": {
    "action": {
      "dtype": "float32",
      "shape": [14],
      "names": [
        "left_shoulder_pan.pos",
        "left_shoulder_lift.pos",
        "left_elbow_flex.pos",
        "left_forearm_roll.pos",
        "left_wrist_flex.pos",
        "left_wrist_roll.pos",
        "left_gripper.pos",
        "right_shoulder_pan.pos",
        "right_shoulder_lift.pos",
        "right_elbow_flex.pos",
        "right_forearm_roll.pos",
        "right_wrist_flex.pos",
        "right_wrist_roll.pos",
        "right_gripper.pos"
      ]
    },
    "observation.state": {
      "dtype": "float32",
      "shape": [14],
      "names": [
        "left_shoulder_pan.pos",
        "left_shoulder_lift.pos",
        "left_elbow_flex.pos",
        "left_forearm_roll.pos",
        "left_wrist_flex.pos",
        "left_wrist_roll.pos",
        "left_gripper.pos",
        "right_shoulder_pan.pos",
        "right_shoulder_lift.pos",
        "right_elbow_flex.pos",
        "right_forearm_roll.pos",
        "right_wrist_flex.pos",
        "right_wrist_roll.pos",
        "right_gripper.pos"
      ]
    },
    "observation.images.front": {
      "dtype": "video",
      "shape": [720, 1280, 3],
      "names": ["height", "width", "channels"]
    },
    "observation.images.left_wrist": {
      "dtype": "video",
      "shape": [720, 1280, 3],
      "names": ["height", "width", "channels"]
    },
    "observation.images.right_wrist": {
      "dtype": "video",
      "shape": [720, 1280, 3],
      "names": ["height", "width", "channels"]
    },
    "observation.images.top": {
      "dtype": "video",
      "shape": [720, 1280, 3],
      "names": ["height", "width", "channels"]
    },
    "timestamp": {
      "dtype": "float32",
      "shape": [1],
      "names": null
    },
    "frame_index": {
      "dtype": "int64",
      "shape": [1],
      "names": null
    },
    "episode_index": {
      "dtype": "int64",
      "shape": [1],
      "names": null
    },
    "index": {
      "dtype": "int64",
      "shape": [1],
      "names": null
    },
    "task_index": {
      "dtype": "int64",
      "shape": [1],
      "names": null
    }
  },
  "episode_length_stats": {
    "min": 770,
    "max": 770,
    "mean": 770.0,
    "sampled": 1
  }
}
```

## Error path — self-merge rejected

_Intent: Try to merge a dataset into itself — clean error so the AI doesn't accidentally double-write._

**→** `merge_into_dataset(source_repo_id='_mcp_merge_fd0929/source', target_repo_id='_mcp_merge_fd0929/source')`

**← error** — `Error executing tool merge_into_dataset: Cannot merge a dataset into itself`

## Error path — unknown source

_Intent: Typo or unopened dataset._

**→** `validate_dataset_merge(source_repo_id='nope/missing', target_repo_id='_mcp_merge_fd0929/target')`

**← error** — `Error executing tool validate_dataset_merge: 'Source dataset not found: nope/missing'`

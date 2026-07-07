## Robots edit-tier MCP — transcript

Captured live against the unified GUI's MCP at `127.0.0.1:8000/mcp/`. The transcript writes a clearly-marked throwaway profile (name prefix `_mcp_transcript_fd44a3`) to the operator's `~/.config/lerobot/robots/` and `teleops/`, walks it through the full CRUD lifecycle, then deletes it. **No motor bus is opened**, no port is connected to, no camera is started — these tools are strictly file CRUD.

## create_robot_profile — write a new profile

_Intent: Create a fresh robot profile. Response carries `created: true` + the on-disk path so the AI can confirm where the file landed._

**→** `create_robot_profile(name='_mcp_transcript_fd44a3_robot', type='bi_so107_follower', fields={'port': '/dev/ttyACM0', 'baudrate': 1000000})`

**←**

```json
{
  "status": "ok",
  "created": true,
  "name": "_mcp_transcript_fd44a3_robot",
  "type": "bi_so107_follower",
  "path": "/home/feit/.config/lerobot/robots/_mcp_transcript_fd44a3_robot.json"
}
```

_Intent: Try to create it again — the name collision is explicit, and the response hints at the right follow-up (`use update_robot_profile`)._

**→** `create_robot_profile(name='_mcp_transcript_fd44a3_robot', type='bi_so107_follower')`

**← error** — `Error executing tool create_robot_profile: Robot profile '_mcp_transcript_fd44a3_robot' already exists. Use update_robot_profile to overwrite, or pick a different name.`

## update_robot_profile — full-replace

_Intent: Replace the profile. Response carries `overwrote: true` and `previous_type` so the AI knows it clobbered an existing one (no silent overwrite)._

**→** `update_robot_profile(name='_mcp_transcript_fd44a3_robot', type='bi_so107_follower_predictive', fields={'port': '/dev/ttyACM2', 'baudrate': 2000000})`

**←**

```json
{
  "status": "ok",
  "name": "_mcp_transcript_fd44a3_robot",
  "type": "bi_so107_follower_predictive",
  "overwrote": true,
  "previous_type": "bi_so107_follower",
  "previous_fields_count": 2
}
```

## assign_port_to_arm — convenience over update

_Intent: Change one port field. Cheaper than reading the whole profile + writing it back. Response carries `previous_port` and `changed` so the AI can see whether the assignment was a no-op._

**→** `assign_port_to_arm(profile_name='_mcp_transcript_fd44a3_robot', port='/dev/ttyACM7')`

**←**

```json
{
  "status": "ok",
  "name": "_mcp_transcript_fd44a3_robot",
  "field": "port",
  "previous_port": "/dev/ttyACM2",
  "new_port": "/dev/ttyACM7",
  "changed": true
}
```

_Intent: Same call again — `changed: false` because the value matches._

**→** `assign_port_to_arm(profile_name='_mcp_transcript_fd44a3_robot', port='/dev/ttyACM7')`

**←**

```json
{
  "status": "ok",
  "name": "_mcp_transcript_fd44a3_robot",
  "field": "port",
  "previous_port": "/dev/ttyACM7",
  "new_port": "/dev/ttyACM7",
  "changed": false
}
```

## rename_robot_profile

_Intent: Rename the file AND the internal `name` field. Both the on-disk path and the JSON contents update._

**→** `rename_robot_profile(old_name='_mcp_transcript_fd44a3_robot', new_name='_mcp_transcript_fd44a3_robot_renamed')`

**←**

```json
{
  "status": "ok",
  "renamed": true,
  "from": "_mcp_transcript_fd44a3_robot",
  "to": "_mcp_transcript_fd44a3_robot_renamed"
}
```

_Intent: Confirm the new name is listed._

**→** `list_robot_profiles()`

**←**

```json
{
  "profiles": [
    {
      "name": "_mcp_transcript_fd44a3_robot_renamed",
      "type": "bi_so107_follower_predictive"
    },
    {
      "name": "virtual_so107_follower",
      "type": "virtual_bi_so107"
    },
    {
      "name": "white",
      "type": "bi_so107_follower"
    },
    {
      "name": "white_pred",
      "type": "bi_so107_follower_predictive"
    }
  ],
  "total": 4
}
```

Renamed profile present in list: `True` (4 total profiles)

## Error path — rename to an existing name

_Intent: Try to rename onto a slot already occupied — clean tool-error message tells the agent what to do next._

**→** `create_robot_profile(name='_mcp_transcript_fd44a3_collision_target', type='bi_so107_follower')`

**←**

```json
{
  "status": "ok",
  "created": true,
  "name": "_mcp_transcript_fd44a3_collision_target",
  "type": "bi_so107_follower",
  "path": "/home/feit/.config/lerobot/robots/_mcp_transcript_fd44a3_collision_target.json"
}
```

**→** `rename_robot_profile(old_name='_mcp_transcript_fd44a3_robot_renamed', new_name='_mcp_transcript_fd44a3_collision_target')`

**← error** — `Error executing tool rename_robot_profile: Robot profile '_mcp_transcript_fd44a3_collision_target' already exists — pick a different name or delete the existing one first.`

**→** `delete_robot_profile(name='_mcp_transcript_fd44a3_collision_target')`

**←**

```json
{
  "status": "ok",
  "deleted": true,
  "name": "_mcp_transcript_fd44a3_collision_target",
  "removed_type": "bi_so107_follower",
  "removed_fields_count": 0
}
```

## delete_robot_profile — gone with diagnostics

_Intent: Delete the throwaway. Response echoes back the removed `type` + `fields_count` so the audit trail shows WHAT was removed, not just THAT something was._

**→** `delete_robot_profile(name='_mcp_transcript_fd44a3_robot_renamed')`

**←**

```json
{
  "status": "ok",
  "deleted": true,
  "name": "_mcp_transcript_fd44a3_robot_renamed",
  "removed_type": "bi_so107_follower_predictive",
  "removed_fields_count": 2
}
```

## Error path — operate on missing profile

_Intent: Try to delete a profile that doesn't exist — the AI sees a clear tool-error message instead of a silent success or HTTP status code._

**→** `delete_robot_profile(name='_mcp_transcript_fd44a3_does_not_exist')`

**← error** — `Error executing tool delete_robot_profile: Robot profile not found: '_mcp_transcript_fd44a3_does_not_exist'`

## Teleop CRUD — same shape, different directory

_Intent: Teleop profiles live in `~/.config/lerobot/teleops/`._

**→** `create_teleop_profile(name='_mcp_transcript_fd44a3_teleop', type='scripted_ee')`

**←**

```json
{
  "status": "ok",
  "created": true,
  "name": "_mcp_transcript_fd44a3_teleop",
  "type": "scripted_ee",
  "path": "/home/feit/.config/lerobot/teleops/_mcp_transcript_fd44a3_teleop.json"
}
```

**→** `delete_teleop_profile(name='_mcp_transcript_fd44a3_teleop')`

**←**

```json
{
  "status": "ok",
  "deleted": true,
  "name": "_mcp_transcript_fd44a3_teleop",
  "removed_type": "scripted_ee",
  "removed_fields_count": 0
}
```

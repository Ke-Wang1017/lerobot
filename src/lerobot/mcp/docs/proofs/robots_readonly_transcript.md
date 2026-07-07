## Robots read-only MCP — transcript

Captured live against the unified GUI's MCP at `127.0.0.1:8000/mcp/`. All calls are pure reads — no motor bus is opened, no port is connected to, no camera is started. The transcript reflects the actual operator's saved profiles and port topology.

## list_robot_profiles — saved robot configs

_Intent: First call any AI agent makes when asked 'what robots does this host know about'. Returns the lightweight name+type pair; full config via get_robot_profile._

**→** `list_robot_profiles()`

**←**

```json
{
  "profiles": [
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
  "total": 3
}
```

_Intent: Inspect the first saved robot profile (`virtual_so107_follower`) to see the full on-disk config — fields, cameras, rest_position. The AI uses this before proposing a teleop or recording session._

**→** `get_robot_profile(name='virtual_so107_follower')`

**←**

```json
{
  "type": "virtual_bi_so107",
  "name": "virtual_so107_follower",
  "fields": {
    "id": null
  },
  "cameras": {},
  "rest_position": {}
}
```

_Intent: Probe a deliberately-missing profile — the AI should see a clean tool-error message it can surface to the operator without parsing exception types._

**→** `get_robot_profile(name='does-not-exist-on-this-host')`

**← error** — `Error executing tool get_robot_profile: Robot profile not found: 'does-not-exist-on-this-host'`

## list_teleop_profiles — saved teleop configs

_Intent: Saved teleop devices (leader arms, Quest 3 VR, scripted EE) for this host. Same name+type shape as robot profiles._

**→** `list_teleop_profiles()`

**←**

```json
{
  "profiles": [
    {
      "name": "blue",
      "type": "bi_so107_leader"
    },
    {
      "name": "blue_highrate",
      "type": "bi_so107_leader_highrate"
    },
    {
      "name": "click",
      "type": "click_target_bimanual_ee"
    },
    {
      "name": "vr",
      "type": "quest_vr"
    }
  ],
  "total": 4
}
```

_Intent: Inspect the first saved teleop profile (`blue`)._

**→** `get_teleop_profile(name='blue')`

**←**

```json
{
  "type": "bi_so107_leader",
  "name": "blue",
  "fields": {
    "id": "blue",
    "left_arm_port": "/dev/ttyACM3",
    "right_arm_port": "/dev/ttyACM1",
    "gripper_bounce": false
  },
  "cameras": {},
  "rest_position": {}
}
```

## list_ports — kernel-level USB serial enumeration

_Intent: Enumerate USB serial adapters via pyserial — kernel device-tree query, no port opened. Used to spot which /dev/ttyACM* / /dev/ttyUSB* are physically connected before assigning them to a profile._

**→** `list_ports()`

**←**

```json
{
  "ports": [
    {
      "path": "/dev/ttyACM0",
      "name": "USB Single Serial",
      "manufacturer": "",
      "vid_pid": "1a86:55d3"
    },
    {
      "path": "/dev/ttyACM1",
      "name": "USB Single Serial",
      "manufacturer": "",
      "vid_pid": "1a86:55d3"
    },
    {
      "path": "/dev/ttyACM2",
      "name": "USB Single Serial",
      "manufacturer": "",
      "vid_pid": "1a86:55d3"
    },
    {
      "path": "/dev/ttyACM3",
      "name": "USB Single Serial",
      "manufacturer": "",
      "vid_pid": "1a86:55d3"
    }
  ],
  "total": 4
}
```

## get_all_port_assignments — saved-profile port map

_Intent: Cross-reference port paths against saved profile configs. Useful for spotting collisions ('two profiles claim /dev/ttyACM0 as their motor bus') and reverse-lookup ('which profile owns /dev/ttyACM2?')._

**→** `get_all_port_assignments()`

**←**

```json
{
  "assignments": [
    {
      "port": "/dev/ttyACM0",
      "profile_name": "white_pred",
      "profile_kind": "robot",
      "field_name": "left_arm_port"
    },
    {
      "port": "/dev/ttyACM2",
      "profile_name": "white_pred",
      "profile_kind": "robot",
      "field_name": "right_arm_port"
    },
    {
      "port": "/dev/ttyACM0",
      "profile_name": "white",
      "profile_kind": "robot",
      "field_name": "left_arm_port"
    },
    {
      "port": "/dev/ttyACM2",
      "profile_name": "white",
      "profile_kind": "robot",
      "field_name": "right_arm_port"
    },
    {
      "port": "/dev/ttyACM3",
      "profile_name": "blue_highrate",
      "profile_kind": "teleop",
      "field_name": "left_arm_port"
    },
    {
      "port": "/dev/ttyACM1",
      "profile_name": "blue_highrate",
      "profile_kind": "teleop",
      "field_name": "right_arm_port"
    },
    {
      "port": "/dev/ttyACM3",
      "profile_name": "blue",
      "profile_kind": "teleop",
      "field_name": "left_arm_port"
    },
    {
      "port": "/dev/ttyACM1",
      "profile_name": "blue",
      "profile_kind": "teleop",
      "field_name": "right_arm_port"
    }
  ],
  "total": 8
}
```

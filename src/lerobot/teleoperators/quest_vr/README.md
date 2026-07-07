# Quest VR teleoperator

WebXR-based 6-DOF Cartesian teleop. The Quest 3's browser streams controller
poses over an HTTPS WebSocket; the teleop converts each frame to an
end-effector delta action. A Cartesian-IK-capable robot turns those deltas
into joint commands itself — see "How the IK is wired" below.

Conforms to LeRobot's `Teleoperator` interface. One config type
`quest_vr` — both Quest controllers drive both arms in a single
immersive session. Action keys are prefixed `left_` / `right_`.
Quest is always bimanual; there is no unimanual variant (a single
WebXR session already streams both controllers, so a one-arm
version would be the bimanual one with one half ignored).

## How it works

```
[Quest 3 browser]                 [PC]
─────────────────                 ─────
WebXR session     ──HTTPS+WSS──>   QuestServer (daemon thread)
controller poses                        │
@ ~90 Hz                                 ▼ on_frame callback
                                   QuestVRTeleop._on_frame
                                         │
                                         ▼ cache update (under lock)
                                   QuestVRTeleop._cached_action
                                         ▲
                                         │ lock-free read
                                   get_action()
                                         │
                                         ▼ (optional) installed transform
                                   CartesianIKController  ──> <motor>.pos
```

Sample rate is decoupled from poll rate: the Quest streams at ~90 Hz; the
run loop polls `get_action()` at whatever rate it wants and always gets the
most recent cached frame.

## How the IK is wired

`get_action()` natively returns **end-effector deltas**, not joint commands.
A Cartesian-IK robot turns them into joints by installing a transform into
the teleop from its `attach_teleop`:

```
robot.attach_teleop(teleop)        # robot builds a CartesianIKController
  └─ teleop.set_action_transform(ik)   # and installs it into the teleop
```

After that, `teleop.get_action()` returns motor-space `<motor>.pos` joints.
The IK is owned by the robot (it has the URDF, the kinematics, the
motor↔URDF alignment); the teleop just applies whatever transform was
installed. Because the teleop ends up emitting joints, the upstream
`lerobot-teleoperate` / `lerobot-record` / `lerobot-replay` loops need no
changes — they call `get_action()` / `send_action()` as usual.

For the bimanual SO-107 the wiring lives in
`BiSO107Follower.attach_teleop`; the IK itself is
`lerobot.robots.so107_description.cartesian_ik.CartesianIKController`.

## Setup

1. **Install the extra** (pulls `aiohttp` for the server and `pin-pink` for the IK):
   ```bash
   pip install 'lerobot[quest-vr]'
   ```
2. **Open the server port in the PC's firewall**: `sudo ufw allow 8443/tcp`.
3. **Quest 3 on the same WiFi as the PC.**
4. **Disable the proximity sensor on the Quest** if you want to wear the headset
   on your chest/neck during collocated teleop (otherwise the immersive session
   pauses when the headset comes off your face). Tape over the sensor (inside,
   top-center between the lenses) is the most reliable approach.

## Action format

Native `get_action()` output (per arm, prefixed `left_` / `right_`):

| key                               | type        | meaning                                                 |
| --------------------------------- | ----------- | ------------------------------------------------------- |
| `enabled`                         | float (0/1) | clutch state (grip pressed)                             |
| `target_x, target_y, target_z`    | float       | position delta from engage snapshot, in robot frame (m) |
| `target_wx, target_wy, target_wz` | float       | rotation delta from engage snapshot, as a rotvec        |
| `gripper_pos`                     | float       | absolute motor-space gripper target (from the trigger)  |

`CartesianIKController` consumes the deprefixed slice of these keys and emits
`<motor>.pos`.

## Usage

The teleop is registered as `quest_vr` and works through the standard tooling,
e.g. driving the bimanual SO-107:

```bash
lerobot-teleoperate --robot.type=bi_so107_follower ... --teleop.type=quest_vr
```

It logs a URL like `https://192.168.x.x:8443/`. Open it in the Quest browser,
accept the self-signed cert warning, tap **Connect** + **Enter VR**, then
squeeze the grip to engage tracking.

## What's tested

- `tests/robots/test_cartesian_ik.py` — controller state machine, motor↔URDF
  joint map, bimanual split/merge, the teleop→IK key contract,
  `attach_teleop` branching, and a real-IK round-trip (skipped without
  `pin-pink`).
- `tests/model/test_pink_kinematics.py` / `test_pink_ik_trajectory.py` — the
  underlying Pink IK.

End-to-end physical testing needs a Quest 3 + SO-107 hardware.

# URDF State Visualization

An interactive 3D view of the robot — orbit and zoom with the mouse —
shown beside the camera feeds during teleop, recording, replay, and policy
runs. It mirrors the robot's **observed** joint pose; it is a viewer, not a
controller, so the operator can read the arm's state at a glance even when
the physical robot is out of view or the camera angle is unhelpful.

![The visualizer tile (bottom-centre) beside the camera feeds during a run.](images/urdf_viz.png)

It is **on by default**: whenever the running robot has a vendored URDF, the
viewer appears automatically as one tile in the observation grid.

## Data flow

```
observation stream (shared mem)
        │  {motor}.pos values
        ▼
resolve_robot(obs_keys)            urdf_viz.py — matches the motor set to a
        │                          vendored *_description package
        ▼
compute_joint_angles(spec, obs)    urdf_viz.py — motor pos -> URDF rad
        │  {prefix: {urdf_joint: radians}}
        ▼
GET /api/run/urdf-viz/meta         run.py — one-shot identity + sources
GET /api/run/urdf-viz?source=X     run.py — per-source frames
        │
        ▼
static/urdf_viz.html               three.js + urdf-loader, in an iframe tile
```

All kinematics lives in **Python** (`urdf_viz.py`), not JavaScript. The
motor->URDF conversion is the correctness-critical part, so it sits under
pytest (`tests/gui/test_urdf_viz.py`) and carries asserts. The browser page
only fetches angles and draws them — it does no math.

## Sources — uniform shape, backend decides

The viewer is parameterised by a **source name**. Today: `"state"` (always)
and `"action"` (when the run has a commanded action stream, or the dataset
has an `action` feature). The names are user-facing concepts. The backend
decides how to fulfil each one — a teleop's "action" is a single commanded
pose; a future chunk-output policy's "action" would be N predicted poses
with a playback fps. The frontend doesn't know which.

The frontend speaks two endpoints:

- `GET .../urdf-viz/meta` → `{name, urdf, bimanual, sources: ["state", "action", ...]}`.
  Fetched once at iframe mount. Determines which sources to expose in the UI.
- `GET .../urdf-viz?source=X` → `{available, arms: [{prefix, frames: [{joints}, ...], fps?}]}`.
  Polled per tick (live mode) or per scrubber move (dataset mode).

The response shape is uniform across sources and modes:

| `frames.length` | Meaning                                   | Renderer behaviour today                            |
| --------------- | ----------------------------------------- | --------------------------------------------------- |
| 1               | A single pose (state, or teleop's action) | Set the URDF's joint values to `frames[0].joints`   |
| > 1             | A chunk of N future poses (`fps` present) | Apply `frames[0]` only; chunk playback is scope-out |

The list-of-frames format is in the API now even though the renderer only
reads `frames[0]`, so future sources (policy chunks, debug-model chunks)
slot in without an API rename.

### Adding a new source

A source is "a producer registered in the backend that can supply joint
values from the same `compute_joint_angles` pipeline". The two existing
ones — state and action — read from `ObservationStreamReader.read_obs()` /
`read_action()` for live, and from the parquet's `observation.state` /
`action` columns for datasets. A future debug-model source would register
a handler that reads from wherever the model's output is buffered, and
return the same `frames[]` shape.

The frontend UI today is a single "show action" toggle. When a second
candidate overlay source appears in meta (e.g. `"prediction"`), the toggle
will need to become a multi-pick — but the API contract doesn't change.

## Trajectory tube (EE path of multi-frame sources)

When a source returns `frames` of length > 1 (a chunk), the renderer
swaps from "ghost body" mode to "trajectory" mode: the ghost body is
hidden and a thin cyan tube (~3 mm radius) is drawn through the EE
link's world position at each frame. Single-frame "action" (today: a
teleop's per-tick commanded pose) keeps the ghost body so the pose
comparison is still available there.

How the tube is computed (entirely in JS, no extra backend work):

1. The frontend already keeps a hidden _ghost_ URDF tree per arm (loaded
   lazily on first "show action" toggle).
2. For each frame in `frames`, the renderer briefly sets the ghost's
   joint values, calls `updateMatrixWorld`, reads the EE link's world
   position, and collects it.
3. The list of positions becomes a `CatmullRomCurve3`; `TubeGeometry`
   wraps the curve into a thin tube mesh.
4. The ghost is restored to `frames[0]` (then made invisible) so the FK
   tree is in a sane state for the next tick.

The visual choice — hide the ghost body when there's a trajectory — is
deliberate. A same-coloured body overlay and a same-coloured tube blur
together; you can't tell whether you're looking at "the body at the next
step" or "the path through space". The body is a pose comparison; the
tube is a path-through-space comparison. They're answering different
questions, so the viewer shows one or the other based on what the data
actually contains, not both.

The **EE link name** comes from the description package's `VIZ_SPEC.ee_link`
field. It should be the tool / gripper tip (SO-107: `L7_1`; SO-101:
`gripper_frame_link`) — the trace then reads as "where the gripper will
be", which is what a user wants from a future-EE-path visualisation.
For SO-107 specifically the tip is downstream of the gripper joint so
it picks up some gripper-open/close jiggle; we accept that for the
intuitive reading. If the description omits `ee_link`, the trajectory
tube is skipped silently (the ghost body still renders as a single-
frame fallback).

WebGL line widths are ignored on most platforms (always 1px) which made
a naive polyline invisible at typical view scale. The tube has actual
3D thickness, so it stays visible at any zoom and any motion magnitude.

## Future sources (out of scope here, intentionally designed-for)

1. **Chunk playback with a paused producer.** If a policy outputs an
   action chunk of N future poses, naively previewing the chunk as a
   _moving_ ghost robot while the _actual_ robot is also moving creates a
   visual mess (two robots both moving, neither overlapping the other in
   time). The right UX is pause-then-step: the policy's producer needs a
   pause/snapshot mode the GUI can drive; the viewer scrubs through the
   captured chunk's frames at the producer's `fps`. Affects both ends — a
   pause hook on the policy side + a scrubber in the viewer. The
   trajectory tube already shows the chunk's path in space; playback adds
   the time dimension.
2. **Chunk-publishing for live policy modes.** Today the trajectory tube
   appears whenever a source returns N frames. In dataset/replay mode we
   already know the future (next N rows of `action`) so the tube lights
   up immediately. In live mode, a teleop's "action" is one frame and an
   ACT/diffusion/HVLA policy currently writes one frame per step too.
   Lighting up the tube live needs the policy to publish the _chunk_, not
   just per-step actions — a small additional shared-memory channel.
   Once it's there, the renderer reads it without changes.

## The two-layer motor->URDF mapping

A robot's motor readings rarely line up with its URDF's joint frame. Two
layers bridge the gap:

- **Layer 1 — the URDF itself.** A well-authored URDF (the vendored SO-101)
  is built to match the motor calibration: joint zero = motor zero, axes
  agree. Nothing else is needed.
- **Layer 2 — a per-joint `(sign, offset_deg)` delta.** A URDF exported
  straight from CAD (the SO-107, with generic `S1..S7` joint names) has a
  joint-zero that does not match the motors and some inverted axes. The
  delta realigns it:

  ```
  urdf_rad = deg2rad(sign * motor_pos + offset_deg)
  ```

Layer-2 alignment is **calibration data, not a model constant** — a
different physical build measures slightly different values, and the two
arms of a bimanual robot differ (some axes are mirror-mounted). It ships in
the description package as the _default_; an editable per-robot calibration
layer is future work.

## The `VIZ_SPEC` contract

`urdf_viz.py` names no robot. Each robot ships a `lerobot.robots.*_description`
package that declares a module-level `VIZ_SPEC` dict, and `resolve_robot`
discovers them at runtime by walking `lerobot.robots`:

```python
VIZ_SPEC = {
    "name":        str,                # human-facing display name
    "motors":      tuple[str, ...],    # motor names this robot exposes
    "urdf_joints": tuple[str, ...],    # URDF joint name per motor (parallel)
    "urdf_file":   str,                # URDF filename within urdf/
    "alignment":   {prefix: {motor: (sign, offset_deg)}} | None,
    "ee_link":     str | None,         # URDF link name for trajectory tube
}
```

`ee_link` should name a link _upstream of the gripper joint_ so the
trajectory tube traces arm motion rather than gripper jiggle. `None`
(or omitted) disables the trajectory tube for this robot.

`alignment` is layer 2; `None` means identity. `prefix` is `"left_"` /
`"right_"` for a bimanual robot; a unimanual robot reuses the `"right_"`
entry.

`resolve_robot` matches a live robot by its **motor set**: the description
whose `motors` are all present wins, and the **most specific** (largest
motor set) wins a tie. That is why a 7-motor SO-107 resolves to SO-107 even
though its motors are a superset of the 6-motor SO-101.

## Adding a new robot

The GUI hardcodes no robot. A URDF is, of course, robot-specific — what is
generic is the GUI code that consumes it. Everything robot-specific (the
URDF, its meshes, any calibration) lives in a self-contained package under
`src/lerobot/robots/`, discovered at runtime:

```
src/lerobot/robots/so107_description/
├── __init__.py          # get_urdf_path(), get_meshes_dir(), VIZ_SPEC
├── joint_alignment.py   # optional — layer-2 (sign, offset_deg) deltas
├── urdf/
│   └── SO107.urdf
├── meshes/
│   └── *.stl            # link meshes the URDF references
└── PROVENANCE.md        # where the URDF + meshes came from
```

To onboard a robot:

1. Create `<name>_description/` with the `urdf/` and `meshes/` files and a
   `PROVENANCE.md` recording the upstream source. Meshes are git-LFS
   tracked — a `.gitignore` negation keeps these packages tracked despite
   the global `*.urdf` / `*.stl` ignore.
2. Add `__init__.py` exposing `get_urdf_path()`, `get_meshes_dir()`, and a
   module-level `VIZ_SPEC` (the contract above).
3. Choose the calibration layer. A URDF authored to match the motor
   calibration uses `"alignment": None`. A CAD-exported URDF needs the
   per-joint `(sign, offset_deg)` deltas measured against the real robot —
   conventionally in a `joint_alignment.py` that the `VIZ_SPEC` imports.
4. That is all. `server.py` mounts every `*_description` under
   `/urdf-assets/`, `resolve_robot` discovers the package by motor set, and
   `tests/gui/test_urdf_viz.py` parametrizes over it automatically —
   asserting the URDF parses, its `VIZ_SPEC` joints exist, and its meshes
   are on disk.

## Vendored browser libraries

`static/vendor/` holds three.js (r169) and urdf-loader, vendored so the
viewer runs offline with no CDN dependency. See `static/vendor/PROVENANCE.md`
for sources and the one local modification. `urdf-loader` clamps
`setJointValue` to the URDF `<limit>` by default; the viewer sets
`ignoreLimits = true` because layer-2 offsets legitimately exceed the CAD
limits, and the goal is to show the _true_ observed pose.

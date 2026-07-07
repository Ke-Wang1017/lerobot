# GUI Architecture

## Tabs

| Tab       | Purpose                                                                                                            |
| --------- | ------------------------------------------------------------------------------------------------------------------ |
| **Data**  | Browse, play, edit, merge datasets. See [data_tab.md](data_tab.md).                                                |
| **Model** | Browse checkpoints, launch training, inspect configs. See [model_tab.md](model_tab.md).                            |
| **Run**   | Teleoperate, run policies, replay episodes. Mode depends on selection (robot+teleop, robot+policy, robot+dataset). |
| **Robot** | Hardware setup: robot/teleop profiles, camera detection, port identification, calibration.                         |

## Config Storage

```
~/.config/lerobot/
  settings.json          # app-level prefs (theme, default dirs, HF token ref)
  robots/{name}.json     # one file per robot profile
  teleops/{name}.json    # one file per teleop profile
  training/defaults.json # default training config
```

- Robots and teleops are separate — any teleop can pair with any robot (chosen in Run tab)
- Secrets stored as references to env vars or keyring, not plaintext
- Calibration data stays at `~/.cache/huggingface/lerobot/calibration/`

## Backend

Single FastAPI app with routers:

```
src/lerobot/gui/
  server.py        # FastAPI app, mount all routers
  state.py         # AppState (robot/model/run/dataset state)
  frame_cache.py   # LRU frame cache for video playback
  api/
    datasets.py    # dataset CRUD, episode listing, frame serving
    edits.py       # trim, delete, merge (pending edits model)
    playback.py    # WebSocket video playback
    models.py      # model listing, training launch, HF sync
    robot.py       # robot/teleop profiles, port/camera detection
    run.py         # teleop, policy eval, replay sessions
```

- `run.py` manages long-lived teleop/inference loops as WebSocket streams
- Training jobs are background subprocesses; API tracks PID, logs, status

## Frontend

- Per-tab vanilla JS modules, top-level tab bar with lazy loading
- Shared patterns: multi-camera grid, entity selectors (dataset, model, robot)

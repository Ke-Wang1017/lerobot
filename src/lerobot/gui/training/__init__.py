# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""Training pipeline submodule of the GUI.

Owns the GUI's training subsystem end-to-end:

- :mod:`.runs`         — Run dataclass, RunPaths, RunRegistry, state machine
- :mod:`.recipes`      — recipe builder (composes ``docker run lerobot-train``
                         and HVLA argv from a Run's args dict)
- :mod:`.transport`    — Subprocess + SSH transports for spawning workers
- :mod:`.hosts`        — TrainingHost dataclass + HostRegistry +
                         GPU auto-detection for the workstation host
- :mod:`.orchestrator` — start / poll / stop entry points; image pre-pull;
                         state machine reconciliation
- :mod:`.runner`       — fake-training runner module (used by unit tests +
                         legacy docker-less smoke path)

Scaffolding modules from earlier exploration of the SSH/Nebius path (kept
for future C7 work; not in the active runtime today):

- :mod:`.jobs`         — older job/host-profile scaffolding
- :mod:`.worker`       — older SSH-polling worker
- :mod:`.providers`    — HostProvider protocol + Nebius / Persistent stubs

Nothing is re-exported here on purpose; callers should ``from
lerobot.gui.training.X import Y`` for the concrete module they use.
"""

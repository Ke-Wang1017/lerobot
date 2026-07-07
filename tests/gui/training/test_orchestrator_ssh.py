# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""Contract test: full orchestrator end-to-end over the SSH transport.

The architectural claim of the prototype is: ``TransportClient`` is the
right abstraction — implementing :class:`SshClient` should let the
orchestrator drive a training run over SSH with **zero orchestrator code
changes**. This test is the proof.

If this test passes, every other orchestrator test in
``test_orchestrator.py`` should also pass over SSH given the same setup
(it's the same orchestrator code). Re-running all 30+ subprocess tests
under SSH would be redundant and slow; we instead pick the single test
that exercises every code path the orchestrator uses to talk to the
transport:

  - launch (image_pull is no-op for the fake recipe)
  - is_alive (polling loop)
  - exit_code (terminal event from exit)
  - append_text (orchestrator event emission)
  - read_text (progress.json + events.jsonl)
  - read_bytes_from_offset (stderr tail incremental read)
  - read_tail (stderr_tail for the snapshot)
  - list_dir + sha256_of (checkpoint manifest sync)
  - stop is exercised in test_ssh_transport.py at unit level

This is "shaped like ``test_end_to_end_natural_completion`` from
``test_orchestrator.py`` but with an SSH host."
"""

from __future__ import annotations

import os
import time
from pathlib import Path

import pytest

from lerobot.gui.training.hosts import HostRegistry, TrainingHost
from lerobot.gui.training.orchestrator import Orchestrator, StartRequest
from lerobot.gui.training.runs import RunRegistry, RunState
from lerobot.gui.training.transport import SshTransport

pytestmark = pytest.mark.requires_ssh_loopback


def _wait_until_state(orch: Orchestrator, run_id: str, want: RunState, *, timeout: float = 60.0):
    """Poll until the run reaches ``want`` (or any terminal state). The
    timeout is generous because SSH ops are ~50-100 ms each over
    loopback (vs ~ms for subprocess); a 10-step fake run that polls ~30
    times can take ~5 s end-to-end."""
    deadline = time.monotonic() + timeout
    last = None
    while time.monotonic() < deadline:
        snap = orch.poll(run_id)
        last = snap
        if snap.run.state == want or snap.run.state.value in {"completed", "failed", "aborted"}:
            return snap
        time.sleep(0.1)
    raise AssertionError(
        f"timed out waiting for state {want.value}; last={last.run.state.value if last else None}"
    )


def test_end_to_end_natural_completion_over_ssh(ssh_loopback: dict, tmp_path: Path) -> None:
    """The contract test. A fake-recipe run started on an SSH-transport
    host completes naturally, with progress.json, checkpoints.jsonl, and
    stderr_tail all surfaced correctly via the SshClient. Mirrors
    ``test_orchestrator.py::test_end_to_end_natural_completion``."""
    # Per-test remote dirs under /tmp so loopback can write freely without
    # asking for anything in $HOME or under the GUI server's runs dir.
    remote_workdir = Path(f"/tmp/lerobot-pytest-orch-ssh-{os.getpid()}-{tmp_path.name}")
    runs_root = Path(f"/tmp/lerobot-pytest-orch-ssh-runs-{os.getpid()}-{tmp_path.name}")

    transport = SshTransport(
        host=ssh_loopback["host"],
        port=ssh_loopback["port"],
        user=ssh_loopback["user"],
    )
    host = TrainingHost(
        id="ssh-loopback-host",
        display_name="SSH Loopback",
        transport=transport,
    )
    hr = HostRegistry(hosts=[host])
    rr = RunRegistry(runs_dir=runs_root)
    orch = Orchestrator(host_registry=hr, run_registry=rr)

    try:
        req = StartRequest(
            host_id="ssh-loopback-host",
            recipe_name="fake",
            dataset_id="fake/ds",
            args={
                "__recipe__": "__fake__",
                "num_steps": 10,
                "save_every": 5,
                "step_seconds": 0.05,
            },
        )
        run = orch.start(req)
        # Synchronous return: PENDING (the prep thread will advance us).
        assert run.state == RunState.PENDING

        # The fake recipe takes ~0.5 s of step work + checkpoint writes;
        # SSH polls add overhead. 60 s is comfortable headroom.
        snap = _wait_until_state(orch, run.run_id, RunState.COMPLETED)

        # State machine landed correctly
        assert snap.run.state == RunState.COMPLETED
        assert snap.run.started_at is not None
        assert snap.run.finished_at is not None
        # session_id is the SSH-encoded form (tmux-name|workdir)
        assert snap.run.session_id is not None
        assert "|" in snap.run.session_id

        # Progress was written (read via SshClient.read_text)
        assert snap.progress is not None
        assert snap.progress["step"] == 10
        assert snap.progress["loss"] > 0

        # Checkpoints surfaced via manifest (list_dir + sha256_of)
        assert len(snap.checkpoints) == 2  # 10 steps / save_every=5 = 2
        assert snap.checkpoints[0].step == 5
        assert snap.checkpoints[1].step == 10
        assert all(c.sha256 for c in snap.checkpoints)

        # stderr_tail surfaced (read_bytes_from_offset / read_tail)
        assert "[runner]" in snap.stderr_tail
    finally:
        # Best-effort cleanup of remote dirs (loopback = local FS).
        import shutil

        shutil.rmtree(remote_workdir, ignore_errors=True)
        shutil.rmtree(runs_root, ignore_errors=True)

# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""Fake training worker — a TEST FIXTURE, not shipped production code.

Lets the orchestrator's unit tests exercise the full start/poll/stop/checkpoint
path on CPU (no docker, GPU, or dataset). It sleeps, writes a decaying-loss
curve, and periodically writes placeholder checkpoints — i.e. the same
structured files a real worker writes (progress.json, events.jsonl,
checkpoints.jsonl, stderr). Invoked by the ``__fake__`` recipe via
``recipes.FAKE_RUNNER_PATH``; see tests/gui/conftest.py.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import os
import signal
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path


@dataclass
class RunnerConfig:
    run_dir: Path
    num_steps: int
    save_every: int
    step_seconds: float


def _parse_args(argv: list[str] | None = None) -> RunnerConfig:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--run-dir", required=True, type=Path)
    p.add_argument("--num-steps", type=int, default=200)
    p.add_argument("--save-every", type=int, default=50)
    p.add_argument("--step-seconds", type=float, default=0.1)
    args = p.parse_args(argv)
    return RunnerConfig(
        run_dir=args.run_dir,
        num_steps=args.num_steps,
        save_every=args.save_every,
        step_seconds=args.step_seconds,
    )


# ── File writers (atomic where appropriate) ───────────────────────────────────


def _atomic_write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), prefix=path.name + ".", suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(data, f)
        os.replace(tmp, path)
    except Exception:
        with contextlib.suppress(FileNotFoundError):
            os.unlink(tmp)  # safe-destruct: our own mkstemp tmp file, failed-write cleanup
        raise


def _append_event(events_path: Path, type_: str, **fields) -> None:
    events_path.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps({"type": type_, "ts": time.time(), **fields}) + "\n"
    with events_path.open("a") as f:
        f.write(line)


def _write_fake_checkpoint(run_dir: Path, step: int) -> tuple[Path, str]:
    """Write a small placeholder checkpoint file, return its path + sha256."""
    ckpt_dir = run_dir / "checkpoints" / f"{step:08d}"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt_file = ckpt_dir / "model.safetensors"
    payload = f"fake-checkpoint step={step} ts={time.time():.3f}".encode()
    ckpt_file.write_bytes(payload)
    digest = hashlib.sha256(payload).hexdigest()
    return ckpt_file, digest


def _append_manifest(manifest: Path, *, step: int, path: str, sha256: str) -> None:
    """Append a checkpoint manifest line (DESIGN.md § Checkpoints)."""
    manifest.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps({"step": step, "path": path, "sha256": sha256, "ts": time.time()}) + "\n"
    with manifest.open("a") as f:
        f.write(line)


# ── Main loop ──────────────────────────────────────────────────────────────────


def _install_signal_handlers(state: dict) -> None:
    """SIGTERM → mark for aborted-by-user exit; finish the current step and stop."""

    def _on_sigterm(signum, frame):  # noqa: ARG001
        state["abort_requested"] = True

    signal.signal(signal.SIGTERM, _on_sigterm)


def main(argv: list[str] | None = None) -> int:
    cfg = _parse_args(argv)
    state = {"abort_requested": False}
    _install_signal_handlers(state)

    progress_path = cfg.run_dir / "progress.json"
    events_path = cfg.run_dir / "events.jsonl"
    manifest_path = cfg.run_dir / "checkpoints.jsonl"

    _append_event(events_path, "started", num_steps=cfg.num_steps)
    print(f"[runner] starting fake training: num_steps={cfg.num_steps}", flush=True)

    final_step = 0
    final_event = "completed_naturally"
    exit_code = 0
    try:
        for step in range(1, cfg.num_steps + 1):
            if state["abort_requested"]:
                final_event = "aborted_by_user"
                break
            # Fake the training step
            time.sleep(cfg.step_seconds)
            # Vaguely plausible loss curve: starts at 2.5, decays toward 0.1
            loss = 0.1 + 2.4 * (0.99**step)
            _atomic_write_json(
                progress_path,
                {
                    "step": step,
                    "num_steps": cfg.num_steps,
                    "loss": round(loss, 4),
                    "ts": time.time(),
                },
            )
            if step % cfg.save_every == 0:
                ckpt_file, digest = _write_fake_checkpoint(cfg.run_dir, step)
                rel = ckpt_file.relative_to(cfg.run_dir).as_posix()
                _append_manifest(manifest_path, step=step, path=rel, sha256=digest)
                print(f"[runner] checkpoint at step {step} → {rel}", flush=True)
            final_step = step
    except Exception as exc:  # noqa: BLE001
        final_event = "crashed"
        exit_code = 1
        _append_event(events_path, "crashed", error=str(exc), final_step=final_step)
        print(f"[runner] crashed: {exc}", file=sys.stderr, flush=True)
    else:
        _append_event(events_path, final_event, final_step=final_step, exit_code=exit_code)

    print(f"[runner] done: event={final_event} final_step={final_step}", flush=True)
    return exit_code


if __name__ == "__main__":
    sys.exit(main())

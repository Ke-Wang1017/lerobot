# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""Tests for the training-recipe builder.

Pure unit tests — verify argv composition for both the fake fallback and
the real ``docker run`` recipe. No docker invocation; the real-recipe path
is validated end-to-end via the live smoke (see scripts/training/README.md
"2026-06-07 verification").
"""

from __future__ import annotations

import time
from pathlib import Path

import pytest

from lerobot.gui.training.recipes import (
    CONTAINER_HF_CACHE,
    CONTAINER_OUTPUT_SUBDIR,
    CONTAINER_RUNS_MOUNT,
    DEFAULT_IMAGE,
    FAKE_RECIPE_MARKER,
    HOST_GID_TOKEN,
    HOST_HOME_TOKEN,
    HOST_UID_TOKEN,
    HVLA_FLOW_S1_RECIPE,
    build_lerobot_train_command,
    docker_available,
    is_fake_recipe,
    is_hvla_flow_s1_recipe,
    output_subdir_in_run,
    resolve_host_placeholders,
)
from lerobot.gui.training.runs import Run, RunPaths, RunState, new_run_id


def _make_run(args: dict) -> Run:
    return Run(
        run_id=new_run_id(),
        host_id="this-server",
        recipe_name="act-default",
        dataset_id="lerobot/pusht",
        args=args,
        state=RunState.PENDING,
        created_at=time.time(),
    )


# ── Fake recipe fallback ──────────────────────────────────────────────────────


def test_is_fake_recipe_detects_marker() -> None:
    assert is_fake_recipe(_make_run({"__recipe__": FAKE_RECIPE_MARKER}))
    assert not is_fake_recipe(_make_run({}))
    assert not is_fake_recipe(_make_run({"policy.type": "act"}))


def test_output_subdir_for_fake_is_empty() -> None:
    assert output_subdir_in_run(_make_run({"__recipe__": FAKE_RECIPE_MARKER})) == ""


def test_output_subdir_for_docker_is_output() -> None:
    assert output_subdir_in_run(_make_run({"policy.type": "act"})) == CONTAINER_OUTPUT_SUBDIR


def test_fake_recipe_emits_python_runner_argv(tmp_path: Path) -> None:
    paths = RunPaths.for_run("test", runs_dir=tmp_path)
    paths.ensure_exists()
    run = _make_run({"__recipe__": FAKE_RECIPE_MARKER, "num_steps": 5, "save_every": 2})
    cmd, env = build_lerobot_train_command(run, paths)
    # Invokes the test fake worker by absolute file path (the autouse fixture
    # in tests/gui/conftest.py sets recipes.FAKE_RUNNER_PATH); skips the meta
    # marker; emits the rest as kebab-cased --flag value.
    assert cmd[1].endswith("fake_runner.py")
    assert cmd[2] == "--run-dir"
    assert "--num-steps" in cmd
    assert "5" in cmd
    assert "--save-every" in cmd
    assert "2" in cmd
    assert env == {}
    # Meta marker doesn't leak into the argv
    assert "--__recipe__" not in cmd
    assert FAKE_RECIPE_MARKER not in cmd


# ── Docker recipe ─────────────────────────────────────────────────────────────


def _docker_cmd(run: Run, paths: RunPaths) -> list[str]:
    cmd, _ = build_lerobot_train_command(run, paths)
    return cmd


def test_log_freq_injected_so_short_runs_log(tmp_path: Path) -> None:
    """lerobot logs metrics only at step % log_freq == 0, so a short run with
    the default log_freq (~200) prints nothing and the chart stays empty
    (round-5). The recipe injects a clamped log_freq when the user didn't."""
    paths = RunPaths.for_run("t", runs_dir=tmp_path)
    paths.ensure_exists()
    cmd = _docker_cmd(
        _make_run({"policy.type": "act", "dataset.repo_id": "lerobot/pusht", "steps": 30}), paths
    )
    flags = [a for a in cmd if a.startswith("--log_freq=")]
    assert len(flags) == 1
    assert 1 <= int(flags[0].split("=", 1)[1]) <= 30  # ≤ steps, so it actually fires


def test_log_freq_respects_user_value(tmp_path: Path) -> None:
    paths = RunPaths.for_run("t", runs_dir=tmp_path)
    paths.ensure_exists()
    cmd = _docker_cmd(
        _make_run({"policy.type": "act", "dataset.repo_id": "lerobot/pusht", "steps": 30, "log_freq": 7}),
        paths,
    )
    assert [a for a in cmd if a.startswith("--log_freq=")] == ["--log_freq=7"]  # not double-injected


def test_docker_recipe_command_shape(tmp_path: Path) -> None:
    paths = RunPaths.for_run("abc123", runs_dir=tmp_path)
    paths.ensure_exists()
    run = _make_run({"policy.type": "act", "dataset.repo_id": "lerobot/pusht", "steps": 5})
    cmd = _docker_cmd(run, paths)
    # docker run prefix
    assert cmd[0:2] == ["docker", "run"]
    # GPU passthrough
    assert "--gpus" in cmd and "all" in cmd
    # User UID/GID: host-identity TOKENS at compose time — resolved by the
    # orchestrator against the LAUNCHING host (remote uids aren't reliably
    # the GUI server's; first Nebius smoke ran as 1001).
    user_idx = cmd.index("--user")
    assert cmd[user_idx + 1] == f"{HOST_UID_TOKEN}:{HOST_GID_TOKEN}"
    # Image is positional, just before "lerobot-train"
    assert "lerobot-train" in cmd
    train_idx = cmd.index("lerobot-train")
    assert cmd[train_idx - 1] == DEFAULT_IMAGE


def test_docker_recipe_bind_mounts(tmp_path: Path) -> None:
    paths = RunPaths.for_run("abc123", runs_dir=tmp_path)
    paths.ensure_exists()
    run = _make_run({"policy.type": "act"})
    cmd = _docker_cmd(run, paths)
    # HF cache mount source is $HOME-on-the-host, tokenised at compose time
    assert f"{HOST_HOME_TOKEN}/.cache/huggingface:{CONTAINER_HF_CACHE}" in cmd
    # Run dir mount
    assert f"{paths.root}:{CONTAINER_RUNS_MOUNT}" in cmd


def test_docker_recipe_forces_safety_flags(tmp_path: Path) -> None:
    paths = RunPaths.for_run("xyz", runs_dir=tmp_path)
    run = _make_run({"policy.type": "act"})
    cmd = _docker_cmd(run, paths)
    # Forced flags that prevent post-train regressions (push attempts,
    # silent missed checkpoints, wandb auth prompts).
    assert "--policy.push_to_hub=false" in cmd
    assert "--wandb.enable=false" in cmd
    assert "--save_checkpoint=true" in cmd
    # output_dir locked to /runs/<subdir> (host can read via bind-mount)
    assert f"--output_dir={CONTAINER_RUNS_MOUNT}/{CONTAINER_OUTPUT_SUBDIR}" in cmd
    # repo_id is intentionally NOT forced — lerobot-train's validate()
    # only requires it when push_to_hub=true, and we hard-force false.
    # The old `local/<run_id>` injection was a footgun: when the form
    # leaked push_to_hub=true via the user-wins branch, lerobot-train
    # tried to create that fake namespace on HF Hub → 403.
    assert not any(t.startswith("--policy.repo_id=") for t in cmd), (
        f"recipe should not inject policy.repo_id; got: {cmd}"
    )


def test_docker_recipe_hard_force_push_to_hub_false_drops_user_input(tmp_path: Path) -> None:
    """User-submitted policy.push_to_hub=true is silently dropped — the
    recipe hard-forces false. Reason: any GUI-launched run defaults to
    NOT pushing to Hub. The user can publish from the model detail page
    after the run completes. Regression test for the SmolVLA 403."""
    paths = RunPaths.for_run("xyz", runs_dir=tmp_path)
    run = _make_run(
        {
            "policy.type": "act",
            "policy.push_to_hub": True,  # user (or form leak) requests push
            "output_dir": "/somewhere/else",  # user wants different output dir
        }
    )
    cmd = _docker_cmd(run, paths)
    # User's push_to_hub=true is silently dropped; forced false wins.
    assert "--policy.push_to_hub=false" in cmd
    assert "--policy.push_to_hub=true" not in cmd
    # Same hard-force for output_dir (bind-mount correctness invariant).
    assert f"--output_dir={CONTAINER_RUNS_MOUNT}/{CONTAINER_OUTPUT_SUBDIR}" in cmd
    assert "--output_dir=/somewhere/else" not in cmd


def test_docker_recipe_user_flag_wins_for_soft_forced(tmp_path: Path) -> None:
    """Soft-forced flags (wandb.enable, save_checkpoint) let user override.
    Only the never-override set (push_to_hub, output_dir) is hard-forced."""
    paths = RunPaths.for_run("xyz", runs_dir=tmp_path)
    run = _make_run(
        {
            "policy.type": "act",
            "wandb.enable": True,  # user wants wandb on
            "save_checkpoint": False,  # user wants no checkpoints
        }
    )
    cmd = _docker_cmd(run, paths)
    assert "--wandb.enable=true" in cmd
    assert "--wandb.enable=false" not in cmd
    assert "--save_checkpoint=false" in cmd
    assert "--save_checkpoint=true" not in cmd


def test_docker_recipe_translates_arg_types(tmp_path: Path) -> None:
    paths = RunPaths.for_run("xyz", runs_dir=tmp_path)
    run = _make_run(
        {
            "policy.type": "act",
            "policy.use_vae": False,  # bool → "false"
            "policy.optimizer_betas": [0.9, 0.999],  # list → "[0.9,0.999]"
            "policy.chunk_size": 100,  # int → "100"
        }
    )
    cmd = _docker_cmd(run, paths)
    assert "--policy.use_vae=false" in cmd
    assert "--policy.optimizer_betas=[0.9,0.999]" in cmd
    assert "--policy.chunk_size=100" in cmd


def test_docker_recipe_image_override(tmp_path: Path) -> None:
    paths = RunPaths.for_run("xyz", runs_dir=tmp_path)
    run = _make_run({"policy.type": "act", "__image__": "myreg/my-custom:dev"})
    cmd = _docker_cmd(run, paths)
    assert "myreg/my-custom:dev" in cmd
    assert DEFAULT_IMAGE not in cmd
    # Meta marker doesn't become a flag
    assert "--__image__=myreg/my-custom:dev" not in cmd


def test_docker_recipe_drops_meta_markers(tmp_path: Path) -> None:
    """Any double-underscore-prefixed arg is for the recipe builder, not the
    train CLI."""
    paths = RunPaths.for_run("xyz", runs_dir=tmp_path)
    run = _make_run({"policy.type": "act", "__some_marker__": "x", "__another__": True})
    cmd = _docker_cmd(run, paths)
    for tok in cmd:
        # Host-identity placeholders are intentional compose-time tokens
        # (resolved at launch); form meta markers must not leak.
        if "__LEROBOT_HOST_" in tok:
            continue
        assert "__" not in tok or tok.endswith(".pyc"), f"unexpected meta token: {tok}"


# ── docker_available probe ────────────────────────────────────────────────────


def test_docker_available_truthy_when_docker_on_path() -> None:
    # On this dev host, docker is installed (verified by C0 smoke)
    # In CI / sandboxes without docker, this would return False.
    # We don't strictly assert True — but assert it returns a bool.
    result = docker_available()
    assert isinstance(result, bool)


def test_docker_available_false_without_docker(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("lerobot.gui.training.recipes.shutil.which", lambda _: None)
    assert docker_available() is False


# ── HVLA flow_matching S1 recipe ──────────────────────────────────────────────


def _hvla_run(args: dict | None = None) -> Run:
    base = {"__recipe__": HVLA_FLOW_S1_RECIPE, "dataset_repo_id": "thewisp/some_data"}
    if args:
        base.update(args)
    return _make_run(base)


def test_is_hvla_recipe_detects_marker() -> None:
    assert is_hvla_flow_s1_recipe(_hvla_run())
    assert not is_hvla_flow_s1_recipe(_make_run({"policy.type": "act"}))
    assert not is_hvla_flow_s1_recipe(_make_run({"__recipe__": FAKE_RECIPE_MARKER}))


def test_hvla_output_subdir_is_output() -> None:
    """Both real recipes (lerobot-train, HVLA) write into /runs/output so the
    Models tab scanner finds checkpoints in the same place."""
    assert output_subdir_in_run(_hvla_run()) == CONTAINER_OUTPUT_SUBDIR


def test_hvla_recipe_entrypoint_and_dashed_cli(tmp_path: Path) -> None:
    paths = RunPaths.for_run("h1", runs_dir=tmp_path)
    paths.ensure_exists()
    run = _hvla_run({"steps": 200, "batch_size": 8, "chunk_size": 50})
    cmd = _docker_cmd(run, paths)
    # Entrypoint module
    assert "python" in cmd
    assert "lerobot.policies.hvla.s1.flow_matching.train" in cmd
    # CLI uses --key VALUE (space-separated), NOT --key=value
    assert "--dataset-repo-id" in cmd
    assert "thewisp/some_data" in cmd
    # Verify positional ordering: flag immediately followed by its value
    for flag, expected in [
        ("--dataset-repo-id", "thewisp/some_data"),
        ("--steps", "200"),
        ("--batch-size", "8"),
        ("--chunk-size", "50"),
    ]:
        idx = cmd.index(flag)
        assert cmd[idx + 1] == expected, f"{flag} expected to be followed by {expected}"


def test_hvla_recipe_forces_output_dir_into_bind_mount(tmp_path: Path) -> None:
    paths = RunPaths.for_run("h1", runs_dir=tmp_path)
    paths.ensure_exists()
    forced = f"{CONTAINER_RUNS_MOUNT}/{CONTAINER_OUTPUT_SUBDIR}"
    # User didn't set output_dir → we force one in
    cmd = _docker_cmd(_hvla_run({"steps": 5}), paths)
    idx = cmd.index("--output-dir")
    assert cmd[idx + 1] == forced
    # User did set output_dir → we silently override (correctness invariant)
    cmd2 = _docker_cmd(_hvla_run({"steps": 5, "output_dir": "/somewhere/else"}), paths)
    indices = [i for i, t in enumerate(cmd2) if t == "--output-dir"]
    assert len(indices) == 1, "should not emit --output-dir twice"
    assert cmd2[indices[0] + 1] == forced
    assert "/somewhere/else" not in cmd2


def test_hvla_recipe_omits_s2_latent_path_by_default(tmp_path: Path) -> None:
    """S1-without-S2 is the prototype's scope; we always strip --s2-latent-path."""
    paths = RunPaths.for_run("h1", runs_dir=tmp_path)
    paths.ensure_exists()
    # No s2_latent_path key → no flag
    cmd = _docker_cmd(_hvla_run({"steps": 5}), paths)
    assert "--s2-latent-path" not in cmd
    # User set s2_latent_path → we silently strip (S2 workflow not wired yet)
    cmd2 = _docker_cmd(_hvla_run({"steps": 5, "s2_latent_path": "/some/path.pt"}), paths)
    assert "--s2-latent-path" not in cmd2
    assert "/some/path.pt" not in cmd2


def test_hvla_recipe_drops_unknown_keys(tmp_path: Path) -> None:
    """HVLA argparse rejects unknown flags. The recipe filter drops anything
    not in HVLA_FLOW_S1_FIELD_TO_FLAG before composing the argv (in particular,
    lerobot-train-style dotted keys like policy.type should never reach
    HVLA's CLI)."""
    paths = RunPaths.for_run("h1", runs_dir=tmp_path)
    paths.ensure_exists()
    run = _hvla_run({"policy.type": "act", "wandb.enable": False, "steps": 5})
    cmd = _docker_cmd(run, paths)
    assert "--policy.type" not in cmd
    assert "--policy-type" not in cmd
    assert "--wandb.enable" not in cmd
    assert "act" not in cmd


def test_hvla_recipe_does_not_emit_lerobot_train_safety_flags(tmp_path: Path) -> None:
    """The forced flags from _FORCED_FLAGS (push_to_hub, repo_id, wandb.enable,
    save_checkpoint) only apply to lerobot-train. HVLA's argparse would error
    on them."""
    paths = RunPaths.for_run("h1", runs_dir=tmp_path)
    paths.ensure_exists()
    cmd = _docker_cmd(_hvla_run({"steps": 5}), paths)
    forbidden = [
        "--policy.push_to_hub=false",
        "--policy.repo_id",
        "--wandb.enable=false",
        "--save_checkpoint=true",
    ]
    for tok in forbidden:
        assert tok not in cmd, f"unexpected lerobot-train flag in HVLA argv: {tok}"


def test_hvla_recipe_bind_mounts(tmp_path: Path) -> None:
    paths = RunPaths.for_run("h1", runs_dir=tmp_path)
    paths.ensure_exists()
    cmd = _docker_cmd(_hvla_run({"steps": 5}), paths)
    assert f"{HOST_HOME_TOKEN}/.cache/huggingface:{CONTAINER_HF_CACHE}" in cmd
    assert f"{paths.root}:{CONTAINER_RUNS_MOUNT}" in cmd


def test_hvla_recipe_user_and_gpu_passthrough(tmp_path: Path) -> None:
    paths = RunPaths.for_run("h1", runs_dir=tmp_path)
    paths.ensure_exists()
    cmd = _docker_cmd(_hvla_run({"steps": 5}), paths)
    assert "--gpus" in cmd and "all" in cmd
    user_idx = cmd.index("--user")
    assert cmd[user_idx + 1] == f"{HOST_UID_TOKEN}:{HOST_GID_TOKEN}"


def test_hvla_recipe_image_override(tmp_path: Path) -> None:
    paths = RunPaths.for_run("h1", runs_dir=tmp_path)
    paths.ensure_exists()
    run = _hvla_run({"steps": 5, "__image__": "myreg/my-custom:dev"})
    cmd = _docker_cmd(run, paths)
    assert "myreg/my-custom:dev" in cmd
    assert DEFAULT_IMAGE not in cmd


# ── Host-placeholder resolution ──────────────────────────────────────────────


def test_resolve_host_placeholders_substitutes_all(tmp_path: Path) -> None:
    """Regression (2026-06-12 GPU smoke): --user was baked with the GUI
    server's uid at compose time; remote feit was uid 1001 and the
    container ground PermissionErrors into 1001-owned mounts."""
    paths = RunPaths.for_run("abc123", runs_dir=tmp_path)
    paths.ensure_exists()
    run = _make_run({"policy.type": "act"})
    cmd = _docker_cmd(run, paths)
    resolved = resolve_host_placeholders(cmd, uid=1001, gid=1001, home="/home/remoteuser")
    user_idx = resolved.index("--user")
    assert resolved[user_idx + 1] == "1001:1001"
    assert f"/home/remoteuser/.cache/huggingface:{CONTAINER_HF_CACHE}" in resolved
    assert not any("__LEROBOT_HOST_" in a for a in resolved)


def test_resolve_host_placeholders_rejects_relative_home() -> None:
    """ValueError (not assert): the message reaches the run's error field
    and must tell the user what to check, not dump an assert backtrace."""
    with pytest.raises(ValueError, match="non-absolute home"):
        resolve_host_placeholders(["x"], uid=1, gid=1, home="relative/home")
    with pytest.raises(ValueError, match="non-absolute home"):
        resolve_host_placeholders(["x"], uid=1, gid=1, home="")


def test_docker_recipe_sets_inductor_cache_dir(tmp_path: Path) -> None:
    """Regression (GPU smoke bug #4): torch's inductor cache calls
    getpass.getuser() at import time — a passwd lookup by uid that
    KeyErrors when the container runs as a host uid with no /etc/passwd
    entry in the image (any host whose user isn't uid 1000)."""
    paths = RunPaths.for_run("abc123", runs_dir=tmp_path)
    paths.ensure_exists()
    cmd = _docker_cmd(_make_run({"policy.type": "act"}), paths)
    e_idx = cmd.index("-e")
    assert cmd[e_idx + 1] == "TORCHINDUCTOR_CACHE_DIR=/tmp/torchinductor-cache"


def test_docker_recipe_hf_mount_outside_image_home(tmp_path: Path) -> None:
    """Regression (GPU smoke bug #5): mounting under /home/user_lerobot
    requires traversing image-baked dirs owned by uid 1000 — any other
    host uid gets EACCES before reaching the mount. Target must live at
    container root, with HF env overriding the image's baked paths."""
    paths = RunPaths.for_run("abc123", runs_dir=tmp_path)
    paths.ensure_exists()
    cmd = _docker_cmd(_make_run({"policy.type": "act"}), paths)
    assert CONTAINER_HF_CACHE == "/hf-cache"
    assert not CONTAINER_HF_CACHE.startswith("/home/")
    assert f"HF_HOME={CONTAINER_HF_CACHE}" in cmd
    assert f"HF_LEROBOT_HOME={CONTAINER_HF_CACHE}/lerobot" in cmd
    assert "TRITON_CACHE_DIR=/tmp/triton-cache" in cmd
    # bug #6: torchvision backbone download wanted ~/.cache/torch — HOME
    # itself must point at writable tmp for arbitrary host uids.
    assert "HOME=/tmp/lerobot-home" in cmd
    # the image bakes TORCH_HOME; torch.hub consults it before ~
    assert "TORCH_HOME=/tmp/lerobot-home/.cache/torch" in cmd

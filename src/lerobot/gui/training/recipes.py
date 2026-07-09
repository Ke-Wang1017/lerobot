# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""Recipe builder — composes the training argv from a Run.

Image-everywhere by default (DESIGN.md § Unified execution): a training run is
``docker run training-image lerobot-train ...`` on every host mode, so a
workstation and a cloud pod run the same image bit-for-bit. The recipe builder
produces that argv (plus an env dict) from a Run's args dict.

Local-native exception: on the workstation host a run may set
``Run.args["__execution__"]="native"`` (see :data:`EXECUTION_MODE_KEY`) to run
the trainer directly in the GUI server's own Python env — ``sys.executable -m
lerobot.scripts.lerobot_train ...`` with no container. Same forced-flag and
output-dir logic; only the wrapper (docker vs bare interpreter) and the
finetune-base / output-dir paths differ.

Run.args convention (flat dict, dotted keys):

    {
        "policy.type": "act",
        "policy.chunk_size": 100,
        "dataset.repo_id": "lerobot/pusht",
        "steps": 5,
        "batch_size": 8,
        "save_freq": 5,
    }

Each entry becomes a ``--key=value`` flag on the lerobot-train command line.

Forced flags (these are always emitted, regardless of what's in args; they
defend against the verified-by-smoke gotchas). Some are HARD-forced (user
input is silently dropped — see :data:`_NEVER_USER_OVERRIDE`); the rest let
the user override.

  --policy.push_to_hub=false       — HARD-forced. SmolVLA (and any future
                                     VLM-family policy) defaults
                                     push_to_hub=True in the dataclass; if
                                     the form leaks that into args, the
                                     post-train ``push_model_to_hub`` call
                                     fires and tries to create a Hub repo
                                     under whatever ``repo_id`` lerobot-train
                                     was given — which 403s for any namespace
                                     the user can't write to. We never want
                                     a GUI-launched run to push automatically.
  --wandb.enable=false             — tracking is opt-in. The start form flips
                                     this to true (user-wins branch) once a
                                     W&B key is connected; the docker recipe
                                     then forwards ``WANDB_API_KEY`` into the
                                     container by NAME (``-e WANDB_API_KEY``),
                                     so the secret never lands in an argv.
                                     See ``wandb_credentials.py``.
  --save_checkpoint=true           — explicit; checkpoints are how we close
                                     the felt loop in C3.
  --output_dir=/runs/output        — HARD-forced. Fixed subdir of the
                                     bind-mount. Must not pre-exist
                                     (lerobot-train refuses to overwrite).
                                     Bind-mount target /runs already exists;
                                     /runs/output does not.

  NOT forced: ``policy.repo_id``. lerobot-train's ``TrainPipelineConfig.
  validate()`` only requires ``policy.repo_id`` non-None when
  ``push_to_hub=True``; with push hard-forced false, ``repo_id=None`` is
  fine and the synthetic ``local/<run_id>`` we used to inject was a
  footgun (it became the 403'd Hub namespace once the leak above fired).

``Run.args["__recipe__"] == "__fake__"`` selects a test-only fake-training
worker (``tests/gui/training/fake_runner.py``, not shipped here) so the
orchestrator's unit tests run without docker. Its path is injected via
:data:`FAKE_RUNNER_PATH`, unset in production.
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path
from typing import Any

from lerobot.gui.training.runs import Run, RunPaths
from lerobot.gui.training.wandb_credentials import WANDB_API_KEY_ENV, wandb_enabled

# Pinned image tag — bumped explicitly via PR. ``latest`` is only published
# on main; per-branch builds publish ``<branch>-<sha>``. This default points
# at the latest verified-by-smoke build. Override per-run via
# Run.args["__image__"]. Content-addressing this tag from the source state
# (Dockerfile + lockfile hash) is a separate follow-up.
DEFAULT_IMAGE = "ghcr.io/thewisp/lerobot-training:feat-gui-training-deploy-proto-e6bf147"

# Marker that selects the fake-training runner instead of real lerobot-train.
# Used by orchestrator unit tests so they don't depend on docker.
FAKE_RECIPE_MARKER = "__fake__"

# Execution mode — how the trainer runs, independent of WHICH trainer (the
# recipe) runs. Set via Run.args["__execution__"]; absent means "docker".
#
#   docker  — `docker run <image> <entrypoint> …` (DESIGN.md § Unified
#             execution). The default: bit-for-bit parity with cloud hosts,
#             at the cost of a Docker install + image pull.
#   native  — the entrypoint runs directly in the GUI server's own Python
#             environment (`sys.executable -m …`), no container. Only valid
#             on the local workstation host — the GUI server's venv is the
#             one training uses, so this is meaningless on a remote host.
#             The orchestrator guards against native-on-remote.
EXECUTION_MODE_KEY = "__execution__"
EXECUTION_DOCKER = "docker"
EXECUTION_NATIVE = "native"

# Absolute path to the fake-training worker (a test fixture in
# ``tests/gui/``, set by its autouse fixture). None in production, so the
# fake recipe is test-only and fails loudly elsewhere.
FAKE_RUNNER_PATH: str | None = None

# Marker that selects the HVLA Flow Matching S1 training script instead of
# lerobot-train. HVLA isn't registered with lerobot-train's draccus policy
# registry — it has its own argparse-based train script with a different CLI
# shape (dashed --key value rather than dotted --key=value). Set via
# Run.args["__recipe__"]; the form does this when "HVLA Flow Matching S1"
# is picked.
HVLA_FLOW_S1_RECIPE = "hvla_flow_s1"

# Inside-container entrypoint for the HVLA Flow Matching S1 trainer.
HVLA_FLOW_S1_ENTRYPOINT = ["python", "-u", "-m", "lerobot.policies.hvla.s1.flow_matching.train"]

# HVLA argparse uses dashed-kebab CLI flag names. Map from the form's
# field key → CLI flag name. (Form keys are clean snake_case so they can
# be re-used by other recipes; per-recipe translation lives here.)
HVLA_FLOW_S1_FIELD_TO_FLAG: dict[str, str] = {
    "dataset_repo_id": "--dataset-repo-id",
    "output_dir": "--output-dir",
    "steps": "--steps",
    "batch_size": "--batch-size",
    "save_freq": "--save-freq",
    "num_workers": "--num-workers",
    "device": "--device",
    "chunk_size": "--chunk-size",
    "num_inference_steps": "--num-inference-steps",
    "rtc_max_delay": "--rtc-max-delay",
    "rtc_drop_prob": "--rtc-drop-prob",
    "max_delay": "--max-delay",
    "resize_images": "--resize-images",
    "hidden_dim": "--hidden-dim",
    "num_decoder_layers": "--num-decoder-layers",
    "s2_latent_path": "--s2-latent-path",  # OMIT to train without S2
}

# Inside-container paths. The bind-mounts in the docker command line map
# host paths to these.
CONTAINER_RUNS_MOUNT = "/runs"
CONTAINER_OUTPUT_SUBDIR = "output"  # /runs/output — lerobot-train writes here
# Mounted at container root, NOT inside the image user's home: the path
# walk to a target under /home/user_lerobot crosses image-baked dirs owned
# by uid 1000 with no world-x, so any other host uid gets EACCES before it
# even reaches the mount (GPU smoke bug #5). "/" is root-owned 755 —
# traversable by every uid.
CONTAINER_HF_CACHE = "/hf-cache"
# A local checkpoint chosen as the finetune base is bind-mounted here (read
# only) and ``--policy.pretrained_path`` is rewritten to this path, so the
# base weights are reachable inside the container regardless of where they
# live on the host (GUI runs dir, ./outputs, HF cache, …). A Hub repo id is
# left untouched — the container resolves it via the mounted HF cache.
CONTAINER_FINETUNE_BASE = "/finetune-base"

# ── Host-identity placeholders ───────────────────────────────────────────────
#
# Recipes are composed on the GUI server but EXECUTE on whichever host the
# transport points at. Anything host-dependent (uid/gid for --user, $HOME
# for the HF-cache bind mount) must therefore not be resolved at compose
# time — the GUI server's uid was baked into --user once, and the first
# remote VM whose user wasn't uid 1000 ground the container's writes into
# somebody else's directories. The recipe emits these tokens instead; the
# orchestrator substitutes them via TransportClient.host_identity() at
# launch time, on the launching host's truth.
HOST_UID_TOKEN = "__LEROBOT_HOST_UID__"  # nosec B105 — substitution token, not a secret
HOST_GID_TOKEN = "__LEROBOT_HOST_GID__"  # nosec B105 — substitution token, not a secret
HOST_HOME_TOKEN = "__LEROBOT_HOST_HOME__"  # nosec B105 — substitution token, not a secret


def resolve_host_placeholders(command: list[str], uid: int, gid: int, home: str) -> list[str]:
    """Substitute the host-identity tokens in a composed argv.

    Pre: ``home`` is an absolute path on the target host.
    Post: no ``__LEROBOT_HOST_*__`` token remains in the result.
    """
    if not home or not home.startswith("/"):
        raise ValueError(
            f"training host reported a non-absolute home directory ({home!r}) — "
            "check the host's $HOME (e.g. `ssh <host> 'echo $HOME'`)"
        )
    out = []
    for arg in command:
        arg = arg.replace(HOST_UID_TOKEN, str(uid))
        arg = arg.replace(HOST_GID_TOKEN, str(gid))
        arg = arg.replace(HOST_HOME_TOKEN, home)
        out.append(arg)
    for arg in out:
        assert "__LEROBOT_HOST_" not in arg, f"unresolved host placeholder in {arg!r}"
    return out


def is_fake_recipe(run: Run) -> bool:
    """Whether this Run uses the fake-training fallback path (no docker)."""
    return run.args.get("__recipe__") == FAKE_RECIPE_MARKER


def is_hvla_flow_s1_recipe(run: Run) -> bool:
    """Whether this Run uses the HVLA Flow Matching S1 training script
    instead of lerobot-train."""
    return run.args.get("__recipe__") == HVLA_FLOW_S1_RECIPE


def is_native_local(run: Run) -> bool:
    """Whether this Run runs the trainer directly in the GUI server's Python
    env (no docker). See :data:`EXECUTION_MODE_KEY`. Only meaningful on the
    local workstation host — the orchestrator rejects it on remote hosts."""
    return run.args.get(EXECUTION_MODE_KEY) == EXECUTION_NATIVE


def output_subdir_in_run(run: Run) -> str:
    """Per-run output subdir name relative to the run's root.

    The orchestrator uses this to know where the worker wrote checkpoints
    (host-side, via the bind-mount): ``paths.root / output_subdir_in_run(run)``.

    Fake recipe writes to ``paths.root`` directly (no subdir) for backwards
    compat with the existing unit tests.

    Real (lerobot-train OR HVLA flow_matching) recipes write to
    ``paths.root / output / ...`` — both honor ``--output-dir /runs/output``
    (HVLA uses dashed form, lerobot-train dotted; same result on disk).
    """
    return "" if is_fake_recipe(run) else CONTAINER_OUTPUT_SUBDIR


def build_lerobot_train_command(run: Run, paths: RunPaths) -> tuple[list[str], dict[str, str]]:
    """Return ``(command_argv, extra_env)`` for a Run.

    For docker recipes: composes ``docker run --gpus all --user UID:GID
    -v HF:HF -v paths.root:/runs IMAGE lerobot-train --key=value ...`` with
    all the forced flags above.

    For the fake recipe (``__recipe__=__fake__``, test-only): returns a
    ``python <FAKE_RUNNER_PATH> --run-dir … …`` argv. Requires the test
    harness to have set :data:`FAKE_RUNNER_PATH` (asserts otherwise).

    Pre: ``paths.root`` exists (created by RunPaths.ensure_exists()).
    Post: returned argv is ready for ``subprocess.Popen``; env dict should
    be merged on top of ``os.environ``.
    """
    if is_fake_recipe(run):
        return _build_fake_command(run, paths)
    # Execution mode (docker vs native-local) is orthogonal to which trainer
    # runs — both the standard lerobot-train recipe and the HVLA recipe can
    # run either wrapped in the image or directly in this venv.
    native = is_native_local(run)
    if is_hvla_flow_s1_recipe(run):
        return _build_hvla_flow_s1_command(run, paths, native=native)
    return _build_docker_command(run, paths, native=native)


def docker_available() -> bool:
    """Cheap probe used at run start to fail loudly if the docker recipe
    is requested but docker isn't installed."""
    return shutil.which("docker") is not None


# ── Fake-training fallback ────────────────────────────────────────────────────


def _build_fake_command(run: Run, paths: RunPaths) -> tuple[list[str], dict[str, str]]:
    assert FAKE_RUNNER_PATH is not None, (
        "the fake recipe is test-only; set recipes.FAKE_RUNNER_PATH (tests/gui/conftest.py does this)"
    )
    # By file path, not `python -m`: tests/ isn't importable from the spawn cwd.
    cmd = [sys.executable, FAKE_RUNNER_PATH, "--run-dir", str(paths.root)]
    for k, v in run.args.items():
        if k.startswith("__"):
            continue  # skip meta markers
        cmd.extend([f"--{k.replace('_', '-')}", _fmt_arg(v)])
    return cmd, {}


# ── Docker recipe ─────────────────────────────────────────────────────────────


# Flags the recipe builder always forces — see the module docstring.
_FORCED_FLAGS: dict[str, str] = {
    "policy.push_to_hub": "false",
    "wandb.enable": "false",
    "save_checkpoint": "true",
    "output_dir": f"{CONTAINER_RUNS_MOUNT}/{CONTAINER_OUTPUT_SUBDIR}",
}

# Subset of :data:`_FORCED_FLAGS` where a user-supplied value is dropped
# (the recipe wins, silently). Everything else in _FORCED_FLAGS lets the
# user override.
#   output_dir         — must live inside the bind-mount or the host
#                        can't read the checkpoints.
#   policy.push_to_hub — used to be overridable, but SmolVLA (and other
#                        VLM-family policies) default push_to_hub=True
#                        in the dataclass. The leak path was: dataclass
#                        default → form pre-fill → user-wins branch →
#                        argv → lerobot-train tries to create
#                        local/<run_id> on HF Hub → 403 at end of train.
#                        Nothing the GUI launches today should push to
#                        Hub automatically; the user can do it from the
#                        model detail page after the run completes.
_NEVER_USER_OVERRIDE: frozenset[str] = frozenset({"output_dir", "policy.push_to_hub"})

# lerobot-train logs metrics only at ``step % log_freq == 0``. Its default
# (~200) means a short run never logs and the dashboard chart stays empty
# (round-5 smoke). When the user doesn't set log_freq, pick a cadence that
# yields ~_TARGET_LOG_POINTS points, capped at the default so long runs aren't
# spammed.
_DEFAULT_LOG_FREQ = 200
_TARGET_LOG_POINTS = 20


def _docker_argv_base(
    image: str,
    paths: RunPaths,
    extra_mounts: list[str] | None = None,
    forward_env: list[str] | None = None,
) -> list[str]:
    """The docker-run prefix shared by every recipe: GPU passthrough,
    host-identity placeholders, the arbitrary-uid env overrides, and the
    two bind mounts. One seam so the GPU-smoke lessons can't drift apart
    between recipe builders (they were patched in parallel six times
    before this was extracted).

    ``extra_mounts`` (already ``["-v", "src:dst:opts", ...]`` tokens) are
    spliced in just before the image — used to mount a local finetune-base
    checkpoint into the container.

    ``forward_env`` names env vars to pass through from the launching process
    (``-e NAME`` with no ``=value``). That form is what keeps a secret out of
    the argv — docker copies the value from its own environment, which the
    orchestrator populated. A name whose var is unset is simply not set in the
    container, so forwarding is safe even when the value is absent.

    Post: ends with the image — callers append their entrypoint + args.
    """
    # Resolved on the LAUNCHING host at launch time (see the placeholder
    # block above) — never expanduser() here, this code runs on the GUI
    # server while the mount source lives on the training host.
    hf_cache_host = f"{HOST_HOME_TOKEN}/.cache/huggingface"
    return [
        "docker",
        "run",
        "--rm",
        "--gpus",
        "all",
        # Docker defaults /dev/shm to 64 MiB, which the PyTorch DataLoader
        # blows through immediately for any camera-using policy (one batch
        # of 4 cameras × 512² × uint8 is ~12 MB per sample). The crash
        # surfaces as "unable to allocate shared memory(shm) ... Resource
        # temporarily unavailable" in a worker process. 8g is conservative
        # for typical ML batches.
        "--shm-size=8g",
        "--user",
        f"{HOST_UID_TOKEN}:{HOST_GID_TOKEN}",
        # The container runs as the HOST's uid, which usually has no
        # /etc/passwd entry inside the image (only user_lerobot=1000 does).
        # torch's inductor cache calls getpass.getuser() AT IMPORT TIME,
        # which is a passwd lookup by uid → KeyError on any host whose
        # user isn't uid 1000. Point the cache somewhere world-writable
        # so the lookup never happens.
        "-e",
        "TORCHINDUCTOR_CACHE_DIR=/tmp/torchinductor-cache",
        # The image bakes HF_HOME/HF_LEROBOT_HOME pointing into the image
        # user's home; override to the world-traversable mount target.
        # TRITON_CACHE_DIR: same passwd-less-uid class as inductor, fires
        # on first triton-compiled kernel.
        "-e",
        f"HF_HOME={CONTAINER_HF_CACHE}",
        "-e",
        f"HF_LEROBOT_HOME={CONTAINER_HF_CACHE}/lerobot",
        "-e",
        "TRITON_CACHE_DIR=/tmp/triton-cache",
        # Closes the whole ~-derived-cache class for arbitrary host uids
        # (torch hub backbones, matplotlib, any XDG default): the image
        # user's home isn't writable (or traversable) for uid != 1000.
        # /tmp is sticky world-writable; libs mkdir what they need. The
        # one cache that must persist, HF, is explicitly mounted above.
        "-e",
        "HOME=/tmp/lerobot-home",
        # The image also bakes TORCH_HOME into the image user's home, and
        # torch.hub checks TORCH_HOME before falling back to ~ — so the
        # HOME redirect alone doesn't cover the backbone-weights cache.
        "-e",
        "TORCH_HOME=/tmp/lerobot-home/.cache/torch",
        *[tok for name in (forward_env or []) for tok in ("-e", name)],
        "-v",
        f"{hf_cache_host}:{CONTAINER_HF_CACHE}",
        "-v",
        f"{paths.root}:{CONTAINER_RUNS_MOUNT}",
        *(extra_mounts or []),
        image,
    ]


def _standard_train_flags(run: Run, paths: RunPaths, *, native: bool) -> tuple[list[str], list[str]]:
    """Translate Run.args → lerobot-train ``--key=value`` flags.

    Shared by the docker and native-local standard recipes. Only two things
    differ between the modes:
      1. ``output_dir`` — a real host path for native, the ``/runs`` bind-mount
         target for docker.
      2. finetune base — native reads the checkpoint from its real host path;
         docker bind-mounts it read-only into the container and rewrites the
         flag to the in-container path.

    Returns ``(train_args, extra_mounts)``. ``extra_mounts`` is always empty
    for native — no container means no bind mounts.
    """
    output_dir = (
        str(paths.root / CONTAINER_OUTPUT_SUBDIR)
        if native
        else f"{CONTAINER_RUNS_MOUNT}/{CONTAINER_OUTPUT_SUBDIR}"
    )
    forced = {**_FORCED_FLAGS, "output_dir": output_dir}

    train_args: list[str] = []
    extra_mounts: list[str] = []
    seen: set[str] = set()
    # User-supplied flags first
    for k, v in run.args.items():
        if k.startswith("__"):
            continue
        if k == "policy.pretrained_path":
            # Finetune base. A Hub repo id (not an existing local dir) passes
            # through untouched in both modes — resolved via the HF cache.
            base = str(v).strip()
            if not base:
                continue
            host_path = Path(base).expanduser()
            if native:
                # No container: the trainer reads the checkpoint directly.
                train_args.append(f"--policy.pretrained_path={host_path if host_path.is_dir() else base}")
            elif host_path.is_dir():
                # A local checkpoint dir isn't under either bind mount in the
                # general case, so mount it read-only and rewrite the flag.
                extra_mounts.extend(["-v", f"{host_path}:{CONTAINER_FINETUNE_BASE}:ro"])
                train_args.append(f"--policy.pretrained_path={CONTAINER_FINETUNE_BASE}")
            else:
                train_args.append(f"--policy.pretrained_path={base}")
            seen.add(k)
            continue
        if k in forced:
            # User explicitly set a flag we'd otherwise force — silently
            # drop iff in the never-override set; otherwise let user win.
            if k in _NEVER_USER_OVERRIDE:
                continue
            train_args.append(f"--{k}={_fmt_arg(v)}")
            seen.add(k)
            continue
        train_args.append(f"--{k}={_fmt_arg(v)}")
        seen.add(k)

    # Forced flags — emit only if user didn't already provide
    for k, v in forced.items():
        if k in seen:
            continue
        train_args.append(f"--{k}={v}")

    # Make metrics actually print (see _DEFAULT_LOG_FREQ note) unless the user
    # set their own log_freq.
    if "log_freq" not in seen:
        try:
            steps = int(run.args.get("steps") or 0)
        except (TypeError, ValueError):
            steps = 0
        if steps > 0:
            train_args.append(f"--log_freq={max(1, min(_DEFAULT_LOG_FREQ, steps // _TARGET_LOG_POINTS))}")

    return train_args, extra_mounts


def _build_docker_command(
    run: Run, paths: RunPaths, *, native: bool = False
) -> tuple[list[str], dict[str, str]]:
    train_args, extra_mounts = _standard_train_flags(run, paths, native=native)
    if native:
        # Run lerobot-train in the GUI server's own interpreter/venv. `-u`
        # keeps stdout unbuffered so the log file fills in real time (docker
        # already line-buffers; native must ask). `-m` guarantees the same
        # environment the GUI server runs in rather than trusting PATH.
        cmd = [sys.executable, "-u", "-m", "lerobot.scripts.lerobot_train", *train_args]
        return cmd, {}
    image = run.args.get("__image__") or DEFAULT_IMAGE
    # Only a tracked run gets the key. Forwarding it unconditionally would put
    # the GUI server's ambient WANDB_API_KEY inside every container we launch.
    forward_env = [WANDB_API_KEY_ENV] if wandb_enabled(run.args) else []
    docker_argv = [
        *_docker_argv_base(image, paths, extra_mounts, forward_env),
        "lerobot-train",
        *train_args,
    ]
    return docker_argv, {}


def _fmt_arg(v: Any) -> str:
    """Format a Python value for lerobot-train's draccus CLI parser.

    Booleans → 'true' / 'false' (draccus accepts both).
    Lists / tuples → '[a,b,c]' (draccus syntax).
    Everything else → str().
    """
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, (list, tuple)):
        return "[" + ",".join(_fmt_arg(x) for x in v) + "]"
    return str(v)


# ── HVLA Flow Matching S1 recipe ──────────────────────────────────────────────


def _build_hvla_flow_s1_command(
    run: Run, paths: RunPaths, *, native: bool = False
) -> tuple[list[str], dict[str, str]]:
    """Compose a `docker run … python -m lerobot.policies.hvla.s1.flow_matching.train …`
    argv for the HVLA Flow Matching S1 training script (or, when ``native``,
    the bare ``python -m …`` argv running in the GUI server's own venv).

    Differs from the lerobot-train recipe in three ways:
      1. Different entrypoint inside the container.
      2. Dashed-kebab argparse CLI (--key value, space-separated), not
         draccus's --key=value dotted-dataclass form.
      3. None of lerobot-train's safety flags (--policy.push_to_hub etc.)
         apply — HVLA's argparse rejects them.

    S2 conditioning: HVLA trains WITHOUT S2 iff ``--s2-latent-path`` is
    omitted from the CLI. The form leaves it out by default; user can
    opt in via Run.args["s2_latent_path"]="..." (NOT supported via the
    form today; would need an upstream extension).
    """
    image = run.args.get("__image__") or DEFAULT_IMAGE

    # Translate the form's flat snake_case args dict → HVLA's dashed CLI flags.
    train_args: list[str] = []
    for k, v in run.args.items():
        if k.startswith("__"):
            continue
        flag = HVLA_FLOW_S1_FIELD_TO_FLAG.get(k)
        if flag is None:
            # Skip unknown keys — HVLA argparse would error on them. Logged
            # at the orchestrator level if we ever want to surface a warning.
            continue
        # Bool / None / list handling: HVLA argparse expects "true"/"false"
        # for bools (same as draccus); list args aren't part of the schema.
        if v is None:
            continue
        train_args.extend([flag, _fmt_arg(v)])

    # Forced: output-dir always lives inside the bind-mount (host needs to
    # read checkpoints back), and we always omit --s2-latent-path so the
    # S1-without-S2 path is taken — that's the prototype's scope per
    # DESIGN.md (S2 latents extraction is a separate workflow not wired
    # to the GUI yet).
    # Native writes to a real host path; docker to the /runs bind-mount target.
    forced_output_dir = (
        str(paths.root / CONTAINER_OUTPUT_SUBDIR)
        if native
        else f"{CONTAINER_RUNS_MOUNT}/{CONTAINER_OUTPUT_SUBDIR}"
    )
    if "--output-dir" not in train_args:
        train_args.extend(["--output-dir", forced_output_dir])
    else:
        # User-provided output-dir — refuse to honor it; we MUST control
        # the path so checkpoints land where the host's manifest scanner
        # looks. Replace silently.
        idx = train_args.index("--output-dir")
        train_args[idx + 1] = forced_output_dir
    # Strip any --s2-latent-path the user shoved in via meta marker — until
    # the GUI properly supports the S2 conditioning workflow, we always
    # train S1-only.
    if "--s2-latent-path" in train_args:
        idx = train_args.index("--s2-latent-path")
        del train_args[idx : idx + 2]

    if native:
        cmd = [sys.executable, "-u", "-m", "lerobot.policies.hvla.s1.flow_matching.train", *train_args]
        return cmd, {}
    docker_argv = [
        *_docker_argv_base(image, paths),
        *HVLA_FLOW_S1_ENTRYPOINT,
        *train_args,
    ]
    return docker_argv, {}

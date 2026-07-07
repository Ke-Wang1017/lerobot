# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""Recipe builder — composes the `docker run … lerobot-train …` argv from a Run.

Image-everywhere (DESIGN.md § Unified execution): every training run is
``docker run training-image lerobot-train ...``, on every host mode. The
recipe builder produces that argv (plus an env dict) from a Run's args dict.

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
  --wandb.enable=false             — no tracking in v1.
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
from typing import Any

from lerobot.gui.training.runs import Run, RunPaths

# Pinned image tag — bumped explicitly via PR. ``latest`` is only published
# on main; per-branch builds publish ``<branch>-<sha>``. This default points
# at the latest verified-by-smoke build. Override per-run via
# Run.args["__image__"]. Content-addressing this tag from the source state
# (Dockerfile + lockfile hash) is a separate follow-up.
DEFAULT_IMAGE = "ghcr.io/thewisp/lerobot-training:feat-gui-training-deploy-proto-e6bf147"

# Marker that selects the fake-training runner instead of real lerobot-train.
# Used by orchestrator unit tests so they don't depend on docker.
FAKE_RECIPE_MARKER = "__fake__"

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
    if is_hvla_flow_s1_recipe(run):
        return _build_hvla_flow_s1_command(run, paths)
    return _build_docker_command(run, paths)


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


def _docker_argv_base(image: str, paths: RunPaths) -> list[str]:
    """The docker-run prefix shared by every recipe: GPU passthrough,
    host-identity placeholders, the arbitrary-uid env overrides, and the
    two bind mounts. One seam so the GPU-smoke lessons can't drift apart
    between recipe builders (they were patched in parallel six times
    before this was extracted).

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
        "-v",
        f"{hf_cache_host}:{CONTAINER_HF_CACHE}",
        "-v",
        f"{paths.root}:{CONTAINER_RUNS_MOUNT}",
        image,
    ]


def _build_docker_command(run: Run, paths: RunPaths) -> tuple[list[str], dict[str, str]]:
    image = run.args.get("__image__") or DEFAULT_IMAGE

    # Translate Run.args → lerobot-train --key=value flags
    train_args: list[str] = []
    seen: set[str] = set()
    # User-supplied flags first
    for k, v in run.args.items():
        if k.startswith("__"):
            continue
        if k in _FORCED_FLAGS:
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
    for k, v in _FORCED_FLAGS.items():
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

    docker_argv = [
        *_docker_argv_base(image, paths),
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


def _build_hvla_flow_s1_command(run: Run, paths: RunPaths) -> tuple[list[str], dict[str, str]]:
    """Compose a `docker run … python -m lerobot.policies.hvla.s1.flow_matching.train …`
    argv for the HVLA Flow Matching S1 training script.

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
    forced_output_dir = f"{CONTAINER_RUNS_MOUNT}/{CONTAINER_OUTPUT_SUBDIR}"
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

    docker_argv = [
        *_docker_argv_base(image, paths),
        *HVLA_FLOW_S1_ENTRYPOINT,
        *train_args,
    ]
    return docker_argv, {}

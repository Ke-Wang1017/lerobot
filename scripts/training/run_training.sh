#!/usr/bin/env bash
# Launch one training run inside a fresh container. Assumes the host has
# been set up via setup_host.sh — pulls/builds nothing here.
#
# This is the prototype-script analog of the future GUI's "Start training"
# button — one container per run, --rm so it cleans up on exit, training
# command forwarded verbatim.
#
# ── Modes ───────────────────────────────────────────────────────────────────
#
# Default mode runs the lerobot we baked into the image (i.e., the commit
# that was on the host's working tree at setup_host.sh time):
#
#     bash run_training.sh -- lerobot-train --policy.type=act ...
#
# Ad-hoc local mode bind-mounts your current src/lerobot/ on top of the
# image's installed source. The image's editable install (pip install -e .)
# resolves through the bind-mount so your live edits run immediately,
# without rebuilding. Use this for fast dev iteration:
#
#     bash run_training.sh --bind-local -- lerobot-train --policy.type=act ...
#
# Caveat: dependencies (torch, transformers, ffmpeg, etc.) still come from
# the image. If you edit pyproject.toml or uv.lock, re-run setup_host.sh
# to rebuild the image — bind-mount doesn't reach pip-installed deps.
#
# Override the image (e.g. to test a tagged custom build):
#
#     bash run_training.sh --image=lerobot-training:my-experiment -- lerobot-train ...
#
# Combine bind-local + custom image for the full "I'm developing" loop.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
DEFAULT_IMAGE="lerobot-training:dev"
HF_CACHE_HOST="${HF_HOME:-${HOME}/.cache/huggingface}"
OUTPUTS_HOST="${REPO_ROOT}/outputs"

log() { printf '\033[1;36m[run]\033[0m %s\n' "$*"; }
err() { printf '\033[1;31m[run]\033[0m %s\n' "$*" >&2; }

# Choose between `docker` and `sudo docker`. On a fresh cloud-init VM, the
# user usually isn't in the `docker` group yet (cloud-init's `sudo:` adds
# a sudoers entry but doesn't change group membership). Fall back to sudo
# instead of failing — the user can fix this permanently by logging out
# and back in (the install_prereqs.sh script adds them to the group).
docker_cmd() {
    if docker info >/dev/null 2>&1; then
        echo "docker"
    elif sudo -n docker info >/dev/null 2>&1; then
        echo "sudo docker"
    else
        err "docker is unreachable both as your user AND via passwordless sudo."
        err "Either log out/in (if you ran install_prereqs.sh) or check the daemon."
        exit 2
    fi
}

# ── Parse args ───────────────────────────────────────────────────────────────

IMAGE_REF="${DEFAULT_IMAGE}"
BIND_LOCAL=0
TRAINING_CMD=()

while [[ $# -gt 0 ]]; do
    case $1 in
        --bind-local)
            BIND_LOCAL=1
            shift
            ;;
        --image=*)
            IMAGE_REF="${1#--image=}"
            shift
            ;;
        --)
            shift
            TRAINING_CMD=("$@")
            break
            ;;
        -h|--help)
            sed -n '2,/^$/p' "$0" | sed 's/^# *//' | head -40
            exit 0
            ;;
        *)
            # Allow training args without explicit -- separator
            TRAINING_CMD=("$@")
            break
            ;;
    esac
done

# Default smoke command if user supplied no training command at all.
# Note: lerobot-train requires --policy.repo_id even when wandb is off
# (its validate() insists; it'll try to push to Hub at the end and 401
# if HF_TOKEN isn't set — but training itself completes successfully).
if [[ ${#TRAINING_CMD[@]} -eq 0 ]]; then
    log "No training command supplied; running bundled smoke (10-step ACT on pusht)"
    TRAINING_CMD=(
        lerobot-train
        --policy.type=act
        --policy.repo_id=local/smoke-test-act-pusht
        --dataset.repo_id=lerobot/pusht
        --output_dir=/lerobot/outputs/smoke_act_pusht
        --job_name=smoke_act_pusht
        --steps=10
        --save_freq=10
        --batch_size=2
        --num_workers=0
        --policy.device=cuda
        --wandb.enable=false
    )
fi

# ── Sanity checks ────────────────────────────────────────────────────────────

if ! command -v docker >/dev/null 2>&1; then
    err "docker not found. Run setup_host.sh first."
    exit 2
fi

DOCKER=$(docker_cmd)

if ! ${DOCKER} image inspect "${IMAGE_REF}" >/dev/null 2>&1; then
    err "Image ${IMAGE_REF} not present locally."
    err "Run setup_host.sh first (or --image=<existing-tag>)."
    exit 2
fi

# ── Container launch ─────────────────────────────────────────────────────────

mkdir -p "${HF_CACHE_HOST}" "${OUTPUTS_HOST}"

# The image's container user (`user_lerobot`, UID 1000 inside) and the host
# user may have different UIDs, so a bind-mounted directory owned by the
# host user may be unwritable from inside the container. Pre-relax the
# permissions on the outputs dir so the container can write checkpoints.
# (The HF cache mount has the same issue but is more delicate — see the
# "HF cache bind-mount" entry in scripts/training/README.md's gap table.)
chmod 0777 "${OUTPUTS_HOST}"

# Build the mounts list. Always:
#  - HF cache (for datasets, models, token; shared with host)
#  - outputs (so checkpoints survive container exit)
MOUNTS=(
    -v "${HF_CACHE_HOST}:/home/user_lerobot/.cache/huggingface"
    -v "${OUTPUTS_HOST}:/lerobot/outputs"
)

# Bind-local overlay: replace the image's /lerobot/src/lerobot with the
# host's working tree. The image's editable install (`pip install -e .`)
# was built against /lerobot/src/lerobot, so any module under
# `import lerobot.X.Y` resolves through the bind-mount.
if [[ $BIND_LOCAL -eq 1 ]]; then
    MOUNTS+=(-v "${REPO_ROOT}/src/lerobot:/lerobot/src/lerobot")
    log "Bind-mount mode: ${REPO_ROOT}/src/lerobot → /lerobot/src/lerobot"
    log "  (your live edits will be picked up; deps still come from the image)"
fi

# HF token: forward as env var. Order of preference:
#   1. HF_TOKEN env var on the host (highest — explicit user intent)
#   2. ~/.cache/huggingface/token (the file `huggingface-cli login` writes)
# Avoids needing to huggingface-cli login inside the container.
HF_TOKEN_ARGS=()
HF_TOKEN_VALUE=""
if [[ -n "${HF_TOKEN:-}" ]]; then
    HF_TOKEN_VALUE="${HF_TOKEN}"
elif [[ -f "${HF_CACHE_HOST}/token" ]]; then
    HF_TOKEN_VALUE=$(cat "${HF_CACHE_HOST}/token")
fi
if [[ -n "${HF_TOKEN_VALUE}" ]]; then
    HF_TOKEN_ARGS+=(-e "HF_TOKEN=${HF_TOKEN_VALUE}")
    log "Forwarding HF_TOKEN to container."
else
    log "No HF_TOKEN found on host. Training will run, but Hub push at"
    log "end-of-run will 401. Use 'huggingface-cli login' to fix."
fi

# Override CI-test ENV vars baked into upstream Dockerfile.internal that
# would leak into our run. (Our Dockerfile.training already drops them;
# this override is for the --image=upstream:tag case.)
ENV_OVERRIDES=(
    -e "CUDA_VISIBLE_DEVICES=" # unset; let the GPU pass-through layer choose
    -e "TEST_TYPE="
)

log "Image:    ${IMAGE_REF}"
log "Outputs:  ${OUTPUTS_HOST}"
log "HF cache: ${HF_CACHE_HOST}"
log "Command:  ${TRAINING_CMD[*]}"
log ""

# --init: Docker injects its own minimal init (effectively tini) as PID 1,
#         so SIGTERM reaches the training process — required for clean
#         shutdown on preemption / cancel.
# --rm:   container is removed on exit (outputs survive via bind-mount).
# --gpus all: NVIDIA Container Toolkit pass-through; ~0% perf overhead.
# --network host: skip Docker bridge for faster HF Hub traffic.
${DOCKER} run --rm \
    --init \
    --gpus all \
    --network host \
    -t \
    "${MOUNTS[@]}" \
    "${HF_TOKEN_ARGS[@]}" \
    "${ENV_OVERRIDES[@]}" \
    "${IMAGE_REF}" \
    "${TRAINING_CMD[@]}"

log ""
log "Done. Outputs at ${OUTPUTS_HOST}"

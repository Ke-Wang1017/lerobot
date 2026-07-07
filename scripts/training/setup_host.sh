#!/usr/bin/env bash
# Setup the host for LeRobot training: verify prerequisites, ensure the
# deployment image is ready, confirm GPU access works inside a container.
# Idempotent — run once per host (or after a new image is published).
#
# This script is "deploy" only. It does not run training; run_training.sh
# handles per-run launches. Mirrors what the future GUI's "Add training
# host" dialog will do.
#
# ── On image source ─────────────────────────────────────────────────────────
#
# Our fork's training image is published to GHCR by CI on every push to
# main (see .github/workflows/docker_publish_fork_training.yml):
#
#     ghcr.io/thewisp/lerobot-training:latest      ← tracks main
#     ghcr.io/thewisp/lerobot-training:main-<sha>  ← per-commit, immutable
#     ghcr.io/thewisp/lerobot-training:<release>   ← release tags
#
# Default mode pulls `:latest`. The image is built from our fork's source
# + pyproject.toml + uv.lock so the deps match our code exactly — unlike
# upstream's `huggingface/lerobot-gpu:latest`, which won't satisfy our
# fork's imports.
#
# Fallback: if the pull fails (network down, image not yet public, etc.)
# the script offers to build locally from docker/Dockerfile.training.
#
# ── Usage ───────────────────────────────────────────────────────────────────
#
#   bash scripts/training/setup_host.sh                  # pull :latest from GHCR
#   bash scripts/training/setup_host.sh --tag=main-abc12 # pull a specific tag
#   bash scripts/training/setup_host.sh --build-local    # build from Dockerfile.training
#   bash scripts/training/setup_host.sh --image=foo:tag  # use any image you specify

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
GHCR_OWNER="thewisp"   # lowercased; matches GHCR namespace
GHCR_REPO="lerobot-training"
DEFAULT_TAG="latest"
LOCAL_IMAGE="lerobot-training:dev"
DOCKERFILE="${REPO_ROOT}/docker/Dockerfile.training"

log() { printf '\033[1;36m[setup]\033[0m %s\n' "$*"; }
err() { printf '\033[1;31m[setup]\033[0m %s\n' "$*" >&2; }

# ── Parse args ───────────────────────────────────────────────────────────────

MODE="pull"             # pull | build | custom
IMAGE_REF=""
GHCR_TAG="${DEFAULT_TAG}"

while [[ $# -gt 0 ]]; do
    case $1 in
        --build-local)
            MODE="build"
            IMAGE_REF="${LOCAL_IMAGE}"
            shift
            ;;
        --tag=*)
            MODE="pull"
            GHCR_TAG="${1#--tag=}"
            shift
            ;;
        --image=*)
            MODE="custom"
            IMAGE_REF="${1#--image=}"
            shift
            ;;
        -h|--help)
            sed -n '2,/^$/p' "$0" | sed 's/^# *//' | head -40
            exit 0
            ;;
        *)
            err "Unknown argument: $1"
            err "See --help."
            exit 2
            ;;
    esac
done

if [[ "${MODE}" == "pull" ]]; then
    IMAGE_REF="ghcr.io/${GHCR_OWNER}/${GHCR_REPO}:${GHCR_TAG}"
fi

# ── Prereq checks ────────────────────────────────────────────────────────────

# Resolve `docker` vs `sudo docker`. On a fresh cloud-init VM, the user
# usually isn't in the `docker` group yet (cloud-init's `sudo:` adds a
# sudoers entry but doesn't change group membership). Falling back to sudo
# avoids a "log out / back in" interruption right after install. The
# permanent fix is handled by install_prereqs.sh which `usermod -aG`s the
# user; this falls back gracefully until the user re-logs in.
DOCKER=""
docker_cmd() {
    if docker info >/dev/null 2>&1; then
        DOCKER="docker"
    elif sudo -n docker info >/dev/null 2>&1; then
        DOCKER="sudo docker"
        log "  (using 'sudo docker' — your user isn't in the docker group yet."
        log "   Log out + in after install_prereqs.sh ran to use 'docker' directly.)"
    else
        err "Docker daemon is unreachable both as your user AND via passwordless sudo. Try:"
        err "    sudo systemctl start docker"
        err "    sudo bash ${REPO_ROOT}/scripts/training/install_prereqs.sh"
        exit 2
    fi
}

check_prereqs() {
    local missing=0

    if ! command -v docker >/dev/null 2>&1; then
        err "docker not found on this host."
        err "Install Docker + nvidia-container-toolkit:"
        err "    sudo bash ${REPO_ROOT}/scripts/training/install_prereqs.sh"
        missing=1
    fi

    if ! command -v nvidia-smi >/dev/null 2>&1; then
        err "nvidia-smi not found — no NVIDIA driver detected."
        missing=1
    fi

    [[ $missing -eq 1 ]] && exit 2

    docker_cmd     # populates $DOCKER with "docker" or "sudo docker"

    # Probe GPU pass-through with a stock cuda image (small, ~150 MB)
    if ! ${DOCKER} run --rm --gpus all \
            nvidia/cuda:12.4.1-base-ubuntu22.04 nvidia-smi >/dev/null 2>&1; then
        err "Docker can't access the GPU. Likely missing nvidia-container-toolkit."
        err "Install it via:"
        err "    sudo bash ${REPO_ROOT}/scripts/training/install_prereqs.sh"
        exit 2
    fi
}

# ── Image acquisition ────────────────────────────────────────────────────────

build_locally() {
    log "Building from ${DOCKERFILE}"
    log "(first build ~5 min for dep solve + install; cached after)"
    # shellcheck disable=SC2086 # $DOCKER intentionally splits to "docker" or "sudo docker"
    DOCKER_BUILDKIT=1 ${DOCKER} build \
        -f "${DOCKERFILE}" \
        -t "${LOCAL_IMAGE}" \
        "${REPO_ROOT}"
    IMAGE_REF="${LOCAL_IMAGE}"
}

acquire_image() {
    case "${MODE}" in
        pull)
            log "Pulling published image: ${IMAGE_REF}"
            log "(published by CI from our fork; see .github/workflows/docker_publish_fork_training.yml)"
            # shellcheck disable=SC2086
            if ! ${DOCKER} pull "${IMAGE_REF}" 2>&1; then
                err ""
                err "Pull failed. Common causes:"
                err "  - The image hasn't been published yet (push the workflow + wait for CI)"
                err "  - The package is private (default for GHCR — go to"
                err "    https://github.com/users/${GHCR_OWNER}/packages/container/${GHCR_REPO}/settings"
                err "    → 'Change visibility' → Public)"
                err "  - You're offline"
                err ""
                err "Falling back to local build. Re-run with --build-local to skip the pull attempt."
                build_locally
            fi
            ;;
        build)
            build_locally
            ;;
        custom)
            log "Using custom image: ${IMAGE_REF}"
            # shellcheck disable=SC2086
            if ! ${DOCKER} image inspect "${IMAGE_REF}" >/dev/null 2>&1; then
                log "Image not present locally — attempting pull..."
                # shellcheck disable=SC2086
                ${DOCKER} pull "${IMAGE_REF}"
            else
                log "Image present locally — skipping pull."
            fi
            log ""
            log "  Note: custom images must already have our fork's deps installed."
            log "  Upstream's huggingface/lerobot-gpu:latest will NOT work for our"
            log "  fork — pyproject.toml and uv.lock have diverged from upstream."
            ;;
    esac
}

# ── Smoke-verify the image works ─────────────────────────────────────────────

verify_image() {
    log "Verifying image: python + cuda + lerobot importable..."
    # shellcheck disable=SC2086
    ${DOCKER} run --rm --gpus all --init "${IMAGE_REF}" \
        python -c "
import sys
import torch
import lerobot
print(f'  python:   {sys.version.split()[0]}')
print(f'  torch:    {torch.__version__}')
print(f'  CUDA:     avail={torch.cuda.is_available()}  visible_devices={torch.cuda.device_count()}')
print(f'  lerobot:  {lerobot.__file__}')
        "
}

# ── Main ─────────────────────────────────────────────────────────────────────

main() {
    check_prereqs
    acquire_image
    verify_image

    log ""
    log "Host setup complete."
    log "  Image: ${IMAGE_REF}"
    log ""
    log "Next: run a training via"
    log "  bash scripts/training/run_training.sh [--bind-local] [--image=...] -- <lerobot-train args>"
}

main

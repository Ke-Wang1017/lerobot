#!/usr/bin/env bash
# One-shot installer for the host-side prereqs of the LeRobot training-deploy
# prototype: Docker Engine + nvidia-container-toolkit. Tested on Ubuntu 22.04,
# 24.04, and the Nebius `ubuntu24.04-cuda13.0` image. Idempotent — re-running
# is safe.
#
# Run on the host:      sudo bash scripts/training/install_prereqs.sh
# Or from the GUI box:  ssh <host> 'sudo bash -s' < scripts/training/install_prereqs.sh
#
# After install:
#   - Log out and back in (so the docker group takes effect for your user), OR
#   - Run `newgrp docker` in your current shell.
#
# Non-interactive runs (over SSH, in CI, agent-driven): all gpg operations
# use `--batch --yes` and read keyrings from temp files, so we never block
# on /dev/tty. DEBIAN_FRONTEND=noninteractive suppresses debconf prompts.
#
# The whole body lives in main(){} and is invoked with stdin redirected to
# /dev/null. This is what makes the `ssh ... 'sudo bash -s' < script` form
# safe: `bash -s` reads the script from stdin INCREMENTALLY, so any child
# that reads stdin (apt/debconf/needrestart) would otherwise swallow the
# remaining script text and silently skip the rest of the install. The
# function is parsed in full before anything executes; children see EOF.

set -euo pipefail

export DEBIAN_FRONTEND=noninteractive

main() {

if [[ ${EUID} -ne 0 ]]; then
    echo "This script needs root. Re-run with: sudo bash $0" >&2
    exit 1
fi

TARGET_USER="${SUDO_USER:-$USER}"

log() { printf '\033[1;36m[install]\033[0m %s\n' "$*"; }

# Download a remote GPG key to a temp file then dearmor it locally with
# --batch --yes. The naïve `curl ... | gpg --dearmor` pattern fails over
# non-interactive SSH with "cannot open '/dev/tty'" — see the gap notes
# in scripts/training/README.md.
download_and_dearmor() {
    local url="$1"
    local dest="$2"
    local tmp
    tmp=$(mktemp)
    curl -fsSL "${url}" -o "${tmp}"
    gpg --batch --yes --dearmor -o "${dest}" "${tmp}"
    rm -f "${tmp}"
}

# ── Docker Engine ────────────────────────────────────────────────────────────

if command -v docker >/dev/null 2>&1; then
    log "Docker already installed: $(docker --version)"
else
    log "Installing Docker Engine via the official Docker apt repo..."

    apt-get update
    apt-get install -y ca-certificates curl gnupg

    install -m 0755 -d /etc/apt/keyrings
    download_and_dearmor \
        "https://download.docker.com/linux/ubuntu/gpg" \
        "/etc/apt/keyrings/docker.gpg"
    chmod a+r /etc/apt/keyrings/docker.gpg

    UBUNTU_CODENAME=$(. /etc/os-release && echo "${VERSION_CODENAME}")
    echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] \
https://download.docker.com/linux/ubuntu ${UBUNTU_CODENAME} stable" \
        > /etc/apt/sources.list.d/docker.list

    apt-get update
    apt-get install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin

    systemctl enable --now docker
    log "Docker installed: $(docker --version)"
fi

# Add target user to docker group so they don't need sudo for docker.
# Cloud-init's `sudo: ALL=(ALL) NOPASSWD:ALL` writes /etc/sudoers.d/ entries
# but does NOT change group membership — so on a fresh cloud VM, the user
# always needs to be explicitly added here.
if ! id -nG "${TARGET_USER}" | grep -qw docker; then
    usermod -aG docker "${TARGET_USER}"
    log "Added ${TARGET_USER} to the docker group."
    log "  → You must log out and back in (or run 'newgrp docker') for this to take effect."
else
    log "${TARGET_USER} is already in the docker group."
fi

# ── nvidia-container-toolkit (GPU pass-through) ──────────────────────────────

# Nebius's `ubuntu24.04-cuda13.0` image ships nvidia-container-toolkit as a
# held package — plain `apt-get install -y` errors out with "Held packages
# were changed and -y was used without --allow-change-held-packages." We
# unhold first, then upgrade with the explicit flag.

held_toolkit_packages() {
    apt-mark showhold 2>/dev/null | grep -E \
        '^(nvidia-container-toolkit|nvidia-container-toolkit-base|libnvidia-container-tools|libnvidia-container1)$' \
        || true
}

if dpkg -l nvidia-container-toolkit 2>/dev/null | grep -q '^ii' && \
   [[ -z $(held_toolkit_packages) ]]; then
    log "nvidia-container-toolkit already installed (and not held). Skipping."
else
    log "Installing / upgrading nvidia-container-toolkit..."

    # Unhold any held versions so apt can change them.
    held=$(held_toolkit_packages)
    if [[ -n "${held}" ]]; then
        log "Unholding pre-installed (held) toolkit packages: $(echo ${held} | tr '\n' ' ')"
        # shellcheck disable=SC2086
        apt-mark unhold ${held}
    fi

    download_and_dearmor \
        "https://nvidia.github.io/libnvidia-container/gpgkey" \
        "/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg"

    curl -fsSL https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list \
        | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' \
        > /etc/apt/sources.list.d/nvidia-container-toolkit.list

    apt-get update -qq
    # --allow-change-held-packages handles the case where we couldn't fully
    # clear the hold above (e.g., the package was held via a different
    # mechanism than apt-mark).
    apt-get install -y --allow-change-held-packages nvidia-container-toolkit

    nvidia-ctk runtime configure --runtime=docker
    systemctl restart docker
    log "nvidia-container-toolkit installed and Docker runtime configured."
fi

# ── Sanity check ─────────────────────────────────────────────────────────────

# Always do the cheap host-side check: is the GPU/driver visible at all?
log "Verifying host GPU (nvidia-smi)..."
if ! nvidia-smi -L >/dev/null 2>&1; then
    log "✗ nvidia-smi failed on the host — GPU/driver not available. Run"
    log "  'nvidia-smi' on the host to diagnose."
    exit 1
fi
log "✓ Host GPU visible."

# The container GPU smoke pulls a CUDA image and runs nvidia-smi inside it.
# Automated bringup skips it (LEROBOT_PREREQS_SKIP_CONTAINER_SMOKE=1): the
# real training run pulls the training image and runs --gpus all anyway, so
# a separate pull here is redundant. Manual setup runs keep it for assurance.
if [[ "${LEROBOT_PREREQS_SKIP_CONTAINER_SMOKE:-0}" == "1" ]]; then
    log "Skipping container GPU smoke (the training run is the end-to-end test)."
else
    log "Verifying GPU access from a container..."
    # Try as the target user first (uses their docker socket access via the
    # group we just added). If they aren't in the group yet (need re-login),
    # fall back to root just to verify the toolkit/driver pair is working.
    SUDO_RUN="sudo -u ${TARGET_USER}"
    if ! ${SUDO_RUN} docker info >/dev/null 2>&1; then
        log "  (target user can't reach docker yet — they need to log out/in;"
        log "   running the smoke check as root to verify the toolkit instead)"
        SUDO_RUN=""
    fi

    if ${SUDO_RUN} docker run --rm --gpus all \
            nvidia/cuda:12.4.1-base-ubuntu22.04 nvidia-smi --query-gpu=name \
            --format=csv,noheader >/dev/null 2>&1; then
        log "✓ GPU access from container works."
    else
        log "✗ Container could not access the GPU. Check the NVIDIA driver version"
        log "  (must be ≥ 525.60 for CUDA 12.4). Run 'nvidia-smi' on the host."
        exit 1
    fi
fi

log ""
log "Done. Next steps:"
log "  1. If you weren't already in the docker group, log out and back in now."
log "  2. Then run:  bash scripts/training/setup_host.sh"

}

# Stdin → /dev/null: children can't eat the script stream (see header).
main "$@" </dev/null

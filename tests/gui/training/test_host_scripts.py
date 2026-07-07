# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""Static guards for the host-setup shell scripts.

The probe dialog tells users to pipe install_prereqs.sh over ssh stdin
(``ssh <host> 'sudo bash -s' < script``). ``bash -s`` reads the script
from stdin INCREMENTALLY, so any child that reads stdin (apt/debconf/
needrestart) would swallow the remaining script text and silently skip
the rest of the install — demonstrated 2026-06-12: an unwrapped script
with a stdin-reading child skipped its tail and still exited 0.

The defense is structural: the whole body lives in ``main() {}`` (parsed
in full before anything executes) and is invoked with stdin redirected
to /dev/null. These tests pin that structure so a refactor can't quietly
reintroduce the hazard.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
INSTALL_PREREQS = REPO_ROOT / "scripts" / "training" / "install_prereqs.sh"


def test_install_prereqs_exists():
    assert INSTALL_PREREQS.is_file()


def test_body_is_wrapped_in_main():
    content = INSTALL_PREREQS.read_text()
    assert "main() {" in content, "body must live in main(){} so bash -s parses it fully before executing"


def test_main_invoked_with_stdin_devnull():
    content = INSTALL_PREREQS.read_text()
    assert 'main "$@" </dev/null' in content, (
        "main must run with stdin → /dev/null so apt/debconf children "
        "can't consume the script stream under `bash -s < script`"
    )


def test_apt_runs_noninteractive():
    content = INSTALL_PREREQS.read_text()
    assert "DEBIAN_FRONTEND=noninteractive" in content


def test_script_parses():
    r = subprocess.run(["bash", "-n", str(INSTALL_PREREQS)], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr


def test_stdin_pipe_form_reaches_main():
    """Run the script via the exact `bash -s < file` form (as non-root).
    The root-check refusal firing proves the full parse + main dispatch
    works through a pipe — the failure mode would be bash dying earlier
    or executing nothing."""
    with open(INSTALL_PREREQS) as f:
        r = subprocess.run(["bash", "-s"], stdin=f, capture_output=True, text=True)
    assert r.returncode == 1
    assert "needs root" in r.stderr

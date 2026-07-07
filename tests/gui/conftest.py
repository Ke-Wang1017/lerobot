# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""Shared fixtures for the GUI test suite.

The fake-training worker is a test fixture (production trains via docker +
``lerobot-train``). The autouse fixture points ``recipes.FAKE_RUNNER_PATH``
at it so ``__recipe__=__fake__`` runs can spawn it.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest

from lerobot.gui.training import recipes

FAKE_RUNNER_PATH = Path(__file__).parent / "training" / "fake_runner.py"


@pytest.fixture(autouse=True)
def _inject_fake_runner() -> Iterator[None]:
    assert FAKE_RUNNER_PATH.exists(), f"fake runner missing at {FAKE_RUNNER_PATH}"
    prev = recipes.FAKE_RUNNER_PATH
    recipes.FAKE_RUNNER_PATH = str(FAKE_RUNNER_PATH)
    try:
        yield
    finally:
        recipes.FAKE_RUNNER_PATH = prev

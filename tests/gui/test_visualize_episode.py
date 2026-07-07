# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Tests for /api/datasets/.../visualize.

The interesting test here is the *cross-module dry-run parse*:
``_build_visualize_cmd`` produces the argv we'll hand to
``lerobot.scripts.lerobot_dataset_viz``. We import that script's actual
argparse parser and feed our argv through it — same as `argparse` would
at runtime. If upstream renames a flag, removes one, or changes a flag
from value-taking to ``store_true`` (which is exactly what bit us last
time), this test fails immediately rather than at runtime when stderr
has been DEVNULL'd by Popen.
"""

import sys

import pytest

from lerobot.gui.api.datasets import _build_visualize_cmd


@pytest.fixture
def cmd() -> list[str]:
    return _build_visualize_cmd(repo_id="user/repo", episode_idx=7, root="/tmp/dataset")


class TestVisualizeCmdShape:
    """Pin the argv shape so future regressions surface here, not at runtime."""

    def test_uses_current_interpreter_via_module_invocation(self, cmd):
        """`sys.executable -m lerobot.scripts.lerobot_dataset_viz` — never the
        bare console script, since PATH lookup can land on a different env's
        binary with mismatched torch/torchcodec."""
        assert cmd[0] == sys.executable
        assert cmd[1] == "-m"
        assert cmd[2] == "lerobot.scripts.lerobot_dataset_viz"

    def test_does_not_pass_display_compressed_images(self, cmd):
        """Upstream changed this from a value-taking arg to a `store_true`
        flag. We omit it entirely — passing the old `--display-compressed-images
        False` form crashes argparse silently because "False" becomes an
        unexpected positional."""
        assert "--display-compressed-images" not in cmd

    def test_required_args_present(self, cmd):
        for required in ("--repo-id", "--episode-index", "--root"):
            assert required in cmd, f"missing {required} in {cmd}"


class TestVisualizeCmdParsesViaTargetScript:
    """The high-value defense: dry-run the argv through the *real* parser
    that lerobot-dataset-viz uses. This is what would have caught the
    `--display-compressed-images False` regression on the upstream merge,
    automatically."""

    def test_argv_parses_cleanly(self, cmd):
        """Import the target script's parser and parse our argv through it.
        argparse raises SystemExit on bad args — pytest will surface it."""
        from lerobot.scripts.lerobot_dataset_viz import _build_parser

        parser = _build_parser()
        # Skip [sys.executable, "-m", "lerobot.scripts.lerobot_dataset_viz"] —
        # that's how Python resolves the module, not argparse input.
        ns = parser.parse_args(cmd[3:])

        # Round-trip the values we set
        assert ns.repo_id == "user/repo"
        assert ns.episode_index == 7
        assert str(ns.root) == "/tmp/dataset"
        # Default for the flag we deliberately omit
        assert ns.display_compressed_images is False

    def test_synthetic_old_form_is_rejected(self):
        """Sanity: the old `--display-compressed-images False` form (which
        is what we used to send) really does break argparse. If this ever
        starts passing, upstream has reverted the flag and we should
        re-evaluate our omission."""
        from lerobot.scripts.lerobot_dataset_viz import _build_parser

        parser = _build_parser()
        with pytest.raises(SystemExit):
            parser.parse_args(
                [
                    "--repo-id",
                    "user/repo",
                    "--episode-index",
                    "0",
                    "--root",
                    "/tmp/x",
                    "--display-compressed-images",
                    "False",
                ]
            )

# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
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

"""Integration invariants pinned after merging upstream huggingface/lerobot.

Each test guards a seam where upstream changes and fork features meet — the
places a future merge is most likely to silently break again:

- the policy factory must expose BOTH upstream's policies (evo1, fastwam,
  lingbot_va, groot) and the fork's (act_vlm),
- the visualization dispatch (rerun/foxglove) must keep routing while the
  fork's rerun customizations (LEROBOT_RERUN_SERVE_PORT) survive in the
  relocated backend module,
- EEBoundsAndSafety must keep the fork's looser 0.2 m step default alongside
  upstream's raise_on_jump escape hatch,
- lerobot-dataset-viz's parser must keep the fork's optional --episode-index
  while exposing upstream's foxglove flags, and main() must only forward
  kwargs its visualize_dataset actually accepts,
- record_loop must accept the fork's interpolator AND upstream's display_mode,
- pyproject extras must keep the fork's (hvla/gui/mcp) and upstream's
  (lingbot_va/evo1/fastwam) wired into `all`.
"""

import inspect
from pathlib import Path

import pytest

PYPROJECT = Path(__file__).resolve().parent.parent / "pyproject.toml"


# ---------------------------------------------------------------------------
# Policy factory / draccus registry
# ---------------------------------------------------------------------------


def test_policy_registry_has_upstream_and_fork_policies():
    import lerobot.policies  # noqa: F401  (triggers config registration)
    from lerobot.configs.policies import PreTrainedConfig

    registered = set(PreTrainedConfig.get_known_choices())
    # Upstream policies added by the 2026-07 merge
    assert {"evo1", "fastwam", "lingbot_va", "groot"} <= registered
    # Pre-existing upstream policies that must not regress
    assert {"act", "diffusion", "smolvla", "pi0"} <= registered
    # Fork policy
    assert "act_vlm" in registered


def test_policy_factory_resolves_new_policy_classes():
    """get_policy_class must map every new name to a PreTrainedPolicy subclass.

    Uses lazy imports internally; skips a policy if its optional heavy deps
    aren't installed in this environment rather than failing the merge check.
    """
    from lerobot.policies.factory import get_policy_class
    from lerobot.policies.pretrained import PreTrainedPolicy

    for name in ("evo1", "fastwam", "lingbot_va", "groot"):
        try:
            cls = get_policy_class(name)
        except ImportError as e:
            pytest.skip(f"optional deps for {name} not installed: {e}")
        assert issubclass(cls, PreTrainedPolicy), name


# ---------------------------------------------------------------------------
# Visualization dispatch (upstream refactor) + fork rerun customizations
# ---------------------------------------------------------------------------


def test_visualization_dispatch_routes_by_display_mode(monkeypatch):
    import lerobot.utils.visualization_utils as vu

    calls = []
    monkeypatch.setattr(vu, "init_rerun", lambda **kw: calls.append(("rerun", kw)))
    monkeypatch.setattr(vu, "init_foxglove", lambda **kw: calls.append(("foxglove", kw)))
    monkeypatch.setattr(vu, "shutdown_rerun", lambda: calls.append(("rerun_down", {})))
    monkeypatch.setattr(vu, "shutdown_foxglove", lambda: calls.append(("foxglove_down", {})))

    vu.init_visualization("rerun", session_name="s", ip="1.2.3.4", port=9)
    vu.init_visualization("foxglove", ip=None, port=None)
    vu.shutdown_visualization("rerun")
    vu.shutdown_visualization("foxglove")

    assert [c[0] for c in calls] == ["rerun", "foxglove", "rerun_down", "foxglove_down"]
    # foxglove defaults to localhost when no ip is given
    assert calls[1][1]["host"] == "127.0.0.1"

    with pytest.raises(ValueError, match="display_mode"):
        vu.init_visualization("nope")
    with pytest.raises(ValueError, match="display_mode"):
        vu.log_visualization_data("nope")
    with pytest.raises(ValueError, match="display_mode"):
        vu.shutdown_visualization("nope")


def test_init_rerun_honors_serve_port_env(monkeypatch):
    """The fork's LEROBOT_RERUN_SERVE_PORT branch must survive in the
    relocated rerun backend: when set, init_rerun hosts a gRPC server instead
    of connecting out or spawning a viewer."""
    rr = pytest.importorskip("rerun")
    from lerobot.utils import rerun_visualization

    calls = []
    monkeypatch.setattr(rr, "init", lambda *a, **kw: calls.append("init"))
    monkeypatch.setattr(rr, "serve_grpc", lambda **kw: calls.append(("serve_grpc", kw)))
    monkeypatch.setattr(rr, "connect_grpc", lambda **kw: calls.append(("connect_grpc", kw)))
    monkeypatch.setattr(rr, "spawn", lambda **kw: calls.append(("spawn", kw)))
    monkeypatch.setenv("LEROBOT_RERUN_SERVE_PORT", "9199")

    rerun_visualization.init_rerun(session_name="merge-test")

    served = [c for c in calls if isinstance(c, tuple) and c[0] == "serve_grpc"]
    assert served and served[0][1]["grpc_port"] == 9199
    assert not any(isinstance(c, tuple) and c[0] in ("connect_grpc", "spawn") for c in calls)


def test_dispatcher_reexports_backend_symbols():
    """Legacy fork import paths went through visualization_utils; the
    dispatcher must keep re-exporting the rerun backend symbols."""
    from lerobot.utils import visualization_utils as vu

    for name in ("init_rerun", "log_rerun_data", "shutdown_rerun", "VISUALIZATION_MODES"):
        assert hasattr(vu, name), name
    assert vu.VISUALIZATION_MODES == ("rerun", "foxglove")


# ---------------------------------------------------------------------------
# EEBoundsAndSafety: fork default + upstream raise_on_jump
# ---------------------------------------------------------------------------


def _make_ee_action(x=0.0, y=0.0, z=0.0):
    return {"ee.x": x, "ee.y": y, "ee.z": z, "ee.wx": 0.0, "ee.wy": 0.0, "ee.wz": 0.0}


def test_ee_bounds_keeps_fork_default_and_upstream_raise_flag():
    from lerobot.robots.robot_kinematic_processor import EEBoundsAndSafety

    fields = {f.name: f for f in EEBoundsAndSafety.__dataclass_fields__.values()}
    assert fields["max_ee_step_m"].default == 0.2  # fork's deliberate loosening (upstream: 0.05)
    assert fields["raise_on_jump"].default is True  # upstream's new escape hatch, safe default

    bounds = {"min": [-1.0, -1.0, -1.0], "max": [1.0, 1.0, 1.0]}

    # Default: an over-limit jump aborts the loop.
    step = EEBoundsAndSafety(end_effector_bounds=bounds)
    step.action(_make_ee_action(0.0, 0.0, 0.0))
    with pytest.raises(ValueError, match="EE jump"):
        step.action(_make_ee_action(0.9, 0.0, 0.0))

    # raise_on_jump=False: the same jump is rate-limited to max_ee_step_m.
    step = EEBoundsAndSafety(end_effector_bounds=bounds, raise_on_jump=False)
    step.action(_make_ee_action(0.0, 0.0, 0.0))
    out = step.action(_make_ee_action(0.9, 0.0, 0.0))
    assert out["ee.x"] == pytest.approx(0.2)


# ---------------------------------------------------------------------------
# lerobot-dataset-viz: fork's optional --episode-index + upstream's foxglove
# ---------------------------------------------------------------------------


def test_dataset_viz_parser_contract():
    from lerobot.scripts.lerobot_dataset_viz import _build_parser

    parser = _build_parser()
    args = parser.parse_args(["--repo-id", "lerobot/pusht"])

    # Fork behavior: --episode-index optional -> all-episodes mode
    assert args.episode_index is None
    # Upstream foxglove flags
    assert args.display_mode == "rerun"
    assert args.host == "127.0.0.1"
    assert args.autoplay is True
    # Upstream removed --ws-port; it must not silently parse
    with pytest.raises(SystemExit):
        parser.parse_args(["--repo-id", "x", "--ws-port", "1"])


def test_dataset_viz_main_forwards_only_accepted_kwargs():
    """main() forwards vars(args) minus popped keys into visualize_dataset;
    every forwarded key must be a real parameter (a renamed flag would
    otherwise only fail at runtime deep in a viz session)."""
    from lerobot.scripts.lerobot_dataset_viz import _build_parser, visualize_dataset

    args = _build_parser().parse_args(["--repo-id", "lerobot/pusht"])
    forwarded = set(vars(args)) - {"repo_id", "root", "tolerance_s", "episode_index"}
    params = inspect.signature(visualize_dataset).parameters
    assert all(p in params for p in forwarded), forwarded - set(params)


# ---------------------------------------------------------------------------
# record/teleoperate loops: fork + upstream params coexist
# ---------------------------------------------------------------------------


def test_record_loop_accepts_fork_and_upstream_params():
    from lerobot.scripts.lerobot_record import record_loop

    # record_loop is wrapped by @safe_stop_image_writer, which doesn't use
    # functools.wraps — recover the undecorated function from the closure.
    inner = next(
        c.cell_contents for c in record_loop.__closure__ if callable(getattr(c, "cell_contents", None))
    )
    params = inspect.signature(inner).parameters
    # fork
    assert {"interpolator", "intervention_dataset", "latency_session", "episode_index"} <= set(params)
    # upstream
    assert "display_mode" in params
    assert params["display_mode"].default == "rerun"


def test_teleop_loop_accepts_fork_and_upstream_params():
    from lerobot.scripts.lerobot_teleoperate import TeleoperateConfig, teleop_loop

    params = inspect.signature(teleop_loop).parameters
    assert {"obs_stream_steps", "latency_session", "motion_logger"} <= set(params)  # fork
    assert "display_mode" in params  # upstream
    assert TeleoperateConfig.__dataclass_fields__["display_mode"].default == "rerun"


# ---------------------------------------------------------------------------
# pyproject extras: union of fork and upstream
# ---------------------------------------------------------------------------


def test_pyproject_extras_union():
    import tomllib

    with open(PYPROJECT, "rb") as f:
        extras = tomllib.load(f)["project"]["optional-dependencies"]

    # fork extras
    assert {"hvla", "gui", "mcp"} <= set(extras)
    # upstream extras from the 2026-07 merge
    assert {"lingbot_va", "evo1", "fastwam"} <= set(extras)

    all_extra = " ".join(extras["all"])
    for name in ("hvla", "gui", "mcp", "lingbot_va", "evo1", "fastwam", "groot"):
        assert f"lerobot[{name}]" in all_extra, name

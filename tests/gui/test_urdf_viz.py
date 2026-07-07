"""Tests for the URDF state visualization (``lerobot.gui.urdf_viz``).

Covers the correctness-critical pieces: the motor->URDF angle conversion,
robot resolution from an observation's motor set, the per-arm angle
computation, the ``/api/run/urdf-viz`` endpoint glue, and the integrity of
every vendored ``*_description`` package (URDF parses, declared joints
exist, referenced meshes are on disk).
"""

import asyncio
import importlib
import math
import xml.etree.ElementTree as ET
from unittest.mock import MagicMock, patch

import pytest

from lerobot.gui.urdf_viz import (
    JointCalibration,
    _discover_descriptions,
    compute_joint_angles,
    obs_to_urdf_rad,
    resolve_robot,
)

# Motor sets of the two vendored robots. SO-101's six motors are a subset of
# SO-107's seven (SO-107 adds forearm_roll) — which is exactly what the
# "most specific match wins" rule in resolve_robot must disambiguate.
SO101_MOTORS = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper")
SO107_MOTORS = (
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "forearm_roll",
    "wrist_flex",
    "wrist_roll",
    "gripper",
)


def _pos_keys(motors, prefix=""):
    """Observation ``<motor>.pos`` keys for the given motors, optionally prefixed."""
    return [f"{prefix}{m}.pos" for m in motors]


# ============================================================================
# obs_to_urdf_rad — the motor->URDF angle conversion
# ============================================================================


class TestObsToUrdfRad:
    """The layer-2 conversion ``urdf_rad = deg2rad(sign * pos + offset_deg)``."""

    def test_identity_calibration(self):
        assert obs_to_urdf_rad(90.0, JointCalibration()) == pytest.approx(math.radians(90.0))

    def test_zero_stays_zero(self):
        assert obs_to_urdf_rad(0.0, JointCalibration()) == 0.0

    def test_sign_flip(self):
        assert obs_to_urdf_rad(90.0, JointCalibration(sign=-1.0)) == pytest.approx(math.radians(-90.0))

    def test_offset_only(self):
        assert obs_to_urdf_rad(0.0, JointCalibration(offset_deg=-90.0)) == pytest.approx(math.radians(-90.0))

    def test_sign_and_offset_combined(self):
        # urdf_deg = -1 * 45 + 78.87 = 33.87
        got = obs_to_urdf_rad(45.0, JointCalibration(sign=-1.0, offset_deg=78.87))
        assert got == pytest.approx(math.radians(33.87))

    @pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
    def test_non_finite_input_raises(self, bad):
        with pytest.raises(AssertionError):
            obs_to_urdf_rad(bad, JointCalibration())


class TestJointCalibration:
    def test_defaults_are_identity(self):
        c = JointCalibration()
        assert c.sign == 1.0
        assert c.offset_deg == 0.0


# ============================================================================
# resolve_robot — matching a live robot to a description package
# ============================================================================


class TestResolveRobot:
    def test_resolves_so101_from_six_motors(self):
        spec = resolve_robot(_pos_keys(SO101_MOTORS))
        assert spec is not None
        assert spec.name == "SO-101"
        assert spec.urdf_url_path.startswith("so101_description/")
        assert len(spec.arms) == 1
        assert spec.arms[0].obs_prefix == ""

    def test_resolves_so107_from_seven_motors(self):
        spec = resolve_robot(_pos_keys(SO107_MOTORS))
        assert spec is not None
        assert spec.name == "SO-107"
        assert len(spec.arms) == 1

    def test_most_specific_match_wins(self):
        # The seven-motor set is a superset of SO-101's six; SO-107 must win.
        spec = resolve_robot(_pos_keys(SO107_MOTORS))
        assert spec is not None
        assert spec.name == "SO-107"

    def test_bimanual_detected_from_left_prefix(self):
        keys = _pos_keys(SO107_MOTORS, "left_") + _pos_keys(SO107_MOTORS, "right_")
        spec = resolve_robot(keys)
        assert spec is not None
        assert len(spec.arms) == 2
        assert {a.obs_prefix for a in spec.arms} == {"left_", "right_"}

    def test_extra_non_motor_keys_ignored(self):
        keys = _pos_keys(SO101_MOTORS) + ["observation.images.cam", "timestamp", "task"]
        spec = resolve_robot(keys)
        assert spec is not None
        assert spec.name == "SO-101"

    def test_no_pos_keys_returns_none(self):
        assert resolve_robot(["observation.images.cam", "task"]) is None
        assert resolve_robot([]) is None

    def test_unknown_motor_set_returns_none(self):
        assert resolve_robot(["mystery_a.pos", "mystery_b.pos"]) is None

    def test_partial_motor_set_returns_none(self):
        # A strict subset of a known robot's motors is not enough to match it.
        assert resolve_robot(_pos_keys(SO101_MOTORS[:3])) is None


# ============================================================================
# compute_joint_angles — per-arm URDF angles from an observation
# ============================================================================


class TestComputeJointAngles:
    def test_so101_identity_mapping(self):
        spec = resolve_robot(_pos_keys(SO101_MOTORS))
        obs = {f"{m}.pos": 10.0 for m in SO101_MOTORS}
        angles = compute_joint_angles(spec, obs)
        assert set(angles.keys()) == {""}
        # SO-101's URDF joint names equal its motor names; alignment is identity.
        for m in SO101_MOTORS:
            assert angles[""][m] == pytest.approx(math.radians(10.0))

    def test_so107_alignment_is_applied(self):
        spec = resolve_robot(_pos_keys(SO107_MOTORS))
        obs = {f"{m}.pos": 0.0 for m in SO107_MOTORS}
        joints = compute_joint_angles(spec, obs)[""]
        # Unimanual SO-107 reuses the right-arm alignment; shoulder_lift there
        # is (sign +1, offset -90), so pos 0 -> -90 deg. Its URDF joint is S2.
        assert joints["S2"] == pytest.approx(math.radians(-90.0))

    def test_partial_observation_omits_missing_joint(self):
        spec = resolve_robot(_pos_keys(SO101_MOTORS))
        obs = {f"{m}.pos": 5.0 for m in SO101_MOTORS if m != "gripper"}
        joints = compute_joint_angles(spec, obs)[""]
        assert "gripper" not in joints
        assert "shoulder_pan" in joints

    def test_bimanual_arms_use_distinct_alignments(self):
        keys = _pos_keys(SO107_MOTORS, "left_") + _pos_keys(SO107_MOTORS, "right_")
        spec = resolve_robot(keys)
        obs = dict.fromkeys(keys, 0.0)
        angles = compute_joint_angles(spec, obs)
        assert set(angles.keys()) == {"left_", "right_"}
        # gripper (URDF joint S7): left is (+1, -90), right is (-1, 0).
        assert angles["left_"]["S7"] == pytest.approx(math.radians(-90.0))
        assert angles["right_"]["S7"] == pytest.approx(math.radians(0.0))


# ============================================================================
# /api/run/urdf-viz {meta, source} endpoints
# ============================================================================


def _call_meta():
    from lerobot.gui.api.run import urdf_viz_meta

    return asyncio.run(urdf_viz_meta())


def _call_source(source: str = "state"):
    from lerobot.gui.api.run import urdf_viz_source

    return asyncio.run(urdf_viz_source(source))


class TestUrdfVizMeta:
    def test_unavailable_when_no_reader(self):
        with patch("lerobot.gui.api.run._get_obs_reader", return_value=None):
            assert _call_meta() == {"available": False}

    def test_unavailable_when_no_observation(self):
        reader = MagicMock()
        reader.read_obs.return_value = []
        with patch("lerobot.gui.api.run._get_obs_reader", return_value=reader):
            assert _call_meta() == {"available": False}

    def test_unavailable_for_unknown_robot(self):
        reader = MagicMock()
        reader.read_obs.return_value = [{"mystery_a.pos": 1.0, "mystery_b.pos": 2.0}]
        with patch("lerobot.gui.api.run._get_obs_reader", return_value=reader):
            assert _call_meta() == {"available": False}

    def test_advertises_state_only_when_no_action_stream(self):
        reader = MagicMock()
        reader.read_obs.return_value = ({f"{m}.pos": 0.0 for m in SO107_MOTORS}, 0.0)
        reader.read_action.return_value = None
        with patch("lerobot.gui.api.run._get_obs_reader", return_value=reader):
            result = _call_meta()
        assert result["available"] is True
        assert result["name"] == "SO-107"
        assert result["urdf"].startswith("/urdf-assets/so107_description/")
        assert result["bimanual"] is False
        # Only state is advertised — frontend will hide the overlay toggle.
        assert result["sources"] == ["state"]
        # ee_link is surfaced so the frontend can FK overlay frames into
        # a polyline (the L6 anchor for SO-107 — gripper-independent).
        assert result["ee_link"] == "L7_1"

    def test_advertises_action_when_reader_supplies_it(self):
        reader = MagicMock()
        reader.read_obs.return_value = ({f"{m}.pos": 0.0 for m in SO107_MOTORS}, 0.0)
        reader.read_action.return_value = ({f"{m}.pos": 10.0 for m in SO107_MOTORS}, 0.0)
        with patch("lerobot.gui.api.run._get_obs_reader", return_value=reader):
            result = _call_meta()
        assert result["sources"] == ["state", "action"]


class TestUrdfVizSource:
    def test_unavailable_when_no_reader(self):
        with patch("lerobot.gui.api.run._get_obs_reader", return_value=None):
            assert _call_source("state") == {"available": False}

    def test_state_returns_unified_frames_shape(self):
        reader = MagicMock()
        reader.read_obs.return_value = ({f"{m}.pos": 0.0 for m in SO107_MOTORS}, 0.0)
        reader.read_action.return_value = None
        with patch("lerobot.gui.api.run._get_obs_reader", return_value=reader):
            result = _call_source("state")
        assert result["available"] is True
        assert len(result["arms"]) == 1
        arm = result["arms"][0]
        # Always a frames[] — len 1 for poses, len N for future chunks. The
        # renderer's contract is "look at frames[0]" today.
        assert "frames" in arm
        assert len(arm["frames"]) == 1
        assert arm["frames"][0]["joints"]["S2"] == pytest.approx(math.radians(-90.0))

    def test_action_returns_distinct_pose_from_state(self):
        reader = MagicMock()
        reader.read_obs.return_value = ({f"{m}.pos": 0.0 for m in SO107_MOTORS}, 0.0)
        reader.read_action.return_value = ({f"{m}.pos": 10.0 for m in SO107_MOTORS}, 0.0)
        with patch("lerobot.gui.api.run._get_obs_reader", return_value=reader):
            state = _call_source("state")
            action = _call_source("action")
        # shoulder_lift carries an offset of -90, so pos=0 -> -90 deg (state)
        # and pos=10 -> -80 deg (action). Each source returns its own data.
        assert state["arms"][0]["frames"][0]["joints"]["S2"] == pytest.approx(math.radians(-90.0))
        assert action["arms"][0]["frames"][0]["joints"]["S2"] == pytest.approx(math.radians(-80.0))

    def test_action_unavailable_when_reader_has_no_action(self):
        reader = MagicMock()
        reader.read_obs.return_value = ({f"{m}.pos": 0.0 for m in SO107_MOTORS}, 0.0)
        reader.read_action.return_value = None
        with patch("lerobot.gui.api.run._get_obs_reader", return_value=reader):
            assert _call_source("action") == {"available": False}

    def test_unknown_source_raises_400(self):
        from fastapi import HTTPException

        reader = MagicMock()
        reader.read_obs.return_value = ({f"{m}.pos": 0.0 for m in SO107_MOTORS}, 0.0)
        reader.read_action.return_value = None
        with (
            patch("lerobot.gui.api.run._get_obs_reader", return_value=reader),
            pytest.raises(HTTPException) as exc,
        ):
            _call_source("policy_chunk")
        assert exc.value.status_code == 400


# ============================================================================
# /api/datasets/.../urdf-viz endpoint — frame-driven sibling of the live one
# ============================================================================


def _make_so107_bimanual_dataset(ep_length: int = 50):
    """Mock the bits of LeRobotDataset the urdf-viz endpoint reads."""
    import numpy as np

    state_names = [f"left_{m}.pos" for m in SO107_MOTORS] + [f"right_{m}.pos" for m in SO107_MOTORS]
    ds = MagicMock()
    ds.repo_id = "test/so107"
    ds.root = "/fake/path"
    ds.fps = 30
    ds.meta.total_episodes = 1
    ds.meta.total_frames = ep_length
    ds.meta.episodes = [{"length": ep_length, "data/chunk_index": 0, "data/file_index": 0}]
    # Identical names for state and action — that's the LeRobot convention
    # for joint-space records (both flow through MOTOR_NAMES on the robot).
    ds.meta.features = {
        "observation.state": {"dtype": "float32", "shape": [14], "names": state_names},
        "action": {"dtype": "float32", "shape": [14], "names": state_names},
    }
    # Per-frame vectors: state at zero, action +10 deg on every motor — the
    # endpoint should round-trip those into distinct URDF joint angles after
    # the alignment is applied.
    ds._state_vec = np.zeros(14, dtype=np.float32)
    ds._action_vec = np.full(14, 10.0, dtype=np.float32)
    return ds


def _call_dataset_meta(dataset_id: str, episode_idx: int):
    from lerobot.gui.api.datasets import get_urdf_viz_dataset_meta

    return asyncio.run(get_urdf_viz_dataset_meta(dataset_id, episode_idx))


def _call_dataset_source(
    dataset_id: str, episode_idx: int, frame: int, source: str = "state", horizon: int = 1
):
    from lerobot.gui.api.datasets import get_urdf_viz_dataset_source

    return asyncio.run(get_urdf_viz_dataset_source(dataset_id, episode_idx, frame, source, horizon))


class TestUrdfVizDatasetEndpoints:
    @pytest.fixture
    def app_with_so107(self):
        import pandas as pd

        from lerobot.gui.api import datasets as datasets_module
        from lerobot.gui.frame_cache import FrameCache
        from lerobot.gui.state import AppState

        state = AppState(frame_cache=FrameCache(max_bytes=1_000_000))
        original = datasets_module._app_state
        datasets_module.set_app_state(state)
        ds = _make_so107_bimanual_dataset(ep_length=50)
        state.datasets["test/so107"] = ds

        # Each parquet row holds the full 14-wide state and action vectors;
        # the endpoint reads one column at the requested frame.
        df = pd.DataFrame(
            {
                "episode_index": [0] * 50,
                "observation.state": [ds._state_vec] * 50,
                "action": [ds._action_vec] * 50,
            }
        )
        with (
            patch("lerobot.gui.api.datasets.pd.read_parquet", return_value=df),
            patch("pathlib.Path.exists", return_value=True),
        ):
            yield ds

        datasets_module._app_state = original

    def test_meta_404_for_unknown_dataset(self, app_with_so107):
        from fastapi import HTTPException

        with pytest.raises(HTTPException) as exc:
            _call_dataset_meta("nope/missing", 0)
        assert exc.value.status_code == 404

    def test_meta_advertises_both_sources_when_action_feature_present(self, app_with_so107):
        result = _call_dataset_meta("test/so107", 0)
        assert result["available"] is True
        assert result["name"] == "SO-107"
        assert result["bimanual"] is True
        assert result["sources"] == ["state", "action"]
        # Same ee_link as the live endpoint: frontend uses it for FK in
        # the trajectory polyline.
        assert result["ee_link"] == "L7_1"

    def test_meta_advertises_state_only_when_action_feature_absent(self, app_with_so107):
        app_with_so107.meta.features.pop("action")
        result = _call_dataset_meta("test/so107", 0)
        assert result["available"] is True
        assert result["sources"] == ["state"]

    def test_meta_unavailable_without_observation_state(self, app_with_so107):
        app_with_so107.meta.features.pop("observation.state")
        assert _call_dataset_meta("test/so107", 0) == {"available": False}

    def test_source_state_returns_unified_frames_shape(self, app_with_so107):
        result = _call_dataset_source("test/so107", 0, 0, "state")
        assert result["available"] is True
        assert len(result["arms"]) == 2
        # left shoulder_lift alignment is (+1, -90): state pos 0 -> -90 deg.
        left = next(a for a in result["arms"] if a["prefix"] == "left_")
        assert len(left["frames"]) == 1
        assert left["frames"][0]["joints"]["S2"] == pytest.approx(math.radians(-90.0))

    def test_source_action_distinct_from_state(self, app_with_so107):
        state = _call_dataset_source("test/so107", 0, 0, "state")
        action = _call_dataset_source("test/so107", 0, 0, "action")
        left_state = next(a for a in state["arms"] if a["prefix"] == "left_")
        left_action = next(a for a in action["arms"] if a["prefix"] == "left_")
        assert left_state["frames"][0]["joints"]["S2"] == pytest.approx(math.radians(-90.0))
        # pos=10 with offset -90 -> -80 deg
        assert left_action["frames"][0]["joints"]["S2"] == pytest.approx(math.radians(-80.0))

    def test_source_action_unavailable_when_dataset_has_no_action_feature(self, app_with_so107):
        app_with_so107.meta.features.pop("action")
        assert _call_dataset_source("test/so107", 0, 0, "action") == {"available": False}

    def test_source_unknown_raises_400(self, app_with_so107):
        from fastapi import HTTPException

        with pytest.raises(HTTPException) as exc:
            _call_dataset_source("test/so107", 0, 0, "policy_chunk")
        assert exc.value.status_code == 400

    def test_source_horizon_returns_n_consecutive_frames(self, app_with_so107):
        # horizon=10 from frame 0 of a 50-frame episode -> exactly 10 frames.
        result = _call_dataset_source("test/so107", 0, 0, "action", horizon=10)
        assert result["available"] is True
        for arm in result["arms"]:
            assert len(arm["frames"]) == 10
            for f in arm["frames"]:
                assert "joints" in f

    def test_source_horizon_clips_at_episode_end(self, app_with_so107):
        # horizon=20 starting at frame 45 of a 50-frame episode -> clip to 5.
        # Frontend has to be OK with shorter-than-requested trajectories.
        result = _call_dataset_source("test/so107", 0, 45, "action", horizon=20)
        assert result["available"] is True
        for arm in result["arms"]:
            assert len(arm["frames"]) == 5

    def test_source_horizon_below_one_raises_400(self, app_with_so107):
        from fastapi import HTTPException

        with pytest.raises(HTTPException) as exc:
            _call_dataset_source("test/so107", 0, 0, "action", horizon=0)
        assert exc.value.status_code == 400


# ============================================================================
# Vendored *_description packages — URDF + mesh integrity
# ============================================================================

_DESCRIPTIONS = _discover_descriptions()


def test_vendored_descriptions_are_discovered():
    names = {pkg for pkg, _ in _DESCRIPTIONS}
    assert "so101_description" in names
    assert "so107_description" in names


def _urdf_path(pkg_name):
    return importlib.import_module(f"lerobot.robots.{pkg_name}").get_urdf_path()


@pytest.mark.parametrize(
    ("pkg_name", "viz_spec"),
    _DESCRIPTIONS,
    ids=[d[0] for d in _DESCRIPTIONS],
)
class TestDescriptionAssets:
    """Integrity checks run against every discovered ``*_description`` package."""

    def test_viz_spec_shape(self, pkg_name, viz_spec):
        assert isinstance(viz_spec["name"], str) and viz_spec["name"]
        assert len(viz_spec["motors"]) == len(viz_spec["urdf_joints"]), (
            "motors and urdf_joints must be parallel sequences"
        )
        assert len(set(viz_spec["motors"])) == len(viz_spec["motors"]), "duplicate motor name"
        assert len(set(viz_spec["urdf_joints"])) == len(viz_spec["urdf_joints"]), "duplicate URDF joint"

    def test_urdf_exists_and_parses(self, pkg_name, viz_spec):
        urdf = _urdf_path(pkg_name)
        assert urdf.name == viz_spec["urdf_file"], "get_urdf_path disagrees with VIZ_SPEC['urdf_file']"
        ET.parse(urdf)  # raises ParseError on malformed XML

    def test_declared_joints_exist_in_urdf(self, pkg_name, viz_spec):
        urdf = _urdf_path(pkg_name)
        joint_names = {j.get("name") for j in ET.parse(urdf).getroot().iter("joint")}
        for jn in viz_spec["urdf_joints"]:
            assert jn in joint_names, f"VIZ_SPEC joint {jn!r} is absent from {urdf.name}"

    def test_referenced_meshes_exist(self, pkg_name, viz_spec):
        urdf = _urdf_path(pkg_name)
        mesh_files = [m.get("filename") for m in ET.parse(urdf).getroot().iter("mesh")]
        assert mesh_files, "URDF references no meshes"
        for fn in mesh_files:
            resolved = (urdf.parent / fn).resolve()
            assert resolved.exists(), f"mesh {fn!r} referenced by {urdf.name} is missing at {resolved}"

    def test_resolve_robot_round_trips(self, pkg_name, viz_spec):
        """A robot exposing exactly this description's motors resolves back to it."""
        spec = resolve_robot([f"{m}.pos" for m in viz_spec["motors"]])
        assert spec is not None
        assert spec.name == viz_spec["name"]

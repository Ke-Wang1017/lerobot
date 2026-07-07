"""Tests for Run tab API: profile→CLI conversion and endpoint arg assembly."""

import asyncio
import json
from unittest.mock import AsyncMock, patch

import pytest
from fastapi import HTTPException

from lerobot.gui.api.run import (
    RecordRequest,
    ReplayRequest,
    TeleoperateRequest,
    _ensure_no_active_process,
    _profile_to_cli_args,
    start_record,
    start_replay,
    start_teleoperate,
)

# ============================================================================
# _profile_to_cli_args
# ============================================================================


class TestProfileToCliArgs:
    """Tests for converting profile dicts to draccus CLI arguments."""

    def test_basic_type(self):
        profile = {"type": "bi_so107_follower"}
        args = _profile_to_cli_args(profile, "robot")
        assert args == ["--robot.type=bi_so107_follower"]

    def test_string_fields(self):
        profile = {"type": "koch_follower", "fields": {"port": "/dev/ttyACM0"}}
        args = _profile_to_cli_args(profile, "robot")
        assert "--robot.type=koch_follower" in args
        assert "--robot.port=/dev/ttyACM0" in args

    def test_bool_fields_lowercase(self):
        profile = {"type": "t", "fields": {"gripper_bounce": True, "enabled": False}}
        args = _profile_to_cli_args(profile, "robot")
        assert "--robot.gripper_bounce=true" in args
        assert "--robot.enabled=false" in args

    def test_none_fields_skipped(self):
        profile = {"type": "t", "fields": {"port": None, "baudrate": 115200}}
        args = _profile_to_cli_args(profile, "robot")
        assert len(args) == 2  # type + baudrate
        assert not any("port" in a for a in args)

    def test_numeric_fields(self):
        profile = {"type": "t", "fields": {"baudrate": 115200, "timeout": 1.5}}
        args = _profile_to_cli_args(profile, "robot")
        assert "--robot.baudrate=115200" in args
        assert "--robot.timeout=1.5" in args

    def test_cameras_json(self):
        cameras = {"cam_0": {"type": "opencv", "index": 0}}
        profile = {"type": "t", "cameras": cameras}
        args = _profile_to_cli_args(profile, "robot")
        cam_arg = next(a for a in args if "cameras" in a)
        assert cam_arg == f"--robot.cameras={json.dumps(cameras)}"

    def test_empty_cameras_omitted(self):
        profile = {"type": "t", "cameras": {}}
        args = _profile_to_cli_args(profile, "robot")
        assert not any("cameras" in a for a in args)

    def test_no_cameras_key(self):
        profile = {"type": "t"}
        args = _profile_to_cli_args(profile, "robot")
        assert not any("cameras" in a for a in args)

    def test_no_fields_key(self):
        profile = {"type": "t"}
        args = _profile_to_cli_args(profile, "robot")
        assert args == ["--robot.type=t"]

    def test_teleop_prefix(self):
        profile = {"type": "so_leader", "fields": {"port": "/dev/ttyACM1"}}
        args = _profile_to_cli_args(profile, "teleop")
        assert "--teleop.type=so_leader" in args
        assert "--teleop.port=/dev/ttyACM1" in args

    def test_multiple_fields_ordering(self):
        profile = {"type": "t", "fields": {"a": 1, "b": 2, "c": 3}}
        args = _profile_to_cli_args(profile, "robot")
        # Type always first
        assert args[0] == "--robot.type=t"
        # All fields present
        assert len(args) == 4

    def test_mixed_field_types(self):
        profile = {
            "type": "bi_so107_follower",
            "fields": {
                "left_arm_port": "/dev/ttyACM0",
                "right_arm_port": "/dev/ttyACM1",
                "left_arm_disable_torque_on_disconnect": True,
                "unused": None,
            },
            "cameras": {"cam_0": {"type": "opencv", "index": 0}},
        }
        args = _profile_to_cli_args(profile, "robot")
        assert "--robot.type=bi_so107_follower" in args
        assert "--robot.left_arm_port=/dev/ttyACM0" in args
        assert "--robot.right_arm_port=/dev/ttyACM1" in args
        assert "--robot.left_arm_disable_torque_on_disconnect=true" in args
        assert not any("unused" in a for a in args)
        assert any("cameras" in a for a in args)

    def test_unknown_fields_filtered_with_warning(self):
        """Fields not in the config class are silently skipped."""
        profile = {
            "type": "bi_so107_follower",
            "fields": {
                "left_arm_port": "/dev/ttyACM0",
                "this_field_will_never_exist_xyz": True,
            },
        }
        args = _profile_to_cli_args(profile, "robot")
        assert "--robot.left_arm_port=/dev/ttyACM0" in args
        assert not any("this_field_will_never_exist_xyz" in a for a in args)

    def test_include_cameras_false(self):
        cameras = {"cam_0": {"type": "opencv", "index": 0}}
        profile = {"type": "t", "fields": {"port": "/dev/ttyACM0"}, "cameras": cameras}
        args = _profile_to_cli_args(profile, "robot", include_cameras=False)
        assert "--robot.port=/dev/ttyACM0" in args
        assert not any("cameras" in a for a in args)

    def test_calibration_dir_round_trips_through_cli(self):
        """Regression for commit 1b49664ba.

        ``calibration_dir`` lives in ``_SKIP_FIELDS`` on the backend schema
        (it's not rendered as a normal form input in the GUI), but the
        profile JSON on disk still carries it and the launcher MUST pass
        it through as a CLI arg when present in the profile data —
        otherwise ``Robot.__init__`` falls back to the wrong default path
        and the calibration JSON isn't found, triggering a recalibration
        prompt that stalls the GUI subprocess silently.

        The dropping bug was on the frontend (``_collectFormFields``
        rebuilt the fields dict from the schema only, erasing any
        non-schema keys before sending to ``/api/run/teleoperate``).
        Fixed by having the frontend preserve loaded-data fields. This
        test pins the backend's contract: given a profile that contains
        ``calibration_dir``, the launcher emits it.
        """
        profile = {
            "type": "bi_so107_follower_predictive",
            "fields": {
                "id": "white",
                "left_arm_port": "/dev/ttyACM0",
                "right_arm_port": "/dev/ttyACM2",
                "calibration_dir": "/home/test/cal/so107_follower",
            },
        }
        args = _profile_to_cli_args(profile, "robot")
        assert "--robot.calibration_dir=/home/test/cal/so107_follower" in args


# ============================================================================
# _ensure_no_active_process
# ============================================================================


class TestEnsureNoActiveProcess:
    """Tests for the active process guard."""

    def test_no_process_passes(self):
        with patch("lerobot.gui.api.run._active_process", None):
            _ensure_no_active_process()  # Should not raise

    def test_exited_process_passes(self):
        mock_proc = AsyncMock()
        mock_proc.returncode = 0  # Process has exited
        mock_proc.pid = 123
        with patch("lerobot.gui.api.run._active_process", mock_proc):
            _ensure_no_active_process()  # Should not raise

    def test_running_process_raises_409(self):
        mock_proc = AsyncMock()
        mock_proc.returncode = None  # Still running
        mock_proc.pid = 456
        with (
            patch("lerobot.gui.api.run._active_process", mock_proc),
            patch("lerobot.gui.api.run._active_command", "teleoperate"),
        ):
            with pytest.raises(HTTPException) as exc_info:
                _ensure_no_active_process()
            assert exc_info.value.status_code == 409


# ============================================================================
# Endpoint arg assembly
# ============================================================================

_ROBOT = {
    "type": "so107_follower",
    "fields": {"port": "/dev/ttyACM0"},
    "cameras": {"cam_0": {"type": "opencv", "index_or_path": "/dev/video0"}},
}
_TELEOP = {"type": "so107_leader", "fields": {"port": "/dev/ttyACM1"}}


def _make_fake_launch(captured_args):
    """Create a fake _launch_subprocess that captures args and sets _active_process."""
    import lerobot.gui.api.run as run_module

    async def fake_launch(args, command, config, extra_env=None):
        captured_args.extend(args)
        # The endpoint reads _active_process.pid after launch, so set a mock
        mock_proc = AsyncMock()
        mock_proc.pid = 9999
        run_module._active_process = mock_proc

    return fake_launch


class TestTeleoperateEndpoint:
    """Tests for the teleoperate endpoint's CLI arg assembly."""

    def test_teleoperate_args(self):
        captured_args = []

        async def run():
            req = TeleoperateRequest(robot=_ROBOT, teleop=_TELEOP, fps=45)
            with (
                patch("lerobot.gui.api.run._active_process", None),
                patch("lerobot.gui.api.run._launch_subprocess", _make_fake_launch(captured_args)),
            ):
                await start_teleoperate(req)

        asyncio.run(run())

        assert captured_args[0] == "lerobot-teleoperate"
        assert "--robot.type=so107_follower" in captured_args
        assert "--robot.port=/dev/ttyACM0" in captured_args
        assert "--teleop.type=so107_leader" in captured_args
        assert "--teleop.port=/dev/ttyACM1" in captured_args
        assert "--fps=45" in captured_args


class TestRecordEndpoint:
    """Tests for the record endpoint's CLI arg assembly."""

    def test_record_args(self):
        captured_args = []

        async def run():
            req = RecordRequest(
                robot=_ROBOT,
                teleop=_TELEOP,
                repo_id="user/my_dataset",
                single_task="pick up the cube",
                fps=30,
                episode_time_s=120,
                reset_time_s=30,
                num_episodes=10,
                video=True,
                vcodec="libsvtav1",
                play_sounds=False,
                resume=False,
            )
            with (
                patch("lerobot.gui.api.run._active_process", None),
                patch("lerobot.gui.api.run._launch_subprocess", _make_fake_launch(captured_args)),
            ):
                await start_record(req)

        asyncio.run(run())

        assert captured_args[0] == "lerobot-record"
        assert "--robot.type=so107_follower" in captured_args
        assert "--teleop.type=so107_leader" in captured_args
        assert "--dataset.repo_id=user/my_dataset" in captured_args
        assert "--dataset.single_task=pick up the cube" in captured_args
        assert "--dataset.fps=30" in captured_args
        assert "--dataset.episode_time_s=120.0" in captured_args
        assert "--dataset.reset_time_s=30.0" in captured_args
        assert "--dataset.num_episodes=10" in captured_args
        assert "--dataset.video=true" in captured_args
        assert "--dataset.push_to_hub=false" in captured_args
        assert "--dataset.vcodec=libsvtav1" in captured_args
        assert "--play_sounds=false" in captured_args
        # resume=False should not add --resume
        assert "--resume=true" not in captured_args
        # No root provided — should not have --dataset.root
        assert not any("--dataset.root" in a for a in captured_args)

    def test_record_with_root(self):
        captured_args = []

        async def run():
            req = RecordRequest(
                robot=_ROBOT,
                teleop=_TELEOP,
                repo_id="user/my_dataset",
                root="/home/user/.cache/huggingface/lerobot/user/my_dataset",
                single_task="pick up the cube",
                resume=True,
            )
            with (
                patch("lerobot.gui.api.run._active_process", None),
                patch("lerobot.gui.api.run._launch_subprocess", _make_fake_launch(captured_args)),
            ):
                await start_record(req)

        asyncio.run(run())

        assert "--dataset.root=/home/user/.cache/huggingface/lerobot/user/my_dataset" in captured_args
        assert "--resume=true" in captured_args

    def test_record_resume_flag(self):
        captured_args = []

        async def run():
            req = RecordRequest(
                robot=_ROBOT,
                teleop=_TELEOP,
                repo_id="user/ds",
                single_task="task",
                resume=True,
            )
            with (
                patch("lerobot.gui.api.run._active_process", None),
                patch("lerobot.gui.api.run._launch_subprocess", _make_fake_launch(captured_args)),
            ):
                await start_record(req)

        asyncio.run(run())

        assert "--resume=true" in captured_args

    def test_record_video_false(self):
        captured_args = []

        async def run():
            req = RecordRequest(
                robot=_ROBOT,
                teleop=_TELEOP,
                repo_id="user/ds",
                single_task="task",
                video=False,
            )
            with (
                patch("lerobot.gui.api.run._active_process", None),
                patch("lerobot.gui.api.run._launch_subprocess", _make_fake_launch(captured_args)),
            ):
                await start_record(req)

        asyncio.run(run())

        assert "--dataset.video=false" in captured_args

    def test_record_with_policy_path(self):
        captured_args = []

        async def run():
            req = RecordRequest(
                robot=_ROBOT,
                teleop=_TELEOP,
                repo_id="eval/eval_my_policy",
                single_task="pick up socket",
                policy_path="/home/user/outputs/act/checkpoints/last/pretrained_model",
            )
            with (
                patch("lerobot.gui.api.run._active_process", None),
                patch("lerobot.gui.api.run._launch_subprocess", _make_fake_launch(captured_args)),
            ):
                await start_record(req)

        asyncio.run(run())

        assert "--policy.path=/home/user/outputs/act/checkpoints/last/pretrained_model" in captured_args
        assert "--teleop.type=so107_leader" in captured_args

    def test_record_policy_only_no_teleop(self):
        captured_args = []

        async def run():
            req = RecordRequest(
                robot=_ROBOT,
                teleop=None,
                repo_id="eval/eval_my_policy",
                single_task="pick up socket",
                policy_path="/home/user/outputs/act/checkpoints/last/pretrained_model",
            )
            with (
                patch("lerobot.gui.api.run._active_process", None),
                patch("lerobot.gui.api.run._launch_subprocess", _make_fake_launch(captured_args)),
            ):
                await start_record(req)

        asyncio.run(run())

        assert "--policy.path=/home/user/outputs/act/checkpoints/last/pretrained_model" in captured_args
        assert not any("--teleop." in a for a in captured_args)

    def test_record_no_teleop_no_policy_raises(self):
        async def run():
            req = RecordRequest(
                robot=_ROBOT,
                teleop=None,
                repo_id="eval/eval_my_policy",
                single_task="task",
            )
            with (
                patch("lerobot.gui.api.run._active_process", None),
            ):
                await start_record(req)

        with pytest.raises(HTTPException) as exc_info:
            asyncio.run(run())
        assert exc_info.value.status_code == 400


class TestUnifiedFps:
    """The GUI uses a single FPS field for both teleop-only and record modes.

    Verify that the same fps value routes to --fps (teleoperate) or
    --dataset.fps (record) depending on the endpoint.
    """

    def test_teleop_uses_fps_flag(self):
        captured_args = []

        async def run():
            req = TeleoperateRequest(robot=_ROBOT, teleop=_TELEOP, fps=25)
            with (
                patch("lerobot.gui.api.run._active_process", None),
                patch("lerobot.gui.api.run._launch_subprocess", _make_fake_launch(captured_args)),
            ):
                await start_teleoperate(req)

        asyncio.run(run())

        assert "--fps=25" in captured_args
        assert not any("--dataset.fps" in a for a in captured_args)

    def test_record_uses_dataset_fps_flag(self):
        captured_args = []

        async def run():
            req = RecordRequest(
                robot=_ROBOT,
                teleop=_TELEOP,
                repo_id="user/ds",
                single_task="task",
                fps=25,
            )
            with (
                patch("lerobot.gui.api.run._active_process", None),
                patch("lerobot.gui.api.run._launch_subprocess", _make_fake_launch(captured_args)),
            ):
                await start_record(req)

        asyncio.run(run())

        assert "--dataset.fps=25" in captured_args
        assert not any(a == "--fps=25" for a in captured_args)


class TestReplayEndpoint:
    """Tests for the replay endpoint's CLI arg assembly."""

    def test_replay_args(self):
        captured_args = []

        async def run():
            req = ReplayRequest(
                robot=_ROBOT,
                repo_id="user/my_dataset",
                episode=5,
            )
            with (
                patch("lerobot.gui.api.run._active_process", None),
                patch("lerobot.gui.api.run._launch_subprocess", _make_fake_launch(captured_args)),
            ):
                await start_replay(req)

        asyncio.run(run())

        assert captured_args[0] == "lerobot-replay"
        assert "--robot.type=so107_follower" in captured_args
        assert "--robot.port=/dev/ttyACM0" in captured_args
        assert "--dataset.repo_id=user/my_dataset" in captured_args
        assert "--dataset.episode=5" in captured_args
        # Replay must NOT pass --dataset.fps: the upstream lerobot-replay loop
        # paces by `dataset.fps` (read from the loaded dataset) regardless,
        # so a CLI-supplied value would be misleading dead config.
        assert not any(a.startswith("--dataset.fps") for a in captured_args)
        # Replay should not have teleop args
        assert not any("teleop" in a for a in captured_args)
        # Replay now includes cameras (for obs-stream live viewer)
        assert any("cameras" in a for a in captured_args)
        # No root provided
        assert not any("--dataset.root" in a for a in captured_args)

    def test_replay_with_root(self):
        captured_args = []

        async def run():
            req = ReplayRequest(
                robot=_ROBOT,
                repo_id="user/my_dataset",
                root="/tmp/datasets/user/my_dataset",
                episode=3,
            )
            with (
                patch("lerobot.gui.api.run._active_process", None),
                patch("lerobot.gui.api.run._launch_subprocess", _make_fake_launch(captured_args)),
            ):
                await start_replay(req)

        asyncio.run(run())

        assert "--dataset.root=/tmp/datasets/user/my_dataset" in captured_args
        assert "--dataset.repo_id=user/my_dataset" in captured_args
        assert "--dataset.episode=3" in captured_args


# ============================================================================
# Stale obs reader cleanup on launch
# ============================================================================


class TestObsReaderCleanupOnLaunch:
    """_launch_subprocess must close the obs reader so the GUI doesn't serve
    stale frames from a previous session's shared memory segments."""

    def test_launch_clears_obs_reader(self):
        """Launching any workflow closes existing obs reader."""
        import lerobot.gui.api.run as run_module

        close_called = False
        original_close = run_module._close_obs_reader

        def tracking_close():
            nonlocal close_called
            close_called = True
            original_close()

        async def run():
            req = TeleoperateRequest(robot=_ROBOT, teleop=_TELEOP, fps=30)
            with (
                patch("lerobot.gui.api.run._active_process", None),
                patch("lerobot.gui.api.run._close_obs_reader", tracking_close),
                patch(
                    "lerobot.gui.api.run._launch_subprocess",
                    wraps=run_module._launch_subprocess,
                ),
                patch("asyncio.create_subprocess_exec") as mock_exec,
            ):
                mock_proc = AsyncMock()
                mock_proc.pid = 9999
                # readline() must return EOF (empty bytes) so the
                # _read_stream tasks exit cleanly. Without this, the
                # tasks loop indefinitely on AsyncMock return values
                # and leak `Task exception was never retrieved` warnings
                # at event-loop teardown (rstrip on a coroutine).
                mock_proc.stdout = AsyncMock()
                mock_proc.stdout.readline = AsyncMock(return_value=b"")
                mock_proc.stderr = AsyncMock()
                mock_proc.stderr.readline = AsyncMock(return_value=b"")
                mock_proc.wait = AsyncMock(return_value=0)
                mock_exec.return_value = mock_proc
                await start_teleoperate(req)

        asyncio.run(run())
        assert close_called, "_close_obs_reader must be called during launch"

    def test_inode_based_reader_reattach(self):
        """_get_obs_reader detects inode change and re-attaches."""
        import lerobot.gui.api.run as run_module

        # Simulate: reader attached with inode 1000, then stream recreated with inode 2000
        mock_reader = AsyncMock()
        mock_reader.close = lambda: None

        run_module._obs_reader = mock_reader
        run_module._obs_reader_meta_ino = 1000

        with (
            patch("os.stat") as mock_stat,
            patch(
                "lerobot.robots.obs_stream.ObservationStreamReader",
            ) as mock_reader_cls,
        ):
            # Current inode is different — stream was recreated
            mock_stat.return_value.st_ino = 2000
            new_reader = AsyncMock()
            new_reader.obs_scalar_keys = ["j.pos"]
            new_reader.image_keys = {}
            mock_reader_cls.return_value = new_reader

            result = run_module._get_obs_reader()

            assert result is new_reader, "Should return newly created reader"
            assert run_module._obs_reader is new_reader
            assert run_module._obs_reader_meta_ino == 2000

        # Cleanup
        run_module._obs_reader = None
        run_module._obs_reader_meta_ino = None

    def test_same_inode_keeps_reader(self):
        """_get_obs_reader returns cached reader when inode hasn't changed."""
        import lerobot.gui.api.run as run_module

        mock_reader = AsyncMock()
        run_module._obs_reader = mock_reader
        run_module._obs_reader_meta_ino = 1000

        with patch("os.stat") as mock_stat:
            mock_stat.return_value.st_ino = 1000  # Same inode
            result = run_module._get_obs_reader()
            assert result is mock_reader, "Should return cached reader"

        # Cleanup
        run_module._obs_reader = None
        run_module._obs_reader_meta_ino = None

    def test_missing_shm_clears_reader(self):
        """_get_obs_reader clears reader if /dev/shm file disappears."""
        import lerobot.gui.api.run as run_module

        mock_reader = AsyncMock()
        mock_reader.close = lambda: None
        run_module._obs_reader = mock_reader
        run_module._obs_reader_meta_ino = 1000

        with patch("os.stat", side_effect=FileNotFoundError):
            result = run_module._get_obs_reader()
            assert result is None
            assert run_module._obs_reader is None

        # Cleanup
        run_module._obs_reader_meta_ino = None


# ============================================================================
# GET /api/run/rlt-config — slider seed endpoint
# ============================================================================


class TestGetRltConfig:
    """The GUI sliders seed from this endpoint on init so they reflect what
    the training process actually has applied, not HTML hardcoded defaults.
    """

    @pytest.fixture
    def reset_active(self):
        import lerobot.gui.api.run as run_module

        prev = run_module._active_config
        run_module._active_config = None
        yield run_module
        run_module._active_config = prev

    def test_no_active_session_returns_defaults(self, reset_active):
        """Before any RLT session starts, endpoint returns RLTConfig defaults.
        Without this the GUI would crash trying to read undefined state."""
        from lerobot.gui.api.run import get_rlt_config
        from lerobot.policies.hvla.rlt.config import RLTConfig

        cfg = asyncio.run(get_rlt_config())
        defaults = RLTConfig()
        assert cfg["beta"] == defaults.beta
        assert cfg["exploration_sigma"] == defaults.exploration_sigma
        assert cfg["target_sigma"] == defaults.target_sigma
        assert cfg["dump_chunks"] is False

    def test_reads_current_override_values(self, reset_active, tmp_path):
        """The whole point: the value on disk must round-trip to the GUI.
        Before this endpoint existed, the slider hardcoded 0.02 regardless
        of what the training process was using (the bug we're fixing)."""
        from lerobot.gui.api.run import get_rlt_config

        (tmp_path / "rlt_overrides.json").write_text(
            json.dumps({"beta": 0.5, "exploration_sigma": 0.0, "dump_chunks": True})
        )
        reset_active._active_config = {"rlt_output_dir": str(tmp_path)}
        cfg = asyncio.run(get_rlt_config())
        assert cfg["beta"] == 0.5
        assert cfg["exploration_sigma"] == 0.0
        assert cfg["dump_chunks"] is True

    def test_legacy_actor_sigma_key_is_accepted(self, reset_active, tmp_path):
        """Older GUI builds wrote ``actor_sigma``. Accept as synonym for
        ``exploration_sigma`` so mid-session upgrades don't drop the value."""
        from lerobot.gui.api.run import get_rlt_config

        (tmp_path / "rlt_overrides.json").write_text(json.dumps({"beta": 0.5, "actor_sigma": 0.0}))
        reset_active._active_config = {"rlt_output_dir": str(tmp_path)}
        cfg = asyncio.run(get_rlt_config())
        assert cfg["exploration_sigma"] == 0.0

    def test_partial_override_fills_missing_keys_from_defaults(self, reset_active, tmp_path):
        """Old override files may predate some keys. Missing keys fall back
        to config defaults so the GUI always has a value to show."""
        from lerobot.gui.api.run import get_rlt_config
        from lerobot.policies.hvla.rlt.config import RLTConfig

        (tmp_path / "rlt_overrides.json").write_text(json.dumps({"beta": 0.3}))
        reset_active._active_config = {"rlt_output_dir": str(tmp_path)}
        cfg = asyncio.run(get_rlt_config())
        assert cfg["beta"] == 0.3
        assert cfg["exploration_sigma"] == RLTConfig().exploration_sigma
        assert cfg["dump_chunks"] is False

    def test_missing_file_returns_defaults(self, reset_active, tmp_path):
        """Fresh RLT session hasn't written the file yet — don't 404."""
        from lerobot.gui.api.run import get_rlt_config
        from lerobot.policies.hvla.rlt.config import RLTConfig

        reset_active._active_config = {"rlt_output_dir": str(tmp_path)}
        cfg = asyncio.run(get_rlt_config())
        assert cfg["beta"] == RLTConfig().beta

    def test_malformed_file_returns_defaults(self, reset_active, tmp_path):
        """Never break the GUI over a corrupt override file."""
        from lerobot.gui.api.run import get_rlt_config
        from lerobot.policies.hvla.rlt.config import RLTConfig

        (tmp_path / "rlt_overrides.json").write_text("not json {{{")
        reset_active._active_config = {"rlt_output_dir": str(tmp_path)}
        cfg = asyncio.run(get_rlt_config())
        assert cfg["beta"] == RLTConfig().beta


# ============================================================================
# RLT mode requires the RL Token Encoder. Caught at the API boundary so a
# missing field never makes it to the subprocess (where it would crash deep
# inside actor.pt state_dict load with a size mismatch). See incident
# 9bf49910f / launch attempt of rlt_online_v2_widened on 2026-04-27.
# ============================================================================


class TestHvlaRltTokenRequired:
    """rlt_mode=True without rlt_token_checkpoint must be rejected at the
    API boundary with a helpful 400, not silently passed through."""

    def _make_request_kwargs(self, **overrides):
        """Minimum HVLARunRequest kwargs to exercise the validation path.
        Real launches need more; the rejection fires before any of it
        matters."""
        base = {
            "robot": _ROBOT,
            "s1_checkpoint": "/tmp/fake_s1",
            "task": "assemble cylinder into ring",
            "fps": 30,
            "rlt_mode": True,
        }
        base.update(overrides)
        return base

    def test_missing_token_checkpoint_raises_400(self):
        from lerobot.gui.api.run import HVLARunRequest, start_hvla

        req = HVLARunRequest(**self._make_request_kwargs())
        with patch("lerobot.gui.api.run._active_process", None), pytest.raises(HTTPException) as excinfo:
            asyncio.run(start_hvla(req))
        assert excinfo.value.status_code == 400
        # Must be actionable — name the field and the recipe
        msg = str(excinfo.value.detail)
        assert "rlt_token_checkpoint" in msg
        assert "RL Token Encoder" in msg

    def test_blank_token_checkpoint_raises_400(self):
        """Empty string must be treated the same as None — JS may send
        '' from a never-filled field."""
        from lerobot.gui.api.run import HVLARunRequest, start_hvla

        req = HVLARunRequest(
            **self._make_request_kwargs(
                rlt_token_checkpoint="   "  # whitespace only
            )
        )
        with patch("lerobot.gui.api.run._active_process", None), pytest.raises(HTTPException) as excinfo:
            asyncio.run(start_hvla(req))
        assert excinfo.value.status_code == 400

    def test_rlt_disabled_does_not_require_token(self):
        """When rlt_mode=False, the token field is irrelevant — must
        not be enforced (regression guard so the validation doesn't
        leak into normal HVLA runs)."""
        from lerobot.gui.api.run import HVLARunRequest, start_hvla

        captured_args = []
        req = HVLARunRequest(
            **self._make_request_kwargs(
                rlt_mode=False,
                rlt_token_checkpoint=None,
            )
        )
        with (
            patch("lerobot.gui.api.run._active_process", None),
            patch("lerobot.gui.api.run._launch_subprocess", _make_fake_launch(captured_args)),
        ):
            asyncio.run(start_hvla(req))
        # Reached the launch path, no rejection
        assert any("--rlt-mode" not in a for a in captured_args)
        assert "--rlt-mode" not in captured_args


class TestTimeoutExcsRegression:
    """Regression guard for the asyncio.TimeoutError vs builtin TimeoutError
    Python-3.10 mismatch.

    On Python 3.10 (the lerobot conda env's actual interpreter, despite
    pyproject's ``requires-python = ">=3.12"``) the two classes are
    *distinct* and not even in the same hierarchy. Pyupgrade and ruff
    UP041 rewrite any literal ``asyncio.TimeoutError`` token in an except
    clause to ``TimeoutError`` under py311+ targets, which silently
    breaks the SSE keepalive on the actual runtime — every 2s wait_for
    timeout escapes and crashes the response generator.

    These tests fail loudly if anyone simplifies _TIMEOUT_EXCS away."""

    def test_constant_includes_asyncio_timeout(self):
        from lerobot.gui.api.run import _TIMEOUT_EXCS

        assert TimeoutError in _TIMEOUT_EXCS
        assert asyncio.TimeoutError in _TIMEOUT_EXCS

    def test_catches_what_wait_for_raises(self):
        """End-to-end: asyncio.wait_for timing out is caught by the tuple."""
        from lerobot.gui.api.run import _TIMEOUT_EXCS

        async def go():
            try:
                await asyncio.wait_for(asyncio.sleep(10), timeout=0.01)
            except _TIMEOUT_EXCS:
                return "caught"
            return "missed"

        assert asyncio.run(go()) == "caught"


# ============================================================================
# /latency-metrics + /latency-sources — multi-source snapshot routing
# ============================================================================


class TestLatencyMetricsSources:
    """The /latency-metrics endpoint reads ``latency_snapshot.json`` from a
    directory chosen by the ``source`` query param. The default ``teleop``
    source preserves backwards-compatible behaviour for existing callers."""

    def test_default_source_is_teleop(self, tmp_path, monkeypatch):
        """Calling without ?source=... reads the teleop snapshot."""
        from lerobot.gui.api import run as run_module

        # Point teleop at a temp dir we control, drop a fake snapshot in it.
        teleop_dir = tmp_path / "teleop"
        teleop_dir.mkdir()
        (teleop_dir / "latency_snapshot.json").write_text(
            json.dumps({"loop_kind": "teleop", "n_records": 42, "stages": {}})
        )
        monkeypatch.setitem(run_module.LATENCY_SOURCES, "teleop", str(teleop_dir))

        result = asyncio.run(run_module.get_latency_metrics())
        assert result["loop_kind"] == "teleop"
        assert result["n_records"] == 42

    def test_explicit_source_routes_correctly(self, tmp_path, monkeypatch):
        from lerobot.gui.api import run as run_module

        hvla_dir = tmp_path / "hvla"
        hvla_dir.mkdir()
        (hvla_dir / "latency_snapshot.json").write_text(
            json.dumps({"loop_kind": "hvla_infer", "n_records": 7, "stages": {}})
        )
        monkeypatch.setitem(run_module.LATENCY_SOURCES, "hvla_infer", str(hvla_dir))

        result = asyncio.run(run_module.get_latency_metrics(source="hvla_infer"))
        assert result["loop_kind"] == "hvla_infer"
        assert result["n_records"] == 7

    def test_unknown_source_returns_empty_stub(self):
        """Unknown sources don't 404 — the dashboard's polling loop
        shouldn't have to special-case errors."""
        from lerobot.gui.api import run as run_module

        result = asyncio.run(run_module.get_latency_metrics(source="does_not_exist"))
        assert result == {
            "n_records": 0,
            "dropped_records": 0,
            "overrun_ratio": 0.0,
            "stages": {},
            "series": {},
        }

    def test_missing_snapshot_returns_empty_stub(self, tmp_path, monkeypatch):
        """Source is registered but no snapshot file exists yet (fresh
        session). Should return the empty stub, not raise."""
        from lerobot.gui.api import run as run_module

        empty_dir = tmp_path / "empty"
        empty_dir.mkdir()
        monkeypatch.setitem(run_module.LATENCY_SOURCES, "teleop", str(empty_dir))

        result = asyncio.run(run_module.get_latency_metrics(source="teleop"))
        assert result["n_records"] == 0


class TestLatencySourcesListing:
    """``/latency-sources`` enumerates which snapshot files actually exist
    and whether they're recent. The dashboard polls this to decide which
    loops to render."""

    def test_lists_only_existing_snapshots(self, tmp_path, monkeypatch):
        """Non-existent snapshot files should still appear in the list with
        ``fresh=False``, so the dashboard can render a placeholder for
        known-but-currently-quiet loops."""
        from lerobot.gui.api import run as run_module

        teleop_dir = tmp_path / "teleop"
        teleop_dir.mkdir()
        (teleop_dir / "latency_snapshot.json").write_text(
            json.dumps({"loop_kind": "teleop", "n_records": 1, "stages": {}})
        )
        monkeypatch.setattr(
            run_module,
            "LATENCY_SOURCES",
            {"teleop": str(teleop_dir), "hvla_infer": str(tmp_path / "no_such_dir")},
        )

        result = asyncio.run(run_module.list_latency_sources())
        sources = {s["key"]: s for s in result["sources"]}
        assert sources["teleop"]["loop_kind"] == "teleop"
        assert sources["teleop"]["fresh"] is True
        assert sources["hvla_infer"]["loop_kind"] is None
        assert sources["hvla_infer"]["fresh"] is False
        assert sources["hvla_infer"]["age_s"] is None

    def test_default_registry_lists_hvla_main_and_inference(self):
        """The shipped LATENCY_SOURCES must include both HVLA tracks so the
        dashboard can render them stacked. Subdirs (not different
        filenames) under outputs/hvla_runs/ keep the snapshot writer
        simple."""
        from lerobot.gui.api import run as run_module

        assert "hvla_main" in run_module.LATENCY_SOURCES
        assert "hvla_infer" in run_module.LATENCY_SOURCES
        assert run_module.LATENCY_SOURCES["hvla_main"].endswith("/main")
        assert run_module.LATENCY_SOURCES["hvla_infer"].endswith("/inference")

    def test_lists_include_process_and_track_when_present(self, tmp_path, monkeypatch):
        """Multi-thread loops publish ``process`` / ``track`` in the
        snapshot envelope so the dashboard can group sources by process.
        Single-track loops omit them; the listing reflects what's there."""
        from lerobot.gui.api import run as run_module

        main_dir = tmp_path / "main"
        main_dir.mkdir()
        (main_dir / "latency_snapshot.json").write_text(
            json.dumps({"loop_kind": "hvla_main", "process": "hvla", "track": "main", "stages": {}})
        )
        infer_dir = tmp_path / "inference"
        infer_dir.mkdir()
        (infer_dir / "latency_snapshot.json").write_text(
            json.dumps({"loop_kind": "hvla_infer", "process": "hvla", "track": "inference", "stages": {}})
        )
        monkeypatch.setattr(
            run_module,
            "LATENCY_SOURCES",
            {"hvla_main": str(main_dir), "hvla_infer": str(infer_dir)},
        )

        result = asyncio.run(run_module.list_latency_sources())
        sources = {s["key"]: s for s in result["sources"]}
        assert sources["hvla_main"]["loop_kind"] == "hvla_main"
        assert sources["hvla_main"]["fresh"] is True
        assert sources["hvla_infer"]["loop_kind"] == "hvla_infer"
        assert sources["hvla_infer"]["fresh"] is True

    def test_stale_snapshot_marked_not_fresh(self, tmp_path, monkeypatch):
        """A snapshot whose mtime is more than 5s old is reported with
        ``fresh=False`` — the loop has likely stopped publishing."""
        import os

        from lerobot.gui.api import run as run_module

        teleop_dir = tmp_path / "teleop"
        teleop_dir.mkdir()
        snap_path = teleop_dir / "latency_snapshot.json"
        snap_path.write_text(json.dumps({"loop_kind": "teleop", "stages": {}}))
        old_mtime = snap_path.stat().st_mtime - 60
        os.utime(snap_path, (old_mtime, old_mtime))
        monkeypatch.setattr(run_module, "LATENCY_SOURCES", {"teleop": str(teleop_dir)})

        result = asyncio.run(run_module.list_latency_sources())
        teleop = result["sources"][0]
        assert teleop["fresh"] is False
        assert teleop["age_s"] >= 60.0


# ============================================================================
# Parent-death detector (PR_SET_PDEATHSIG)
# ============================================================================


class TestParentDeathDetector:
    """The GUI launches subprocesses with preexec_fn=_set_pdeathsig_preexec
    so a SIGKILL on the GUI server doesn't leave teleop/record processes
    running indefinitely. Regression for the orphan-subprocess scenario
    documented in gui/TODO.md."""

    def test_launch_subprocess_passes_preexec(self):
        """The teleop launch path must pass our pdeathsig preexec_fn."""
        import lerobot.gui.api.run as run_module

        async def run():
            req = TeleoperateRequest(robot=_ROBOT, teleop=_TELEOP, fps=30)
            with (
                patch("lerobot.gui.api.run._active_process", None),
                patch("asyncio.create_subprocess_exec") as mock_exec,
            ):
                mock_proc = AsyncMock()
                mock_proc.pid = 9999
                mock_proc.stdout = AsyncMock()
                mock_proc.stdout.readline = AsyncMock(return_value=b"")
                mock_proc.stderr = AsyncMock()
                mock_proc.stderr.readline = AsyncMock(return_value=b"")
                mock_proc.wait = AsyncMock(return_value=0)
                mock_exec.return_value = mock_proc
                await start_teleoperate(req)
                return mock_exec

        mock_exec = asyncio.run(run())
        assert mock_exec.called
        kwargs = mock_exec.call_args.kwargs
        assert "preexec_fn" in kwargs, (
            "create_subprocess_exec must receive preexec_fn for the parent-death detector"
        )
        assert kwargs["preexec_fn"] is run_module._set_pdeathsig_preexec

    def test_set_pdeathsig_preexec_is_callable_on_linux(self):
        """The preexec function itself must run without raising on Linux.

        It's called in the forked child between fork and exec; any
        exception there would prevent the subprocess from starting at all.
        """
        import sys

        from lerobot.gui.api.run import _set_pdeathsig_preexec

        # On Linux this hits libc.prctl; on other platforms it's a no-op.
        # Either way it must not raise.
        _set_pdeathsig_preexec()
        if sys.platform != "linux":
            pytest.skip("PR_SET_PDEATHSIG integration test only runs on Linux")

    @pytest.mark.skipif(
        __import__("sys").platform != "linux",
        reason="PR_SET_PDEATHSIG is Linux-only",
    )
    def test_child_dies_when_grandparent_killed(self, tmp_path):
        """End-to-end: spawn a 'fake GUI' subprocess that itself spawns a
        sleeping child with our preexec; kill the fake GUI; confirm the
        child died. Proves the kernel-level mechanism actually fires."""
        import os
        import signal
        import subprocess
        import sys
        import textwrap
        import time

        pidfile = tmp_path / "child.pid"

        # The fake-GUI script: spawn a sleep-60 subprocess with our
        # preexec_fn, write its PID to a file, then sleep forever.
        fake_gui_src = textwrap.dedent(f"""
            import subprocess, time, signal, ctypes

            def preexec():
                libc = ctypes.CDLL("libc.so.6", use_errno=True)
                libc.prctl(1, signal.SIGTERM, 0, 0, 0)

            child = subprocess.Popen(
                ["sleep", "60"],
                preexec_fn=preexec,
            )
            with open({str(pidfile)!r}, "w") as f:
                f.write(str(child.pid))
            time.sleep(60)
        """)

        fake_gui = subprocess.Popen([sys.executable, "-c", fake_gui_src])
        try:
            # Wait for the child to be spawned and pid recorded.
            for _ in range(50):
                if pidfile.exists():
                    break
                time.sleep(0.05)
            assert pidfile.exists(), "fake GUI never reported child pid"
            child_pid = int(pidfile.read_text().strip())

            # Sanity check: child is alive.
            os.kill(child_pid, 0)  # no-op, just probes existence

            # Now kill the fake GUI with SIGKILL — no chance for clean shutdown.
            fake_gui.kill()
            fake_gui.wait(timeout=5.0)

            # PR_SET_PDEATHSIG should have sent SIGTERM to the child the
            # moment its parent (the fake GUI) died. Within a short window
            # the child must no longer exist.
            for _ in range(50):
                try:
                    os.kill(child_pid, 0)
                except ProcessLookupError:
                    return  # Child is gone — test passes
                time.sleep(0.05)
            # Reaching here means the child outlived its parent.
            try:
                os.kill(child_pid, signal.SIGKILL)  # cleanup
            except ProcessLookupError:
                pass
            pytest.fail(
                f"Child pid {child_pid} survived after grandparent died — PR_SET_PDEATHSIG did not fire."
            )
        finally:
            if fake_gui.poll() is None:
                fake_gui.kill()
                fake_gui.wait(timeout=2.0)


# ============================================================================
# Launch-lock serialization (TOCTOU on _active_process)
# ============================================================================


class TestLaunchLockSerializes:
    """Regression for the TOCTOU race documented in gui/TODO.md:
    `_ensure_no_active_process()` is synchronous, then `await _launch_subprocess(...)`
    is the first await — without a lock, a second request arriving during the
    fork sees `_active_process is None`, passes the check, and overwrites the
    in-flight launch — orphaning the first subprocess holding cameras /
    serial. `_launch_lock` makes the check+launch atomic."""

    def test_concurrent_launches_serialize(self):
        """Second concurrent /teleoperate must wait on _launch_lock until the
        first finishes, then observe _active_process and get 409."""
        import asyncio as _asyncio

        import lerobot.gui.api.run as run_module

        first_started = _asyncio.Event()
        release_first = _asyncio.Event()

        async def slow_launch(args, command, config, extra_env=None):
            # Mimic _launch_subprocess: set _active_process, then yield to the
            # event loop and hold until the test releases. Without _launch_lock,
            # the second start_teleoperate could observe _active_process is
            # still None (or already set) inconsistently. With the lock, the
            # second call blocks until this returns.
            mock_proc = AsyncMock()
            mock_proc.pid = 9999
            mock_proc.returncode = None
            run_module._active_process = mock_proc
            run_module._active_command = command
            first_started.set()
            await release_first.wait()

        async def run():
            req1 = TeleoperateRequest(robot=_ROBOT, teleop=_TELEOP, fps=30)
            req2 = TeleoperateRequest(robot=_ROBOT, teleop=_TELEOP, fps=30)
            with (
                patch("lerobot.gui.api.run._active_process", None),
                patch("lerobot.gui.api.run._active_command", None),
                patch("lerobot.gui.api.run._launch_subprocess", slow_launch),
            ):
                t1 = _asyncio.create_task(start_teleoperate(req1))
                await first_started.wait()

                t2 = _asyncio.create_task(start_teleoperate(req2))
                # Give t2 a slice of time to attempt the lock.
                await _asyncio.sleep(0.05)
                assert not t2.done(), "Second launch ran without waiting for the first — _launch_lock missing"

                release_first.set()
                await t1
                with pytest.raises(HTTPException) as exc_info:
                    await t2
                assert exc_info.value.status_code == 409
            # Restore module state for other tests.
            run_module._active_process = None
            run_module._active_command = None

        asyncio.run(run())

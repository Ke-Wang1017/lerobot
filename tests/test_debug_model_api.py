"""Tests for debug model load/unload API guards.

Covers:
  - Cannot load when already loaded (409)
  - Cannot double-unload (returns not_loaded)
  - Status reflects actual state
  - Lock prevents concurrent load/unload
"""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


@pytest.fixture(autouse=True)
def _reset_state():
    """Reset debug model state between tests."""
    import lerobot.gui.api.run as run_mod

    run_mod._debug_process = None
    # Reset the lock (may be held from a failed test)
    run_mod._debug_lock = asyncio.Lock()
    yield
    run_mod._debug_process = None


def _make_fake_process(alive=True):
    proc = MagicMock()
    proc.pid = 12345
    proc.returncode = None if alive else 0
    proc.terminate = MagicMock()
    proc.kill = MagicMock()
    proc.wait = AsyncMock(return_value=0)
    return proc


def _run(coro):
    """Run an async function synchronously."""
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


class TestLoadGuards:
    def test_load_when_already_loaded_returns_409(self):
        import lerobot.gui.api.run as run_mod

        run_mod._debug_process = _make_fake_process(alive=True)

        from fastapi import HTTPException

        from lerobot.gui.api.run import DebugModelConfig, load_debug_model

        config = DebugModelConfig(checkpoint="/fake/path", policy_type="hvla_s2_vlm", task="test")
        with pytest.raises(HTTPException) as exc_info:
            _run(load_debug_model(config))
        assert exc_info.value.status_code == 409

    def test_load_unsupported_type_returns_400(self):
        from fastapi import HTTPException

        from lerobot.gui.api.run import DebugModelConfig, load_debug_model

        config = DebugModelConfig(checkpoint="/fake/path", policy_type="unknown_type", task="test")
        with pytest.raises(HTTPException) as exc_info:
            _run(load_debug_model(config))
        assert exc_info.value.status_code == 400


class TestUnloadGuards:
    def test_unload_when_not_loaded(self):
        from lerobot.gui.api.run import unload_debug_model

        result = _run(unload_debug_model())
        assert result["status"] == "not_loaded"

    def test_unload_already_exited_process(self):
        import lerobot.gui.api.run as run_mod

        run_mod._debug_process = _make_fake_process(alive=False)

        from lerobot.gui.api.run import unload_debug_model

        result = _run(unload_debug_model())
        assert result["status"] == "not_loaded"

    def test_unload_running_process(self):
        import lerobot.gui.api.run as run_mod

        run_mod._debug_process = _make_fake_process(alive=True)

        from lerobot.gui.api.run import unload_debug_model

        with patch.object(run_mod, "_stop_debug_process", new_callable=AsyncMock) as mock_stop:

            async def _do_stop():
                run_mod._debug_process = None

            mock_stop.side_effect = _do_stop
            result = _run(unload_debug_model())
        assert result["status"] == "unloaded"
        mock_stop.assert_called_once()


class TestStatus:
    def test_status_not_loaded(self):
        from lerobot.gui.api.run import debug_model_status

        result = _run(debug_model_status())
        assert result["loaded"] is False
        assert result["pid"] is None

    def test_status_loaded(self):
        import lerobot.gui.api.run as run_mod

        run_mod._debug_process = _make_fake_process(alive=True)

        from lerobot.gui.api.run import debug_model_status

        result = _run(debug_model_status())
        assert result["loaded"] is True
        assert result["pid"] == 12345

    def test_status_after_process_exits(self):
        import lerobot.gui.api.run as run_mod

        run_mod._debug_process = _make_fake_process(alive=False)

        from lerobot.gui.api.run import debug_model_status

        result = _run(debug_model_status())
        assert result["loaded"] is False


class TestConcurrency:
    def test_concurrent_loads_one_succeeds_one_409(self):
        """Two simultaneous loads — first succeeds, second gets 409."""
        from fastapi import HTTPException

        import lerobot.gui.api.run as run_mod
        from lerobot.gui.api.run import DebugModelConfig, load_debug_model

        config = DebugModelConfig(checkpoint="/fake/path", policy_type="hvla_s2_vlm", task="test")

        load_count = 0

        async def _fake_launch(cfg):
            nonlocal load_count
            load_count += 1
            await asyncio.sleep(0.1)
            run_mod._debug_process = _make_fake_process(alive=True)

        results = []

        async def _test():
            with patch.object(run_mod, "_launch_debug_s2", side_effect=_fake_launch):

                async def try_load():
                    try:
                        r = await load_debug_model(config)
                        results.append(("ok", r))
                    except HTTPException as e:
                        results.append(("error", e.status_code))

                await asyncio.gather(try_load(), try_load())

        _run(_test())

        statuses = sorted(r[0] for r in results)
        assert statuses == ["error", "ok"], f"Expected one ok + one error, got {results}"
        assert load_count == 1

"""Tests for :func:`lerobot.utils.utils.setup_run_logging`."""

from __future__ import annotations

import logging
import re
import sys
import threading
from pathlib import Path

import pytest

from lerobot.utils.utils import _install_exception_logging, setup_run_logging


@pytest.fixture
def _restore_logging_and_hooks():
    """Save and restore globals that ``setup_run_logging`` mutates.

    ``init_logging`` clears the root logger's handlers and ``setup_run_logging``
    installs ``sys.excepthook`` / ``threading.excepthook`` hooks — both
    process-global. Without restoration, pytest's own logging plumbing breaks
    for subsequent tests in the same process.
    """
    root = logging.getLogger()
    saved_handlers = list(root.handlers)
    saved_level = root.level
    saved_sys_hook = sys.excepthook
    saved_thread_hook = threading.excepthook
    try:
        yield
    finally:
        root.handlers = saved_handlers
        root.setLevel(saved_level)
        sys.excepthook = saved_sys_hook
        threading.excepthook = saved_thread_hook


def test_setup_run_logging_creates_log_file(tmp_path: Path, _restore_logging_and_hooks) -> None:
    log_path = setup_run_logging(tmp_path, "myrun")
    assert log_path.exists(), "log file should be created on disk"
    assert log_path.parent == tmp_path
    # Filename must follow <run_name>_<YYYYMMDD_HHMMSS>.log
    assert re.match(r"^myrun_\d{8}_\d{6}\.log$", log_path.name), log_path.name


def test_setup_run_logging_creates_missing_output_dir(tmp_path: Path, _restore_logging_and_hooks) -> None:
    out_dir = tmp_path / "nested" / "outputs"
    assert not out_dir.exists()
    log_path = setup_run_logging(out_dir, "run")
    assert log_path.parent == out_dir
    assert out_dir.is_dir()


def test_setup_run_logging_accepts_str_path(tmp_path: Path, _restore_logging_and_hooks) -> None:
    log_path = setup_run_logging(str(tmp_path), "run")
    assert log_path.parent == tmp_path


def test_setup_run_logging_routes_logs_to_file(tmp_path: Path, _restore_logging_and_hooks) -> None:
    log_path = setup_run_logging(tmp_path, "run")
    logging.info("sentinel-line-2026")
    # File handler is buffered; flush via the handler chain.
    for h in logging.getLogger().handlers:
        h.flush()
    content = log_path.read_text()
    assert "sentinel-line-2026" in content


def test_install_exception_logging_routes_main_uncaught(tmp_path: Path, _restore_logging_and_hooks) -> None:
    """``sys.excepthook`` should route non-KeyboardInterrupt exceptions through
    ``logging.error`` so the per-run log file captures the traceback.
    """
    log_path = setup_run_logging(tmp_path, "run")
    try:
        raise RuntimeError("boom-sentinel")
    except RuntimeError:
        exc_type, exc_value, exc_tb = sys.exc_info()
    # Invoke the installed hook directly — simulates an uncaught exception
    # without actually crashing the test process.
    sys.excepthook(exc_type, exc_value, exc_tb)
    for h in logging.getLogger().handlers:
        h.flush()
    content = log_path.read_text()
    assert "Uncaught exception" in content
    assert "boom-sentinel" in content


def test_install_exception_logging_passes_keyboard_interrupt_through(
    _restore_logging_and_hooks,
) -> None:
    """KeyboardInterrupt must reach the default excepthook (so Ctrl-C still
    terminates the process cleanly) and NOT route through logging.
    """
    _install_exception_logging()
    calls: list[type[BaseException]] = []
    real_default = sys.__excepthook__

    def _fake_default(exc_type, exc_value, exc_tb):
        calls.append(exc_type)

    sys.__excepthook__ = _fake_default
    try:
        try:
            raise KeyboardInterrupt()
        except KeyboardInterrupt:
            sys.excepthook(*sys.exc_info())
    finally:
        sys.__excepthook__ = real_default

    assert calls == [KeyboardInterrupt]


def test_install_exception_logging_routes_thread_uncaught(tmp_path: Path, _restore_logging_and_hooks) -> None:
    """``threading.excepthook`` must capture thread crashes into the log file.

    Without this, ``Exception in thread …`` would only print to stderr.
    """
    log_path = setup_run_logging(tmp_path, "run")

    def _boom():
        raise RuntimeError("thread-sentinel-X")

    t = threading.Thread(target=_boom, name="boom-thread")
    t.start()
    t.join()
    for h in logging.getLogger().handlers:
        h.flush()
    content = log_path.read_text()
    assert "boom-thread" in content
    assert "thread-sentinel-X" in content

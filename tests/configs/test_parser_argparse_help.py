"""Regression test for the argparse --help safety patch in `lerobot.configs.parser`.

dataclass field comments are surfaced as argparse help strings by draccus, and
argparse then runs `help_text % vars(action)` on them. Literal `%` in those
comments (e.g. "50% open", "99.5%") used to crash `--help` with a TypeError
or ValueError. The patch installed at parser import time wraps argparse's
substitution so literal `%` renders verbatim.
"""

from __future__ import annotations

import argparse

import pytest


@pytest.fixture(autouse=True)
def _ensure_patch_installed():
    """Importing `lerobot.configs.parser` installs the safety patch as a side
    effect. The fixture imports it so test ordering doesn't matter.
    """
    import lerobot.configs.parser  # noqa: F401


def _build_parser_with_help(help_text: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="test")
    parser.add_argument("--flag", help=help_text)
    return parser


def test_literal_percent_open_does_not_crash_help():
    """`%o` would otherwise parse as an octal-format directive."""
    parser = _build_parser_with_help("Gripper bounce back to neutral position (50% open)")
    text = parser.format_help()
    assert "(50% open)" in text


def test_literal_percent_dot_does_not_crash_help():
    """`%.` would otherwise parse as a precision-format directive."""
    parser = _build_parser_with_help("overrun ratio falling from 99.5% to 0.0%")
    text = parser.format_help()
    assert "99.5%" in text
    assert "0.0%" in text


def test_literal_percent_backtick_does_not_crash_help():
    """`%`` is an entirely-unsupported format character."""
    parser = _build_parser_with_help("the `%` symbol is literal here")
    text = parser.format_help()
    assert "`%`" in text


def test_standard_default_substitution_still_works():
    """`%(default)s` is the supported argparse substitution; the patch must
    not break it.
    """
    parser = argparse.ArgumentParser(prog="test")
    parser.add_argument("--count", type=int, default=42, help="how many (default: %(default)s)")
    text = parser.format_help()
    assert "(default: 42)" in text


def test_unmatched_paren_falls_back_to_verbatim():
    """`%(missing)` without a corresponding key in vars(action) would raise
    KeyError. The patch's second-pass escape doesn't help (the `%(` survives
    escaping), so the final fallback must return the original help verbatim.
    """
    parser = _build_parser_with_help("see %(nonexistent_key) for details")
    text = parser.format_help()
    # The exact form returned by the verbatim fallback (no substitution).
    assert "%(nonexistent_key)" in text or "nonexistent_key" in text

"""Helpers for safety-regression tests: snapshot a directory and assert no data loss."""

from __future__ import annotations

from pathlib import Path


def snapshot_tree(root: Path) -> dict[str, int]:
    """Return ``{relative_path: file_size_in_bytes}`` for every file under ``root``.

    Used by safety-regression tests to verify that an exception path did not
    silently destroy or modify user data. Compare two snapshots to detect any
    file removal, addition, or size change.

    Args:
        root: Directory to snapshot.

    Returns:
        Mapping from relative POSIX-style path to file size. Empty if root
        does not exist.
    """
    if not root.exists():
        return {}
    return {
        str(p.relative_to(root).as_posix()): p.stat().st_size for p in sorted(root.rglob("*")) if p.is_file()
    }


def assert_no_data_loss(before: dict[str, int], after: dict[str, int]) -> None:
    """Assert that no files were removed or shrunk between the two snapshots.

    Files added or grown are allowed (e.g. log lines appended). Files removed
    or shrunk fail the assertion — those are the destructive cases this guard
    is intended to catch.
    """
    removed = set(before) - set(after)
    if removed:
        raise AssertionError(
            f"DESTRUCTIVE REGRESSION: {len(removed)} file(s) removed: {sorted(removed)[:10]}"
        )
    shrunk = [(k, before[k], after[k]) for k in before if k in after and after[k] < before[k]]
    if shrunk:
        raise AssertionError(f"DESTRUCTIVE REGRESSION: {len(shrunk)} file(s) shrunk: {shrunk[:5]}")

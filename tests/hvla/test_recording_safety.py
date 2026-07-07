"""Safety regression tests for the HVLA recording flow.

Specifically guards against the May 2026 incident where a load failure
in ``_create_or_resume_dataset`` triggered a silent ``shutil.rmtree`` of
the user's dataset directory.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pytest

from tests.fixtures.dataset_snapshot import assert_no_data_loss, snapshot_tree


def _build_partial_dataset(root: Path) -> dict[str, int]:
    """Create a dataset dir that is partially valid — info.json present but
    meta/episodes/ has no parquet files. ``_load_metadata`` will raise
    ``FileNotFoundError`` from ``load_nested_dataset``.

    Returns the file snapshot for assertion comparison.
    """
    (root / "meta").mkdir(parents=True)
    (root / "meta" / "info.json").write_text(
        json.dumps({"codebase_version": "v3.0", "fps": 30, "features": {}})
    )
    (root / "meta" / "episodes").mkdir()  # empty — load_nested_dataset raises here
    (root / "data" / "chunk-000").mkdir(parents=True)
    (root / "data" / "chunk-000" / "file-000.parquet").write_bytes(b"placeholder")
    return snapshot_tree(root)


def test_create_or_resume_does_not_destroy_on_corrupted_metadata(tmp_path, monkeypatch):
    """Regression: a dataset dir that fails to load must NOT be auto-deleted.

    This is the exact failure mode that destroyed `eval_cylinder_ring_assembly_apr_24`
    in May 2026. The fix replaced the ``except: rmtree`` fallback with a
    fail-loud ``LeRobotDataset.resume()`` call.
    """
    repo_id = "test/safety_corrupted"
    dataset_root = tmp_path / repo_id
    snapshot = _build_partial_dataset(dataset_root)

    # Redirect HF_LEROBOT_HOME so dataset_root is what `_create_or_resume_dataset` finds.
    monkeypatch.setattr(
        "lerobot.utils.constants.HF_LEROBOT_HOME",
        tmp_path,
    )
    # The s1_process module captures this constant inside the function — patch its lookup too.

    from lerobot.policies.hvla.s1_process import _create_or_resume_dataset

    # The function must raise rather than silently destroy the dataset.
    with pytest.raises(Exception):
        _create_or_resume_dataset(repo_id=repo_id, fps=30, features={}, robot_type="test")

    # Files must still exist — none removed, none shrunk.
    assert_no_data_loss(snapshot, snapshot_tree(dataset_root))


def test_create_or_resume_does_not_probe_hub_when_root_exists(tmp_path, monkeypatch, lerobot_dataset_factory):
    """When a valid local dataset exists, no Hub call should happen.

    Otherwise a Hub-side rename or 404 (network glitch, repo migration) could
    cascade into the same destructive fallback path.
    """
    repo_id = "test/safety_local_only"
    dataset_root = tmp_path / repo_id
    dataset = lerobot_dataset_factory(root=dataset_root, total_episodes=2, total_frames=20)
    snapshot = snapshot_tree(dataset_root)

    monkeypatch.setattr("lerobot.utils.constants.HF_LEROBOT_HOME", tmp_path)

    from lerobot.policies.hvla.s1_process import _create_or_resume_dataset

    # Patch huggingface_hub.snapshot_download so any Hub probe raises loudly.
    with patch("lerobot.datasets.dataset_metadata.snapshot_download") as mock_snapshot:
        mock_snapshot.side_effect = AssertionError(
            "Hub probe attempted for a local-only dataset — this is the regression"
        )
        # Should succeed using the local files; resume() with explicit root must not probe.
        ds = _create_or_resume_dataset(
            repo_id=repo_id,
            fps=dataset.meta.fps,
            features=dataset.meta.features,
            robot_type=dataset.meta.robot_type,
        )
        assert ds.num_episodes == 2

    # Snapshot must be unchanged — no temp files leaked, no metadata rewritten silently.
    after = snapshot_tree(dataset_root)
    assert set(snapshot) == set(after), (
        f"file set changed: removed={set(snapshot) - set(after)}, added={set(after) - set(snapshot)}"
    )

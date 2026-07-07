"""Tests for lerobot.gui.dataset_reload helpers.

These test the extracted reload helpers that replaced the duplicated
reload logic in api/edits.py, api/datasets.py, and api/playback.py.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest


@pytest.fixture
def fake_dataset():
    """A minimal mock LeRobotDataset with the attributes the helpers touch."""
    ds = SimpleNamespace()
    ds.root = Path("/fake/dataset/root")
    ds.meta = SimpleNamespace(
        info={"marker": "old_info"},
        episodes=None,
        stats={"marker": "old_stats"},
        tasks={"marker": "old_tasks"},
        features={"marker": "features"},
    )
    ds.hf_dataset = None
    ds._lazy_loading = True
    return ds


def test_ensure_episodes_loaded_no_op_when_already_loaded(fake_dataset):
    """If episodes are already populated, don't call load_episodes again."""
    fake_dataset.meta.episodes = {0: {"length": 100}}

    with patch("lerobot.datasets.io_utils.load_episodes") as mock_load:
        from lerobot.gui.dataset_reload import ensure_episodes_loaded

        ensure_episodes_loaded(fake_dataset)

    mock_load.assert_not_called()
    assert fake_dataset.meta.episodes == {0: {"length": 100}}


def test_ensure_episodes_loaded_calls_load_when_none(fake_dataset):
    """If episodes is None, load from disk."""
    fake_dataset.meta.episodes = None

    with patch("lerobot.datasets.io_utils.load_episodes") as mock_load:
        mock_load.return_value = {0: {"length": 50}}
        from lerobot.gui.dataset_reload import ensure_episodes_loaded

        ensure_episodes_loaded(fake_dataset)

    mock_load.assert_called_once_with(fake_dataset.root)
    assert fake_dataset.meta.episodes == {0: {"length": 50}}


def test_ensure_episodes_loaded_custom_root(fake_dataset):
    """Explicit root override is used instead of dataset.root."""
    fake_dataset.meta.episodes = None
    custom_root = Path("/other/root")

    with patch("lerobot.datasets.io_utils.load_episodes") as mock_load:
        mock_load.return_value = {}
        from lerobot.gui.dataset_reload import ensure_episodes_loaded

        ensure_episodes_loaded(fake_dataset, root=custom_root)

    mock_load.assert_called_once_with(custom_root)


def test_reload_metadata_refreshes_all_four_fields(fake_dataset):
    """reload_metadata must call load_info, load_episodes, load_stats, load_tasks."""
    with (
        patch("lerobot.datasets.io_utils.load_info") as m_info,
        patch("lerobot.datasets.io_utils.load_episodes") as m_eps,
        patch("lerobot.datasets.io_utils.load_stats") as m_stats,
        patch("lerobot.datasets.io_utils.load_tasks") as m_tasks,
    ):
        m_info.return_value = {"marker": "new_info"}
        m_eps.return_value = {0: {"length": 10}}
        m_stats.return_value = {"marker": "new_stats"}
        m_tasks.return_value = {"marker": "new_tasks"}

        from lerobot.gui.dataset_reload import reload_metadata

        reload_metadata(fake_dataset)

    assert fake_dataset.meta.info == {"marker": "new_info"}
    assert fake_dataset.meta.episodes == {0: {"length": 10}}
    assert fake_dataset.meta.stats == {"marker": "new_stats"}
    assert fake_dataset.meta.tasks == {"marker": "new_tasks"}


def test_reload_hf_dataset_noop_when_reader_inactive(fake_dataset):
    """Skip reload when reader is None or its hf_dataset is None (lazy dataset)."""
    fake_dataset.reader = None

    from lerobot.gui.dataset_reload import reload_hf_dataset

    reload_hf_dataset(fake_dataset)  # must not raise

    # Reader-with-no-hf_dataset is also a no-op
    fake_dataset.reader = MagicMock()
    fake_dataset.reader.hf_dataset = None
    reload_hf_dataset(fake_dataset)
    fake_dataset.reader.load_and_activate.assert_not_called()


def test_reload_hf_dataset_routes_through_reader(fake_dataset):
    """When reader.hf_dataset exists, cleanup cache, disable caching, call
    ``reader.load_and_activate()``, re-enable caching.

    Regression: pre-fix, this helper tried `dataset.hf_dataset = ...` which
    raises AttributeError on the post-refactor read-only property.
    """
    mock_hf = MagicMock()
    fake_dataset.reader = MagicMock()
    fake_dataset.reader.hf_dataset = mock_hf
    fake_dataset.reader.root = fake_dataset.root

    with patch("datasets.disable_caching") as m_disable, patch("datasets.enable_caching") as m_enable:
        from lerobot.gui.dataset_reload import reload_hf_dataset

        reload_hf_dataset(fake_dataset)

    mock_hf.cleanup_cache_files.assert_called_once()
    m_disable.assert_called_once()
    m_enable.assert_called_once()
    fake_dataset.reader.load_and_activate.assert_called_once()


def test_reload_hf_dataset_reenables_caching_on_error(fake_dataset):
    """If reader.load_and_activate() raises, caching must still be re-enabled."""
    fake_dataset.reader = MagicMock()
    fake_dataset.reader.hf_dataset = MagicMock()
    fake_dataset.reader.root = fake_dataset.root
    fake_dataset.reader.load_and_activate.side_effect = RuntimeError("boom")

    with patch("datasets.disable_caching"), patch("datasets.enable_caching") as m_enable:
        from lerobot.gui.dataset_reload import reload_hf_dataset

        with pytest.raises(RuntimeError, match="boom"):
            reload_hf_dataset(fake_dataset)

    m_enable.assert_called_once()


def test_reload_dataset_from_disk_calls_both(fake_dataset):
    """reload_dataset_from_disk = reload_metadata + reload_hf_dataset."""
    with (
        patch("lerobot.gui.dataset_reload.reload_metadata") as m_meta,
        patch("lerobot.gui.dataset_reload.reload_hf_dataset") as m_hf,
    ):
        from lerobot.gui.dataset_reload import reload_dataset_from_disk

        reload_dataset_from_disk(fake_dataset)

    m_meta.assert_called_once_with(fake_dataset, root=None)
    m_hf.assert_called_once_with(fake_dataset, root=None)


def test_reload_dataset_from_disk_passes_custom_root(fake_dataset):
    """Custom root is forwarded to both helpers."""
    custom = Path("/other/root")

    with (
        patch("lerobot.gui.dataset_reload.reload_metadata") as m_meta,
        patch("lerobot.gui.dataset_reload.reload_hf_dataset") as m_hf,
    ):
        from lerobot.gui.dataset_reload import reload_dataset_from_disk

        reload_dataset_from_disk(fake_dataset, root=custom)

    m_meta.assert_called_once_with(fake_dataset, root=custom)
    m_hf.assert_called_once_with(fake_dataset, root=custom)

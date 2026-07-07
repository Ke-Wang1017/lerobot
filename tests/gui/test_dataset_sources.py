"""Tests for dataset source management (folder browser)."""

import json
from pathlib import Path
from unittest.mock import patch

from lerobot.gui.api.datasets import (
    _read_sources,
    _scan_source,
    _write_sources,
)

# ============================================================================
# _read_sources / _write_sources
# ============================================================================


class TestReadSources:
    """Tests for reading source folder config."""

    def test_returns_default_when_no_file(self, tmp_path):
        fake_file = tmp_path / "nonexistent" / "dataset_sources.json"
        with patch("lerobot.gui.api.datasets.SOURCES_FILE", fake_file):
            sources = _read_sources()
        assert len(sources) == 1
        assert sources[0]["removable"] is False
        assert sources[0]["expanded"] is True

    def test_reads_from_file(self, tmp_path):
        fake_file = tmp_path / "dataset_sources.json"
        hf_home = "/home/test/.cache/huggingface/lerobot"
        data = {
            "version": 1,
            "sources": [
                {"path": hf_home, "removable": False, "expanded": True},
                {"path": "/tmp/extra", "removable": True, "expanded": False},
            ],
        }
        fake_file.write_text(json.dumps(data))
        with (
            patch("lerobot.gui.api.datasets.SOURCES_FILE", fake_file),
            patch("lerobot.gui.api.datasets.HF_LEROBOT_HOME", Path(hf_home)),
        ):
            sources = _read_sources()
        assert len(sources) == 2
        assert sources[1]["path"] == "/tmp/extra"
        assert sources[1]["removable"] is True

    def test_ensures_default_source_present(self, tmp_path):
        """If config file exists but doesn't include default, it gets added."""
        fake_file = tmp_path / "dataset_sources.json"
        hf_home = "/home/test/.cache/huggingface/lerobot"
        data = {
            "version": 1,
            "sources": [
                {"path": "/tmp/custom", "removable": True, "expanded": False},
            ],
        }
        fake_file.write_text(json.dumps(data))
        with (
            patch("lerobot.gui.api.datasets.SOURCES_FILE", fake_file),
            patch("lerobot.gui.api.datasets.HF_LEROBOT_HOME", Path(hf_home)),
        ):
            sources = _read_sources()
        assert len(sources) == 2
        assert sources[0]["path"] == hf_home  # default inserted first
        assert sources[0]["removable"] is False

    def test_handles_corrupt_file(self, tmp_path):
        fake_file = tmp_path / "dataset_sources.json"
        fake_file.write_text("not valid json{{{")
        with patch("lerobot.gui.api.datasets.SOURCES_FILE", fake_file):
            sources = _read_sources()
        # Falls back to default
        assert len(sources) == 1
        assert sources[0]["removable"] is False


class TestWriteSources:
    """Tests for writing source folder config."""

    def test_creates_file(self, tmp_path):
        fake_file = tmp_path / "subdir" / "dataset_sources.json"
        sources = [{"path": "/tmp/test", "removable": True, "expanded": True}]
        with patch("lerobot.gui.api.datasets.SOURCES_FILE", fake_file):
            _write_sources(sources)
        assert fake_file.exists()
        data = json.loads(fake_file.read_text())
        assert data["version"] == 1
        assert len(data["sources"]) == 1

    def test_roundtrip(self, tmp_path):
        fake_file = tmp_path / "dataset_sources.json"
        hf_home = "/home/test/.cache/huggingface/lerobot"
        original = [
            {"path": hf_home, "removable": False, "expanded": True},
            {"path": "/tmp/extra", "removable": True, "expanded": False},
        ]
        with (
            patch("lerobot.gui.api.datasets.SOURCES_FILE", fake_file),
            patch("lerobot.gui.api.datasets.HF_LEROBOT_HOME", Path(hf_home)),
        ):
            _write_sources(original)
            result = _read_sources()
        assert len(result) == 2
        assert result[0]["path"] == hf_home
        assert result[1]["path"] == "/tmp/extra"
        assert result[1]["expanded"] is False


# ============================================================================
# _scan_source
# ============================================================================


class TestScanSource:
    """Tests for scanning directories for datasets."""

    def _make_dataset(self, root: Path, name: str, **info_overrides) -> Path:
        """Create a minimal dataset directory with meta/info.json."""
        ds_dir = root / name
        meta_dir = ds_dir / "meta"
        meta_dir.mkdir(parents=True)
        info = {
            "total_episodes": 10,
            "total_frames": 300,
            "fps": 30,
            "robot_type": "so101_follower",
            **info_overrides,
        }
        (meta_dir / "info.json").write_text(json.dumps(info))
        return ds_dir

    def test_finds_datasets(self, tmp_path):
        self._make_dataset(tmp_path, "dataset_a")
        self._make_dataset(tmp_path, "dataset_b", total_episodes=5)
        result = _scan_source(str(tmp_path))
        assert len(result) == 2
        names = [d["name"] for d in result]
        assert "dataset_a" in names
        assert "dataset_b" in names

    def test_reads_metadata(self, tmp_path):
        self._make_dataset(tmp_path, "my_dataset", total_episodes=42, total_frames=1000, fps=15)
        result = _scan_source(str(tmp_path))
        assert len(result) == 1
        ds = result[0]
        assert ds["name"] == "my_dataset"
        assert ds["total_episodes"] == 42
        assert ds["total_frames"] == 1000
        assert ds["fps"] == 15

    def test_nested_datasets(self, tmp_path):
        """Datasets can be nested (e.g. user/dataset_name)."""
        user_dir = tmp_path / "thewisp"
        user_dir.mkdir()
        self._make_dataset(user_dir, "pick_ball")
        self._make_dataset(user_dir, "push_cube")
        result = _scan_source(str(tmp_path))
        assert len(result) == 2
        names = sorted(d["name"] for d in result)
        assert names == ["thewisp/pick_ball", "thewisp/push_cube"]

    def test_respects_max_depth(self, tmp_path):
        """Datasets deeper than max_depth are not found."""
        deep = tmp_path / "a" / "b" / "c" / "d"
        deep.mkdir(parents=True)
        self._make_dataset(deep, "too_deep")
        result = _scan_source(str(tmp_path), max_depth=3)
        # a/b/c/d/too_deep = depth 4, should not be found
        assert len(result) == 0

    def test_does_not_recurse_into_datasets(self, tmp_path):
        """Once a dataset is found, its subdirs are not scanned."""
        ds = self._make_dataset(tmp_path, "parent_ds")
        # Put another dataset-like structure inside
        nested = ds / "nested" / "meta"
        nested.mkdir(parents=True)
        (nested / "info.json").write_text(json.dumps({"total_episodes": 1}))
        result = _scan_source(str(tmp_path))
        assert len(result) == 1
        assert result[0]["name"] == "parent_ds"

    def test_skips_dot_directories(self, tmp_path):
        self._make_dataset(tmp_path, ".hidden_dataset")
        self._make_dataset(tmp_path, "visible_dataset")
        result = _scan_source(str(tmp_path))
        assert len(result) == 1
        assert result[0]["name"] == "visible_dataset"

    def test_handles_missing_directory(self):
        result = _scan_source("/nonexistent/path/12345")
        assert result == []

    def test_handles_null_robot_type(self, tmp_path):
        self._make_dataset(tmp_path, "null_robot", robot_type=None)
        result = _scan_source(str(tmp_path))
        assert len(result) == 1
        assert result[0]["robot_type"] == ""

    def test_handles_missing_robot_type(self, tmp_path):
        """info.json without robot_type key."""
        ds_dir = tmp_path / "no_robot"
        meta_dir = ds_dir / "meta"
        meta_dir.mkdir(parents=True)
        (meta_dir / "info.json").write_text(
            json.dumps(
                {
                    "total_episodes": 5,
                    "total_frames": 100,
                    "fps": 10,
                }
            )
        )
        result = _scan_source(str(tmp_path))
        assert len(result) == 1
        assert result[0]["robot_type"] == ""

    def test_handles_corrupt_info_json(self, tmp_path):
        """Datasets with unreadable info.json are skipped."""
        ds_dir = tmp_path / "bad_dataset"
        meta_dir = ds_dir / "meta"
        meta_dir.mkdir(parents=True)
        (meta_dir / "info.json").write_text("not json")
        self._make_dataset(tmp_path, "good_dataset")
        result = _scan_source(str(tmp_path))
        assert len(result) == 1
        assert result[0]["name"] == "good_dataset"

    def test_empty_directory(self, tmp_path):
        result = _scan_source(str(tmp_path))
        assert result == []

    def test_sorted_by_name(self, tmp_path):
        self._make_dataset(tmp_path, "zebra")
        self._make_dataset(tmp_path, "alpha")
        self._make_dataset(tmp_path, "middle")
        result = _scan_source(str(tmp_path))
        names = [d["name"] for d in result]
        assert names == ["alpha", "middle", "zebra"]

    def test_root_path_in_result(self, tmp_path):
        self._make_dataset(tmp_path, "my_ds")
        result = _scan_source(str(tmp_path))
        assert result[0]["root"] == str(tmp_path / "my_ds")

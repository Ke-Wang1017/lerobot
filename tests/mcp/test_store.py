"""Unit tests for the annotation sidecar (no external deps)."""

from __future__ import annotations

from pathlib import Path

import pytest

from lerobot.mcp.store import AnnotationStore


@pytest.fixture
def store(tmp_path: Path) -> AnnotationStore:
    return AnnotationStore(tmp_path / "annotations.sqlite")


def test_creates_db_file(tmp_path: Path) -> None:
    db = tmp_path / "sub" / "annotations.sqlite"
    AnnotationStore(db)
    assert db.exists()


def test_set_and_get_tag_roundtrip(store: AnnotationStore) -> None:
    store.set_tag("org/ds", 0, "outcome", "success")
    assert store.get_tags("org/ds", 0) == {"outcome": "success"}


def test_get_tags_empty_for_unknown_episode(store: AnnotationStore) -> None:
    assert store.get_tags("org/ds", 99) == {}


def test_multiple_tags_per_episode(store: AnnotationStore) -> None:
    store.set_tag("org/ds", 5, "outcome", "failure")
    store.set_tag("org/ds", 5, "failure_mode", "gripper_miss")
    store.set_tag("org/ds", 5, "score", 0.42)
    tags = store.get_tags("org/ds", 5)
    assert tags == {"outcome": "failure", "failure_mode": "gripper_miss", "score": 0.42}


def test_upsert_overwrites_value(store: AnnotationStore) -> None:
    store.set_tag("org/ds", 1, "outcome", "success")
    store.set_tag("org/ds", 1, "outcome", "failure")
    assert store.get_tags("org/ds", 1) == {"outcome": "failure"}


def test_tags_are_scoped_per_repo(store: AnnotationStore) -> None:
    store.set_tag("org/a", 0, "k", 1)
    store.set_tag("org/b", 0, "k", 2)
    assert store.get_tags("org/a", 0) == {"k": 1}
    assert store.get_tags("org/b", 0) == {"k": 2}


def test_complex_values_roundtrip(store: AnnotationStore) -> None:
    store.set_tag("org/ds", 0, "phases", [{"name": "approach", "start": 0, "end": 50}])
    assert store.get_tags("org/ds", 0) == {"phases": [{"name": "approach", "start": 0, "end": 50}]}


def test_delete_tag(store: AnnotationStore) -> None:
    store.set_tag("org/ds", 0, "k", 1)
    assert store.delete_tag("org/ds", 0, "k") is True
    assert store.get_tags("org/ds", 0) == {}
    assert store.delete_tag("org/ds", 0, "k") is False


def test_list_tagged_episodes_all(store: AnnotationStore) -> None:
    store.set_tag("org/ds", 0, "a", 1)
    store.set_tag("org/ds", 2, "b", 2)
    store.set_tag("org/ds", 1, "a", 3)
    eps = store.list_tagged_episodes("org/ds")
    assert eps == [{"episode_id": 0}, {"episode_id": 1}, {"episode_id": 2}]


def test_list_tagged_episodes_by_key(store: AnnotationStore) -> None:
    store.set_tag("org/ds", 0, "outcome", "success")
    store.set_tag("org/ds", 1, "outcome", "failure")
    store.set_tag("org/ds", 2, "note", "skip")  # different key, excluded
    eps = store.list_tagged_episodes("org/ds", key="outcome")
    assert [e["episode_id"] for e in eps] == [0, 1]
    assert eps[0]["value"] == "success"
    assert eps[1]["value"] == "failure"


def test_invalid_episode_id_rejected(store: AnnotationStore) -> None:
    with pytest.raises(AssertionError):
        store.set_tag("org/ds", -1, "k", 1)


def test_empty_key_rejected(store: AnnotationStore) -> None:
    with pytest.raises(AssertionError):
        store.set_tag("org/ds", 0, "", 1)


def test_persistence_across_instances(tmp_path: Path) -> None:
    db = tmp_path / "annotations.sqlite"
    s1 = AnnotationStore(db)
    s1.set_tag("org/ds", 0, "k", "v")
    s2 = AnnotationStore(db)
    assert s2.get_tags("org/ds", 0) == {"k": "v"}

#!/usr/bin/env python

# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json
from pathlib import Path
from typing import Any

import pytest

from lerobot.utils.io_utils import deserialize_json_into_object


@pytest.fixture
def tmp_json_file(tmp_path: Path):
    """Writes `data` to a temporary JSON file and returns the file's path."""

    def _write(data: Any) -> Path:
        file_path = tmp_path / "data.json"
        with file_path.open("w", encoding="utf-8") as f:
            json.dump(data, f)
        return file_path

    return _write


def test_simple_dict(tmp_json_file):
    data = {"name": "Alice", "age": 30}
    json_path = tmp_json_file(data)
    obj = {"name": "", "age": 0}
    assert deserialize_json_into_object(json_path, obj) == data


def test_nested_structure(tmp_json_file):
    data = {"items": [1, 2, 3], "info": {"active": True}}
    json_path = tmp_json_file(data)
    obj = {"items": [0, 0, 0], "info": {"active": False}}
    assert deserialize_json_into_object(json_path, obj) == data


def test_tuple_conversion(tmp_json_file):
    data = {"coords": [10.5, 20.5]}
    json_path = tmp_json_file(data)
    obj = {"coords": (0.0, 0.0)}
    result = deserialize_json_into_object(json_path, obj)
    assert result["coords"] == (10.5, 20.5)


def test_type_mismatch_raises(tmp_json_file):
    data = {"numbers": {"bad": "structure"}}
    json_path = tmp_json_file(data)
    obj = {"numbers": [0, 0]}
    with pytest.raises(TypeError):
        deserialize_json_into_object(json_path, obj)


def test_missing_key_raises(tmp_json_file):
    data = {"one": 1}
    json_path = tmp_json_file(data)
    obj = {"one": 0, "two": 0}
    with pytest.raises(ValueError):
        deserialize_json_into_object(json_path, obj)


def test_extra_key_raises(tmp_json_file):
    data = {"one": 1, "two": 2}
    json_path = tmp_json_file(data)
    obj = {"one": 0}
    with pytest.raises(ValueError):
        deserialize_json_into_object(json_path, obj)


def test_list_length_mismatch_raises(tmp_json_file):
    data = {"nums": [1, 2, 3]}
    json_path = tmp_json_file(data)
    obj = {"nums": [0, 0]}
    with pytest.raises(ValueError):
        deserialize_json_into_object(json_path, obj)


# ── write_json atomic-write behaviour ───────────────────────────────────────


from lerobot.utils.io_utils import write_json  # noqa: E402


def test_write_json_round_trips(tmp_path: Path):
    """Happy path: data written via write_json can be read back."""
    p = tmp_path / "out.json"
    payload = {"a": 1, "nested": {"b": [1, 2, 3]}}
    write_json(payload, p)
    assert json.loads(p.read_text()) == payload


def test_write_json_creates_parent_dirs(tmp_path: Path):
    p = tmp_path / "nested" / "dir" / "out.json"
    write_json({"x": 1}, p)
    assert p.exists()
    assert p.parent.is_dir()


def test_write_json_preserves_previous_file_on_serialization_failure(tmp_path: Path):
    """The atomic-write contract: if `json.dump` raises mid-write, the
    destination file must still hold its previous content — not a half-
    written file the next reader would reject with JSONDecodeError.
    """
    p = tmp_path / "stats.json"
    # Seed with valid content
    write_json({"good": True}, p)
    original_text = p.read_text()

    # Try to write an unserializable payload — set() isn't JSON-encodable.
    unserializable = {"bad": {1, 2, 3}}
    with pytest.raises(TypeError):
        write_json(unserializable, p)

    # Destination still holds the original content; no garbage was committed.
    assert p.read_text() == original_text
    # And the .tmp sibling was cleaned up; no stale staging file is left
    # behind to confuse a later observer.
    assert list(tmp_path.iterdir()) == [p]


def test_write_json_does_not_leave_tmp_sibling_on_success(tmp_path: Path):
    """The `.tmp` staging file must be renamed away on success, not left behind."""
    p = tmp_path / "out.json"
    write_json({"k": "v"}, p)
    # No `.tmp` sibling should remain.
    siblings = list(tmp_path.iterdir())
    assert siblings == [p], siblings

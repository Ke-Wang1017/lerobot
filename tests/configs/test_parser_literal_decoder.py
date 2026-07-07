#!/usr/bin/env python

# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Regression tests for the ``typing.Literal`` draccus decoder registered
in ``lerobot.configs.parser`` (commit eef8d9da0).

draccus has no built-in Literal handler. Without the registered decoder
any config field annotated with ``Literal[...]`` (e.g.
``camera_read_strategy=Literal["latest", "blocking"]`` on
BiSO107FollowerConfig) crashes CLI parse with "No decoding function for
type typing.Literal[...]". These tests pin the decoder so a draccus
upgrade or accidental decoder unregistration is caught early.
"""

from dataclasses import dataclass
from typing import Literal

import draccus
import pytest
from draccus.utils import DecodingError

# Importing lerobot.configs.parser registers the decoder as a side effect.
from lerobot.configs.parser import _decode_literal  # noqa: F401 (side-effect import)


class TestDecodeLiteralDirect:
    def test_accepts_string_literal(self):
        assert _decode_literal(Literal["a", "b"], "a") == "a"
        assert _decode_literal(Literal["a", "b"], "b") == "b"

    def test_rejects_unknown_value(self):
        with pytest.raises(DecodingError, match="not in allowed values"):
            _decode_literal(Literal["a", "b"], "c")

    def test_int_literal_accepts_int(self):
        assert _decode_literal(Literal[1, 2], 1) == 1

    def test_int_literal_accepts_str_via_fallback(self):
        # CLI / YAML often deliver "1" as a string even when the field
        # is Literal[1, 2]. The decoder normalizes back to the int.
        result = _decode_literal(Literal[1, 2], "1")
        assert result == 1
        assert isinstance(result, int)

    def test_error_message_includes_value_and_allowed_list(self):
        with pytest.raises(DecodingError) as exc:
            _decode_literal(Literal["a", "b"], "nope")
        msg = str(exc.value)
        assert "nope" in msg
        assert "a" in msg and "b" in msg


class TestDecodeLiteralViaDraccus:
    """The decoder must be picked up by draccus.decode for real dataclasses."""

    def test_decode_dataclass_with_literal_field(self):
        @dataclass
        class Cfg:
            mode: Literal["fast", "slow"] = "fast"

        cfg = draccus.decode(Cfg, {"mode": "slow"})
        assert cfg.mode == "slow"

    def test_decode_rejects_invalid_literal(self):
        @dataclass
        class Cfg:
            mode: Literal["fast", "slow"] = "fast"

        with pytest.raises(DecodingError, match="not in allowed values"):
            draccus.decode(Cfg, {"mode": "medium"})

    def test_decode_default_preserved(self):
        @dataclass
        class Cfg:
            mode: Literal["fast", "slow"] = "fast"

        cfg = draccus.decode(Cfg, {})
        assert cfg.mode == "fast"

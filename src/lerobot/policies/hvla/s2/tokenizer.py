"""PaliGemma tokenizer for S2 prompt encoding.

Lightweight sentencepiece-based tokenizer that builds the Pi0.5 prompt format:
  "Task: {task}; State: {state}; Subtask: {subtask};\nAction: "

Ported from openpi/models/tokenizer_lite.py — no JAX dependency.
"""

import logging
import string
from pathlib import Path

import numpy as np
import sentencepiece

logger = logging.getLogger(__name__)

DEFAULT_TOKENIZER_PATH = str(Path.home() / ".cache/openpi/big_vision/paligemma_tokenizer.model")


class PaligemmaTokenizer:
    """SentencePiece tokenizer for PaliGemma Pi0.5 prompt format."""

    def __init__(self, max_len: int = 256, tokenizer_path: str = DEFAULT_TOKENIZER_PATH):
        self._max_len = max_len
        self._sp = sentencepiece.SentencePieceProcessor()
        self._sp.Load(tokenizer_path)
        logger.info("Loaded PaliGemma tokenizer (vocab=%d) from %s", self._sp.GetPieceSize(), tokenizer_path)

    def detokenize(self, tokens: np.ndarray) -> str:
        """Decode token IDs to string, filtering out EOS (1) and padding (0)."""
        valid = [int(t) for t in tokens if t not in (0, 1)]
        return self._sp.decode(valid)

    def tokenize_prompt(
        self,
        high_prompt: str,
        low_prompt: str = "",
        state: np.ndarray | None = None,
        subtask_only: bool = False,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Tokenize task prompt for S2 VLM.

        Returns: (token_ids [max_len], attention_mask [max_len])
        """
        cleaned_high = high_prompt.lower().strip().replace("_", " ").replace("\n", " ")
        cleaned_low = low_prompt.lower().strip().replace("_", " ").replace("\n", " ")

        # Discretize state for prompt
        if state is not None:
            discretized = np.digitize(state, bins=np.linspace(-1, 1, 257)[:-1]) - 1
            state_str = " ".join(map(str, discretized))
        else:
            state_str = " ".join(["127"] * 32)

        # Normalize punctuation
        if cleaned_high and cleaned_high[-1] in string.punctuation:
            cleaned_high = cleaned_high[:-1]
        cleaned_high += "."

        # Segment 1: task + state (prefix, no loss)
        sub_prompt_1 = f"Task: {cleaned_high}; State: {state_str}; Subtask: "
        tokens_1 = self._sp.encode(sub_prompt_1, add_bos=True)

        # Segment 2: subtask (for AR decoding)
        if cleaned_low and cleaned_low[-1] in string.punctuation:
            cleaned_low = cleaned_low[:-1]
        if cleaned_low:
            cleaned_low += "."
        sub_prompt_2 = cleaned_low + ";\nAction: "
        tokens_2 = self._sp.encode(sub_prompt_2) + [1]  # EOS

        if subtask_only:
            # For subtask decoding: stop at "Subtask: " so the model
            # AR-decodes subtask text, not FAST action tokens.
            tokens = tokens_1
        else:
            tokens = tokens_1 + tokens_2

        # Pad / truncate
        n = len(tokens)
        if n < self._max_len:
            pad = self._max_len - n
            tokens = tokens + [0] * pad
            mask = [True] * n + [False] * pad
        else:
            if n > self._max_len:
                logger.warning("Token length %d exceeds max_len %d, truncating", n, self._max_len)
            tokens = tokens[: self._max_len]
            mask = [True] * self._max_len

        return (
            np.asarray(tokens, dtype=np.int32),
            np.asarray(mask, dtype=bool),
        )

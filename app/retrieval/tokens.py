"""Token counting with tiktoken cl100k_base, as the task requires for the corpus budget."""
from __future__ import annotations

import functools


@functools.lru_cache(maxsize=1)
def _encoding():
    import tiktoken

    return tiktoken.get_encoding("cl100k_base")


def count_tokens(text: str) -> int:
    return len(_encoding().encode(text, disallowed_special=()))

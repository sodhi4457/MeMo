"""Token-aware document chunking with sliding-window overlap.

Uses tiktoken (cl100k_base) for a tokenizer-agnostic chunk-size budget.
For Step A this only needs to keep chunks small enough to fit the
Generator's context — exact alignment with the Memory Model's tokenizer
is not required here.
"""

from __future__ import annotations

import tiktoken


def get_encoder(name: str = "cl100k_base"):
    return tiktoken.get_encoding(name)


def chunk_document(
    text: str,
    chunk_size: int = 512,
    overlap: int = 64,
    encoder=None,
) -> list[str]:
    """Split `text` into overlapping chunks of approximately `chunk_size` tokens.

    Overlap (default 64 tokens) preserves boundary context so facts spanning
    a split aren't lost.
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size must be > 0")
    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("overlap must satisfy 0 <= overlap < chunk_size")

    encoder = encoder or get_encoder()
    token_ids = encoder.encode(text)
    if not token_ids:
        return []

    chunks: list[str] = []
    start = 0
    stride = chunk_size - overlap
    while start < len(token_ids):
        end = min(start + chunk_size, len(token_ids))
        chunks.append(encoder.decode(token_ids[start:end]))
        if end == len(token_ids):
            break
        start += stride
    return chunks

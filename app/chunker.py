from __future__ import annotations

import re


def chunk_text(text: str, chunk_size: int = 520, overlap: int = 80) -> list[str]:
    """Split a document into overlapping chunks.

    The chunker keeps paragraphs together when possible and falls back to a
    fixed-size sliding window for long paragraphs.
    """
    normalized = re.sub(r"\n{3,}", "\n\n", text.strip())
    if not normalized:
        return []

    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", normalized) if p.strip()]
    chunks: list[str] = []
    buffer = ""

    def flush_buffer() -> None:
        nonlocal buffer
        if buffer.strip():
            chunks.append(buffer.strip())
        buffer = ""

    for paragraph in paragraphs:
        if len(paragraph) > chunk_size:
            flush_buffer()
            start = 0
            step = max(1, chunk_size - overlap)
            while start < len(paragraph):
                piece = paragraph[start : start + chunk_size].strip()
                if piece:
                    chunks.append(piece)
                start += step
            continue

        candidate = f"{buffer}\n\n{paragraph}".strip() if buffer else paragraph
        if len(candidate) <= chunk_size:
            buffer = candidate
        else:
            flush_buffer()
            buffer = paragraph

    flush_buffer()
    return chunks


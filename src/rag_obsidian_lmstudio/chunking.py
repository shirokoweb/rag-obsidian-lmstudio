"""Heading-aware markdown chunking (pure functions, no I/O).

Split on H1/H2 headings so chunks follow the note's own structure, then
word-window any section larger than ``chunk_words``. Known limitation:
``# `` lines inside fenced code blocks are treated as headings.
"""

import re

from .config import CHUNK_OVERLAP, CHUNK_WORDS

_HEADING_RE = re.compile(r"(?m)^(#{1,2} .*)$")


def chunk_markdown(
    text: str,
    chunk_words: int = CHUNK_WORDS,
    overlap: int = CHUNK_OVERLAP,
) -> list[str]:
    """Split markdown text into retrieval-sized chunks.

    Raises:
        ValueError: if ``overlap`` is not smaller than ``chunk_words``.
    """
    if overlap >= chunk_words:
        raise ValueError(f"overlap ({overlap}) must be smaller than chunk_words ({chunk_words})")

    chunks: list[str] = []
    for section in _split_sections(text):
        chunks.extend(_window(section, chunk_words, overlap))
    return chunks


def _split_sections(text: str) -> list[str]:
    """Split on H1/H2 headings; each heading keeps its body for context."""
    parts = _HEADING_RE.split(text)
    # re.split with a capture group interleaves [pre, head, body, head, body...]
    sections = []
    if parts[0].strip():
        sections.append(parts[0].strip())
    for i in range(1, len(parts), 2):
        head = parts[i].strip()
        body = parts[i + 1].strip() if i + 1 < len(parts) else ""
        sections.append(f"{head}\n{body}".strip())
    return sections


def _window(section: str, chunk_words: int, overlap: int) -> list[str]:
    """Word-window one section into pieces of at most ``chunk_words`` words."""
    words = section.split()
    if not words:
        return []
    if len(words) <= chunk_words:
        return [section]
    step = chunk_words - overlap
    out = []
    for start in range(0, len(words), step):
        out.append(" ".join(words[start : start + chunk_words]))
        if start + chunk_words >= len(words):
            break
    return out

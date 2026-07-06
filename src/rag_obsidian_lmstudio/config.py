"""Configuration: explicit arguments > ``RAG_*`` env vars > defaults.

``docs_dir`` is required and validated here so every downstream module
can assume it points at an existing directory.
"""

import os
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

from .errors import ConfigError

DEFAULT_BASE_URL = "http://localhost:1234/v1"
DEFAULT_TOP_K = 4
MAX_TOP_K = 20

# Chunking parameters (used by chunking.py; central so they stay consistent).
CHUNK_WORDS = 600
CHUNK_OVERLAP = 100

# LM Studio HTTP behaviour. Connect fails fast (server down is instant to
# detect); read is generous because large local models can take minutes to
# finish a completion.
EMBED_BATCH_SIZE = 64
HTTP_CONNECT_TIMEOUT_SECONDS = 5.0
HTTP_READ_TIMEOUT_SECONDS = 600.0


@dataclass(frozen=True)
class Config:
    docs_dir: Path
    base_url: str = DEFAULT_BASE_URL
    chat_model: str | None = None
    embed_model: str | None = None
    top_k: int = DEFAULT_TOP_K


def load_config(
    *,
    docs_dir: str | None = None,
    base_url: str | None = None,
    chat_model: str | None = None,
    embed_model: str | None = None,
    top_k: int | None = None,
    env: Mapping[str, str] = os.environ,
) -> Config:
    """Build a validated :class:`Config`.

    Raises:
        ConfigError: if ``docs_dir`` is missing/invalid or ``top_k`` is out of range.
    """
    raw_docs = docs_dir or env.get("RAG_DOCS_DIR")
    if not raw_docs:
        raise ConfigError(
            "No documents directory configured. "
            "Pass --docs-dir or set the RAG_DOCS_DIR environment variable."
        )
    path = Path(raw_docs).expanduser().resolve()
    if not path.exists():
        raise ConfigError(f"Documents directory does not exist: {path}")
    if not path.is_dir():
        raise ConfigError(f"Documents path is not a directory: {path}")

    return Config(
        docs_dir=path,
        base_url=base_url or env.get("RAG_BASE_URL") or DEFAULT_BASE_URL,
        chat_model=chat_model or env.get("RAG_CHAT_MODEL"),
        embed_model=embed_model or env.get("RAG_EMBED_MODEL"),
        top_k=_parse_top_k(top_k, env),
    )


def _parse_top_k(explicit: int | None, env: Mapping[str, str]) -> int:
    raw = str(explicit) if explicit is not None else env.get("RAG_TOP_K")
    if raw is None:
        return DEFAULT_TOP_K
    try:
        value = int(raw)
    except ValueError:
        raise ConfigError(
            f"top_k must be an integer, got {raw!r} (set via --top-k or RAG_TOP_K)."
        ) from None
    if not 1 <= value <= MAX_TOP_K:
        raise ConfigError(
            f"top_k must be between 1 and {MAX_TOP_K}, got {value} (set via --top-k or RAG_TOP_K)."
        )
    return value

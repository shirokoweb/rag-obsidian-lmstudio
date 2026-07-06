"""Build, cache, and query the vector index over a markdown vault.

Cache design (replaces the prototype's pickle — an arbitrary-code-execution
format — with inert data files):

- ``manifest.json``: schema version, embed model, per-file mtime + chunk texts
- ``vectors.npz``: one float32 matrix, rows aligned with manifest chunk order

Both are written atomically (tmp + ``os.replace``). Any inconsistency —
corrupt JSON, unreadable npz, row-count mismatch, different embed model —
invalidates the cache and triggers a full rebuild; the cache is disposable
by design, so rebuild is always the correct recovery.

Cache lives under the OS user-cache dir keyed by a hash of the vault path:
never inside the vault, never inside the repo.
"""

import hashlib
import json
import logging
import os
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import numpy as np
from platformdirs import user_cache_dir

from .chunking import chunk_markdown
from .config import Config
from .errors import IndexingError
from .lmstudio import ModelPair

logger = logging.getLogger(__name__)

_APP_NAME = "rag-obsidian-lmstudio"
_SCHEMA_VERSION = 2  # v2: shared generation ID ties vectors.npz to manifest.json
_MANIFEST = "manifest.json"
_VECTORS = "vectors.npz"


class EmbeddingProvider(Protocol):
    """What index building/retrieval needs from a client (see LMStudioClient)."""

    def resolve_models(self) -> ModelPair: ...

    def embed_documents(self, texts: list[str]) -> np.ndarray: ...

    def embed_queries(self, texts: list[str]) -> np.ndarray: ...


@dataclass(frozen=True)
class Hit:
    path: str  # relative to the vault root
    chunk: str
    score: float


@dataclass(frozen=True)
class Index:
    vectors_norm: np.ndarray  # (n_chunks, dim), rows L2-normalized
    metas: list[tuple[str, str]]  # (relative path, chunk text), row-aligned


def default_cache_dir(docs_dir: Path) -> Path:
    key = hashlib.sha256(str(docs_dir).encode("utf-8")).hexdigest()[:16]
    return Path(user_cache_dir(_APP_NAME)) / key


def build_index(
    client: EmbeddingProvider,
    config: Config,
    cache_dir: Path | None = None,
) -> Index:
    """Build the index incrementally: unchanged files reuse cached vectors.

    Raises:
        IndexingError: if the vault yields no chunks at all.
        LMStudioError: if embedding requests fail.
    """
    cache_dir = cache_dir or default_cache_dir(config.docs_dir)
    embed_model = client.resolve_models().embed
    cached_files, cached_vectors, cached_offsets = _load_cache(cache_dir, embed_model)

    files: dict[str, dict[str, Any]] = {}
    parts: list[np.ndarray] = []
    embedded = reused = 0

    for path in sorted(config.docs_dir.rglob("*.md")):
        rel = path.relative_to(config.docs_dir).as_posix()
        try:
            mtime = path.stat().st_mtime
        except OSError as e:
            logger.warning("Skipping %s: %s", rel, e)
            continue

        entry = cached_files.get(rel)
        vectors: np.ndarray | None
        if entry is not None and entry["mtime"] == mtime:
            chunks = entry["chunks"]
            offset = cached_offsets[rel]
            vectors = cached_vectors[offset : offset + len(chunks)]
            reused += 1
        else:
            try:
                text = path.read_text(encoding="utf-8")
            except (OSError, UnicodeDecodeError) as e:
                logger.warning("Skipping %s: %s", rel, e)
                continue
            chunks = chunk_markdown(text)
            vectors = client.embed_documents(chunks) if chunks else None
            embedded += 1

        files[rel] = {"mtime": mtime, "chunks": chunks}
        if chunks and vectors is not None:
            parts.append(vectors)

    _save_cache(cache_dir, embed_model, files, parts)
    logger.info("Indexed %d files (%d embedded, %d from cache)", len(files), embedded, reused)

    if not parts:
        raise IndexingError(
            f"No markdown content found under {config.docs_dir} — nothing to index."
        )
    vectors_all = np.vstack(parts)
    metas = [(rel, chunk) for rel, entry in files.items() for chunk in entry["chunks"]]
    return Index(vectors_norm=_normalize(vectors_all), metas=metas)


def retrieve(client: EmbeddingProvider, index: Index, query: str, k: int) -> list[Hit]:
    """Return the k most similar chunks for a query (cosine similarity)."""
    q = client.embed_queries([query])[0]
    q = q / (np.linalg.norm(q) + 1e-9)
    scores = index.vectors_norm @ q
    top = np.argsort(scores)[::-1][:k]
    return [
        Hit(path=index.metas[i][0], chunk=index.metas[i][1], score=float(scores[i])) for i in top
    ]


# -- cache I/O -----------------------------------------------------------------


def _load_cache(
    cache_dir: Path, embed_model: str
) -> tuple[dict[str, dict[str, Any]], np.ndarray, dict[str, int]]:
    """Load the cache; on any inconsistency return an empty one (full rebuild)."""
    empty: tuple[dict[str, dict[str, Any]], np.ndarray, dict[str, int]] = ({}, np.empty((0, 0)), {})
    try:
        manifest = json.loads((cache_dir / _MANIFEST).read_text(encoding="utf-8"))
        if manifest["schema"] != _SCHEMA_VERSION or manifest["embed_model"] != embed_model:
            return empty
        files: dict[str, dict[str, Any]] = manifest["files"]
        with np.load(cache_dir / _VECTORS, allow_pickle=False) as npz:
            vectors = npz["vectors"]
            generation = str(npz["generation"])
        if manifest["generation"] != generation:
            # A crash between the two os.replace calls in _write_cache paired
            # a stale manifest with new vectors; row counts alone can't always
            # detect that, the shared generation ID can.
            logger.info("Cache generation mismatch; rebuilding index from scratch.")
            return empty
        offsets: dict[str, int] = {}
        total = 0
        for rel, entry in files.items():
            offsets[rel] = total
            total += len(entry["chunks"])
        if total != vectors.shape[0]:
            logger.info("Cache row count mismatch; rebuilding index from scratch.")
            return empty
        return files, vectors, offsets
    except FileNotFoundError:
        return empty
    except Exception as e:  # cache is disposable; rebuild on any damage
        logger.info("Cache unreadable (%s); rebuilding index from scratch.", e)
        return empty


def _save_cache(
    cache_dir: Path,
    embed_model: str,
    files: dict[str, dict[str, Any]],
    parts: list[np.ndarray],
) -> None:
    """Best-effort persistence: the index is already in memory, so a failure
    to write the cache (disk full, permissions) degrades to a warning —
    it must never fail the build or leak a raw OSError past RagError."""
    try:
        _write_cache(cache_dir, embed_model, files, parts)
    except OSError as e:
        logger.warning("Could not persist index cache to %s: %s", cache_dir, e)


def _write_cache(
    cache_dir: Path,
    embed_model: str,
    files: dict[str, dict[str, Any]],
    parts: list[np.ndarray],
) -> None:
    cache_dir.mkdir(parents=True, exist_ok=True)
    os.chmod(cache_dir, 0o700)  # manifest holds note contents — owner only

    # One generation ID stamped into both files: on load, a mismatch proves
    # a crash landed between the two os.replace calls below.
    generation = uuid.uuid4().hex
    manifest = {
        "schema": _SCHEMA_VERSION,
        "generation": generation,
        "embed_model": embed_model,
        "files": files,
    }

    vectors_tmp = cache_dir / (_VECTORS + ".tmp")
    manifest_tmp = cache_dir / (_MANIFEST + ".tmp")
    try:
        with open(vectors_tmp, "wb") as fh:
            np.savez_compressed(
                fh,
                vectors=np.vstack(parts) if parts else np.empty((0, 0), dtype=np.float32),
                generation=np.array(generation),
            )
        os.replace(vectors_tmp, cache_dir / _VECTORS)
        manifest_tmp.write_text(json.dumps(manifest), encoding="utf-8")
        os.replace(manifest_tmp, cache_dir / _MANIFEST)
    finally:
        vectors_tmp.unlink(missing_ok=True)
        manifest_tmp.unlink(missing_ok=True)


def _normalize(matrix: np.ndarray) -> np.ndarray:
    normalized: np.ndarray = matrix / (np.linalg.norm(matrix, axis=1, keepdims=True) + 1e-9)
    return normalized

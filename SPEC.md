# Spec: rag-obsidian-lmstudio

Fully local RAG over an Obsidian (or any markdown) vault, powered by LM Studio.
Nothing leaves the machine. Public, installable, no personal paths in the repo.

## Objective

Rebuild of a working prototype (`~/rag-lmstudio`) to public-release quality.

**Users:** anyone running LM Studio locally who wants to ask questions grounded
in their own markdown notes — via a terminal REPL or directly inside the
LM Studio chat GUI (MCP tool).

**User stories:**
- As a user, I install one package, point it at my vault, and ask questions in
  a REPL; answers cite source filenames.
- As an LM Studio user, I register one MCP server in `mcp.json` and the chat
  model can search my notes and answer grounded.
- As a returning user, re-indexing is incremental — only changed files are
  re-embedded.

**Success looks like:** a fresh machine can `uv tool install` from GitHub, set
one env var, and both entry points work. CI is green. Every finding from the
2026-07-03 code review of the prototype is structurally impossible or tested.

## Tech Stack

- Python ≥ 3.11 (CI matrix: 3.11 / 3.12 / 3.13)
- `uv` for env + lockfile; `pyproject.toml` (hatchling backend)
- Runtime deps (pinned in lockfile, ranged in pyproject):
  - `mcp` (FastMCP stdio server)
  - `numpy` (vector math)
  - `httpx` (LM Studio HTTP client — already a transitive dep of `mcp`; no `requests`)
  - `platformdirs` (cache location in the OS user-cache dir, not in the vault)
- Dev deps: `pytest`, `pytest-cov`, `ruff`, `mypy`
- License: MIT

## Commands

```
Setup:      uv sync --all-extras
Test:       uv run pytest --cov=rag_obsidian_lmstudio
Lint:       uv run ruff check . && uv run ruff format --check .
Typecheck:  uv run mypy src
Run CLI:    uv run obsidian-rag --docs-dir <path>
Run MCP:    uv run obsidian-rag-mcp        (docs dir via RAG_DOCS_DIR)
Install:    uv tool install git+https://github.com/shirokoweb/rag-obsidian-lmstudio
```

## Project Structure

```
rag-obsidian-lmstudio/
├── pyproject.toml              # metadata, deps, entry points, ruff/mypy/pytest config
├── uv.lock
├── SPEC.md                     # this file (living document)
├── README.md                   # install + LM Studio setup (incl. mcp.json snippet)
├── LICENSE                     # MIT
├── .gitignore                  # .venv/, __pycache__/, .coverage, dist/
├── .github/workflows/ci.yml    # lint + typecheck + tests on push/PR, 3.11–3.13
├── src/rag_obsidian_lmstudio/
│   ├── __init__.py             # version
│   ├── config.py               # Config dataclass; CLI args > env vars > defaults
│   ├── errors.py               # RagError hierarchy (never sys.exit in library code)
│   ├── lmstudio.py             # httpx client: list_models, embed, chat
│   ├── chunking.py             # heading-aware markdown chunker (pure functions)
│   ├── index.py                # build/load/save index, cache, retrieval
│   ├── cli.py                  # REPL entry point (obsidian-rag)
│   └── mcp_server.py           # stdio MCP entry point (obsidian-rag-mcp)
└── tests/
    ├── test_chunking.py
    ├── test_config.py
    ├── test_index.py           # cache round-trip, eviction, corruption recovery
    ├── test_lmstudio.py        # httpx.MockTransport: embed order, batching, errors
    └── test_mcp_server.py      # top_k clamping, graceful error strings, no SystemExit
```

## Architecture & Key Decisions

Retrieval design is carried over from the prototype (it works):
heading-aware chunking (H1/H2 split, ~600-word windows, 100 overlap),
nomic task prefixes (`search_document:` / `search_query:`), cosine top-k,
per-file mtime cache. No vector DB, no reranker — out of scope.

Decisions that differ from the prototype (each maps to a review finding):

| # | Decision | Review finding |
|---|----------|----------------|
| 1 | Library raises `RagError` subclasses; only `cli.py` converts to exit codes | C1 (sys.exit killed MCP server) |
| 2 | Cache = `index.npz` (vectors) + `manifest.json` (paths, mtimes, chunks, schema version, embed model), written atomically (tmp + `os.replace`), stored under `platformdirs.user_cache_dir`, keyed by hash of docs-dir path | I2, S6, S7 (pickle RCE, corruption, cache in repo) |
| 3 | Corrupt/missing/version-mismatched cache → silent full rebuild, never a crash | I2 |
| 4 | Embeddings sorted by response `index` field; batched (≤ 64 texts/request) | I5, S4 |
| 5 | Per-file read/decode errors → warn to stderr, skip file, keep indexing | I3 |
| 6 | `top_k` clamped to [1, 20] in the MCP tool; query length capped | I4 |
| 7 | All config via `--flags` / `RAG_*` env vars; `docs_dir` **required**, validated (exists, is dir); no personal defaults anywhere | S1, public release |
| 8 | Cache evicts entries for deleted files on every save | S2 |
| 9 | Chat/embed model resolution: auto-detect like prototype, overridable via `RAG_CHAT_MODEL` / `RAG_EMBED_MODEL` | S3 |
| 10 | MCP tool description is generic ("the user's notes"), not vault-specific | S1 |

Env vars: `RAG_DOCS_DIR` (required for MCP), `RAG_BASE_URL`
(default `http://localhost:1234/v1`), `RAG_CHAT_MODEL`, `RAG_EMBED_MODEL`,
`RAG_TOP_K` (default 4).

## Code Style

`ruff` (lint + format, line length 100), `mypy` on `src/` (strict minus
`disallow_any_generics`). Full type hints. Google-style docstrings on public
functions only.

```python
def retrieve(
    query: str,
    embedding: EmbeddingClient,
    index: Index,
    k: int = DEFAULT_TOP_K,
) -> list[Hit]:
    """Return the k most similar chunks for a query.

    Raises:
        EmbeddingError: if the embedding request fails.
    """
    vec = embedding.embed_queries([query])[0]
    scores = index.vectors_norm @ vec
    top = np.argsort(scores)[::-1][:k]
    return [Hit(index.metas[i], float(scores[i])) for i in top]
```

Conventions: dataclasses for structured data (`Config`, `Hit`, `CacheEntry`);
no module-level mutable state in the library; constants in `config.py`.

## Testing Strategy

- `pytest`, tests in `tests/`, no network — LM Studio HTTP mocked with
  `httpx.MockTransport`; filesystem via `tmp_path`.
- Coverage gate: ≥ 85% on `chunking.py`, `index.py`, `config.py` (the pure
  logic). Entry-point glue (REPL loop) excluded via pragma.
- Required regression tests (one per review finding):
  - chunking: no headings, oversized section windows/overlap, heading-only file,
    empty file, heading inside fenced code block (documents known behavior)
  - index: cache round-trip, mtime-based reuse, eviction of deleted files,
    corrupt manifest → rebuild, embed-model change → re-embed
  - lmstudio: out-of-order embedding response reordered correctly; batching
    splits at 64; timeout → `EmbeddingError`
  - mcp: `top_k=-1`, `top_k=10_000` clamped; LM Studio down → error *string*
    returned (assert no `SystemExit` escapes)
- CI (GitHub Actions): ruff + mypy + pytest on push/PR across 3.11–3.13.

## Boundaries

- **Always:** run lint + typecheck + tests before every commit; atomic commits
  whose messages answer *why*; validate all external input (env vars, CLI args,
  MCP tool args, LM Studio responses) at the boundary; timeouts on every HTTP call.
- **Ask first:** adding any dependency beyond the five listed; publishing to
  PyPI; changing the cache schema after v0.1; force-push or history rewrites.
- **Never:** personal paths, vault names, or secrets in the repo; `pickle` for
  anything; `sys.exit` outside `cli.py`/`mcp_server.py` `main()`; writes inside
  the user's vault; network calls to anything but the configured LM Studio URL.

## Success Criteria

1. `uv sync && uv run pytest` green; ruff and mypy clean; CI green on GitHub.
2. Fresh-machine install: `uv tool install git+…` → `obsidian-rag --docs-dir ~/notes`
   answers a question with source citations against a running LM Studio.
3. MCP server registered in LM Studio `mcp.json` answers `search_notes` calls;
   with **no models loaded** it returns a readable error string and the process
   stays alive (C1 regression test, verified manually once against real LM Studio).
4. Killing the process mid-index leaves a cache that loads cleanly next run
   (rebuilds if partial).
5. No personal identifiers anywhere in the repo or its history.
6. README lets a stranger go from zero → working MCP tool without reading source.

## Resolved Decisions

1. Distribution: GitHub-install only for v0.1 (no PyPI yet). — 2026-07-03
2. Repo: `https://github.com/shirokoweb/rag-obsidian-lmstudio`, public. — 2026-07-03

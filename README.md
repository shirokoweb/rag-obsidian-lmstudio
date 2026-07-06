# rag-obsidian-lmstudio

Fully local RAG over an Obsidian (or any markdown) vault, powered by
[LM Studio](https://lmstudio.ai). **Nothing leaves your machine.**

Two ways to use it:

- **`obsidian-rag`** — a terminal REPL: ask questions, get answers grounded in
  your notes with source citations.
- **`obsidian-rag-mcp`** — an MCP server for the LM Studio GUI: the chat model
  gets a `search_notes` tool and answers from your vault, inside the app.

Indexing is incremental: only new or edited files are re-embedded on each run.

## Prerequisites

1. [LM Studio](https://lmstudio.ai) with the local server running
   (**Developer** tab → **Start Server**, default `http://localhost:1234`).
2. Two models loaded:
   - a **chat model** (e.g. any Gemma / Llama / Qwen instruct model)
   - an **embedding model** (e.g. `nomic-embed-text-v1.5`)
3. Python ≥ 3.11 and [`uv`](https://docs.astral.sh/uv/) (or `pipx`).

## Install

```sh
uv tool install git+https://github.com/shirokoweb/rag-obsidian-lmstudio
# or: pipx install git+https://github.com/shirokoweb/rag-obsidian-lmstudio
```

## Use the terminal REPL

```sh
obsidian-rag --docs-dir ~/path/to/your/vault
```

```
Chat:  google/gemma-4-e4b
Embed: text-embedding-nomic-embed-text-v1.5
Indexed 317 chunks from /Users/you/vault

Ask a question (blank line or Ctrl-D to quit).
? What is the CIA triad?

The CIA triad is a model that helps organizations consider risk ...

  sources: Module 2/25. Explore the CIA triad.md, ...
  top score: 0.830
```

## Use inside LM Studio (MCP)

Add the server to LM Studio's `mcp.json` (**Program** tab → **Install** →
**Edit mcp.json**):

```json
{
  "mcpServers": {
    "obsidian-rag": {
      "command": "obsidian-rag-mcp",
      "env": {
        "RAG_DOCS_DIR": "/Users/you/path/to/your/vault"
      }
    }
  }
}
```

Then ask the chat model anything about your notes — it calls `search_notes`
and answers grounded, with source filenames.

> If LM Studio can't find the command, use the absolute path
> (`which obsidian-rag-mcp`) in the `command` field.

## Configuration

CLI flags take precedence over environment variables.

| Flag | Env var | Default | Purpose |
|------|---------|---------|---------|
| `--docs-dir` | `RAG_DOCS_DIR` | *(required)* | Vault / notes directory |
| `--base-url` | `RAG_BASE_URL` | `http://localhost:1234/v1` | LM Studio server URL |
| `--chat-model` | `RAG_CHAT_MODEL` | auto-detect | Chat model id |
| `--embed-model` | `RAG_EMBED_MODEL` | auto-detect | Embedding model id |
| `--top-k` | `RAG_TOP_K` | `4` | Retrieved chunks per question (1–20) |

Auto-detection picks the first loaded model whose id contains `embed` as the
embedder and the first other model for chat. With several chat models loaded,
set `RAG_CHAT_MODEL` explicitly.

The embedding cache lives in your OS user-cache directory (e.g.
`~/Library/Caches/rag-obsidian-lmstudio` on macOS) — never inside your vault.
Deleting it is always safe; it will be rebuilt.

## Troubleshooting

| Symptom | Fix |
|---------|-----|
| `Cannot reach LM Studio` | Start the server: LM Studio → Developer tab → Start Server |
| `Need both a chat and an embedding model loaded` | Load an embedding model *and* a chat model in LM Studio |
| `request timed out ... responding slowly` | The chat model is too large/slow — set `RAG_CHAT_MODEL` to a smaller one |
| `No documents directory configured` | Pass `--docs-dir` or set `RAG_DOCS_DIR` |
| Stale answers after editing notes | Nothing to do — the index refreshes on every run/query |

## Privacy & security notes

- All traffic goes to your configured LM Studio URL (localhost by default);
  there are no other network calls, no telemetry.
- The cache uses plain JSON + NumPy `.npz` — no pickle, nothing executable.
- The tool only ever *reads* your vault; it never writes into it.

## Development

```sh
git clone https://github.com/shirokoweb/rag-obsidian-lmstudio
cd rag-obsidian-lmstudio
uv sync --all-extras
uv run pytest                 # tests
uv run ruff check . && uv run ruff format --check .
uv run mypy src               # typecheck
```

See [SPEC.md](SPEC.md) for design decisions. MIT license.

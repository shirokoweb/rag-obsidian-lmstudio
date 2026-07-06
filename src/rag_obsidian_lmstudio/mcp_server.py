"""stdio MCP server: exposes vault search to LM Studio's chat GUI.

stdout is the JSON-RPC channel; all diagnostics go through logging
(stderr). Every failure — config, server down, anything — is returned
to the model as a string: nothing may escape the tool and kill the
process (the prototype died on sys.exit; RagError + containment here
make that structurally impossible).

Index freshness: rebuilt on every call. The mtime cache makes an
unchanged rebuild cheap, and the model never sees a stale index.
"""

import logging
import sys

from mcp.server.fastmcp import FastMCP

from .config import MAX_TOP_K, load_config
from .errors import RagError
from .index import build_index, retrieve
from .lmstudio import LMStudioClient

logger = logging.getLogger(__name__)

MAX_QUERY_CHARS = 2000

mcp = FastMCP("obsidian-rag")


@mcp.tool()
def search_notes(query: str, top_k: int = 4) -> str:
    """Search the user's markdown notes for passages relevant to the query.
    Use this for any question about the user's notes, then answer using only
    the returned passages and cite their source filenames.

    Args:
        query: A natural-language question or topic to search for.
        top_k: How many passages to return (default 4, max 20).
    """
    try:
        return _search(query, top_k)
    except RagError as e:
        return f"Search unavailable: {e}"
    except Exception as e:  # containment: a tool result, never a dead server
        logger.exception("search_notes failed")
        return f"Search failed unexpectedly: {e!r}"


def _search(query: str, top_k: int) -> str:
    query = query.strip()[:MAX_QUERY_CHARS]
    if not query:
        return "Empty query — provide a question or topic to search for."
    k = max(1, min(top_k, MAX_TOP_K))

    config = load_config()  # env-only: the GUI passes RAG_* via mcp.json
    with LMStudioClient(config) as client:
        index = build_index(client, config)
        hits = retrieve(client, index, query, k)

    if not hits:
        return "No relevant passages found in the notes."
    return "\n\n---\n\n".join(f"[source: {h.path} | score {h.score:.3f}]\n{h.chunk}" for h in hits)


def main() -> None:
    logging.basicConfig(stream=sys.stderr, level=logging.INFO, format="%(levelname)s %(message)s")
    mcp.run()  # stdio transport


if __name__ == "__main__":
    main()

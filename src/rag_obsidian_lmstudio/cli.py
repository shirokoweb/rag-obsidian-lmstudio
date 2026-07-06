"""Terminal REPL: grounded Q&A over a markdown vault.

The only module (with mcp_server) allowed to exit the process: it maps
RagError to exit code 1 with a plain message. Library code never exits.
"""

import argparse
import logging
import sys
import textwrap

from . import __version__
from .config import Config, load_config
from .errors import RagError
from .index import Hit, Index, build_index, retrieve
from .lmstudio import LMStudioClient

SYSTEM_PROMPT = (
    "You answer strictly from the provided context. If the context does not "
    "contain the answer, say so plainly. Cite the source filename(s) you used."
)


def build_user_prompt(question: str, hits: list[Hit]) -> str:
    context = "\n\n".join(f"[{h.path}]\n{h.chunk}" for h in hits)
    return f"Context:\n{context}\n\nQuestion: {question}"


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="obsidian-rag",
        description="Ask questions grounded in your markdown vault, fully locally via LM Studio.",
    )
    parser.add_argument("--docs-dir", help="vault directory (env: RAG_DOCS_DIR)")
    parser.add_argument("--base-url", help="LM Studio server URL (env: RAG_BASE_URL)")
    parser.add_argument("--chat-model", help="chat model id override (env: RAG_CHAT_MODEL)")
    parser.add_argument("--embed-model", help="embedding model id override (env: RAG_EMBED_MODEL)")
    parser.add_argument("--top-k", type=int, help="retrieved chunks per question (env: RAG_TOP_K)")
    parser.add_argument("-v", "--verbose", action="store_true", help="show indexing progress")
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    return parser.parse_args(argv)


def run(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    logging.basicConfig(
        stream=sys.stderr,
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(levelname)s %(message)s",
    )
    try:
        config = load_config(
            docs_dir=args.docs_dir,
            base_url=args.base_url,
            chat_model=args.chat_model,
            embed_model=args.embed_model,
            top_k=args.top_k,
        )
        with LMStudioClient(config) as client:
            models = client.resolve_models()
            print(f"Chat:  {models.chat}\nEmbed: {models.embed}")
            index = build_index(client, config)
            print(f"Indexed {len(index.metas)} chunks from {config.docs_dir}\n")
            _repl(client, config, index)
    except RagError as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:  # pragma: no cover
        print()
        return 130
    return 0


def _repl(client: LMStudioClient, config: Config, index: Index) -> None:
    print("Ask a question (blank line or Ctrl-D to quit).")
    while True:
        try:
            question = input("? ").strip()
        except EOFError:
            return
        if not question:
            return
        hits = retrieve(client, index, question, config.top_k)
        answer = client.chat(SYSTEM_PROMPT, build_user_prompt(question, hits))
        print()
        print(textwrap.fill(answer, width=100))
        print(f"\n  sources: {', '.join(sorted({h.path for h in hits}))}")
        print(f"  top score: {hits[0].score:.3f}\n")


def main() -> None:
    raise SystemExit(run())

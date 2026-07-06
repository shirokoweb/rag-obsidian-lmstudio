"""Typed httpx client for LM Studio's OpenAI-compatible local server.

All failures surface as :class:`LMStudioError` subclasses (never raw
httpx exceptions, never SystemExit) so callers can recover uniformly.
Embeddings are batched and re-sorted by the response ``index`` field —
response order is not part of the OpenAI API contract.
"""

from dataclasses import dataclass
from types import TracebackType
from typing import Any

import httpx
import numpy as np

from .config import (
    EMBED_BATCH_SIZE,
    HTTP_CONNECT_TIMEOUT_SECONDS,
    HTTP_READ_TIMEOUT_SECONDS,
    Config,
)
from .errors import ChatError, EmbeddingError, LMStudioError

# nomic-embed requires these task prefixes; harmless no-op for other models.
DOC_PREFIX = "search_document: "
QUERY_PREFIX = "search_query: "

CHAT_TEMPERATURE = 0.2


@dataclass(frozen=True)
class ModelPair:
    chat: str
    embed: str


class LMStudioClient:
    """Client for one LM Studio server; resolves models once and caches."""

    def __init__(self, config: Config, transport: httpx.BaseTransport | None = None) -> None:
        self._config = config
        self._client = httpx.Client(
            base_url=config.base_url,
            timeout=httpx.Timeout(
                connect=HTTP_CONNECT_TIMEOUT_SECONDS,
                read=HTTP_READ_TIMEOUT_SECONDS,
                write=HTTP_READ_TIMEOUT_SECONDS,
                pool=HTTP_CONNECT_TIMEOUT_SECONDS,
            ),
            transport=transport,
        )
        self._models: ModelPair | None = None

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> "LMStudioClient":
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.close()

    # -- models ---------------------------------------------------------------

    def list_models(self) -> list[str]:
        payload = self._request("GET", "models", error=LMStudioError)
        try:
            return [m["id"] for m in payload["data"]]
        except (KeyError, TypeError) as e:
            raise LMStudioError(f"Malformed /models response: {e!r}") from e

    def resolve_models(self) -> ModelPair:
        """Pick chat + embedding model ids: explicit config wins, else auto-detect.

        Raises:
            LMStudioError: if the server is unreachable or a pair can't be resolved.
        """
        if self._models is not None:
            return self._models

        chat, embed = self._config.chat_model, self._config.embed_model
        if not (chat and embed):
            ids = self.list_models()
            embed = embed or next((m for m in ids if "embed" in m.lower()), None)
            chat = chat or next((m for m in ids if m != embed), None)
            if not (chat and embed):
                raise LMStudioError(
                    "Need both a chat and an embedding model loaded in LM Studio "
                    f"(currently loaded: {ids or 'none'}). Load them in the app, or set "
                    "RAG_CHAT_MODEL / RAG_EMBED_MODEL to override auto-detection."
                )
        self._models = ModelPair(chat=chat, embed=embed)
        return self._models

    # -- embeddings -----------------------------------------------------------

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        return self._embed([DOC_PREFIX + t for t in texts])

    def embed_queries(self, texts: list[str]) -> np.ndarray:
        return self._embed([QUERY_PREFIX + t for t in texts])

    def _embed(self, texts: list[str]) -> np.ndarray:
        """Embed texts -> (n, dim) float32 array, batched, order preserved."""
        if not texts:
            return np.empty((0, 0), dtype=np.float32)
        model = self.resolve_models().embed
        rows: list[list[float]] = []
        for start in range(0, len(texts), EMBED_BATCH_SIZE):
            batch = texts[start : start + EMBED_BATCH_SIZE]
            payload = self._request(
                "POST",
                "embeddings",
                error=EmbeddingError,
                json_body={"model": model, "input": batch},
            )
            try:
                data = sorted(payload["data"], key=lambda d: d["index"])
                rows.extend(d["embedding"] for d in data)
            except (KeyError, TypeError) as e:
                raise EmbeddingError(f"Malformed embeddings response: {e!r}") from e
        return np.asarray(rows, dtype=np.float32)

    # -- chat -----------------------------------------------------------------

    def chat(self, system: str, user: str) -> str:
        payload = self._request(
            "POST",
            "chat/completions",
            error=ChatError,
            json_body={
                "model": self.resolve_models().chat,
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                "temperature": CHAT_TEMPERATURE,
            },
        )
        try:
            content = payload["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as e:
            raise ChatError(f"Malformed chat response: {e!r}") from e
        if not isinstance(content, str):
            raise ChatError(f"Malformed chat response: content is {type(content).__name__}")
        return content

    # -- plumbing ---------------------------------------------------------------

    def _request(
        self,
        method: str,
        path: str,
        *,
        error: type[LMStudioError],
        json_body: dict[str, Any] | None = None,
    ) -> Any:
        try:
            r = self._client.request(method, path, json=json_body)
            r.raise_for_status()
            return r.json()
        except httpx.HTTPStatusError as e:
            raise error(
                f"LM Studio returned HTTP {e.response.status_code} for {path}: "
                f"{e.response.text[:200]}"
            ) from e
        except httpx.TimeoutException as e:
            raise error(
                f"LM Studio request timed out ({path}): {e!r}. The server is up but "
                "the model is responding slowly — a smaller model may help "
                "(set RAG_CHAT_MODEL to choose one)."
            ) from e
        except (httpx.HTTPError, ValueError) as e:
            raise error(
                f"Cannot reach LM Studio at {self._config.base_url} ({e!r}). "
                "Is the LM Studio server running? (Developer tab -> Start Server)"
            ) from e

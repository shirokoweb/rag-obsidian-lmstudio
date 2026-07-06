import json
from pathlib import Path

import httpx
import numpy as np
import pytest

from rag_obsidian_lmstudio.config import Config
from rag_obsidian_lmstudio.errors import ChatError, EmbeddingError, LMStudioError
from rag_obsidian_lmstudio.lmstudio import DOC_PREFIX, QUERY_PREFIX, LMStudioClient


def make_client(
    handler: httpx._transports.mock.SyncHandler,
    chat_model: str | None = None,
    embed_model: str | None = None,
) -> LMStudioClient:
    cfg = Config(docs_dir=Path("."), chat_model=chat_model, embed_model=embed_model)
    return LMStudioClient(cfg, transport=httpx.MockTransport(handler))


def models_response(ids: list[str]) -> httpx.Response:
    return httpx.Response(200, json={"data": [{"id": i} for i in ids]})


class TestResolveModels:
    def test_auto_detects_chat_and_embed(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            return models_response(["gemma-3n", "nomic-embed-text-v1.5"])

        pair = make_client(handler).resolve_models()
        assert pair.chat == "gemma-3n"
        assert pair.embed == "nomic-embed-text-v1.5"

    def test_no_models_loaded_raises(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            return models_response([])

        with pytest.raises(LMStudioError, match="chat and an embedding"):
            make_client(handler).resolve_models()

    def test_only_embed_model_raises(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            return models_response(["nomic-embed-text-v1.5"])

        with pytest.raises(LMStudioError, match="RAG_CHAT_MODEL"):
            make_client(handler).resolve_models()

    def test_explicit_overrides_skip_detection(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            raise AssertionError("no HTTP call expected when both models are configured")

        pair = make_client(handler, chat_model="my-chat", embed_model="my-embed").resolve_models()
        assert pair.chat == "my-chat"
        assert pair.embed == "my-embed"

    def test_server_down_raises_lmstudio_error(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            raise httpx.ConnectError("connection refused")

        with pytest.raises(LMStudioError, match="Is the LM Studio server running"):
            make_client(handler).resolve_models()

    def test_slow_model_timeout_is_diagnosed_as_slow_not_down(self) -> None:
        """Found live at CP3: a 31B chat model exceeding the read timeout was
        misreported as 'server not running'. Timeouts get their own message."""

        def handler(request: httpx.Request) -> httpx.Response:
            raise httpx.ReadTimeout("timed out")

        with pytest.raises(LMStudioError, match="responding slowly"):
            make_client(handler).resolve_models()


def embedding_handler(record: list[list[str]]) -> httpx._transports.mock.SyncHandler:
    """Echo handler: embedding of "... tN" is [N]. Returns data out of order
    to prove the client re-sorts by the response `index` field (I5)."""

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/models"):
            return models_response(["chat-model", "the-embedder"])
        inputs = json.loads(request.content)["input"]
        record.append(inputs)
        data = [
            {"index": i, "embedding": [float(text.rsplit("t", 1)[-1])]}
            for i, text in enumerate(inputs)
        ]
        return httpx.Response(200, json={"data": list(reversed(data))})

    return handler


class TestEmbeddings:
    def test_out_of_order_response_is_reordered(self) -> None:
        record: list[list[str]] = []
        vecs = make_client(embedding_handler(record)).embed_documents(["t0", "t1", "t2"])
        assert vecs[:, 0].tolist() == [0.0, 1.0, 2.0]

    def test_document_prefix_applied(self) -> None:
        record: list[list[str]] = []
        make_client(embedding_handler(record)).embed_documents(["t7"])
        assert record[0] == [f"{DOC_PREFIX}t7"]

    def test_query_prefix_applied(self) -> None:
        record: list[list[str]] = []
        make_client(embedding_handler(record)).embed_queries(["t7"])
        assert record[0] == [f"{QUERY_PREFIX}t7"]

    def test_batches_of_at_most_64_preserving_global_order(self) -> None:
        record: list[list[str]] = []
        texts = [f"t{i}" for i in range(100)]
        vecs = make_client(embedding_handler(record)).embed_documents(texts)
        assert [len(b) for b in record] == [64, 36]
        assert vecs.shape == (100, 1)
        assert vecs[:, 0].tolist() == [float(i) for i in range(100)]

    def test_returns_float32(self) -> None:
        record: list[list[str]] = []
        vecs = make_client(embedding_handler(record)).embed_queries(["t1"])
        assert vecs.dtype == np.float32

    def test_empty_input_makes_no_request(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            raise AssertionError("no HTTP call expected for empty input")

        vecs = make_client(handler, chat_model="c", embed_model="e").embed_documents([])
        assert vecs.shape[0] == 0

    def test_server_error_raises_embedding_error(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            if request.url.path.endswith("/models"):
                return models_response(["chat-model", "the-embedder"])
            return httpx.Response(500, text="boom")

        with pytest.raises(EmbeddingError):
            make_client(handler).embed_documents(["t1"])

    def test_short_response_raises_embedding_error(self) -> None:
        """A response with fewer embeddings than inputs must fail typed —
        not crash later in np.vstack with an opaque shape error."""

        def handler(request: httpx.Request) -> httpx.Response:
            if request.url.path.endswith("/models"):
                return models_response(["chat-model", "the-embedder"])
            return httpx.Response(200, json={"data": []})

        with pytest.raises(EmbeddingError, match="0 embeddings for 2 inputs"):
            make_client(handler).embed_documents(["t1", "t2"])

    def test_malformed_response_raises_embedding_error(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            if request.url.path.endswith("/models"):
                return models_response(["chat-model", "the-embedder"])
            return httpx.Response(200, json={"unexpected": True})

        with pytest.raises(EmbeddingError, match=r"[Mm]alformed"):
            make_client(handler).embed_documents(["t1"])


class TestChat:
    def test_returns_message_content(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            if request.url.path.endswith("/models"):
                return models_response(["chat-model", "the-embedder"])
            body = json.loads(request.content)
            assert body["messages"][0]["role"] == "system"
            return httpx.Response(200, json={"choices": [{"message": {"content": "the answer"}}]})

        assert make_client(handler).chat("sys", "hi") == "the answer"

    def test_http_error_raises_chat_error(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            if request.url.path.endswith("/models"):
                return models_response(["chat-model", "the-embedder"])
            return httpx.Response(500, text="boom")

        with pytest.raises(ChatError):
            make_client(handler).chat("sys", "hi")

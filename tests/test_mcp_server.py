import asyncio
from pathlib import Path
from types import TracebackType
from typing import Any

import pytest

from rag_obsidian_lmstudio import mcp_server
from rag_obsidian_lmstudio.index import Hit, Index
from rag_obsidian_lmstudio.lmstudio import ModelPair


class FakeClient:
    def __init__(self, config: Any) -> None:
        pass

    def __enter__(self) -> "FakeClient":
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        pass

    def resolve_models(self) -> ModelPair:
        return ModelPair(chat="c", embed="e")


@pytest.fixture
def wired(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    """Point the server at a tmp vault and record what reaches retrieve()."""
    (tmp_path / "a.md").write_text("alpha")
    monkeypatch.setenv("RAG_DOCS_DIR", str(tmp_path))
    seen: dict[str, Any] = {}
    fake_index = Index.__new__(Index)  # retrieval is faked; contents don't matter

    def fake_build(client: Any, config: Any, cache_dir: Any = None) -> Index:
        return fake_index

    def fake_retrieve(client: Any, index: Index, query: str, k: int) -> list[Hit]:
        seen["query"], seen["k"] = query, k
        return [Hit(path="a.md", chunk="alpha chunk", score=0.9)]

    monkeypatch.setattr(mcp_server, "LMStudioClient", FakeClient)
    monkeypatch.setattr(mcp_server, "build_index", fake_build)
    monkeypatch.setattr(mcp_server, "retrieve", fake_retrieve)
    return seen


class TestTopKClamping:
    """I4 regression: top_k is model-supplied input and must be clamped."""

    def test_negative_clamped_to_1(self, wired: dict[str, Any]) -> None:
        mcp_server.search_notes("q", top_k=-1)
        assert wired["k"] == 1

    def test_huge_clamped_to_max(self, wired: dict[str, Any]) -> None:
        mcp_server.search_notes("q", top_k=10_000)
        assert wired["k"] == 20

    def test_reasonable_value_passes_through(self, wired: dict[str, Any]) -> None:
        mcp_server.search_notes("q", top_k=7)
        assert wired["k"] == 7


class TestQueryHandling:
    def test_query_length_capped(self, wired: dict[str, Any]) -> None:
        mcp_server.search_notes("x" * 10_000)
        assert len(wired["query"]) == mcp_server.MAX_QUERY_CHARS

    def test_empty_query_returns_message_without_search(self, wired: dict[str, Any]) -> None:
        out = mcp_server.search_notes("   ")
        assert "query" in out.lower()
        assert "k" not in wired  # retrieve never called

    def test_hits_are_formatted_with_source_and_score(self, wired: dict[str, Any]) -> None:
        out = mcp_server.search_notes("anything")
        assert "a.md" in out
        assert "alpha chunk" in out
        assert "0.9" in out


class TestErrorContainment:
    """C1 regression: failures return strings; the server process never dies."""

    def test_missing_docs_dir_returns_string(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("RAG_DOCS_DIR", raising=False)
        out = mcp_server.search_notes("q")
        assert isinstance(out, str)
        assert "RAG_DOCS_DIR" in out

    def test_lmstudio_down_returns_string_not_exception(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        (tmp_path / "a.md").write_text("alpha")
        monkeypatch.setenv("RAG_DOCS_DIR", str(tmp_path))
        # port 9 (discard): connection refused — the prototype died here (sys.exit)
        monkeypatch.setenv("RAG_BASE_URL", "http://127.0.0.1:9/v1")
        out = mcp_server.search_notes("q")
        assert isinstance(out, str)
        assert "unavailable" in out.lower()

    def test_unexpected_exception_returns_string(
        self, wired: dict[str, Any], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def boom(client: Any, config: Any, cache_dir: Any = None) -> Index:
            raise RuntimeError("totally unexpected")

        monkeypatch.setattr(mcp_server, "build_index", boom)
        out = mcp_server.search_notes("q")
        assert isinstance(out, str)
        assert "unexpected" in out.lower()


class TestToolRegistration:
    def test_search_notes_is_registered_with_generic_description(self) -> None:
        tools = asyncio.run(mcp_server.mcp.list_tools())
        (tool,) = [t for t in tools if t.name == "search_notes"]
        desc = (tool.description or "").lower()
        assert "notes" in desc
        # public tool: description must not reference any specific person/vault
        assert "cybersecurity" not in desc

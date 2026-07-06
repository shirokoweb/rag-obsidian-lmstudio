import io
from pathlib import Path
from types import TracebackType
from typing import Any

import numpy as np
import pytest

from rag_obsidian_lmstudio import __version__, cli
from rag_obsidian_lmstudio.index import Hit, Index
from rag_obsidian_lmstudio.lmstudio import ModelPair


class TestErrorPaths:
    """CLI must translate RagError into exit 1 + message — never a traceback."""

    def test_missing_docs_dir_exits_1_with_hint(
        self, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv("RAG_DOCS_DIR", raising=False)
        assert cli.run([]) == 1
        err = capsys.readouterr().err
        assert "RAG_DOCS_DIR" in err
        assert "Traceback" not in err

    def test_nonexistent_docs_dir_exits_1(self, capsys: pytest.CaptureFixture[str]) -> None:
        assert cli.run(["--docs-dir", "/nonexistent/vault"]) == 1
        assert "does not exist" in capsys.readouterr().err

    def test_lmstudio_down_exits_1_with_message(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        (tmp_path / "a.md").write_text("hello")
        # port 9 (discard) is never an LM Studio server; connection is refused fast
        rc = cli.run(["--docs-dir", str(tmp_path), "--base-url", "http://127.0.0.1:9/v1"])
        assert rc == 1
        err = capsys.readouterr().err
        assert "LM Studio" in err
        assert "Traceback" not in err

    def test_out_of_range_top_k_exits_1(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        assert cli.run(["--docs-dir", str(tmp_path), "--top-k", "999"]) == 1
        assert "top_k" in capsys.readouterr().err


class TestFlags:
    def test_version_flag(self, capsys: pytest.CaptureFixture[str]) -> None:
        with pytest.raises(SystemExit) as excinfo:
            cli.run(["--version"])
        assert excinfo.value.code == 0
        assert __version__ in capsys.readouterr().out


class TestPromptBuilding:
    def test_prompt_contains_question_chunks_and_paths(self) -> None:
        hits = [
            Hit(path="a.md", chunk="alpha text", score=0.9),
            Hit(path="sub/b.md", chunk="beta text", score=0.5),
        ]
        prompt = cli.build_user_prompt("what is alpha?", hits)
        assert "what is alpha?" in prompt
        assert "alpha text" in prompt
        assert "beta text" in prompt
        assert "a.md" in prompt
        assert "sub/b.md" in prompt


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
        return ModelPair(chat="fake-chat", embed="fake-embed")

    def chat(self, system: str, user: str) -> str:
        return "the grounded answer"


class TestHappyPath:
    def test_repl_answers_one_question_then_eof(
        self,
        tmp_path: Path,
        capsys: pytest.CaptureFixture[str],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        (tmp_path / "a.md").write_text("alpha content")
        fake_index = Index(
            vectors_norm=np.eye(1, dtype=np.float32),
            metas=[("a.md", "alpha content")],
        )
        monkeypatch.setattr(cli, "LMStudioClient", FakeClient)
        monkeypatch.setattr(cli, "build_index", lambda client, config: fake_index)
        monkeypatch.setattr(
            cli,
            "retrieve",
            lambda client, index, q, k: [Hit(path="a.md", chunk="alpha content", score=0.87)],
        )
        monkeypatch.setattr("sys.stdin", io.StringIO("what is alpha?\n"))

        assert cli.run(["--docs-dir", str(tmp_path)]) == 0
        out = capsys.readouterr().out
        assert "the grounded answer" in out
        assert "a.md" in out
        assert "0.87" in out

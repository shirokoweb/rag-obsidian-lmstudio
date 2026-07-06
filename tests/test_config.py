from pathlib import Path

import pytest

from rag_obsidian_lmstudio.config import DEFAULT_BASE_URL, load_config
from rag_obsidian_lmstudio.errors import ConfigError, RagError


class TestErrorHierarchy:
    def test_config_error_is_rag_error(self) -> None:
        assert issubclass(ConfigError, RagError)

    def test_rag_error_is_not_system_exit(self) -> None:
        # C1 regression guard: library errors must be catchable via Exception.
        assert issubclass(RagError, Exception)
        assert not issubclass(RagError, SystemExit)


class TestDocsDir:
    def test_missing_docs_dir_raises_with_actionable_hint(self) -> None:
        with pytest.raises(ConfigError, match="RAG_DOCS_DIR"):
            load_config(env={})

    def test_nonexistent_docs_dir_raises(self, tmp_path: Path) -> None:
        with pytest.raises(ConfigError, match="does not exist"):
            load_config(docs_dir=str(tmp_path / "nope"), env={})

    def test_docs_dir_that_is_a_file_raises(self, tmp_path: Path) -> None:
        f = tmp_path / "notes.md"
        f.write_text("x")
        with pytest.raises(ConfigError, match="not a directory"):
            load_config(docs_dir=str(f), env={})

    def test_tilde_is_expanded(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("HOME", str(tmp_path))
        (tmp_path / "vault").mkdir()
        cfg = load_config(docs_dir="~/vault", env={})
        assert cfg.docs_dir == tmp_path / "vault"


class TestPrecedence:
    def test_arg_beats_env(self, tmp_path: Path) -> None:
        arg_dir = tmp_path / "from-arg"
        env_dir = tmp_path / "from-env"
        arg_dir.mkdir()
        env_dir.mkdir()
        cfg = load_config(docs_dir=str(arg_dir), env={"RAG_DOCS_DIR": str(env_dir)})
        assert cfg.docs_dir == arg_dir

    def test_env_used_when_no_arg(self, tmp_path: Path) -> None:
        cfg = load_config(env={"RAG_DOCS_DIR": str(tmp_path)})
        assert cfg.docs_dir == tmp_path

    def test_defaults(self, tmp_path: Path) -> None:
        cfg = load_config(docs_dir=str(tmp_path), env={})
        assert cfg.base_url == DEFAULT_BASE_URL
        assert cfg.top_k == 4
        assert cfg.chat_model is None
        assert cfg.embed_model is None

    def test_base_url_and_models_from_env(self, tmp_path: Path) -> None:
        env = {
            "RAG_DOCS_DIR": str(tmp_path),
            "RAG_BASE_URL": "http://localhost:9999/v1",
            "RAG_CHAT_MODEL": "some-chat",
            "RAG_EMBED_MODEL": "some-embed",
        }
        cfg = load_config(env=env)
        assert cfg.base_url == "http://localhost:9999/v1"
        assert cfg.chat_model == "some-chat"
        assert cfg.embed_model == "some-embed"


class TestTopK:
    def test_top_k_from_env(self, tmp_path: Path) -> None:
        cfg = load_config(env={"RAG_DOCS_DIR": str(tmp_path), "RAG_TOP_K": "7"})
        assert cfg.top_k == 7

    @pytest.mark.parametrize("bad", ["abc", "0", "-3", "999", "4.5"])
    def test_invalid_top_k_raises(self, tmp_path: Path, bad: str) -> None:
        with pytest.raises(ConfigError, match="RAG_TOP_K"):
            load_config(env={"RAG_DOCS_DIR": str(tmp_path), "RAG_TOP_K": bad})

    def test_config_is_immutable(self, tmp_path: Path) -> None:
        cfg = load_config(docs_dir=str(tmp_path), env={})
        with pytest.raises(AttributeError):
            cfg.top_k = 9  # type: ignore[misc]

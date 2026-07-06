import logging
from pathlib import Path

import numpy as np
import pytest

from rag_obsidian_lmstudio.config import Config
from rag_obsidian_lmstudio.errors import IndexingError
from rag_obsidian_lmstudio.index import Index, build_index, default_cache_dir, retrieve
from rag_obsidian_lmstudio.lmstudio import ModelPair


class FakeClient:
    """Embedding provider with controllable vectors and call recording."""

    def __init__(self, embed_model: str = "embed-x") -> None:
        self.embed_model = embed_model
        self.mapping: dict[str, list[float]] = {}
        self.calls: list[list[str]] = []

    def resolve_models(self) -> ModelPair:
        return ModelPair(chat="chat-x", embed=self.embed_model)

    def _vectors(self, texts: list[str]) -> np.ndarray:
        return np.asarray([self.mapping.get(t, [1.0, 0.0, 0.0]) for t in texts], dtype=np.float32)

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        self.calls.append(texts)
        return self._vectors(texts)

    def embed_queries(self, texts: list[str]) -> np.ndarray:
        return self._vectors(texts)


@pytest.fixture
def vault(tmp_path: Path) -> Path:
    docs = tmp_path / "vault"
    docs.mkdir()
    (docs / "alpha.md").write_text("alpha content")
    sub = docs / "sub"
    sub.mkdir()
    (sub / "beta.md").write_text("beta content")
    return docs


def build(docs: Path, cache: Path, client: FakeClient) -> Index:
    cfg = Config(docs_dir=docs)
    return build_index(client, cfg, cache_dir=cache)


class TestBuild:
    def test_indexes_all_markdown_files(self, vault: Path, tmp_path: Path) -> None:
        idx = build(vault, tmp_path / "cache", FakeClient())
        assert sorted(m[0] for m in idx.metas) == ["alpha.md", "sub/beta.md"]
        assert idx.vectors_norm.shape == (2, 3)

    def test_vectors_are_normalized(self, vault: Path, tmp_path: Path) -> None:
        client = FakeClient()
        client.mapping["alpha content"] = [3.0, 4.0, 0.0]
        idx = build(vault, tmp_path / "cache", client)
        norms = np.linalg.norm(idx.vectors_norm, axis=1)
        assert np.allclose(norms, 1.0, atol=1e-5)

    def test_non_markdown_files_ignored(self, vault: Path, tmp_path: Path) -> None:
        (vault / "notes.txt").write_text("not markdown")
        idx = build(vault, tmp_path / "cache", FakeClient())
        assert all(not m[0].endswith(".txt") for m in idx.metas)

    def test_empty_vault_raises_indexing_error(self, tmp_path: Path) -> None:
        empty = tmp_path / "empty"
        empty.mkdir()
        with pytest.raises(IndexingError, match=r"[Nn]o"):
            build(empty, tmp_path / "cache", FakeClient())

    def test_unreadable_file_is_skipped_with_warning(
        self, vault: Path, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        (vault / "bad.md").write_bytes(b"\xff\xfe\x00broken")
        with caplog.at_level(logging.WARNING):
            idx = build(vault, tmp_path / "cache", FakeClient())
        assert sorted(m[0] for m in idx.metas) == ["alpha.md", "sub/beta.md"]
        assert any("bad.md" in r.message for r in caplog.records)


class TestCacheReuse:
    def test_unchanged_files_are_not_reembedded(self, vault: Path, tmp_path: Path) -> None:
        cache = tmp_path / "cache"
        build(vault, cache, FakeClient())
        second = FakeClient()
        idx = build(vault, cache, second)
        assert second.calls == []
        assert len(idx.metas) == 2

    def test_round_trip_preserves_vectors_and_metas(self, vault: Path, tmp_path: Path) -> None:
        cache = tmp_path / "cache"
        client = FakeClient()
        client.mapping["alpha content"] = [0.0, 2.0, 0.0]
        first = build(vault, cache, client)
        second = build(vault, cache, FakeClient())
        assert second.metas == first.metas
        assert np.allclose(second.vectors_norm, first.vectors_norm)

    def test_modified_file_is_reembedded(self, vault: Path, tmp_path: Path) -> None:
        import os

        cache = tmp_path / "cache"
        build(vault, cache, FakeClient())
        alpha = vault / "alpha.md"
        alpha.write_text("alpha v2")
        os.utime(alpha, (0, 9_999_999_999))  # force distinct mtime
        second = FakeClient()
        idx = build(vault, cache, second)
        assert second.calls == [["alpha v2"]]
        assert ("alpha.md", "alpha v2") in idx.metas

    def test_deleted_file_is_evicted(self, vault: Path, tmp_path: Path) -> None:
        cache = tmp_path / "cache"
        build(vault, cache, FakeClient())
        (vault / "alpha.md").unlink()
        idx = build(vault, cache, FakeClient())
        assert [m[0] for m in idx.metas] == ["sub/beta.md"]
        # and it stays evicted from the persisted cache
        third = FakeClient()
        build(vault, cache, third)
        assert third.calls == []

    def test_embed_model_change_forces_full_reembed(self, vault: Path, tmp_path: Path) -> None:
        cache = tmp_path / "cache"
        build(vault, cache, FakeClient(embed_model="model-a"))
        second = FakeClient(embed_model="model-b")
        build(vault, cache, second)
        assert len(second.calls) == 2  # both files re-embedded


class TestCacheCorruption:
    """I2 regression: a damaged cache must never crash — always rebuild."""

    def _corrupt_and_rebuild(self, vault: Path, cache: Path, filename: str, data: bytes) -> None:
        build(vault, cache, FakeClient())
        target = next(cache.rglob(filename))
        target.write_bytes(data)
        second = FakeClient()
        idx = build(vault, cache, second)
        assert len(second.calls) == 2  # full rebuild
        assert len(idx.metas) == 2

    def test_corrupt_manifest_triggers_rebuild(self, vault: Path, tmp_path: Path) -> None:
        self._corrupt_and_rebuild(vault, tmp_path / "cache", "manifest.json", b"{ not json")

    def test_corrupt_vectors_triggers_rebuild(self, vault: Path, tmp_path: Path) -> None:
        self._corrupt_and_rebuild(vault, tmp_path / "cache", "vectors.npz", b"garbage")

    def test_missing_vectors_triggers_rebuild(self, vault: Path, tmp_path: Path) -> None:
        cache = tmp_path / "cache"
        build(vault, cache, FakeClient())
        next(cache.rglob("vectors.npz")).unlink()
        second = FakeClient()
        build(vault, cache, second)
        assert len(second.calls) == 2

    def test_count_mismatch_triggers_rebuild(self, vault: Path, tmp_path: Path) -> None:
        import json

        cache = tmp_path / "cache"
        build(vault, cache, FakeClient())
        manifest_path = next(cache.rglob("manifest.json"))
        manifest = json.loads(manifest_path.read_text())
        first_file = next(iter(manifest["files"]))
        manifest["files"][first_file]["chunks"].append("phantom chunk")
        manifest_path.write_text(json.dumps(manifest))
        second = FakeClient()
        build(vault, cache, second)
        assert len(second.calls) == 2


class TestRetrieve:
    def test_most_similar_chunk_ranks_first(self, vault: Path, tmp_path: Path) -> None:
        client = FakeClient()
        client.mapping["alpha content"] = [1.0, 0.0, 0.0]
        client.mapping["beta content"] = [0.0, 1.0, 0.0]
        client.mapping["about beta"] = [0.0, 1.0, 0.0]
        idx = build(vault, tmp_path / "cache", client)
        hits = retrieve(client, idx, "about beta", k=1)
        assert len(hits) == 1
        assert hits[0].path == "sub/beta.md"
        assert hits[0].chunk == "beta content"
        assert hits[0].score == pytest.approx(1.0, abs=1e-5)

    def test_k_larger_than_index_returns_all(self, vault: Path, tmp_path: Path) -> None:
        client = FakeClient()
        idx = build(vault, tmp_path / "cache", client)
        assert len(retrieve(client, idx, "anything", k=50)) == 2


class TestCacheDir:
    def test_distinct_vaults_get_distinct_cache_dirs(self, tmp_path: Path) -> None:
        a = default_cache_dir(tmp_path / "vault-a")
        b = default_cache_dir(tmp_path / "vault-b")
        assert a != b
        assert a.parent == b.parent

    def test_same_vault_same_cache_dir(self, tmp_path: Path) -> None:
        assert default_cache_dir(tmp_path) == default_cache_dir(tmp_path)

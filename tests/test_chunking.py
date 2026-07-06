from itertools import pairwise

import pytest

from rag_obsidian_lmstudio.chunking import chunk_markdown


class TestBasicSplitting:
    def test_empty_text_yields_no_chunks(self) -> None:
        assert chunk_markdown("") == []

    def test_whitespace_only_yields_no_chunks(self) -> None:
        assert chunk_markdown("  \n\n  ") == []

    def test_text_without_headings_is_one_chunk(self) -> None:
        text = "Just a paragraph.\n\nAnd another one."
        assert chunk_markdown(text) == [text]

    def test_heading_only_file(self) -> None:
        assert chunk_markdown("# Title") == ["# Title"]

    def test_h1_and_h2_start_new_chunks(self) -> None:
        text = "intro\n# One\nbody one\n## Two\nbody two"
        chunks = chunk_markdown(text)
        assert chunks == ["intro", "# One\nbody one", "## Two\nbody two"]

    def test_h3_does_not_split(self) -> None:
        text = "# One\nbody\n### Sub\nmore"
        assert chunk_markdown(text) == [text]

    def test_heading_keeps_its_body_for_retrieval_context(self) -> None:
        chunks = chunk_markdown("# Networks\nTCP handshake details")
        assert chunks[0].startswith("# Networks")
        assert "TCP handshake" in chunks[0]


class TestWindowing:
    def test_oversized_section_is_windowed_with_overlap(self) -> None:
        words = [f"w{i}" for i in range(25)]
        chunks = chunk_markdown(" ".join(words), chunk_words=10, overlap=3)
        # step = 7 → windows at 0, 7, 14, 21
        assert len(chunks) == 4
        assert chunks[0].split() == words[0:10]
        assert chunks[1].split() == words[7:17]
        assert chunks[3].split() == words[21:25]

    def test_consecutive_windows_share_overlap_words(self) -> None:
        words = [f"w{i}" for i in range(30)]
        chunks = chunk_markdown(" ".join(words), chunk_words=10, overlap=3)
        for a, b in pairwise(chunks):
            assert a.split()[-3:] == b.split()[:3]

    def test_every_chunk_nonempty_and_within_limit(self) -> None:
        text = "# H\n" + " ".join(f"w{i}" for i in range(1000))
        chunks = chunk_markdown(text, chunk_words=50, overlap=10)
        assert chunks
        for c in chunks:
            assert c.strip()
            assert len(c.split()) <= 50

    def test_exact_window_size_yields_single_chunk(self) -> None:
        text = " ".join(f"w{i}" for i in range(10))
        assert len(chunk_markdown(text, chunk_words=10, overlap=3)) == 1

    def test_overlap_must_be_smaller_than_window(self) -> None:
        with pytest.raises(ValueError, match="overlap"):
            chunk_markdown("some text", chunk_words=10, overlap=10)


class TestKnownLimitations:
    def test_hash_line_in_fenced_code_block_splits(self) -> None:
        """Documented limitation: '# ' inside a fenced code block is treated
        as a heading. Acceptable for note vaults; revisit if chunks look bad."""
        text = "# Real\nbody\n```\n# not a heading\ncode\n```"
        chunks = chunk_markdown(text)
        assert len(chunks) == 2  # currently splits inside the fence

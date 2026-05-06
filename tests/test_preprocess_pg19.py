"""Smoke tests for scripts/preprocess_pg19.py — the chunking helper.

The HF download path (`_pg19_token_iter`) is not exercised here — it
needs network and a gated model token. We test the pure chunking logic
(`chunk_token_stream_to_memmaps`) with synthetic per-document token
lists, which is what determines whether downstream PPL evaluation
sees correctly-sized chunks.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

# scripts/ is not a package — load by file path
SCRIPTS = Path(__file__).resolve().parent.parent / "scripts"
sys.path.insert(0, str(SCRIPTS))
import preprocess_pg19  # noqa: E402

chunk_token_stream_to_memmaps = preprocess_pg19.chunk_token_stream_to_memmaps


# ---------------------------------------------------------------------------
# Basic chunking math
# ---------------------------------------------------------------------------

class TestChunkTokenStreamToMemmaps:
    def test_single_doc_one_chunk(self, tmp_path):
        """A single doc whose token count == chunk_size produces 1 chunk."""
        meta = chunk_token_stream_to_memmaps(
            iter([list(range(100))]),
            out_dir=tmp_path, prefix="t", chunk_size=100,
        )
        assert len(meta["chunks"]) == 1
        assert meta["chunks"][0]["length"] == 100

    def test_single_doc_partial_chunk(self, tmp_path):
        """A doc shorter than chunk_size produces a partial chunk."""
        meta = chunk_token_stream_to_memmaps(
            iter([list(range(50))]),
            out_dir=tmp_path, prefix="t", chunk_size=100,
        )
        assert len(meta["chunks"]) == 1
        assert meta["chunks"][0]["length"] == 50

    def test_single_doc_multiple_chunks(self, tmp_path):
        """A doc much longer than chunk_size produces N+remainder chunks."""
        meta = chunk_token_stream_to_memmaps(
            iter([list(range(250))]),
            out_dir=tmp_path, prefix="t", chunk_size=100,
        )
        assert len(meta["chunks"]) == 3
        assert [c["length"] for c in meta["chunks"]] == [100, 100, 50]

    def test_multiple_docs_concatenated(self, tmp_path):
        """Multiple docs concatenate into the chunk stream — boundaries
        are NOT preserved (PG-19 PPL is over the streamed token corpus)."""
        meta = chunk_token_stream_to_memmaps(
            iter([list(range(60)), list(range(70)), list(range(70))]),
            out_dir=tmp_path, prefix="t", chunk_size=100,
        )
        # 60 + 70 + 70 = 200 → 2 chunks of 100, no remainder
        assert len(meta["chunks"]) == 2
        assert [c["length"] for c in meta["chunks"]] == [100, 100]

    def test_empty_input_no_chunks(self, tmp_path):
        meta = chunk_token_stream_to_memmaps(
            iter([]),
            out_dir=tmp_path, prefix="t", chunk_size=100,
        )
        assert meta["chunks"] == []


# ---------------------------------------------------------------------------
# Round-trip: chunks reassemble to the original token stream
# ---------------------------------------------------------------------------

class TestRoundTrip:
    def test_tokens_preserved_exactly(self, tmp_path):
        """Reading every chunk back and concatenating must equal the
        input token sequence — no padding, no reordering, int32 lossless."""
        docs = [list(range(60)), list(range(60, 130)), list(range(130, 250))]
        flat = [t for doc in docs for t in doc]
        meta = chunk_token_stream_to_memmaps(
            iter(docs), out_dir=tmp_path, prefix="t", chunk_size=100,
        )
        reassembled: list[int] = []
        for c in meta["chunks"]:
            mm = np.memmap(c["file"], dtype="int32", mode="r")
            reassembled.extend(mm.tolist())
        assert reassembled == flat

    def test_metadata_lengths_match_files(self, tmp_path):
        """meta["chunks"][i]["length"] must equal the on-disk file length."""
        meta = chunk_token_stream_to_memmaps(
            iter([list(range(250))]),
            out_dir=tmp_path, prefix="t", chunk_size=100,
        )
        for c in meta["chunks"]:
            mm = np.memmap(c["file"], dtype="int32", mode="r")
            assert mm.shape[0] == c["length"]

    def test_metadata_json_persisted(self, tmp_path):
        """metadata.json on disk must equal the returned meta dict."""
        meta = chunk_token_stream_to_memmaps(
            iter([list(range(150))]),
            out_dir=tmp_path, prefix="abc", chunk_size=100,
        )
        on_disk = json.loads((tmp_path / "abc_metadata.json").read_text())
        assert on_disk == meta


# ---------------------------------------------------------------------------
# Behavior at scale
# ---------------------------------------------------------------------------

class TestScale:
    def test_buffer_compaction_does_not_corrupt_tokens(self, tmp_path):
        """The internal buffer compaction triggers when start > 4 * chunk_size.
        Force it with many small docs and assert tokens still round-trip."""
        # chunk_size=10 → compaction threshold = 40
        # 10 docs of 8 tokens = 80 tokens → compaction fires after ~5 chunks
        docs = [list(range(i * 8, (i + 1) * 8)) for i in range(10)]
        flat = [t for doc in docs for t in doc]
        meta = chunk_token_stream_to_memmaps(
            iter(docs), out_dir=tmp_path, prefix="t", chunk_size=10,
        )
        reassembled: list[int] = []
        for c in meta["chunks"]:
            mm = np.memmap(c["file"], dtype="int32", mode="r")
            reassembled.extend(mm.tolist())
        assert reassembled == flat
        # 80 tokens / 10 chunk_size = exactly 8 chunks
        assert len(meta["chunks"]) == 8
        assert all(c["length"] == 10 for c in meta["chunks"])


# ---------------------------------------------------------------------------
# Output filenames
# ---------------------------------------------------------------------------

class TestFilenames:
    def test_prefix_and_index_in_filename(self, tmp_path):
        meta = chunk_token_stream_to_memmaps(
            iter([list(range(250))]),
            out_dir=tmp_path, prefix="pg19_validation", chunk_size=100,
        )
        for i, c in enumerate(meta["chunks"]):
            assert Path(c["file"]).name == f"pg19_validation_chunk_{i}.dat"
            assert Path(c["file"]).exists()


# ---------------------------------------------------------------------------
# Type guarantees
# ---------------------------------------------------------------------------

class TestDtype:
    def test_chunks_are_int32(self, tmp_path):
        meta = chunk_token_stream_to_memmaps(
            iter([[1, 2, 3, 4, 5]]),
            out_dir=tmp_path, prefix="t", chunk_size=10,
        )
        mm = np.memmap(meta["chunks"][0]["file"], dtype="int32", mode="r")
        assert mm.dtype == np.int32

    def test_token_ids_lossless(self, tmp_path):
        """Llama-2 vocab is 32000; gpt2 is 50257. Both fit comfortably in
        int32. Confirm round-trip on values up to 32000."""
        big = list(range(0, 32000, 137))  # spans the Llama-2 vocab
        meta = chunk_token_stream_to_memmaps(
            iter([big]),
            out_dir=tmp_path, prefix="t", chunk_size=10_000,
        )
        mm = np.memmap(meta["chunks"][0]["file"], dtype="int32", mode="r")
        assert mm.tolist() == big

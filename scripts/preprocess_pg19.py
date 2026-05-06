#!/usr/bin/env python3
"""Download PG-19 via `datasets` and chunk into N-token memmap files.

Writes int32 memmap files of length `chunk_size` (default 1_000_000) plus
a `pg19_{split}_metadata.json` describing every chunk's path + length.

Usage:
    python scripts/preprocess_pg19.py --split validation --limit 5
    python scripts/preprocess_pg19.py --tokenizer meta-llama/Llama-2-7b-hf
    python scripts/preprocess_pg19.py --chunk-size 65536  # 64k for ctx=64k

Notes:
    * The default tokenizer is the LLaMA-2 SentencePiece (vocab=32000).
      The pod experiments use the same tokenizer as the target model so
      PG-19 token IDs are directly compatible. Override with --tokenizer
      if you want gpt2 / a public model for offline tests.
    * --limit N processes only the first N documents from the stream.
      Useful for local verification (the full validation split has ~50
      books; train is too large to materialise locally).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable, Iterator

import numpy as np


def chunk_token_stream_to_memmaps(
    token_iter: Iterator[list[int]],
    out_dir: Path,
    prefix: str,
    chunk_size: int,
) -> dict:
    """Pure helper: stream per-document token lists into fixed-size
    int32 memmap chunks. Returns metadata listing each chunk's
    relative path + length.

    Each input element is one document's token list; the helper
    concatenates them into a single token stream and slices into
    `chunk_size`-token chunks. The final chunk may be shorter than
    `chunk_size` (it is *not* padded; downstream code reads memmaps
    of variable length).

    No tokenizer / dataset dependency — easy to unit-test with
    synthetic input.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    meta = {"chunks": []}
    chunk_idx = 0
    buffer: list[int] = []

    def _flush(arr: np.ndarray) -> None:
        nonlocal chunk_idx
        fname = out_dir / f"{prefix}_chunk_{chunk_idx}.dat"
        mm = np.memmap(fname, dtype="int32", mode="w+", shape=(arr.shape[0],))
        mm[:] = arr[:]
        mm.flush()
        meta["chunks"].append({"file": str(fname), "length": int(arr.shape[0])})
        chunk_idx += 1

    # Use a list buffer + index so we don't pay O(n) on `buffer = buffer[k:]`
    # at large chunk sizes. We slice from `start` instead of mutating.
    start = 0
    for tokens in token_iter:
        buffer.extend(tokens)
        while len(buffer) - start >= chunk_size:
            arr = np.fromiter(
                (buffer[i] for i in range(start, start + chunk_size)),
                dtype=np.int32, count=chunk_size,
            )
            _flush(arr)
            start += chunk_size
        # Periodic compaction so buffer doesn't grow without bound
        if start > chunk_size * 4:
            del buffer[:start]
            start = 0

    # Final partial chunk
    remainder = buffer[start:]
    if remainder:
        arr = np.array(remainder, dtype=np.int32)
        _flush(arr)

    meta_path = out_dir / f"{prefix}_metadata.json"
    with meta_path.open("w") as fh:
        json.dump(meta, fh, indent=2)

    return meta


def _pg19_token_iter(split: str, tokenizer_name: str, limit: int | None):
    """Yield per-document token lists from the PG-19 dataset (streaming)."""
    from datasets import load_dataset
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, use_fast=True)
    ds = load_dataset("pg19", split=split, streaming=True)
    for i, example in enumerate(ds):
        if limit is not None and i >= limit:
            break
        text = example.get("text") or example.get("text_plain") or example.get("content")
        if not text:
            continue
        yield tokenizer.encode(text, add_special_tokens=False)


def preprocess(split: str, tokenizer_name: str, out_dir: str | Path,
               chunk_size: int, limit: int | None = None) -> dict:
    """Top-level driver: stream PG-19 → memmap chunks. Returns metadata."""
    return chunk_token_stream_to_memmaps(
        _pg19_token_iter(split, tokenizer_name, limit),
        Path(out_dir),
        prefix=f"pg19_{split}",
        chunk_size=chunk_size,
    )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--tokenizer", default="meta-llama/Llama-2-7b-hf",
                   help="Tokenizer name or path (default: Llama-2-7b-hf — "
                        "matches the pod experiments' target/draft vocab)")
    p.add_argument("--out", default="data/processed/pg19", help="Output directory")
    p.add_argument("--split", default="validation",
                   help="PG-19 split: train/validation/test "
                        "(default: validation — small enough to materialise locally)")
    p.add_argument("--chunk-size", type=int, default=1_000_000,
                   help="Chunk size in tokens (default: 1M to match 1M-context M4 target)")
    p.add_argument("--limit", type=int, default=None,
                   help="Process only the first N documents (default: all). "
                        "Useful for local verification.")
    args = p.parse_args()

    meta = preprocess(args.split, args.tokenizer, args.out,
                      args.chunk_size, args.limit)
    print(f"Wrote {len(meta['chunks'])} chunk file(s) to {args.out}")
    total = sum(c["length"] for c in meta["chunks"])
    print(f"Total tokens: {total:,}")


if __name__ == "__main__":
    main()

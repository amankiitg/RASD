#!/usr/bin/env python3
"""Download PG-19 via `datasets` and chunk into 1M-token memmap files.

Writes token id memmap files and a `metadata.json` describing chunks.
"""
import argparse
import json
import os
from pathlib import Path

import numpy as np
from datasets import load_dataset
from transformers import AutoTokenizer


def chunk_tokens_to_memmap(tokens, out_path, prefix, chunk_size=1_000_000):
    out_path = Path(out_path)
    out_path.mkdir(parents=True, exist_ok=True)
    meta = {"chunks": []}
    buffer = []
    chunk_i = 0

    def flush_full_chunk(buf):
        nonlocal chunk_i
        arr = np.array(buf[:chunk_size], dtype=np.int32)
        fname = out_path / f"{prefix}_chunk_{chunk_i}.dat"
        mm = np.memmap(fname, dtype='int32', mode='w+', shape=(arr.shape[0],))
        mm[:] = arr[:]
        mm.flush()
        meta["chunks"].append({"file": str(fname), "length": int(arr.shape[0])})
        chunk_i += 1

    # extend buffer with input tokens and flush as needed
    buffer.extend(tokens)
    while len(buffer) >= chunk_size:
        flush_full_chunk(buffer)
        buffer = buffer[chunk_size:]

    # final smaller chunk
    if len(buffer) > 0:
        arr = np.array(buffer, dtype=np.int32)
        fname = out_path / f"{prefix}_chunk_{chunk_i}.dat"
        mm = np.memmap(fname, dtype='int32', mode='w+', shape=(arr.shape[0],))
        mm[:] = arr[:]
        mm.flush()
        meta["chunks"].append({"file": str(fname), "length": int(arr.shape[0])})

    # save metadata
    with open(out_path / f"{prefix}_metadata.json", "w") as fh:
        json.dump(meta, fh, indent=2)

    return meta


def preprocess(split: str, tokenizer_name: str, out_dir: str, chunk_size: int):
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, use_fast=True)
    ds = load_dataset("pg19", split=split, streaming=True)

    # accumulate tokens streaming, write chunks periodically
    tokens_buffer = []
    chunk_count = 0
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for example in ds:
        text = example.get("text") or example.get("text_plain") or example.get("content")
        if not text:
            continue
        encoded = tokenizer.encode(text, add_special_tokens=False)
        tokens_buffer.extend(encoded)
        # flush while we have at least one full chunk
        while len(tokens_buffer) >= chunk_size:
            arr = np.array(tokens_buffer[:chunk_size], dtype=np.int32)
            fname = out_dir / f"pg19_{split}_chunk_{chunk_count}.dat"
            mm = np.memmap(fname, dtype='int32', mode='w+', shape=(arr.shape[0],))
            mm[:] = arr[:]
            mm.flush()
            chunk_count += 1
            tokens_buffer = tokens_buffer[chunk_size:]

    # write remainder
    if len(tokens_buffer) > 0:
        arr = np.array(tokens_buffer, dtype=np.int32)
        fname = out_dir / f"pg19_{split}_chunk_{chunk_count}.dat"
        mm = np.memmap(fname, dtype='int32', mode='w+', shape=(arr.shape[0],))
        mm[:] = arr[:]
        mm.flush()

    # write simple metadata listing
    meta = {"chunks": []}
    for p in sorted(out_dir.glob(f"pg19_{split}_chunk_*.dat")):
        mm = np.memmap(p, dtype='int32', mode='r')
        meta["chunks"].append({"file": str(p), "length": int(mm.shape[0])})

    with open(out_dir / f"pg19_{split}_metadata.json", "w") as fh:
        json.dump(meta, fh, indent=2)

    print(f"Wrote {len(meta['chunks'])} chunk files to {out_dir}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--tokenizer", default="gpt2", help="Tokenizer name or path (default: gpt2)")
    p.add_argument("--out", default="data/processed/pg19", help="Output directory")
    p.add_argument("--split", default="train", help="PG-19 split: train/validation/test")
    p.add_argument("--chunk-size", type=int, default=1_000_000, help="Chunk size in tokens")
    args = p.parse_args()

    preprocess(args.split, args.tokenizer, args.out, args.chunk_size)


if __name__ == "__main__":
    main()

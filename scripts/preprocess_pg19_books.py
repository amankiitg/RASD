#!/usr/bin/env python3
"""Scan PG-19 and record per-book token counts.

Answers one question: how many *distinct books* are long enough to serve as a
document at each rung? The published PG-19 pipeline concatenated documents into
1M-token chunks, which erases book boundaries; a rung built from offsets into a
concatenated stream is not 10 independent documents.

Streams the corpus (no full materialisation), tokenizes each book, and writes
`book_lengths.json` with one row per book.

Usage:
    python scripts/preprocess_pg19_books.py --tokenizer meta-llama/Llama-3.1-8B
    python scripts/preprocess_pg19_books.py --min-tokens 131072 --want 10
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def scan(tokenizer_name: str, split: str, out_path: Path,
         dataset_name: str = "emozilla/pg19", limit: int | None = None,
         min_chars: int = 0) -> dict:
    """Tokenize each book and record its length.

    `min_chars` pre-screens on character count before tokenizing: text is
    already in memory so `len()` is free, while `encode()` is not. Screening
    first is what makes a scan of the train split tractable.
    """
    from datasets import load_dataset
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(tokenizer_name, use_fast=True)
    ds = load_dataset(dataset_name, split=split, streaming=True)

    books = []
    seen = skipped = 0
    for i, ex in enumerate(iter(ds)):
        if limit is not None and seen >= limit:
            break
        seen += 1
        text = ex.get("text") or ""
        if not text:
            continue
        if min_chars and len(text) < min_chars:
            skipped += 1
            continue
        n = len(tok.encode(text, add_special_tokens=False))
        books.append({
            "index": i,
            "title": ex.get("short_book_title"),
            "url": ex.get("url"),
            "publication_date": ex.get("publication_date"),
            "chars": len(text),
            "tokens": n,
        })
        print(f"  [{i:5d}] {n:>9,} tok  {len(text):>10,} ch  "
              f"{(ex.get('short_book_title') or '')[:52]}", flush=True)

    out = {"tokenizer": tokenizer_name, "split": split,
           "dataset": dataset_name, "limit": limit,
           "min_chars": min_chars, "books_scanned": seen,
           "books_screened_out": skipped, "books": books}
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2))
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--tokenizer", default="meta-llama/Llama-3.1-8B")
    p.add_argument("--split", default="validation")
    p.add_argument("--dataset", default="emozilla/pg19")
    p.add_argument("--out", default="data/processed/pg19_books/book_lengths.json")
    p.add_argument("--min-tokens", type=int, default=None,
                   help="Also print a count of books at or above this length")
    p.add_argument("--limit", type=int, default=None,
                   help="Scan at most N books (streaming; for bounded probes)")
    p.add_argument("--min-chars", type=int, default=0,
                   help="Skip books shorter than this many chars without "
                        "tokenizing them (cheap pre-screen)")
    p.add_argument("--want", type=int, default=10)
    args = p.parse_args()

    res = scan(args.tokenizer, args.split, Path(args.out), args.dataset,
               args.limit, args.min_chars)
    books = sorted(res["books"], key=lambda b: -b["tokens"])
    print(f"\n{len(books)} books scanned "
          f"({res['books_screened_out']} screened out on chars); wrote {args.out}")

    print("\nlongest books:")
    for b in books[:15]:
        print(f"  {b['tokens']:>9,} tok  {(b['title'] or '')[:58]}")

    if args.min_tokens:
        ok = [b for b in books if b["tokens"] >= args.min_tokens]
        print(f"\nbooks with >= {args.min_tokens:,} tokens: {len(ok)} "
              f"(need {args.want} for a rung) -> "
              f"{'OK' if len(ok) >= args.want else 'INSUFFICIENT'}")


if __name__ == "__main__":
    main()

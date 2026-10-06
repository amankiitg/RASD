#!/usr/bin/env python3
"""Build the per-book document pool the rungs consume.

Why this exists rather than reusing `preprocess_pg19.py`: that script
concatenates documents into fixed-size chunks, which erases book boundaries. A
rung built from offsets into a concatenated stream is not "10 distinct
documents", and a slice can straddle two books. Here **one book is one memmap**,
so the document is a real unit and its prompt is provably a contiguous span of a
single book.

Selection (deterministic, RNG-free): order the books eligible at the largest
rung by descending token count and take the longest `--documents` of them. The
plan fixes the rule; this script does not choose.

Emits, under `--out`:
  doc_000.dat ... doc_00N.dat   int32 memmaps, one per book
  documents.json                pool metadata + per-rung prompt hashes

The prompt hash per (rung, document) is computed here, from the same token IDs
the engine will see, so a result can be tied to its exact input.

Usage:
    python scripts/mlsys_build_documents.py \
        --lengths data/processed/pg19_books/book_lengths_train.json \
        --out data/processed/pg19_docs --rungs 131072 262144 524288
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

# Tokens the model generates per run. The scored sequence is prompt
# (C - GEN) + continuation (GEN) = exactly C, so a document needs C tokens.
GEN = 1024


def spread(books: list[dict], need_docs: int, seed: int) -> list[dict]:
    """A seeded uniform sample without replacement from an eligible pool.

    The paired design takes the LONGEST books, which at 512k leaves only 14
    candidates and makes the set Bible- and omnibus-heavy. This is the other
    selection: an actual random sample of the eligible pool, so a rung's
    headline number can be checked against one that is not an artefact of the
    set the paired design is forced into. Seeded, so it is reproducible.
    """
    import numpy as np
    if len(books) < need_docs:
        raise RuntimeError(
            f"need {need_docs} documents, pool has {len(books)}"
        )
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(books), size=need_docs, replace=False)
    return [books[int(i)] for i in sorted(idx)]


def eligible(books: list[dict], max_rung: int, need_docs: int) -> list[dict]:
    """Books that can serve every rung up to `max_rung`, longest first.

    A document at context C must supply C tokens: C - GEN for the prompt and
    GEN for the scored continuation, which is a single contiguous forward of
    exactly C. Taking the longest first means the selected set also has the
    most headroom at the top rung.
    """
    # A book must cover the rung plus a generation, so that after the
    # source-length adjustment there is still a full continuation left.
    need = max_rung + GEN
    ok = [b for b in books if b["tokens"] >= need]
    ok.sort(key=lambda b: -b["tokens"])
    if len(ok) < need_docs:
        raise RuntimeError(
            f"need {need_docs} documents of >= {need:,} tokens, found {len(ok)} "
            f"in {len(books)} scanned books; the rung set is not feasible"
        )
    return ok


def build(lengths_path: Path, out_dir: Path, rungs: list[int], n_docs: int,
          tokenizer_name: str, split: str, dataset_name: str,
          selection: str = "longest", seed: int = 20261006) -> dict:
    import numpy as np
    from datasets import load_dataset
    from transformers import AutoTokenizer

    lengths = json.loads(Path(lengths_path).read_text())
    max_rung = max(rungs)
    # `eligible` returns the whole sorted pool and checks feasibility against
    # the number of documents actually needed, so pass n_docs, not the pool size.
    pool = eligible(lengths["books"], max_rung, n_docs)
    if selection == "longest":
        chosen = pool[:n_docs]
    elif selection == "spread":
        chosen = spread(pool, n_docs, seed)
    else:
        raise ValueError(f"unknown selection {selection!r}")
    want = {b["index"]: b for b in chosen}
    # Memmaps carry a full generation of tail headroom beyond the largest rung.
    #
    # This is not slack. The engine re-tokenises the prompt STRING, and for some
    # books the source slice must be LONGER than the prompt it produces (measured
    # drift on this pool: up to ~1100 tokens at the 512k rung, from tokens that
    # decode to text re-encoding to more tokens than they came from). The
    # continuation is then taken after that longer source slice, so a pool capped
    # at exactly `max_rung + GEN` leaves too little room and the last rung fails
    # with a short continuation. The cap is therefore `max_rung + 2 * GEN`, and a
    # book must supply at least `max_rung + GEN` to be eligible.
    max_len = max_rung + 2 * GEN
    print(f"selected {len(chosen)} books (>= {max_len:,} tokens), longest "
          f"{chosen[0]['tokens']:,}, shortest {chosen[-1]['tokens']:,}")

    out_dir.mkdir(parents=True, exist_ok=True)
    tok = AutoTokenizer.from_pretrained(tokenizer_name, use_fast=True)
    ds = load_dataset(dataset_name, split=split, streaming=True)

    docs: list[dict] = []
    for i, ex in enumerate(iter(ds)):
        if i > max(want):
            break
        if i not in want:
            continue
        ids = tok.encode(ex.get("text") or "", add_special_tokens=False)
        if len(ids) < max_len:
            # Disagrees with the lengths scan; fail rather than emit a short
            # document, which would silently break the prompt/continuation split.
            raise RuntimeError(
                f"book {i} tokenized to {len(ids):,}, expected >= {max_len:,} "
                f"from {lengths_path}"
            )
        ids = ids[:max_len]
        slot = len(docs)
        fname = out_dir / f"doc_{slot:03d}.dat"
        arr = np.asarray(ids, dtype=np.int32)
        mm = np.memmap(fname, dtype="int32", mode="w+", shape=arr.shape)
        mm[:] = arr[:]
        mm.flush()
        del mm
        docs.append({
            "doc_id": f"pg19_{split}_{i}",
            "slot": slot,
            "book_index": i,
            "title": ex.get("short_book_title"),
            "url": ex.get("url"),
            "file": str(fname),
            "length": int(arr.shape[0]),
            "book_tokens": want[i]["tokens"],
        })
        print(f"  [{slot:2d}] {arr.shape[0]:>9,} tok  {docs[-1]['title'][:54]}",
              flush=True)

    if len(docs) != len(chosen):
        raise RuntimeError(
            f"materialized {len(docs)} of {len(chosen)} selected books"
        )

    # Per-rung prompt provenance: hash the exact prompt token IDs.
    rung_meta = {}
    for ctx in rungs:
        p_len = ctx - GEN
        per_doc = {}
        for d in docs:
            arr = np.memmap(d["file"], dtype="int32", mode="r")
            pids = arr[:p_len].astype(int).tolist()
            cont = arr[ctx:ctx + GEN].astype(int).tolist()
            per_doc[d["doc_id"]] = {
                "prompt_tokens": p_len,
                "prompt_sha256": hashlib.sha256(
                    ",".join(map(str, pids)).encode()).hexdigest(),
                "continuation_sha256": hashlib.sha256(
                    ",".join(map(str, cont)).encode()).hexdigest(),
            }
            del arr
        rung_meta[str(ctx)] = per_doc

    meta = {
        "tokenizer": tokenizer_name,
        "dataset": dataset_name,
        "split": split,
        "gen_tokens": GEN,
        "rungs": sorted(rungs),
        "selection": (
            f"{selection}: {len(docs)} books from {len(pool)} eligible "
            f"(>= {max_rung:,} tokens required, prompt + continuation = "
            f"context)"
            + (f", seed {seed}" if selection == "spread" else "")
        ),
        "selection_mode": selection,
        "selection_seed": seed if selection == "spread" else None,
        "documents": docs,
        "per_rung": rung_meta,
    }
    (out_dir / "documents.json").write_text(json.dumps(meta, indent=2))
    print(f"\nwrote {out_dir}/documents.json ({len(docs)} documents)")
    uniq = {r: len({v['prompt_sha256'] for v in m.values()})
            for r, m in rung_meta.items()}
    print(f"unique prompt hashes per rung: {uniq}")
    return meta


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--lengths",
                   default="data/processed/pg19_books/book_lengths_train.json")
    p.add_argument("--out", default="data/processed/pg19_docs")
    p.add_argument("--rungs", type=int, nargs="+",
                   default=[131072, 262144, 524288])
    p.add_argument("--documents", type=int, default=10)
    p.add_argument("--selection", choices=["longest", "spread"], default="longest",
                   help="longest = the paired core set; spread = a seeded "
                        "uniform sample of the eligible pool")
    p.add_argument("--seed", type=int, default=20261006)
    p.add_argument("--tokenizer", default="meta-llama/Llama-3.1-8B")
    p.add_argument("--split", default="train")
    p.add_argument("--dataset", default="emozilla/pg19")
    args = p.parse_args()
    build(Path(args.lengths), Path(args.out), args.rungs, args.documents,
          args.tokenizer, args.split, args.dataset, args.selection, args.seed)


if __name__ == "__main__":
    main()

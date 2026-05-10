#!/usr/bin/env python3
"""PG-19 perplexity sanity check for the M4 paper.

Computes target-model PPL on PG-19 validation chunks under our M4
configuration (NF4 4-bit weights via bitsandbytes). Compares against
a published / baseline number to sanity-check that our quantization
+ long-context infra has not degraded language modelling quality.

Scope (M4):
  * Single-process. We run PPL at moderate contexts (4k / 8k / 16k /
    32k) where Llama-2-7B + bf16 KV fits a single 40 GB GPU. This
    catches the most common failure mode (NF4 weights destroyed
    training-time NLL).
  * Long-context PPL (128k+) requires sequence-parallel forward
    through the RASD ring attention path, which is an order of
    magnitude more complex to wire up here. That's deferred to a
    follow-up paper iteration; documented as a known gap.

Output: CSV with columns
    timestamp,seed,context_length,quantize_target,ppl,eval_time_sec,n_tokens

Usage on pod (after PG-19 is preprocessed):
    python scripts/preprocess_pg19.py \\
        --split validation \\
        --limit 5 \\
        --tokenizer meta-llama/Llama-2-7b-hf \\
        --chunk-size 65536

    python scripts/eval_perplexity_matrix.py \\
        --target meta-llama/Llama-2-7b-hf \\
        --contexts 4096 8192 16384 32768 \\
        --seeds 42 \\
        --quantize-target \\
        --pg19-meta data/processed/pg19_validation_metadata.json \\
        --out results/perplexity/m4_ppl.csv
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

# Make sure src/ is importable when run from repo root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.analysis.perplexity import compute_perplexity  # noqa: E402


CSV_HEADER = [
    "timestamp",
    "seed",
    "context_length",
    "quantize_target",
    "ppl",
    "eval_time_sec",
    "n_tokens",
    "stride",
    "model",
]


def _load_target(name: str, revision: str | None, quantize: bool, dtype: torch.dtype):
    """Load the target Llama as RASD does (NF4 weights when quantize=True)."""
    bnb = None
    if quantize and torch.cuda.is_available():
        bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=dtype)
    model = AutoModelForCausalLM.from_pretrained(
        name,
        revision=revision,
        torch_dtype=dtype,
        quantization_config=bnb,
        device_map={"": 0} if torch.cuda.is_available() else None,
    )
    model.eval()
    return model


def _load_pg19_chunk(meta_path: Path, target_len: int, seed: int) -> torch.Tensor:
    """Read a PG-19 chunk and slice/pad to exactly target_len tokens.

    Picks a chunk pseudorandomly seeded from `seed` so results are
    reproducible across seeds. Returns (1, target_len) int64 tensor.
    """
    meta = json.loads(meta_path.read_text())
    chunks = meta["chunks"]
    if not chunks:
        raise RuntimeError(f"{meta_path}: no chunks in metadata")
    # Pick the longest chunk that has enough tokens; if none, fall
    # back to the longest chunk and concatenate as needed.
    suitable = [c for c in chunks if c["length"] >= target_len]
    rng = np.random.default_rng(seed)
    if suitable:
        c = suitable[rng.integers(0, len(suitable))]
        arr = np.memmap(meta_path.parent / c["path"], dtype="int32", mode="r")
        # Pick a random start within the chunk
        start = int(rng.integers(0, c["length"] - target_len + 1))
        ids = np.array(arr[start:start + target_len], dtype=np.int64)
    else:
        # Concatenate chunks until target_len reached
        joined = []
        for c in chunks:
            arr = np.memmap(meta_path.parent / c["path"], dtype="int32", mode="r")
            joined.append(np.array(arr, dtype=np.int64))
            if sum(len(x) for x in joined) >= target_len:
                break
        ids = np.concatenate(joined)[:target_len]
    return torch.from_numpy(ids).unsqueeze(0)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--target", default="meta-llama/Llama-2-7b-hf")
    p.add_argument("--revision", default=None)
    p.add_argument("--contexts", type=int, nargs="+",
                   default=[4096, 8192, 16384, 32768],
                   help="PPL evaluation context lengths (single-rank fits "
                        "Llama-2-7B + bf16 KV up to ~32k on 40GB; longer "
                        "needs sequence parallelism, deferred future work)")
    p.add_argument("--seeds", type=int, nargs="+", default=[42, 123, 456])
    p.add_argument("--quantize-target", action="store_true",
                   help="Load target with NF4 4-bit weights (matches RASD config). "
                        "Without this flag, loads bf16 weights as a comparison baseline.")
    p.add_argument("--dtype", default="bfloat16",
                   choices=["bfloat16", "float16", "float32"])
    p.add_argument("--pg19-meta", required=True,
                   help="Path to pg19_<split>_metadata.json from preprocess_pg19.py")
    p.add_argument("--out", default="results/perplexity/m4_ppl.csv")
    p.add_argument("--stride", type=int, default=None,
                   help="PPL sliding-window stride (default ctx/2)")
    args = p.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("eval_perplexity_matrix requires CUDA (NF4 weights need bnb)")

    dtype = getattr(torch, args.dtype)
    print(f"Loading target: {args.target} quantize={args.quantize_target}")
    model = _load_target(args.target, args.revision, args.quantize_target, dtype)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not out_path.exists()
    with open(out_path, "a", newline="") as fh:
        w = csv.writer(fh)
        if write_header:
            w.writerow(CSV_HEADER)

        for seed in args.seeds:
            for ctx in args.contexts:
                # Defensive cap: skip cells that will OOM single-rank
                # (Llama-2-7B + bf16 KV at 64k is already 32 GB; on
                # 40GB pod that's the limit).
                if ctx > 32768 and not args.quantize_target:
                    print(f"  [skip] seed={seed} ctx={ctx} bf16 baseline OOMs single-rank past 32k")
                    continue

                print(f"  [eval] seed={seed} ctx={ctx} quant={args.quantize_target}")
                ids = _load_pg19_chunk(
                    Path(args.pg19_meta), target_len=ctx, seed=seed,
                ).to("cuda:0")
                t0 = time.perf_counter()
                ppl = compute_perplexity(
                    model, ids, max_length=ctx, stride=args.stride,
                )
                t1 = time.perf_counter()
                w.writerow([
                    datetime.now(timezone.utc).isoformat(),
                    seed,
                    ctx,
                    args.quantize_target,
                    f"{ppl:.4f}",
                    f"{t1 - t0:.2f}",
                    ids.shape[1],
                    args.stride or ctx // 2,
                    args.target,
                ])
                fh.flush()
                print(f"    ppl={ppl:.4f}  time={t1-t0:.2f}s")

    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Vanilla HF Llama-2-7B + FA-2 generate() ceiling baseline.

Single-rank, single-seed OOM-ceiling test for the M4 paper. Documents
the contention "vanilla HF generate cannot reach long context on
commodity hardware" without sequence parallelism or KV quantization.

Expected outcome (Llama-2-7B, bf16 weights + bf16 KV):
  *  32k context:  passes (~30 GB peak, fits 80GB)
  * 128k context:  borderline (~78 GB; may pass or OOM)
  * 256k context:  OOMs (KV alone is 128 GB)
  * 512k context:  OOMs
  *   1M context:  OOMs

The point is the 32k/128k clean tok/s number (apples-to-apples vs
RASD at the only context where vanilla HF fits) plus the explicit
OOM evidence for ≥256k. NO sequence parallelism, NO bnb 4-bit
weights — that's the "vanilla" setup the paper is contrasting RASD
against.

Output csv:
    timestamp,context_length,seed,attn_impl,quantize_target,status,
    time_sec,throughput_tps,tokens_generated,gpu_peak_mem_mb,error

`status` is one of: ok | oom | runtime_error.

Usage on pod:
    python scripts/benchmark_hf_baseline.py \\
        --target meta-llama/Llama-2-7b-hf \\
        --contexts 32768 131072 262144 524288 1048576 \\
        --max-new-tokens 64 \\
        --seed 42 \\
        --out results/baselines/hf_ceiling.csv
"""
from __future__ import annotations

import argparse
import csv
import os
import time
from datetime import datetime, timezone
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


CSV_HEADER = [
    "timestamp",
    "context_length",
    "seed",
    "attn_impl",
    "quantize_target",
    "status",
    "time_sec",
    "throughput_tps",
    "tokens_generated",
    "gpu_peak_mem_mb",
    "error",
]


def _load_target(name: str, revision, dtype, attn_impl, quantize):
    """Load Llama-2-7B for the ceiling test. Vanilla = bf16 weights,
    FA-2 attention, single-rank. Optionally NF4 weights for the
    "vanilla + 4-bit" variant if --quantize-target is passed."""
    bnb = None
    if quantize:
        bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=dtype)
    model = AutoModelForCausalLM.from_pretrained(
        name,
        revision=revision,
        torch_dtype=dtype,
        attn_implementation=attn_impl,
        quantization_config=bnb,
        device_map={"": 0},
    )
    model.eval()
    return model


def _build_prompt_ids(tokenizer, ctx: int, seed: int):
    """Build a synthetic prompt of approximately ctx tokens via the
    same template run_experiment.py uses for matrix runs (ensures the
    HF baseline is fed analogous input)."""
    template = (
        "The quick brown fox jumps over the lazy dog. "
        "Pack my box with five dozen liquor jugs. "
    )
    rep = max(1, ctx // len(tokenizer.encode(template)) + 1)
    text = template * rep
    ids = tokenizer.encode(text)[:ctx]
    return torch.tensor([ids], dtype=torch.long, device="cuda:0")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--target", default="meta-llama/Llama-2-7b-hf")
    p.add_argument("--revision", default=None)
    p.add_argument("--contexts", type=int, nargs="+",
                   default=[32_768, 131_072, 262_144, 524_288, 1_048_576])
    p.add_argument("--max-new-tokens", type=int, default=64)
    p.add_argument("--seed", type=int, default=42,
                   help="Single seed (this is a ceiling test, not a 3-seed eval)")
    p.add_argument("--attn-impl", default="flash_attention_2",
                   choices=["flash_attention_2", "sdpa", "eager"])
    p.add_argument("--quantize-target", action="store_true",
                   help="Optional: load with NF4 4-bit weights. Default False "
                        "(true 'vanilla HF' = bf16 weights). With NF4, the "
                        "ceiling shifts up by ~10 GB so 128k may pass cleanly.")
    p.add_argument("--dtype", default="bfloat16",
                   choices=["bfloat16", "float16"])
    p.add_argument("--out", default="results/baselines/hf_ceiling.csv")
    args = p.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required")

    dtype = getattr(torch, args.dtype)
    print(f"Loading {args.target} (attn_impl={args.attn_impl}, quantize={args.quantize_target})")
    model = _load_target(
        args.target, args.revision, dtype, args.attn_impl, args.quantize_target,
    )
    tokenizer = AutoTokenizer.from_pretrained(args.target, revision=args.revision)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not out_path.exists()
    with open(out_path, "a", newline="") as fh:
        w = csv.writer(fh)
        if write_header:
            w.writerow(CSV_HEADER)
        fh.flush()

        torch.manual_seed(args.seed)
        for ctx in args.contexts:
            print(f"\n[ctx={ctx:,}]")
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.empty_cache()

            try:
                input_ids = _build_prompt_ids(tokenizer, ctx, args.seed)
                t0 = time.perf_counter()
                with torch.no_grad():
                    out = model.generate(
                        input_ids,
                        max_new_tokens=args.max_new_tokens,
                        do_sample=False,
                        use_cache=True,
                    )
                torch.cuda.synchronize()
                t1 = time.perf_counter()

                tokens_gen = out.shape[1] - input_ids.shape[1]
                elapsed = t1 - t0
                peak_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)
                print(f"  ok: {tokens_gen} tokens in {elapsed:.2f}s "
                      f"(peak={peak_mb:.0f} MB, {tokens_gen/elapsed:.3f} tok/s)")
                w.writerow([
                    datetime.now(timezone.utc).isoformat(),
                    ctx, args.seed, args.attn_impl, args.quantize_target,
                    "ok", f"{elapsed:.4f}",
                    f"{tokens_gen/elapsed:.4f}" if elapsed > 0 else "0.0",
                    tokens_gen, f"{peak_mb:.2f}", "",
                ])
            except torch.cuda.OutOfMemoryError as e:
                msg = str(e).split("\n", 1)[0][:160]
                print(f"  OOM: {msg}")
                w.writerow([
                    datetime.now(timezone.utc).isoformat(),
                    ctx, args.seed, args.attn_impl, args.quantize_target,
                    "oom", "-1.0", "-1.0", "0", "-1.0", msg,
                ])
                # OOM corrupts the GPU state for subsequent allocations;
                # explicitly empty cache so the next ctx (which we EXPECT
                # to OOM at deeper depth) at least starts clean.
                torch.cuda.empty_cache()
            except RuntimeError as e:
                msg = str(e).split("\n", 1)[0][:160]
                print(f"  runtime_err: {msg}")
                w.writerow([
                    datetime.now(timezone.utc).isoformat(),
                    ctx, args.seed, args.attn_impl, args.quantize_target,
                    "runtime_error", "-1.0", "-1.0", "0", "-1.0", msg,
                ])
                torch.cuda.empty_cache()
            fh.flush()

    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    raise SystemExit(main())

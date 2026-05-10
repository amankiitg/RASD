#!/usr/bin/env python3
"""Benchmark RingAttention and SlidingWindowAttention at different context lengths.

IMPORTANT — THIS METRIC IS NOT GENERATION THROUGHPUT (Phase C blocker
#6, 2026-05-10). The CSV column is `forward_tps` (forward-pass tokens
per second from a single attention forward), NOT `throughput_tps` (the
column name in run_experiment.py results, which is RASD's generation
tokens per second from a full speculative decoding loop). Different
units. Phase D Figure 1 must NOT overlay these columns on the same
y-axis without an explicit caption note.

Produces a CSV at `results/baselines/baselines.csv` with per-run timing,
forward-pass throughput, and latency metrics required by Milestone 2.

Usage (single GPU / CPU):
    python scripts/benchmark_baselines.py

Usage (multi-GPU via torchrun, e.g. 8 GPUs):
    torchrun --nproc_per_node=8 scripts/benchmark_baselines.py --distributed

Columns written to CSV:
    timestamp, baseline, context_length, seed, device, world_size,
    time_s, forward_tps, latency_ms
"""
import argparse
import csv
import os
import sys
import time
from datetime import datetime, timezone
from typing import Optional

import torch
import torch.nn as nn

# Allow running from project root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.baselines.ring_attention import RingAttention
from src.baselines.sliding_window import SlidingWindowAttention


# ---------------------------------------------------------------------------
# Timing helpers
# ---------------------------------------------------------------------------

def _sync(device: torch.device):
    if device.type == "cuda":
        torch.cuda.synchronize()


def benchmark_module(
    mod: nn.Module,
    seq_len: int,
    dim: int,
    device: torch.device,
    runs: int = 3,
    total_len: Optional[int] = None,
) -> dict:
    """Run `mod` for `runs` forward passes and return timing stats.

    For distributed runs the input is a local shard (`seq_len` already
    divided by `world_size` by the caller). Pass `total_len` so that
    throughput is reported as total-context tokens/sec, not per-shard
    — this matches RASD's metric semantics in run_experiment.py and
    makes the baselines column comparable for Figure 1.

    Returns a dict with keys: time_s, forward_tps, latency_ms.

    NOTE: `forward_tps` is forward-pass tokens/sec, NOT generation tps.
    Different from RASDInference.generate_text's `throughput_tps`.
    See module docstring.
    """
    mod = mod.to(device).eval()
    x = torch.randn(1, seq_len, dim, device=device)

    # Warmup
    with torch.no_grad():
        mod(x)
    _sync(device)

    elapsed = []
    with torch.no_grad():
        for _ in range(runs):
            _sync(device)
            t0 = time.perf_counter()
            mod(x)
            _sync(device)
            elapsed.append(time.perf_counter() - t0)

    avg_s = sum(elapsed) / len(elapsed)
    # Throughput is over the full context. In distributed mode all
    # ranks process their shard in parallel and finish (≈)
    # simultaneously, so wall time corresponds to total_len tokens.
    # Falls back to seq_len for single-rank runs (caller passes
    # total_len=None, which == seq_len). (Fix for finding #4 from
    # 2026-05-10 review: previously underreported by world_size×.)
    measured_len = total_len if total_len is not None else seq_len
    return {
        "time_s": avg_s,
        "forward_tps": measured_len / avg_s,
        "latency_ms": (avg_s / measured_len) * 1_000,
    }


# ---------------------------------------------------------------------------
# CSV helpers
# ---------------------------------------------------------------------------

CSV_HEADER = [
    "timestamp",
    "baseline",
    "context_length",
    "seed",
    "device",
    "world_size",
    "time_s",
    "forward_tps",
    "latency_ms",
]


def write_row(writer, fh, row: dict):
    writer.writerow([row[c] for c in CSV_HEADER])
    fh.flush()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(description="Benchmark Ring and Sliding-Window attention baselines")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu",
                   help="Torch device string (default: cuda if available)")
    p.add_argument("--dim", type=int, default=1024, help="Model hidden dimension")
    p.add_argument("--heads", type=int, default=8, help="Number of attention heads")
    p.add_argument("--block-size", type=int, default=4096,
                   help="Ring attention block size (local path)")
    p.add_argument("--window-size", type=int, default=128,
                   help="Sliding window size (keep small: unfold materialises O(S*window*head_dim) tensor)")
    p.add_argument("--runs", type=int, default=3, help="Timed forward passes per config")
    p.add_argument("--out", default="results/baselines/baselines.csv", help="Output CSV path")
    p.add_argument(
        "--lengths", nargs="+", type=int,
        default=[131_072, 262_144, 524_288, 1_048_576],
        help="Context lengths to benchmark (default: 128k 256k 512k 1M — "
             "matches the M4 final-matrix grid)",
    )
    p.add_argument(
        "--seeds", nargs="+", type=int, default=[42, 123, 456],
        help="Seeds to run per (baseline, length). Default matches the "
             "M4 final matrix's 3-seed grid.",
    )
    p.add_argument("--distributed", action="store_true",
                   help="Initialise torch.distributed (use with torchrun)")
    p.add_argument("--ring-max-ctx", type=int, default=131_072,
                   help="Skip the naive Ring baseline above this context "
                        "(default 128k). Single-rank Ring materialises (S, S) "
                        "attention scores; at S=256k that's ~33 GB per head, "
                        "OOMs on 40GB and 80GB alike. Skipping past the "
                        "ceiling avoids polluting session.log with 'CUDA out "
                        "of memory' which can trigger upstream failure-guards. "
                        "Pre-2026-05-10 was implicit (try/except + print)— "
                        "the explicit gate is safer.")
    p.add_argument("--sliding-max-ctx", type=int, default=262_144,
                   help="Skip the Sliding-Window baseline above this context "
                        "(default 256k). Sliding's unfold materialises "
                        "O(S * window * head_dim); at S=768k with window=128 "
                        "and 32 heads that's ~48 GiB single-rank — exactly "
                        "the OOM that terminated Track A on 40GB.")
    args = p.parse_args()

    # Optional distributed init
    world_size = 1
    if args.distributed:
        import torch.distributed as dist
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        if args.device == "cuda":
            torch.cuda.set_device(local_rank)
        dist.init_process_group(
            backend="nccl" if args.device == "cuda" else "gloo",
            device_id=torch.device(f"cuda:{local_rank}") if args.device == "cuda" else None,
        )
        world_size = dist.get_world_size()
        rank = dist.get_rank()
        if args.device == "cuda":
            args.device = f"cuda:{rank}"
    else:
        rank = 0

    device = torch.device(args.device)

    # Only rank 0 writes the CSV
    if rank == 0:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        write_header = not os.path.exists(args.out)
        csv_fh = open(args.out, "a", newline="")
        writer = csv.writer(csv_fh)
        if write_header:
            writer.writerow(CSV_HEADER)
    else:
        csv_fh = writer = None

    for seed in args.seeds:
        torch.manual_seed(seed)
        if device.type == "cuda":
            torch.cuda.manual_seed_all(seed)

        for total_len in args.lengths:
            # Each rank processes a shard in distributed mode
            local_len = total_len // world_size if world_size > 1 else total_len

            models = {
                "ring": RingAttention(
                    dim=args.dim, num_heads=args.heads, block_size=args.block_size
                ),
                "sliding": SlidingWindowAttention(
                    dim=args.dim, num_heads=args.heads, window_size=args.window_size
                ),
            }

            for name, mod in models.items():
                # Pre-skip baselines past their known memory ceiling so no
                # OOM ever fires. Important for two reasons:
                #   (1) The naive single-rank Ring/Sliding implementations
                #       are FAIRNESS baselines for the paper — readers
                #       expect them to OOM at long context, that's the
                #       contribution being demonstrated.
                #   (2) Even with try/except, an OOM message in stdout
                #       trips upstream failure-guards (Phase C 2026-05-10:
                #       a Sliding 768k OOM tripped the 40GB pod's terminate
                #       even though the Python code recovered).
                ceiling = (
                    args.ring_max_ctx if name == "ring"
                    else args.sliding_max_ctx
                )
                if total_len > ceiling:
                    if rank == 0:
                        print(
                            f"  [{name}] context_length={total_len:,}  "
                            f"SKIPPED (past ceiling {ceiling:,}; expected "
                            f"to OOM single-rank, this is the paper claim)",
                            flush=True,
                        )
                        # Row dict MUST cover every CSV_HEADER field —
                        # write_row does `[row[c] for c in CSV_HEADER]`
                        # and KeyError takes torchrun down with exit 1
                        # (caught and recovered 2026-05-10 19:48 UTC after
                        # commit 465d714 missed timestamp + device fields,
                        # killing Track B's orchestrator post-p33).
                        write_row(writer, csv_fh, {
                            "timestamp": datetime.now(timezone.utc).isoformat(),
                            "baseline": name,
                            "context_length": total_len,
                            "seed": seed,
                            "device": args.device,
                            "world_size": world_size,
                            "time_s": -1.0,
                            "forward_tps": -1.0,
                            "latency_ms": -1.0,
                        })
                    continue

                if rank == 0:
                    print(
                        f"  [{name}] context_length={total_len:,}  "
                        f"local_shard={local_len:,}  seed={seed}",
                        flush=True,
                    )

                try:
                    stats = benchmark_module(
                        mod, local_len, args.dim, device,
                        runs=args.runs, total_len=total_len,
                    )
                except (RuntimeError, torch.cuda.OutOfMemoryError) as exc:
                    # Report the failure but DON'T print "CUDA out of memory"
                    # verbatim — keep the failure-guard's grep clean. Truncate
                    # the exc message and tag it with our own prefix.
                    msg = str(exc).split('\n', 1)[0][:80]
                    if rank == 0:
                        print(f"    BASELINE_RUNTIME_ERR ({name}): {msg}", flush=True)
                    stats = {"time_s": -1.0, "forward_tps": -1.0, "latency_ms": -1.0}
                finally:
                    del mod
                    if device.type == "cuda":
                        torch.cuda.empty_cache()

                if rank == 0:
                    row = {
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                        "baseline": name,
                        "context_length": total_len,
                        "seed": seed,
                        "device": args.device,
                        "world_size": world_size,
                        **stats,
                    }
                    write_row(writer, csv_fh, row)
                    print(
                        f"    time={stats['time_s']:.3f}s  "
                        f"forward_tps={stats['forward_tps']:.1f} tok/s  "
                        f"latency={stats['latency_ms']:.4f} ms/tok",
                        flush=True,
                    )

    if rank == 0:
        csv_fh.close()
        print(f"\nResults written to {args.out}")

    if args.distributed:
        import torch.distributed as dist
        dist.destroy_process_group()


if __name__ == "__main__":
    main()

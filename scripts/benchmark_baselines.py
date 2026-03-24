#!/usr/bin/env python3
"""Benchmark RingAttention and SlidingWindowAttention at different context lengths.

Produces a CSV at `results/baselines.csv` with per-run timing, throughput,
and latency metrics required by Milestone 2.

Usage (single GPU / CPU):
    python scripts/benchmark_baselines.py

Usage (multi-GPU via torchrun, e.g. 8 GPUs):
    torchrun --nproc_per_node=8 scripts/benchmark_baselines.py --distributed

Columns written to CSV:
    timestamp, baseline, context_length, device, world_size,
    time_s, throughput_tps, latency_ms
"""
import argparse
import csv
import os
import sys
import time
from datetime import datetime, timezone

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
) -> dict:
    """Run `mod` for `runs` forward passes and return timing stats.

    For distributed runs the input is a local shard (seq_len already divided
    by world_size by the caller).

    Returns a dict with keys: time_s, throughput_tps, latency_ms
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
    return {
        "time_s": avg_s,
        "throughput_tps": seq_len / avg_s,          # tokens generated per second
        "latency_ms": (avg_s / seq_len) * 1_000,    # ms per token
    }


# ---------------------------------------------------------------------------
# CSV helpers
# ---------------------------------------------------------------------------

CSV_HEADER = [
    "timestamp",
    "baseline",
    "context_length",
    "device",
    "world_size",
    "time_s",
    "throughput_tps",
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
    p.add_argument("--window-size", type=int, default=2048,
                   help="Sliding window size")
    p.add_argument("--runs", type=int, default=3, help="Timed forward passes per config")
    p.add_argument("--out", default="results/baselines.csv", help="Output CSV path")
    p.add_argument(
        "--lengths", nargs="+", type=int,
        default=[131_072, 262_144, 524_288],
        help="Context lengths to benchmark (default: 128k 256k 512k)",
    )
    p.add_argument("--distributed", action="store_true",
                   help="Initialise torch.distributed (use with torchrun)")
    args = p.parse_args()

    # Optional distributed init
    world_size = 1
    if args.distributed:
        import torch.distributed as dist
        dist.init_process_group(backend="nccl" if args.device == "cuda" else "gloo")
        world_size = dist.get_world_size()
        rank = dist.get_rank()
        # Pin each process to its own GPU
        if args.device == "cuda":
            torch.cuda.set_device(rank)
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
            if rank == 0:
                print(f"  [{name}] context_length={total_len:,}  local_shard={local_len:,}", flush=True)

            try:
                stats = benchmark_module(mod, local_len, args.dim, device, runs=args.runs)
            except (RuntimeError, torch.cuda.OutOfMemoryError) as exc:
                if rank == 0:
                    print(f"    FAILED: {exc}", flush=True)
                stats = {"time_s": -1.0, "throughput_tps": -1.0, "latency_ms": -1.0}

            if rank == 0:
                row = {
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "baseline": name,
                    "context_length": total_len,
                    "device": args.device,
                    "world_size": world_size,
                    **stats,
                }
                write_row(writer, csv_fh, row)
                print(
                    f"    time={stats['time_s']:.3f}s  "
                    f"throughput={stats['throughput_tps']:.1f} tok/s  "
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

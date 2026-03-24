#!/usr/bin/env python3
"""Benchmark RingAttention and SlidingWindowAttention at different context lengths.

Produces a CSV log at `results/baselines.csv` with timings.
"""
import argparse
import csv
import os
import time
from datetime import datetime

import torch
from torch import nn

from src.baselines.ring_attention import RingAttention
from src.baselines.sliding_window import SlidingWindowAttention


def benchmark_module(mod: nn.Module, seq_len: int, dim: int, device: torch.device, runs: int = 3):
    x = torch.randn(1, seq_len, dim, device=device)
    mod = mod.to(device)
    # warmup
    with torch.no_grad():
        for _ in range(1):
            _ = mod(x)

    torch.cuda.synchronize() if device.type == 'cuda' else None
    times = []
    with torch.no_grad():
        for _ in range(runs):
            t0 = time.time()
            _ = mod(x)
            if device.type == 'cuda':
                torch.cuda.synchronize()
            times.append(time.time() - t0)
    return sum(times) / len(times)


def ensure_results_dir(path: str = 'results'):
    os.makedirs(path, exist_ok=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    p.add_argument('--dims', type=int, default=1024)
    p.add_argument('--heads', type=int, default=8)
    p.add_argument('--runs', type=int, default=3)
    p.add_argument('--out', default='results/baselines.csv')
    p.add_argument('--lengths', nargs='+', type=int, default=[131072, 262144, 524288])
    args = p.parse_args()

    device = torch.device(args.device)
    ensure_results_dir(os.path.dirname(args.out) or '.')

    header = ['timestamp', 'baseline', 'context_length', 'time_s', 'device']
    write_header = not os.path.exists(args.out)

    with open(args.out, 'a', newline='') as fh:
        writer = csv.writer(fh)
        if write_header:
            writer.writerow(header)

        for L in args.lengths:
            # keep model dims small to reduce memory; change as needed
            ring = RingAttention(dim=args.dims, num_heads=args.heads, block_size=4096)
            sliding = SlidingWindowAttention(dim=args.dims, num_heads=args.heads, window_size=2048)

            for name, mod in [('ring', ring), ('sliding', sliding)]:
                try:
                    t = benchmark_module(mod, L, args.dims, device, runs=args.runs)
                except RuntimeError as e:
                    t = -1.0
                writer.writerow([datetime.utcnow().isoformat(), name, L, t, args.device])
                fh.flush()


if __name__ == '__main__':
    main()

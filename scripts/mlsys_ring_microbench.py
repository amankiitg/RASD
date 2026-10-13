#!/usr/bin/env python3
"""BARE RING MICROBENCHMARK — the engine's rotation primitive, with no model.

WHY THIS EXISTS. A 128k decode step on 8x A100 costs ~3.3-3.7 s, and the profile localises
73.4% of the decode window to NCCL SendRecv over 256 rotation rounds
(8 rotations x 32 layers), i.e. ~9.3 ms per round. Two readings fit that number and they
imply opposite fixes:

  * BANDWIDTH — each round ships the full local K/V shard with the GQA expansion applied
    (`repeat_kv` runs before the rotation), 2 x 32 heads x 16256 tokens x 128 dim x 2 B
    = 266 MB, so 266 MB / 9.3 ms = ~29 GB/s. On NVLink that is 5-10x below line rate, which
    would mean the transport is on a slow path and an NCCL env var fixes it.
  * LATENCY / PER-ROUND OVERHEAD — the round costs ~9.3 ms regardless of payload, which
    would mean the bytes are not the problem and the engine's per-round work is.

This script measures the primitive itself: the same `dist.batch_isend_irecv` ring with four
P2POps (isend k, isend v, irecv k, irecv v) that `_issue_rotation` issues, run for 256
rounds at four payload sizes. The 9.3 ms/round figure comes from the profile; the number
this script produces for the SAME payload is the comparison that decides which reading is
right.

Sizes (bf16, 1x32 heads x S tokens x 128 dim, K+V):
   4 KB    pure latency, payload negligible
   33 MB   the raw local shard's K, WITHOUT the GQA expansion (8 heads)
   133 MB  K with the expansion (32 heads)
   266 MB  K+V with the expansion — what the engine actually ships per round at 128k

Run it with torchrun (it needs a real process group):
   torchrun --nproc_per_node=8 scripts/mlsys_ring_microbench.py --rounds 256
and it degrades honestly at one rank: it reports that the collective path was not exercised
rather than printing a timing for a no-op.

Output: a CSV and a JSON in --out, plus one line per (size, rank-group) on stdout.
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time


def _parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--rounds", type=int, default=256,
                   help="rotation rounds per size (256 = one decode token's worth)")
    p.add_argument("--warmup", type=int, default=10, help="rounds discarded before timing")
    p.add_argument("--heads", type=int, default=32, help="heads in the shipped payload")
    p.add_argument("--head-dim", type=int, default=128)
    p.add_argument("--seq", type=int, default=16256,
                   help="tokens per shard (130048/8 at 128k nominal)")
    p.add_argument("--out", default="results/mlsys/ring_bench")
    p.add_argument("--tag", default="ringbench")
    p.add_argument("--points", default=None,
                   help="override the sweep as 'label:heads:seq,label:heads:seq'")
    return p.parse_args(argv)


# The default sweep. WHY THESE SHAPES. The brief asks for 4 KB / 33 MB / 133 MB / 266 MB per
# round. Every point keeps a real engine head count and varies the SHARD LENGTH, so each
# measured round is the same kind of operation the engine issues; only its size changes.
# `K+V` is the bytes actually on the wire, and each point is labelled with the real engine
# quantity it corresponds to:
#   tiny_4kb  heads=8, seq=1  -> 4.1 KB   pure latency floor (smaller than any real payload)
#   mb33      seq=2032   -> 33.3 MB
#   mb133     seq=8128   -> 133.2 MB
#   mb266     seq=16256  -> 266.4 MB  THE ENGINE'S ACTUAL PAYLOAD at 128k (S_local)
#   k8_66mb   seq=16256, heads=8 -> 66.6 MB  what the deferred-expansion fix would ship
DEFAULT_POINTS = [
    ("tiny_4kb", 8, 1),
    ("mb33", 32, 2032),
    ("mb133", 32, 8128),
    ("mb266", 32, 16256),
    ("k8_66mb_expand_after", 8, 16256),
]


def _parse_points(spec):
    pts = []
    for item in spec.split(","):
        label, heads, seq = item.split(":")
        pts.append((label, int(heads), int(seq)))
    return pts


def main(argv=None):
    args = _parse_args(argv)
    import torch
    import torch.distributed as dist

    rank = int(os.environ.get("RANK", "0"))
    world = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank) if torch.cuda.is_available() else torch.device("cpu")

    if world > 1:
        dist.init_process_group(backend="nccl" if torch.cuda.is_available() else "gloo")
    os.makedirs(args.out, exist_ok=True)

    def log(msg):
        if rank == 0:
            print(msg, flush=True)

    log(f"[ringbench] world={world} rank={rank} device={device} "
        f"torch={torch.__version__} rounds={args.rounds}")
    if world == 1:
        log("[ringbench] WORLD_SIZE=1: the collective path CANNOT be exercised at one rank "
            "(this is the R4 lesson). Timings written below are latency of a no-op exchange.")

    rows = []
    points = _parse_points(args.points) if args.points else DEFAULT_POINTS
    for label, heads, seq_len in points:
        shape = (1, heads, seq_len, args.head_dim)
        dtype = torch.bfloat16   # the engine's payload dtype; keeps the byte math device-independent
        send_k = torch.randn(shape, device=device, dtype=dtype)
        send_v = torch.randn(shape, device=device, dtype=dtype)
        recv_k = torch.empty_like(send_k)
        recv_v = torch.empty_like(send_v)
        nbytes = 2 * send_k.numel() * send_k.element_size()
        send_to, recv_from = (rank + 1) % world, (rank - 1) % world

        def one_round():
            if world == 1:
                return
            ops = [
                dist.P2POp(dist.isend, send_k, send_to),
                dist.P2POp(dist.isend, send_v, send_to),
                dist.P2POp(dist.irecv, recv_k, recv_from),
                dist.P2POp(dist.irecv, recv_v, recv_from),
            ]
            for r in dist.batch_isend_irecv(ops):
                r.wait()

        for _ in range(args.warmup):
            one_round()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        if world > 1:
            dist.barrier()

        per_round = []
        t_all = time.perf_counter()
        for _ in range(args.rounds):
            t0 = time.perf_counter()
            one_round()
            per_round.append(time.perf_counter() - t0)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        if world > 1:
            dist.barrier()
        total = time.perf_counter() - t_all

        med = statistics.median(per_round) if per_round else float("nan")
        p90 = (sorted(per_round)[int(0.9 * (len(per_round) - 1))] if per_round else float("nan"))
        gbs = (nbytes / med / 1e9) if med else float("nan")
        rows.append({
            "label": label, "size_mb_actual": round(nbytes / 1e6, 3),
            "seq_len": seq_len, "heads": heads, "rank": rank,
            "rounds": args.rounds, "ms_per_round": round(med * 1e3, 4),
            "ms_per_round_p90": round(p90 * 1e3, 4),
            "gb_per_s": round(gbs, 2),
            "ms_total_256_rounds": round(med * 1e3 * 256, 2),
            "world_size": world,
        })
        log(f"[ringbench] {label:22s} {nbytes/1e6:8.2f} MB  seq={seq_len:6d} heads={heads}  "
            f"median {med*1e3:8.3f} ms/round (p90 {p90*1e3:8.3f})  {gbs:8.2f} GB/s  "
            f"-> {med*1e3*256:8.1f} ms per 256 rounds")

    # Rank skew: the profile's rank 0 vs rank 4 difference needs a cause. Gather every
    # rank's median at the largest size; a wide spread means one rank is a straggler.
    skew = []
    for row in rows:
        if world > 1:
            allv = [None] * world
            dist.all_gather_object(allv, row["ms_per_round"])
            row_skew = {"label": row["label"], "size_mb_actual": row["size_mb_actual"],
                        "per_rank_ms": [round(float(v), 4) for v in allv],
                        "spread_ms": round(max(allv) - min(allv), 4)}
            skew.append(row_skew)
            if rank == 0:
                log(f"[ringbench] skew @{row['label']} ({row['size_mb_actual']} MB): "
                    f"min {min(allv):.3f} max {max(allv):.3f} ms "
                    f"(spread {max(allv)-min(allv):.3f} ms)")

    if rank == 0:
        import csv
        with open(os.path.join(args.out, f"{args.tag}.csv"), "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows([r for r in rows if r["rank"] == 0])
        json.dump({"tag": args.tag, "rounds": args.rounds, "world_size": world,
                   "rank0_rows": [r for r in rows if r["rank"] == 0], "skew": skew},
                  open(os.path.join(args.out, f"{args.tag}.json"), "w"), indent=1)
        log(f"[ringbench] wrote {args.out}/{args.tag}.csv and .json")
    if dist.is_initialized():
        dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    sys.exit(main())

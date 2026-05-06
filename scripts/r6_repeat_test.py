"""
Diagnostic: call engine.generate() repeatedly with small max_new.
If the bug is per-CALL state (e.g., dist init), repeated calls succeed.
If the bug is per-ROUND state, accumulating rounds across calls also fails.
Run via torchrun --nproc_per_node=8.
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import torch
import torch.distributed as dist


def _init_dist():
    if "WORLD_SIZE" not in os.environ:
        return 0, 1
    world_size = int(os.environ["WORLD_SIZE"])
    rank       = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    if world_size > 1:
        dist.init_process_group(
            backend="nccl",
            device_id=torch.device(f"cuda:{local_rank}"),
        )
    return rank, world_size


def _build_prompt(ctx_len, tok):
    seed = "The quick brown fox jumps over the lazy dog. " * 32
    full = seed * max(1, ctx_len // len(tok.encode(seed)))
    return tok.decode(tok.encode(full)[:ctx_len], skip_special_tokens=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-call-max-new", type=int, default=8)
    ap.add_argument("--n-calls", type=int, default=10)
    ap.add_argument("--prefetch-depth", type=int, default=0)
    args = ap.parse_args()

    rank, world_size = _init_dist()
    is_chief = (rank == 0)

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from src.models.rasd_inference import RASDConfig, RASDInference

    cfg = RASDConfig(
        target_model_name = "meta-llama/Llama-2-7b-hf",
        draft_model_name  = "princeton-nlp/Sheared-LLaMA-1.3B",
        spec_steps        = 4,
        max_new_tokens    = args.per_call_max_new,
        dtype             = "bfloat16",
        quantize_target   = True,
        quantize_draft    = True,
        kv_block_size     = 999999,
        prefetch_depth    = args.prefetch_depth,
        context_length    = 8192,
        seed              = 42,
    )

    if is_chief:
        print(f"[repeat_test] world_size={world_size} prefetch={args.prefetch_depth} "
              f"per_call={args.per_call_max_new} n_calls={args.n_calls}")

    engine = RASDInference(cfg)
    if is_chief:
        print("[repeat_test] engine loaded")

    # Tokenize prompt once
    raw = _build_prompt(8192, engine.tokenizer)
    ids = engine.tokenizer(raw, return_tensors="pt", add_special_tokens=False).input_ids
    S = (ids.shape[1] // world_size) * world_size
    ids = ids[:, :S].to(f"cuda:{int(os.environ.get('LOCAL_RANK', 0))}")

    # Repeated calls
    for i in range(args.n_calls):
        t0 = time.perf_counter()
        _, metrics = engine.generate(ids)
        dt = time.perf_counter() - t0
        if is_chief:
            print(f"[repeat_test] call {i+1}/{args.n_calls}: "
                  f"alpha={metrics.get('acceptance_rate', float('nan')):.3f} "
                  f"tokens={metrics.get('tokens_generated', 0)} "
                  f"elapsed={dt:.2f}s "
                  f"peak_mb={torch.cuda.max_memory_allocated()/1024**2:.0f}",
                  flush=True)

    if is_chief:
        print("[repeat_test] done")
    if world_size > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()

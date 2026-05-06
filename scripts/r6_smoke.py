"""
R6 smoke test — multi-rank ring-attention + spec decoding.

Run with torchrun:
    # R6.2: 2-rank 8k smoke
    torchrun --nproc_per_node=2 --master_port=29510 scripts/r6_smoke.py \
        --context-length 8192 --max-new-tokens 256 --spec-steps 4 \
        --target-quant fp16 --draft-quant nf4

    # R6.3: 8-rank 8k smoke
    torchrun --nproc_per_node=8 --master_port=29510 scripts/r6_smoke.py \
        --context-length 8192 --max-new-tokens 256 --spec-steps 4 \
        --target-quant fp16 --draft-quant nf4

    # R6.4: 8-rank 64k memory check (small max_new for speed)
    torchrun --nproc_per_node=8 --master_port=29510 scripts/r6_smoke.py \
        --context-length 65536 --max-new-tokens 64 --spec-steps 4 \
        --target-quant fp16 --draft-quant nf4

Reports α, tps, and per-rank peak GPU memory. The ring-attention path is
exercised when WORLD_SIZE > 1.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import torch
import torch.distributed as dist


def _init_dist():
    """Initialise NCCL process group if torchrun launched us; return (rank, world_size)."""
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


def _build_prompt(ctx_len: int, tokenizer) -> str:
    """Build a synthetic prompt of approximately `ctx_len` tokens."""
    seed = "The quick brown fox jumps over the lazy dog. " * 32
    tokens = tokenizer.encode(seed)
    repeats = max(1, ctx_len // len(tokens))
    full = (seed * repeats)
    full_tokens = tokenizer.encode(full)[:ctx_len]
    return tokenizer.decode(full_tokens, skip_special_tokens=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target-model", default="meta-llama/Llama-2-7b-hf")
    ap.add_argument("--draft-model",  default="princeton-nlp/Sheared-LLaMA-1.3B")
    ap.add_argument("--target-quant", choices=["fp16", "nf4"], default="fp16")
    ap.add_argument("--draft-quant",  choices=["fp16", "nf4"], default="nf4")
    ap.add_argument("--context-length", type=int, default=8192)
    ap.add_argument("--spec-steps",  type=int, default=4)
    ap.add_argument("--max-new-tokens", type=int, default=256)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--kv-block-size",  type=int, default=512)
    ap.add_argument("--prefetch-depth", type=int, default=1)
    ap.add_argument("--out", default=None,
                    help="JSON output path (rank 0 writes); default stdout only")
    args = ap.parse_args()

    rank, world_size = _init_dist()
    is_chief = (rank == 0)

    if is_chief:
        print(f"[r6_smoke] world_size={world_size} rank={rank}")
        print(f"[r6_smoke] config: ctx={args.context_length}, "
              f"k={args.spec_steps}, max_new={args.max_new_tokens}, "
              f"target={args.target_quant}, draft={args.draft_quant}")

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from src.models.rasd_inference import RASDConfig, RASDInference

    cfg = RASDConfig(
        target_model_name = args.target_model,
        draft_model_name  = args.draft_model,
        spec_steps        = args.spec_steps,
        max_new_tokens    = args.max_new_tokens,
        dtype             = "bfloat16" if args.target_quant == "fp16" else "bfloat16",
        quantize_target   = (args.target_quant == "nf4"),
        quantize_draft    = (args.draft_quant  == "nf4"),
        temperature       = args.temperature,
        kv_block_size     = args.kv_block_size,
        prefetch_depth    = args.prefetch_depth,
        context_length    = args.context_length,
        seed              = args.seed,
    )

    if is_chief:
        print("[r6_smoke] loading models (cold first launch downloads ~16 GB)…")
    t0 = time.perf_counter()
    engine = RASDInference(cfg)
    t_load = time.perf_counter() - t0
    if is_chief:
        print(f"[r6_smoke] load done in {t_load:.1f}s")

    # Build a prompt and tokenize. Crucial: truncate token count to be
    # divisible by world_size so the contiguous sequence shard math in
    # _prefill works (rasd_inference.py asserts S % W == 0 under multi-rank).
    raw_prompt = _build_prompt(args.context_length, engine.tokenizer)
    input_ids = engine.tokenizer(
        raw_prompt, return_tensors="pt", add_special_tokens=False,
    ).input_ids
    S = input_ids.shape[1]
    if world_size > 1 and S % world_size != 0:
        S = (S // world_size) * world_size
        input_ids = input_ids[:, :S]
    input_ids = input_ids.to(f"cuda:{int(os.environ.get('LOCAL_RANK', 0))}")
    if is_chief:
        print(f"[r6_smoke] prompt tokens: {S} (divisible by world_size={world_size}: "
              f"{S % world_size == 0})")

    if is_chief:
        print("[r6_smoke] starting generate()…")
    torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    _, metrics = engine.generate(input_ids)
    elapsed = time.perf_counter() - t0
    peak_mb = torch.cuda.max_memory_allocated() / 1024**2

    # Each rank reports its own peak; rank 0 prints aggregate
    peak_t = torch.tensor([peak_mb], device=f"cuda:{int(os.environ.get('LOCAL_RANK', 0))}")
    if world_size > 1:
        peaks = [torch.zeros_like(peak_t) for _ in range(world_size)]
        dist.all_gather(peaks, peak_t)
        peaks_mb = [p.item() for p in peaks]
    else:
        peaks_mb = [peak_mb]

    if is_chief:
        print()
        print("=" * 70)
        print(f"R6 smoke results — world_size={world_size}, ctx={args.context_length}")
        print("=" * 70)
        print(f"  alpha (acceptance rate):  {metrics.get('acceptance_rate', float('nan')):.3f}")
        print(f"  tokens generated:         {metrics.get('tokens_generated', 0)}")
        print(f"  throughput tps:           {metrics.get('throughput_tps', float('nan')):.2f}")
        print(f"  total time:               {elapsed:.1f} s")
        print(f"  per-rank peak memory MB:  {[f'{p:.0f}' for p in peaks_mb]}")
        print(f"  per-rank peak memory GB:  {[f'{p/1024:.1f}' for p in peaks_mb]}")
        print("=" * 70)

        if args.out:
            os.makedirs(os.path.dirname(args.out), exist_ok=True)
            with open(args.out, "w") as f:
                json.dump({
                    "world_size":    world_size,
                    "context_length": args.context_length,
                    "spec_steps":    args.spec_steps,
                    "max_new_tokens": args.max_new_tokens,
                    "target_quant":  args.target_quant,
                    "draft_quant":   args.draft_quant,
                    "alpha":         metrics.get("acceptance_rate"),
                    "tps":           metrics.get("throughput_tps"),
                    "tokens":        metrics.get("tokens_generated"),
                    "load_sec":      t_load,
                    "elapsed_sec":   elapsed,
                    "peak_mem_mb":   peaks_mb,
                    "peak_mem_gb":   [p/1024 for p in peaks_mb],
                }, f, indent=2)
            print(f"[r6_smoke] wrote {args.out}")

    if world_size > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()

"""
Memory scaling validation for RingAttentionFlash.

Runs both the naive baseline and the FA-2 variant across increasing context
lengths on a single GPU (simulates one rank's local shard) and reports peak
GPU memory. Confirms that the FA variant no longer OOMs at 256k/512k.

Usage
-----
    python scripts/validate_flash_memory.py

Requirements: CUDA GPU, flash-attn installed.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
import gc

# Context lengths to test (per-GPU shard lengths)
# At 8 GPUs and total lengths [64k, 128k, 256k, 512k]:
#   shard = total / 8 = [8k, 16k, 32k, 64k]
SHARD_LENGTHS = [8_192, 16_384, 32_768, 65_536]
TOTAL_LENGTHS  = [s * 8 for s in SHARD_LENGTHS]

BATCH      = 1
DIM        = 1024
NUM_HEADS  = 8
DTYPE      = torch.bfloat16
DEVICE     = "cuda"


def peak_mem_mb():
    return torch.cuda.max_memory_allocated() / 1024 ** 2


def run_single(module, shard_len, batch, dim, dtype, device):
    torch.cuda.reset_peak_memory_stats()
    gc.collect()
    torch.cuda.empty_cache()

    x = torch.randn(batch, shard_len, dim, dtype=dtype, device=device)
    with torch.no_grad():
        _ = module(x)
    torch.cuda.synchronize()
    return peak_mem_mb()


def main():
    if not torch.cuda.is_available():
        print("ERROR: CUDA not available. Run on a GPU node.")
        sys.exit(1)

    from src.baselines.ring_attention import RingAttention
    from src.models.ring_attention_flash import RingAttentionFlash, _FLASH_AVAILABLE

    print(f"flash_attn available : {_FLASH_AVAILABLE}")
    print(f"dtype                : {DTYPE}")
    print(f"batch                : {BATCH}")
    print(f"dim / heads          : {DIM} / {NUM_HEADS}")
    print()
    print(f"{'Context':>10}  {'Shard':>8}  {'Baseline MB':>12}  {'FA-2 MB':>10}  {'Savings':>10}")
    print("-" * 60)

    naive_mod = RingAttention(DIM, NUM_HEADS).to(DTYPE).to(DEVICE).eval()
    flash_mod = RingAttentionFlash(DIM, NUM_HEADS).to(DTYPE).to(DEVICE).eval()

    for shard_len, total_len in zip(SHARD_LENGTHS, TOTAL_LENGTHS):
        label = f"{total_len // 1024}k"

        # Naive baseline
        try:
            naive_mb = run_single(naive_mod, shard_len, BATCH, DIM, DTYPE, DEVICE)
            naive_str = f"{naive_mb:>10.0f} MB"
        except torch.cuda.OutOfMemoryError:
            naive_str = "      OOM"
            naive_mb  = float("inf")
            torch.cuda.empty_cache()

        # FlashAttention variant
        try:
            flash_mb = run_single(flash_mod, shard_len, BATCH, DIM, DTYPE, DEVICE)
            flash_str = f"{flash_mb:>8.0f} MB"
        except torch.cuda.OutOfMemoryError:
            flash_str = "    OOM"
            flash_mb  = float("inf")
            torch.cuda.empty_cache()

        if naive_mb != float("inf") and flash_mb != float("inf"):
            savings = f"{(1 - flash_mb / naive_mb) * 100:>8.1f}%"
        else:
            savings = "       N/A"

        print(f"{label:>10}  {shard_len:>8,}  {naive_str}  {flash_str}  {savings}")

    print()
    print("Expected: FA-2 variant should NOT OOM at shard lengths where baseline does.")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Validate Ring Attention communication primitives in isolation.

Tests that the ring send/recv primitives correctly rotate tensors around the
process ring before running the full benchmark. Run this first to catch any
driver/NCCL/communication issues early.

Usage:
    torchrun --nproc_per_node=8 scripts/validate_ring_comm.py
"""
import os
import sys
import torch
import torch.distributed as dist

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


def validate_ring_rotation():
    """Test 1: Each rank sends its rank-id tensor around the full ring.
    After world_size steps every rank should have received from every other rank.
    """
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    device = torch.device(f"cuda:{rank}")

    send_to   = (rank + 1) % world_size
    recv_from = (rank - 1) % world_size

    # Each rank starts with a tensor filled with its own rank id
    buf = torch.full((4,), float(rank), device=device)
    recv_buf = torch.empty_like(buf)

    received_from = []
    for step in range(world_size):
        dist.barrier()
        if step < world_size - 1:
            reqs = [
                dist.isend(buf, dst=send_to),
                dist.irecv(recv_buf, src=recv_from),
            ]
            for r in reqs:
                r.wait()
            buf = recv_buf.clone()
            received_from.append(int(buf[0].item()))

    dist.barrier()
    expected_sources = sorted([(rank - s - 1) % world_size for s in range(world_size - 1)])
    actual_sources   = sorted(received_from)
    ok = expected_sources == actual_sources

    if rank == 0:
        status = "PASS" if ok else "FAIL"
        print(f"[Test 1: Ring rotation]  {status}")
        if not ok:
            print(f"  expected sources: {expected_sources}")
            print(f"  actual   sources: {actual_sources}")

    return ok


def validate_online_softmax_consistency():
    """Test 2: Single-GPU blockwise attention output matches full attention output.
    Checks that the online softmax accumulator is numerically equivalent to
    standard scaled dot-product attention.
    """
    import math
    from src.baselines.ring_attention import RingAttention

    rank = dist.get_rank()
    if rank != 0:
        dist.barrier()
        return True

    device = torch.device("cuda:0")
    dim, heads, seq = 64, 4, 256
    model = RingAttention(dim=dim, num_heads=heads, block_size=64).to(device).eval()

    x = torch.randn(1, seq, dim, device=device)
    with torch.no_grad():
        # Our blockwise output
        out_blockwise = model._local_forward(x)

        # Reference: standard full attention
        qkv = model.qkv(x).reshape(1, seq, 3, heads, dim // heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        scores = torch.einsum("bhid,bhjd->bhij", q, k) / math.sqrt(dim // heads)
        attn = torch.softmax(scores, dim=-1)
        out_ref = (attn @ v).permute(0, 2, 1, 3).reshape(1, seq, dim)
        out_ref = model.out(out_ref)

    max_err = (out_blockwise - out_ref).abs().max().item()
    ok = max_err < 1e-4
    status = "PASS" if ok else "FAIL"
    print(f"[Test 2: Online softmax]  {status}  (max_err={max_err:.2e})")

    dist.barrier()
    return ok


def validate_all_gather_ranks():
    """Test 3: all_gather correctness — every rank sees data from all ranks."""
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    device = torch.device(f"cuda:{rank}")

    local = torch.full((2,), float(rank), device=device)
    gathered = [torch.empty_like(local) for _ in range(world_size)]
    dist.all_gather(gathered, local)
    dist.barrier()

    ok = all(int(gathered[r][0].item()) == r for r in range(world_size))
    if rank == 0:
        status = "PASS" if ok else "FAIL"
        print(f"[Test 3: all_gather]      {status}")

    return ok


def main():
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl", device_id=torch.device(f"cuda:{local_rank}"))
    rank = dist.get_rank()

    if rank == 0:
        world_size = dist.get_world_size()
        print(f"\n=== Ring Communication Validation ({world_size} GPUs) ===\n")

    results = []
    results.append(validate_ring_rotation())
    dist.barrier()
    results.append(validate_online_softmax_consistency())
    dist.barrier()
    results.append(validate_all_gather_ranks())
    dist.barrier()

    if rank == 0:
        all_ok = all(results)
        print(f"\n{'All tests PASSED' if all_ok else 'SOME TESTS FAILED'} — "
              f"{'safe to run benchmark' if all_ok else 'fix issues before benchmarking'}\n")
        sys.exit(0 if all_ok else 1)

    dist.destroy_process_group()


if __name__ == "__main__":
    main()

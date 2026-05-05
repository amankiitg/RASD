"""
Tests for the ring attention kernel (src/models/ring_attention_kernel.py).

R1 of M3_RING_INTEGRATION_PLAN.md.

Layers:
  1. Single-process math tests (no dist) — causal mask, lse merge, layout.
  2. Multi-process gloo tests — actual ring P2P with W in {2, 4} on CPU.

The multi-process tests use torch.multiprocessing.spawn with gloo backend,
so they run on CPU without needing CUDA/NCCL. They are the gate for
"the kernel produces the same answer as single-process attention."
"""

from __future__ import annotations

import os
import socket
import tempfile

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp


# ---------------------------------------------------------------------------
# Single-process math tests
# ---------------------------------------------------------------------------

class TestSingleProcessMath:
    """No dist needed; verify the kernel's math under W=1."""

    def setup_method(self):
        torch.manual_seed(0)
        self.B, self.H, self.D = 2, 4, 32

    def test_prefill_w1_matches_reference(self):
        from src.models.ring_attention_kernel import ring_attention_prefill, reference_attention
        S = 16
        q = torch.randn(self.B, self.H, S, self.D)
        k = torch.randn(self.B, self.H, S, self.D)
        v = torch.randn(self.B, self.H, S, self.D)
        out_ring = ring_attention_prefill(q, k, v, rank=0, world_size=1)
        out_ref  = reference_attention(q, k, v, causal=True)
        assert torch.allclose(out_ring, out_ref, atol=1e-5)

    def test_decode_w1_matches_reference(self):
        from src.models.ring_attention_kernel import ring_attention_decode, reference_attention
        S_q, S_k = 5, 32
        q = torch.randn(self.B, self.H, S_q, self.D)
        k = torch.randn(self.B, self.H, S_k, self.D)
        v = torch.randn(self.B, self.H, S_k, self.D)
        out_ring = ring_attention_decode(
            q, k, v, rank=0, world_size=1,
            new_kv_owner_rank=0, new_kv_count=S_q,
        )
        out_ref = reference_attention(q, k, v, causal=True)
        assert torch.allclose(out_ring, out_ref, atol=1e-5)

    def test_combine_associativity(self):
        """Online-softmax merge should be associative: combining (a,b),c == a,(b,c)."""
        from src.models.ring_attention_kernel import _combine, _sdpa_step
        S_q, S_k = 4, 12
        q = torch.randn(self.B, self.H, S_q, self.D)
        k = torch.randn(self.B, self.H, S_k, self.D)
        v = torch.randn(self.B, self.H, S_k, self.D)
        scale = 1.0 / self.D ** 0.5

        # Split K/V into three blocks
        k1, k2, k3 = k.split([4, 4, 4], dim=2)
        v1, v2, v3 = v.split([4, 4, 4], dim=2)
        o1, l1 = _sdpa_step(q, k1, v1, scale, causal=False)
        o2, l2 = _sdpa_step(q, k2, v2, scale, causal=False)
        o3, l3 = _sdpa_step(q, k3, v3, scale, causal=False)

        # Combine left-associative
        oA, lA = _combine(o1, l1, o2, l2)
        oA, lA = _combine(oA, lA, o3, l3)

        # Reference: full-sequence attention (no chunking)
        o_ref, _ = _sdpa_step(q, k, v, scale, causal=False)
        assert torch.allclose(oA, o_ref, atol=1e-5)

    def test_causal_mask_alignment(self):
        """SDPA step's causal=True must align last query to last key (FA-2 convention)."""
        from src.models.ring_attention_kernel import _sdpa_step
        S_q, S_k = 3, 7
        q = torch.randn(self.B, self.H, S_q, self.D)
        k = torch.randn(self.B, self.H, S_k, self.D)
        v = torch.randn(self.B, self.H, S_k, self.D)
        scale = 1.0 / self.D ** 0.5
        # Force last value of v to a sentinel; q[-1] should attend to it freely,
        # q[0] should be able to attend up to S_k - S_q + 0 = 4 (i.e. v[0..4]),
        # so v[5] and v[6] (the "future" wrt q[0]) are masked out for q[0].
        v_test = v.clone()
        v_test[..., -1, :] = 1e3  # sentinel large value
        out, _ = _sdpa_step(q, k, v_test, scale, causal=True)
        # q[-1] should pull in the sentinel (one of its valid keys is the last)
        assert out[..., -1, :].abs().max() > 1.0, \
            "Last query must attend to last key when S_k>=S_q causal"


# ---------------------------------------------------------------------------
# Multi-process ring tests via torch.multiprocessing + gloo
# ---------------------------------------------------------------------------

def _worker_prefill(rank: int, world_size: int, init_file: str,
                    ref_q: torch.Tensor, ref_k: torch.Tensor, ref_v: torch.Tensor,
                    expected_out: torch.Tensor, atol: float, rtol: float):
    """Multi-process worker: each rank runs ring prefill on its slice."""
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    dist.init_process_group(
        backend="gloo",
        init_method=f"file://{init_file}",
        rank=rank,
        world_size=world_size,
    )
    try:
        from src.models.ring_attention_kernel import ring_attention_prefill
        S_total = ref_q.shape[2]
        assert S_total % world_size == 0, "Sequence must divide world size"
        S_local = S_total // world_size
        start, end = rank * S_local, (rank + 1) * S_local
        q_local = ref_q[:, :, start:end, :].contiguous()
        k_local = ref_k[:, :, start:end, :].contiguous()
        v_local = ref_v[:, :, start:end, :].contiguous()

        out_local = ring_attention_prefill(
            q_local, k_local, v_local,
            rank=rank, world_size=world_size,
        )
        # Compare this rank's slice to expected
        expected_local = expected_out[:, :, start:end, :]
        if not torch.allclose(out_local, expected_local, atol=atol, rtol=rtol):
            diff = (out_local - expected_local).abs().max().item()
            raise AssertionError(
                f"rank {rank}: ring prefill output diverges from reference "
                f"(max abs diff = {diff:.2e}, atol={atol})"
            )
    finally:
        dist.destroy_process_group()


def _worker_decode(rank: int, world_size: int, init_file: str,
                   q_global: torch.Tensor, full_k: torch.Tensor, full_v: torch.Tensor,
                   new_kv_owner_rank: int, new_kv_count: int,
                   expected_out: torch.Tensor, atol: float, rtol: float):
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    dist.init_process_group(
        backend="gloo",
        init_method=f"file://{init_file}",
        rank=rank,
        world_size=world_size,
    )
    try:
        from src.models.ring_attention_kernel import ring_attention_decode
        S_k = full_k.shape[2]
        S_local = S_k // world_size
        start, end = rank * S_local, (rank + 1) * S_local
        k_local = full_k[:, :, start:end, :].contiguous()
        v_local = full_v[:, :, start:end, :].contiguous()

        out = ring_attention_decode(
            q_global, k_local, v_local,
            rank=rank, world_size=world_size,
            new_kv_owner_rank=new_kv_owner_rank,
            new_kv_count=new_kv_count,
        )
        # Output should be the same on every rank (replicated)
        if not torch.allclose(out, expected_out, atol=atol, rtol=rtol):
            diff = (out - expected_out).abs().max().item()
            raise AssertionError(
                f"rank {rank}: ring decode output diverges from reference "
                f"(max abs diff = {diff:.2e}, atol={atol})"
            )
    finally:
        dist.destroy_process_group()


def _spawn(world_size: int, target, args):
    """Run `target(rank, world_size, init_file, *args)` across world_size workers."""
    init_file = tempfile.NamedTemporaryFile(delete=False).name
    os.unlink(init_file)  # init_file must NOT exist when gloo opens it
    try:
        mp.spawn(
            target,
            args=(world_size, init_file) + tuple(args),
            nprocs=world_size,
            join=True,
        )
    finally:
        if os.path.exists(init_file):
            os.unlink(init_file)


@pytest.mark.parametrize("world_size", [2, 4])
class TestMultiProcessRing:
    """Run ring kernel in a real distributed group (gloo, CPU)."""

    def _make_inputs(self, world_size: int):
        from src.models.ring_attention_kernel import reference_attention
        torch.manual_seed(world_size * 7)
        B, H, D = 1, 4, 16
        S_total = 8 * world_size  # divisible by W
        q = torch.randn(B, H, S_total, D, dtype=torch.float32)
        k = torch.randn(B, H, S_total, D, dtype=torch.float32)
        v = torch.randn(B, H, S_total, D, dtype=torch.float32)
        ref_out = reference_attention(q, k, v, causal=True)
        return q, k, v, ref_out

    def test_prefill_matches_reference(self, world_size):
        q, k, v, ref_out = self._make_inputs(world_size)
        _spawn(world_size, _worker_prefill, (q, k, v, ref_out, 1e-5, 1e-5))

    def test_decode_matches_reference(self, world_size):
        from src.models.ring_attention_kernel import reference_attention
        torch.manual_seed(world_size * 13)
        B, H, D = 1, 4, 16
        S_k = 8 * world_size
        S_q = 3
        q_global = torch.randn(B, H, S_q, D, dtype=torch.float32)
        # Build K such that the LAST S_q positions are the "new" tail aligned
        # with q_global, and they live on rank world_size-1
        full_k = torch.randn(B, H, S_k, D, dtype=torch.float32)
        full_v = torch.randn(B, H, S_k, D, dtype=torch.float32)
        ref_out = reference_attention(q_global, full_k, full_v, causal=True)

        new_kv_owner_rank = world_size - 1
        new_kv_count = S_q  # last S_q K/V positions are the new tail
        _spawn(
            world_size, _worker_decode,
            (q_global, full_k, full_v, new_kv_owner_rank, new_kv_count, ref_out,
             1e-5, 1e-5),
        )

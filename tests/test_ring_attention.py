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

    def test_decode_w1_no_tail_matches_reference(self):
        """Decode against pure-sharded cache (immediately after prefill)."""
        from src.models.ring_attention_kernel import ring_attention_decode, reference_attention
        S_q, S_k = 5, 32
        q = torch.randn(self.B, self.H, S_q, self.D)
        k = torch.randn(self.B, self.H, S_k, self.D)
        v = torch.randn(self.B, self.H, S_k, self.D)
        out_ring = ring_attention_decode(
            q, k, v, rank=0, world_size=1, tail_count=0,
        )
        # No tail means K positions are all "past" wrt Q — unconditional attend.
        out_ref = reference_attention(q, k, v, causal=False)
        assert torch.allclose(out_ring, out_ref, atol=1e-5)

    def test_decode_w1_with_tail_matches_reference(self):
        """Decode against (prefill + replicated tail). Tail positions align causally with Q."""
        from src.models.ring_attention_kernel import ring_attention_decode, reference_attention
        S_q = 5
        S_prefill, S_tail = 32, S_q
        q = torch.randn(self.B, self.H, S_q, self.D)
        k_prefill = torch.randn(self.B, self.H, S_prefill, self.D)
        v_prefill = torch.randn(self.B, self.H, S_prefill, self.D)
        k_tail = torch.randn(self.B, self.H, S_tail, self.D)
        v_tail = torch.randn(self.B, self.H, S_tail, self.D)
        # Concatenate as the kernel sees it (prefill | tail)
        k_local = torch.cat([k_prefill, k_tail], dim=2)
        v_local = torch.cat([v_prefill, v_tail], dim=2)
        out_ring = ring_attention_decode(
            q, k_local, v_local, rank=0, world_size=1, tail_count=S_tail,
        )
        # Reference: full sequence with causal mask. Q sees all prefill (no
        # mask) + causal-aligned tail. The same `causal=True` semantics on
        # the full (S_prefill + S_tail) sequence does this when S_q == S_tail.
        out_ref = reference_attention(q, k_local, v_local, causal=True)
        assert torch.allclose(out_ring, out_ref, atol=1e-5)

    def test_decode_w1_only_tail(self):
        """Degenerate case: no prefill, only tail."""
        from src.models.ring_attention_kernel import ring_attention_decode, reference_attention
        S_q = S_tail = 4
        q = torch.randn(self.B, self.H, S_q, self.D)
        k = torch.randn(self.B, self.H, S_tail, self.D)
        v = torch.randn(self.B, self.H, S_tail, self.D)
        out_ring = ring_attention_decode(
            q, k, v, rank=0, world_size=1, tail_count=S_tail,
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
                    expected_out: torch.Tensor, atol: float, rtol: float,
                    chunk_size: object = None, prefetch_depth: int = 0):
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
            chunk_size=chunk_size, prefetch_depth=prefetch_depth,
        )
        # Compare this rank's slice to expected
        expected_local = expected_out[:, :, start:end, :]
        if not torch.allclose(out_local, expected_local, atol=atol, rtol=rtol):
            diff = (out_local - expected_local).abs().max().item()
            raise AssertionError(
                f"rank {rank}: ring prefill output diverges from reference "
                f"(max abs diff = {diff:.2e}, atol={atol}) "
                f"chunk_size={chunk_size} prefetch_depth={prefetch_depth}"
            )
    finally:
        dist.destroy_process_group()


def _worker_decode(rank: int, world_size: int, init_file: str,
                   q_global: torch.Tensor,
                   full_prefill_k: torch.Tensor, full_prefill_v: torch.Tensor,
                   tail_k: torch.Tensor, tail_v: torch.Tensor,
                   tail_count: int,
                   expected_out: torch.Tensor, atol: float, rtol: float,
                   chunk_size: object = None, prefetch_depth: int = 0):
    """Each rank holds (its prefill slice) + (the same replicated tail)."""
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    dist.init_process_group(
        backend="gloo",
        init_method=f"file://{init_file}",
        rank=rank,
        world_size=world_size,
    )
    try:
        from src.models.ring_attention_kernel import ring_attention_decode
        S_prefill = full_prefill_k.shape[2]
        S_local = S_prefill // world_size
        start, end = rank * S_local, (rank + 1) * S_local
        # Each rank's local cache: own prefill slice + replicated tail
        k_local = torch.cat([
            full_prefill_k[:, :, start:end, :], tail_k,
        ], dim=2).contiguous()
        v_local = torch.cat([
            full_prefill_v[:, :, start:end, :], tail_v,
        ], dim=2).contiguous()

        out = ring_attention_decode(
            q_global, k_local, v_local,
            rank=rank, world_size=world_size, tail_count=tail_count,
            chunk_size=chunk_size, prefetch_depth=prefetch_depth,
        )
        if not torch.allclose(out, expected_out, atol=atol, rtol=rtol):
            diff = (out - expected_out).abs().max().item()
            raise AssertionError(
                f"rank {rank}: ring decode output diverges from reference "
                f"(max abs diff = {diff:.2e}, atol={atol}) "
                f"chunk_size={chunk_size} prefetch_depth={prefetch_depth}"
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
        """Decode under dual-cache: sharded prefill + replicated tail.

        Reference: a single-process full attention over (full_prefill + tail)
        with causal=True (Q's last positions align with tail).
        """
        from src.models.ring_attention_kernel import reference_attention
        torch.manual_seed(world_size * 13)
        B, H, D = 1, 4, 16
        S_prefill = 8 * world_size
        S_tail = 3
        S_q = S_tail
        q_global = torch.randn(B, H, S_q, D, dtype=torch.float32)
        full_prefill_k = torch.randn(B, H, S_prefill, D, dtype=torch.float32)
        full_prefill_v = torch.randn(B, H, S_prefill, D, dtype=torch.float32)
        tail_k = torch.randn(B, H, S_tail, D, dtype=torch.float32)
        tail_v = torch.randn(B, H, S_tail, D, dtype=torch.float32)
        # Single-process reference: concat as the model would see it.
        full_k = torch.cat([full_prefill_k, tail_k], dim=2)
        full_v = torch.cat([full_prefill_v, tail_v], dim=2)
        ref_out = reference_attention(q_global, full_k, full_v, causal=True)

        _spawn(
            world_size, _worker_decode,
            (q_global, full_prefill_k, full_prefill_v, tail_k, tail_v,
             S_tail, ref_out, 1e-5, 1e-5),
        )

    def test_decode_no_tail_matches_reference(self, world_size):
        """Decode against pure-sharded cache (no tail yet — first verify call)."""
        from src.models.ring_attention_kernel import reference_attention
        torch.manual_seed(world_size * 17)
        B, H, D = 1, 4, 16
        S_prefill = 8 * world_size
        S_q = 3
        q_global = torch.randn(B, H, S_q, D, dtype=torch.float32)
        full_prefill_k = torch.randn(B, H, S_prefill, D, dtype=torch.float32)
        full_prefill_v = torch.randn(B, H, S_prefill, D, dtype=torch.float32)
        # No tail — reference is unconditional attention (Q is past all prefill)
        ref_out = reference_attention(q_global, full_prefill_k, full_prefill_v, causal=False)
        empty_tail = torch.zeros(B, H, 0, D, dtype=torch.float32)

        _spawn(
            world_size, _worker_decode,
            (q_global, full_prefill_k, full_prefill_v, empty_tail, empty_tail,
             0, ref_out, 1e-5, 1e-5),
        )


# ---------------------------------------------------------------------------
# R3.5 — A3/A4 redefinition correctness invariance
#
# The kernel output MUST be bit-equivalent (within accumulator precision)
# regardless of (chunk_size, prefetch_depth). Only timing differs. We
# parameterize over the cross product to catch any silent breakage.
# ---------------------------------------------------------------------------

# Cross product: chunk_size ∈ {None, 2, 4, 8} × prefetch_depth ∈ {0, 1}
# Single world_size=2 to keep CI cost bounded; the math is layout-agnostic.
@pytest.mark.parametrize("prefetch_depth", [0, 1])
@pytest.mark.parametrize("chunk_size", [None, 2, 4, 8])
class TestKnobInvariance:
    """A3 (chunk_size) and A4 (prefetch_depth) must NOT change kernel output."""

    def test_prefill_invariant_under_knobs(self, chunk_size, prefetch_depth):
        from src.models.ring_attention_kernel import reference_attention
        torch.manual_seed(101)
        world_size = 2
        B, H, D = 1, 4, 16
        S_total = 8 * world_size
        q = torch.randn(B, H, S_total, D, dtype=torch.float32)
        k = torch.randn(B, H, S_total, D, dtype=torch.float32)
        v = torch.randn(B, H, S_total, D, dtype=torch.float32)
        ref_out = reference_attention(q, k, v, causal=True)
        _spawn(
            world_size, _worker_prefill,
            (q, k, v, ref_out, 1e-5, 1e-5, chunk_size, prefetch_depth),
        )

    def test_decode_invariant_under_knobs(self, chunk_size, prefetch_depth):
        from src.models.ring_attention_kernel import reference_attention
        torch.manual_seed(103)
        world_size = 2
        B, H, D = 1, 4, 16
        S_prefill = 8 * world_size
        S_q = S_tail = 3
        q_global = torch.randn(B, H, S_q, D, dtype=torch.float32)
        full_prefill_k = torch.randn(B, H, S_prefill, D, dtype=torch.float32)
        full_prefill_v = torch.randn(B, H, S_prefill, D, dtype=torch.float32)
        tail_k = torch.randn(B, H, S_tail, D, dtype=torch.float32)
        tail_v = torch.randn(B, H, S_tail, D, dtype=torch.float32)
        full_k = torch.cat([full_prefill_k, tail_k], dim=2)
        full_v = torch.cat([full_prefill_v, tail_v], dim=2)
        ref_out = reference_attention(q_global, full_k, full_v, causal=True)
        _spawn(
            world_size, _worker_decode,
            (q_global, full_prefill_k, full_prefill_v, tail_k, tail_v,
             S_tail, ref_out, 1e-5, 1e-5, chunk_size, prefetch_depth),
        )

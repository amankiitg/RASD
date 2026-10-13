"""repeat_kv after the rotation must be bit-identical, and must actually shrink the wire.

The change is small and its justification is structural: `repeat_kv` duplicates GQA heads,
the ring rotation permutes the SEQUENCE dimension, and the two operations commute — so
expanding on the receiving side lets the rotation ship `num_kv_heads` copies instead of
`num_heads` (32 -> 8 at 128k, 4x fewer bytes) while the values reaching attention, and their
head order, are unchanged.

"Commute" is an argument, not a measurement, and the failure modes it hides are exactly the
ones that matter: expanding the SEND buffer (no payload saving at all), expanding before the
rotation (`kv_groups` silently ignored), or expanding the wrong buffer after the receive
(wrong shard). So this test runs the REAL `_ring_against_sharded` loop across two emulated
ranks and asserts three things:

  1. the deferred run's output is BIT-IDENTICAL to the eager run's;
  2. both equal a reference attention over the concatenated shards;
  3. the tensors handed to the collective in the deferred run carry `num_kv_heads`, not
     `num_heads` -- the payload claim, which a correct-but-useless implementation fails.

The distributed emulation is the same kind of faithful fake as
`test_mlsys_split_kv_decode.py`: a blocking P2P exchange, where every participant that
issues a receive ends up holding the peer's bytes.
"""
from __future__ import annotations

import math
import sys
import threading
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

torch = pytest.importorskip("torch")
from src.models.ring_attention_kernel import (  # noqa: E402
    _attn_step,
    _expand_kv_heads,
    _ring_against_sharded,
)

_TLS = threading.local()


class _RingExchange:
    """A blocking two-rank ring exchange, one slot per rank, one step at a time.

    WHY THREADS. `_ring_against_sharded` is written for ranks that run concurrently: with
    `prefetch_depth=0` each rank issues its rotation at the end of step 0 and waits at the
    top of step 1. Running the two ranks sequentially would deadlock at the wait. Two
    threads with a barrier reproduce the real ordering.
    """

    def __init__(self, world_size: int, groups: int):
        self.world_size = world_size
        self.groups = groups
        self.barrier = threading.Barrier(world_size)
        self.sent_shapes: list[tuple[int, int, int, int]] = []
        self.lock = threading.Lock()
        self.mailbox: dict[int, tuple] = {}

    # -- torch.distributed surface used by _issue_rotation -------------------
    def is_initialized(self):
        return True

    def isend(self, *a, **k):
        return "isend"

    def irecv(self, *a, **k):
        return "irecv"

    def P2POp(self, op, tensor, peer, group=None):        # noqa: N802
        # `_issue_rotation` passes `dist.isend` / `dist.irecv` as VALUES, not as calls, so
        # the op arrives as the monkeypatched bound method. Its __name__ is the only stable
        # way to tell the two apart (each attribute access makes a fresh bound method).
        kind = op if isinstance(op, str) else getattr(op, "__name__", "irecv")
        return (kind, tensor, peer)

    def batch_isend_irecv(self, ops, group=None):
        rank = _TLS.rank
        send_k = send_v = recv_k = recv_v = None
        for op, tensor, peer in ops:
            if op == "isend":
                if send_k is None:
                    send_k = tensor
                else:
                    send_v = tensor
            else:
                if recv_k is None:
                    recv_k = tensor
                else:
                    recv_v = tensor
        with self.lock:
            self.sent_shapes.append(tuple(send_k.shape))
            self.mailbox[rank] = (send_k, send_v, recv_k, recv_v)
        # Both ranks arrive here once per step; the exchange happens after both have
        # published their buffers, which is what makes the copies well defined.
        self.barrier.wait()
        peer = (rank + 1) % self.world_size
        p_k, p_v, _, _ = self.mailbox[peer]
        recv_k.copy_(p_k)
        recv_v.copy_(p_v)
        return [_NoopReq()]


class _NoopReq:
    """The exchange above is synchronous, so waiting is a no-op (it over-synchronises,
    which is safe for a correctness test)."""

    def wait(self):
        return None


def _install_fake_dist(monkeypatch, exchange, rank):
    import torch.distributed as dist

    _TLS.rank = rank
    monkeypatch.setattr(dist, "batch_isend_irecv", exchange.batch_isend_irecv, raising=False)
    monkeypatch.setattr(dist, "P2POp", exchange.P2POp, raising=False)
    monkeypatch.setattr(dist, "isend", exchange.isend, raising=False)
    monkeypatch.setattr(dist, "irecv", exchange.irecv, raising=False)
    monkeypatch.setattr(dist, "is_initialized", exchange.is_initialized, raising=False)


def _run_two_ranks(k_local, v_local, q, scale, kv_groups, prefetch_depth=0):
    """Run `_ring_against_sharded` on two emulated ranks; returns {rank: (out, lse)}.

    The caller installs the fake `torch.distributed` (pointing at the SAME exchange it then
    reads `sent_shapes` from) before calling this.
    """
    world = 2
    per = k_local.shape[2] // world
    results: dict[int, tuple] = {}
    errors: list[BaseException] = []

    def worker(rank):
        _TLS.rank = rank
        try:
            sl = slice(rank * per, (rank + 1) * per)
            results[rank] = _ring_against_sharded(
                q, k_local[:, :, sl, :], v_local[:, :, sl, :],
                rank=rank, world_size=world, scale=scale, dropout_p=0.0,
                process_group=None, chunk_size=None, prefetch_depth=prefetch_depth,
                kv_groups=kv_groups,
            )
        except BaseException as exc:                       # noqa: BLE001 - re-raised below
            errors.append(exc)

    threads = [threading.Thread(target=worker, args=(r,)) for r in range(world)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=60)
    if errors:
        raise errors[0]
    assert len(results) == world, "a rank thread did not finish"
    return results


def test_deferred_expansion_is_bit_identical_and_ships_fewer_heads(monkeypatch):
    torch.manual_seed(0)
    B, S_q, D = 1, 1, 8
    num_kv_heads, groups = 2, 4
    num_heads = num_kv_heads * groups
    S_local = 6
    q = torch.randn(B, num_heads, S_q, D, dtype=torch.float32)
    k8 = torch.randn(B, num_kv_heads, 2 * S_local, D, dtype=torch.float32)
    v8 = torch.randn(B, num_kv_heads, 2 * S_local, D, dtype=torch.float32)
    scale = 1.0 / math.sqrt(D)

    ref, _ = _attn_step(q, k8.repeat_interleave(groups, dim=1), v8.repeat_interleave(groups, dim=1),
                        scale=scale, dropout_p=0.0, causal=False)

    # (a) eager: expand BEFORE the rotation, as every published row does today.
    eager_ex = _RingExchange(2, 1)
    _install_fake_dist(monkeypatch, eager_ex, 0)
    eager_out = _run_two_ranks(_expand_kv_heads(k8, groups),
                               _expand_kv_heads(v8, groups), q, scale, kv_groups=1)
    eager_heads = {s[1] for s in eager_ex.sent_shapes}

    # (b) deferred: rotate the num_kv_heads tensor, expand after receipt.
    deferred_ex = _RingExchange(2, groups)
    _install_fake_dist(monkeypatch, deferred_ex, 0)
    deferred_out = _run_two_ranks(k8, v8, q, scale, kv_groups=groups)
    deferred_heads = {s[1] for s in deferred_ex.sent_shapes}

    assert eager_heads, "the eager run issued no rotation — the emulation is not exercising it"
    assert torch.equal(deferred_out[0][0], eager_out[0][0]), \
        "deferred expansion must be bit-identical to expanding before the rotation"
    assert torch.equal(deferred_out[1][0], eager_out[1][0])
    assert torch.allclose(deferred_out[0][0], ref, atol=1e-5, rtol=1e-4), \
        f"ring result differs from full attention by {(deferred_out[0][0] - ref).abs().max():.3e}"

    assert eager_heads == {num_heads}, f"eager run should ship {num_heads} heads, saw {eager_heads}"
    assert deferred_heads == {num_kv_heads}, \
        f"deferred run must ship {num_kv_heads} heads (the payload claim), saw {deferred_heads}"


def test_deferred_expansion_with_async_prefetch(monkeypatch):
    """prefetch_depth>=1 takes a different branch inside the loop (issue before compute);
    the payload and the result must be unchanged."""
    torch.manual_seed(1)
    B, S_q, D = 1, 2, 8
    num_kv_heads, groups = 2, 2
    S_local = 4
    q = torch.randn(B, num_kv_heads * groups, S_q, D, dtype=torch.float32)
    k8 = torch.randn(B, num_kv_heads, 2 * S_local, D, dtype=torch.float32)
    v8 = torch.randn(B, num_kv_heads, 2 * S_local, D, dtype=torch.float32)
    scale = 1.0 / math.sqrt(D)
    ex = _RingExchange(2, groups)
    _install_fake_dist(monkeypatch, ex, 0)
    out = _run_two_ranks(k8, v8, q, scale, kv_groups=groups, prefetch_depth=1)
    assert {s[1] for s in ex.sent_shapes} == {num_kv_heads}
    ref, _ = _attn_step(q, k8.repeat_interleave(groups, dim=1), v8.repeat_interleave(groups, dim=1),
                        scale=scale, dropout_p=0.0, causal=False)
    assert torch.allclose(out[0][0], ref, atol=1e-5, rtol=1e-4)


def test_expand_kv_heads_matches_transformers_repeat_kv():
    hf = pytest.importorskip("transformers.models.llama.modeling_llama")
    torch.manual_seed(2)
    t = torch.randn(2, 3, 5, 4)
    got = _expand_kv_heads(t, 4)
    want = hf.repeat_kv(t, 4)
    assert got.shape == want.shape
    assert torch.equal(got, want), "the kernel's expansion must match HF's head order exactly"


def test_groups_of_one_is_a_passthrough():
    t = torch.randn(1, 4, 3, 2)
    assert _expand_kv_heads(t, 1) is t, "groups<=1 must not copy"

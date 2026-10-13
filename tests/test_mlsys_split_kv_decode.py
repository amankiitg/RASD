"""Split-KV decode: the merge must equal one attention over the concatenated shards.

The counterfactual replaces the ring rotation with "each rank attends to its own shard,
then merge the partials". The rotation is easy to trust; the merge is not -- a shard
counted twice, a max taken after the weights are formed, or an lse left in the wrong
convention all produce output that still looks like attention. So the merge is a pure
function of the partial lists and is tested here against the ground truth it is supposed to
reproduce: a single `_attn_step` over the concatenation of the shards.

These run on CPU: the merge is arithmetic, and the distributed path differs only in how the
partials are collected (one all_gather), which needs a pod rather than a unit test.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

torch = pytest.importorskip("torch")
from src.models.ring_attention_kernel import (  # noqa: E402
    _attn_step,
    merge_split_kv_partials,
    ring_attention_decode_split_kv,
)


def _partials(q, k, v, shards, scale):
    """Each shard's (out, lse), exactly as a rank would compute them."""
    parts = []
    per = k.shape[2] // shards
    for r in range(shards):
        sl = slice(r * per, (r + 1) * per)
        parts.append(_attn_step(q, k[:, :, sl, :], v[:, :, sl, :],
                                scale=scale, dropout_p=0.0, causal=False))
    return parts


def test_merge_reproduces_full_attention():
    torch.manual_seed(0)
    B, H, S_q, D, S_local, shards = 1, 2, 1, 8, 24, 3
    q = torch.randn(B, H, S_q, D, dtype=torch.float32)
    k = torch.randn(B, H, S_local * shards, D, dtype=torch.float32)
    v = torch.randn(B, H, S_local * shards, D, dtype=torch.float32)
    scale = 1.0 / math.sqrt(D)

    ref, ref_lse = _attn_step(q, k, v, scale=scale, dropout_p=0.0, causal=False)
    parts = _partials(q, k, v, shards, scale)
    merged, merged_lse = merge_split_kv_partials([p[0] for p in parts],
                                                 [p[1] for p in parts])
    assert torch.allclose(merged, ref, atol=1e-5, rtol=1e-4), \
        f"merge differs from full attention: max abs {(merged - ref).abs().max():.3e}"
    assert torch.allclose(merged_lse, ref_lse, atol=1e-4), \
        "the merged lse must be the logsumexp of the whole score set"


def test_a_single_partial_is_a_passthrough():
    """world_size == 1 must not change anything, so the switch is safe to leave on."""
    out = torch.randn(1, 2, 1, 8)
    lse = torch.randn(1, 2, 1)
    o, l = merge_split_kv_partials([out], [lse])
    assert torch.equal(o, out) and torch.equal(l, lse)


def test_a_shard_counted_twice_does_not_match():
    """The failure mode the merge could hide: a duplicated shard changes the answer, so a
    merge that double-counted the replicated tail would NOT pass the test above by luck."""
    torch.manual_seed(1)
    B, H, S_q, D, S_local = 1, 2, 1, 8, 24
    q = torch.randn(B, H, S_q, D, dtype=torch.float32)
    k = torch.randn(B, H, S_local * 2, D, dtype=torch.float32)
    v = torch.randn(B, H, S_local * 2, D, dtype=torch.float32)
    scale = 1.0 / math.sqrt(D)
    ref, _ = _attn_step(q, k, v, scale=scale, dropout_p=0.0, causal=False)
    parts = _partials(q, k, v, 2, scale)
    doubled, _ = merge_split_kv_partials([parts[0][0], parts[0][0]],
                                         [parts[0][1], parts[0][1]])
    assert not torch.allclose(doubled, ref, atol=1e-4), \
        "a duplicated shard must not look correct -- otherwise the merge test proves nothing"


def test_split_kv_decode_merges_the_tail_once():
    """The replicated tail is identical on every rank; merged after the combine it is
    counted once. Compare against full attention over [shard | tail]."""
    torch.manual_seed(2)
    B, H, S_q, D, S_pre, tail = 1, 2, 1, 8, 24, 5
    q = torch.randn(B, H, S_q, D, dtype=torch.float32)
    k_pre = torch.randn(B, H, S_pre, D, dtype=torch.float32)
    v_pre = torch.randn(B, H, S_pre, D, dtype=torch.float32)
    k_tail = torch.randn(B, H, tail, D, dtype=torch.float32)
    v_tail = torch.randn(B, H, tail, D, dtype=torch.float32)
    scale = 1.0 / math.sqrt(D)

    ref, _ = _attn_step(q, torch.cat([k_pre, k_tail], dim=2),
                        torch.cat([v_pre, v_tail], dim=2),
                        scale=scale, dropout_p=0.0, causal=False)
    k_local = torch.cat([k_pre, k_tail], dim=2)
    v_local = torch.cat([v_pre, v_tail], dim=2)
    got = ring_attention_decode_split_kv(q, k_local, v_local, rank=0, world_size=1,
                                         tail_count=tail, scale=scale, dropout_p=0.0)
    assert torch.allclose(got, ref, atol=1e-5, rtol=1e-4), \
        f"tail counted wrong: max abs {(got - ref).abs().max():.3e}"

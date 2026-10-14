"""The NF4 tail fold must be exact, and must bound the per-step chunk count.

A target-only decode appends one token per step. Every append used to create its
own chunk, so after N steps a layer held N+1 chunks and `_dequantize_layer`
concatenated all of them on EVERY forward: per-step cost growing with the number
of steps, which is the one shape a decode loop cannot have.

The fold concatenates two small chunks along the sequence axis. That is exact --
not an approximation -- because `quantize_nf4` blocks run along the HEAD
dimension, so a position's codes and scales are independent of its neighbours and
the merged tensor holds the same per-position blocks in the same order. These
tests assert that equality bitwise, which is what makes the optimisation safe to
apply to a number that goes in a paper.
"""
from __future__ import annotations

import pytest
import torch

from src.models.nf4_dynamic_cache import NF4DynamicCache

D = 32          # head dim; must be divisible by block_size
H = 2
BLOCK = 8


def _cache(**kw) -> NF4DynamicCache:
    return NF4DynamicCache(block_size=BLOCK, dtype=torch.float32,
                           bf16_prefix_size=0, update_chunk_size=0, **kw)


def _tokens(n: int, seed: int = 0) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    return torch.randn(1, H, n, D, generator=g)


def test_a_token_by_token_fill_matches_a_single_shot_fill():
    """Same values, whether the sequence arrives at once or one token at a time."""
    whole = _tokens(128)
    one_shot = _cache()
    one_shot.update(whole, whole.clone(), layer_idx=0)
    k_whole, v_whole = one_shot._dequantize_layer(0)

    incremental = _cache()
    for i in range(whole.shape[2]):
        incremental.update(whole[:, :, i:i + 1, :].contiguous(),
                           whole[:, :, i:i + 1, :].contiguous(), layer_idx=0)
    k_inc, v_inc = incremental._dequantize_layer(0)

    assert torch.equal(k_whole, k_inc), (
        "the tail fold changed the stored values; it must only change how they "
        "are laid out in chunks")
    assert torch.equal(v_whole, v_inc)


def test_the_fold_bounds_the_chunk_count_of_a_long_decode():
    cache = _cache()
    n = 512
    for i in range(n):
        t = _tokens(1, seed=i)
        cache.update(t, t.clone(), layer_idx=0)

    n_chunks = len(cache._k_codes[0])
    assert cache.get_seq_length(0) == n, "every token must still be stored"
    assert n_chunks <= n / cache.tail_merge_below + 2, (
        f"{n_chunks} chunks for {n} one-token appends: the per-step dequantize "
        f"cost still grows with the number of steps")
    assert n_chunks > 1, "the fold must not collapse the whole cache into one"


def test_disabling_the_fold_restores_one_chunk_per_append():
    cache = _cache(tail_merge_below=0)
    for i in range(32):
        t = _tokens(1, seed=i)
        cache.update(t, t.clone(), layer_idx=0)
    assert len(cache._k_codes[0]) == 32


def test_truncation_still_sees_a_consistent_cache_after_folding():
    cache = _cache()
    whole = _tokens(200, seed=7)
    for i in range(whole.shape[2]):
        cache.update(whole[:, :, i:i + 1, :].contiguous(),
                     whole[:, :, i:i + 1, :].contiguous(), layer_idx=0)

    cache.truncate(50)
    assert cache.get_seq_length(0) == 50
    k, _ = cache._dequantize_layer(0)
    assert k.shape[2] == 50
    # The surviving prefix must equal the first 50 tokens dequantized on their
    # own: folding must not have moved a value across a chunk boundary.
    reference = _cache()
    reference.update(whole[:, :, :50, :].contiguous(),
                     whole[:, :, :50, :].contiguous(), layer_idx=0)
    k_ref, _ = reference._dequantize_layer(0)
    assert torch.equal(k, k_ref)


def test_a_negative_threshold_is_refused():
    with pytest.raises(ValueError):
        _cache(tail_merge_below=-1)

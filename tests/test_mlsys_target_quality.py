"""Target-quality perplexity: chunked NLL, and shard bounds that tile exactly."""
from __future__ import annotations

import pytest
import torch

from src.analysis.target_quality import (
    head_chunked_nll, local_bounds, perplexity_from_sums,
)


class _Tiny(torch.nn.Module):
    """Minimal stand-in exposing the two attributes the scorer touches."""

    def __init__(self, dim=4, vocab=7, seed=0):
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        self.lm_head = torch.nn.Linear(dim, vocab, bias=False)
        with torch.no_grad():
            self.lm_head.weight.copy_(torch.randn(vocab, dim, generator=g))


def test_chunked_nll_matches_reference_and_bounds_tile_the_sequence():
    model = _Tiny()
    torch.manual_seed(0)
    hidden = torch.randn(1, 6, 4)
    targets = torch.randint(0, 7, (1, 6))

    # Chunking must be the identical cross-entropy, not an approximation: the
    # whole point of chunking is to bound memory without changing the number.
    # The tolerance is 1e-6 rather than tighter because the chunked path
    # accumulates each chunk's fp32 loss into a float64 total, so it is
    # marginally MORE accurate than this all-in-fp32 reference rather than
    # identical to it.
    ref = torch.nn.functional.cross_entropy(
        model.lm_head(hidden).reshape(-1, 7), targets.reshape(-1),
        reduction="sum").item()
    for chunk in (1, 2, 6, 64):
        total, n = head_chunked_nll(model, hidden, targets, chunk=chunk)
        assert n == 6
        assert total == pytest.approx(ref, rel=1e-6)

    assert perplexity_from_sums(0.0, 4) == 1.0

    # Every scored position must be claimed by exactly one rank, and no rank may
    # score a position before `score_from`. A gap or an overlap would silently
    # change the denominator of the perplexity. A rank whose whole slice lies
    # before `score_from` correctly scores nothing, so lo == hi is allowed.
    n_total, world, score_from = 64, 4, 40
    claimed: list[int] = []
    for rank in range(world):
        slice_start, slice_end, lo, hi = local_bounds(
            rank, world, n_total, score_from)
        assert slice_start >= 0 and slice_end <= n_total
        assert 0 <= lo <= hi <= slice_end - slice_start
        claimed.extend(range(slice_start + lo, slice_start + hi))
        assert slice_end - slice_start <= n_total // world + 1
    assert sorted(claimed) == list(range(score_from, n_total))

    # Rank 0 of that layout owns [0, 16) and scores nothing: below score_from
    # the sequence is prompt, which supplies context but not loss. The bounds
    # stay inside the local slice rather than running off its end.
    assert local_bounds(0, 4, 64, 40)[2:4] == (16, 16)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))

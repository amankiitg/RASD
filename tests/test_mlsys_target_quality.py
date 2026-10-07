"""Target-quality perplexity: chunked NLL, and shard bounds that tile exactly."""
from __future__ import annotations

import math

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
    #
    # The scored global position is `own_start + local_index + 1`: hidden[i]
    # predicts the token one position after it.
    n_total, world, score_from = 64, 4, 40
    claimed: list[int] = []
    for rank in range(world):
        b = local_bounds(rank, world, n_total, score_from)
        assert b.own_start >= 0 and b.own_end <= n_total
        assert 0 <= b.lo <= b.hi <= b.fwd_end - b.own_start
        claimed.extend(range(b.own_start + b.lo + 1, b.own_start + b.hi + 1))
        assert b.fwd_end - b.own_start == math.ceil(n_total / world)
    assert sorted(claimed) == list(range(score_from, n_total))

    # Rank 0 of that layout owns [0, 16) and scores nothing: below score_from
    # the sequence is prompt, which supplies context but not loss. The bounds
    # stay inside the local slice rather than running off its end.
    b0 = local_bounds(0, 4, 64, 40)
    assert (b0.lo, b0.hi) == (16, 16)
    assert (b0.score_lo, b0.score_hi) == (16, 16)


@pytest.mark.parametrize("n_total,world", [(13, 4), (17, 8), (129, 8),
                                           (1000, 8), (9, 8), (1, 8)])
def test_uneven_shards_keep_every_rank_on_the_same_collective_count(n_total,
                                                                   world):
    """Equal forwarded length is what makes the ring safe with a remainder.

    The ring rotates each rank's own KV slice, and `_issue_rotation` splits that
    slice into `ceil(S_local / chunk_size)` chunks, so a rank with a different
    local length would submit a different number of P2P ops and the ring would
    hang rather than merely slow down. The padding in `local_bounds` is the
    mechanism that prevents it, so it is asserted here rather than assumed.
    """
    chunk = 512
    lengths, op_counts, scored = set(), set(), []
    for rank in range(world):
        b = local_bounds(rank, world, n_total, score_from=1)
        local_len = b.fwd_end - b.own_start
        lengths.add(local_len)
        op_counts.add(len(range(0, local_len, chunk)))
        scored.extend(range(b.score_lo, b.score_hi))
        assert b.own_end <= b.fwd_end
        assert b.fwd_end - b.own_end <= local_len
    assert len(lengths) == 1, (
        f"unequal forward lengths {lengths} -> unequal ring op counts")
    assert len(op_counts) == 1
    # Every position except 0 (no predictor) is scored exactly once, in order.
    assert sorted(scored) == list(range(1, n_total))


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))

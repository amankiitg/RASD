"""Target quality measured beside acceptance: conditioned continuation NLL.

Why this is a separate module from `src/analysis/perplexity.py`: that module's
`compute_perplexity` calls the model with `labels=` and reads HF's mean CE loss,
which materialises the full logit tensor in fp32. At 32k positions that is
4.2 GiB on top of the KV cache and it OOMs, which is why the earlier long-context
probes had to be rewritten before they could run at all.

Two things are different here:

* the LM head is applied in position chunks, which is the *identical* shifted
  cross-entropy with bounded memory rather than an approximation;
* only the target is scored, over a window defined by the caller, and under
  sequence-parallel sharding the NLL is computed per rank over the positions it
  owns and summed across ranks, because no single rank holds the full sequence.

`continuation_nll` is deliberately shaped so the sharding logic is a pure
function (`local_bounds`) that can be tested without any distributed setup.
"""
from __future__ import annotations

import math
from typing import Callable, NamedTuple, Optional

import torch

HEAD_CHUNK = 1024


class ShardBounds(NamedTuple):
    """One rank's slice of the sequence, and the positions it scores.

    `own_start`/`own_end` are the REAL tokens the rank forwards;
    `own_end <= fwd_end` because a rank at the end of an uneven split forwards
    pad positions (`fwd_end - own_end` of them) so that every rank's forwarded
    length is identical. Identical lengths are what make every rank issue the
    same number of ring collectives: the ring rotates KV slices of the local
    length, and `_issue_rotation` splits that slice into
    `ceil(S_local / chunk_size)` chunks, so unequal lengths mean unequal op
    counts -- i.e. a hang, not a slow path.

    `lo`/`hi` index the rank's own hidden states: `hidden[:, lo:hi, :]` predicts
    the global targets `[score_lo, score_hi)`, which the caller reads from the
    FULL id tensor (`ids[:, score_lo:score_hi]`). The target at a rank's last
    scored position is predicted by hidden at `own_end - 1`, so it is a target
    the rank never forwarded -- but every rank holds the full ids, so reading it
    costs no communication. That is the whole point: the loss is exact, and the
    only communication is the ring's own KV rotation.

    Pad positions are never scored (`score_hi <= n_total < fwd_end`), and no real
    position attends to them (attention is causal within the ring), so the pad
    cannot change a scored hidden state.
    """

    own_start: int
    own_end: int
    fwd_end: int
    lo: int
    hi: int
    score_lo: int
    score_hi: int


def local_bounds(rank: int, world_size: int, n_total: int,
                 score_from: int) -> ShardBounds:
    """Slice and scoring range for one rank, contiguous and DISJOINT.

    Ranks own contiguous slices `[own_start, own_end)` of the sequence, matching
    `RASDInference.generate`'s prefill layout. Slices do not overlap: this
    function used to pull the slice back by one token so that the hidden state
    predicting a rank's first token was available locally, which meant every
    rank forwarded one position it did not own. Since each rank can read any
    target token from the full ids, that pull-back bought nothing and cost a
    duplicated position per rank per forward.

    Scoring: global positions `[max(own_start + 1, score_from, 1), own_end + 1)`
    clamped to `n_total`. Position 0 has no predictor, so rank 0 starts at 1.
    The union across ranks is exactly `[score_from, n_total)` -- every scored
    position is claimed once, so the perplexity denominator is the same whether
    it is computed sharded or whole.

    Uneven splits are supported: `n_total` need not be divisible by
    `world_size`. The remainder becomes pad positions on the last rank's forward
    (`fwd_end > own_end`), which is what keeps the collective count identical on
    every rank.
    """
    if world_size < 1:
        raise ValueError("world_size must be >= 1")
    if not 0 <= rank < world_size:
        raise ValueError(f"rank {rank} outside world_size {world_size}")
    if n_total < 1:
        raise ValueError("n_total must be >= 1")
    s_local = math.ceil(n_total / world_size)
    own_start = min(rank * s_local, n_total)
    own_end = min(own_start + s_local, n_total)
    # Every rank forwards exactly s_local positions, pad included, so shapes and
    # therefore collective counts match. Ranks whose slice is entirely pad (only
    # possible when world_size > n_total) forward s_local pads and score nothing.
    fwd_end = own_start + s_local
    score_lo = max(own_start + 1, score_from, 1)
    score_hi = min(own_end + 1, n_total)
    empty = score_hi <= score_lo
    if empty:
        # Keep lo/hi as an ordered, in-range pair at the end of the slice so a
        # caller that slices blindly gets an empty tensor rather than a range
        # that runs off its own slice.
        lo = hi = own_end - own_start
        score_lo = score_hi = own_end
    else:
        # hidden[i] (i local) predicts global position own_start + i + 1, so the
        # hidden index for global target q is q - own_start - 1.
        lo = score_lo - own_start - 1
        hi = score_hi - own_start - 1
    return ShardBounds(own_start, own_end, fwd_end, lo, hi, score_lo, score_hi)



def head_chunked_nll(model, hidden: torch.Tensor, target_ids: torch.Tensor,
                     chunk: int = HEAD_CHUNK) -> tuple[float, int]:
    """Sum of negative log-likelihood and token count over aligned positions.

    `hidden[:, i, :]` is the state that predicts `target_ids[:, i]`. The LM head
    is applied in `chunk`-sized position blocks and each block's logits are
    released before the next, so peak extra memory is `chunk x vocab` instead of
    `seq_len x vocab`.
    """
    if hidden.shape[1] != target_ids.shape[1]:
        raise ValueError(
            f"hidden has {hidden.shape[1]} positions but {target_ids.shape[1]} "
            f"targets were supplied; they must be already aligned"
        )
    total = torch.zeros((), dtype=torch.float64)
    n = 0
    for i in range(0, hidden.shape[1], chunk):
        j = min(i + chunk, hidden.shape[1])
        logits = model.lm_head(hidden[:, i:j, :]).float()
        total += torch.nn.functional.cross_entropy(
            logits.reshape(-1, logits.shape[-1]),
            target_ids[:, i:j].reshape(-1),
            reduction="sum",
        ).double().cpu()
        n += int(target_ids[:, i:j].numel())
        del logits
    return float(total), n


def continuation_nll(
    model,
    input_ids: torch.Tensor,
    score_from: int,
    forward: Optional[Callable] = None,
    rank: int = 0,
    world_size: int = 1,
    chunk: int = HEAD_CHUNK,
) -> tuple[float, int, torch.Tensor]:
    """Local (sum NLL, count, hidden) for the positions this rank scores.

    Runs under `torch.no_grad()`: this is a measurement, and building an
    autograd graph over a 128k-token forward would add activation memory to a
    stage that is already close to the card's limit for no benefit.

    `forward(local_ids, abs_positions)` must return the local hidden states. It
    exists as a parameter so the sharded case can pass the ring-attention
    forward (which needs absolute position ids and an empty cache) while the
    single-rank case uses a plain model forward. `abs_positions` are absolute so
    RoPE is applied at the true global positions of the shard, matching the
    engine's prefill layout.

    Returns the *local* sums; the caller all-reduces. Returning local values
    keeps this function free of distributed state and testable on one process.

    The third return value is the hidden slice that was scored (not the whole
    forwarded window), so a caller that wants to inspect what produced the loss
    does not have to reproduce the bounds.
    """
    ids = input_ids if input_ids.dim() == 2 else input_ids.unsqueeze(0)
    n_total = int(ids.shape[1])
    b = local_bounds(rank, world_size, n_total, score_from)
    # Forward EXACTLY the rank's slice, padded to a common length so that every
    # rank issues the same ring collectives. Pad positions are never scored and
    # no real position attends to them (causal), so they cannot move a number.
    local_ids = ids[:, b.own_start:b.own_end]
    n_pad = b.fwd_end - b.own_end
    if n_pad > 0:
        local_ids = torch.cat(
            [local_ids, local_ids[:, -1:].expand(-1, n_pad)], dim=1)
    local_ids = local_ids.contiguous()
    abs_pos = torch.arange(b.own_start, b.fwd_end,
                           device=ids.device).unsqueeze(0)
    with torch.no_grad():
        hidden = forward(local_ids, abs_pos)
        pred = hidden[:, b.lo:b.hi, :]
        # The targets come from the FULL ids: a rank's last scored position is
        # predicted by its own last hidden state but the token itself belongs to
        # the next slice. Every rank holds the full ids, so this needs no
        # communication.
        targets = ids[:, b.score_lo:b.score_hi]
        total, n = head_chunked_nll(model, pred, targets, chunk=chunk)
    return total, n, pred


def perplexity_from_sums(total_nll: float, count: int) -> float:
    if count <= 0:
        raise ValueError("cannot form a perplexity from zero scored tokens")
    return float(torch.exp(torch.tensor(total_nll / count, dtype=torch.float64)))

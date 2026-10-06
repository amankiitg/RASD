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

from typing import Callable, Optional

import torch

HEAD_CHUNK = 1024


def local_bounds(rank: int, world_size: int, n_total: int, score_from: int):
    """Slice and scoring range for one rank under contiguous sharding.

    Ranks own contiguous equal slices `[rank*S, (rank+1)*S)` of a sequence of
    `n_total` tokens, matching `RASDInference.generate`'s prefill layout.

    Returns `(slice_start, slice_end, score_from_local, score_to_local)`.

    `slice_start` is pulled back by one token so that the hidden state which
    *predicts* the rank's first owned token is available locally: the shifted
    cross-entropy for token `p` needs the hidden state at `p - 1`, and under
    sharding `p - 1` may belong to the previous rank. Rank 0 cannot pull back
    and so scores from the second token it owns, which is also the earliest
    position that has a predictor at all.

    Only positions at or after the global `score_from` are scored, so the
    prompt contributes context without contributing loss.
    """
    if world_size < 1:
        raise ValueError("world_size must be >= 1")
    if not 0 <= rank < world_size:
        raise ValueError(f"rank {rank} outside world_size {world_size}")
    s_local = n_total // world_size
    if s_local == 0:
        raise ValueError(f"n_total {n_total} too short for {world_size} ranks")
    own_start = rank * s_local
    own_end = own_start + s_local if rank < world_size - 1 else n_total
    slice_start = max(0, own_start - 1)
    score_lo = max(own_start, score_from, 1)
    score_hi = max(score_lo, own_end)
    # Clamp to the local slice so the returned bounds are always valid indices
    # into it, and so a rank whose whole slice lies before `score_from` reports
    # an empty range rather than one that runs off the end of its own slice.
    local_len = own_end - slice_start
    lo = min(score_lo - slice_start, local_len)
    hi = min(max(score_hi - slice_start, lo), local_len)
    return slice_start, own_end, lo, hi


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

    `forward(local_ids, abs_positions)` must return the local hidden states. It
    exists as a parameter so the sharded case can pass the ring-attention
    forward (which needs absolute position ids and an empty cache) while the
    single-rank case uses a plain model forward. `abs_positions` are absolute so
    RoPE is applied at the true global positions of the shard, matching the
    engine's prefill layout.

    Returns the *local* sums; the caller all-reduces. Returning local values
    keeps this function free of distributed state and testable on one process.
    """
    ids = input_ids if input_ids.dim() == 2 else input_ids.unsqueeze(0)
    n_total = int(ids.shape[1])
    slice_start, slice_end, lo, hi = local_bounds(
        rank, world_size, n_total, score_from
    )
    local_ids = ids[:, slice_start:slice_end].contiguous()
    abs_pos = torch.arange(slice_start, slice_end, device=ids.device).unsqueeze(0)
    hidden = forward(local_ids, abs_pos)[:, lo - 1:hi - 1, :]
    targets = local_ids[:, lo:hi]
    total, n = head_chunked_nll(model, hidden, targets, chunk=chunk)
    return total, n, hidden


def perplexity_from_sums(total_nll: float, count: int) -> float:
    if count <= 0:
        raise ValueError("cannot form a perplexity from zero scored tokens")
    return float(torch.exp(torch.tensor(total_nll / count, dtype=torch.float64)))

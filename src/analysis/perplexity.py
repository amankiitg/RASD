"""Perplexity evaluator (mentor M4 deliverable C4).

PPL = exp(-1/N * Σ log P(x_i | x_{<i})) over a held-out token sequence.
Used for: (a) the headline PPL metric the mentor lists alongside tps /
α / TTFT, (b) the C11 NF4 KV-cache validation gate (target ≤ 2% PPL
degradation on PG-19 vs bf16 baseline).

Implementation notes:
- For sequences ≤ model.max_position_embeddings, runs a single-pass
  forward with `labels=input_ids` and uses HF's built-in shifted CE
  loss (out.loss is mean NLL over non-padding tokens).
- For longer sequences, slides a max_length window with `stride`
  steps. Each window scores only the (end - prev_end) tokens that
  are newly visible — earlier positions are masked with label=-100
  (HF cross-entropy ignores). This is the canonical HF tutorial
  formulation; gives the per-token NLL averaged over the full
  sequence. See https://huggingface.co/docs/transformers/perplexity
- Pure function: model + input_ids in, scalar PPL out. No I/O, no
  global state, easy to test on CPU.
"""
from __future__ import annotations

from typing import Optional

import torch


def compute_perplexity(
    model,
    input_ids: torch.Tensor,
    max_length: Optional[int] = None,
    stride: Optional[int] = None,
) -> float:
    """Sliding-window perplexity over a sequence.

    Args:
        model        : HuggingFace causal LM (must accept labels= kwarg
                       and return out.loss as mean NLL).
        input_ids    : (1, L) tensor of token IDs. Batch size > 1 is
                       not supported — PPL is naturally a per-sequence
                       metric and aggregation across multiple sequences
                       happens at the caller (e.g., over PG-19 chunks).
        max_length   : Window size for the sliding pass. Defaults to
                       model.config.max_position_embeddings.
        stride       : How far to advance the window each step. Smaller
                       stride = more overlap = more accurate (each
                       prediction has more context), but more compute.
                       Defaults to max_length // 2.

    Returns
        ppl          : float, exp(mean NLL).
    """
    if input_ids.dim() != 2 or input_ids.shape[0] != 1:
        raise ValueError(
            f"compute_perplexity expects (1, L); got {tuple(input_ids.shape)}"
        )

    if max_length is None:
        max_length = int(model.config.max_position_embeddings)
    if stride is None:
        stride = max_length // 2 if max_length > 1 else 1
    # Defensive: stride must make forward progress
    stride = max(1, stride)

    L = input_ids.shape[1]
    device = input_ids.device

    # ----- Short sequence: single-pass -----
    if L <= max_length:
        with torch.no_grad():
            out = model(input_ids, labels=input_ids)
        # HF causal LM internally shifts labels; out.loss is the mean
        # NLL over tokens with label != -100. For input_ids labels,
        # the first token has no preceding context so it's effectively
        # unscored after the shift — that's intentional and matches
        # the standard PPL formulation.
        loss = out.loss.detach()
        return float(torch.exp(loss).item())

    # ----- Long sequence: sliding window -----
    nlls: list[torch.Tensor] = []
    counts: list[int] = []
    prev_end = 0
    for begin in range(0, L, stride):
        end = min(begin + max_length, L)
        # Number of *new* tokens scored in this window. Earlier
        # positions in [begin, prev_end) were already scored in
        # previous windows; we mask them to -100 so HF's CE ignores.
        trg_len = end - prev_end
        if trg_len <= 0:
            break
        chunk = input_ids[:, begin:end]
        target = chunk.clone()
        # Mask the (window_len - trg_len) leading tokens to -100
        target[:, :-trg_len] = -100
        with torch.no_grad():
            out = model(chunk, labels=target)
        # out.loss is mean NLL over non-(-100) tokens in this window.
        # Multiply by trg_len to convert mean-over-this-window back to
        # sum-NLL contributed by this window.
        nlls.append(out.loss.detach() * trg_len)
        counts.append(trg_len)
        prev_end = end
        if end >= L:
            break

    total_nll   = torch.stack(nlls).sum()
    total_count = sum(counts)
    if total_count == 0:
        # Defensive: shouldn't happen for L > 0, but avoid div by zero
        return float("nan")
    return float(torch.exp(total_nll / total_count).item())

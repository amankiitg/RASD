"""
Ring attention free function — layout-agnostic kernel for sequence-parallel
attention with FlashAttention-2 intra-rank kernels and online-softmax merge.

Extracted from src/models/ring_attention_flash.py (R1 of M3_RING_INTEGRATION_PLAN.md)
so that both the standalone RingAttentionFlash module and the future
LlamaAttention patch can call the same kernel.

Layout
------
Contiguous slice per rank (R0.1 in the plan doc):
    Rank r holds K/V for absolute positions [r * S_local, (r+1) * S_local).

Causal masking per ring step
----------------------------
Rank r at step s reads K/V from source rank sr = (r - s) mod W:
    sr <  r : full attend (past rank)            — causal=False, no mask
    sr == r : self-step, FA-2 causal=True        — within-slice diagonal mask
    sr >  r : skip (future rank, would violate causality)

This pattern only handles **prefill** where every rank's queries are at the
"end" of its own slice. For autoregressive decode, queries can be at
positions strictly past the prefilled K/V — see ring_attention_decode below.
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
import torch.nn.functional as F

try:
    import torch.distributed as dist
    _DIST_AVAILABLE = True
except ImportError:
    _DIST_AVAILABLE = False

try:
    from flash_attn import flash_attn_func as _flash_attn_func
    _FLASH_AVAILABLE = True
except ImportError:
    _FLASH_AVAILABLE = False


# ---------------------------------------------------------------------------
# Online softmax merge — log-sum-exp identity
# ---------------------------------------------------------------------------

def _combine(out_acc: torch.Tensor, lse_acc: torch.Tensor,
             out_step: torch.Tensor, lse_step: torch.Tensor
             ) -> Tuple[torch.Tensor, torch.Tensor]:
    """Merge one ring step's (out, lse) into the running accumulator.

    out_acc, out_step : (B, H, S_q, D)  model dtype (bf16/fp16)
    lse_acc, lse_step : (B, H, S_q)     fp32 (kept high-precision)
    """
    lse_new = torch.logaddexp(lse_acc, lse_step)
    w_acc   = torch.exp(lse_acc  - lse_new).unsqueeze(-1).to(out_acc.dtype)
    w_step  = torch.exp(lse_step - lse_new).unsqueeze(-1).to(out_acc.dtype)
    out_new = w_acc * out_acc + w_step * out_step.to(out_acc.dtype)
    return out_new, lse_new


def _flash_step(q_bshd: torch.Tensor, k_bshd: torch.Tensor, v_bshd: torch.Tensor,
                scale: float, dropout_p: float, causal: bool
                ) -> Tuple[torch.Tensor, torch.Tensor]:
    """One FA-2 call. q/k/v in (B, S, H, D). Returns (out (B,H,S,D), lse (B,H,S) fp32)."""
    out_fa, lse, _ = _flash_attn_func(
        q_bshd, k_bshd, v_bshd,
        dropout_p=dropout_p,
        softmax_scale=scale,
        causal=causal,
        return_attn_probs=True,
    )
    out = out_fa.permute(0, 2, 1, 3).contiguous()  # (B, H, S, D)
    return out, lse  # lse is already (B, H, S) fp32


def _sdpa_step(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
               scale: float, causal: bool
               ) -> Tuple[torch.Tensor, torch.Tensor]:
    """Reference attention (no FA). Computes scores explicitly so we can return lse.

    q, k, v : (B, H, S_q/S_k, D)
    Returns (out (B,H,S_q,D), lse (B,H,S_q) fp32).

    Used as a fallback when flash_attn is unavailable, and as the test reference.
    """
    scores = torch.einsum("bhqd,bhkd->bhqk", q.float(), k.float()) * scale
    if causal:
        S_q, S_k = scores.shape[-2], scores.shape[-1]
        # Causal mask aligned to the right (last position): standard attention
        # convention is q[i] attends to k[j] iff j <= i + (S_k - S_q).
        # Same as FA-2's causal=True semantics.
        mask = torch.ones(S_q, S_k, device=scores.device, dtype=torch.bool).tril(
            diagonal=S_k - S_q
        )
        scores = scores.masked_fill(~mask, float("-inf"))
    lse = torch.logsumexp(scores, dim=-1)                 # (B, H, S_q) fp32
    probs = torch.softmax(scores, dim=-1)
    out = torch.einsum("bhqk,bhkd->bhqd", probs, v.float()).to(q.dtype)
    return out, lse


def _attn_step(q_bhsd: torch.Tensor, k_bhsd: torch.Tensor, v_bhsd: torch.Tensor,
               scale: float, dropout_p: float, causal: bool
               ) -> Tuple[torch.Tensor, torch.Tensor]:
    """Dispatch to FA-2 if available, else SDPA reference. Layout-normalising wrapper.

    Inputs in (B, H, S, D); returns (out (B,H,S_q,D), lse (B,H,S_q) fp32).
    """
    if _FLASH_AVAILABLE and q_bhsd.is_cuda and q_bhsd.dtype in (torch.float16, torch.bfloat16):
        q_bshd = q_bhsd.permute(0, 2, 1, 3).contiguous()
        k_bshd = k_bhsd.permute(0, 2, 1, 3).contiguous()
        v_bshd = v_bhsd.permute(0, 2, 1, 3).contiguous()
        return _flash_step(q_bshd, k_bshd, v_bshd, scale, dropout_p, causal)
    return _sdpa_step(q_bhsd, k_bhsd, v_bhsd, scale, causal)


# ---------------------------------------------------------------------------
# Ring attention — prefill path (Q is also sequence-sharded)
# ---------------------------------------------------------------------------

def ring_attention_prefill(
    q_local: torch.Tensor,        # (B, H, S_local, D)  — this rank's Q slice
    k_local: torch.Tensor,        # (B, H, S_local, D)  — this rank's K slice
    v_local: torch.Tensor,        # (B, H, S_local, D)  — this rank's V slice
    rank: int,
    world_size: int,
    scale: Optional[float] = None,
    dropout_p: float = 0.0,
    process_group: Optional["dist.ProcessGroup"] = None,
) -> torch.Tensor:
    """Causal ring attention for prefill (Q sharded contiguously like K/V).

    Each rank's Q sees K/V from all ranks at "past" or "self" position;
    future ranks are skipped (causal).

    Args
        q_local, k_local, v_local : (B, H, S_local, D), S_local = N / world_size
        rank, world_size          : ring topology
        scale                     : 1/sqrt(D) if None
        process_group             : optional dist process group

    Returns
        out_local : (B, H, S_local, D) — attention output for this rank's queries

    Communication: world_size - 1 batched isend/irecv rounds.
    Memory: O(B*H*S_local*D) per ring step — no full (S, S) materialised.
    """
    B, H, S_local, D = q_local.shape
    if scale is None:
        scale = 1.0 / math.sqrt(D)
    assert k_local.shape == (B, H, S_local, D), "k_local shape mismatch"
    assert v_local.shape == (B, H, S_local, D), "v_local shape mismatch"

    if world_size == 1:
        out, _lse = _attn_step(q_local, k_local, v_local, scale, dropout_p, causal=True)
        return out

    assert _DIST_AVAILABLE and dist.is_initialized(), \
        "ring_attention_prefill requires torch.distributed initialised when world_size > 1"

    send_to   = (rank + 1) % world_size
    recv_from = (rank - 1) % world_size

    # The K/V slice we currently hold rotates: at step s, we hold the slice
    # from source rank sr = (rank - s) % W.
    k_cur = k_local.contiguous()
    v_cur = v_local.contiguous()
    k_buf = torch.empty_like(k_cur)
    v_buf = torch.empty_like(v_cur)

    out_acc: Optional[torch.Tensor] = None
    lse_acc: Optional[torch.Tensor] = None

    for step in range(world_size):
        sr = (rank - step) % world_size

        if sr <= rank:
            # sr < rank → past rank (full attend, causal=False)
            # sr == rank → self (causal=True within slice)
            causal_this_step = (sr == rank)
            out_step, lse_step = _attn_step(
                q_local, k_cur, v_cur,
                scale=scale, dropout_p=dropout_p, causal=causal_this_step,
            )
            if out_acc is None:
                out_acc, lse_acc = out_step, lse_step
            else:
                out_acc, lse_acc = _combine(out_acc, lse_acc, out_step, lse_step)
        # else: sr > rank → future, skip (causality)

        # Rotate K/V to next rank for the next ring step
        if step < world_size - 1:
            ops = [
                dist.P2POp(dist.isend, k_cur, send_to, group=process_group),
                dist.P2POp(dist.isend, v_cur, send_to, group=process_group),
                dist.P2POp(dist.irecv, k_buf, recv_from, group=process_group),
                dist.P2POp(dist.irecv, v_buf, recv_from, group=process_group),
            ]
            reqs = dist.batch_isend_irecv(ops)
            for r in reqs:
                r.wait()
            k_cur, k_buf = k_buf, k_cur
            v_cur, v_buf = v_buf, v_cur

    if out_acc is None:
        # Only happens if world_size == 1 (handled above) or rank 0 with all
        # future-rank skips, which can't happen for sr=rank=0 (always self).
        raise RuntimeError("ring_attention_prefill produced no output — rank/world_size mismatch")

    return out_acc


# ---------------------------------------------------------------------------
# Ring attention — decode path (Q is replicated across ranks at the "tail")
# ---------------------------------------------------------------------------

def ring_attention_decode(
    q_global: torch.Tensor,       # (B, H, S_q, D)  — Q for newly-decoded positions, replicated
    k_local: torch.Tensor,        # (B, H, S_k_local, D) — local K slice
    v_local: torch.Tensor,        # (B, H, S_k_local, D) — local V slice
    rank: int,
    world_size: int,
    new_kv_owner_rank: int,       # which rank owns the just-appended K/V tail
    new_kv_count: int,            # how many positions at the END of that rank's slice are "new"
    scale: Optional[float] = None,
    dropout_p: float = 0.0,
    process_group: Optional["dist.ProcessGroup"] = None,
) -> torch.Tensor:
    """Ring attention for decode/verify: Q at the tail attending to all past K/V.

    Q is replicated on every rank (S_q tokens, typically 1 for greedy decode
    or k+1 for spec-decode verify). K/V is sharded contiguously.

    The k+1 NEW K/V positions live at the end of `new_kv_owner_rank`'s slice.
    For everyone except that rank, all local K positions are strictly < all
    Q positions → unconditional attend. For new_kv_owner_rank, the last
    `new_kv_count` K positions need a causal mask vs. their corresponding Q.

    Args
        q_global              : (B, H, S_q, D), same on every rank
        k_local, v_local      : (B, H, S_k_local, D)
        rank, world_size      : ring topology
        new_kv_owner_rank     : the rank whose slice contains the just-appended Q-aligned K/V
        new_kv_count          : 0 if K cache is "frozen" (no new appends this step) else S_q
        scale                 : 1/sqrt(D) if None
        process_group         : optional dist process group

    Returns
        out_global : (B, H, S_q, D) — replicated on every rank (up to bf16 noise)
    """
    B, H, S_q, D = q_global.shape
    if scale is None:
        scale = 1.0 / math.sqrt(D)

    if world_size == 1:
        # Q sees all K/V; the last new_kv_count positions need causal alignment
        # with Q's own positions. _attn_step's causal=True already does this
        # when S_k > S_q (mask diagonal=S_k-S_q), so just call once.
        out, _lse = _attn_step(
            q_global, k_local, v_local, scale=scale, dropout_p=dropout_p,
            causal=(new_kv_count == S_q),
        )
        return out

    assert _DIST_AVAILABLE and dist.is_initialized(), \
        "ring_attention_decode requires torch.distributed initialised when world_size > 1"

    send_to   = (rank + 1) % world_size
    recv_from = (rank - 1) % world_size

    k_cur = k_local.contiguous()
    v_cur = v_local.contiguous()
    k_buf = torch.empty_like(k_cur)
    v_buf = torch.empty_like(v_cur)

    out_acc: Optional[torch.Tensor] = None
    lse_acc: Optional[torch.Tensor] = None

    for step in range(world_size):
        sr = (rank - step) % world_size  # source rank of currently-held K/V

        # During decode, Q's absolute positions are *all* past every rank's
        # prefilled K/V. So we always attend (no skip). The only causal
        # consideration is the new-tail block when sr == new_kv_owner_rank.
        if sr == new_kv_owner_rank and new_kv_count > 0 and new_kv_count == S_q:
            # The last new_kv_count K positions are pairwise causal with Q.
            out_step, lse_step = _attn_step(
                q_global, k_cur, v_cur,
                scale=scale, dropout_p=dropout_p, causal=True,
            )
        else:
            out_step, lse_step = _attn_step(
                q_global, k_cur, v_cur,
                scale=scale, dropout_p=dropout_p, causal=False,
            )

        if out_acc is None:
            out_acc, lse_acc = out_step, lse_step
        else:
            out_acc, lse_acc = _combine(out_acc, lse_acc, out_step, lse_step)

        if step < world_size - 1:
            ops = [
                dist.P2POp(dist.isend, k_cur, send_to, group=process_group),
                dist.P2POp(dist.isend, v_cur, send_to, group=process_group),
                dist.P2POp(dist.irecv, k_buf, recv_from, group=process_group),
                dist.P2POp(dist.irecv, v_buf, recv_from, group=process_group),
            ]
            reqs = dist.batch_isend_irecv(ops)
            for r in reqs:
                r.wait()
            k_cur, k_buf = k_buf, k_cur
            v_cur, v_buf = v_buf, v_cur

    return out_acc


# ---------------------------------------------------------------------------
# Reference: full-sequence attention (single-process, used by tests)
# ---------------------------------------------------------------------------

def reference_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                        scale: Optional[float] = None, causal: bool = True
                        ) -> torch.Tensor:
    """Reference (B,H,S,D) attention via _sdpa_step. Returns (B,H,S_q,D)."""
    if scale is None:
        scale = 1.0 / math.sqrt(q.shape[-1])
    out, _ = _sdpa_step(q, k, v, scale=scale, causal=causal)
    return out

"""
Ring attention free function — layout-agnostic kernel for sequence-parallel
attention with FlashAttention-2 intra-rank kernels and online-softmax merge.

Extracted from src/models/ring_attention_flash.py (R1 of M3_RING_INTEGRATION_PLAN.md)
so that both the standalone RingAttentionFlash module and the future
LlamaAttention patch can call the same kernel.

Layout
------
Contiguous slice per rank (R0.1):
    Rank r holds K/V for prefill positions [r * S_local, (r+1) * S_local).

Decode tail (replicated across ranks, R3 design refinement 2026-05-05):
    During decode, every rank appends new K/V positions to its cache
    (since hidden_states is replicated, the new K/V is identical on every
    rank). These trailing positions form a "replicated tail" that the
    kernel attends to ONCE locally — not in the ring rotation.

Causal masking per ring step (prefill only)
-------------------------------------------
Rank r at step s reads K/V from source rank sr = (r - s) mod W:
    sr <  r : full attend (past rank)            — causal=False, no mask
    sr == r : self-step, FA-2 causal=True        — within-slice diagonal mask
    sr >  r : skip (future rank, would violate causality)

Decode path
-----------
The cache layout each rank holds is `[sharded_prefill_local | replicated_tail]`
of total length `S_local_prefill + tail_count`. The decode kernel:
  1. Splits the local K/V into prefill (first S_local_prefill positions) and
     tail (last tail_count positions).
  2. Runs ring attention against the sharded prefill — W steps, each rotating
     prefill K/V to the next rank. Returns partial (out, lse).
  3. Runs local attention against the replicated tail (no comm) with causal
     masking when S_q == tail_count (the verify case). Returns partial
     (out, lse).
  4. Combines the two via the online-softmax `_combine` identity.

This ensures the tail is counted exactly once across the whole attention
computation, while every rank produces the same final output (Q is replicated).
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
    chunk_size: Optional[int] = None,
    prefetch_depth: int = 0,
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
    pending_reqs: Optional[List] = None
    pending_assemble = None
    use_async = prefetch_depth >= 1

    for step in range(world_size):
        # Drain prior in-flight rotation (sync or async).
        if pending_reqs is not None:
            for r in pending_reqs:
                r.wait()
            if pending_assemble is not None:
                pending_assemble()
            pending_reqs = None
            pending_assemble = None
            k_cur, k_buf = k_buf, k_cur
            v_cur, v_buf = v_buf, v_cur

        # Async: schedule next rotation BEFORE compute.
        if use_async and step < world_size - 1:
            pending_reqs, pending_assemble = _issue_rotation(
                k_cur, v_cur, k_buf, v_buf,
                send_to, recv_from, chunk_size, process_group,
            )

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

        # Sync: schedule rotation AFTER compute (wait at top of next iter).
        if not use_async and step < world_size - 1:
            pending_reqs, pending_assemble = _issue_rotation(
                k_cur, v_cur, k_buf, v_buf,
                send_to, recv_from, chunk_size, process_group,
            )

    if out_acc is None:
        # Only happens if world_size == 1 (handled above) or rank 0 with all
        # future-rank skips, which can't happen for sr=rank=0 (always self).
        raise RuntimeError("ring_attention_prefill produced no output — rank/world_size mismatch")

    return out_acc


# ---------------------------------------------------------------------------
# Ring attention — decode path (Q is replicated across ranks at the "tail")
# ---------------------------------------------------------------------------

def _issue_rotation(
    send_k: torch.Tensor, send_v: torch.Tensor,
    recv_k: torch.Tensor, recv_v: torch.Tensor,
    send_to: int, recv_from: int,
    chunk_size: Optional[int],
    process_group,
):
    """Issue a single ring rotation as one or more chunked batch_isend_irecv ops.

    `chunk_size` controls A3 — transmission granularity. None or >= S_local
    means a single 4-op batch (legacy behavior). Smaller values split the
    K/V slice into N = ceil(S_local / chunk_size) chunks of contiguous
    positions; all 4*N ops are submitted in a single batch_isend_irecv so
    NCCL still sees them atomically (no eager-mode deadlock risk).

    Slicing (B, H, S, D) along dim 2 produces non-contiguous views, which
    backends like gloo's recv reject. So under chunking we recv into
    fresh per-chunk contiguous buffers and assemble them into recv_k/recv_v
    after the caller waits on the returned reqs. The caller invokes the
    returned `assemble_fn()` after wait; pass-through (no-op) when not
    chunked.

    Returns
        (reqs, assemble_fn)
            reqs: List[Work] from dist.batch_isend_irecv
            assemble_fn: callable; safe to call multiple times; no-op
                         in the unchunked case.
    """
    S = send_k.shape[2]
    if chunk_size is None or chunk_size <= 0 or chunk_size >= S:
        ops = [
            dist.P2POp(dist.isend, send_k, send_to, group=process_group),
            dist.P2POp(dist.isend, send_v, send_to, group=process_group),
            dist.P2POp(dist.irecv, recv_k, recv_from, group=process_group),
            dist.P2POp(dist.irecv, recv_v, recv_from, group=process_group),
        ]
        reqs = dist.batch_isend_irecv(ops)
        return reqs, lambda: None

    ops = []
    chunks: List[Tuple[int, int, torch.Tensor, torch.Tensor]] = []
    for start in range(0, S, chunk_size):
        end = min(start + chunk_size, S)
        sk = send_k[:, :, start:end, :].contiguous()
        sv = send_v[:, :, start:end, :].contiguous()
        rk = torch.empty_like(sk)
        rv = torch.empty_like(sv)
        ops.append(dist.P2POp(dist.isend, sk, send_to,   group=process_group))
        ops.append(dist.P2POp(dist.isend, sv, send_to,   group=process_group))
        ops.append(dist.P2POp(dist.irecv, rk, recv_from, group=process_group))
        ops.append(dist.P2POp(dist.irecv, rv, recv_from, group=process_group))
        chunks.append((start, end, rk, rv))
    reqs = dist.batch_isend_irecv(ops)

    def assemble():
        for start, end, rk, rv in chunks:
            recv_k[:, :, start:end, :].copy_(rk)
            recv_v[:, :, start:end, :].copy_(rv)

    return reqs, assemble


def _ring_against_sharded(
    q_global: torch.Tensor,       # (B, H, S_q, D) — replicated Q
    k_sharded_local: torch.Tensor,  # (B, H, S_local_prefill, D)
    v_sharded_local: torch.Tensor,
    rank: int,
    world_size: int,
    scale: float,
    dropout_p: float,
    process_group,
    chunk_size: Optional[int] = None,
    prefetch_depth: int = 0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Ring W steps over the sharded prefill K/V.

    Returns (out, lse) where lse is in fp32. All ranks' K/V slices are
    same-sized (S_local_prefill); causal masking is unconditional (no
    self-step diagonal) because Q's absolute positions are strictly past
    all prefill positions during decode.

    Ablation knobs (A3/A4 — see M3_RING_INTEGRATION_PLAN.md, R3.5):
        chunk_size      A3. Per-step batch_isend_irecv transmission chunk
                        size in tokens. None / 0 / >= S_local = single chunk
                        (no chunking).
        prefetch_depth  A4. 0 = synchronous rotation (wait between steps).
                        >= 1 = async overlap: rotation s+1 is issued before
                        compute s, waited on at top of next iteration.
                        Saturates at 1 under the standard ring P2P pattern;
                        higher values are accepted but produce identical
                        behaviour to 1.
    """
    if world_size == 1:
        return _attn_step(q_global, k_sharded_local, v_sharded_local,
                          scale=scale, dropout_p=dropout_p, causal=False)

    assert _DIST_AVAILABLE and dist.is_initialized(), \
        "_ring_against_sharded requires torch.distributed initialised when world_size > 1"

    send_to   = (rank + 1) % world_size
    recv_from = (rank - 1) % world_size

    k_cur = k_sharded_local.contiguous()
    v_cur = v_sharded_local.contiguous()
    k_buf = torch.empty_like(k_cur)
    v_buf = torch.empty_like(v_cur)

    out_acc: Optional[torch.Tensor] = None
    lse_acc: Optional[torch.Tensor] = None
    pending_reqs: Optional[List] = None
    pending_assemble = None  # callable invoked after wait if chunked recv
    use_async = prefetch_depth >= 1

    for step in range(world_size):
        # Drain any in-flight rotation from the previous iteration and swap
        # so k_cur/v_cur point to the just-received slice.
        if pending_reqs is not None:
            for r in pending_reqs:
                r.wait()
            if pending_assemble is not None:
                pending_assemble()
            pending_reqs = None
            pending_assemble = None
            k_cur, k_buf = k_buf, k_cur
            v_cur, v_buf = v_buf, v_cur

        # Async path: schedule next rotation BEFORE compute so it overlaps.
        if use_async and step < world_size - 1:
            pending_reqs, pending_assemble = _issue_rotation(
                k_cur, v_cur, k_buf, v_buf,
                send_to, recv_from, chunk_size, process_group,
            )

        out_step, lse_step = _attn_step(
            q_global, k_cur, v_cur,
            scale=scale, dropout_p=dropout_p, causal=False,
        )
        if out_acc is None:
            out_acc, lse_acc = out_step, lse_step
        else:
            out_acc, lse_acc = _combine(out_acc, lse_acc, out_step, lse_step)

        # Sync path: schedule rotation AFTER compute and stash reqs for the
        # top of the next iteration to wait on (uniform with async path).
        if not use_async and step < world_size - 1:
            pending_reqs, pending_assemble = _issue_rotation(
                k_cur, v_cur, k_buf, v_buf,
                send_to, recv_from, chunk_size, process_group,
            )

    return out_acc, lse_acc


def ring_attention_decode(
    q_global: torch.Tensor,       # (B, H, S_q, D)  — replicated Q
    k_local: torch.Tensor,        # (B, H, S_local_prefill + tail_count, D)
    v_local: torch.Tensor,        # (B, H, S_local_prefill + tail_count, D)
    rank: int,
    world_size: int,
    tail_count: int,              # number of REPLICATED tail positions at the end
    scale: Optional[float] = None,
    dropout_p: float = 0.0,
    process_group: Optional["dist.ProcessGroup"] = None,
    chunk_size: Optional[int] = None,
    prefetch_depth: int = 0,
) -> torch.Tensor:
    """Ring attention for decode/verify under the dual-cache layout.

    Each rank's local K/V is structured as:
        [sharded_prefill_local | replicated_tail]
    where the prefill part has S_local_prefill = N/W positions (rank r holds
    [r*S_local..(r+1)*S_local) of the global prefill) and the tail has
    `tail_count` positions that are IDENTICAL on every rank (the decoded
    tokens accumulated since prefill).

    The kernel:
      Pass 1 — ring W steps against the sharded prefill (K/V slices rotate
               between ranks). Each step contributes attention against
               1/W of the prefill positions; combined via online-softmax.
      Pass 2 — local attention against the replicated tail (no comm). When
               S_q == tail_count we mask causally (new K/V is the same
               positions as Q); otherwise attend unconditionally (tail
               positions are all "past" wrt Q).
      Combine — merge pass-1 and pass-2 outputs via the log-sum-exp identity.

    Args
        q_global              : (B, H, S_q, D), same on every rank
        k_local, v_local      : (B, H, S_local_prefill + tail_count, D)
        rank, world_size      : ring topology
        tail_count            : number of trailing positions in k_local that
                                are the replicated tail (NOT in the ring)
        scale, dropout_p      : standard attention args
        process_group         : optional dist process group

    Returns
        out_global : (B, H, S_q, D) — replicated on every rank (up to bf16 noise)
    """
    B, H, S_q, D = q_global.shape
    if scale is None:
        scale = 1.0 / math.sqrt(D)
    S_total_local = k_local.shape[2]
    S_prefill_local = S_total_local - tail_count
    assert S_prefill_local >= 0, \
        f"tail_count={tail_count} exceeds k_local.shape[2]={S_total_local}"

    has_prefill = S_prefill_local > 0
    has_tail    = tail_count > 0

    if not has_prefill and not has_tail:
        raise ValueError("ring_attention_decode: empty K/V")

    if not has_prefill:
        # Only tail (degenerate; no prefill positions to ring over).
        out, _ = _attn_step(
            q_global, k_local, v_local,
            scale=scale, dropout_p=dropout_p, causal=(S_q == tail_count),
        )
        return out

    # Pass 1: ring against sharded prefill
    k_prefill = k_local[:, :, :S_prefill_local, :].contiguous()
    v_prefill = v_local[:, :, :S_prefill_local, :].contiguous()
    out_prefill, lse_prefill = _ring_against_sharded(
        q_global, k_prefill, v_prefill,
        rank=rank, world_size=world_size,
        scale=scale, dropout_p=dropout_p,
        process_group=process_group,
        chunk_size=chunk_size,
        prefetch_depth=prefetch_depth,
    )

    if not has_tail:
        return out_prefill

    # Pass 2: local against replicated tail (causal when S_q matches tail length)
    k_tail = k_local[:, :, S_prefill_local:, :]
    v_tail = v_local[:, :, S_prefill_local:, :]
    out_tail, lse_tail = _attn_step(
        q_global, k_tail, v_tail,
        scale=scale, dropout_p=dropout_p, causal=(S_q == tail_count),
    )

    # Combine the two passes via online-softmax identity.
    out_combined, _ = _combine(out_prefill, lse_prefill, out_tail, lse_tail)
    return out_combined


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

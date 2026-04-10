"""
Blockwise Ring Attention with FlashAttention-2 intra-rank kernels.

Problem solved
--------------
The naive ring attention in src/baselines/ring_attention.py computes a full
(S_local × S_local) attention score matrix at each ring step. This is O(S_local²)
in memory. At 512k total tokens across 8 GPUs, S_local = 64k — the score matrix
alone is ~32 GB in fp32 per head, causing the OOMs seen at 256k/512k context.

Fix
---
Replace the inner einsum + manual online-softmax with a FlashAttention-2 call.
FA tiles the Q×K^T computation internally, never materialising the full score
matrix. Memory cost per ring step drops from O(S_local²) to O(S_local).

The ring communication layer (batch_isend_irecv, buffer swap) is unchanged.
Cross-step combination still uses the log-sum-exp identity — FA returns the
per-step log-sum-exp (lse) needed for numerically stable merging.

Architecture
------------
  Ring Attention  — shards the full sequence across GPUs        (supply chain)
  FlashAttention  — tiles each GPU's local QK^T computation     (factory floor)

Requirements
------------
  flash-attn >= 2.0   pip install flash-attn --no-build-isolation
  torch >= 2.1        (fallback uses F.scaled_dot_product_attention)
  CUDA, fp16 or bf16 inputs
"""

import math
import warnings

import torch
import torch.nn as nn
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
    warnings.warn(
        "flash_attn not found. RingAttentionFlash will fall back to the naive "
        "O(S²) kernel, which OOMs at context lengths > 128k on 80 GB A100s. "
        "Install: pip install flash-attn --no-build-isolation",
        stacklevel=1,
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _flash_step(q_fa, k_fa, v_fa, scale, dropout_p, causal):
    """Run one FlashAttention-2 step and return (out, lse).

    q_fa / k_fa / v_fa : (B, S, H, D)  — flash_attn layout
    Returns
        out : (B, H, S, D)              — standard (B,H,S,D) layout
        lse : (B, H, S)  fp32           — log-sum-exp for cross-step merging
    """
    result = _flash_attn_func(
        q_fa, k_fa, v_fa,
        dropout_p=dropout_p,
        softmax_scale=scale,
        causal=causal,
        return_attn_probs=True,   # gives (out, softmax_lse, S_dmask)
    )
    out_fa, lse, _ = result      # lse: (B, H, S)
    out = out_fa.permute(0, 2, 1, 3)  # -> (B, H, S, D)
    return out, lse


def _sdpa_step(q, k, v, scale, dropout_p, causal):
    """PyTorch SDPA fallback (uses FA-2 on CUDA fp16/bf16 when available).

    Does NOT return lse — cannot be used for cross-step combination in the
    ring path. Used only in the single-GPU local forward.

    q / k / v : (B, H, S, D)
    Returns out : (B, H, S, D)
    """
    return F.scaled_dot_product_attention(
        q, k, v,
        dropout_p=dropout_p,
        is_causal=causal,
        scale=scale,
    )


def _combine(out_acc, lse_acc, out_step, lse_step):
    """Merge one ring step's output into the running accumulator.

    Uses the log-sum-exp identity so combining N partial softmaxes is
    equivalent to a single softmax over the full concatenated KV sequence.

      lse_new = logaddexp(lse_acc, lse_step)
      out_new = exp(lse_acc  - lse_new) * out_acc
              + exp(lse_step - lse_new) * out_step

    All lse tensors are kept in fp32 for numerical stability; out tensors
    stay in the model's native dtype (fp16 / bf16).

    Args
        out_acc   : (B, H, S, D)  accumulated output (model dtype)
        lse_acc   : (B, H, S)     accumulated lse    (fp32)
        out_step  : (B, H, S, D)  new step output    (model dtype)
        lse_step  : (B, H, S)     new step lse       (fp32)
    Returns
        updated (out_acc, lse_acc)
    """
    lse_new  = torch.logaddexp(lse_acc, lse_step)                   # (B,H,S)
    w_acc    = torch.exp(lse_acc  - lse_new).unsqueeze(-1)          # (B,H,S,1)
    w_step   = torch.exp(lse_step - lse_new).unsqueeze(-1)          # (B,H,S,1)
    out_new  = w_acc * out_acc + w_step * out_step.to(out_acc.dtype)
    return out_new, lse_new


def _naive_step(q, k_cur, v_cur, out_acc, lse_acc):
    """Fallback online-softmax accumulator (O(S²) memory) — mirrors baseline."""
    scale   = 1.0 / math.sqrt(q.shape[-1])
    scores  = torch.einsum("bhid,bhjd->bhij", q, k_cur) * scale
    blk_max = scores.amax(dim=-1)                                    # (B,H,S)
    exp_s   = torch.exp(scores - blk_max.unsqueeze(-1))
    step_lse = blk_max + torch.log(exp_s.sum(dim=-1))               # (B,H,S)
    new_lse  = torch.logaddexp(lse_acc, step_lse)
    w_old    = torch.exp(lse_acc  - new_lse).unsqueeze(-1)
    w_new    = torch.exp(blk_max  - new_lse).unsqueeze(-1)
    out_acc  = w_old * out_acc + w_new * (exp_s @ v_cur)
    return out_acc, new_lse


# ---------------------------------------------------------------------------
# Module
# ---------------------------------------------------------------------------

class RingAttentionFlash(nn.Module):
    """Ring Attention with FlashAttention-2 intra-rank kernels.

    Drop-in replacement for src/baselines/ring_attention.RingAttention.
    Keeps identical QKV projection weights and ring communication protocol
    (batch_isend_irecv). Replaces the O(S²) local computation with FA-2.

    Args
        dim       : model hidden dimension
        num_heads : number of attention heads
        causal    : apply causal (autoregressive) masking in the local step.
                    Note: full causal ring attention requires per-step masking
                    based on rank offset — set causal=False for the ring path
                    and handle masking at the model level when needed.
        dropout   : attention dropout probability (applied in FA kernel)
    """

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        causal: bool = False,
        dropout: float = 0.0,
    ):
        super().__init__()
        assert dim % num_heads == 0, "dim must be divisible by num_heads"
        self.dim       = dim
        self.num_heads = num_heads
        self.head_dim  = dim // num_heads
        self.causal    = causal
        self.dropout_p = dropout
        self.scale     = 1.0 / math.sqrt(self.head_dim)

        self.qkv      = nn.Linear(dim, dim * 3, bias=False)
        self.out_proj = nn.Linear(dim, dim, bias=False)

    # ------------------------------------------------------------------
    # Distributed check
    # ------------------------------------------------------------------

    def _is_distributed(self) -> bool:
        return (
            _DIST_AVAILABLE
            and dist.is_available()
            and dist.is_initialized()
            and dist.get_world_size() > 1
        )

    # ------------------------------------------------------------------
    # Single-GPU forward
    # ------------------------------------------------------------------

    def _local_forward(self, x: torch.Tensor) -> torch.Tensor:
        B, S, D = x.shape
        qkv = (
            self.qkv(x)
            .reshape(B, S, 3, self.num_heads, self.head_dim)
            .permute(2, 0, 3, 1, 4)                        # (3, B, H, S, D)
        )
        q, k, v = qkv[0], qkv[1], qkv[2]                 # each (B, H, S, D)

        if _FLASH_AVAILABLE:
            # flash_attn expects (B, S, H, D)
            q_fa = q.permute(0, 2, 1, 3).contiguous()
            k_fa = k.permute(0, 2, 1, 3).contiguous()
            v_fa = v.permute(0, 2, 1, 3).contiguous()
            out_fa = _flash_attn_func(
                q_fa, k_fa, v_fa,
                dropout_p=self.dropout_p if self.training else 0.0,
                softmax_scale=self.scale,
                causal=self.causal,
            )
            out = out_fa.permute(0, 2, 1, 3)              # (B, H, S, D)
        else:
            # Fallback: PyTorch SDPA (uses FA-2 on CUDA fp16/bf16 when available)
            out = _sdpa_step(
                q, k, v,
                scale=self.scale,
                dropout_p=self.dropout_p if self.training else 0.0,
                causal=self.causal,
            )

        out = out.permute(0, 2, 1, 3).reshape(B, S, D)
        return self.out_proj(out)

    # ------------------------------------------------------------------
    # Multi-GPU ring forward
    # ------------------------------------------------------------------

    def _ring_forward(self, x: torch.Tensor) -> torch.Tensor:
        """Distributed ring attention with FA-2 intra-rank kernels.

        Communication protocol (batch_isend_irecv) is unchanged from baseline.
        Only the inner attention computation is replaced with FA.
        """
        rank       = dist.get_rank()
        world_size = dist.get_world_size()
        send_to    = (rank + 1) % world_size
        recv_from  = (rank - 1) % world_size

        B, S_local, D = x.shape

        qkv = (
            self.qkv(x)
            .reshape(B, S_local, 3, self.num_heads, self.head_dim)
            .permute(2, 0, 3, 1, 4)
        )
        q, k, v = qkv[0], qkv[1], qkv[2]  # each (B, H, S_local, head_dim)

        # Pre-permute Q once — stays fixed across all ring steps
        q_fa = q.permute(0, 2, 1, 3).contiguous() if _FLASH_AVAILABLE else None

        k_cur = k.contiguous()
        v_cur = v.contiguous()
        k_buf = torch.empty_like(k_cur)
        v_buf = torch.empty_like(v_cur)

        out_acc = torch.zeros_like(q)                           # (B, H, S_local, D)
        # Keep lse in fp32 regardless of model dtype for numerical stability
        lse_acc = torch.full(
            (B, self.num_heads, S_local), float("-inf"),
            device=x.device, dtype=torch.float32,
        )

        for step in range(world_size):
            if _FLASH_AVAILABLE:
                k_fa = k_cur.permute(0, 2, 1, 3).contiguous()
                v_fa = v_cur.permute(0, 2, 1, 3).contiguous()
                out_step, lse_step = _flash_step(
                    q_fa, k_fa, v_fa,
                    scale=self.scale,
                    dropout_p=self.dropout_p if self.training else 0.0,
                    causal=False,   # full causal ring needs per-rank masking; TODO
                )
                out_acc, lse_acc = _combine(out_acc, lse_acc, out_step, lse_step)
            else:
                out_acc, lse_acc = _naive_step(q, k_cur, v_cur, out_acc, lse_acc)

            # Rotate KV ring — batch_isend_irecv posts all send/recv atomically,
            # preventing the NCCL eager-mode serialisation deadlock.
            if step < world_size - 1:
                ops = [
                    dist.P2POp(dist.isend, k_cur, send_to),
                    dist.P2POp(dist.isend, v_cur, send_to),
                    dist.P2POp(dist.irecv, k_buf, recv_from),
                    dist.P2POp(dist.irecv, v_buf, recv_from),
                ]
                reqs = dist.batch_isend_irecv(ops)
                for r in reqs:
                    r.wait()
                k_cur, k_buf = k_buf, k_cur
                v_cur, v_buf = v_buf, v_cur

        out = out_acc.permute(0, 2, 1, 3).reshape(B, S_local, D)
        return self.out_proj(out)

    # ------------------------------------------------------------------
    # Entry point
    # ------------------------------------------------------------------

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args
            x : (B, S, D) — full sequence (single-GPU) or local shard (multi-GPU)
        Returns
            (B, S, D) attention output
        """
        if self._is_distributed():
            return self._ring_forward(x)
        return self._local_forward(x)

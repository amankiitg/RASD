"""
Ring attention patch for Llama target model (R2 of M3_RING_INTEGRATION_PLAN.md).

Replaces each `LlamaAttention.forward` with a ring-aware version that:
  - Splits K/V across `world_size` ranks (contiguous slices, R0.1)
  - Uses ring_attention_prefill / ring_attention_decode from
    src/models/ring_attention_kernel.py for the actual attention math
  - Falls through to the original forward when world_size == 1

Targets transformers 4.44.x's LlamaAttention forward signature:
    forward(hidden_states, attention_mask=None, position_ids=None,
            past_key_value=None, output_attentions=False, use_cache=False,
            cache_position=None, position_embeddings=None, **kwargs)

Calling convention (driver responsibilities)
--------------------------------------------
  Prefill: hidden_states is each rank's local slice (S_local = N/W tokens).
           position_ids must be the absolute positions
           [rank*S_local, (rank+1)*S_local).
           After prefill, the driver MUST set `_ring_prefill_len` on every
           attention module to S_local — this freezes the prefill boundary
           for subsequent decode calls.
  Decode:  hidden_states is the SAME on every rank (Q replicated). Every
           rank appends the new K/V to its local cache (the tail is
           "replicated" identical across ranks since hidden_states is
           identical). The kernel splits prefill vs tail via `_ring_prefill_len`.

Dual-cache layout (R3 design refinement 2026-05-05):
  Each layer's HF past_key_value contains, in order:
      [sharded_prefill_local | replicated_tail]
  The prefill part has _ring_prefill_len positions (rank's contiguous slice).
  The tail part grows monotonically during decode (subject to truncation on
  partial rejection) and is identical on every rank.

Tested locally for the world_size=1 fall-through path. Multi-rank validation
runs on pod (R6) since transformers==4.44.2 is not installed locally.
"""

from __future__ import annotations

import logging
import math
from types import MethodType
from typing import Optional, Tuple

import torch

from src.models.ring_attention_kernel import (
    ring_attention_decode,
    ring_attention_prefill,
)

logger = logging.getLogger(__name__)


def _ring_llama_attention_forward(
    self,
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_value=None,
    output_attentions: bool = False,
    use_cache: bool = False,
    cache_position: Optional[torch.LongTensor] = None,
    position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    **kwargs,
):
    """Ring-aware LlamaAttention.forward. Bound to instances via install_ring_attention."""

    # world_size==1 → fall through to the original forward (no ring overhead)
    if getattr(self, "_ring_world_size", 1) <= 1:
        return self._ring_original_forward(
            hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
            **kwargs,
        )

    # Lazy import: only required when running multi-rank
    from transformers.models.llama.modeling_llama import (
        apply_rotary_pos_emb,
        repeat_kv,
    )

    rank = self._ring_rank
    world_size = self._ring_world_size
    chunk_size     = getattr(self, "_ring_chunk_size", None)
    prefetch_depth = getattr(self, "_ring_prefetch_depth", 0)

    bsz, q_len, _ = hidden_states.size()
    head_dim = self.head_dim
    num_heads = self.num_heads
    num_kv_heads = getattr(self, "num_key_value_heads", num_heads)
    num_kv_groups = getattr(self, "num_key_value_groups", 1)

    # 1. QKV projection
    q = self.q_proj(hidden_states).view(bsz, q_len, num_heads,    head_dim).transpose(1, 2)
    k = self.k_proj(hidden_states).view(bsz, q_len, num_kv_heads, head_dim).transpose(1, 2)
    v = self.v_proj(hidden_states).view(bsz, q_len, num_kv_heads, head_dim).transpose(1, 2)

    # 2. RoPE
    if position_embeddings is not None:
        cos, sin = position_embeddings
    else:
        cos, sin = self.rotary_emb(v, position_ids)
    q, k = apply_rotary_pos_emb(q, k, cos, sin)

    # 3. Cache update — every rank appends new K/V (dual-cache layout).
    #    During prefill: hidden_states is each rank's local slice → each
    #    rank's appended K/V is its own contiguous prefill slice.
    #    During decode:  hidden_states is replicated across ranks → each
    #    rank appends the SAME new K/V positions as the global "tail".
    is_prefill = (past_key_value is None) or _cache_is_empty(past_key_value, self.layer_idx)
    k_full, v_full = _cache_append(
        past_key_value, k, v, self.layer_idx, sin, cos, cache_position,
    )

    # 3a. M4 C11 — NF4 KV-cache lossy bf16 path.
    # When the installer set _ring_kv_quant=True, the K/V returned from
    # the cache is round-tripped through NF4 quantize→dequantize before
    # being passed to the ring kernel. This exercises the codec on real
    # attention activations and lets us measure α/PPL impact end-to-end.
    # The actual cache *storage* still holds bf16 — true memory savings
    # require subclassing DynamicCache or an external NF4 store, which
    # is M4 Phase C work. Default is False; M3 byte-identical when off.
    if getattr(self, "_ring_kv_quant", False):
        k_full, v_full = _kv_quant_round_trip(k_full, v_full)

    # 4. GQA expansion (no-op for Llama-2 where num_kv_heads == num_heads)
    k_full = repeat_kv(k_full, num_kv_groups)
    v_full = repeat_kv(v_full, num_kv_groups)

    # 5. Ring attention
    scale = 1.0 / math.sqrt(head_dim)
    if is_prefill:
        # Q is sharded (each rank holds local slice); K/V is sharded.
        attn_out = ring_attention_prefill(
            q, k_full, v_full,
            rank=rank, world_size=world_size, scale=scale,
            chunk_size=chunk_size, prefetch_depth=prefetch_depth,
        )
    else:
        # Q is replicated; K/V is [sharded_prefill | replicated_tail].
        # The driver sets `_ring_prefill_len` after prefill so we can split.
        prefill_len = getattr(self, "_ring_prefill_len", k_full.shape[2] - q_len)
        tail_count = k_full.shape[2] - prefill_len
        attn_out = ring_attention_decode(
            q, k_full, v_full,
            rank=rank, world_size=world_size,
            tail_count=tail_count, scale=scale,
            chunk_size=chunk_size, prefetch_depth=prefetch_depth,
        )

    # 6. Output projection
    attn_out = attn_out.transpose(1, 2).contiguous().view(bsz, q_len, -1)
    attn_out = self.o_proj(attn_out)

    # Attention weights are not returned (output_attentions ignored)
    return attn_out, None, past_key_value


# ---------------------------------------------------------------------------
# Cache helpers — handle both DynamicCache (4.44+) and legacy tuple
# ---------------------------------------------------------------------------

def _cache_is_empty(past_key_value, layer_idx: int) -> bool:
    """True if there are no past keys for this layer."""
    if past_key_value is None:
        return True
    if hasattr(past_key_value, "key_cache"):
        # DynamicCache
        if layer_idx >= len(past_key_value.key_cache):
            return True
        kc = past_key_value.key_cache[layer_idx]
        return kc is None or kc.numel() == 0
    # Legacy tuple of (k, v) per layer
    if isinstance(past_key_value, (tuple, list)):
        if layer_idx >= len(past_key_value):
            return True
        kv = past_key_value[layer_idx]
        if kv is None or len(kv) == 0:
            return True
        return kv[0] is None or kv[0].numel() == 0
    return False


def _kv_quant_round_trip(
    k_full: torch.Tensor, v_full: torch.Tensor, block_size: int = 64,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """NF4 quantize→dequantize round-trip on the full K and V tensors.

    Used by _ring_llama_attention_forward when `attn._ring_kv_quant` is
    True. Pure transformation — input shape and dtype preserved, values
    perturbed by NF4 reconstruction error (~3-5% relative L2 on real
    KV activations per KIVI/KVQuant).

    Defensive: if the head_dim isn't divisible by `block_size`, skip the
    round-trip (return inputs unchanged) so a misconfigured model doesn't
    crash the verify loop. The pod-side validation gate (C11) will
    surface this case early if it ever fires for the M4 target models.
    """
    from src.models.nf4_kv import dequantize_nf4, quantize_nf4

    head_dim = k_full.shape[-1]
    if head_dim % block_size != 0:
        return k_full, v_full
    dtype = k_full.dtype
    k_codes, k_scales = quantize_nf4(k_full.contiguous(), block_size=block_size)
    v_codes, v_scales = quantize_nf4(v_full.contiguous(), block_size=block_size)
    k_dq = dequantize_nf4(k_codes, k_scales, block_size=block_size, dtype=dtype)
    v_dq = dequantize_nf4(v_codes, v_scales, block_size=block_size, dtype=dtype)
    return k_dq, v_dq


def _cache_append(past_key_value, k_new: torch.Tensor, v_new: torch.Tensor,
                  layer_idx: int, sin, cos, cache_position
                  ) -> Tuple[torch.Tensor, torch.Tensor]:
    """Append k_new, v_new to past_key_value at layer_idx; return full (k, v)."""
    if past_key_value is None:
        return k_new, v_new
    if hasattr(past_key_value, "update"):
        cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
        return past_key_value.update(k_new, v_new, layer_idx, cache_kwargs)
    # Legacy tuple
    past_k, past_v = past_key_value[layer_idx]
    return torch.cat([past_k, k_new], dim=2), torch.cat([past_v, v_new], dim=2)


def _cache_read(past_key_value, layer_idx: int) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
    """Read existing (k, v) for layer_idx without mutating the cache."""
    if past_key_value is None:
        return None, None
    if hasattr(past_key_value, "key_cache"):
        if layer_idx >= len(past_key_value.key_cache):
            return None, None
        return past_key_value.key_cache[layer_idx], past_key_value.value_cache[layer_idx]
    if isinstance(past_key_value, (tuple, list)) and layer_idx < len(past_key_value):
        kv = past_key_value[layer_idx]
        if kv is None or len(kv) < 2:
            return None, None
        return kv[0], kv[1]
    return None, None


# ---------------------------------------------------------------------------
# Public installer
# ---------------------------------------------------------------------------

def install_ring_attention(
    target_model,
    world_size: int,
    rank: int,
    chunk_size: Optional[int] = None,
    prefetch_depth: int = 0,
    kv_quant: bool = False,
) -> int:
    """Replace each LlamaAttention.forward with the ring-aware version.

    No-op when world_size <= 1. Returns the number of attention layers patched.

    Args:
        chunk_size      A3. Per-step batch_isend_irecv chunk size in tokens.
                        None / 0 / >= S_local = single chunk (no chunking).
        prefetch_depth  A4. 0 = synchronous; >= 1 = async overlap (one
                        rotation in flight while compute proceeds).
        kv_quant        M4 C11. When True, every K/V tensor returned from
                        the cache is round-tripped through NF4
                        quantize→dequantize before being passed to the
                        ring kernel. This "lossy bf16 path" exercises
                        the codec on real activations; actual cache-
                        storage savings are deferred to Phase C.

    The original forward is stored as `_ring_original_forward` on each
    instance so multi-rank=1 falls through unchanged. Six attributes are
    added to each attention module: `_ring_rank`, `_ring_world_size`,
    `_ring_prefill_len`, `_ring_chunk_size`, `_ring_prefetch_depth`,
    `_ring_kv_quant`.
    """
    if world_size is None or world_size <= 1:
        return 0

    layers = getattr(getattr(target_model, "model", target_model), "layers", None)
    if layers is None:
        raise RuntimeError(
            "install_ring_attention: could not find target_model.model.layers — "
            "is this a Llama-style model?"
        )

    n_patched = 0
    for layer in layers:
        attn = getattr(layer, "self_attn", None)
        if attn is None:
            continue
        attn._ring_rank = rank
        attn._ring_world_size = world_size
        # Initial prefill_len = 0; the driver sets the real value after prefill
        # completes (RASDInference._prefill is responsible).
        attn._ring_prefill_len = 0
        # A3 / A4 ablation knobs (see kernel docstring).
        attn._ring_chunk_size = chunk_size
        attn._ring_prefetch_depth = prefetch_depth
        attn._ring_kv_quant = bool(kv_quant)
        attn._ring_original_forward = attn.forward
        attn.forward = MethodType(_ring_llama_attention_forward, attn)
        n_patched += 1

    logger.info(
        "[ring] patched %d LlamaAttention layers (rank=%d, world_size=%d, "
        "chunk_size=%s, prefetch_depth=%d, kv_quant=%s)",
        n_patched, rank, world_size, chunk_size, prefetch_depth, kv_quant,
    )
    return n_patched


def set_prefill_len(target_model, prefill_len: int) -> int:
    """Freeze the prefill boundary on every patched attention module.

    Called by RASDInference after prefill completes. Each subsequent decode
    forward will treat positions [0, prefill_len) of its local cache as the
    sharded prefill (rotated through the ring) and positions
    [prefill_len, end) as the replicated tail (attended locally).

    Returns the number of attention modules updated. No-op (returns 0) if
    install_ring_attention was never called or world_size <= 1.
    """
    layers = getattr(getattr(target_model, "model", target_model), "layers", None)
    if layers is None:
        return 0
    n_set = 0
    for layer in layers:
        attn = getattr(layer, "self_attn", None)
        if attn is None or not hasattr(attn, "_ring_world_size"):
            continue
        attn._ring_prefill_len = prefill_len
        n_set += 1
    return n_set

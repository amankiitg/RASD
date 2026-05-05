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
  Decode:  hidden_states is the SAME on every rank (Q replicated). Only the
           rank world_size-1 actually appends the new K/V to its local cache;
           other ranks leave their cache untouched.

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

    # 3. Cache update — only the owning rank appends new K/V
    #    During prefill, hidden_states is the local slice → each rank appends
    #    its own slice. During decode, hidden_states is replicated → only
    #    rank world_size-1 appends; others leave their cache untouched.
    is_prefill = (past_key_value is None) or _cache_is_empty(past_key_value, self.layer_idx)
    new_kv_owner_rank = world_size - 1  # contiguous layout: tail goes to last rank

    if is_prefill:
        # Each rank appends its local slice
        k_full, v_full = _cache_append(past_key_value, k, v, self.layer_idx, sin, cos, cache_position)
    else:
        if rank == new_kv_owner_rank:
            k_full, v_full = _cache_append(past_key_value, k, v, self.layer_idx, sin, cos, cache_position)
        else:
            # Non-owner ranks: read existing cache, don't mutate
            k_full, v_full = _cache_read(past_key_value, self.layer_idx)
            if k_full is None:
                # No cache yet — shouldn't happen during decode, but guard anyway
                k_full, v_full = k, v

    # 4. GQA expansion (no-op for Llama-2 where num_kv_heads == num_heads)
    k_full = repeat_kv(k_full, num_kv_groups)
    v_full = repeat_kv(v_full, num_kv_groups)

    # 5. Ring attention
    scale = 1.0 / math.sqrt(head_dim)
    if is_prefill:
        attn_out = ring_attention_prefill(
            q, k_full, v_full,
            rank=rank, world_size=world_size, scale=scale,
        )
    else:
        new_kv_count = q_len  # number of new K/V positions appended this call
        attn_out = ring_attention_decode(
            q, k_full, v_full,
            rank=rank, world_size=world_size,
            new_kv_owner_rank=new_kv_owner_rank,
            new_kv_count=new_kv_count, scale=scale,
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

def install_ring_attention(target_model, world_size: int, rank: int) -> int:
    """Replace each LlamaAttention.forward with the ring-aware version.

    No-op when world_size <= 1. Returns the number of attention layers patched.

    The original forward is stored as `_ring_original_forward` on each
    instance so multi-rank=1 falls through unchanged. Two new attributes are
    added to each attention module: `_ring_rank`, `_ring_world_size`.
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
        attn._ring_original_forward = attn.forward
        attn.forward = MethodType(_ring_llama_attention_forward, attn)
        n_patched += 1

    logger.info("[ring] patched %d LlamaAttention layers (rank=%d, world_size=%d)",
                n_patched, rank, world_size)
    return n_patched

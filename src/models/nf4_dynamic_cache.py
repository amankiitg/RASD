"""NF4-storage K/V cache, duck-typed as HuggingFace `DynamicCache` (M4 C11).

Replaces the bf16 cache backing with NF4-packed bytes + per-block fp32
scales. Reads (via `update()` and `key_cache`/`value_cache` properties)
dequantize on demand; the rest of the model never sees the NF4 form.

Per-rank memory savings vs bf16 storage:
  bf16: 2 bytes/element
  NF4:  0.5 + 4/block_size bytes/element ≈ 0.5625 at block_size=64
  ratio: ~3.55x compression

For the M4 1M ring KV budget (per-rank K/V at ctx=1M, W=8 = ~68 GB
in bf16), NF4 brings this to ~17 GB — fits 40 GB SXM2 with the rest
of the per-rank memory budget.

Why this is "true" NF4 storage (vs the earlier round-trip path):
- Earlier `_kv_quant_round_trip` quantize→dequantized the FULL cache
  every forward step → cache stayed bf16 → no actual memory savings
  AND repeated quantization accumulated error on existing entries.
- This cache stores NF4 bytes once on append. Every subsequent read
  dequantizes the same bytes → no error accumulation.

Compatibility:
- HF transformers 4.44.2 (pod env) and 5.5.4 (local dev env) both
  call `update(key_states, value_states, layer_idx, cache_kwargs)`
  and read via `key_cache` / `value_cache` lists. We duck-type on
  both surfaces.
- `_cache_is_empty` and `_cache_read` in src/models/ring_llama_attention.py
  go through `key_cache[layer_idx]` — handled by the lazy view.
- `_truncate_kv` in src/models/rasd_inference.py iterates the cache
  via `for layer in past_kv: k, v = layer[0], layer[1]` — handled by
  `__iter__`. To preserve NF4 storage across rounds, `_truncate_kv`
  delegates to our `truncate()` method for in-place truncation
  (vs producing a new bf16 legacy tuple).
"""
from __future__ import annotations

from typing import List, Optional, Tuple

import torch

from src.models.nf4_kv import dequantize_nf4, quantize_nf4


class NF4DynamicCache:
    """A DynamicCache-compatible cache that stores K/V as packed NF4.

    Internal layout:
      _k_codes[layer_idx]  : list[uint8 tensor]   — packed NF4 codes per append chunk
      _k_scales[layer_idx] : list[fp32 tensor]    — per-block scales per append chunk
      _v_codes[layer_idx]  : list[uint8 tensor]
      _v_scales[layer_idx] : list[fp32 tensor]

    Each append corresponds to one `update()` call (one new K/V chunk
    of shape (B, H, S_new, D)). Reads dequantize-and-concatenate all
    chunks for a layer.
    """

    def __init__(
        self,
        block_size: int = 64,
        dtype: torch.dtype = torch.bfloat16,
    ):
        self.block_size = block_size
        self.dtype = dtype
        self._k_codes: List[List[torch.Tensor]] = []
        self._k_scales: List[List[torch.Tensor]] = []
        self._v_codes: List[List[torch.Tensor]] = []
        self._v_scales: List[List[torch.Tensor]] = []

    # ------------------------------------------------------------------
    # Lifecycle helpers
    # ------------------------------------------------------------------

    def _ensure_layer(self, layer_idx: int) -> None:
        """Grow the per-layer storage so layer_idx is valid."""
        while len(self._k_codes) <= layer_idx:
            self._k_codes.append([])
            self._k_scales.append([])
            self._v_codes.append([])
            self._v_scales.append([])

    # ------------------------------------------------------------------
    # DynamicCache compat surface — `update()`
    # ------------------------------------------------------------------

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        cache_kwargs: Optional[dict] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Quantize key_states / value_states, append to layer_idx, and
        return the dequantized FULL (k, v) for the kernel.

        cache_kwargs is accepted for HF compat but ignored — by the
        time we get here, RoPE has already been applied to k_states
        in the patched LlamaAttention forward.
        """
        if key_states.shape != value_states.shape:
            raise ValueError(
                f"key_states.shape={tuple(key_states.shape)} != "
                f"value_states.shape={tuple(value_states.shape)}"
            )
        if key_states.shape[-1] % self.block_size != 0:
            raise ValueError(
                f"head_dim={key_states.shape[-1]} not divisible by "
                f"block_size={self.block_size}"
            )

        self._ensure_layer(layer_idx)

        # Quantize the new K/V chunk
        kc, ks = quantize_nf4(key_states.contiguous(),   block_size=self.block_size)
        vc, vs = quantize_nf4(value_states.contiguous(), block_size=self.block_size)

        self._k_codes[layer_idx].append(kc)
        self._k_scales[layer_idx].append(ks)
        self._v_codes[layer_idx].append(vc)
        self._v_scales[layer_idx].append(vs)

        return self._dequantize_layer(layer_idx)

    # ------------------------------------------------------------------
    # Internal: dequantize a layer's accumulated chunks
    # ------------------------------------------------------------------

    def _dequantize_layer(
        self, layer_idx: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Concatenate all stored chunks for `layer_idx` and dequantize
        to the cache's `dtype`. Returns (k, v) of shape (B, H, S, D)."""
        if layer_idx >= len(self._k_codes) or not self._k_codes[layer_idx]:
            empty = torch.empty(0, dtype=self.dtype)
            return empty, empty

        k_chunks = [
            dequantize_nf4(c, s, block_size=self.block_size, dtype=self.dtype)
            for c, s in zip(self._k_codes[layer_idx], self._k_scales[layer_idx])
        ]
        v_chunks = [
            dequantize_nf4(c, s, block_size=self.block_size, dtype=self.dtype)
            for c, s in zip(self._v_codes[layer_idx], self._v_scales[layer_idx])
        ]
        # Position dim is axis 2 in (B, H, S, D)
        k_full = torch.cat(k_chunks, dim=2)
        v_full = torch.cat(v_chunks, dim=2)
        return k_full, v_full

    # ------------------------------------------------------------------
    # DynamicCache compat surface — read accessors
    # ------------------------------------------------------------------

    @property
    def key_cache(self) -> "_LazyDequantList":
        return _LazyDequantList(self, is_value=False)

    @property
    def value_cache(self) -> "_LazyDequantList":
        return _LazyDequantList(self, is_value=True)

    def get_seq_length(self, layer_idx: int = 0) -> int:
        """Total stored positions for `layer_idx`. Sums across chunks."""
        if layer_idx >= len(self._k_codes):
            return 0
        # codes shape is (B, H, S_chunk, D//2); position dim is axis 2
        return sum(c.shape[2] for c in self._k_codes[layer_idx])

    def get_max_length(self) -> Optional[int]:
        """HF compat: unbounded for DynamicCache-style caches."""
        return None

    def get_usable_length(
        self, new_seq_length: int, layer_idx: int = 0
    ) -> int:
        """HF compat: how much existing context is usable for a new
        forward of `new_seq_length` tokens."""
        # DynamicCache has no max length, so the entire stored context is usable
        return self.get_seq_length(layer_idx)

    # ------------------------------------------------------------------
    # Iteration (mirrors HF DynamicCache + legacy tuple)
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self._k_codes)

    def __iter__(self):
        for i in range(len(self)):
            yield self._dequantize_layer(i)

    def __getitem__(self, layer_idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self._dequantize_layer(layer_idx)

    # ------------------------------------------------------------------
    # M4 C6 — in-place truncation (preserves NF4 storage between rounds)
    # ------------------------------------------------------------------

    def truncate(self, new_seqlen: int) -> "NF4DynamicCache":
        """Trim every layer to keep only the first `new_seqlen` positions.

        Mirrors `_truncate_kv` for the speculative-decoding partial-
        rejection rollback. Mutates in place; returns self for chaining.

        Without this, `_truncate_kv`'s default behavior (build a new
        legacy bf16 tuple) would defeat NF4 storage between rounds —
        the next forward would see a bf16 tuple and stay bf16 forever.
        """
        if new_seqlen < 0:
            raise ValueError(f"new_seqlen={new_seqlen} must be >= 0")

        for layer_idx in range(len(self._k_codes)):
            self._truncate_layer(layer_idx, new_seqlen)
        return self

    def _truncate_layer(self, layer_idx: int, new_seqlen: int) -> None:
        """Per-layer truncation: drop full chunks past `new_seqlen`,
        partially trim the boundary chunk."""
        if layer_idx >= len(self._k_codes):
            return
        kc_list = self._k_codes[layer_idx]
        ks_list = self._k_scales[layer_idx]
        vc_list = self._v_codes[layer_idx]
        vs_list = self._v_scales[layer_idx]

        new_kc, new_ks = [], []
        new_vc, new_vs = [], []
        cumulative = 0
        for kc, ks_, vc, vs_ in zip(kc_list, ks_list, vc_list, vs_list):
            chunk_len = kc.shape[2]
            if cumulative + chunk_len <= new_seqlen:
                new_kc.append(kc); new_ks.append(ks_)
                new_vc.append(vc); new_vs.append(vs_)
                cumulative += chunk_len
            else:
                keep = new_seqlen - cumulative
                if keep > 0:
                    new_kc.append(kc[:, :, :keep, :].contiguous())
                    new_ks.append(ks_[:, :, :keep, :].contiguous())
                    new_vc.append(vc[:, :, :keep, :].contiguous())
                    new_vs.append(vs_[:, :, :keep, :].contiguous())
                break

        self._k_codes[layer_idx] = new_kc
        self._k_scales[layer_idx] = new_ks
        self._v_codes[layer_idx] = new_vc
        self._v_scales[layer_idx] = new_vs

    # ------------------------------------------------------------------
    # Memory accounting (informational; used by tests)
    # ------------------------------------------------------------------

    def memory_bytes(self) -> int:
        """Total bytes stored on the device for K + V codes + scales,
        across all layers and chunks."""
        total = 0
        for kc_list, ks_list, vc_list, vs_list in zip(
            self._k_codes, self._k_scales, self._v_codes, self._v_scales
        ):
            for kc in kc_list:
                total += kc.numel()                         # uint8 = 1 byte/elem
            for ks_ in ks_list:
                total += ks_.numel() * 4                    # fp32 = 4 bytes/elem
            for vc in vc_list:
                total += vc.numel()
            for vs_ in vs_list:
                total += vs_.numel() * 4
        return total


# ---------------------------------------------------------------------------
# Lazy dequantize view for `cache.key_cache[i]` / `cache.value_cache[i]`
# ---------------------------------------------------------------------------

class _LazyDequantList:
    """List-like view that dequantizes on access.

    HF code paths and our own `_cache_is_empty` / `_cache_read` reach
    for `cache.key_cache[layer_idx]` / `cache.value_cache[layer_idx]`.
    Those calls return the dequantized bf16 tensor for that layer; the
    underlying NF4 storage stays untouched.
    """

    def __init__(self, cache: NF4DynamicCache, *, is_value: bool):
        self._cache = cache
        self._is_value = is_value

    def __len__(self) -> int:
        return len(self._cache)

    def __getitem__(self, idx: int) -> torch.Tensor:
        k, v = self._cache._dequantize_layer(idx)
        return v if self._is_value else k

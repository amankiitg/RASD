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

# We MUST subclass transformers.cache_utils.Cache. Without that,
# transformers/llama/modeling_llama.py LlamaModel.forward triggers:
#   if use_cache and not isinstance(past_key_values, Cache):
#       past_key_values = DynamicCache.from_legacy_cache(past_key_values)
# i.e. our duck-typed cache would be replaced by a vanilla bf16
# DynamicCache before any attention layer sees it, defeating the
# whole NF4 storage point. (Verified 2026-05-10 against transformers
# v4.44.2 source.) Cache is the canonical base class on both pod
# (transformers 4.44.2) and local dev (transformers 5.5.4).
try:
    from transformers.cache_utils import Cache as _HFCache
except ImportError:  # transformers not installed in some test envs
    _HFCache = object


class NF4DynamicCache(_HFCache):
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
        bf16_prefix_size: int = 0,
    ):
        """Args:
            block_size: NF4 block size for the codec.
            dtype: dequantization output dtype (bf16 for inference).
            bf16_prefix_size: number of leading tokens (per layer) that
                bypass NF4 and are stored in bf16 instead. Used by
                StreamingLLM-style outlier preservation (Xiao et al.
                2024): the first ~128 tokens act as attention sinks and
                are disproportionately attended to throughout the model;
                quantizing them is the largest source of acceptance
                loss in speculative decoding (M4 Phase C 2026-05-10).
                Pass 0 (default) for pure-NF4 storage. Only the rank
                that holds global position 0 (rank 0 under
                sequence-parallel sharding) should be constructed with
                bf16_prefix_size > 0; other ranks' caches don't see the
                first global-tokens at all.
                Memory cost: 32 layers * 8 KV-heads * 128 tokens * 128
                head_dim * 2 (K+V) * 2 bytes = 8 MB per rank, negligible.
        """
        # We can't call super().__init__() unconditionally — Cache base
        # class signature is INCOMPATIBLE between transformers versions:
        #   * 4.44.2 (pod):  Cache.__init__() takes no args; Cache extends nn.Module
        #   * 5.5.4 (local): Cache.__init__(layers=None,
        #                                   layer_class_to_replicate=None,
        #                                   ...) and raises ValueError if
        #                                   both layers and
        #                                   layer_class_to_replicate are None.
        #                                   Cache no longer extends nn.Module.
        #
        # On 4.44.2 specifically, Cache extends nn.Module, so without
        # nn.Module.__init__ being called the subclass instance lacks
        # _parameters / _modules / _buffers dicts. Anything that calls
        # cache.to(device), .train(), or recursively walks .modules()
        # would crash. We don't currently do any of those on our hot
        # path, but explicitly run nn.Module.__init__ when applicable
        # so future HF code that does won't blow up. (Fix for blocker
        # #5 from 2026-05-10 third-pass review.)
        if isinstance(self, torch.nn.Module):
            torch.nn.Module.__init__(self)
        self.block_size = block_size
        self.dtype = dtype
        if bf16_prefix_size < 0:
            raise ValueError(f"bf16_prefix_size={bf16_prefix_size} must be >= 0")
        self.bf16_prefix_size = bf16_prefix_size
        self._k_codes: List[List[torch.Tensor]] = []
        self._k_scales: List[List[torch.Tensor]] = []
        self._v_codes: List[List[torch.Tensor]] = []
        self._v_scales: List[List[torch.Tensor]] = []
        # bf16 prefix store. None entry = no prefix yet for that layer;
        # tensor entry = (B, H, P, D) where P <= bf16_prefix_size.
        self._k_bf16_prefix: List[Optional[torch.Tensor]] = []
        self._v_bf16_prefix: List[Optional[torch.Tensor]] = []

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
            self._k_bf16_prefix.append(None)
            self._v_bf16_prefix.append(None)

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

        # Outlier-keep: route the first `bf16_prefix_size` tokens
        # (across the lifetime of this layer's cache) into a bf16
        # prefix store; route the rest into NF4. When
        # bf16_prefix_size=0 (default), this branch reduces to
        # pure NF4 storage with one extra integer comparison.
        new_tokens = key_states.shape[2]
        prefix_used = (
            0 if self._k_bf16_prefix[layer_idx] is None
            else self._k_bf16_prefix[layer_idx].shape[2]
        )
        prefix_room = max(0, self.bf16_prefix_size - prefix_used)
        n_to_prefix = min(prefix_room, new_tokens)

        if n_to_prefix > 0:
            new_bf16_k = key_states[:, :, :n_to_prefix, :].contiguous()
            new_bf16_v = value_states[:, :, :n_to_prefix, :].contiguous()
            if self._k_bf16_prefix[layer_idx] is None:
                self._k_bf16_prefix[layer_idx] = new_bf16_k
                self._v_bf16_prefix[layer_idx] = new_bf16_v
            else:
                self._k_bf16_prefix[layer_idx] = torch.cat(
                    [self._k_bf16_prefix[layer_idx], new_bf16_k], dim=2
                )
                self._v_bf16_prefix[layer_idx] = torch.cat(
                    [self._v_bf16_prefix[layer_idx], new_bf16_v], dim=2
                )

        if n_to_prefix < new_tokens:
            nf4_k = key_states[:, :, n_to_prefix:, :].contiguous()
            nf4_v = value_states[:, :, n_to_prefix:, :].contiguous()
            kc, ks = quantize_nf4(nf4_k, block_size=self.block_size)
            vc, vs = quantize_nf4(nf4_v, block_size=self.block_size)
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
        """Concatenate all stored content for `layer_idx`. Layout:
        [bf16_prefix (if any) | dequant(NF4 chunk 0) | ... | dequant(NF4 chunk N)].
        Returns (k, v) of shape (B, H, S, D) where S = prefix_len + sum(chunk_len)."""
        if layer_idx >= len(self._k_codes):
            empty = torch.empty(0, dtype=self.dtype)
            return empty, empty
        has_nf4 = bool(self._k_codes[layer_idx])
        has_prefix = (
            layer_idx < len(self._k_bf16_prefix)
            and self._k_bf16_prefix[layer_idx] is not None
        )
        if not has_nf4 and not has_prefix:
            empty = torch.empty(0, dtype=self.dtype)
            return empty, empty

        k_chunks: List[torch.Tensor] = []
        v_chunks: List[torch.Tensor] = []
        if has_prefix:
            # Prefix is already in self.dtype; just append.
            k_chunks.append(self._k_bf16_prefix[layer_idx])
            v_chunks.append(self._v_bf16_prefix[layer_idx])
        if has_nf4:
            for c, s in zip(self._k_codes[layer_idx], self._k_scales[layer_idx]):
                k_chunks.append(
                    dequantize_nf4(c, s, block_size=self.block_size, dtype=self.dtype)
                )
            for c, s in zip(self._v_codes[layer_idx], self._v_scales[layer_idx]):
                v_chunks.append(
                    dequantize_nf4(c, s, block_size=self.block_size, dtype=self.dtype)
                )
        # Position dim is axis 2 in (B, H, S, D); skip cat for trivial cases.
        if len(k_chunks) == 1:
            return k_chunks[0], v_chunks[0]
        return torch.cat(k_chunks, dim=2), torch.cat(v_chunks, dim=2)

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
        """Total stored positions for `layer_idx`. Sums bf16 prefix
        plus NF4 chunks."""
        if layer_idx >= len(self._k_codes):
            return 0
        prefix_len = (
            0 if (layer_idx >= len(self._k_bf16_prefix)
                  or self._k_bf16_prefix[layer_idx] is None)
            else self._k_bf16_prefix[layer_idx].shape[2]
        )
        # codes shape is (B, H, S_chunk, D//2); position dim is axis 2
        nf4_len = sum(c.shape[2] for c in self._k_codes[layer_idx])
        return prefix_len + nf4_len

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
    # Beam search compat — RASD doesn't use beam search; no-op
    # ------------------------------------------------------------------

    def reorder_cache(self, beam_idx: torch.Tensor) -> None:
        """No-op: RASD does temperature sampling, not beam search."""
        return None

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
        """Per-layer truncation. Layout is [bf16_prefix | NF4_chunks];
        truncation chops from the tail. Three cases:
          (a) new_seqlen >= prefix_len + nf4_len: nothing to do
          (b) prefix_len <= new_seqlen <  prefix_len + nf4_len: keep
              the full prefix, truncate the NF4 portion to
              new_seqlen - prefix_len
          (c) new_seqlen < prefix_len: truncate the prefix to
              new_seqlen, drop all NF4 chunks
        """
        if layer_idx >= len(self._k_codes):
            return
        prefix_len = (
            0 if (layer_idx >= len(self._k_bf16_prefix)
                  or self._k_bf16_prefix[layer_idx] is None)
            else self._k_bf16_prefix[layer_idx].shape[2]
        )

        if new_seqlen <= prefix_len:
            # case (c): truncate prefix, drop all NF4
            if new_seqlen == 0:
                self._k_bf16_prefix[layer_idx] = None
                self._v_bf16_prefix[layer_idx] = None
            else:
                self._k_bf16_prefix[layer_idx] = (
                    self._k_bf16_prefix[layer_idx][:, :, :new_seqlen, :].contiguous()
                )
                self._v_bf16_prefix[layer_idx] = (
                    self._v_bf16_prefix[layer_idx][:, :, :new_seqlen, :].contiguous()
                )
            self._k_codes[layer_idx] = []
            self._k_scales[layer_idx] = []
            self._v_codes[layer_idx] = []
            self._v_scales[layer_idx] = []
            return

        # cases (a)/(b): keep full prefix, truncate NF4 portion
        nf4_target = new_seqlen - prefix_len
        kc_list = self._k_codes[layer_idx]
        ks_list = self._k_scales[layer_idx]
        vc_list = self._v_codes[layer_idx]
        vs_list = self._v_scales[layer_idx]

        new_kc, new_ks = [], []
        new_vc, new_vs = [], []
        cumulative = 0
        for kc, ks_, vc, vs_ in zip(kc_list, ks_list, vc_list, vs_list):
            chunk_len = kc.shape[2]
            if cumulative + chunk_len <= nf4_target:
                new_kc.append(kc); new_ks.append(ks_)
                new_vc.append(vc); new_vs.append(vs_)
                cumulative += chunk_len
            else:
                keep = nf4_target - cumulative
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
    # Serialization for checkpointing (M4 C6 + finding #1, 2026-05-10)
    #
    # CRITICAL: do NOT dequantize during checkpoint save. The naive code
    # path `for layer in cache: ...` calls __iter__ which dequantizes
    # the entire cache to bf16 — at ctx=1M × W=8 that's ~64 GB / rank
    # materialized on the GPU before .cpu() copies it off, and the
    # checkpoint file balloons to bf16 size, defeating NF4's whole
    # ~3.55x compression story. Worse, the resumed legacy bf16 tuple
    # would force the next round into the bf16 cache path, losing NF4
    # storage permanently after first reload. Use these two methods
    # for save/load instead.
    # ------------------------------------------------------------------

    _SERIALIZED_VERSION_KEY = "_nf4_dynamic_cache_v1"

    def to_serializable(self) -> dict:
        """Return a dict-of-CPU-tensors that round-trips through torch.save
        WITHOUT dequantizing. The on-disk size is the same as the
        in-memory NF4 size (~3.55x smaller than bf16 storage at
        block_size=64). Reconstruct via NF4DynamicCache.from_serializable()."""
        return {
            self._SERIALIZED_VERSION_KEY: True,
            "block_size": int(self.block_size),
            "dtype": _dtype_name(self.dtype),
            "bf16_prefix_size": int(self.bf16_prefix_size),
            "k_codes": [
                [c.detach().cpu() for c in layer] for layer in self._k_codes
            ],
            "k_scales": [
                [s.detach().cpu() for s in layer] for layer in self._k_scales
            ],
            "v_codes": [
                [c.detach().cpu() for c in layer] for layer in self._v_codes
            ],
            "v_scales": [
                [s.detach().cpu() for s in layer] for layer in self._v_scales
            ],
            "k_bf16_prefix": [
                None if t is None else t.detach().cpu()
                for t in self._k_bf16_prefix
            ],
            "v_bf16_prefix": [
                None if t is None else t.detach().cpu()
                for t in self._v_bf16_prefix
            ],
        }

    @classmethod
    def from_serializable(cls, d: dict) -> "NF4DynamicCache":
        """Reconstruct an NF4DynamicCache from `to_serializable()` output.

        Tensors come back on CPU; the caller is expected to move them
        to the model's device via `.move_tensors_to(device)` if needed.

        Backwards-compatible with payloads from before outlier-keep was
        added: missing 'bf16_prefix_size' / 'k_bf16_prefix' / 'v_bf16_prefix'
        keys default to a pure-NF4 cache.
        """
        if not is_nf4_serialized(d):
            raise ValueError(
                "from_serializable expected a dict produced by "
                "NF4DynamicCache.to_serializable()"
            )
        cache = cls(
            block_size=d.get("block_size", 64),
            dtype=_dtype_from_name(d.get("dtype", "bfloat16")),
            bf16_prefix_size=d.get("bf16_prefix_size", 0),
        )
        cache._k_codes  = [list(layer) for layer in d["k_codes"]]
        cache._k_scales = [list(layer) for layer in d["k_scales"]]
        cache._v_codes  = [list(layer) for layer in d["v_codes"]]
        cache._v_scales = [list(layer) for layer in d["v_scales"]]
        n_layers = len(cache._k_codes)
        k_prefix = d.get("k_bf16_prefix") or [None] * n_layers
        v_prefix = d.get("v_bf16_prefix") or [None] * n_layers
        # Pad with None for any missing layers (tolerate length mismatch).
        cache._k_bf16_prefix = list(k_prefix) + [None] * (
            n_layers - len(k_prefix)
        )
        cache._v_bf16_prefix = list(v_prefix) + [None] * (
            n_layers - len(v_prefix)
        )
        return cache

    def move_tensors_to(self, device) -> "NF4DynamicCache":
        """Move all stored NF4 tensors AND the bf16 prefix to `device`.
        In-place; returns self. Mirrors GenerationCheckpoint.move_tensors_to
        for cache restore on resume."""
        device = torch.device(device)
        for li in range(len(self._k_codes)):
            self._k_codes[li]  = [t.to(device) for t in self._k_codes[li]]
            self._k_scales[li] = [t.to(device) for t in self._k_scales[li]]
            self._v_codes[li]  = [t.to(device) for t in self._v_codes[li]]
            self._v_scales[li] = [t.to(device) for t in self._v_scales[li]]
            if li < len(self._k_bf16_prefix) and self._k_bf16_prefix[li] is not None:
                self._k_bf16_prefix[li] = self._k_bf16_prefix[li].to(device)
            if li < len(self._v_bf16_prefix) and self._v_bf16_prefix[li] is not None:
                self._v_bf16_prefix[li] = self._v_bf16_prefix[li].to(device)
        return self

    # ------------------------------------------------------------------
    # Memory accounting (informational; used by tests)
    # ------------------------------------------------------------------

    def memory_bytes(self) -> int:
        """Total bytes stored on the device for K + V codes + scales
        + bf16 prefix, across all layers and chunks."""
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
        for kbp, vbp in zip(self._k_bf16_prefix, self._v_bf16_prefix):
            if kbp is not None:
                total += kbp.numel() * kbp.element_size()
            if vbp is not None:
                total += vbp.numel() * vbp.element_size()
        return total


# ---------------------------------------------------------------------------
# Serialization helpers (module-level so callers can sniff payloads)
# ---------------------------------------------------------------------------

_DTYPE_NAMES = {
    torch.bfloat16: "bfloat16",
    torch.float16:  "float16",
    torch.float32:  "float32",
    torch.float64:  "float64",
}


def _dtype_name(dtype: torch.dtype) -> str:
    return _DTYPE_NAMES.get(dtype, str(dtype).rsplit(".", 1)[-1])


def _dtype_from_name(name: str) -> torch.dtype:
    return getattr(torch, name, torch.bfloat16)


def is_nf4_serialized(payload) -> bool:
    """Return True if `payload` is a dict produced by
    NF4DynamicCache.to_serializable(). Used by GenerationCheckpoint
    save/load to detect the NF4-native form without dequantizing."""
    return (
        isinstance(payload, dict)
        and payload.get(NF4DynamicCache._SERIALIZED_VERSION_KEY) is True
    )


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

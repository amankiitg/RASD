"""Tests for src/models/nf4_dynamic_cache.py — true NF4-storage K/V cache.

CPU tests covering the cache class's HF DynamicCache compat surface +
the M4-specific in-place truncation. Real cache memory savings are
verified directly via .memory_bytes() vs equivalent bf16 storage.
"""
from __future__ import annotations

import pytest
import torch

from src.models.nf4_dynamic_cache import NF4DynamicCache


def _kv(B=1, H=4, S=8, D=128, seed=0):
    torch.manual_seed(seed)
    return (
        torch.randn(B, H, S, D).bfloat16(),
        torch.randn(B, H, S, D).bfloat16(),
    )


# ---------------------------------------------------------------------------
# Cache subclass — CRITICAL invariant
# ---------------------------------------------------------------------------

class TestCacheSubclass:
    """transformers/llama/modeling_llama.py LlamaModel.forward checks
    `not isinstance(past_key_values, Cache)` and replaces non-Cache
    objects via DynamicCache.from_legacy_cache(...). If our cache isn't
    a Cache subclass, the NF4 storage is silently swapped out for bf16
    DynamicCache before attention sees it, defeating the whole point.

    This test is the single most important assertion in this file.
    """

    def test_isinstance_transformers_cache(self):
        from transformers.cache_utils import Cache
        cache = NF4DynamicCache()
        assert isinstance(cache, Cache), (
            "C11 (b) regression: NF4DynamicCache no longer subclasses "
            "transformers.cache_utils.Cache. LlamaModel.forward will "
            "replace it with DynamicCache.from_legacy_cache() before "
            "attention sees it; kv_quant=True will silently store bf16. "
            "1M context will OOM at 40 GB SXM4."
        )

    def test_subclass_relationship_at_class_level(self):
        from transformers.cache_utils import Cache
        assert issubclass(NF4DynamicCache, Cache)


# ---------------------------------------------------------------------------
# Initial state
# ---------------------------------------------------------------------------

class TestInitialState:
    def test_empty_cache(self):
        cache = NF4DynamicCache()
        assert len(cache) == 0
        assert cache.get_seq_length(0) == 0
        assert cache.memory_bytes() == 0

    def test_get_seq_length_unknown_layer(self):
        cache = NF4DynamicCache()
        assert cache.get_seq_length(layer_idx=99) == 0


# ---------------------------------------------------------------------------
# Append via update()
# ---------------------------------------------------------------------------

class TestUpdate:
    def test_returns_dequantized_full_kv(self):
        """update() must return (k, v) bf16 tensors of shape matching input."""
        cache = NF4DynamicCache()
        k, v = _kv(B=1, H=32, S=8, D=128)
        out_k, out_v = cache.update(k, v, layer_idx=0)
        assert out_k.shape == k.shape
        assert out_v.shape == v.shape
        assert out_k.dtype == torch.bfloat16
        assert out_v.dtype == torch.bfloat16

    def test_update_grows_seq_length(self):
        cache = NF4DynamicCache()
        # First chunk: 8 positions
        cache.update(*_kv(S=8), layer_idx=0)
        assert cache.get_seq_length(0) == 8
        # Append 4 more
        cache.update(*_kv(S=4, seed=1), layer_idx=0)
        assert cache.get_seq_length(0) == 12

    def test_update_returns_concatenated_kv(self):
        """After two updates, returned (k, v) must include both chunks."""
        cache = NF4DynamicCache()
        cache.update(*_kv(S=8), layer_idx=0)
        out_k, _ = cache.update(*_kv(S=4, seed=1), layer_idx=0)
        # Second call should return shape (B, H, 12, D) — both chunks
        assert out_k.shape[2] == 12

    def test_update_per_layer_independent(self):
        """update() at layer_idx=2 should grow that layer without touching layer 0."""
        cache = NF4DynamicCache()
        cache.update(*_kv(S=4), layer_idx=0)
        cache.update(*_kv(S=8), layer_idx=2)
        assert cache.get_seq_length(0) == 4
        assert cache.get_seq_length(1) == 0
        assert cache.get_seq_length(2) == 8

    def test_rejects_mismatched_kv_shape(self):
        cache = NF4DynamicCache()
        k = torch.randn(1, 4, 8, 128).bfloat16()
        v = torch.randn(1, 4, 7, 128).bfloat16()
        with pytest.raises(ValueError, match="key_states.shape"):
            cache.update(k, v, layer_idx=0)

    def test_rejects_misaligned_head_dim(self):
        cache = NF4DynamicCache(block_size=64)
        k = torch.randn(1, 4, 8, 60).bfloat16()
        v = torch.randn(1, 4, 8, 60).bfloat16()
        with pytest.raises(ValueError, match="head_dim=60"):
            cache.update(k, v, layer_idx=0)

    def test_cache_kwargs_ignored(self):
        """cache_kwargs is accepted for HF compat but should not affect result."""
        cache = NF4DynamicCache()
        k, v = _kv()
        out_a = cache.update(k, v, layer_idx=0, cache_kwargs=None)
        cache2 = NF4DynamicCache()
        out_b = cache2.update(k, v, layer_idx=0,
                              cache_kwargs={"sin": None, "cos": None})
        # Same input -> identical quantization -> identical output
        assert torch.equal(out_a[0], out_b[0])


# ---------------------------------------------------------------------------
# Round-trip fidelity
# ---------------------------------------------------------------------------

class TestRoundTripFidelity:
    def test_dequantized_close_to_input(self):
        """NF4 round-trip on real-shaped K/V should preserve ~85%+ of magnitude."""
        cache = NF4DynamicCache()
        k, v = _kv(B=1, H=32, S=64, D=128, seed=42)
        out_k, out_v = cache.update(k, v, layer_idx=0)
        rel_err_k = (out_k.float() - k.float()).norm() / k.float().norm()
        rel_err_v = (out_v.float() - v.float()).norm() / v.float().norm()
        assert rel_err_k < 0.15, f"K rel_err {rel_err_k:.3f}"
        assert rel_err_v < 0.15, f"V rel_err {rel_err_v:.3f}"

    def test_no_error_accumulation_on_re_read(self):
        """Reading the same cache twice must produce IDENTICAL bf16
        tensors (no extra quantization on read)."""
        cache = NF4DynamicCache()
        k, v = _kv(B=1, H=4, S=8, D=128)
        out_k1, _ = cache.update(k, v, layer_idx=0)
        # Read again via key_cache property
        out_k2 = cache.key_cache[0]
        assert torch.equal(out_k1, out_k2)


# ---------------------------------------------------------------------------
# DynamicCache compat surface
# ---------------------------------------------------------------------------

class TestDynamicCacheCompat:
    def test_key_cache_value_cache_lazy_views(self):
        cache = NF4DynamicCache()
        cache.update(*_kv(S=8), layer_idx=0)
        cache.update(*_kv(S=4, seed=1), layer_idx=1)
        assert len(cache.key_cache) == 2
        assert len(cache.value_cache) == 2
        k0 = cache.key_cache[0]
        assert k0.shape == (1, 4, 8, 128)

    def test_iter_yields_kv_tuples(self):
        """`for layer in past_kv: k, v = layer[0], layer[1]` pattern from
        _truncate_kv must work."""
        cache = NF4DynamicCache()
        cache.update(*_kv(S=8), layer_idx=0)
        cache.update(*_kv(S=8), layer_idx=1)
        layers = list(cache)
        assert len(layers) == 2
        for k, v in layers:
            assert k.shape == (1, 4, 8, 128)
            assert v.shape == (1, 4, 8, 128)

    def test_getitem_returns_kv_tuple(self):
        cache = NF4DynamicCache()
        cache.update(*_kv(S=8), layer_idx=0)
        k, v = cache[0]
        assert k.shape == (1, 4, 8, 128)
        assert v.shape == (1, 4, 8, 128)

    def test_get_max_length_returns_none(self):
        """DynamicCache convention: unbounded length."""
        cache = NF4DynamicCache()
        assert cache.get_max_length() is None

    def test_get_usable_length_matches_seq_length(self):
        cache = NF4DynamicCache()
        cache.update(*_kv(S=12), layer_idx=0)
        assert cache.get_usable_length(8, layer_idx=0) == 12


# ---------------------------------------------------------------------------
# Truncation (M4 C6 + spec-decoding partial rejection)
# ---------------------------------------------------------------------------

class TestTruncate:
    def test_truncate_drops_full_chunks(self):
        cache = NF4DynamicCache()
        for _ in range(4):
            cache.update(*_kv(S=4), layer_idx=0)
        # Layer 0 has 16 positions; truncate to 8
        cache.truncate(8)
        assert cache.get_seq_length(0) == 8

    def test_truncate_partial_chunk(self):
        cache = NF4DynamicCache()
        cache.update(*_kv(S=8), layer_idx=0)
        cache.truncate(5)
        assert cache.get_seq_length(0) == 5

    def test_truncate_zero(self):
        cache = NF4DynamicCache()
        cache.update(*_kv(S=8), layer_idx=0)
        cache.truncate(0)
        assert cache.get_seq_length(0) == 0

    def test_truncate_in_place_returns_self(self):
        """truncate() should return self for chaining and to make the
        contract explicit (in-place mutation, not a new instance)."""
        cache = NF4DynamicCache()
        cache.update(*_kv(S=4), layer_idx=0)
        result = cache.truncate(2)
        assert result is cache

    def test_truncate_negative_raises(self):
        cache = NF4DynamicCache()
        with pytest.raises(ValueError, match="must be >= 0"):
            cache.truncate(-1)

    def test_truncate_applies_to_all_layers(self):
        cache = NF4DynamicCache()
        cache.update(*_kv(S=12), layer_idx=0)
        cache.update(*_kv(S=12), layer_idx=1)
        cache.update(*_kv(S=12), layer_idx=2)
        cache.truncate(6)
        for li in range(3):
            assert cache.get_seq_length(li) == 6

    def test_truncate_preserves_dequantized_values(self):
        """After truncation, the kept positions should dequantize to the
        SAME values they had before (no re-quantization)."""
        cache = NF4DynamicCache()
        cache.update(*_kv(S=12, seed=7), layer_idx=0)
        before_k, before_v = cache._dequantize_layer(0)
        cache.truncate(6)
        after_k, after_v = cache._dequantize_layer(0)
        assert torch.equal(before_k[:, :, :6, :], after_k)
        assert torch.equal(before_v[:, :, :6, :], after_v)


# ---------------------------------------------------------------------------
# Memory accounting (the headline claim)
# ---------------------------------------------------------------------------

class TestMemoryAccounting:
    def test_nf4_cache_uses_3_to_4x_less_than_bf16(self):
        """The whole point of C11. NF4 storage at block_size=64 should
        deliver ~3.5-3.7x compression vs bf16 storage."""
        cache = NF4DynamicCache(block_size=64)
        # Llama-2-7B-shaped per-rank slice: B=1, H=32, S=64, D=128
        # bf16 storage:  2 * 1 * 32 * 64 * 128 = 524288 bytes per K
        #                * 2 (K and V) = 1048576 bytes
        cache.update(*_kv(B=1, H=32, S=64, D=128), layer_idx=0)
        bf16_bytes = 1 * 32 * 64 * 128 * 2 * 2  # k + v
        nf4_bytes = cache.memory_bytes()
        ratio = bf16_bytes / nf4_bytes
        assert 3.0 < ratio < 4.0, (
            f"NF4 cache compression {ratio:.2f}x out of band — "
            f"expected ~3.5x at block_size=64"
        )

    def test_memory_grows_linearly_with_appends(self):
        """Two equal-sized appends should roughly double the memory."""
        cache = NF4DynamicCache()
        cache.update(*_kv(B=1, H=4, S=64, D=128), layer_idx=0)
        m1 = cache.memory_bytes()
        cache.update(*_kv(B=1, H=4, S=64, D=128, seed=1), layer_idx=0)
        m2 = cache.memory_bytes()
        # Within 5% of doubling
        assert 1.95 * m1 < m2 < 2.05 * m1

    def test_truncation_reduces_memory(self):
        cache = NF4DynamicCache()
        cache.update(*_kv(S=64), layer_idx=0)
        full_mem = cache.memory_bytes()
        cache.truncate(16)
        truncated_mem = cache.memory_bytes()
        # ~1/4 of original (16/64 of positions)
        assert truncated_mem < full_mem
        assert truncated_mem < 0.30 * full_mem


# ---------------------------------------------------------------------------
# Block-size + dtype variations
# ---------------------------------------------------------------------------

class TestSerialization:
    """to_serializable / from_serializable / is_nf4_serialized — the
    NF4-native checkpoint path. Critical because the legacy code path
    (`for layer in cache: ...`) would dequantize the cache during
    save and produce a bf16 checkpoint that loses NF4 storage on
    resume. Fix for high-risk finding #1, 2026-05-10 third-pass review."""

    def test_to_serializable_preserves_nf4_storage_size(self):
        """The serialized payload must NOT dequantize. Total bytes
        should match in-memory NF4 size, not bf16 size."""
        from src.models.nf4_dynamic_cache import NF4DynamicCache
        cache = NF4DynamicCache(block_size=64)
        # 32 layers × 64 positions per layer × Llama-2-7B head shape
        for layer in range(32):
            cache.update(*_kv(B=1, H=32, S=64, D=128), layer_idx=layer)

        nf4_in_memory_bytes = cache.memory_bytes()
        bf16_equivalent_bytes = (
            32     # layers
            * 1    # B
            * 32   # H
            * 64   # S
            * 128  # D
            * 2    # bytes per bf16 element
            * 2    # K and V
        )
        # Compute serialized bytes by adding up tensor sizes
        ser = cache.to_serializable()
        serialized_bytes = 0
        for key in ("k_codes", "k_scales", "v_codes", "v_scales"):
            for layer_chunks in ser[key]:
                for tensor in layer_chunks:
                    serialized_bytes += tensor.element_size() * tensor.numel()

        # Serialized must be within ~5% of in-memory NF4 size
        assert 0.95 * nf4_in_memory_bytes <= serialized_bytes <= 1.05 * nf4_in_memory_bytes
        # And FAR less than the bf16 equivalent
        ratio = bf16_equivalent_bytes / serialized_bytes
        assert ratio > 3.0, (
            f"to_serializable not preserving NF4 compression: "
            f"{ratio:.2f}x vs bf16 (expected >3.0x)"
        )

    def test_to_serializable_marker_present(self):
        from src.models.nf4_dynamic_cache import NF4DynamicCache, is_nf4_serialized
        cache = NF4DynamicCache()
        cache.update(*_kv(), layer_idx=0)
        ser = cache.to_serializable()
        assert is_nf4_serialized(ser), (
            "is_nf4_serialized must detect to_serializable() output "
            "so checkpoint loaders can dispatch to from_serializable"
        )

    def test_is_nf4_serialized_false_on_legacy_tuple(self):
        from src.models.nf4_dynamic_cache import is_nf4_serialized
        legacy_tuple = ((torch.zeros(1, 4, 8, 128).bfloat16(),
                         torch.zeros(1, 4, 8, 128).bfloat16()),)
        assert not is_nf4_serialized(legacy_tuple)
        assert not is_nf4_serialized(None)
        assert not is_nf4_serialized({"foo": "bar"})

    def test_round_trip_preserves_dequantized_values(self):
        """Save → load must produce a cache that dequantizes to the same
        values as the original — no extra quantization noise on the
        round trip (bytes are preserved verbatim)."""
        from src.models.nf4_dynamic_cache import NF4DynamicCache
        cache = NF4DynamicCache()
        for layer in range(4):
            cache.update(*_kv(seed=layer), layer_idx=layer)

        before_layers = [cache._dequantize_layer(i) for i in range(4)]
        ser = cache.to_serializable()
        restored = NF4DynamicCache.from_serializable(ser)
        after_layers = [restored._dequantize_layer(i) for i in range(4)]

        for (k_b, v_b), (k_a, v_a) in zip(before_layers, after_layers):
            assert torch.equal(k_b, k_a)
            assert torch.equal(v_b, v_a)

    def test_round_trip_preserves_block_size_and_dtype(self):
        from src.models.nf4_dynamic_cache import NF4DynamicCache
        cache = NF4DynamicCache(block_size=32, dtype=torch.float16)
        cache.update(
            torch.randn(1, 4, 8, 128, dtype=torch.float16),
            torch.randn(1, 4, 8, 128, dtype=torch.float16),
            layer_idx=0,
        )
        restored = NF4DynamicCache.from_serializable(cache.to_serializable())
        assert restored.block_size == 32
        assert restored.dtype == torch.float16

    def test_round_trip_through_torch_save_load(self, tmp_path):
        """End-to-end: torch.save the serialized form, torch.load it,
        reconstruct, and verify dequantized values match."""
        from src.models.nf4_dynamic_cache import NF4DynamicCache
        cache = NF4DynamicCache()
        cache.update(*_kv(B=1, H=8, S=16, D=128), layer_idx=0)
        cache.update(*_kv(B=1, H=8, S=16, D=128, seed=1), layer_idx=1)

        path = tmp_path / "ckpt.pt"
        torch.save(cache.to_serializable(), path)
        loaded = torch.load(path, weights_only=False)
        restored = NF4DynamicCache.from_serializable(loaded)

        for i in range(2):
            k_orig, v_orig = cache._dequantize_layer(i)
            k_back, v_back = restored._dequantize_layer(i)
            assert torch.equal(k_orig, k_back)
            assert torch.equal(v_orig, v_back)

    def test_from_serializable_rejects_non_nf4_payload(self):
        from src.models.nf4_dynamic_cache import NF4DynamicCache
        with pytest.raises(ValueError, match="from_serializable"):
            NF4DynamicCache.from_serializable({"random": "dict"})
        with pytest.raises(ValueError, match="from_serializable"):
            NF4DynamicCache.from_serializable(((torch.zeros(1, 4, 8, 128),),))

    def test_no_dequantization_during_to_serializable(self):
        """Spot-check: to_serializable must not invoke __iter__ /
        _dequantize_layer. We verify by reading the source — it must
        access self._k_codes / self._v_codes / etc. directly, not via
        the read API."""
        import inspect
        from src.models.nf4_dynamic_cache import NF4DynamicCache
        src = inspect.getsource(NF4DynamicCache.to_serializable)
        assert "_dequantize_layer" not in src, (
            "to_serializable invoking _dequantize_layer defeats the "
            "memory-saving guarantee — the whole point is to skip "
            "dequant during save"
        )
        assert "key_cache" not in src
        assert "value_cache" not in src


class TestBlockSizes:
    @pytest.mark.parametrize("block_size", [32, 64, 128])
    def test_round_trip_at_various_block_sizes(self, block_size):
        cache = NF4DynamicCache(block_size=block_size)
        torch.manual_seed(0)
        # head_dim must be divisible by block_size
        D = 128
        assert D % block_size == 0
        k = torch.randn(1, 4, 16, D).bfloat16()
        v = torch.randn(1, 4, 16, D).bfloat16()
        out_k, out_v = cache.update(k, v, layer_idx=0)
        assert out_k.shape == k.shape

    def test_fp32_dtype(self):
        cache = NF4DynamicCache(dtype=torch.float32)
        k = torch.randn(1, 4, 8, 128, dtype=torch.float32)
        v = torch.randn(1, 4, 8, 128, dtype=torch.float32)
        out_k, _ = cache.update(k, v, layer_idx=0)
        assert out_k.dtype == torch.float32

    def test_bf16_dtype_default(self):
        cache = NF4DynamicCache()
        assert cache.dtype == torch.bfloat16

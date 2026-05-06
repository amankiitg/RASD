"""Tests for src/models/nf4_kv.py — the M4 C11 NF4 codec.

CPU tests covering:
1. Packing/unpacking byte layout (lower-nibble = even, upper = odd)
2. Round-trip error bounds on normal-distributed data
3. Per-block scale independence
4. Dtype preservation through dequantize
5. Memory accounting helpers (validates the paper claim of ~3.6x)
6. Edge cases: zero block, constant block, all-positive, all-negative
"""
from __future__ import annotations

import math

import pytest
import torch

from src.models.nf4_kv import (
    NF4_CODES_FLOAT,
    NF4KVCache,
    compression_ratio,
    dequantize_nf4,
    packed_size_bytes,
    quantize_nf4,
)


# ---------------------------------------------------------------------------
# Codepoint constants
# ---------------------------------------------------------------------------

class TestCodepoints:
    def test_sixteen_codepoints(self):
        assert len(NF4_CODES_FLOAT) == 16

    def test_endpoints_are_minus_one_and_plus_one(self):
        assert NF4_CODES_FLOAT[0] == -1.0
        assert NF4_CODES_FLOAT[-1] == 1.0

    def test_strictly_increasing(self):
        """Codepoints must be sorted ascending — argmin(distance) needs that."""
        for i in range(len(NF4_CODES_FLOAT) - 1):
            assert NF4_CODES_FLOAT[i] < NF4_CODES_FLOAT[i + 1]

    def test_zero_in_codepoints(self):
        """Zero must be exactly representable."""
        assert 0.0 in NF4_CODES_FLOAT


# ---------------------------------------------------------------------------
# Quantize: shape + dtype invariants
# ---------------------------------------------------------------------------

class TestQuantizeShape:
    def test_codes_shape_half_input(self):
        x = torch.randn(64)
        codes, scales = quantize_nf4(x, block_size=64)
        assert codes.shape == (32,)
        assert codes.dtype == torch.uint8

    def test_scales_shape_one_per_block(self):
        x = torch.randn(256)  # 4 blocks of 64
        codes, scales = quantize_nf4(x, block_size=64)
        assert scales.shape == (4,)

    def test_scales_dtype_fp32(self):
        x = torch.randn(64).bfloat16()
        codes, scales = quantize_nf4(x)
        assert scales.dtype == torch.float32

    def test_2d_input(self):
        x = torch.randn(8, 128)  # 2 blocks of 64 along last dim
        codes, scales = quantize_nf4(x, block_size=64)
        assert codes.shape == (8, 64)
        assert scales.shape == (8, 2)

    def test_3d_input(self):
        x = torch.randn(2, 4, 256)  # 4 blocks of 64
        codes, scales = quantize_nf4(x, block_size=64)
        assert codes.shape == (2, 4, 128)
        assert scales.shape == (2, 4, 4)

    def test_rejects_non_floating_input(self):
        with pytest.raises(ValueError, match="floating"):
            quantize_nf4(torch.randint(0, 100, (64,), dtype=torch.int32))

    def test_rejects_unaligned_last_dim(self):
        x = torch.randn(63)  # not divisible by 64
        with pytest.raises(ValueError, match="not divisible"):
            quantize_nf4(x, block_size=64)


# ---------------------------------------------------------------------------
# Pack/unpack byte layout
# ---------------------------------------------------------------------------

class TestPacking:
    def test_lower_nibble_is_even_index(self):
        """Code at index 0 occupies the LOWER nibble of byte 0."""
        # Construct a tensor whose first 64 values quantize to known codes.
        # Easier: round-trip a simple input and verify packing convention.
        x = torch.zeros(64)
        codes, scales = quantize_nf4(x, block_size=64)
        # All zeros: scale=1 (clamped), normed=0, nearest codepoint is 0.0
        # which is index 7 in NF4_CODES_FLOAT.
        # So every code should be 7. Packed byte = (7 & 0xF) | ((7 & 0xF) << 4) = 7 + 112 = 119
        expected_byte = (7 & 0x0F) | ((7 & 0x0F) << 4)
        assert int(codes[0]) == expected_byte

    def test_pack_unpack_round_trips_indices(self):
        """The packing convention itself round-trips correctly."""
        # Inject distinct values such that quantization indices differ
        # at even vs odd positions
        block = torch.tensor([
            -1.0, 0.0, -1.0, 0.0, -1.0, 0.0, -1.0, 0.0,  # alternating -1, 0
            -1.0, 0.0, -1.0, 0.0, -1.0, 0.0, -1.0, 0.0,
            -1.0, 0.0, -1.0, 0.0, -1.0, 0.0, -1.0, 0.0,
            -1.0, 0.0, -1.0, 0.0, -1.0, 0.0, -1.0, 0.0,
            -1.0, 0.0, -1.0, 0.0, -1.0, 0.0, -1.0, 0.0,
            -1.0, 0.0, -1.0, 0.0, -1.0, 0.0, -1.0, 0.0,
            -1.0, 0.0, -1.0, 0.0, -1.0, 0.0, -1.0, 0.0,
            -1.0, 0.0, -1.0, 0.0, -1.0, 0.0, -1.0, 0.0,
        ])
        codes, scales = quantize_nf4(block, block_size=64)
        # Index 0 (-1.0) -> code 0; index 7 (0.0) -> code 7
        # Packed: lower=0, upper=7 -> byte = 0 | (7 << 4) = 112
        assert int(codes[0]) == (0 & 0x0F) | ((7 & 0x0F) << 4)


# ---------------------------------------------------------------------------
# Round-trip
# ---------------------------------------------------------------------------

class TestRoundTrip:
    def test_zeros_dequant_to_zero(self):
        """All-zero input round-trips to all-zero output."""
        x = torch.zeros(128)
        codes, scales = quantize_nf4(x)
        out = dequantize_nf4(codes, scales)
        assert torch.all(out == 0)

    def test_shape_preserved(self):
        x = torch.randn(2, 4, 256)
        codes, scales = quantize_nf4(x)
        out = dequantize_nf4(codes, scales)
        assert out.shape == x.shape

    def test_dtype_argument_respected(self):
        x = torch.randn(64).bfloat16()
        codes, scales = quantize_nf4(x)
        out = dequantize_nf4(codes, scales, dtype=torch.bfloat16)
        assert out.dtype == torch.bfloat16

    def test_normal_dist_relative_error_under_12_pct(self):
        """NF4 quality bound on normally-distributed inputs.

        Pure absmax-scaling NF4 (without QLoRA's "double quantization"
        of the scales themselves) achieves ~10% relative L2 error on
        N(0,1) inputs at block_size=64. The 12% bound here gives
        headroom for seed noise; if it ever exceeds 12% the codepoints
        or per-block math has regressed.

        For ~2-3% PPL degradation on KV-cache use (the actual M4 C11
        target per KIVI/KVQuant prior art), this 10% per-element error
        is well within budget — the attention dot-product averages
        across head_dim, so per-token error is much smaller.
        """
        torch.manual_seed(0)
        x = torch.randn(8192)
        codes, scales = quantize_nf4(x, block_size=64)
        recon = dequantize_nf4(codes, scales, dtype=torch.float32)
        rel_err = (recon - x).norm() / x.norm()
        assert rel_err < 0.12, (
            f"NF4 relative L2 error {rel_err.item():.4f} exceeds 12% — "
            f"either codepoints or block math regressed"
        )

    def test_max_value_not_clipped(self):
        """The max-magnitude value in each block must dequantize to
        ±scale (since the scale is set to absmax)."""
        x = torch.zeros(64)
        x[0] = 1.5   # absmax
        codes, scales = quantize_nf4(x, block_size=64)
        recon = dequantize_nf4(codes, scales)
        # x[0] / scale = 1.5 / 1.5 = 1.0 -> NF4 code 15 (+1.0)
        # Reconstruction: code +1.0 * scale 1.5 = 1.5
        assert abs(recon[0].item() - 1.5) < 1e-5

    def test_block_independence(self):
        """Per-block scaling — large value in block 1 does NOT affect
        block 0's reconstruction precision."""
        x = torch.zeros(128)
        x[64] = 1000.0  # huge value, but only in block 1
        x[5] = 0.5      # small value in block 0
        codes, scales = quantize_nf4(x, block_size=64)
        recon = dequantize_nf4(codes, scales)
        # Block 0's scale is 0.5; precision should match small-value regime
        assert abs(recon[5].item() - 0.5) < 1e-5
        # Block 1's scale is 1000; precision is ~0.07 * 1000 = 70 around
        # the 1000 value (one NF4 code-step at top end)
        assert abs(recon[64].item() - 1000.0) < 80


# ---------------------------------------------------------------------------
# Memory accounting (paper claim)
# ---------------------------------------------------------------------------

class TestMemoryAccounting:
    def test_packed_size_64_block(self):
        """64 elements -> 32 bytes codes + 4 bytes scale = 36 bytes."""
        assert packed_size_bytes(64, block_size=64) == 36

    def test_packed_size_1024_block_64(self):
        """1024 elements / 64 block -> 512 bytes codes + 16*4 = 576 bytes."""
        assert packed_size_bytes(1024, block_size=64) == 512 + 64

    def test_compression_ratio_at_block_64(self):
        """At block_size=64, ratio = 2 / (0.5 + 0.0625) = ~3.55x."""
        assert abs(compression_ratio(64) - 3.555555) < 1e-3

    def test_compression_better_at_larger_blocks(self):
        """Larger blocks amortize the scale cost — better compression."""
        assert compression_ratio(128) > compression_ratio(64)
        assert compression_ratio(256) > compression_ratio(128)
        # Asymptote: 4x exact at infinite block_size
        assert compression_ratio(2**20) < 4.0
        assert compression_ratio(2**20) > 3.99

    def test_packed_size_rejects_unaligned(self):
        with pytest.raises(ValueError, match="not divisible"):
            packed_size_bytes(63, block_size=64)


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:
    def test_constant_positive_block(self):
        """Block of constant +1.0 — scale=1, all values quant to code 15."""
        x = torch.full((64,), 0.5)
        codes, scales = quantize_nf4(x)
        recon = dequantize_nf4(codes, scales)
        # All values normalize to +1.0 (since absmax = 0.5 -> normed = 1)
        # code 15 (+1.0), scale=0.5 -> reconstruction = 0.5
        assert torch.allclose(recon, x, atol=1e-5)

    def test_constant_negative_block(self):
        x = torch.full((64,), -0.5)
        codes, scales = quantize_nf4(x)
        recon = dequantize_nf4(codes, scales)
        assert torch.allclose(recon, x, atol=1e-5)

    def test_block_with_one_outlier(self):
        """Outlier sets the scale — small values get coarser quantization
        in that block. Validate the math; not a correctness fail."""
        x = torch.zeros(64)
        x[0] = 100.0
        codes, scales = quantize_nf4(x)
        recon = dequantize_nf4(codes, scales)
        # The outlier gets exact reconstruction
        assert abs(recon[0].item() - 100.0) < 1e-3
        # Other (zero) values: normed = 0, nearest code = 7 (0.0),
        # reconstruction = 0 * 100 = 0
        assert torch.allclose(recon[1:], torch.zeros(63), atol=1e-5)

    def test_bf16_input_dtype(self):
        x = torch.randn(128).bfloat16()
        codes, scales = quantize_nf4(x)
        recon = dequantize_nf4(codes, scales, dtype=torch.bfloat16)
        # bf16 limits us to ~3 decimal digits of precision; loosen the
        # bound vs the float32 test
        rel_err = (recon.float() - x.float()).norm() / x.float().norm()
        assert rel_err < 0.10


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------

class TestDeterminism:
    def test_same_input_same_output(self):
        torch.manual_seed(0)
        x = torch.randn(256)
        c1, s1 = quantize_nf4(x)
        c2, s2 = quantize_nf4(x)
        assert torch.equal(c1, c2)
        assert torch.equal(s1, s2)

    def test_dequant_deterministic(self):
        torch.manual_seed(0)
        x = torch.randn(256)
        codes, scales = quantize_nf4(x)
        r1 = dequantize_nf4(codes, scales)
        r2 = dequantize_nf4(codes, scales)
        assert torch.equal(r1, r2)


# ---------------------------------------------------------------------------
# Block-size variations
# ---------------------------------------------------------------------------

class TestBlockSizes:
    @pytest.mark.parametrize("block_size", [16, 32, 64, 128, 256])
    def test_round_trip_at_various_blocks(self, block_size):
        torch.manual_seed(0)
        N = 1024
        assert N % block_size == 0
        x = torch.randn(N)
        codes, scales = quantize_nf4(x, block_size=block_size)
        recon = dequantize_nf4(codes, scales, block_size=block_size)
        rel_err = (recon - x).norm() / x.norm()
        # Smaller blocks = better fidelity (more scales / amortized).
        # Pure-absmax NF4 typically lands 5-12%; bound at 15% gives
        # comfortable headroom on the larger blocks where one scale
        # covers more values.
        assert rel_err < 0.15, (
            f"block_size={block_size}: rel_err={rel_err.item():.4f}"
        )


# ---------------------------------------------------------------------------
# NF4KVCache wrapper
# ---------------------------------------------------------------------------

class TestNF4KVCacheBasic:
    def test_initial_state(self):
        cache = NF4KVCache()
        assert cache.num_chunks == 0
        assert len(cache) == 0
        assert cache.memory_bytes() == 0

    def test_add_kv_single_chunk(self):
        cache = NF4KVCache(block_size=64)
        torch.manual_seed(0)
        # Llama-2-7B-shaped: B=1, H=32, S=8, D=128
        k = torch.randn(1, 32, 8, 128).bfloat16()
        v = torch.randn(1, 32, 8, 128).bfloat16()
        cache.add_kv(k, v)
        assert cache.num_chunks == 1
        assert len(cache) == 8

    def test_add_kv_multiple_chunks(self):
        cache = NF4KVCache(block_size=64)
        torch.manual_seed(0)
        for _ in range(3):
            k = torch.randn(1, 32, 4, 128).bfloat16()
            v = torch.randn(1, 32, 4, 128).bfloat16()
            cache.add_kv(k, v)
        assert cache.num_chunks == 3
        assert len(cache) == 12  # 3 chunks of 4 positions

    def test_get_kv_returns_correct_shape(self):
        cache = NF4KVCache(block_size=64)
        for _ in range(2):
            cache.add_kv(
                torch.randn(1, 32, 5, 128).bfloat16(),
                torch.randn(1, 32, 5, 128).bfloat16(),
            )
        k, v = cache.get_kv()
        assert k.shape == (1, 32, 10, 128)
        assert v.shape == (1, 32, 10, 128)
        assert k.dtype == torch.bfloat16

    def test_get_kv_returns_dequantized(self):
        """Output values are reconstructed (not equal to input but close)."""
        cache = NF4KVCache(block_size=64)
        torch.manual_seed(42)
        k_orig = torch.randn(1, 4, 8, 128).bfloat16()
        v_orig = torch.randn(1, 4, 8, 128).bfloat16()
        cache.add_kv(k_orig, v_orig)
        k_out, v_out = cache.get_kv()
        # Should be close but not equal (lossy)
        rel_err_k = (k_out.float() - k_orig.float()).norm() / k_orig.float().norm()
        rel_err_v = (v_out.float() - v_orig.float()).norm() / v_orig.float().norm()
        assert rel_err_k < 0.15
        assert rel_err_v < 0.15
        # NOT byte-equal
        assert not torch.equal(k_out, k_orig)


class TestNF4KVCacheValidation:
    def test_rejects_mismatched_k_v_shape(self):
        cache = NF4KVCache()
        k = torch.randn(1, 32, 8, 128).bfloat16()
        v = torch.randn(1, 32, 7, 128).bfloat16()  # mismatched S
        with pytest.raises(ValueError, match="k.shape"):
            cache.add_kv(k, v)

    def test_rejects_3d_input(self):
        cache = NF4KVCache()
        with pytest.raises(ValueError, match="\\(B,H,S,D\\)"):
            cache.add_kv(
                torch.randn(32, 8, 128).bfloat16(),
                torch.randn(32, 8, 128).bfloat16(),
            )

    def test_rejects_misaligned_head_dim(self):
        cache = NF4KVCache(block_size=64)
        with pytest.raises(ValueError, match="head_dim=63"):
            cache.add_kv(
                torch.randn(1, 32, 8, 63).bfloat16(),
                torch.randn(1, 32, 8, 63).bfloat16(),
            )

    def test_rejects_inconsistent_dimensions_across_appends(self):
        cache = NF4KVCache(block_size=64)
        cache.add_kv(
            torch.randn(1, 32, 4, 128).bfloat16(),
            torch.randn(1, 32, 4, 128).bfloat16(),
        )
        with pytest.raises(ValueError, match="shape mismatch"):
            cache.add_kv(
                torch.randn(1, 16, 4, 128).bfloat16(),  # H changed
                torch.randn(1, 16, 4, 128).bfloat16(),
            )


class TestNF4KVCacheTruncate:
    def test_truncate_drops_full_chunks(self):
        cache = NF4KVCache(block_size=64)
        for _ in range(4):  # 4 chunks of 4 positions = 16 total
            cache.add_kv(
                torch.randn(1, 32, 4, 128).bfloat16(),
                torch.randn(1, 32, 4, 128).bfloat16(),
            )
        cache.truncate(8)  # keep first 8 positions = first 2 chunks
        assert len(cache) == 8
        assert cache.num_chunks == 2

    def test_truncate_partial_chunk(self):
        cache = NF4KVCache(block_size=64)
        cache.add_kv(
            torch.randn(1, 32, 8, 128).bfloat16(),
            torch.randn(1, 32, 8, 128).bfloat16(),
        )
        cache.truncate(5)  # keep first 5 of the 8 in this chunk
        assert len(cache) == 5
        # Output reflects truncation
        k, v = cache.get_kv()
        assert k.shape[2] == 5

    def test_truncate_zero_clears(self):
        cache = NF4KVCache(block_size=64)
        cache.add_kv(
            torch.randn(1, 32, 8, 128).bfloat16(),
            torch.randn(1, 32, 8, 128).bfloat16(),
        )
        cache.truncate(0)
        assert len(cache) == 0
        assert cache.num_chunks == 0

    def test_truncate_to_existing_length_is_noop(self):
        cache = NF4KVCache(block_size=64)
        cache.add_kv(
            torch.randn(1, 32, 8, 128).bfloat16(),
            torch.randn(1, 32, 8, 128).bfloat16(),
        )
        cache.truncate(8)  # keep all
        assert len(cache) == 8

    def test_truncate_negative_raises(self):
        cache = NF4KVCache()
        with pytest.raises(ValueError, match="must be >= 0"):
            cache.truncate(-1)


class TestNF4KVCacheMemory:
    def test_memory_compresses_vs_bf16(self):
        """Per the C11 design contract: NF4 cache uses ~3-4x less memory
        than bf16 storage of the same K/V shape."""
        cache = NF4KVCache(block_size=64)
        # Llama-2-7B per-rank slice at ctx=64k×W=8: B=1, H=32, S=8192, D=128
        # bf16: 2 * 8192 * 32 * 128 * 2 (k+v) = 134 MB
        # Smaller test shape to keep test fast: S=64
        cache.add_kv(
            torch.randn(1, 32, 64, 128).bfloat16(),
            torch.randn(1, 32, 64, 128).bfloat16(),
        )
        bf16_bytes = 1 * 32 * 64 * 128 * 2 * 2  # k + v
        nf4_bytes = cache.memory_bytes()
        ratio = bf16_bytes / nf4_bytes
        # Conservative bound — exact ratio depends on scale overhead
        assert 3.0 < ratio < 4.0, (
            f"NF4 cache compression ratio {ratio:.2f}x out of expected band"
        )


class TestNF4KVCacheEmptyGet:
    def test_empty_cache_get_kv_raises(self):
        """Until at least one add_kv, dimensions are unknown."""
        cache = NF4KVCache()
        with pytest.raises(RuntimeError, match="empty"):
            cache.get_kv()

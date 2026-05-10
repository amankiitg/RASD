"""Tests for the C11 kv_quant flag plumbing.

The flag goes through three layers:
  RASDConfig.kv_quant = False (default)
    -> _load_models -> install_ring_attention(kv_quant=...)
    -> attn._ring_kv_quant
    -> _ring_llama_attention_forward branch
    -> _kv_quant_round_trip helper

CPU tests cover the helper directly + source-inspection guards on the
plumbing layer-by-layer. Full end-to-end exercise (kv_quant=True under
multi-rank with real KV activations) is M4 Phase C pod-side work; the
codec itself is locked by tests/test_nf4_kv.py.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest
import torch

from src.models.rasd_inference import RASDConfig
from src.models.ring_llama_attention import _kv_quant_round_trip

REPO_ROOT = Path(__file__).resolve().parent.parent
RASD_INF_SRC = (REPO_ROOT / "src" / "models" / "rasd_inference.py").read_text()
RING_PATCH_SRC = (REPO_ROOT / "src" / "models" / "ring_llama_attention.py").read_text()
RUN_EXP_SRC = (REPO_ROOT / "run_experiment.py").read_text()


# ---------------------------------------------------------------------------
# Layer 1: RASDConfig default
# ---------------------------------------------------------------------------

class TestConfigDefault:
    def test_kv_quant_default_false(self):
        """M3 byte-identical invariant: default off."""
        cfg = RASDConfig()
        assert cfg.kv_quant is False

    def test_kv_quant_settable(self):
        cfg = RASDConfig(kv_quant=True)
        assert cfg.kv_quant is True


# ---------------------------------------------------------------------------
# Layer 2: _kv_quant_round_trip helper
# ---------------------------------------------------------------------------

class TestRoundTripHelper:
    def test_shape_preserved(self):
        torch.manual_seed(0)
        k = torch.randn(1, 32, 16, 128).bfloat16()
        v = torch.randn(1, 32, 16, 128).bfloat16()
        k_out, v_out = _kv_quant_round_trip(k, v)
        assert k_out.shape == k.shape
        assert v_out.shape == v.shape

    def test_dtype_preserved(self):
        torch.manual_seed(0)
        k = torch.randn(1, 32, 16, 128).bfloat16()
        v = torch.randn(1, 32, 16, 128).bfloat16()
        k_out, v_out = _kv_quant_round_trip(k, v)
        assert k_out.dtype == torch.bfloat16
        assert v_out.dtype == torch.bfloat16

    def test_values_lossy_but_close(self):
        """Round-trip introduces NF4 quantization error — output is
        close but not equal to input."""
        torch.manual_seed(0)
        k = torch.randn(1, 32, 16, 128).bfloat16()
        v = torch.randn(1, 32, 16, 128).bfloat16()
        k_out, v_out = _kv_quant_round_trip(k, v)
        # Not byte-equal (codec is lossy)
        assert not torch.equal(k_out, k)
        # But within reasonable error bound for real-shaped K/V
        rel_err_k = (k_out.float() - k.float()).norm() / k.float().norm()
        rel_err_v = (v_out.float() - v.float()).norm() / v.float().norm()
        assert rel_err_k < 0.15
        assert rel_err_v < 0.15

    def test_misaligned_head_dim_passthrough(self):
        """head_dim not divisible by block_size -> returned unchanged
        (defensive — don't crash the verify loop on misconfig)."""
        k = torch.randn(1, 32, 16, 60).bfloat16()  # 60 not div by 64
        v = torch.randn(1, 32, 16, 60).bfloat16()
        k_out, v_out = _kv_quant_round_trip(k, v, block_size=64)
        assert torch.equal(k_out, k)
        assert torch.equal(v_out, v)

    def test_llama2_7b_shape_works(self):
        """Llama-2-7B head_dim=128 — must work with default block_size=64."""
        torch.manual_seed(0)
        k = torch.randn(1, 32, 64, 128).bfloat16()
        v = torch.randn(1, 32, 64, 128).bfloat16()
        k_out, v_out = _kv_quant_round_trip(k, v)
        assert k_out.shape == k.shape

    def test_custom_block_size(self):
        torch.manual_seed(0)
        k = torch.randn(1, 32, 8, 128).bfloat16()
        v = torch.randn(1, 32, 8, 128).bfloat16()
        # block_size=32: 128 / 32 = 4 blocks per head
        k_out, v_out = _kv_quant_round_trip(k, v, block_size=32)
        assert k_out.shape == k.shape


# ---------------------------------------------------------------------------
# Layer 3: install_ring_attention plumbing
# ---------------------------------------------------------------------------

class TestInstallerPlumbing:
    def test_install_ring_attention_accepts_kv_quant_kwarg(self):
        """The installer's signature must include kv_quant."""
        assert re.search(
            r"def install_ring_attention\([^)]*kv_quant[^)]*\)",
            RING_PATCH_SRC, re.DOTALL,
        ), (
            "C11 regression: install_ring_attention lost the kv_quant "
            "kwarg — flag would never reach the patched module"
        )

    def test_attn_ring_kv_quant_attribute_set(self):
        """Each patched attention module must get _ring_kv_quant set."""
        assert "attn._ring_kv_quant = bool(kv_quant)" in RING_PATCH_SRC, (
            "C11 regression: _ring_kv_quant not stored on attention module"
        )

    def test_kv_quant_logged_at_install_time(self):
        """The install_ring_attention log line must include kv_quant so
        we can confirm the flag is set correctly on real pod runs."""
        assert "kv_quant=%s" in RING_PATCH_SRC, (
            "C11 regression: install log doesn't include kv_quant"
        )


# ---------------------------------------------------------------------------
# Layer 4: forward path (true NF4 storage replaces the round-trip helper)
# ---------------------------------------------------------------------------

class TestForwardPath:
    """The 2026-05-10 (b)-scope refactor moved NF4 from a round-trip in
    the forward to true storage in NF4DynamicCache. The forward no
    longer calls _kv_quant_round_trip — the cache itself does the
    quantize-on-append + dequantize-on-read. Quantizing the already-
    dequantized cache output again would just compound error.
    """

    def test_round_trip_helper_no_longer_invoked_from_forward(self):
        """The forward path must NOT call _kv_quant_round_trip — true
        NF4 storage in the cache makes it redundant. (Helper itself
        kept in the module for backward-compat tests.)"""
        # Search for the helper invocation in the forward function.
        # The function spans from `_ring_llama_attention_forward` to
        # the next top-level `def`. Pull just that slice.
        m = re.search(
            r"def _ring_llama_attention_forward\(.*?\n(?=def |\Z)",
            RING_PATCH_SRC, re.DOTALL,
        )
        assert m is not None, "could not locate _ring_llama_attention_forward"
        forward_body = m.group(0)
        assert "_kv_quant_round_trip(k_full, v_full)" not in forward_body, (
            "C11 regression: forward still invokes _kv_quant_round_trip; "
            "with true NF4 storage in the cache, this would compound "
            "quantization error on every read"
        )

    def test_round_trip_helper_still_exposed(self):
        """The _kv_quant_round_trip helper itself is kept (used by
        legacy unit tests + as a debugging tool). Just shouldn't be
        wired into the forward path anymore."""
        assert "def _kv_quant_round_trip(" in RING_PATCH_SRC


class TestNF4DynamicCacheWired:
    """The (b) refactor: when cfg.kv_quant=True, generate() supplies an
    NF4DynamicCache as initial past_key_values."""

    def test_nf4_cache_imported_in_generate(self):
        from pathlib import Path
        rasd_inf = (Path(__file__).resolve().parent.parent
                    / "src" / "models" / "rasd_inference.py").read_text()
        assert "from src.models.nf4_dynamic_cache import NF4DynamicCache" in rasd_inf, (
            "C11 (b) regression: NF4DynamicCache import missing from "
            "rasd_inference.py — generate() can't construct one"
        )

    def test_nf4_cache_constructed_under_kv_quant_flag(self):
        """generate() must construct an NF4DynamicCache only when
        cfg.kv_quant=True. Default off must leave initial_cache=None
        so HF creates its own bf16 DynamicCache."""
        from pathlib import Path
        rasd_inf = (Path(__file__).resolve().parent.parent
                    / "src" / "models" / "rasd_inference.py").read_text()
        # Look for the gated construction (allow generous whitespace
        # for the outlier-keep comment block added 2026-05-10).
        assert re.search(
            r"if cfg\.kv_quant:[\s\S]{0,800}NF4DynamicCache\(",
            rasd_inf,
        ), (
            "C11 (b) regression: NF4DynamicCache construction not gated "
            "by `if cfg.kv_quant:`"
        )

    def test_initial_cache_passed_to_target_model(self):
        """target_model(...) call in prefill must pass past_key_values=initial_cache."""
        from pathlib import Path
        rasd_inf = (Path(__file__).resolve().parent.parent
                    / "src" / "models" / "rasd_inference.py").read_text()
        assert re.search(
            r"target_out\s*=\s*self\.target_model\([\s\S]{0,400}past_key_values\s*=\s*initial_cache",
            rasd_inf,
        ), (
            "C11 (b) regression: prefill target_model() call doesn't pass "
            "past_key_values=initial_cache; HF would construct its own "
            "bf16 cache and the kv_quant flag would have no effect"
        )

    def test_prefill_uses_num_logits_to_keep_1(self):
        """The target prefill must pass num_logits_to_keep=1 so HF
        only materializes the last position's logits. Without this,
        a full (B, S_local, 32000) tensor is allocated — 8.2 GB at
        1M context (S_local=128k). Required since we only use
        ...[:, -1, :] downstream.

        LlamaForCausalLM.forward gained this kwarg in transformers
        4.45+; we bumped requirements-lock.txt from 4.44.2 to 4.46.3
        specifically to enable this. (Earlier attempt with 4.44.2
        failed: 'unexpected keyword argument num_logits_to_keep'.)

        The verify forward (separate call) is intentionally NOT
        constrained — its t_input is only spec_steps+1 tokens, so
        keeping all logits there is cheap and we use them all.
        """
        from pathlib import Path
        rasd_inf = (Path(__file__).resolve().parent.parent
                    / "src" / "models" / "rasd_inference.py").read_text()
        match = re.search(
            r"target_out\s*=\s*self\.target_model\("
            r"[\s\S]{0,600}past_key_values\s*=\s*initial_cache,?"
            r"[\s\S]{0,300}num_logits_to_keep\s*=\s*1",
            rasd_inf,
        )
        assert match, (
            "M4 Phase C lever #1 regression: prefill target_model() call "
            "must pass num_logits_to_keep=1 to skip materializing the "
            "(B, S_local, 32000) logits tensor. Saves 8.2 GB at 1M context."
        )

    def test_truncate_kv_preserves_nf4_storage(self):
        """_truncate_kv must call cache.truncate() in place when the
        cache has a truncate method — otherwise the next round drops
        back to bf16 legacy tuple."""
        from pathlib import Path
        rasd_inf = (Path(__file__).resolve().parent.parent
                    / "src" / "models" / "rasd_inference.py").read_text()
        # Look for the hasattr-based dispatch
        assert re.search(
            r"hasattr\(past_kv,\s*[\"\']truncate[\"\'][\s\S]{0,200}past_kv\.truncate\(new_len\)",
            rasd_inf,
        ), (
            "C11 (b) regression: _truncate_kv doesn't dispatch to "
            "cache.truncate() for cache types that support it; NF4 "
            "storage would convert to bf16 legacy tuple every round"
        )


# ---------------------------------------------------------------------------
# Layer 5: RASDInference._load_models propagates cfg.kv_quant
# ---------------------------------------------------------------------------

class TestLoadModelsPlumbing:
    def test_install_ring_called_with_kv_quant(self):
        """RASDInference._load_models must pass cfg.kv_quant through to
        the installer."""
        assert "kv_quant=cfg.kv_quant" in RASD_INF_SRC, (
            "C11 regression: cfg.kv_quant not propagated to "
            "install_ring_attention call"
        )


# ---------------------------------------------------------------------------
# Layer 6: run_experiment.py reads kv_quant from run dict
# ---------------------------------------------------------------------------

class TestRunExperimentPlumbing:
    def test_run_experiment_passes_kv_quant_to_rasd_config(self):
        """run_experiment._run_single_worker pulls kv_quant from the
        run dict (default False) and passes to RASDConfig."""
        assert re.search(
            r'kv_quant\s*=\s*bool\(run\.get\([\"\']kv_quant[\"\'],\s*False\)\)',
            RUN_EXP_SRC,
        ), (
            "C11 regression: kv_quant not read from run dict in "
            "_run_single_worker"
        )

    def test_run_experiment_propagates_outlier_keep_knobs(self):
        """run_experiment._run_single_worker must explicitly pass the
        2026-05-10 NF4 mitigation knobs to RASDConfig so they take
        effect on the pod. Without this, the RASDConfig defaults
        would still apply, but a YAML override (for A/B testing) would
        be silently dropped."""
        assert re.search(
            r"kv_outlier_prefix_size\s*=\s*int\(run\.get\([\"\']kv_outlier_prefix_size[\"\']",
            RUN_EXP_SRC,
        ), (
            "kv_outlier_prefix_size not propagated from run dict to "
            "RASDConfig in run_experiment.py — YAML override would be ignored"
        )
        assert re.search(
            r"kv_block_size_nf4\s*=\s*int\(run\.get\([\"\']kv_block_size_nf4[\"\']",
            RUN_EXP_SRC,
        ), (
            "kv_block_size_nf4 not propagated from run dict to "
            "RASDConfig in run_experiment.py — YAML override would be ignored"
        )

    def test_m4_yamls_set_kv_quant_true(self):
        """The M4 phase-C YAMLs must set kv_quant: true.

        History (2026-05-10):
          1. Original plan: kv_quant=true on 40GB SXM4 (memory lever)
          2. After moving to 80GB SXM4, briefly flipped to kv_quant=false
             reasoning that bf16 KV would fit. THIS WAS WRONG: per-rank
             memory at 512k bf16 hit ~80 GB (hardware cap) and OOMed.
             Ring attention IS sharded but per-rank intermediates scale
             super-linearly with S_local, so even bf16 doesn't fit at 1M.
          3. Final: kv_quant=true with two acceptance-recovery levers:
             (a) block_size=32 (better codec rel_err: ~7% vs ~11%)
             (b) StreamingLLM-style outlier-keep on rank 0 (first 128
                 global tokens stored in bf16; the rest in NF4).

        So the production setting is kv_quant: true. The mitigation
        levers are wired through RASDConfig (kv_block_size_nf4=32,
        kv_outlier_prefix_size=128) and applied at NF4DynamicCache
        construction time in rasd_inference.py:generate(). The codec
        round-trip is still gated by scripts/c11_validation.py.
        """
        for cfg_file in ("configs/m4_phase_c_long_smoke.yml",
                         "configs/m4_final_matrix.yml"):
            text = (REPO_ROOT / cfg_file).read_text()
            assert re.search(r"kv_quant:\s*true", text), (
                f"{cfg_file} should set kv_quant: true under defaults "
                "(see test docstring for the journey)"
            )


# ---------------------------------------------------------------------------
# Layer 7: outlier-keep behavior in NF4DynamicCache (M4 Phase C 2026-05-10)
# ---------------------------------------------------------------------------

class TestNF4OutlierKeep:
    """The bf16-prefix store inside NF4DynamicCache: first N tokens
    bypass NF4 quantization. Used for StreamingLLM-style attention-sink
    preservation on the rank that holds global position 0."""

    def _make_kv(self, S, B=1, H=8, D=128, seed=0):
        torch.manual_seed(seed)
        k = torch.randn(B, H, S, D, dtype=torch.bfloat16)
        v = torch.randn(B, H, S, D, dtype=torch.bfloat16)
        return k, v

    def test_default_prefix_size_is_zero(self):
        """Default constructor (no bf16_prefix_size kwarg) is pure NF4."""
        from src.models.nf4_dynamic_cache import NF4DynamicCache
        cache = NF4DynamicCache()
        assert cache.bf16_prefix_size == 0

    def test_negative_prefix_size_rejected(self):
        from src.models.nf4_dynamic_cache import NF4DynamicCache
        with pytest.raises(ValueError):
            NF4DynamicCache(bf16_prefix_size=-1)

    def test_bf16_prefix_stores_first_n_exactly(self):
        """First `bf16_prefix_size` tokens must round-trip BIT-EXACT
        (not lossy) since they go through bf16 path, not NF4."""
        from src.models.nf4_dynamic_cache import NF4DynamicCache
        cache = NF4DynamicCache(block_size=32, bf16_prefix_size=64)
        k, v = self._make_kv(S=200)
        # update returns the dequantized full (k, v) for the kernel
        k_out, v_out = cache.update(k, v, layer_idx=0)
        assert k_out.shape == k.shape
        # The first 64 tokens must be EXACTLY equal (bf16 prefix path).
        assert torch.equal(k_out[:, :, :64, :], k[:, :, :64, :]), (
            "First 64 tokens should be bit-exact — they bypass NF4"
        )
        assert torch.equal(v_out[:, :, :64, :], v[:, :, :64, :])
        # Tokens past prefix must be close-but-not-equal (NF4 lossy)
        nf4_diff = (k_out[:, :, 64:, :] - k[:, :, 64:, :]).abs().mean().item()
        assert nf4_diff > 0, "NF4 portion should be lossy (got bit-exact)"
        assert nf4_diff < 0.5, f"NF4 rel_err too high: {nf4_diff:.3f}"

    def test_prefix_smaller_than_first_chunk(self):
        """When the first update() exceeds the prefix capacity, only
        the first `bf16_prefix_size` tokens are bf16 — the rest land
        in NF4."""
        from src.models.nf4_dynamic_cache import NF4DynamicCache
        cache = NF4DynamicCache(block_size=32, bf16_prefix_size=8)
        k, v = self._make_kv(S=64)
        cache.update(k, v, layer_idx=0)
        assert cache._k_bf16_prefix[0].shape[2] == 8
        # NF4 codes shape is (B, H, S_chunk, D//2), so axis 2 = remaining
        assert cache._k_codes[0][0].shape[2] == 64 - 8

    def test_prefix_fills_across_multiple_updates(self):
        """If prefill is broken into chunks (decode-style), the prefix
        fills until full, then subsequent updates go to NF4."""
        from src.models.nf4_dynamic_cache import NF4DynamicCache
        cache = NF4DynamicCache(block_size=32, bf16_prefix_size=128)
        k1, v1 = self._make_kv(S=32, seed=1)
        k2, v2 = self._make_kv(S=32, seed=2)
        k3, v3 = self._make_kv(S=128, seed=3)
        cache.update(k1, v1, layer_idx=0)
        cache.update(k2, v2, layer_idx=0)
        cache.update(k3, v3, layer_idx=0)
        # Prefix should hold first 128 tokens (32 + 32 + 64 from chunk 3)
        assert cache._k_bf16_prefix[0].shape[2] == 128
        # NF4 holds the trailing 64 tokens of chunk 3
        nf4_total = sum(c.shape[2] for c in cache._k_codes[0])
        assert nf4_total == 64

    def test_seq_length_includes_prefix(self):
        from src.models.nf4_dynamic_cache import NF4DynamicCache
        cache = NF4DynamicCache(block_size=32, bf16_prefix_size=16)
        k, v = self._make_kv(S=128)
        cache.update(k, v, layer_idx=0)
        assert cache.get_seq_length(0) == 128

    def test_truncate_into_prefix(self):
        """Truncating below the prefix length should keep only the
        first new_seqlen prefix tokens and drop ALL NF4."""
        from src.models.nf4_dynamic_cache import NF4DynamicCache
        cache = NF4DynamicCache(block_size=32, bf16_prefix_size=64)
        k, v = self._make_kv(S=256)
        cache.update(k, v, layer_idx=0)
        cache.truncate(32)
        assert cache.get_seq_length(0) == 32
        assert cache._k_bf16_prefix[0].shape[2] == 32
        assert cache._k_codes[0] == [], "NF4 chunks must be dropped"

    def test_truncate_into_nf4(self):
        """Truncating above the prefix length keeps the full prefix and
        trims the NF4 portion."""
        from src.models.nf4_dynamic_cache import NF4DynamicCache
        cache = NF4DynamicCache(block_size=32, bf16_prefix_size=64)
        k, v = self._make_kv(S=256)
        cache.update(k, v, layer_idx=0)
        cache.truncate(100)
        assert cache.get_seq_length(0) == 100
        assert cache._k_bf16_prefix[0].shape[2] == 64
        assert sum(c.shape[2] for c in cache._k_codes[0]) == 36

    def test_truncate_to_zero(self):
        from src.models.nf4_dynamic_cache import NF4DynamicCache
        cache = NF4DynamicCache(block_size=32, bf16_prefix_size=64)
        k, v = self._make_kv(S=128)
        cache.update(k, v, layer_idx=0)
        cache.truncate(0)
        assert cache.get_seq_length(0) == 0
        assert cache._k_bf16_prefix[0] is None

    def test_serialize_round_trip_preserves_prefix(self):
        """Save+load must preserve both NF4 storage AND bf16 prefix."""
        from src.models.nf4_dynamic_cache import NF4DynamicCache
        cache = NF4DynamicCache(block_size=32, bf16_prefix_size=32)
        k, v = self._make_kv(S=128)
        cache.update(k, v, layer_idx=0)
        d = cache.to_serializable()
        new_cache = NF4DynamicCache.from_serializable(d)
        assert new_cache.bf16_prefix_size == 32
        assert new_cache.get_seq_length(0) == 128
        assert new_cache._k_bf16_prefix[0].shape[2] == 32
        # Dequantized full output must match
        k1, _ = cache._dequantize_layer(0)
        k2, _ = new_cache._dequantize_layer(0)
        assert torch.equal(k1, k2)

    def test_legacy_serialized_payload_loads_as_pure_nf4(self):
        """Backwards compat: a payload from before outlier-keep was
        added (no bf16_prefix_size key) must still load and behave as
        a pure-NF4 cache."""
        from src.models.nf4_dynamic_cache import NF4DynamicCache
        cache = NF4DynamicCache(block_size=32)  # bf16_prefix_size=0 default
        k, v = self._make_kv(S=64)
        cache.update(k, v, layer_idx=0)
        d = cache.to_serializable()
        # Strip the new keys to simulate legacy payload
        d.pop("bf16_prefix_size", None)
        d.pop("k_bf16_prefix", None)
        d.pop("v_bf16_prefix", None)
        new_cache = NF4DynamicCache.from_serializable(d)
        assert new_cache.bf16_prefix_size == 0
        assert all(p is None for p in new_cache._k_bf16_prefix)
        assert new_cache.get_seq_length(0) == 64

    def test_memory_bytes_includes_prefix(self):
        from src.models.nf4_dynamic_cache import NF4DynamicCache
        cache_no_prefix = NF4DynamicCache(block_size=32, bf16_prefix_size=0)
        cache_w_prefix  = NF4DynamicCache(block_size=32, bf16_prefix_size=64)
        k, v = self._make_kv(S=128)
        cache_no_prefix.update(k, v, layer_idx=0)
        cache_w_prefix.update(k, v, layer_idx=0)
        # The prefix variant must be larger (64 tokens × 8 H × 128 D ×
        # 2 (K+V) × 2 bytes = 256 KB more, minus the savings from
        # NOT NF4-quantizing those 64 tokens).
        assert cache_w_prefix.memory_bytes() > cache_no_prefix.memory_bytes()

    def test_chunked_update_default_disabled(self):
        """update_chunk_size defaults to 0 (legacy single-shot path)
        so existing constructors without the kwarg are byte-identical
        to before."""
        from src.models.nf4_dynamic_cache import NF4DynamicCache
        c = NF4DynamicCache()
        assert c.update_chunk_size == 0

    def test_chunked_update_negative_rejected(self):
        from src.models.nf4_dynamic_cache import NF4DynamicCache
        with pytest.raises(ValueError):
            NF4DynamicCache(update_chunk_size=-1)

    def test_chunked_update_produces_multiple_chunks(self):
        """With update_chunk_size=64 and S_new=200, the NF4 portion
        should be split into 4 chunks (64+64+64+8). Each chunk is
        stored as a separate (codes, scales) entry in the layer's list."""
        from src.models.nf4_dynamic_cache import NF4DynamicCache
        c = NF4DynamicCache(block_size=32, update_chunk_size=64)
        torch.manual_seed(0)
        k = torch.randn(1, 8, 200, 128, dtype=torch.bfloat16)
        v = torch.randn(1, 8, 200, 128, dtype=torch.bfloat16)
        c.update(k, v, layer_idx=0)
        assert len(c._k_codes[0]) == 4, (
            f"expected 4 chunks at chunk_size=64 over S=200, got "
            f"{len(c._k_codes[0])}"
        )
        chunk_sizes = [chunk.shape[2] for chunk in c._k_codes[0]]
        assert chunk_sizes == [64, 64, 64, 8]
        # Total stored positions equals S_new
        assert c.get_seq_length(0) == 200

    def test_chunked_update_bit_close_to_unchunked(self):
        """The output of the chunked path must be numerically close
        to the unchunked path (the only difference is per-chunk
        absmax scaling — within the codec's intrinsic rel_err)."""
        from src.models.nf4_dynamic_cache import NF4DynamicCache
        torch.manual_seed(7)
        k = torch.randn(1, 8, 256, 128, dtype=torch.bfloat16)
        v = torch.randn(1, 8, 256, 128, dtype=torch.bfloat16)

        unchunked = NF4DynamicCache(block_size=32, update_chunk_size=0)
        unchunked.update(k.clone(), v.clone(), layer_idx=0)
        k_unchunked, v_unchunked = unchunked._dequantize_layer(0)

        chunked = NF4DynamicCache(block_size=32, update_chunk_size=64)
        chunked.update(k.clone(), v.clone(), layer_idx=0)
        k_chunked, v_chunked = chunked._dequantize_layer(0)

        assert k_chunked.shape == k_unchunked.shape
        # Per-chunk absmax can differ slightly from global absmax,
        # but both are within ~10% rel_err of original. Allow 2%
        # difference between paths themselves.
        diff_k = (k_chunked - k_unchunked).abs().mean().item()
        norm_k = k_unchunked.abs().mean().item()
        assert diff_k / max(norm_k, 1e-9) < 0.05, (
            f"chunked vs unchunked rel diff={diff_k/norm_k:.4f}, expected <0.05"
        )

    def test_chunked_update_with_outlier_prefix(self):
        """Outlier-keep + chunked update must compose: prefix takes
        the first N tokens (bf16 exact), the remaining S_new - N
        tokens are chunked into NF4 codes."""
        from src.models.nf4_dynamic_cache import NF4DynamicCache
        c = NF4DynamicCache(
            block_size=32,
            bf16_prefix_size=32,
            update_chunk_size=64,
        )
        torch.manual_seed(11)
        k = torch.randn(1, 8, 200, 128, dtype=torch.bfloat16)
        v = torch.randn(1, 8, 200, 128, dtype=torch.bfloat16)
        c.update(k, v, layer_idx=0)
        # 32 in bf16 prefix + 168 in NF4 chunks of size 64
        assert c._k_bf16_prefix[0].shape[2] == 32
        # NF4 chunks: 64 + 64 + 40 = 168
        chunk_sizes = [chunk.shape[2] for chunk in c._k_codes[0]]
        assert chunk_sizes == [64, 64, 40]
        # Round-trip exact bf16 prefix
        k_full, _ = c._dequantize_layer(0)
        assert torch.equal(k_full[:, :, :32, :], k[:, :, :32, :])

    def test_run_experiment_propagates_nf4_chunk_size(self):
        """run_experiment.py must wire nf4_update_chunk_size from the
        run dict into RASDConfig."""
        assert re.search(
            r"nf4_update_chunk_size\s*=\s*int\(run\.get\([\"\']nf4_update_chunk_size[\"\']",
            RUN_EXP_SRC,
        ), (
            "nf4_update_chunk_size not propagated from run dict to "
            "RASDConfig in run_experiment.py — YAML override would be ignored"
        )

    def test_rank0_only_construction_in_generate(self):
        """generate() must construct outlier-keep ONLY for rank 0; other
        ranks get pure NF4 (since they don't hold global position 0)."""
        assert re.search(
            r"prefix_size\s*=\s*\(\s*cfg\.kv_outlier_prefix_size\s+if\s+self\._rank\s*==\s*0\s+else\s+0",
            RASD_INF_SRC,
        ), (
            "rasd_inference.generate() should set bf16_prefix_size only "
            "on rank 0; other ranks should pass 0 (pure NF4)"
        )

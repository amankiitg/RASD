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
        # Look for the gated construction
        assert re.search(
            r"if cfg\.kv_quant:[\s\S]{0,200}NF4DynamicCache\(",
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

    def test_m4_yamls_set_kv_quant_false(self):
        """The M4 phase-C YAMLs must set kv_quant: false.

        Background (2026-05-10): the original M4 plan ran kv_quant=True
        (NF4 KV-cache) as a 40GB-class memory lever. After moving to the
        80GB SXM4 instance for the 1M-context headline run, NF4 is no
        longer needed (per-rank budget at 1M is ~30-35 GB / 80 GB with
        bf16 KV) and the simple absmax NF4 codec was costing ~5-10 pts
        of acceptance vs the bf16 baseline (real K/V rel_err ~11% vs
        KIVI's 3-5% — KIVI requires double-quant which we don't ship).

        So the production setting is kv_quant: false; the codec itself
        is still validated end-to-end by scripts/c11_validation.py
        (which builds an NF4DynamicCache directly), but the
        long-context smoke + final matrix YAMLs run bf16 KV.
        """
        for cfg_file in ("configs/m4_phase_c_long_smoke.yml",
                         "configs/m4_final_matrix.yml"):
            text = (REPO_ROOT / cfg_file).read_text()
            assert re.search(r"kv_quant:\s*false", text), (
                f"{cfg_file} should set kv_quant: false under defaults "
                "(see test docstring for rationale)"
            )

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
# Layer 4: forward gate
# ---------------------------------------------------------------------------

class TestForwardGate:
    def test_forward_branches_on_ring_kv_quant(self):
        """The forward must call _kv_quant_round_trip only when
        _ring_kv_quant is True."""
        # Search for the call site
        assert "_kv_quant_round_trip(k_full, v_full)" in RING_PATCH_SRC, (
            "C11 regression: _kv_quant_round_trip not invoked from forward"
        )

    def test_round_trip_gated_by_ring_kv_quant(self):
        """The round-trip call must be inside `if getattr(self, "_ring_kv_quant", False):`
        so default-off (no attribute set) means M3-byte-identical."""
        assert re.search(
            r'if getattr\(self,\s*[\"\']_ring_kv_quant[\"\'],\s*False\)\s*:',
            RING_PATCH_SRC,
        ), (
            "C11 regression: kv_quant branch not gated by getattr default-False; "
            "could trigger spuriously if attribute missing"
        )

    def test_round_trip_after_cache_append(self):
        """The round-trip must come AFTER _cache_append returns the full
        tensors — round-tripping the new k/v BEFORE cache append would
        produce wrong attention because the cached prefill would still
        be exact-bf16."""
        idx_cache_append = RING_PATCH_SRC.find("k_full, v_full = _cache_append")
        idx_round_trip = RING_PATCH_SRC.find("_kv_quant_round_trip(k_full, v_full)")
        assert idx_cache_append > 0
        assert idx_round_trip > 0
        assert idx_round_trip > idx_cache_append, (
            "C11 regression: kv_quant round-trip placed BEFORE _cache_append — "
            "would attend on quantized-new-k/v but exact-cached prefill"
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

    def test_m4_yamls_set_kv_quant_true(self):
        """The M4 phase-C YAMLs must enable kv_quant (Phase C validation
        of the codec on real activations)."""
        for cfg_file in ("configs/m4_phase_c_long_smoke.yml",
                         "configs/m4_final_matrix.yml"):
            text = (REPO_ROOT / cfg_file).read_text()
            assert re.search(r"kv_quant:\s*true", text), (
                f"{cfg_file} should set kv_quant: true under defaults"
            )

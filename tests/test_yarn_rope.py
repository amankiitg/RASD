"""Tests for C2b — YaRN RoPE config wiring.

The numeric correctness of YaRN at factor=256 needs CUDA + a real
forward pass to verify (no NaN / inf, no quality collapse). That's a
pod-side validation gate. Locally we test the **wiring**:

- `_build_rope_scaling_dict(rope_type, factor, native_max)` returns
  the canonical key set transformers ≥ 4.42 expects for each strategy
- `RASDConfig.rope_type` defaults to "linear" (M3 byte-identical)
- The full `_build_hf_config` end-to-end honors rope_type and produces
  the right `hf_cfg.rope_scaling` dict
- factor=256 (1M target over Llama-2's 4k) doesn't crash
"""
from __future__ import annotations

import re
from pathlib import Path
from unittest.mock import patch

import pytest

from src.models.rasd_inference import (
    RASDConfig,
    RASDInference,
    _build_rope_scaling_dict,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
RASD_INF_SRC = (REPO_ROOT / "src" / "models" / "rasd_inference.py").read_text()
RUN_EXP_SRC = (REPO_ROOT / "run_experiment.py").read_text()


# ---------------------------------------------------------------------------
# _build_rope_scaling_dict — pure helper
# ---------------------------------------------------------------------------

class TestBuildRopeScalingDict:
    def test_linear_default(self):
        d = _build_rope_scaling_dict("linear", factor=16.0, native_max=4096)
        assert d == {"type": "linear", "factor": 16.0}

    def test_linear_lowercase_independent(self):
        a = _build_rope_scaling_dict("linear", 16.0, 4096)
        b = _build_rope_scaling_dict("LINEAR", 16.0, 4096)
        c = _build_rope_scaling_dict("Linear", 16.0, 4096)
        assert a == b == c

    def test_dynamic_ntk(self):
        d = _build_rope_scaling_dict("dynamic", factor=16.0, native_max=4096)
        assert d == {"type": "dynamic", "factor": 16.0}

    def test_yarn_minimal_keys(self):
        """YaRN dict must include type, factor, and original_max_position_embeddings."""
        d = _build_rope_scaling_dict("yarn", factor=16.0, native_max=4096)
        assert d["type"] == "yarn"
        assert d["factor"] == 16.0
        assert d["original_max_position_embeddings"] == 4096

    def test_yarn_factor_256_for_1M(self):
        """1M context over Llama-2-7B's 4k native max. factor=256.
        YaRN should accept factor=256 without crashing."""
        d = _build_rope_scaling_dict("yarn", factor=256.0, native_max=4096)
        assert d["type"] == "yarn"
        assert d["factor"] == 256.0
        assert d["original_max_position_embeddings"] == 4096

    def test_factor_coerced_to_float(self):
        """factor as int input must come out as float (transformers
        validation expects float)."""
        d = _build_rope_scaling_dict("linear", factor=16, native_max=4096)
        assert isinstance(d["factor"], float)

    def test_native_max_coerced_to_int(self):
        d = _build_rope_scaling_dict("yarn", factor=16.0, native_max=4096.0)
        assert isinstance(d["original_max_position_embeddings"], int)

    def test_unknown_rope_type_raises(self):
        with pytest.raises(ValueError, match="Unknown rope_type"):
            _build_rope_scaling_dict("longrope", factor=16.0, native_max=4096)
        with pytest.raises(ValueError, match="Unknown rope_type"):
            _build_rope_scaling_dict("", factor=16.0, native_max=4096)


# ---------------------------------------------------------------------------
# RASDConfig defaults
# ---------------------------------------------------------------------------

class TestConfigDefaults:
    def test_rope_type_default_linear(self):
        """M3 invariant: rope_type defaults to "linear" so M3 replay
        is byte-identical when the field isn't set."""
        cfg = RASDConfig()
        assert cfg.rope_type == "linear"

    def test_rope_type_overridable(self):
        cfg = RASDConfig(rope_type="yarn")
        assert cfg.rope_type == "yarn"


# ---------------------------------------------------------------------------
# _build_hf_config integration
# ---------------------------------------------------------------------------

class TestBuildHfConfigIntegration:
    @pytest.fixture
    def fake_llama_cfg(self):
        from transformers import LlamaConfig
        return LlamaConfig(max_position_embeddings=4096)

    def _build(self, rope_type, ctx, fake_cfg):
        with patch("transformers.AutoConfig.from_pretrained",
                   return_value=fake_cfg):
            inst = RASDInference.__new__(RASDInference)
            return inst._build_hf_config(
                model_name="ignored",
                revision=None,
                context_length=ctx,
                label="target",
                apply_rope_scaling=True,
                rope_type=rope_type,
            )

    def test_linear_at_64k(self, fake_llama_cfg):
        result = self._build("linear", 65536, fake_llama_cfg)
        assert result.rope_scaling == {"type": "linear", "factor": 16.0}
        assert result.max_position_embeddings == 65536

    def test_yarn_at_64k(self, fake_llama_cfg):
        result = self._build("yarn", 65536, fake_llama_cfg)
        assert result.rope_scaling["type"] == "yarn"
        assert result.rope_scaling["factor"] == 16.0
        assert result.rope_scaling["original_max_position_embeddings"] == 4096
        assert result.max_position_embeddings == 65536

    def test_yarn_at_1M(self, fake_llama_cfg):
        """factor=256 — the 1M target case. Must not crash."""
        result = self._build("yarn", 1_048_576, fake_llama_cfg)
        assert result.rope_scaling["type"] == "yarn"
        assert result.rope_scaling["factor"] == 256.0
        assert result.max_position_embeddings == 1_048_576

    def test_dynamic_at_64k(self, fake_llama_cfg):
        result = self._build("dynamic", 65536, fake_llama_cfg)
        assert result.rope_scaling == {"type": "dynamic", "factor": 16.0}

    def test_default_rope_type_kwarg_is_linear(self, fake_llama_cfg):
        """If a caller invokes _build_hf_config without rope_type, they
        get linear (M3 byte-identical behavior)."""
        with patch("transformers.AutoConfig.from_pretrained",
                   return_value=fake_llama_cfg):
            inst = RASDInference.__new__(RASDInference)
            result = inst._build_hf_config(
                model_name="ignored", revision=None,
                context_length=65536, label="target",
                apply_rope_scaling=True,
                # rope_type omitted on purpose
            )
        assert result.rope_scaling["type"] == "linear"

    def test_draft_skip_unaffected_by_rope_type(self, fake_llama_cfg):
        """Option B (apply_rope_scaling=False) must still skip scaling
        regardless of rope_type — saves the ~11 GB/rank draft KV blow-up."""
        for rt in ("linear", "yarn", "dynamic"):
            with patch("transformers.AutoConfig.from_pretrained",
                       return_value=fake_llama_cfg):
                inst = RASDInference.__new__(RASDInference)
                result = inst._build_hf_config(
                    model_name="ignored", revision=None,
                    context_length=65536, label="draft",
                    apply_rope_scaling=False, rope_type=rt,
                )
            rs = getattr(result, "rope_scaling", None) or {}
            assert rs.get("type") != "yarn", f"rope_type={rt} leaked into draft"
            assert rs.get("type") != "dynamic", f"rope_type={rt} leaked into draft"
            assert "factor" not in rs, f"rope_type={rt} leaked factor into draft"
            assert result.max_position_embeddings == 4096


# ---------------------------------------------------------------------------
# run_experiment.py wiring
# ---------------------------------------------------------------------------

class TestRunExperimentWiring:
    def test_rope_type_propagated_to_rasd_config(self):
        """run_experiment._run_single_worker must pass rope_type from the
        run dict (default "linear") into RASDConfig."""
        assert re.search(
            r'rope_type\s*=\s*str\(run\.get\([\"\']rope_type[\"\'],\s*[\"\']linear[\"\']\)\)',
            RUN_EXP_SRC,
        ), (
            "C2b regression: rope_type not propagated from run dict to "
            "RASDConfig in _run_single_worker"
        )

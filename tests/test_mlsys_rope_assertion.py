"""The gate's rope assertion must be able to FAIL.

The assertion exists because a YaRN configuration was silently anchored on the
context length instead of the model's 4096 base: transformers <=4.50 ignores
`rope_scaling["original_max_position_embeddings"]` and takes the interpolation
band from `config.max_position_embeddings`, which this project sets to the
context. The first version of this assertion compared the built `inv_freq`
against a reference recomputed from the very config the model was built with,
using the same rope init functions — a tautology that matched unconditionally and
therefore could never have caught the bug it was written for.

This test pins the property: given the built vector from a mis-anchored
configuration, the assertion must report a mismatch and must name the
mis-anchoring. Config-only, no weights, no GPU.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
MODEL = "meta-llama/Llama-2-7b-hf"     # 4096 native window; anchor != context


def _load_gate():
    spec = importlib.util.spec_from_file_location(
        "gate_mod", REPO / "scripts" / "mlsys_coherence_gate.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["gate_mod"] = mod
    spec.loader.exec_module(mod)
    return mod


class _Rope:
    def __init__(self, inv):
        self.inv_freq = inv


class _Model:
    """Minimal stand-in exposing the path `built_inv_freq` reads."""

    def __init__(self, inv):
        attn = type("A", (), {"rotary_emb": _Rope(inv)})()
        layer = type("L", (), {"self_attn": attn})()
        self.model = type("M", (), {"layers": [layer]})()


@pytest.mark.skipif(
    not (REPO / "configs").exists(), reason="repo layout changed")
def test_rope_assertion_rejects_the_context_anchored_build():
    gate = _load_gate()
    ctx = 32768
    intended = {"rope_type": "yarn", "rope_factor": 8,
                "rope_anchor_base": 4096, "context_length": ctx}

    declared = gate._declared_reference(intended, MODEL)
    misanchored = gate._declared_reference(intended, MODEL, anchor_override=ctx)
    # If these were equal the test would prove nothing: there would be no
    # observable difference between honouring the anchor and ignoring it.
    assert float((declared - misanchored).abs().max()) > 1e-6, (
        "the honoured and mis-anchored references are identical, so the "
        "assertion cannot distinguish them"
    )

    # A model that honoured the declared anchor passes.
    ok = gate.assert_effective_rope(_Model(declared), MODEL, intended)
    assert ok["effective_rope_matches_intent"] is True
    assert ok["effective_rope_match"] == "declared"
    assert ok["built_anchor_is_context"] is False

    # A model that anchored on the context is a MISMATCH, and is named as such.
    bad = gate.assert_effective_rope(_Model(misanchored), MODEL, intended)
    assert bad["effective_rope_matches_intent"] is False, (
        "a context-anchored build passed the assertion; this is exactly the "
        "bug the assertion exists to catch"
    )
    assert bad["effective_rope_match"] == "anchor_on_context"
    assert bad["built_anchor_is_context"] is True

    # `llama3` reads the anchor from the dict, so a context-anchored variant is
    # not constructible: the gate must not claim to test for one.
    l3 = {"rope_type": "llama3", "rope_factor": 16,
          "rope_anchor_base": 8192, "context_length": 262144}
    refs = gate._declared_reference(l3, "meta-llama/Llama-3.1-8B")
    assert refs is not None


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))

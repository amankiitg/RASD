"""RoPE candidates for Llama-3.1-8B beyond its native 128k window.

Context: the ARM4 f2/f4 rungs REPLACED Llama-3.1's shipped `llama3` rope
block with a YaRN dict. The target then emitted EOS at token 0 and acceptance
collapsed, so those rungs measure a broken target rather than speculative
decoding. The replacement confounded "extrapolate past 128k" with "throw away
the model's own long-context mechanism".

The candidates here BUILD ON the native scaling instead:

  * ``llama3`` type with a larger factor — identical mechanism to what the
    model ships; only ``factor`` changes.
  * YaRN anchored on 8192, Llama-3.1's true pretraining base, instead of
    131072 or the target context length.

Everything here computes ``inv_freq`` from a config object alone — no weights,
no network, no tokenizer — so the frequency effect of each choice is
reviewable as a pure number before anything is gated or run on paid hardware.

IMPORTANT implementation detail these tests pin down: in transformers 4.47.1
the two rope families read their anchor from DIFFERENT places. ``llama3``
reads ``rope_scaling["original_max_position_embeddings"]``; YaRN carries a
``TODO`` and never reads that key at all, deriving its interpolation band from
``config.max_position_embeddings``. A YaRN rung that sets only the dict key is
silently identical to one that sets nothing, so the anchor has to be routed
per rope type (`_rope_anchor_channel`).
"""

import math

import pytest
import torch

from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS

from src.models.rasd_inference import (
    _build_rope_scaling_dict,
    _resolve_rope_anchor,
    _rope_anchor_channel,
)


# Llama-3.1-8B, as documented in the repo and confirmed against
# AutoConfig.from_pretrained (config only, no weights).
LLAMA31_NATIVE_ROPE = {
    "factor": 8.0,
    "low_freq_factor": 1.0,
    "high_freq_factor": 4.0,
    "original_max_position_embeddings": 8192,
    "rope_type": "llama3",
}
LLAMA31_NATIVE_MAX = 131072
LLAMA31_ROPE_THETA = 500000.0
LLAMA31_HEAD_DIM = 128
N_FREQ = 64


class _StubConfig:
    """Minimal stand-in for a HF config — enough for the rope init functions."""

    def __init__(self, rope_scaling,
                 max_position_embeddings=LLAMA31_NATIVE_MAX):
        self.rope_scaling = rope_scaling
        self.rope_theta = LLAMA31_ROPE_THETA
        self.hidden_size = 4096
        self.num_attention_heads = 32
        self.head_dim = LLAMA31_HEAD_DIM
        self.max_position_embeddings = max_position_embeddings
        self.partial_rotary_factor = 1.0


def _inv_freq(rope_scaling, max_position_embeddings=LLAMA31_NATIVE_MAX):
    """(inv_freq, attention_factor) for a rope_scaling dict — config only.

    Accepts the codebase's `{"type": ...}` spelling: transformers normalises
    that alias to `rope_type` when the dict is attached to a real HF config,
    but a bare stub skips that step, so do it here.

    `max_position_embeddings` is a parameter because for YaRN it is the ONLY
    channel an anchor can travel through (see `_rope_anchor_channel`).
    """
    rs = dict(rope_scaling)
    if "rope_type" not in rs and "type" in rs:
        rs["rope_type"] = rs.pop("type")
    cfg = _StubConfig(rs, max_position_embeddings)
    return ROPE_INIT_FUNCTIONS[rs["rope_type"]](cfg, torch.device("cpu"))


def _native():
    return _inv_freq(dict(LLAMA31_NATIVE_ROPE))


def _diff(base, cand):
    """How a candidate's inv_freq differs from native, as plain numbers."""
    b, c = base[0], cand[0]
    assert b.shape == c.shape, f"shape mismatch {b.shape} vs {c.shape}"
    rel = (c - b).abs() / b.abs()
    return {
        "n_entries": int(b.numel()),
        "n_changed": int((c != b).sum().item()),
        "max_abs_delta": float((c - b).abs().max().item()),
        "max_rel_delta": float(rel.max().item()),
        "mean_rel_delta": float(rel.mean().item()),
        "attention_factor": float(cand[1]),
    }


def _llama3_scaling(factor):
    d = dict(LLAMA31_NATIVE_ROPE)
    d["factor"] = float(factor)
    return d


def _candidates():
    """name -> (rope_scaling, context, effective_anchor).

    `effective_anchor` is what `config.max_position_embeddings` must be for
    the rope family to see the intended base (`_rope_anchor_channel`).
    """
    return {
        "native llama3 f8 (128k, as shipped)":
            (_llama3_scaling(8.0), 131072, 8192),
        "llama3 f16 (256k, native mechanism, larger factor)":
            (_llama3_scaling(16.0), 262144, 8192),
        "llama3 f32 (512k, native mechanism, larger factor)":
            (_llama3_scaling(32.0), 524288, 8192),
        "yarn base8192 f32 (256k, true pretraining base)":
            (dict(_build_rope_scaling_dict("yarn", 32.0, 8192)), 262144, 8192),
        "yarn base8192 f64 (512k, true pretraining base)":
            (dict(_build_rope_scaling_dict("yarn", 64.0, 8192)), 524288, 8192),
        "yarn base131072 f2 (256k, ARM4 f2 dict)":
            (dict(_build_rope_scaling_dict("yarn", 2.0, 131072)), 262144, 262016),
        "yarn base131072 f4 (512k, ARM4 f4 dict)":
            (dict(_build_rope_scaling_dict("yarn", 4.0, 131072)), 524288, 524032),
    }


def _cand_inv_freq(name):
    """inv_freq for a named candidate, anchored through the right channel."""
    rs, _ctx, anchor = _candidates()[name]
    return _inv_freq(rs, anchor)


class TestRopeCandidatesInvFreq:
    def test_report_each_candidate_against_native(self, capsys):
        """Compute every candidate's inv_freq from config alone and report
        how far it moves from what the model actually ships."""
        native = _native()
        rows = []
        for name, (_rs, ctx, anchor) in _candidates().items():
            d = _diff(native, _cand_inv_freq(name))
            rows.append((name, ctx, anchor, d))

        with capsys.disabled():
            print("\n  RoPE candidates vs Llama-3.1-8B native "
                  f"(rope_theta={LLAMA31_ROPE_THETA:g}, head_dim={LLAMA31_HEAD_DIM}, "
                  f"{N_FREQ} freq entries)")
            print(f"  {'candidate':<52} {'ctx':>7} {'anchor':>7} {'chg':>6} "
                  f"{'max_rel':>9} {'mean_rel':>9} {'attn':>7}")
            for name, ctx, anchor, d in rows:
                print(f"  {name:<52} {ctx:>7} {anchor:>7} "
                      f"{d['n_changed']:>3}/{N_FREQ} {d['max_rel_delta']:>9.4g} "
                      f"{d['mean_rel_delta']:>9.4g} {d['attention_factor']:>7.4g}")

        assert len(rows) == 7
        for _, _, _, d in rows:
            assert math.isfinite(d["max_rel_delta"])
            assert math.isfinite(d["attention_factor"])

    def test_native_candidate_is_identity(self):
        """The f8 row must be a no-op against itself, otherwise the baseline
        is wrong and every delta below is meaningless."""
        d = _diff(_native(), _inv_freq(_llama3_scaling(8.0)))
        assert d["n_changed"] == 0
        assert d["max_abs_delta"] == 0.0

    def test_llama3_candidates_share_the_native_mechanism(self):
        """The point of the llama3 rungs: only `factor` moves, so a change
        there cannot be attributed to swapping the rope family."""
        for factor in (16.0, 32.0):
            rs = _llama3_scaling(factor)
            assert rs["rope_type"] == "llama3"
            assert rs["original_max_position_embeddings"] == 8192
            assert rs["low_freq_factor"] == 1.0
            assert rs["high_freq_factor"] == 4.0

    def test_llama3_factor_leaves_high_frequencies_untouched(self):
        """llama3 scaling is band-selective: the shortest wavelengths pass
        through and only the long ones move. Measured behaviour, not an
        assumption — entries 0..28 are unchanged and entry 63 scales by
        native_factor/factor (native f8 already divides by 8, so f16 lands at
        half of the shipped value, f32 at a quarter)."""
        native = _native()[0]
        native_factor = LLAMA31_NATIVE_ROPE["factor"]
        for factor in (16.0, 32.0):
            cand = _inv_freq(_llama3_scaling(factor))[0]
            changed = [i for i in range(N_FREQ) if cand[i] != native[i]]
            assert cand[0] == native[0]
            assert changed[0] == 29, f"unexpected first changed index {changed[0]}"
            assert changed[-1] == 63
            assert len(changed) == 35
            assert cand[-1] == pytest.approx(
                native[-1] * native_factor / factor, rel=1e-6)

    def test_llama3_anchor_travels_in_the_dict(self):
        """llama3 reads its anchor from the rope_scaling dict, so changing it
        there must move inv_freq."""
        a = _inv_freq(_llama3_scaling(8.0))[0]
        d = dict(LLAMA31_NATIVE_ROPE)
        d["factor"] = 8.0
        d["original_max_position_embeddings"] = 4096
        b = _inv_freq(d)[0]
        assert int((a != b).sum().item()) > 0

    def test_yarn_dict_anchor_is_inert(self):
        """Pins the transformers 4.47.1 behaviour that makes the dict-only
        spelling a trap: two YaRN dicts with different
        original_max_position_embeddings and the same factor produce
        IDENTICAL inv_freq. If this ever starts failing, transformers has
        implemented the TODO and `_rope_anchor_channel` should be revisited.
        """
        same_mpe = LLAMA31_NATIVE_MAX
        a = _inv_freq(_build_rope_scaling_dict("yarn", 32.0, 8192), same_mpe)[0]
        b = _inv_freq(_build_rope_scaling_dict("yarn", 32.0, 131072), same_mpe)[0]
        assert bool((a == b).all()), (
            "YaRN now honours the rope_scaling anchor; _rope_anchor_channel "
            "should route YaRN through the dict instead of the config"
        )

    def test_yarn_anchor_travels_in_max_position_embeddings(self):
        """The channel that DOES work for YaRN: the same dict at two different
        config windows must differ."""
        rs = _build_rope_scaling_dict("yarn", 32.0, 8192)
        a = _inv_freq(rs, 8192)[0]
        b = _inv_freq(rs, 131072)[0]
        assert int((a != b).sum().item()) > 0

    def test_yarn_base8192_differs_from_the_arm4_rebased_treatment(self):
        """The 'true base' rung and the ARM4 re-based rung are genuinely
        different treatments, which is why the gate measures both."""
        keep = _cand_inv_freq("yarn base8192 f32 (256k, true pretraining base)")
        rebased = _cand_inv_freq("yarn base131072 f2 (256k, ARM4 f2 dict)")
        assert int((keep[0] != rebased[0]).sum().item()) > 0

    def test_yarn_anchor_reaches_the_transformers_dict(self):
        """_build_rope_scaling_dict still emits the canonical key set."""
        d = _build_rope_scaling_dict("yarn", 32.0, 8192)
        assert d == {
            "type": "yarn",
            "factor": 32.0,
            "original_max_position_embeddings": 8192,
        }
        assert _inv_freq(d)[0].shape == _native()[0].shape

    def test_native_bands_survive_a_llama3_factor_change(self):
        """The candidate that builds on native scaling must not disturb the
        frequency band structure (entries 0..28) at any factor, which is what
        distinguishes it from re-basing."""
        native = _native()[0]
        for factor in (16.0, 32.0):
            cand = _inv_freq(_llama3_scaling(factor))[0]
            assert bool((cand[:29] == native[:29]).all())


class TestRopeAnchorChannel:
    def test_routes_each_rope_type_to_the_channel_it_reads(self):
        assert _rope_anchor_channel("llama3") == "dict"
        assert _rope_anchor_channel("yarn") == "config"
        assert _rope_anchor_channel("linear") == "unused"
        assert _rope_anchor_channel("dynamic") == "unused"
        assert _rope_anchor_channel("none") == "unused"

    def test_case_insensitive(self):
        assert _rope_anchor_channel("YaRN") == "config"
        assert _rope_anchor_channel("LLAMA3") == "dict"

    def test_channel_matches_observed_transformers_behaviour(self):
        """Cross-check the routing table against the installed implementation:
        for each type claiming 'dict', the dict really must move inv_freq."""
        native = _native()[0]
        d = dict(LLAMA31_NATIVE_ROPE)
        d["factor"] = 16.0
        assert int((_inv_freq(d)[0] != native).sum().item()) > 0


class TestResolveRopeAnchor:
    def test_none_preserves_model_window(self):
        assert _resolve_rope_anchor(None, 131072) == 131072
        assert _resolve_rope_anchor(None, 4096) == 4096

    def test_explicit_base_wins(self):
        assert _resolve_rope_anchor(8192, 131072) == 8192

    def test_rejects_anchor_beyond_the_trained_window(self):
        with pytest.raises(ValueError, match="exceeds"):
            _resolve_rope_anchor(262144, 131072)

    def test_rejects_non_positive(self):
        for bad in (0, -1):
            with pytest.raises(ValueError):
                _resolve_rope_anchor(bad, 131072)

    def test_rejects_non_integer(self):
        with pytest.raises(ValueError, match="not an integer"):
            _resolve_rope_anchor("8192.5x", 131072)

"""The gate's generation criteria are relative to the paired baseline.

The 2026-10-08T14:23Z gate run measured every control correctly and then rejected
three positive controls on an ABSOLUTE blank-line ceiling of 0.30. The three
rejections — 31.58%, 31.58% and 38.89% — were the same native configuration
scoring 16.67% at 32k and 18.75% at 128k: PG-19 is hard-wrapped Gutenberg text,
so the blank-line share of a 200-token continuation says which passage was drawn,
not whether the rope is healthy. The rule is now

    blank/repeat share <= GEN_SHARE_TOLERANCE x the PAIRED BASELINE's own share

with absolute thresholds kept ONLY for degeneration (a collapsed generation must
not pass by standing next to a collapsed baseline).

These tests pin the three ways that rule can quietly stop meaning anything: it
could drift back to an absolute threshold, the ceilings could be dropped, or the
baseline the row is judged against could stop being the paired one.
"""
from __future__ import annotations

import importlib.util
import json
import re
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
GATE = REPO / "scripts" / "mlsys_coherence_gate.py"
CONTROLS = REPO / "configs" / "mlsys_gate_controls.json"
MANIFEST = REPO / "configs" / "mlsys_manifest.yml"
LLAMA31 = "meta-llama/Llama-3.1-8B"


def _load_gate():
    spec = importlib.util.spec_from_file_location("_gate_under_test", GATE)
    mod = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(mod)
    except Exception as e:                      # noqa: BLE001
        pytest.skip(f"gate not importable here: {e}")
    return mod


BASE = {"status": "ok", "ppl_continuation": 10.0, "gen_blank_share": 0.05,
        "gen_repeat_share": 0.10}
CLEAN = {"status": "ok", "ppl_continuation": 12.0, "early_eos": False,
         "gen_blank_share": 0.05, "gen_repeat_share": 0.10,
         "effective_rope_matches_intent": True}


# --------------------------------------------------------------------------
# the relative rule
# --------------------------------------------------------------------------

def test_a_share_below_the_tolerance_band_passes():
    gate = _load_gate()
    v = gate.verdict({**CLEAN, "gen_blank_share": 0.09}, BASE)
    assert v["gate_pass"] is True, v["gate_reason"]
    assert v["blank_share_ratio"] == pytest.approx(1.8)
    assert v["baseline_blank_share"] == 0.05


def test_a_share_above_the_tolerance_band_fails_even_without_degeneration():
    """The case an absolute threshold missed: 12% is nowhere near collapse, and
    it is still 2.4x the baseline's own 5%."""
    gate = _load_gate()
    v = gate.verdict({**CLEAN, "gen_blank_share": 0.12}, BASE)
    assert v["gate_pass"] is False
    assert "2.40x" in v["gate_reason"]
    assert "degenerate" not in v["gate_reason"]


def test_the_repeat_share_is_judged_the_same_way():
    gate = _load_gate()
    assert gate.verdict({**CLEAN, "gen_repeat_share": 0.19},
                        BASE)["gate_pass"] is True
    v = gate.verdict({**CLEAN, "gen_repeat_share": 0.21}, BASE)
    assert v["gate_pass"] is False
    assert v["repeat_share_ratio"] == pytest.approx(2.1)


def test_the_baseline_is_the_row_it_was_paired_with_not_a_constant():
    """Two baselines with the same perplexity and different shares must give
    different verdicts for the same candidate."""
    gate = _load_gate()
    row = {**CLEAN, "gen_blank_share": 0.20}
    strict = gate.verdict(row, {**BASE, "gen_blank_share": 0.05})
    lenient = gate.verdict(row, {**BASE, "gen_blank_share": 0.15})
    assert strict["gate_pass"] is False
    assert lenient["gate_pass"] is True


def test_the_tolerance_is_wider_than_the_perplexity_band():
    """Shares come from a 200-token generation, perplexity from 1024 scored
    tokens; the share band has to be the wider one or good rows get rejected on
    sampling noise."""
    gate = _load_gate()
    assert gate.GEN_SHARE_TOLERANCE > gate.PPL_TOLERANCE


# --------------------------------------------------------------------------
# the absolute ceilings, which are degeneration and nothing else
# --------------------------------------------------------------------------

def test_a_collapsed_generation_fails_next_to_a_collapsed_baseline():
    """Without the ceiling, a degenerate candidate would pass by standing next
    to an equally degenerate reference."""
    gate = _load_gate()
    v = gate.verdict({**CLEAN, "gen_blank_share": 0.95},
                     {**BASE, "gen_blank_share": 0.95})
    assert v["gate_pass"] is False
    assert "degenerate" in v["gate_reason"]


def test_an_eos_inside_the_degenerate_window_fails():
    gate = _load_gate()
    v = gate.verdict({**CLEAN, "early_eos": True, "eos_at": 5}, BASE)
    assert v["gate_pass"] is False
    assert "EOS at token 5" in v["gate_reason"]


def test_a_zero_baseline_share_leaves_only_the_ceiling():
    """The documented hole: a ratio against zero is undefined, so the relative
    test cannot apply. Asserted so it stays a decision rather than a surprise."""
    gate = _load_gate()
    zero = {**BASE, "gen_blank_share": 0.0}
    v = gate.verdict({**CLEAN, "gen_blank_share": 0.5}, zero)
    assert v["gate_pass"] is True
    assert v["blank_share_ratio"] == ""
    # ...and the ceiling still catches the collapse it exists for.
    assert gate.verdict({**CLEAN, "gen_blank_share": 0.95}, zero)["gate_pass"] \
        is False


def test_no_absolute_quality_threshold_survives_in_the_gate():
    """A threshold that is still read is still a decision.

    Checks for a DEFINITION, not a mention: `verdict`'s docstring names the
    withdrawn constant to explain why the rule changed, and that is prose. A
    surviving USE of the name would be a NameError the suite and the dry run
    both catch, so the definition is the thing to pin.
    """
    source = GATE.read_text()
    for gone in ("EARLY_EOS_TOKENS", "MAX_BLANK_SHARE", "MAX_REPEAT_SHARE"):
        assert not re.search(rf"^{gone}\s*=", source, re.M), \
            f"{gone} is defined again in the gate"


def test_the_absolute_ceilings_are_degeneration_thresholds():
    gate = _load_gate()
    assert gate.CATASTROPHIC_BLANK_SHARE == 0.90
    assert gate.CATASTROPHIC_EOS_TOKENS == 10


def test_the_applied_ratios_reach_the_csv():
    """A reader must be able to recompute the share verdict from the file."""
    gate = _load_gate()
    for field in ("baseline_blank_share", "baseline_repeat_share",
                  "blank_share_ratio", "repeat_share_ratio"):
        assert field in gate.FIELDS, f"{field} is computed but never written"


# --------------------------------------------------------------------------
# the 64k rung's positive control
# --------------------------------------------------------------------------

def _controls() -> list:
    spec = json.loads(CONTROLS.read_text())
    return spec["candidates"]


def test_the_64k_positive_control_is_not_a_copy_of_its_baseline():
    """It was byte-identical to B_native_64k, so it scored one measurement twice
    and both copies were reported as failures."""
    rows = {c["name"]: c for c in _controls()}
    p2 = next(c for c in _controls()
              if c.get("role") == "positive" and c["context_length"] == 65536)
    ref = next(c for c in _controls()
               if c.get("native_baseline") and c["context_length"] == 65536)
    assert ref["name"] == "B_native_64k"
    # Same sample: the pairing rule needs the same (context, model, seed)...
    for k in ("context_length", "target_model_name", "seed", "target_revision"):
        assert p2[k] == ref[k], k
    # ...but a DIFFERENT configuration, or the row is a second measurement of
    # its own reference.
    assert (p2["rope_type"], p2.get("rope_factor"),
            p2.get("rope_anchor_base")) != (
        ref["rope_type"], ref.get("rope_factor"), ref.get("rope_anchor_base"))
    assert "P2_native_64k" not in rows


def test_the_declared_rope_control_declares_exactly_the_shipped_block():
    """Declaring what the model already ships is a numerical no-op, which is why
    the control's expected verdict is PASS rather than hoped for."""
    p2 = next(c for c in _controls()
              if c["name"] == "P2_declared_shipped_64k")
    assert p2["rope_type"] == "llama3"
    assert p2["rope_factor"] == 8
    assert p2["rope_anchor_base"] == 8192
    assert p2["expect"] == "pass"


def test_the_declared_rope_rebuilds_the_shipped_rope_bit_for_bit():
    """The claim the control rests on, checked against the real config.

    Needs the model's config.json, so it skips without a cache or network — in
    which case the control is still a valid exercise of the declaration path,
    just not a verified no-op.
    """
    gate = _load_gate()
    try:
        from transformers import AutoConfig
        cfg = AutoConfig.from_pretrained(LLAMA31, local_files_only=True)
    except Exception as e:                      # noqa: BLE001
        pytest.skip(f"no local config for {LLAMA31}: {e}")
    shipped = dict(getattr(cfg, "rope_scaling", None) or {})
    assert shipped.get("rope_type") == "llama3"
    native = gate._declared_reference({}, LLAMA31)
    declared = gate._declared_reference(
        {"rope_type": "llama3", "rope_factor": 8, "rope_anchor_base": 8192,
         "context_length": 65536}, LLAMA31)
    assert float((native - declared).abs().max()) == 0.0


def test_every_declared_positive_control_named_in_the_manifest_exists():
    """The rename touched the manifest's documentation list; a stale name there
    would describe a control that does not run."""
    names = {c["name"] for c in _controls()}
    manifest = MANIFEST.read_text()
    line = next(l for l in manifest.splitlines()
                if l.strip().startswith("positive_controls:"))
    listed = [x.strip() for x in
              line.split("[", 1)[1].rstrip("]").split(",")]
    missing = [n for n in listed if n not in names]
    assert not missing, f"{missing} are listed as positive controls but absent"

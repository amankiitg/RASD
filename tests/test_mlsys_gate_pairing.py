"""The correction evidence must compare ONE SAMPLE with itself.

The correction note's claim is that the rope's ANCHORING explains the perplexity
damage at out-of-distribution contexts. That claim is a paired contrast: the same
document, offset, prompt and continuation, scored twice with only the anchor
differing. If the candidate and its reference score different windows, the ratio
contains the difference between two samples -- which is far larger than the
effect under test -- and the number is reported as an error rather than a ratio.

Three things are checked here:

  * the shipped candidate file pairs every OOD candidate with a reference at the
    same seed;
  * `pairing_verdict` refuses a ratio when either hash differs, and marks the
    in-distribution row descriptive;
  * the gate's window call passes the tokenizer's BOS on every path, because the
    engine prepends one and a window built without it is a different sample as
    well as a different length.
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
GATE = REPO / "scripts" / "mlsys_coherence_gate.py"
CORRECTION = REPO / "configs" / "mlsys_correction_candidates.json"


def _load_gate():
    spec = importlib.util.spec_from_file_location("_gate_under_test", GATE)
    mod = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(mod)
    except Exception as e:                      # noqa: BLE001
        pytest.skip(f"gate not importable here: {e}")
    return mod


# ---------------------------------------------------------------------------
# the shipped pairing
# ---------------------------------------------------------------------------

def test_every_ood_candidate_shares_its_reference_seed():
    d = json.loads(CORRECTION.read_text())
    cands = d["candidates"]
    refs = {}
    for c in cands:
        if c.get("reference_role") == "unscaled_ood_reference":
            refs[(c["context_length"], c["target_model_name"])] = c
    assert refs, "no unscaled-OOD reference rows are declared"
    for c in cands:
        if c.get("reference_role"):
            continue
        ref = refs.get((c["context_length"], c["target_model_name"]))
        assert ref is not None, f"{c['name']} has no reference at its context"
        assert c.get("seed") == ref.get("seed") == 42, (
            f"{c['name']} uses seed {c.get('seed')} but its reference uses "
            f"{ref.get('seed')}: they would score different samples")


def test_the_in_distribution_row_is_descriptive():
    d = json.loads(CORRECTION.read_text())
    row = [c for c in d["candidates"]
           if c.get("reference_role") == "in_distribution_reference"]
    assert len(row) == 1, "the 4096 in-distribution row is missing"
    assert row[0]["context_length"] == 4096


# ---------------------------------------------------------------------------
# the verdict
# ---------------------------------------------------------------------------

def _pair(**over):
    cand = {"prompt_sha256": "a" * 64, "continuation_sha256": "b" * 64,
            "seed": 42}
    base = {"prompt_sha256": "a" * 64, "continuation_sha256": "b" * 64,
            "seed": 42}
    cand.update({k: v for k, v in over.items() if k in
                 ("prompt_sha256", "continuation_sha256", "seed",
                  "reference_role")})
    return cand, base


def test_matching_hashes_are_paired():
    gate = _load_gate()
    cand, base = _pair()
    pairing, reason = gate.pairing_verdict(cand, base)
    assert pairing == "paired", reason


def test_a_different_continuation_is_unpaired():
    """Same prompt, different continuation: still two different samples."""
    gate = _load_gate()
    cand, base = _pair(continuation_sha256="c" * 64)
    pairing, reason = gate.pairing_verdict(cand, base)
    assert pairing == "unpaired"
    assert "continuation_sha256" in reason


def test_a_different_prompt_is_unpaired():
    gate = _load_gate()
    cand, base = _pair(prompt_sha256="d" * 64)
    pairing, reason = gate.pairing_verdict(cand, base)
    assert pairing == "unpaired" and "prompt_sha256" in reason


def test_a_missing_hash_is_unpaired_not_matched():
    """Two blanks are not agreement."""
    gate = _load_gate()
    cand, base = _pair(prompt_sha256="")
    pairing, _ = gate.pairing_verdict(cand, base)
    assert pairing == "unpaired"


def test_different_seeds_are_unpaired_even_with_matching_hashes():
    gate = _load_gate()
    cand, base = _pair(seed=7)
    pairing, reason = gate.pairing_verdict(cand, base)
    assert pairing == "unpaired" and "seeds differ" in reason


def test_the_in_distribution_row_is_descriptive_not_paired():
    gate = _load_gate()
    cand, base = _pair(reference_role="in_distribution_reference")
    pairing, reason = gate.pairing_verdict(cand, base)
    assert pairing == "descriptive"
    assert "not a denominator" in reason


# ---------------------------------------------------------------------------
# item 3: the BOS on every window path
# ---------------------------------------------------------------------------

def test_the_gate_window_call_passes_the_bos(monkeypatch):
    """Tested through the CALLER the gate uses, not the helper's default.

    `load_pg19_window(bos_id=None)` is a legal call that produces a window one
    token short of what the engine sees, so a test of the helper's default would
    prove nothing about what the gate does.
    """
    gate = _load_gate()
    seen = {}

    def fake(meta_path, ctx, seed, bos_id=None):
        seen["bos_id"] = bos_id
        return [1, 2, 3], [4, 5, 6]

    monkeypatch.setattr(gate, "load_pg19_window", fake)

    class Tok:
        bos_token_id = 128000

    gate.gate_sample("meta.json", 131072, 42, Tok())
    assert seen["bos_id"] == 128000, (
        "gate_sample did not pass the tokenizer's BOS, so the gate would score "
        "a window the engine never saw")


def test_run_candidate_uses_the_single_window_call():
    src = GATE.read_text()
    assert "load_pg19_window(\n" not in src.split("def gate_sample")[1], (
        "run_candidate calls load_pg19_window directly, so a second caller path "
        "can drift from the BOS-passing one")
    assert "gate_sample(" in src.split("def run_candidate")[1]


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))


def test_an_extension_is_labelled_cross_context_not_unpaired():
    """>128k is judged against the native window: different windows by design.

    The plan declares that comparison ("does extending the window keep the
    target coherent, where coherent is defined by the native configuration"), so
    it is labelled rather than refused -- but it must never be presented as a
    same-sample contrast.
    """
    gate = _load_gate()
    cand = {"prompt_sha256": "a" * 64, "continuation_sha256": "b" * 64,
            "seed": 42, "context_length": 262144, "reference_context": 131072}
    base = {"prompt_sha256": "c" * 64, "continuation_sha256": "d" * 64,
            "seed": 42, "context_length": 131072}
    pairing, reason = gate.pairing_verdict(cand, base)
    assert pairing == "cross_context"
    assert "cross-window" in reason


def test_every_same_context_reference_shares_its_candidates_seed():
    """Asserted on the SHIPPED configs, for all four candidate files."""
    problems = []
    for f in ("configs/mlsys_gate_controls.json",
              "configs/mlsys_rope_candidates.json",
              "configs/mlsys_rope_intervention_candidates.json",
              "configs/mlsys_correction_candidates.json"):
        d = json.loads((REPO / f).read_text())
        key = "candidates" if "candidates" in d else "coherence_gate"
        rows = d[key]
        refs = {(r["context_length"], r["target_model_name"]): r
                for r in rows if r.get("native_baseline")}
        for r in rows:
            if r.get("reference_role") == "in_distribution_reference":
                continue
            ref_ctx = r.get("reference_context") or r["context_length"]
            if int(ref_ctx) != int(r["context_length"]):
                continue                      # cross-window: not a paired contrast
            ref = refs.get((ref_ctx, r["target_model_name"]))
            if ref is None:
                continue
            if r.get("seed") != ref.get("seed"):
                problems.append(f"{f}: {r['name']} seed {r.get('seed')} vs "
                                f"{ref['name']} seed {ref.get('seed')}")
    assert not problems, problems

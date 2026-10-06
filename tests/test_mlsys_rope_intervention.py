"""rope_intervention_128k must isolate the rope, and must be pre-registered.

Every other comparison in the plan moves the context length and the target's
rope together, so none of them isolates a rope intervention. This stage holds the
context at 128k and moves only the rope, and it can only make that claim if the
two arms are otherwise identical -- so the isolation is asserted here rather than
described in a comment.

The stage was also required to be pre-registered before it was implemented, and
the comparison is a pre-registered one with a pre-registered equivalence margin,
so the plan is checked to contain the margin that the analysis will apply.
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest
import yaml

REPO = Path(__file__).resolve().parent.parent
CFG = REPO / "configs" / "mlsys_rope_intervention_128k.yml"
CAND = REPO / "configs" / "mlsys_rope_intervention_candidates.json"
MANIFEST = REPO / "configs" / "mlsys_manifest.yml"
PLAN = REPO / "docs" / "mlsys_analysis_plan.md"

ROPES = {"rope_type", "rope_factor", "rope_anchor_base"}
# Not run parameters: the level id, the arm's own label, and the naming fields.
# `rope_arm` IS expected to differ between arms -- identifying which arm a row
# belongs to is its whole purpose -- so it is excluded from the isolation check,
# not from the comparison.
NOT_A_PARAM = {"id", "group", "run_id", "level_id", "name", "notes", "rope_arm"}


def _gate():
    spec = importlib.util.spec_from_file_location(
        "gate_ri", REPO / "scripts" / "mlsys_coherence_gate.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["gate_ri"] = mod
    spec.loader.exec_module(mod)
    return mod


def test_the_arms_differ_only_in_the_rope():
    assert CFG.exists(), f"the approved stage has no config: {CFG}"
    cfg = yaml.safe_load(CFG.read_text())
    groups = [k for k in cfg if k != "defaults"]
    assert groups == ["RI_native_128k_SPEC", "RI_llama3_f16_128k_SPEC",
                      "RI_llama3_f32_128k_SPEC"], groups
    native = cfg[groups[0]]["levels"][0]
    treated = [cfg[g]["levels"][0] for g in groups[1:]]

    # the control runs the SHIPPED block -- declaring nothing
    assert "rope_type" not in native, "the control arm is not the shipped rope"

    for t in treated:
        # everything that is not the rope must be identical to the control
        for key in set(native) | set(t):
            if key in ROPES or key in NOT_A_PARAM:
                continue
            assert native.get(key) == t.get(key), (
                f"a treated arm differs from the control in {key!r}, so the "
                f"rope is not isolated"
            )
        assert t["rope_type"] == "llama3"
        # the anchor stays where the model ships it, or the factor is
        # confounded with the anchor -- the error the correction note documents
        assert t["rope_anchor_base"] == 8192
        assert len(t["documents"]) == 10
        assert native["documents"] == t["documents"]

    # two factors, because one could not distinguish "nothing moved" from
    # "coherence broke"
    assert sorted(int(t["rope_factor"]) for t in treated) == [16, 32]

    # every arm is labelled: all three are speculative, so arm_role alone
    # cannot tell them apart and the analysis would pair the wrong two
    for g in groups:
        assert cfg[g]["levels"][0].get("rope_arm"), f"{g} has no rope_arm label"
    arms = [cfg[g]["levels"][0]["rope_arm"] for g in groups]
    assert arms == ["native", "factor16", "factor32"], arms

    core = yaml.safe_load(MANIFEST.read_text())["documents"]["core_10"]
    assert set(native["documents"]) == set(core)


def test_the_declared_anchor_reproduces_the_shipped_block():
    """factor 16 at the shipped anchor is the ONLY change.

    If declaring factor 8 at 8192 did not reproduce the model's shipped
    inv_freq, then the treated arm would differ in more than the factor, and the
    'only the factor moved' claim would be false. Measured, not asserted from
    the config text.
    """
    g = _gate()
    model = "meta-llama/Llama-3.1-8B"
    shipped = g._declared_reference({}, model)
    as_f8 = g._declared_reference(
        {"rope_type": "llama3", "rope_factor": 8, "rope_anchor_base": 8192}, model)
    as_f16 = g._declared_reference(
        {"rope_type": "llama3", "rope_factor": 16, "rope_anchor_base": 8192}, model)
    as_f32 = g._declared_reference(
        {"rope_type": "llama3", "rope_factor": 32, "rope_anchor_base": 8192}, model)

    assert float((shipped - as_f8).abs().max()) < 1e-12, (
        "declaring factor 8 at 8192 does not reproduce the shipped rope, so the "
        "shipped factor or the anchor assumption is wrong"
    )
    # and the intervention really does move the rope, or the arms are the same
    assert float((as_f8 - as_f16).abs().max()) > 1e-6, (
        "factor 16 produces the same inv_freq as factor 8, so the intervention "
        "changes nothing"
    )
    assert float((as_f8 - as_f32).abs().max()) > 1e-6, (
        "factor 32 produces the same inv_freq as factor 8"
    )


def test_the_equivalence_rule_is_the_pre_registered_three_way_decision():
    """Inside / outside / inconclusive, and never rounded to a boundary."""
    sys.path.insert(0, str(REPO))
    from src.analysis.document_bootstrap import equivalence_verdict as ev

    m = 0.05
    # entirely inside -> equivalent
    assert ev({"lo": -0.01, "hi": 0.02}, m) == "equivalent"
    assert ev({"lo": 0.0, "hi": 0.0}, m) == "equivalent"
    assert ev({"lo": -0.05, "hi": 0.05}, m) == "equivalent"      # closed
    # entirely outside -> an effect, with a direction
    assert ev({"lo": 0.07, "hi": 0.12}, m) == "effect"
    assert ev({"lo": -0.12, "hi": -0.07}, m) == "effect"
    # overlapping a boundary -> inconclusive, NOT rounded to the nearer side
    assert ev({"lo": -0.02, "hi": 0.07}, m) == "inconclusive"
    assert ev({"lo": -0.07, "hi": 0.02}, m) == "inconclusive"
    assert ev({"lo": -0.06, "hi": -0.05}, m) == "inconclusive"
    # a single point inside the margin is NOT equivalence on its own: the point
    # is not the interval, and treating it as such is the failure this rule
    # exists to prevent
    assert ev({"lo": float("-inf"), "hi": 0.0}, m) == "inconclusive"
    assert ev({"lo": float("nan"), "hi": 0.0}, m) == "inconclusive"


def test_the_stage_is_pre_registered_with_its_margin():
    assert CFG.exists(), f"the approved stage has no config: {CFG}"
    plan = PLAN.read_text()
    assert "rope_intervention_128k" in plan, "the stage is not pre-registered"
    assert "equivalence margin: 0.05" in plan.lower(), (
        "the pre-registered equivalence margin is missing from the plan"
    )
    assert "document-bootstrap" in plan and "n_boot = 10000" in plan, (
        "the pre-registered interval method is missing"
    )
    # the stage must be listed as approved on both sides
    assert "rope_intervention_128k" in MANIFEST.read_text()
    for f in ("scripts/mlsys_manifest.sh", "scripts/mlsys_watch_and_run.sh"):
        assert "rope_intervention_128k" in (REPO / f).read_text(), f


def test_every_gate_candidate_has_a_baseline_at_its_own_context():
    assert CAND.exists(), f"the stage has no gate candidates file: {CAND}"
    cands = json.loads(CAND.read_text())["candidates"]
    baselines = {(c["context_length"], c["target_model_name"])
                 for c in cands if c.get("native_baseline")}
    assert baselines, "no declared baseline, so no ratio can be computed"
    for c in cands:
        key = (c["context_length"], c["target_model_name"])
        assert key in baselines, (
            f"{c['name']} has no declared baseline at {key}, so it cannot be "
            f"measured and its verdict would be meaningless"
        )


def test_the_stage_has_no_target_only_arm():
    """Spec-only by design: nothing here is a ratio against a baseline."""
    assert CFG.exists(), f"the approved stage has no config: {CFG}"
    cfg = yaml.safe_load(CFG.read_text())
    groups = [k for k in cfg if k != "defaults"]
    assert len(groups) == 3, groups
    for g in groups:
        assert "TARGET" not in g, f"{g} is a target-only arm"
        assert cfg[g]["levels"][0].get("spec_steps", 1) > 0, g
    stage = [s for s in yaml.safe_load(MANIFEST.read_text())["stages"]
             if s["id"] == "rope_intervention_128k"][0]
    assert stage.get("spec_only") is True
    assert stage.get("losslessness") == "not_required"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))


def _ri_mod():
    spec = importlib.util.spec_from_file_location(
        "ri_mod", REPO / "scripts" / "mlsys_rope_intervention.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["ri_mod"] = mod
    spec.loader.exec_module(mod)
    return mod


def _row(doc, arm, value, status="ok"):
    return {"run_id": f"RI_{arm}_{doc}", "doc_id": doc, "pair_id": f"131072:{doc}",
            "rope_arm": arm, "status": status, "acceptance_rate": value,
            "spec_steps": 4}


def test_the_comparison_is_a_paired_difference_over_shared_documents():
    m = _ri_mod()
    rows = []
    for d in ("d0", "d1", "d2"):
        rows += [_row(d, "native", 0.90), _row(d, "factor16", 0.902),
                 _row(d, "factor32", 0.60)]
    by_arm, _ = m.load_arms(rows, "acceptance_rate")
    r = m.compare(by_arm, "native", "factor16", "acceptance_rate", 0.05, 200, 1)
    assert r["n_documents"] == 3
    assert r["point"] == pytest.approx(0.002, abs=1e-9)
    # a small move on every document -> the interval is inside the margin
    assert r["verdict"] == "equivalent", r

    r32 = m.compare(by_arm, "native", "factor32", "acceptance_rate", 0.05, 200, 1)
    assert r32["verdict"] == "effect", r32
    assert r32["point"] < 0


def test_an_arm_that_did_not_run_is_absent_rather_than_filled_in():
    """A gate failure leaves an arm with no rows. That is its result."""
    m = _ri_mod()
    rows = []
    for d in ("d0", "d1"):
        rows += [_row(d, "native", 0.90), _row(d, "factor16", 0.902)]
    by_arm, _ = m.load_arms(rows, "acceptance_rate")
    assert "factor32" not in by_arm
    # the contrast that CAN be formed is formed
    assert m.compare(by_arm, "native", "factor16", "acceptance_rate",
                     0.05, 100, 1) is not None
    # and the one that cannot is absent, not a zero
    assert m.compare(by_arm, "native", "factor32", "acceptance_rate",
                     0.05, 100, 1) is None


def test_rows_with_no_arm_label_are_excluded_not_guessed():
    """All three arms are speculative, so a label-less row is unrecoverable."""
    m = _ri_mod()
    rows = [_row("d0", "native", 0.90), _row("d0", "factor16", 0.902)]
    unlabelled = _row("d1", "factor16", 0.91)
    unlabelled["rope_arm"] = ""
    rows.append(unlabelled)
    by_arm, _ = m.load_arms(rows, "acceptance_rate")
    # d1 is dropped from factor16 rather than being assigned to an arm
    assert set(by_arm["factor16"]) == {"d0"}, by_arm["factor16"]
    assert by_arm["native"] == {"d0": 0.90}


def test_only_ok_rows_are_compared():
    """A document whose row failed in one arm is not compared at all.

    load_arms keeps the ok rows of each arm independently; the restriction that
    matters happens in `compare`, which uses the INTERSECTION of the two arms'
    documents. A document present in one arm and not the other is dropped, not
    compared against a missing or substituted value.
    """
    m = _ri_mod()
    rows = [_row("d0", "native", 0.90), _row("d0", "factor16", 0.902),
            _row("d1", "native", 0.88, status="error"),
            _row("d1", "factor16", 0.881)]
    by_arm, _ = m.load_arms(rows, "acceptance_rate")
    assert "d1" not in by_arm["native"], "an error row entered the comparison"
    assert "d1" in by_arm["factor16"]          # that row is itself fine
    r = m.compare(by_arm, "native", "factor16", "acceptance_rate", 0.05, 100, 1)
    assert r["n_documents"] == 1, r          # only d0 survives the intersection
    assert r["documents"] == "d0", r

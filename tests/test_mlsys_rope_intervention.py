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
# Not run parameters: the level id and the run's own naming fields.
NOT_A_PARAM = {"id", "group", "run_id", "level_id", "name", "notes"}


def _gate():
    spec = importlib.util.spec_from_file_location(
        "gate_ri", REPO / "scripts" / "mlsys_coherence_gate.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["gate_ri"] = mod
    spec.loader.exec_module(mod)
    return mod


def test_the_two_arms_differ_only_in_the_rope():
    assert CFG.exists(), f"the approved stage has no config: {CFG}"
    cfg = yaml.safe_load(CFG.read_text())
    groups = [k for k in cfg if k != "defaults"]
    assert groups == ["RI_native_128k_SPEC", "RI_llama3_f16_128k_SPEC"], groups
    native = cfg[groups[0]]["levels"][0]
    treated = cfg[groups[1]]["levels"][0]

    # everything that is not the rope must be identical
    for key in set(native) | set(treated):
        if key in ROPES or key in NOT_A_PARAM:
            continue
        assert native.get(key) == treated.get(key), (
            f"the arms differ in {key!r}, so the rope is not isolated"
        )

    # the control arm runs the SHIPPED block -- declaring nothing
    assert "rope_type" not in native, "the control arm is not the shipped rope"
    assert treated["rope_type"] == "llama3"
    assert treated["rope_factor"] == 16
    assert treated["rope_anchor_base"] == 8192

    # the same ten documents, so the paired difference is per document
    assert len(native["documents"]) == 10
    assert native["documents"] == treated["documents"]
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

    assert float((shipped - as_f8).abs().max()) < 1e-12, (
        "declaring factor 8 at 8192 does not reproduce the shipped rope, so the "
        "shipped factor or the anchor assumption is wrong"
    )
    # and the intervention really does move the rope, or the arms are the same
    assert float((as_f8 - as_f16).abs().max()) > 1e-6, (
        "factor 16 produces the same inv_freq as factor 8, so the intervention "
        "changes nothing"
    )


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
    for g in (k for k in cfg if k != "defaults"):
        assert "TARGET" not in g, f"{g} is a target-only arm"
        assert cfg[g]["levels"][0].get("spec_steps", 1) > 0, g
    stage = [s for s in yaml.safe_load(MANIFEST.read_text())["stages"]
             if s["id"] == "rope_intervention_128k"][0]
    assert stage.get("spec_only") is True
    assert stage.get("losslessness") == "not_required"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))

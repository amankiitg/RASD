"""A control with no baseline at its own context was NOT measured.

The gate refuses to compute a ratio when no native baseline is declared at the
candidate's own context and model, and reports `gate_pass = False` with the
reason "no declared native baseline...". For a NEGATIVE control that is
indistinguishable, from the CSV alone, from the gate successfully detecting a
broken configuration -- so the naive checker counted an unmeasured negative as
"detected". The ARM4 f2 collapse was published once already on the strength of a
number that had not been checked; a control that cannot be measured must not be
able to certify the gate.

`no baseline` is therefore not a detection, and the missing baseline is a
calibration failure in its own right.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent


def _load():
    spec = importlib.util.spec_from_file_location(
        "calib_mod", REPO / "scripts" / "mlsys_gate_calibration_check.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["calib_mod"] = mod
    spec.loader.exec_module(mod)
    return mod


CONTROLS = {
    "candidates": [
        {"name": "POS", "expect": "pass"},
        {"name": "NEG", "expect": "fail"},
    ]
}


def _write(tmp_path, rows):
    import csv
    ct = tmp_path / "controls.json"
    ct.write_text(__import__("json").dumps(CONTROLS))
    csv_path = tmp_path / "gate.csv"
    fields = ["candidate", "status", "gate_pass", "gate_reason",
              "ppl_continuation", "native_ppl_reference", "ppl_ratio", "error"]
    with csv_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fields})
    return csv_path, ct


def _row(name, passed, ppl="10.0", baseline="10.1", reason=""):
    return {"candidate": name, "status": "ok",
            "gate_pass": "True" if passed else "False",
            "ppl_continuation": ppl, "native_ppl_reference": baseline,
            "ppl_ratio": "", "gate_reason": reason}


@pytest.mark.skipif(not (REPO / "configs").exists(),
                    reason="repo layout changed")
def test_unmeasured_negative_control_is_not_a_detection(tmp_path):
    calib = _load()

    # The failure mode: "no baseline" reported as a passed-negative detection.
    csv_path, ct = _write(tmp_path, [
        _row("POS", True),
        _row("NEG", False, ppl="", baseline="",
             reason="no declared native baseline at this context and model"),
    ])
    problems, summary = calib.check(csv_path, ct)
    assert problems, (
        "a negative control with no measured ratio was accepted as detected"
    )
    assert any("NOT MEASURED" in p for p in problems), problems
    assert summary["negatives"] == []

    # A genuinely measured negative failure IS a detection.
    csv_path, ct = _write(tmp_path, [
        _row("POS", True),
        _row("NEG", False, ppl="826.9", baseline="14.45",
             reason="perplexity 57.2x the native baseline (tolerance 1.5x)"),
    ])
    problems, summary = calib.check(csv_path, ct)
    assert problems == [], problems
    assert len(summary["negatives"]) == 1


@pytest.mark.skipif(not (REPO / "configs").exists(),
                    reason="repo layout changed")
def test_positive_control_must_actually_pass(tmp_path):
    calib = _load()
    csv_path, ct = _write(tmp_path, [
        _row("POS", False, ppl="30.0", baseline="10.0",
             reason="perplexity 3.0x the native baseline"),
        _row("NEG", False, ppl="826.9", baseline="14.45", reason="57.2x"),
    ])
    problems, _ = calib.check(csv_path, ct)
    assert any("POS" in p and "expected pass" in p for p in problems), problems


@pytest.mark.skipif(not (REPO / "configs").exists(),
                    reason="repo layout changed")
def test_shipped_controls_declare_a_baseline_at_every_control_context():
    """The shipped file must not rely on a context where no baseline exists."""
    import json
    doc = json.loads((REPO / "configs" / "mlsys_gate_controls.json").read_text())
    cands = doc["candidates"]
    baselines = {(c["context_length"], c["target_model_name"])
                 for c in cands if c.get("native_baseline")}
    for c in cands:
        key = (c["context_length"], c["target_model_name"])
        assert key in baselines, (
            f"{c['name']} has no declared native baseline at its own context "
            f"{key}, so it cannot be measured"
        )
    # A control that is its own baseline would give ratio 1.0 and could not fail.
    for c in cands:
        if c.get("native_baseline"):
            assert c.get("role") == "baseline" or c["name"].startswith("B_"), (
                f"{c['name']} is both a control and its own baseline"
            )


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))

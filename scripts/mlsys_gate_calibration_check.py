#!/usr/bin/env python3
"""Check the coherence gate against controls whose answers are already known.

The gate decides which rope configurations are allowed to run, so an uncalibrated
gate is worse than no gate: it converts "we did not check" into "we checked and
it passed". This script is the check on the checker.

It reads the calibration CSV (written by the gate from
`configs/mlsys_gate_controls.json`) and requires every control's measured verdict
to match its declared `expect`:

  * a POSITIVE control (a native configuration, which must be coherent) that
    FAILS means the gate is too strict — most likely the perplexity tolerance was
    set too tight, and it would have rejected good configurations;
  * a NEGATIVE control (a configuration already known to be broken) that PASSES
    means the gate cannot detect the failure it exists to detect — and the ARM4
    f2 cell is proof that such a failure can otherwise be published as a result.

Either way, no candidate may be gated until this passes. Neither is fixed by
relaxing a threshold to make a control agree.

A control counts as MEASURED only when the gate produced a perplexity for it
AND a declared native baseline to compare against at its own context. "no
baseline" is not a detection: without it the ratio is undefined, so a negative
control could otherwise be "detected" by never having been measured at all.

The tolerance is FIXED at 1.5x by the 2026-10-06 plan revision, so the
positives' spread no longer sets it. The spread is still reported, as the
evidence that the gate's perplexity measurement sits well inside the tolerance.

Usage:
    python scripts/mlsys_gate_calibration_check.py results/mlsys/gate_calibration.csv
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
CONTROLS = REPO / "configs" / "mlsys_gate_controls.json"


def load_expectations(path: Path) -> dict:
    data = json.loads(path.read_text())
    return {c["name"]: c.get("expect") for c in data["candidates"]}


def check(csv_path: Path, controls_path: Path) -> tuple[list[str], dict]:
    expect = load_expectations(controls_path)
    rows = list(csv.DictReader(csv_path.open()))
    problems: list[str] = []
    summary: dict = {"positives": [], "negatives": [], "missing": []}

    seen = set()
    for r in rows:
        name = r.get("candidate", "")
        seen.add(name)
        want = expect.get(name)
        if want is None:
            problems.append(f"{name}: not a declared control")
            continue
        if r.get("status") not in ("ok", ""):
            # A control the gate could not measure cannot be judged. Report it
            # rather than skipping it silently: an unmeasured control is not a
            # passing control.
            problems.append(
                f"{name}: gate could not measure it "
                f"({str(r.get('error', 'no error recorded'))[:80]})"
            )
            continue
        got = "pass" if str(r.get("gate_pass")) == "True" else "fail"
        ppl = r.get("ppl_continuation", "")
        reason = r.get("gate_reason", "")
        # MEASURED means the gate produced both a perplexity and a declared
        # baseline to compare it with. A row whose ratio is undefined was not
        # measured, and "no baseline" must never be read as a detected failure
        # for a negative control.
        measured = bool(ppl not in ("", None)) and bool(
            r.get("native_ppl_reference") not in ("", None))
        if not measured:
            problems.append(
                f"{name}: NOT MEASURED -- no perplexity and/or no declared "
                f"native baseline at its own context ({reason[:70]}). An "
                f"unmeasured control is neither a pass nor a detected failure."
            )
            continue
        entry = {"name": name, "want": want, "got": got, "ppl": ppl,
                 "reason": reason, "measured": True,
                 "baseline": r.get("native_ppl_reference", ""),
                 "ratio": r.get("ppl_ratio", "")}
        (summary["positives"] if want == "pass" else summary["negatives"]).append(entry)
        if got != want:
            problems.append(
                f"{name}: expected {want}, got {got} ({reason}); "
                f"ppl={ppl}"
            )

    for name, want in expect.items():
        if name not in seen:
            summary["missing"].append(name)
            problems.append(f"{name}: no verdict row in {csv_path}")

    # What tolerance the positive spread would justify, as a sanity number to
    # compare against the a-priori 1.5x.
    try:
        vals = [float(e["ppl"]) for e in summary["positives"]
                if e["ppl"] not in ("", None)]
        if vals and min(vals) > 0:
            summary["positive_ppl_spread"] = round(max(vals) / min(vals), 4)
    except (TypeError, ValueError):
        pass
    return problems, summary


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("csv")
    p.add_argument("--controls", default=str(CONTROLS))
    args = p.parse_args()

    problems, summary = check(Path(args.csv), Path(args.controls))

    for e in summary["positives"]:
        print(f"  positive {e['name']:<28} want={e['want']:<5} got={e['got']:<5} "
              f"ppl={e['ppl']:<12} baseline={e.get('baseline', ''):<12} "
              f"ratio={e.get('ratio', '')}")
    for e in summary["negatives"]:
        print(f"  negative {e['name']:<28} want={e['want']:<5} got={e['got']:<5} "
              f"ppl={e['ppl']:<12} baseline={e.get('baseline', ''):<12} "
              f"ratio={e.get('ratio', '')}")
    if "positive_ppl_spread" in summary:
        print(f"  positive-control ppl spread max/min = "
              f"{summary['positive_ppl_spread']}  (informational: the tolerance "
              f"is FIXED at 1.5x by the plan revision; this is the evidence the "
              f"measurement sits well inside it)")

    if problems:
        print("\nCALIBRATION FAILED:")
        for pr in problems:
            print(f"  - {pr}")
        print("\nThe gate is wrong, not the control. Fix the gate before gating "
              "any candidate; do not relax the threshold to make a control agree.")
        return 1
    print("\nCALIBRATION OK: every control matched its expected verdict.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

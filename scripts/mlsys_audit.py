#!/usr/bin/env python
"""A8 audit — every arm CSV must have exactly one spec row and one
target-only row per seed, and nothing else.

Why this exists: the Arm2/Arm3 memory comparison was originally computed over
"4 spec rows" for a 3-seed arm. The extra row was the CANARY (a 15-token smoke
cell with an empty context_length), which inflated the mean and produced the
wrong "~2x memory" figure. Excluding it gives 5.57x. That class of error — a
row that is not a real experimental cell silently entering an aggregate — is
exactly what an audit should catch, automatically, every time.

Rules enforced per arm-level CSV (files that have group/level_id/run_id):

  1. Canaries (group == "canary" or empty context_length) are acknowledged and
     EXCLUDED from the per-seed counts.
  2. For each seed present: exactly ONE spec row (spec_steps > 0) and exactly
     ONE target-only row (spec_steps == 0).
  3. No duplicate (seed, spec_steps) pairs.
  4. Every non-canary row has a non-empty context_length.

Exit 0 = clean, 1 = violations found (printed).

Usage:
    python scripts/mlsys_audit.py --results-dir results/mlsys
"""
from __future__ import annotations

import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent

# Files that are NOT arm result CSVs.
SKIP_SUFFIXES = {
    "gpu_hours.csv", "acceptance_accounting.csv", "summary_with_ci.csv",
    "trace_summary.csv", "dip_test.csv", "dip_by_context.csv", "arm_dip.csv",
    "vllm_baseline.csv",
}
# Multi-context matrices legitimately carry several rows per seed (one per
# context level); the invariant there is one row per (seed, context, kind).
SKIP_FILES = {"shakedown.csv"}


def is_canary(r: dict) -> bool:
    return (str(r.get("group", "")).lower() == "canary"
            or not str(r.get("context_length", "")).strip())


def audit_file(path: Path) -> tuple[list[str], list[str]]:
    rows = list(csv.DictReader(path.open()))
    if not rows or "spec_steps" not in rows[0]:
        return [], []

    problems: list[str] = []
    info: list[str] = []
    # Key on (seed, context_length): a single-context arm CSV must have
    # exactly one spec + one target-only per seed, while a multi-context
    # matrix must have exactly one of each per (seed, context).
    per_cell: dict[tuple[str, str], dict[str, int]] = defaultdict(
        lambda: {"spec": 0, "tgt": 0})
    seen: set[tuple[str, str, str, str]] = set()
    n_canary = 0

    for r in rows:
        if is_canary(r):
            n_canary += 1
            continue
        seed = str(r.get("seed", "")).strip()
        k = str(r.get("spec_steps", "")).strip()
        if not str(r.get("context_length", "")).strip():
            problems.append(f"{r.get('run_id')}: non-canary row with empty context_length")
        ctx = str(r.get("context_length", "")).strip()
        # draft_dtype/cap distinguish the intentional FP4-vs-bf16 variants
        # that share a (seed, ctx, spec_steps); they are not duplicates.
        variant = f"{r.get('draft_dtype','')}|{r.get('draft_window_cap','')}"
        key = (seed, ctx, k, variant)
        if key in seen:
            problems.append(f"duplicate (seed={seed}, ctx={ctx}, spec_steps={k}) "
                            f"e.g. {r.get('run_id')}")
        seen.add(key)
        per_cell[(seed, ctx)][("spec" if k not in ("0", "") else "tgt")] += 1

    # The strict one-spec-one-target rule applies to the native-vs-YaRN ARM
    # CSVs. Other file types legitimately differ:
    #   * bf16_draft_isolation carries TWO spec rows per seed by design (the
    #     FP4 and bf16 draft variants) and no baseline.
    #   * the multiseed matrices are spec-only; their target-only baselines
    #     live in results/final.
    # Those are reported for information, not counted as failures.
    is_arm = all(str(r.get("run_id", "")).upper().startswith("ARM")
                 for r in rows if not is_canary(r))
    for (seed, ctx), c in sorted(per_cell.items()):
        if c["spec"] != 1:
            msg = f"seed {seed} ctx {ctx}: {c['spec']} spec rows (expected 1)"
            (problems if is_arm else info).append(msg)
        if c["tgt"] != 1 and is_arm:
            problems.append(
                f"seed {seed} ctx {ctx}: {c['tgt']} target-only rows (expected 1)")
    if not per_cell:
        problems.append("no non-canary rows at all")

    return problems, info


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", default=str(REPO / "results" / "mlsys"))
    args = ap.parse_args()

    d = Path(args.results_dir)
    files = sorted(p for p in d.glob("*.csv")
                   if p.name not in SKIP_SUFFIXES and p.name not in SKIP_FILES)
    if not files:
        print(f"[fatal] no arm CSVs under {d}")
        return 1

    total_bad = 0
    print(f"A8 audit — {len(files)} CSV(s) under {d}\n")
    for f in files:
        try:
            problems, info = audit_file(f)
        except Exception as e:  # noqa: BLE001
            print(f"  [ERROR] {f.name}: {e}")
            total_bad += 1
            continue
        if not problems and not info:
            print(f"  [OK]   {f.name}")
        elif not problems:
            print(f"  [OK]   {f.name}  (informational: {len(info)} note(s))")
            for i in info[:3]:
                print(f"           . {i}")
        else:
            total_bad += 1
            print(f"  [FAIL] {f.name}")
            for p in problems:
                print(f"           - {p}")

    print()
    if total_bad:
        print(f"AUDIT FAILED: {total_bad} file(s) with violations")
        return 1
    print("AUDIT PASSED: every arm CSV has exactly one spec and one "
          "target-only row per seed (canaries excluded)")
    return 0


if __name__ == "__main__":
    sys.exit(main())

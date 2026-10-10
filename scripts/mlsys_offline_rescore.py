#!/usr/bin/env python3
"""Apply the new repetition rules to results that were already produced.

Nothing here re-runs a model. It reads the sidecars the 2026-10-10 campaign pulled
off the pod and answers one question per row: does the new absolute rule change the
verdict, and does the acceptance number survive?

Three inputs, each optional, so one invocation can cover a whole stage family:

  --stage-csv     the stage's result CSV (for the OLD verdict/value)
  --tokens-dir    token sidecars, giving periodicity and token repetition
  --per-token-dir per-round traces, giving acceptance by window
  --generated-dir a directory of decoded continuations (`gen_*.txt`, written by the
                  coherence gate). Used for the word-n-gram ceiling when there are
                  no token ids.

Output: one row per (run, criterion) with old/new values and `changed`, plus a
plain-text summary of what actually moved.

Usage:
    python scripts/mlsys_offline_rescore.py \
        --stage-csv results/mlsys/incident_.../coherence_gate.csv \
        --generated-dir results/mlsys/incident_.../gate_generated \
        --out results/mlsys/rescore_coherence_gate.csv
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from src.analysis.repetition import (  # noqa: E402
    non_degenerate_acceptance, row_reasons, row_stats,
)

FIELDS = ["run_id", "criterion", "old", "new", "changed", "note"]

#: How much the acceptance may move before the old number is called misleading. A
#: row whose acceptance only comes from loop windows has no honest value at all,
#: and that is reported as a change regardless of the size of the gap.
ACCEPTANCE_TOL = 0.02


def _rows(csv_path: Path) -> list:
    return list(csv.DictReader(csv_path.open())) if csv_path.exists() else []


def _id_key(row: dict) -> str:
    return (row.get("run_id") or row.get("candidate") or "").strip()


def _modes(csv_rows: list, tokens: Path | None, per_token: Path | None,
           generated: Path | None) -> list:
    out = []
    for r in csv_rows:
        rid = _id_key(r)
        if not rid:
            continue
        rec = {"run_id": rid}

        ids = None
        if tokens is not None:
            f = tokens / f"{rid}.json"
            if f.exists():
                ids = json.loads(f.read_text()).get("generated_token_ids") or []
        text = None
        if generated is not None:
            for cand in (generated / f"gen_{rid}.txt",
                         generated / f"{rid}.txt"):
                if cand.exists():
                    text = cand.read_text()
                    break
        # The gate's `gen_*.txt` is the continuation alone; a stage's
        # `generated/*.txt` is prompt + continuation, and measuring repetition
        # over that dilutes a loop 128x. Where both a sidecar and a text exist,
        # the TOKENS win, because only they are unambiguously the continuation.
        if ids:
            rec.update(row_stats(ids))
        elif text is not None:
            rec.update(row_stats([], text))
        else:
            # NOT a failure: an unmeasurable row is reported as unmeasurable. A
            # row nothing can be said about must not be filed as a row that
            # failed, or the rescore invents verdict changes out of missing
            # evidence.
            rec["rep_unmeasurable"] = ("no token sidecar and no continuation "
                                       "text: the new rule cannot be applied")
            out.append(rec)
            continue
        rec["rep_reasons"] = "; ".join(row_reasons(rec))
        rec["rep_degenerate"] = bool(row_reasons(rec))

        if per_token is not None and ids:
            tr = []
            f = per_token / f"{rid}.jsonl"
            if f.exists():
                tr = [json.loads(l) for l in f.read_text().splitlines()
                      if l.strip()]
            if tr:
                res = non_degenerate_acceptance(tr, ids, 128)
                rec["windows_total"] = res["windows_total"]
                rec["windows_used"] = res["windows_used"]
                rec["acceptance_nondegenerate"] = res["acceptance"]
        out.append(rec)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage-csv", required=True)
    ap.add_argument("--tokens-dir", default=None)
    ap.add_argument("--per-token-dir", default=None)
    ap.add_argument("--generated-dir", default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("--label", default="")
    args = ap.parse_args()

    csv_rows = _rows(Path(args.stage_csv))
    modes = _modes(csv_rows,
                   Path(args.tokens_dir) if args.tokens_dir else None,
                   Path(args.per_token_dir) if args.per_token_dir else None,
                   Path(args.generated_dir) if args.generated_dir else None)
    by_id = {m["run_id"]: m for m in modes}

    comparisons = []
    for r in csv_rows:
        rid = _id_key(r)
        m = by_id.get(rid)
        if m is None:
            continue
        # 1) the repetition verdict the stage recorded only as a RELATIVE ratio
        old_ratio = r.get("repeat_share_ratio", "")
        old_share = r.get("gen_repeat_share", "")
        reasons = m.get("rep_reasons", "")
        passed_before = None
        if str(r.get("gate_pass", "")) != "":
            passed_before = str(r["gate_pass"]).strip().lower() in ("true", "1")
        comparisons.append({
            "run_id": rid, "criterion": "absolute_repetition",
            "old": (f"pass (no absolute rule; ratio {old_ratio or 'n/a'}"
                    f", share {old_share or 'n/a'})"),
            "new": ("unmeasurable: " + m["rep_unmeasurable"]
                    if m.get("rep_unmeasurable")
                    else ("pass" if not reasons else f"FAIL: {reasons}")),
            "changed": bool(reasons) and passed_before is True,
            "note": "" if passed_before is not None
                    else "stage CSV carries no pass field",
        })
        # 2) the acceptance number
        if r.get("acceptance_rate"):
            reported = r.get("acceptance_rate", "")
            new = m.get("acceptance_nondegenerate")
            if new is None and m.get("windows_used") == 0:
                comparisons.append({
                    "run_id": rid, "criterion": "acceptance",
                    "old": reported,
                    "new": "none (every window is degenerate)",
                    "changed": True,
                    "note": (f"{m.get('windows_used')}/{m.get('windows_total')} "
                             f"windows survive; the reported number is a loop"),
                })
            elif new is not None:
                try:
                    moved = abs(float(new) - float(reported)) > ACCEPTANCE_TOL
                except (TypeError, ValueError):
                    moved = False
                comparisons.append({
                    "run_id": rid, "criterion": "acceptance",
                    "old": reported, "new": new, "changed": moved,
                    "note": f"{m.get('windows_used')}/{m.get('windows_total')} "
                            f"non-degenerate windows",
                })
        # 3) rows that cannot be judged at all
        if m.get("rep_unmeasurable"):
            comparisons.append({
                "run_id": rid, "criterion": "unmeasurable", "old": "",
                "new": m["rep_unmeasurable"], "changed": False, "note": "",
            })

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS)
        w.writeheader()
        for c in comparisons:
            w.writerow(c)

    label = args.label or Path(args.stage_csv).stem
    n_rep = sum(1 for c in comparisons if c["criterion"] == "absolute_repetition")
    n_deg = sum(1 for m in modes if m.get("rep_degenerate"))
    changed = [c for c in comparisons if c["changed"]]
    print(f"=== {label} ===")
    print(f"rows: {len(csv_rows)}   degenerate now: {n_deg}/{n_rep}   "
          f"verdicts changed: {len(changed)}")
    for c in comparisons:
        if c["criterion"] == "absolute_repetition" and c["changed"]:
            print(f"  CHANGED {c['run_id']}: {c['old']} -> {c['new']}")
    for c in comparisons:
        if c["criterion"] == "acceptance":
            print(f"  acceptance {c['run_id'][:44]:44s} {str(c['old'])[:8]:>8s} -> "
                  f"{str(c['new'])[:34]:34s} {c['note'][:40]}")
    unmeasurable = [c for c in comparisons if c["criterion"] == "unmeasurable"]
    if unmeasurable:
        print(f"  {len(unmeasurable)} rows cannot be judged offline "
              f"(no tokens, no text)")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

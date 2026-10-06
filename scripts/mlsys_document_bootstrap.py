#!/usr/bin/env python3
"""Report acceptance, speedup and target quality with document-level intervals.

Reads a result CSV whose rows are per (rung, document, arm) and emits:

* per group, per metric: the document mean with a percentile bootstrap interval
  and the t-based cluster interval beside it;
* per group: the **paired** speedup against the target-only arm, matched by
  document, with its paired interval and a verdict against the 1.0 threshold.

The verdict is the plan's "stops paying off" rule: the interval must lie
entirely below 1.0. `inconclusive` is reported as such — the plan forbids
rounding an interval that contains the threshold toward the favourable side.

Usage:
    python scripts/mlsys_document_bootstrap.py \
        --results results/mlsys/natural.csv \
        --group-by level_id context_length \
        --spec-arm-column spec_steps --spec-arm 4 --target-arm 0 \
        --metrics acceptance_rate throughput_tps target_ppl \
        --out results/mlsys/natural_doc_intervals.csv
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from src.analysis.document_bootstrap import (
    clears, document_mean_bootstrap, paired_bootstrap, t_cluster_interval,
)


def _f(x):
    s = str(x).strip()
    if not s:
        return None
    try:
        return float(s)
    except ValueError:
        return None


def group_rows(rows: list[dict], keys: list[str]) -> dict[tuple, list[dict]]:
    out: dict[tuple, list[dict]] = {}
    for r in rows:
        out.setdefault(tuple(str(r.get(k, "")) for k in keys), []).append(r)
    return out


def summarise(rows: list[dict], keys: list[str], metrics: list[str],
              arm_column: str, spec_arm: str, target_arm: str,
              seed: int = 20261006) -> list[dict]:
    out: list[dict] = []
    for gkey, grows in sorted(group_rows(rows, keys).items()):
        label = dict(zip(keys, gkey))
        for metric in metrics:
            # One value per document; if a document appears twice inside the
            # same arm the mean would be weighted by row count, not by document.
            by_doc: dict[str, list[float]] = {}
            for r in grows:
                v = _f(r.get(metric))
                if v is None:
                    continue
                by_doc.setdefault(str(r.get("doc_id", "")), []).append(v)
            vals = [sum(v) / len(v) for v in by_doc.values()]
            if not vals:
                continue
            boot = document_mean_bootstrap(vals, seed=seed)
            t_ci = t_cluster_interval(vals)
            out.append({
                **label, "estimate": "mean", "metric": metric,
                "point": round(boot["mean"], 6),
                "ci_lo": round(boot["lo"], 6), "ci_hi": round(boot["hi"], 6),
                "t_ci_lo": round(t_ci["lo"], 6), "t_ci_hi": round(t_ci["hi"], 6),
                "n_documents": boot["n_documents"],
                "verdict": clears(boot, 1.0) if metric == "throughput_tps" else "",
            })

        spec = {str(r.get("doc_id", "")): r for r in grows
                if str(r.get(arm_column, "")) == spec_arm}
        targ = {str(r.get("doc_id", "")): r for r in grows
                if str(r.get(arm_column, "")) == target_arm}
        paired = sorted(set(spec) & set(targ))
        if not paired:
            continue
        a, b, dropped = [], [], []
        for d in paired:
            sa, tb = _f(spec[d].get("throughput_tps")), _f(targ[d].get("throughput_tps"))
            if not sa or not tb:
                dropped.append(d)
                continue
            a.append(sa)
            b.append(tb)
        if not a:
            continue
        pr = paired_bootstrap(a, b, kind="ratio", seed=seed)
        out.append({
            **label, "estimate": "paired_speedup", "metric": "throughput_tps",
            "point": round(pr["point"], 6),
            "ci_lo": round(pr["lo"], 6), "ci_hi": round(pr["hi"], 6),
            "t_ci_lo": "", "t_ci_hi": "",
            "n_documents": pr["n_documents"],
            "verdict": clears(pr, 1.0),
            "note": f"dropped {dropped}" if dropped else "",
        })
    return out


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--results", required=True)
    p.add_argument("--group-by", nargs="+", default=["level_id"])
    p.add_argument("--spec-arm-column", default="spec_steps")
    p.add_argument("--spec-arm", default="4")
    p.add_argument("--target-arm", default="0")
    p.add_argument("--metrics", nargs="+",
                   default=["acceptance_rate", "throughput_tps", "target_ppl"])
    p.add_argument("--seed", type=int, default=20261006)
    p.add_argument("--out", default=None)
    args = p.parse_args()

    path = Path(args.results)
    with path.open() as fh:
        rows = [r for r in csv.DictReader(fh) if r.get("status") == "ok"]

    res = summarise(rows, args.group_by, args.metrics, args.spec_arm_column,
                    args.spec_arm, args.target_arm, seed=args.seed)

    out_path = Path(args.out) if args.out else path.with_name(
        path.stem + "_doc_intervals.csv")
    fields = list(args.group_by) + [
        "estimate", "metric", "point", "ci_lo", "ci_hi", "t_ci_lo", "t_ci_hi",
        "n_documents", "verdict", "note"]
    with out_path.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for r in res:
            w.writerow(r)

    print(f"{'group':<30} {'estimate':<14} {'metric':<16} "
          f"{'point':>9} {'bootstrap ci':>24} {'t ci':>24} {'n':>3}  verdict")
    for r in res:
        lbl = "/".join(str(r[k]) for k in args.group_by)
        boot = f"[{r['ci_lo']}, {r['ci_hi']}]"
        tci = (f"[{r['t_ci_lo']}, {r['t_ci_hi']}]"
               if r["t_ci_lo"] != "" else "-")
        print(f"  {lbl:<28} {r['estimate']:<14} {r['metric']:<16} "
              f"{r['point']:>9} {boot:>24} {tci:>24} {r['n_documents']:>3}  "
              f"{r['verdict']}")
    print(f"\nwrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

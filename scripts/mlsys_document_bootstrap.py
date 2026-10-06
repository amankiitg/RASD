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
    clears, document_mean_bootstrap, paired_bootstrap, paired_ratio_t_interval,
    t_cluster_interval,
)


def _f(x):
    s = str(x).strip()
    if not s:
        return None
    try:
        return float(s)
    except ValueError:
        return None


def _per_document_values(rows: list[dict], metric: str, arm_column: str):
    """One value per document, averaging rows within the same arm.

    Returns `(values, saturated)`. Saturation is the share of rounds fully
    accepted, averaged over documents; a rung above 0.90 is flagged because a
    saturated rung cannot support a comparative claim.
    """
    by_doc: dict[str, list[float]] = {}
    sat: list[float] = []
    for r in rows:
        v = _f(r.get(metric))
        if v is None:
            continue
        key = (str(r.get("doc_id", "")), str(r.get(arm_column, "")))
        by_doc.setdefault(key, []).append(v)
        s = _f(r.get("full_accept_share"))
        if s is not None:
            sat.append(s)
    vals = [sum(v) / len(v) for v in by_doc.values()]
    mean_sat = round(sum(sat) / len(sat), 6) if sat else ""
    return vals, mean_sat


def _per_document_first(rows: list[dict], metric: str, arm_column: str,
                        arm: str) -> dict[str, float]:
    """One value per document for one arm, averaging duplicate rows."""
    by_doc: dict[str, list[float]] = {}
    for r in rows:
        if str(r.get(arm_column, "")) != arm:
            continue
        v = _f(r.get(metric))
        if v is None:
            continue
        by_doc.setdefault(str(r.get("doc_id", "")), []).append(v)
    return {d: sum(v) / len(v) for d, v in by_doc.items() if v}


def group_rows(rows: list[dict], keys: list[str]) -> dict[tuple, list[dict]]:
    out: dict[tuple, list[dict]] = {}
    for r in rows:
        out.setdefault(tuple(str(r.get(k, "")) for k in keys), []).append(r)
    return out


def summarise(rows: list[dict], keys: list[str], metrics: list[str],
              arm_column: str, spec_arm: str, target_arm: str,
              seed: int = 20261006,
              ratio_metric: str = "decode_tps") -> list[dict]:
    out: list[dict] = []
    for gkey, grows in sorted(group_rows(rows, keys).items()):
        label = dict(zip(keys, gkey))
        for metric in metrics:
            vals, saturated = _per_document_values(grows, metric, arm_column)
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
                "saturated": saturated,
                # No verdict here. A 1.0 threshold is meaningful only for the
                # paired RATIO; applying `clears(..., 1.0)` to a raw throughput
                # mean in tok/s produced spurious "above" verdicts.
                "verdict": "",
            })

        # One row per (document, arm). A plain dict keyed on doc_id kept only
        # the LAST row, so with more than one row per document the paired
        # speedup silently used an arbitrary subset while the marginal means
        # above averaged them all.
        # Primary ratio on the pre-registered decode-only rate; the end-to-end
        # ratio is reported beside it for the documents that have a full-length
        # target-only partner. See the plan's revision block.
        spec = _per_document_first(grows, ratio_metric, arm_column, spec_arm)
        targ = _per_document_first(grows, ratio_metric, arm_column, target_arm)
        paired = sorted(set(spec) & set(targ))
        if not paired:
            continue
        a, b = [spec[d] for d in paired], [targ[d] for d in paired]
        pr = paired_bootstrap(a, b, kind="ratio", seed=seed)
        tr = paired_ratio_t_interval(a, b)
        # The plan's payoff rule: the interval must lie entirely below 1.0. It
        # also requires that a DISAGREEMENT with the robust interval be reported
        # as not reached, so the two are compared rather than one being quoted.
        v_boot, v_t = clears(pr, 1.0), clears(tr, 1.0)
        verdict = v_boot if v_boot == v_t else "inconclusive"
        out.append({
            **label, "estimate": "paired_speedup", "metric": ratio_metric,
            "point": round(pr["point"], 6),
            "ci_lo": round(pr["lo"], 6), "ci_hi": round(pr["hi"], 6),
            "t_ci_lo": round(tr["lo"], 6), "t_ci_hi": round(tr["hi"], 6),
            "n_documents": pr["n_documents"],
            "verdict": verdict,
            "note": ("bootstrap and t interval disagree"
                     if v_boot != v_t else ""),
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
                   default=["acceptance_rate", "decode_tps", "throughput_tps",
                            "target_ppl"])
    p.add_argument("--ratio-metric", default="decode_tps",
                   help="Pre-registered primary metric for the paired ratio "
                        "(decode_tps; throughput_tps for the end-to-end view)")
    p.add_argument("--seed", type=int, default=20261006)
    p.add_argument("--out", default=None)
    args = p.parse_args()

    path = Path(args.results)
    with path.open() as fh:
        rows = [r for r in csv.DictReader(fh) if r.get("status") == "ok"]

    res = summarise(rows, args.group_by, args.metrics, args.spec_arm_column,
                    args.spec_arm, args.target_arm, seed=args.seed,
                    ratio_metric=args.ratio_metric)
    # The end-to-end ratio is reported BESIDE the primary one, so a reader can
    # see the prefill weighting rather than take it on trust.
    other = "throughput_tps" if args.ratio_metric != "throughput_tps" else "decode_tps"
    for r in summarise(rows, args.group_by, args.metrics, args.spec_arm_column,
                       args.spec_arm, args.target_arm, seed=args.seed,
                       ratio_metric=other):
        if r["estimate"] == "paired_speedup":
            r["estimate"] = f"paired_speedup_{other}"
            r["verdict"] = ""
            res.append(r)

    out_path = Path(args.out) if args.out else path.with_name(
        path.stem + "_doc_intervals.csv")
    fields = list(args.group_by) + [
        "estimate", "metric", "point", "ci_lo", "ci_hi", "t_ci_lo", "t_ci_hi",
        "n_documents", "saturated", "verdict", "note"]
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

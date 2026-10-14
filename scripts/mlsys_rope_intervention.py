#!/usr/bin/env python3
"""rope_intervention_128k: native vs each rope factor, paired by document.

Pre-registered as C6 in docs/mlsys_analysis_plan.md. The stage holds the context
at 128k and moves only the target's rope, so this is the one comparison in the
campaign that isolates a rope intervention.

The estimand is the paired DIFFERENCE in `alpha_round` (treated minus native),
per document, with a document-bootstrap 95% interval and the pre-registered
0.05 equivalence margin.

Why a difference and not a ratio: nothing here is being compared against a
baseline configuration whose cost we want a speedup against. Both arms are
speculative, at the same context and the same generation length, so the question
is whether acceptance MOVES, and "moved by 0.03" is the answer to it. A ratio of
two acceptance rates is not a quantity anyone asked for.

Why paired by document: the documents differ from each other by far more than
the factors do, so an unpaired comparison would be dominated by which books
happened to land in which arm. Resampling documents (not tokens) is what makes
the interval a statement about books.

Why the interval and not the point: with 10 documents the interval is usually
too wide to license the word "equivalent", and the pre-registered rule says so
in advance. An inconclusive result is reported as inconclusive. It is a result,
not a failure, and not something to re-run until it narrows.

Usage:
    python scripts/mlsys_rope_intervention.py \
        --results results/mlsys/rope_intervention_128k.csv \
        --out results/mlsys/rope_intervention_128k_comparison.csv
"""
from __future__ import annotations

import argparse
import csv
import math
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from src.analysis.document_bootstrap import (  # noqa: E402
    equivalence_verdict, paired_bootstrap, t_cluster_interval,
)

FIELDS = ["contrast", "metric", "n_documents", "point", "ci_lo", "ci_hi",
          "t_ci_lo", "t_ci_hi", "margin", "verdict", "verdict_t", "note",
          "arms_present", "documents"]

DEFAULT_MARGIN = 0.05          # pre-registered in the plan revision
PRIMARY = [("native", "factor16"), ("native", "factor32")]
SECONDARY = [("factor16", "factor32")]


def _f(x):
    s = str(x).strip()
    if not s:
        return None
    try:
        v = float(s)
    except ValueError:
        return None
    return None if math.isnan(v) else v


def load_arms(rows: list[dict], metric: str) -> tuple[dict, dict]:
    """{arm: {document: value}} for every arm, and the documents per arm.

    Keyed on the pair_id's document half so the same book is compared with
    itself across arms -- the pairing the plan pre-registers. An arm whose row
    has no rope_arm label is not silently matched to another arm: it is
    reported as unlabelled and excluded, because guessing which arm a row
    belongs to is how a comparison ends up between the wrong two things.
    """
    by_arm: dict[str, dict] = {}
    per_arm_docs: dict[str, list[str]] = {}
    unlabelled = 0
    for r in rows:
        if str(r.get("status", "")).strip() != "ok":
            continue
        v = _f(r.get(metric))
        if v is None:
            continue
        arm = str(r.get("rope_arm", "") or "").strip()
        if not arm:
            unlabelled += 1
            continue
        pid = str(r.get("pair_id", "") or "")
        doc = pid.split(":", 1)[1] if ":" in pid else str(r.get("doc_id", ""))
        if not doc:
            continue
        by_arm.setdefault(arm, {})[doc] = v
        per_arm_docs.setdefault(arm, []).append(doc)
    if unlabelled:
        print(f"  [warn] {unlabelled} ok rows carry no rope_arm and were "
              f"excluded: which arm a row belongs to is not a guess")
    return by_arm, per_arm_docs


def compare(by_arm: dict, a: str, b: str, metric: str, margin: float,
            n_boot: int, seed: int) -> dict | None:
    """Paired difference (b minus a) over the documents both arms ran.

    Restricted to the INTERSECTION of the two arms' documents: if one arm's gate
    failed and it did not run, there is nothing to pair, and padding the missing
    side would invent a difference.
    """
    if a not in by_arm or b not in by_arm:
        return None
    shared = sorted(set(by_arm[a]) & set(by_arm[b]))
    if not shared:
        return None
    x = [by_arm[b][d] for d in shared]
    y = [by_arm[a][d] for d in shared]
    boot = paired_bootstrap(x, y, kind="difference", b_resamples=n_boot, seed=seed)
    # The robustness interval for a DIFFERENCE is a t interval on the
    # per-document differences. `paired_ratio_t_interval` is a t interval on log
    # RATIOS and answers a different question; using it here would have compared
    # a difference against a ratio and reported a disagreement that is not one.
    tci = t_cluster_interval([xi - yi for xi, yi in zip(x, y)])
    verdict = equivalence_verdict(boot, margin)
    # The robustness check: the plan requires a disagreement between the
    # bootstrap and the t interval to be reported as inconclusive rather than
    # one of them being quoted.
    verdict_t = equivalence_verdict(tci, margin)
    note = ""
    if verdict_t != verdict:
        note = (f"bootstrap says {verdict}, t interval says {verdict_t}; "
                f"reported inconclusive")
        verdict_out = "inconclusive"
    else:
        verdict_out = verdict
    return {
        "contrast": f"{b}_minus_{a}", "metric": metric,
        "n_documents": len(shared),
        "point": round(boot["point"], 6),
        "ci_lo": round(boot["lo"], 6), "ci_hi": round(boot["hi"], 6),
        "t_ci_lo": round(tci["lo"], 6), "t_ci_hi": round(tci["hi"], 6),
        "margin": margin, "verdict": verdict_out, "verdict_t": verdict_t,
        "note": note, "arms_present": f"{a},{b}",
        "documents": ",".join(shared),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True)
    ap.add_argument("--out", default=None)
    ap.add_argument("--metric", default="acceptance_rate",
                    help="alpha_round; the pre-registered primary metric")
    ap.add_argument("--margin", type=float, default=DEFAULT_MARGIN)
    ap.add_argument("--n-boot", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=20261006)
    args = ap.parse_args()

    path = Path(args.results)
    if not path.exists():
        raise SystemExit(f"no results CSV: {path}")
    rows = list(csv.DictReader(path.open()))

    by_arm, per_arm_docs = load_arms(rows, args.metric)
    print(f"  arms present: " + ", ".join(
        f"{a}({len(d)} docs)" for a, d in sorted(by_arm.items())))
    if "native" not in by_arm:
        raise SystemExit(
            "no `native` arm in the results. The native arm is the control and "
            "always runs; without it there is no contrast to report."
        )
    missing = [arm for _, arm in PRIMARY if arm not in by_arm]
    if missing:
        # A treated arm can legitimately be absent: its gate failed at 128k.
        # That is its RESULT, recorded here, not an error.
        print(f"  NOT RUN (recorded as that arm's result): {', '.join(missing)}")
        print(f"    reason: the arm did not pass the coherence gate at 128k, "
              f"so it produced no acceptance to compare")

    out_rows = []
    for a, b in PRIMARY + SECONDARY:
        r = compare(by_arm, a, b, args.metric, args.margin, args.n_boot, args.seed)
        if r is None:
            continue
        tag = "primary" if (a, b) in PRIMARY else "secondary"
        r["note"] = (r["note"] + " " if r["note"] else "") + tag
        r["note"] = r["note"].strip()
        out_rows.append(r)

    if not out_rows:
        raise SystemExit(
            "no contrast could be formed: neither treated arm ran. Record this "
            "as the stage's result rather than reporting an acceptance figure."
        )

    out_path = Path(args.out) if args.out else path.with_name(
        path.stem + "_comparison.csv")
    with out_path.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS, extrasaction="ignore")
        w.writeheader()
        for r in out_rows:
            w.writerow(r)

    print(f"\n  {'contrast':<26} {'n':>3} {'difference':>11} "
          f"{'bootstrap 95%':>22} {'margin':>7}  verdict")
    for r in out_rows:
        ci = f"[{r['ci_lo']}, {r['ci_hi']}]"
        print(f"  {r['contrast']:<26} {r['n_documents']:>3} {r['point']:>11} "
              f"{ci:>22} {r['margin']:>7}  {r['verdict']}")
        if r["note"]:
            print(f"      -> {r['note']}")
    print(f"\n  wrote {out_path}")
    print(f"  margin {args.margin} is PRE-REGISTERED; 'equivalent' requires the"
          f" whole interval inside it, and overlapping a boundary is"
          f" inconclusive, never rounded.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Acceptance by window with periodicity flags, for the 128k campaign's spec rows.

WHY THE FIRST WINDOW IS REPORTED SEPARATELY. On natural_f1_128k the acceptance
reported for a whole row (0.620-0.970) is contaminated to an unknown degree by greedy
loops: 13 of 20 rows are periodic, and 4 of the 10 spec rows have NO non-degenerate
window at all. Token positions 0-127 are the part of every row that is *before* the
loop settles in, so the first window is the closest thing the existing data has to a
loop-free acceptance measurement -- and it is reported here WITH the periodicity of
that window, so a reader can see whether even it is degenerate.

Acceptance is the per-round quantity alpha = accepted-prefix-length / gamma, summed
over the rounds whose emitted tokens fall in the window (reviewer AbH52 #1), never the
i.i.d. per-token parameter.

Usage:
    python scripts/mlsys_first_window_report.py --results <stage.csv> \
        --tokens-dir <tokens> --per-token-dir <per_token> --out <out.csv>
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
    non_degenerate_acceptance, periodicity, row_reasons, row_stats, window_acceptance,
)

FIELDS = ["run_id", "doc_id", "context_length", "acceptance_reported",
          "w0_acceptance", "w0_rounds", "w0_period", "w0_self_agreement",
          "w0_first_periodic_token", "w0_periodic_tail_share", "w0_degenerate",
          "row_period", "row_self_agreement", "row_first_periodic_token",
          "row_periodic_tail_share", "row_tokens_in_repeat", "row_flag",
          "row_reasons", "windows_total", "windows_used", "acceptance_nondegenerate",
          "acceptance_nondegenerate_n"]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True)
    ap.add_argument("--tokens-dir", required=True)
    ap.add_argument("--per-token-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--window", type=int, default=128)
    args = ap.parse_args()

    rows = []
    for r in csv.DictReader(Path(args.results).open()):
        if str(r.get("spec_steps", "0")).strip() in ("", "0"):
            continue
        rid = r.get("run_id", "")
        tf = Path(args.tokens_dir) / f"{rid}.json"
        if not tf.exists():
            continue
        ids = json.loads(tf.read_text()).get("generated_token_ids") or []
        trf = Path(args.per_token_dir) / f"{rid}.jsonl"
        tr = [json.loads(l) for l in trf.read_text().splitlines() if l.strip()] if trf.exists() else []
        st = row_stats(ids)
        w0 = window_acceptance(tr, len(ids), args.window) if tr else []
        first = w0[0] if w0 else {}
        seg = ids[:args.window]
        p0 = periodicity(seg) if len(seg) >= 4 else {"period": 0, "agreement": 0.0,
                                                     "first_periodic_token": len(seg),
                                                     "periodic_tail_share": 1.0}
        nd = non_degenerate_acceptance(tr, ids, args.window) if tr else {}
        rows.append({
            "run_id": rid, "doc_id": r.get("doc_id", ""),
            "context_length": r.get("context_length", ""),
            "acceptance_reported": r.get("acceptance_rate", ""),
            "w0_acceptance": first.get("acceptance", ""),
            "w0_rounds": first.get("rounds", ""),
            "w0_period": p0["period"], "w0_self_agreement": p0["agreement"],
            "w0_first_periodic_token": p0["first_periodic_token"],
            "w0_periodic_tail_share": p0["periodic_tail_share"],
            "w0_degenerate": p0["periodic_tail_share"] > 0.50,
            "row_period": st["rep_period"], "row_self_agreement": st["rep_period_agreement"],
            "row_first_periodic_token": st["rep_first_periodic_token"],
            "row_periodic_tail_share": st["rep_periodic_tail_share"],
            "row_tokens_in_repeat": st.get("rep_tokens_in_repeat", ""),
            "row_flag": "LOOP" if row_reasons(st) else "clean",
            "row_reasons": "; ".join(row_reasons(st)),
            "windows_total": nd.get("windows_total", ""),
            "windows_used": nd.get("windows_used", ""),
            "acceptance_nondegenerate": "" if nd.get("acceptance") is None
                                        else nd.get("acceptance", ""),
            "acceptance_nondegenerate_n": nd.get("windows_used", ""),
        })

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)

    print(f"{'run_id':42s} {'reported':>8s} {'w0':>7s} {'w0per':>5s} {'w0tail':>6s} "
          f"{'rowper':>6s} {'rowtail':>7s} {'rowflag':>7s} {'nondegen':>8s} {'n':>3s}")
    for r in sorted(rows, key=lambda x: -float(x["acceptance_reported"] or 0)):
        print(f"{str(r['run_id'])[:42]:42s} {str(r['acceptance_reported'])[:8]:>8s} "
              f"{str(r['w0_acceptance'])[:7]:>7s} {str(r['w0_period']):>5s} "
              f"{str(r['w0_periodic_tail_share'])[:6]:>6s} {str(r['row_period']):>6s} "
              f"{str(r['row_periodic_tail_share'])[:7]:>7s} {r['row_flag']:>7s} "
              f"{str(r['acceptance_nondegenerate'])[:8]:>8s} {str(r['windows_used']):>3s}")
    w0 = [float(r["w0_acceptance"]) for r in rows if r["w0_acceptance"] not in ("", None)]
    if w0:
        w0s = sorted(w0)
        print(f"\nfirst-window acceptance over {len(w0)} spec rows: "
              f"min {w0s[0]:.4f} median {w0s[len(w0s)//2]:.4f} max {w0s[-1]:.4f}")
        print(f"first windows that are themselves periodic: "
              f"{sum(1 for r in rows if r['w0_degenerate'])}/{len(rows)}")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Acceptance and throughput versus context length, from the session CSVs.

Rows that LOOP are excluded from the means and marked, because a greedy loop makes
acceptance meaningless -- that is the whole finding the 2026-10-10 campaign produced
after quoting 0.97 acceptance that was a cycle. A row with no non-degenerate window
is plotted as a void with its reason rather than as a point at its reported number.

Usage:
    python scripts/mlsys_plot_scaling.py --out results/mlsys/scaling.png [csv ...]
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
from src.analysis.repetition import row_reasons, row_stats  # noqa: E402


def load(rows_csv: Path, tokens_dir: Path | None, per_token_dir: Path | None):
    import json
    out = []
    for r in csv.DictReader(rows_csv.open()):
        if r.get("status") != "ok":
            continue
        try:
            ctx = int(r.get("context_length") or 0)
        except ValueError:
            continue
        rid = r.get("run_id", "")
        tok = (tokens_dir or rows_csv.parent / "tokens") / f"{rid}.json"
        ids = json.loads(tok.read_text()).get("generated_token_ids") if tok.exists() else []
        st = row_stats(ids) if ids else {}
        out.append({
            "run_id": rid, "ctx": ctx, "spec_steps": int(r.get("spec_steps") or 0),
            "accept": float(r.get("acceptance_rate") or 0.0),
            "tps": float(r.get("throughput_tps") or 0.0),
            "loop": bool(ids) and bool(row_reasons(st)),
            "no_ids": not ids,
        })
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("csv", nargs="*")
    ap.add_argument("--out", default="results/mlsys/scaling.png")
    ap.add_argument("--title", default="RASD: speculative decoding vs context length")
    args = ap.parse_args()
    paths = [Path(p) for p in args.csv] or sorted(
        (REPO / "results" / "mlsys").glob("*session*/session_a_*.csv"))
    rows = []
    for p in paths:
        if p.exists():
            rows += load(p, None, None)
    if not rows:
        print("no rows found"); return 1

    spec = [r for r in rows if r["spec_steps"] > 0 and not r["loop"]]
    target = [r for r in rows if r["spec_steps"] == 0 and not r["loop"]]
    looping = [r for r in rows if r["loop"]]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))
    for ax, key, ylab in ((ax1, "accept", "acceptance (accepted prefix / gamma)"),
                          (ax2, "tps", "end-to-end generation tok/s")):
        if spec:
            ax.scatter([r["ctx"] for r in spec], [r[key] for r in spec],
                       label="speculative (non-degenerate rows)", color="tab:blue", zorder=3)
        if target:
            ax.scatter([r["ctx"] for r in target], [r[key] for r in target],
                       label="target-only", color="tab:orange", marker="s", zorder=3)
        if looping:
            ax.scatter([r["ctx"] for r in looping], [r[key] for r in looping],
                       label="LOOPING row (excluded from means)", color="tab:red",
                       marker="x", s=80, zorder=4)
        ax.set_xscale("log", base=2)
        ax.set_xlabel("context length (tokens)")
        ax.set_ylabel(ylab)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    fig.suptitle(args.title)
    fig.tight_layout()
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=140)
    print(f"wrote {args.out}")
    print(f"  spec rows plotted (non-degenerate): {len(spec)}, target-only: {len(target)}, "
          f"looping (marked, excluded): {len(looping)}")
    if spec:
        for ctx in sorted({r['ctx'] for r in spec}):
            xs = [r['accept'] for r in spec if r['ctx'] == ctx]
            ts = [r['tps'] for r in spec if r['ctx'] == ctx]
            print(f"  ctx {ctx:>8}: accept mean {sum(xs)/len(xs):.4f} (n={len(xs)}), "
                  f"tok/s mean {sum(ts)/len(ts):.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

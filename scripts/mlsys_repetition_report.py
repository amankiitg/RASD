#!/usr/bin/env python3
"""Per-row repetition and non-degenerate acceptance for a stage's outputs.

WHY THIS IS A SEPARATE, OFFLINE STAGE. The 2026-10-10 campaign reported
acceptance 0.620-0.970 for natural_f1_128k and the number was quoted as a result.
Reading the traces afterwards showed that from token 128 on the top rows sit at
acceptance exactly 1.000, in windows where 100% of tokens lie inside a repeated
16-gram -- the model was in a greedy loop and the draft was "predicting" a
constant. The stage CSV recorded no repetition statistic at all, so nothing in the
campaign could have noticed. This reads the sidecars the stage already writes and
reports, per row: the repetition statistics, whether the row is degenerate, and
acceptance recomputed over the NON-DEGENERATE windows only, with the window count
it was computed from.

It changes no measurement: it reads what exists.

Input per row: the token sidecar (`tokens_slim/`, which carries
`generated_token_ids`) and, when present, the per-round trace
(`per_token/`, which carries `n_acc`/`spec_steps`/`n_emitted` per round).
A row with no trace gets its repetition block and no acceptance.

Usage:
    python scripts/mlsys_repetition_report.py \
        --results results/mlsys/natural_f1_128k.csv \
        --tokens-dir results/mlsys/tokens_slim \
        --per-token-dir results/mlsys/per_token \
        --out results/mlsys/natural_f1_128k_repetition.csv
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
    IN_REPEAT_CEILING, PERIODIC_TAIL_CEILING, REPEAT_CEILING,
    VERBATIM_SPAN_CEILING, non_degenerate_acceptance, row_reasons, row_stats,
)

FIELDS = [
    "run_id", "doc_id", "spec_steps", "context_length",
    "tokens_generated", "acceptance_rate", "acceptance_reported",
    "rep_repeat_share", "rep_tok_repeat_share", "rep_tokens_in_repeat",
    "rep_longest_repeat_span", "rep_period", "rep_period_agreement",
    "rep_first_periodic_token", "rep_periodic_tail_share",
    "rep_degenerate", "rep_reasons",
    "windows_total", "windows_used", "windows_dropped", "windows_dropped_list",
    "acceptance_nondegenerate", "status", "error",
]


def _read_json(path: Path):
    try:
        return json.loads(path.read_text())
    except Exception:
        return None


def _trace(d: Path | None, run_id: str) -> list:
    """The per-round trace for one row, from either on-disk shape.

    The engine writes one JSON object per round, so the file on disk is `.jsonl`;
    a `.json` list is accepted too because the loader is not the place to be
    strict about which of the two a given stage emitted.
    """
    if d is None:
        return []
    for name in (f"{run_id}.jsonl", f"{run_id}.json"):
        f = d / name
        if not f.exists():
            continue
        if f.suffix == ".jsonl":
            out = []
            for line in f.read_text().splitlines():
                line = line.strip()
                if not line:
                    continue
                try:
                    out.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
            if out:
                return out
            continue
        blob = _read_json(f)
        if blob is None:
            continue
        if isinstance(blob, list):
            return blob
        for k in ("rounds", "per_token", "trace"):
            if isinstance(blob.get(k), list):
                return blob[k]
    return []


def report(rows: list[dict], tokens_dir: Path, per_token_dir: Path | None,
           window: int) -> list[dict]:
    out = []
    for r in rows:
        rid = r.get("run_id", "")
        rec = {
            "run_id": rid,
            "doc_id": r.get("doc_id", ""),
            "spec_steps": r.get("spec_steps", ""),
            "context_length": r.get("context_length", ""),
            "tokens_generated": r.get("tokens_generated", ""),
            "acceptance_rate": r.get("acceptance_rate", ""),
            "status": r.get("status", ""),
            "error": r.get("error", ""),
        }
        sc = _read_json(tokens_dir / f"{rid}.json") if rid else None
        ids = (sc or {}).get("generated_token_ids") or []
        if not ids:
            rec["rep_reasons"] = ("no token sidecar: repetition cannot be "
                                  "measured, and an unmeasurable row is not a "
                                  "clean row")
            rec["acceptance_reported"] = ""
            out.append(rec)
            continue
        st = row_stats(ids)
        rec.update({k: st[k] for k in st if k.startswith("rep_")})
        rec["rep_reasons"] = "; ".join(row_reasons(st))
        rec["rep_degenerate"] = bool(row_reasons(st))
        tr = _trace(per_token_dir, rid)
        if tr:
            res = non_degenerate_acceptance(tr, ids, window)
            rec["windows_total"] = res["windows_total"]
            rec["windows_used"] = res["windows_used"]
            rec["windows_dropped"] = res["windows_dropped"]
            rec["windows_dropped_list"] = ",".join(str(i) for i in res["dropped"])
            rec["acceptance_nondegenerate"] = (
                "" if res["acceptance"] is None else res["acceptance"])
            rec["acceptance_reported"] = rec["acceptance_nondegenerate"]
        else:
            rec["acceptance_reported"] = ""
        out.append(rec)
    return out


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--results", required=True)
    p.add_argument("--tokens-dir", required=True)
    p.add_argument("--per-token-dir", default=None)
    p.add_argument("--window", type=int, default=128)
    p.add_argument("--out", default=None)
    args = p.parse_args()

    results = Path(args.results)
    rows = list(csv.DictReader(results.open()))
    rep = report(rows, Path(args.tokens_dir),
                 Path(args.per_token_dir) if args.per_token_dir else None,
                 args.window)

    out_path = Path(args.out) if args.out else results.with_name(
        results.stem + "_repetition.csv")
    with out_path.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS, extrasaction="ignore")
        w.writeheader()
        for r in rep:
            w.writerow(r)

    print(f"ceilings: word-n-gram repeat {REPEAT_CEILING:.2f}, "
          f"tokens-in-repeat {IN_REPEAT_CEILING:.2f}, "
          f"verbatim span {VERBATIM_SPAN_CEILING}, "
          f"periodic tail {PERIODIC_TAIL_CEILING:.2f}")
    print(f"{'run_id':44s} {'accept':>7s} {'nondegen':>9s} {'nwin':>5s} "
          f"{'lspan':>6s} {'period':>7s} {'tail':>6s} {'inrep':>6s} flag")
    for r in rep:
        print(f"{str(r['run_id'])[:44]:44s} "
              f"{str(r.get('acceptance_rate',''))[:7]:>7s} "
              f"{str(r.get('acceptance_nondegenerate',''))[:9]:>9s} "
              f"{str(r.get('windows_used','')):>5s} "
              f"{str(r.get('rep_longest_repeat_span','')):>6s} "
              f"{str(r.get('rep_period','')):>7s} "
              f"{str(r.get('rep_periodic_tail_share','')):>6s} "
              f"{str(r.get('rep_tokens_in_repeat','')):>6s} "
              f"{'LOOP' if r.get('rep_degenerate') else 'clean'}")
    n_loop = sum(1 for r in rep if r.get("rep_degenerate"))
    with_acc = [r for r in rep
                if r.get("acceptance_nondegenerate") not in ("", None)]
    print(f"\n{n_loop}/{len(rep)} rows degenerate; "
          f"{len(with_acc)} rows have a non-degenerate acceptance")
    if with_acc:
        vals = sorted(float(r["acceptance_nondegenerate"]) for r in with_acc)
        print(f"non-degenerate acceptance: min {vals[0]:.3f} "
              f"median {vals[len(vals)//2]:.3f} max {vals[-1]:.3f}")
    print(f"wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

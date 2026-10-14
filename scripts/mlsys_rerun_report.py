#!/usr/bin/env python3
"""Part 3 tables: the candidate engine against the old engine, per context, from ROWS.

Reads only committed/pulled evidence, no GPU. Every number is recomputed from a row:
`tokens_generated / time_sec` for the end-to-end rate, `(time_sec - ttft) / (tokens - 1)` for
the decode-only rate, and the periodicity from the row's own generated-token sidecar using the
project's `src/analysis/repetition.py` -- one implementation, so this table and the CSV column
cannot drift apart.

Usage:
  python3 scripts/mlsys_rerun_report.py --new results/mlsys/rerun_r1 results/mlsys/rerun_r2 \
      --old results/mlsys/session_a results/mlsys/seeds_ablations
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import statistics as st
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.analysis.repetition import periodicity   # noqa: E402

PROTOCOL_TEMP = "0.7"


def _f(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _load(dirs, pattern="*.csv"):
    """Every row in every CSV under the given directories."""
    out = []
    for d in dirs:
        for p in sorted(glob.glob(os.path.join(d, pattern))):
            try:
                rows = list(csv.DictReader(open(p)))
            except Exception:                                 # noqa: BLE001
                continue
            for r in rows:
                r["_csv"] = os.path.basename(p)
                r["_dir"] = d
                out.append(r)
    return out


def _periodicity_for(row, dirs):
    """The row's dominant cycle, from its own token sidecar. None when it has no sidecar."""
    rid = row.get("run_id")
    if not rid:
        return None
    for d in dirs:
        for cand in glob.glob(os.path.join(d, "tokens", f"{rid}*.json")):
            try:
                blob = json.load(open(cand))
            except Exception:                                 # noqa: BLE001
                continue
            ids = blob.get("generated_token_ids") or blob.get("generated_ids")
            if not ids and isinstance(blob.get("metrics"), dict):
                ids = blob["metrics"].get("generated_token_ids")
            if ids:
                return periodicity(ids)
    return None


def _ok(rows):
    return [r for r in rows if r.get("status") == "ok"]


def _stats(rows, dirs):
    """Aggregate ok rows: mean decode-only s/tok, mean e2e tok/s, acceptance, periodicity."""
    recs = []
    for r in _ok(rows):
        tok = _f(r.get("tokens_generated"))
        t = _f(r.get("time_sec"))
        ttft = _f(r.get("ttft_ms"))
        if not tok or t is None:
            continue
        recs.append({
            "ctx": int(_f(r.get("context_length")) or 0),
            "spec": int(_f(r.get("spec_steps")) or 0),
            "seed": r.get("seed"), "book": r.get("doc_id") or r.get("book"),
            "decode": (t - (ttft or 0) / 1000.0) / (tok - 1) if tok > 1 else None,
            "e2e": tok / t, "acc": _f(r.get("acceptance_rate")),
            "ttft": ttft, "mem": _f(r.get("gpu_peak_mem_mb")),
            "temp": r.get("temperature"),
            "engine": r.get("ring_decode_mode"), "chunk": r.get("kv_block_size"),
            "expand": r.get("ring_expand_after_rotation"),
            "commit": (r.get("git_commit") or "")[:7],
            "periodicity": _periodicity_for(r, dirs),
            "csv": r["_csv"],
        })
    return recs


def _mean(vals):
    vals = [v for v in vals if v is not None]
    return st.mean(vals) if vals else None


def _fmt(v, n=3):
    return "-" if v is None else f"{v:.{n}f}"


def _md(headers, rows):
    out = ["| " + " | ".join(headers) + " |", "|" + "|".join(["---"] * len(headers)) + "|"]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(out)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--new", nargs="+", required=True)
    ap.add_argument("--old", nargs="*", default=[])
    a = ap.parse_args(argv)

    new = _stats(_load(a.new), a.new)
    old = _stats(_load(a.old), a.old) if a.old else []

    print("# Part 3 — the candidate engine (split-KV) against the old engine\n")
    if not new:
        print("(no ok rows found under --new)\n")

    # Engine provenance, first: a row that cannot prove its engine is void (S2).
    print("## Engine provenance (stop condition S2)\n")
    prov = {}
    for r in new:
        prov.setdefault((r["engine"], r["chunk"], r["expand"], r["commit"]), 0)
        prov[(r["engine"], r["chunk"], r["expand"], r["commit"])] += 1
    print(_md(["ring_decode_mode", "kv_block_size", "expand_after", "git_commit", "rows"],
              [[k[0], k[1], k[2], k[3], v] for k, v in sorted(prov.items(), key=lambda kv: str(kv[0]))]))
    print()

    # Per context, candidate vs old.
    print("## Per context: candidate vs old engine\n")
    tab = []
    for ctx in sorted({r["ctx"] for r in new} | {r["ctx"] for r in old}):
        for spec in (4, 0):
            nn = [r for r in new if r["ctx"] == ctx and r["spec"] == spec]
            oo = [r for r in old if r["ctx"] == ctx and r["spec"] == spec
                  and str(r.get("temp")) == PROTOCOL_TEMP]
            if not nn and not oo:
                continue
            nd, od = _mean([r["decode"] for r in nn]), _mean([r["decode"] for r in oo])
            tab.append([
                ctx, "spec" if spec else "target-only", len(nn), len(oo),
                _fmt(nd, 4), _fmt(od, 4),
                (_fmt(od / nd, 3) + "x") if (nd and od) else "-",
                _fmt(_mean([r["e2e"] for r in nn]), 4), _fmt(_mean([r["e2e"] for r in oo]), 4),
                _fmt(_mean([r["acc"] for r in nn if spec]), 4),
                _fmt(_mean([r["ttft"] for r in nn]) / 1000.0 if any(r["ttft"] for r in nn) else None, 1),
                _fmt(_mean([r["mem"] for r in nn]), 0),
            ])
    print(_md(["ctx", "arm", "n new", "n old", "new decode s/tok", "old decode s/tok",
               "speedup new vs old", "new e2e tok/s", "old e2e tok/s", "acceptance",
               "ttft s", "peak mem MB"], tab))
    print()

    # Spec speedup against the SAME engine's target-only arm -- the paper's metric.
    print("## Spec vs target-only, within the candidate engine\n")
    tab = []
    for ctx in sorted({r["ctx"] for r in new}):
        s = [r for r in new if r["ctx"] == ctx and r["spec"] == 4]
        t = [r for r in new if r["ctx"] == ctx and r["spec"] == 0]
        if not s or not t:
            continue
        sd, td = _mean([r["decode"] for r in s]), _mean([r["decode"] for r in t])
        se, te = _mean([r["e2e"] for r in s]), _mean([r["e2e"] for r in t])
        tab.append([ctx, len(s), len(t), _fmt(sd, 4), _fmt(td, 4), _fmt(td / sd, 3) + "x",
                    _fmt(se, 4), _fmt(te, 4),
                    # BOTH RATES ARE "PER SECOND", so the e2e speedup is se/te -- the other
                    # direction is a slowdown expressed as a speedup.
                    _fmt(se / te, 3) + "x"])
    print(_md(["ctx", "n spec", "n target", "spec decode s/tok", "target decode s/tok",
               "decode speedup", "spec e2e tok/s", "target e2e tok/s", "e2e speedup"], tab))
    print()

    # The seed spread at 1M.
    print("## Seed spread at 1M (the headline)\n")
    seeds = {}
    for r in new:
        if r["ctx"] >= 1048576 and r["spec"] == 4:
            seeds.setdefault(str(r["seed"]), []).append(r)
    tab = []
    for seed, rs in sorted(seeds.items()):
        tab.append([seed, len(rs), _fmt(_mean([r["decode"] for r in rs]), 4),
                    _fmt(_mean([r["e2e"] for r in rs]), 4),
                    _fmt(_mean([r["acc"] for r in rs]), 4),
                    _fmt(_mean([r["mem"] for r in rs]), 0)])
    if tab:
        print(_md(["seed", "books", "decode s/tok", "e2e tok/s", "acceptance", "peak mem MB"], tab))
        allr = [r for rs in seeds.values() for r in rs]
        accs = [r["acc"] for r in allr if r["acc"] is not None]
        dec = [r["decode"] for r in allr if r["decode"] is not None]
        if accs:
            print(f"\nacceptance over {len(accs)} rows: mean {st.mean(accs):.4f} "
                  f"min {min(accs):.4f} max {max(accs):.4f} sd {st.pstdev(accs):.4f}")
        if dec:
            print(f"decode s/tok over {len(dec)} rows: mean {st.mean(dec):.4f} "
                  f"min {min(dec):.4f} max {max(dec):.4f}")
    print()

    # Periodicity, per row: the brief asks for it on every row.
    print("## Periodicity per row (src/analysis/repetition.py)\n")
    tab = []
    degen = 0
    for r in new:
        p = r["periodicity"]
        if not p:
            continue
        tail = p.get("periodic_tail_share", 0.0)
        flag = "**loop**" if tail >= 0.5 else ""
        if tail >= 0.5:
            degen += 1
        tab.append([r["ctx"], "spec" if r["spec"] else "target", r["seed"], r["book"],
                    p.get("period"), _fmt(p.get("agreement"), 3), _fmt(tail, 3), flag])
    if tab:
        print(_md(["ctx", "arm", "seed", "book", "period", "agreement", "periodic tail", "flag"], tab))
        print(f"\n{degen} row(s) with a periodic tail >= 0.50 (the project's degenerate threshold).")
    else:
        print("(no token sidecars found: the periodicity check needs generated_token_ids)")
    return 0


if __name__ == "__main__":
    sys.exit(main())

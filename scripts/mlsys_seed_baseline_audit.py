#!/usr/bin/env python3
"""The old-engine 1M seed baseline, recomputed from rows the seed bug did not touch.

THE BUG. `build_run_configs` merges a level into the defaults and then sets
`run["seed"] = seed` from the DEFAULTS' `seeds:` list (run_experiment.py:590-612). A level that
declares its own `seeds:` therefore has it recorded in the run dict and then overwritten, so the
stage runs the default seed. Exactly two stages ever did this -- both in
`configs/mlsys_seeds_ablations.yml` -- and both ran **seed 42** while their run ids said
`s123` and `s456`.

WHAT THAT BROKE. RESULTS_OCT 10.3 and CLAIMS C18a read those rows as real seed-123/456
replicates and concluded the old engine's 1M acceptance was identical across seeds, i.e. that
the old engine's seed replicates were duplicates of seed 42 rather than seed variation. The
rows *are* duplicates -- of seed 42 -- but that is an artefact of the bug, not a property of
the engine, and the conclusion drawn from it ("these are the first real 1M seed replicates this
campaign has") was wrong: a fixed run exists.

THE CORRECTED BASELINE, per seed, from rows that were never affected:
  seed 42   results/mlsys/sampling_session/session_a_SAMP_1M_SPEC.csv   (5 books)
  seed 123  results/mlsys/fp4_seeds/session_a_FP4_SEED_1M_SPEC.csv      (5 books)
  seed 456  results/mlsys/fp4_seeds/session_a_FP4_SEED_1M_SPEC.csv      (3 books)
The fixed run's stage `FP4_SEED_1M_SPEC` carries `seeds: [123, 456]` in the config's DEFAULTS,
which is the form the merge honours, so it was never affected.

EXCLUDED ENTIRELY (listed so the exclusion is auditable):
  results/mlsys/seeds_ablations/session_a_SA_SEED_123_1M_SPEC.csv
  results/mlsys/seeds_ablations/session_a_SA_SEED_456_1M_SPEC.csv
  results/mlsys/seeds_ablations/repetition_SA_SEED_123_1M_SPEC.csv
  results/mlsys/seeds_ablations/repetition_SA_SEED_456_1M_SPEC.csv
  results/mlsys/seeds_ablations/tokens_slim/SA_SEED_{123,456}_1M_SPEC_*.json
Their acceptance values reproduce seed 42's exactly, which is the fingerprint of the bug.

ALSO NOT POOLABLE, and excluded for a different reason: `results/mlsys/llama2_matrix_multiseed.csv`
and `results/mlsys/pg19_multiseed.csv` carry seeds 123/456 at 1M for `meta-llama/Llama-2-7b-hf`,
a different target family. They are not comparable with the Llama-3 headline and entering them
into a Llama-3 pool would mix model families, which is a worse error than the one being fixed.

Usage:  python3 scripts/mlsys_seed_baseline_audit.py [--new results/mlsys/rerun_r1 results/mlsys/rerun_r2]
"""
from __future__ import annotations

import argparse
import csv
import glob
import statistics as st

OLD_SEED42 = ["results/mlsys/sampling_session/session_a_SAMP_1M_SPEC.csv"]
OLD_SEEDS_123_456 = ["results/mlsys/fp4_seeds/session_a_FP4_SEED_1M_SPEC.csv"]
NEW = ["results/mlsys/rerun_r1/session_a_*.csv", "results/mlsys/rerun_r2/session_a_*.csv"]

EXCLUDED_SEED_BUG = [
    "results/mlsys/seeds_ablations/session_a_SA_SEED_123_1M_SPEC.csv",
    "results/mlsys/seeds_ablations/session_a_SA_SEED_456_1M_SPEC.csv",
    "results/mlsys/seeds_ablations/repetition_SA_SEED_123_1M_SPEC.csv",
    "results/mlsys/seeds_ablations/repetition_SA_SEED_456_1M_SPEC.csv",
    "results/mlsys/seeds_ablations/tokens_slim/SA_SEED_123_1M_SPEC_pg19_train_*.json",
    "results/mlsys/seeds_ablations/tokens_slim/SA_SEED_456_1M_SPEC_pg19_train_*.json",
]
EXCLUDED_OTHER_MODEL = [
    "results/mlsys/llama2_matrix_multiseed.csv  (meta-llama/Llama-2-7b-hf)",
    "results/mlsys/pg19_multiseed.csv           (meta-llama/Llama-2-7b-hf)",
]

MATCHED_3 = ["pg19_train_0", "pg19_train_1", "pg19_train_915"]


def decode(r):
    tok = int(r["tokens_generated"]); t = float(r["time_sec"]); tt = float(r["ttft_ms"])
    return (t - tt / 1000.0) / (tok - 1)


def load(paths, ctx=1048576, spec=4):
    """(seed, book) -> (acceptance, decode). Duplicate reps of a cell are averaged."""
    cells = {}
    for p in paths:
        for f in glob.glob(p):
            for r in csv.DictReader(open(f)):
                if r.get("status") != "ok" or not r.get("ttft_ms") or not r.get("time_sec"):
                    continue
                try:
                    if int(r["context_length"]) != ctx or int(r["spec_steps"]) != spec:
                        continue
                    cells.setdefault((r["seed"], r["doc_id"]), []).append(
                        (float(r["acceptance_rate"]), decode(r)))
                except Exception:
                    continue
    return {k: (st.mean(x[0] for x in v), st.mean(x[1] for x in v)) for k, v in cells.items()}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--new", nargs="+", default=NEW)
    a = ap.parse_args(argv)

    old = {}
    old.update({("42", b): v for b, v in
                {k[1]: v for k, v in load(OLD_SEED42).items()}.items()})
    for (s, b), v in load(OLD_SEEDS_123_456).items():
        old[(s, b)] = v
    new = load(a.new)

    print("# The old-engine 1M seed baseline, corrected\n")
    print("## Excluded from every old-engine pool\n")
    print("Seed bug (their run ids say s123/s456; they ran seed 42):")
    for p in EXCLUDED_SEED_BUG:
        print(f"  - `{p}`")
    print("\nDifferent target family (Llama-2-7b, not poolable with the Llama-3 headline):")
    for p in EXCLUDED_OTHER_MODEL:
        print(f"  - `{p}`")

    print(f"\n{len(old)} old cells over seeds {sorted({s for s, _ in old})}; "
          f"{len(new)} new cells over seeds {sorted({s for s, _ in new})}; "
          f"{len(set(old) & set(new))} matched (seed, book) cells\n")

    print("## Per seed, matched books only\n")
    print("| seed | books | old acceptance | new acceptance | delta | old s/tok | new s/tok | speedup |")
    print("|---|---|---|---|---|---|---|---|")
    po_a, pn_a, po_d, pn_d = [], [], [], []
    for seed in ("42", "123", "456"):
        books = sorted(b for (s, b) in old if s == seed and (s, b) in new)
        if not books:
            continue
        oa = [old[(seed, b)][0] for b in books]; na = [new[(seed, b)][0] for b in books]
        od = [old[(seed, b)][1] for b in books]; nd = [new[(seed, b)][1] for b in books]
        po_a += oa; pn_a += na; po_d += od; pn_d += nd
        print(f"| {seed} | {len(books)} | {st.mean(oa):.4f} | {st.mean(na):.4f} | "
              f"{100*(st.mean(na)-st.mean(oa))/st.mean(oa):+.1f}% | {st.mean(od):.4f} | "
              f"{st.mean(nd):.4f} | {st.mean(od)/st.mean(nd):.3f}x |")
    print(f"| **pooled** | **{len(po_a)}** | **{st.mean(po_a):.4f}** | **{st.mean(pn_a):.4f}** | "
          f"**{100*(st.mean(pn_a)-st.mean(po_a))/st.mean(po_a):+.1f}%** | **{st.mean(po_d):.4f}** | "
          f"**{st.mean(pn_d):.4f}** | **{st.mean(po_d)/st.mean(pn_d):.3f}x** |")

    print("\n## Is the new acceptance inside the old engine's own seed spread?\n")
    o_seed = {s: st.mean([v[0] for (ss, _), v in old.items() if ss == s]) for s in ("42", "123", "456")}
    n_seed = {s: st.mean([v[0] for (ss, _), v in new.items() if ss == s]) for s in ("42", "123", "456")}
    lo, hi = min(o_seed.values()), max(o_seed.values())
    print(f"- old seed means: " + ", ".join(f"{s} **{v:.4f}**" for s, v in sorted(o_seed.items())))
    print(f"- new seed means: " + ", ".join(f"{s} **{v:.4f}**" for s, v in sorted(n_seed.items())))
    print(f"- old range **{lo:.4f} - {hi:.4f}**, i.e. **{100*(hi-lo)/st.mean(o_seed.values()):.1f}%** "
          f"of the mean; new range {min(n_seed.values()):.4f} - {max(n_seed.values()):.4f}")
    o3 = {s: st.mean([old[(s, b)][0] for b in MATCHED_3]) for s in ("42", "123", "456")}
    n3 = {s: st.mean([new[(s, b)][0] for b in MATCHED_3]) for s in ("42", "123", "456")}
    print(f"- on the matched three books (0/1/915): old range {max(o3.values())-min(o3.values()):.4f} "
          f"= **{100*(max(o3.values())-min(o3.values()))/st.mean(o3.values()):.1f}%** of the mean; "
          f"new range {max(n3.values())-min(n3.values()):.4f} = "
          f"{100*(max(n3.values())-min(n3.values()))/st.mean(n3.values()):.1f}%")
    inside = [f"{b} s{s}" for (s, b) in set(old) & set(new) if lo <= new[(s, b)][0] <= hi]
    print(f"- **{len(inside)} of {len(set(old) & set(new))}** new cells fall inside the old range")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

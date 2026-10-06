#!/usr/bin/env python3
"""MLSys experiment program — aggregation and statistics pass.

Consumes whatever the pod session has written into results/mlsys/ and
produces the derived artefacts:

  summary_with_ci.csv        per (group, level) bootstrap CIs on every
                             numeric metric, from the arm CSVs
  dip_test.csv               Hartigan dip test, one row per trace file
  dip_by_context.csv         dip aggregated across seeds per context
  acceptance_accounting.csv  CSV acceptance_rate vs recomputed per-round
                             alpha, i.e. the AbH52 #1 verification
  MASTER_TABLE.txt           old vs new numbers per reviewer/roadmap item

Safe to run at any time: missing inputs produce empty outputs with a
warning rather than a traceback, so it can run before, during and after
the GPU session.

Usage:
    python scripts/mlsys_analysis.py \
        --results-dir results/mlsys \
        --trace-dir   results/mlsys/per_token
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from src.analysis.bootstrap import bootstrap_mean_ci            # noqa: E402
from src.analysis.acceptance import (                          # noqa: E402
    summarize_trace_dir,
    verify_csv_acceptance,
)
from src.analysis.dip_test import (                            # noqa: E402
    aggregate_by_context,
    dip_over_trace_dir,
)

# Metrics the summary aggregates. Anything absent from a CSV is skipped.
METRICS = [
    "tokens_generated", "time_sec", "throughput_tps", "acceptance_rate",
    "mean_latency_ms", "ttft_ms", "gpu_peak_mem_mb", "n_rounds",
]

# CSVs the analyser *writes*; excluded from the inputs it reads. Other
# derived artefacts are additionally filtered by required-column check in
# load_arm_frames, so a new by-product can never crash the aggregation.
#
# pg19_multiseed.csv and bf16_draft_isolation.csv are deliberately NOT
# listed here: they are RAW run output (one row per executed cell), and
# treating them as derived aggregates meant the analyser silently ignored the
# very cells Phase 2 and Phase 3 produced.
OUTPUT_NAMES = {
    "summary_with_ci.csv", "dip_test.csv", "dip_by_context.csv",
    "acceptance_accounting.csv", "trace_summary.csv",
    "acceptance_accounting_seed42_pg19.csv",
    "arm_dip.csv", "vllm_baseline.csv",
    "seed_coverage.csv",
}

# A results CSV must carry these to be aggregatable as an arm frame.
REQUIRED_ARM_COLUMNS = {"run_id", "group", "level_id"}


def load_arm_frames(results_dir: Path) -> pd.DataFrame:
    """Concatenate every input results CSV under results_dir."""
    frames = []
    for p in sorted(results_dir.glob("*.csv")):
        if p.name in OUTPUT_NAMES:
            continue
        try:
            df = pd.read_csv(p)
        except Exception as e:  # noqa: BLE001
            print(f"[warn] could not read {p.name}: {e}")
            continue
        if df.empty:
            continue
        missing = REQUIRED_ARM_COLUMNS - set(df.columns)
        if missing:
            print(f"[skip] {p.name}: not an arm result CSV "
                  f"(missing {sorted(missing)})")
            continue
        df["source_csv"] = p.name
        frames.append(df)
        print(f"[info] loaded {len(df):>3} rows from {p.name}")
    if not frames:
        print("[warn] no arm result CSVs found — nothing to aggregate")
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def summarize_with_ci(df: pd.DataFrame, ci: float = 0.95,
                      seed: int = 42) -> pd.DataFrame:
    """Bootstrap CI on the mean of each metric per (group, level_id)."""
    if df.empty:
        return pd.DataFrame(
            columns=["group", "level_id", "metric", "n", "mean",
                     "ci_lo", "ci_hi", "ci_half"]
        )
    ok = df[df["status"].astype(str).str.lower() == "ok"] if "status" in df else df
    rows = []
    for (group, level), sub in ok.groupby(["group", "level_id"], dropna=False):
        for metric in METRICS:
            if metric not in sub.columns:
                continue
            vals = pd.to_numeric(sub[metric], errors="coerce").dropna().to_numpy()
            if vals.size == 0:
                continue
            mean, lo, hi = bootstrap_mean_ci(vals, ci=ci, seed=seed)
            rows.append({
                "group":   group,
                "level_id": level,
                "metric":  metric,
                "n":       int(vals.size),
                "mean":    mean,
                "ci_lo":   lo,
                "ci_hi":   hi,
                "ci_half": (hi - lo) / 2.0,
                "n_seeds": int(sub["seed"].nunique(dropna=True))
                           if "seed" in sub else 0,
            })
    return pd.DataFrame(rows)


def load_seed42_from_final(final_dir: Path) -> pd.DataFrame:
    """A1: recover the seed-42 rows that live in results/final.

    The Phase 2 multiseed cells only ran seeds 123 and 456 — the seed-42 point
    of each series already existed from the original Phase D / p35d work. Left
    unjoined, a group labelled "3 seeds" carries n=2, which is precisely the
    "largely single-seed" complaint the multiseed run exists to answer.

    Only seed-42 rows are taken; seeds 123/456 must come from the new run, so
    an old value can never masquerade as a new one.
    """
    if not final_dir.is_dir():
        print(f"[warn] {final_dir} not found — seed-42 join skipped")
        return pd.DataFrame()

    frames = []
    for p in sorted(final_dir.glob("*.csv")):
        try:
            df = pd.read_csv(p)
        except Exception as e:  # noqa: BLE001
            print(f"[warn] could not read {p.name}: {e}")
            continue
        if df.empty or "seed" not in df.columns:
            continue
        missing = REQUIRED_ARM_COLUMNS - set(df.columns)
        if missing:
            continue
        keep = df[pd.to_numeric(df["seed"], errors="coerce") == 42].copy()
        if keep.empty:
            continue
        keep["source_csv"] = f"final/{p.name}"
        frames.append(keep)
        print(f"[info] seed-42 join: {len(keep):>2} rows from final/{p.name}")
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def join_seed42(raw: pd.DataFrame, seed42: pd.DataFrame) -> pd.DataFrame:
    """Append seed-42 rows, never overwriting a seed-42 row that already ran.

    If the new run produced its own seed-42 cell (the ARM4-f1 rung does), that
    row is the authority and the historical copy is dropped, so a re-run is
    never silently replaced by a stale number.
    """
    if seed42.empty:
        return raw
    if raw.empty:
        return seed42
    if "run_id" in raw.columns:
        have = set(raw["run_id"].astype(str))
        seed42 = seed42[~seed42["run_id"].astype(str).isin(have)]
    if seed42.empty:
        return raw
    print(f"[info] joined {len(seed42)} historical seed-42 row(s)")
    return pd.concat([raw, seed42], ignore_index=True, sort=False)


# Historical seed-42 rows carry the ORIGINAL Phase-D labels for the same
# series the multiseed config re-ran at 123/456:
#   group "M4",        level "M4_ctx128k"                vs M4_MULTISEED/M4_ctx128k
#   group "M4",        level "RASD_ctx4k_pg19_phaseD"    vs PG19_MULTISEED/PG19_ctx4k
#   group "M4",        level "P35D_ctx1M_pg19"           vs PG19_MULTISEED/PG19_ctx1M
# Joining on run_id succeeds while leaving the seeds split across two buckets,
# so the coverage check would still see n=2. The mapping below is EXPLICIT
# rather than pattern-inferred so it can be audited and cannot silently
# mis-pair two different cells.
_PG19_LEVEL_BY_CTX = {4096: "PG19_ctx4k", 8192: "PG19_ctx8k",
                      1048576: "PG19_ctx1M"}
_SERIES_GROUP_ALIAS = {"M4_MULTISEED": "M4"}


def add_series(df: pd.DataFrame) -> pd.DataFrame:
    """Attach the canonical series key used by the seed-coverage assertion."""
    if df.empty:
        return df
    out = df.copy()
    n = len(out)
    grp = (out["group"].astype(str) if "group" in out
           else pd.Series([""] * n, index=out.index))
    lvl = (out["level_id"].astype(str) if "level_id" in out
           else pd.Series([""] * n, index=out.index))
    rid = (out["run_id"].astype(str) if "run_id" in out else lvl)
    ctx = (pd.to_numeric(out["context_length"], errors="coerce")
           if "context_length" in out else pd.Series([np.nan] * n, index=out.index))
    is_pg19 = (lvl.str.contains("pg19", case=False)
               | rid.str.contains("pg19", case=False)
               | grp.str.contains("pg19", case=False))
    # Target-only rows are a DIFFERENT series from their spec siblings: the
    # baseline exists to divide by, not to pool with. spec_steps==0 marks
    # them, and the historical TARGET_*_pg19_phaseD rows would otherwise be
    # relabelled as spec cells.
    spec = (pd.to_numeric(out["spec_steps"], errors="coerce")
            if "spec_steps" in out else pd.Series([np.nan] * n, index=out.index))

    series = []
    for g, l, c, pg, sp in zip(grp, lvl, ctx, is_pg19, spec):
        ng = _SERIES_GROUP_ALIAS.get(g, g)
        if pg:
            ng = "PG19_MULTISEED"
            if pd.notna(c):
                base = _PG19_LEVEL_BY_CTX.get(int(c))
                if base:
                    l = base if not (pd.notna(sp) and int(sp) == 0) \
                        else f"{base}_targetonly"
        series.append(f"{ng}::{l}")
    out["series"] = series
    return out


def check_seed_coverage(df: pd.DataFrame, expected: int = 3,
                        out_dir: Path | None = None) -> pd.DataFrame:
    """A1: assert every multi-seed group really carries n=3.

    A group that has more than one seed is claiming to be a multi-seed result,
    so it must have exactly `expected` of them. Anything else is printed as a
    failing row rather than quietly reported as a 3-seed CI computed over two
    seeds.
    """
    if df.empty or "seed" not in df.columns:
        print("[warn] seed coverage check skipped (no seed column)")
        return pd.DataFrame()

    # Group by the canonical series key (see add_series), NOT by the raw
    # group/level_id labels, so a historical seed-42 row lands in the same
    # bucket as its 123/456 siblings.
    if "series" in df.columns:
        grouped = df.groupby("series", dropna=False)
    else:
        keys = [c for c in ("group", "level_id") if c in df.columns]
        grouped = (df.groupby(keys, dropna=False) if keys
                   else [("", df)])

    rows = []
    for k, sub in grouped:
        seeds = sorted({int(s) for s in
                        pd.to_numeric(sub["seed"], errors="coerce").dropna()})
        rows.append({
            "series": str(k),
            "n_seeds": len(seeds),
            "seeds": ",".join(str(s) for s in seeds),
        })
    cov = pd.DataFrame(rows)
    if cov.empty:
        return cov

    # Multi-seed = more than one seed present. A single-seed group is reported
    # but not treated as a failure; it is simply not a "3-seed" claim.
    multi = cov[cov["n_seeds"] > 1]
    bad = multi[multi["n_seeds"] != expected]
    if not bad.empty:
        print()
        print("[ERROR] A1 SEED COVERAGE FAILED — these groups claim to be "
              f"multi-seed but do not have n={expected}:")
        print(bad.to_string(index=False))
    else:
        print(f"[ok] seed coverage: {len(multi)} multi-seed group(s), "
              f"all have n={expected}")
    if out_dir is not None:
        cov.to_csv(out_dir / "seed_coverage.csv", index=False)
    return cov


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results-dir", default=str(REPO / "results" / "mlsys"))
    ap.add_argument("--final-dir", default=str(REPO / "results" / "final"),
                    help="A1: source of the historical seed-42 rows that "
                         "complete each multiseed series")
    ap.add_argument("--trace-dir", default=None,
                    help="Per-round trace dir. Defaults to "
                         "<results-dir>/per_token.")
    ap.add_argument("--out-dir", default=None, help="Defaults to --results-dir.")
    ap.add_argument("--ci", type=float, default=0.95)
    ap.add_argument("--boot-pval", action="store_true",
                    help="Use the uniform-null Monte-Carlo dip p-value "
                         "(seed-controlled) instead of Hartigan's table.")
    ap.add_argument("--n-boot", type=int, default=10_000)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    results_dir = Path(args.results_dir)
    out_dir = Path(args.out_dir) if args.out_dir else results_dir
    trace_dir = Path(args.trace_dir) if args.trace_dir else results_dir / "per_token"
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 72)
    print(f"MLSys analysis   results={results_dir}   traces={trace_dir}")
    print("=" * 72)

    # --- 1. bootstrap CIs over the arm CSVs -----------------------------
    df = load_arm_frames(results_dir)
    # A1: complete the multiseed series with their historical seed-42 rows,
    # then verify no group over-claims its seed count.
    df = join_seed42(df, load_seed42_from_final(Path(args.final_dir)))
    df = add_series(df)
    check_seed_coverage(df, out_dir=out_dir)
    summary = summarize_with_ci(df, ci=args.ci, seed=args.seed)
    summary.to_csv(out_dir / "summary_with_ci.csv", index=False)
    print(f"[write] summary_with_ci.csv  ({len(summary)} rows)")

    # --- 2. per-round acceptance accounting (AbH52 #1) ------------------
    acc_rows = []
    trace_summary = summarize_trace_dir(trace_dir)
    if not trace_summary.empty:
        trace_summary.to_csv(out_dir / "trace_summary.csv", index=False)
        print(f"[write] trace_summary.csv  ({len(trace_summary)} traces)")
    for p in sorted(results_dir.glob("*.csv")):
        if p.name in OUTPUT_NAMES:
            continue
        try:
            ver = verify_csv_acceptance(p, trace_dir)
        except Exception as e:  # noqa: BLE001
            print(f"[warn] acceptance check skipped for {p.name}: {e}")
            continue
        if not ver.empty:
            ver["source_csv"] = p.name
            acc_rows.append(ver)
    if acc_rows:
        acc = pd.concat(acc_rows, ignore_index=True)
        acc.to_csv(out_dir / "acceptance_accounting.csv", index=False)
        checked = acc["ok"].notna().sum()
        bad = (acc["ok"] == False).sum()  # noqa: E712
        print(f"[write] acceptance_accounting.csv  "
              f"({checked} verified, {bad} mismatched)")
        if bad:
            print("[ERROR] CSV acceptance_rate does NOT match the per-round "
                  "alpha for some rows — see acceptance_accounting.csv")
    else:
        print("[warn] no acceptance rows verified (no traces yet)")

    # --- 3. dip test on the per-round acceptance distributions ----------
    dip = dip_over_trace_dir(trace_dir, boot_pval=args.boot_pval,
                             n_boot=args.n_boot, seed=args.seed)
    if not dip.empty:
        dip.to_csv(out_dir / "dip_test.csv", index=False)
        agg = aggregate_by_context(dip)
        agg.to_csv(out_dir / "dip_by_context.csv", index=False)
        print(f"[write] dip_test.csv ({len(dip)} runs), "
              f"dip_by_context.csv ({len(agg)} contexts)")
        print()
        print(agg.to_string(index=False))
    else:
        print("[warn] no per-round traces — dip test skipped")

    print()
    print(f"Done. Artefacts in {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

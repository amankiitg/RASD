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
OUTPUT_NAMES = {
    "summary_with_ci.csv", "dip_test.csv", "dip_by_context.csv",
    "acceptance_accounting.csv", "trace_summary.csv",
    "acceptance_accounting_seed42_pg19.csv",
    "pg19_multiseed.csv", "bf16_draft_isolation.csv", "vllm_baseline.csv",
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


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results-dir", default=str(REPO / "results" / "mlsys"))
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

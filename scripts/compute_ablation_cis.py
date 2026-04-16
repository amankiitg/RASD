#!/usr/bin/env python3
"""Compute bootstrap 95% CIs on results/ablations/ablations.csv.

Outputs:
  results/final/ablation_cis.csv  — long-form: (group, level_id, metric, mean, ci_lo, ci_hi, n)

Reads the M3 ablation CSV, drops deterministic short-run outliers (see
src/analysis/metrics.py SHORT_RUN_THRESHOLD), and emits percentile-bootstrap
CIs for throughput_tps + acceptance_rate. Used by Figure 2 and the LaTeX
ablation summary table.

Usage:
    python scripts/compute_ablation_cis.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

# Allow running from project root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.analysis.bootstrap import per_level_ci
from src.analysis.metrics import filter_valid, load_ablations


METRICS = ["throughput_tps", "acceptance_rate", "mean_latency_ms", "gpu_peak_mem_mb"]


def main():
    df = load_ablations()
    valid = filter_valid(df)
    print(f"Loaded {len(df)} rows, {len(valid)} after short-run filter "
          f"(threshold=tokens_generated>=20).")

    pieces = [per_level_ci(valid, m) for m in METRICS]
    out = pd.concat(pieces, ignore_index=True)

    out_path = Path("results/final/ablation_cis.csv")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False, float_format="%.4f")
    print(f"Wrote {out_path} ({len(out)} rows, {len(METRICS)} metrics)")

    # Print per-axis winners (max mean) for tps + acceptance
    for metric in ("throughput_tps", "acceptance_rate"):
        print(f"\nPer-axis winner by {metric} (mean):")
        m = out[out["metric"] == metric]
        for g in m["group"].unique():
            sub = m[m["group"] == g]
            best = sub.loc[sub["mean"].idxmax()]
            print(f"  {g}: {best['label']:<16s}  mean={best['mean']:.4f}  "
                  f"CI=[{best['ci_lo']:.4f}, {best['ci_hi']:.4f}]  n={best['n']}")


if __name__ == "__main__":
    main()

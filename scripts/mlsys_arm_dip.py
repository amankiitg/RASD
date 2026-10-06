#!/usr/bin/env python
"""Per-arm Hartigan dip test over the native-vs-YaRN arms (1, 2, 3, 4).

Why this exists separately from mlsys_analysis.py
-------------------------------------------------
mlsys_analysis.py aggregates the M4 dose-response. The Phase-1/Phase-A arms
are a DIFFERENT experiment (they change the target model and the RoPE regime,
not just the context length), so they get their own aggregation and their own
file: results/mlsys/arm_dip.csv.

It also bypasses any run-id parsing subtleties entirely by discovering traces
from the arm prefixes and reading context/seed through parse_run_id(), which
now handles the bare `_<N>k_` dialect.

Output columns (per seed, plus one pooled row per arm):
    arm, seed, n_rounds, spec_steps, alpha_mean, p_zero, frac_full_accept,
    dip, p_value, rejects_005

The pooled row is what supports the headline shape claim:
    Arm1 (YaRN/OOD)  -> zero-inflated, bimodal      (reject unimodality)
    Arm2/Arm3/Arm4-128k (native) -> unimodal-high   (cannot reject)

Usage:
    python scripts/mlsys_arm_dip.py --trace-dir results/mlsys/per_token \
        --out results/mlsys/arm_dip.csv
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

import numpy as np  # noqa: E402

from src.analysis.acceptance import (per_round_alpha, load_trace,  # noqa: E402
                                    parse_run_id, run_family)

FIELDS = ["arm", "seed", "n_rounds", "spec_steps", "alpha_mean", "p_zero",
          "frac_full_accept", "dip", "p_value", "rejects_005", "pooled"]


def _alpha(trace: list[dict]) -> np.ndarray:
    """Per-round acceptance, from the SHARED estimator.

    Recomputed inline here it would keep the truncated final round (whose
    accepted prefix was verified but only partly emitted), so the dip test would
    run on a different sample from the alpha it is reported beside. Same helper
    as the cluster bootstrap and the CSV metric.
    """
    return per_round_alpha(trace)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--trace-dir", default=str(REPO / "results" / "mlsys" / "per_token"))
    ap.add_argument("--out", default=str(REPO / "results" / "mlsys" / "arm_dip.csv"))
    ap.add_argument("--alpha", type=float, default=0.05)
    args = ap.parse_args()

    try:
        from diptest import diptest
    except ImportError:
        print("[fatal] the `diptest` package is required "
              "(pip install diptest)")
        return 2

    trace_dir = Path(args.trace_dir)
    traces = sorted(trace_dir.glob("*.jsonl"))
    if not traces:
        print(f"[fatal] no traces in {trace_dir}")
        return 2

    # family -> seed -> alpha array
    by_arm: dict[str, dict[int, np.ndarray]] = {}
    for p in traces:
        rid = p.stem
        fam = run_family(rid)
        if not fam.startswith("arm"):
            continue                     # matrix families handled elsewhere
        ctx, seed = parse_run_id(rid)
        if ctx is None or seed is None:
            print(f"[skip] {rid}: could not parse context/seed")
            continue
        t = load_trace(p)
        if not t:
            print(f"[skip] {rid}: empty trace")
            continue
        # Keep the real spec_steps from the trace rather than assuming k=4.
        by_arm.setdefault(fam, {})[seed] = (_alpha(t), int(t[0]["spec_steps"]))

    if not by_arm:
        print(f"[fatal] no arm traces found under {trace_dir}")
        return 2

    rows: list[dict] = []
    for fam in sorted(by_arm):
        pooled_parts = []
        for seed in sorted(by_arm[fam]):
            a, k = by_arm[fam][seed]
            pooled_parts.append(a)
            d, pv = diptest(a)
            rows.append({
                "arm": fam, "seed": seed, "n_rounds": len(a),
                "spec_steps": k,
                "alpha_mean": round(float(a.mean()), 6),
                "p_zero": round(float((a == 0).mean()), 4),
                "frac_full_accept": round(float((a == 1).mean()), 4),
                "dip": round(float(d), 6), "p_value": float(pv),
                "rejects_005": bool(pv < args.alpha), "pooled": False,
            })
        pooled = np.concatenate(pooled_parts)
        d, pv = diptest(pooled)
        rows.append({
            "arm": fam, "seed": "ALL", "n_rounds": len(pooled),
            "spec_steps": "", "alpha_mean": round(float(pooled.mean()), 6),
            "p_zero": round(float((pooled == 0).mean()), 4),
            "frac_full_accept": round(float((pooled == 1).mean()), 4),
            "dip": round(float(d), 6), "p_value": float(pv),
            "rejects_005": bool(pv < args.alpha), "pooled": True,
        })

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)

    print(f"\nPer-arm dip test ({len(rows)} rows) -> {out}")
    print(f"{'arm':<8}{'seed':<6}{'n':<5}{'mean':<9}{'P(a=0)':<9}"
          f"{'full':<8}{'dip':<9}{'p':<12}reject?")
    for r in rows:
        print(f"{r['arm']:<8}{str(r['seed']):<6}{r['n_rounds']:<5}"
              f"{r['alpha_mean']:<9.4f}{r['p_zero']:<9.4f}"
              f"{r['frac_full_accept']:<8.3f}{r['dip']:<9.4f}"
              f"{r['p_value']:<12.3g}"
              f"{'YES' if r['rejects_005'] else 'no'}"
              f"{'   <- POOLED' if r['pooled'] else ''}")

    print("\nShape comparison (pooled):")
    for r in rows:
        if r["pooled"]:
            verdict = ("zero-inflated / REJECT unimodal"
                       if r["rejects_005"] else "unimodal-high (not rejected)")
            print(f"  {r['arm']:<8} P(alpha=0)={r['p_zero']:<7.4f} "
                  f"full-accept={r['frac_full_accept']:<7.4f} -> {verdict}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

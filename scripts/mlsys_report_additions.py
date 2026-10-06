#!/usr/bin/env python
"""Report-gate additions D1-D3.

D1  Paired seed-level Arm2 - Arm3 acceptance differences with a bootstrap CI.
    Paired because both arms run on the SAME seed, so the seed's prompt and
    decoding randomness cancel in the difference. The word "equivalent" is
    deliberately not produced anywhere: a CI containing zero is failure to
    detect a difference at this n, which is not the same claim, and with
    three seeds the interval is wide enough that saying "equivalent" would
    overstate what was measured.

D2  a_iid reported alongside alpha_round for each arm and each ARM4 cell,
    with the truncated-geometric GOF p-value. alpha_round is the quantity the
    paper reports; a_iid is the fitted i.i.d. parameter that the round-level
    structure must be compared against. Printing them together is what stops
    a reader from reading alpha_round as if it were a per-token probability.

D3  The matched-context read: acceptance vs rope factor 1 -> 2 -> 4 at fixed
    128k context, plus the llama3-factor rungs (16/32) that vary ONLY the
    factor inside Meta's own mechanism. At fixed context, any change in
    acceptance is attributable to the rope change rather than to context
    length. Cells whose data has not been produced yet are listed as pending
    rather than omitted, so a missing rung is visible instead of implied.

Usage:
    python scripts/mlsys_report_additions.py --results-dir results/mlsys \
        --trace-dir results/mlsys/per_token --out-dir results/mlsys
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from src.analysis.acceptance import (              # noqa: E402
    run_family, run_labels, summarize_trace_dir,
)
from src.analysis.bootstrap import bootstrap_mean_ci   # noqa: E402

# ARM4 cells in the order the factor rises at FIXED 128k context.
ARM4_128K_LADDER = [
    ("f1 (native, factor 1)", 130944, 1.0, "yarn"),
    ("f2 (YaRN factor 2)", 130944, 2.0, "yarn"),
    ("f4 (YaRN factor 4)", 130944, 4.0, "yarn"),
    ("llama3 factor 16", 130944, 16.0, "llama3"),
    ("llama3 factor 32", 130944, 32.0, "llama3"),
]


def d1_paired_arm2_arm3(trace_dir: Path) -> pd.DataFrame:
    """Paired seed-level Arm2 - Arm3 acceptance differences."""
    ts = summarize_trace_dir(trace_dir)
    if ts.empty or "run_id" not in ts.columns:
        print("[warn] D1: no traces")
        return pd.DataFrame()

    ts = ts.copy()
    ts["family"] = ts["run_id"].astype(str).map(run_family)
    a2 = ts[ts["family"] == "arm2"].set_index("seed")["alpha_round"]
    a3 = ts[ts["family"] == "arm3"].set_index("seed")["alpha_round"]

    # Paired on seed. An inner join is required: an unpaired difference would
    # be dominated by seed-to-seed prompt variance rather than by the arm.
    paired = pd.concat({"arm2": a2, "arm3": a3}, axis=1, join="inner").dropna()
    if paired.empty:
        print("[warn] D1: no seeds shared by arm2 and arm3")
        return pd.DataFrame()

    diff = (paired["arm2"] - paired["arm3"]).to_numpy()
    mean, lo, hi = bootstrap_mean_ci(diff) if diff.size > 1 else (
        float(diff.mean()) if diff.size else np.nan, np.nan, np.nan)

    rows = [{
        "comparison":   "arm2_minus_arm3",
        "pairing":      "same seed (paired)",
        "n_seeds":      int(diff.size),
        "seeds":        ",".join(str(int(s)) for s in paired.index),
        "mean_diff":    float(mean),
        "ci_lo":        float(lo),
        "ci_hi":        float(hi),
        "ci_half":      float((hi - lo) / 2.0) if np.isfinite(hi) else np.nan,
        "arm2_mean":    float(paired["arm2"].mean()),
        "arm3_mean":    float(paired["arm3"].mean()),
        # The honest reading. NOT "equivalent": with n=3 a CI spanning zero is
        # a failure to detect a difference, not evidence of sameness.
        "read": ("CI excludes zero: the arms differ"
                 if np.isfinite(lo) and (lo > 0 or hi < 0)
                 else "CI includes zero: no difference detected at this n "
                      "(NOT evidence of equivalence)"),
    }]
    for s, r in paired.iterrows():
        rows.append({
            "comparison": "per_seed", "pairing": f"seed {int(s)}",
            "n_seeds": 1, "seeds": str(int(s)),
            "mean_diff": float(r["arm2"] - r["arm3"]),
            "ci_lo": np.nan, "ci_hi": np.nan, "ci_half": np.nan,
            "arm2_mean": float(r["arm2"]), "arm3_mean": float(r["arm3"]),
            "read": "",
        })
    return pd.DataFrame(rows)


def d2_alpha_iid(trace_dir: Path) -> pd.DataFrame:
    """a_iid next to alpha_round, with the GOF p-value, per arm/ARM4 cell."""
    ts = summarize_trace_dir(trace_dir)
    if ts.empty:
        print("[warn] D2: no traces")
        return pd.DataFrame()
    ts = ts.copy()
    ts["family"] = ts["run_id"].astype(str).map(run_family)
    if "gof_p_value" not in ts.columns:
        ts["gof_p_value"] = np.nan
    cols = ["run_id", "family", "seed", "gamma", "n_rounds", "alpha_round",
            "alpha_iid", "iid_ks", "gof_stat", "gof_p_value", "gof_method",
            "p_zero", "frac_full_accept"]
    out = ts[[c for c in cols if c in ts.columns]].copy()
    # A3's GOF is the test of the fitted i.i.d. model; flag it next to the
    # value so alpha_iid is never read as a descriptive statistic. Tri-state
    # on purpose: a NaN p-value (e.g. gamma too small for any bin to reach
    # expected>=5) means the test was uninformative, and reporting that as
    # "did not reject" would turn absence of power into a positive result.
    if "gof_p_value" in out.columns:
        p = pd.to_numeric(out["gof_p_value"], errors="coerce")
        out["gof_reject_5pct"] = np.where(
            p.isna(), "unknown",
            np.where(p < 0.05, "yes", "no"))
    return out.sort_values(["family", "run_id"]).reset_index(drop=True)


def d3_matched_context(trace_dir: Path, results_dir: Path) -> pd.DataFrame:
    """Acceptance vs rope factor at fixed 128k, from the ARM4 cells."""
    ts = summarize_trace_dir(trace_dir)
    have: dict[str, float] = {}
    if not ts.empty:
        for _, r in ts.iterrows():
            rid = str(r["run_id"])
            if rid.startswith("ARM4"):
                have[rid] = float(r["alpha_round"])

    rows = []
    for label, ctx, factor, rtype in ARM4_128K_LADDER:
        prefix = {1.0: "ARM4_f1", 2.0: "ARM4_f2", 4.0: "ARM4_f4"}.get(
            factor, "ARM4_llama3f%d" % int(factor))
        vals = [v for k, v in have.items()
                if k.startswith(prefix) and f"_{ctx}_" in k]
        rows.append({
            "rung":        label,
            "rope_type":   rtype,
            "factor":      factor,
            "context":     ctx,
            "n_runs":      len(vals),
            "acceptance_mean": float(np.mean(vals)) if vals else np.nan,
            "status":      "ok" if vals else "pending (cell not yet run)",
        })
    df = pd.DataFrame(rows)

    # The llama3 rungs vary ONLY the factor within Meta's own mechanism, so
    # agreement/disagreement with the YaRN rungs separates "any rope change
    # hurts" from "YaRN specifically hurts".
    yarn = df[df["rope_type"] == "yarn"]["acceptance_mean"].dropna()
    l3 = df[df["rope_type"] == "llama3"]["acceptance_mean"].dropna()
    if yarn.size >= 2 and l3.size >= 2:
        print(f"[D3] 128k YaRN ladder acceptance: "
              f"{[round(float(v), 4) for v in yarn]}")
        print(f"[D3] 128k llama3-factor acceptance: "
              f"{[round(float(v), 4) for v in l3]}")
        print("[D3] both mechanisms fall as the factor rises"
              if yarn.iloc[-1] < yarn.iloc[0] and l3.iloc[-1] < l3.iloc[0]
              else "[D3] the two mechanisms do NOT agree in direction")
    else:
        print("[D3] matched-context read pending: ARM4 cells have not run yet")
    return df


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", default=str(REPO / "results" / "mlsys"))
    ap.add_argument("--trace-dir", default=None)
    ap.add_argument("--out-dir", default=None)
    args = ap.parse_args()

    results_dir = Path(args.results_dir)
    trace_dir = Path(args.trace_dir) if args.trace_dir else results_dir / "per_token"
    out_dir = Path(args.out_dir) if args.out_dir else results_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    d1 = d1_paired_arm2_arm3(trace_dir)
    if not d1.empty:
        d1.to_csv(out_dir / "d1_paired_arm2_arm3.csv", index=False)
        print(f"[write] d1_paired_arm2_arm3.csv ({len(d1)} rows)")

    d2 = d2_alpha_iid(trace_dir)
    if not d2.empty:
        d2.to_csv(out_dir / "d2_alpha_iid.csv", index=False)
        print(f"[write] d2_alpha_iid.csv ({len(d2)} rows)")

    d3 = d3_matched_context(trace_dir, results_dir)
    if not d3.empty:
        d3.to_csv(out_dir / "d3_matched_context.csv", index=False)
        print(f"[write] d3_matched_context.csv ({len(d3)} rows)")
        print(d3.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

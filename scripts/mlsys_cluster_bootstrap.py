#!/usr/bin/env python3
"""Acceptance reporting with cluster-bootstrap intervals over rounds.

Existing summary tables report a single point estimate per cell, which is what
drew the "largely single-seed" and "bimodality not established" reviewer
comments. Two things are wrong with that:

* Tokens within a verify round are accepted or rejected TOGETHER — the round is
  the independent unit, not the token. A token-level bootstrap would understate
  the interval, sometimes by a lot, because it treats ~gamma correlated draws
  as independent.
* A run's acceptance is a mean over its rounds, and runs differ in how many
  rounds they complete. Comparing a 127-round run against a 6-round run on raw
  point estimates compares two different amounts of evidence.

So: resample ROUNDS with replacement, B times, and report the percentile
interval of the resampled mean. `alpha_total_ratio` (identical to alpha_round when gamma is constant) and
  `alpha_iid` (the memoryless parameter, solved from the mean prefix length),
total_accepted / total_proposed) is reported alongside `alpha_round` (the
per-round accepted-prefix-length / gamma) because they answer different
questions and only one of them is the speculative-decoding acceptance rate.

Optionally reports a common-window estimate: the first N rounds, N = the
shortest run compared, so rungs are compared on equal footing regardless of
output length.

Usage:
    python scripts/mlsys_cluster_bootstrap.py --traces results/mlsys/per_token \
        --out results/mlsys/acceptance_bootstrap.csv [--window 15]
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path

import numpy as np

# This script imports the project's own acceptance code, so the repo root must be
# importable when it is run as a file (sys.path[0] is then `scripts/`).
REPO = Path(__file__).resolve().parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))


def read_rounds(path: Path) -> list[dict]:
    rows = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _window(rows: list[dict], upto: int | None) -> list[dict]:
    """The first `upto` rounds, then the truncated round dropped.

    Order matters. `upto` implements the window-matched comparison ("the first N
    rounds", N = the shortest generation among the runs compared), so it must be
    applied to the rounds as they happened; dropping the partial round first
    would let a short run contribute fewer than N rounds to a window it was
    supposed to fill.
    """
    sel = rows if upto is None else rows[:upto]
    return [r for r in sel if not r.get("round_truncated")]


def round_accepted(rows: list[dict], upto: int | None = None) -> np.ndarray:
    """Per-round accepted counts over the window, excluding the partial round."""
    return np.asarray([int(r["n_acc"]) for r in _window(rows, upto)], dtype=float)


def round_drafted(rows: list[dict], upto: int | None = None) -> np.ndarray:
    return np.asarray([int(r.get("spec_steps", 0))
                       for r in _window(rows, upto)], dtype=float)


def cluster_bootstrap_ci(acc: np.ndarray, drafted: np.ndarray,
                         n_boot: int = 10000, alpha: float = 0.05,
                         seed: int = 0) -> tuple[float, float, float]:
    """Percentile CI for the round-mean of alpha, resampling ROUNDS.

    alpha_round per round is accepted/gamma; gamma is constant within a run, so
    the mean of per-round alpha equals mean(accepted)/gamma when gamma > 0.
    Rounds with gamma == 0 (target-only runs) carry no acceptance signal and
    are excluded rather than counted as zero.
    """
    ok = drafted > 0
    if not ok.any():
        return float("nan"), float("nan"), float("nan")
    a = acc[ok] / drafted[ok]
    n = len(a)
    if n == 0:
        return float("nan"), float("nan"), float("nan")
    point = float(a.mean())
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(n_boot, n))
    means = a[idx].mean(axis=1)
    lo, hi = np.percentile(means, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return point, float(lo), float(hi)


def alpha_total_ratio(acc: np.ndarray, drafted: np.ndarray) -> float:
    """Total accepted / total proposed.

    NOTE: with gamma constant this is algebraically IDENTICAL to the per-round
    mean `alpha_round`, because mean(n_acc/gamma) == sum(n_acc)/(R*gamma). It is
    therefore NOT the i.i.d. per-token parameter and must not be labelled as
    one; see `alpha_iid` below. This function used to be called `a_iid`, which
    printed the same number twice under "per-round" and "i.i.d." headings —
    asserting memorylessness by construction, which is precisely the reviewer
    objection `src/analysis/acceptance.py` was written to answer.
    """
    tot_d = float(drafted.sum())
    return float(acc.sum() / tot_d) if tot_d > 0 else float("nan")


def alpha_iid(acc: np.ndarray, drafted: np.ndarray, gamma: int) -> float:
    """The memoryless per-token alpha that reproduces this mean prefix length.

    Solves E[N] = mean(n_acc) for alpha under P(N >= i) = alpha^i, i.e.
    E[N] = alpha + ... + alpha^gamma. This is the parameter to quote when
    describing acceptance as an i.i.d. per-token probability, and it is
    strictly greater than the per-round mean whenever the trace is not
    memoryless. Returns NaN when it cannot be identified.
    """
    if gamma <= 0 or acc.size == 0:
        return float("nan")
    # Deliberately NOT wrapped in a broad try/except. Swallowing the error here
    # returned NaN for all 57 existing traces while the column printed as if it
    # had been computed, which is the "looks like success" failure mode this
    # project has already been bitten by.
    from src.analysis.acceptance import iid_alpha_for_mean
    return float(iid_alpha_for_mean(float(acc.mean()), gamma))


def split_at_eos(rows: list[dict]) -> int | None:
    """Index of the round that ended on EOS, if the trace recorded one.

    Requires the per-round `ended_on_eos` field added alongside this script.
    Returns None when no round is flagged, including for runs whose traces
    predate the field — in which case the before/after split is not
    reconstructible and the caller must not pretend otherwise.
    """
    for i, r in enumerate(rows):
        if r.get("ended_on_eos"):
            return i
    return None


def p_zero(acc: np.ndarray, drafted: np.ndarray) -> float:
    ok = drafted > 0
    return float((acc[ok] == 0).mean()) if ok.any() else float("nan")


def full_accept_share(acc: np.ndarray, drafted: np.ndarray) -> float:
    """Share of rounds where EVERY drafted token was accepted (saturation)."""
    ok = drafted > 0
    if not ok.any():
        return float("nan")
    return float((acc[ok] == drafted[ok]).mean())


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--traces", default="results/mlsys/per_token")
    ap.add_argument("--out", default="results/mlsys/acceptance_bootstrap.csv")
    ap.add_argument("--window", type=int, default=None,
                    help="also report the first N rounds (common window)")
    ap.add_argument("--n-boot", type=int, default=10000)
    ap.add_argument("--pattern", default="*.jsonl")
    args = ap.parse_args()

    files = sorted(Path(args.traces).glob(args.pattern))
    if not files:
        print(f"no traces matching {args.traces}/{args.pattern}")
        return 1

    rows = []
    for p in files:
        rounds = read_rounds(p)
        if not rounds:
            continue
        acc = round_accepted(rounds)
        drf = round_drafted(rounds)
        if (drf > 0).sum() == 0:
            continue                      # target-only: no acceptance to report

        pt, lo, hi = cluster_bootstrap_ci(acc, drf, n_boot=args.n_boot)
        eos = split_at_eos(rounds)
        rec = {
            "trace": p.stem,
            "n_rounds": len(rounds),
            "n_rounds_used": int((drf > 0).sum()),
            "gamma": int(drf[drf > 0][0]) if (drf > 0).any() else "",
            "alpha_round": round(pt, 6),
            "alpha_round_ci_lo": round(lo, 6),
            "alpha_round_ci_hi": round(hi, 6),
            "ci_half_width": round((hi - lo) / 2, 6),
            "alpha_total_ratio": round(alpha_total_ratio(acc, drf), 6),
            "alpha_iid": round(alpha_iid(acc, drf, int(drf[drf > 0][0])), 6),
            "p_alpha_zero": round(p_zero(acc, drf), 6),
            "full_accept_share": round(full_accept_share(acc, drf), 6),
            "total_accepted": int(acc.sum()),
            "total_drafted": int(drf.sum()),
            "ended_on_eos_round": eos if eos is not None else "",
            "rounds_before_eos": eos if eos is not None else "",
            "rounds_after_eos": (len(rounds) - eos - 1) if eos is not None else "",
        }
        if eos is not None:
            b = acc[:eos + 1] / drf[:eos + 1]
            a = acc[eos + 1:] / drf[eos + 1:] if eos + 1 < len(rounds) else np.array([])
            rec["alpha_before_eos"] = round(float(b.mean()), 6) if b.size else ""
            rec["alpha_after_eos"] = round(float(a.mean()), 6) if a.size else ""

        if args.window:
            wa, wd = round_accepted(rounds, args.window), round_drafted(rounds, args.window)
            wpt, wlo, whi = cluster_bootstrap_ci(wa, wd, n_boot=args.n_boot)
            rec.update({
                f"alpha_window{args.window}": round(wpt, 6),
                f"alpha_window{args.window}_ci_lo": round(wlo, 6),
                f"alpha_window{args.window}_ci_hi": round(whi, 6),
                f"n_rounds_window{args.window}": int((wd > 0).sum()),
            })
        rows.append(rec)

    if not rows:
        print("no spec-decoding traces found")
        return 1

    fields = list(rows[0].keys())
    for r in rows:
        for k in r:
            if k not in fields:
                fields.append(k)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow(r)

    print(f"\n  {'trace':<34} {'n':>4} {'alpha':>8} {'95% CI':>18} {'ratio':>8} {'alpha_iid':>9} "
          f"{'P(a=0)':>7} {'full':>6}")
    for r in rows:
        ci = f"[{r['alpha_round_ci_lo']:.3f}, {r['alpha_round_ci_hi']:.3f}]"
        print(f"  {r['trace']:<34} {r['n_rounds_used']:>4} {r['alpha_round']:>8.4f} "
              f"{ci:>18} {r['alpha_total_ratio']:>8.4f} {r['alpha_iid']:>8.4f} "
              f"{r['p_alpha_zero']:>7.3f} "
              f"{r['full_accept_share']:>6.3f}")
    print(f"\n  wrote {out}  ({len(rows)} traces)")
    print("  NOTE: tokens within a round are accepted/rejected together, so the")
    print("        interval is computed over ROUNDS, not tokens.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Per-round acceptance accounting (MLSys Phase 4, reviewer AbH52 #1).

Two different quantities get called "acceptance rate" in the
speculative-decoding literature, and conflating them is what the
reviewer flagged:

  alpha_round = (1/R) * SUM_r (n_acc_r / gamma)          <-- per-round
      The mean fraction of the gamma = spec_steps proposed tokens that
      form a *prefix* that is accepted, averaged over the R verify
      rounds. This is what the RASD tables report (`acceptance_rate` in
      rasd_inference.generate(): SUM n_acc / (R * gamma)) and the
      quantity the MLSys tables must keep reporting.

      Note it is algebraically identical to
      (total accepted draft tokens) / (total proposed draft tokens),
      i.e. SUM n_acc / SUM gamma, since gamma is constant. The
      "per-round" and "per-proposed-token" ratios are the SAME number.
      They are NOT, however, the i.i.d. parameter below.

  alpha_iid                                                    <-- i.i.d.
      The single per-token acceptance probability alpha of the
      memoryless model of Leviathan et al. (2023). Under that model the
      number of consecutively accepted tokens N has
          P(N=j) = (1-alpha) alpha^j   for j = 0..gamma-1
          P(N=gamma) = alpha^gamma
      with E[N] = (1 - alpha^(gamma+1)) / (1 - alpha).
      Unlike alpha_round, alpha_iid is a *derived model parameter*: the
      alpha you would have to assume for the i.i.d. formula to reproduce
      the observed mean n_acc.

The point of this module is that alpha_round and alpha_iid are equal
ONLY under memorylessness. `summarize_trace()` reports both, plus a KS
distance between the empirical n_acc distribution and the fitted
geometric one, so the i.i.d. assumption can be rejected explicitly
instead of silently inherited.

`verify_csv_acceptance()` recomputes alpha_round from the C13 per-round
traces and asserts it matches the scalar CSV column, so no table can
silently start reporting the derived i.i.d. parameter instead.

Trace schema (one JSON object per line, written by run_experiment.py
under --log-per-token; see `_build_per_token_record` in rasd_inference.py):

    {"round_idx": int, "global_pos_start": int, "spec_steps": int,
     "n_acc": int, "draft_tokens": [int, ...], "accepted": [bool, ...]}
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import pandas as pd

# Matches the run-id dialects used in this repo: the M4 matrix
# ("M4_ctx128k_s42") and the Phase D / PG-19 runs
# ("RASD_ctx4k_pg19_phaseD_s42", "P35D_ctx1M_pg19_s42").
_CTX_RE = re.compile(r"ctx(\d+)([kM])", re.IGNORECASE)
# Phase-1 / Phase-A arm run ids carry the context as a bare `_<N>k_` /
# `_<N>M_` token instead of a `ctx<N>` token, e.g.
#   ARM1_llama2_yarn_128k_s42
#   ARM2_llama3_native_128k_cap4k_s42
#   ARM3_llama3_native_128k_nativedraft_s42
#   ARM4_llama3_yarn_256k_s123
# Without this second pattern those traces silently parsed to (None, None)
# and were dropped from every by-context aggregate, so the 131072 row
# reported the M4 matrix while appearing to cover the arms too.
#
# Anchoring on `_` before the digits and `_`/end after the unit is what
# keeps it from mis-reading a draft-cap suffix: in
# "ARM2_llama3_native_128k_cap4k_s42" the `4k` is preceded by "cap", not
# by an underscore, so only the intended `_128k_` matches.
_CTX_TRAILING_RE = re.compile(r"_(\d+)([kM])(?=_|$)", re.IGNORECASE)
_SEED_RE = re.compile(r"_s(\d+)$", re.IGNORECASE)

# Experiment family, so arm runs are never pooled with the M4
# dose-response cells that happen to share a context length.
_ARM_RE = re.compile(r"^ARM(\d+)", re.IGNORECASE)


def run_family(run_id: str) -> str:
    """Group a run_id into an experiment family.

    "arm1".."armN" for the native-vs-YaRN arms; "matrix" for the M4
    dose-response / Phase-D / PG-19 cells. Arms and matrix cells at the
    SAME context length are different experiments and must be aggregated
    separately.
    """
    m = _ARM_RE.match(run_id)
    if m:
        return f"arm{int(m.group(1))}"
    return "matrix"


def parse_run_id(run_id: str) -> tuple[Optional[int], Optional[int]]:
    """Extract (context_length_tokens, seed) from a run_id.

    "ctx128k" -> 131072, "ctx1M" -> 1048576, and the arm dialect
    "_128k_" -> 131072. Returns (None, None) for pieces that are absent
    (canary rows carry no context).
    """
    ctx: Optional[int] = None
    m = _CTX_RE.search(run_id) or _CTX_TRAILING_RE.search(run_id)
    if m:
        magnitude = 1024 if m.group(2).lower() == "k" else 1024 * 1024
        ctx = int(m.group(1)) * magnitude
    s = _SEED_RE.search(run_id)
    seed = int(s.group(1)) if s else None
    return ctx, seed


def load_trace(path: str | Path) -> list[dict]:
    """Read one .jsonl per-round trace. Missing file -> empty list."""
    p = Path(path)
    if not p.exists():
        return []
    records = []
    with p.open() as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def load_trace_dir(trace_dir: str | Path) -> dict[str, list[dict]]:
    """Load every .jsonl in a directory, keyed by run_id (the file stem)."""
    d = Path(trace_dir)
    if not d.is_dir():
        return {}
    return {p.stem: load_trace(p) for p in sorted(d.glob("*.jsonl"))}


def per_round_alpha(trace: Iterable[dict]) -> np.ndarray:
    """Per-round alpha_r = n_acc_r / gamma, one value per verify round.

    This is the sample the dip test and the bimodality analysis consume.
    Rounds with gamma == 0 are skipped.
    """
    out = []
    for rec in trace:
        gamma = int(rec.get("spec_steps", 0))
        if gamma > 0:
            out.append(int(rec["n_acc"]) / gamma)
    return np.asarray(out, dtype=float)


def iid_alpha_for_mean(mean_n_acc: float, gamma: int) -> float:
    """Solve E[N] = mean_n_acc for alpha under the memoryless model.

    Under i.i.d. per-token acceptance with probability alpha, the number
    of consecutively accepted tokens N is min(Geometric(alpha), gamma),
    whose pmf is in `_iid_pmf` and whose mean is
        E[N] = alpha + alpha^2 + ... + alpha^gamma
    (since P(N >= i) = alpha^i). E[N] is strictly increasing on [0, 1],
    so bisection is exact to machine tolerance.

    The expectation is evaluated from the pmf itself rather than a closed
    form, so this function and `_iid_pmf` cannot drift apart.

    Returns nan when the mean is outside the achievable range.
    """
    if gamma < 1 or not np.isfinite(mean_n_acc):
        return float("nan")
    if mean_n_acc <= 0.0:
        return 0.0
    if mean_n_acc >= gamma:
        return 1.0

    j = np.arange(gamma + 1, dtype=float)

    def expected(a: float) -> float:
        return float((_iid_pmf(a, gamma) * j).sum())

    lo, hi = 0.0, 1.0
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if expected(mid) < mean_n_acc:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def _iid_pmf(alpha: float, gamma: int) -> np.ndarray:
    """P(N = j) for j = 0..gamma under the memoryless model."""
    if not np.isfinite(alpha) or gamma < 1:
        return np.full(gamma + 1, np.nan)
    pmf = np.empty(gamma + 1)
    pmf[:gamma] = (1.0 - alpha) * alpha ** np.arange(gamma)
    pmf[gamma] = alpha ** gamma
    return pmf


def iid_ks_distance(n_acc: np.ndarray, gamma: int) -> float:
    """One-sample KS distance between empirical n_acc and the fitted
    geometric law. Large values reject the i.i.d./memoryless model."""
    if n_acc.size == 0 or gamma < 1:
        return float("nan")
    alpha = iid_alpha_for_mean(float(n_acc.mean()), gamma)
    model = np.cumsum(_iid_pmf(alpha, gamma))
    counts = np.bincount(n_acc.astype(int), minlength=gamma + 1)[: gamma + 1]
    empirical = np.cumsum(counts) / n_acc.size
    return float(np.max(np.abs(empirical - model)))


def summarize_trace(trace: list[dict]) -> dict:
    """Both acceptance conventions plus the round / zero structure.

    Target-only runs have no rounds; every field is then nan / 0 so the
    row stays visible in an aggregate rather than being dropped.
    """
    alpha_r = per_round_alpha(trace)
    n_rounds = int(alpha_r.size)
    if n_rounds == 0:
        return {
            "n_rounds": 0, "gamma": 0,
            "alpha_round": float("nan"), "alpha_round_sem": float("nan"),
            "alpha_iid": float("nan"), "iid_ks": float("nan"),
            "mean_n_acc": float("nan"), "p_zero": float("nan"),
            "frac_full_accept": float("nan"),
            "total_accepted": 0, "total_proposed": 0,
        }

    gamma = int(trace[0].get("spec_steps", 0))
    n_acc = np.asarray([int(r["n_acc"]) for r in trace], dtype=float)

    return {
        "n_rounds":            n_rounds,
        "gamma":               gamma,
        # the reported quantity: mean per-round accepted-prefix / gamma
        "alpha_round":         float(alpha_r.mean()),
        # stderr of the per-round mean — basis of the 3-seed CIs
        "alpha_round_sem":     float(alpha_r.std(ddof=1) / np.sqrt(n_rounds))
                               if n_rounds > 1 else 0.0,
        # the DERIVED i.i.d. parameter, not a separate estimator
        "alpha_iid":           iid_alpha_for_mean(float(n_acc.mean()), gamma),
        # distance to the memoryless model; large => i.i.d. rejected
        "iid_ks":              iid_ks_distance(n_acc, gamma),
        "mean_n_acc":          float(n_acc.mean()),
        "p_zero":              float((alpha_r == 0).mean()),
        "frac_full_accept":    float((alpha_r == 1).mean()),
        "total_accepted":      int(n_acc.sum()),
        # denominator that makes alpha_round = total_accepted / total_proposed
        "total_proposed":      gamma * n_rounds,
    }


def summarize_trace_dir(trace_dir: str | Path) -> pd.DataFrame:
    """One row per trace file: alpha_round, alpha_iid, iid_ks, p_zero, ..."""
    rows = []
    for run_id, trace in load_trace_dir(trace_dir).items():
        ctx, seed = parse_run_id(run_id)
        rec = summarize_trace(trace)
        rec.update({"run_id": run_id, "context_length": ctx, "seed": seed})
        rows.append(rec)
    if not rows:
        return pd.DataFrame(
            columns=["run_id", "context_length", "seed", "n_rounds", "gamma",
                     "alpha_round", "alpha_iid", "iid_ks", "p_zero",
                     "frac_full_accept"]
        )
    return (pd.DataFrame(rows)
            .sort_values(["context_length", "seed"])
            .reset_index(drop=True))


def verify_csv_acceptance(csv_path: str | Path, trace_dir: str | Path,
                          tol: float = 1e-4) -> pd.DataFrame:
    """Assert the CSV `acceptance_rate` equals the per-round alpha.

    Recomputes SUM(n_acc) / (n_rounds * gamma) straight from the traces
    and compares it to the scalar column. `ok=True` means the table is
    reporting the per-round accepted-prefix/gamma quantity.

    `tol` defaults to 1e-4, NOT a tight numerical tolerance, because
    run_experiment.py writes `acceptance_rate` rounded to 4 decimal
    places (see the round(..., 4) in the CSV row builder). Storage alone
    can therefore differ by up to 5e-5, so an exact comparison would flag
    every correctly-accounted row. The default is 2x that bound — still
    far below a genuine mix-up (reporting alpha_iid instead differs by
    0.1-0.3 on the committed traces), so real errors are still caught.

    Rows that exist in only one of the two sources get ok=None (with a
    note) so they are visibly unverified rather than silently dropped.
    """
    csv_path, trace_dir = Path(csv_path), Path(trace_dir)
    if not csv_path.exists():
        raise FileNotFoundError(f"no such results CSV: {csv_path}")

    df = pd.read_csv(csv_path)
    traces = load_trace_dir(trace_dir)

    rows = []
    seen = set()
    if "run_id" in df.columns:
        for _, r in df.iterrows():
            run_id = str(r["run_id"])
            seen.add(run_id)
            csv_val = _as_float(r.get("acceptance_rate"))
            if run_id not in traces:
                rows.append({"run_id": run_id,
                             "csv_acceptance_rate": csv_val,
                             "trace_alpha_round": float("nan"),
                             "alpha_iid": float("nan"),
                             "abs_diff": float("nan"), "ok": None,
                             "note": "no trace (target-only run?)"})
                continue
            summary = summarize_trace(traces[run_id])
            trace_val = summary["alpha_round"]
            diff = abs(csv_val - trace_val) if np.isfinite(csv_val) else float("nan")
            rows.append({
                "run_id":              run_id,
                "csv_acceptance_rate": csv_val,
                "trace_alpha_round":   trace_val,
                "alpha_iid":           summary["alpha_iid"],
                "abs_diff":            diff,
                "ok":                  bool(diff <= tol) if np.isfinite(diff) else None,
                "note":                "",
            })
    for run_id in sorted(set(traces) - seen):
        rows.append({"run_id": run_id, "csv_acceptance_rate": float("nan"),
                     "trace_alpha_round": float("nan"), "alpha_iid": float("nan"),
                     "abs_diff": float("nan"), "ok": None,
                     "note": "trace present, no CSV row"})
    return pd.DataFrame(rows)


def _as_float(v) -> float:
    try:
        return float(v)
    except (TypeError, ValueError):
        return float("nan")

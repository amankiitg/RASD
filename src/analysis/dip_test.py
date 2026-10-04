"""Hartigan dip test for the per-round acceptance distributions
(MLSys Phase 4, reviewer AbH52 #3 / roadmap T2.3).

The reviewer's objection: the paper's bimodality claim was supported only
by a zero/nonzero split of the per-round acceptance (a spike at
alpha = 0 alongside a nonzero mode). That split is a *description*, not a
test — a continuous unimodal distribution with mass near zero produces
the same picture. The correct question is whether the per-round
acceptance sample is consistent with unimodality at all.

The Hartigan & Hartigan (1985) dip statistic answers exactly that:
    D_n = min over unimodal G of  sup_x |F_n(x) - G(x)|
i.e. the L-infinity distance from the empirical CDF to the closest
unimodal CDF. It is location/scale invariant, works on continuous data
(so alpha_r = n_acc/gamma needs no binning or zero/nonzero threshold),
and D_n <= 1/4 always.

We do NOT reimplement Hartigan's GCM/LCM algorithm by hand. The
`diptest` package is a validated C port of the original Fortran and is
used directly; hand-rolling a numerically delicate hull algorithm for a
paper's headline significance claim would be worse than the bug it is
meant to avoid. Tests in tests/test_dip_test.py pin the behaviour
(unimodal -> large p, separated bimodal -> small p, D <= 1/4).

P-value convention
------------------
`diptest.dip` defaults to linear interpolation of Hartigan's tabulated
critical values, which were computed under a *uniform* null. The dip
test is conservative for other unimodal nulls (the uniform tends to
produce larger dips than a peaked unimodal law), so a small tabulated
p-value is strong evidence of multimodality. A uniform-null Monte-Carlo
p-value is available via `boot_pval=True` when a seed-controlled,
assumption-explicit alternative is preferred; it agrees with the
tabulated value up to Monte-Carlo error.

Aggregation
-----------
`dip_over_trace_dir()` runs the test once per trace file (one run =
one seed at one context) and `aggregate_by_context()` summarises across
the 3 seeds, reporting the mean dip, the per-seed p-values, and how many
of the seeds reject unimodality. No seeds => no aggregate row, so a
1-seed context is never presented as if it were 3-seed evidence.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from .acceptance import (
    load_trace_dir,
    parse_run_id,
    per_round_alpha,
)


def dip_test(samples, boot_pval: bool = False, n_boot: int = 10_000,
             seed: int = 42, alpha: float = 0.05,
             only_unique: bool = False) -> dict:
    """Hartigan dip test on one sample.

    Args:
        samples     : 1-D sequence of per-round acceptance values.
        boot_pval   : if True use a uniform-null Monte-Carlo p-value
                      (seed-controlled) instead of Hartigan's table.
        n_boot      : bootstrap draws when boot_pval is set.
        seed        : RNG seed for the bootstrap.
        alpha       : significance level for the `reject_unimodal` flag.
        only_unique : use each distinct value once (see dips on tied
                      data below).

    Returns a dict with the dip statistic, p-value and decision. Degenerate
    inputs (n < 4, or all values identical) return a dip of 0 with a
    non-significant p-value, flagged via `n` / `degenerate` rather than
    raising, because a run with very few rounds is a real (if uninformative)
    outcome that the aggregate must still account for.

    Ties: alpha_r = n_acc/gamma takes only gamma+1 distinct values, so a
    real trace is heavily tied. The dip statistic is defined on the
    empirical CDF and remains valid with ties; `only_unique` is exposed
    only for sensitivity reporting, not as the default.
    """
    from diptest import diptest as _diptest

    x = np.asarray(samples, dtype=float)
    x = x[np.isfinite(x)]
    if only_unique:
        x = np.unique(x)

    n = int(x.size)
    if n < 4 or (n > 0 and np.all(x == x[0])):
        return {
            "n": n, "dip": 0.0, "p_value": 1.0,
            "reject_unimodal": False, "degenerate": True,
            "modal_lo": float("nan"), "modal_hi": float("nan"),
            "p_method": "degenerate",
        }

    # ONE call with full_output: it returns (dip, pval, res), so the modal
    # interval is free. Calling diptest twice (once plain, once for
    # full_output) was both wasteful and the trigger for a segfault inside
    # the C extension when other libraries were already loaded — observed
    # via tests/test_dip_test.py running after the rest of the suite.
    kwargs: dict = {}
    if boot_pval:
        # n_threads=1 keeps the bootstrap single-threaded. That makes the
        # p-value exactly seed-reproducible (the multithreaded path splits
        # the RNG stream across workers) and avoids the multithreaded
        # branch that crashed.
        kwargs = {"boot_pval": True, "n_boot": n_boot, "seed": seed,
                  "n_threads": 1}
    stat, pval, res = _diptest(x, full_output=True, **kwargs)
    lo = float(res.get("xl", float("nan")))
    hi = float(res.get("xu", float("nan")))

    return {
        "n":                n,
        "dip":              float(stat),
        "p_value":          float(pval),
        "reject_unimodal":  bool(pval < alpha),
        "degenerate":       False,
        "modal_lo":         lo,
        "modal_hi":         hi,
        "p_method":         "uniform_bootstrap" if boot_pval else "hartigan_table",
    }


def dip_over_trace_dir(trace_dir: str | Path, boot_pval: bool = False,
                       n_boot: int = 10_000, seed: int = 42,
                       alpha: float = 0.05) -> pd.DataFrame:
    """One dip-test row per .jsonl trace in a directory.

    Adds the descriptive statistics (p_zero, frac_full_accept) alongside
    the formal test so the zero/nonzero split the reviewer objected to is
    reported *next to* the test rather than instead of it.
    """
    from .acceptance import summarize_trace

    rows = []
    for run_id, trace in load_trace_dir(trace_dir).items():
        ctx, s = parse_run_id(run_id)
        alpha_r = per_round_alpha(trace)
        rec = dip_test(alpha_r, boot_pval=boot_pval, n_boot=n_boot,
                       seed=seed, alpha=alpha)
        summary = summarize_trace(trace)
        rec.update({
            "run_id":           run_id,
            "context_length":   ctx,
            "seed":             s,
            "alpha_round":      summary["alpha_round"],
            "alpha_iid":        summary["alpha_iid"],
            "iid_ks":           summary["iid_ks"],
            "p_zero":           summary["p_zero"],
            "frac_full_accept": summary["frac_full_accept"],
            "n_rounds":         summary["n_rounds"],
        })
        rows.append(rec)
    if not rows:
        return pd.DataFrame(columns=[
            "run_id", "context_length", "seed", "n", "dip", "p_value",
            "reject_unimodal", "alpha_round", "alpha_iid", "iid_ks",
            "p_zero", "frac_full_accept", "n_rounds",
        ])
    return (pd.DataFrame(rows)
            .sort_values(["context_length", "seed"])
            .reset_index(drop=True))


def aggregate_by_context(per_run: pd.DataFrame) -> pd.DataFrame:
    """Across-seed summary of the dip test for each context.

    Reports the mean dip with a bootstrap CI, the number of seeds whose
    test rejects unimodality, and the per-seed p-values so a reader can
    see the spread rather than only a verdict. `n_seeds` is reported
    explicitly: a context backed by 1 seed must not read like a 3-seed
    result.
    """
    from .bootstrap import bootstrap_mean_ci

    if per_run.empty:
        return pd.DataFrame(columns=[
            "context_length", "n_seeds", "dip_mean", "dip_ci_lo", "dip_ci_hi",
            "n_reject", "min_p_value", "p_values", "n_rounds_total",
        ])

    rows = []
    for ctx, sub in per_run.groupby("context_length", dropna=False):
        dips = sub["dip"].dropna().to_numpy()
        mean, lo, hi = bootstrap_mean_ci(dips) if dips.size else (np.nan, np.nan, np.nan)
        rows.append({
            "context_length": int(ctx) if pd.notna(ctx) else None,
            "n_seeds":        int(sub["seed"].nunique(dropna=True)),
            "dip_mean":       float(mean),
            "dip_ci_lo":      float(lo),
            "dip_ci_hi":      float(hi),
            "n_reject":       int(sub["reject_unimodal"].sum()),
            "min_p_value":    float(sub["p_value"].min()),
            "p_values":       ",".join(f"{p:.4g}" for p in sub["p_value"]),
            "n_rounds_total": int(sub["n_rounds"].sum()),
        })
    return pd.DataFrame(rows).sort_values("context_length").reset_index(drop=True)

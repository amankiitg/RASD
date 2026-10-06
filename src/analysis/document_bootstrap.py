"""Intervals that resample documents, not tokens and not rounds.

The analysis plan fixes the document as the unit of independence. That choice
has a mechanical consequence here: tokens inside a verify round are accepted or
rejected together, and rounds inside one generation share a prompt and a KV
state, so neither is independent. Documents are.

Two intervals are computed for every estimate, on purpose:

* a **percentile bootstrap** over documents, which is what the plan specifies;
* a **t-based cluster interval** on the same document means.

With 10 clusters a percentile bootstrap is optimistic and its tails are coarse,
so the two can disagree. When they disagree about whether an interval clears a
decision threshold, the plan requires the result to be reported as inconclusive
rather than as the favourable one — which is why the disagreement is a returned
field instead of a judgement made at the call site.

A paired estimate resamples the *same* documents in both arms, so the interval
is on the difference (or ratio) rather than on two marginal intervals.
"""
from __future__ import annotations

import math
from typing import Sequence

import numpy as np

B_DEFAULT = 10000


def _bootstrap_index(n: int, b: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.integers(0, n, size=(b, n))


def document_mean_bootstrap(values: Sequence[float], b: int = B_DEFAULT,
                            seed: int = 20261006,
                            alpha: float = 0.05) -> dict:
    """Percentile bootstrap interval for the mean over documents."""
    v = np.asarray(values, dtype=float)
    if v.size == 0:
        raise ValueError("no documents supplied")
    idx = _bootstrap_index(v.size, b, seed)
    means = v[idx].mean(axis=1)
    lo, hi = np.percentile(means, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return {"mean": float(v.mean()), "lo": float(lo), "hi": float(hi),
            "n_documents": int(v.size)}


def t_cluster_interval(values: Sequence[float], alpha: float = 0.05) -> dict:
    """Student-t interval on the document mean, n-1 degrees of freedom.

    The robustness column. With few clusters this is the more conservative of
    the two, which is the point of reporting it.
    """
    v = np.asarray(values, dtype=float)
    n = v.size
    if n < 2:
        return {"mean": float(v.mean()) if n else float("nan"),
                "lo": float("nan"), "hi": float("nan"), "n_documents": int(n)}
    se = v.std(ddof=1) / math.sqrt(n)
    # Two-sided t quantile via the normal for large n; n is small here (10),
    # so use the exact t table values for the DOF we actually meet and fall
    # back to a normal approximation beyond that.
    t = _t_crit(n - 1, alpha)
    m = float(v.mean())
    return {"mean": m, "lo": m - t * se, "hi": m + t * se,
            "n_documents": int(n)}


_T95 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447,
        7: 2.365, 8: 2.306, 9: 2.262, 10: 2.228, 11: 2.201, 12: 2.179,
        13: 2.160, 14: 2.145, 15: 2.131, 16: 2.120, 17: 2.110, 18: 2.101,
        19: 2.093, 20: 2.086, 25: 2.060, 30: 2.042}


def _t_crit(dof: int, alpha: float) -> float:
    if alpha != 0.05:
        # Only the 95% level is pre-registered; anything else would be an
        # analysis choice made after seeing data, so make it explicit.
        raise ValueError("only alpha=0.05 is pre-registered for cluster intervals")
    if dof in _T95:
        return _T95[dof]
    if dof <= 30:
        return max(_T95[k] for k in _T95 if k <= dof)
    return 1.96


def paired_bootstrap(a: Sequence[float], b: Sequence[float],
                     kind: str = "difference", b_resamples: int = B_DEFAULT,
                     seed: int = 20261006, alpha: float = 0.05) -> dict:
    """Interval for a paired difference or ratio over documents.

    `a` and `b` must already be aligned: element i of each is the same document
    under the two conditions. Resampling indices, not the two arms
    independently, is what makes this a paired estimate.
    """
    x = np.asarray(a, dtype=float)
    y = np.asarray(b, dtype=float)
    if x.shape != y.shape:
        raise ValueError(
            f"paired arms must be aligned; got {x.shape} and {y.shape}"
        )
    if x.size == 0:
        raise ValueError("no documents supplied")
    idx = _bootstrap_index(x.size, b_resamples, seed)
    if kind == "difference":
        stat = (x - y)[idx].mean(axis=1)
        point = float((x - y).mean())
    elif kind == "ratio":
        # Ratio of paired means. A per-document ratio would be undefined
        # whenever a target-only throughput is zero, and would weight
        # slow documents more heavily than fast ones.
        stat = x[idx].mean(axis=1) / y[idx].mean(axis=1)
        point = float(x.mean() / y.mean())
    else:
        raise ValueError(f"unknown kind {kind!r}")
    lo, hi = np.percentile(stat, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return {"kind": kind, "point": point, "lo": float(lo), "hi": float(hi),
            "n_documents": int(x.size)}


def clears(interval: dict, threshold: float) -> str:
    """Verdict for an interval against a decision threshold.

    Returns "above", "below" or "inconclusive". "below" is the verdict the
    plan's payoff rule needs (a speedup ratio whose interval is entirely below
    1.0); an interval containing the threshold is inconclusive, never rounded
    to the nearer side.
    """
    if math.isnan(interval["lo"]) or math.isnan(interval["hi"]):
        return "inconclusive"
    if interval["lo"] > threshold:
        return "above"
    if interval["hi"] < threshold:
        return "below"
    return "inconclusive"

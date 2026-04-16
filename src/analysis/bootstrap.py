"""Percentile-bootstrap confidence intervals.

With only 3 seeds per level we use the empirical percentile bootstrap
(not the BCa or t-bootstrap). Accurate enough for the "which level wins"
question; worth revisiting with >=10 seeds.
"""
from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd

from .metrics import GROUPS, LEVEL_ORDER, LEVEL_LABELS


def bootstrap_mean_ci(values: Sequence[float],
                      n_resamples: int = 10_000,
                      ci: float = 0.95,
                      seed: int = 42) -> tuple[float, float, float]:
    """Return (mean, lo, hi) for a percentile bootstrap on the sample mean."""
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        return (float("nan"), float("nan"), float("nan"))
    if arr.size == 1:
        v = float(arr[0])
        return (v, v, v)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, arr.size, size=(n_resamples, arr.size))
    means = arr[idx].mean(axis=1)
    alpha = (1.0 - ci) / 2.0
    lo = float(np.quantile(means, alpha))
    hi = float(np.quantile(means, 1.0 - alpha))
    return (float(arr.mean()), lo, hi)


def per_level_ci(df: pd.DataFrame, metric: str,
                 ci: float = 0.95, seed: int = 42) -> pd.DataFrame:
    """Per-level bootstrap CI. One row per (group, level_id)."""
    rows = []
    for g in GROUPS:
        sub = df[df["group"] == g]
        for level_id in LEVEL_ORDER[g]:
            vals = sub[sub["level_id"] == level_id][metric].dropna().to_numpy()
            if vals.size == 0:
                continue
            mean, lo, hi = bootstrap_mean_ci(vals, ci=ci, seed=seed)
            rows.append({
                "group":    g,
                "level_id": level_id,
                "label":    LEVEL_LABELS[level_id],
                "metric":   metric,
                "n":        int(vals.size),
                "mean":     mean,
                "ci_lo":    lo,
                "ci_hi":    hi,
                "ci_half":  (hi - lo) / 2.0,
            })
    return pd.DataFrame(rows)

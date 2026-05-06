#!/usr/bin/env python3
"""Figure 2 — M3 ablation grid: throughput + acceptance per axis with 95% CIs.

2 rows (tps, acceptance) x 5 cols (A1..A5).
Reads results/final/ablation_cis.csv (produced by compute_ablation_cis.py).

Usage:
    python scripts/compute_ablation_cis.py   # regenerates CIs
    python scripts/plot_figure2.py
Output: figures/fig2_ablation_bars.{pdf,png}
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from src.analysis.figures import apply_rcparams, save
from src.analysis.metrics import GROUP_LABELS, GROUPS, LEVEL_ORDER


def _plot_axis(ax, df_metric: pd.DataFrame, group: str, ylabel: str, show_ylabel: bool):
    levels = [lv for lv in LEVEL_ORDER[group]
              if lv in df_metric["level_id"].values]
    xs = range(len(levels))
    means = [df_metric[df_metric["level_id"] == lv]["mean"].iloc[0] for lv in levels]
    lows  = [df_metric[df_metric["level_id"] == lv]["ci_lo"].iloc[0] for lv in levels]
    highs = [df_metric[df_metric["level_id"] == lv]["ci_hi"].iloc[0] for lv in levels]
    # yerr expects distance from mean, not absolute values
    yerr_low  = [m - lo for m, lo in zip(means, lows)]
    yerr_high = [hi - m for m, hi in zip(means, highs)]
    labels = [df_metric[df_metric["level_id"] == lv]["label"].iloc[0] for lv in levels]

    ax.bar(xs, means, yerr=[yerr_low, yerr_high],
           capsize=3, color="#3b78b0", edgecolor="black", linewidth=0.5)
    ax.set_xticks(list(xs))
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_title(f"{group}: {GROUP_LABELS[group]}")
    if show_ylabel:
        ax.set_ylabel(ylabel)
    ax.margins(x=0.08)


def main():
    apply_rcparams()
    cis = pd.read_csv("results/final/ablation_cis.csv")

    fig, axes = plt.subplots(2, 5, figsize=(13, 5.5), sharex=False)
    for col, group in enumerate(GROUPS):
        tps = cis[(cis["group"] == group) & (cis["metric"] == "throughput_tps")]
        acc = cis[(cis["group"] == group) & (cis["metric"] == "acceptance_rate")]
        _plot_axis(axes[0, col], tps, group,
                   "Throughput (tok/s)",    show_ylabel=(col == 0))
        _plot_axis(axes[1, col], acc, group,
                   "Acceptance rate",       show_ylabel=(col == 0))

    fig.suptitle("M3 ablation — 95% bootstrap CIs "
                 "(short-run rows tokens_generated<20 excluded)",
                 fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.96))

    out = Path("figures/fig2_ablation_bars")
    save(fig, out)
    print(f"Wrote {out}.pdf and {out}.png")


if __name__ == "__main__":
    main()

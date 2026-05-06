#!/usr/bin/env python3
"""Emit tables/ablation_summary.tex — booktabs table of M3 ablation results.

Rows: axis × level. Columns: throughput mean [95% CI], acceptance mean [95% CI], n.

Usage:
    python scripts/compute_ablation_cis.py   # regenerates CIs
    python scripts/emit_ablation_table.py
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.analysis.metrics import GROUP_LABELS, GROUPS
from src.analysis.tables import df_to_booktabs


def _fmt_ci(row, digits: int) -> str:
    fmt = f"{{:.{digits}f}}"
    return (f"{fmt.format(row['mean'])} "
            f"[{fmt.format(row['ci_lo'])}, {fmt.format(row['ci_hi'])}]")


def main():
    cis = pd.read_csv("results/final/ablation_cis.csv")

    rows = []
    for g in GROUPS:
        tps = cis[(cis["group"] == g) & (cis["metric"] == "throughput_tps")]
        acc = cis[(cis["group"] == g) & (cis["metric"] == "acceptance_rate")]
        axis_label = f"{g}: {GROUP_LABELS[g]}"
        for _, tps_row in tps.iterrows():
            acc_row = acc[acc["level_id"] == tps_row["level_id"]].iloc[0]
            rows.append({
                "Axis":                    axis_label,
                "Level":                   tps_row["label"],
                "Throughput (tok/s) [95\\% CI]": _fmt_ci(tps_row, 2),
                "Acceptance [95\\% CI]":        _fmt_ci(acc_row, 3),
                "n": int(tps_row["n"]),
            })
            axis_label = ""  # only label first row of each axis group

    df = pd.DataFrame(rows)
    out = Path("tables/ablation_summary.tex")
    df_to_booktabs(
        df,
        out,
        caption=("M3 ablation results (R6.5 re-ablation, 64k context, "
                 "8$\\times$A100-SXM4-40GB). Mean and 95\\% bootstrap CI "
                 "per level over 3 seeds (deterministic early-EOS rows "
                 "with tokens\\_generated $<$ 20 excluded; see "
                 "\\S Error Analysis)."),
        label="tab:m3-ablation",
        column_format="llrrr",
    )
    print(f"Wrote {out}")
    print()
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()

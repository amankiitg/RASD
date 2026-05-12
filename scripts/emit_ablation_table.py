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
        # Forward to base emitter; midrules-between-groups added in
        # post-process below since df_to_booktabs is groupless.
        df,
        out,
        caption=("M3 ablation results (64k context, "
                 "8$\\times$A100-SXM4-40GB). Mean and 95\\% bootstrap CI "
                 "per level over 3 seeds; horizontal rules separate the "
                 "five sweep axes. Deterministic early-EOS rows with "
                 "tokens\\_generated $<$ 20 are excluded; see "
                 "\\S Error Analysis."),
        label="tab:m3-ablation",
        column_format="llrrr",
    )
    # Inject \midrule between axis groups. The first two body rows (the
    # header line and the very first data row) are already separated by
    # the header \midrule, so we only insert extra rules before group
    # leaders that come after the first data row.
    text = out.read_text()
    out_lines = []
    data_row_count = 0
    for line in text.splitlines():
        stripped = line.lstrip()
        is_data_row = (line.endswith("\\\\")
                       and "&" in line
                       and not stripped.startswith("\\toprule")
                       and not stripped.startswith("\\bottomrule")
                       and not stripped.startswith("\\midrule"))
        if is_data_row:
            data_row_count += 1
            first_col = line.split("&", 1)[0].strip()
            is_group_leader = bool(first_col)
            # data_row_count == 1 is the header; data_row_count == 2 is
            # the first real data row. From row 3 onward, a non-empty
            # first column marks a new axis group → insert \midrule.
            if is_group_leader and data_row_count > 2:
                out_lines.append("\\midrule")
        out_lines.append(line)
    out.write_text("\n".join(out_lines) + ("\n" if text.endswith("\n") else ""))

    print(f"Wrote {out}")
    print()
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()

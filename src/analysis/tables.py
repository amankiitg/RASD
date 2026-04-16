"""LaTeX booktabs table emitter (hand-rolled — no jinja2 dep)."""
from __future__ import annotations

from pathlib import Path

import pandas as pd


def _fmt(val, float_format: str) -> str:
    if isinstance(val, float):
        if pd.isna(val):
            return "--"
        return float_format % val
    return str(val)


def df_to_booktabs(df: pd.DataFrame,
                   path: str | Path,
                   caption: str,
                   label: str,
                   column_format: str | None = None,
                   float_format: str = "%.3f") -> Path:
    """Write a booktabs-style LaTeX table.

    Requires \\usepackage{booktabs} in the main .tex.
    Columns are taken in DataFrame order; header uses the column names as-is
    (caller is responsible for escaping special LaTeX chars like % or _).
    """
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    if column_format is None:
        column_format = "l" + "r" * (len(df.columns) - 1)

    header = " & ".join(str(c) for c in df.columns) + r" \\"
    body_lines = []
    for _, row in df.iterrows():
        cells = [_fmt(v, float_format) for v in row.tolist()]
        body_lines.append(" & ".join(cells) + r" \\")
    body = "\n".join(body_lines)

    out = (
        "\\begin{table}[h]\n"
        "\\centering\n"
        f"\\caption{{{caption}}}\n"
        f"\\label{{{label}}}\n"
        f"\\begin{{tabular}}{{{column_format}}}\n"
        "\\toprule\n"
        f"{header}\n"
        "\\midrule\n"
        f"{body}\n"
        "\\bottomrule\n"
        "\\end{tabular}\n"
        "\\end{table}\n"
    )
    p.write_text(out)
    return p

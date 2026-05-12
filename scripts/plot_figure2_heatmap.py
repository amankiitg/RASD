#!/usr/bin/env python3
"""Figure 2 — M3 ablation heatmap (mentor M4 spec).

Throughput as a function of Draft Model Size × Speculative Steps (k).

Important data note: M3 ablations are 1-D sweeps along a cross shape,
not a full Cartesian grid. A1 varied draft with k=4 fixed; A2 varied k
with draft=Sheared-LLaMA fixed. So the heatmap is filled along one row
(Sheared, all k) and one column (k=4, both drafts) and intentionally
LEFT BLANK at the un-measured (TinyLlama, k != 4) cells. Those are
shown hatched + annotated "n/m" (not measured) so the figure does not
mislead by interpolating absent data.

Reads the R6.5 ablation CSV directly (does NOT use the bootstrap CIs
file — heatmap shows means; CIs are in tables/ablation_summary.tex).

Usage:
    python scripts/plot_figure2_heatmap.py
    python scripts/plot_figure2_heatmap.py --csv path/to/other.csv
Output: figures/fig2_ablation_heatmap.{pdf,png}
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.analysis.figures import apply_rcparams, save
from src.analysis.metrics import filter_valid, load_ablations


# Row/col ordering for the heatmap. Drafts on the y-axis, k on the x-axis.
DRAFTS = [
    ("A1_tinyllama_1b",   "TinyLlama-1.1B"),
    ("A1_sheared_1b",     "Sheared-LLaMA-1.3B"),
]
K_VALUES = [
    ("A2_k2",   "k=2"),
    ("A2_k4",   "k=4"),       # default-cell — appears in both A1 and A2
    ("A2_k6",   "k=6"),
    ("A2_k8",   "k=8"),
    ("A2_k12",  "k=12"),
]


def _cell_value(valid: pd.DataFrame, level_id: str, metric: str) -> float | None:
    """Mean of `metric` across seeds for a given level_id, or None if absent."""
    sub = valid[valid["level_id"] == level_id][metric].dropna()
    return float(sub.mean()) if len(sub) > 0 else None


def _build_grid(valid: pd.DataFrame, metric: str) -> tuple[np.ndarray, np.ndarray]:
    """Return (values, mask) arrays of shape (len(DRAFTS), len(K_VALUES)).

    Mask is True where the cell is MEASURED (filled in the heatmap), False
    where it is not (hatched out).

    Cell-fill semantics:
      - (Sheared, k=*)        from A2 (Sheared is A2's fixed draft)
      - (*, k=4)              from A1 (k=4 is A1's fixed spec_steps)
      - (TinyLlama, k != 4)   never measured → hatched
    """
    nr, nc = len(DRAFTS), len(K_VALUES)
    values = np.full((nr, nc), np.nan)
    mask   = np.zeros((nr, nc), dtype=bool)
    for i, (draft_id, _) in enumerate(DRAFTS):
        for j, (k_id, _) in enumerate(K_VALUES):
            # The k=4 column reads from A1 levels (which fix k=4 by design)
            # because A2_k4 is duplicate-data with A1_sheared_1b. For the
            # Sheared row we prefer A2's level_id since it has 3 seeds
            # under A2 itself.
            if k_id == "A2_k4":
                # k=4 column: read from A1's level_id for both drafts
                v = _cell_value(valid, draft_id, metric)
            elif draft_id == "A1_sheared_1b":
                # Sheared row, k != 4: read from A2's level_id (k sweep)
                v = _cell_value(valid, k_id, metric)
            else:
                # TinyLlama × k != 4: never measured
                v = None
            if v is not None:
                values[i, j] = v
                mask[i, j] = True
    return values, mask


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--csv", default="results/ablations/ablations_r65.csv",
                   help="Input ablation CSV (default: R6.5)")
    p.add_argument("--metric", default="throughput_tps",
                   help="Metric to plot in the heatmap")
    p.add_argument("--out", default="figures/fig2_ablation_heatmap",
                   help="Output path stem (no extension)")
    args = p.parse_args()

    apply_rcparams()
    df = load_ablations(args.csv)
    valid = filter_valid(df)
    print(f"Loaded {len(df)} rows, {len(valid)} valid after short-run filter")

    values, mask = _build_grid(valid, args.metric)

    fig, ax = plt.subplots(figsize=(7.5, 3.0))
    # Two-tone sequential (light to deep blue) so the figure reads as
    # a single-axis quantity rather than a multi-hue thermal map.
    cmap = plt.get_cmap("Blues").copy()
    cmap.set_bad(color="#e5e5e5")  # un-measured cells
    masked = np.ma.array(values, mask=~mask)
    im = ax.imshow(masked, aspect="auto", cmap=cmap, origin="lower")

    # Annotate measured cells with the mean value. Use auto-contrast:
    # Blues cmap goes from very light (low values) to deep blue (high
    # values), so we pick white text only where the cell is dark.
    vmin, vmax = float(np.nanmin(values)), float(np.nanmax(values))
    vmid = (vmin + vmax) / 2.0
    for i in range(values.shape[0]):
        for j in range(values.shape[1]):
            if mask[i, j]:
                txt_color = "white" if values[i, j] > vmid else "#1a1a1a"
                ax.text(j, i, f"{values[i, j]:.2f}",
                        ha="center", va="center", color=txt_color,
                        fontsize=10, fontweight="bold")
            else:
                # Denser, darker hatching + bigger bolder n/m label so
                # the un-measured cells are unmistakable.
                ax.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1,
                                           fill=True, facecolor="#dddddd",
                                           hatch="xxxx",
                                           edgecolor="#666666",
                                           linewidth=0.6))
                ax.text(j, i, "n/m", ha="center", va="center",
                        color="#222222", fontsize=12, fontweight="bold",
                        style="italic")

    ax.set_xticks(range(len(K_VALUES)))
    ax.set_xticklabels([lbl for _, lbl in K_VALUES])
    ax.set_yticks(range(len(DRAFTS)))
    ax.set_yticklabels([lbl for _, lbl in DRAFTS])
    ax.set_xlabel("Speculative steps (k)")
    ax.set_ylabel("Draft model")

    metric_titles = {
        "throughput_tps":   "Throughput (tok/s)",
        "acceptance_rate":  "Acceptance rate",
        "mean_latency_ms":  "Latency (ms / token)",
        "gpu_peak_mem_mb":  "Peak GPU memory (MB)",
    }
    title = metric_titles.get(args.metric, args.metric)
    ax.set_title(f"M3 ablation: {title}  (ctx=64k, 8×A100-SXM4-40GB)",
                 fontsize=10)

    cb = fig.colorbar(im, ax=ax, shrink=0.85)
    cb.set_label(title)

    # Footer note explaining the hatched cells (larger + darker so it
    # is actually readable next to the heatmap).
    fig.text(0.5, 0.02,
             "Hatched cells (n/m) were not measured: A1 fixed k=4; "
             "A2 fixed draft=Sheared-LLaMA-1.3B.",
             ha="center", fontsize=10, color="#222222")
    fig.tight_layout(rect=(0, 0.07, 1, 1))

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    save(fig, out)
    print(f"Wrote {out}.pdf and {out}.png")


if __name__ == "__main__":
    main()

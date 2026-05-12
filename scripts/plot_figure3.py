#!/usr/bin/env python3
"""Phase D F3: stacked bar — compute / comm / idle / other time breakdown
across contexts (mentor M4 Fig 3).

x-axis: context length (canary @ 32k / 128k / 256k / 512k)
y-axis: percentage of wall-clock time spent in each bucket
stack:  compute, comm, idle, other (sums to 100% per bar)

Reads results/final/final_results.json -> data["profiler"][run_id]
which holds the RoundProfiler summary JSON written by Phase C
(commits 8a9dd00, 61508ac, 699b93d).

Output:
  figures/fig3_time_breakdown.pdf
  figures/fig3_time_breakdown.png
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt


REPO_ROOT = Path(__file__).resolve().parent.parent

# Run-id → display label (x-axis tick)
LABEL_MAP = {
    "matrix_profiled_canary_s42": "32k\n(canary)",
    "PROFILED_ctx128k_s42":       "128k",
    "PROFILED_ctx256k_s42":       "256k",
    "PROFILED_ctx512k_s42":       "512k",
    "PROFILED_ctx1M_s42":         "1M",  # if it ever lands
}
LABEL_ORDER = list(LABEL_MAP.keys())


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--in",  dest="in_path",
                   default="results/final/final_results.json")
    p.add_argument("--out-pdf",
                   default="figures/fig3_time_breakdown.pdf")
    p.add_argument("--out-png",
                   default="figures/fig3_time_breakdown.png")
    args = p.parse_args()

    data = json.loads((REPO_ROOT / args.in_path).read_text())
    prof = data["profiler"]

    bars = []  # (label, compute_pct, comm_pct, idle_pct, other_pct)
    for run_id in LABEL_ORDER:
        if run_id not in prof:
            continue
        d = prof[run_id]
        bars.append((
            LABEL_MAP[run_id],
            d.get("compute_pct", 0.0) * 100,
            d.get("comm_pct",    0.0) * 100,
            d.get("idle_pct",    0.0) * 100,
            d.get("other_pct",   0.0) * 100,
        ))

    if not bars:
        raise SystemExit("No profiler data found in final_results.json")

    labels   = [b[0] for b in bars]
    compute  = [b[1] for b in bars]
    comm     = [b[2] for b in bars]
    idle_    = [b[3] for b in bars]
    other    = [b[4] for b in bars]

    fig, ax = plt.subplots(figsize=(6.0, 4.0), dpi=140)
    bar_width = 0.65
    x = list(range(len(labels)))

    # Stack: compute (bottom) -> comm -> idle -> other (top)
    p1 = ax.bar(x, compute, bar_width, label="compute",
                color="#2ca02c", edgecolor="white", linewidth=0.4)
    p2 = ax.bar(x, comm, bar_width, bottom=compute, label="comm",
                color="#ff7f0e", edgecolor="white", linewidth=0.4)
    bot3 = [a + b for a, b in zip(compute, comm)]
    p3 = ax.bar(x, idle_, bar_width, bottom=bot3, label="idle",
                color="#9467bd", edgecolor="white", linewidth=0.4)
    bot4 = [a + b for a, b in zip(bot3, idle_)]
    p4 = ax.bar(x, other, bar_width, bottom=bot4, label="other",
                color="#7f7f7f", edgecolor="white", linewidth=0.4)

    # Annotate compute % on each bar
    for xi, (cpct, comm_pct) in enumerate(zip(compute, comm)):
        ax.text(xi, cpct / 2, f"{cpct:.0f}%",
                ha="center", va="center", color="white",
                fontsize=9, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Wall-clock time share (rank 0, %)")
    ax.set_title("Time breakdown across contexts (RASD, spec_steps=4)")
    ax.set_ylim(0, 100)
    ax.set_yticks([0, 25, 50, 75, 100])
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5),
              frameon=True, framealpha=0.95, fontsize=9,
              borderaxespad=0.0)
    ax.grid(True, axis="y", alpha=0.3, linewidth=0.5)
    plt.tight_layout()

    out_pdf = REPO_ROOT / args.out_pdf
    out_png = REPO_ROOT / args.out_png
    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_pdf, format="pdf", bbox_inches="tight")
    plt.savefig(out_png, format="png", bbox_inches="tight", dpi=160)
    print(f"Wrote {out_pdf}")
    print(f"Wrote {out_png}")
    print()
    print("Bar values (per context):")
    for lab, c, cm, i, o in zip(labels, compute, comm, idle_, other):
        print(f"  {lab:<14}  compute={c:5.2f}%  comm={cm:5.2f}%  "
              f"idle={i:5.2f}%  other={o:5.2f}%")


if __name__ == "__main__":
    raise SystemExit(main())

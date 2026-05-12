#!/usr/bin/env python3
"""Phase D F1: throughput vs context length (line plot with 3-seed CI bands).

Three series:
  - RASD spec mode (p35 final_matrix)
  - Target-only baseline (p35b, spec_steps=0)
  - HF FA-2 ceiling (p37, single-rank, vanilla generate)

x-axis: context length (log scale, 128k / 256k / 512k / 1M)
y-axis: generation throughput (tokens/sec)
shaded bands: 95% CI from 3-seed bootstrap (RASD + target-only only;
              HF FA-2 has 1 seed and OOM markers).

Reads from results/final/final_results.json. Output:
  figures/fig1_throughput_vs_context.pdf
  figures/fig1_throughput_vs_context.png
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


REPO_ROOT = Path(__file__).resolve().parent.parent


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--in",  dest="in_path",
                   default="results/final/final_results.json")
    p.add_argument("--out-pdf",
                   default="figures/fig1_throughput_vs_context.pdf")
    p.add_argument("--out-png",
                   default="figures/fig1_throughput_vs_context.png")
    p.add_argument("--show-hf", action="store_true",
                   help="Overlay vanilla HF FA-2 ceiling (1 seed, OOM markers).")
    args = p.parse_args()

    data = json.loads((REPO_ROOT / args.in_path).read_text())

    # ---- Pull series ----
    def _series(matrix_key: str) -> tuple[list[int], list[float], list[float]]:
        """Return (ctx, mean_tps, ci_half) sorted by ctx."""
        m = data["matrix"][matrix_key]
        ctxs, means, cis = [], [], []
        for level_id, cells in m.items():
            if not cells:
                continue
            tps_vals = [c["throughput_tps"] for c in cells]
            ctx = cells[0]["context_length"]
            mean = float(np.mean(tps_vals))
            sem  = float(np.std(tps_vals, ddof=1) / (len(tps_vals) ** 0.5)) \
                   if len(tps_vals) > 1 else 0.0
            ci_half = 1.96 * sem
            ctxs.append(ctx); means.append(mean); cis.append(ci_half)
        order = sorted(range(len(ctxs)), key=lambda i: ctxs[i])
        return ([ctxs[i] for i in order],
                [means[i] for i in order],
                [cis[i]   for i in order])

    rasd_ctx, rasd_tps, rasd_ci = _series("rasd")
    tgt_ctx,  tgt_tps,  tgt_ci  = _series("target_only")

    # HF FA-2 ceiling: single seed, includes OOM rows
    hf_rows = data["matrix"]["hf_ceiling"]
    hf_ok = [r for r in hf_rows if r["status"] == "ok"]
    hf_oom = [r for r in hf_rows if r["status"] == "oom"]
    hf_ctx_ok = sorted(r["context_length"] for r in hf_ok)
    hf_tps_ok = [next(r["throughput_tps"] for r in hf_ok
                      if r["context_length"] == c) for c in hf_ctx_ok]
    hf_ctx_oom = sorted(r["context_length"] for r in hf_oom)

    # ---- Plot ----
    fig, ax = plt.subplots(figsize=(6.5, 4.2), dpi=140)

    def _band(ax, ctx, mean, ci, color, label, marker):
        m_arr = np.array(mean); ci_arr = np.array(ci)
        ax.plot(ctx, m_arr, marker=marker, color=color, label=label,
                linewidth=1.8, markersize=6)
        ax.fill_between(ctx, m_arr - ci_arr, m_arr + ci_arr,
                        alpha=0.18, color=color, linewidth=0)

    _band(ax, rasd_ctx, rasd_tps, rasd_ci, "#1f77b4",
          "RASD (spec_steps=4)", "o")
    _band(ax, tgt_ctx,  tgt_tps,  tgt_ci,  "#ff7f0e",
          "Target-only (spec_steps=0)", "s")

    if args.show_hf:
        # OK rows: filled circle. OOM rows: red X markers at y=0 with
        # an annotation that they OOM'd.
        if hf_tps_ok:
            ax.plot(hf_ctx_ok, hf_tps_ok, marker="^", color="#2ca02c",
                    label="HF FA-2 + generate (vanilla)", linewidth=1.8,
                    markersize=7)
        for c in hf_ctx_oom:
            ax.scatter([c], [0.005], marker="x", color="#d62728",
                       s=70, zorder=5)
        if hf_ctx_oom:
            ax.scatter([], [], marker="x", color="#d62728", s=70,
                       label="HF FA-2 OOM")

    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xlabel("Context length (tokens)")
    ax.set_ylabel("Throughput (generation tokens/sec)")
    ax.set_title("Long-context inference throughput "
                 "(Llama-2-7B, 8×A100, NF4 + ring SP + YaRN)")
    # Tick labels in human-readable form (128k / 256k / 512k / 1M)
    xticks = [131072, 262144, 524288, 1048576]
    if args.show_hf and 32768 not in xticks:
        xticks = [32768] + xticks
    xtick_labels = {32768: "32k", 131072: "128k", 262144: "256k",
                    524288: "512k", 1048576: "1M"}
    ax.set_xticks(xticks)
    ax.set_xticklabels([xtick_labels[c] for c in xticks])
    ax.grid(True, which="both", alpha=0.3, linewidth=0.5)
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5),
              frameon=True, framealpha=0.95, borderaxespad=0.0)
    plt.tight_layout()

    out_pdf = REPO_ROOT / args.out_pdf
    out_png = REPO_ROOT / args.out_png
    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_pdf, format="pdf", bbox_inches="tight")
    plt.savefig(out_png, format="png", bbox_inches="tight", dpi=160)
    print(f"Wrote {out_pdf}")
    print(f"Wrote {out_png}")
    print()
    print("Series summary:")
    for label, ctxs, means, cis in [
        ("RASD",   rasd_ctx, rasd_tps, rasd_ci),
        ("Target", tgt_ctx,  tgt_tps,  tgt_ci),
    ]:
        for c, m, ci in zip(ctxs, means, cis):
            print(f"  {label:<6}  ctx={c:>8,}  {m:.3f} ± {ci:.3f} tps")


if __name__ == "__main__":
    raise SystemExit(main())

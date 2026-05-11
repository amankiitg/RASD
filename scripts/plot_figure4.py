#!/usr/bin/env python3
"""Phase D F4: α (acceptance rate) vs token position with smoothing.

For each RASD spec_steps=4 run, the per-position .jsonl sidecar has
one record per verify round:
    {round_idx, global_pos_start, spec_steps, n_acc,
     draft_tokens, accepted, ...}

We compute α per round as n_acc/spec_steps and plot α vs round_idx
(or vs global_pos_start) for each context length. With smoothing
(rolling mean over a 3-round window) so the line is readable.

Reads from results/final/per_token/*.jsonl OR via final_results.json
data['per_position_accept'] (the same data, indexed by run_id).

Output:
  figures/fig4_acceptance_vs_position.pdf
  figures/fig4_acceptance_vs_position.png
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


REPO_ROOT = Path(__file__).resolve().parent.parent


CTX_LABEL = {
    131072:   "128k",
    262144:   "256k",
    524288:   "512k",
    1048576:  "1M",
}
CTX_COLOR = {
    131072:   "#1f77b4",
    262144:   "#ff7f0e",
    524288:   "#2ca02c",
    1048576:  "#d62728",
}


def _alpha_from_record(rec: dict) -> float:
    spec_steps = rec.get("spec_steps", 0)
    n_acc      = rec.get("n_acc", rec.get("n_accepted", 0))
    if spec_steps <= 0:
        return 0.0
    return min(1.0, n_acc / spec_steps)


def _rolling_mean(xs: list[float], k: int = 3) -> list[float]:
    if k <= 1 or len(xs) <= k:
        return list(xs)
    out = []
    half = k // 2
    for i in range(len(xs)):
        lo = max(0, i - half)
        hi = min(len(xs), i + half + 1)
        out.append(float(np.mean(xs[lo:hi])))
    return out


def _read_jsonl(path: Path) -> list[dict]:
    records: list[dict] = []
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return records


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--per-token-dir", default="results/final/per_token",
                   help="Directory of <run_id>.jsonl files.")
    p.add_argument("--out-pdf", default="figures/fig4_acceptance_vs_position.pdf")
    p.add_argument("--out-png", default="figures/fig4_acceptance_vs_position.png")
    p.add_argument("--smooth", type=int, default=3,
                   help="Rolling-mean window size for smoothing.")
    p.add_argument("--prefer", default="RASD",
                   choices=["RASD", "M4"],
                   help="Run-id prefix to include (RASD_* for Phase D rerun "
                        "or M4_* for Phase C matrix).")
    args = p.parse_args()

    pt_dir = REPO_ROOT / args.per_token_dir
    if not pt_dir.exists() or not list(pt_dir.glob("*.jsonl")):
        raise SystemExit(
            f"No .jsonl sidecars in {pt_dir}. "
            f"Run with --log-per-token to generate them."
        )

    # Group files by ctx
    series: dict[int, list[float]] = {}
    for jf in sorted(pt_dir.glob("*.jsonl")):
        name = jf.stem
        # Only the matching prefix + skip target-only (n_acc would be 0 since
        # spec_steps=0; α undefined)
        if args.prefer == "RASD" and not name.startswith("RASD_"):
            continue
        if args.prefer == "M4" and not name.startswith("M4_"):
            continue
        # Pull ctx from CTX_LABEL keys by name substring match
        ctx_found: int | None = None
        for ctx, lab in CTX_LABEL.items():
            if lab in name:
                ctx_found = ctx; break
        if ctx_found is None:
            continue
        recs = _read_jsonl(jf)
        if not recs:
            continue
        alphas = [_alpha_from_record(r) for r in recs]
        series.setdefault(ctx_found, []).extend(alphas)

    if not series:
        raise SystemExit("No matching per-token files found "
                         f"(prefer={args.prefer}).")

    fig, ax = plt.subplots(figsize=(7.0, 4.2), dpi=140)
    for ctx in sorted(series):
        alphas = series[ctx]
        smoothed = _rolling_mean(alphas, k=args.smooth)
        rounds = list(range(1, len(smoothed) + 1))
        ax.plot(rounds, smoothed, color=CTX_COLOR[ctx],
                linewidth=2.0, alpha=0.95, label=CTX_LABEL[ctx])
        # Raw points lightly drawn behind
        ax.scatter(rounds, alphas, color=CTX_COLOR[ctx],
                   s=12, alpha=0.25, edgecolors="none")

    ax.set_xlabel("Verify round")
    ax.set_ylabel(r"Acceptance rate $\alpha = n_{\mathrm{acc}}/k$")
    ax.set_title("Per-round acceptance rate across contexts "
                 "(RASD, spec_steps=4, k=3-round smoothing)")
    ax.set_ylim(-0.02, 1.02)
    ax.grid(True, alpha=0.3, linewidth=0.5)
    ax.legend(title="Context", loc="upper right", frameon=True,
              framealpha=0.95)
    plt.tight_layout()

    out_pdf = REPO_ROOT / args.out_pdf
    out_png = REPO_ROOT / args.out_png
    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_pdf, format="pdf", bbox_inches="tight")
    plt.savefig(out_png, format="png", bbox_inches="tight", dpi=160)
    print(f"Wrote {out_pdf}")
    print(f"Wrote {out_png}")
    print()
    print(f"Per-context α summary (mean over all rounds):")
    for ctx in sorted(series):
        a = series[ctx]
        print(f"  {CTX_LABEL[ctx]:>5}  n_rounds={len(a):>3}  "
              f"mean_α={np.mean(a):.3f}  median_α={np.median(a):.3f}")


if __name__ == "__main__":
    raise SystemExit(main())

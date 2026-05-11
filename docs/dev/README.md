# Developer & Working Notes

These are the working-tree plans, runbooks, and engineering chronicles
that produced the published RASD paper. They are kept for
reproducibility and to provide the full audit trail for any reader
who wants to verify how the system's design point was derived.

| Document | What it is |
|---|---|
| [`M3_RING_INTEGRATION_PLAN.md`](M3_RING_INTEGRATION_PLAN.md) | Phase-by-phase chronicle of integrating ring attention + speculative decoding (R1 → R6.5). Contains the M3 ablation methodology and findings that fixed every knob in the system. |
| [`M4_PLAN.md`](M4_PLAN.md) | M4 (1M-context inference) plan: long-context memory equation, NF4 KV strategy, the 36-cell final matrix protocol, Phase C runbook, and the headline-numbers + limitations sections used as source for the manuscript. |
| [`PHASE_C_RUNBOOK.md`](PHASE_C_RUNBOOK.md) | Single-page operator checklist for running Phase C (pod bootstrap, env vars, stage marker semantics, post-session cleanup). Source for the paper's Reproduction Appendix. |
| [`M5_PLAN.md`](M5_PLAN.md) | M5 (manuscript) plan: multi-track submission strategy, abstract draft, three research questions, week-by-week task list. |
| [`PUBLICATION_STRATEGY.md`](PUBLICATION_STRATEGY.md) | Long-range venue strategy: arXiv + NeurIPS 2026 workshop now, MLSys 2027 as the primary archival target, NeurIPS 2027 as backup. |
| [`experiments.md`](experiments.md) | The Phase B analysis on the M3 ablations (3-seed CIs, error analysis, replication of mentor's expected design point). |

## Link conventions

Inline backticked paths (e.g., `` `tables/main_speedup.tex` ``) are
prose and remain readable regardless of file location. Clickable
Markdown links inside these documents are relative to the repository
root --- they may not resolve correctly when viewed in GitHub's web
UI because the documents now live at `docs/dev/` rather than the
repo root.  Most readers do not need to follow the links; the prose
is self-contained. If you do need to click one, prefix with the
repo URL or read the document locally.

## Why these are not at the repo root

These are internal-process documents. They contain implementation
chronicles, debugging notes, cost discipline, and engineering risk
registers --- material that is valuable for reproducibility audits
but distracts from the paper. The repository root is reserved for
`README.md`, `LICENSE`, `REPRODUCE.md`, and the published artifacts
that a first-time reader would touch.

## Authoritative pointers

For the paper itself, the architecture description, and the figure-to-
data-to-W&B mapping, see:

* The compiled manuscript at [`manuscript/arxiv/main.pdf`](../../manuscript/arxiv/main.pdf).
* The reproduction recipe at [`REPRODUCE.md`](../../REPRODUCE.md).
* The canonical results JSON at [`results/final/final_results.json`](../../results/final/final_results.json).

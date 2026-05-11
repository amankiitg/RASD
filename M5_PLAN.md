# Milestone 5 — Manuscript Writing & Workshop Submission

**Weeks 9–10 (2026-05-12 → 2026-05-25).**
Target venue: **NeurIPS 2026 workshop** (ML for Systems / ENLSP /
Efficient Foundation Models — whichever has the next live deadline
in the May–August window).
8-page manuscript, MIT-licensed code release.

This is the **tactical** plan for M5. The long-range venue strategy
lives in `PUBLICATION_STRATEGY.md`; the upstream data work is in
`M4_PLAN.md`.

---

## Inputs from M4 (frozen at tag `m4-phase-d-complete`)

All paper inputs are committed in `main` and pointed to by the tag.

### Numerical results (paper-grade, 3-seed where applicable)

| Source | What it says | File |
|---|---|---|
| **p35** RASD final matrix | RASD throughput at 128k/256k/512k/1M, 3 seeds | `results/final/final_matrix.csv` |
| **p35b** target-only baseline | Apples-to-apples baseline (same RASD infra, `spec_steps=0`), 3 seeds | `results/final/target_only_matrix.csv` |
| **p35c** PG-19 perplexity | Quant + YaRN quality sanity at 4k–32k | `results/perplexity/m4_ppl.csv` |
| **p35c (control)** vanilla-RoPE PPL | Methodology appendix — shows 11–18× YaRN benefit | `results/perplexity/m4_ppl_vanilla_rope.csv` |
| **p35d** PG-19 prompt at 1M | 10.8% acceptance (lower than synthetic mean) — fundamental, not prompt-driven | `results/final/p35d_pg19_prompt.csv` |
| **p36** profiler matrix | compute/comm/idle/other at canary + 128k/256k/512k | `results/final/profiler_pass/profiler/*.json` |
| **p37** vanilla HF FA-2 ceiling | 32k passes (30.8 GB), ≥128k all OOM | `results/baselines/hf_ceiling.csv` |
| **Phase D rerun** | Per-token sidecars + saved generations for F4/F5/F8 | `results/final/phase_d_rerun.csv` + sidecars |
| **Phase D short-ctx PG-19** | **4.4× speedup, 70.6% acceptance at 4k native** | `results/final/phase_d_short_ctx_pg19.csv` |

### Figures (all in `figures/`)

- `fig1_throughput_vs_context.{pdf,png}` — RASD vs target-only vs HF ceiling, log-log, 3-seed CI bands
- `fig2_ablation_heatmap.{pdf,png}` — M3 ablation surface (carried forward)
- `fig3_time_breakdown.{pdf,png}` — compute/comm/idle/other across 4 contexts
- `fig4_acceptance_vs_position.{pdf,png}` — two-panel: per-round trace + α=0 vs α>0 bimodality bar, includes short-ctx PG-19 reference
- `fig5_qualitative_examples.txt` — RASD vs target side-by-side at 6 contexts (text version)

### Tables (all in `tables/`)

- `main_speedup.tex` — RASD vs target speedup per context (1.24× → 1.80× → 0.97× → 1.00× synth; 4.4× at 4k PG-19)
- `main_memory.tex` — peak GB per context + RASD overhead column
- `main_ppl.tex` — YaRN vs vanilla PPL benefit ratio
- `main_profiler.tex` — compute/comm/idle/other breakdown
- `main_ceiling.tex` — HF FA-2 OOM ceiling
- `qualitative_examples.tex` — F5 in LaTeX form
- `ablation_summary.tex` — M3 carry-forward

### Analysis docs

- `analysis/error_analysis.md` — F8 with bimodality headline + mechanistic interpretation
- `M4_PLAN.md` headline + Limitations sections — source for Methods / Discussion prose
- `PHASE_C_RUNBOOK.md` — source for Reproduction section

### Single canonical aggregate

- `results/final/final_results.json` — JSON aggregator that downstream
  figure scripts cite. Built by `scripts/aggregate_final_results.py`.

### Codebase state

- 470 tests passing
- Tags: `m4-phase-c-complete`, `m4-phase-d-complete`
- `requirements-lock.txt` pinned (Phase C captured)

---

## Headline findings to frame around

Three claims the paper makes, ranked by paper-value:

1. **Capability** — RASD enables Llama-2-7B inference at 1M context on
   8×A100 80GB at 40 GB peak per rank. Vanilla HF FA-2 + `generate()`
   OOMs at 128k single-rank (p37 evidence). **Only RASD reaches 1M.**

2. **Speedup is base-model-bounded, not architecture-bounded.** At
   Llama-2's native 4k context with real text, RASD delivers **4.4×
   speedup with median per-round acceptance of 1.0** (most rounds
   accept all 4 draft tokens). At long contexts under YaRN factor=256,
   acceptance drops to 12–21% and speedup flattens — but Phase D's
   short-ctx PG-19 cells (4k/8k) show this is **OOD-driven** (Llama-2
   wasn't trained at >4k), not a fundamental RASD limit. A
   long-context-trained base model (Llama-3.1-128k, Qwen-2.5-1M)
   would carry the 4× headline through to longer contexts.

3. **Honest characterization of failure modes.** Per-round acceptance
   is **bimodal** — about half of verify rounds fully reject the
   draft (α=0), the other half accept 1–2 of 4 draft tokens. Mean
   acceptance hides this; median + the F4 right-panel bar surface
   it. The bimodality is mechanistically tied to target-model logit
   entropy in the OOD regime.

---

## Three research questions (for risk-mitigation per the mentor brief)

1. **RQ1 (capability):** Can speculative decoding be combined with
   ring-attention sequence parallelism and NF4 KV quantization to
   enable 1M-context inference on commodity hardware (8×A100 80GB)?
2. **RQ2 (performance):** Does the spec-decoding speedup transfer
   to the long-context regime, and what limits it?
3. **RQ3 (characterization):** What are the failure modes of the
   speedup at extreme context lengths, and how do they decompose
   into architecture vs base-model contributions?

**Action 9.1 below: send these three RQs to the mentor BEFORE
writing prose**, per the brief's risk-mitigation note.

---

## Outputs (mapped from the M5 brief)

- `manuscript/main.pdf` — final camera-ready PDF
- `manuscript/supplementary.pdf` — extra plots (vanilla-RoPE PPL,
  memory traces, RULER-niah infra description as future work)
- Public GitHub repo (separate clean fork — not the working repo)
  under **MIT license**
- `README.md` with reproduction instructions
- Submission portal confirmation + ID

---

## Task list

Each task tagged with `[ID effort]` for estimation. `H` = ~half-day,
`F` = ~full day. Order is dependency-aware.

### Week 9 (2026-05-12 → 2026-05-18) — drafting

- [ ] **9.1 [H]** Write three RQs (above) + 1-page bullet outline of
      the paper. Email mentor for sign-off. **BLOCKS prose work.**
- [ ] **9.2 [H]** Create Overleaf project, paste NeurIPS 2026 template,
      set up `\input{}` shims for all `tables/*.tex` files. Verify
      build.
- [ ] **9.3 [H]** Add `LICENSE` file (MIT) to repo root.
- [ ] **9.4 [F]** Draft **Methods** section (Architecture):
      ring attention + chunked NF4 cache + speculative verify loop +
      YaRN. Source: §M3_RING_INTEGRATION_PLAN.md + §M4_PLAN.md
      "Long-context memory equation". Include 1 architecture
      diagram (Mermaid → TikZ or Inkscape export).
- [ ] **9.5 [F]** Draft **Experiments** section (Setup + Results):
      pod hardware, model + tokenizer, p35/p35b/p36/p37 protocol,
      headline tables. Mostly rewrite from `M4_PLAN.md` "Phase C
      headline numbers" with paper voice.
- [ ] **9.6 [H]** Write figure + table captions for F1–F8. Captions
      must be **self-contained** (reviewer can read a figure without
      reading the text).
- [ ] **9.7 [H]** Mentor sign-off on the outline (response from 9.1).
      Adjust prose if needed.

### Week 10 (2026-05-19 → 2026-05-25) — finishing

- [ ] **10.1 [F]** Draft **Related Work** — cite ~20 papers:
      - Speculative decoding: Leviathan, Chen, SpecInfer, Medusa, EAGLE
      - Long-context inference: Ring Attention (Liu et al.), Tree
        Attention, BurstAttention, Striped Attention
      - RoPE extension: YaRN, Linear, NTK, LongRoPE, PoSE
      - KV quantization: KIVI, KVQuant, NF4 (bitsandbytes)
      - Long-context evaluation: LongBench, L-Eval, RULER, ∞Bench
      - StreamingLLM (outlier-keep)
      - Compare positioning against each (table or prose).
- [ ] **10.2 [H]** Draft **Limitations** subsection. Source: §M4_PLAN.md
      "Known limitations" (4 items: RULER deferred, 1M profile compute-
      bound, p35b orchestrator NoneType, single-instance only).
- [ ] **10.3 [H]** Draft **Abstract** (≤200 words) — 3 claims +
      headline number (4.4× at 4k, 1.0× scaling at 1M).
- [ ] **10.4 [H]** Draft **Introduction** — motivation, claim, three
      contributions matching the three RQs.
- [ ] **10.5 [H]** Draft **Conclusion** — restate the three claims +
      one paragraph on future work (long-context-fine-tuned base
      model + RULER/LongBench eval + multi-node TP).
- [ ] **10.6 [F]** **Public GitHub repo prep:**
      - New repo `rasd-paper-2026` (or similar)
      - `git filter-repo` or fresh history to drop wandb logs, runpod
        creds, internal debug chronicles
      - Copy `LICENSE`, `README.md`, `requirements-lock.txt`, src/,
        scripts/, tests/, configs/, results/final/*.csv, results/final/
        per_token/, results/final/generated/, figures/, tables/,
        analysis/error_analysis.md, M4_PLAN.md (trimmed)
      - **EXCLUDE**: `runpod_creds.md`, `checkpoint.md`, `*.log`,
        `wandb/`, `data/processed*/`, `results/m4_smoke/`,
        `results/phase_c/logs/`, `results/final/memory_trace_80gb/`
        (Phase B leftovers)
      - Push to GitHub, make public.
- [ ] **10.7 [H]** Write the new repo's **`README.md`** (M5 deliverable):
      Quick start (clone + venv + pip install), 1-page architecture
      overview with link to the paper, how to reproduce headline
      results (script + expected output), citation BibTeX placeholder.
- [ ] **10.8 [H]** Supplementary PDF: vanilla-RoPE PPL appendix,
      bimodality longer discussion, hardware/cost notes, full
      profiler tables.
- [ ] **10.9 [F]** **Full read-through revision cycle with mentor.**
      Mentor reads the assembled draft. Address comments. Repeat
      once if needed.
- [ ] **10.10 [H]** Final polish pass: grammar, citation formatting,
      figure positioning, page-budget check.
- [ ] **10.11 [H]** **Submit.** Upload `main.pdf` + `supplementary.pdf`
      to the venue portal. Capture submission ID. Email mentor
      confirmation.

### Internal deadlines (3-day buffer per brief)

| Milestone | Internal target | Real deadline |
|---|---|---|
| Outline + 3 RQs sent to mentor | 2026-05-13 (W9-D2) | — |
| Methods + Experiments drafted | 2026-05-15 (W9-D4) | — |
| Related Work + Abstract done | 2026-05-21 (W10-D3) | — |
| Repo public + README done | 2026-05-22 (W10-D4) | — |
| Mentor revision cycle done | 2026-05-23 (W10-D5) | — |
| **Internal submission target** | **2026-05-23** | — |
| **Workshop deadline** | (venue-dependent) | TBD |

---

## Risks + mitigations (from the mentor brief, augmented)

| Risk | Mitigation |
|---|---|
| **Story is weak.** | 9.1 + 9.7 mentor sign-off before prose. Three-RQ arc forces narrative structure. |
| **Time crunch.** | 3-day buffer (10.11 internal at 2026-05-23). 80% draft by start of Week 10. |
| **Repo cleanup leaks credentials.** | Use `git filter-repo` against a known-bad list (runpod_creds.md, *.log, wandb/). Verify with `git log --all --full-history -- runpod_creds.md` returning empty before push. |
| **Reviewer asks "where's LongBench / RULER?"** | Cited in 10.2 Limitations with `scripts/score_ruler_niah.py` referenced. RULER infrastructure ships with the paper; full eval is future work. The mentor signed off on this scope. |
| **Reviewer asks "what about a long-ctx-trained base model?"** | Section 10.5 Future Work: Llama-3.1-128k, Qwen-2.5-1M, MPT-7B-128k. RASD's contribution is the inference *system*; substituting the base model is straightforward. |
| **PDF page limit overflow.** | 10.10 has explicit page-budget check. Move excess content to `supplementary.pdf`. NeurIPS workshop typical limit is 4 + refs; if our target is 8 + refs (per the mentor brief), we have headroom. |
| **Figure/table rendering breaks Overleaf.** | 9.2 includes a build verification. Use vector PDFs not PNGs in the manuscript. |

---

## Companion docs

- `M4_PLAN.md` — upstream data work (frozen at tag `m4-phase-d-complete`)
- `PUBLICATION_STRATEGY.md` — long-range venue strategy (MLSys 2027
  primary, this M5 workshop submission is a stepping stone)
- `PHASE_C_RUNBOOK.md` — source for reproduction section
- `M3_RING_INTEGRATION_PLAN.md` — source for ring-attention prose

## Revision log

| Date | Change |
|---|---|
| 2026-05-11 | Initial draft post-M4 completion. Three-RQ arc locked. |

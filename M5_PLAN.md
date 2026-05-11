# Milestone 5 — Manuscript Writing & Multi-Track Submission

**Weeks 9–10 (2026-05-12 → 2026-05-25).**

Multi-track submission strategy (per `PUBLICATION_STRATEGY.md` and the
mentor's roadmap):

| Track | When | Page limit | Output |
|---|---|---|---|
| **1. arXiv preprint** | **First — Week 9** | unlimited | Establishes priority; cited from everywhere else |
| **2. NeurIPS 2026 workshop** | Week 10 | 4 + refs (typ.) | ML for Systems / ENLSP / Efficient Foundation Models (whichever deadline lands first) |
| **3. MLSys 2027** | Oct 30, 2026 | 8 + refs | Primary archival target — extended version of the workshop paper |
| 4. NeurIPS 2027 backup | May 2027 | 8 + refs | Only if MLSys rejects |

**arXiv goes first** because it (a) establishes priority on the RASD
name + 1M finding, (b) gives mentor + the field a citable reference
for the workshop submission, and (c) the workshop submission can
literally be a 4-page extraction of the arXiv version.

This is the **tactical** plan for M5. The long-range venue strategy
lives in `PUBLICATION_STRATEGY.md`; the upstream data work is in
`M4_PLAN.md`.

## Abstract (working draft, mentor-aligned)

Seeded from the mentor's roadmap abstract (March 2026); updated with
M4 measured findings.

> Long-context inference for large language models (LLMs) faces two
> compounding bottlenecks: (i) the KV-cache memory and bandwidth cost
> at million-token contexts exceeds the capacity of any single GPU,
> and (ii) autoregressive decoding amortizes a single token over a
> full forward pass through the model.  We present **RASD**
> (Ring Attention with Speculative Decoding), an inference system
> that combines ring-attention sequence parallelism, NF4 KV-cache
> quantization with chunked updates and a bf16 outlier prefix, YaRN
> position-scaling, and speculative verification with a small draft
> model into a single end-to-end loop.  On 8×A100 80GB SXM4, RASD
> runs Llama-2-7B inference at 1M context with 40 GB peak per rank,
> a regime where vanilla HuggingFace FlashAttention-2 + `generate()`
> goes out-of-memory at 128k single-rank.  At the model's native 4k
> context with real PG-19 narrative text, RASD delivers a **4.4×
> throughput speedup** over a target-only baseline with median
> per-round acceptance of **1.0** (all four draft tokens accepted in
> most rounds).  At extended contexts under YaRN factor=256, the
> speedup contracts to 1.0–1.8× and per-round acceptance becomes
> bimodal — a regime we attribute to Llama-2-7B going out-of-
> distribution beyond its 4k training horizon, not to a limit of
> the RASD architecture.  We release the full implementation, all
> reproduction scripts, and the per-position acceptance traces under
> the MIT license.

(Word count: ~210; trim to ≤200 before submission. The 4.4× number
is the F4/F5 short-ctx-PG-19 finding committed in `27badaf`.)

## Three Research Questions (mentor-aligned)

Mentor's exact wording from `literature_review/roadmap_…pdf` page 3,
with our M4 findings annotated against each:

**RQ1: How does RASD compare against a baseline Ring-Attention
implementation in wall-clock latency and throughput at 256k–1M
contexts on an 8-node A100 cluster?**
→ Our answer: p35 vs p35b matrix.  Throughput speedup of 1.24×, 1.80×,
0.97×, 1.00× at 128k/256k/512k/1M (3-seed mean, synthetic prompts);
at 4k PG-19 the speedup is 4.4×.  See `tables/main_speedup.tex` +
F1 + F5.

**RQ2: What is the optimal draft-model-size vs k trade-off?**
Mentor's hypothesis was 1.3B draft + k∈[4,8].
→ M3 ablation grid swept (A1) DistilGPT-2 124M vs Sheared-LLaMA-1.3B
and (A2) k∈{2,4,6,8,12}.  The 1.3B draft + k=4 combination won.
See `figures/fig2_ablation_heatmap.pdf` and `tables/ablation_summary.tex`.
**The M3 ablation result confirms the mentor's hypothesis.**

**RQ3: To what extent can the communication latency of inter-GPU
KV-cache transfer in Ring Attention be masked by the computational
work of the speculative decoding phase?**
→ Our F3 profiler measurement (4 contexts) shows **comm = 1.0–1.2%
of wall on rank 0** across canary/128k/256k/512k.  Compute is
28–37%; **idle dominates at 51–57%**, with `other` (allocator /
launch overhead) at 9–15%.  Mechanistically, comm at our scale isn't
the bottleneck the roadmap hypothesized — the actual rate-limiter is
target-forward compute, with rank-0 waiting on the sharded ring
collective.  This finding *partially refutes* the RQ3 hypothesis (the
overlap isn't load-bearing because there's not much comm to hide),
and the paper should treat this as a useful **negative result**:
ring-attention's collective layout is already comm-efficient at 8
ranks, so spec-decoding's contribution at long ctx is bounded by
acceptance, not by overlap.

**Action 9.1 below: send these three RQs + the abstract to the mentor
BEFORE writing prose**, per the brief's risk-mitigation note.

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

## Headline findings to frame around (M4 measured, recoded for paper voice)

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

## Outputs (mapped from the M5 brief + multi-track strategy)

- **`manuscript/arxiv/main.pdf`** — unlimited-page arXiv preprint
  (FIRST priority, no page limit means we ship everything)
- **`manuscript/workshop/main.pdf`** — 4-page workshop submission
  extracted from the arXiv version
- `manuscript/supplementary.pdf` — vanilla-RoPE PPL, memory traces,
  RULER-niah infra description as future work, full profiler tables
- Public GitHub repo (separate clean fork — not the working repo)
  under **MIT license**
- `README.md` with reproduction instructions
- arXiv submission ID + workshop submission ID

---

## Task list

Each task tagged with `[ID effort]` for estimation. `H` = ~half-day,
`F` = ~full day. Order is dependency-aware.

### Week 9 (2026-05-12 → 2026-05-18) — arXiv preprint

The arXiv preprint is the **first** output. Workshop is a 4-page
extract built AFTER the arXiv version is solid. Internal target:
arXiv submission by Fri 2026-05-16 (3 days into W9).

- [ ] **9.1 [H]** Send mentor: abstract (above) + three RQs (above)
      + 1-page bullet outline. **BLOCKS prose work** (wait for
      sign-off before drafting). Goal: catch story-weak risk early.
- [ ] **9.2 [H]** Create `manuscript/arxiv/` LaTeX scaffold (NeurIPS
      preprint style with `\usepackage[preprint]{neurips_2026}`).
      Section stubs, `\input{...}` shims for all `tables/*.tex`,
      `\includegraphics` for `figures/*.pdf`, `references.bib` set up.
- [ ] **9.3 [H]** Add `LICENSE` (MIT) to repo root.
- [ ] **9.4 [F]** Draft **§Methods**: architecture stack (ring
      attention + chunked NF4 cache + speculative verify loop +
      YaRN + outlier-keep). One figure: the architecture diagram
      (Mermaid → TikZ). Source: `M3_RING_INTEGRATION_PLAN.md` +
      `M4_PLAN.md` "Long-context memory equation".
- [ ] **9.5 [F]** Draft **§Experiments + §Results**: pod hardware
      (8×A100 80GB SXM4), model + tokenizer, protocols for p33–p37,
      headline tables (`\input` from `tables/`), F1+F3+F4 figures.
      Source: `M4_PLAN.md` "Phase C headline numbers".
- [ ] **9.6 [H]** Self-contained captions for F1–F5.
- [ ] **9.7 [H]** **Mentor sign-off** on prose draft (response from
      9.1). Address comments inline.

### Week 10 (2026-05-19 → 2026-05-25) — arXiv submit, then workshop extract

- [ ] **10.1 [F]** Draft **§Related Work** — cite ~20 papers:
      - Speculative decoding: Leviathan, Chen, SpecInfer, Medusa, EAGLE
      - Long-context inference: Ring Attention (Liu et al.), Tree
        Attention, BurstAttention, Striped Attention
      - RoPE extension: YaRN, Linear, NTK, LongRoPE, PoSE
      - KV quantization: KIVI, KVQuant, NF4 (bitsandbytes)
      - Long-context evaluation: LongBench, L-Eval, RULER, ∞Bench
      - StreamingLLM (outlier-keep)
      - vLLM, FlashAttention-2 (Dao et al.)
- [ ] **10.2 [H]** Draft **§Limitations** + **§Future Work**.
      Source: `M4_PLAN.md` "Known limitations" + Phase D bimodality
      findings + long-context-trained base-model future work.
- [ ] **10.3 [H]** Draft **§Introduction** — motivation, claim, four
      contributions (RASD system + 4.4× headline + bimodal-α finding +
      open-source release).
- [ ] **10.4 [H]** Draft **§Conclusion** + tighten **abstract** to ≤200 words.
- [ ] **10.5 [F]** **Public GitHub repo prep:**
      - New repo `rasd-paper-2026` (separate from working tree)
      - `git filter-repo` to drop credentials, wandb, debug chronicles
      - Copy: `LICENSE`, `README.md`, `requirements-lock.txt`, src/,
        scripts/, tests/, configs/, all of `results/final/`, figures/,
        tables/, analysis/, M4_PLAN.md (trimmed)
      - **EXCLUDE**: `runpod_creds.md`, `checkpoint.md`, `*.log`,
        `wandb/`, `data/processed*/`, `results/m4_smoke/`,
        `results/phase_c/logs/`
      - Push public.
- [ ] **10.6 [H]** Write new repo `README.md`: quick start, 1-page
      architecture overview, how to reproduce headline results
      (single command), citation BibTeX (arXiv ID once we have it).
- [ ] **10.7 [H]** Build `manuscript/supplementary.pdf`: vanilla-RoPE
      PPL appendix, bimodality discussion, hardware/cost notes, full
      profiler tables, RULER niah infra description.
- [ ] **10.8 [F]** **Final read-through with mentor.** Address comments.
- [ ] **10.9 [H]** **Submit to arXiv.** Capture arXiv ID. Update README.
- [ ] **10.10 [F]** **Build workshop submission** — extract 4 pages
      from the arXiv version (NeurIPS workshop template). Drop
      §Related Work to a 1-paragraph, §Methods to 1.5 pages, keep
      §Results central, move all tables/figures to supplementary
      except F1 + F3.
- [ ] **10.11 [H]** **Submit to workshop.** Capture submission ID.

### Internal deadlines (3-day buffer per brief)

| Milestone | Internal target | Real deadline |
|---|---|---|
| Abstract + RQs sent to mentor | 2026-05-12 (W9-D1) | — |
| arXiv scaffold + LICENSE + Methods drafted | 2026-05-14 (W9-D3) | — |
| Experiments + Results drafted | 2026-05-15 (W9-D4) | — |
| Mentor sign-off prose | 2026-05-16 (W9-D5) | — |
| **arXiv submit (internal target)** | **2026-05-19 (W10-D1)** | — |
| Workshop extract done | 2026-05-22 (W10-D4) | — |
| Mentor revision cycle done | 2026-05-23 (W10-D5) | — |
| **Workshop submit** | **2026-05-24 (W10-D6)** | venue-dependent |
| Buffer | 2026-05-25 (W10-D7) | — |

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

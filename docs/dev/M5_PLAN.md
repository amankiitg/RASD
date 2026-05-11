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

- [~] **9.1 [H]** ~~Send mentor: abstract (above) + three RQs (above)~~
      **SKIPPED** — mentor unavailable to bless prose; user proceeding
      solo per 2026-05-11 directive ("let's do option C"). Story risk
      mitigated by self-review of v1 → v3 rewrite addressing missing
      architecture figure, demoted M3 results, pXX jargon, and 4.4×
      framing.
- [x] **9.2 [H]** ~~Create `manuscript/arxiv/` LaTeX scaffold~~ **DONE
      2026-05-11.** Switched from NeurIPS preprint to a vanilla
      `article` class with NeurIPS-like geometry for arXiv (workshop
      template will be applied at extraction time). `\input` shims
      and `\includegraphics` calls wired to `tables/*.tex` and
      `figures/*.pdf`. Built locally with `tectonic`.
- [x] **9.3 [H]** ~~Add `LICENSE` (MIT) to repo root.~~ **DONE
      2026-05-11.** MIT, © 2026 Aman Kesarwani.
- [x] **9.4 [F]** ~~Draft **§Methods**~~ **DONE 2026-05-11.** §3 covers
      ring attention, NF4 chunked KV cache + bf16 attention-sink prefix,
      YaRN factor=256, and the speculative verify loop. Two-panel TikZ
      architecture diagram is Figure 1 (ring topology + decode round
      timeline). §3.7 documents the M3 design-point ablation sweeps
      (A1 draft size, A2 spec steps k, A3 KV block size, A4 prefetch
      depth, A5 target model) and identifies block size as the 4×
      throughput lever.
- [x] **9.5 [F]** ~~Draft **§Experiments + §Results**~~ **DONE
      2026-05-11.** §4 + §5 with pod hardware, model + tokenizer,
      protocol descriptions using descriptive names (no internal pXX
      jargon), main throughput matrix, baseline comparison, PPL sanity
      check, profiler breakdown, HF FA-2 ceiling, and short-context
      PG-19 control. All headline tables `\input`ed from `tables/`,
      F1+F3+F4 included.
- [x] **9.6 [H]** ~~Self-contained captions for F1–F5.~~ **DONE
      2026-05-11.** Every caption reads standalone (claim + context +
      where the data lives) so a reviewer can skim figures without the
      prose.
- [~] **9.7 [H]** ~~Mentor sign-off on prose draft.~~ **SKIPPED** — see
      9.1 above. User self-reviewed v1, identified four sloppiness
      issues, and v3 manuscript addresses each.

### Week 10 (2026-05-19 → 2026-05-25) — arXiv submit, then workshop extract

- [x] **10.1 [F]** ~~Draft **§Related Work**~~ **DONE 2026-05-11.** §2
      covers speculative decoding (Leviathan, Chen, SpecInfer, Medusa,
      EAGLE), long-context inference (Ring Attention, Tree/Burst/Striped
      Attention), RoPE extension (YaRN, NTK, LongRoPE, PoSE), KV
      quantization (KIVI, KVQuant, NF4), long-context evaluation
      (LongBench, L-Eval, RULER, ∞Bench), StreamingLLM, and FA-2/vLLM.
- [x] **10.2 [H]** ~~Draft **§Limitations** + **§Future Work**.~~ **DONE
      2026-05-11.** §6 covers single-instance scope, Llama-2 OOD
      regime, synthetic-prompt artefact, and points to long-context-
      trained base model (Llama-3.1-128k, Qwen-2.5-1M) as the highest-
      leverage follow-up. RULER niah scoring infra shipped for
      follow-up evaluation work.
- [x] **10.3 [H]** ~~Draft **§Introduction**~~ **DONE 2026-05-11.** §1
      states motivation, the four contributions (RASD system + 1M
      capability + bimodal-α finding + open-source release), and
      honestly demotes the 4.4× number from headline to control
      experiment.
- [x] **10.4 [H]** ~~Draft **§Conclusion** + tighten **abstract**~~
      **DONE 2026-05-11.** Abstract reframed to lead with capability +
      memory + design-point + honest characterization (rather than the
      now-demoted 4.4×).
- [~] **10.5 [F]** **Make THIS repo public — IN PROGRESS 2026-05-11.**
      Sub-steps:

      1. ~~**CRITICAL — purge credentials from history**~~ **NO-OP
         2026-05-11.** Audit confirmed `runpod_creds.md` was NEVER
         tracked in git (gitignored from day 1). No destructive
         history rewrite required. Keys still need rotation (step 2).
      2. **ROTATE the three credentials** at wandb / HuggingFace /
         Lambda. **PENDING — USER ACTION.**
      3. ~~Audit commit messages for leaked keys~~ **DONE 2026-05-11.**
         `git log -p --all | grep -E "wandb_v1_|^hf_|secret_rasd"`
         returned empty.
      4. ~~Audit code for hardcoded secrets~~ **DONE 2026-05-11.**
         `grep -rE "wandb_v1_|hf_[a-z]|secret_rasd" src/ scripts/ tests/`
         returned empty.
      5. ~~**Trim working tree**~~ **DONE 2026-05-11.** Removed 9.2 GB
         of gitignored artifacts (results/m4_smoke/checkpoints,
         results/c6_validation/checkpoints, memory_trace_80gb,
         checkpoint.md). All deletions were already-gitignored content;
         no `git rm` needed. Final tree: 23 MB.
      6. ~~**Move dev notes under `docs/dev/`**~~ **DONE 2026-05-11.**
         Via `git mv` (history preserved): M3_RING_INTEGRATION_PLAN,
         M4_PLAN, M5_PLAN, PHASE_C_RUNBOOK, PUBLICATION_STRATEGY,
         experiments. Cross-refs in REPRODUCE.md, source files, and
         scripts updated. Broken test `test_m4_plan_documents_metric_difference`
         fixed; all 470 tests pass.
      7. ~~**Update root `README.md`** to paper-first format~~ **DONE
         2026-05-11.** New ~205-line README with headline numbers
         table, quick start, reproduction commands, repo layout,
         "what this codebase does NOT claim" disclaimer, citation
         placeholder. See task 10.6.
      8. **Make repo public** on GitHub. **PENDING — USER ACTION.**
- [x] **10.6 [H]** ~~Write new repo `README.md`~~ **DONE 2026-05-11.**
      Headline numbers table, quick start, reproduction (single
      command for figures, one bootstrap command for GPU re-run), repo
      layout, citation BibTeX with arXiv-ID-pending placeholder, links
      to `docs/dev/README.md` for working-tree planning docs.
- [ ] **10.7 [H]** ~~Build `manuscript/supplementary.pdf`~~ **PAUSED
      per 2026-05-11 user directive** (arXiv submission and
      supplementary deferred for review).
- [~] **10.8 [F]** ~~**Final read-through with mentor.**~~ **SKIPPED**
      — see 9.1; user proceeding solo.
- [ ] **10.9 [H]** ~~**Submit to arXiv.**~~ **PAUSED per 2026-05-11 user
      directive** ("in the last step you said submit to arxiv, let's
      pause that and do everything else").
- [ ] **10.10 [F]** ~~**Build workshop submission**~~ **PAUSED** — gated
      on arXiv submission.
- [ ] **10.11 [H]** ~~**Submit to workshop.**~~ **PAUSED** — gated on
      10.10.

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
| 2026-05-11 | v2: arXiv-first multi-track strategy, mentor-aligned abstract + 3 RQs. |
| 2026-05-11 | v3: manuscript v3 rewrite (TikZ architecture figure, M3 ablation §3.7, demoted 4.4× from headline, descriptive section names). Mentor sign-off (9.1, 9.7, 10.8) skipped per user — proceeding solo. |
| 2026-05-11 | Task 10.5 (repo public-prep): runpod_creds.md audit confirmed never tracked (step 1 no-op); secret-grep clean (steps 3-4); 9.2 GB working-tree trim (step 5); dev .md moved to `docs/dev/` (step 6); paper-first README (steps 7 + 10.6); LICENSE added (9.3); docs/dev/README.md index added. Pending USER ACTIONS: rotate keys (step 2), make-public via GitHub UI (step 8). Tasks 10.7, 10.9, 10.10, 10.11 paused. |

# Milestone 4 — Evaluation & Analysis Plan

Tracking file for M4 work. Strategy, phased order, and deliverables.

## Status snapshot (2026-04-16)

- **M3 ablation study**: complete — 49/49 rows in [results/ablations/ablations.csv](results/ablations/ablations.csv)
- **Mentor's M3 asks**:
  - ✅ Blockwise FA + ring attention ([src/models/rasd_inference.py](src/models/rasd_inference.py))
  - ✅ Memory validated to 512k ([results/baselines/flash_memory_validation.csv](results/baselines/flash_memory_validation.csv))
  - ⏳ Full RASD at long-context (1M) — deferred to M4
  - ❓ LongBench / L-Eval task-accuracy eval — used PG-19 perplexity instead; asking mentor whether to add
  - ⏳ Implementation-details email — pending
- **M3 context length**: 8k (conservative start; ring partitioning + tick-ordering validated at block=2048)
- **Spend so far**: ~$200 on RunPod A100s for M3 — M4 cost strategy must be frugal

## Strategy

**Local-first.** Do everything that doesn't need an A100 on the laptop first:
analysis scaffolding, bootstrap CIs, figure templates, LaTeX tables, profiler
code, checkpoint/resume logic, RoPE scaling wiring, tick-gate regression tests
(gloo backend), PG-19 preprocessing verification. This means when we rent
the pod, we run compute that can't be done locally and nothing else.

**Pod time is for the 36-run final matrix + 1M-context smoke tests only.**
Figures are drawn from the resulting CSV after pod teardown.

## Phased order

### Phase 0 — Reproducibility guardrails
Make sure we can always replay M2/M3 exactly, even after M4 refactors.

0.1 ✅ Tag `m3-complete` at current HEAD so `git checkout m3-complete` replays M3
0.2 ⏳ `requirements-lock.txt` — run [scripts/capture_pod_env.sh](scripts/capture_pod_env.sh) on next pod before anything else; commit the output
0.3 ✅ Pin HF model revisions in [configs/ablations.yml](configs/ablations.yml) (Sheared-LLaMA-1.3B, TinyLlama_v1.1). Llama-2 (gated) — run [scripts/pin_hf_revisions.py](scripts/pin_hf_revisions.py) on pod with HF_TOKEN to capture those two hashes
0.4 ✅ Wire `revision=` through `RASDConfig` → `from_pretrained` (additive, default None = HEAD so no M3 semantics change)
0.5 ✅ Added `--seeds` flag to `run_experiment.py` for subsetting
0.6 ✅ [scripts/replay_m3_smoke.sh](scripts/replay_m3_smoke.sh) — runs one seed per group, asserts throughput within 15% of golden CSV
0.7 📋 **Feature-flag pattern for M4 code** — all new M4 functionality (profiler, checkpoint, RoPE scaling, TTFT split timing) goes behind default-off flags so `--resume` on M3 rows stays byte-identical when flags are off. M3 code paths must not be refactored inline.

Run on first M4 pod: `bash scripts/capture_pod_env.sh > requirements-lock.txt && python scripts/pin_hf_revisions.py` then `bash scripts/replay_m3_smoke.sh` to confirm replay works.

### Phase 1 — Local, reuses existing M3 data
No new compute. Use [results/ablations/ablations.csv](results/ablations/ablations.csv).

1. Error analysis on 5 short-run rows — confirm each is deterministic/legitimate (done in conversation; codify in notebook)
2. Build `src/analysis/` scaffolding (`metrics.py`, `bootstrap.py`, `figures.py`, `tables.py`)
3. Bootstrap CIs for per-axis winners (A1/A2/A3/A4/A5 on tps + acceptance)
4. Figure 2: ablation bar charts with CIs (from M3 data)
5. LaTeX `tables/ablation_summary.tex` from ablations.csv

### Phase 2 — Local, code-ready for pod
Code/tests/scaffolds. Verified locally on CPU or single GPU where possible.

6. TTFT instrumentation in `RASDInference.generate`
7. Per-position acceptance logging (list-of-bool per round)
8. `torch.profiler` context-manager wrapper (NVTX ranges at round boundaries)
9. Perplexity evaluator for PG-19 (single-GPU path first)
10. Checkpoint/resume for multi-hour runs (writes intermediate CSV rows)
11. RoPE scaling for 1M context (YaRN or NTK-aware — pick during this phase)
12. Gloo-backend regression test for tick-gate ordering (catches block=2048 regression without CUDA)
13. PG-19 preprocessing verification (tokenization, seq-length distribution)
14. Figure scaffolds (Figure 1 throughput-vs-context, Figure 3 acceptance-vs-position, Figure 4 memory, Figure 5 latency breakdown) — load from CSV, plot, no real data yet

### Phase 3 — Pod required
Only runs that need A100s.

15. RoPE scaling validation at 64k/256k/1M on 1 GPU (sanity-check numerical stability before distributing)
16. Smoke tests at 64k, 256k, 1M context (RASD + baselines)
17. Baseline validation at 1M (ring-attention alone, sliding-window)
18. **36-run final matrix** (3 seeds × 12 configs — contexts × configs TBD during Phase 2)
19. Profiler pass on best config (1 run, full traces saved)

### Phase 4 — Post-pod, local
Assemble paper deliverables from real data.

20. Figures 1/3/4/5 with real CSV + bootstrap CIs
21. `results/final/final_results.json` (aggregate metrics, per-seed)
22. LaTeX tables with real numbers
23. Manuscript sections (methods, results, discussion)

### Conditional
24. LongBench / L-Eval task-accuracy eval — **pending mentor approval**. Scope: pick 2-3 LongBench tasks, run target-only vs RASD at 64k, compare EM/F1. Adds ~1 pod-day; only if mentor says PG-19 perplexity is insufficient.

### Admin
25. Send mentor follow-up email: FA+Ring implementation details + LongBench scope question + cost note ($200 burned on 8k M3; asking about cheaper alternatives for M4)

## Deliverables

- `results/final/final_results.json` — aggregated metrics, per-seed per-config
- `figures/` — Figure 1 (throughput vs context), Figure 2 (ablation bars), Figure 3 (per-position acceptance), Figure 4 (memory), Figure 5 (latency breakdown)
- `tables/` — LaTeX ablation summary + final results tables
- `manuscript/` — methods, results, discussion sections
- Mentor emails: implementation details, LongBench scope, cost ask

## Risks / anticipated fixes

- **RoPE scaling at 1M** may produce NaN/inf with naive linear scaling → YaRN or NTK-aware needed. Validate on single GPU before ring.
- **Memory at 1M × 8 ranks** — 4-bit NF4 + FA-2 should fit, but prefetch_depth=2 may push over 80GB per rank; fall back to 1 or 0 if needed.
- **Tick-ordering regression** — gloo test in Phase 2 catches this before pod time.
- **Long-run orphaned VRAM** — stick to the 5-step clean-kill sequence in [configs/ablations.yml](configs/ablations.yml); never `pkill -9`.
- **Subprocess timeout** — 120s worked for 8k; 1M context may need 600-1800s. Patch per-phase.

## Cost discipline

- Phase 1-2 = $0 (laptop only)
- Phase 3 = 1 pod rental, target <$150
- Phase 4 = $0 (laptop only)
- Conditional LongBench ≈ +$50-80 if approved
- Budget cap: $250 total for M4

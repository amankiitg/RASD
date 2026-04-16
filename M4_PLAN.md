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

**Local-first, priority-ordered.** Two parallel tracks:

- **Analysis track** (local, $0): extract everything possible from existing
  M3 data — bootstrap CIs, Figure 2, ablation tables, error analysis.
  Unblocks paper writing immediately.
- **Compute track** (local code → pod runs): priority-ordered by paper
  evidence value:
  1. **RoPE scaling** — blocker for 1M context; no eval possible without it
  2. **Perplexity + throughput** — the two numbers the paper needs
  3. **Checkpoint/resume** — cost protection at 20+ min/run on pod
  4. **Minimal profiler** — only if we need a "why fast" story
  5. **Tick-gate gloo test** — cheap, guards commit dc14915
  (deprioritized: TTFT split timing, per-position acceptance sidecar)

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

### Two parallel tracks

**Analysis track** and **Compute track** run in parallel. Analysis is all
local, runs on existing M3 data, and unblocks paper writing independently.
Compute track is ordered by priority + hard dependencies so we spend the
minimum pod-$ for the strongest story.

### Analysis track — local, reuses existing M3 data
No new compute. Use [results/ablations/ablations.csv](results/ablations/ablations.csv).

A1. Error analysis on 5 short-run rows — confirm each is deterministic/legitimate (done in conversation; codify in notebook)
A2. Build `src/analysis/` scaffolding (`metrics.py`, `bootstrap.py`, `figures.py`, `tables.py`)
A3. Bootstrap CIs for per-axis winners (A1/A2/A3/A4/A5 on tps + acceptance)
A4. Figure 2: ablation bar charts with CIs (from M3 data)
A5. LaTeX `tables/ablation_summary.tex` from ablations.csv

### Compute track — priority-ordered (reprioritized 2026-04-16)

Ordering reflects paper-evidence value: **RoPE is the 1M blocker, PPL+throughput
is the core claim, checkpoint/resume is pod-$ protection, everything else is
secondary.** TTFT and per-position acceptance are deprioritized — TTFT is a
product-serving metric (not our research angle), per-position Fig 4 can be
approximated from existing round-level logs.

#### P1 — RoPE scaling (BLOCKER for 1M)
C1. RoPE scaling code path (YaRN or dynamic-NTK — pick when starting this item) behind `--rope-scaling` flag, default off → M3 unchanged
C2. Smoke-test RoPE numerically on single GPU at 64k locally (if laptop can fit; else first pod action)

#### P2 — Perplexity + throughput (core evidence)
C3. PG-19 preprocessing verification (tokenization, seq-length distribution, produces 1M-token chunks)
C4. Perplexity evaluator `src/analysis/perplexity.py` — tested on `sshleifer/tiny-random-llama` locally before pod
C5. Wire PPL logging into `run_experiment.py` alongside existing throughput metrics (sidecar, additive column)

#### P3 — Checkpoint/resume (pod-$ protection)
C6. Generation checkpoint/resume — write state every N tokens, resume on failure. At 1M context a single run is 20+ min; one crash = one pod-hour lost without this.

#### P4 — Minimal profiler (conditional)
C7. `torch.profiler` context-manager wrapper with NVTX ranges at round boundaries — **only build if P1-P3 land and we still need a "why fast" story for the paper**. Skip entirely if results speak for themselves.

#### P5 — Tick-gate regression test (cheap, high-value)
C8. Gloo-backend regression test for tick-gate ordering (catches a block=2048 regression without CUDA). Tiny, guards commit dc14915.

#### Deprioritized / likely dropped
~~TTFT split timing~~ — product-serving metric, not research-paper evidence
~~Per-position acceptance sidecar .jsonl~~ — approximate from round-level logs for Figure 4

### Phase 3 — Pod required
Only runs that need A100s. Order reflects dependencies on Compute-track items above.

P3.1. Run Phase 0 completion on first pod (capture_pod_env.sh → `requirements-lock.txt`, pin_hf_revisions.py → Llama-2 hashes, replay_m3_smoke.sh → confirm drift ≤15%)
P3.2. RoPE scaling validation: PPL at 32k/128k/512k/1M on 1 GPU (needs C1+C4)
P3.3. Smoke tests: single RASD run at 32k, 128k, 512k, 1M context
P3.4. Baseline validation: Ring + Sliding end-to-end at 128k, 1M
P3.5. **Final 36-run matrix** — RASD+Ring+Sliding × {128k, 256k, 512k, 1M} × 3 seeds
P3.6. Profiler pass (only if C7 built)

### Phase 4 — Post-pod, local
Assemble paper deliverables from real data.

F1. Figure 1 throughput vs context (real CSV + 95% CI bands)
F2. Figure 3 stacked time breakdown (only if profiler ran; else drop)
F3. Figure 4 acceptance vs token position (approximated from round-level logs)
F4. Figure 5 memory footprint RASD vs baselines
F5. `results/final/final_results.json` (aggregate metrics, per-seed)
F6. LaTeX tables with real numbers
F7. Manuscript sections (methods, results, discussion)

### Conditional
X1. LongBench / L-Eval task-accuracy eval — **pending mentor approval**. Scope: pick 2-3 LongBench tasks, run target-only vs RASD at 64k, compare EM/F1. Adds ~1 pod-day; only if mentor says PG-19 perplexity is insufficient.

### Admin
Z1. Send mentor follow-up email: FA+Ring implementation details + LongBench scope question + cost note ($200 burned on 8k M3; asking about cheaper alternatives for M4)

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

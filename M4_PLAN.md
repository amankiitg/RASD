# Milestone 4 — Evaluation & Analysis Plan

Tracking file for M4 work. Strategy, phased order, and deliverables.

## Current state (2026-05-06 — updated post Phase B)

Four phases total. **Phase A and Phase B are complete; Phase C is next; Phase D is the last.**

| Phase | What | Status | Tag |
|---|---|---|---|
| **A** | Analysis track (figures + sidecars + PPL + profiler primitives) | ✅ DONE | rolled into `m4-phase-b-complete` |
| **B** | Compute track local (C3 + C5 + C2b + C6 + C11 codec) | ✅ DONE | [`m4-phase-b-complete`](../../tree/m4-phase-b-complete) |
| **C** | Pod session: validation gates + matrix + profiler pass | ⚠ blockers fixed; **#1 scope decision pending** | TBD post-pod |
| **D** | Post-pod paper deliverables (Fig 1/3/4/5, tables, manuscript) | ⏳ blocked on Phase C data | — |

**Phase B local commits (in order):**
- `28d4517` C3 — PG-19 preprocess refactor + smoke tests (12 tests)
- `75b7ff8` C5 — TTFT + per-position trace wired into launcher (16 tests)
- `6e3bdb0` C2b — YaRN RoPE config wiring (17 tests)
- `611045c` C6 — Generation checkpoint/resume rank-aware (41 tests)
- `55efa28` C11 — NF4 KV-cache codec + cache wrapper (51 tests)

**Phase A commits (in order):**
- `69ecbac` A1 — M3 invariant regression tests (Option B / Fix2 / Fix3 / Fix4)
- `9eace0e` A2 — TTFT (C12) + per-position acceptance trace (C13) inside `generate()`
- `aabb53c` A3 — Fig 2 heatmap rewrite, R6.5 outputs, R6.5-aligned `error_analysis.md`
- `07f3807` A4 — `torch.profiler` wrapper for Fig 3 stacked time breakdown
- `b71b65d` A5 — Perplexity evaluator (sliding-window)

**Default-off invariants preserved:** `cfg.checkpoint_every == 0`,
`cfg.log_per_token == False`, `cfg.rope_type == "linear"` all give
M3-byte-identical execution. The M3 invariant tests
(`tests/test_m3_invariants.py`) all still pass — Option B / Fix2 / Fix3 /
Fix4 are locked.

**Phase C prep landed (scripts ready for the pod session):**
- `scripts/phase_c_pod_session.sh` — master orchestration
- `scripts/c11_validation.py` — bf16-vs-NF4 validation gate
- `scripts/yarn_numeric_validation.py` — factor=128/256 NaN/PPL check
- `scripts/c6_resume_validation.py` — multi-rank checkpoint+resume sanity
- `PHASE_C_RUNBOOK.md` — single-page operator checklist
- Pod-side reproducibility lockdown bundled into Phase C P3.1
  (capture_pod_env.sh + pin_hf_revisions.py)

Test count: **283 passed** (104 M3 invariants + 179 M4 additions).

## Mentor M4 spec alignment (2026-05-06)

The mentor's M4 brief asks for: final 1M-context runs vs baselines × 3 seeds,
bootstrap CIs, 5 publication figures, LaTeX tables, error analysis on
low-α sequences, and 5 metrics (tps, latency/token, α, PPL, TTFT).
Mapping each mentor ask to a plan section so nothing falls through:

| Mentor ask | Plan section | Status |
|---|---|---|
| Final 1M × baselines × 3 seeds | P3.5 (36-run matrix) | code: pending |
| Bootstrap CIs | A3 + `src/analysis/bootstrap.py` | scaffolded |
| Throughput (tps) | already logged in M3 CSV | ✅ |
| Latency per output token (ms) | derived from tps + token count; add column | pending |
| Acceptance rate α | already logged | ✅ |
| Perplexity (PPL) | C3-C5 (PG-19 + evaluator + sidecar) | pending |
| **TTFT** (un-deprioritized 2026-05-06) | new sidecar timer in `_prefill` | pending |
| **Fig 1** throughput vs context, 95% CI bands | F1 | pending — needs Phase 3 data |
| **Fig 2** heatmap throughput × draft_size × spec_steps | A4 (revised from bars to heatmap) | rewrite needed |
| **Fig 3** stacked time breakdown compute/comm/idle | C7 (promoted from conditional → required) | pending |
| **Fig 4** α vs token position | F3 + per-position sidecar (un-deprioritized) | pending |
| **Fig 5** qualitative text comparison table | F5 (new) | pending |
| LaTeX tables | A5 / F6 (`pandas.to_latex()`) | scaffolded |
| `analysis/error_analysis.md` on low-α sequences | F4 (consolidated path) | pending |
| Checkpoint/resume for long runs | C6 (mentor risk mitigation) | pending |
| GPU health checks before final runs | new Phase 3 preflight | pending |
| Negative-result pivot path | bottleneck story = Fig 3 + profiler insurance | covered by C7 promotion |

## M3 → M4 configuration inheritance (2026-05-06)

R6.5 settled the M3 ablation axes. M4's 36-run final matrix should NOT
re-ablate them — it should **lock all knobs at M3's winners** and vary
only the cross-method × cross-context grid that the paper actually
needs. This saves ~3-4× the compute and keeps the matrix's signal clean.

| M3 axis | R6.5 finding | M4 lock |
|---|---|---|
| **A2** spec_steps k | k=4 / k=8 tps-equivalent; α monotonic in k | `spec_steps=4` |
| **A3** kv_block_size | tps monotonic, 2048 wins ~40% over default 512 | **`kv_block_size=2048`** (revise from M3 default 512) |
| **A4** prefetch_depth | sync == async-1 == async-2 (identical to logged precision) | `prefetch_depth=1`; do NOT sweep |
| **A1** draft size | TinyLlama-1.1B ≈ Sheared-LLaMA-1.3B (within seed noise) | `draft = princeton-nlp/Sheared-LLaMA-1.3B` |
| **A5** target | 7B works at 64k×W=8 NF4; 13B OOMs on 40 GB | `target = meta-llama/Llama-2-7b-hf`; 13B revisits *after* C11 lands |

**M4 final-matrix shape (36 runs)**: method × context × seed
= {RASD, Ring, Sliding} × {128k, 256k, 512k, 1M} × {42, 123, 456}.
All other knobs locked at M3 winners above.

This is a structural simplification: M3 was an ablation along 5 axes
at fixed ctx; M4 is a method-comparison along 1 axis (ctx) at fixed
ablation winners. The two milestones produce complementary evidence.

## Status snapshot (2026-05-06 — major revision)

> **What changed since the 2026-04-16 snapshot below this section:** the
> 2026-04-16 audit ([results/quant_mini/check4_audit.md](results/quant_mini/check4_audit.md))
> found that the original M3 implementation had **four coupled defects**
> in the speculative-verify path that were suppressing α to 0.06–0.11
> across all 49 ablation rows, plus a deeper architectural gap where
> **ring attention's prefetched K/V was never consumed by the target's
> attention forward** — at world_size>1 each rank actually held the full
> sequence's K/V locally and the ring P2P was decorative. Three weeks
> of redesign work landed on main as the **R0–R3.5 + R5 + R6.x** task
> sequence in [M3_RING_INTEGRATION_PLAN.md](M3_RING_INTEGRATION_PLAN.md).
> Ring attention now correctly lives inside `LlamaAttention.forward`
> with a dual-cache (sharded prefill + replicated decode tail) layout.

**Current state of M3 (2026-05-06):**

- **M3 ablation study**: ❌ INVALIDATED 2026-04-16 (post-analysis bug found).
  Original 49-row CSV not paper-defensible. Re-ablation **R6.5 PENDING**;
  gated on Issues #1, #2 in the M3 plan doc.
- **Code (R0–R3.5, R5)**: ✅ landed across 11 commits since 2026-05-05.
  78/78 unit tests green. Architecture is now correct.
- **R6 validation matrix** (Lambda Cloud, 8x A100 SXM):
  - R6.1 single-rank smoke (NF4 α=0.654 / bf16 α=0.682) ✅ 2026-05-05
  - R6.2 2-rank 8k smoke (α=0.312, 28.4 GB/rank) ✅ 2026-05-06
  - R6.3 8-rank 8k sync (α=0.625, 13.4 GB/rank) ✅ 2026-05-06
  - R6.3 8-rank 8k async (max_new=16) ✅; max_new ≥ 32 inconclusive ⚠
  - R6.4 8-rank 64k OOM at 35.7 GB / 40 GB ❌; ctx=16k fits ✅
  - **Option B fix landed** 2026-05-06 (commit eb9297a): don't RoPE-scale
    the draft. Cuts per-rank draft KV at ctx=64k from ~12 GB → ~770 MB.
    Predicted new per-rank usage at 64k×8 NF4: ~25 GB → fits 40 GB SXM2.
    Pod-side empirical verification pending (Issue #2 in M3 plan).
- **Mentor's M3 asks** (revised):
  - ⚠ Blockwise FA + ring attention — REWRITTEN. Original was structurally
    broken; new implementation in
    [src/models/ring_attention_kernel.py](src/models/ring_attention_kernel.py)
    + [src/models/ring_llama_attention.py](src/models/ring_llama_attention.py)
    is empirically validated through R6.3 sync at 8 ranks and ctx=16k.
    64k empirical confirmation pending Option B verification on pod.
  - ⏳ Memory validated to 512k — old [results/baselines/flash_memory_validation.csv](results/baselines/flash_memory_validation.csv)
    was for the standalone `RingAttentionFlash` kernel, not the production
    target+ring path. Needs re-validation with the new architecture.
  - ⏳ Full RASD at long-context (1M) — deferred to M4. **Tensor
    parallelism (weight sharding) needed for this.** See M3 plan Issue #4.
  - ❓ LongBench / L-Eval task-accuracy eval — still pending mentor input.
  - ⏳ Implementation-details email — pending.
- **M3 context length**: validated to 16k×W=8 in R6 (2026-05-06).
  64k×W=8 fits Option B prediction; pod-side confirmation pending.
- **Spend so far**: ~$224 cumulative ($200 RunPod M3 + $1.50 Lambda R6.1 +
  $22 Lambda R6.2-R6.4). M4 budget cap revised below.

## Status snapshot (2026-04-16) — preserved for history

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
0.2 ⏳ **(d)** `requirements-lock.txt` — run [scripts/capture_pod_env.sh](scripts/capture_pod_env.sh) on next pod before anything else; commit the output. **Now bundled into Phase C P3.1** — kept here as the canonical TODO marker for reproducibility.
0.3 ⏳ **(e)** Pin HF model revisions in [configs/ablations.yml](configs/ablations.yml). Sheared-LLaMA-1.3B + TinyLlama_v1.1 done; Llama-2 (gated) hashes still missing — run [scripts/pin_hf_revisions.py](scripts/pin_hf_revisions.py) on pod with HF_TOKEN. **Bundled into Phase C P3.1.**
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

### Compute track — priority-ordered (re-revised 2026-05-06)

**Top priority is now closing M3** — finishing R6.5 (49-row re-ablation)
on the new architecture. Without that, no published numbers from M3 are
defensible. Then M4 proper proceeds with PPL + 1M context + TP.

#### P0 — Close M3 first (NEW, ahead of everything else)

C0a. Resolve M3 plan Issue #1 (A4 async-ring inconclusive). Run async at
`max_new ∈ {16, 32, 64} × seed ∈ {42, 123}` on 8-GPU pod with full stderr
captured per run (no grep filtering). Script:
[scripts/r6_verify_runner.sh](scripts/r6_verify_runner.sh). ~6 runs ×
3-5 min ≈ 30 min pod time = ~$8.

C0b. Resolve M3 plan Issue #2 (Option B verification). Re-run R6.4 at
ctx=64k×W=8 NF4 sync; confirm per-rank usage drops from 35.7 GB to ~25 GB
matching the prediction. Bundled into the same runner script.

C0c. **R6.5 — Full 49-row re-ablation at 64k×W=8** with new architecture.
Gated on C0a + C0b passing. Wandb project `rasd-m3-reablation-64k`.
~3-4 hr pod time = ~$50–80 at $15.92/hr.

#### P1 — RoPE scaling (BLOCKER for 1M)

C1. ~~RoPE scaling code path behind `--rope-scaling` flag~~
**C1 partially landed in M3 redesign:** `_build_hf_config` in
[src/models/rasd_inference.py](src/models/rasd_inference.py) supports
linear RoPE scaling for the target. Validated end-to-end at ctx=8192
in R6.1. **Open work:** scaling to 64k+ with verified PPL (no NaN/inf).
The R6.4 OOM blocked the 64k empirical check — Option B (2026-05-06)
unblocks it.

C2. ~~Smoke-test RoPE numerically on single GPU at 64k~~ — fold into
C0b above (R6.4 verification with Option B). 64k single-GPU was ruled
out by R6.4 memory analysis; needs ring sharding which we now have.

C2b. **YaRN or NTK-aware scaling for 1M** — current Option B uses linear
RoPE scaling on target only. Linear interpolation is known to degrade
quality at 16x+ scaling factor. For 1M (factor=256 over Llama-2's native
4k), YaRN is widely recommended. Add when starting the 1M push.

#### P2 — Perplexity + throughput (core evidence)
C3. PG-19 preprocessing verification (tokenization, seq-length distribution, produces 1M-token chunks)
C4. Perplexity evaluator `src/analysis/perplexity.py` — tested on `sshleifer/tiny-random-llama` locally before pod
C5. Wire PPL logging into `run_experiment.py` alongside existing throughput metrics (sidecar, additive column)

#### P3 — Checkpoint/resume (pod-$ protection)
C6. Generation checkpoint/resume — write state every N tokens, resume on failure. At 1M context a single run is 20+ min; one crash = one pod-hour lost without this.

#### P4 — NF4 KV-cache quantization (NEW 2026-05-06, primary 1M lever)

**Replaces TP (C9) as the critical-path lever for 1M memory budget.**
Per-rank memory analysis showed K/V cache (post-ring sharding) is ~80%
of the budget at 1M; weights are <5%. NF4 KV cuts the KV term by ~4x
(~51 GB/rank saved at 1M); TP cuts the weight term by ~3.5 GB/rank.
NF4 KV is ~14x bigger savings AND lower engineering risk.

C11. **NF4 KV-cache quantization for target model.** Quantize target
K/V cache entries to 4-bit NF4 format (bitsandbytes-compatible) at write
time; dequantize at read time inside the ring kernel. Draft KV stays
bf16 (it's already small under Option B).

  Implementation outline:
  - Dual-cache integration: extend the existing sharded-prefill +
    replicated-tail layout in [src/models/ring_attention_kernel.py](src/models/ring_attention_kernel.py)
    to store both halves as NF4 with per-block scale factors.
  - Quantize at KV-append time (after each verify round). Per-block
    granularity (e.g. 64 positions/block) for scale factor amortization.
  - Dequantize-into-SRAM inside the ring step's `_attn_step` before the
    FA-2 call. FA-2 still operates on bf16 tiles; quantization is a
    storage/transport optimization, not a kernel-level change.
  - For the cross-rank rotation, send NF4-packed K/V (4x less wire
    traffic). Each rank dequantizes the received block before its
    `_attn_step`. Net P2P bandwidth savings: ~4x.

  Reference: bitsandbytes 4-bit NF4 kernels (already a project dependency).
  KIVI (Liu et al. 2024), KVQuant (Hooper et al. 2024) for prior art on
  per-channel/per-block quantization at long context.

  Engineering: ~1-2 weeks. Quality validation gate before committing:
  - Validation smoke (~30 min pod time, ~$10): ctx ∈ {8k, 64k} ×
    kv_dtype ∈ {bf16, NF4} × seed ∈ {42, 123, 456}. Confirms α drop ≤
    0.03 absolute.
  - PG19-long PPL benchmark (~20 min, ~$5): bf16 vs NF4 KV at ctx=64k×W=8.
    Targets <2% PPL increase per published KIVI/KVQuant results.
  - Throughput micro-benchmark (~15 min, ~$5): tps at ctx ∈ {8k, 32k, 64k}.
    Confirms NF4 net-positive at long context (memory bandwidth dominates
    dequant overhead per FA-2 access pattern).

  **Escape hatch — NF8 KV** if NF4 PPL degradation is unacceptable
  (>3% on PG19): switch to 8-bit symmetric quant. 2x KV savings instead
  of 4x — at 1M×W=8 that's 34 GB/rank instead of 17 GB, still fits 40 GB
  SXM2 with headroom (~38 GB total per-rank). Implementation reuses the
  same NF4 plumbing with int8 dtype.

  **Paper framing:** appendix subsection ("Appendix B.3: NF4 KV-cache
  validation"), not a 7th ablation axis. Three small tables (α, PPL,
  tps) defending the choice. Avoids 2x'ing the headline 49-row grid.

#### P5 — Tensor parallelism (DEMOTED to future-work / 70B+ regime)

C9. ~~**Megatron-style tensor parallelism for weight sharding.**~~
**Demoted 2026-05-06.** Per-rank memory analysis showed TP saves
~3.5 GB/rank at 1M while NF4 KV saves ~51 GB/rank. TP also competes
with ring for the fixed 8-GPU budget (TP=2 + SP=4 halves your ring
degree vs SP=8 + no TP), which actively *hurts* the 1M context-per-memory
budget on a fixed GPU count.

  **TP becomes critical at target ≥ 30B** where weights become the
  dominant memory cost (Llama-2-70B NF4 = ~35 GB/rank replicated, OOM).
  For the **7B target** evaluated in this paper, NF4 KV alone is the
  right lever. Paper framing: "Tensor parallelism is orthogonal and
  would benefit larger model targets (70B+) where weight memory
  dominates KV memory. Out of scope for the 7B target studied here."

  Future work item for follow-up paper (M5+): RASD with Llama-2-70B
  target + TP + ring + spec + NF4 KV at 1M. Different paper, different
  scope.

C10. **Install flash-attn explicitly** — currently we get FA-2 implicitly
because PyTorch's `F.scaled_dot_product_attention` dispatches to the FA-2
backend on Ampere+ GPUs with bf16/fp16 inputs. The explicit
`pip install --no-build-isolation flash-attn` is typically 10–20% faster
than SDPA's FA-2 backend (less dispatch overhead, more aggressive kernel
fusion). Required for the strongest 1M tps story. Note: flash-attn build
strips torch from the isolated env, so use `--no-build-isolation` per
the [reference_lambda_setup.md](.claude/projects/-Users-amankesarwani-PycharmProjects-RASD/memory/reference_lambda_setup.md)
gotcha. The kernel already has the dispatch logic at
[src/models/ring_attention_kernel.py:_attn_step](src/models/ring_attention_kernel.py)
— installing flash-attn flips `_FLASH_AVAILABLE = True`, no other code change needed.

### Long-context memory equation (added 2026-05-06)

Per-rank memory at context N, world_size W, with each ingredient
contributing or shrinking specific terms. Target weights (bf16) and
LLaMA-2-7B (32 layers × 32 heads × 128 head_dim) used for arithmetic.

| Ingredient | Without it | With it | Why |
|---|---|---|---|
| **FlashAttention** (FA-2) | O(N²) for the score matrix per layer per rank — at 1M, **~32 GB/layer = ~1 TB/rank** total | O(N) blocked tiling — **a few hundred MB/rank** total | tiles Q,K,V into SRAM; never materialises S = Q@Kᵀ in HBM |
| **Ring (sequence-parallel)** | full N positions of K/V on every rank — at 1M, **~540 GB/rank** | 1/W of K/V — at W=8, 1M, ~**68 GB/rank** | each rank holds N/W positions; rotation moves them through the ring during the forward |
| **Option B (don't RoPE-scale draft)** | draft KV scales to target's N — at 1M, **~190 GB/rank** | draft capped at native 4k — **~770 MB/rank** | draft only ever sees `draft_max_len` recent tokens; speculative decoding tolerates this asymmetry |
| **Tensor-parallel weights (C9)** | weights replicated W× — **4 GB/rank NF4** or 13 GB bf16 | weights split W ways — at W=8, **0.5 GB/rank NF4** or 1.6 GB bf16 | column/row-parallel linears with all-reduce after o_proj and down_proj |

**Net per-rank budget at 1M, W=8, NF4 weights, FA-2 + ring + Option B + NF4 KV (no TP):**

| Component | GB |
|---|---|
| Target weights (NF4, replicated) | 4.0 |
| Target K/V cache (ring-sharded, NF4) | ~17 |
| Draft weights (replicated, NF4) | 0.7 |
| Draft KV (Option B, bf16, 4k cap) | 0.8 |
| Activations + FA tile workspace | ~3 |
| Allocator fragmentation | ~5 |
| **Total** | **~30** — fits 40 GB SXM2 with headroom |

**Crucial insight (revised 2026-05-06):** at 1M, K/V cache is ~80% of the
post-ring budget. The four scale-out levers and their savings at 1M, W=8:

| Lever | Saves per rank | Engineering cost |
|---|---|---|
| FA-2 (attention scores) | ~1 TB | already shipped |
| Ring W=8 (K/V sharding) | ~470 GB | already shipped (M3) |
| Option B (draft KV cap) | ~190 GB | already shipped (M3) |
| **NF4 KV cache (target K/V)** | **~51 GB** | **~1-2 weeks (M4 critical path)** |
| TP W=8 (weight sharding) | ~3.5 GB | ~2-6 weeks (DEMOTED — see C9) |

**TP is ~14x smaller savings than NF4 KV with similar-or-higher engineering
cost.** TP also competes with ring for fixed 8-GPU budget (TP=2 + SP=4
halves ring degree vs SP=8). For the **7B target evaluated here, NF4 KV
alone closes the 1M memory budget on 40 GB SXM2 hardware** — TP is
unnecessary. TP becomes critical only at 30B+ targets where weight
memory dominates.

**Hardware tier:** 40 GB SXM2 sufficient at 1M with NF4 KV. 80 GB SXM4
provides additional headroom but is not strictly required for the M4
target.

#### P6 — Profiler / time-breakdown (REQUIRED — mentor Fig 3)

C7. **`torch.profiler` context-manager wrapper with NVTX ranges at round
boundaries** — promoted from conditional → required 2026-05-06 to satisfy
mentor's **Figure 3 (stacked bar: compute / comm / idle for Ring vs RASD)**.
Also serves as insurance for the negative-result pivot path called out in
the mentor's risks section: if final tps gains are marginal, the time-
breakdown becomes the headline story. Build locally; tested on CPU
gloo at small ctx; run on pod during Phase 3.5 final matrix as a sidecar
on a subset of cells (one seed × all 4 contexts × {RASD, Ring}).

#### P7 — Tick-gate regression test (cheap, high-value, mostly OBSOLETE)

C8. ~~Gloo-backend regression test for tick-gate ordering~~ —
**partially obsoleted by M3 redesign**: the entire `AsyncKVRingPrefetcher`
class (which owned the tick gate) was deleted in commit 09f7d98 when ring
moved into the attention forward. The tick-gate ordering bug from
commit dc14915 cannot recur because the code path no longer exists.
What DOES still need a regression test is the new ring kernel's
batched_isend_irecv ordering at world_size > 4 — already partially
covered by `tests/test_ring_attention.py::TestKnobInvariance` at
W∈{2,4} via gloo. Could extend to W=8 (gloo CPU) for cheap.

#### P8 — Mentor-required sidecars (RESTORED 2026-05-06 from deprioritized)

C12. **TTFT split timer** — instrument `_prefill` start/end and emit the
delta to wandb + CSV as `ttft_ms`. Mentor lists TTFT as a core metric.
Implementation: ~30 lines, additive column, no behavior change. Tested
locally on CPU gloo.

C13. **Per-position acceptance sidecar `.jsonl`** — log `{round_idx,
position, accepted: bool, draft_token, target_token}` per spec round
to a per-run sidecar file. Required for Figure 4 (α vs token position)
to be rigorous rather than approximated from round-level logs.
Implementation: ~50 lines in `rasd_inference.py` behind `--log-per-token`
flag (default off so M3 replay stays byte-identical).

### Phase 3 — Pod required (Lambda 8x A100 SXM4 40 GB, gpu_8x_a100)

**Status (2026-05-10):** four blocker bugs surfaced by external code
review on 2026-05-10 have been fixed (commit `b993f67`); a fifth
high-risk finding remains as a **pending scope decision** (see
"Phase C blockers" subsection below). Master script
`scripts/phase_c_pod_session.sh` runs every P3.x stage in sequence,
aborts on first failure, and writes per-stage marker files so re-runs
skip already-completed stages. See [PHASE_C_RUNBOOK.md](PHASE_C_RUNBOOK.md)
for the bundled-session checklist.

#### Phase C blockers (audit log)

External code review on 2026-05-10 surfaced 5 high-risk findings.
Current state:

| # | Finding | Status | Fix commit |
|---|---|---|---|
| 1 | `kv_quant=True` is round-trip only, not storage NF4 → 1M premise breaks | ⚠ **scope decision pending** | not yet |
| 2 | Double `torchrun` on orchestrator stages (outer + inner) | ✅ fixed | `b993f67` |
| 3 | `build_run_configs` `A*` prefix filter drops M4 YAMLs | ✅ fixed | `b993f67` |
| 4 | Baseline stage uses `bash` + wrong flag name | ✅ fixed | `b993f67` |
| 5 | `rng_state` field never populated → resume divergence under temp>0 | ✅ fixed | `b993f67` |

**Finding #1 scope question (pending):** the in-kernel round-trip
quantize→dequantize doesn't reduce cache memory — `kv_quant=True`
is currently a **lossy bf16 path** that only validates codec
correctness. True memory savings require the cache itself to hold
NF4 bytes (subclass DynamicCache or external NF4 store). Per the
M4_PLAN memory equation, this matters at 1M:

| ctx | bf16 KV / rank (W=8) | NF4 KV / rank | fits 40 GB? |
|---|---|---|---|
| 64k | ~13 GB | ~3.5 GB | ✅ either way (R6.5 confirmed) |
| 128k | ~37 GB | ~10 GB | ❌ bf16 / ✅ NF4 |
| 512k | (OOM bf16) | ~25 GB | ❌ bf16 / ✅ NF4 (tight) |
| 1M | ~68 GB | ~17 GB | ❌ bf16 / ✅ NF4 |

Three remediation paths:

1. **(a) Drop ctx > 64k from Phase C.** Validate at 64k only;
   admit 1M needs the real C11 storage integration first; defer
   the headline 1M number to a follow-up. Cheapest, smallest scope
   reduction. Phase D figures degrade to "1M is future work".
2. **(b) Implement true NF4 storage.** Subclass DynamicCache or
   write an external NF4-packed store. ~1-2 weeks engineering;
   needs careful unit tests + multi-rank gloo tests to land safely.
   Then run Phase C at all four contexts. Hits the original 1M
   target.
3. **(c) Move to 80 GB SXM4 hardware** (`gpu_8x_a100_80gb_sxm4`,
   $22.32/hr). bf16 1M KV at 68 GB / rank fits with headroom on
   80 GB. No code change needed. But Lambda's 80 GB SKU has been
   at zero capacity throughout the May 2026 polling window —
   capacity availability is the blocker.

Recommendation written by external code reviewer (synthesized):
"For the M4 1M target, real NF4 storage (b) is the proper path;
(a) is a defensible scope reduction if the timeline doesn't permit
1-2 weeks of engineering."



**Hardware decision 2026-05-06**: stay on Lambda ($2000 credit) using
the 40 GB SXM2 tier ($15.92/hr). With NF4 KV (C11), per-rank memory at
1M fits comfortably (~30 GB / 40 GB). 80 GB SXM4 would give more
headroom but isn't strictly required. Capacity is volatile but
consistently shows up in europe-central-1 (no rasd-fs filesystem there
— scp results back each session).

P3.0. **GPU health-check preflight** (mentor risk mitigation) — before
any long-running 1M cell, run `nvidia-smi -q | grep -i "ecc\|xid\|throttl"`
on all 8 GPUs; confirm 0 MiB used at idle; run a 1-min NCCL all-reduce
loopback to catch flapping interconnect. Aborts session if any rank
fails. Script: `scripts/gpu_health_check.sh`.
P3.1. **Reproducibility lockdown** — first thing on the pod, before any
experiment work, close the two open reproducibility gaps:
  - **(d)** `bash scripts/capture_pod_env.sh > requirements-lock.txt` —
    captures exact transitive pip+conda versions on the pod that produces
    M4 results. Commit immediately. Without this, third-party
    reproduction relies on `environment_gpu.yml` resolving to the same
    transitive versions, which is not guaranteed (HuggingFace/PyPI
    packages move).
  - **(e)** `python scripts/pin_hf_revisions.py` (with `HF_TOKEN` set
    inline) — captures Llama-2-7b-hf and Llama-2-13b-hf model commit
    hashes into `configs/ablations.yml`. The non-gated drafts
    (Sheared-LLaMA-1.3B, TinyLlama_v1.1) are already pinned. Commit.
  - Then run `bash scripts/replay_m3_smoke.sh` (one seed per group) —
    asserts throughput within 15% of golden CSV. Catches semantic
    regressions before burning $80 on the full M4 grid.
P3.2. RoPE scaling validation: PPL at 32k/128k/512k/1M on 1 GPU (needs C1+C4)
P3.3. Smoke tests: single RASD run at 32k, 128k, 512k, 1M context (validate NF4 KV doesn't OOM at the long end)
P3.4. Baseline validation: Ring + Sliding end-to-end at 128k, 1M
P3.5. **Final 36-run matrix** — RASD+Ring+Sliding × {128k, 256k, 512k, 1M} × 3 seeds. Use `max_new_tokens=64` for memory + tps (sufficient for headline numbers); optionally subset of cells re-run at `max_new_tokens=256` for α stability. **Per-position sidecar (C13) enabled on all RASD cells**; TTFT (C12) on all cells.
P3.6. **Profiler sidecar pass** (C7 — REQUIRED for Fig 3): one seed × all
4 contexts × {RASD, Ring} = 8 cells with `torch.profiler` enabled,
capturing compute / comm / idle breakdown. Adds ~10% overhead, run
separately from P3.5 to keep headline tps numbers clean.

### Phase 4 (= Phase D) — Post-pod, local — **LAST PHASE**

This is the final M4 phase: assemble paper deliverables from real data.
Estimated effort: ~3-5 days, all local. **Blocked on Phase C producing
the matrix CSV + profiler sidecars + per-position .jsonl files.**

Figure list is mentor's 5 figures verbatim (alignment matrix at top of file).

F1. **Figure 1 — Throughput vs context length** (line plot, RASD vs Ring
    vs Sliding, contexts ∈ {128k, 256k, 512k, 1M}, 95% CI bands from
    bootstrap over 3 seeds). Source: Phase 3.5 CSV.
F2. **Figure 2 — Heatmap of throughput × draft_size × spec_steps**
    summarizing the M3 ablation. Source: `ablations_r65.csv` (existing).
    *Note: replaces the earlier "ablation bars" plan; rewrite required.*
F3. **Figure 3 — Stacked bar: time breakdown (compute / comm / idle)**
    Ring vs RASD. Source: C7 profiler sidecar from Phase 3.5 subset.
F4. **Figure 4 — α vs token position** (line, with smoothing). Source:
    C13 per-position sidecar `.jsonl` from Phase 3.5 final-matrix runs.
F5. **Figure 5 — Qualitative text comparison table** (3-5 sample
    prompts × {target-only, RASD} generated continuations, side by side).
    Source: capture sample generations during Phase 3.5; manual curation.
F6. `results/final/final_results.json` (aggregate metrics, per-seed,
    includes tps / latency_ms_per_token / α / PPL / TTFT).
F7. LaTeX tables via `pandas.to_latex()` — main results + ablation
    summary + per-context breakdown.
F8. **`analysis/error_analysis.md`** — examine sequences with abnormally
    low α from per-position sidecar. Hypothesize failure modes (topic
    shifts, complex syntax, etc.). Mentor deliverable verbatim.
F9. Manuscript sections (methods, results, discussion).

### Conditional
X1. LongBench / L-Eval task-accuracy eval — **pending mentor approval**. Scope: pick 2-3 LongBench tasks, run target-only vs RASD at 64k, compare EM/F1. Adds ~1 pod-day; only if mentor says PG-19 perplexity is insufficient.

### Admin
Z1. Send mentor follow-up email: FA+Ring implementation details + LongBench scope question + cost note ($200 burned on 8k M3; asking about cheaper alternatives for M4)

## Deliverables (mentor-aligned 2026-05-06)

- `results/final/final_results.json` — aggregated metrics, per-seed per-config
  (tps, latency_ms_per_token, α, PPL, TTFT)
- `figures/` — **5 publication-quality PDFs** matching mentor's Fig 1-5:
  - `fig1_throughput_vs_context.pdf` — line plot, 95% CI bands
  - `fig2_ablation_heatmap.pdf` — heatmap throughput × draft_size × spec_steps
  - `fig3_time_breakdown.pdf` — stacked bar compute/comm/idle, Ring vs RASD
  - `fig4_acceptance_vs_position.pdf` — line plot from per-position sidecar
  - `fig5_qualitative_examples.pdf` — text comparison table
- `tables/` — LaTeX via `pandas.to_latex()`: ablation summary + main results
  + per-context breakdown
- `analysis/error_analysis.md` — low-α sequence analysis (mentor deliverable)
- `manuscript/` — methods, results, discussion sections
- Mentor emails: implementation details, LongBench scope, cost ask

## Risks / anticipated fixes (updated 2026-05-06)

- **Marginal performance gain at 1M (mentor risk)** — if final tps gain
  vs Ring baseline is within CI overlap, paper pivots to **bottleneck
  analysis as the primary contribution**: Figure 3 (compute/comm/idle
  breakdown via C7 profiler) becomes the headline figure, supported by
  per-position acceptance fall-off (Figure 4 from C13 sidecar). This is
  why C7 was promoted from conditional → required and C13 was restored
  from deprioritized. Both must land before Phase 3.5 to keep this
  pivot path open.
- **Hardware instability during long runs (mentor risk)** — addressed
  by: (a) C6 checkpoint/resume, (b) P3.0 GPU health-check preflight,
  (c) per-rank memory ceiling already validated at ~25 GB / 40 GB in
  M3 R6.5, leaving 15 GB headroom for fragmentation under long runs.

- **A4 async-ring inconclusive at W=8 max_new ≥ 32** (M3 plan Issue #1).
  Worst case: R6.5 runs sync-only (A4 levels 1, 2 dropped from paper). Best
  case: it was transient; characterization in C0a clarifies.
- **64k×W=8 fits 40 GB hardware with Option B but only just** (~25 GB
  used, ~15 GB headroom). Any additional growth (longer ablation
  configs, FA-2 install, fragmentation) could push over. C0b
  empirically confirms; if tight, drop ablation rows that grow memory
  the most (e.g. `kv_block_size=2048` builds bigger ring buffers).
- **RoPE scaling at 1M** may produce NaN/inf with linear scaling
  (factor=256 is far past linear's regime of validity). C2b adds YaRN
  before the 1M push.
- **Memory at 1M × 8 ranks** — K/V cache is ~80% of post-ring budget.
  **NF4 KV-cache quantization (C11) is the critical-path lever**, cuts
  K/V from ~68 GB → ~17 GB per rank. With NF4 KV, 1M fits 40 GB SXM2 at
  ~30 GB/rank. TP (C9) demoted to future work (saves only ~3.5 GB/rank
  for 7B target; relevant only at 30B+).
- **NF4 KV quality regression** — published KIVI/KVQuant results show
  <2% PPL degradation, but α impact for spec decoding has not been
  characterized. C11 includes a validation smoke before committing
  (8k+64k × bf16/NF4). NF8 escape hatch documented if PPL exceeds 3%.
- **Lambda multi-GPU capacity is volatile.** R6 work used europe-central-1
  (no rasd-fs filesystem) because that's where capacity returned. For R6.5
  prefer us-west-2 if available; otherwise live with cold model loads
  each session and remember to scp results before terminating.
- **Long-run orphaned VRAM** — Lambda's pod-per-instance design means
  each launch is a clean GPU. No equivalent of RunPod's orphaned-memory
  problem.
- **Subprocess timeout** — 120s worked for 8k; 1M context may need
  600-1800s. Patch per-phase. Already extended to 600s in
  [scripts/r6_verify_runner.sh](scripts/r6_verify_runner.sh).

## Cost discipline (revised 2026-05-06 with 1M target + Lambda commitment)

User confirmed 2026-05-06: $2000 Lambda credit available; stay on Lambda
even with capacity volatility. 1M context is in scope. NF4 KV (C11)
chosen over TP (C9) — saves ~14x more memory for ~half the eng cost.

| Phase | Detail | Estimated $ | Status |
|---|---|---|---|
| Phase A (local analysis) | A1-A5 + experiments.md + reproducibility | $0 | ✅ done |
| **R6 verification** | C0a (Issue #1) + C0b (Issue #2 / Option B) | $11 | ✅ done 2026-05-06 |
| **R6.5 49-row re-ablation** at ctx=64k×W=8 | ~5 hr live now | ~$80 | ✅ done 2026-05-06 |
| Phase B (local engineering) | C3 + C5 + C2b + C6 + C11 codec/cache | $0 | ✅ done 2026-05-06 |
| **Phase C — bundled pod session** | health check + repro lockdown + C11/C2b/C6 validation + smokes + 36-run matrix + profiler | ~$160 | 📋 prepped (`PHASE_C_RUNBOOK.md`) |
| Phase D (local) | Fig 1/3/4/5 + tables + final_results.json + manuscript | $0 | ⏳ blocked on Phase C |
| Conditional LongBench | pending mentor approval | +$50-80 | pending |
| **M4 total (with 1M, Lambda 40 GB tier)** | | **~$160 remaining** | |

**Cumulative project spend (2026-05-06 post Phase B):**
- M3 (RunPod 8k baseline, INVALIDATED): $200
- R6.1 (Lambda 1x A100): $1.50
- R6.2-R6.4 (Lambda 8x A100, partial): $22
- R6.5 + verification (Lambda 8x A100): ~$95
- Phase A + B (entirely local): $0
- **Subtotal**: ~$320

**Phase C estimate**: ~$160 (bundled session ~10 hr at $15.92/hr).
**Phase D**: $0 (local).
**Total project budget on 1M-context target (Lambda 40 GB tier):** ~$320
spent + ~$160 Phase C + $0 Phase D = **~$480 total, well under the
$2000 Lambda credit**.

**Engineering-time impact of NF4 KV vs TP:** TP would have been 2-6
weeks critical-path; NF4 KV is 1-2 weeks. Saves ~3 weeks → reduces
deadline pressure significantly.

**Hardware tier rationale:** Lambda 8x A100 SXM4 80 GB is $22.32/hr
when available (rare). Lambda 8x A100 SXM2 40 GB is $15.92/hr (more
reliably available). 30% cheaper. With NF4 KV, 1M fits 40 GB, so the
80 GB tier provides headroom but no functional advantage for our 7B
target. Net: 40 GB SXM2 is the right tier for M4.

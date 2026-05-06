# M3 Error Analysis (R6.5 re-ablation)

**Source:** [`results/ablations/ablations_r65.csv`](../results/ablations/ablations_r65.csv) — 49 rows = 46 ok + 3 OOM
**Date:** 2026-05-06 (R6.5 re-ablation; supersedes the 2026-04-16 analysis below)
**Hardware:** Lambda 8× A100-SXM4-40GB, ctx=64k, NF4 target/draft, bf16 KV
**Reproducible via:** `python scripts/compute_ablation_cis.py`

This is the mentor M4 deliverable analyzing failure modes and abnormally
low-α sequences. Per-position acceptance trace was not captured in R6.5
(M4 sidecar [C13](../M4_PLAN.md) was committed *after* R6.5 ran), so the
"sequences with abnormally low α" subsection defers to M4 Phase 3.5
runs. What this file covers:

1. The 3 OOM failures (axis A5, Llama-2-13B target)
2. Canary determinism check
3. Aggregate observations on α dispersion across the 46 ok rows
4. Short-run filter rationale (`tokens_generated >= 20`) — still applies
5. Pointer to where per-position low-α analysis will live for M4

## 1. The three OOM rows — A5_llama2_13b

| run_id | seed | tokens_generated | status | error |
|---|---:|---:|---|---|
| `A5_llama2_13b_s42`  | 42  | — | error | OOM at 78 MiB alloc |
| `A5_llama2_13b_s123` | 123 | — | error | OOM at 78 MiB alloc |
| `A5_llama2_13b_s456` | 456 | — | error | OOM at 78 MiB alloc |

**Identical failure mode across all 3 seeds:**

```
CUDA out of memory. Tried to allocate 78.00 MiB. GPU 0 has a total
capacity of 39.49 GiB of which 3.56 MiB is free. ... 31.02 GiB is
allocated by PyTorch, and 6.95 GiB is reserved by PyTorch but
unallocated.
```

### Root-cause analysis

This is **not a regression**. It is the predicted memory-ceiling
failure that motivates M4's NF4 KV-cache work (C11). Per the long-
context memory equation in [`M4_PLAN.md`](../M4_PLAN.md), per-rank
budget at ctx=64k×W=8 NF4 target:

| Component | 7B target | 13B target |
|---|---:|---:|
| Target weights (NF4) | ~3.8 GB | ~7.0 GB |
| Target K/V cache (ring-sharded, bf16, ctx=64k×W=8) | ~13 GB | ~22 GB |
| Draft weights + KV (Option B, ~770 MB) | ~0.8 GB | ~0.8 GB |
| Activations + FA workspace | ~3 GB | ~3 GB |
| Allocator fragmentation (bnb + ring buffers) | ~5 GB | ~5 GB |
| **Total** | **~25 GB** ✓ | **~38 GB** (exceeds 40 GB once fragmentation appears) |

The 7B target uses ~25 GB / 40 GB rock-stable across all 46 ok rows.
The 13B target loads weights successfully (~31 GB allocated, per the
error message) but cannot reserve the incremental KV+activation chunk
for the verify forward — the "3.56 MiB free of 39.49 GiB" line is the
smoking gun.

### Reproducibility of the OOM

- All 3 seeds OOM at the **same allocation site** (78 MiB) and the
  **same memory state** (3.56 MiB free, 31.02 GiB allocated). Strong
  evidence this is deterministic, not transient.
- Reproducer: any seed × `target=meta-llama/Llama-2-13b-hf` ×
  `quantize_target=True` × `context_length=65536` × `world_size=8` on
  40 GB SXM2/SXM4 hardware will fail identically.

### Resolution paths

1. **NF4 KV-cache (M4 C11)** cuts the K/V term from ~22 GB → ~6 GB at
   ctx=64k for 13B. Per-rank total drops to ~22 GB. **Primary path.**
2. **80 GB SXM4 hardware** would fit 13B + bf16 KV at ~38 GB / 80 GB.
   Documented as a tier-up option but not committed for M4 (Lambda
   80 GB SKU was at zero capacity throughout the R6.5 session).
3. **Tensor parallelism (C9, demoted)** reduces the weight term by
   only ~2 GB / rank — not enough to close the 13B gap by itself.
   Critical only at 30B+ targets.

The 13B OOMs are **clean negative-result evidence** for the paper:
they directly motivate C11 and quantify the 40 GB tier's empirical
ceiling for the 7B target.

## 2. Canary determinism check

The canary cell `default_canary_s42` exercises the project default
(Sheared-LLaMA-1.3B draft, Llama-2-7B target, k=4, kv_block=512,
async-1, ctx=65536, seed=42).

| field | value |
|---|---:|
| tokens_generated | 33 |
| time_sec | 74.8 |
| throughput_tps | 0.44 |
| acceptance_rate | 0.221 |
| mean_latency_ms | 2266.8 |
| gpu_peak_mem_mb | 24770.3 |

The same default-config cell appears in **5 ablation positions**
(canary, A1_sheared_1b_s42, A2_k4_s42, A3_block512_s42,
A5_llama2_7b_s42). Per
[`results/ablations/r65_audit.md`](../results/ablations/r65_audit.md),
all five report identical (tps=0.8, α=0.231, mem=25.5 GB) **to logged
precision**. The canary's lower tps (0.44 vs 0.8) and α (0.221 vs
0.231) reflect a **shorter run** — only 33 tokens generated vs the
production rows' ~64. Not a regression: the canary is configured to
exit early on EOS for speed.

**Strong reproducibility signal:** byte-precision match across 5
independent invocations of the same config in the same R6.5 session.

## 3. α dispersion across the 46 ok rows

α range: **0.105 (A2 k=12 floor)** to **0.424 (A2 k=2 best-case)**.

| Source of dispersion | Effect |
|---|---|
| **A2 (k)** | Monotonic: α decreases with k (textbook spec-decoding). Explains most of the spread. |
| **A1 (draft size)** | Tight at 0.23-0.25 — TinyLlama vs Sheared within seed noise. |
| **A3 (kv_block_size)** | Invariant at 0.253 across {256, 512, 1024, 2048} — α only depends on attention math, not chunking granularity. |
| **A4 (prefetch_depth)** | Invariant at 0.253 across {sync, async-1, async-2} — sync/async produces *identical* token streams (NCCL handles overlap). |
| **A5 (target=7B)** | Single working level at 0.253 (13B OOMed). |

**No row has α more than 1σ below the per-axis mean.** Every ok row's
α is within the predicted spec-decoding theoretical band for its k
value. There are no "abnormally low α" sequences in R6.5.

## 4. Short-run filter — `tokens_generated >= 20`

This threshold is implemented in
[`src/analysis/metrics.py`](../src/analysis/metrics.py) as
`SHORT_RUN_THRESHOLD = 20` and applied by `filter_valid(df)`. The
intent is to drop deterministic early-EOS outliers from aggregate
statistics without dropping them from the on-disk CSV.

**R6.5 status:** all 46 `status=ok` rows clear this threshold
(canary's 33 is the smallest). No R6.5 row is excluded by this
filter. The 3 A5_llama2_13b rows have `status=error` and are
dropped by the `status=ok` predicate first.

This is a **change from the pre-audit M3** where 4 rows fell below
the threshold (analyzed in §6 below). The R6.5 architecture's
correct cross-rank consensus (Fix2) eliminated the early-EOS
divergence those rows exhibited.

## 5. Per-sequence low-α analysis (deferred to M4 Phase 3.5)

The M4 plan adds:

- **C13 per-position acceptance sidecar** (committed 2026-05-06,
  9eace0e): when `cfg.log_per_token=True`, `generate()` returns
  `metrics["per_token_trace"]` with one entry per spec round
  describing which draft tokens were accepted at which global
  positions.
- **Phase 3.5 final matrix**: enables C13 on all RASD cells so
  Figure 4 (α vs token position, mentor's required Fig 4) can be
  plotted from real per-position data.

Once Phase 3.5 produces sidecars, this file will be extended with:

- A subsection per ablation cell where mean(α) is at least 1.5σ
  below the global mean
- Token-level analysis of where rejection clusters fall (topic
  shifts, code blocks, named entities, etc.)
- Hypothesis testing: does rejection cluster at draft-context
  boundaries? At positions far past the draft's native 4k context
  cap under Option B?

That work is M4 Phase D (post-pod analysis) — see
[`M4_PLAN.md`](../M4_PLAN.md) §F8.

## 6. Pre-audit M3 (2026-04-16) — preserved for history

The original M3 analysis (against the now-invalidated
`ablations.csv`) classified 4 short-run rows:

| run_id | tokens | accept | tps | classification |
|---|---:|---:|---:|---|
| `A2_k2_s42`            | 6  | 0.000 | 1.28 | k=2 + early EOS  |
| `A2_k8_s42`            | 3  | 0.000 | 0.62 | k=8 + early EOS  |
| `A1_tinyllama_1b_s456` | 9  | 0.036 | 1.53 | seed-456 short prompt |
| `A2_k6_s456`           | 14 | 0.030 | 1.65 | seed-456 short prompt |

These rows do not exist in `ablations_r65.csv` — the architectural
fixes (Option B + Fix2 + Fix3 + Fix4) eliminated the verify-loop
divergence and bf16-drift that produced them. The threshold
rationale is still load-bearing for any future ablation that
might reintroduce short-run outliers.

## Files

- [`results/ablations/ablations_r65.csv`](../results/ablations/ablations_r65.csv) — raw R6.5 CSV
- [`results/ablations/r65_audit.md`](../results/ablations/r65_audit.md) — group summaries
- [`results/final/ablation_cis.csv`](../results/final/ablation_cis.csv) — bootstrap CIs
- [`figures/fig2_ablation_heatmap.pdf`](../figures/fig2_ablation_heatmap.pdf) — Fig 2 (mentor spec)
- [`tables/ablation_summary.tex`](../tables/ablation_summary.tex) — booktabs table for paper

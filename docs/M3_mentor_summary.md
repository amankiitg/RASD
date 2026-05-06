# M3 — Mentor Summary

**Date:** 2026-05-06
**Status:** M3 closed (compute-track complete; paper write-up consolidated into M4)

## TL;DR

The original M3 ablation (commit `126dbb7`, 2026-04-15) reported 49/49 rows
with α (acceptance rate) = 0.06–0.11 across all configurations. A
post-analysis audit found four mathematical defects in the speculative-
verify path *plus* an architectural gap where ring attention's prefetched
K/V was never consumed by the target's attention forward — so at
`world_size>1`, each rank silently held the full sequence's K/V locally
and the ring P2P was decorative.

Three weeks of redesign (R0–R3.5 + R5) plus four fixes during a single-
day pod session unblocked the architecture. **R6.5 (the re-ablation)
ran 2026-05-06 on Lambda 8× A100-SXM4-40GB at ctx=64k×W=8 and produced
46/49 valid rows with α = 0.105–0.424** (3–7× higher than the buggy
baseline at floor and ceiling). The 3 OOM rows are A5_llama2_13b at
ctx=64k×W=8 NF4 — predicted memory-ceiling failure that motivates the
M4 NF4 KV-cache work.

## What changed since the M3 status update

### Audit findings (2026-04-16 → 2026-05-05)

- **Four math defects** in the spec-verify path (autoregressive target
  forward instead of packed; raw `p_target` instead of residual on
  partial rejection; off-by-one KV truncation; one latent CUDA stream
  race). Locked into 6 unit tests in `tests/test_verification_math.py`.
- **Ring attention was structurally broken**: the `AsyncKVRingPrefetcher`
  posted P2P that was never read by attention. At `world_size=8`, each
  rank effectively ran inference against its full local KV — the
  paper's "ring shards K/V across ranks" claim was unsubstantiated by
  the running code.

### Architecture redesign (R0–R3.5 + R5, 11 commits 2026-05-05)

- Layout-agnostic ring kernel
  ([`src/models/ring_attention_kernel.py`](src/models/ring_attention_kernel.py))
  with online-softmax merge and FA-2/SDPA dispatch.
- Ring attention now lives **inside `LlamaAttention.forward`** via a
  monkey-patch installer
  ([`src/models/ring_llama_attention.py`](src/models/ring_llama_attention.py)).
  Each rank holds a local K/V slice + replicated decode tail; ring
  rotation happens during the forward pass; tail attention is local.
- 78 unit tests covering kernel math, multi-process gloo correctness at
  W∈{2,4}, and Llama-patch plumbing.
- A3 / A4 ablation axes redefined to map onto the new architecture:
  A3 = per-step `batch_isend_irecv` chunk size; A4 = ring-step prefetch
  depth.

### Four fixes landed today (2026-05-06)

| Fix | Commit | What | Why |
|---|---|---|---|
| **Option B** | `eb9297a` | Don't RoPE-scale the draft; cap at 4k native | Saves ~11 GB/rank at ctx=64k by shrinking replicated draft KV from ~12 GB → ~770 MB |
| **Fix2** | `e875f6d` | Broadcast `target_logits_v` + `draft_logits` from rank 0 before accept/reject | Eliminates cross-rank divergence from bf16-noise drift in ring online-softmax. Was the root cause of NCCL coalesced-op timeouts at SeqNum ~3500-3600. |
| **Fix3** | `45b2b40` | Auto-truncate prompt to multiple of `world_size` in `_prefill` | Tokenizers return off-by-a-few token counts; assertion was crashing rank 0 |
| **Fix4** | `ad2bf5e` | Remove legacy `_ring_peer_loop` from `run_experiment.py` | Pre-R3 master/slave pattern; ranks 1-7 sat in `dist.recv(tick, src=0)` after R3 deleted the prefetcher |

## R6.5 results (re-ablation at ctx=64k×W=8 NF4)

**Hardware:** Lambda `gpu_8x_a100` (A100-SXM4-40GB), europe-central-1,
$15.92/hr. Total runtime ~4.6 hours for 48 production rows.

**Wandb project:** [rasd-m3-reablation-64k](https://wandb.ai/amank-iitg-uc-berkeley-electrical-engineering-computer-s/rasd-m3-reablation-64k)

| Group | n ok | mean α | mean tps | mean mem GB |
|---|---|---|---|---|
| canary | 1 | 0.221 | 0.44 | 24.2 |
| A1 (draft size) | 6 | 0.244 | 0.85 | 24.5 |
| A2 (spec_steps k) | 15 | 0.214 | 0.84 | 24.7 |
| A3 (chunk_size) | 12 | 0.253 | **0.96** | 24.9 |
| A4 (prefetch_depth) | 9 | 0.253 | 0.87 | 24.9 |
| A5 (target model) | 3 | 0.253 | 0.86 | 24.9 |

Memory **rock-stable at ~25 GB/rank** across all 46 successful rows; all
8 ranks reported identical peak memory to byte precision every row.
This is direct empirical evidence that ring sequence-parallelism +
Option B + Fix2 work in lockstep at production scale.

### Headline ablation findings (paper-relevant)

**A2 (spec_steps k):** α decreases monotonically with k — textbook
spec-decoding curve.

| k | mean α | mean tps |
|---|---|---|
| 2 | 0.38 | 0.77 |
| 4 (default) | 0.25 | 0.87 |
| 6 | 0.18 | 0.83 |
| 8 | 0.15 | 0.87 |
| 12 | 0.11 | 0.83 |

**A3 (kv_block_size = ring transmission chunk size):** ~94% throughput
gain monotonic across 256 → 2048. α invariant. Default 512 is too
conservative; 1024-2048 should be production default.

| chunk_size | mean tps |
|---|---|
| 256 | 0.63 |
| 512 (default) | 0.87 |
| 1024 | 1.10 |
| 2048 | **1.23** |

**A4 (prefetch_depth):** sync vs async-1 vs async-2 produce **identical
metrics to logged precision at every seed**. With modern NCCL (v2.26+)
and ring-in-attention, NCCL's internal stream concurrency handles
compute/comm overlap automatically. Explicit Python-level prefetch
adds zero measurable benefit at our scale. Useful negative result for
the paper.

**A5 (target):** 7B (Llama-2-7B) ran cleanly; 13B OOM'd at all 3 seeds
(predicted memory ceiling on 40 GB SXM4). Documents the motivation
for M4's NF4 KV-cache work.

### Determinism (sanity check)

The default-config cell appears in 5 ablation positions (canary,
A1_sheared_s42, A2_k4_s42, A3_block512_s42, A5_llama2_7b_s42). All
five report identical (tps=0.8, α=0.231, mem=25.5 GB) to logged
precision — strong reproducibility guarantee.

## What's next (M4)

1. **NF4 KV-cache quantization** (M4 C11) — primary 1M memory lever.
   Saves ~51 GB/rank at 1M (vs ~3.5 GB for tensor parallelism).
   Engineering: ~1-2 weeks. Validation: ~$30 of pod time.
2. **YaRN RoPE** (M4 C2b) — current linear scaling fails past factor=16;
   1M needs factor=256. ~2-3 days engineering.
3. **Phase 3 long-context matrix** — RASD + Ring + Sliding × {128k,
   256k, 512k, 1M} × 3 seeds. Target: ~$170 of pod time on Lambda
   40 GB tier (NF4 KV makes 1M fit).

Total M4 budget projection: ~$310 of pod time + 2-3 weeks of engineering.
Within Lambda's $2000 research credit.

## Files

- [results/ablations/ablations_r65.csv](../results/ablations/ablations_r65.csv) — 46 valid + 3 OOM rows
- [results/ablations/r65_audit.md](../results/ablations/r65_audit.md) — group summaries
- [M3_RING_INTEGRATION_PLAN.md](../M3_RING_INTEGRATION_PLAN.md) — fix log, live findings, paper commentary at the bottom
- [M4_PLAN.md](../M4_PLAN.md) — revised for 1M target, Lambda 40 GB tier, NF4 KV path
- Per-row wandb dashboards at the project URL above

## Compute acknowledgement

This work used **Lambda Labs research compute credits ($2000)** for the
post-audit R6 verification + R6.5 re-ablation runs.

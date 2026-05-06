# R6.2–R6.4 — Multi-rank Lambda smokes (results + open issues)

**Date:** 2026-05-06
**Hardware:** 1× `gpu_8x_a100` (40 GB SXM2) on Lambda Cloud, europe-central-1, $15.92/hr
**Filesystem:** none — ephemeral storage (rasd-fs lives in us-west-2; region mismatch)
**Cost (this session):** ~$22 estimated (~80 min)

## Verdict

| Phase | Status | α | Per-rank memory |
|---|---|---|---|
| **R6.2** 2-rank 8k smoke | ✅ PASS | 0.312 | 28.4 GB |
| **R6.3** 8-rank 8k sync | ✅ PASS | 0.625 (max_new=8) | 13.4 GB |
| **R6.3** 8-rank 8k async max_new=16 | ✅ PASS | 0.812 | 14.9 GB |
| **R6.3** 8-rank 8k async max_new=32 | ❌ Traceback (flaky) | — | — |
| **R6.4** 8-rank 64k sync | ❌ **OOM** at 35.7 GB / 40 GB | — | partial alloc |
| **R6.4b** 8-rank 16k sync | ✅ PASS | 0.875 | 20.9 GB |

**Architecture validated**: ring-attention + speculative decoding produces
consistent metrics across ranks at world_size ≤ 8 in sync mode. The dual-cache
design is working — sharded K/V scales linearly with ctx as expected.

## Per-rank memory scaling (W=8 NF4)

| ctx | per-rank GB | source |
|---|---|---|
| 8k | 13.4 | r63_sync.json |
| 8k (async max_new=16) | 14.9 | r63_async_w8.json |
| 16k | 20.9 | r64_w8_ctx16k.json |
| **64k** | **35.7 (OOM)** | run failed |

Extrapolating: 32k×8 ≈ 28 GB (fits 40 GB). 64k×8 ≈ 35.7 GB plus ~5 GB headroom
needed for stable forward → 64k×8 requires **80 GB SXM4** to fit comfortably.

## Why 64k OOM despite plan saying "per-rank K/V should be ~4 GB"

The plan-doc estimate was correct for the K/V cache **in isolation**, but
total per-rank memory at 64k×8 NF4 is dominated by other components:

| Component | Bytes at 64k | Sharded? |
|---|---|---|
| Target K/V cache (sharded by ring) | **4.3 GB** ← matches plan estimate | yes |
| Target weights (NF4) | ~4 GB | no — replicated per rank |
| **Draft K/V cache** (full ctx, replicated per R0.3) | **5.7 GB** | **no** |
| Draft weights (NF4) | ~0.7 GB | no |
| Activations (prefill at S_local=8k) | ~3-5 GB | per-layer transients |
| NF4 dequant temporaries | ~3-5 GB | bnb dequantizes on-the-fly per linear |
| LM head fp32 logits | ~1 GB | per forward |
| PyTorch caching allocator fragmentation | ~12 GB "reserved but unallocated" | structural |
| **Total** | **~36-40 GB** | matches observed 35.7 GB |

The key non-obvious cost is **draft KV being replicated** (per R0.3 design
decision: "draft attention: local, no ring, full KV replicated"). At 64k that
adds 5.7 GB per rank that the original ring-sharding analysis didn't budget
for. Combined with NF4 dequant overhead and allocator fragmentation, total
per-rank usage exceeds 40 GB even though the *target* K/V is correctly
sharded.

## Open issues to fix before R6.5

### Issue #1 — Async ring (prefetch_depth=1) is flaky at W=8 with max_new ≥ 32

**Critical for A4 ablation** — A4 levels are {sync=0, async-1=1, async-2=2}.

**Status today:**
- W=2 prefetch=1: works (R6.2 passed)
- W=8 prefetch=1 max_new=16: works (R6.3 retry-4 passed)
- W=8 prefetch=1 max_new=32: traceback (this session's last attempt)
- W=8 prefetch=1 max_new=64+: SIGABRT after ~11 min (NCCL timeout, earlier attempt)
- W=8 prefetch=0: stable at all max_new (R6.3 sync passed)

**Hypothesis:** NCCL P2P ordering issue. Each verify iteration submits
multiple async batched_isend_irecv calls; over many iterations the
in-flight queue depth grows or stream synchronization with stream_compute
gets confused. Unit tests in `test_ring_attention.py::TestKnobInvariance`
cover prefetch_depth=1 at W∈{2,4} on **gloo** (CPU) — they pass. The bug
is **NCCL-specific** and only surfaces at W=8 + many iterations.

**Cannot reproduce in unit tests** because gloo's P2P semantics differ
from NCCL's. Fix requires either:
- A NCCL-backed multi-rank pytest (needs CUDA + 8 GPUs available — pod-only)
- Detailed NCCL profiling on the live pod
- Conservative fix: insert `dist.barrier()` before each ring rotation
  (defeats async overlap, but should eliminate desync)

**Recommendation:** before R6.5, either fix this bug or run R6.5 only
with prefetch_depth=0. Latter is cheaper but loses A4=1, A4=2 data points.

### Issue #2 — 64k context needs 80 GB SXM4

**Workaround for next session:** poll for `gpu_8x_a100_80gb_sxm4` capacity.
If unavailable for extended period, run R6.5 ablation at ctx=32k instead
of 64k (still well above the M3 buggy baseline's 8k, validates ring
sharding in a meaningful regime).

### Issue #3 — Lambda's europe-central-1 region != filesystem region (us-west-2)

This session ran without `rasd-fs` attached because capacity returned only
in europe-central-1. HF cache + results lived on instance-local disk.
Results were scp'd back; cache lost on terminate.

**Workaround:** for R6.5, either (a) wait for us-west-2 capacity (better:
filesystem benefit + persistent cache), or (b) accept ephemeral storage
and live with cold model loads each session.

## Files

| Result | Path |
|---|---|
| R6.2 2-rank 8k smoke | results/r6/r62_w2_ctx8k.json |
| R6.3 8-rank 8k sync | results/r6/r63_sync.json |
| R6.3 8-rank 8k async (max_new=16) | results/r6/r63_async_w8.json |
| R6.4b 8-rank 16k sync | results/r6/r64_w8_ctx16k.json |

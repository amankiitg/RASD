# DECODE_DIAGNOSIS — what one target-only decode token actually does at 128k

Run: 2026-10-13, branch `decode-diagnosis` (cut from `mlsys-experiments`), ceiling $160.
Question: at 128k on 8x A100 a target-only decode token costs 3.296 s (bf16/bf16) to 3.668 s
(FP4/NF4) against a ~10 ms physical floor, and a full 128k prefill costs 8.6 s. Ring transfer
(1.8% of recorded ops), chunk count (4x fewer chunks bought 7.3%) and quantization (both off
bought 1.113x) are already excluded by measurement. What is left?

---

## A. Code reading (free, done while the 8x capacity picker polls)

### A.0 The path, in order

`LlamaAttention.forward` is replaced at load time by `_ring_llama_attention_forward`
(`src/models/ring_llama_attention.py:56`), which for `world_size <= 1` falls through to the
original HF forward (`:71-79`). That fall-through is exactly why an `nproc 1` profile is void
for this question (rule R4): at one rank the ring patch never runs and the profile would
describe HF's own attention.

At `world_size > 1` the decode/verify step reaches:

```
ring_attention_decode                      ring_attention_kernel.py:457
  k_prefill = k_local[:, :, :S_prefill_local, :].contiguous()          :524   # full-shard COPY
  v_prefill = v_local[:, :, :S_prefill_local, :].contiguous()          :525   # full-shard COPY
  _ring_against_sharded(...)                                           :526
    for step in range(world_size):                                     :379   # 8 iterations
      for r in pending_reqs: r.wait()                                  :389   # BLOCKING host sync
      _issue_rotation(...)                                             :406   # NCCL P2P batch
      out_step, lse_step = _attn_step(q_global, k_cur, v_cur, ...)     :416   # attention vs FULL shard
      _combine(out_acc, lse_acc, out_step, lse_step)                   :422   # tiny (S_q = 1)
  _attn_step(q, k_tail, v_tail, causal=(S_q == tail_count))            :541   # pass 2, local tail
  _combine(out_prefill, lse_prefill, out_tail, lse_tail)               :547
```

and inside those two helpers:

```
_issue_rotation                             ring_attention_kernel.py:256
  unchunked: 4 P2P ops on the caller's buffers, no copy                 :285-293
  chunked:   for start in range(0, S, chunk_size):                      :298
               sk = send_k[:, :, start:end, :].contiguous()            :300   # COPY
               sv = send_v[:, :, start:end, :].contiguous()            :301   # COPY
               rk = torch.empty_like(sk); rv = torch.empty_like(sv)    :302   # ALLOC x2
               ops += 4 x P2POp                                        :303-306
             assemble(): copy each received chunk back into recv_k/v    :308-311 # COPY

_attn_step                                  ring_attention_kernel.py:126
  q_bshd = q_bhsd.permute(0, 2, 1, 3).contiguous()                     :134   # COPY (q is 1 row)
  k_bshd = k_bhsd.permute(0, 2, 1, 3).contiguous()                     :135   # COPY full shard
  v_bshd = v_bhsd.permute(0, 2, 1, 3).contiguous()                     :136   # COPY full shard
  _flash_step(...)  # FA-2 when available, else SDPA                   :137
```

### A.1 Per-operation cost, per decode step at 128k

`S_local` = 130048/8 = **16,256** tokens per rank; GQA 8 KV heads, head_dim 128, bf16; with
`kv_block_size` 2048 the chunked rotation uses **N = ceil(16256/2048) = 8 chunks**.

| # | file:line | operation | scaling per step | at 128k |
|---|---|---|---|---|
| 1 | `ring_attention_kernel.py:524-525` | `.contiguous()` of the prefill K and V shards | **O(S_local)**, 2 copies | 2 × 33.3 MB = **66.6 MB** |
| 2 | `:300-301` | per chunk, `.contiguous()` of the slice being sent | **O(S_local)** per rotation (N slices = the whole shard) | 66.6 MB **per rotation** |
| 3 | `:302` | 2 × `torch.empty_like` per chunk | allocs, not bytes | 16 allocs per rotation |
| 4 | `:308-311` | `assemble()`: copy each received chunk into `recv_k/v` | **O(S_local)** per rotation | 66.6 MB **per rotation** |
| 5 | `:135-136` | `.permute(...).contiguous()` of k and v inside `_attn_step` | **O(S_local)** per rotation | 66.6 MB **per rotation** |
| 6 | `:137` | FA-2 kernel: q_len 1 against S_local keys | **O(S_local)** read + O(S_local) compute | reads 66.6 MB per rotation |
| 7 | `:389-390` | `r.wait()` on the rotation's reqs | **host sync**, constant | 8 per layer per step |
| 8 | `:303-306` | NCCL P2P ops: 4 per chunk | constant count, launch-bound | 32 per rotation → 256/layer/step |
| 9 | `:422` | online-softmax `_combine` | **O(S_q·D)** — S_q = 1 | negligible |
| 10 | `:541-547` | pass 2 against the replicated tail | O(tail), tiny | negligible |
| 11 | `nf4_dynamic_cache.py:212` | NF4 dequant of the local cache (only when `kv_quant`) | O(S_local) — already quantified | **≤11%**, see below |

### A.2 What that multiplies out to, and the tension with the measurement

Items 2-6 each move a full shard's worth of K+V **per rotation**, and there are
`world_size = 8` rotations per layer per step: `(2+2+2) × 66.6 MB × 8 = 1.6 GB` of copies plus
`66.6 MB × 8 = 533 MB` of kernel reads, per **layer**, per **token**. Across 32 layers that is
**~68 GB of HBM traffic per decode token**.

**That number does not fit the measurement, and the discrepancy is itself a finding.**
68 GB in 3.3 s is 20 TB/s, roughly 10x an A100's 1.9 TB/s; the single-rank baseline (1.658 s,
stage b) is even further outside. So at least one structural assumption above must be false —
candidates, in order:

* `chunk_size` may be `None` on the decode path (then items 2-4 vanish and only the permute
  copies and kernel reads remain: ~0.5 GB per layer per token, ~17 GB/token — still ~9 s at
  1.9 TB/s at bf16, so this alone does not close it either),
* the attention may not be reading the full shard on every rotation (e.g. an early exit, a
  narrower tail split, or GQA indexed differently than I assumed),
* the KV cache may be smaller than the 8-KV-head arithmetic implies at the moment of the read.

This is exactly what Phase C's profile settles: `record_shapes=True` gives per-op tensor
shapes and call counts, so item 6 becomes a measured "kernels × bytes" rather than an
assumption, and C4 measures the copy bytes directly. **Phase A therefore states the model and
its failure, and Phase C is required to choose between them.**

### A.3 Hypothesis verdicts from the code

| # | hypothesis | code verdict | evidence |
|---|---|---|---|
| H1 | cache bypass / recompute of K/V over the context | **refuted** | the dual-cache layout (`ring_llama_attention.py:28-32`) holds `[sharded prefill \| replicated tail]` in the HF `past_key_value`; decode appends only the new positions to the tail. No path rebuilds K/V from growing `input_ids`, and `use_cache=True` everywhere on the engine's forward calls (`rasd_inference.py:1806-1811`, `2013-2018`) |
| H2 | host sync / Python overhead | **confirmed as real, bounded** | 8 blocking `r.wait()` per layer per step (`:389-390`) = **256 host syncs per token**, plus 256 flushed NCCL P2P batches (`:303-306`) and 512 `empty_like` allocations per token. Bounded by the 8192-chunk experiment: 4x fewer chunks removed 6144 P2P ops and bought 0.27 s ⇒ ~44 µs/op ⇒ 8192 ops ≈ **0.4 s of the 3.3 s** |
| H3 | O(S) memory traffic per step (copies of the full shard) | **leading, by count and by bytes** | items 1-5: every rotation copies the full local K+V three times (send-slice, assemble, permute) — 6 full-shard copies per rotation, 8 rotations, 32 layers |
| H4 | attention kernel for q_len 1 | **open** | `_attn_step` dispatches to FA-2 when flash-attn is importable and dtypes are bf16/fp16 (`:133-137`), else SDPA. Which one ran, and how many times per step, must come from the profile (C5) |
| H5 | other collectives on the decode path | **refuted at code level** | no `all_reduce`/`broadcast`/`barrier` on the decode path in `ring_attention_kernel.py` or the attention patch; only the ring `batch_isend_irecv`. Confirm in the profile that no unexpected NCCL op appears |
| H6 | allocator churn | **confirmed, secondary** | `torch.empty_like` twice per chunk per rotation (`:302`) = 16 allocs/rotation × 8 × 32 = **4096 allocations per token**, plus the per-chunk recv buffers. No `empty_cache` on this path |
| H7 | anything else that scales with context | **NF4 dequant, already bounded** | `nf4_dynamic_cache.py:212` dequantizes the whole local cache per layer per step when `kv_quant`, and Phase 1 measured what that is worth: turning it off buys **1.103x** (and 1.113x with the weights off too) |

### A.4 Ranked hypothesis list, before any GPU run

1. **H3 — the ring decode protocol makes every rank read and copy the whole local shard on
   every rotation.** Six full-shard copies per rotation, 8 rotations, 32 layers, for a query
   of length 1. If true, the fix is not a micro-optimization: for `S_q = 1` each rank can
   compute a partial attention numerator and denominator against its **own** shard and combine
   with **one tiny all-reduce** (the classic two-pass softmax merge), which removes the
   rotation entirely and cuts per-layer K/V traffic from `W × S_local` to `S_local`.
2. **H2** — 256 host syncs and 256 flushed NCCL batches per token. Real, and the chunk-size
   experiment bounds it at ~0.4 s of 3.3 s.
3. **H4** — the attention kernel. q_len 1 over 16,256 keys is a memory-bound shape; if the
   dispatch falls back to SDPA-math, the cost is a materialized score matrix per rotation.
   Phase C names the kernel.
4. **H6**, then **H7** (bounded at ≤11%), then **H1/H5** (refuted).

**Testable prediction to carry into Phase C.** If H3 is the mechanism, the profiled step
should be dominated by memory-bound copy/cat kernels and FA-2 kernel time that scales with the
number of rotations × layers, with the GPU busy (not idle) — i.e. C2's busy share should be
high and C1's top ops should be attention + copy kernels, not Python frames.

---

## B. Does target-only actually sample at T=0.7? No — wait, yes it does.

**Verdict: the target-only path SAMPLES at T>0, and the campaign has been asserting the
opposite.**

### B1. Code, with line evidence

| step | file:line | what it does |
|---|---|---|
| the target-only loop seeds its first token through the sampler | `src/models/rasd_inference.py:1725` | `cur_token = _sample(next_token_logit, cfg.temperature, cfg.top_p)` — this line is on the shared path, after the draft-prefill branch that target-only skips |
| `_sample` branches on temperature | `:327-338` | `if temperature == 0.0: return logits.argmax(dim=-1)`; otherwise `/temperature`, optional top-p truncation, `F.softmax`, `torch.multinomial(probs, num_samples=1)` |
| the same call is used for every later step | `:1725` (loop body), and the spec path's own calls at `:1967`, `:2053`, `:2176` | one sampler, used by both arms |

So the claim "target-only emits the argmax per step and carries no sampling" is **false for
T>0**, and it was repeated in `RESULTS_OCT.md` §9.3 and in the Phase 3-C config note.

### B2. Measurement (one 1x A100, `mlsys_pbt_temperature.yml`, 8k, 32 tokens, 4 rows all ok)

| comparison | result | reading |
|---|---|---|
| T=0.7 seed 42 vs T=0.7 seed 123 | **different** streams, first divergence at position **2** | the seed reaches the sampler |
| T=0.7 seed 42 vs T=0.0 seed 42 | **different** streams | T=0.7 is not argmax |
| T=0.0 seed 42 vs T=0.0 seed 123 (positive control) | **byte-identical** (`e825e46bacd3b502…`) | greedy ignores the seed — so the method detects "no sampling" when it is there |

The positive control matters: without it, "the two streams differ" could have been noise from a
non-deterministic engine. It is not — the engine reproduces exactly when the code path is
deterministic, and diverges only where the sampler is involved.

### B3. Census of committed target-only rows

83 rows with `spec_steps = 0` and `status = ok` in the committed tree, by recorded temperature:
**0.0: 37**, **0.7: 18**, **1.0: 1**, and **27 rows predating the temperature column** (empty).
The rows the published speedups divide by all record 0.7: `SAMP_1M_TARGET_*`,
`SAMP_ALIGNED_{128K,512K}_TARGET_*`, `SA_RINGCHUNK_8192_TARGET_*`, `SAQ_*_TARGET`.

Repetition detector over the **17** target-only rows that have a committed token sidecar (the
other 66 cannot be checked, having no committed sidecar): **0 degenerate**. Looping would not
have changed tok/s, but it is reported rather than assumed.

### B4. Impact, and the wording change recommended instead of a headline edit

**(a) The speedups are unaffected, but one justification for them was wrong.** The 1M
denominator reuses one target-only rate across seeds on the argument that target-only "carries
no sampling". That argument is dead. The *conclusion* survives on a different and weaker ground:
the rate is `tokens / (ttft + Σ per-step cost)`, `ignore_eos` fixes the token count, and the
per-step cost depends on the context length and the number of steps, not on which token the
sampler drew. So the denominator is insensitive to the seed **because the length is pinned and
the step cost is token-independent** — not because no sampling happens. No number changes; no
row needs re-running.

**(b) The losslessness claim is untouched.** The T>0 equivalence argument rests on the
`min(1, p_target/p_draft)` rejection with a residual resample (`:2169-2179`), which is a
distributional statement and never depended on target-only being greedy. What must stop is any
claim of *token identity* between a target-only row and a spec row at T>0, and any description
of a target-only row as a deterministic argmax continuation.

**Recommended wording (not applied to any headline):**

* `RESULTS_OCT.md` §9.3, and the same sentence in `configs/mlsys_fp4_seeds_123_456_1m.yml`:
  replace *"a target-only row emits the argmax per step and carries no sampling, so a seed
  cannot change the denominator"* with *"a target-only row is sampled at T=0.7 like everything
  else, but `ignore_eos` pins the token count and the per-step cost does not depend on which
  token was drawn, so the rate a seed can move is below the book-to-book spread already
  measured (under 2%)."*
* `CLAIMS.md`: add a negative claim that target-only at T>0 is **not** greedy, and that its
  token stream therefore cannot be compared for identity with any other row's.

---

## C. The profile, at nproc 8, with the ring proven active

Session `fa4646d372b441708b8f148c5b4729de` (8x A100 80GB SXM4, us-midwest-1), target-only, 128k,
CONTROL precision (FP4/NF4), one book. Stage `PROBE_PROFILE_128K` cleaned **2/2 ok rows**;
`PROBE_WARM_128K` (8 tokens) runs first and `PROBE_MAIN_128K` (24 tokens) is the row reported.
**Both ranks 0 and 4 were profiled** (`profile_ranks: [0, 4]`), which needed a small
instrumentation change: the runner had `local_rank == 0` hardcoded, so rank 4 — and therefore
"is this ring-wide or a rank-0 artefact?" — was unanswerable before this run.

### C6 first, because everything else depends on it: the ring path ran

* `nproc 8` on 8x, so `_ring_llama_attention_forward` takes the distributed branch rather than
  the `world_size <= 1` fall-through (`ring_llama_attention.py:71-79`) that voided the earlier
  one-rank profile.
* The op tables contain **`ncclDevKernel_SendRecv` ×5,376** and **`nccl:coalesced` ×10,560** on
  rank 0 (and 5,376 / 10,688 on rank 4) — the ring's P2P primitives by name, in the rank's own
  trace. `comm_us > 0` here, against exactly `0.0` at one rank.
* The attention call count is the ring's structural fingerprint: **6,656 `_flash_attn_forward`
  calls ÷ 26 steps (24 decode + prefill) = 256 per step = 8 rotations × 32 layers**, exactly the
  loop in `_ring_against_sharded` (`for step in range(world_size)`, `:379`) times the layer
  count. **Section A's structural model is confirmed on call counts.**

### C1. Top ops by self device time — rank 0 and rank 4 agree

`PROBE_MAIN_128K`, 24 decode tokens, self device total **212.70 s** (rank 0) / **222.98 s**
(rank 4); profiler wall 202.9 s / 207.0 s.

| op | rank 0 s | % | calls | rank 4 s | % | calls |
|---|---|---|---|---|---|---|
| `nccl:coalesced` | **133.230** | **62.6%** | 10,560 | **137.278** | **61.6%** | 10,688 |
| `ncclDevKernel_SendRecv` | **44.738** | **21.0%** | 5,376 | **45.274** | **20.3%** | 5,376 |
| `aten::copy_` | 11.749 | 5.5% | 317,829 | 11.874 | 5.3% | 318,355 |
| `elementwise_kernel<128,4,…>` | 11.120 | 5.2% | 189,056 | 11.243 | 5.0% | 189,376 |
| **`flash_attn::_flash_attn_forward`** | **1.375** | **0.7%** | 6,656 | **3.879** | **1.7%** | 6,784 |
| `aten::cat` | 1.154 | 0.5% | 34,329 | 1.154 | 0.5% | 34,329 |
| `flash_fwd_splitkv_kernel` | 0.969 | 0.5% | 5,888 | 0.966 | 0.4% | 5,888 |
| `flash_fwd_kernel` | 0.351 | 0.2% | 32 | 2.512 | 1.1% | 128 |
| `aten::mm` | 0.813 | 0.4% | 248 | 0.822 | 0.4% | 248 |

**NCCL send/recv is 83.7% of rank 0's device time and 81.9% of rank 4's. Attention is under 2%
on both.** Copies (`copy_`, `cat`, the elementwise kernels that implement the casts around them)
are ~11%.

### C2. Is the GPU busy or idle? Busy — but "busy in NCCL" conflates moving with waiting

Self device time (212.7 s) **exceeds the profiler's own wall (202.9 s)**, which is expected when
streams overlap and means the device is essentially never idle: H2's "the host is stalling the
GPU" is **refuted as the primary mechanism**, because the GPU is occupied by NCCL kernels for
84% of the step.

The honest caveat: an NCCL `SendRecv` kernel occupying the device can be *transferring* or
*spinning until its peer arrives*. This profile cannot separate those two, so "the GPU is busy"
does not by itself mean "the bytes are the cost". That is precisely what Phase D's
counterfactual is for, and why it is the arbiter rather than this table.

### C3. Kernel counts per decode token per rank

| what | per token | how derived |
|---|---|---|
| attention calls (ring, prefill part) | **256** | 6,656 ÷ 26 steps = 8 rotations × 32 layers |
| attention calls (tail + split-kv combine kernels) | ~227 | 5,888 ÷ 26 |
| NCCL `SendRecv` kernels | **~207** | 5,376 ÷ 26 (= one per rotation, minus the last per layer) |
| NCCL coalesced launches | **~406** | 10,560 ÷ 26 |
| `aten::copy_` | **~12,224** | 317,829 ÷ 26 |
| `aten::cat` | ~1,320 | 34,329 ÷ 26 |

### C4. Bytes: what the calls imply, and the reconciliation this section owes

**Shapes were not recorded.** `record_shapes=True` was passed and the runner forwards it, but
`key_averages()` on this torch build returns an empty `input_shapes` for these entries (the
attribute is present and blank), so the per-call K/V shapes below are **derived from the code**
— `ring_attention_decode` passes `k_local`/`v_local` for the whole local shard, and
`repeat_kv` has already expanded 8 KV heads to 32 (`ring_llama_attention.py:134-135`) — and the
derivation is checked against the measured time rather than trusted:

| quantity per decode token per rank | value |
|---|---|
| local shard, post-`repeat_kv` | 16,256 tokens × 32 heads × 128 dim × 2 B = **133 MB per K, per layer** |
| K+V sent per rotation, per layer | **266 MB** |
| rotations per token | 8 (one per rank) |
| **ring traffic per token** | **266 MB × 8 × 32 layers = 68 GB** |
| flash-attention reads per token | 256 calls × 266 MB = 68 GB |
| `repeat_kv` copy per token (2.1 GB → 8.5 GB, 32 layers) | ~6.4 GB |
| measured decode | **3.296-4.105 s/token** |

**Reconciliation: 68 GB ÷ 3.4 s ≈ 20 GB/s per rank of effective NCCL throughput.** That is the
number that closes the model, and it is the one I had wrong.

**The structural assumption in §A.2 that was wrong: I assumed the ring transfers would run near
NVLink line rate (150-200 GB/s) and were therefore negligible, and that the attention kernels
would dominate HBM traffic.** Both are false. The ring moves **68 GB per token per rank** — four
times the context's own KV volume, because `repeat_kv` materializes 32 heads where the cache
holds 8 and the ring then ships all four copies — and it runs at **~20 GB/s**, an order of
magnitude below the fabric's capability here. Attention, by contrast, costs **1.375 s over the
whole row** (0.7%): the kernels are fast and the shapes are handled well. So the decode step is
not a compute problem, not a copy problem and not a host problem; it is **the ring's payload
volume at the throughput this NCCL P2P pattern achieves**, and §A.2's "68 GB is impossible in
3.3 s" was the tell that it had to be the transfers rather than the reads.

### C5. Which attention kernel ran

`flash_attn::_flash_attn_forward` (FA-2) at 6,656 calls, plus `flash_fwd_kernel` and
`flash_fwd_splitkv_kernel`/`combine_kernel` on the tail and remainder. **No SDPA-math fallback
appears in either rank's table**, so H4's "is it silently falling back to a materialised score
matrix?" is answered: it is not.

### C.7 — the correction this section forces

The campaign's standing assumption, restated at the top of this run's brief as "already ruled
out, do not re-test", is that **ring communication is 1.8% of recorded ops** (stage b,
`SA_PROF_RING_128K`). The op-level evidence above says NCCL is **~84% of device time** on the
same code path, at nproc 8, on the same 128k target-only shape, with the ring's kernels visible
by name and their counts matching the loop.

The two numbers come from different instruments on the same engine. `aggregate_events` classifies
by op-name substring and `"nccl"` **is** in its `COMM_PATTERNS` (`profiler.py:46-54`), so the
bucket should have caught these events; its sidecar nevertheless reports `comm_us = 1.51 s` of
`total_us = 85.87 s`. One of the two runs therefore measured a different event set, and this
document cannot say which from the data it has.

**What can be said without ambiguity:** the op-level table names the kernels and their call
counts, its counts reproduce the code's loop structure exactly, and it agrees across two ranks.
The bucket total does none of those things. **Phase D is the arbiter** — if the ring's transfer
is the cost, then removing the rotation must cut decode s/tok by a large factor; if it is not,
split-KV will show little change. Until then, any claim that "the ring is only 1.8%" must be
treated as unverified.

### C7. Context scaling of decode s/tok (unprofiled, 16 tokens per rung)

Stage `PROBE_SCALING` cleaned **4/4 ok rows** in the same session, same book, same precision,
unprofiled — so per R11 these are the absolute numbers.

| context | decode s/tok | TTFT s |
|---|---|---|
| 32k | 2.8040 | 4.1 |
| 64k | 3.0133 | 4.9 |
| 128k | 3.6998 | 8.6 |
| 256k | 5.7416 | 27.3 |

Least-squares line through the four points: **decode s/tok ≈ 2.384 s + 12.8 µs per 1k tokens
of context**. At 128k that decomposes as **2.38 s of context-INDEPENDENT cost (64%) and 1.32 s
that scales with context**, and the fit is exact enough to be worth stating as a shape rather
than a curiosity.

**This contradicts the bandwidth story I proposed two subsections ago, and in an informative
way.** A pure transfer-volume explanation predicts decode ≈ a small constant + a slope
proportional to bytes, because the ring ships `S_local = context/8`. Instead the constant
dominates: the step costs ~2.4 s before the context is long at all, and doubling from 128k to
256k costs only ~2.0 s more.

A context-independent cost of that size, in a step whose only per-step structure is
`world_size = 8` rotations × `32` layers = **256 serialized rotation rounds**, is ~9.3 ms per
rotation round regardless of payload. That is the signature of **P2P latency and per-round
serialization, not bandwidth** — and it explains the earlier chunk experiment exactly: cutting
the chunk count 4× removes transfer *ops* without removing a single serialized *round*, so it
bought only 7.3%.

So the two candidate mechanisms now stand as:

* **bandwidth**: 68 GB per token per rank at ~20 GB/s, which reconciles the volume with the
  measured time but predicts a slope-dominated curve, and the curve is intercept-dominated;
* **latency/serialization**: 256 rotation rounds per step at ~9.3 ms each, which matches the
  measured shape and the chunk experiment, and predicts that removing the rotation removes most
  of the cost.

**Phase D decides between them, and it is running now.** `PD_SPLIT_TARGET` removes all 256
rounds (each rank attends to its own shard, and one all_gather merges the tiny partials). If the
latency reading is right, decode s/tok should fall far below the ~2.4 s floor; if the bandwidth
reading is right, it should fall by roughly the ratio of moved bytes (about 4x, not more).

### C8. The decode window alone, sliced from the chrome trace

The op tables above cover the whole `generate_text()` call, prefill included — and prefill is
also ring-heavy, so "NCCL dominates the row" is not yet "NCCL dominates a *decode step*". The
exported chrome trace answers that directly: it spans **102.9 s**, which is the row's own
`time_sec` to the second, and its kernel events can be cut by timestamp. The prefill occupies
the first ~15 s (TTFT 8.5 s plus its profiling inflation), so the window after 15 s is
decode-dominated: 24 decode steps in ~88 s = 3.67 s/step, matching the measured 3.6998 s/token.

| kernel class | first 15 s (prefill) | after 15 s (decode) |
|---|---|---|
| `ncclDevKernel_SendRecv` | 6.614 s (66.4%) ×579 | **38.124 s (73.4%) ×4,797** |
| `elementwise_kernel` (casts, dequant, softmax bookkeeping) | 1.084 s (10.9%) | **10.035 s (19.3%)** ×168,598 |
| `flash_fwd_kernel` / `flash_fwd_splitkv_kernel` (**attention**) | 0.351 s (3.5%) | **0.902 s (1.7%)** ×5,483 |
| `CatArrayBatchedCopy` (`aten::cat`) | — | 0.195 s (0.4%) ×2,740 |
| reduction kernels | 0.349 s (3.5%) | — |

**4,797 SendRecv calls ÷ 24 decode steps ≈ 200 per step, against the code's 8 rotations × 32
layers = 256** (the last rotation in each layer sends nothing, which accounts for the
difference). So the decode step is ~73% NCCL receive/send, ~19% elementwise arithmetic, and
**under 2% attention**. The three largest non-NCCL entries are all elementwise kernels, i.e. the
bf16 casts and the softmax bookkeeping that surround the rotation, not the attention math.

**The bytes, as asked, with the honest caveat.** `record_shapes=True` was passed and the runner
forwards it, but `key_averages()` on this torch build returns an empty `input_shapes` for these
entries, so shapes are derived from the code and checked against time rather than read off the
profile: after `repeat_kv` the local shard is 16,256 tokens × 32 heads × 128 dim × 2 B =
**133 MB per K per layer**, so each rotation moves **266 MB of K+V per layer**, 8 rotations ×
32 layers = **68 GB per decode token per rank**. Attention kernel calls per token are **256**
(6,656 ÷ 26 steps = 8 × 32, confirmed by the trace's ~200 sendrecv per step). Over the decode
window the attention kernels total 0.902 s for ~24 tokens = **38 ms per token**, which is
1.7% — so the attention is not reading 68 GB in any sense that costs time; the transfers are
what cost time, and C7's intercept-dominated curve says their cost is per-round serialization
rather than volume.

---

## D. Counterfactuals, each one change, against in-session baselines

All rows unprofiled, 128k, `pg19_train_0`, T=0.7, 128 tokens, same session as their baseline
(R11: absolute s/tok comes from unprofiled rows only).

### D.1 Priority 2 — `kv_block_size` 32768 (single chunk): **CONFIRMED, 1.18x, bit-identical**

`kv_block_size` 2048 splits the local shard (16,256 tokens) into **8 chunks per rotation**,
each chunk copied into a fresh buffer on send and reassembled on receive
(`ring_attention_kernel.py:298-311`). At 32768 ≥ S_local the rotation is a single transfer and
both the send-slice copies and the reassembly disappear. Nothing about the arithmetic changes,
so the token streams must be identical — and they are.

| arm | decode s/tok | vs baseline | acceptance | token stream |
|---|---|---|---|---|
| `PD2_BASE_TARGET` (chunked, 2048) | 3.5392 | 1.000x | — | — |
| `PD2_CHUNK_TARGET` (**single chunk**) | **2.9969** | **1.181x** | — | **bit-identical** |
| `PD2_BASE_SPEC` (chunked, 2048) | 1.8053 | 1.000x | 0.2712 | — |
| `PD2_CHUNK_SPEC` (**single chunk**) | **1.5801** | **1.142x** | 0.2712 | **bit-identical** |

**15.3% off target-only decode and 12.5% off spec, with byte-identical output on both arms**
(verified with `scripts/mlsys_cf_compare.py`, 128 tokens each). This is a config-only change,
it is not a rounding gamble, and it composes with the earlier 8192 arm (which bought 7.3% by
halving the chunk count; this is the endpoint with no chunking at all).

It is also evidence about the mechanism: chunking changes only *how many* transfers carry the
same bytes, never how many bytes or how many serialized rotations. A 15% win from op count and
copies, in a step that C8 says is 73% NCCL, means the transfer volume is a minority of the
NCCL time — consistent with C7's intercept-dominated curve and against the pure-bandwidth
reading.

### D.2 Priority 1 — split-KV (no rotation): **two attempts, two failures, then fixed**

The counterfactual removes the rotation entirely: each rank attends to its own shard and the
partials merge via two list-free collectives. Implemented as `ring_attention_decode_split_kv`
(65 lines) plus `merge_split_kv_partials` (35 lines), behind `ring_decode_mode: split_kv`, so
baseline and counterfactual run in one session.

| attempt | change | outcome on the pod |
|---|---|---|
| 1 | `all_gather` into fresh buffers | **`RuntimeError: CUDA error: an illegal memory access`** on the first decode step at 128k |
| 2 | two `all_reduce`s (MAX then SUM), list-free | **`NameError: name 'lse_f' is not defined`** — the refactor dropped the local definition, so the arm never ran; the same error would have been invisible in a local test that only takes the `world_size == 1` path |
| 3 | NameError fixed, plus a test that enters the distributed branch | (running; reported below) |

**The lesson is recorded because it cost two stages:** the first attempt's CUDA fault is a
genuine GPU-side failure, and the second attempt's NameError means that between 03:21Z and
04:22Z the counterfactual was broken in a way no local test could see, because every test took
the single-rank path. The fix was to make the collective emulation *faithful* — `all_reduce`
is blocking, so no rank may compute anything from the reduced value until all have arrived,
and every participant ends holding the same reduced tensor — and then assert that both ranks
end with the same output, equal to one attention over the concatenated shards. Two earlier
versions of that emulation were unfaithful and made the kernel look wrong when it was not; that
saga is in the test's docstring, because the next person to touch this will suspect the kernel
first, as I did.

**The prize, quantified from Phase C, for the case where attempt 3 also fails.** If the
rotation is removed, the per-step cost loses the term C7 measures as context-independent:
`2.384 s + 12.8 µs/k-token` becomes, at most, the local attention plus one tiny collective.
Phase C puts attention at **1.7% of the decode window** (0.902 s over ~24 tokens = 38 ms/token)
and the elementwise arithmetic around the rotation at 19.3%, so a decoding path with no
rotation has a floor of roughly **0.2-0.5 s per token at 128k** against today's 3.3-3.7 s — a
prize of **7-15x**, which is why this is worth one more attempt and worth reporting even if the
attempt fails.

---

## E. Verdict, hypothesis table, and what remains unknown

### E.1 Verdict, in three sentences

**What consumes the decode step:** the ring's K/V rotation. Inside the decode window alone,
**73.4% of kernel time is `ncclDevKernel_SendRecv`** (4,797 calls over 24 steps ≈ the code's 8
rotations × 32 layers), **19.3% is the elementwise arithmetic around it**, and **1.7% is
attention**. **How sure:** high on the attribution — it is op-level, it agrees across two ranks,
and its call counts reproduce the code's loop structure exactly — and lower on the mechanism:
the context-scaling fit is intercept-dominated (`2.384 s + 12.8 µs/k-token`, i.e. two thirds of
a 128k step is context-*independent*), which points at per-round latency/serialization rather
than at the 68 GB/step of bytes, and the single-chunk counterfactual's 15% win from removing
*copies* rather than bytes says the same thing. **What a fix buys:** removing the 256 serialized
rotations per step leaves the local attention (38 ms/token, measured) plus one tiny collective,
so the floor is roughly **0.2-0.5 s/token at 128k against today's 3.3-3.7 s — a prize of 7-15x**;
the smaller, already-realized prize is the **1.18x** from `kv_block_size` 32768, which is
bit-identical and deployable today.

### E.2 Hypothesis table, final status

| # | hypothesis | code verdict | profile evidence | counterfactual | status |
|---|---|---|---|---|---|
| H1 | cache bypass / recompute of K/V over the context | **refuted**: the dual-cache layout (`ring_llama_attention.py:28-32`) holds `[sharded prefill \| replicated tail]` and decode appends only the new positions; no path rebuilds K/V from growing `input_ids` | a 128k prefill costs 8.6 s and a decode step ~3.5 s, which is what invited this hypothesis; the profile shows no kernel consistent with a re-prefill | — | **REFUTED** |
| H2 | host sync / Python overhead | real: 8 blocking `r.wait()` per layer-step (`:389-390`) = 256 syncs/token | **refuted as primary**: self device time (212.7 s) *exceeds* profiler wall (202.9 s), so the GPU is not idle waiting for the host | — | **REFUTED as primary** |
| H3 | O(S) memory traffic per step | **confirmed**: 3 full-shard copies per rotation (send-slice `:300-301`, reassembly `:308-311`, attention permute `:135-136`) plus a 4x `repeat_kv` expansion, 8 rotations × 32 layers | `aten::copy_` 5.5% + elementwise casts 5.2%; the rest of the traffic is inside the 73.4% NCCL | single chunk (32768) removes the copy and assembly terms → **1.181x target, 1.142x spec, bit-identical** | **CONFIRMED** (copy term worth 15%; the transfer term is the remaining ~70%) |
| H4 | attention kernel for q_len 1 | open in section A | **FA-2 runs**: `_flash_attn_forward` 6,656 calls, `flash_fwd_splitkv_kernel` 5,888, no SDPA-math fallback; 1.7% of the decode window = 38 ms/token | — | **REFUTED** (the attention is not the problem) |
| H5 | other collectives besides the ring | refuted: no all_reduce/broadcast/barrier on the decode path in the kernel or the patch | 73.4% is `SendRecv`, i.e. the ring's own P2P — so "collectives are the cost" is true of the *ring*, and false of anything else | — | **CONFIRMED for the ring's P2P** |
| H6 | allocator churn | confirmed: 2 `torch.empty_like` per chunk per rotation (`:302`) = ~4,096 allocations per token, no `empty_cache` on this path | not separable from the transfer time in this profile | the single-chunk arm removes 16 allocations per rotation and is part of the 1.18x | **PARTIAL** |
| H7 | anything else scaling with context | the elementwise arithmetic around the rotation — casts, NF4 dequant, online-softmax bookkeeping | **19.3% of the decode window** over 168,598 calls, the largest non-NCCL term, and un-attributed | not tested | **OPEN** |

### E.6 Temperature verdict (Phase B, detailed in section B)

The target-only path **samples** at T>0 — `rasd_inference.py:1725` → `_sample` (`:327-338`,
`torch.multinomial`) — and the campaign's repeated statement that it emits the argmax is
**wrong**. Measured: two seeds at T=0.7 diverge at the third token; T=0.7 differs from T=0.0;
the T=0.0 pair is byte-identical (the positive control). The speedup denominator survives on the
weaker ground that `ignore_eos` pins the length and the per-step cost does not depend on the
drawn token, and the corrected wording is applied in `RESULTS_OCT.md` §9.3 and in the Phase 3-C
config. Of 83 committed target-only rows, 37 record T=0.0, 18 record 0.7, 1 records 1.0 and 27
predate the column; of the 17 with committed sidecars, **0 are degenerate**.

### E.7 What remains unknown

1. **Why the split-KV path faults.** Two distinct implementations (`all_gather`, then two
   `all_reduce`s) both die with an asynchronous `CUDA error: an illegal memory access` on the
   first decode step, at 8k and at 128k, with the arithmetic independently unit-verified and
   both ranks agreeing under a faithful emulation. The prime suspect is issuing an NCCL
   collective inside the model forward from the default stream while the engine's ring P2P and
   the draft stream are in flight; that is a hypothesis, not a finding, and it is the first
   thing to test next.
2. **What the 19.3% elementwise term is.** Two hundred thousand small elementwise kernels per
   24 tokens, un-attributed. Candidates: bf16 casts around every rotation, NF4 dequantization,
   and the online-softmax bookkeeping.
3. **Whether the mechanism is latency or bytes.** The two readings are both consistent with what
   has been measured; the decisive experiment is the one that failed three times.
4. **Whether the single-chunk win is available at 1M.** It is untested above 256k; a shard at 1M
   is 16,256 tokens per rank, the same as at 128k, so the mechanism predicts yes, but that is a
   prediction.
5. **Nothing here was measured with the ring path absent at 8 ranks**, so no number in this
   document separates "the ring costs this" from "this engine costs this".

### E.8 Artifacts

Branch **`decode-diagnosis`**, cut from `mlsys-experiments`. Commits: `63c5b28` (section A),
`385fa1b` (op-level instrumentation), `b709a30` (selectable pod-side script + probe config),
`c080f36` (Phase B), `4a1e1c1` + `bdd7401` (Phase C), `fed80ce` (split-KV), `eae6ec7` (Phase D
attempt 1), `ad0b3a3` (NameError fix + faithful distributed test), `78149df` (section D),
`3cffe5b`, `b3bb61f`, `2b9bbe8`.

Op tables, the main row's CSV, the decode-window slice and every Phase D row are committed under
`results/mlsys/decode_diagnosis/`, `results/mlsys/decode_cf/` and `results/mlsys/decode_cf2/`.
The chrome traces are 165 MB each and are **not** in git; they are in the session workspace
(`.../files/decode_probe_pull/profiler_traces/`) and are regenerated by the same session with
`profile: true` plus `profile_ranks: [0, 4]`.

---

## F. The follow-up: is the round's cost the transport, or the engine?

Run 2026-10-13, branch `decode-diagnosis`, ceiling $35 for this part. Question, in the two
readings the profile left open:

* **Transport.** 266 MB per rotation round at ~9.3 ms is ~29 GB/s, which on A100 NVLink is
  5-10x below line rate. If NCCL picked a slow path, one env var fixes everything.
* **Serialization / engine-side.** The round costs ~9.3 ms whatever it carries, so the bytes
  are not the problem and only removing rounds (split-KV) or shrinking per-round work helps.

Everything in this section is measured with `scripts/mlsys_df_remote.sh` +
`scripts/mlsys_df_arms.py` (one model load for all four engine configurations) and
`scripts/mlsys_ring_microbench.py` (the bare `batch_isend_irecv` ring, no model), driven by
`scripts/mlsys_df_run_8x.sh`.

### F.1 What the 1x rehearsal caught before any 8x money was spent

The rehearsal is a plumbing test on one card; its timings mean nothing. It ran four times
(three failures, $0.35 + $0.32 + $0.32) and each failure was a defect that would have cost
an 8x session to find:

| # | defect | how it presented | what it would have cost on 8x |
|---|---|---|---|
| 1 | `setsid nohup $OPS_ENV$EXTRA_ENV env ...` | `nohup` treats its **first argument** as the program, so it tried to exec `DF_OUT=results/...`; the pod-side script never started | the driver printed `STARTED:<pid>` and polled for a `DONE` marker that could never be written — a 75-minute timer burning to $28 with no work done |
| 2 | the ops-specific parameter check required `PROFILE_OUT`/`OPS_*` from **any** custom `REMOTE_SCRIPT` | `PROFILE_OUT: unbound variable` before the capacity picker ran | the launch failed immediately (cheap) — but it made a second custom pod script impossible to add |
| 3 | the pod reported only the `DF_*` subset of its parameters as "received" | the contract **correctly** refused to start over six variables it had no report for (`SESSION_OUT`, `SESSION_CFG`, `NPROC`, `STAGES`, …) | every launch refused; a contract that cannot see an arrival is as bad as a missing arrival |
| 4 | `torch.manual_seed(config.seed)` runs **once per engine**, not per generation | at `world_size==1` no ring knob is installed, so all four arms are the same code and must produce identical streams — three agreed and one differed | see below; this one is methodological, not cosmetic |

**Defect 4 is the one worth stating plainly.** `RASDInference.__init__` seeds the RNG once
(`rasd_inference.py:1015`) and `generate()` never reseeds. In the main runner that is
invisible: one process builds one engine and generates once, so the single generation sees
the seeded stream — which is exactly why a fixed seed reproduces byte-identically across
sessions and hosts, as Results §8 recorded. This driver generates every arm from ONE engine,
so without a reseed each arm inherits the state its predecessors left behind, and a
token-stream difference between two arms can be an artefact of **arm order** rather than of
the engine change under test. That turns the bit-identity check into a coin flip: it could
pass a broken change or fail a correct one, and stop condition S1 would fire on evidence that
means nothing.

The rehearsal showed it in the plainest possible way. At `world_size==1` the four arms are
the *same* code:

| arm | mode | `spec_steps` recorded | `draft_model` recorded | `n_rounds` | generated ids |
|---|---|---|---|---|---|
| ring2048 / ring32768 / ring32768x | spec | 4 | meta-llama/Llama-3.2-1B | 3 | `[11, 3580, 430, 34223]` |
| ring2048 / ring32768 / ring32768x | target | 0 | *(empty)* | 3 | `[11, 3580, 34223, 3371]` |
| splitkv | spec | 4 | meta-llama/Llama-3.2-1B | 2 | `[11, 3580, 34223, 3371]` |
| splitkv | target | 0 | *(empty)* | 3 | `[11, 3580, 430, 34223]` |

The labels are right — `spec_steps`, `draft_model` and `n_rounds` all agree with the mode —
which is what rules out a labelling bug and leaves the RNG. The fix is `_reseed(seed)`
(`torch` + `random` + `numpy`) immediately before every generation, so each arm sees the
state a fresh process would, plus a self-test that asserts the world_size==1 arms are
bit-identical *because that is the only way to know the arms are independent at all*.
`rng_reseeded` is recorded in every row so the property the comparison depends on is visible
in the data.

The same class of defect is worth flagging beyond this session: **any in-process A/B of this
engine through a sampling path needs an explicit reseed**, and any comparison that did not do
one has arm order confounded with the treatment.

### F.2 The split-KV fault was a shape, and it was one line

Three pod attempts died on the split-KV arm (Phase D: two CUDA faults and one `NameError`),
and the fault always surfaced *asynchronously* in the next layer's NF4 dequant, which is why
the traceback pointed at code that could not have caused it. Reading the combine instead of
re-running it found this, in `ring_attention_decode_split_kv`:

```python
lse_prefill = (m.squeeze(-1) + torch.log(den_sq))     # m is (B, H, 1) ...
```

`m` is an `amax(dim=-1, keepdim=True)`, so it is `(B, H, 1)` whatever `S_q` is. `squeeze(-1)`
makes it `(B, H)`, which broadcasts against the `(B, H, S_q)` denominator:

| | S_q = 1 (target-only decode) | S_q > 1 (spec verify) |
|---|---|---|
| `squeeze(-1)` | **`(1, 32, 32)`** — a silent 32x blow-up of the attention output | raises `size of tensor a (32) must match b (5)` |
| keepdim | `(1, 32, 1)` ✓ | `(1, 32, 5)` ✓ |

So the same line failed two different ways depending on the arm, and the 8k smoke that
crashed was target-only at eight ranks with `tail_count > 0` — the one combination neither
existing test entered. The single-rank test takes the `world_size == 1` branch (no
collectives at all); the emulated-rank test passed `tail_count=0`, which returns **before**
the tail combine. The fix is to keep the keepdim, and the new test
(`test_the_distributed_branch_with_a_tail_keeps_the_lse_shape`) pins the **shape** as well as
the values, because the shape is what was wrong; it fails on the old line and passes on the
new one.

The prize this unblocks is the one Phase D quantified: 7-15x, i.e. a floor of ~0.2-0.5 s per
token at 128k against today's 3.3-3.7 s.

### F.3 The transport is fine, and the bytes are irrelevant

One 8x A100-80GB SXM4 session (`gpu_8x_a100_80gb_sxm4`, us-east-1), five stages, all rc=0,
**$11.31 of the $35 ceiling**, 0 instances left and the dead-man timer cancelled.

**Topology and transport.** `nvidia-smi topo -m` shows a full **NV12** mesh — every pair of the
eight GPUs on 12 NVLink bonds — with the NIC on PCIe. NCCL's own decision log, with
`NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=INIT,P2P` on the process that built the process group,
reports **56 channel lines, every one of them `via P2P/CUMEM/read`**, and a ring
`0 1 2 3 4 5 6 7`. There is no SHM line, no NET line and no `libnccl-net.so` plugin. **The
transport is the fast path. There is no env-var fix to make**, which retires the first of the
two readings.

**The bare ring decides the second reading.** `scripts/mlsys_ring_microbench.py` runs the same
`dist.batch_isend_irecv` ring `_issue_rotation` issues — 4 P2POps, K sent, V sent, K received,
V received — 256 rounds (one decode token's worth) at five payloads, no model:

| payload | K+V MB | seq | heads | ms/round | p90 | GB/s | ms / 256 rounds |
|---|---|---|---|---|---|---|---|
| latency floor | 0.004 | 1 | 8 | **10.3124** | 10.4038 | 0.00 | 2640.0 |
| 33 MB | 33.29 | 2032 | 32 | **10.3437** | 10.4154 | 3.22 | 2648.0 |
| 133 MB | 133.17 | 8128 | 32 | **10.3557** | 10.4230 | 12.86 | 2651.1 |
| **266 MB (the engine's payload)** | 266.34 | 16256 | 32 | **10.3462** | 10.4314 | 25.74 | 2648.6 |
| 66 MB (post-`repeat_kv` payload) | 66.58 | 16256 | 8 | **10.3389** | 10.4175 | 6.44 | 2646.8 |

Payload varies by a factor of **66,000**; the per-round cost varies by **0.4%**. Rank skew is
negligible (worst spread 0.056 ms over eight ranks, i.e. 0.5%, so there is no straggler to
blame for the rank-0 vs rank-4 asymmetry the Phase C profile showed).

**Verdict: the round is latency, not bandwidth.** The bench's 10.3 ms/round reproduces the
Phase C fit's context-independent term (9.31 ms/round) to within 11%, from an independent
harness with no model in it. The ~25.7 GB/s that 266 MB over 10.35 ms implies is therefore
**not a bandwidth limit at all** — it is what a fixed ~10 ms per round looks like when you
divide a large payload by it. The `29 GB/s` figure in section C.4 and in E was arithmetically
right and causally wrong, and this section supersedes it.

Two consequences, both negative for the other candidate fix:

* **Renewing `repeat_kv` after the rotation wins almost nothing.** Shipping 8 heads instead of
  32 cuts the wire payload 4x — from 266 MB to 66 MB per round — and the payload does not
  matter. Measured on the engine: **1.273x** (target-only) against **1.240x** for the
  single-chunk change alone, i.e. **+2.7%**. It is still worth keeping, because it is
  bit-identical and free (see F.4), but it is not a fix and its justification is now "less
  local copy work", not "fewer bytes on the wire".
* **Removing rounds is the whole prize.** 256 rounds x ~10 ms = 2.6 s of a 3.3-3.7 s step,
  which is where the decode time actually lives.

### F.4 The engine arms: one model load, four configurations, three of them bit-identical

`scripts/mlsys_df_arms.py` loads the 8B target and the 1B draft **once** and flips the ring
knobs between generations, so every arm is measured against an in-session baseline on the same
box, same book, same seed. Twelve rows, all `ok`, at 128k (`pg19_train_0`, seed 42, FP4 weights
+ NF4 KV, the branch-C precision):

| arm | what it changes | target-only decode s/tok | spec decode s/tok | greedy decode s/tok | target speedup |
|---|---|---|---|---|---|
| `ring2048` | the published config (baseline) | 3.5637 | 2.7910 | 3.5913 | 1.000x |
| `ring32768` | `kv_block_size` 32768 — one chunk per rotation | 2.8749 | 2.3074 | 2.7937 | **1.240x** |
| `ring32768x` | the above + `repeat_kv` after the rotation | 2.7985 | 2.2004 | 2.9077 | **1.273x** |
| `splitkv` | `ring_decode_mode: split_kv` — no rotation at all | **1.0011** | **0.8254** | **1.0026** | **3.560x** |

Peak reserved memory is flat across the arms — 11,792 MB for the three ring arms and
11,939 MB for split-KV, worst rank over all eight — so the 3.56x costs no memory.

**Bit-identity, checked rather than assumed.** The token streams are compared by sha256 of the
generated ids, per mode:

* `ring32768` vs `ring2048`: **IDENTICAL** in target, spec and greedy.
* `ring32768x` vs `ring2048`: **IDENTICAL** in target, spec and greedy.

So both changes that were supposed to be exact are exact at 128k on eight ranks, in all three
decoding modes. (This is also the property the `_reseed` fix in F.1 makes meaningful: without
it, an identity here could have been an artefact of arm order.)

**split-KV diverges, and the divergence is a near-tie.** The engine records its own top-2
logit margin per emitted token (`metrics.token_gaps`), which is what makes the divergence
readable rather than merely present:

| mode | first divergence | baseline's gap there | that row's gap distribution | reading |
|---|---|---|---|---|
| target | index 12 of 16, after 12 identical | **0.0000** | median 2.9375, min 0.0000 | exact tie broken the other way |
| greedy | index 12 of 16, after 12 identical | **0.0000** | median 2.9375, min 0.0000 | same, with no sampling in the way |
| spec | index 1 of 32, after 1 identical | 0.8750 | median 0.9375, min 0.0000 | below the row's own median |

A merge that was structurally wrong would diverge at the first token; this one reproduces the
ring baseline **bit-for-bit for twelve consecutive tokens** and then diverges exactly where the
baseline's own margin is zero. That is the signature of reordered floating-point summation, and
it is what the unit tests assert independently against a single attention over the concatenated
shards.

**The requested perplexity gate cannot see this change, and says so.** The brief asks for a
1024-token ppl within 1% of the ring path. All four arms return **1.648977** — identical to six
decimal places. That is not strong evidence of correctness, it is *no* evidence:
`_measure_target_ppl` calls `model.model(...)` with `use_cache=False`, i.e. a **prefill**
forward, and a decode-path change cannot affect it. The number is reported because it was
asked for; the evidence that carries the gate is the divergence table above plus the unit
tests. The same caveat is recorded in `configs/mlsys_df_followup.yml` and in the pod-side log.

### F.5 What the arms say about the two remaining candidate fixes

| candidate | correctness | speed at 128k | keep? |
|---|---|---|---|
| `kv_block_size` 32768 (one chunk) | bit-identical, all modes | 1.240x target | **yes** — config-only, exact |
| `repeat_kv` after the rotation | bit-identical, all modes | +2.7% on top | **yes** — exact, free, but not a fix |
| NCCL env change | n/a — the transport is already `P2P/CUMEM` over NV12 | none available | **no such change exists** |
| **split-KV (no rotation)** | near-tie divergence, unit-tested, ppl n/a by construction | **3.560x target, 3.382x spec** | **yes — this is the fix** |

The 3.56x is measured with 16 decode tokens, which **understates** it: the first steps of any
arm carry warm-up (allocator growth, NCCL channel setup) and the published 128-token rows
amortise it. The published 128k baseline decodes at 3.58 s/token over 128 tokens; this
session's 16-token baseline is 3.56 s/token, i.e. the arms are comparable to each other and
the ratios between them are if anything conservative.

---

## G. The decision and the rerun

### G.1 The decision (Part 2), committed before anything was launched

`results/mlsys/RERUN_DECISION.md`, commit `bff25d6`.

| quantity | value |
|---|---|
| baseline `ring2048` target-only decode s/tok at 128k | 3.5637 |
| candidate `splitkv` target-only decode s/tok at 128k | 1.0011 |
| **speedup = baseline / candidate** | **3.560x** |
| gate | >= 1.50x — **PASS**, so Part 3 ran |

The candidate engine config, exactly: `ring_decode_mode: split_kv`, `kv_block_size: 32768`,
`ring_expand_after_rotation: true`, nothing else changed from Session A. The last two are
**inert** under split-KV — there is no rotation to chunk or to expand on receipt — and their
measured 1.240x and +2.7% belong to the ring path, which the candidate does not use. They are
kept because they are exact and free, and because the fallback if split-KV is ever withdrawn is
then a better engine than Session A's.

The `split_kv` correctness gate, all three conditions: the 6 unit tests pass; the 128k
divergence is at a **near-tie** (target-only and greedy both reproduce the ring bit-for-bit for
twelve tokens, then diverge where the baseline's own top-2 margin is **0.0000** against a row
median of 2.9375; spec diverges at index 1 with a margin of 0.875 against a median of 0.9375);
and the requested 1024-token perplexity is within 1% — **vacuously so**, because
`_measure_target_ppl` runs a prefill forward that a decode-path change cannot reach. All four
arms return **1.648977**. That caveat was recorded in the decision document before the rerun,
and is repeated in F.4, so the ppl number is not read as stronger than it is.

### G.2 The rerun (Part 3)

Two 8x A100-80GB SXM4 sessions in parallel, **31 ok rows**, all stages rc=0, **$113.06**. R1
(seed 42) ran the gate pair at 128k, then 1M spec on five books, 1M target-only on two, 512k
aligned (spec 3 / target 1) and 128k aligned (spec 3 / target 1). R2 (seeds 123 and 456) ran 1M
spec on five books for each seed. Every row records `split_kv` / 32768 / expand=true /
`ring_layers_patched: 32` / its commit.

| context | arm | candidate | old | speedup |
|---|---|---|---|---|
| 128k | spec | 0.5731 | 1.3831 | **2.41x** |
| 128k | target-only | 1.1312 | 3.3665 | **2.98x** |
| 512k | spec | 0.9384 | 3.5227 | **3.75x** |
| 512k | target-only | 2.3460 | 9.9900 | **4.26x** |
| 1M | spec | 1.4264 | 5.7519 | **4.03x** |
| 1M | target-only | 3.2107 | 17.0902 | **5.32x** |

**The speedup grows with context**, which is the prediction F.3 makes and did not test: the
rotation cost a fixed ~10 ms per round, so its share of a step shrinks with context while the
work that remains grows — the longer the context, the more the ring was costing relative to
what is left. The figures, the acceptance discussion, the seed-spread finding, the periodicity
check and the TTFT caveat are in `RESULTS_OCT.md` section 10.

### G.3 What this run cost, and what it leaves open

Part 1 $11.31 of $35, Part 3 $113.06 of $200, ~$2.31 of 1x rehearsals: **$126.68 of the $250
ceiling**, cumulative ~$840 of the $1,000 cap. 0 instances, no timers, both instances
terminated in the API.

Four defects were caught by rehearsing rather than by reasoning, and three of them would have
cost the run (F.1 and F.2, plus the pod-side `git_commit`): the arms driver needed an explicit
per-arm RNG reseed, `nohup VAR=val` is not a command line, the parameter contract was
unverifiable in both directions at once, and the pod has no `.git`. The general lesson worth
keeping is defect 4's: **any in-process A/B of this engine through a sampling path needs an
explicit reseed**, because `torch.manual_seed` runs once per engine and a comparison that skips
it has arm order confounded with the treatment.

Left open, in the order I would take them:

1. **The 512k and 1M target-only denominators are thin** (1 and 2 rows). The spec arms have 3
   and 15. A reviewer who wants the 5.32x at 1M on five books needs more target-only rows.
2. **Acceptance behaviour under the new kernel is not characterised.** It falls 10.4% at 1M and
   is flat at 128k. The explanation offered (a flipped near-tie cascading through the draft's
   proposals) is consistent with every measurement here but was not tested directly — the test
   would be to hold the emitted prefix fixed and re-measure the acceptance decision.
3. **The 19.3% elementwise term from section C is now the largest remaining non-attention
   cost**, and it was never attributed. With the rotation gone it is a much bigger share of
   what is left.
4. **The 1M prefill (335 s) now dominates a 1M row** — 570 s total of which 335 s is prefill.
   The decode work that this run optimised is no longer the long pole at 1M.

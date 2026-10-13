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

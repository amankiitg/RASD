# The bandwidth floor for target-only decode, and where the time actually goes

Local analysis, 2026-10-12, written while the 8x sampling session ran. **No engine code
was changed**: any change would invalidate every timing already measured. Findings only.

Everything below is a *lower bound*: it counts bytes that must move and divides by the
hardware's peak rate. It assumes perfect overlap, zero latency, no dequantization cost and
no Python. A real engine is above its floor, and the ratio is the finding.

---

## 1. Assumptions (every one of them)

| # | Assumption | Value used | Where it comes from |
|---|---|---|---|
| A1 | Target weights | 8.03e9 parameters | Llama-3.1-8B / Gradient-1048k class |
| A2 | Weight storage | NF4, `0.5 + 4/block_size` bytes/element | `src/models/nf4_dynamic_cache.py:9` |
| A3 | Weight block size | 64 → 0.5625 B/elem → **4.52 GB** | that docstring's default; see A8 note |
| A4 | Weight replication | **full copy on every rank** (not sharded) | ring shards the SEQUENCE, not the weights; consistent with the measured `post_load` 6,775.8 MB at 256k |
| A5 | KV storage | NF4, `kv_block_size_nf4 = 32` → 0.625 B/elem | `src/models/rasd_inference.py:293` (`kv_block_size_nf4: int = 32`) |
| A6 | KV shape | 32 layers × 8 kv_heads × 128 head_dim × 2 (K and V) = 65,536 values/token | Llama-3.1-8B geometry |
| A7 | **KV bytes per token, total** | 65,536 × 0.625 = **40,960 B = 40 KiB** | this reproduces the stated 40 KiB/token exactly |
| A8 | KV sharding | sequence sharded 8 ways → 5 KiB/token/rank *held*; a rank *reads* all 40 KiB/token when it attends across the ring | §3 |
| A9 | A100 80GB SXM4 HBM | 2,039 GB/s | datasheet |
| A10 | A100 SXM4 NVLink 3 | 600 GB/s bidirectional aggregate = **300 GB/s per direction** | datasheet |
| A11 | Ring payload dtype | **bf16**, not NF4 | `_ring_forward` sends `k_cur`/`v_cur` derived from `self.qkv(x)`, before any NF4 codec — §4 |
| A12 | Ring rotation count | `world_size` = 8 steps per layer per forward | `for step in range(world_size)` |
| A13 | Transmission chunking | `chunk_size = kv_block_size` = 2,048 tokens | `rasd_inference.py:1273` |
| A14 | Contexts | 128k / 512k / 1M tokens = 131,072 / 524,288 / 1,048,576 | session configs |

**A8 is the assumption with the most leverage, and it is the one to check first.** If each
rank only ever touched its own shard, per-rank KV traffic would be 5 KiB/token and the KV
term would be negligible. Ring attention requires the opposite: every rank must attend to
every token, so every rank's bytes cross the ring (A11-A13).

---

## 2. The floor, per token, per rank

**Weights.** 4.52 GB must be read once per token: `4.52 / 2039 GB/s = 2.22 ms`.
At NF4 block 32 (A5 style) it is 5.02 GB → 2.46 ms. **~2.2-2.5 ms.**

**KV read from HBM.** A rank reads its own shard once: `S_local × 5 KiB`.
At 1M: 131,072 × 5 KiB = 671 MB → **0.33 ms**. Negligible.

**Ring communication** (the dominant term). Per layer, the ring sends and receives
`S_local × 8 heads × 128 dim × 2 B` per tensor, for K and V, at each of 8 steps:

| context | S_local per rank | K or V tensor | bytes sent per layer | bytes sent per token (32 layers) | floor at 300 GB/s |
|---|---|---|---|---|---|
| 128k | 16,384 | 33.6 MB | 537 MB | 17.2 GB | **57 ms** |
| 512k | 65,536 | 134.2 MB | 2.15 GB | 68.7 GB | **229 ms** |
| 1M | 131,072 | 268.4 MB | 4.29 GB | 137.4 GB | **458 ms** |

Send and receive are concurrent on a duplex link, so the floor is the larger direction
(458 ms at 1M), not their sum.

**Total floor:** 128k ≈ **59 ms**, 512k ≈ **232 ms**, 1M ≈ **461 ms** per token, and the
ring is 97-99% of it at every context.

---

## 3. Measured time per token, and the ratio

Measured as `time_sec / tokens_generated`, never the rounded `throughput_tps` field.
Sources: the 2026-10-10/11 greedy session, `gradientai/...-Gradient-1048k` target-only,
8x A100 80GB SXM4, NF4 KV, `kv_block_size` 2048.

| context | measured s/token | measured tok/s | floor ms/token | **ratio** | ring chunk transfers per token | measured ms **per transfer** |
|---|---|---|---|---|---|---|
| 128k | 3.51 (sanity target-only, 224.7 s / 64 tok) | 0.285 | 59 | **59x** | 4,096 | 0.86 |
| 512k | 9.98 (1277.0 s / 128 tok) | 0.100 | 232 | **43x** | 16,384 | 0.61 |
| 1M | 19.10 (2444.8 s / 128 tok) | 0.052 | 461 | **42x** | 32,768 | 0.58 |

**The last column is the finding.** The measured time per token is very nearly *linear in
the number of ring chunk transfers*, with a slope of **0.58-0.86 ms per transfer** across a
factor of 8 in context. A 2.1 MB K-chunk moved in 0.58 ms is **3.6 GB/s**, roughly 80x
below what the link can do per direction. So the engine is not moving bytes slowly because
the bytes are large; it is paying a large fixed cost per transfer.

---

## 4. Likely causes of the gap, with the code

**C1 — the ring is chunked, so one layer's rotation is many transfers.**

```python
# src/models/rasd_inference.py:1273
            chunk_size=cfg.kv_block_size,
```
with `kv_block_size: int = 512` (`:141`) and 2048 in the session configs. The payload per
layer is the whole local shard (268 MB at 1M), split into 2,048-token chunks: **64 chunks
per tensor, 128 per step, 1,024 per layer, 32,768 per token.**

**C2 — every transfer is waited on individually, in Python, on the critical path.**

```python
# src/models/ring_attention_flash.py, _ring_forward
            if step < world_size - 1:
                ops = [
                    dist.P2POp(dist.isend, k_cur, send_to),
                    dist.P2POp(dist.isend, v_cur, send_to),
                    dist.P2POp(dist.irecv, k_buf, recv_from),
                    dist.P2POp(dist.irecv, v_buf, recv_from),
                ]
                reqs = dist.batch_isend_irecv(ops)
                for r in reqs:
                    r.wait()
                k_cur, k_buf = k_buf, k_cur
                v_cur, v_buf = v_buf, v_cur
```
`batch_isend_irecv` posts atomically (the comment says it exists to avoid an NCCL
eager-mode deadlock), but then **all four requests are waited on immediately** before the
next step can begin. There is no overlap between the transfer and the attention that
follows it, and no pipelining across the 64 chunks. `prefetch_depth` exists in the config
(`prefetch_depth: 1`) but the loop above has no double-buffered prefetch: `k_buf`/`v_buf`
are swapped only after the waits complete.

**C3 — attention is recomputed per ring step against a rotating KV, so each rank performs
`world_size` attention passes per layer** rather than one pass over its own shard:

```python
        for step in range(world_size):
            if _FLASH_AVAILABLE:
                out_step, lse_step = _flash_step(q_fa, k_fa, v_fa, ...)
                out_acc, lse_acc = _combine(out_acc, lse_acc, out_step, lse_step)
```
This is the correctness requirement of ring attention, not a bug — but it means the work
per layer is `world_size` attention calls plus `world_size × 64 × 2` transfers, all on the
critical path.

**C4 — the KV is dequantized on demand, so a decode step touches far more than the bytes it
attends to.**

```python
# src/models/nf4_dynamic_cache.py:5
dequantize on demand; the rest of the model never sees the NF4 form.
```
and at `:318-322` the read path calls `dequantize_nf4(...)`. If the whole cached KV is
dequantized each step, the traffic is 40 GB read + 128 GB written back as bf16 at 1M
(≈82 ms) — small against the ring term, but it doubles the HBM pressure and adds a
kernel-launch burst per layer.

**C5 — per-step synchronizations.** Beyond the explicit `r.wait()`, the decode loop
broadcasts logits and round state between ranks (`:924-926`, `:2044-2045`) and this is
documented as "~1 MB broadcast per round. Negligible" *in bytes*. In latency it is one more
blocking collective per round, which matters only if the round is otherwise fast — at 19 s
it is not.

**What the numbers already rule out.** Not bandwidth (bytes/rate is 42-59x below
measured), and not attention FLOPs: a decode step at 1M is ~137 GFLOP of attention per
rank, ~0.44 ms at bf16 dense rates. The gap is per-transfer overhead, which is what the
profiles in `PROFILE_PLAN.md` are designed to confirm or refute.

---

## 5. What would close the gap (findings only, no engine changes)

1. **Fewer, larger transfers.** The measured slope is per *transfer*, not per byte, so
   raising the chunk size from 2,048 tokens to the whole shard should cut the count by 64x
   at 1M. Expected effect: if the slope is really fixed cost, 1M decode moves from ~19 s
   toward the byte floor (~0.5 s) — a 38x change on the target-only path. This is the single
   largest lever and the first thing to test.
2. **Overlap transfer with compute** (double-buffered prefetch, wait only before use),
   which the current `wait()`-per-step structure prevents by construction.
3. **Keep the ring payload in NF4** instead of bf16 (A11): 40 KiB/token instead of
   128 KiB/token, an 3.2x reduction in ring bytes. Only worth it if bytes, not latency,
   turn out to dominate — which this analysis says they currently do not.

The honest caveat: these are arithmetic and code-reading conclusions. The profile decides.

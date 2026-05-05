# M3 — Real Ring Attention Integration Plan

Tracking file for finishing the ring-attention/spec-decoding integration that was
deferred during M3. Created 2026-05-05 after the Check-4 audit surfaced the gap.

## Why this exists

During the post-stream-race audit (2026-05-05) we discovered that the verify
forward in [src/models/rasd_inference.py:659-663](src/models/rasd_inference.py#L659-L663)
passes the **local-only** `past_kv` directly to `self.target_model(...)`. The
ring prefetcher receives KV blocks into `pending_block.keys/values` but those
buffers are **never merged back into the attention input**. In other words:

- `world_size=8` today: each rank holds the full N-token KV locally and runs
  the full target verify against its own copy. Ring P2P runs but is decorative.
- `kv_block_size` (A3) and `prefetch_depth` (A4) ablation axes are not
  measuring what the paper claims they measure.
- Memory: at 64k context, target KV is ~33 GB per rank — fits 80 GB but
  defeats the purpose of sharding.

This contradicts the M3 mentor ask ("Blockwise FA + ring attention") in
[M4_PLAN.md](M4_PLAN.md#L9). The kernel exists at
[src/models/ring_attention_flash.py](src/models/ring_attention_flash.py) but
is a **standalone `nn.Module`** with its own QKV projections — it has never
been patched into Llama's transformer block.

## Decision

Pursue **option (c)** — finish the integration. Re-ablation at 64k×8 is
deferred until ring is actually wired. Single-GPU 64k results would be
publishable on their own but don't tell the M3 story; the paper's whole
contribution is sequence-parallel ring + spec decoding.

## Architecture target

Sequence-parallel KV. Each of `world_size` ranks holds an equal slice of the
sequence dimension:

```
Rank 0: positions [0,        N/W)
Rank 1: positions [N/W,      2N/W)
...
Rank W-1: positions [(W-1)N/W, N)
```

Where `W = world_size` and `N` = context length. Per-rank KV memory drops by
factor `W` (at 64k×8: 33 GB → ~4 GB target KV per rank).

**Attention forward (per layer):**
- Each rank holds full Q (`k+1` tokens during verify, 1 token during draft)
  and its local K/V slice.
- Ring step `s = 0..W-1`:
  1. Compute partial attention `O_s, lse_s = flash_attn(Q, K_current, V_current)`
     using the FA-2 varlen API that returns log-sum-exp.
  2. Online-softmax combine `(O_s, lse_s)` into running `(O, lse)`.
  3. P2P-rotate `K_current, V_current` to next rank via `batch_isend_irecv`.
- After W steps, `O` is the full attention output.

The math (Liu et al. RingAttention 2023, Korthikanti et al. SeqPar) is exactly
what `_local_forward` in `ring_attention_flash.py` does today for `S` tokens
across one rank — generalised to W rotation steps.

**KV append after verify:** the `k+1` new positions produced by each verify
round must be appended to *exactly one* rank. Round-robin keeps slices balanced
(token at position `p` goes to rank `p mod W`). This means each verify round
appends `(k+1)/W` tokens per rank on average, with the host rank alternating.

## Task list

### Phase R0 — Design lock (locked 2026-05-05)

- [x] **R0.1 — KV layout: contiguous slices** (revised 2026-05-05 during R1
  reading). Rank `r` holds positions `[r*N/W, (r+1)*N/W)`. Reason for the
  flip from round-robin: causal masking under contiguous layout has the
  standard literature pattern (skip future-rank steps, full-attend past-rank
  steps, FA's built-in `causal=True` flag at the self-step). Round-robin
  causal masking interleaves positions and breaks per-step bulk masking —
  would require either custom CUDA or per-position attention-bias tensors,
  doubling implementation cost. The 256 generated tokens flowing to the
  last rank produces ~3% memory imbalance at 64k base — negligible. New
  decode tokens are appended to the rank whose contiguous range covers
  `current_global_seqlen`, which is rank `W-1` until that slice is full,
  then the model's seqlen has exceeded the prefilled budget (we don't go
  there in this study).
- [x] **R0.2 — Draft model: replicated per rank.** Draft is 1.3B params,
  ~2.6 GB bf16 — cheap to duplicate × 8. Avoids the sequential bottleneck of
  rank-0-only drafting and keeps speculative determinism trivially aligned
  across ranks (same seed → same draft tokens on every rank).
- [x] **R0.3 — Draft attention: local (no ring), full KV replicated.**
  Each rank holds the draft's full KV cache. 64k × 1.3B → ~6 GB/rank, fits
  comfortably. M4 1M-context will need ring on draft too; deferred until then.
- [x] **R0.4 — Sequence diagram below.**

#### Sequence diagram — one verify round, W=4 ring step

```
                      Ring topology: rank r sends to (r+1) % W,
                                     receives from (r-1) % W

Initial KV layout (contiguous, prefill of N=16 positions, W=4):
  Rank 0 holds positions [0,  4)   = {0, 1, 2, 3}
  Rank 1 holds positions [4,  8)   = {4, 5, 6, 7}
  Rank 2 holds positions [8, 12)   = {8, 9, 10, 11}
  Rank 3 holds positions [12, 16)  = {12, 13, 14, 15}

Causal masking per ring step (rank r reads K from source rank sr=(r-s)%W):
  sr <  r : full attend (past rank, all K positions < all Q positions)
  sr == r : FA-2 causal=True at the self-step (within-slice diagonal mask)
  sr >  r : skip — Q at rank r cannot see future positions on rank sr

Each rank holds Q for ALL its query tokens during verify (k+1 of them,
broadcast/replicated). Each rank holds only its K/V slice.

Ring attention forward (per Llama layer):

  step s=0  rank r reads its OWN K,V slice
            ┌───────────────────────────────────┐
            │ O_partial, lse_partial =          │
            │   flash_attn(Q, K_local, V_local) │  ← FA-2 with absolute
            │ O, lse = combine(O_partial, lse_  │    pos info for causal
            │                  partial, +∞)     │    masking
            └───────────────────────────────────┘
            Concurrent: post P2P send K,V → next rank
                        post P2P recv K,V ← prev rank

  step s=1  rank r now sees K,V from rank (r-1) mod W
            (waits on P2P recv to complete)
            O_partial_1 = flash_attn(Q, K_recv, V_recv)
            O, lse = combine(O, lse, O_partial_1, lse_partial_1)
            Concurrent: rotate again

  step s=2..W-1  same pattern

  After W steps: O is the full attention output. K_local/V_local has
  rotated all the way around the ring and is back to its original owner.

After full forward of all layers: target_logits_v ready on each rank
(replicated, since Q was replicated and we accumulated full O).

Accept/reject: runs identically on every rank (deterministic given
same RNG state). No further comm needed.

KV append after acceptance:
  n_acc + 1 new tokens land at absolute positions
    [prior_global_seqlen, prior_global_seqlen + n_acc + 1)
  Under contiguous layout, all new tokens go to the rank whose range
  contains those positions. For a 64k prefill on W=8, that's rank 7
  (positions [56k, 64k)) until decode pushes seqlen past 64k. We cap
  at max_new_tokens=256 so global seqlen ≤ 64k+256 — all decoded
  tokens go to rank 7. Other ranks' KV stays static during decode.
```

Key correctness invariants captured here:

1. Q is replicated; K/V is sharded. Output O is replicated after all ring
   steps complete (same value on every rank up to bf16 noise).
2. Causal masking uses **absolute positions**, not slice-local indices —
   each ring step provides (Q_pos, K_pos) tensors to the FA-2 kernel.
3. The W ring steps inside the attention forward replace the
   `AsyncKVRingPrefetcher`'s prefetch-during-draft pattern. Streams
   `stream_comm` may collapse into the attention forward's own usage.
4. Draft model runs entirely locally on each rank with full replicated
   KV — no comm during draft phase.

### Phase R1 — Ring attention forward, single-layer correctness ✅ done 2026-05-05

- [x] **R1.1** Extracted [src/models/ring_attention_kernel.py](src/models/ring_attention_kernel.py)
  with `ring_attention_prefill` and `ring_attention_decode` free functions.
  Pre-projected Q/K/V interface, FA-2 + SDPA dispatch, online-softmax merge.
- [x] **R1.2** [tests/test_ring_attention.py](tests/test_ring_attention.py) — 8/8 green.
  Single-process math (4 tests) plus real multi-process gloo (4 tests) at
  W∈{2,4} for both prefill and decode.
- [x] **R1.3** Numerical tolerance: single-rank kernel matches reference bit-exactly
  (atol < 1e-5 fp32). bf16 numerical tolerance check deferred to R6 (CUDA-only).

### Phase R2 — Patch into Llama target attention ✅ done 2026-05-05

- [x] **R2.1** Monkey-patch `LlamaAttention.forward` per instance via
  [src/models/ring_llama_attention.py](src/models/ring_llama_attention.py).
  `world_size==1` falls through to original forward (preserves single-rank
  exactness). Cache helpers handle DynamicCache and legacy tuple.
- [x] **R2.2** `install_ring_attention(target_model, world_size, rank)` walks
  `target_model.model.layers` and replaces `.self_attn.forward`. Will be
  called from `_load_models` during R3.
- [x] **R2.3** Plumbing tests in [tests/test_ring_llama_attention.py](tests/test_ring_llama_attention.py) —
  10/10 green. Multi-rank correctness on real Llama gated on R6 pod time
  (transformers==4.44.2 not installed locally).

### Phase R3 — KV cache layout + spec integration (merged with R4)

R3 and R4 are now one phase — `_extract_kv_block` and `_alloc_kv_buffers`
exist solely to feed `AsyncKVRingPrefetcher`, so the prefetcher deletion
and the layout refactor belong in the same commit.

**Two preserved invariants throughout R3:**

1. **`world_size=1` byte-equivalence.** The current single-rank verify path
   has α≈0.4 NF4 / ≈0.7 bf16 and 6/6 verify-math tests green. R3 must not
   regress this — every multi-rank code branch is gated on `world_size > 1`,
   and `tests/test_verification_math.py` stays as the regression gate.
2. **Driver-agnostic sharding.** Each rank receives the full prompt and
   slices internally inside `_prefill`. `run_experiment.py` does not need
   to know about sharding — the same `engine.generate_text(prompt)` call
   works at world_size 1 or 8.

**Tasks:**

- [ ] **R3.1** Wire `install_ring_attention(target_model, world_size, rank)`
  into [`_load_models`](src/models/rasd_inference.py#L364) right after
  `target_model = AutoModelForCausalLM.from_pretrained(...)`. No-op when
  `world_size <= 1`.
- [ ] **R3.2** Refactor `_prefill` to slice `input_ids` per rank:
  `local_ids = input_ids[:, rank*S/W : (rank+1)*S/W]`,
  `position_ids = arange(rank*S/W, (rank+1)*S/W)`. Pass these to
  `target_model(...)`. Draft model continues to receive full `input_ids`
  (replicated KV per R0.3).
- [ ] **R3.3** Update `_truncate_kv` for the contiguous layout: only the
  rank whose slice owns the just-appended tail truncates
  `(k - 1 - n_acc)` positions. With `new_kv_owner_rank = world_size-1`,
  that's just rank W-1; other ranks' caches are unchanged during decode.
- [ ] **R3.4** Delete `AsyncKVRingPrefetcher` (the entire class) and its
  invocation block at [generate_text:601-635](src/models/rasd_inference.py#L601-L635).
  Ring rotation now lives inside `LlamaAttention.forward`, not as a separate
  prefetch loop.
- [ ] **R3.5** Delete `_extract_kv_block` and `_alloc_kv_buffers` — only
  the prefetcher used them.
- [ ] **R3.6** Re-evaluate the three CUDA streams. With ring-inside-attention,
  `stream_comm` no longer has work to do — it can be removed. `stream_compute`
  and `stream_draft` stay (they cover the parallel target verify vs. draft
  generation pipeline). Update the stream-ordering invariants in
  `feedback_spec_verify_fix.md` once R3 lands.
- [ ] **R3.7** Tests added in the same commit:
  - `test_local_slice_indexing` — verify the `_prefill` slice math
    (positions, edge cases at world_size that doesn't divide S).
  - `test_truncate_kv_owner_only` — `_truncate_kv` only modifies rank W-1's
    cache during decode partial rejection.
- [ ] **R3.8** Run `tests/test_verification_math.py` (6 tests) on the
  ring-integrated path. The math invariants (packed verify, residual resample,
  KV truncation) must still hold; only the attention computation changes.

### Phase R5 — Multi-rank stream audit (1 day)

- [ ] **R5.1** With ring inside attention, audit the remaining stream
  interactions: `stream_compute` (verify forward including its internal P2P),
  `stream_draft` (draft forward, no ring), `default` (accept/reject). Ensure
  the verify forward's output is committed before accept/reject reads it
  (the existing `current_stream().wait_stream(stream_compute)` should still
  cover this).
- [ ] **R5.2** Run a 2-rank smoke at 8k context, confirm α matches
  single-rank baseline within bootstrap CIs.

### Phase R6 — Validation matrix (2-3 days, mostly pod time)

- [ ] **R6.1** Single-rank 8k smoke (regression check vs current main).
- [ ] **R6.2** 2-rank 8k smoke. α vs single-rank: should be statistically
  indistinguishable.
- [ ] **R6.3** 8-rank 8k smoke. Same α target.
- [ ] **R6.4** 8-rank 64k smoke (1 prompt, 64 tokens). Memory check:
  per-rank target KV should be ~4 GB, not ~33 GB.
- [ ] **R6.5** Full 49-row re-ablation at 64k×8. Now A3 (`kv_block_size`)
  and A4 (`prefetch_depth`) actually mean something — ring step granularity
  and pipeline depth.

## Open design questions

1. ~~**Causal masking with round-robin layout.**~~ Resolved during R1
   reading (2026-05-05): flipped R0.1 to contiguous slices. Round-robin's
   per-position causal masking would have required custom CUDA. Contiguous
   uses the standard literature pattern (skip future-rank, full-attend
   past-rank, FA-2 `causal=True` at self-step). See R0.1 above for full
   rationale.

1a. **A3/A4 ablation axes after R3.4 deletes the prefetcher.** With ring
   rotation now inside `LlamaAttention.forward`, `kv_block_size` (A3) and
   `prefetch_depth` (A4) lose their original meaning. Two options for
   the re-ablation: (a) drop A3/A4 entirely from the 49-row grid (down to
   ~30 rows of A1/A2/A5), (b) redefine A3 as ring step micro-batching and
   A4 as compute/comm overlap depth and re-design the experiments. Decide
   before R6.5 launches; not a blocker for R3.

2. **Draft KV growth.** If draft is replicated, its KV grows linearly per
   rank (no sharding). At 64k that's 4 GB/rank — fine. At 1M (M4 target),
   that's 64 GB/rank — won't fit. Will need ring on draft too for M4.
3. **GQA/MQA for Llama-2-7b.** Llama-2-7b uses standard MHA (num_kv_heads=32).
   No GQA complications. Llama-3 would need head-group-aware ring.

## Testing discipline

Every code change in phases R1–R5 lands with tests in the same commit.
No phase is marked done until its tests are green locally.

Mapping of code → tests:

| Code | Tests |
|---|---|
| [src/models/ring_attention_kernel.py](src/models/ring_attention_kernel.py) | [tests/test_ring_attention.py](tests/test_ring_attention.py) — single-process math + multi-process gloo at W∈{2,4} |
| [src/models/ring_llama_attention.py](src/models/ring_llama_attention.py) | [tests/test_ring_llama_attention.py](tests/test_ring_llama_attention.py) — installer + cache-helper plumbing |
| Verify path in [src/models/rasd_inference.py](src/models/rasd_inference.py) | [tests/test_verification_math.py](tests/test_verification_math.py) — six locked invariants from feedback_spec_verify_fix.md |
| Future R3/R4 changes to KV layout / spec integration | extend the existing test files; do not bypass |

What CI cannot cover (and is therefore explicitly gated on R6 pod time):

- Real `LlamaAttention` integration (transformers 4.44.2 only on pod).
- Multi-rank correctness with actual model weights vs. mocked tensors.
- CUDA stream timing under bf16/NF4 (the bug class that fooled M3).
- End-to-end α and tps numbers vs. the M3 buggy baseline.

For these, the discipline is: **smoke first, sweep second.** Never launch a
49-row sweep before the corresponding smoke (single rank, then multi-rank,
then 64k) has produced expected α and memory profile.

## Validation strategy

Three layers, in order:

1. **Unit tests (fast):** ring vs reference attention, single-rank determinism.
   `tests/test_ring_attention.py`. Runs on CPU with gloo backend.
2. **Integration tests (CUDA, slow):** target output equivalence between
   `world_size=1` and `world_size=8` on the same prompt+seed.
3. **End-to-end (pod):** spec-decoding α match between single-rank and
   multi-rank at 8k context. Memory profile match at 64k.

## Done criteria

This work is complete when:

- `tests/test_ring_attention.py` passes for W in {1, 2, 4, 8}.
- `tests/test_verification_math.py` (existing 6 tests) still passes.
- An 8-rank 64k smoke shows per-rank target KV memory ≈ 4 GB (not 33 GB).
- 49-row ablation re-runs cleanly with A3/A4 axes producing distinguishable
  results (i.e., kv_block_size and prefetch_depth actually affect throughput).

## What's deferred until this lands

- 49-row re-ablation (was todo #10 — paused).
- New wandb project `rasd-m3-reablation-64k` setup (will create when launching
  the re-ablation, not before).
- Lambda 8× A100 capacity hunting (no point until R6.4 passes).

The single-GPU smoke (todos #4-7) can still proceed if you want a
sanity-check that the RoPE+stream-fix changes didn't regress single-rank
behavior. Otherwise it's strictly optional.

## Effort estimate

| Phase | Days | Status |
|---|---|---|
| R0 design lock | 1 | ✅ done 2026-05-05 |
| R1 kernel correctness | 2-3 | ✅ done 2026-05-05 (ring_attention_kernel.py + 8 tests) |
| R2 Llama patch | 2-3 | ✅ done 2026-05-05 (ring_llama_attention.py + 10 tests) |
| R3+R4 KV layout + prefetcher removal | 2-3 | merged; in progress |
| R5 stream audit | 1 | smaller now ring is in-layer |
| R6 validation matrix | 2-3 | mostly pod cost |
| **Total** | **10-15 days** | plus ~$100 pod budget for R6 |

## References

- [src/models/ring_attention_flash.py](src/models/ring_attention_flash.py) —
  existing standalone kernel
- [src/models/rasd_inference.py:659-663](src/models/rasd_inference.py#L659-L663) —
  current verify forward (local-only `past_kv`)
- [analysis/m3_post_analysis_plan.md](analysis/m3_post_analysis_plan.md) —
  prior audit context
- [tests/test_verification_math.py](tests/test_verification_math.py) —
  spec-decoding math invariants (must remain green)
- [memory/feedback_spec_verify_fix.md](.claude/projects/-Users-amankesarwani-PycharmProjects-RASD/memory/feedback_spec_verify_fix.md) —
  the four invariants we already locked

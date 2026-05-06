# M3 — Real Ring Attention Integration Plan

Tracking file for finishing the ring-attention/spec-decoding integration that was
deferred during M3. Created 2026-05-05 after the Check-4 audit surfaced the gap.

## Status snapshot (2026-05-05, end of day)

**Phases R0–R3.5 complete. 76/76 tests green locally. Pushed to main on three
commits: c7800c1 (R0–R2), 09f7d98 (R3 dual-cache integration), d45ccbc (R3.5
A3/A4 redefinition).**

Architecture target (sequence-sharded KV via ring-in-attention with online
softmax) is met in code:
- `src/models/ring_attention_kernel.py` — `ring_attention_prefill` and
  `ring_attention_decode` free functions with online-softmax merge, dual-cache
  decode (sharded prefill + replicated tail), and chunked + prefetched
  rotation.
- `src/models/ring_llama_attention.py` — `install_ring_attention` monkey-patches
  every `LlamaAttention.forward`. `set_prefill_len` freezes the prefill
  boundary after prefill so decode forwards know the sharded/tail split.
- `src/models/rasd_inference.py` — ring is fully wired into `generate()`:
  prefill slices `input_ids` per rank, broadcasts the last logit, sets
  `_ring_prefill_len`; decode passes absolute `position_ids`; the legacy
  `AsyncKVRingPrefetcher` and its helpers are deleted (~370 lines removed,
  ~70 added). Stream count went 3 → 2.
- A3/A4 ablation knobs preserved with **real** semantics in the new
  architecture: `cfg.kv_block_size` controls per-step batch_isend_irecv
  chunk size; `cfg.prefetch_depth` toggles compute/comm overlap.

What's left:
- **R5** — multi-rank stream re-audit on the integrated path + a 2-rank
  smoke. Smaller now ring is in-layer (only `stream_compute` ↔ `stream_draft`
  to coordinate; the third stream is gone).
- **R6** — Lambda pod validation matrix (1/2/8-rank smokes at 8k, 64k memory
  check, full 49-row re-ablation at 64k×8). This is where real Llama,
  real CUDA, real NCCL get tested. Budget ~$70–90.

Also-pending (deferred but not blockers):
- Enrich wandb per-round telemetry (do during R6).
- Lambda SSH key registration, persistent filesystem, instance provisioning.

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

### Phase R5 — Multi-rank stream audit ✅ done 2026-05-05

- [x] **R5.1** Audit found three iteration-boundary cross-stream races
  (default → stream_draft for `cur_token`, default → stream_compute for
  `past_kv`, default → stream_draft for `draft_past_kv` after partial
  rejection's `_truncate_kv`). Same class as Check 4's α=0.018 bug —
  small writes on default stream, fast bf16 readers on draft/compute
  streams. Fix: two cheap `wait_stream` calls at the top of each loop
  iteration:
  ```python
  self.stream_draft.wait_stream(torch.cuda.current_stream())
  self.stream_compute.wait_stream(torch.cuda.current_stream())
  ```
  No-op on the first iteration (default stream is idle post-prefill-sync).
  Top-of-file architecture docstring updated with the full 4-rule
  stream-ordering contract (down from 5+ pre-R3). Memory
  `feedback_spec_verify_fix.md` updated with the new invariants.
- [ ] **R5.2** Run a 2-rank smoke at 8k context, confirm α matches
  single-rank baseline within bootstrap CIs. **This is pod work; folded
  into R6.1/R6.2.**

### Phase R6 — Validation matrix (mostly done; R6.5 deferred)

- [x] **R6.1** Single-rank 8k smoke ✅ 2026-05-05 (NF4 α=0.654, bf16 α=0.682).
- [x] **R6.2** 2-rank 8k smoke ✅ 2026-05-06 (α=0.312, peak 28.4 GB/rank).
- [x] **R6.3** 8-rank 8k smoke ✅ 2026-05-06 (sync stable; async stable at
  max_new ≤ 16, flaky at max_new ≥ 32 — see Open Issues).
- [⚠] **R6.4** 8-rank 64k smoke — **OOM at 35.7 GB / 40 GB on the SXM2
  variant**. Sharding works as designed (target K/V correctly reduced to
  4.3 GB/rank), but other components (replicated draft KV, NF4 dequant
  temporaries, allocator fragmentation) push total per-rank usage past
  40 GB. Backed off to ctx=16k, which passed at 20.9 GB/rank ✅. 64k
  needs the 80 GB SXM4 SKU (`gpu_8x_a100_80gb_sxm4`). See
  [results/r6/r6_audit.md](results/r6/r6_audit.md) Issue #2 for the full
  per-rank memory breakdown.
- [ ] **R6.5** Full 49-row re-ablation — **deferred** until either (a) 80
  GB capacity available on Lambda OR (b) sweep is rescoped to ctx=32k
  to fit 40 GB. Also gated on Issue #1 below.

## R6 Open Issues (must address before R6.5)

### Issue #1 — Async ring (prefetch_depth=1) at W=8: status inconclusive

**A4-critical.** The A4 ablation has 3 levels: sync=0, async-1=1, async-2=2.

**What the data actually shows from the 2026-05-06 session (honest readout):**

| max_new | mode | result |
|---|---|---|
| 64 | async (fp16 target) | SIGABRT after 11 min (NCCL timeout) |
| 32 | async (NF4) | hung silently after prefill, killed by 480s timeout |
| 8 | **sync** (NF4) | ✅ works |
| 2 | async + RING_TRACE | ✅ works (single verify iter only) |
| 16 | async (NF4) | ✅ works (multiple verify iters) |
| 32 | async (NF4) | Traceback in output but my grep filter chopped it |

So sync is solid at every config tested. Async works at max_new ≤ 16,
flaky/inconclusive at max_new ≥ 32. **One failed run at max_new=32 isn't
enough evidence to say the bug is real** — could be transient NCCL issue,
env-var difference, or genuine bug at high iteration counts.

**To resolve:** re-run async at max_new ∈ {16, 32, 64, 128} × seed ∈
{42, 123, 456} on a multi-GPU pod with full stderr captured (no grep
filter). 12 runs ≈ 1 hr pod time at $15.92/hr ≈ $16. Cheap diagnostic.

**Fix options if it IS a real bug:**
1. Run R6.5 in sync-only mode → loses A4 levels 1, 2 (drops paper claim
   from 3-level to 1-level for that axis).
2. Insert `dist.barrier()` between ring rotations → eliminates async
   overlap (defeats A4=1 purpose) but unblocks correctness.
3. Deep NCCL profiling on a replacement pod.

### Issue #2 — 64k context exceeds 40 GB SXM2 per-rank budget

**Memory math correction:** plan-doc estimate of "per-rank K/V ~4 GB" was
correct in isolation but total per-rank usage at ctx=64k×W=8 NF4 is ~36 GB.
Corrected breakdown of the dominant non-target components:
- target weights replicated: ~4 GB
- **draft KV replicated per R0.3: ~12 GB at 64k** (was originally
  estimated as 5.7 GB — the underestimate was the biggest miss)
- NF4 dequant temporaries: ~3-5 GB
- activations + LM head: ~5-6 GB
- allocator fragmentation: ~12 GB

**Partial fix landed 2026-05-06 — Option B: don't RoPE-scale the draft.**

Commit (forthcoming) — `_build_hf_config(..., apply_rope_scaling=False)`
when called for the draft model. Caps the draft at its native context
(Sheared-LLaMA-1.3B = 4096) regardless of `cfg.context_length`. The
existing `draft_ids = raw_draft_ids[:, -self.draft_max_len:]` truncation
in `generate()` already feeds only the recent native_max tokens to the
draft. Per-rank draft KV at ctx=64k drops from ~12 GB → ~770 MB
(`4096 × 24 × 16 × 128 × 2 × 2`). Saves **~11 GB/rank**. New per-rank
total at ctx=64k×W=8: ~25 GB → fits 40 GB SXM2 with headroom.

Tradeoff: draft sees only the last 4k tokens of context regardless of
target ctx. Speculative decoding tolerates this asymmetry (drafts
typically have shorter native contexts than targets). May cost a small
amount of α at very long contexts but is the standard practice for
production spec-decoding setups. Option A (full draft KV sharding via
ring) is deferred to M4 1M-context where draft sharding is mandatory.

Tests added: `tests/test_rasd_components.py::TestRoPEScalingGate`
(2 tests) — regression guard that target stays scaled and draft does
not, even when ctx > native_max.

**Still needed for R6.5 at ctx=64k×W=8 on 40 GB hardware:** confirm
empirically that per-rank usage now fits in next pod session.

### Issue #3 — Region mismatch (compute ≠ filesystem)

R6.2-R6.4 ran in europe-central-1 because that's where 8x A100 capacity
appeared. Filesystem `rasd-fs` is in us-west-2 — not attachable. We ran
on ephemeral storage and scp'd results back. For R6.5, prefer launching
in us-west-2 if capacity returns there; otherwise live with cold model
loads each session and remember to scp results before terminating.

## Open design questions

1. ~~**Causal masking with round-robin layout.**~~ Resolved during R1
   reading (2026-05-05): flipped R0.1 to contiguous slices. Round-robin's
   per-position causal masking would have required custom CUDA. Contiguous
   uses the standard literature pattern (skip future-rank, full-attend
   past-rank, FA-2 `causal=True` at self-step). See R0.1 above for full
   rationale.

1a. ~~**A3/A4 ablation axes after R3.4 deletes the prefetcher.**~~ Resolved
   2026-05-05: keep both axes with redefined semantics that map onto the
   new ring-in-attention architecture. The 49-row grid stays intact.

   - **A4 — `prefetch_depth`** → ring-step prefetch depth (compute/comm
     overlap). Inside `_ring_against_sharded` and `ring_attention_prefill`,
     the loop body becomes:
       - prefetch_depth=0 (sync): wait after each rotation, then compute
         the next step.
       - prefetch_depth≥1 (async): issue rotation s+1's `batch_isend_irecv`
         BEFORE computing step s, wait at top of next iteration. One
         rotation in flight while compute proceeds.
       - prefetch_depth=2 saturates the same as 1 under the ring P2P
         pattern (each rotation sends what was just received from the
         previous rotation, so two cannot be in flight simultaneously
         without changing the comm pattern). We keep the value plumbed
         through and document the saturation in the paper.
   - **A3 — `kv_block_size`** → ring transmission chunk size. Each ring
     step's `batch_isend_irecv` of the per-rank K/V slice is split into
     contiguous chunks of `kv_block_size` positions. Smaller chunks =
     more NCCL launch overhead; larger chunks = better bandwidth
     amortization. The chunking is purely a transmission concern; the
     receiving rank still computes one FA-2 over the full reassembled
     slice (correctness is invariant under chunk size).

   **Correctness invariant:** the kernel output must be bit-equivalent
   (within accumulator precision) regardless of (chunk_size,
   prefetch_depth). Tests parameterize over the cross product to enforce.

   **Why this redefinition is meaningful:** the OLD A3/A4 numbers came from
   a prefetcher whose output was never consumed by attention (see audit
   in this doc's "Why this exists" section), so they reflected pure
   comm overhead. The NEW numbers reflect actual ring throughput with
   correctness coupling — strictly more publishable.

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
| R1.5 dual-cache decode kernel refactor | (in-line) | ✅ done 2026-05-05 (caught during R3 reading; 4 new tests) |
| R2 Llama patch | 2-3 | ✅ done 2026-05-05 (ring_llama_attention.py + 10 tests) |
| R2.5 dual-cache patch + set_prefill_len helper | (in-line) | ✅ done 2026-05-05 (2 new tests) |
| R3+R4 KV layout + prefetcher removal | 2-3 | ✅ done 2026-05-05 (rasd_inference.py surgery; 5 new tests; commit 09f7d98) |
| R3.5 A3/A4 redefinition + chunked/prefetched ring | 0.5 | ✅ done 2026-05-05 (commit d45ccbc; 16+2 new tests) |
| R5 stream audit | 1 | ✅ done 2026-05-05 (R5.1 code audit + iteration-boundary wait fix; R5.2 smoke folded into R6) |
| R6 validation matrix | 2-3 | pending — mostly pod cost |
| **Total** | **~10 days actual / 10-15 estimated** | plus ~$100 pod budget for R6 |

## Recent updates (chronological)

**2026-05-05** — Three commits land all of R0–R3.5:

1. `c7800c1` — *Ring/spec integration R0–R2: kernel + Llama patch + RoPE scaling.*
   Audit during the multi-rank stream review revealed the original "ring
   attention" path never fed prefetched KV blocks into the target's attention
   forward — `world_size>1` had each rank holding the full local KV with
   ring P2P being decorative. This commit captured the finding in
   `M3_RING_INTEGRATION_PLAN.md`, locked R0 design (contiguous KV slices,
   replicated draft, no ring on draft), shipped the layout-agnostic ring
   kernel free functions (R1) with FA-2 + online-softmax merge and 8 tests
   (single-process math + multi-process gloo at W∈{2,4}), and shipped the
   `install_ring_attention` monkey-patch with 10 plumbing tests (R2). Also
   threaded `context_length` through `RASDConfig` for 64k RoPE scaling
   (`_build_hf_config` helper applies linear rope_scaling factor when ctx
   exceeds the model's native 4096). 18/18 tests green at this point.

2. `09f7d98` — *Ring/spec integration R3 (merged R3+R4): finish dual-cache
   wiring.* Surfaced two design refinements during R3 reading that fed back
   into R1/R2:

   - **R0.1 flip**: round-robin → contiguous slice layout. Round-robin
     causal masking would have required custom CUDA; contiguous matches the
     standard literature pattern (skip future-rank steps, full-attend
     past-rank, FA-2 `causal=True` at self-step). The 256-token decode
     imbalance is ~3% at 64k base — negligible.
   - **R1.5 + R2.5 dual-cache decode**: the original R1 decode kernel
     implicitly required equal-size K/V across ranks during decode. Under
     the contiguous design, only rank W-1 grows, breaking this. Solution:
     during decode, every rank appends new K/V to a "replicated tail"
     (identical on all ranks since hidden_states is replicated). The kernel
     splits the per-rank cache into [sharded_prefill | replicated_tail],
     runs ring W steps over the prefill, runs one local pass over the tail,
     and combines via online softmax. Adds ~33 MB per rank for 256 decode
     tokens × 32 layers — negligible. Test layer added.

   Then the actual rasd_inference.py surgery: deleted `AsyncKVRingPrefetcher`
   class, `_KVBlock`, `_extract_kv_block`, `_alloc_kv_buffers`, the 40-line
   P2P+tick block in the verify loop, and the post-loop "drain remaining
   P2P" block. Added: install_ring_attention call in `_load_models`, sliced
   prefill with absolute position_ids, broadcast of the last logit from
   rank W-1, `set_prefill_len` call after prefill, `global_seqlen` tracking
   for verify position_ids. Stream count: 3 → 2. ~370 lines removed, ~70
   added. 59/59 tests green.

3. `d45ccbc` — *R3.5: redefine A3/A4 ablation axes onto ring-in-attention
   architecture.* The 49-row ablation grid is preserved by remapping
   `kv_block_size` and `prefetch_depth` onto the new architecture (they
   were going to be obsolete after R3 deleted the prefetcher). New
   semantics: A3 = per-step batch_isend_irecv chunk size; A4 = ring-step
   prefetch depth. `_issue_rotation` returns `(reqs, assemble_fn)` with the
   assemble_fn handling chunked recv (gloo's recv requires contiguous
   tensors, so chunked path uses per-chunk buffers and copies into the
   destination after wait). 16 parameterized invariance tests cover
   `(chunk_size ∈ {None, 2, 4, 8}) × (prefetch_depth ∈ {0, 1}) × {prefill,
   decode}` to enforce that varying knobs only changes timing, never
   correctness. 76/76 tests green.

## A3/A4 are real, not cosmetic

The redefined A3/A4 axes do **genuinely different work** in the kernel —
this is not just preserving a knob name to keep the YAML happy. Validating
that the ablation will produce meaningful curves on the pod:

**A3 (chunk_size) — what differs at runtime:**
- Number of NCCL ops per ring rotation: `4` (unchunked) vs `4 * ceil(S_local
  / chunk_size)` (chunked). At 64k context with W=8 (S_local=8k):
  chunk_size=2048 → 16 ops/rotation; chunk_size=256 → 128 ops/rotation.
  Each op carries CUDA launch overhead.
- Allocations: unchunked path allocates zero buffers per rotation; chunked
  path allocates `2 * ceil(S_local/chunk_size)` contiguous recv buffers
  each rotation.
- Memory bandwidth: chunked path's `assemble_fn` copies temp buffers into
  the destination after wait — O(S_local × H × D × bf16) per rotation per
  layer. At 64k×W=8 across 32 layers, that's nontrivial memory-bus traffic.
- Expected curve: U-shape with a sweet spot around 512–1024. Smaller
  chunks suffer launch overhead; very large chunks (= unchunked) are
  fastest in raw bandwidth but show NCCL bandwidth saturation differences
  at the limit.

**A4 (prefetch_depth) — what differs at runtime:**
- Sync (depth=0): host calls `r.wait()` then launches `_attn_step` — comm
  and compute serialize on the host timeline.
- Async (depth≥1): rotation s+1's `batch_isend_irecv` lands on NCCL's
  internal stream BEFORE the FA-2 kernel for step s launches on the
  compute stream. `r.wait()` happens at the top of the next iteration.
- On A100 SXM with NCCL: NCCL P2P runs ~50–100 GB/s intra-node; FA-2 at
  64k×8 heads is ~1–2 ms per layer. The two genuinely overlap on the GPU.
- Expected effect: meaningful tps win for async (likely 10–25% at 64k
  scales where comm time is non-trivial relative to compute). depth=2
  saturates at depth=1 behaviour because each ring rotation depends on
  the previous one's recv as its send — documented limit.

**Why this is more meaningful than the M3 numbers:**
The OLD prefetcher's output was never consumed by attention (the audit that
started this whole plan). M3's A3/A4 numbers reflected only the comm
scheduling overhead of an unused mechanism. The NEW A3/A4 numbers will
reflect actual ring throughput because the rotation output IS consumed by
the FA-2 kernel inside the attention forward — there's a genuine
correctness coupling now. Strictly more publishable.

## Plan-to-implementation deviations

The original M3 milestone document specified three things that the
post-audit implementation deviates from. Each deviation was surfaced
during the work and signed off; this section captures them so the
plan-vs-as-built mapping is traceable when writing experiments.md.

### 1. A1 draft model: TinyLlama-1.1B instead of DistilGPT-2 (124M)

**Plan said:** A1 levels = `DistilGPT-2 (124M params)`, `Sheared-LLaMA (1.3B params)`.

**Implemented:** `TinyLlama-1.1B`, `Sheared-LLaMA-1.3B`.

**Why:** DistilGPT-2 uses the GPT-2 tokenizer (vocab=50257), incompatible
with the LLaMA-2 SentencePiece tokenizer (vocab=32000) used by both target
options (Llama-2-7b, Mistral-7b — same SentencePiece). Speculative decoding
requires a shared vocab between draft and target so accept/reject can
operate token-for-token. TinyLlama-1.1B is the closest-in-size LLaMA-tokenizer
draft (1.1B params, LLaMA-2 SP vocab). Documented in
`feedback_dep_versions.md` and pinned via HF revisions in
[configs/ablations.yml](configs/ablations.yml).

**Impact on results:** A1 still studies "small vs medium draft" but the
small endpoint is 1.1B not 124M. The ratio is smaller, so A1's expected
acceptance-rate gap will be smaller than what 124M-vs-1.3B would show —
but the trend direction (larger draft → higher α, lower tps) is preserved.

### 2. CUDA stream count: 2 explicit + NCCL internal, not 3 explicit

**Plan said:** "Use three separate CUDA streams: one for target model
computation, one for draft model computation, and one for D2D/inter-GPU
communication."

**Implemented:** Two explicit streams (`stream_compute`, `stream_draft`),
plus NCCL's internal P2P stream which is implicitly managed by
`dist.batch_isend_irecv` inside the target's `LlamaAttention.forward`.

**Why:** R3's audit revealed that the original `AsyncKVRingPrefetcher`
(which owned the explicit `stream_comm`) posted P2P that was never
consumed by attention — the prefetched K/V blocks landed in
`pending_block.keys/values` but were never merged into the target's
attention input. Deleting the prefetcher (and its `stream_comm`) and
moving ring rotation INSIDE `LlamaAttention.forward` is the only way to
get correct ring attention with FA-2 + online-softmax merge. NCCL still
runs P2P on its own internal stream, so the *parallelism intent* of the
original plan is preserved — comm and compute still overlap (A4=1) — just
structurally co-located inside the kernel rather than orchestrated as a
separate stream from `generate()`.

**Impact on results:** zero on observable metrics; the GPU still runs comm
and compute concurrently. The change is internal — fewer Python-level stream
objects, same hardware-level concurrency. Stream-ordering invariants (Rule 4
in `feedback_spec_verify_fix.md`) collapsed from three explicit waits to two.

### 3. A3/A4 semantics: ring-in-attention knobs, not prefetcher knobs

**Plan said:**
- A3 = "KV Cache Block Size (for communication)" with levels {256, 512, 1024, 2048}
- A4 = "Communication/Computation Overlap Strategy" with levels {Sync, Async-1, Async-2}

**Implemented:** Same numerical levels, redefined semantics:
- A3 = per-step `batch_isend_irecv` chunk size inside the ring kernel
- A4 = ring-step prefetch depth (sync vs async overlap of rotation s+1
  with compute s); async-2 saturates as async-1 under the standard ring
  P2P pattern (each rotation depends on the previous one's recv as its
  send), documented as a known limit.

**Why:** the plan's A3/A4 measured the deleted prefetcher's mechanics.
After R3 collapse, those mechanics no longer exist. The redefinition maps
the *spirit* of A3 (granularity of comm) and A4 (compute/comm overlap)
onto the new architecture, where they actually steer kernel behaviour
(Python-level NCCL op count, allocations, stream concurrency) — see the
"A3/A4 are real, not cosmetic" section above.

**Impact on results:** the 49-row grid stays unchanged. Numerical levels
and reported variable names are preserved; semantics are explicitly
documented in the kernel docstring, the `RASDConfig` field comments,
the `rasd_inference.py` module header, and this plan doc.

### Net effect on milestone scorecard

All four M3 acceptance items still hold:
1. **"Implement core RASD algorithm + ring attention KV-cache integration."**
   ✅ done — actually *more* correct than the original implementation
   would have been, since the audit caught the silent unused-prefetcher bug.
2. **"Comprehensive grid search of ablations."** ⏸ pending R6 pod work; the
   49-row grid in `configs/ablations.yml` is unchanged.
3. **"Identify optimal RASD configuration for max throughput."** ⏸
   produced from R6 outputs. The redefined A3/A4 axes will give cleaner
   signals than the M3 ablation did.
4. **"Detailed analysis in `experiments.md`."** ⏸ produced in R6 wrap-up.

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

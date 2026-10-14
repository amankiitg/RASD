# RERUN_DECISION — the Part 2 decision, its numbers, and the exact candidate config

Run 2026-10-13, branch `decode-diagnosis`. Written and committed **before** anything in Part 3
was launched.

## The decision

**Speedup 3.560x — above the 1.5x gate — so Part 3 runs.**

| quantity | value | where it comes from |
|---|---|---|
| baseline: `ring2048` target-only decode s/tok at 128k | **3.5637** | `results/mlsys/decode_df/arms/df_arms.csv`, row `DF_ring2048_target_pg19_train_0_ctx131072_s42` |
| candidate: `splitkv` target-only decode s/tok at 128k | **1.0011** | same CSV, row `DF_splitkv_target_pg19_train_0_ctx131072_s42` |
| **speedup = baseline / candidate** | **3.560x** | both rows from ONE model load on ONE box, same book, same seed |
| gate | >= 1.50x | brief, Part 2 |

Both numbers are decode-only, `(time_sec - ttft) / (tokens_generated - 1)`, from unprofiled rows
with 16 decode tokens. The spec arm shows the same thing: 2.7910 -> 0.8254 = **3.382x**.

## The candidate engine config

Every change that passed its correctness gate, and nothing else:

| key | published Session A value | candidate value | why |
|---|---|---|---|
| `ring_decode_mode` | `ring` | **`split_kv`** | the fix: no rotation, each rank attends to its own shard, partials merged with two tiny collectives |
| `kv_block_size` | 2048 | **32768** | `>= S_local`, so the rotation is a single chunk. Bit-identical. Inert under `split_kv` (there is no rotation to chunk) but kept so the config states one engine |
| `ring_expand_after_rotation` | false | **true** | the GQA expansion moves to the receiving side. Bit-identical. Inert under `split_kv` |
| any `NCCL_*` | none | **none** | NCCL already uses `P2P/CUMEM/read` over a full NV12 mesh; there is no env fix to make |

Everything else is unchanged from Session A: FP4 weights + NF4 KV (the branch-C precision),
`dtype` bfloat16, `spec_steps` 4, `prefetch_depth` 1, `draft_window_cap` 4096, `temperature` 0.7,
`max_new_tokens` 128, the same target/draft revisions, the same five books and offsets, and the
periodicity check on every row.

**The two exact changes are inert under `split_kv`, and the report will say so rather than
claim their speedup.** They were kept because they are bit-identical and cost nothing; the
measured 1.240x and +2.7% belong to the ring path, which the candidate does not use. Part 3's
gate stage measures the combined config, so if the combination were NOT equal to `split_kv`
alone that would show up as a difference in the gate row.

**Commit:** the tree the candidate runs is the one `git rev-parse HEAD` names at launch,
forwarded to the pod as `DF_COMMIT` and asserted by the parameter contract, and recorded in
every CSV row and every token sidecar. The pod's `~/RASD` is an rsync with no `.git`, which is
why the Part 1 rows recorded `git_commit: unknown`; that gap is fixed before this rerun.

## Why this is the fix, in one paragraph

The bare ring costs **10.3 ms per round whether it carries 4 KB or 266 MB** (66,000x payload
range, 0.4% variation; worst rank spread 0.056 ms over eight ranks). So the rotation's cost is
per-round latency, not bytes: `repeat_kv` after the rotation, which cuts the wire payload 4x, is
worth +2.7%, and no NCCL setting can help because the transport is already `P2P/CUMEM/read` over
NV12. 256 rounds x ~10 ms = **2.6 s of a 3.3-3.7 s decode step**, so the only thing that can pay
is removing rounds — which is exactly what `split_kv` does: it has no rounds at all.

## The correctness gate for `split_kv`, as required by the brief

Three conditions, all of them checked:

1. **Unit tests pass.** `tests/test_mlsys_split_kv_decode.py` — 6 tests, including the merge
   against a single attention over the concatenated shards, the tail counted once, and the
   faithful two-rank emulation of the collectives.
2. **The 128k divergence is at a near-tie, below that row's own median top-2 gap.** Target-only:
   first divergence at index **12** of 16, after twelve identical tokens, at a position where the
   baseline's own margin is **0.0000** against a row median of 2.9375. Greedy (no sampling in
   the way): same index, same 0.0000. Spec: divergence at index 1, margin 0.8750 against a
   median of 0.9375. A structurally wrong merge diverges at the first token; this one reproduces
   the ring **bit-for-bit for twelve consecutive tokens**.
3. **Greedy perplexity within 1%.** All four arms return **1.648977**, identical to six decimal
   places — **and this condition is vacuous as a check on this change**.
   `_measure_target_ppl` calls `model.model(...)` with `use_cache=False`, i.e. a **prefill**
   forward, so a decode-path change cannot alter it. It was measured because it was asked for;
   the evidence that carries the gate is (1) and (2). This is recorded here so the ppl number is
   not read as a stronger result than it is.

## Artifacts

* `results/mlsys/decode_df/` — every row, sidecar, log, the NCCL transport evidence and
  `nvidia-smi topo -m` for this session.
* `results/mlsys/DECODE_DIAGNOSIS.md` section F — the full analysis.
* Session: one 8x A100 80GB SXM4 instance, five stages all rc=0, **$11.31** of the $35 ceiling,
  0 instances left, dead-man timer cancelled.

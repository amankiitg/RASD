# LCFM_RECONCILE — the arXiv/LCFM manuscript against this week's evidence and the current code

Read-only audit, 2026-10-12. No GPU, no spend. No running process, instance or other file was touched.
Nothing here is reused as evidence for the MLSys paper. The old Llama-2 + YaRN + Sheared-LLaMA
artifacts are analysed only to decide what the public preprint must correct.

**What was compared**

* `manuscript/arxiv/main.tex` (1858 lines). Its content is identical to
  `manuscript/arxiv/arxiv-submission/main.tex`, the uploaded source; only include paths differ.
  Line numbers below refer to `manuscript/arxiv/main.tex`.
* `manuscript/workshop/main.tex`, the **LCFM camera-ready** at tag `lcfm-camera-ready` (2026-10-02).
  It is a separate 8-page extract, but it repeats nearly every claim corrected here (cross-references
  are marked `[W:line]`).
* Code at the paper's tags (`m3-reablation`, `m4-phase-c-complete`, `m4-phase-d-complete`) and at
  `HEAD` (`34b955d`). The ring kernel has not changed since `m4-phase-c-complete`: only two docstring
  lines differ.
* Evidence: `results/final/*` (the paper's own data), `results/mlsys/BANDWIDTH_FLOOR.md`,
  `PROFILE.md`, `RESULTS_OCT.md`, `CLAIMS.md`, `MORNING_REPORT.txt` (read only), commit `d89cf41`
  (draft window), `docs/lcfm_correction_note.md` (YaRN anchor), `docs/manuscript_fixes_pending.md`
  (FP4 weights), and `results/mlsys/README.md` (nominal context).

Scratch scripts used for §2 are in the session scratchpad and are not committed. Every number below
can be recomputed from the committed CSVs, JSONs and token files named next to it.

---

## 0. Summary

1. **The "comm ≤ 1.2% of wall" result is wrong, and the conclusion drawn from it must be withdrawn.**
   At 128k/256k/512k the profiler's comm bucket recorded 0.84/4.39/10.57 s. The minimum time NVLink
   needs to move the K/V bytes the ring kernel sends in those same runs is 2.7/13.4/28.4 s. The bucket
   recorded 0.31–0.37× that floor, so it did not capture the transfers. The "idle ≥ 50%" figure is
   57–80% profiler time spent outside `generate()`'s own timer.
2. **The 512k/1M "regression" (0.94×/0.97×) is not an acceptance effect, and the paper's own re-run
   does not reproduce it.** Decode tokens per verify round are flat from 256k to 1M (1.80 → 1.69 →
   1.71). The regression comes from per-round cost: in the main-matrix pod a spec round cost 1.8× a
   target-only step at 512k/1M, against 0.97× at 128k/256k. The same-code Phase D re-run (seed 42,
   same session) gives **1.61×/1.90×/1.53×/1.56×** at 128k/256k/512k/1M. The 1M seed-42 trajectory is
   identical in both pods (acceptance 0.2014, 36 rounds, 66 tokens) but cost 31.9 s/round in one and
   17.3 s/round in the other.
3. **The draft model never saw the end of the prompt.** In every spec run whose prompt exceeded 4096
   tokens (8k PG-19, all of 128k–1M, and the M3 64k ablation grid), the draft was conditioned on the
   prompt's *first* 4096 tokens (`d89cf41`). The paper and LCFM say the opposite.
4. **The 4.4× claim passes the loop check but fails on other grounds.** Both 4k rows are clean under
   the periodicity detector (T = 1.0, 64-token cap). But it is one pair from one session. Its
   decode-only ratio (6.1×) exceeds the 3.82 tokens per round that speculation can deliver, because a
   whole spec round cost 0.62× of one target-only step. As a control for "bimodality is YaRN", it
   differs from the long-context cells in at least five ways at once.
5. **YaRN:** every ≥ 8k cell was anchored on its own context length, not on 4096 (transformers ≤ 4.50
   ignores `original_max_position_embeddings`). The PPL table's "YaRN" rows all used factor 8, not
   ⌈N/4096⌉. At 4k the engine applies no YaRN at all.
6. **The paper's overlap description is wrong, but not for the reason BANDWIDTH_FLOOR.md gives.** The
   real kernel issues the next rotation before each step's attention (`prefetch_depth` 1).
   BANDWIDTH_FLOOR quoted `ring_attention_flash.py`, which is not on the inference path. What the code
   never implemented is the paper's central overlap, **draft generation hidden behind ring traffic**:
   the draft and the verify run back to back.
7. Smaller errata: FP4 (not NF4) weights; the reported memory peak is rank 0 only (allocated max over
   ranks is 41.2 GiB and *reserved* is 69 GiB, not "half of 80"); a test the paper cites does not
   exist; the "1M" cells were 1,002,984 tokens; the
   "3 seeds" used one synthetic prompt; CIs use ±1.96 SEM at n = 3.
8. **LCFM status:** the repo calls it non-archival and double-blind (README, a self-statement). The
   repo cannot settle the rest (see §4).

---

## 1. ERRATA for the public preprint

Status labels: **WRONG** (contradicted by the code or the paper's own data), **UNSUPPORTED** (no
evidence for it), **CONFOUNDED** (the evidence cannot separate the stated cause from others).

### A. Communication, profiler, overlap

#### E1 — "communication is ≤ 1.2% of wall on the speculator rank" — WRONG
Locations: abstract L88-91, L206-208, L510-523, §5.4 L1237-1308, L1514-1531, conclusion L1602-1606.
In LCFM: [W:61-63, 125-129, 538-567, 638].

**What the old comm bucket counted.** `src/analysis/profiler.py` (unchanged since the paper):

* rank 0 only (`profile_enabled = ... and local_rank == 0`, `run_experiment.py` at
  `m4-phase-c-complete`);
* one continuous `torch.profiler` capture wrapping `engine.generate_text(prompt)`, not just the
  decode loop;
* each event key from `key_averages()` is bucketed by substring (`nccl`, `c10d`, `isend`, `irecv`,
  `broadcast`, `p2p`, …). The value added is **CPU self-time + CUDA self-time**;
* `comm_pct = comm_us / wall_us`, where `wall_us` is `perf_counter` around the context manager. It
  includes the profiler's own `__exit__` (trace stop and collection) and the tokenisation and decode
  inside `generate_text`.

Only the four-bucket summary JSONs were saved (`results/final/profiler_pass/profiler/*.json`). With no
raw trace, the exact event set cannot be reconstructed. What can be computed is a lower bound on the
transfer time the bucket should have contained.

**The bucket is below the physical minimum.** The decode ring (`ring_attention_kernel.py`
`_ring_against_sharded`) rotates the whole local prefill shard 7 times (W − 1) per layer per target
forward. The payload is bf16 K and V with 32 heads (Llama-2 is MHA) × 128 dim. The same rotations
happen at prefill. Using the actual prompt lengths (`nominal_vs_actual_context.csv`), the n_rounds in
`profiler_pass.csv`, and 300 GB/s per direction (A100 SXM4 NVLink peak):

| ctx | S_local | K+V per rotation | per forward (32 layers × 7) | forwards (prefill + rounds) | bytes sent per rank | **wire floor** | **recorded comm** | recorded / floor | floor / run's `time_sec` |
|---|---|---|---|---|---|---|---|---|---|
| 128k | 15,670 | 256.7 MB | 57.5 GB | 14 | 0.81 TB | **2.7 s** | 0.84 s | **0.31×** | 5.8% |
| 256k | 31,339 | 513.5 MB | 115.0 GB | 35 | 4.03 TB | **13.4 s** | 4.39 s | **0.33×** | 6.2% |
| 512k | 62,684 | 1,027 MB | 230.1 GB | 37 | 8.51 TB | **28.4 s** | 10.57 s | **0.37×** | 3.8% |

A bucket that contained the NCCL transfer kernels could not sit below the time the link needs at
100% efficiency. The bucket therefore missed most of the transfer time. Plausible reasons that cannot
be checked without the raw trace: CUPTI dropping device activity over long captures, or NCCL kernel
time on NCCL's internal stream not landing in `self_cuda_time_total`. Two further structural problems
remain. CPU-side blocking on the device (`cudaStreamSynchronize`, `aten::_local_scalar_dense`, …)
matches no comm or skip pattern, so it falls into **compute**. And CPU and CUDA self-times are added,
so the shares are not time shares.

**Two more sentences fall with it:**

* L1247-1252, "issued comm upper-bounds exposed comm, so ≤ 1.2% rules out overlap regardless of
  definition". Wrong: the figure is neither issued nor exposed comm.
* L1282-1284, "agrees with theoretical NCCL bandwidth estimates". Wrong: the theoretical floor alone
  is 3.8–6.2% of the measured run time, and real P2P runs well below peak.

**Correct statement for v2:**
> Our profiler's communication bucket summed the self-time of NCCL/c10d-named events on rank 0. It
> recorded 0.84/4.39/10.57 s at 128k/256k/512k, which is 0.31–0.37× the minimum time NVLink needs to
> move the K/V bytes our ring kernel sends in those runs (2.7/13.4/28.4 s at 300 GB/s per direction).
> The bucket therefore did not capture the transfers. The ≤ 1.2% figure is withdrawn, and so is the
> conclusion that communication is negligible and overlap is ruled out. The wire-time floor alone is
> 3.8–6.2% of measured run time at these contexts. The actual exposed communication share was not
> measured.

**What this week's evidence adds (different model, so context only, not a v2 replacement number).**
On the Gradient-1048k target, target-only decode time grows nearly linearly with the number of ring
chunk transfers (`BANDWIDTH_FLOOR.md` §3). Going from 1 rank to 8 ranks multiplies decode s/token by
about 2.1× at 128k (`PROFILE.md` §4). Both point the same way as E1: communication-related cost on
this engine is not ~1%. Neither measures the old configuration.

#### E2 — "Idle is ≥ 50% of rank-0 wall … rank 0 waiting on collective synchronization with the other 7 ranks, which are doing useful compute on their shards" — WRONG
Locations: L1291-1297, Fig. 3 caption L1261-1266, [W:548-550, 567].

* `idle` is `wall_us − (compute + comm + other self-time)`, i.e. *unrecorded* time. It is not a
  measurement of waiting.
* `wall_us` exceeds the run's own `time_sec` by **31.4 s (128k), 154.5 s (256k), 314.6 s (512k)**.
  That is time inside the profiler context but outside `generate()`'s timer, and it is **78%, 80% and
  57% of the idle bucket** respectively.
* In decode, the other ranks are not "doing useful compute on their shards" while rank 0 waits. Q is
  replicated, every rank runs the identical full-model forward on the same new tokens, and each rank
  attends to its 1/W prefill shard in lock-step ring steps (`ring_llama_attention.py` docstring;
  `ring_attention_kernel.py:457`).
* There is no distinct "speculator rank" in the M4 code: every rank runs the draft and the verify.
  The term comes from the retired M3 `_ring_peer_loop` design.

#### E3 — the tick-gate barrier — WRONG for the code that produced the M4 numbers
Locations: L699-701 ("we add a tick-gate barrier from rank 0 before the batch_isend_irecv call") and
L1252-1256 (tick sends "included in the comm bucket"); [W:546].

The tick gate (`dc14915`, 2026-04-15) belonged to the M3 `_ring_peer_loop` architecture. It is absent
from `src/models/rasd_inference.py` at `m3-reablation`, `m4-phase-c-complete` and HEAD (`grep tick` →
nothing). The R3 dual-cache rewrite replaced it. Barriers that do exist: a full `torch.cuda.synchronize()`
at the start of every round when `prefetch_depth ≥ 1` (`rasd_inference.py:1950`) and the round-state
broadcast (`:2045`).

#### E4 — §3.2's ring-exchange/overlap description vs the code — partly WRONG
Location: L689-697 (and the gap analysis L456-477).

Correcting the premise first. `BANDWIDTH_FLOOR.md` §4 C2 says every transfer is `wait()`ed in Python
with no overlap and that `prefetch_depth` is unused. That quote comes from
`src/models/ring_attention_flash.py:235-295` (`RingAttentionFlash._ring_forward`). **That module is
not on the inference path.** It is used only by `scripts/validate_flash_memory.py`. The Llama path is
`ring_llama_attention.py` → `ring_attention_kernel.ring_attention_decode` → `_ring_against_sharded`,
where:

* with `prefetch_depth ≥ 1` (all paper and session configs use 1), rotation s+1 is **issued before**
  step s's attention (`ring_attention_kernel.py:403`) and waited at the **top of the next step**
  (`:389`). `k_buf`/`v_buf` form a real double buffer. Depth 2 is documented as identical to depth 1
  (`:344`);
* each rotation is one `batch_isend_irecv` of 4 × ⌈S_local / c⌉ ops (`_issue_rotation`, `:256`). The
  waits are per rotation (per ring step), on every request of that batch;
* direction: rank r **sends to r + 1 and receives from r − 1** (`:355`).

So the paper's "issue async, compute while in flight" is roughly what the kernel does. These specifics
are wrong:

| paper (L689-701) | code |
|---|---|
| "send to rank i−1, recv from rank i+1" | send to r+1, recv from r−1 (Fig. 1 at L599-601 has it right; the text contradicts it) |
| "at decode round r … the r-th K/V block" | every target forward does W − 1 = 7 rotations **per layer**, each moving the whole shard; there is no one-block-per-decode-round schedule |
| "the two streams synchronize at the end of the round" | the wait is per ring step; in addition, a device-wide `torch.cuda.synchronize()` runs at every round start (`rasd_inference.py:1950`) |
| tick-gate barrier | absent (E3) |
| (implied) overlap hides the transfer | the overlap window is one step's attention, i.e. a 1–5-query FA-2 call per step, far shorter than moving a 32-head bf16 shard. Nearly all transfer time stays on the critical path |

**The paper's central overlap was never implemented.** The gap analysis (§2.5 L456-477, Eq. 2) and the
positioning table (L483-508, "Decode Overlap ✓") propose hiding ring communication behind the
*draft's* γ steps. In the code the verify forward waits for the draft
(`stream_compute.wait_stream(self.stream_draft)`, `rasd_inference.py:1988`). The ring traffic happens
inside that verify forward, so draft generation and ring exchange never overlap. The profiler result
was presented as ruling out a mechanism that the system did not contain.

#### E5 — positioning table and related-work framing — WRONG / UNSUPPORTED
* L505, "RASD … Decode Overlap ✓". Wrong (E4).
* L504, "vLLM … Spec Decoding —". vLLM ships speculative decoding.
* L383-390, "All prior distributed inference work treats decode as strictly sequential … order
  10–100 ms per transfer on NVLink". Unsupported (no source, no measurement).
* L416-417 and L427-429, speculative decoding "not measured at million-token context". The LCFM
  camera-ready added TriForce/MagicDec (`4e5403b`); the arXiv text predates that and overclaims
  novelty.

#### Corrections this audit found in BANDWIDTH_FLOOR.md / PROFILE.md (not edited, flagged for the MLSys text)
1. **The ring payload is 4× what A6/A11 assume.** `ring_llama_attention.py:134` calls
   `repeat_kv(k_full, num_kv_groups)` *before* `ring_attention_decode`. For the Gradient target
   (32 Q heads, 8 KV heads) the rotated tensors carry 32 heads, not 8. There are also 7 rotations per
   layer, not 8 (A12). Corrected ring floor: **≈ 200 / 800 / 1,600 ms** per token at 128k/512k/1M
   (not 57/229/458). The measured/floor ratio becomes **≈ 17× / 12× / 12×** (not 59/43/42×). Each
   chunk is 16.8 MB (not 2.1 MB), at 28,672 chunk-transfers per token at 1M (not 32,768): ≈ 0.67 ms
   each, ≈ 25 GB/s effective (not 3.6 GB/s). The qualitative finding (a large gap; a cost that tracks
   transfer count) survives. The magnitudes do not.
2. **C2 quotes the wrong module** (E4). C4 (whole-layer dequantize on every read) is right:
   `nf4_dynamic_cache.py:285` returns `_dequantize_layer` from every `update()`.
3. **Profile (a) at `--nproc 1` does not run the ring kernel at all.** `install_ring_attention` is a
   no-op at world size 1, and the patched forward falls through to HF's original attention
   (`ring_llama_attention.py:71`). The 1-rank vs 8-rank "2.1× ring multiplier" therefore also changes
   the attention implementation, not only the presence of communication. Also, "28× the 59 ms floor"
   divides a 1-rank time by the 8-rank *ring* floor. A 1-rank run has no ring term, so its own floor is
   a few ms and the local gap is even larger than stated.

### B. What the draft model actually saw

#### E6 — draft conditioning — WRONG; CONFOUNDS every long-context acceptance number
Locations: implicit throughout §3.1/§3.4. LCFM states it outright: [W:237-241] says the draft "attends
over only the most recent 4,096 tokens".

* `generate_text` built the draft input as `tokenizer(prompt, max_length=draft_max_len,
  truncation=True)` (added 2026-04-10 in `24879f0`; present at `m3-reablation` and
  `m4-phase-c/d-complete`). HF's default `truncation_side="right"` **keeps the first 4096 tokens**.
  The later `raw_draft_ids[:, -draft_max_len:]` in `generate()` was then a no-op. `run_experiment.py`
  at those tags calls `engine.generate_text(prompt)`.
* The draft (Sheared-LLaMA-1.3B, `max_position_embeddings` 4096, deliberately not RoPE-scaled) was
  prefilled on **prompt tokens 0..4095** and then decoded from position 4096 onward, a few dozen
  positions past its own window. Its weights were 4-bit FP4 (E19); its KV was a bf16 HF cache.
* Measured on a real 131,072-token PG-19 window (`d89cf41`): the draft input contained **none** of the
  prompt's final 4096 tokens.
* **Affected:** `final_matrix.csv` (128k–1M, all seeds), `phase_d_rerun.csv` spec rows, the 8k PG-19
  cell, the profiler rows, `ablations_r65.csv` (64k, the whole M3 grid behind §3.6), and everything in
  `results/mlsys/` built on Llama-2 or on a capped draft at > 4096. **Not affected:** the 4k PG-19 cell
  (prompt = 4096 tokens including BOS, so truncation was a no-op) and every target-only row.
* For the synthetic repeated-paragraph prompt, the draft saw the same paragraph cycle but ending at a
  **different phase** from the target's context. It was therefore predicting the continuation of a
  different point in the cycle.
* Fixed at HEAD: one rule, `_draft_window` (`rasd_inference.py:868`), BOS + the most recent
  `draft_max_len − 1` tokens, applied once at `:1705`.

**Correct statement:** acceptance at every context > 4096 was measured with a draft conditioned on the
opening of the prompt, not its end. Those acceptance numbers, the bimodality, and the speedups derived
from them cannot be attributed to context length or to YaRN.

### C. YaRN configuration

#### E7 — the anchor — WRONG configuration, disclosed nowhere in the paper
Source: `docs/lcfm_correction_note.md`. Pinned transformers is **4.46.3** (`requirements-lock.txt` at
`m4-phase-c-complete` and HEAD; the note says 4.47.1, which is a slip in the note, but both are
≤ 4.50). It never reads `rope_scaling.original_max_position_embeddings`. The engine sets
`max_position_embeddings = context_length`, so every YaRN cell was anchored on its own context. The
slowest channels were stretched 2.6× (f = 8) to 8.2× (f = 32) more than YaRN intends. Measured at 32k,
f = 8: PPL **826.9** as published vs **14.45** correctly anchored (native 4k: 2.97); generation still
collapses into newlines under both anchors. Affects L547-548, L789-802 and every "YaRN factor 256"
statement. [W:53, 146-147, 259].

#### E8 — the PPL table's YaRN rows all used factor 8 — WRONG as described
Locations: L795-802, L1404-1442, Appendix B L1763-1782 ("YaRN with f = ⌈N/4096⌉").

`scripts/eval_perplexity_matrix.py` at `m4-phase-c-complete` loads the model **once** with
`factor = ceil(max_ctx / 4096) = 8` and `max_position_embeddings = 32768` for all rows (lines
178-190 of that script). So:

* the 4k and 8k "YaRN" PPLs (12.1, 13.2) are factor-8, mis-anchored YaRN evaluated at 4k/8k. They are
  not the configuration the engine used at those contexts;
* the "**13% YaRN penalty at native 4k**" (L1416-1421) is the cost of mis-anchored f = 8 at 4k. The
  engine applies **no** YaRN at 4k: `_build_hf_config` scales only when `context_length >
  max_position_embeddings`. YaRN at f = 1 would be the identity;
* the "11–18× YaRN benefit" (L378-380, L1428-1434) compares vanilla RoPE with mis-anchored f = 8. It
  does not measure the matrix's per-context factors.

#### E9 — the 4k/8k control's rope state
At 4k: no scaling. At 8k: YaRN f = 2, anchored on 8192 (E7). At 128k–1M: f = 32–256, anchored on the
context. The "YaRN factor 256" framing is accurate only for the 1M cell, and that cell ran
1,002,984 tokens (E13).

### D. Verify loop and measurement defects

#### E10 — verify-loop defects
* **Known and disclosed:** the three 2026-04-16 defects (packed forward, residual resample, KV
  truncation; `2e45518`) and the stream race (`9e13f1f`). Both commits precede `m3-reablation` and
  `m4-phase-c-complete`, and L924-928/L1852-1856 disclose the re-run. No erratum needed beyond keeping
  that disclosure.
* **Found after the preprint, applicable to its runs (T = 1.0):**
  - `draft_seq` was not in the round-state broadcast (fixed in `d89cf41`, R4). At T > 0 each rank
    samples its own draft proposal, so ranks could test different proposals. L766-769 says the
    broadcast makes acceptance "bit-identical everywhere"; that was true of the logits, not of the
    proposals.
  - The acceptance estimand counted the cap-truncated final round (`1408d92`), biasing per-round α
    slightly low.
  - EOS early-stop was on, so rows have different lengths: the 128k spec rows generated **20/27/16**
    tokens against 64/64/22 for target-only. Spec rounds could also overshoot the cap (66 vs 64).
    Throughput includes prefill (§E15), so rows of different length amortize prefill differently.
* L776-778: "a unit test (`tests/test_verification_math.py`) that runs the verify loop with two
  different seeds and confirms byte-identical KV state". **No such test exists.** That file holds
  verify-math specs (conditioning on draft tokens, KV truncation, residual resample, bonus, ratio
  math) and no cross-seed or cross-rank KV comparison.

#### E11 — "3 seeds" — CONFOUNDED
The synthetic prompt builder ignored `seed` until `2836a64` (A7). The 3-seed matrix is therefore one
prompt × 3 decoding seeds at T = 1.0. The CIs reflect sampling noise on one prompt, not prompt
variation. For PG-19 cells the seed did pick the document.

#### E12 — confidence intervals — WRONG method
L1166, L1128, [W:58-59, 480-483]: "95% CI (± 1.96 SEM)" with n = 3. The t-quantile for 2 degrees of
freedom is 4.30, so the stated intervals are about 2.2× too narrow.

#### E13 — nominal context — WRONG as stated
Synthetic cells ran **~95.6% of nominal**: 125,360 / 250,716 / 501,472 / 1,002,984 tokens for
"128k/256k/512k/1M" (`results/mlsys/nominal_vs_actual_context.csv`; the memory trace confirms
S_local 125,373 at "1M"). `_prefill` also drops the prompt's last `S mod W` tokens to make it
divisible by W (`rasd_inference.py:1579`), so "each rank holds ⌈N/W⌉ tokens" (L545, L592, L606) is
⌊N/W⌋. Paired ratios are unaffected (same builder in both arms). Every "1M-token context" sentence
needs the footnote staged in `docs/manuscript_fixes_pending.md`.

### E. The speedup and acceptance claims

#### E14 — "the speedup regresses to 0.94× at 512k and 0.97× at 1M" — CONFOUNDED; not reproduced by the paper's own re-run
Locations: abstract L83-85, L194-198, L1173-1177, L1589-1596, [W:58-59, 138-141, 451, 517, 598-599].

Computed from `final_matrix.csv` / `target_only_matrix.csv` (Phase C, 3-seed means) and
`phase_d_rerun.csv` / `phase_d_short_ctx_pg19.csv` (Phase D, seed 42, same session). Decode-only rate
is `(tokens − 1) / (time_sec − ttft)`. "Tokens/round" is `(tokens − 1) / n_rounds`, the decode tokens
emitted per verify round.

| source | ctx | e2e speedup | decode-only | tokens/round | spec round (s) | target-only step (s) | round / step |
|---|---|---|---|---|---|---|---|
| Phase C | 128k | 1.256× | 1.375× | 1.36 | 4.49 | 4.60 | 0.98 |
| Phase C | 256k | 1.761× | 1.869× | 1.80 | 6.84 | 7.09 | 0.97 |
| Phase C | 512k | **0.940×** | 0.928× | **1.69** | 17.52 | 9.63 | **1.82** |
| Phase C | 1M | **0.972×** | 0.965× | **1.71** | 31.32 | 17.68 | **1.77** |
| Phase D | 4k | 4.383× | 6.122× | 3.82 | 1.142 | 1.829 | **0.62** |
| Phase D | 8k | 2.251× | 2.518× | 2.03 | 1.428 | 1.769 | 0.81 |
| Phase D | 128k | 1.612× | 1.894× | 1.46 | 2.696 | 3.494 | 0.77 |
| Phase D | 256k | 1.896× | 2.030× | 1.85 | 5.114 | 5.603 | 0.91 |
| Phase D | 512k | **1.526×** | 1.607× | 1.78 | 16.624 | 15.023 | 1.11 |
| Phase D | 1M | **1.556×** | 1.840× | 1.81 | 17.299 | 17.625 | 0.98 |

Readings:

* Tokens per round **do not fall** from 256k to 1M (1.80 → 1.69 → 1.71 in Phase C). The 512k/1M
  "regression" comes entirely from the round/step cost ratio jumping from ~0.97 to ~1.8 in the
  Phase C pod. The abstract's causal chain (acceptance becomes bimodal, so speedup regresses) is not
  what the numbers show.
* The **same seed-42 1M spec trajectory** appears in both pods: acceptance 0.2014, 36 rounds,
  66 tokens. It cost **31.87 s/round** in Phase C and **17.30 s/round** in Phase D. Within each pod,
  the three seeds agree to ±3%. Between pods, identical work differs by 1.84×. The Phase D 512k
  target-only step (15.02 s, a retry after a `TypeError`) is likewise 1.56× its Phase C value
  (9.63 s).
* In 5 of 6 Phase D pairs, a whole spec round (4 draft forwards plus a 5-token verify) costs **less**
  than one 1-token target-only step. With matched implementations that cannot happen. The two arms run
  different loops (target-only: per-step `stream_compute.synchronize()` + broadcast, no draft model
  loaded; spec: per-round `torch.cuda.synchronize()` + round broadcast). So "differs in exactly one
  variable" (L239-240, L1019-1023) is an overstatement, and part of every speedup is an unexplained
  per-forward cost difference, not speculation.
* All speedups are end-to-end tok/s *including prefill* (L1062-1063 defines T = N_tokens / T_wall
  without saying so). Prefill is identical in both arms and compresses the ratio toward 1.

**Correct statement:** across 128k–1M, speculation emitted 1.36–1.85 decode tokens per verify round
(n = 3 Phase C, n = 1 Phase D), with no decline from 256k to 1M. End-to-end speedups against our own
target-only path ranged from 0.94× to 1.90× depending on which session's rows are paired. The
per-round cost of identical work varied by up to 1.84× between sessions, so no context-dependent
trend in speedup is supported.

#### E15 — "a short-context PG-19 control attributes the bimodality to base-model YaRN extrapolation" — CONFOUNDED
Locations: abstract L86-87, L200-203, L1179-1197, L1359-1388, L1500-1512, [W:61, 387-417].

The 4k control and the long-context cells differ simultaneously in:

1. rope: none at 4k, mis-anchored YaRN at ≥ 8k (E7, E9);
2. draft context: the full prompt at 4k, the prompt's opening at ≥ 8k (E6);
3. prompt source: PG-19 vs synthetic repeated paragraph (one prompt for all seeds, E11);
4. document: `_build_pg19_prompt` draws a random chunk and offset per context, so 4k and 8k read
   different books (the saved prompts confirm this);
5. context length.

The 8k PG-19 cell, which changes (1), (2), (4) and (5) at once, already drops α from 0.706 to 0.258.
None of the five can be singled out. The "bimodality" itself: the later dip test
(`results/mlsys/dip_by_context.csv`) also rejects unimodality at 8k in 3/3 seeds and at 4k in 1/3. The
"not present at 4k" statement (L1362-1364) is therefore also only partly right.

**Correct statement:** the 4k-vs-long-context contrast is confounded with draft conditioning, rope
configuration, prompt source and document. It does not identify YaRN as the cause.

#### E16 — the 4.4× claim — see §2.4 (survives the loop check; does not survive as stated)

#### E17 — the speedup formula — WRONG application
L402-409 and L1300-1303 apply `(1 − α^{γ+1}) / (1 − α)` with the CSV α. That formula takes the
i.i.d. per-token acceptance probability, while the CSV reports the per-round mean accepted fraction
(`results/mlsys/README.md`, AbH52 #1). With α_round ≈ 0.20 the formula gives 1.25 tokens/forward,
against 1.69–1.85 measured. Separately, at α = 0.8, γ = 4 the formula gives 3.36, not "≈ 3.5" (L407).

### F. Memory and capability

#### E18 — peak memory — WRONG as defined
Locations: L99-100, L211-212, L1072-1073, L1111-1112, [W: abstract].

* `gpu_peak_mem_mb` is **rank 0's `torch.cuda.max_memory_allocated`** (`rasd_inference.py:1873/2284`;
  only rank 0 writes the CSV). It is not "nvidia-smi-reported" and not "maximum over the 8 ranks"
  (L1072-1073).
* From the committed per-rank memory traces at "1M" (`results/final/memory_trace/RASD_ctx1M_phaseD_s42.rank*.json`):
  allocated peak is **40,283 MiB on rank 0 but 42,229 MiB on ranks 2–7 (41.2 GiB)**. **Reserved**
  peak, which nvidia-smi tracks, is **67,144–70,780 MiB (65.6–69.1 GiB) per rank**. "40 GiB peak per
  rank, half the available 80 GiB" understates device occupancy by ~28 GiB.
* L733-735, sink prefix: `128 · 2 · 2 · 4096 · 32` B = 64 MiB is **correct** (checked; no erratum).
* L742-751, memory equation: "½" is bytes per element, not "the NF4 compression ratio relative to
  bf16" (that is ¼). The equation also omits the per-block fp32 scales: 0.625 B/element at block 32,
  so NF4 KV at 1M is ~20 GiB/rank, not 16 (L1143).
* L226-228, "validate it within rounding across the matrix", contradicts L752-758 / L1209-1214
  (measured ~15 GiB higher).
* L229-232, "without either component the configuration does not fit". The bf16 sink prefix *adds*
  memory; it exists for accuracy. There is no no-sink/no-chunk memory measurement in the repo.
* L1132-1141, "prefill activations (quadratic in sequence length even with FlashAttention…)". Wrong:
  FA-2's memory is linear in sequence length.
* L183-186, L224-225, L1039-1044: the HF ceiling test used **one GPU** of the 8-GPU node. "The
  standard open-source inference stack does not reach 128k on this hardware" is not what was tested.

### G. Other methods and experiment sentences

#### E19 — weights are FP4, not NF4 — WRONG
L97, L540-541, L607-608, L975-976, L983, keywords L108. The loader never set `bnb_4bit_quant_type`,
so bitsandbytes' default **fp4** applied (`2836a64`; pinned explicitly at HEAD,
`rasd_inference.py:1244, 1335`). The KV cache is genuinely NF4 (custom codec).
`docs/manuscript_fixes_pending.md` has staged replacements.

#### E20 — NF4 KV description — WRONG in three places
* L715-716: "block size 32 … the inner dimension blocking we found gave the best acceptance-rate vs.
  memory trade-off in M3 ablations". The M3 grid (A1–A5) never swept the NF4 block size. Unsupported.
* L716-717: "dequantization to bf16 happens on-the-fly inside the FlashAttention-2 kernel call".
  Wrong: `NF4DynamicCache.update` returns `_dequantize_layer(...)` (`nf4_dynamic_cache.py:285`). That
  is the whole layer, dequantized to bf16 in PyTorch on every forward, and then passed to FA-2.
* L719-726: "re-quantizes the full per-layer block … chunks of 2048 elements … transient under
  100 MiB". Wrong on mechanism and units. `update` quantizes only the new tokens. The chunking is
  2048 **tokens** along the sequence axis, and it bounds the *prefill* quantization transient. Every
  read still materializes the full layer in bf16 (~2 GiB per layer at 1M for Llama-2), so the decode
  transient is not under 100 MiB.

#### E21 — §3.6 design-point sweeps
All rest on `ablations_r65.csv` (64k, 40 GB SXM4). Those runs were affected by E6 (draft saw the
opening), E11 (one prompt), and the synthetic-prompt length shortfall. Specifics:

* A3 (L858-871): "c is the size of each rank's KV shard exchanged in one ring round". Wrong: c is the
  **transmission chunk** within a rotation, and the whole shard moves every rotation. "c = 2048 as the
  largest value that still fits at 1M" is wrong too: `_issue_rotation` posts every chunk in one batch,
  so all chunk buffers are live at once and memory does not depend on c.
* A4 (L873-887): "three communication strategies". Depth 2 runs the same code as depth 1
  (`ring_attention_kernel.py:344`), so there were two.
* A2 (L780-784, L844-856): "γ = 4 the acceptance/throughput optimum". Throughput was flat
  (0.81–0.86 tok/s) across γ, so "optimum" is unsupported.
* L899-903: "if communication is 12% of wall at 64k". There is no 64k profile, and the number
  contradicts the paper's own 1.2%.

#### E22 — prompts and citations
* L990-992: the synthetic repeated paragraph "matches the protocol used in prior speculative-decoding
  papers (Leviathan et al.; Miao et al.)". Misattribution: those papers evaluate on standard datasets,
  not repeated synthetic text.
* L1421-1423: "Published bf16 Llama-2-7B PPL on natural narrative text is in the 7–12 range
  (Touvron 2023)". I could not find a PG-19 perplexity in the Llama-2 paper. Verify or drop.

#### E23 — qualitative table (§5.7) — stands, with one caveat
The long-context generations are degenerate in both arms, and the 4k/8k ones are English-like (§2).
"RASD does not collapse the distribution" holds only in the weak sense that both arms are equally
degenerate. The draft's conditioning (E6) does not change the target's distribution, so the
distribution-preservation argument is unaffected.

---

## 2. The periodicity detector on the LCFM traces and generations

### 2.1 Method
* Detector: `src/analysis/repetition.py` (`row_stats`, `row_reasons`, `non_degenerate_acceptance`)
  with the campaign's absolute ceilings: verbatim span ≥ 128, word-trigram repeat > 0.60, tokens in a
  repeated 16-gram > 0.60, periodic tail > 0.50.
* Inputs: `results/final/generated/*.txt` are `decode(prompt + continuation)` (`generate_text` at
  `m4-phase-d-complete`). The continuation was recovered by re-encoding with the pinned Llama-2
  tokenizer (local HF cache, offline) and taking the last `tokens_generated` ids. For the three
  garbled long-context spec rows, the start was refined by searching the offset that best matches the
  trace.
* **Alignment check:** each round's accepted draft tokens (from `results/final/per_token/*.jsonl`)
  must reappear at their positions in the recovered continuation. Exact for 4k (48/48), 8k (32/32) and
  256k (27/29). Partial for 128k (4/6), 512k (9/28) and 1M (12/28), where byte-fallback and NUL
  characters in degenerate text do not round-trip through decode/encode. The verdicts for those rows
  do not depend on alignment: no token span longer than 2 repeats anywhere, and no repeated 16-gram.
* Sampling settings: `scripts/phase_d_short_ctx_pg19.sh` and `phase_d_rerun_session.sh` at
  `m4-phase-d-complete` set **temperature 1.0, top_p 1.0, max_new_tokens 64** (EOS early-stop on).

### 2.2 Per row (Phase D, seed 42, T = 1.0, cap 64)

| run | prompt tokens | tokens out | rounds | accept | period | agreement | first periodic | tail | longest repeat | in-16-gram | word-3gram repeat | **verdict** |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| RASD 4k PG-19 | 4,096 | 66 | 17 | 0.7059 | 5 | 0.098 | 66 (= len) | 0.00 | 3 | 0.00 | 0.02 | **no cycle** |
| TARGET 4k PG-19 | 4,096 | 64 | 63 | — | 31 | 0.091 | 64 (= len) | 0.00 | 2 | 0.00 | 0.00 | **no cycle** |
| RASD 8k PG-19 | 8,192 | 64 | 31 | 0.2581 | 31 | 0.061 | 64 | 0.00 | 1 | 0.00 | 0.00 | **no cycle** |
| TARGET 8k PG-19 | 8,192 | 64 | 63 | — | 26 | 0.053 | 64 | 0.00 | 1 | 0.00 | 0.00 | **no cycle** |
| RASD "128k" | 125,360 | **20** (EOS) | 13 | 0.1154 | 5 | 0.071 | = len | 0.00 | 1 | 0.00 | 0.00 | **no cycle** (degenerate text) |
| TARGET "128k" | 125,360 | 64 | 63 | — | 25 | 0.026 | 64 | 0.00 | 1 | 0.00 | 0.00 | **no cycle** (degenerate text) |
| RASD "256k" | 250,712 | 64 | 34 | 0.2132 | 5 | 0.051 | 58 | 0.09 | 1 | 0.00 | 0.00 | **no cycle** (degenerate text) |
| TARGET "256k" | 250,712 | 64 | 63 | — | 17 | 0.043 | 64 | 0.00 | 1 | 0.00 | 0.00 | **no cycle** (degenerate text) |
| RASD "512k" | 501,472 | 65 | 36 | 0.1944 | 30 | 0.029 | 65 | 0.00 | 1 | 0.00 | 0.00 | **no cycle** (degenerate text) |
| TARGET "512k" | 501,472 | 64 | 63 | — | 31 | 0.030 | 64 | 0.00 | 1 | 0.00 | 0.00 | **no cycle** (degenerate text) |
| RASD "1M" | 1,002,984 | 66 | 36 | 0.2014 | 13 | 0.078 | = len | 0.00 | 2 | 0.00 | 0.00 | **no cycle** (degenerate text) |
| TARGET "1M" | 1,002,984 | 64 | 63 | — | 11 | 0.038 | 64 | 0.00 | 1 | 0.00 | 0.00 | **no cycle** (degenerate text) |

"Degenerate text" means the long-context outputs are near-random multilingual fragments (for example
`…!OEO\0OfOtr\nA además over n ä H8`). They are broken, but not looping. The detector's verdict is
about cycles, and there are none. The nonzero "period" column is the argmax of near-noise agreement
(≤ 0.10), the same case `RESULTS_OCT.md` §1 calls "no cycle".

**Not testable:** the Phase C main-matrix rows behind Tables 2–3 (all seeds) have no saved generations
and no per-round traces. Only the Phase D seed-42 re-run does.

### 2.3 The 4k/8k control, other seeds (traces only, weaker test)
`results/mlsys/per_token/PG19_ctx{4k,8k}_s{123,456}.jsonl` exist, but no generations were saved. On the
accepted-token stream, with each unknown bonus token replaced by a unique sentinel (so spans cannot
cross rounds):

| run | rounds | accept | full-accept rounds | period agreement | periodic tail | verdict |
|---|---|---|---|---|---|---|
| PG19 4k s123 | 17 | 0.7206 | 10/17 | 0.044 | 0.00 | no cycle among accepted tokens |
| PG19 4k s456 | 17 | 0.6912 | 8/17 | 0.067 | 0.00 | no cycle among accepted tokens |
| PG19 8k s123 | 31 | 0.2742 | 4/31 | 0.025 | 0.00 | no cycle among accepted tokens |
| PG19 8k s456 | 33 | 0.2500 | 2/33 | 0.040 | 0.00 | no cycle among accepted tokens |

No same-session target-only partner exists for these seeds (`mlsys_pg19_short_targetonly_canary_s123`
errored), so they give acceptance but no speedup.

### 2.4 Does the 4.4× claim survive?

**The loop check: yes.** At T = 1.0 neither 4k row cycles. The greedy-loop artifact that inflated the
October greedy campaign (9/15 rows) does not apply here.

**The claim as written: no.** "Speedup at 4k native context: 4.4×" (L1189) and its uses (L1196,
L1501, L1601, L1626, [W:435]) fail for these reasons:

1. **n = 1, one session.** 2.4061 / 0.5489 tok/s = 4.383×. The same 4k spec cell ran at
   **3.40–3.50 tok/s** in a later session (seeds 123/456), 1.4× faster than the seed-42 row. A ratio is
   meaningful only within a session, and only one same-session pair exists.
2. **It exceeds what speculation can deliver.** Decode-only, the ratio is 6.12×, while the spec arm
   emitted 3.82 decode tokens per verify round. A whole spec round (4 draft forwards + a 5-token
   verify) took 1.142 s, against 1.829 s for one target-only step (0.62×). At least ~1.6× of the
   speedup is a per-step cost difference between the two code paths, not speculation. Its cause is not
   identifiable from the saved artifacts.
3. **As a control it is confounded five ways** (E15). The 4k cell is the only one with no rope
   scaling *and* a draft that saw the end of its prompt.
4. **The "1.0 median per-round α" is real**: 11 of 17 rounds fully accepted, n = 1.

**What can still be said:** "In one 4k PG-19 pair at T = 1.0, the speculative arm accepted 70.6% of
drafted tokens (3.8 tokens per verify round) and ran 4.4× faster end to end. Part of that ratio comes
from a per-step cost difference between the arms that we did not isolate." Nothing about long context
or YaRN follows from it.

---

## 3. TEXT REUSE for the MLSys paper

Rule applied: text is reusable only if (a) it makes no claim resting on old results, and (b) every
mechanism it describes matches the code at HEAD. The new setup is the Gradient-1048k target
(`rope_type: none`, θ = 3.58e9), the Llama-3.2-1B draft with `draft_window_cap` 4096, FP4 weights,
NF4 KV, and T = 0.7 sampling.

### 3.1 By section

| section (lines) | verdict | notes |
|---|---|---|
| Title, abstract, keywords (66-109) | **Drop** | Built on E1, E14, E15. Keywords name NF4 weights and YaRN. |
| §1 Bottleneck 1/2 (129-154) | **Reuse with edits** | Motivation is fine. Change the KV numbers to the new target's GQA geometry (bf16 KV for Llama-3-8B: 128 KiB/token → 128 GiB at 1M, not 512 GiB), and change "2–4× speedups" to a cited figure. |
| §1 question / system / findings / contributions (156-255) | **Rewrite** | Every finding is withdrawn or superseded. Contribution bullets are wrong (E5, E18, E21). |
| §2.1 attention efficiency (267-310) | **Reuse** | Sound background; keep the citation fixes (`c2e5ec3`). |
| §2.2 Ring Attention, system-level, long-context, KV quant (315-368) | **Reuse with edits** | Ring Attention summary OK as a description of Liu et al. Drop "outlier-keep … rank 0" detail from related work. |
| §2.2 position scaling (370-380) | **Rewrite** | Drop the "11–18× YaRN benefit" sentence (E8). Keep a short RoPE-scaling background paragraph, since the rope section of RESULTS_OCT needs it. |
| §2.2 "Limitation of prior distributed methods" (382-390) | **Drop** | Unsupported (E5). |
| §2.3 speculative decoding (395-429) | **Reuse with edits** | Fix 3.5 → 3.4 (E17), state that α in Eq. 1 is i.i.d. per token, add TriForce/MagicDec (and any 2025-26 long-context speculative work) as the LCFM camera-ready did, and remove "not measured at million-token context". |
| §2.4 long-context evaluation (434-445) | **Reuse with edits** | Update to what the MLSys paper evaluates (perplexity coherence, no RULER). |
| §2.5 gap analysis, Eq. 2, positioning table, "empirical surprise" (450-523) | **Drop** | The overlap mechanism was never implemented (E4); the table is wrong (E5); the surprise rests on E1. |
| §3.1 overview + Figs 1-2 (531-681) | **Rewrite** | Models, quantization, rope and shard math all changed. The figures' structure (ring of 8, 5-step round) can be redrawn; Fig. 2 must add the draft-token broadcast and the cap-truncated commit. |
| §3.2 ring attention (683-705) | **Rewrite** | See §3.2 below. The new text must describe the dual-cache layout, the replicated tail, W − 1 rotations per layer per forward, chunked `batch_isend_irecv`, one-rotation lookahead, and `repeat_kv` before the ring. |
| §3.3 NF4 KV (707-758) | **Rewrite** | E18, E20. Keep the three-part structure (codec, prefill chunking, sink prefix); fix every fact. |
| §3.4 verify loop (760-784) | **Reuse the core with edits** | The acceptance rule, packed forward, residual resample and broadcast rationale match HEAD. Fix the broadcast set, the RNG statement, the test claim and the γ justification. |
| §3.5 YaRN (786-802) | **Drop** | The new target needs no scaling. The rope intervention on Llama-3.1-8B (RESULTS_OCT §4) gets its own new text. |
| §3.6 ablation sweeps (804-912) | **Drop** | Old evidence, and affected by E6/E11/E21. If the MLSys paper needs design-point justification, it must be re-measured. |
| §3.7 implementation (914-937) | **Rewrite** | Versions still match the lock (torch 2.5.1+cu124, transformers 4.46.3, bnb 0.49.2, flash-attn 2.8.3). The test count (470 → 855+), scripts and tags all change. |
| §4.1 hardware (945-968) | **Reuse with edits** | Same instance type. Replace pod narrative, versions and W&B projects. |
| §4.2 models, data (970-997) | **Rewrite** | New models. Drop the misattributed synthetic-prompt protocol (E22). PG-19 aligned pools are new. |
| §4.3 matrix, §4.4 metrics (999-1078) | **Rewrite** | Metric definitions are partly reusable (tok/s, latency, α). Fix "peak memory" (E18), say throughput includes prefill, define α as per-round, and drop the profiler buckets. |
| §5 results (1083-1492) | **Drop** | All old. |
| §6 discussion (1497-1581) | **Drop** | Rests on E1/E14/E15. The "reproducibility threats" paragraph can be rewritten. |
| §7 conclusion (1586-1652) | **Drop** | |
| Appendix A repro (1673-1758) | **Rewrite** | New scripts, CSVs, tags. |
| Appendix B vanilla PPL (1760-1782) | **Drop** | E8. |
| Appendix C RULER (1784-1804) | **Drop** | Not part of the new evaluation. |
| Appendix D per-run metrics (1806-1856) | **Rewrite** | New evidence tree. |

### 3.2 Every methods sentence that does not match the current code

"Methods" means §3 plus the methods-bearing sentences in §1, §4 and the figure captions. Each entry
is the arXiv line, then the fact at HEAD.

**Models, precision, rope**
1. L537-541 / L590-591 / L607-608: target Llama-2-7B, draft Sheared-LLaMA-1.3B. Now the Gradient
   Llama-3-8B-Instruct-1048k target and the Llama-3.2-1B draft.
2. L540-541, L975-976, L983: "quantized to NF4 4-bit weights". Weights are **fp4**, pinned
   (`rasd_inference.py:1244, 1335`).
3. L547-548, L789-792, L1015-1016: "YaRN with factor f = ⌈N/4096⌉ / f = 256". The new target runs
   `rope_type: "none"`: no scaling dict, the model's own θ, and a refusal if the context exceeds
   `max_position_embeddings`.
4. L793-794: "YaRN combines NTK-aware β-scaling … attention-temperature factor". Irrelevant to the new
   target; drop.

**Sharding and ring**
5. L545, L592, L606: "each rank holds ⌈N/W⌉ tokens". It is ⌊N/W⌋ after dropping the prompt's last
   `S mod W` tokens (`:1579`). The new aligned pools land on multiples of W, but the sentence is still
   wrong.
6. L546-547: "ring attention with FlashAttention-2 inside each local block". True, but incomplete for
   decode: Q is replicated, the prefill KV rotates, and the decode tail is replicated and attended
   locally once (`ring_attention_decode`, `ring_attention_kernel.py:457`). Every rank runs the full
   model on the decode tokens.
7. L691-692: "send to rank i−1 and recv from rank i+1". Code: send to r+1, recv from r−1.
8. L689-692: "at decode round r … the r-th K/V block". Code: 7 rotations per layer per forward, each
   moving the whole shard in ⌈S_local / c⌉ chunks.
9. L693-696: "while that exchange is in flight, the local attention kernel computes … synchronize at
   the end of the round". Code: the next rotation is issued before each step's attention and waited at
   the top of the next step (prefetch_depth 1), plus a device-wide synchronize at each round start
   (`:1950`).
10. L699-701: tick-gate barrier. Absent (E3).
11. L703-705 / L858-871: "c = 2048 … size of each rank's KV shard exchanged in one ring round".
    c is the transmission-chunk size (`chunk_size = cfg.kv_block_size`). It does not bound memory.
12. Not in the paper, but required: `repeat_kv` runs before the ring (`ring_llama_attention.py:134`),
    so for the GQA target the ring payload is 32-head bf16, 4× the stored KV heads.

**NF4 cache**
13. L541-544, L593-594: "bf16 attention-sink prefix of the first 128 tokens held only on rank 0".
    Matches HEAD (`kv_outlier_prefix_size` 128, rank 0 only, `:2357`). Keep.
14. L713-716: "block size 32 … found best in M3 ablations". The block size is 32
    (`kv_block_size_nf4`), but no ablation chose it.
15. L716-717: "dequantization … inside the FlashAttention-2 kernel call". Code: the whole layer is
    dequantized in PyTorch on every `update()` (`nf4_dynamic_cache.py:285`).
16. L720-725: "re-quantizes the full per-layer block … chunks of 2048 elements … under 100 MiB".
    Code: quantizes only new tokens, in chunks of 2048 **tokens**, which bounds the prefill
    quantization transient only.
17. L733-735: 64 MiB is right for Llama-2. For the new target (8 KV heads × 128) it is 16 MiB.
18. L742-751: equation terms. It needs the 0.625 B/element NF4 with scales, the new geometry
    (32 layers × 8 KV heads × 128), and the replicated draft (Llama-3.2-1B, bf16 KV over a 4096
    window, i.e. ~128 MiB/rank, per `MORNING_REPORT` step 1a).

**Draft and verify loop**
19. L673-675, L763-765: "Every rank runs the draft model". True. Missing: the draft is conditioned on
    BOS + the most recent `draft_max_len − 1` tokens (`_draft_window`, `:868`), where
    `draft_max_len = min(native, draft_window_cap)` = 4096 in the new runs. It is windowed at prefill
    and grows by the generated tokens.
20. L646, L664, L676-678, L765-768: "rank 0 broadcasts its draft and target logits". At HEAD the
    broadcast is `draft_seq`, `target_logits_v` and `draft_logits` (`_broadcast_round_state`,
    `:900-924`).
21. L648-649, L768-769: "every rank samples using a shared RNG seed". At HEAD, ranks share the
    broadcast proposal and logits, and the accept uniforms come from each rank's identically seeded
    RNG (`_acceptance_mask`). Say both.
22. L651-653, L679-681: "accept matching prefix; KV updated by accepted tokens". At T > 0 it is
    rejection sampling with residual resample (the text at L679-681 says so; Fig. 2 does not). HEAD
    also truncates the final round to the cap (`_round_commit_plan`) and records `kv_len_before/after`.
23. L767: "~1 MB collective per round". For a 128,256-vocab target in bf16, the logits alone are
    ≈ 2.3 MB per round (5 + 4 rows). Recompute for the dtype actually broadcast.
24. L776-778: the cited unit test does not exist (E10).
25. L780-784: γ = 4 justified by the M3 grid. Drop the justification, or re-measure.

**Measurement**
26. L1062-1063: throughput definition. State that it includes prefill. The MLSys docs compute tok/s
    as `tokens_generated / time_sec` (not the rounded field) and should say so.
27. L1066-1067: α definition. At HEAD it is `acc_verified / (acc_rounds · γ)` over untruncated rounds
    (`1408d92`).
28. L1072-1073: peak memory. It is rank 0's `max_memory_allocated`. Report max over ranks from the
    memory traces, and reserved memory if occupancy is claimed.
29. L1075-1077 and L1237-1256: profiler buckets. Do not reuse. `PROFILE.md`'s own caveats apply, plus
    §1A above.
30. L1046-1054: EOS early-stop. The new runs use `ignore_eos`/fixed-length generation where
    configured. State which.

---

## 4. Dual-submission facts (from the repo only)

What the repo says:

* `README.md:17-22`: "submitted to the Workshop on Long Context Foundation Models (LCFM) @ NeurIPS 2026
  (**non-archival, double-blind**)". This is the project's own statement, not a quotation of the CFP.
* `manuscript/workshop/main.tex:1-12`: `\workshoptitle{Workshop on Long Context Foundation Models
  (LCFM) @ NeurIPS 2026}`, NeurIPS 2026 style with the `dblblindworkshop, final` options; a page budget
  of "≤ 8 pages of body … per the LCFM CFP (long-paper option)".
* Tag `lcfm-camera-ready` (2026-10-02) and commits `44038b2`, `4e5403b`, `130ef1f` ("Camera-ready: …").
  These imply acceptance, but the repo holds no acceptance notification, no OpenReview ID, and no
  statement of whether camera-ready papers appear on a public site or in any proceedings.
* `docs/dev/PUBLICATION_STRATEGY.md:118-123` (written 2026-05, before LCFM was chosen): "Workshops are
  exempt — non-archival … arXiv is allowed concurrently with all venues." That is a general planning
  assumption, not a fact about LCFM or MLSys 2027.
* `docs/dev/M5_PLAN.md:12`: MLSys 2027 deadline **2026-10-30**, "extended version of the workshop
  paper".
* arXiv: `README.md:14` says "arXiv ID pending". No arXiv ID appears anywhere in the repo. Tag
  `v1-arxiv` (2026-05-11) lists Raj Dandekar as a co-author. The later `45d377a` moved him to the
  acknowledgments, and `arxiv-submission.tar.gz` (2026-05-16) matches the sole-author `main.tex`. The
  repo cannot tell which version was posted, or whether a v2 exists.
* Affiliation differs: arXiv says "University of California, Berkeley (alumnus)", LCFM says
  "Independent Researcher".

**What cannot be determined from the repo:** whether LCFM is in fact non-archival under its CFP;
whether accepted papers are published on OpenReview or the workshop site and in what form; whether
there are proceedings; MLSys 2027's dual-submission, prior-publication and anonymity rules, including
how it treats a non-archival workshop paper and a public arXiv preprint with the same system name and
code link. **You will need to check these yourself.**

One repo-derived fact bears on overlap. The MLSys paper as decided (new target, new draft, new data)
shares with LCFM and arXiv only the *system description*. It shares no results. §3 above marks which
system text has to be rewritten anyway because it is wrong.

---

## 5. Recommendation: what an arXiv v2 should change

A correction, not a new paper. Keep the new results out of v2, so that the MLSys submission's evidence
is not pre-published. Concretely:

1. **Withdraw** these headline claims, in the abstract, §1, §5 and §7: comm ≤ 1.2% and overlap "ruled
   out" (E1–E3); "speedup regresses to 0.94×/0.97×" as an acceptance or context effect (E14); "the PG-19
   control attributes bimodality to YaRN" (E15); "4.4× at native context" as a speculation result
   (§2.4); and "40 GiB peak, half of 80" (E18).
2. **Disclose the three configuration defects** that confound every long-context acceptance number:
   the draft saw the prompt's opening (E6); YaRN was anchored on the context (E7); the "3 seeds" used
   one synthetic prompt at ~95.6% of nominal length (E11, E13). Add a short "Known defects" section
   near the top.
3. **Restate what survives**, narrowly: the system ran a Llama-2-7B target forward at a nominal 1M
   (1,002,984) tokens on 8 × A100-80GB, at 41.2 GiB allocated / ~69 GiB reserved peak per rank. Tokens
   per verify round of 1.36–1.85 were measured under the confounds above. Same-session speedups were
   1.5–1.9× at 128k–1M (Phase D, n = 1), with up to 1.84× between-session variation in per-round cost.
4. **Fix the factual methods errors** even where no headline depends on them: FP4 weights; ring
   direction and schedule; no tick gate; dequantization outside FA-2; chunk units; the
   memory-equation factor and missing scale overhead; c as transmission chunk; the non-existent test; the single-GPU HF
   ceiling; the PPL table's factor 8; the misattributed prompt protocol; and ±1.96 SEM at n = 3.
5. **Change the title.** "When Speculative Decoding Stops Paying Off at Long Context" asserts the
   withdrawn finding.
6. **Add a v2 comments line** on arXiv summarising the corrections. Align the LCFM camera-ready text
   if the workshop still accepts changes: it carries E1, E2, E6, E14, E15 and the 4.4× row.
7. **Do not** cite `BANDWIDTH_FLOOR.md` or `PROFILE.md` numbers in v2 without the corrections in §1A
   (the 4× payload, 7 rotations, the wrong module for C2, and the nproc-1 fall-through).

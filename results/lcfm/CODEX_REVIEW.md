NOT READY

# Independent review of `lcfm-camera-ready-v1` (manuscript/lcfm/main.tex)

Reviewed: tag `lcfm-camera-ready-v1` (commit c0fa9c7), on branch `review/codex-lcfm`. I recomputed
everything from the raw CSVs and sidecars with my own code: `review_scratch/recompute.py`,
`review_scratch/census.py` and `review_scratch/f5_paired.py`. Only then did I compare against the
paper and the summary documents. No GPU was used. Nothing under manuscript/, results/mlsys/,
results/lcfm/figs|tables/, src/ or configs/ was changed. Line numbers below refer to `main.tex` at
the tag.

**The short version.** Most of the arithmetic reproduces exactly: every table regenerates
byte-identically and every figure is pixel-identical. The problems are of four kinds:

1. **Labels and pools.** The F2 census mixes temperatures. The vLLM ratio is mislabelled. The F5
   128k ratio breaks H6.
2. **Mechanism.** The F4 explanation is backwards. The ~10.3 ms "per-round latency" is very likely
   a `TORCH_NCCL_BLOCKING_WAIT=1` polling artefact. The round count is 224, not 256.
3. **Disclosures that are wrong for this paper's own runs.** H4 (draft window) and the 41.2 GiB
   memory figure both describe the reviewed build.
4. **One misdescribed citation** (Meta's pass-Q).

Most fixes are wording changes and can be made length-neutral. Page 8 is full, so they have to be.
F4 needs one small piece of new evidence before its mechanism claim can stand (see B2).

---

## 2. BLOCKERS

**B1. F4's "speedup grows with context because the per-round cost is context-independent" is
backwards, and the paper's own numbers contradict it.**
*Where:* lines 135-136, 390-391 (Fig. 4 caption), 398-400, 535-536.

*What is wrong:* if a fixed cost F is removed from a step whose other terms g(C) grow with context,
the ratio (F+g)/(h+g) *shrinks* as context grows. It cannot grow. The data show the removed time is
strongly context-dependent:

| Context | old − new, target-only (s/token) | ms per round (÷224) | K+V payload per round |
|---|---|---|---|
| 128k | 2.43 | 10.9 | 0.27 GB |
| 512k | 7.64 | 34.1 | 1.07 GB |
| 1M | 13.88 | 62.0 | 2.15 GB |

The bare benchmark only covers payloads up to 266 MB (the 128k shard). So "a round costs ~10.3 ms
regardless of payload" holds at 128k only. At 512k and 1M the removed cost scales with the bytes
shipped.

*Fix:* replace the mechanistic sentence with:

> The removed time grows with context (2.4 / 7.6 / 13.9 s per token target-only), so beyond 128k the
> rotation's cost scales with the shard it ships; the flat ~10 ms per round is measured at the 128k
> payload only.

Make the same change in the Fig. 4 caption and the conclusion.

**B2. The flat ~10.3 ms per round is very likely a host-side polling artefact of
`TORCH_NCCL_BLOCKING_WAIT=1`, not NCCL or NVLink latency.**
*Where:* lines 73-75, 128-131, 345-354, 360-364, 497-498 (multi-node extrapolation), 531-533.

*Evidence:*
- `scripts/mlsys_pod_env.sh:32-35` writes `TORCH_NCCL_BLOCKING_WAIT=1` into `.pod_env.sh`.
  `scripts/mlsys_df_remote.sh:82` (the ring bench and diagnostic arms) and
  `scripts/mlsys_session_a_remote.sh:62` (the campaign stages) both source that file.
- The campaign pod's own PyTorch log confirms it was active:
  `results/mlsys/incident_20261009T001856Z/pod_results/pod_home/manifest.log:200`, "timed out in
  blocking wait (TORCH_NCCL_BLOCKING_WAIT=1)".
- `scripts/mlsys_ring_microbench.py:136-150` times a host-side `r.wait()` per round. In blocking-wait
  mode, `WorkNCCL::wait` polls for completion with a fixed millisecond sleep. To my knowledge that
  sleep is 10 ms (`kSynchronizeBusyWaitMillis`); I could not re-read the torch 2.5.1 source (see §8).
- The data have exactly the shape a 10 ms poll produces. A 4 KB round takes 10.31 ms (p90 10.40),
  while an NVLink P2P of 4 KB normally completes in tens of microseconds. 266 MB takes 10.35 ms.
  The rank-to-rank spread is ≤0.056 ms.
- The 128k old-engine intercept from DECODE_DIAGNOSIS (≈2.38 s) ≈ 224 × 10.3 ms.
- Split-KV's two all-reduces per layer are synchronous (`ring_attention_kernel.py`, `dist.all_reduce`
  without `async_op`). That is 64 waits per token, ≈0.66 s, which is also about the intercept of the
  fixed engine's own decode curve.

*Consequence:* F4's diagnosis ("latency per round, not bandwidth") and the multi-node extrapolation
may be describing a configuration flag. Every absolute decode number, the 2.80-5.32× speedups, the
F5 ratios and the 130-162× vLLM decode gap are all measured under it.

*Minimal evidence that settles it:* re-run `scripts/mlsys_ring_microbench.py` on one 8×A100 node
with `TORCH_NCCL_BLOCKING_WAIT` unset. That takes minutes. Ideally also re-time the 128k
`ring2048` / `splitkv` target-only arms (`df_arms`).

*If no rerun is possible:* disclose in §3 and §4.4 that all runs used `TORCH_NCCL_BLOCKING_WAIT=1`.
Restate the 10.3 ms as "a fixed ~10 ms per synchronous wait under our NCCL configuration", and drop
"latency, not bandwidth" and the multi-node sentence.

**B3. "8 rotations × 32 layers = 256 rounds" should be 7 × 32 = 224.**
*Where:* lines 73-74, 127-128, 345-346, 531-532.

*Evidence:* `src/models/ring_attention_kernel.py:402-455` issues a rotation only when
`step < world_size - 1`, which is 7 per layer. The project's own profile also counts ≈200-207
SendRecv kernels per decode step (`DECODE_DIAGNOSIS.md:263,389`), not 256. (256 is just the
benchmark's chosen loop length, which is fine in Fig. 3 and Table 7.)

*Fix:* say "7 rotations over 32 layers, so 224 rounds".

**B4. The vLLM "our prefill beats vLLM's entire run by 1.92 / 2.52 / 2.56×" uses prefill ÷ prefill,
not entire run ÷ prefill.**
*Where:* lines 78-79, 474-476, 538.

*Evidence:* 16.85 / 8.76 = 1.92, 222.80 / 88.56 = 2.52, and 859.60 / 335.34 = 2.56. Those are
vLLM's TTFT over RASD's TTFT (as in `CAMERA_READY_KIT.md:204-206`). The entire-run ratio
(`end_to_end_wall_s` from vllm_run2 ÷ RASD TTFT, same books) is **2.04 / 2.54 / 2.57×**.

*Fix:* keep the sentence and change the numbers to **2.04 to 2.57×**, or keep the numbers and say
"our prefill is 1.92 to 2.56× faster than vLLM's prefill".

**B5. F2: "1 of 83 sampled rows at T=0.7" includes 6 greedy rows.**
*Where:* lines 66, 112-113, 248, 259-260, 523; Fig. 1 legend.

*Evidence:* the census glob (`scripts/lcfm_figures.py:91-95`) pulls in
`repetition_QUALITY_FIX_{128K,1M}.csv`. Those are 6 rows at **T=0.0** with 8 generated tokens each.
The census also counts `RR_SANITY_*` twice (r1 and r2) and includes GATE/SANITY/SAMP_CHECK repeats.
On T=0.7 rows only, the count is **1 of 77**, and the loop is the F16 row the paper names.

*Fix:* "1 of 77", and filter the census on `temperature == 0.7`.

**B6. F2: "under greedy, 1 of 10 in the later temperature arms" counts 5 rows at T=1.0 as greedy.**
*Where:* lines 65-66, 112, 246-247; Fig. 1 caption (258-259) and x-label "greedy temperature arms".

*Evidence:* `SA_TEMP_1_0_128K_SPEC` rows have `temperature=1.0`. The greedy arm is `SA_TEMP_0`, 5
rows, and its one loop is `SA_TEMP_0_128K_SPEC_pg19_train_915_s42` (69% periodic tail). This is
confirmed by `repetition.py` and by my own periodicity check.

*Fix:* "1 of 5 greedy rows in the 128k temperature arm". Also note that the greedy arm's mean
acceptance (0.4604) is no higher than T=0.7 (0.4681). That weakens "a greedy acceptance number
measures the degeneracy" as a general statement; see 5a.

**B7. The looping F16 row is inside the F1 headline mean, but the paper says it is not.**
*Where:* lines 250-251 ("outside the headline pools") and 268 ("never smoothed into a mean"),
against Table 1 and Fig. 2.

*Evidence:* `ROPE_EXT_SAMP_F16_SPEC_pg19_train_2768_s42` (acceptance 0.7121; 77% of tokens inside a
repeated 16-gram) is one of the five rows in the factor-16 mean of 0.5336
(`lcfm_figures.py ROPE_POOLS`).
- Excluding it, factor 16 = **0.4890**, a 32.8% relative loss.
- The direction still holds (4/4 books).
- Book 2768 is also the weakest per-book drop (−0.030), so "five books of five" leans on a looping
  row for factor 16.

*Fix:* say the row is in the factor-16 mean and give both values (0.5336 with it, 0.4890 without).
Or exclude it and say so.

**B8. The H4 disclosure is wrong for this paper's runs.**
*Where:* lines 224-229 and 494-495 (§3 paragraph "The draft saw the prompt's first 4096 tokens";
limitation "every acceptance level here is a lower bound").

*Evidence:*
- `_draft_window` (BOS + the **most recent** 4095 tokens) has been in `rasd_inference.py` since
  d89cf41 (2026-10-07). It is present at fbd41fb and 429bfad, the commits recorded by every
  new-engine row, and `generate_text` passes `draft_input_ids=None`.
- The run logs show the fixed path running, e.g. `rerun_r1/group_RR_1M_SPEC.log`: "draft prefill
  over 4096 of 1047552 prompt tokens" (40 times), and the 2026-10-10 greedy session shows the same
  pattern.
- The first-4096 defect (AUDIT E6) applied to the **reviewed** version's runs (Llama-2 /
  Sheared-LLaMA, `final_matrix.csv` and similar), not to the October Llama-3 rows reported here.

*Fix:* replace the §3 paragraph with one sentence:

> In the reviewed version the draft saw the prompt's first 4096 tokens (E6); every run reported
> here uses the corrected window, BOS plus the most recent 4095 tokens.

Then delete the limitation bullet. Appendix A's E6 row is correct as a statement about the reviewed
version. *(The brief's H4 premise does not hold for this paper; see §7.)*

**B9. The 1M spec speedup is printed two ways.**
*Where:* the text says 4.37× (lines 75, 133, 396, 534). Table 2 says 4.38×. Appendix B (line 438)
says 4.375×.

*Evidence:* 6.0506 / 1.3830 = 4.3751, which rounds to 4.38.

*Fix:* "4.38" everywhere, or "2.80 to 4.38×".

**B10. F5's 128k "2.01×" compares a 3-book spec mean with a 1-book target-only value (H6).**
*Where:* lines 76, 140-141, 416-418, 536.

*Evidence:* on the only matched book (train_0), target-only / spec = 1.0596 / 0.6861 = **1.54×**.
512k (3 vs 3 books) gives 2.29× and 1M (5 vs 5, seed 42) gives 2.20×, both matched. The paper
discloses "three books against one", but H6 says pooled means over different book mixes are an
error, and the book spread (1.27-1.30×) is the same size as the effect.

*Fix:* report 128k on the matched book (1.54×), making the range "1.54 to 2.29×". Or drop 128k from
the range and keep it as a labelled unmatched figure.

**B11. Misdescription of Meta's pass-Q.**
*Where:* lines 149-150 and 166-171 ("the query replicated in the pass-Q form"; "We adopt the pass-Q
form as split-KV").

*Evidence:* arXiv:2411.01783, §3.6 and Algorithm 4 (fetched). Pass-Q decode is still a **ring**:
each of N steps sends Q to the next rank, followed by an All2All of partial outputs. The query is
not replicated with a single collective. RASD's split-KV (replicated query, local attention against
the own shard, LSE merge by collectives) matches **vLLM DCP** (all-gather Q, then all-gather of
partials and LSE plus reduce-scatter; fetched blog). Under the paper's own diagnosis, pass-Q would
not remove the per-round cost, so the attribution matters.

*Fix:*

> Meta's report introduces exact ring variants (pass-KV, pass-Q) for million-token inference;
> split-KV, which replicates the query and merges per-rank partials by log-sum-exp, is the scheme
> vLLM ships as Decode Context Parallelism.

Keep both citations.

**B12. The 41.2 GiB / 69 GiB peak is from the reviewed build, not this system.**
*Where:* lines 194-196.

*Evidence:* `LCFM_RECONCILE.md:409-413` takes it from
`results/final/memory_trace/RASD_ctx1M_phaseD_s42.rank*.json`. That trace is the Llama-2 build with
a 1,002,984-token prompt. This campaign's 1M rows (Gradient-1048k, 1,047,551 tokens) record a rank-0
`gpu_peak_mem_mb` of 26,837 MB; the per-rank maximum is not recorded.

*Fix:* either state the provenance ("in the reviewed build's 1M trace") or report "26.2 GiB allocated
on rank 0 at 1M (per-rank maximum not recorded)".

**B13. The F1 perplexity ratio is n=1, on a different prompt, near its floor, and the text implies
otherwise.**
*Where:* lines 69-71, 101-107, 283-285 (Table 1 caption "five books per arm ... paired measurement
against the native arm on the same continuation"), 297-298, 303-304, 525-526; "Acceptance is not a
proxy for perplexity" (line 71).

*Evidence:*
- `session_a_rope_fix_gate.csv` has **one row per arm**. Its prompt (sha 574e…, 130,048 tokens)
  matches none of the five acceptance prompts. `mlsys_coherence_gate.py` draws a seeded random book
  and offset from `data/processed/pg19_docs`.
- Native perplexity is **1.013**, meaning near-deterministic or memorised text, so there is almost no
  room for perplexity to move.

*Fix:* in the caption and text, say the ratio is one 1024-token continuation of one book (native
perplexity 1.013), not the five acceptance prompts. Soften "Acceptance is not a proxy for
perplexity" to "on this model and context, a perplexity check on one passage did not reveal the
acceptance loss". The direction of F1 stands; the generalisation does not.

---

## 3. SHOULD-FIX

- **S1. "Acceptance unchanged" (lines 77, 139, 142, 440, 500, 537) overclaims.** The 13 paired cells
  give a mean difference of −0.0005 with a 95% t-interval of [−0.061, +0.060]: about ±12.5% relative
  on a 0.484 base. Per-cell changes range from −31% to +30%.
  *Say:* "no detectable net change (−0.1%; 95% CI ±12% relative)".
- **S2. Seed spread "1.04×" (line 431) should be 1.02×.** The fixed-engine 1M per-seed decode means
  are 1.4370 / 1.4078 / 1.4344, so 1.4370 / 1.4078 = 1.021. The 1.04 also appears in
  RESULTS_OCT.md:1170 and does not reproduce.
- **S3. Benchmark range (line 349) should end at 10.3557, not 10.3462.** The maximum is 10.3557 ms
  (mb133); 10.3462 is the 266 MB point. The spread is still 0.4%.
- **S4. "The bare primitive accounts for the whole cost" (lines 351-354).** The 9.3 ms "profiled per
  round" comes from dividing by 256; with 224 it is ~10.6 ms. A benchmark that is *slower* than the
  profiled engine round is itself the signature of B2. Rewrite once B2/B3 are settled.
- **S5. Seed-bug description.** The run ids of the bug-hit rows end in `_s42`, so "their run ids
  said otherwise" (line 613) is inaccurate: the *stage names* said otherwise. `run_experiment.py:590-612`
  (line 609) is `_measure_target_ppl` at the tag; the bug is in `build_run_configs`, line 666
  (seed list popped from defaults) and lines 686-687 (level merged, then `run["seed"]` overwritten). The
  repo script prints the same wrong "run ids say s123/s456".
- **S6. Draft model.** "a 1.3B draft" (line 191) is the reviewed version's Sheared-LLaMA-1.3B. Every
  row here uses `meta-llama/Llama-3.2-1B` (≈1.24B). Name it.
- **S7. Rotary-scaling sentence (lines 197-199)** describes the reviewed engine (E7/E8). No reported
  run uses it: the Gradient target runs `rope_type: none`, and the F1 arms use an explicit factor.
  Delete it or label it as reviewed-version only.
- **S8. "Every row also records the engine" (lines 219-221) is false for old-engine rows.**
  `sampling_session`, `n5_extension` and `fp4_seeds` have no `ring_decode_mode`,
  `ring_layers_patched` or `git_commit` columns, and `decode_df/arms/df_arms.csv` has
  `git_commit=unknown`. All 31 r1/r2 rows and 6 top-up rows pass the check (split_kv, 32 layers,
  commit fbd41fb / 429bfad). *Say:* "every fixed-engine row".
- **S9. Fig. 1 plots only the periodic-tail share.** 4 of the 9 looping first-window rows and the one
  looping sampled row sit at ≈0, below the 0.50 line. The figure visually shows 5/10 and 0/83 against
  legends of 9/10 and 1/83. Mark the verdict with a filled/hollow marker or plot the triggering
  statistic.
- **S10. Unreferenced floats.** Table 1 (rope), Fig. 2 (rope), Table 3 (vLLM) and appendix
  Tables 4, 6-10 are never `\ref`'d (4e). Table 2's file is also `\input` twice (Tables 2 and 5).
- **S11. Drift between "a quarter" and "almost a third".** Abstract (line 69) and conclusion (524)
  say "a quarter of acceptance"; results (315-316) say "almost a third"; the data are 26.6% / 31.2%.
  Use "27 to 31%".
- **S12. Line 403 matched-cell counts.** "The matched old-against-new comparison is one cell at 512k
  and two at 1M" is true only for target-only. Spec is 3 and 13.
- **S13. "Roughly 70% of the decode deficit" (line 480)** is 69.7 / 76.5 / 81.2% at 128k / 512k / 1M.
  *Say:* "70 to 81%".
- **S14. F3 aligned book.** The single exactly aligned book is the KJV Bible (perplexity 1.65 → 1.81,
  very likely memorised). The other two sit at ≈11-13. Name it. "Within 9.5%" is a +9.5% degradation
  on the only exact evidence.
- **S15. Rotary framing.** The native arm is the shipped llama3 factor 8. The intervention changes the
  llama3 factor to 16 / 32; it is not YaRN, which the related-work paragraph (183-185) implies. YaRN
  is also not "validated by perplexity alone" (it reports passkey and benchmark results).
- **S16. Speedup formula (lines 176-177).** (1−α^{γ+1})/(1−α) is expected tokens per target forward
  (Leviathan et al.'s improvement factor also divides by (γc+1)), not "the single-device speedup".
- **S17. Appendix D.** The ablation table caption cites "§ Error Analysis", which does not exist in
  this paper. Those 64k runs predate the draft-window fix and are affected by E6 (LCFM_RECONCILE:448);
  Appendix D does not say so.
- **S18. vLLM configuration.** vLLM ran `1-default-tp8` (untuned chunked-prefill and batch-token
  settings). A reviewer will read "our prefill beats vLLM" as possibly reflecting defaults. One clause
  saying the configuration was default would pre-empt that.
- **S19. "570 s of which about 340 s is prefill" (lines 420, 504).** No subset gives 570: all 20 1M
  rows average 577 s, spec rows 524 s, target-only 740 s; prefill is 335-342 s. *Say:* "about
  520-740 s, of which about 340 s".
- **S20. Top-up table includes a plumbing row.** `TP_GATE_128K_TARGET` (Table 9) is a GATE row, which
  the paper excludes elsewhere, and it widened nothing (128k stays one book).
- **S21. SRC hygiene (not rendered).** Line 311's SRC comment is truncated (`results/lcfy...`). Several
  SRC comments point at summary documents (RESULTS_OCT.md, CAMERA_READY_KIT.md) rather than CSVs.
  `results/lcfm/FINAL_REPORT.txt` is cited in comments (lines 11, bib header) but does not exist at
  the tag.

## 4. NITS

- **Abstract grammar (lines 65-66).** "9 of 10 rows ... loop and 1 of 10 temperature-arm rows does":
  verb agreement. Same at lines 112 and 523.
- **Line 306.** "a rotary effect at n=3 arms" is an odd phrase; "across three arms" or drop it.
- **Fig. 2.** The right panel's y-label crowds the left panel. Neither panel says the right panel is n=1.
- **Line 160.** Ring Attention's own evaluation is training/long-context capability, not "a prefill
  story". "with FlashAttention-2 as the local kernel" reads as if it were Ring Attention's choice.
- **Overfull boxes.** 15 overfull \hbox of 1.85 pt each, Table 4 (lines 570-587); column widths sum
  past \linewidth.
- **Bibliography.** See §6: RoFormer author order and missing volume; StreamingLLM typed `@article`
  with journal = ICLR; `dao2023` key vs 2024 year.

## 5. Number-recompute table (Task 1)

"Mine" is from `review_scratch/recompute.py`, `census.py` or `f5_paired.py`; ✓ means it matches at the
printed precision.

**F1, rotary (Llama-3.1-8B, 128k)**

| # | Claim (line) | Paper | SRC | Mine | Match |
|---|---|---|---|---|---|
| 1 | Native acceptance (70, 104, 302, T1) | 0.7272 | ROPE_SAMP + ROPE_EXT | 0.7272 | ✓ |
| 2 | f16 / f32 acceptance | 0.5336 / 0.5006 | same | 0.5336 / 0.5006 | ✓ (f16 includes a looping row; B7) |
| 3 | ppl ratio f16 / f32 | +0.05% / +1.56% | rope_fix_gate | 1.00049 / 1.01560 | ✓ (n=1, different prompt; B13) |
| 4 | Relative loss (303, Fig. 2) | 27 / 31% | — | 26.61 / 31.16% | ✓ |
| 5 | "a quarter" / "almost a third" | — | — | 27-31% | drift (S11) |
| 6 | tok/s old engine (T1) | 0.9109 / 0.7699 / 0.7473 | ROPE CSVs | same | ✓ |
| 7 | f16-f32 gap at n=3 / n=5; "6.6% apart" | 0.0012 / 0.033; 6.6% | — | 0.0012 / 0.0330; 6.60% | ✓ |
| 8 | Direction on 5/5 books | 5/5 | — | 5/5 both factors (f16 on 2768 is the loop row, −0.030) | ✓ (B7) |

**F2, degeneracy**

| # | Claim (line) | Paper | SRC | Mine | Match |
|---|---|---|---|---|---|
| 9 | Greedy first-window 9 of 10 loop | 9/10 | natural_f1…csv | 9/10 by repetition.py from sidecars; config T=0.0 | ✓ |
| 10 | Greedy temperature arms 1 of 10 | 1/10 | SA_TEMP_{0,1_0} | 1/5 greedy + 0/5 at T=1.0 | ✗ (B6) |
| 11 | Sampled T=0.7: 1 of 83 | 1/83 | Fig. C census | 83 rows include 6 at T=0.0; 1/77 at T=0.7 | ✗ (B5) |
| 12 | Looping sampled row = ROPE_EXT F16 book 2768, 77%, no periodic tail | — | n5 repetition CSV | 0.7656 in 16-gram, tail 0.0; sidecar not released | ✓ (CSV only) |
| 13 | T=1.0 vs T=0.7 acceptance (269-270) | 0.7369 / 0.4681 | SA_TEMP_1_0; SAMP_ALIGNED+EXT 128k | 0.7369 / 0.4681 | ✓ |
| 14 | 0 of 5 degenerate in both arms | 0/5, 0/5 | — | 0/5, 0/5 | ✓ |

**F3, coherence (Gradient-1048k)**

| # | Claim (line) | Paper | SRC | Mine | Match |
|---|---|---|---|---|---|
| 15 | Aligned book ppl 1M vs 128k | 1.805 / 1.649, +9.5% | QUALITY_FIX | 1.805 / 1.649, +9.49% | ✓ |
| 16 | Other two books within 1.1%; share 91-95% | — | QUALITY_FIX; alignment.csv | −1.07% / +0.43%; overlap 91.0 / 95.3 | ✓ |
| 17 | "Exactly aligned" | train_0 | alignment.csv | continuation_sha256 equal (889a47…) in alignment.csv **and** both sidecars; offset 0 | ✓ |

**F4, ring bench and decode fix**

| # | Claim (line) | Paper | SRC | Mine | Match |
|---|---|---|---|---|---|
| 18 | Rounds per token | 256 | code | 224 (7 × 32) | ✗ (B3) |
| 19 | 4 KB / 266 MB ms per round | 10.3124 / 10.3462 | ringbench.csv | same | ✓ |
| 20 | "Range spans 10.3124 to 10.3462" (349) | — | ringbench.csv | max is 10.3557 | ✗ (S3) |
| 21 | Spread / payload range | 0.4% / 66,000× | ringbench.csv | 0.420% / 66,584× | ✓ |
| 22 | Worst rank spread | 0.056 ms | ringbench.json | 0.0562 | ✓ |
| 23 | Table 7 rows (sizes, GB/s) | — | ringbench.csv | identical | ✓ |
| 24 | Headline decode, 128k spec | 1.4779 / 0.5275, 2.80×, n=3 | matched cells | 1.4779 / 0.5275, 2.8018×, n=3 | ✓ |
| 25 | Headline decode, 128k target | 3.4925 / 1.0596, 3.30×, n=1 | — | same, 3.2960× | ✓ |
| 26 | Headline decode, 512k spec | 3.5344 / 0.9384, 3.77×, n=3 | — | same, 3.7666× | ✓ |
| 27 | Headline decode, 512k target | 9.9900 / 2.3460, 4.26×, n=1 | — | same, 4.2583× | ✓ |
| 28 | Headline decode, 1M spec | 6.0506 / 1.3830, **4.37×** (text) / 4.38 (table), n=13 | — | 4.3751× | text ✗ (B9) |
| 29 | Headline decode, 1M target | 17.0902 / 3.2107, 5.32×, n=2 | — | 5.3229× | ✓ |
| 30 | "Grows with context because cost is context-independent" | — | — | removed cost grows 2.4 → 13.9 s | ✗ (B1) |
| 31 | Diagnostic arms (Table 8) | 3.5637 / 2.8749 / 2.7985 / 1.0011; 1.240, 1.273, 3.560, 2.872, 1.027 | df_arms.csv | identical | ✓ (git_commit=unknown; S8) |

**F5 and the corrected baseline**

| # | Claim (line) | Paper | SRC | Mine | Match |
|---|---|---|---|---|---|
| 32 | Spec over target, decode-only | 2.01 / 2.29 / 2.20 | RR + topup | 2.0089 (3 vs 1 books) / 2.2891 / 2.1958 (s42, 5 vs 5); matched 128k = 1.545 | numerically ✓; 128k breaks H6 (B10) |
| 33 | End-to-end spec over target | 1.95 / 1.76 / 1.41 | — | 1.9522 / 1.7588 / 1.4123 (mean tok/s, s42) | ✓ |
| 34 | 1M row 570 s, ~340 s prefill | — | — | 577 s all / 524 spec / 740 target; 341 s | ~ (S19) |
| 35 | 512k target mean, top-up | 2.148 vs 2.346 | topup | 2.1480 vs 2.3460 | ✓ |
| 36 | 1M target mean, top-up | 3.155 vs 3.211 | topup | 3.1553 vs 3.2107 | ✓ |
| 37 | "Flattering by 8%" | 8% | — | 8.4% (or 9.2% depending on base) | ✓ |
| 38 | Book spread 512k / 1M | 1.27 / 1.30× | — | 1.2699 / 1.2988 | ✓ |
| 39 | Seed spread | 1.04× | — | 1.0207 | ✗ (S2) |
| 40 | Corrected 1M speedup | 4.375× | audit script | 4.3751× | ✓ |
| 41 | Buggy 1M speedup | 4.033× | RESULTS_OCT | 4.037× (stage-labelled pool, 15 vs 15 rows); not reproduced at 4 sig. figs | ✗ minor |
| 42 | Acceptance −0.1% (vs −10.4%) | — | audit | −0.10% (buggy −10.45%) | ✓ |
| 43 | Per seed −15.7 / +10.9 / +11.3%; 0.5246 / 0.4424 etc. (Table 6) | — | — | identical | ✓ |
| 44 | Old seed spread 23.0% | 23.0% | — | 23.03% on the 3 common books (16.3% over 13 cells) | ✓ (say "on the three books all seeds share") |
| 45 | 13 matched cells | 13 | — | 13 | ✓ |
| 46 | Bit-identical 12 tokens, tie margin 0.0000, median 2.9375 | — | DECODE_DIAGNOSIS F.4/F.5 | token_digests.json present but not independently re-derived | not verified (§8) |
| 47 | Top-up table (Table 9) | 1.051 / 2.826 / 3.126 / 3.402 / 1.847 / 2.251 | topup | identical | ✓ |

**vLLM comparison**

| # | Claim (line) | Paper | SRC | Mine | Match |
|---|---|---|---|---|---|
| 48 | Prefill vLLM / RASD (Table 3) | 16.85 / 8.76; 222.80 / 88.56; 859.60 / 335.34 | vllm_run2; RR target | identical | ✓ |
| 49 | Decode vLLM (Table 3) | 0.0081 / 0.0145 / 0.0234 | — | 0.00814 / 0.01449 / 0.02338 | ✓ |
| 50 | "Prefill beats entire run" | 1.92 / 2.52 / 2.56 | — | 2.04 / 2.54 / 2.57 | ✗ (B4) |
| 51 | vLLM decode faster (fixed / old engine) | 130-162× / 429-731× | — | 130.2 / 161.9 / 137.3; 429 / 690 / 731 | ✓ |
| 52 | Spec decode vs vLLM | 65 / 65 / 59× | — | 64.8 / 64.8 / 59.1 (13-cell mean) | ✓ |
| 53 | "Roughly 70% of deficit" | — | — | 69.7 / 76.5 / 81.2% | ~ (S13) |
| 54 | vllm_run end-to-end agrees | 2.4% / 0.1% | vllm_run | 2.45% / 0.01% / 0.08% | ✓ |

**Setup, limitations and appendix**

| # | Claim (line) | Paper | SRC | Mine | Match |
|---|---|---|---|---|---|
| 55 | 41.2 GiB / 69 GiB at 1M | — | LCFM_RECONCILE | from the Llama-2 reviewed trace; this campaign's rank 0 is 26.2 GiB | ✗ (B12) |
| 56 | Limitations cell counts (490-492) | 3/1, 3/1, 13/2; 1/3/5 books | — | identical | ✓ |
| 57 | 31 ok rows in r1/r2 | 31 | — | 31 (19 + 12, including GATE/SANITY) | ✓ |
| 58 | 1,047,551 vs 1,002,984 prompt tokens (Table 4) | — | — | 1,047,551 in every 1M row | ✓ (other side not re-derived) |
| 59 | 19.3% elementwise, 9.3 ms profiled | — | DECODE_DIAGNOSIS | profile not re-derivable from CSVs | not verified |
| 60 | HF ceiling 30.1 GiB, 7.25 tok/s | — | results/baselines/hf_ceiling.csv | not re-derived (appendix, Llama-2) | not verified |

`scripts/lcfm_figures.py` (regenerated into /tmp) and my independent code agree on every shared
number. `scripts/mlsys_seed_baseline_audit.py` agrees with my seed table exactly.

**Regeneration (1f).** All four .tex tables are byte-identical to `results/lcfm/tables/`. All four
figures, rasterised at 110 dpi, are pixel-identical to the committed PDFs.

### 1c. Pools used

All pools drop `status != ok` rows and run_ids containing GATE or SANITY.

- **Headline, old engine.**
  - 128k spec: `sampling_session/session_a_SAMP_ALIGNED_128K_SPEC.csv` (books 0 / 1 / 915) plus
    `n5_extension/session_a_SAMP_EXT_ALIGNED_128K_SPEC.csv` (2204 / 2768, no new-engine partner,
    so they drop out of the match).
  - 128k target: `SAMP_ALIGNED_128K_TARGET` (book 0).
  - 512k spec: `SAMP_ALIGNED_512K_SPEC` plus `SAMP_EXT_ALIGNED_512K_SPEC`.
  - 512k target: `SAMP_ALIGNED_512K_TARGET` (book 0).
  - 1M spec: `SAMP_1M_SPEC` (seed 42, books 0 / 1 / 915 / 2204 / 2768; the 1726 rows are errors)
    plus `fp4_seeds/session_a_FP4_SEED_1M_SPEC.csv` (seed 123 × 5 books, seed 456 × books
    0 / 1 / 915). **H1 rows excluded.**
  - 1M target: `SAMP_1M_TARGET` (books 0 / 1).
- **Headline, new engine.** `rerun_r1/session_a_RR_{128K,512K,1M}_{SPEC,TARGET}.csv` and
  `rerun_r2/session_a_RR_SEEDS_1M_SPEC.csv`. Dropped as plumbing: RR_GATE ×2 and RR_SANITY ×4.
- **Matched n.** 3 / 1 / 3 / 1 / 13 / 2 cells (lists in `recompute.py` output).
- **F5.** New-engine spec (seed 42) against r1 target plus `rerun_topup/TP_{512K,1M}_TARGET` (the
  TP_GATE row is dropped as plumbing).
- **F1.** ROPE_SAMP_{NATIVE,F16,F32}_SPEC (books 0 / 1 / 915) plus ROPE_EXT_* (2204 / 2768), n=5
  per arm; perplexity from `session_a_rope_fix_gate.csv` (3 rows).
- **vLLM.** `vllm_run2/vllm_baseline.csv` status ok (4 rows; `vllm_prefill.csv` is
  UnitMismatch and unused).

## 6. Citation table (Task 3)

| Key | Source checked | Title | Authors | Year / venue | Use in paper | Verdict |
|---|---|---|---|---|---|---|
| yang2024contextparallelism | arXiv:2411.01783 abs + PDF | ✓ | ✓ 9 authors, order ✓ | 2024 arXiv ✓ | **Misdescribed**: pass-Q decode is a Q ring plus All2All, not a replicated query | ✗ (B11) |
| lee2026dcp | vllm.ai/blog/2026-08-07-decode-context-parallelism (301 from the bib URL) | ✓ | ✓ Lee, Ha, Almog, Wilkinson | Aug 7 2026 ✓ | Motivation (KV duplication when heads < TP) ✓; split-KV ≈ DCP ✓ | ✓ (update URL to the redirect target) |
| liu2023ringattention | arXiv:2310.01889 | ✓ | ✓ | 2023 ✓ | "prefill story" loose | ✓ (nit) |
| dao2023flashattention2 | arXiv:2307.08691 | ✓ | ✓ | ICLR 2024 (from knowledge; not on the abs page) | ✓ | ✓ |
| leviathan2023fast | arXiv:2211.17192 | ✓ | ✓ | ICML 2023 ✓ | formula overstated (S16) | ✓ |
| chen2023accelerating | arXiv:2302.01318 | ✓ | ✓ 6 authors | 2023 ✓ | ✓ | ✓ |
| cai2024medusa | arXiv:2401.10774 | ✓ | ✓ 7 authors | 2024 ✓ | ✓ | ✓ |
| dettmers2023qlora | arXiv:2305.14314 | ✓ | ✓ | 2023 ✓ (NeurIPS 2023 version exists) | NF4 ✓ | ✓ |
| xiao2024streamingllm | arXiv:2309.17453 | ✓ | ✓ | ICLR 2024 ✓ | ✓ | ✓ (should be `@inproceedings`) |
| su2024rope | arXiv:2104.09864 + Crossref 10.1016/j.neucom.2023.127063 | ✓ | Bib uses the arXiv order; Neurocomputing order is Su, **Ahmed (Murtadha)**, Lu, Pan, Bo, Liu | Neurocomputing 568:127063, 2024 | ✓ | ✗ minor: match the order to the venue, add volume and article |
| peng2023yarn | arXiv:2309.00071 | ✓ | ✓ | 2023 ✓ | "validated by perplexity alone" overstated; the F1 intervention is not YaRN (S15) | ~ |
| rae2019pg19 | arXiv:1911.05507 | ✓ | arXiv lists 4 authors; bib adds Chloe Hillier (the ICLR version reportedly includes her) | ICLR 2020 | ✓ | **unverified** (OpenReview, dblp and Semantic Scholar blocked or rate-limited) |

**3c.** All 13 bib entries are cited. Every `\cite` key resolves. The clean build reports no
undefined citations.

## 7. Hazards H1-H6

| Hazard | Handled correctly? | Evidence |
|---|---|---|
| **H1** seed bug | **Yes** | Bug confirmed in `build_run_configs` (line 666; lines 686-687). The SA_SEED_123 / 456 rows have `generated_tokens_sha256` byte-identical to SAMP_1M_SPEC seed 42 on all 5 books, and run_ids `_s42`. They are excluded from every old-engine pool (`lcfm_figures.py OLD_POOLS`, audit script, my code). The paper's line citation (590-612) and "run ids said otherwise" are wrong (S5). |
| **H2** vllm_run | **Yes** | `vllm_run/` ttft ≈ 1.79e9 (absolute clock), decode_only negative; only `vllm_run2` is used; end-to-end agrees within 2.45% / 0.01% / 0.08%. |
| **H3** Llama-2 CSVs | **Yes** | Neither `lcfm_figures.py` nor any table reads `llama2_matrix_multiseed.csv` or `pg19_multiseed.csv`; the audit script lists them as excluded; Llama-2 appears only in the labelled appendix ceiling test and Appendix D. |
| **H4** draft saw first 4096 | **No: the premise is false for this paper, and the paper's disclosure is wrong** | The first-4096 defect is real for the reviewed version (E6). The camera-ready rows ran with `_draft_window` = BOS + last 4095 (code at fbd41fb / 429bfad; `generate_text` passes `draft_input_ids=None`; logs show "draft prefill over 4096 of 1047552 / 130048 prompt tokens"). Old-engine sampling rows have no committed logs, but were produced on 2026-10-11/12, after the fix (2026-10-07), and the 2026-10-10 log already shows the fixed path. See B8. |
| **H5** FP4 / NF4 | **Yes** | Configs set `quantize_target: true` (bitsandbytes default `fp4`, `rasd_inference.py:1240`) and `kv_quant: true`; sidecars record `weight_precision: fp4` and `kv_dtype: nf4`; df_arms has `quantize_target=True`, `kv_quant=True`. The paper says FP4 weights and NF4 KV in the abstract, §3, and the Table 1 and 3 captions. |
| **H6** same book sets | **Partly** | Headline (F4) and vLLM are on matched cells and books ✓. F5 128k divides a 1-book target by a 3-book spec mean (B10). |

## 8. What I could NOT verify, and why

- **The PyTorch blocking-wait sleep constant.** The brief allows network access only for citations,
  so I did not fetch the torch 2.5.1 `ProcessGroupNCCL.cpp`. The local wheels' headers do not contain
  it. B2 rests on the recorded environment, the pod log and the timing shape. No GPU, so I could not
  run the decisive microbenchmark.
- **The rescore of the looping F16 row.** `n5_extension/` ships no token sidecars, so it is confirmed
  from `repetition_ROPE_EXT_SAMP_F16_SPEC.csv` only. The same applies to all n5_extension rows in
  the census.
- **The draft window for old-engine rows.** sampling_session, n5 and fp4_seeds have no committed
  logs or git commit. I infer it from production dates and the adjacent session logs.
- **The F1 perplexity-gate book.** Which book and offset the seeded gate drew is not recorded in the
  CSV.
- **The kernel-equivalence claims.** "Bit-identical for 12 tokens, tie margin 0.0000, median
  2.9375", the 19.3% elementwise share and the 9.3 ms profiled round come from DECODE_DIAGNOSIS and
  profiles that cannot be re-derived from the released CSVs. I did not re-derive them from
  `token_digests.json`.
- **Appendix-only sources.** The HF ceiling (30.1 GiB, 7.25 tok/s) and the Appendix A/D audit items
  (E1-E23) were not re-derived.
- **Two citation details.** The PG-19 ICLR author list (sources blocked), and the FA-2 ICLR venue,
  which I took from knowledge.

## 5a-5c. Reviewer-style read (Task 5)

**The three weakest points.**

1. **The F4 mechanism** (B1 + B2 + B3). A systems reviewer will see that 10 ms for a 4 KB NVLink
   send is about 1000× too slow, ask about NCCL settings, and notice that the speedup growing with
   context contradicts a context-independent cost. *Not answered.*
2. **F1's perplexity side** (B13). One passage, ppl 1.013, a different prompt from the acceptance
   rows. "Acceptance is not a proxy for perplexity" is a general claim from n=1. *Not answered*;
   the table caption implies the opposite.
3. **F2 as a general statement.** The cleanest controlled comparison in the release (the 128k
   temperature arm, same 5 books) shows greedy acceptance 0.460 against 0.468 at T=0.7. Greedy is
   not inflated there; 1 of 5 greedy rows loops. The 9/10 evidence comes from a different
   protocol: 1024-token generations on a different 10-book pool. *Partly answered*: the paper
   labels the census, but the abstract states the inference generally.

**Unclear or misleading sentences.** Lines 250-251 and 268 (B7). Lines 224-229 (B8). Line 403
(S12). Lines 351-354 (S4). The Table 1 caption on perplexity pairing (B13). "Removing it made our
decode merely slow" may be literally true for a reason the paper does not state (B2).

**Typos and broken references.** No "??" in the PDF. Dangling "§ Error Analysis" in Appendix D
(S17). Agreement errors (Nits). Units are consistent (s/token, ms/round, GiB vs GB used
correctly).

## Task 4 summary

- **4a.** A clean tectonic build into /tmp: 0 undefined references or citations; 15 overfull boxes,
  all 1.85 pt, in Table 4.
- **4b.** pdftotext: §6 Conclusion ends on page 8, and "References" starts at the top of page 9 ✓.
  Page 8 is full, so fixes must be net-neutral.
- **4c.** The author block is present; the NeurIPS 2026 / LCFM footer is present; no line numbers,
  "TODO", "XXX", "FIXME", "anonymous" or watermark. The committed main.pdf text is identical to the
  fresh build ✓.
- **4d.** No `---` in main.tex, no U+2014 in main.tex, the included tables or the PDF text ✓.
- **4e.** Floats are numbered in order. Several are unreferenced (S10). Axes are labelled with units
  (Fig. 1's y-axis is a unitless share). Readable at print size.

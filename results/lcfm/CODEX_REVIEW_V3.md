NOT READY

# Independent re-review of `lcfm-camera-ready-v3` (manuscript/lcfm/main.tex)

**Scope.** Tag `lcfm-camera-ready-v3` (e139981), reviewed on branch `review/codex-v3`. I recomputed
everything from the raw CSVs, sidecars and bw_diag rows with my own code, in
`review_scratch_v3/{census_v3,bw_recompute,token_streams}.py` with their `*_out.txt` outputs. My v1
scripts in `review_scratch/` reproduce their committed output byte for byte on v3. No GPU was used
and nothing protected was edited. Line numbers below refer to `main.tex` at the tag.

**The short version.**
- **Fixed.** 10 of the 13 previous blockers are fixed and verified. The F2 regrouping, the F5
  matched ratios, the vLLM ratio definitions, the 224 rounds, the draft-window disclosure, the
  single-passage perplexity disclosure and the pass-Q attribution are all correct now.
- **What's wrong.** Three things block submission, and all three are text fixes that need no new
  experiments:
  1. A `%` comment swallows half a sentence, so the PDF renders a fragment in §4.4.
  2. The headline F4 mechanism ("the bottleneck was a host-side poll, not the ring"; "the poll is
     the whole story at 128k") is contradicted by the paper's own diagnostic.
  3. One new quantitative claim about the benchmark is false.
- A set of smaller v2/v3 patches left stale or inconsistent text behind; see SHOULD-FIX.
- Page 8 is full, so every fix has to be length-neutral.

---

## 2. Task 0: previous items

| Item | Status | Evidence (v3 line; recomputed value) |
|---|---|---|
| B1 backwards "context-independent cost" | **VERIFIED FIXED** | No "context-independent", "latency, not bandwidth" or "overlap" remains in the main text, captions or figures (pdftotext grep; Appendix A uses "overlap" only for reviewed-version E4/E5 rows). Removed time 2.43 / 7.64 / 13.88 s and 10.9 / 34.1 / 62.0 ms per round reproduce. |
| B2 blocking-wait artefact | **FIX INTRODUCED A NEW PROBLEM** | The flag is disclosed everywhere, and the constant is confirmed (`kSynchronizeBusyWaitMillis = 10`, ProcessGroupNCCL.cpp:426 at torch v2.5.1, fetched; torch==2.5.1 pinned). But the abstract (74), the F4 title (124) and lines 414-423 / 571-573 overstate it. See NB2. |
| B3 224 rounds | **VERIFIED FIXED** | ring_attention_kernel.py:402 loops over ranks; 426 and 455 rotate only while `step < world_size-1`. 7 × 32 = 224. The "256" survivors are only the bench's loop count. |
| B4 vLLM ratio label | **PARTLY FIXED** | Text is correct (2.04 / 2.54 / 2.57 run-over-prefill; 2.52 / 2.56 prefill-over-prefill). But Table 4's header calls the first ratio column "vLLM run / RASD prefill", and the table prints the 128k prefill ratio **1.92×** that §4.6 says is not quoted. See S-V1. |
| B5 1 of 77 | **VERIFIED FIXED** | By each row's own recorded temperature: 77 rows at T=0.7, 1 loop. The 6 dropped rows are T=0.0 with 8 tokens. (Those 77 contain 75 unique run ids and 9 plumbing or repeat rows; nit.) |
| B6 1 of 5 / 0 of 5 | **VERIFIED FIXED** | SA_TEMP_0: 1/5 loops (train_915), confirmed from the sidecar by repetition.py **and** my periodicity check. SA_TEMP_1_0: 0/5. |
| B7 looping row in the F1 mean | **PARTLY FIXED** | Both values given: 0.5336 and 0.4890 (32.8%). Lines 269-270 and 317-324 are correct. But line 289 still says a looping row is "never smoothed into a mean", while the conservative claim (0.5336) is exactly such a mean. See S-F1. |
| B8 draft-window disclosure | **VERIFIED FIXED** | Line 243-245 is correct (rasd_inference.py:878 `_draft_window`; `src/` unchanged since v1; logs unchanged). Only Appendix A E6 retains the defect, and it adds an unsupported "lower bounds" (nit). |
| B9 4.37 vs 4.38 | **VERIFIED FIXED** | 4.38 everywhere; 6.0506 / 1.3830 = 4.3751. |
| B10 F5 128k | **VERIFIED FIXED** | 1.5445 → 1.54 on train_0; 512k 2.2891; 1M 2.1958. End-to-end 1.4956 / 1.7588 / 1.4123 → 1.50 / 1.76 / 1.41. |
| B11 pass-Q | **VERIFIED FIXED** | Lines 155-159 and 175-180 match arXiv:2411.01783 §3.6 / Alg. 4 (Q passed around the ring plus all-to-all) and the vLLM DCP blog (replicated query, LSE merge). |
| B12 41.2 GiB | **PARTLY FIXED** | 29.6 / 32.8 / 35.6 / 57.0 GiB reproduce, but they come from the **ring2048 diagnostic rows** (bw_diag_p2, commit 135f8fb), not the split-KV campaign rows. Those record only rank 0 (26.2 GiB max). Line 207 says "this campaign's rows". See S-M. |
| B13 F1 perplexity n=1 | **VERIFIED FIXED** | Abstract, F1, Fig. 2 label, Table 3 caption, §4.2 and Limitations all say one passage, a different prompt, native perplexity 1.013. |
| S1 "unchanged" | **PARTLY FIXED** | The F5 body (148-150) is correct, but "unchanged" survives at 80, 144 (F5 title), 476, 535 and 574. |
| S2 seed spread | **VERIFIED FIXED** | 1.02 (1.4370 / 1.4078). |
| S3 benchmark range | **PARTLY FIXED** | Same pattern as before: 10.3347-10.3520 are the 4 KB and 266 MB endpoints; the blocking maximum is **10.3802** (66 MB). Fig. 4 itself correctly says "10.33 to 10.38". |
| S4 9.3 ms paragraph | **VERIFIED FIXED** | Gone. |
| S5 seed-bug citation | **VERIFIED FIXED** (nit) | Lines 693 / 713 / 714 are correct at v3. The documents-based run id is built at **line 722**; 725 is the no-documents branch. |
| S6 draft model | **VERIFIED FIXED** | `meta-llama/Llama-3.2-1B`. |
| S7 rotary-scaling sentence | **PARTLY FIXED** | Line 210-212 still describes a "fixed factor 8 at 8k and above, anchored on each context's own length" branch that no reported row uses. The F1 arms set an explicit factor 16 / 32 with anchor 8192. |
| S8 "every row records the engine" | **VERIFIED FIXED** | "Every fixed-engine row". All 31 r1/r2 rows and 6 top-up rows, plus all bw_diag rows, record mode, 32 layers and a commit. |
| S9 Fig. 1 markers | **VERIFIED FIXED** | Diamonds mark the verdict. |
| S10 unreferenced floats | **PARTLY FIXED** | Main-text floats are referenced, but Tables 7-13 (`tab:topup, bench, bwarms, rotationcost, arms, memory`) have 0 `\ref`. The decode-vs-context figure (`figA`) is **not included anywhere**, yet line 410-411 says the cells are "plotted in Appendix C" (V2_REPORT says it moved there). Table 3's caption references itself. |
| S11 "a quarter" | **VERIFIED FIXED** | "27 to 31%" throughout. |
| S12 matched-cell counts sentence | **FIX INTRODUCED A NEW PROBLEM** | The sentence is now cut in half by a comment and renders as a fragment. See NB1. |
| S13 70 to 81% | **VERIFIED FIXED** | 69.7 / 76.5 / 81.2%. |
| S14 Bible caveat | **VERIFIED FIXED** | Book identity confirmed from `data/processed/pg19_1m/documents.json`; alignment from alignment.csv `continuation_sha256` and offset 0. |
| S15 YaRN framing | **VERIFIED FIXED** (nit) | New unsupported generalisation: "usually validated by a perplexity curve alone in practice" (196-197). |
| S16 formula | **VERIFIED FIXED** | Line 186-188. |
| S17 Appendix D | **VERIFIED FIXED** | Caption now points to `\S\ref{app:ablations}`; Appendix D discloses E6. |
| S18 vLLM defaults | **VERIFIED FIXED** | Line 499 and Table 4 caption. |
| S19 520-740 s | **VERIFIED FIXED** | 524 / 740 s; prefill 341 s. |
| S20 TP_GATE row | **VERIFIED FIXED** | Omitted. |
| S21 SRC hygiene | **PARTLY FIXED** | SRC comments now name artifacts, but lines 142, 378 and 424 credit `lcfm_figures.py removed_time()` / `bench_two_modes()` for numbers that script does not produce (1.72, 1.75, 1.93, 3.79, 16.43, 15.96). They are hand-transcribed from BLOCKING_WAIT_DIAG.md. The header (19-20) claim "all of them are generated ... by lcfm_figures.py" is therefore false. |
| Nits (grammar, "across three arms", Ring Attention, overfull boxes, bib) | **VERIFIED FIXED** | 0 overfull boxes. StreamingLLM `@inproceedings`. `dao2024`. RoFormer order, vol. 568, art. 127063 match Crossref. The vLLM URL returns 200. Fig. 2's right y-label still crowds the left panel. |

---

## 3. BLOCKERS

**NB1. A rendered sentence fragment in §4.4.**
*Where:* main.tex:431-436.

*What is wrong:* the `%` lines swallow "…And the matched old-against-new comparison is one". The
PDF prints:

> (Appendix C). cell at 512 k and two at 1 M, because that is how many books the old engine
> measured there, …

(pdftotext, page 7). "Two caveats we state rather than bury" (427) now introduces only one caveat.
The sentence is also still the S12 inaccuracy (true only for target-only; spec is 3 and 13).

*Fix:* delete lines 431-436 and make line 427 "One caveat we state rather than bury". The matched
counts are already in Limitations (520-521).

**NB2. The F4 headline overstates the poll, and the paper's own diagnostic contradicts it.**
*Where:* abstract 73-74 ("our ring's decode bottleneck was not the ring but a host-side poll"),
F4 title 124 ("a host-side poll, not the ring"), 414-416 ("10.9 ms per round, which is the poll
period itself"), 422-423 ("the poll is the whole story at 128k"), 571-573 ("the poll is essentially
all of it at 128k").

*Evidence (`bw_recompute.py`):*
- At 128k, removing the flag saves the ring 1.749 s (p1) or 1.869 s (p2).
- Split-KV removes 2.433 s (headline) or 2.71 / 2.79 s (same-session, bw_diag).
- So the poll is **65-72%** of what split-KV removes at 128k, not "the whole story". The "1.4%"
  agreement is poll-model versus flag-removal on the ring, not versus the split-KV saving.
- At 512k and 1M the paper itself says the poll is a small share: 3% at 1M.
- The §4.4 heading ("The ring's key/value rotation was the decode step") contradicts the F4 title.

*Fix:* abstract:

> our ring's decode step was dominated by the key/value rotation, and at 128k mostly by a host-side
> poll: …

F4 title: "The ring's rotation, inflated by a host-side poll". In 414-423:

> at 128k removing the flag saves 1.75 s of the 2.43 s split-KV removes (about 70%); at 1M it
> saves 3%.

Make the same change at 571-573.

**NB3. A false quantitative claim about the benchmark.**
*Where:* main.tex:376-377, "the two modes' post-sync times agree to within 3% at every payload, so
the wire never changed".

*Evidence:* `bw_diag_p1/bench` post-sync times, blocking against no flag:

| Payload | Blocking (ms) | No flag (ms) | Difference |
|---|---|---|---|
| 4 KB | 0.1487 | 0.1340 | **+11.0%** |
| 33 MB | 0.8373 | 0.8135 | +2.9% |
| 66 MB | 1.4371 | 1.3633 | **+5.4%** |
| 133 MB | — | — | +0.5% |
| 266 MB | — | — | +0.8% |

*Fix:*

> agree to within 1% at 133 MB and 266 MB and within 11% at every payload

or drop the clause; "the wire never changed" does not need it.

---

## 4. SHOULD-FIX

- **S-P (prediction arithmetic, line 416).** The printed formula "224 × (10.3347 − 2.6191) ms" is
  1.728 s, which rounds to **1.73**, not 1.72. The 1.72 and the "1.4%" come from
  224 × (10.3347 − **2.6404**), the *blocking*-mode post-sync time (BLOCKING_WAIT_DIAG.md:279).
  *Fix:* print 2.6404, or keep 2.6191 and say "1.73 s against 1.75 s, within 1.2%". Also say it is
  session p1: in session p2 the same prediction is **7.5%** off (3.7950 − 1.9257 = 1.869 s).
- **S-Mix (sessions mixed, lines 416 and 429).** 1.75 s is p1 (3.7805 / 2.0319). "1.93 s ... against
  3.79 s" is p2 (1.9257 / 3.7950). Table 10 shows the p1 pair, not the p2 pair the text quotes. Use
  one session, or name both.
- **S-1M (overclaim, lines 419-421 and the Table 11 caption).** "Most of what split-KV removes at
  [1M] is the rotation's own transfer and per-chunk work" is an inference from one row per mode,
  and transfer is not most of it:
  - 224 rounds × 2.15 GB at the bench's 101.7 GB/s = **4.7 s** of the 13.9 s removed.
  - "Per-chunk work" is unmeasured.
  - The diagnostic itself says the 1M poll saving (0.47 s) is 2.4× below its model and calls it
    unresolved at n=1 per arm (BLOCKING_WAIT_DIAG.md:290-293).

  *Fix:* "at 1M the poll is about 3% of decode (one row per mode); the rest of the removed time is
  not attributed by these measurements."
- **S-BW ("bandwidth-bound above a few hundred kilobytes", line 133).** No payload was measured
  between 4 KB and 33 MB, and the curve is not bandwidth-linear: 133 MB costs 2.53 ms and 266 MB
  costs 2.62 ms. *Fix:* "rises with payload to 2.62 ms (101.7 GB/s) at 266 MB".
- **S-V1 (vLLM 128k, lines 506-509 and Table 4).**
  - The 128k prefill ratio is "not quoted" because the 128k first-token time varied 8.7-12.4 s, but
    the 128k **run-over-prefill** 2.04× (abstract, conclusion, the 2.04-2.57 range) divides by the
    same 8.76 s. On 12.4 s it would be 1.44×.
  - Table 4 also prints the dropped 1.92×, and its first ratio column is headed "vLLM run / RASD
    prefill" when it is prefill / prefill.

  *Fix:* quote 2.54-2.57× at 512k / 1M (or a 128k range), relabel the column "vLLM prefill / RASD
  prefill", and drop or footnote the 1.92.
- **S-M (memory provenance, lines 207-208).** Say "the 1M ring2048 diagnostic rows (worst rank)".
  The split-KV campaign rows record only rank 0 (26.2 GiB).
- **S-F1 (line 289).** "A looping row is named wherever one appears, never smoothed into a mean"
  contradicts 317-324. Write instead: "…and where one sits inside a mean, the mean is reported with
  and without it."
- **S-U ("unchanged", lines 80, 144, 476, 535, 574).** Replace with "no detectable change" in the
  same places. ±0.061 / 0.4839 is ±12.6%, so "about ±13%" is more exact than ±12%.
- **S-Spec (line 773).** "On a greedy seeded row" is wrong: both p2 speculation rows record
  `temperature=0.7` (`bw_diag_p2/spec/*/df_arms.csv`). Write "on a seeded T=0.7 row". The
  divergence at index 8 itself is confirmed, and the target-only streams are identical across
  modes, sessions and boxes (`token_streams_out.txt`).
- **S-Bench caption (Table 7, lines 745-746).** "Worst rank-to-rank spread 0.056 ms" is from the
  *old* decode_df bench. The new two-mode bench the table shows has a worst spread of 0.091 ms
  (blocking) and 0.110 ms (no flag).
- **S-Env (line 235).** `scripts/mlsys_pod_env.sh:32` at v3 is a comment, and the script now
  defaults to **async** (`MLSYS_NCCL_WAIT_MODE:-async`). Cite the file at the run commit
  (`fbd41fb:scripts/mlsys_pod_env.sh:34`) and use the past tense.
- **S-Fig (line 410-411).** "With the same cells plotted in Appendix C": there is no such figure.
  Re-include `figA_decode_vs_context.pdf` in Appendix C or delete the clause.
- **S-Refs.** Tables 7-13 are never `\ref`'d; add one clause each in Appendices B and C. Table 3's
  caption should not reference Table 3.
- **S-Every ("every timing", lines 160, 234, 427, 524).** The paper quotes flag-free timings
  (0.134 ms, 2.619 ms, 1.93 s, 15.96 s) and vLLM timings. *Write:* "every RASD timing except the
  no-flag diagnostic arm".
- **S-SRC.** Point the SRC comments at lines 142, 378 and 424 at `BLOCKING_WAIT_DIAG.md` and the
  bw_diag CSVs (not `removed_time()`), and fix the header claim (19-20).
- **S7 remainder (210-212).** Delete the unused "factor 8 at 8k" branch description.

## 5. NITS

- Line 498: "NCLL" should be "NCCL".
- Line 673: missing period ("NF4 KV cache Acceptance").
- Line 599: duplicated `\setlength{\tabcolsep}`.
- Line 725 should be 722 (S5).
- Line 474: "rather than 4.03×". The stage-labelled buggy pool gives 4.037, so "4.04" (RESULTS_OCT
  says 4.033; not reproduced at the printed precision).
- Line 268: "too short for a detector whose shortest repeat span is 128 tokens". The 16-gram and
  periodicity rules work on 8+ tokens, and the T=0.7 census keeps six 64-token SANITY rows. Say
  "too short to judge (8 tokens)".
- Line 196-197: unsupported "usually … alone in practice".
- Appendix A E6 (608): "so its acceptance levels are lower bounds" is unsupported.
- Line 337: "both arms" should be "all three arms".
- Fig. 2: the right panel's y-label overlaps the left panel.
- Fig. 1: the legend box sits over the T=0.7 column near y≈0.47.
- 128k TTFT range (507-508): the bw_diag 128k rows go as low as 7.0 s (splitkv p1), so "7.0 to
  12.4 s" is more accurate.
- The conclusion SRC comment (577) names "Tables 2 and 5 and Figures 1 to 4"; it is stale and not
  rendered.

## 6. Task 1: number table

Recomputed by my code; ✓ means it matches at the printed precision.

**Abstract, introduction and F2-F3**

| Claim (lines) | Paper | Mine | Match |
|---|---|---|---|
| F2 counts (65-66, 109-110, 264-265, 280, 557-558) | 9/10, 1/5, 0/5, 1/77 | 9/10, 1/5, 0/5, 1/77 (per-row temperature; 75 unique ids in 77) | ✓ |
| Excluded rows (268) | 6 rows, T=0.0, 8 tokens | 6 QUALITY_FIX rows, T=0.0, 8 tokens | ✓ |
| Looping row; 77%; zero tail (265-267) | ROPE_EXT…F16…2768 | 0.7656, tail 0.0 (CSV; no sidecar released) | ✓ |
| T=1.0 vs T=0.7 acceptance; 0/5 both (290-291) | 0.7369 / 0.4681 | 0.7369 / 0.4681 | ✓ |
| F3 (118-119, 350-355) | 1.805 / 1.649, +9.5%; ≤1.1%; 10.6 / 12.8 | 1.805 / 1.649, +9.49%; −1.07 / +0.43%; 10.58-10.63 / 12.71-12.85 | ✓ |
| F3 aligned book | train_0 = KJV Bible | sha 889a47… equal in alignment.csv and both sidecars; offset 0; documents.json title | ✓ |

**F1, rotary**

| Claim (lines) | Paper | Mine | Match |
|---|---|---|---|
| Acceptance (102, 314) | 0.7272 / 0.5336 / 0.5006 | same | ✓ |
| Relative loss (69, 101, 314, 559) | 27-31% (with the loop row) | 26.61 / 31.16% | ✓ |
| Without the loop row (320-321) | 0.4890, 32.8% | 0.4890, 32.76% | ✓ |
| Loop row is highest in arm (319) | 0.7121 | f16 values 0.317 / 0.572 / 0.611 / 0.456 / 0.712 | ✓ |
| Per-book direction (316) | 5/5 | 5/5 both factors | ✓ |
| Factor gap (325-327) | 6.6%; 0.0012 → 0.033 | 6.60%; 0.0012 / 0.0330 | ✓ |
| Perplexity ratio (70, 103-104, 315) | +0.05% / +1.56%, native 1.013 | 1.00049 / 1.01560; 1.013; 3 gate rows, prompt 574e… | ✓ |
| Arms differ only in rope_factor | — | configs differ only in `rope_type` / `rope_factor` / `rope_anchor_base` (native = shipped llama3 factor 8); same prompts, draft, revision | ✓ |

**F4, ring bench and decode fix**

| Claim (lines) | Paper | Mine | Match |
|---|---|---|---|
| Rounds per token (74, 125, 365, 566) | 224 | 224 (code) | ✓ |
| Blocking bench (131, 372-373) | 10.3347 / 10.3520; 0.4% | same; range max 10.3802; 0.44% | ✓ (S3) |
| Payload range (374) | 66,584× | 66,584× | ✓ |
| No-flag bench (132, 375-376) | 0.134 → 2.619 ms; 101.7 GB/s | 0.1340 / 2.6191; 101.69 | ✓ |
| Post-sync agreement (377) | within 3% at every payload | 11.0 / 2.9 / 5.4 / 0.5 / 0.8% | ✗ (NB3) |
| "Bandwidth-bound above a few hundred KB" (133) | — | no data between 4 KB and 33 MB | ✗ (S-BW) |
| Headline speedups (77-78, 135-136, 411-412) | 2.80 / 3.77 / 4.38; 3.30 / 4.26 / 5.32 | 2.8018 / 3.7666 / 4.3751; 3.2960 / 4.2583 / 5.3229 | ✓ |
| Removed time (137, 414, 417) | 2.43 / 7.64 / 13.88 s; 10.9 / 34.1 / 62.0 ms | 2.4329 / 7.6440 / 13.8795; same | ✓ |
| Prediction vs measured (416) | 1.72 vs 1.75, 1.4% | formula as printed = 1.728; with 2.6404 = 1.724; p1 1.749 (1.2-1.4%); p2 1.869 (7.5%) | ✗ minor (S-P) |
| "Poll is the whole story at 128k" (422, 572) | — | 65-72% | ✗ (NB2) |
| 1M flag effect (419-420, 573) | 16.43 → 15.96, 3% | 16.4348 → 15.9613, −2.88% | ✓ |
| 128k flag effect (428-429) | 1.93 vs 3.79 | p2 1.9257 / 3.7950 | ✓ (S-Mix) |

**F5 and the corrected baseline**

| Claim (lines) | Paper | Mine | Match |
|---|---|---|---|
| Spec over target (79, 145-147, 447-448) | 1.54 / 2.29 / 2.20; 1 / 3 / 5 books | 1.5445 / 2.2891 / 2.1958 | ✓ |
| End-to-end (451) | 1.50 / 1.76 / 1.41 | 1.4956 / 1.7588 / 1.4123 | ✓ |
| Row time (451-452, 539) | 520-740 s, ~340 s prefill | 524 / 740; 341 | ✓ |
| Top-up means (459-462) | 2.148 / 2.346; 3.155 / 3.211; 8% | same; 8.4% | ✓ |
| Spreads (463) | 1.27 / 1.30; seeds 1.02 | 1.2699 / 1.2988; 1.0207 | ✓ |
| Acceptance change (148-150) | −0.1%, about ±12%; −31 to +30% | −0.10%; ±0.061 = ±12.6%; −31.0 / +30.2% | ✓ (say ±13%) |
| Corrected baseline (474-475) | 4.38 vs 4.03; −0.1 vs −10.4% | 4.3751; buggy 4.037; −10.45% | 4.03 ✗ nit |
| Seed table (477-478, Table 2) | 23.0%; −15.7 / +10.9 / +11.3 | identical (23.0% on the 3 common books) | ✓ |
| Kernel equivalence (486-488) | 12 tokens, 0.0000, 2.9375 | not re-derived (DECODE_DIAGNOSIS) | — |

**vLLM**

| Claim (lines) | Paper | Mine | Match |
|---|---|---|---|
| Run over prefill (81, 504, 575) | 2.04-2.57× | 2.041 / 2.537 / 2.572 | ✓ (S-V1) |
| Prefill over prefill (506) | 2.52 / 2.56 | 2.516 / 2.563 | ✓ |
| 128k TTFT variation (507-508) | 8.7-12.4 s, ~40% | RR_128K_TARGET 8.76; bw_diag p2 12.39; ratio 1.41 | ✓ (nit: min 7.0) |
| Decode gap and deficit (510-513) | 130-162×; 429-731×; 70-81%; 65 / 65 / 59× | 130.2 / 161.9 / 137.3; 429 / 690 / 731; 69.7 / 76.5 / 81.2; 64.8 / 64.8 / 59.1 | ✓ |

**Setup, Limitations and appendix**

| Claim (lines) | Paper | Mine | Match |
|---|---|---|---|
| Memory (207-208, Table 13) | 29.6 / 32.8 / 35.6; 57.0 | 29.60 / 32.83 / 35.56 / 57.04 GiB (ring2048 diag rows) | ✓ value; provenance S-M |
| Limitations cell counts (520-521) | 3/1, 3/1, 13/2 | same | ✓ |
| Token-stream claims (769-773) | identical target streams; spec diverges at 8 | confirmed; but the row is T=0.7, not greedy | ✓ / S-Spec |
| Assertion failures (774) | 3 of 5 async stages | p1 arms_async and p2 onem_async logs carry `indexSelectLargeIndex`; p2 B1 is per the diag table (trimmed log shows errors, not the string) | ✓ (partly from the diag) |
| Table 7 rank spread | 0.056 ms | 0.091 / 0.110 ms in the new bench | ✗ (S-Bench) |
| All tables | — | regenerate byte-identically from `lcfm_figures.py` | ✓ |
| All figures | — | pixel-identical | ✓ |
| Fig. 4 | two wait modes; caption describes squares (blocking, host wait) and triangles (no flag, after device sync) | matches `ringbench_{blocking,async}.csv` (wait vs sync columns) | ✓ |

**Pools (1c).** Unchanged from my v1 review (`review_scratch/recompute_out.txt`): GATE/SANITY are
dropped and H1 rows excluded everywhere. New pools in v3:
- **F5 matched books:** r1 seed-42 spec against r1 target plus top-up target, books 0 | 0 / 1 / 915 |
  0 / 1 / 915 / 2204 / 2768.
- **bw_diag:** p1 bench and arms; p2 arms, onem and spec.
- **Census:** grouped by each row's own session-CSV temperature (the first-window temperature comes
  from `incident_20261010T113537Z_repetition_stop/pod_results/natural_f1_128k.csv`, T=0.0).

**Provenance (1d).** Every r1, r2 and top-up row has `split_kv`, 32 layers and commit fbd41fb or
429bfad. Every bw_diag row records mode, 32 layers and commit 31724c97 or 135f8fb8. `src/` is
identical to fbd41fb at both diagnostic commits. No void rows.

## 7. Task 3: citations

| Key | Source | Title / authors / year / venue | Use | Verdict |
|---|---|---|---|---|
| yang2024contextparallelism | arXiv:2411.01783 abs + PDF | ✓ / 9 authors ✓ / 2024 ✓ | pass-Q now described correctly (Q ring + all-to-all) | ✓ |
| lee2026dcp | vllm.ai/blog/2026-08-07-decode-context-parallelism (HTTP 200) | ✓ / Lee, Ha, Almog, Wilkinson ✓ / Aug 2026 ✓ | rotation-free LSE merge ✓ | ✓ |
| su2024rope | Crossref 10.1016/j.neucom.2023.127063 | ✓ / Su, Ahmed, Lu, Pan, (Wen) Bo, Liu ✓ / Neurocomputing 568:127063, 2024 ✓ | ✓ | ✓ |
| dao2024flashattention2 | arXiv:2307.08691 | ✓ / Dao ✓ / ICLR 2024 (venue from knowledge) | local kernel ✓ | ✓ |
| xiao2024streamingllm | arXiv:2309.17453 | ✓ / 5 authors ✓ / ICLR 2024 ✓, `@inproceedings` | ✓ | ✓ |
| liu2023ringattention, leviathan2023fast, chen2023accelerating, cai2024medusa, dettmers2023qlora, peng2023yarn | arXiv (checked in v1; entries unchanged) | ✓ | uses now accurate; the YaRN sentence has one unsupported clause (nit) | ✓ |
| rae2019pg19 | arXiv:1911.05507 lists 4 authors; bib adds Hillier (ICLR version) | **unverified** (OpenReview / S2 blocked or rate-limited) | ✓ | ? |

Also checked: torch v2.5.1 `ProcessGroupNCCL.cpp` (cited in the text). Line 426 is
`kSynchronizeBusyWaitMillis = 10` and line 684 is the sleep inside the blocking-wait loop ✓.
**3c:** cited keys equal bib keys (13); none missing or uncited.

## 8. Hazards

| Hazard | Handled? | Evidence |
|---|---|---|
| **H1** seed bug | **Yes** | SA_SEED rows are excluded from every old-engine pool; their token hashes equal seed 42's; v3 line citations are correct except 725 → 722. |
| **H2** vllm_run | **Yes** | Only vllm_run2 is used; vllm_run ttft ≈ 1.79e9. |
| **H3** Llama-2 CSVs | **Yes** | Not read by `lcfm_figures.py`; Llama-2 appears only in the labelled ceiling test and Appendix D. |
| **H4** draft window | **Yes** | `_draft_window` (BOS + last 4095) at fbd41fb and 429bfad; the logs show "draft prefill over 4096 of 1047552"; `src/` unchanged; the first-4096 defect appears only in Appendix A E6 (reviewed build). |
| **H5** FP4 / NF4 | **Yes** | `quantize_target` / `kv_quant` true in configs and in every df_arms row; sidecars say fp4 / nf4; the paper agrees. |
| **H6** matched book sets | **Yes** | Headline, F5 (now 1 / 3 / 5 matched books) and vLLM are all matched. One gap: the 128k vLLM run-over-prefill uses an unstable single TTFT (S-V1). |
| **H7** blocking wait | **Yes, with wording issues** | v1-era pod env exported the flag (`lcfm-camera-ready-v1:scripts/mlsys_pod_env.sh:32-35`); the campaign pod log says "TORCH_NCCL_BLOCKING_WAIT=1"; every bw_diag row records `nccl_env` (blocking vs `<unset>`). Every flag-free number the paper quotes (0.134 / 2.619 ms, 1.93 / 2.03 s, 15.96 s, 57.0 GiB) comes from the bw_diag async arm and is labelled as such. Blanket "every timing" wording (S-Every) and the stale `pod_env.sh:32` citation (S-Env) remain. |

## 9. Could not verify

- **The PG-19 ICLR author list** (Hillier): sources blocked or rate-limited.
- **The F16 looping row's tokens:** n5_extension ships no sidecar; the row is checked from its CSV
  only.
- **Kernel equivalence** (12 tokens, margin 0.0000, median 2.9375) and the **19.3%** profile share:
  not re-derivable from released CSVs.
- **Which async stage died on the assert in p2 arms:** the trimmed log carries no
  `indexSelectLargeIndex` string, so the "3 of 5" partly rests on BLOCKING_WAIT_DIAG.md.
- **Whether vLLM's own runs inherited the flag:** this does not affect any number, since vLLM's TP
  collectives do not go through ProcessGroupNCCL's wait.

## Task 4 and 5 summary

- **4a:** clean build, 0 undefined references or citations, **0 overfull boxes**.
- **4b:** §6 ends on page 8; References at the top of page 9 (15 pages) ✓.
- **4c:** author block present; no TODO / XXX / FIXME / "??" (all 4 TODO AMAN markers are gone);
  the committed PDF equals the fresh build. **But one rendered comment fragment** (NB1).
- **4d:** 0 em dashes in the .tex, the tables and the PDF ✓.
- **4e:** see S-Refs and S-Fig.

**5a. The weakest points a reviewer will attack.**
1. **F4's mechanism** (NB2, S-1M). The poll explains about 70% at 128k and 3% at 1M, and most of the
   1M saving is unattributed. The paper answers it only partly, and the abstract states the
   opposite.
2. **F2's generality.** In the controlled 128k temperature arm, greedy acceptance (0.460) is not
   above T=0.7 (0.468). The abstract's "a greedy acceptance number measures the text" rests on the
   1024-token first-window audit, a different protocol and book pool. V2_REPORT says this
   limitation is "carried by those two counts", but no sentence in the paper states it.
3. **F3.** It rests on one near-deterministic (memorised) book. The caveat is now stated (adequate
   for a workshop), but "stays coherent" is still the framing.

**5d. Coherence.** §4.4 is the clearly patched section:
- the title contradicts the F4 title;
- "Two caveats" introduces one;
- a comment cuts a sentence;
- the sentence "the fix is to remove the rounds rather than to move more data per round"
  (Appendix C, 756-757) no longer follows from a poll diagnosis, where removing the flag is the
  obvious fix.

§3's last paragraph and Appendix B/C captions also carry cross-version residue (S-Env, S-Bench,
S-Fig). The rest reads as written.

READY WITH MINOR FIXES

# Final review of `lcfm-camera-ready-v5`

**Scope.** Tag `lcfm-camera-ready-v5` (4fbb242), reviewed on branch `review/codex-v5`. The raw data
is unchanged since v3 (no diff under `results/mlsys/`). My v1 and v3 recompute scripts reproduce
their committed output byte for byte on v5. New checks are in `review_scratch_v5/v5_checks.py`.
`scripts/lcfm_figures.py` regenerates all 9 tables byte-identically and all 4 figures
pixel-identically.

**Venue rules: all met.**
- The main text ends on page 9 and References start on page 10 (pdftotext, 16 pages).
- `[dblblindworkshop, final]` prints the author block and "Workshop: Long Context Foundation
  Models (LCFM) @ NeurIPS 2026."; there are no line numbers.
- Every reference verifies against its canonical source.
- 0 em dashes; 0 undefined references; **0 overfull boxes**.
- Every float is referenced and numbered in order.

**What remains.** About ten one-line text fixes. One is a false statement introduced by a v4 edit
(F-1), and the rest are dangling or stale references, a garbled phrase and caption mismatches. No
headline number is wrong, and nothing needs new data. Two of the remaining items (F-7, and part of
F-6) trace back to an imprecise suggestion in my own v3 report, which I correct below.

## Task 1: every CODEX_REVIEW_V3 item

| v3 item | Status | Evidence (v5 main.tex line) |
|---|---|---|
| NB1 rendered fragment | VERIFIED FIXED | 473 "One caveat we state rather than bury"; the fragment scan of the PDF is clean. |
| NB2 poll overstated | VERIFIED FIXED | Abstract 85-91, F4 137-152, §4.4 heading 387 and text 452-468, conclusion 619-622 all say about 70% at 128k and about 3% at 1M. (New caption problem: F-5.) |
| NB3 "within 3% at every payload" | VERIFIED FIXED | 402-404 and 420-421; recomputed +2.9 / +0.5 / +0.8% (33 / 133 / 266 MB), +5.4% (66 MB), +11.0% (4 KB). |
| S-P prediction | VERIFIED FIXED | 459-460: 224 × (10.3347 − 2.6191) ms = 1.7283 → 1.73 s, against 1.7485 → 1.75 s. |
| S-Mix sessions | VERIFIED FIXED | Main text uses p1 throughout (455; 474-475: 2.0319 / 3.7805 → 2.03 / 3.78). |
| S-1M "mostly transfer" | VERIFIED FIXED | 465-466: 224 × 2.147 GB / 101.69 GB/s = 4.74 s of 13.88 s, "not attributed". (Appendix C contradicts it: F-6.) |
| S-BW "bandwidth-bound" | VERIFIED FIXED | Clause gone. |
| S-V1 vLLM 128k | PARTLY FIXED | 128k ratios dropped (2.537 / 2.572; 2.516 / 2.563 ✓) and the columns relabelled ✓. Two problems remain: 557-558 now says "both divide by that same unstable number" with no antecedent in the text (F-2), and the footnote's "7.0 to 12.4 s for one configuration" mixes configurations (F-7). |
| S-M memory provenance | PARTLY FIXED | 218-220 and Table 13 are correct (29.60 / 32.83 / 35.56 / 57.04 GiB from ring2048 diagnostic rows; split-KV n=20, rank-0 max 26.21 GiB). Appendix A E18 (671) still says "This campaign's 1M rows record 29.6 GiB … worst rank" (F-8). |
| S-F1 "never smoothed" | VERIFIED FIXED | 301-303. |
| S-U "unchanged" | VERIFIED FIXED | 92, 155, 159, 515, 585, 623; ±0.061 / 0.4839 = ±12.6% → "±13%" ✓. |
| S-Spec greedy label | VERIFIED FIXED | 799-800 "seeded T=0.7 row". |
| S-Bench spread | NEW PROBLEM | Table 9 caption (769-771): see F-6b. My v3 value of "0.110 ms (no flag)" was itself wrong for this table. |
| S-Env pod_env citation | VERIFIED FIXED | 246: `fbd41fb:scripts/mlsys_pod_env.sh:34` is `TORCH_NCCL_BLOCKING_WAIT=1`. |
| S-Fig missing figA | VERIFIED FIXED | Fig. 4 is back in §4.4 (437-446). (Its caption has a new wording problem: F-5.) |
| S-Refs | VERIFIED FIXED | All 16 labelled floats are `\ref`'d at least once; Table 1 no longer cites itself. |
| S-Every | PARTLY FIXED | 244-246 is precise. "Every timing in this paper" (171) and "Every timing was taken under NCCL blocking wait" (574) remain blanket; the paper quotes flag-free timings and vLLM timings (nit). |
| S-SRC | MOSTLY FIXED | `bw_numbers()` / `bw_transfer_check()` now print the bench, post-sync, p1 / p2 poll shares (64.5 / 66.9%), the 1M −2.88% and the 4.74 s. The 1.73 s prediction and the 72% pairing share are still not printed by the script (nit). |
| S7 unused factor-8 branch | VERIFIED FIXED | 221-223. |
| Nit "NCLL" | VERIFIED FIXED | 538. |
| Nit missing period | VERIFIED FIXED | 330. |
| Nit duplicated tabcolsep | VERIFIED FIXED | 651. |
| Nit line 725 → 722 | PARTLY FIXED | Main text 511 ✓; **Appendix B 697 still says "line 725"** (F-4). |
| Nit 4.03 → 4.04 | VERIFIED FIXED | 513 (buggy pool 4.037). |
| Nit 8-token rationale | VERIFIED FIXED | 279. |
| Nit YaRN clause | VERIFIED FIXED | 206-208. |
| Nit E6 "lower bounds" | VERIFIED FIXED | 660. |
| Nit "both arms" (rotary §, 337 in v3) | NEW PROBLEM | The edit landed on the wrong sentence: line 305 now says "0 of 5 rows degenerate in **all three arms**", which is false (the greedy arm is 1/5). Line 364 still says "both arms paid the same blocking-wait cost" for three rotary arms (F-1). |
| Nit Fig. 1 legend | NEW PROBLEM (nit) | The legend moved to the upper left and now covers the first-window diamonds at y≈0.95. |
| Nit Fig. 2 y-label | PARTLY FIXED (nit) | Shortened, but it still runs into the left panel. |
| Nit 128k TTFT range | NEW PROBLEM | See F-7 (caused by my v3 suggestion). |
| Nit stale conclusion SRC | VERIFIED FIXED | 627-628. |
| 5a.2 F2 generality | PARTLY FIXED | Scoped correctly at 79-80 and 298-300 (0.4604 vs 0.4681 → 0.460 / 0.468 ✓). The F2 paragraph (125-127: "A greedy acceptance number therefore measures the degeneracy") and the conclusion (606-607) are still unscoped (F-3). |
| 5a.3 F3 framing | DELIBERATE, accepted | Caveat stated at 379-384. Not re-litigated. |
| PG-19 authors unverified | VERIFIED | arXiv PDF 1911.05507v1 title page and the ICLR 2020 listing (iclr.cc/virtual_2020/poster_SylKikSYDH) both list Hillier. |
| 5d patched §4.4 / Appendix C | PARTLY FIXED | §4.4 now reads as one argument. Appendix C's replacement sentence (779-782) is wrong (F-6), and 837-838 ("the four-arm comparison all of that rests on") is unclear. |

## Blockers

None. The venue's hard rules are met, and no headline number is wrong.

## Fix before upload (all one-line, text-only)

**F-1. A false statement introduced by the v4 nit edit.**
*Where:* main.tex:305.

*Evidence:* "with 0 of 5 rows degenerate in all three arms". The 128k greedy arm has 1 of 5
degenerate (SA_TEMP_0 book 915, `census_v3_out.txt`), as the same section says at 275.

*Fix:* "with 0 of 5 rows degenerate at $T{=}1.0$ and at $T{=}0.7$". At line 364, change "both arms"
to "all three arms", which is the edit the nit actually asked for.

**F-2. A dangling reference.**
*Where:* main.tex:557-558.

*Evidence:* "We quote neither 128k ratio, because both divide by that same unstable number": no
unstable number is introduced in the text (the TTFT sentence was removed in v4). The explanation
survives only in the Table 3 footnote.

*Fix:* "…because both divide by our 128k first-token time, which varies by about 20% for one
configuration (Table 3)".

**F-3. F2 is still stated generally in two places.**
*Where:* main.tex:125-127 and 606-607.

*Evidence:* these contradict the scoping at 298-300 ("the inflation we report belongs to the
first-window protocol rather than to greedy decoding in general").

*Fix:* at 125-126, "A greedy acceptance number from this audit therefore measures…". At 606, "Under
greedy decoding the first-window audit's acceptance numbers are measured on degenerate text".

**F-4. Appendix B line citation.**
*Where:* main.tex:697.

*Evidence:* "(line 725)" should be **722**, as the main text already says (511).
`run_experiment.py:722` builds the documents-based run id.

**F-5. The Fig. 4 caption mis-states the mechanism.**
*Where:* main.tex:442-443.

*Evidence:* "Both engines are dominated by the same rotation path, which split-KV shortens". Split-KV
has no rotation; it removes it (406-408).

*Fix:* "The old engine is dominated by the rotation, which split-KV removes, and the gap widens as
the shard a round ships grows."

**F-6. Two Appendix C statements disagree with the data.**

*(a) What removing the rotation eliminates.*
*Where:* main.tex:781-782.

*Evidence:* "What the wire costs is what the no-flag column shows, and that is the part of the ring's
cost that removing the rotation eliminates." But removing the rotation eliminates the poll (about
70% at 128k), the transfer, and an unattributed remainder (§4.4, Table 11).

*Fix:* "…so the campaign's flat line was the poll rather than the wire; what removing the rotation
eliminates is broken down in Table 11."

*(b) The Table 9 caption spread.*
*Where:* main.tex:769-771.

*Evidence:* the caption says the largest no-flag rank spread is "0.110 ms". The table's no-flag
column holds the **post-sync** spreads (max 0.0286 ms; JSON skew entries are wait / sync pairs).
0.110 is the no-flag *wait-timing* spread, which the table does not show. `bw_numbers()` takes the
max over both, and so did my v3 report.

*Fix:* "the largest are 0.091 ms under the blocking wait and 0.029 ms without it".

**F-7. The 128k TTFT footnote mixes configurations (my v3 suggestion was wrong).**
*Where:* `results/lcfm/tables/vllm.tex` footnote.

*Evidence:* "varies 7.0 to 12.4 s for one configuration". 7.0 s is `splitkv` (kv_block_size 32768,
bw_diag p1) and 12.4 s is `ring2048` (bw_diag p2). Within one configuration:
- the split-KV campaign's own 128k rows span **8.0-9.6 s** (target 8.02-9.57; spec 8.27-9.49);
- ring2048 under blocking spans **8.9-12.4 s**.

*Fix:* "varies 8.0 to 9.6 s across this release's split-KV 128k rows (8.9 to 12.4 s for one ring
configuration)", or just "varies by up to 40% between sessions". The decision to drop the ratio
still stands.

**F-8. Appendix A E18 provenance.**
*Where:* main.tex:671.

*Evidence:* "This campaign's 1M rows record 29.6 GiB allocated and 32.8 GiB reserved on the worst
rank".

*Fix:* "This release's 1M ring2048 diagnostic rows record … on the worst rank; the split-KV campaign
rows record rank 0 only (26.2 GiB)".

**F-9. Garbled phrase.**
*Where:* main.tex:274.

*Evidence:* "the long-context first-window audit loop loops in 9 of its 10 rows".

*Fix:* "the long-context first-window audit loops in 9 of its 10 rows".

## Nits

- **"Every timing"** (171, 574) and **"every run here used"** (617): append "of RASD's" / "except
  the no-flag diagnostic arm" (S-Every).
- **Line 614-615**, "our diagnosis was the measurement condition": true at 128k, not at 1M (3%).
  "…was largely the measurement condition at 128k" would match §4.4.
- **Line 837-838**, "Table 12 is the four-arm comparison all of that rests on": unclear antecedent.
  Table 11 uses the matched cells, not Table 12.
- **Figures.** Fig. 1's legend covers data; Fig. 2's right y-label crowds the left panel.
- **The bw_arms caption (790)** describes "two flagged stages marked n/a", but the table has no n/a
  rows. I missed this in v3; it is unchanged since v2.
- **`bw_numbers()`** should report the post-sync spread to match the table (F-6b), and could print
  the 1.73 s prediction and the 72% pairing share (S-SRC).
- **The references.bib header** points to "V2_REPORT.txt, v5 section" for field-by-field results,
  but V2_REPORT ends at v4.
- **RoFormer author.** Crossref lists the fifth author as given "Wen", family "Bo"; the bib renders
  "Bo Wen" (the arXiv form). Either is defensible.

## Task 2: recomputed numbers (own code)

| Number | Paper | Mine | Match |
|---|---|---|---|
| 128k poll share, pairing | 72% | 1.7485 / 2.4329 = 71.9% | ✓ |
| 128k poll share, within session | 65% / 67% | p1 1.7485 / 2.7098 = 64.5%; p2 1.8694 / 2.7938 = 66.9% | ✓ |
| Prediction vs measured | 1.73 vs 1.75 s | 1.7283 vs 1.7485 | ✓ |
| 1M flag effect | 3%, 16.43 → 15.96 | −2.88%, 16.4348 → 15.9613 | ✓ |
| 1M transfer | 4.7 of 13.9 s | 4.74 of 13.88 s | ✓ |
| Post-sync agreement | <3% / 5.4% / 11% | 2.9, 0.5, 0.8 / 5.4 / 11.0% | ✓ |
| vLLM ratios | 2.54-2.57×; 2.52 / 2.56× | 2.537 / 2.572; 2.516 / 2.563 | ✓ |
| Memory | 29.6 / 32.8 / 35.6 / 57.0; 26.2 (n=20) | 29.60 / 32.83 / 35.56 / 57.04 (ring2048 diag, worst rank); 26.21 rank-0 max over 20 split-KV 1M rows | ✓ (provenance F-8) |
| Greedy vs T=0.7 | 0.460 vs 0.468 | 0.4604 vs 0.4681 | ✓ |
| Rank spread | 0.091 / 0.110 ms | blocking wait 0.0906 ✓; no-flag post-sync 0.0286 (0.1096 is wait timing) | ✗ (F-6b) |
| Acceptance interval | ±13% | ±12.6% | ✓ |
| Buggy 1M speedup | 4.04 | 4.037 | ✓ |
| 128k TTFT range | 7.0-12.4 s | mixes configurations | ✗ (F-7) |

All other numbers are unchanged from v3 and reproduce: F1, F2, F3, the headline, F5, the seed table
and the top-up (see `review_scratch*/` outputs).

## References (Task 3): every entry fetched

| Key | Canonical source fetched | Title / authors (order) / year / venue / URL |
|---|---|---|
| yang2024contextparallelism | arXiv API 2411.01783 | ✓ / 9 ✓ / 2024 ✓ / arXiv id ✓ |
| lee2026dcp | vllm.ai/blog/2026-08-07-decode-context-parallelism (HTTP 200) | ✓ / Lee, Ha, Almog, Wilkinson ✓ / Aug 7 2026 ✓ / URL ✓ |
| liu2023ringattention | arXiv API 2310.01889 | ✓ / 3 ✓ / 2023 ✓ |
| dao2024flashattention2 | ICLR 2024 official listing (iclr.cc data; poster 17889) and arXiv 2307.08691 | ✓ / Dao ✓ / ICLR 2024 ✓ |
| leviathan2023fast | PMLR v202 (leviathan23a) | ✓ / 3 ✓ / ICML 2023 ✓ |
| chen2023accelerating | arXiv API 2302.01318 | ✓ / 6 ✓ / 2023 ✓ |
| cai2024medusa | arXiv API 2401.10774 | ✓ / 7 ✓ / 2024 ✓ (cited as arXiv; valid) |
| dettmers2023qlora | arXiv API 2305.14314 | ✓ / 4 ✓ / 2023 ✓ (cited as arXiv; valid) |
| xiao2024streamingllm | ICLR 2024 listing (poster 18794) and arXiv 2309.17453 | ✓ / 5 ✓ / ICLR 2024 ✓ |
| su2024rope | Crossref 10.1016/j.neucom.2023.127063 | ✓ / Su, Ahmed, Lu, Pan, Bo/Wen, Liu ✓ / Neurocomputing 568:127063, 2024 ✓ |
| peng2023yarn | arXiv API 2309.00071 | ✓ / 4 ✓ / 2023 ✓ |
| rae2020pg19 | arXiv PDF 1911.05507v1 title page and ICLR 2020 listing (iclr.cc/virtual_2020/poster_SylKikSYDH) | ✓ / 5, including Hillier ✓ (the arXiv API metadata omits her) / ICLR 2020 ✓ |

Cited keys equal bib keys (12 entries in the reference list; no missing or uncited key). The cited
torch source (ProcessGroupNCCL.cpp:426 at v2.5.1, `kSynchronizeBusyWaitMillis = 10`) was verified
in v3 and is unchanged.

## Could not verify

- **The OpenReview and dblp records** for the ICLR 2024 entries (bot challenge). I used the
  official iclr.cc listing instead, which is sufficient.
- **The kernel-equivalence claims** (12 tokens, margin 0.0000) and the **19.3%** profile share: not
  re-derivable from released CSVs (unchanged since v1).
- **The F16 looping row's tokens:** no sidecar is released; the row is checked from its CSV only.

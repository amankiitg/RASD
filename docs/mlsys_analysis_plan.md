# MLSys analysis plan (pre-registration)

**Status:** pre-registered before any new GPU run for this campaign.
**Frozen:** 2026-10-06. **Replications of the plan itself** go in the revisions
block at the bottom, dated, with the reason.

This document fixes the measurements, the unit of independence, the interval
method, the comparisons, and the pass/fail rules *before* the data exist. Once a
number is produced under this plan it may not be re-analysed under a different
rule without a dated revision. The purpose is to make the results survive a
reviewer who looks for the analysis choice that was made after seeing the data.

---

## 1. Model policy

| role | model | rope |
|---|---|---|
| target | `meta-llama/Llama-3.1-8B` | native shipped `llama3` block, untouched |
| draft | `meta-llama/Llama-3.2-1B` | its own native window |
| target (correction-note arm) | `meta-llama/Llama-2-7b-hf` | YaRN, both anchorings, for the correction note only |

Binding rules:

1. At 128k and below the target runs its **native shipped configuration with no
   `rope_scaling` modification**. Anything else is a different experiment.
2. Above 128k, a rope configuration may be used **only** if it passes the
   coherence gate (§6). The gate is a precondition, not a diagnostic.
3. **No training-free YaRN on a 4k-native model.** Measured: 4.9× native
   perplexity and newline collapse even with the anchor corrected.
4. `transformers==4.47.1` with the project's `rope_anchor_base`. Version-pinned
   because 4.51–4.55 read the anchor but override `factor`, which would silently
   change every YaRN cell.
5. The draft reads its own positional frame. Speculative decoding requires the
   draft's *tokens* to match the target's choices, not that the two models share
   an anchoring.

## 2. Decoding settings (fixed)

| setting | value |
|---|---|
| sampling | **greedy** (`temperature=0`, `top_p=1.0`) |
| generated tokens | **1024** (`max_new_tokens=1024`) |
| EOS | **ignored** (`ignore_eos=true`) |
| prompt length | `context_length - 1024` |
| total sequence | exactly `context_length` |
| spec steps | gamma = 4 |

Two consequences that must hold in every cell:

* `prompt_tokens + tokens_generated == context_length`. At native 128k this keeps
  the whole sequence inside the model's window; a longer sequence would measure
  extrapolation, not the rung. Every row records `prompt_tokens` and
  `tokens_generated`, and the sum is asserted.
* Greedy decoding is **deterministic**. `seed` is therefore *not* a replication
  axis: two seeds on the same prompt produce identical output and identical
  acceptance. Replication is by **document** (§3). `seed` is fixed at 42 and is
  recorded only for reproducibility of prompt construction.

## 3. Unit of independence: the document

**The document — one PG-19 book — is the independent unit.** Not the token, not
the round, not the seed.

Rationale: tokens within a verify round are accepted or rejected together, so
tokens are not independent; rounds within one generation share a prompt and a KV
state, so they are not independent either. Different books are the only units
here that vary independently of the model and of our own choices.

**Documents per rung: 10.**

### Document pool (measured, not assumed)

Source: PG-19 train split (`emozilla/pg19`), streamed; books pre-screened at
≥600,000 characters before tokenizing (a character screen is free, `encode` is
not). First 3,000 books scanned → **535 books tokenized**.

A document at context `C` must supply `C - 1024` prompt tokens, 1024 generated
tokens, and a separate 1024-token held-out continuation, i.e. **≥ C + 1024
tokens**. Eligible books:

| rung `C` | tokens needed | eligible books | 10 available? |
|---|---:|---:|---|
| 128k | 132,096 | 534 | yes |
| 256k | 263,168 | 109 | yes |
| 512k | 525,312 | **14** | yes, but only by taking 10 of 14 |
| 1M | 1,049,600 | **6** | **no** |

Two limitations are therefore structural and are reported with the results
rather than worked around:

* **The validation split cannot be used.** It has 50 books total and only 7
  reach 128k tokens. Any "10 documents" design built on it is impossible; the
  earlier PG-19 pipeline concatenated documents into 1M-token chunks, which
  erases book boundaries and cannot supply documents at all.
* **A 1M natural-text rung cannot have 10 documents.** Only 6 books reach 1M
  tokens. No natural-text cell is reported at 1M under this plan. The 1M rung
  remains synthetic-only and is labelled as such.

### Document selection and the paired design (primary)

**Primary analysis uses a nested core-10:** the same 10 books at every rung.
Selection is deterministic and RNG-free: order the books eligible at the largest
rung (14 at 512k) by descending token count and take the **10 longest**. The
prompt hash of every document at every rung is logged. Each book contributes its
**first** `C - 1024` tokens as the prompt; the continuation is the next 1024
tokens.

The same documents at every rung makes the rung comparison **paired** — the
dose-response claim is a within-document claim, and pairing removes document
difficulty, which is the dominant variance source at n=10.

The cost of this choice is stated plainly: because only 14 books are eligible at
512k, the core-10 is dominated by Bibles and collected/omnibus volumes. That
genre bias is **constant across rungs**, so it cancels in the paired rung
comparison; it does limit *generalisation* to long prose in general, which is a
scope limitation on the headline number and is reported as one.

**Secondary (budget-permitting):** 10 documents drawn independently per rung
from that rung's own eligible pool (534 / 109 / 14), to test whether the core-10's
composition drives the headline acceptance figure. Reported as a robustness
check, never as the primary number.

## 4. Primary metrics

All three are reported for **every cell**. A cell missing any of them is not a
result.

### 4.1 Acceptance

Two distinct quantities, never conflated (reviewer AbH52 #1):

* **`alpha_round`** — the per-round accepted prefix length divided by gamma,
  averaged over rounds. This is the acceptance parameter of speculative
  decoding and the only one used for claims.
* **`a_iid`** — total accepted tokens / total drafted tokens. This is the i.i.d.
  per-token parameter. Reported beside `alpha_round` for comparability with
  other work, and **never** substituted for it.

Rounding: exclude a partial final round (drafted `< gamma`) from the per-round
mean; the number of rounds used is always reported. `alpha_round` is also
reported over a **common window** (the first `N` rounds, `N` = the shortest
generation among the compared runs) so cells with different output lengths are
compared on equal footing.

Also reported: `p_alpha_zero` (share of rounds with zero accepted tokens),
`full_accept_share` (share of rounds with `n_acc == gamma`), and the `n_acc`
distribution. A rung with `full_accept_share > 0.90` is flagged **saturated**,
because a saturated rung cannot show a difference between configurations.

### 4.2 Speedup against the matched target-only run

**speedup = spec `throughput_tps` / paired target-only `throughput_tps`**

* Unit: **end-to-end generation tokens per second**, i.e. `tokens_generated /
  total_wall_time` including prefill. This is the RASD `throughput_tps` field.
  It is **not** `forward_tps`. Any comparison against another system (vLLM) is
  unit-checked and the result carries a `unit_matched` flag; a comparison that
  is not unit-matched is reported as invalid rather than as a speedup.
* Pairing: the target-only run uses **the same document, the same prompt, the
  same context, the same generated-token count, and the same request shape**.
  A speedup against an unpaired baseline is not reported.

### 4.3 Target quality beside every acceptance figure

Perplexity of the **target** on a held-out 1024-token continuation, at that
rung's context. Reported per document, with the document-bootstrap interval,
beside the acceptance figure for the same document — acceptance without target
quality is not interpretable, because a degenerate target can accept *or* reject
a great deal and this project has observed both.

Windows inside one book (the book supplies ≥ `C + 1024` tokens):

| window | tokens | used for |
|---|---|---|
| prompt | `[0, C - 1024)` | both the speculative run and the perplexity measurement |
| generation | `[C - 1024, C)` | the speculative and target-only runs generate here |
| continuation | `[C, C + 1024)` | teacher-forced perplexity, conditioned on the prompt |

So the sequence scored for perplexity is exactly `C` tokens (prompt +
continuation), inside the configured context. The continuation is **not** the
window the model generates, so perplexity and generation quality are measured on
different held-out text; it sits one generation-length past the prompt, which
makes it a test that the context carries forward across the whole span rather
than a local next-token continuation.

* "Held out" means the scored continuation is never part of the prompt.
* Computed with head-chunked cross-entropy. `ForCausalLMLoss` upcasts the full
  logit tensor to fp32 (4.2 GiB at 32k) and OOMs at long context; applying the
  LM head in position chunks is the identical shifted cross-entropy with bounded
  memory.
* Prompt and continuation are contiguous slices of **one** book, so the
  continuation is a genuine continuation of that book's text.

## 5. Interval method

**Percentile bootstrap resampling documents**, `B = 10,000`, 95% intervals.
Reproducible: seeded per cell (`20261006`), and the seed is logged.

* For a single rung: resample the 10 documents with replacement, recompute the
  document-mean metric, take the 2.5th and 97.5th percentiles.
* For a comparison: resample documents and recompute the **paired difference**
  (same document in both arms), so the interval is on the difference, not on two
  marginal intervals.
* For a ratio (speedup): resample documents, recompute the ratio of the paired
  means.

**Robustness column.** With only 10 clusters a percentile bootstrap is
optimistic and its tails are coarse. Every interval is therefore reported
alongside a **t-based cluster interval** (Student-t on 9 degrees of freedom over
the 10 document means). If the two disagree on whether the interval excludes the
decision threshold, the result is reported as **inconclusive**, not as the
favourable one. Both numbers are reported; the plan does not hide the disagreeing
one.

**Power, stated up front.** At n = 10 the 95% interval on a document-mean
proportion near 0.9 has a half-width on the order of ±0.15, and near 0.5 on the
order of ±0.30. This plan can therefore detect *large* effects. It cannot resolve
small ones, and any claim of a small difference is reported as not established.

## 6. Validity gates (preconditions, not results)

### 6.1 Losslessness (mandatory, every rung)

With greedy decoding and `ignore_eos=true`, speculative decoding must be
**exactly** lossless: the spec run's token stream must be **token-identical** to
the paired target-only run's over the full generation. Compared on token IDs, not
on decoded text (text is not injective).

* The first mismatch position is reported. **Any mismatch fails the cell.**
* A failing cell is reported as an implementation failure, with the mismatch
  position. Its acceptance and speedup are **not** reported as results.
* This is the check that would have caught the earlier ARM4 rope corruption
  regardless of what the acceptance number looked like.

### 6.2 Coherence gate (configurations above 128k only)

A rope configuration may only be used if its target is coherent at the context it
will run at:

* the **effective** rope is asserted from the built model's `inv_freq`, never
  read back from the config dict (on 4.47.1 the dict key is inert, so reading it
  proves nothing);
* perplexity on a held-out continuation after a **full-length** prompt, within
  1.5× the native baseline at 128k;
* no EOS within the first 16 generated tokens;
* the generation is not dominated by blank lines or repeated n-grams.

Measured at the candidate's real context. A 512-token prompt cannot exercise
long context: the two anchorings were indistinguishable at 512/4096 tokens (PPL
4.89 vs 4.80) yet 57× apart at 32k.

**Calibration (S0).** The gate's thresholds are set from the **measured spread
of the positive controls on real weights** — native at 32k, 64k, 128k — not
chosen a priori. The negative controls (the ARM4 f2 construction, and
mis-anchored Llama-2) must fail. The threshold actually used is recorded in the
revisions block. A gate that has never seen real weights has not been calibrated,
so no candidate is gated before S0.

## 7. Pre-registered comparisons

| # | comparison | estimand | decision rule |
|---|---|---|---|
| C1 | native 128k: RASD vs vLLM speculative decoding, same model pair, same documents, greedy | acceptance, and losslessness on both | descriptive; a disagreement in acceptance is a finding, not an error to hide |
| C2 | native 128k vs 256k vs 512k, same documents | paired difference in `alpha_round` | interval on the paired difference per adjacency |
| C3 | native 128k vs the mis-anchored YaRN 128k target (arm 1) | paired difference in `alpha_round` and in speedup | this is the primary scientific claim: does acceptance recover when the target is native? |
| C4 | draft window capped at 4k vs full native draft window | paired difference in acceptance and speedup | isolates the draft-window effect from the context-length effect |
| C5 | correction-note arm: Llama-2 YaRN factor {32, 64, 128, 256} × correctly anchored vs mis-anchored | target perplexity, and generation quality at full prompt length | evidence for the correction note; never reported as a speculation result |

## 8. "Speculative decoding stops paying off" — exact definition

**Declared as reached at rung `C` when the 95% interval on the paired
spec/target-only throughput ratio lies entirely below 1.0.**

Stated precisely, because the definition is the claim:

* the interval must lie **entirely** below 1.0; an interval containing 1.0 is
  **inconclusive**, not a payoff boundary;
* it is a **throughput** ratio, generation tok/s, prefill included — not a
  per-token latency ratio and not a `forward_tps` ratio;
* the ratio is **paired** by document;
* the interval is the document bootstrap; if the t-based robustness interval
  disagrees (§5), the boundary is reported as not reached;
* a rung flagged **saturated** (§4.1) cannot support this claim, because a
  saturated rung's speedup is dominated by the draft cost rather than by
  acceptance.

Equivalently, "stops paying off" means the speedup is credibly below 1.0 — not
merely "less than at the previous rung".

## 9. Threats to validity, stated before the results

1. **Pretraining contamination.** PG-19 is a standard pretraining corpus. The
   continuation is held out from the *prompt*, not necessarily from the model's
   training data. Perplexity here is a *relative* instrument — it compares
   configurations on the same text — and must not be read as a clean
   generalisation number.
2. **Pool composition at 512k.** 14 eligible books, dominated by scripture and
   omnibus volumes. Constant across rungs, so it cancels in paired comparisons;
   it limits generalisation.
3. **n = 10.** Detects large effects only (§5).
4. **Greedy only.** All conclusions are for greedy decoding. A sampled-decoding
   regime changes acceptance and is out of scope.
5. **One hardware configuration.** 8×A100-80GB SXM4. Speedups are hardware-bound
   and are not claimed to transfer.
6. **Single-book prompts.** Ten books is ten samples of "long English prose", not
   a random sample of text.
7. **Saturation.** Several rungs already measured above 90% fully-accepted rounds.
   A saturated rung is reported as saturated and cannot support a comparative
   claim, whatever its interval says.

## 10. Reporting contract

Every table reports: acceptance (`alpha_round`) with its document-bootstrap
interval **and** the t-based interval, `a_iid`, `p_alpha_zero`,
`full_accept_share`, the `n_acc` distribution, speedup with its paired interval,
target perplexity with its interval, prompt hashes, prompt and generated token
counts, and the losslessness verdict with the first mismatch position if any.
Cells that failed a gate are listed with the failure and are excluded from the
aggregate tables, not silently dropped.

---

## Revisions

*(append-only; dated; state the reason and whether it was made before or after
seeing the affected data)*

- **2026-10-06 — plan frozen** before any run. No data seen under this plan.
  - Gate threshold: **not yet set** — set in S0 from the positive controls'
    measured spread, recorded below.
- **2026-10-06 — 1M natural-text rung declared unavailable**, before any data.
  6 eligible books, 10 required. Measured in
  `data/processed/pg19_books/book_lengths_train.json`.
  (Opened 2026-10-06, closed 2026-10-06.)

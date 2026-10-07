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
| prompt length | `context_length - 1024 - 1` |
| total sequence | exactly `context_length` (prompt + leading BOS + generated) |
| spec steps | gamma = 4 |

The `- 1` is not decoration. `generate_text` tokenizes the prompt without
`add_special_tokens=False`, so the engine prepends one BOS token; the model
therefore sees one token more than the prompt ids the caller counted. Without the
`- 1` the sequence would be `context_length + 1`, one position past a native 128k
window.

Two consequences that must hold in every cell:

* `prompt_tokens + 1 + tokens_generated == context_length`, where
  `prompt_tokens` is the count **excluding** the engine's BOS (that is the field
  the CSV records) and the row also carries `sequence_tokens`, the total the
  engine actually built. At native 128k this keeps the whole sequence inside the
  model's window; a longer sequence would measure extrapolation, not the rung.
  The builder asserts the identity and **refuses to run** if it cannot be met,
  rather than warning: a run at a context nobody measured is worse than no run.
  Note the engine re-tokenises the prompt *string*, so for some books the source
  slice must be adjusted to make the re-encoded prompt exactly `context - 1025`
  tokens; the adjustment is logged as `prompt_source_len` and is up to ~1100
  tokens at the 512k rung.
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

A document at context `C` must supply **≥ C tokens**: `C - 1024` prompt tokens
plus 1024 continuation tokens, which is a single contiguous forward of exactly
`C`. Eligible books (streamed and tokenized; `book_lengths_train.json`):

| rung `C` | tokens needed | eligible books | 10 available? |
|---|---:|---:|---|
| 128k | 131,072 | 534 | yes |
| 256k | 262,144 | 109 | yes |
| 512k | 524,288 | **14** | yes, but only by taking 10 of 14 |
| 1M | 1,048,576 | **6** | **no** |

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
* **`alpha_total_ratio`** — total accepted tokens / total drafted tokens. With
  gamma constant this is *algebraically identical* to `alpha_round`, so it is a
  consistency check on the accounting, not an independent quantity. It was
  previously labelled `a_iid`, which printed the same number twice under
  "per-round" and "i.i.d." headings and thus asserted memorylessness by
  construction.
* **`alpha_iid`** — the memoryless per-token acceptance that reproduces the
  observed mean accepted-prefix length, obtained by solving
  `E[N] = alpha + ... + alpha^gamma` (see
  [acceptance.py](/Users/amankesarwani/PycharmProjects/RASD/src/analysis/acceptance.py)).
  This is the parameter to quote when describing acceptance as an i.i.d. per-token
  probability, and it is strictly greater than `alpha_round` whenever the trace is
  not memoryless. Measured on the 57 committed traces it exceeds `alpha_round` in
  53 of them, i.e. the distinction is not cosmetic.

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

Perplexity of the **target** on a held-out continuation, at that rung's context.
Reported per document, with the document-bootstrap interval, beside the
acceptance figure for the same document — acceptance without target quality is
not interpretable, because a degenerate target can accept *or* reject a great
deal and this project has observed both.

Windows inside one book (the book supplies ≥ `C` tokens):

| window | tokens | used for |
|---|---|---|
| prompt | `[0, C - 1024 - 1)` after re-encoding | both the speculative run and the perplexity measurement |
| continuation | next 1024 tokens of the book | scored teacher-forced, **and** the window the runs generate into |

The scored sequence is exactly `C` tokens, contiguous, so it is one forward
inside the configured context. The continuation must be contiguous with the
prompt for a single forward to score it; a continuation placed *after* a
generation-length gap would need a forward of `C + 1024`, which at native 128k
would measure extrapolation rather than the rung.

Consequence, stated because it is a design choice a reader will question:
perplexity and generation cover the **same window**. They are different
measurements of it — perplexity is teacher-forced on the **book's own human
text**, while the run generates the model's greedy continuation. Measuring
perplexity on human text rather than on the model's own output is deliberate: it
cannot be flattered by degenerate self-repetition.

* "Held out" means the scored continuation is never part of the prompt.
* Computed with head-chunked cross-entropy. `ForCausalLMLoss` upcasts the full
  logit tensor to fp32 (4.2 GiB at 32k) and OOMs at long context; applying the
  LM head in position chunks is the identical shifted cross-entropy with bounded
  memory.
* Under sequence-parallel sharding the NLL is computed per rank over the
  positions that rank owns and all-reduced, because no single rank holds the
  full sequence. Each rank carries one extra left-context token so that the
  hidden state predicting its first owned token is local.

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

**Calibration (S0).** The tolerance is **fixed at 1.5×** and pre-registered in the
revisions block; the earlier rule that derived it from the positive controls'
measured spread was **withdrawn before any run** (the three positives are single
measurements at three different contexts, so their spread does not estimate the
gate's noise, and every available estimator can only loosen the gate). The
controls therefore act as a **falsification test of the gate at the fixed
threshold**: native at 32k, 64k and 128k must pass, and the negative controls
(the ARM4 f2 construction, and mis-anchored Llama-2) must fail. Each control must
be **measured at its own context against a declared native baseline at that same
context and model**; a control reporting "no baseline" was not measured and is
neither a pass nor a detected failure, so it stops the campaign. A gate that has
never seen real weights has not been calibrated, so no candidate is gated before
S0.

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
  - Gate threshold: **fixed at 1.5×** in the 2026-10-06 revision below, before any
    run. The "derive it from the positive controls' spread" rule was withdrawn;
    the controls falsify the gate at the fixed threshold instead.
- **2026-10-06 — 1M natural-text rung declared unavailable**, before any data.
  6 eligible books, 10 required. Measured in
  `data/processed/pg19_books/book_lengths_train.json`.
  (Opened 2026-10-06, closed 2026-10-06.)

- **2026-10-06 — three corrections from an independent review pass, before any
  data.** An independent reviewer (fresh context) audited this plan, the
  manifest and the pipeline. The following were changed as a result; each is a
  correction of an error in the plan or the code, not a change of intent.

  1. **The prompt/BOS identity was stated wrongly.** The plan said
     `prompt_tokens + tokens_generated == context_length`. The engine prepends a
     BOS, so the true identity is
     `prompt_tokens + 1 + tokens_generated == context_length`, and the prompt is
     `context_length - 1024 - 1` tokens. §2 and §4.3 corrected, and
     `sequence_tokens` is now recorded on every row. (Constant-1 error; no data
     existed.)

  2. **`a_iid` was not the i.i.d. parameter.** The plan defined it as total
     accepted / total drafted, which with gamma constant is algebraically
     identical to `alpha_round`. The plan now separates `alpha_total_ratio`
     (a consistency check) from `alpha_iid` (the memoryless parameter solved
     from the mean prefix length). Measured on the 57 committed traces,
     `alpha_iid > alpha_round` in 53 of them.

  3. **The effective-rope assertion could not fail.** It compared the built
     `inv_freq` against a reference recomputed from the same built config, so it
     matched unconditionally and could not have detected the mis-anchoring it
     exists to catch. It now compares against a reference derived from the
     candidate's DECLARED type/factor/anchor, and reports separately when the
     built rope is anchored on the context.

  Also corrected, without changing the design: the gate scored a candidate over
  `context + 1024` positions (which would have measured the 128k native control
  past its own window); the gate's baseline selection could pick the 32k control
  as the 128k reference; the generation cap was not enforced in the engine, so a
  speculative run could exceed `max_new_tokens`; the stage configs shipped
  `temperature: 1.0` against this plan's greedy contract; and the document pool
  was too short at the 512k rung once the prompt re-encoding adjustment was
  accounted for.

- **2026-10-06 — REVISION: baseline design, rung set, ordering, and scope.
  Recorded before any run of this campaign; no data has been seen under this
  plan.** Every change below is a decision by the operator, taken on the cost
  estimate in `configs/mlsys_manifest.yml`. The pre-registration is only honest
  if the change precedes the data, so this block is dated and committed ahead of
  the first GPU cell.

  **1. Target-only baselines are measured two ways per rung.** For each rung:

  | arm | documents | generated tokens | used for |
  |---|---:|---:|---|
  | target-only, full | 3 | 1024 | losslessness, and the end-to-end paired ratio |
  | target-only, short | 7 | 128 | decode-only tok/s for the paired ratio |
  | speculative | 10 | 1024 | acceptance, and both ratios |

  **Paired speedup is reported on decode-only tok/s for all 10 documents**, with
  the end-to-end ratio reported beside it for the 3 full pairs.

  Justification, as required: **losslessness is an implementation property**, not
  a throughput measurement. Under greedy decoding the speculative output must be
  token-identical to the target-only output, and that can only be checked against
  a target-only run of the *same length*: **3 full-length checks at 1024 tokens,
  plus a 128-token prefix check on the other 7**. Throughput, by contrast, needs
  a steady-state rate, and a rate does not need the same token count on both
  sides. **Prefill is identical across
  arms and is under 5% of the decode wall** — measured 10 s of prefill against
  4495 s of decode at 128k (0.22%) and 97 s against 18022 s at 512k (0.54%) — so
  the end-to-end ratio is a prefill-weighted restatement of the decode ratio
  rather than an independent quantity, and dropping prefill from the primary
  metric costs nothing while saving ~35% of the target-only wall time.

  **`decode_tps` is pre-registered here**: `tokens_generated / (time_sec −
  ttft_sec)`, i.e. generated tokens over the post-prefill wall. The same
  definition is used in both arms, so any convention cancels in the ratio.

  **2. The gated rungs are 256k and 512k, run as separate stages.** 256k runs
  first. 512k is conditional on 256k completing cleanly and on remaining credit,
  and runs in its **own instance session with the watchdog raised to 40 h**,
  because it is ~26 h at this design and the 20 h watchdog would kill it
  mid-stage.

  **3. The diverse-pool arm at 128k is kept**, at the same baseline design as the
  core-10 rungs.

  **4. Seeds are settled at `[42]`.** Greedy decoding is deterministic, so a
  second seed would repeat a run rather than replicate it. The **document** is the
  replication unit and the basis of every interval. This question is closed.

  **5. Ordering changed**: the cross-implementation validation now runs AFTER
  `natural_f1_128k` and is **vLLM-only**, on that stage's first 3 documents,
  compared against `natural_f1_128k`'s own RASD rows and token sidecars. The
  duplicated RASD runs are removed from it.

  **6. A new first stage, `engine_cap_smoke`, runs immediately after gate
  calibration and before every speculative stage.** The generation cap has never
  executed on a GPU, and it is the one change that, if wrong, would silently
  invalidate every cell rather than fail loudly.

  **7. Scope.** `synthetic_spec_gated` remains optional and is dropped from the
  primary claim if it does not run; the payoff rule is evaluated on the natural
  arm alone. The vLLM ladder is retained, scoped to **plain (non-speculative) vLLM
  decode at the same rungs and documents**, because the cross-implementation
  stage covers only 128k and only speculative decoding: without the ladder there
  is no production-stack reference at the 256k and 512k rungs, which is where the
  payoff boundary is claimed.

### Revision: the gate tolerance is FIXED at 1.5x, not derived (2026-10-06)

*Before any GPU run; no data seen.*

The plan contradicted itself: section 6.2 states the normative tolerance as
"within 1.5x the native baseline", while the calibration note said the
threshold was "not yet set -- set in S0 from the positive controls' measured
spread". One of the two had to go.

**Decision: the tolerance is fixed at 1.5x, pre-registered here, and the
"derive it from the positive-control spread" rule is withdrawn.** The
derivation was abandoned for three reasons, none of which is that 1.5 is more
convenient:

1. It cannot be estimated from the controls we have. There are three
   positives, measured at three *different contexts* (32k, 64k, 128k) with one
   seed each and no replication unit. Their spread is dominated by the context
   change and by single-measurement noise, so it does not estimate the
   quantity the threshold needs (the gate's measurement noise at a *fixed*
   context).
2. Every available estimator is monotone in the direction of loosening. Both
   `max(ratio)` and `mean + k*sd` can only raise the tolerance above the
   positives' ratios; neither can produce a tighter gate than 1.5x unless the
   positives happen to sit far below it. A looser gate admits exactly the
   configurations the gate exists to exclude, so the failure mode is silent.
3. It made the threshold tunable *after* seeing a candidate's ratio, which is
   the thing a pre-registration exists to prevent.

The S0 controls therefore change role: they are no longer the source of the
threshold, they are a **falsification test of the gate at the fixed
threshold**. Each positive control must pass at 1.5x and each negative control
must fail at 1.5x, and a disagreement means the gate is wrong.

Two consequences of fixing the threshold are recorded here because they are
claimed in the results:

* Each control must be **measured at its own context** against a **declared**
  native baseline at that same context and model. A control that reports "no
  baseline" was not measured, so it is neither a pass nor a detected failure --
  it is a failure of the calibration and stops the campaign.
* The positive controls at 32k and 64k are within Llama-3.1-8B's native
  window, so they test the gate's machinery (rope assertion, early-EOS
  detection, degeneration shares, the full-length prompt path) rather than a
  rope extension. That is deliberate: the gate must be shown to pass a
  configuration known to be correct before its verdicts on extensions mean
  anything.

### Revision: new stage `rope_intervention_128k` (2026-10-06)

*Before any GPU run; no data seen.*

**Motivation.** Every comparison in section 7 changes either the context
length or the target's rope configuration together with other things, so none
of them isolates what a *rope intervention alone* does to acceptance. If
acceptance at 128k is a property of the target's positional frame rather than
of the context length, that should be visible with the context HELD FIXED and
only the rope changed.

**Design.** Llama-3.1-8B target with Llama-3.2-1B draft, the same 10 core
documents as `natural_f1_128k`, the same 128k context, greedy, 1024 tokens,
`ignore_eos=true`, speculative runs only (no target-only arm: this is a
comparison between two speculative arms, and the losslessness partner is not
required because neither arm is being reported as a baseline ratio).

Two arms, paired by document:

| arm | target rope |
|---|---|
| `R_native` | the model's shipped `llama3` block, untouched |
| `R_llama3_f16` | `rope_type: llama3`, the **shipped dict** with only `factor` changed to 16 |

The shipped dict is used deliberately, so the only difference between the arms
is the factor. Nothing else is varied.

**Gate.** The factor-16 arm must first pass the coherence gate at **128k** at
the fixed 1.5x tolerance. If it fails, that failure **is the result**: the arm
is not run, and it is reported as "a llama3 factor-16 intervention at 128k does
not produce a coherent target", not as an acceptance difference. This stage is
the first to gate an intervention rather than an extension, and it is reported
as such.

**Pre-registered comparison (C6).** Per document, the paired difference in
`alpha_round` (factor-16 minus native), with a document-bootstrap 95% interval,
`n_boot = 10000`, resampling documents, the same interval machinery as every
other comparison in section 5.

**Pre-registered equivalence margin: 0.05.** The two arms are declared
**equivalent** when the 95% document-bootstrap interval on the paired
difference lies entirely inside **[-0.05, +0.05]**. The rule is asymmetric on
purpose and is fixed now, before the data:

* interval entirely inside [-0.05, +0.05] -> **equivalent**: acceptance is
  insensitive to a factor-16 rope intervention at fixed context.
* interval entirely outside [-0.05, +0.05] (both bounds above +0.05 or both
  below -0.05) -> **a rope effect exists**, reported with its direction.
* interval overlapping a margin boundary -> **inconclusive**, never rounded to
  whichever side is more interesting. An inconclusive result is reported as
  inconclusive.

Absence of a detected difference is not evidence of equivalence: only the
interval-inclusion rule above licenses the word "equivalent", and with n = 10
documents the interval will often be too wide to license it. That outcome is
expected, is reported as inconclusive, and is not a failure of the stage.

**Cost.** ~$82, not the ~$35 first estimated. The arms alone are 2 x 10 = 20
speculative runs; spec decode is 0.55 s/token at 128k, so 20 x ~570 s = 3.2h at
$22.32/h = $71, and the gate pass at 128k adds ~0.5h = ~$11, giving ~$82 -- the
figure the manifest records as `est_cost_usd`. The stage is approved at that
figure. The earlier "$35" was an arithmetic slip in this revision (3.2h was
costed as if each hour were ~$11); it is corrected here BEFORE the stage runs,
and the correction is the reason the number is stated with its derivation.

### Revision: session ceiling raised to $850 (2026-10-06)

*Before the affected stages run; no data seen.*

The reviewed cost projections for the approved set totalled **$729** against the
$700 session ceiling, so the aggregate guard would have refused the LAST
approved stage (`vllm_ladder`, $56) on arithmetic alone -- a production-stack
baseline dropped by a ceiling rather than by a judgement about its value.

Two things changed the total since the ceiling was set:

* `gate_calibration` rose from $13 to $22, because every control now carries a
  declared native baseline at its own context (9 rows instead of 5, three of
  them at 128k). A control with no baseline cannot be measured, so the extra
  rows are the price of the verdicts meaning anything.
* `rope_intervention_128k` was added at $82, and gains a third arm below, taking
  it to ~$120.

**The ceiling is now $850**, which covers the approved set (~$767) with ~$83 of
headroom. Recorded here before the stages run. The per-stage approval threshold
($300) is unchanged: the ceiling bounds the session, it does not authorise any
single stage that has not been approved by name.

### Revision: rope_intervention_128k gains a factor-32 arm (2026-10-06)

*Before the arm exists in any config; no data seen.*

The stage compares the shipped `llama3` rope against interventions that change
only its `factor`. With one treated arm, a difference (or an equivalence) is a
statement about factor 16 alone, and cannot be read as a property of the
intervention -- factor 16 may sit close enough to the shipped 8 that nothing
moves, or far enough that coherence breaks, and one point cannot tell those
apart.

**Three arms, the same 10 core documents, paired by document, spec-only, at
128k, greedy, 1024 tokens, `ignore_eos=true`:**

| arm | target rope |
|---|---|
| `R_native` | the shipped `llama3` block, untouched |
| `R_llama3_f16` | shipped dict, only `factor` 8 -> 16 |
| `R_llama3_f32` | shipped dict, only `factor` 8 -> 32 |

The shipped dict is spread and one key overridden, so `factor` is the only
difference in each treated arm. The anchor stays where the model ships it
(`original_max_position_embeddings = 8192`); moving it as well would confound
the factor with the anchor, which is the error the correction note documents.

**Primary comparisons.** `R_native` vs `R_llama3_f16` and `R_native` vs
`R_llama3_f32`, each a paired difference in `alpha_round` (treated minus native)
with a document-bootstrap 95% interval, `n_boot = 10000`, resampling documents.
The `f16` vs `f32` difference is reported as a secondary contrast and is NOT
part of the primary claim.

**Equivalence margin: 0.05, unchanged and as pre-registered for this stage.** The
three-way decision applies to each primary comparison independently:

* interval entirely inside [-0.05, +0.05] -> **equivalent** at that factor;
* interval entirely outside -> **a rope effect exists**, reported with direction;
* interval overlapping a boundary -> **inconclusive**, never rounded.

"Equivalent at factor 16" and "equivalent at factor 32" are separate findings.
Absence of a detected difference is not evidence of equivalence, and with
n = 10 documents an inconclusive result is likely at one or both factors; it is
reported as inconclusive and is not a failure of the stage.

**Gate.** Each treated arm must pass the coherence gate at **128k** on its own
before it runs. **A gate failure is recorded as THAT ARM's result**: the failing
arm is not run and is reported as "a llama3 factor-N intervention at 128k does
not produce a coherent target", while the remaining arms still run. The
threshold is never relaxed to make an arm agree -- a threshold adjusted to admit
the thing it excluded is no longer a gate.

**Cost.** ~$120: 30 speculative runs (3 arms x 10 documents) x ~570 s = 4.75h =
$106, plus 3 gate rows at 128k ~= $13.

### Losslessness verdicts

`losslessness: required` on a stage means, exactly:

* every speculative cell has a **verified prefix of at least 128 tokens**, and
* every cell whose partner reached the full 1024 tokens is **LOSSLESS over all
  1024**.

The verdict vocabulary is fixed:

| verdict | meaning |
|---|---|
| `LOSSLESS` | the partner reached 1024 and every token matches |
| `LOSSLESS_PREFIX_n` | the partner produced n < 1024 tokens and the first n match |
| `MISMATCH` | the first divergence, with position and both tokens named |
| `NO_PAIR` | genuinely no same-document partner exists |
| `BAD_PAIR` | a partner exists but the request fields disagree, so the pair is refused |

The pair guard deliberately does **not** compare `max_new_tokens`: the short
baselines generate 128 tokens against a 1024-token speculative run, which is the
case the prefix verdict exists for. It requires instead that the partner is not
**longer** than the run it checks, since a longer partner cannot be a prefix
comparison.

---

### Revision: the prompt window is defined by the RUNG, not by the arm's own cap (2026-10-06)

*Before any of these stages has run; no data seen.*

**What was wrong.** The PG-19 prompt builder took its prompt length from the
run's own `max_new_tokens` (`prompt = [0, C - gen_tokens - 1)`). Every arm was
therefore given a *different* prompt whenever the arms generate different
numbers of tokens. This plan requires the three full-length target-only arms and
the seven short ones to be checked against the speculative run token-by-token,
and the pair guard in `src/analysis/losslessness.py` refuses a pair whose
`prompt_sha256` or `prompt_tokens` disagree — as it must, because token equality
across two different contexts means nothing. The consequence was that the seven
128-token baselines could never have paired: they would have come back
`BAD_PAIR`, the prefix verdict this plan introduces would never have been
produced, and a stage declaring `losslessness: required` would have failed on a
defect introduced by the prompt builder.

**The rule, now.** The prompt window is a property of the rung:

```
prompt length = context_length - prompt_gen_tokens - 1
```

where `prompt_gen_tokens` is the rung's generation length (1024 for the 128k and
256k rungs), and every arm of a rung sets it to the same value regardless of how
many tokens it itself generates. A level that omits it keeps the previous
behaviour (`max_new_tokens`), so no existing single-arm config changes.

The four stage configs that carry a `TARGET_SHORT` arm declare it explicitly.
The important consequence for the tables: a short baseline's `sequence_tokens` is
`context_length - 1024 + 128`, NOT `context_length`. It is a 128-token
generation into the same 1024-token prompt, which is what makes it a prefix
partner; nothing is reported about its sequence length, and it is not a rung of
its own.

### Revision: a helper follows its parent, and an unmeasured gate is not a failed arm (2026-10-06)

*Before any of the affected stages has run; no data seen.*

**Allowlist.** Helper stages (`*_losslessness`, `*_doc_intervals`,
`*_round_acceptance`, `rope_intervention_gate`,
`rope_intervention_128k_comparison`) are admitted by their PARENT's presence on
`MLSYS_ONLY_STAGES`, never by their own name. `rope_intervention_gate` is
declared as a stage (it is what decides whether the intervention's treated arms
may run), so it did not match the suffix rule, was refused by the allowlist, and
the stage then **recorded both treated arms as having failed a gate that was
never measured**. That is the exact failure mode this plan's negative-control
rule exists to prevent — a control counts as detected only if it was measured
and failed. A helper's cost is priced under its parent.

**A gate that produced no verdicts is an INVALID stage, not a verdict.** The
intervention stage now distinguishes two cases it previously conflated:

| gate outcome | meaning |
|---|---|
| the arm's row is present and fails | that arm's measured result: no coherent target at 128k, not run |
| the gate produced no row for the arm | a plumbing mismatch: `STAGE_INVALID`, no arm's fate recorded |
| the gate produced no file at all | `STAGE_INVALID`, and the speculative stage is skipped rather than spending $120 on arms of unknown coherence |

### Revision: the correction-note evidence declares its baselines (2026-10-06)

*Before the stage has run; no data seen.*

`configs/mlsys_correction_candidates.json` now carries two native rows
(`B_llama2_native_32k`, `B_llama2_native_128k`, `rope_type: none`,
`native_baseline: true`). The gate refuses to compute any ratio when no row is
flagged `native_baseline`, so without them the evidence stage could not have
produced a single number. Each mis-anchored / correctly-anchored pair is scored
against the native row at its own context, which keeps the correction note's
comparison ("the anchor explains the perplexity damage") at a fixed context
rather than across two.

**Cost.** The two extra rows raise this stage's projection from $33 to $41
(8 -> 10 measured candidates, 1.5h -> 1.9h), so the approved nine-stage total is
**$775** against the $850 session ceiling, headroom **$75**.

### Revision: the correction-note ratios are against an unscaled-OOD reference (2026-10-06)

*Before the stage has run; no data seen. Supersedes the labels introduced by the
previous correction-note revision.*

**What the ratios measure.** Each YaRN candidate is scored against the unscaled
Llama-2 at the SAME context, so the ratio isolates the effect of the ANCHORING
CHOICE for a model that is already being shown a context far outside the 4096
tokens it was trained on. It is *not* a quality comparison against an
in-distribution model: at 32k and 128k no anchoring choice puts Llama-2 back in
distribution, so no such comparison exists to make.

**Labels.** The reference rows are therefore named and labelled
`unscaled_ood_reference`, not `native`:

| row | context | role |
|---|---|---|
| `B_llama2_native_4096` | 4096 (the training window) | `in_distribution_reference` |
| `B_llama2_unscaled_ood_32k` | 32768 | `unscaled_ood_reference` |
| `B_llama2_unscaled_ood_128k` | 131072 | `unscaled_ood_reference` |

The gate's CSV carries `role`, `reference_role` and `baseline_role`, so every
ratio row states which kind of reference it was divided by. `native_baseline`
remains the mechanical flag the baseline lookup uses, and reading it as a claim
of in-distribution validity is the misreading this revision exists to prevent.

**The in-distribution measurement is reported DESCRIPTIVELY, beside the ratios**
and never as their denominator. It answers "what does this model's perplexity
look like where it is valid", and with n = 1 context and n = 1 seed the answer is
a single number to quote alongside the table, not a baseline.

Consequence for the correction note's wording: the claim it can support is
"correct anchoring removes most of the perplexity damage *at these
out-of-distribution lengths*", not "correct anchoring restores quality".

### Revision: a stage's outputs belong to the attempt that wrote them (2026-10-06)

*Before any of these stages has run; no data seen.*

Three contract changes, all of which make "the stage passed" a statement about
THIS run rather than about the filesystem:

1. **Fresh attempt.** Every stage archives its own outputs (CSV, helper CSVs,
   partials) into `attempts/<utc>/<stage>/` before it runs, and marks the attempt
   with a fresh-attempt file. Prerequisite checks — the calibration, the cap
   smoke, the coherence gate — read the stage's recorded exit code and its
   attempt marker, never a file that a previous attempt may have left. A
   prerequisite that was refused or failed is a HARD STOP for its dependents.
   `attempts/` is excluded from the results pull: it is diagnostic, and a
   previous attempt's CSV in the delivered corpus would be indistinguishable
   from this run's.

2. **Row counts.** `expected_rows` counts the planner's RUN lines (the "N runs
   total." trailer is not a run, and counting it made every expectation one too
   high), and a plan that yields zero runs is a stage FAILURE, never a reason to
   skip the row check. A stage is complete only if it wrote exactly the planned
   number of `status=ok` rows.

3. **The sequence identity, on every stage's rows.** Every row must satisfy
   `prompt_tokens + 1 (BOS) + tokens_generated == sequence_tokens`, where
   `sequence_tokens` is the sequence the engine actually built. It is recorded
   from the metrics, not from the document plan: the plan's value is
   `prompt + 1 + the rung's generation length`, which is wrong for a 128-token
   baseline (a shorter generation into the rung's prompt) and for any run that
   stops early.

A stage's exit code now reaches its caller, so a failure cannot be silently
treated as a success by a check that only knew how to recognise a refusal.

**Cost (updated).** The in-distribution row is a third reference measurement, so
`correction_note_evidence` now measures 11 candidates rather than 10: its
projection is **$45** (2.1h), and the approved nine-stage total is **$779**
against the $850 session ceiling, headroom **$71**.

### Revision: `sequence_tokens` is measured by the engine, not reconstructed (2026-10-06)

*Before any of these stages has run; no data seen.*

The identity every row must satisfy is unchanged:

```
prompt_tokens + 1 (BOS) + tokens_generated == sequence_tokens
```

What changes is WHERE the right-hand side comes from. It was written by the
runner as `prompt_tokens + 1 + tokens_generated` and asserted by the checker as
the same expression, so the check was a restatement of the row's own arithmetic:
it could not fail, and it therefore verified nothing. The three numbers are now
obtained independently:

| number | source |
|---|---|
| `prompt_tokens` | the runner, from the prompt's token ids |
| `tokens_generated` | the run's token SIDECAR (`tokens/<run_id>.json`), cross-checked against the CSV's count |
| `sequence_tokens` | the ENGINE: `int(generated_ids.shape[1])`, the length of the final sequence tensor the generation loop holds |

The check is therefore a cross-check between the runner's tokenizer, the ids the
engine emitted, and the engine's own tensor -- including the BOS accounting,
which is the quantity that has been wrong on this project before. A divergence
is a real defect: a BOS counted twice, a prompt that is not the prompt the engine
saw, or a loop that stopped somewhere other than where it reported.

The stub measures the same object: it builds the held sequence by running its
loop (the seed the first round verifies, then each round's accepted prefix and
bonus token) and takes its length, and the sidecar is that same object's slice
after the engine's prompt tensor. A first attempt used the trace's final
`kv_len_after`, which is ONE LESS than the emitted count -- the last token's keys
and values are computed by a forward that never happens -- and the identity check
caught it, which is the evidence that the check now has power.

### Revision: the correction contrast is PAIRED, and the claim it supports (2026-10-06)

*Before the stage has run; no data seen. Supersedes the "two native rows" wording
of the earlier correction-note revision.*

**The claim this evidence supports is a PAIRED anchored-vs-misanchored contrast
at each out-of-distribution context, relative to the unscaled OOD reference
measured on the SAME sample. Nothing about restored quality follows.** At 32k and
128k no anchoring choice puts Llama-2 back in distribution, so there is no
in-distribution target for a "restored" claim to be measured against.

**Pairing is enforced, not assumed.** Every candidate and its same-context
reference carry the SAME SEED, which in this pipeline means the same document,
the same offset inside it, the same prompt ids and the same continuation ids. The
gate records both hashes on every row (`prompt_sha256`, `continuation_sha256`,
plus the reference's two) and applies `pairing_verdict` BEFORE computing any
ratio:

| pairing | meaning | ratio |
|---|---|---|
| `paired` | prompt and continuation hashes agree | computed |
| `unpaired` | either hash differs, or a hash is missing | **not computed**; the row is recorded as an error (`status=unpaired_reference`) |
| `cross_context` | an extension judged against the declared native baseline at a different context | computed, and labelled: the plan's coherence comparison, in which the windows differ by construction |
| `descriptive` | the 4096 in-distribution row | **none**; reported beside the ratios |

Two blank hashes are not agreement: a missing hash is `unpaired`. The reference
seeds in all four candidate files were aligned to the candidates they score, and
a test asserts it for every file, so the rule cannot be satisfied by accident in
one file and violated in another.

**The in-distribution row.** `B_llama2_native_4096` (4096, the training window,
shipped rope, `in_distribution_reference`) answers "what does this model's
perplexity look like where it is valid". It is descriptive: no ratio, no
denominator role, not paired with anything.

### Revision: the vLLM comparison uses the ids the TARGET was fed (2026-10-06)

*Before the stage has run; no data seen.*

The token sidecar now records `engine_input_ids`: the ids the target was actually
fed, taken from the engine's own tensor, INCLUDING the leading BOS.
`prompt_tokens` and `prompt_sha256` remain the prompt WITHOUT the BOS, because
that is the sequence the engine's prompt builder is defined over.

The distinction is not cosmetic. The engine's input is one token longer than the
recorded prompt, and that token sits at the FRONT, so it shifts every position's
rotary phase. A vLLM row given the no-BOS ids is a comparison against a sequence
the target never conditioned on, and it would have looked perfectly matched
because both sides agreed with each other.

So: vLLM receives `engine_input_ids` as `prompt_token_ids`; the row records
whether the ids came from the engine (`prompt_ids_from_engine`); the verdict
refuses a row whose ids are not the engine's; and the loader refuses a sidecar
that lacks the field, or whose `len(engine_input_ids) != prompt_tokens + 1`.

### Revision: the baseline row must be the attempt that ran, at the rung it claims (2026-10-06)

*Before the vLLM stage has run; no data seen. No metric, unit of independence,
document count, interval method or comparison changes.*

Four corrections to how a vLLM row is produced and how a vLLM stage is judged.
None of them touches a headline number; all of them decide whether a number that
gets into the table is the number it says it is.

1. **The engine-provenance marker is the parent's to set.** Whether the ids fed
   to vLLM are the ids the target engine was fed is a fact about the RASD token
   sidecar, not about vLLM's run, and the worker never sees the sidecar, so it
   cannot report it. The parent now sets `prompt_ids_from_engine` from the cell
   it is running. Before this, the field was empty on every row the parent
   wrote, so **no vLLM row could be unit-matched at all** and the stage would
   have reported success while producing a comparison table with zero comparable
   rows. Failing closed, silently.

2. **The attempt ladder retries.** The loop that tries the three configurations
   had its execution outside its own body: all three specs were built and only
   the LAST was run, so every row would have come from the 64k fallback attempt
   while `attempt=3`, `config_used` and the row's rung said otherwise. The
   attempt that ran is now the attempt that is recorded.

3. **A stage with no unit-matched row is failed, not ok.** "Reported as invalid
   rather than as a speedup" (§4.2) is enforced at the stage level: if no row is
   `unit_matched=yes`, the stage exits non-zero and the manifest records it
   failed. A baseline that cannot be compared is not a baseline.

4. **A shorter-context fallback is reported but never certified.** The ladder's
   last resort runs `max_model_len=65536`. That is a legitimate row to report --
   "vLLM could not load 128k here" is exactly the honest result §5 asks for --
   but it is not the 128k rung's counterpart, and certifying it would put a 64k
   throughput in the 128k speedup column with nothing else in the row to show
   it. Rows now record `max_model_len`, and the verdict refuses a row whose
   model length is shorter than its rung.

Consequence for reading the results: a vLLM row that ran at a shorter context
than its rung appears in the CSV, is labelled `unit_matched=no`, and does not
enter any speedup ratio. If that happens at 128k, the vLLM reference for that
rung is reported as unavailable rather than as a slower or faster number.

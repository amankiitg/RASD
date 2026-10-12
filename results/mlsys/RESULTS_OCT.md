# RASD — October results

**Headline: at 1M context, temperature 0.7 sampling gives 2.37x over our own target-only
path, on 5 books, with 0 of 26 sampled rows degenerate.** The greedy campaign that
preceded this run could not say that: 9 of its 15 rows looped, which both inflated
acceptance and left the headline resting on a single document. Sampling fixed that, and
the loop detector is applied to every row here.

Two conditions, reported separately and never mixed:
**A. sampling, T=0.7, seed 42** (the new session, 2026-10-12) — §1, §3, §4, §5.
**B. greedy, T=0.0, seed 42** (the Oct 10-11 session) — §2, kept as its own condition
with its own corrected headline.

Target `gradientai/Llama-3-8B-Instruct-Gradient-1048k` (native 1M, `rope_type: none`, no
scaling), draft `meta-llama/Llama-3.2-1B`, 8x A100 80GB SXM4 (`gpu_8x_a100_80gb_sxm4`,
us-east-1, $22.32/hr), NF4 KV, `max_new_tokens` 128, `spec_steps` 4, 3-6 PG-19 books.

**Every tok/s in this document is `tokens_generated / time_sec`, computed here, never the
CSV's `throughput_tps` field**, which carries two significant figures (0.31 vs 0.31475).
Speedups divide by the MEAN target-only tok/s **of the same session at the same context**.

---

## 1. Condition A — the 1M sampling rows (T=0.7), per row

`SAMP_1M_SPEC`, 1M context, 6 books configured. Periodicity is from the per-row token
traces (`src/analysis/repetition.py`, the same detector the incident stop installed).
A row with a nonzero dominant period and **`first_periodic_token = 128`** — i.e. the row
is never periodic to its end — is **"no cycle"**: 128 tokens is the row length, so the
periodic tail is empty. "Verdict" is the detector's absolute rule, not this table's.

| document | accept | tok/s (exact) | rounds | s/round | period | agreement | first periodic | tail | verdict |
|---|---|---|---|---|---|---|---|---|---|
| pg19_train_0 | 0.3776 | 0.10801 | 50 | 23.7 | 34 | 0.0532 | **128** | 0.00 | no cycle |
| pg19_train_1 | 0.6838 | 0.14080 | 34 | 26.7 | 48 | 0.1125 | **128** | 0.00 | no cycle |
| pg19_train_915 | 0.6250 | 0.12687 | 37 | 27.3 | 54 | 0.0541 | **128** | 0.00 | no cycle |
| pg19_train_2204 | 0.5437 | 0.12208 | 40 | 26.2 | 43 | 0.1059 | **128** | 0.00 | no cycle |
| pg19_train_2768 | 0.3929 | 0.10608 | 50 | 24.1 | 64 | 0.0781 | **128** | 0.00 | no cycle |
| pg19_train_1726 | — | — | — | — | — | — | — | — | **EXCLUDED — data, see below** |

The dominant period column is non-zero for every row and is **not** evidence of a cycle:
it is the argmax over candidate periods of a near-noise agreement (0.05-0.11), and the
detector says so by `first_periodic_token = 128`. That is exactly the case the operator's
rule describes.

**`pg19_train_1726` is excluded, and it is a data defect, not a dropped row.** The engine
refused it twice (original attempt and the step-e retry) with
`RuntimeError: pg19_train_1726: continuation is 765 tokens, need 1024; the document is too
short for this rung` — book 1726 has only 765 tokens after the 1M window, and the config
asks for a 1024-token scored continuation. It must be reported as a data exclusion, and the
book either dropped from the pool or given a shorter scored continuation.

### Clean-only means, and the speedup against this session's own target-only path

| context | arm | n | mean accept | mean tok/s | documents |
|---|---|---|---|---|---|
| 1M | spec | **5** | **0.5246** | **0.12077** | train_0, 1, 915, 2204, 2768 |
| 1M | target-only (**subsampled, 2 of 5**) | 2 | — | **0.05103** | train_0, 1 |

**1M clean speedup = 0.12077 / 0.05103 = 2.37x (spec n=5).** Per book against the same
mean target-only rate: 2.12x, 2.76x, 2.49x, 2.39x, 2.08x.

**The target-only rows are subsampled and this is deliberate.** Measured target-only
throughput varied under 2% across books in the greedy condition at 1M
(0.05113 / 0.05132 / 0.05236), so two books at 1M and one at each aligned rung were run
instead of one per book. The speedup therefore divides the spec mean by a two-book mean; it
is not a paired per-book ratio. The two measured rates agree to 2.8% (0.05175 and 0.05031),
which is the support for subsampling.

### The greedy loop does not survive sampling

| condition | rows analysed | degenerate (loop) | acceptance vs context |
|---|---|---|---|
| B. greedy (Oct 10-11) | 15 | **9 (60%)** | falls: 0.3273 → 0.2705 → 0.2132 (n=1 each) |
| **A. sampling T=0.7 (this run)** | **26** | **0 (0%)** | **rises: 0.4349 → 0.4892 → 0.5246** (n=3/3/5) |

Sampling at T=0.7 was chosen precisely to attack the greedy cycle, and it worked: zero rows
in the whole session trip any repetition ceiling. The 26 rows are every ok row of the
session — 21 with speculation (1M / 512k / 128k / rope) and 5 target-only. Longest verbatim
repeat span anywhere: **8 tokens** (the ceiling is 128); share of tokens inside a repeated
16-gram everywhere: **0.00** (the ceiling is 0.60).

One honest exception: `ROPE_SAMP_NATIVE_SPEC_pg19_train_1_s42` has a periodic tail of
**0.2734** from token 93 (period 34). That is below the 0.50 periodic-tail ceiling and
below every other ceiling, so the detector calls it clean — but it is the only sampled row
with any periodic structure at all, and it is named here rather than averaged away.

---

## 2. Condition B — the greedy results (Oct 10-11), kept separate

**Not comparable row-for-row with §1**: different temperature, and the greedy session read
each context from the start of the book, so its continuations were different passages at
different contexts (the aligned pools of §3 did not exist yet).

| context | arm | n | accept | tok/s | speedup |
|---|---|---|---|---|---|
| 1M | spec | **1** | **0.2132** | **0.08411** | **1.61x** (vs target-only 0.05236) |
| 1M | target-only | 1 | — | 0.05236 | — |
| 512k | spec | 1 | 0.2705 | 0.18454 | 1.84x (vs target-only mean 0.09931, n=2) |
| 512k | target-only | 2 | — | 0.09931 | — |
| 128k | spec | 1 | 0.3273 | 0.60384 | **not computable — the 128k target-only arm was lost** |

**The corrected headline stands: 1M clean n=1, acceptance 0.2132, 0.08411 vs 0.05236 =
1.61x.** Two claims from earlier drafts are withdrawn and stay withdrawn: the 1M "2.60x
mean" and "acceptance does not degrade with context". Both came from averaging
loop-inflated rows (0.6429 and 0.7419 on the looping books against 0.2132 on the clean one)
— a greedy cycle is trivially accepted by a greedy draft. §1 replaces both with n=5 under
sampling. The 128k arm was lost to the Mac-side SSH drop documented in the session record,
not to the pod.

---

## 3. Condition A — aligned scaling at 512k and 128k

The aligned pools are built so that every rung's window **ends at the same document
position as the 1M arm**, and the scored continuation is the last 1024 tokens of the
window. Alignment is verified from the token records, not from the pool's name:

* `continuation_sha256` from the token sidecar, per book and per arm;
* the engine's window landing drift, `prompt_source_len - (context - 1024 - 1)`;
* the cross-arm offset is the **difference of the two drifts**, and the token overlap is
  `(1024 - offset)/1024`. Table: `results/mlsys/sampling_session/alignment.csv`.

| book | drifts (1M / 512k / 128k) | 1M vs 128k | 1M vs 512k |
|---|---|---|---|
| pg19_train_0 | 0 / 0 / 0 | `continuation_sha256` **identical**, overlap **100%** | identical, **100%** |
| pg19_train_1 | 101 / 38 / 9 | offset **92** → overlap **91.0%** | offset 63 → 93.8% |
| pg19_train_915 | 65 / 61 / 17 | offset **48** → overlap **95.3%** | offset 4 → **99.6%** |

So the cross-context rows share 91-100% of their scored tokens — a real improvement on the
greedy session, where different contexts read entirely different passages, but **not
identical windows**, and the numbers below carry that caveat. Only book 0 is exactly
aligned, and it is the only book for which a cross-context acceptance difference is
strictly like-for-like.

| context | arm | n | accept (mean) | tok/s (mean) | per-book tok/s | target-only | speedup | vLLM 0.6.3 bf16 TP=8 (n) | vLLM / target-only | vLLM / spec |
|---|---|---|---|---|---|---|---|---|---|---|
| 128k aligned | spec | **5** | **0.4681** | **0.71557** | 0.53084 / 0.65789 / 0.83175 / 0.86744 / 0.68993 | **0.28310** (n=1) | **2.53x** | **7.1575** (1) | **25.3x** | 10.0x |
| 512k aligned | spec | **5** | **0.4745** | **0.24187** | 0.20607 / 0.22362 / 0.30197 / 0.24791 / 0.22976 | **0.09411** (n=1) | **2.57x** | **0.56979** (1) | **6.1x** | 2.4x |
| 1M | spec | **5** | 0.5246 | 0.12077 | §1 | 0.05103 (n=2) | **2.37x** | **0.14840** (2) | **2.9x** | 1.2x |

**ALL THREE RUNGS NOW COVER THE SAME FIVE BOOKS** (train_0, 1, 915, 2204, 2768), which is
what the n=5 extension was for. Nothing was re-measured for books already present: the
three Oct 12 rows per rung are joined by the two the rung lacked, and the target-only
rates are the Oct 12 ones (measured variation across books under 2%). The extension ran
2026-10-12 on `gpu_8x_a100_80gb_sxm4` (5 stages, 10 new rows, all `ok`, $25.36) with new
stage ids (`SAMP_EXT_*`, `ROPE_EXT_*`) so nothing could be overwritten or confused.

**The acceptance-vs-context trend is unchanged and now rests on matched books:
0.4681 -> 0.4745 -> 0.5246.** The n=3 figures it replaces (0.4349 / 0.4892 / 0.5246) were
computed over unequal book sets, so a "trend" there could have been a book effect; on the
identical five it cannot be. Acceptance **rises** with context; the speedup stays flat
(2.53x / 2.57x / 2.37x); the 128k speedup moves from 2.38x to **2.53x** purely because
books 2204 and 2768 are faster on that rung than the three they join.

One of the ten new rows is flagged by the repetition pass: `ROPE_EXT_SAMP_F16_SPEC` on
`pg19_train_2768` has 77% of its tokens inside a repeated 16-gram (above the 60% ceiling),
with a 36-token verbatim span. It has **no periodic tail** (`rep_periodic_tail_share` 0.0,
`first_periodic_token` 128, i.e. no cycle), and its single acceptance window survives the
non-degenerate filter, so its reported acceptance is unchanged at 0.7121; it is reported
as a text-repetition observation, not as a looping row.

Per-row acceptance at 128k: 0.2712 / 0.4255 / 0.6081 (train_0 / 1 / 915); at 512k:
0.3349 / 0.4062 / 0.7266. All 11 spec rows at the three contexts are non-degenerate.

### The vLLM column (2026-10-12, 8x A100 80GB SXM4, `results/mlsys/vllm_run` → pulled to the session pull dir)

Every number in it is `tokens_generated / time_sec` on the **same prompts** — the
sidecars' `engine_input_ids`, the ids the target was actually fed — with
`unit_matched=yes` and `prompt_ids_verified=yes` on all four rows. It is a different
engine at a different precision, so it is a **column beside** the RASD numbers and
never inside them:

* **vLLM is bf16 weights and bf16 KV, greedy**; RASD is **FP4 weights, NF4 KV, T=0.7**.
  The gap is therefore "production engine at full precision" vs "4-bit research path",
  not two engines at equal precision — and the 4-bit path is what makes RASD's memory
  peak 2.7-8.3x smaller (8.3 GB / 15.8 GB / 25.8 GB against vLLM's ~68.8 GB
  preallocated pool).
* **The comparison is target-only against target-only.** vLLM has no speculative
  decoding in this venv, so the ratio bounds **one direction only**: how much slower
  RASD's target-only path is than a production engine on the same prompt. It says
  nothing about RASD's spec path against a production spec path.
* **Book sets differ.** The vLLM rows cover 1 book at 128k/512k and 2 at 1M while the
  spec means cover 3 and 5, so the `vLLM / spec` column mixes book sets. On the one
  book present in every arm (`pg19_train_0`) the ratios are **24.7x / 6.1x / 2.9x**
  against target-only and **13.2x / 2.8x / 1.4x** against spec.
* **vLLM's TTFT is not available.** Its CSV's `ttft_s` and `decode_only_wall_s` are
  unusable (vLLM 0.6.3 reports absolute monotonic timestamps; the values are ~1.79e9
  and ~-1.79e9). What the run supports is that vLLM's *total* wall is 2.1x / 2.5x /
  2.5x longer than RASD's *prefill-only* `ttft_ms`, so vLLM's prefill is the slower of
  the two here — the opposite of the decode ordering.
* **Convention matches:** RASD's `time_sec` contains its own `ttft_ms` (verified:
  `decode_tps = (tokens-1)/(time_sec - ttft)` reproduces the recorded value), so both
  columns are end-to-end including prefill.

Full ten-point caveat list, the per-row table and the provenance of the vLLM rows
(HEAD `d11fdbc`) are in [VLLM_PLAN.md](./VLLM_PLAN.md) §5b-§6.

### The vLLM comparison with PREFILL AND DECODE SEPARATED (rerun, 2026-10-12)

The retry could only report an end-to-end wall, because vLLM 0.6.3's offline metrics
came back as absolute monotonic stamps. The rerun fixes the recorder and also MEASURES
the prefill, by repeating the same prompts with `max_new_tokens=1` (one book per
context) so `prefill ~= end_to_end(1 token)`. The two agree to **0.03 % at 1M and
0.10 % at 512k** (1.3% at 128k, where the prefill is only 17 s), which is what makes
either of them usable:

| context | vLLM prefill (s) | 1-token probe (s) | **vLLM decode s/token** | **vLLM decode tok/s** | RASD target-only dec tok/s | RASD spec dec tok/s | vLLM / RASD-T | vLLM / RASD-S |
|---|---|---|---|---|---|---|---|---|
| 128k | 16.85 | 17.08 | **0.00814** | **122.90** | 0.28632 | 0.6766 | **429x** | 182x |
| 512k | 222.81 | 222.58 | **0.01449** | **69.02** | 0.10010 | 0.2829 | **690x** | 244x |
| 1M | 859.60 | 859.60 | **0.02337** | **42.80** | 0.05853 | 0.1732 | **731x** | 247x |

End-to-end for the same run: **7.1575 / 0.56979 / 0.14840 tok/s** at 128k / 512k / 1M
(against the retry's 6.9865 / 0.56986 / 0.14828 — reproducible to within 2.4% at 128k
and 0.1% elsewhere).

**THE STRUCTURAL RESULT, and the reason the two tables disagree so violently.** The
same wall is dominated by opposite things in the two engines:

| context | vLLM: prefill share of its wall | RASD target-only: prefill share |
|---|---|---|
| 128k | 94.2% | 1.9% |
| 512k | 99.2% | 6.7% |
| 1M | 99.7% | 13.7% |

vLLM at these contexts is a **prefill benchmark**: 99.7% of its 1M wall is reading the
prompt, and its prefill throughput FALLS with length (7.7k tok/s at 128k, 2.3k at 512k,
1.2k at 1M). RASD is the other way round — 86% of its 1M wall is the decode loop — so
its end-to-end number is mostly a decode number, and the two engines' end-to-end ratio
(24.7x / 6.1x / 2.9x) is a ratio between a prefill-bound wall and a decode-bound one,
which is why it shrinks with context instead of holding. Separated, vLLM's decode is
**429x to 731x** RASD's target-only decode and **182x to 247x** RASD's spec decode.

Reported as a bound with the same caveats as the column above (bf16 vs FP4/NF4, fused
CUDA kernels and TP=8 vs a Python ring loop, no speculative decoding on the vLLM side),
and it says nothing that can be folded into the RASD-vs-RASD speedups.

**What this supports:** throughput falls steeply with context (spec 0.674 → 0.244 → 0.121
tok/s; target-only 0.283 → 0.094 → 0.051), the **speedup is roughly flat across a factor of
8 in context (2.38x / 2.59x / 2.37x)**, and under sampling acceptance **rises** with
context. The mechanism holds up at 1M. It does not "get better" with context — the speedup
is flat, not increasing.

**What it does not support:** an exact like-for-like scaling curve. Two of three books have
offset scored windows; n=3 per rung; one seed; one temperature.

### Decode-only numbers, per row (prefill excluded)

The end-to-end rates above include the prompt prefill, which both arms pay identically and
which at 1M is **339 s against a 1185 s spec wall**. To separate the effect from the cost of
reading the prompt, every row is also reported as **decode seconds per token**:

```
decode_s_per_token = (time_sec - ttft_ms/1000) / (tokens_generated - 1)
decode_only_tok/s  = (tokens_generated - 1) / (time_sec - ttft_ms/1000)
```

The `-1` is the campaign's existing convention (the first token is a prefill product). The
formula reproduces each CSV's own `decode_tps` column to its 2-4 recorded digits — e.g. 1M
target `0.05951` against a recorded `0.06`, 128k target `0.28632` against `0.286` — so this
is a re-expression of the recorded numbers, not a new measurement.

| context | n target / spec | target-only dec s/tok | spec dec s/tok (mean) | **decode-only speedup** | end-to-end speedup (for contrast) |
|---|---|---|---|---|---|
| 128k aligned | 1 / **5** | 3.4925 | **1.3808** | **2.53x** | 2.53x |
| 512k aligned | 1 / **5** | 9.9900 | **3.5227** | **2.84x** | 2.57x |
| 1M | 2 / 5 | 17.0902 | **5.7725** | **2.96x** | 2.37x |

Per row (all 15 spec rows at the three contexts plus the four target-only rows):

| context | doc | target-only s/tok | spec s/tok | spec decode-only speedup vs that book's window |
|---|---|---|---|---|
| 128k | pg19_train_0 | 3.4925 | 1.8320 | 1.91x |
| 128k | pg19_train_1 | — | 1.4573 | 2.40x (vs train_0's window) |
| 128k | pg19_train_915 | — | 1.1443 | 3.05x (vs train_0's window) |
| 128k | pg19_train_2204 | — | 1.0303 | 3.39x (vs train_0's window) |
| 128k | pg19_train_2768 | — | 1.4945 | 2.34x (vs train_0's window) |
| 512k | pg19_train_0 | 9.9900 | 4.1901 | 2.38x |
| 512k | pg19_train_1 | — | 3.7766 | 2.65x (vs train_0's window) |
| 512k | pg19_train_915 | — | 2.6365 | 3.79x (vs train_0's window) |
| 512k | pg19_train_2204 | — | 3.8483 | 2.60x (vs train_0's window) |
| 512k | pg19_train_2768 | — | 4.1182 | 2.43x (vs train_0's window) |
| 1M | pg19_train_0 | 16.8039 | 6.6353 | 2.53x |
| 1M | pg19_train_1 | 17.3765 | 4.5226 | 3.84x |
| 1M | pg19_train_915 | — | 5.2687 | 3.24x (vs train_0's window) |
| 1M | pg19_train_2204 | — | 5.5767 | 3.06x (vs train_0's window) |
| 1M | pg19_train_2768 | — | 6.8591 | 2.49x (vs train_0's window) |

Only one book per rung has a target-only row (two at 1M), so the per-book speedups for the other books
compare their decode against **that rung's single target-only rate**, which is legitimate
because the target-only rate is book-independent (measured variation under 2% across books,
§1) — but it is a comparison against a rate, not against a matched pair, and the table says
so rather than implying otherwise.

**The result this exposes.** The decode-only speedup is **larger than the end-to-end
speedup and rises with context: 2.36x → 2.83x → 2.96x**, while end-to-end is flat
(2.38x / 2.59x / 2.37x). The two are not in conflict: the prefill is identical in absolute
terms for both arms on a given window, but it is a far larger share of the *faster* arm's
wall, so it dilutes the ratio more as context grows (at 1M prefill is 339/1185 s = 29% of
the spec wall against 339/2473 s = 14% of the target-only wall). The mechanism the campaign
is testing acts on the **decoding loop**, and the honest headline for the mechanism is the
decode-only number; the end-to-end number is the honest headline for *this engine's wall
time*. Both are reported, and neither is substituted for the other.

The rope arms, on the same formula (their target-only row is `ROPE_SAMP_NATIVE_TARGET`,
3.6493 s/tok decode = 0.27402 tok/s decode-only against 0.27125 tok/s end-to-end):

| rope arm | dec s/tok (mean) | decode-only speedup | end-to-end speedup (§4) |
|---|---|---|---|
| NATIVE | 1.0441 | **3.50x** | 3.33x |
| F16 | 1.3452 | **2.71x** | 2.67x |
| F32 | 1.3239 | **2.76x** | 2.79x |

These arms run at 128k on `meta-llama/Llama-3.1-8B`, whose prefill is small, so decode-only
and end-to-end nearly agree (unlike the 1M rungs above) — the divergence between the two
columns is a property of the context, not of the arm.

---

## 4. Condition A — the RoPE section, with the gate's verdicts

Three arms at 128k on `meta-llama/Llama-3.1-8B` — which *ships* the `rope_scaling` dict the
intervention modifies. The Gradient-1048k target ships `rope_scaling: None` and the engine
refuses `rope_type: llama3` on it outright, so the intervention cannot be applied to the
headline model at all; this section is about the mechanism, on a model that can take it.
Only `factor` changes between the two candidate arms. The coherence gate ran immediately
before these layers and its verdict is reported, never enforced.

| arm | rope | factor | gate `ppl_ratio` vs native | gate window | matches intent | accept (mean, **n=5**) | tok/s (mean) | speedup vs native target |
|---|---|---|---|---|---|---|---|---|
| ROPE_SAMP_NATIVE_SPEC | none | – | 1.000 | 131072 | True | **0.7272** | 0.91087 | **3.36x** |
| ROPE_SAMP_F16_SPEC | llama3 | 16 | **1.0005** | 131072 | True | **0.5336** | 0.76995 | 2.84x |
| ROPE_SAMP_F32_SPEC | llama3 | 32 | **1.0156** | 131072 | True | **0.5006** | 0.74733 | 2.76x |
| ROPE_SAMP_NATIVE_TARGET | none | – | 1.000 | 131072 | True | — | 0.27125 (n=1) | — |

**The three arms are now n=5**, on the same five books as every other rung
(`ROPE_EXT_*` added books 2204 and 2768, 6 new rows, all `ok`). Per book
(native / f16 / f32): 0.5878 / 0.3170 / 0.3578, 0.7121 / 0.5724 / 0.3650,
0.8276 / 0.6111 / 0.7742, 0.7661 / 0.4556 / 0.5000, 0.7422 / 0.7121 / 0.5060.

**`gate_pass = True` on all three arms, `status = ok`, and no arm was excluded.** The
intervention is nearly free in perplexity at these factors: **+0.05% at factor 16 and
+1.56% at factor 32** on the gate's paired 128k continuation, with `effective_rope_match`
and `matches_intent` true and the native window unchanged at 131072 throughout. The gate's
own repetition block is clean on all three (periodic tail 0.0, `first_periodic_token` 200
= the gate's generation length, i.e. no cycle).

**The acceptance drop IS a rope effect, and the arms are a controlled comparison.**
Native **0.7272** vs f16 **0.5336** and f32 **0.5006** — a **0.19-0.23 absolute fall
(27-31% relative)** that
appears the moment the positional basis is stretched. It is attributable to the
intervention because **nothing else differs between the three arms: the same target
(`meta-llama/Llama-3.1-8B`, same revision), the same draft
(`meta-llama/Llama-3.2-1B`, `draft_window_cap` 4096), the same book, seed, temperature
(0.7), `spec_steps` and token budget; only `rope_factor` changes (none / 16 / 32).** The
mechanism is that speculation acceptance is a function of how peaked the target's
next-token distribution is, and a stretched rotary basis flattens it. **The direction
holds on every book individually**, not only in the mean: native ≥ stretched on all
**five** books (native >= f16 and native >= f32 on 5 of 5, at 128k: 0.5878 vs
0.3170 / 0.3578; 0.7121 vs 0.5724 / 0.3650; 0.8276 vs 0.6111 / 0.7742; 0.7661 vs
0.4556 / 0.5000; 0.7422 vs 0.7121 / 0.5060). Reported as a rope finding, **n=5**, with
the caveats that only the 128k rung was measured and 32 is the largest factor tried.

**One n=3 claim is weakened by the fifth book and is restated rather than repeated.** At
n=3 the two stretched factors landed within **0.0012** of each other and that was offered
as what a saturating effect looks like. At n=5 they are 0.5336 and 0.5006 — a 0.033 gap
(6.6% relative), which is still much smaller than either's distance from native (0.19 and
0.23) but is no longer "within noise of each other". The saturation reading survives as
"both stretched factors land far closer to each other than to native"; the specific
0.0012 is withdrawn. `pg19_train_2768` under f16 is also the one row the repetition pass
flags (77% of tokens inside a repeated 16-gram, no periodic tail).

---

## 5. Why the engine is slow, measured (Phase 3)

`BANDWIDTH_FLOOR.md` derived a per-token traffic floor of **59 / 232 / 461 ms** at
128k / 512k / 1M for target-only decode, against measured **3.51 / 9.98 / 19.10 s per
token** — a **42-59x gap** — and showed the measured time is nearly *linear in the number
of ring chunk transfers* (0.58-0.86 ms each across an 8x range of context, ~3.6 GB/s on a
link capable of ~300 GB/s per direction).

**Profile (a) then measured it on one GPU** (`gpu_1x_a100_sxm4`, `--nproc 1`, so the ring
is `world_size = 1` and there is no cross-rank transfer of any kind), target-only decode at
128k, 24 tokens, 60-minute dead-man timer, 0 instances after terminate, **$1.28**:

| what | value |
|---|---|
| measured row | 24 tokens / 76.2503 s = 0.31475 tok/s = **3177 ms/token** (38.12 s of it prefill) |
| decode steps only | 38.13 s / 23 = **1.658 s/token** |
| `comm_us` | **0.0 — exactly zero, in both profiled rows** |
| attributable to recorded ops | 66.34 s = **87.0%** of the measured row: compute 53.96 s (70.8%) + other 12.38 s (16.2%), communication **0.00 s** |
| unrecorded gaps | 9.91 s = **13.0%** |
| profiler overhead | the profiler's own wall is 125.40 s vs the measured 76.25 s — **the profile inflates the run by 1.64x**, so shares of the profiled wall (compute 43.0 / comm 0.0 / idle 47.1 / other 9.9) must not be quoted as shares of the measured time |

**The finding is that the gap is not a ring-communication phenomenon.** With the ring
removed entirely the engine is still **28x above its own bandwidth floor** (1.658 s/token
against 59 ms) with **zero** communication recorded, and 87% of the row is inside recorded
local ops rather than unmeasured gaps. Adding the sampling session's own 8-rank target-only
row at 128k (235.0941 s, ttft 9.4253 s → 3.526 s/token) gives the ring's multiplier:
**~2.1x**. Real, and consistent with the floor analysis — but the residual **local** cost is
of the same order as the whole 8-rank per-token time, so a fix aimed only at the ring would
leave most of the gap in place.

**The honest limit on that comparison:** the one-device run holds the whole 128k KV on a
single rank, so its per-rank local work is larger than an 8-rank rank's. It is **not** a
like-for-like per-rank baseline, and 1.658 s must not be subtracted from 3.526 s to give
"the ring's share". What it proves is the weaker, sufficient statement above. Splitting the
local cost into dequantization vs Python vs kernel launches needs engine changes, which
this plan forbids because they would invalidate every timing quoted here.

### 5a. Profile (b), finally measured at 8 ranks (SA_PROF_RING_128K, 2026-10-12)

Profile (a) could only say that the ring was *not* needed to be slow: at one rank `comm_us`
was 0.0 exactly. It could not say what the ring costs, because the ring is a no-op at
`world_size <= 1`. The seeds/ablations session opens with the row that closes that gap —
`configs/mlsys_profile_128k.yml` field for field, run at the rank count it was written for
(nproc 8), target-only, 128k, a 8-token warm row then a 24-token measured row, `profile:
true` on both.

| what | measured row (24 tokens, nproc 8) |
|---|---|
| the row's OWN `time_sec` | 96.059 s (ttft 8.805 s) |
| the profiler's window (`wall_us`) | **137.487 s** |
| recorded ops (compute + comm + other) | 85.866 s |
| compute | 75.461 s = **87.9%** of recorded ops |
| **communication** | **1.512 s = 1.8% of recorded ops** (1.1% of the profiler window) |
| other | 8.894 s = 10.4% |
| no recorded op (idle + the profiler's own stop/flush) | 51.621 s = 37.5% of the window |

**The profiler does capture the ring.** `comm_us` is **1.512 s, not 0.0** — the same bucket
that was exactly zero at one rank — and the ring's primitives (`batch_isend_irecv`, `nccl`,
`c10d`) are exactly what `src/analysis/profiler.py` bins as communication. So the bucket the
withdrawn 1-GPU profile could not fill is filled, and it is **small**: at nproc 8 with the
ring actually running, communication is under 2% of the recorded time while compute is 88%.
That is the 8-rank version of profile (a)'s conclusion, and it is the last word on C8.

**Two cautions, both measured.** (1) `RoundProfiler` wraps the whole `generate_text()`
call, so its window is a **superset** of the row's own timer: 137.487 s of window against
96.059 s of `time_sec`, the ~41 s difference being the profiler's stop/export, which
happens after `t_end`. The buckets are inside the same call as the row, and the row's timer
is inside the window — but the window is not the row. (2) The JSON's `*_pct` fields are
shares of the **profiler window**, not of the recorded total, which is why "compute 54.9%"
and "compute 87.9% of recorded ops" are both in the JSON's arithmetic and only one of them
means what it sounds like. The table above gives both, labelled.

The row's tok/s is **not** a throughput measurement (24 tokens, and profiled), and is not
quoted as one anywhere in this document.

**Profile (b) — the 2-rank run — was SKIPPED, as a stated gap rather than a silent one.**
The plan says to run it only if (a) does not explain the time; (a) attributes 0% to
communication, so there is no remainder to attribute. Independently, there is no 2-GPU A100
SXM instance to run it on: the API has no `gpu_2x_a100_sxm4` type at all, and the only
2-GPU A100 (`gpu_2x_a100`, 40 GB **PCIe**) has **zero capacity in every region** (checked
08:26Z, 09:11Z and 09:15Z); the plan rules a PCIe pair out as "worse than none". The SXM NVLink
alternatives are 8-GPU boxes at $15.92 and $22.32/hr, both over (b)'s $10 cap. Details:
`results/mlsys/PROFILE.md`.

---

## 6. Condition A — quality redo: does the native-1M model hold up at 1M?

Greedy, `prompt_gen_tokens` 1024, so **`ppl_tokens` = 1024** (this replaces the superseded
8-token quality rows). Perplexity is scored on the last 1024-token window at each context,
and the alignment between arms is verified from `continuation_sha256` plus the recorded
offsets (§3): the scored block is `doc[prompt_source_len : prompt_source_len+1024]`, so the
cross-arm offset is the drift difference.

| book | ppl @128k | ppl @1M | change | alignment | overlap of scored tokens |
|---|---|---|---|---|---|
| pg19_train_0 | **1.648977** | **1.805428** | **+9.5%** | `continuation_sha256` identical | **100% (exact)** |
| pg19_train_1 | 12.845739 | 12.708859 | −1.1% | offsets 9 vs 101 | **91.0%** |
| pg19_train_915 | 10.582580 | 10.628461 | +0.4% | offsets 17 vs 65 | **95.3%** |

**Framing: the native-1M model stays coherent at 1M — within ~10% of its own 128k
perplexity on the same tokens.** On the only book whose scored tokens are *identical*
between the two arms, 1M perplexity is 9.5% higher; on the two books whose windows share
91-95% of their scored tokens, the difference is under 1.1%. That is the claim. **It is
not** "the model benefits from long context": there is no book on which 1M is materially
better, the one exactly-aligned comparison is worse, and it is one book, one seed, greedy,
with the 128k arm at 1/8 the context.

---

## 7. Hardware, spend, and what is left

| item | value |
|---|---|
| Condition A session | instance `cad0d8cdaf2e4994ae7a70322ef1efe3`, `gpu_8x_a100_80gb_sxm4`, us-east-1, $22.32/hr, 2026-10-12T00:11:06Z → 06:39:41Z = **6.82 h = $152.24** |
| Condition A timer | 480-minute detached dead-man timer, PPID 1, cancelled after 0 instances were confirmed via the API |
| Phase 3 profile | `gpu_1x_a100_sxm4` (1x A100 40 GB SXM4), $1.99/hr, 0.65 h = **$1.28**, 60-minute timer, 0 instances confirmed |
| Phase 5 vLLM rehearsals | 2 x `gpu_1x_a100_sxm4` (40 GB SXM4), $1.99/hr, 0.18 h + 0.31 h = **$0.96** total; first attempt found two install blockers, second passed; both times 0 instances confirmed and the timer cancelled |
| Condition B session | `gpu_8x_a100_80gb_sxm4`, us-east-1, $22.32/hr, 6.08 h = **$135.62**, 0 instances, timer cancelled |
| **cumulative campaign spend** | **~$329 of the $450 cap** |

Condition A ran every planned stage — sanity, the 128k and 1M quality redo, the 1M sampling
check, the 1M headline and its target-only partner, both aligned rungs, the rope gate and
all four rope arms — and **nothing was skipped**. The single failure in the session was the
data defect on `pg19_train_1726` of §1. The session was judged from the stage log lines and
the stage CSVs, not from the completion marker's return code: a marker of 0 is not evidence
that the stages ran. Full report: `results/mlsys/SESSION_A_JUDGMENT.txt`.

Evidence tree for condition A: `results/mlsys/sampling_session/` — every stage CSV, slim
token sidecars, per-round traces, and one repetition report per stage.

## 8. The seeds and ablations session (2026-10-12)

One 8x A100 80GB SXM4 instance (`6d8cd2c13c364cddab1de3009e0f40c7`, us-east-1, detached
dead-man timer confirmed by PPID 1, 480-minute deadline, $200 ceiling), **one setting changed
per stage** from the headline config, new stage ids and CSVs (`SA_*`) so nothing here can be
confused with or overwrite the sampling session's rows, stages ordered by priority so a run
cut short loses the LAST stage and never the first.

**The rows are committed**, in the shape the campaign already uses for the October
sampling sessions: `results/mlsys/seeds_ablations/` holds the 12 stage CSVs, a
`repetition_<STAGE>.csv` for every stage (regenerated from the committed tree, so the
degeneracy flags in this section are reproducible from the repository rather than only
from the pod), and `tokens_slim/` with the 48 token sidecars. The per-round traces and the
full sidecars stay in the session pull directory, as they do for the other sessions.

**All 12 stages ran and every stage is ok: 48/48 ok rows, 0 bad.** In order, with rows:
sanity pair 2/2; `SA_PROF_RING_128K` 2/2; `SA_RINGCHUNK_8192_TARGET` 2/2;
`SA_RINGCHUNK_8192_SPEC` 2/2; `SA_SEED_123_1M_SPEC` 5/5 and `SA_SEED_456_1M_SPEC` 5/5 (both
void, §8.3); `SA_DRAFTWIN_2048_128K_SPEC` 5/5; `SA_DRAFTWIN_16384_128K_SPEC` 5/5;
`SA_TEMP_0_128K_SPEC` 5/5; `SA_TEMP_1_0_128K_SPEC` 5/5; `SA_STEPS_2_128K_SPEC` 5/5;
`SA_STEPS_6_128K_SPEC` 5/5. Session wall **5.95 h = $132.70** of the $200 ceiling; the
instance was terminated, the API read **0 instances**, and the timer was cancelled, all
before any of these numbers were read. Two of the twelve stages are consumable only as a
determinism result rather than as the experiment they were designed for (§8.3), so the
priority ordering did its job: the stages that were meant to be droppable were the ones whose
result was already known.

Rehearsed on one 1x A100 first (11 stages, 24/24 ok rows, $1.00) —
and that rehearsal earned its $1.00 twice: the first attempt used 4-token rows, every
acceptance came out 0.0/4, and the session's own gate stopped it at stage 3, so eight of the
eleven stages went unexercised.

### 8.1 Stage b — the profiled ring row

See §5a: at nproc 8 the communication bucket is **1.512 s, not the 0.0 the 1-GPU profile
could report**, compute is 87.9% of recorded ops, and the profiler's window is a superset of
the row's own timer (137.487 s against 96.059 s). The row's tok/s is not a throughput
measurement and is not quoted as one.

### 8.2 Stage c — ring chunk size, `kv_block_size` 8192 against 2048, at 128k

The knob moves the size of each ring transmission (2048 → 8192 tokens), which the floor
analysis says the per-token time is nearly linear in. Same two books in both arms.

| arm | book | `kv_block_size` | accept | e2e tok/s | decode s/token | peak MB |
|---|---|---|---|---|---|---|
| target-only | pg19_train_0 | 2048 (headline session) | — | 0.28310 | 3.4925 | 8256.4 |
| target-only | pg19_train_0 | **8192** | — | **0.30375** | **3.2524** | **8256.4** |
| spec | pg19_train_0 | 2048 (headline session) | 0.2712 | 0.53084 | 1.8320 | — |
| spec | pg19_train_0 | **8192** | **0.2712** | **0.58621** | **1.6461** | 10113.6 |
| spec | pg19_train_1 | 2048 (headline session) | 0.4255 | 0.65789 | 1.4573 | — |
| spec | pg19_train_1 | **8192** | **0.4255** | **0.74480** | **1.2853** | 10112.8 |

**Fewer, larger transfers are faster, and the change is numerically transparent.** Target-only
**+7.3%** end-to-end (**-6.9%** per decode token); spec **+10.4%** and **+13.2%**
(**-10.1%** and **-11.8%** per decode token). The acceptance of each spec row is
**unchanged to four decimal places** (0.2712, 0.4255), which is what a purely scheduling
change should look like — the same arithmetic, the same sampled tokens, less overhead between
them. Peak memory is **identical** on the target-only arm (8256.4 MB) and 10.1 GB on the spec
arm, against 80 GB: the memory question the plan said to answer first is answered by
measurement, not by an estimate.

**The caveat on this one is a session boundary, not a measurement.** The 2048 arm is the
Oct 12 sampling session's rows; this session did not re-measure 2048, so part of a 7-13%
difference could be host or session effect. The identical acceptance is evidence that the two
arms compute the same thing, and the direction is the same in all three pairs, but a
2048-vs-8192 A/B inside one session is the experiment that would close it.

### 8.3 Stage d — the seed stages are VOID as seed variation, and that is itself a result

`SA_SEED_123_1M_SPEC` completed **5/5 ok rows** and is **not a seed replicate**. Every row
records `seed=42`, and every row's `generated_tokens_sha256` is **byte-identical** to its
seed-42 counterpart; the acceptance mean is 0.5246 with the same per-book values to four
decimals (0.3776 / 0.6838 / 0.6250 / 0.5437 / 0.3929).

**Cause, from the code rather than from the output:** `run_experiment.py` pops the seed list
from `defaults` (590), merges each level over the defaults (611) and then sets
`run["seed"] = seed` from that list (612) -- so a level's own `seeds:` key is overwritten on
the next line, and the comment there says the rule was written because "greedy decoding makes
a seed change a no-op". That is true for greedy and false for this session's T=0.7 arms. The
config-only fix is `defaults.seeds`, one config per seed
(`configs/mlsys_seed{123,456}_1m.yml`), not yet run: two seeds x 5 books x 1M is ~3.1 h of 8x
time (~$69, needing approval). `SA_SEED_456` in the same session re-runs seed 42 the same way,
for the same ~$36 as `SA_SEED_123` spent.

**What it does support, and nothing else was needed to get it:** the engine is
**bit-reproducible for a fixed seed across sessions and hosts**. The seed-42 rows were
produced on 2026-10-12 in one 8x instance (`cad0d8cd…`, us-east-1); the seed-123 stage's rows
were produced hours later on a different 8x instance (`d6cc…`, same region) and generated the
identical token streams. Two consequences worth stating: every other delta in this section can
be read as an effect of the setting under test rather than of the machine, and a "3 seeds x 5
books" spread cannot be obtained by re-labelling a run -- it needs the seed to reach the
sampler, which is exactly what the void stage demonstrated it did not.

**The seeds/ablations session's other stages are unaffected, and that was checked rather than
assumed.** Only `seeds` is popped and re-applied, so a level's `temperature`, `spec_steps` and
`kv_block_size` survive the merge, and the rows prove it: `SA_RINGCHUNK_8192_*` records
**`kv_block_size = 8192`** (headline 2048) with `spec_steps` 0 and 4 on its two arms;
`QUALITY_FIX_128K` records `temperature 0.0` where its level set it and the sampling rows
record 0.7. `draft_window_cap` has the same code path but **no row-level precedent anywhere in
the committed results** (every row that carries the column says 4096), so
`SA_DRAFTWIN_2048/16384` are the first rows that will test it; if it silently fell back, those
two stages would be ~20-minute re-runs of 4096 each (~$7 apiece).

### 8.4 Stage e — draft window (`draft_window_cap`), 128k, 5 books

The draft is a 1B model attending over a window capped at this many tokens. The headline is
4096; this moves it to 2048 and 16384, changing nothing else. **2048 and 16384 are an
in-session pair**; 4096 is the cross-session reference and is flagged as such.

| `draft_window_cap` | n | accept (mean) | e2e tok/s | decode s/token | per-book acceptance |
|---|---|---|---|---|---|
| 2048 | 5 | 0.4981 | 0.72666 | 1.3710 | 0.2421 / 0.5641 / 0.6765 / 0.5250 / 0.4826 |
| **4096 (headline, other session)** | 5 | 0.4681 | 0.71557 | 1.3808 | §3 |
| **16384** | 5 | **0.5598** | **0.79637** | **1.2086** | 0.4115 / 0.5641 / 0.6838 / 0.4826 / 0.6571 |

**A larger draft window helps, and the effect is real but modest and NOT monotone.** Against
2048 in-session, 16384 buys **+0.062 acceptance** (+12.4% relative), **+9.6% end-to-end
throughput** and **−11.8% per decode token**, winning on 3 of the 5 books, tying on one
(`pg19_train_1`, 0.5641 in both) and losing on one (`pg19_train_2204`, 0.5250 → 0.4826). The
4096 reference sits *below both*, which is the interesting part: if the effect were monotone
in the cap, 4096 would lie between them. It does not, and 4096 is from a different session, so
the defensible claim is the in-session one — **16384 > 2048** — plus the observation that
**4096 is not obviously better than either**, i.e. the cap is not the free win one would reach
for first. Per-level `draft_window_cap` is also recorded in the rows themselves (2048 and
16384 in the column), which is the first row-level evidence anywhere in these results that the
key reaches the engine rather than falling back to the headline 4096.

### 8.5 Stage f — temperature, 128k, 5 books, periodicity per row

Greedy (T=0.0) and T=1.0 against the headline T=0.7. **0.0 and 1.0 are an in-session pair**;
0.7 is the cross-session reference. Because a looping row's acceptance is not a measurement of
anything, each arm is reported twice: over every ok row, and over the clean rows only, with the
degenerate row named.

| temperature | n | degenerate rows | accept (all ok rows) | accept (clean only) | e2e tok/s | decode s/token |
|---|---|---|---|---|---|---|
| **0.0 (greedy)** | 5 | **1** — `pg19_train_915` | 0.4604 | **0.3818** (n=4) | 0.69717 | 1.4469 |
| 0.7 (headline, other session) | 5 | 0 | 0.4681 | 0.4681 | 0.71557 | 1.3808 |
| **1.0** | 5 | **0** | **0.7369** | 0.7369 | **0.97418** | **0.9660** |

Per-book acceptance (reported / non-degenerate): T=0 `0.2621/0.2621`, `0.4255/0.4255`,
**`0.7750` degenerate**, `0.5000/0.5116`, `0.3396/0.3519`; T=1.0 `0.6429/0.6528`,
`0.7045/0.7132`, `0.8190/0.8250`, `0.7917/0.7984`, `0.7266/0.7348`.

**Correction to the first version of this table.** It reported the degenerate row as
`pg19_train_1` and the clean-only mean as 0.4692, and both were wrong: the detector flags
`pg19_train_915` (69% of tokens inside a repeated 16-gram > 60% absolute; periodic tail 69% of
the row, period 15, > 50% absolute), while `pg19_train_1` is clean -- it merely carries a
42% periodic tail, below the ceiling. The mistaken row made greedy look level with the
headline (0.4692 vs 0.4681). **Dropping the right row, greedy is 0.086 behind (0.3818 vs
0.4681, -18% relative), and the looping row is what was holding its mean up** -- the same
inflation mechanism the 1M headline was withdrawn for, reappearing at 128k in 1 row of 5
where it moves the mean by 0.079. The T=1.0 arm is unaffected: 0/5 degenerate.

**Temperature is the largest single knob measured in this session, and it points UP.** T=1.0
against the headline T=0.7: **+0.269 acceptance** (+57% relative) and **+36% end-to-end
throughput** (0.97418 vs 0.71557), with **-30% per decode token** (0.9660 vs 1.3808 s) -- and
it happens on all five books individually: 0.6429 vs 0.2712, 0.7045 vs 0.4255, 0.8190 vs
0.6081, 0.7917 vs 0.5855, 0.7266 vs 0.4500. Greedy buys nothing at 128k once its looping row
is set aside.

**The greedy result is context-dependent, and it is worth stating against the earlier
headline.** At 1M the greedy condition looped on 9 of 15 rows, which is why the campaign moved
to sampling. At 128k only **1 of 5** rows loops. So "greedy degenerates" is a statement about
long context in this engine, not a universal one -- and it is a statement about the *mean*,
not only about the looping row: the row that loops is the row that inflates.

### 8.6 Stage g — draft length (`spec_steps`), 128k, 5 books, periodicity per row

How many tokens the draft proposes per verify step. **2 and 6 are an in-session pair** on the
same five books; **4 is the headline** and comes from the Oct 12 sampling session. The
target-only reference is that session's 128k book-0 row (0.28310 tok/s, 3.4925 s/token), the
same reference §8.2 uses.

| `spec_steps` | n | degenerate | accept (all / non-degenerate) | e2e tok/s | decode s/token | speedup vs 128k target-only | decode speedup |
|---|---|---|---|---|---|---|---|
| **2** | 5 | 0/5 | 0.6558 / 0.6606 | 0.58904 | 1.6382 | 2.08x | 2.13x |
| 4 (headline, other session) | 5 | 0/5 | 0.4681 / 0.4681 | 0.71557 | 1.3808 | 2.53x | 2.53x |
| **6** | 5 | 0/5 | 0.4270 / 0.4392 | **0.83685** | **1.1621** | **2.96x** | **3.01x** |

Per-book (accept reported / non-degenerate): steps=2 `0.6792/0.6852`, `0.5678/0.5750`,
`0.8021/0.8061`, `0.6140/0.6140`, `0.6161/0.6228`; steps=6 `0.3106/0.3259`, `0.3625/0.3625`,
`0.6282/0.6420`, `0.4510/0.4667`, `0.3829/0.3991`.

**Acceptance falls with draft length while throughput rises, and both monotone.** 0.6558 →
0.4681 → 0.4270 acceptance, 0.58904 → 0.71557 → **0.83685** tok/s. Accepting fewer tokens per
proposal is worth it here because a verify step costs one target forward over a KV cache that
is already paid for: the marginal cost of checking 5 more candidate tokens against 128k of
context is small, so a longer draft amortizes the same expensive step over more emitted
tokens. Decode time per token falls **15.8%** from 4 steps to 6 (1.1621 vs 1.3808 s) and
**18.6%** the other way to 2 steps (1.6382 s) — the ordering is the same in all five books
individually for both comparisons.

**steps=6 is the fastest arm measured at 128k in the whole campaign: 2.96x end-to-end and
3.01x decode-only over target-only**, against the headline's 2.53x. Nothing here says 6 is
optimal -- 6 is simply the largest value tested, and the trend is still rising at the edge of
the tested range.

**The interaction to test, not to assume.** Temperature (§8.5) raises acceptance by 0.269 and
this stage lowers it by 0.042 per added step, so the throughput-optimal step count almost
certainly moves with temperature: at T=1.0 the draft is rewarded far more often per proposal,
which is the regime where longer drafts pay most. Any final configuration should fix one
before reading the other, and neither should be carried into a 1M headline without being
re-measured at 1M, where the per-step cost is much larger.

### Limitations

1. **5 books at 1M** (6 configured; one excluded for a data defect), 3 books per aligned
   rung, and 1-2 target-only books per rung because target-only rows are subsampled.
   Nothing here is a distribution over documents.
2. **One seed (42), one temperature (0.7)** — greedy only where greedy is stated. That
   sampling removes the greedy loop is measured at one temperature.
3. **The speedup is against our own target-only path**, which is itself very slow (0.05103
   tok/s at 1M). It is not a speedup against a production engine, and **no vLLM baseline
   exists** for this model or these prompts. "2.37x" says the speculation mechanism works;
   it does not say the engine is competitive.
4. **Sampling losslessness is distributional**, not bit-exact: at T>0 the engine uses the
   standard `min(1, p_target/p_draft)` rejection criterion with a residual resample, tested
   at T=1.0 and exercised end-to-end by earlier campaigns, but a sampled row cannot be
   compared to target-only by token identity the way a greedy row can. Throughput is
   unaffected; the *output* comparison is statistical.
5. **Two of three books at the aligned rungs do not score identical tokens** (91-100%
   overlap, §3), so cross-context acceptance is close to, but not exactly, like-for-like.
6. **The profiling covers 128k only** — one row per configuration, one book, 24 measured
   tokens, warm-up at row granularity rather than step granularity, and the profiler
   inflating the run by 1.64x. The ring's 2.1x is a 1-rank/8-rank comparison across two
   config files, not a 2-rank measurement.
7. **`pg19_train_1726` cannot serve the 1M rung** at a 1024-token scored continuation, and
   is excluded from §1 rather than silently missing.

### Figure

`results/mlsys/scaling_sampling.png`, regenerated **from the sampling session's CSVs
only**, with the repetition verdict for every row computed from its token sidecar and
looping rows marked and excluded from the means (none were). Command:

```bash
python scripts/mlsys_plot_scaling.py \
  results/mlsys/sampling_session/session_a_SAMP_1M_SPEC.csv \
  results/mlsys/sampling_session/session_a_SAMP_1M_TARGET.csv \
  results/mlsys/sampling_session/session_a_SAMP_ALIGNED_512K_SPEC.csv \
  results/mlsys/sampling_session/session_a_SAMP_ALIGNED_512K_TARGET.csv \
  results/mlsys/sampling_session/session_a_SAMP_ALIGNED_128K_SPEC.csv \
  results/mlsys/sampling_session/session_a_SAMP_ALIGNED_128K_TARGET.csv \
  --tokens-dir results/mlsys/sampling_session/tokens_slim \
  --out results/mlsys/scaling_sampling.png \
  --title "RASD (T=0.7 sampling): acceptance and throughput vs context"
```

It reports 11 spec rows and 4 target-only rows plotted, 0 looping. Two defects in that
script were fixed for this: it had been drawing the rounded `throughput_tps` field, and
with no token sidecar it treated a row as clean without testing it. It now computes
`tokens_generated/time_sec` and excludes unmeasurable rows loudly.

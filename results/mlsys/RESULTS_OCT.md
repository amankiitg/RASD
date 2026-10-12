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

| context | arm | n | accept (mean) | tok/s (mean) | per-book tok/s | target-only | speedup |
|---|---|---|---|---|---|---|---|
| 128k aligned | spec | 3 | 0.4349 | **0.67350** | 0.53084 / 0.65789 / 0.83175 | **0.28310** (n=1) | **2.38x** |
| 512k aligned | spec | 3 | 0.4892 | **0.24389** | 0.20607 / 0.22362 / 0.30197 | **0.09411** (n=1) | **2.59x** |
| 1M | spec | 5 | 0.5246 | 0.12077 | §1 | 0.05103 (n=2) | **2.37x** |

Per-row acceptance at 128k: 0.2712 / 0.4255 / 0.6081 (train_0 / 1 / 915); at 512k:
0.3349 / 0.4062 / 0.7266. All 11 spec rows at the three contexts are non-degenerate.

**What this supports:** throughput falls steeply with context (spec 0.674 → 0.244 → 0.121
tok/s; target-only 0.283 → 0.094 → 0.051), the **speedup is roughly flat across a factor of
8 in context (2.38x / 2.59x / 2.37x)**, and under sampling acceptance **rises** with
context. The mechanism holds up at 1M. It does not "get better" with context — the speedup
is flat, not increasing.

**What it does not support:** an exact like-for-like scaling curve. Two of three books have
offset scored windows; n=3 per rung; one seed; one temperature.

---

## 4. Condition A — the RoPE section, with the gate's verdicts

Three arms at 128k on `meta-llama/Llama-3.1-8B` — which *ships* the `rope_scaling` dict the
intervention modifies. The Gradient-1048k target ships `rope_scaling: None` and the engine
refuses `rope_type: llama3` on it outright, so the intervention cannot be applied to the
headline model at all; this section is about the mechanism, on a model that can take it.
Only `factor` changes between the two candidate arms. The coherence gate ran immediately
before these layers and its verdict is reported, never enforced.

| arm | rope | factor | gate `ppl_ratio` vs native | gate window | matches intent | accept (mean) | tok/s (mean) | speedup vs native target |
|---|---|---|---|---|---|---|---|---|
| ROPE_SAMP_NATIVE_SPEC | none | – | 1.000 | 131072 | True | 0.7092 | 0.90239 | **3.33x** |
| ROPE_SAMP_F16_SPEC | llama3 | 16 | **1.0005** | 131072 | True | 0.5002 | 0.72339 | 2.67x |
| ROPE_SAMP_F32_SPEC | llama3 | 32 | **1.0156** | 131072 | True | 0.4990 | 0.75686 | 2.79x |
| ROPE_SAMP_NATIVE_TARGET | none | – | 1.000 | 131072 | True | — | 0.27125 (n=1) | — |

**`gate_pass = True` on all three arms, `status = ok`, and no arm was excluded.** The
intervention is nearly free in perplexity at these factors: **+0.05% at factor 16 and
+1.56% at factor 32** on the gate's paired 128k continuation, with `effective_rope_match`
and `matches_intent` true and the native window unchanged at 131072 throughout. The gate's
own repetition block is clean on all three (periodic tail 0.0, `first_periodic_token` 200
= the gate's generation length, i.e. no cycle).

The acceptance difference (native 0.7092 vs f16 0.5002 / f32 0.4990) is **not** a rope
finding: all four arms share the same target model, but the stretched positional basis
changes how peaked the target's next-token distribution is, and speculation acceptance
depends on exactly that. It is reported as an observation, with n=3.

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
| Condition B session | `gpu_8x_a100_80gb_sxm4`, us-east-1, $22.32/hr, 6.08 h = **$135.62**, 0 instances, timer cancelled |
| **cumulative campaign spend** | **~$328 of the $450 cap** |

Condition A ran every planned stage — sanity, the 128k and 1M quality redo, the 1M sampling
check, the 1M headline and its target-only partner, both aligned rungs, the rope gate and
all four rope arms — and **nothing was skipped**. The single failure in the session was the
data defect on `pg19_train_1726` of §1. The session was judged from the stage log lines and
the stage CSVs, not from the completion marker's return code: a marker of 0 is not evidence
that the stages ran. Full report: `results/mlsys/SESSION_A_JUDGMENT.txt`.

Evidence tree for condition A: `results/mlsys/sampling_session/` — every stage CSV, slim
token sidecars, per-round traces, and one repetition report per stage.

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

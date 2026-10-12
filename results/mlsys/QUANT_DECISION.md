# QUANT DECISION — Phase 2 of the Oct 12-13 autonomous run

This file holds the precision decision that drives the rest of the run: which of the four
quantization configurations the bf16 campaign is run in, the numbers the decision was made
from, and the branch taken. It is written **before** anything is launched on the strength of
it, so the decision cannot be reverse-fitted to whatever the campaign later produced.

Sections fill in as the phases land. Phase 0 below is complete.

---

## Phase 0 — rehearsal, PASSED (1x A100 40GB SXM4, `dc8a037d00a14b1eae799d872290db86`)

**Verdict: PASS on every criterion.** Judged from the four stage log lines and the eight CSV
rows, not from the DONE marker.

| criterion | result |
|---|---|
| all four configs produce rows | **4/4 stages, 2 rows each, 8 rows, all `status=ok`, no `error` text** |
| every row records the flags its config set (R2) | **yes — see the table below, read from each row's sidecar** |
| no errors | yes: 0 bad rows, 0 non-empty `error` cells |
| a non-default setting exercised the forwarding path | **yes, twice**: `quantize_target: false` and `kv_quant: false` both reached the engine and are visible in the loaded model |

### The R2 check, from the rows' own sidecars

`weight_precision` and `kv_dtype` are **measured off the loaded model and cache**
(`src/models/rasd_inference.py:numerics_report` → `detect_weight_precision`,
`detect_kv_precision`), not echoed from the config, so they detect a silent fallback in either
direction. Each arm's recorded pair:

| stage | config asked for | `weight_precision` recorded | `kv_dtype` recorded | verdict |
|---|---|---|---|---|
| SAQ_CONTROL | both on (headline) | `fp4` | `nf4` | matches |
| SAQ_W_OFF | `quantize_target: false` | **`bfloat16`** | `nf4` | matches: only the weights moved |
| SAQ_KV_OFF | `kv_quant: false` | `fp4` | **`bfloat16`** | matches: only the KV moved |
| SAQ_BOTH_OFF | both false | **`bfloat16`** | **`bfloat16`** | matches |

The sanity pair in the same session records `fp4`/`nf4`, i.e. the rehearsal's own denominator
is the headline precision, as intended.

**HOW THIS IS CHECKED AT 8x, SINCE THE CSV HAS NO quantize_* COLUMN.** The CSV carries
`dtype` and `draft_dtype` (both `bfloat16`, the compute dtype, in every arm — so they cannot
distinguish the arms); the row's **token sidecar** carries the measured `weight_precision`
and `kv_dtype`. Phase 1 and Phase 3 therefore check those two columns per row against the
table above, and any row that disagrees with its arm stops the run (S2).

### What this rehearsal does and does not prove

It proves the two per-level precision keys survive config → driver → pod → engine on this
card, that all four stage ids pass the stage-naming gate, and that a T>0 arm produces
non-degenerate acceptance (0.14-0.20) rather than the 0.0/4 the seeds rehearsal's first
attempt produced. It proves **nothing about timing**: 8192 context is ~16x cheaper per step
than the 128k Phase 1 measures, the card is one 40GB A100, and at one rank the ring is a
no-op.

**A hint, recorded as a hint and not as a result.** Even at 8192/48 tokens on this card, the
target-only arms separated in the direction the physics predicts: CONTROL 7.88 s, W_OFF
6.75 s, KV_OFF 3.57 s, BOTH_OFF 2.68 s (same book, same protocol, one setting apart). That is
a 2.9x spread at a context 16x too small to be decisive, on hardware Phase 1 does not use. It
predicts the Phase 1 ordering and is quoted only as a prediction.

**Cost**: 1053 s = 0.29 h at $1.99/hr = **$0.58** of the $3 rehearsal cap (the earlier
attempt that never got a card cost $0.00, and the misconfigured `nproc 8` launch cost $0.00 —
its VM never booted, confirmed by the terminate response's `vm-has-not-launched`).

---

## Phase 1 — the ablation, measured (8x A100 80GB SXM4 `0a110718553b497cb8c799ffa3289252`, us-midwest-1)

128k, `pg19_train_0`, T=0.7, 128 tokens, spec_steps 4, draft_window_cap 4096, kv_block_size
2048, seed 42. All four arms in one session, so every difference is the setting and not the
host; each arm has its own target-only and spec row. **8/8 rows `ok`, 0 bad, 0 errors**, plus
the sanity pair 2/2. R2 verified per row from the sidecars and reported below.

| arm | target-only decode s/tok | speedup vs CONTROL | spec decode s/tok | spec speedup | spec acceptance | peak MB (rank 0, allocated) |
|---|---|---|---|---|---|---|
| SAQ_CONTROL (fp4/nf4) | 3.6684 | 1.000x | 1.9086 | 1.000x | 0.2712 | 8256 / 10114 |
| SAQ_W_OFF (bf16/nf4) | 3.6079 | **1.017x** | 1.7500 | 1.091x | 0.3170 | 17824 / 19681 |
| SAQ_KV_OFF (fp4/bf16) | 3.3262 | **1.103x** | 1.3438 | **1.420x** | 0.4096 | 13159 / 17633 |
| SAQ_BOTH_OFF (bf16/bf16) | 3.2959 | **1.113x** | 1.8530 | 1.030x | 0.2422 | 22727 / 27207 |

R2, from each row's sidecar (`weight_precision` / `kv_dtype`, read off the loaded model and
cache): CONTROL `fp4`/`nf4`; W_OFF `bfloat16`/`nf4`; KV_OFF `fp4`/`bfloat16`; BOTH_OFF
`bfloat16`/`bfloat16`. **Every arm recorded exactly what it asked for — no silent fallback,
nothing to stop for.**

### The decision, by the stated rule

1. Primary metric = target-only decode s/tok. Speedups: W_OFF 1.017x, KV_OFF 1.103x,
   BOTH_OFF **1.113x**. **BEST = BOTH_OFF** (3.2959 s/tok).
2. "If BOTH_OFF is within 10% of BEST, choose BOTH_OFF": BOTH_OFF *is* BEST, so the chosen
   config is **BOTH_OFF — bf16 weights, bf16 KV**.
3. Memory gate, projected peak per rank at 1M <= 70 GB reserved: bf16 weights are replicated
   per rank (16.1 GB); the measured 128k allocated peak of 22.7 GB (target arm) contains
   ~2.0 GB of per-rank KV, and KV is sequence-sharded, so at 1M it becomes ~16.1 GB. Projected
   allocated at 1M ~36.8 GB target / ~41 GB spec; reserved is *not* measured at 128k (the CSV
   carries rank-0 allocated only), and at a typical 1.2-1.4x reservation that projects to
   ~44-57 GB. **The gate would pass**, with the reservation assumption stated rather than hidden.
4. **Branch: C.** The speedup is 1.113x, below the 1.5x threshold, so the bf16 Phase 3 campaign
   is **NOT run**. Phase 3 becomes Phase 3-C: the parked FP4 seed replicates, ceiling $75.

**Consequence recorded now, before it is felt later:** because branch C skips the campaign,
there are no bf16 rows, so the vLLM comparison stays FP4-vs-bf16 and no precision-matched
column will exist. That is a cost of the measured result, not an oversight.

### What the ablation actually says, including where it contradicts the prediction

**The quantization round-trip is not what makes the decode step slow.** Turning BOTH off saves
**11.3%** of a target-only decode token at 128k, and turning the FP4 weights off alone saves
**1.7%**. Stage b of the seeds/ablations session measured 87-98% of a decode step as local
compute and 3.79 s/step at 128k; this says that almost none of that local compute is the FP4 or
NF4 round-trip. Whatever the remaining ~88% is — attention over a 128k cache, the Python ring
loop's per-step work, kernel launches — it is not quantization.

**The rehearsal's spread did not survive 16x more context.** At 8192/48 tokens the same four
arms separated by 2.9x (7.88 / 6.75 / 3.57 / 2.68 s); at 128k the same ordering holds —
CONTROL > W_OFF > KV_OFF > BOTH_OFF — but the spread collapses to 1.11x. The ordering was
predicted in Phase 0 and that prediction is confirmed; its magnitude was over-read there, and
Phase 0 says so.

**The spec arms disagree with the target-only ordering, strongly.** KV_OFF is the best spec arm
by a wide margin (acceptance **0.4096** and 1.420x decode), while BOTH_OFF is the *worst*
(acceptance **0.2422**, 1.030x) — worse than the control it improves on in the target arm. On
one book, bf16 weights *lower* acceptance (0.2422 vs 0.2712) and bf16 KV *raises* it (0.4096 vs
0.2712). This is a single-book observation (n=1 per arm): it is recorded as the reason
`KV_OFF` deserves its own replicate if precision work resumes, not as a trend.

### Cost

Phase 1: 90-minute timer, launched 21:43:57Z. Rehearsals: quant $0.58 + bf16 $0.67 = **$1.25**
of the $10 rehearsal ceiling (two earlier attempts never got a card and cost $0.00; the
misconfigured `nproc 8` launch also cost $0.00, its VM never booted).

### Phase 3-C scheduling, decided here rather than improvised later

The deliverable is the FP4 seed replicates. A config's `seeds` can only live in `defaults`, and
`defaults` applies to every stage in the config, so one session cannot run seed 42 stages
alongside seed 123 stages; but one session CAN run both replicates by listing
`seeds: [123, 456]` on a single 1M spec stage. Two parked-config sessions would cost the same
and need two capacity waits. Phase 3-C therefore runs **one** session from a new config
(`configs/mlsys_fp4_seeds_123_456_1m.yml`) that lists both seeds; the parked
`mlsys_seed123_1m.yml` and `mlsys_seed456_1m.yml` are left untouched, as instructed.

Budget arithmetic: the FP4 seeds/ablations session measured a 1M/5-book spec stage at
**96m40s (19.3 min/row)**. Two seeds = 193 min of stage time; with the sanity pair and
provisioning that is ~205 min = **$76** at $22.32/hr against the $75 ceiling. The session
therefore runs with a timer capped by the ceiling (201 min), seed 123 first: if the timer
intervenes mid-second-replicate the first replicate's rows are already pulled and committed
(R8) and the truncation is reported as such.

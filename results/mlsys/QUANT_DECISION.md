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

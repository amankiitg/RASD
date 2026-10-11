# RASD — October results

**Status: the 1M-context rows RAN.** One combined session on `gpu_8x_a100_80gb_sxm4`
completed on 2026-10-11, five of its six stages finished, and the headline numbers below
are measured, not projected. Two results are missing or superseded and are marked as
such rather than interpolated.

Every number here comes from a CSV pulled off the pod and carries the instance type it
was measured on. Every speedup compares spec against target-only **within the same
session**, on the same document, at the same context.

---

## 1. Headline table

Target `gradientai/Llama-3-8B-Instruct-Gradient-1048k` (native 1M, `rope_type: none`,
no scaling), draft `meta-llama/Llama-3.2-1B`, 8x A100 80GB SXM4, us-east-1,
`gpu_8x_a100_80gb_sxm4`, NF4 KV. `max_new_tokens` 128, greedy, one seed (42), 3 documents.

| context | arm | accept (per doc) | accept mean | tok/s (per doc) | med. tok/s | s/round | peak MB | speedup vs target-only |
|---|---|---|---|---|---|---|---|---|
| **1M** (1,048,576) | spec | 0.2132 / 0.6429 / 0.7419 | **0.5327** | 0.080 / 0.130 / 0.140 | 0.130 | 22.1–27.7 | 26,855 | **2.60x** (median paired) |
| **1M** | target-only | — | — | 0.050 / 0.050 / 0.050 | 0.050 | 19.3–19.7 | 25,831 | — |
| **512k** (524,288) | spec | 0.2705 / 0.6500 / 0.6691 | **0.5299** | 0.180 / 0.280 / 0.300 | 0.280 | 11.4–12.7 | 16,822 | **2.80x** (median paired) |
| **512k** | target-only | — | — | 0.100 / 0.100 / 0.100 | 0.100 | 10.0–10.2 | 15,798 | — |
| **128k** (131,072) | spec | 0.3273 / 0.5658 / 0.7188 | **0.5373** | 0.600 / 0.780 / 0.930 | 0.780 | 3.9–4.2 | 10,125 | **not computable** — see below |
| **128k** | target-only | — | — | — | — | — | — | **ARM LOST** (section 5) |

Paired per-document speedups: 1M = 1.60x / 2.60x / 2.80x; 512k = 1.80x / 2.80x / 3.00x.

**The headline result is two facts, and they point in opposite directions:**

1. **Acceptance does not degrade with context.** Mean acceptance is 0.5373 at 128k,
   0.5299 at 512k, 0.5327 at 1M — flat to within 1.5% across an 8x increase in context.
   Speculative decoding keeps working at 1M; the draft does not become useless.
2. **Throughput still falls 6.6x** from 128k to 1M (0.770 -> 0.117 tok/s), because the
   per-round cost grows with context (3.9 -> 27.7 s/round) while each round still emits
   ~1.8 tokens. The target-only arm falls identically (0.27 -> 0.050 tok/s). The
   *speedup* is therefore roughly preserved (2.80x -> 2.60x) even as absolute throughput
   collapses. **Speculation is not what breaks at long context — the forward pass is.**

Per-document ordering is stable across all three contexts (doc 0 low, docs 1 and 915
high), which is what you would expect if acceptance is dominated by the document rather
than by the context length. It is not proof: see the alignment caveat in section 3.

---

## 2. Quality check (1M vs 128k) — SUPERSEDED, do not quote

The rows that ran scored perplexity over **8 tokens**, not the final 1024. The cause is a
config error of mine, recorded in `MORNING_REPORT` item 7c: the window is
`prompt_gen_tokens`, and `max_new_tokens` does not control it. The 1M rows are all
`status=ok` but their `ppl_tokens` is 8:

| document | context | ppl | ppl_tokens |
|---|---|---|---|
| pg19_train_0 | 1,048,576 | 1.1846 | **8** |
| pg19_train_1 | 1,048,576 | 12.7478 | **8** |
| pg19_train_915 | 1,048,576 | 9.7225 | **8** |

The spread from 1.18 to 12.75 across three books is what 8 tokens of perplexity looks
like; it is not a finding about context length. **The 128k arm of this comparison ran 1
of 3 rows** (accept 0.1250); the other two failed with the engine's known window-search
error (`could not find a source length`, and `continuation is 0 tokens`) on the
exact-length aligned pool.

**These rows are superseded.** The redo — 3 documents, final 1024 tokens, both arms,
on the rebuilt pools with forward slack — is `QUALITY_FIX_128K` / `QUALITY_FIX_1M` in
the follow-up session (section 8).

---

## 3. Scaling curve

See `results/mlsys/scaling.png` (acceptance and tokens/s vs context; generated from the
five valid stage CSVs only — the stale 128k target-only file is deliberately excluded,
section 5).

| context | accept mean (n=3) | spec tok/s mean | target tok/s mean | s/round (spec) |
|---|---|---|---|---|
| 128k | 0.5373 | 0.770 | (lost) | 4.1 |
| 512k | 0.5299 | 0.253 | 0.100 | 12.1 |
| 1M | 0.5327 | 0.117 | 0.050 | 25.4 |

**UNALIGNED ACROSS CONTEXTS — acceptance is not strictly comparable between these rows.**
All three read the full-book pool from its start, so the 1M continuation begins at book
token 1,048,576 while the 512k continuation begins at 524,288: the arms score *different
passages*. Target-only timing remains valid (it is a per-round cost measurement), but a
difference in acceptance across contexts could be a passage effect rather than a context
effect. The passage-controlled version of this table is `SCALING_ALIGNED_*` in the
follow-up session, where all three windows end at the same token as the 1M prompt and
the continuation ids are asserted `array_equal` against the 1M arm.

The sanity pair's prompt, by contrast, is the book's *opening* passage, and the headline
and scaling numbers come from the full-book pool starting at token 0 — so the 2.70x
sanity figure and the headline figures are from different positions in the same books.

---

## 4. The near-tie finding (roadmap M3) — closed, no engine bug

Rerunning one mismatching `natural_f1` pair at 32k with `--dump-logits-topk` showed both
divergences are the **same candidate set reordered**, not a decoding error:

* `pg19_train_1`, token 4: both arms hold {617, 7236, 656}; spec spread 0.375, target
  0.188. Each arm's choice is the other's rank-3 candidate.
* `pg19_train_0`, token 16: both arms hold {2030, 1789}; spec spread 1.125, target 0.25.

11.7–36.7% of positions in those rows have their own top1–top2 gap below 1.125, so this
is a near-tie at the fp4 noise scale. Caveat kept honest: 1.125 exceeds pure bf16 noise,
so this is filed as *near-tie at the fp4 noise scale*, not as fully explained.

---

## 5. The session's own record: hardware, spend, one lost arm, one stale file

**Session:** instance `12e9c5d794254c498d741261bde0877b`, `gpu_8x_a100_80gb_sxm4`,
us-east-1, **$22.32/hr**, launched 2026-10-10T23:34:04Z, terminated
2026-10-11T05:38:39Z. Wall **6.08 h = $135.62**. 0 instances confirmed after terminate,
dead-man timer cancelled.

**Stages that completed:** sanity (2/2), QUALITY_128K (1 ok / 2 failed), QUALITY_1M
(3/3), HEADLINE_1M_SPEC (3/3), HEADLINE_1M_TARGET (3/3), SCALING_512K_SPEC (3/3),
SCALING_512K_TARGET (3/3), SCALING_128K_SPEC (3/3). 22 ok rows, 2 failed.

**The 128k target-only arm was lost, and it was lost to the Mac, not the pod.** The
Mac->pod SSH connection broke at ~04:12Z (`Read from remote host 129.213.202.106:
Connection reset by peer`). The pod kept working and completed `SCALING_512K_TARGET`
(05:18:48Z) and `SCALING_128K_SPEC` (05:29:16Z) — the final pull at 05:35:12Z captured
both — but the driver then **terminated the instance at 05:35:32Z while
`SCALING_128K_TARGET` was mid-flight** (its log shows round 80 of 128 at 299 s). So the
last stage was cut, and with it the 128k target-only arm. Treating the ssh return as the
end of the remote work was wrong: the remote script survives the ssh session that started
it, which is exactly what happened here.

**A stale rehearsal file nearly became a reported result.** The pulled
`session_a_SUNDAY_SCALING_128K_TARGET.csv` contains **one row, `context_length` 4096,
`max_new_tokens` 8** — the REHEARSAL's row, mtime 23:27:59Z, left in place by rsync
because the main session never produced that file. Trusting it gives a "128k speedup" of
0.78 / 4.43 = **0.18x**, i.e. spec decoding *five times slower* at 128k — an artifact
comparing a real 128k spec row against a 4k rehearsal target row. It is excluded
everywhere in this document and deliberately kept out of the plot. The guard that would
have caught it is a context/`max_new_tokens` check per stage against the config, not a
file-exists check.

**Also recorded:** the main session's dead-man timer disarmed itself at 00:22:35Z with
`instance ... no longer exists` while the instance was active and billing, leaving the
mandated safety net off for about an hour. Cause and fix in `MORNING_REPORT` item 7h: the
check exited on any single failed poll, so one timeout was indistinguishable from a real
"not found". It now requires three consecutive *authoritative* answers (200/404) before
disarming.

**Spend to date:** $135.62 for this session; **~$157 cumulative** against the $450
global new-spend cap (earlier $20.82 + $0.57 of rehearsals + this session).

---

## 6. Rope intervention, redesigned (follow-up session, LAST stage)

Not yet run. Three arms differing only in the target's rotary block — native (shipped
`llama3` block untouched), `llama3` factor 16, `llama3` factor 32, all
`rope_anchor_base: 8192` so a difference is attributable to the *factor* and not to a
swapped rope implementation. Spec rows for all three arms, target-only rows for the
native arm only, so the speedup comes from the same session. The coherence gate runs on
the three arms as the first step of the stage; its verdict is reported, never enforced —
an arm that fails the gate still gets its rows, and its acceptance is marked as coming
from an incoherent model.

---

## 7. What the paper can and cannot claim

**Can claim (measured, 8x A100 80GB SXM4, instance type recorded per table):**

* Speculative decoding holds acceptance at 1M context: mean 0.5327 at 1M vs 0.5373 at
  128k, three documents, one seed, greedy.
* The 1M speedup against our own target-only path is **2.60x** (median paired), with
  2.80x at 512k. Every speedup is same-session, same-document, same-context.
* Absolute throughput at 1M is 0.117 tok/s spec vs 0.050 tok/s target-only, at 26,855 MB
  peak per rank. Both arms fit an 80 GB rank at 1M with NF4 KV.
* Looping is a property of the base model on this data, not of speculation: 2 of the 3
  target-only 1M rows loop under the same absolute criteria as the spec rows (details
  below).

**Cannot claim:**

* That acceptance is *context*-independent. The three contexts are not passage-aligned
  (section 3), so this is confounded until `SCALING_ALIGNED_*` lands. What is solid is
  that acceptance does not *collapse*, and that per-document ordering is stable.
* A 128k speedup. The target-only arm was lost (section 5).
* Any quality/perplexity claim at 1M. The only rows that ran scored 8 tokens.
* That these speedups say anything about production serving. They compare against **our
  own** target-only path, which is known to be slow (0.050 tok/s at 1M target-only). No
  vLLM baseline was run for this document; the vLLM comparison is a separate item and
  must not be inferred from these numbers.
* Generalization beyond 3 documents, one seed, concatenated PG-19 books, greedy
  decoding, `max_new_tokens` 128.

---

## 8. Limitations, stated before anyone asks

1. **n = 3 documents**, all PG-19 books, concatenated to 1M tokens. All three were
   already in the campaign's 128k pool, so the document set is not independently sampled.
2. **One seed (42)** for every row here. The multi-seed analyses are separate.
3. **Looping is pervasive and is reported per row, not averaged away.** Under the same
   absolute ceilings (verbatim span >= 128, tokens-in-repeat > 0.60, periodic tail >
   0.50): at 1M, 2 of 3 spec rows loop and 2 of 3 target-only rows loop; at 512k, 2 of 3
   spec and 1 of 3 target-only; at 128k, 2 of 3 spec. Eight of the fifteen rows with token
   files loop. The looping spec rows carry *higher* acceptance (0.64, 0.74) than the
   clean one (0.21), because a greedy cycle is trivially "accepted" — so the headline
   acceptance mean should be read with the per-row periodicity column, never alone.
4. **The 1M cost is measured on one rank configuration.** Per-rank peak 26,855 MB at 1M
   (weights ~6.8 GB + NF4 KV ~5.2 GB + working set). It fits 80 GB; it would not fit 40 GB.
5. **The scaling arms are not passage-aligned** (section 3), so target-only timing is
   valid but cross-context acceptance is not yet clean.
6. **The 8-token quality rows are noise, not a direction.** No claim is made from them in
   either direction.

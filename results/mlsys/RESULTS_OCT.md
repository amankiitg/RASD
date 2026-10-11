# RASD — October results

**Status: the 1M rows ran, but the headline must be read one row at a time.** Most rows
loop, and our protocol excludes looping rows from headline means. When that rule is
applied, the 1M headline is **one row**, and it is **1.61x**, not 2.60x. Everything below
is per-row first, means second.

Target `gradientai/Llama-3-8B-Instruct-Gradient-1048k` (native 1M, `rope_type: none`, no
scaling), draft `meta-llama/Llama-3.2-1B`, 8x A100 80GB SXM4 (`gpu_8x_a100_80gb_sxm4`,
us-east-1, $22.32/hr). NF4 KV, `max_new_tokens` 128, greedy, seed 42, 3 PG-19 books.

---

## 1. Every 1M / 512k / 128k row, with its periodicity

tok/s is `tokens_generated / time_sec` computed here, **not** the CSV's `throughput_tps`
field, which is rounded to two significant figures (0.08 vs 0.08411 for the 1M clean row —
enough to move a ratio from 1.61x to 1.68x). Periodicity columns: dominant period,
first periodic token (128 = never, i.e. clean), periodic-tail share, self-agreement.

| ctx | arm | document | accept | tok/s | rounds | s/round | period | first | tail | verdict |
|---|---|---|---|---|---|---|---|---|---|---|
| **1M** | spec | pg19_train_0 | **0.2132** | **0.08411** | 69 | 22.1 | 46 | 128 | 0.00 | **CLEAN** |
| 1M | spec | pg19_train_1 | 0.6429 | 0.13233 | 36 | 26.9 | 21 | 75 | 0.41 | LOOPING |
| 1M | spec | pg19_train_915 | 0.7419 | 0.14416 | 32 | 27.7 | 18 | 31 | 0.76 | LOOPING |
| **1M** | target-only | pg19_train_0 | — | **0.05236** | 127 | 19.3 | 44 | 128 | 0.00 | **CLEAN** |
| 1M | target-only | pg19_train_1 | — | 0.05113 | 127 | 19.7 | 50 | 38 | 0.70 | LOOPING |
| 1M | target-only | pg19_train_915 | — | 0.05132 | 127 | 19.6 | 19 | 81 | 0.37 | LOOPING |
| **512k** | spec | pg19_train_0 | **0.2705** | **0.18454** | 61 | 11.4 | 57 | 128 | 0.00 | **CLEAN** |
| 512k | spec | pg19_train_1 | 0.6500 | 0.28012 | 36 | 12.7 | 26 | 61 | 0.52 | LOOPING |
| 512k | spec | pg19_train_915 | 0.6691 | 0.30211 | 35 | 12.1 | 16 | 25 | 0.80 | LOOPING |
| **512k** | target-only | pg19_train_0 | — | **0.10024** | 127 | 10.1 | 49 | 128 | 0.00 | **CLEAN** |
| 512k | target-only | pg19_train_1 | — | 0.10075 | 127 | 10.0 | 3 | 49 | 0.62 | LOOPING |
| **512k** | target-only | pg19_train_915 | — | **0.09837** | 127 | 10.2 | 56 | 128 | 0.00 | **CLEAN** |
| **128k** | spec | pg19_train_0 | **0.3273** | **0.60384** | 55 | 3.9 | 28 | 128 | 0.00 | **CLEAN** |
| 128k | spec | pg19_train_1 | 0.5658 | 0.77862 | 39 | 4.2 | 10 | 61 | 0.52 | LOOPING |
| 128k | spec | pg19_train_915 | 0.7188 | 0.92820 | 33 | 4.2 | 37 | 41 | 0.68 | LOOPING |
| 128k | target-only | — | — | — | — | — | — | — | — | **ARM LOST** (section 5) |

**6 of 15 rows are clean. 9 loop.** Note the target-only rows loop too (2 of 3 at 1M):
looping is a property of the greedy decode on this data, not of speculation.

Correction to what this document said earlier: it reported a 1M mean speedup of 2.60x and
a mean acceptance of 0.5327 by averaging the looping rows in. Those rows reach high
"acceptance" precisely because a greedy cycle is trivially accepted (0.6429 and 0.7419 vs
0.2132 on the clean row). **Averaging them in inflates acceptance.** That claim is
withdrawn.

---

## 2. Clean-rows-only means, with n

| context | arm | n | mean accept | mean tok/s | documents |
|---|---|---|---|---|---|
| 128k | spec | **1** | 0.3273 | 0.60384 | train_0 |
| 128k | target-only | 0 | — | — | (arm lost) |
| 512k | spec | **1** | 0.2705 | 0.18454 | train_0 |
| 512k | target-only | 2 | — | 0.09931 | train_0, train_915 |
| 1M | spec | **1** | **0.2132** | **0.08411** | train_0 |
| 1M | target-only | 1 | — | **0.05236** | train_0 |

**Clean paired speedup:** 1M = 0.08411 / 0.05236 = **1.61x (n=1)**. 512k =
0.18454 / 0.10024 = **1.84x (n=1)**. 128k: not computable.

**The 1M clean result is one row.** Acceptance 0.2132, 0.08411 vs 0.05236 tok/s, 1.61x.
It is a single document and must be quoted as such.

**"Acceptance does not degrade with context" is withdrawn.** On the only document clean at
all three contexts (`pg19_train_0`), acceptance *falls* as context grows:

| context | accept | tok/s | paired speedup |
|---|---|---|---|
| 128k | 0.3273 | 0.60384 | not computable |
| 512k | 0.2705 | 0.18454 | 1.84x |
| 1M | 0.2132 | 0.08411 | 1.61x |

0.3273 -> 0.2705 -> 0.2132 is a 35% relative decline from 128k to 1M, on n=1. The earlier
"flat to 1.5%" came entirely from including loop-inflated rows. What survives is weaker
and narrower: **absolute throughput still falls with context** (0.604 -> 0.185 -> 0.084
tok/s) while the *speedup* against our own target-only path declines gently (1.84x ->
1.61x). Speculation keeps working at 1M; it does not keep working equally well.

---

## 3. Free salvage: acceptance before the loop

For each looping spec row, acceptance recomputed over only those rounds **before the first
periodic token**, from the per-round traces. **Per-round timing does not exist in these
traces** — a record has `n_acc`, `n_committed`, `n_emitted`, `spec_steps`, `global_pos_start`,
`accepted[]`, `emitted_gaps`, `kv_len_before/after`, `ended_on_eos`, `round_truncated` and
`round_idx`, and no time field of any kind. So this salvage yields **acceptance only; no
throughput can be recovered this way.**

| context | document | first periodic | rounds (pre/total) | accept pre-loop | accept whole row | tokens pre-loop |
|---|---|---|---|---|---|---|
| 1M | pg19_train_1 | 75 | 24/36 | 0.5417 | 0.6429 | 76 |
| 1M | pg19_train_915 | 31 | 11/32 | 0.4545 | 0.7419 | 31 |
| 512k | pg19_train_1 | 61 | 21/36 | 0.4762 | 0.6500 | 61 |
| 512k | pg19_train_915 | 25 | 14/35 | 0.2679 | 0.6691 | 29 |
| 128k | pg19_train_1 | 61 | 25/39 | 0.3700 | 0.5658 | 62 |
| 128k | pg19_train_915 | 41 | 12/33 | 0.6042 | 0.7188 | 41 |

The salvage confirms the direction: **the whole-row acceptance is inflated by the loop**,
by 0.10 to 0.29 absolute. Substituting the pre-loop value for the looping rows and keeping
the clean rows whole-row gives mixed means of 0.4338 (128k), 0.3382 (512k), 0.4031 (1M) —
**not monotone**, so no trend may be claimed in either direction even with the salvage.
The salvage is best used to bound the inflation, not to add n.

---

## 4. Can the engine sample at temperature > 0? Yes — and it is tested

The engine implements proper speculative rejection sampling, not just greedy matching:

* `_sample(logits, temperature, top_p)` (`src/models/rasd_inference.py:327`): `temperature == 0.0`
  returns `argmax`; otherwise it divides by temperature, applies optional nucleus `top_p`
  filtering, and returns `torch.multinomial(probs, 1)`. Used for the seed token, the
  target-only loop, and the draft proposals (`.py:1725`, `:1819`, `:1967`).
* `_acceptance_mask(..., temperature)` (`.py:346`): at `temperature == 0.0` it accepts iff
  the target argmax equals the draft token; at `temperature > 0` it computes
  `accept_prob = clamp(p_target(x) / (p_draft(x) + eps), max=1.0)` and draws
  `r = torch.rand_like(accept_prob)`, accepting where `r < accept_prob` — the standard
  Leviathan et al. criterion.
* The bonus/residual resample on rejection is the correct `normalize(max(0, p_t - p_d))`,
  and it is pinned by an **empirical test at temperature 1.0**
  (`tests/test_verification_math.py::test_residual_resample_on_rejection`), which fails if
  the bonus is drawn from raw `p_target` instead.
* Unit tests at temperature 1.0 also cover the accept mask and its bounded rate
  (`tests/test_rasd_components.py:103,130,157`), and `_sample` at several temperatures and
  `top_p` values. RNG state is saved and restored for both CPU and CUDA precisely because
  at temperature > 0 each rank samples its own proposal (`.py:1470`), covered by
  `tests/test_mlsys_round_consensus.py` and `tests/test_m3_invariants.py`.

**Has it been run end-to-end? Yes** — `configs/mlsys_arm4_f1_128k.yml` (the `natural_f1`
campaign) and the `m4_*` matrix configs all set `temperature: 1.0`, so rank-consensus
sampling has been exercised at scale. The two 1M sessions and the follow-up all run
`temperature: 0.0` (greedy). **Not tested:** sampling at 1M context with 8-rank ring
attention, and sampling for the target-only arm at 1M.

---

## 5. Session record: hardware, spend, one lost arm, one stale file

Instance `12e9c5d794254c498d741261bde0877b`, `gpu_8x_a100_80gb_sxm4`, us-east-1,
**$22.32/hr**, 2026-10-10T23:34:04Z -> 2026-10-11T05:38:39Z = **6.08 h = $135.62**.
0 instances after terminate, dead-man timer cancelled. Cumulative ~$157 of the $450 cap.

**The 128k target-only arm was lost to the Mac, not the pod.** The Mac->pod SSH broke at
~04:12Z (`Connection reset by peer`); the pod kept working and finished
`SCALING_512K_TARGET` (05:18:48Z) and `SCALING_128K_SPEC` (05:29:16Z), both captured by
the 05:35:12Z final pull, but the driver then terminated the instance at 05:35:32Z while
`SCALING_128K_TARGET` was mid-flight (round 80 of 128). A remote script outlives the SSH
session that started it, so an SSH return is not evidence that remote work ended.

**A stale rehearsal file nearly became a number.** The pulled
`session_a_SUNDAY_SCALING_128K_TARGET.csv` holds one row with `context_length` 4096 and
`max_new_tokens` 8 — the REHEARSAL's row, mtime 23:27:59Z, left by rsync (no `--delete`)
because the killed stage never wrote that file. Believing it gives 0.78 / 4.43 = **0.18x**,
"spec decoding five times slower at 128k". Excluded everywhere.

**Timer incident.** The dead-man timer disarmed itself at 00:22:35Z with `instance ... no
longer exists` while the instance was active and billing, because the check exited on any
single failed poll. It now requires three consecutive authoritative answers (200/404).
See `MORNING_REPORT` item 7h.

**Superseded:** the quality rows scored perplexity over **8 tokens** (`ppl_tokens` 8), not
the final 1024, because the window is `prompt_gen_tokens` and I left it to `max_new_tokens`.
The 128k arm ran 1 of 3 rows. Redo is `QUALITY_FIX_*` in the follow-up.

**Unresolved and marked as such:** the three contexts are **not passage-aligned**. All
scaling rows read the full-book pool from its start, so the 1M continuation begins at book
token 1,048,576 while the 512k continuation begins at 524,288 — different passages.
Target-only *timing* is valid; cross-context *acceptance* is confounded. The follow-up's
`SCALING_ALIGNED_*` groups end all windows at the 1M prompt's final token with the
continuation ids asserted `array_equal` to the 1M arm.

---

## 6. What the paper can and cannot claim

**Can claim:** at 1M context on 8x A100 80GB SXM4, with NF4 KV and a windowed 1B draft, a
single clean document showed acceptance 0.2132 and 0.08411 vs 0.05236 tok/s target-only =
**1.61x** (n=1); at 512k, 1.84x (n=1). Both arms fit one 80 GB rank at 1M (peak 26,855 MB).
Looping is the greedy decode's, not speculation's: 2 of 3 target-only 1M rows loop under
the same criteria as the spec rows.

**Cannot claim:** any 1M acceptance *mean* over documents (n=1 clean). Any statement that
acceptance is flat or improving with context — on the single document clean at all three
contexts it declines 0.3273 -> 0.2705 -> 0.2132. Any 128k speedup (arm lost). Any
perplexity claim at 1M (8-token rows). Anything about production serving: the comparator
is our own target-only path, which is very slow (0.05236 tok/s at 1M), and no vLLM
baseline exists for this document. Nothing generalizes beyond 3 documents, 1 seed, greedy.

---

## 7. Limitations

1. **n = 3 documents**, all PG-19 books already in the campaign's 128k pool. After
   exclusion, **n = 1** per arm for the headline.
2. **One seed (42)**, greedy decode. Greedy is what produces the loops.
3. **9 of 15 rows loop**, and looping rows inflate acceptance by 0.10–0.29 absolute. Three
   documents cannot support a distributional claim about how often this happens.
4. **No per-round timing** anywhere in the traces, so no row can be re-timed after the fact
   and no pre-loop throughput can be recovered.
5. **Cross-context acceptance is confounded** by passage (section 5).
6. **Sampling is available and tested but never used at 1M** (section 4).

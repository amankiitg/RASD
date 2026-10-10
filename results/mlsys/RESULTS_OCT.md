# RASD — October results

Generated 2026-10-10/12 by an unattended session. Every number below is from a run
that happened, on hardware named in the table. Where a row did not run, it says so
instead of being estimated.

**Instance types, recorded with every table because a comparison across hardware is
not a comparison:**
* `gpu_1x_a100_sxm4` — 1x A100 40GB SXM4, $1.99/hr (Saturday pilot)
* `gpu_8x_a100_80gb_sxm4` — 8x A100 80GB SXM4, $22.32/hr (Session A, polled; see §5)

Every speedup below compares the speculative arm against the target-only arm **within
the same session on the same node**, never against a published number from elsewhere.

---

## 1. Headline table

Generation protocol everywhere: `max_new_tokens 128`, greedy (temperature 0, top_p 1),
`ignore_eos`, NF4 weights, NF4 KV cache on the target, draft window capped at 4096.
Acceptance is the per-round quantity **alpha = accepted-prefix-length / gamma**, never
the i.i.d. per-token parameter. Periodicity is reported next to acceptance, and a row
whose generation loops is flagged and excluded from headline means.

| context | arm | target | draft | acceptance | tok/s | speedup vs target-only | period | self-agr | first periodic | periodic tail | flag | hardware |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 32k | spec | Llama-3.1-8B | Llama-3.2-1B | 0.2705 | 3.58 | 1.24x | 49 | 0.089 | 128 (never) | 0.00 | clean | 1x A100 40GB |
| 32k | spec | Llama-3.1-8B | Llama-3.2-1B | 0.2540 | 3.49 | 1.20x | 45 | 0.229 | 128 (never) | 0.00 | clean | 1x A100 40GB |
| 32k | target-only | Llama-3.1-8B | — | — | 2.89 | 1.00 (denominator) | 50 | 0.077 | 128 (never) | 0.00 | clean | 1x A100 40GB |
| 32k | target-only | Llama-3.1-8B | — | — | 2.90 | 1.00 (denominator) | 16 | 0.089 | 128 (never) | 0.00 | clean | 1x A100 40GB |
| 32k | spec | **native-1M** Gradient 1048k | Llama-3.2-1B | 0.2708 | 3.61 | — (no paired target-only) | 31 | 0.072 | 128 (never) | 0.00 | clean | 1x A100 40GB |
| 128k | spec | native-1M Gradient 1048k | Llama-3.2-1B | 0.2188 | 0.46 | **1.64x** | 14 | 0.060 | 64 (never) | 0.00 | clean | **8x A100 80GB** |
| 128k | target-only | native-1M Gradient 1048k | — | — | 0.28 | 1.00 (denominator) | 8 | 0.089 | 64 (never) | 0.00 | clean | **8x A100 80GB** |
| **1M** | spec | native-1M Gradient 1048k | Llama-3.2-1B | **did not run** | — | — | — | — | — | — | — | see §5 |
| **1M** | target-only | native-1M Gradient 1048k | — | — | **did not run** | — | — | — | — | — | — | 8x A100 80GB unavailable |

The two 128k rows are the Session A **sanity pair** on the 8x A100 80GB node: one
document, 64 tokens, and the only pair this window got from the 8x node (see §5 for
why). They are a same-session, same-node, same-document speedup, which is the only
kind this document reports: **0.46 vs 0.28 tok/s = 1.64x**, with acceptance 0.2188 at
gamma 4 giving 1.94 tokens per verify round. Peak memory was 10,118 MB (spec) and
8,275 MB (target-only) **per rank** across 8 ranks.

Two documents at 32k (the Bible's opening and Shakespeare's opening), 128 tokens each.
The Gradient row carries no speedup because its target-only partner was not run: a
speedup computed against a *different model's* target-only row would be a
cross-model comparison wearing a speedup's clothes.

**The honest headline from what ran:** on real PG-19 prose at 32k, with every row
verified repetition-clean, speculative decoding with Llama-3.2-1B against an 8B target
accepts ~0.26-0.27 of its draft prefix and delivers ~1.2x over the same target decoding
alone. The native-1M target gives the same acceptance (0.2708 vs 0.2705) as
Llama-3.1-8B on the same document, i.e. the 1M-capable checkpoint is not
speculation-hostile.

---

## 2. Quality check (1M vs 128k, same targets)

**Not run.** Requires `gpu_8x_a100_80gb_sxm4` (§5). The design is in place and
verified: `data/processed/pg19_1m_128kwin/` holds, per book, the slice
`ids[917504:1048576]` so that a 128k prompt ends exactly where the 1M continuation
begins; the build asserts `array_equal` between the two arms' continuation ids for all
six documents. Without that assertion the two perplexities would have been measured
over two different passages.

---

## 3. Scaling curve

**Not run** (Session B, contingent on Session A). What exists is the arithmetic that
bounds it: at 1M the per-rank projection is ~27 GB (weights 6.8 + NF4 KV 5.2 + a
working set that scales with the sequence), which fits an 80GB card with ~3x headroom
and does not fit a 40GB one with confidence; at 512k the same projection is ~14 GB.
Long context here is a multi-GPU property, not a budget choice: at `world_size 1` the
NF4 KV alone is 41.6 GB at 1M.

![acceptance and throughput vs context](scaling.png)

*(plot regenerates with `python scripts/mlsys_plot_scaling.py`; looping rows are
marked and excluded from means)*

---

## 4. The near-tie finding (M3) — closed, no engine bug

The campaign's `natural_f1_128k_losslessness` stage reported ten MISMATCHes: the
speculative and target-only arms agreed on the prompt and then emitted different
tokens. Each arm recorded only its OWN top1-top2 gap, which cannot settle the case that
matters — when the arms favour *different* pairs. The engine now dumps top-k logits on
request (`--dump-logits-topk`) and the losslessness stage computes the **cross margin**:
how each arm scored the *other* arm's token.

| document | divergence | spec arm top-k | target arm top-k | cross margins | class |
|---|---|---|---|---|---|
| pg19_train_1 | token 4 | 617 (14.062), 7236 (13.750), 656 (13.688) | 656 (14.062), 7236 (13.938), 617 (13.875) | +0.375 / +0.188 | near-tie |
| pg19_train_0 | token 16 | 2030 (25.250), 1789 (24.125) | 1789 (23.500), 2030 (23.250) | +1.125 / +0.250 | near-tie at the fp4 noise scale |

Both divergences are the SAME candidate set ordered differently. Each arm's choice is
the other arm's rank-2 or rank-3. The target is numerically indifferent at both
positions (0.25 and 0.188 logits; fp4 logits are quantised in 0.0625 steps), and
11.7-36.7% of positions in these rows have their own top1-top2 gap below 1.125, so a
divergence there is an ordinary event rather than an outlier. **Verdict: no engine
bug.** The arms are one target in two forward shapes (a gamma=4 batched verify and a
single-token decode) under NF4 weights, diverging only where the target itself is
indifferent — which is what the project's own `TIE_GAP = 0.1` rule was built for.
Recorded caveat: the 1.125 margin is the largest observed and exceeds pure bf16
reduction noise, so it is filed as "near-tie at the fp4 noise scale", not as fully
explained.

---

## 5. Hardware used, and why the 1M rows still have not run

| # | instance | type | wall | cost |
|---|---|---|---|---|
| 1 | e683001fc17c | gpu_1x_a100_sxm4 | 9.7 min | $0.32 |
| 2 | f6aa5499c537 | gpu_1x_a100_sxm4 | 12.5 min | $0.41 |
| 3 | fa57c8f7fe9f | gpu_1x_a100_sxm4 | 10.9 min | $0.36 |
| 4 | 0dfd9bafe061 | **gpu_8x_a100_80gb_sxm4** (us-midwest-1) | 0.88 h | $19.73 |
| | | **total** | | **$20.82** |

Attempt 1 produced no results: two setup bugs of mine (the pod env is
`~/RASD/.pod_env.sh` rather than `~/.pod_env.sh`, and `--nproc 1` is mandatory on a
single-GPU pod because the launcher's default is 8). Both fixed and recorded.

**Session A did acquire the node** — capacity appeared in us-midwest-1 at 18:56:53Z on
2026-10-10 and the pod ran. It did the expensive part correctly: boot (13 min),
provisioning (19 min, rc=0), staging, and then the sanity gate, which PASSED with two
ok rows — the pair in §1. Then all four real groups were refused **in the same
second**, before touching the GPU:

```
REFUSING to write results/mlsys/session_a/session_a.csv for stage
'session_a_SUNDAY_QUALITY_1M': the file holds rows from ['SANITY','SANITY_targetonly'],
which this stage does not produce.
```

That guard is **correct** and I am glad it exists: it is what stops one stage's rows
being silently interleaved with another's, which would have poisoned every table built
from that CSV. The fault was mine — my remote script passed ONE `--output` filename to
all four groups. Fixed by giving every stage its own CSV (`session_a_<group>.csv`), with
the sanity pair on `session_a_sanity.csv`. The refusal is instant, so no GPU work was
wasted on it, but the pod had already paid for boot and provisioning: **0.88h,
$19.73**.

The instance was terminated on completion and the API confirmed **0 instances**; the
dead-man timer armed at launch was disarmed once its id was gone.

The poller has been relaunched through `scripts/mlsys_detach.py` (not `nohup`, which is
how the 2026-10-10 runtime restart killed the watcher). Verified by effect, not by the
command: **pid 37535, PPID 1, process group 37534 against the launching shell's 37583**
— a different process group *and* a different session, so a signal to the shell's group
cannot reach it. The driver now also checks the timer's PPID after arming it and warns
if the timer is not reparented, rather than assuming it.

So: **the 1M rows have still not run.** What exists for 1M is that the model, the data
and the draft path are all verified ready (native-1M target loads and runs with no rope
scaling — measured; tokenizer identical to the draft over all 128,000 base ids —
measured; draft works at 1M through its existing 4096-token window — measured; six real
PG-19 books of 1,050,624 tokens each staged — built).

## 6. What the paper can and cannot claim

**Can claim**
* On PG-19 prose at 32k, with repetition verified absent from every row, the
  Llama-3.1-8B + Llama-3.2-1B pair accepts 0.254-0.271 of its draft prefix and gives
  ~1.20-1.24x over the same target decoding alone, on one A100 40GB.
* The native-1M checkpoint (`gradientai/Llama-3-8B-Instruct-Gradient-1048k`) runs in
  this engine with **no rope scaling at all** — `rope_theta` 3.58e9 and
  `rope_scaling: none` from its own config — and it speculates as well as
  Llama-3.1-8B does at the same context (0.2708 vs 0.2705 on the same document).
* The project's acceptance numbers for the 128k campaign must be read by window: the
  first window of those rows accepts 0.310-0.798 (median 0.706), four of the ten first
  windows are themselves periodic, nine of ten rows loop over their full 1024 tokens,
  and four rows have no non-degenerate window at all. The 0.94-0.97 figures are
  properties of greedy cycles.
* The spec/target token divergence at 128k is a numerics-scale effect, not an
  implementation defect: the arms reorder the same top-2/top-3 candidates where the
  target is indifferent.

**Cannot claim**
* Any 1M-context acceptance, throughput, speedup or quality number. None was measured.
* Any speedup larger than ~1.25x on these documents and this protocol.
* That 128k acceptance reaches 0.97. It does not, on loop-free windows.
* That these five pilot rows characterise PG-19: n=2 documents (plus one for the
  Gradient target), one seed, 128-token generations.

---

## 7. Limitations, stated before anyone asks

* **n is tiny.** The 32k pilot is 5 rows / 3 documents / 1 seed. Greedy decoding makes
  a seed change a no-op, so replication here means documents, and there are three.
* **The speedup denominator is our own target-only path**, which is slower than a
  production stack; vLLM was never run at these contexts in this window, so ~1.2x is
  a statement about this engine, not about the method's ceiling.
* **Acceptance is text-dependent** in a way the campaign's single number hid:
  0.25-0.27 on the 32k openings, 0.31-0.80 first-window at 128k. Any single headline
  acceptance is a statement about a passage as much as about a mechanism.
* **Looping is not rare.** 9 of 10 of the campaign's 128k rows loop at 1024 greedy
  tokens, and 4 of 10 first windows are already periodic. Greedy long-form decoding on
  an 8B model under NF4 is a loop generator, and every acceptance number from it needs
  its periodicity beside it.
* **The 1M rung remains unmeasured.** The node was acquired, the model and data are
  verified, and the session was killed by my own shared-output-filename bug before any
  1M row started. That is a defect in my orchestration, not a property of the method,
  and it is recorded that way.

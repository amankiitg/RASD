# LCFM CAMERA-READY KIT

Everything needed to write or check the camera-ready from one document. Sources are named per
row; the audit document is `results/mlsys/LCFM_RECONCILE.md` (cited as **AUDIT E<n>**).

Run date 2026-10-14. Branch `decode-diagnosis` (corrections) and `lcfm-camera-ready`
(manuscript). Engine vocabulary used throughout:

* **old engine** — the ring rotates each layer's K/V shard around the 8 ranks: 8 rotations x
  32 layers = **256 rounds per decode token**. Every row before 2026-10-13.
* **candidate engine** — `ring_decode_mode: split_kv`: no rotation; each rank attends to its own
  shard and the partials merge with a MAX and a SUM `all_reduce`. Rows from 2026-10-13.

Precision is **FP4 weights + NF4 KV** in every RASD row in this kit. Where a number is
old-engine only, it says so.

---

## 3a. CLAIM TABLE — F1 to F5

### F1 — Stretching RoPE barely moves perplexity but costs ~30% of speculative acceptance

| field | value |
|---|---|
| **exact wording** | On `meta-llama/Llama-3.1-8B`, which ships the `rope_scaling` dict the intervention modifies, stretching the rotary basis at 128k moves paired perplexity by **+0.05% (factor 16)** and **+1.56% (factor 32)** while speculative acceptance falls from **0.7272 (native)** to **0.5336 / 0.5006** — a **27-31% relative** loss. |
| **numbers** | ppl_ratio 1.0005 / 1.0156; acceptance 0.7272 / 0.5336 / 0.5006; tok/s 0.91087 / 0.76995 / 0.74733 |
| **n** | **5 books** per arm (native / f16 / f32), same five books as every other rung |
| **seeds** | 42 |
| **engine** | **old engine only.** No RoPE arm has been run under the candidate engine. |
| **precision** | FP4 weights + NF4 KV |
| **source** | `results/mlsys/followup_rope/`, stage ids `ROPE_SAMP_NATIVE_SPEC`, `ROPE_SAMP_F16_SPEC`, `ROPE_SAMP_F32_SPEC`, `ROPE_SAMP_NATIVE_TARGET`; table transcribed in `RESULTS_OCT.md` section 4 |
| **caveats** | (i) Only the 128k rung was measured; factor 32 is the largest tried. (ii) The three arms differ in **nothing** but `rope_factor` — same target and same revision, same draft, same book, seed, temperature, `spec_steps` and token budget — which is what makes the acceptance drop attributable to the rope. (iii) The direction holds on **5 of 5 books individually**, not only in the mean. (iv) At n=3 the two stretched factors were within 0.0012 of each other and that was offered as saturation; at n=5 they are 0.033 apart (6.6% relative), so **the specific 0.0012 is withdrawn** and the claim becomes "both stretched factors land far closer to each other than to native". (v) `pg19_train_2768` under f16 is the one row the repetition pass flags (77% of tokens inside a repeated 16-gram, no periodic tail). |
| **speedups** | 3.36x (native) / 2.84x (f16) / 2.76x (f32) **vs a n=1 target-only row**. These are **old-engine** numbers and their denominator is a single row. **Recommendation: keep them but label them old-engine and n=1, and do not put them in the abstract.** The finding is the acceptance/ppl contrast; the speedups are secondary, and under the candidate engine the acceptance contrast is unaffected (acceptance is unchanged by split-KV — see F4) while the speedups would be different. |

### F2 — Greedy decoding loops at long context and inflates acceptance

| field | value |
|---|---|
| **exact wording** | Under greedy decoding **9 of 15** long-context rows degenerate into repetition; under T=0.7 sampling **0 of 26** do. |
| **numbers** | greedy 9/15; sampled 0/26 (old engine) — extended by this run to **0 of 31** more sampled rows under the candidate engine, i.e. **0 of 57** sampled rows in total |
| **n** | 15 greedy, 26 sampled, plus 31 sampled candidate-engine rows |
| **seeds** | mixed; the greedy pool is `SA_TEMP_0_*` and `SA_TEMP_1_0_*` at 128k |
| **engine** | the 9/15 and 0/26 figures are **old engine**; the 0/31 extension is **candidate engine** |
| **precision** | FP4 weights + NF4 KV |
| **source** | `repetition_*.csv` in `results/mlsys/sampling_session/` (one per stage); the 31 new rows are `results/mlsys/rerun_r1|r2/session_a_*.csv` scored with `src/analysis/repetition.py`. Longest verbatim repeat span 8 tokens against a 128 ceiling; tokens-in-repeat 0.00 against a 0.60 ceiling. |
| **rule** | the campaign's **absolute** repetition rule, plus the periodicity detector's periodic-tail-share >= 0.50 threshold |
| **caveats** | The candidate-engine extension used 0.50 as the degeneracy threshold and found a **maximum periodic tail of 0.172** across the 31 rows, so nothing came close. It measures T=0.7 rows only — no greedy row has been run under the candidate engine. |

### F3 — The native-1M model stays coherent at 1M

| field | value |
|---|---|
| **exact wording** | The native-1M target stays coherent at 1M: within ~10% of its own 128k perplexity on the same tokens. |
| **numbers** | **1.805 (1M) vs 1.649 (128k)** on the one exactly aligned book — **+9.5%**; within **1.1%** on the two books sharing 91-95% of their scored tokens |
| **n** | 1 book exactly aligned, 2 books partially aligned |
| **seeds** | 42 |
| **engine** | **old engine** (prefill-only measurement — `measure_target_ppl` scores a prefill forward, so split-KV cannot touch it, and **the number is unchanged by the candidate engine**) |
| **precision** | FP4 weights + NF4 KV |
| **source** | `QUALITY_FIX_1M_*` vs `QUALITY_FIX_128K_*` (`ppl_tokens` 1024) + `results/mlsys/sampling_session/alignment.csv` |
| **caveats** | "Within ~10%" rests on **one** exactly-aligned book; the two partially-aligned books are tighter but share only 91-95% of their scored tokens, so they are weaker evidence. Say the n. Alignment rule is `continuation_sha256` + recorded offsets, never `generated_tokens_sha256`. |

### F4 — The ring's K/V rotation was the decode step

| field | value |
|---|---|
| **exact wording** | At 128k on 8x A100-80GB the decode step was dominated by the ring's per-token K/V rotation: **256 rounds per token** (8 rotations x 32 layers) at a **measured 10.3 ms per round that does not depend on payload** (10.3124 ms for 4 KB, 10.3462 ms for 266 MB — a 66,000x payload range, 0.4% variation, worst rank spread 0.056 ms over eight ranks). Removing the rotation with split-KV (prior work: Meta pass-Q/pass-KV, vLLM Decode Context Parallelism) is worth **2.8-5.3x**. |
| **numbers** | see the table below |
| **n** | 6 spec and 4 target-only rows at 128k; 3 spec and 1 target-only at 512k; 15 spec and 2 target-only at 1M; plus 3 seeds x 5 books at 1M |
| **seeds** | 42 in R1; 42 / 123 / 456 at 1M |
| **engine** | **candidate**, against the old engine measured in the same session or the same protocol |
| **precision** | FP4 weights + NF4 KV |
| **source** | `results/mlsys/rerun_r1/`, `results/mlsys/rerun_r2/`; diagnosis in `DECODE_DIAGNOSIS.md` section F; gate in `RERUN_DECISION.md`; tables in `RESULTS_OCT.md` section 10 |

| context | arm | candidate decode s/token | old decode s/token | speedup |
|---|---|---|---|---|
| 128k | spec | 0.5275 (3 matched books) | 1.4779 | **2.80x** |
| 128k | target-only | 1.0596 | 3.4925 | **3.30x** |
| 512k | spec | 0.9384 | 3.5227 | **3.75x** |
| 512k | target-only | 2.3460 | 9.9900 | **4.26x** |
| 1M | spec | 1.3830 (13 matched cells) | 6.0506 | **4.375x** |
| 1M | target-only | 3.2107 | 17.0902 | **5.32x** |

**Caveats.** (i) The speedup **grows with context**, which is what a context-independent
per-round cost predicts, and that is the mechanistic claim — not just "faster". (ii) Every row
records its engine: `ring_decode_mode: split_kv`, `ring_layers_patched: 32`, and the commit.
(iii) The 512k and 1M **target-only** denominators are n=1 and n=2 (Part 2 of this run tops them
up to n=3 and n=5 if capacity appears; otherwise the paper states the n). (iv) `kv_block_size:
32768` and `ring_expand_after_rotation: true` are in the config and are **inert** under split-KV;
their 1.240x and +2.7% belong to the ring path and **no claim is made for them here**. (v) Single
node, one region, A100-80GB SXM4 only.

### F5 — Speculative decoding still pays on top of split-KV

| field | value |
|---|---|
| **exact wording** | With the rotation removed, speculation remains worth **~2-2.5x** over the same engine's own target-only path. |
| **numbers** | decode-only **1.97x / 2.50x / 2.25x** at 128k / 512k / 1M; end-to-end **1.91x / 1.88x / 1.42x** |
| **n** | 6 spec vs 4 target-only (128k); 3 vs 1 (512k); 15 vs 2 (1M) |
| **seeds** | 42 (and 123/456 at 1M) |
| **engine** | **candidate** |
| **precision** | FP4 weights + NF4 KV |
| **source** | `RESULTS_OCT.md` section 10.2; `scripts/mlsys_rerun_report.py` |
| **caveats** | The **end-to-end** figures are much lower at 1M (1.42x) because prefill is 335 s of a 570 s row — with the decode step fixed, **1M prefill is now the long pole**. Quote decode-only for the speculation claim and say why the end-to-end number is smaller. |

---

## 3b. WHAT CHANGED FROM THE REVIEWED VERSION

Statuses from AUDIT E1-E23, mapped to **KEPT / CORRECTED (old -> new) / REMOVED**. The nine
items the brief names are marked **★**.

| # | claim in the reviewed version | status | old -> new |
|---|---|---|---|
| **E1 ★** | **"communication is <= 1.2% of wall on the speculator rank"** | **REMOVED** | The bucket recorded 0.84 / 4.39 / 10.57 s at 128k / 256k / 512k while the wire floor for the same runs is **2.7 / 13.4 / 28.4 s** — it recorded **0.31-0.37x** of the physical minimum, so it did not capture the transfers. Two dependent sentences go with it: "issued comm upper-bounds exposed comm, so <= 1.2% rules out overlap regardless of definition" and "agrees with theoretical NCCL bandwidth estimates". |
| E2 | "Idle >= 50% of rank-0 wall ... waiting on collective synchronisation" | **REMOVED** | The idle figure is 57-80% **profiler time spent outside `generate()`'s own timer**. |
| E3 | the tick-gate barrier | **REMOVED** | It is WRONG for the code that produced the M4 numbers. |
| E4 | §3.2's ring-exchange/overlap description | **CORRECTED** | The kernel issues the next rotation **before** each step's attention (`prefetch_depth` 1). What was never implemented is the paper's central overlap, **draft generation hidden behind ring traffic**: the draft and the verify run back to back. |
| E5 | positioning table and related-work framing | **CORRECTED** | CRediT-first positioning replaced; add the three split-KV prior-work citations (3b, below). |
| **E6 ★** | **the draft saw the last 4096 tokens of the prompt** | **CORRECTED** | **The draft was conditioned on the prompt's FIRST 4096 tokens** in every spec run whose prompt exceeded 4096 tokens (8k PG-19, all of 128k-1M, and the M3 64k grid): `d89cf41`. The paper says the opposite. This **confounds every long-context acceptance number**, including F1's. |
| **E7 ★** | the YaRN anchor | **CORRECTED** | Every >= 8k cell was anchored on **its own context length**, not 4096, because transformers <= 4.50 ignores `original_max_position_embeddings`. Disclosed nowhere in the paper. |
| **E8 ★** | the PPL table's "YaRN" rows | **CORRECTED** | All used **factor 8**, not `ceil(N/4096)`. At 4k the engine applies **no YaRN at all**. |
| E9 | the 4k/8k control's rope state | **CORRECTED** | state it explicitly |
| E10 | verify-loop defects | **CORRECTED** | per AUDIT E10 |
| E11 | "3 seeds" | **CORRECTED** | The three seeds used **one synthetic prompt**, so they are not three independent samples of a distribution. |
| E12 | confidence intervals | **CORRECTED** | The CIs use **+/-1.96 SEM at n=3**; report the range instead, or state the n and that the interval is not meaningful at that n. |
| E13 | nominal context | **CORRECTED** | The "1M" cells were **1,002,984 tokens**, not 1,048,576. |
| **E14 ★** | **"the speedup regresses to 0.94x at 512k and 0.97x at 1M"** | **REMOVED** | Not an acceptance effect and **not reproduced by the paper's own re-run**. Decode tokens per verify round are flat 256k -> 1M (1.80 / 1.69 / 1.71); the regression was per-round **cost** in one pod (a spec round cost 1.8x a target-only step there vs 0.97x at 128k/256k), and the same-code re-run gives 1.61 / 1.90 / 1.53 / 1.56x. Replace with the F4/F5 tables, which are measured against in-session baselines. |
| E15 | "a short-context PG-19 control attributes the bimodality to base-model YaRN extrapolation" | **CORRECTED** | CONFOUNDED. |
| **E16 ★** | **the 4.4x claim (n=1)** | **REMOVED as stated** | Both 4k rows are clean under the periodicity detector, but it is **one pair from one session**, its decode-only ratio (6.1x) exceeds the 3.82 tokens per round speculation can deliver (a whole spec round cost 0.62x of one target-only step), and as a control for "bimodality is YaRN" it differs from the long-context cells in at least five ways at once. |
| E17 | the speedup formula | **CORRECTED** | WRONG application. Use `tokens_generated / time_sec` for end-to-end and `(time_sec - ttft) / (tokens - 1)` for decode-only, and say which is which. |
| **E18 ★** | **peak memory** | **CORRECTED** | The reported peak is **rank 0 only**. Allocated max over ranks is **41.2 GiB** and **reserved** is **69 GiB**, not "half of 80". |
| **E19 ★** | **"NF4 weights"** | **CORRECTED** | The weights are **FP4**, not NF4. (NF4 is the **KV** codec.) |
| E20 | NF4 KV description | **CORRECTED** | WRONG in three places, per AUDIT E20. |
| E21 | §3.6 design-point sweeps | **CORRECTED** | per AUDIT E21 |
| E22 | prompts and citations | **CORRECTED** | include **★ the cited test that does not exist** — remove it or cite something that does. |
| E23 | qualitative table (§5.7) | **KEPT** | stands, with one caveat (AUDIT E23) |

**Cannot classify (S4): none.** Every claim the audit covers is listed above. Two audit items are
about the *audit's own* earlier documents rather than the manuscript — E5's CRediT note and the
"corrections this audit found in `BANDWIDTH_FLOOR.md` / `PROFILE.md`" block — and are carried as
corrections to those files, not as manuscript claims.

### 3b-bis. Prior work that must be cited and never claimed

The split-KV technique is **prior work**. Cite it for the technique; the contribution here is the
diagnosis and the measurement.

| work | what it is | bib entry |
|---|---|---|
| Meta, "Context Parallelism for Scalable Million-Token Inference" | pass-KV / pass-Q ring-context parallelism | `% TODO VERIFY` — I can verify the title and the technique; **not** the venue, year, pages or DOI. Write the entry with author list, title and technique, and mark the rest `% TODO VERIFY`. |
| vLLM, "Decode Context Parallelism" (engineering blog, Aug 2026) | DCP in vLLM | `% TODO VERIFY` — blog, so cite as a blog with URL and access date; **do not invent a venue**. |
| NVIDIA, "Co-Designing AI Model Attention for Fast, Interactive Long-Context Inference" | attention co-design for long context | `% TODO VERIFY` — blog/technical report; verify author list and date. |

**No bibliographic field beyond the title and the technique is asserted here.** Fill each entry
only from a source you have opened.

---

## 3c. SUMMARY OF CHANGES (to send the organizers)

> This revision changes one thing and records another. The change: the reviewed version's
> systems section reported that ring communication was <= 1.2% of wall time and that the
> speedup regressed to 0.94x/0.97x at 512k/1M. Both are withdrawn. A bare-process benchmark of
> the exact ring primitive shows each rotation round costs a fixed ~10.3 ms **regardless of
> payload** (4 KB and 266 MB are within 0.4% of each other), so the cost is latency per round,
> not bandwidth; the profiler bucket that produced the 1.2% figure recorded less time than the
> link physically needs to move those bytes, so it did not capture the transfers. The
> regression does not reproduce on a same-code re-run. Harnessing the diagnosis, removing the
> rotation with split-KV (prior work: Meta pass-Q, vLLM DCP) yields 2.8-5.3x decode-only and the
> speedup grows with context. The recording: a configuration bug meant the two long-context
> "seed 123/456" stages had re-run seed 42, because a level's `seeds:` key is overwritten by the
> defaults' during config merge. Rebuilt from unaffected rows, the 1M speedup is 4.375x (not
> 4.033x) and the 1M acceptance is **unchanged** (not -10.4%). Two further claims are withdrawn
> or corrected: the draft saw the prompt's *first* 4096 tokens, not its last, which confounds the
> long-context acceptance numbers; and the cited test does not exist. Perplexity findings
> (RoPE acceptance loss, 1M coherence) are unaffected.

---

## 3d. vLLM COMPARISON, recomputed

Setup differences that must appear **in the same sentence as every ratio**: vLLM 0.6.3, TP=8,
**bf16 weights and bf16 KV, greedy decoding (T=0.0)**; RASD **FP4 weights and NF4 KV, T=0.7
sampling with speculative decoding**. Prompts are matched (`unit_matched=yes`: vLLM was fed the
sidecars' `engine_input_ids`). **This is not a precision-matched comparison and must not be
called one.**

| context | vLLM e2e tok/s (bf16, greedy) | candidate spec | candidate target-only | old-engine spec | candidate / vLLM | old / vLLM |
|---|---|---|---|---|---|---|
| 128k | 6.9865 | 1.6064 | 0.8435 | 0.67350 | **0.23x** | 0.096x |
| 512k | 0.56986 | 0.6208 | 0.3312 | 0.24389 | **1.09x** | 0.43x |
| 1M | 0.14828 | 0.2460 | 0.1731 | 0.12077 | **1.66x** | 0.81x |

All rates are end-to-end (`tokens_generated / time_sec`, prefill included), because **vLLM's
first-token time is unusable**: vLLM 0.6.3 reports `RequestMetrics` timestamps as absolute
monotonic clock readings and the recorder stored `1.79e9` for `ttft_s` and `-1.79e9` for
`decode_only_wall_s`. **A vLLM decode-per-token figure therefore cannot be computed from the
recorded data, and this kit does not invent one.** What can be compared is the end-to-end wall
for the same 128 tokens at the same prompt, which is what the table above does.

**Prefill, and it is unchanged by split-KV** (split-KV is a decode-path switch; the candidate
engine's 1M prefill measures 335 s against the old engine's 344 s, a 2.6% difference):

| context | vLLM **total wall** | RASD **prefill only** (`ttft_ms`) | vLLM total / RASD prefill |
|---|---|---|---|
| 128k | 18.321 s | 8.589 s | 2.13x |
| 512k | 224.615 s | 91.447 s | 2.46x |
| 1M | 863.576 s | 339.284 s | 2.55x |

So vLLM's *entire run* — its own prefill plus 128 decode steps — takes longer than RASD's
prefill alone at every context, which means vLLM's prefill is the slower of the two and it is
the opposite of the decode ordering. The decode ordering flips with the candidate engine:
RASD overtakes vLLM at 512k (**1.09x**) and 1M (**1.66x**) and remains behind at 128k (0.23x).

**Peak memory per GPU.** vLLM recorded 68,798.7 / 68,798.8 / 68,799.0 MB — flat across contexts,
which is the shape a fixed `gpu_memory_utilization` preallocation gives, so it is the **KV pool
size and not the working set**. RASD's `gpu_peak_mem_mb` on the same cells is 8,256 / 15,781 /
25,814 MB, and it **does** rise with context. The two are not the same quantity, so they are
reported side by side and **no ratio is claimed**.

---

## 3e. LIMITATIONS

1. **n per cell.** 128k spec n=6 and target n=4; 512k spec n=3 and **target n=1**; 1M spec n=15
   (3 seeds x 5 books) and **target n=2**. The 512k and 1M target-only figures are the
   **denominators** of the 4.26x and 5.32x speedups. Part 2 of this run tops them up to n=3 and
   n=5 if 8x capacity appears before Wed 10:00 ET; if it does not, **say n=1 and n=2 in the
   paper**.
2. **Single node, one region.** All measured rows are one 8x A100-80GB SXM4 node. No
   multi-node result exists, and the ring's per-round latency would change with a network.
3. **FP4/NF4 only — no bf16 row exists.** The campaign took branch C, so every RASD row here is
   4-bit weights with an NF4 KV codec. There is no same-protocol bf16 measurement of the
   candidate engine, which is exactly why the vLLM comparison in 3d is **not**
   precision-matched. A bf16 rerun was planned and never authorised.
4. **One target model family.** `gradientai/Llama-3-8B-Instruct-Gradient-1048k` (plus
   `meta-llama/Llama-3.1-8B` for the RoPE arms, which is a different model on purpose). Nothing
   here establishes that the technique or the acceptance behaviour transfers.
5. **PG-19 only**, five books at 1M. The document is the unit of independence; the book-to-book
   spread (2.3x in decode rate, 0.30-0.68 in acceptance at 1M) is larger than the seed-to-seed
   spread (1.04x), so "n books" and "n seeds" are not interchangeable.
6. **Acceptance under split-KV is explained but not tested.** The pooled 1M acceptance is
   unchanged (-0.1% over 13 matched cells) and per seed it moves both ways (-15.7% / +10.9% /
   +11.3%) inside an old-engine seed spread that is itself 23.0% of the mean. The mechanism
   offered — one flipped near-tie changing the emitted prefix, and with it every later draft
   proposal — is consistent with every measurement but was not tested directly; the test would
   hold the emitted prefix fixed and re-measure the decision.
7. **1M prefill is now the long pole.** A candidate-engine 1M row is 570 s of which **335 s is
   prefill**, so the end-to-end spec speedup at 1M is 1.42x even though decode-only is 2.25x.
   Any claim about 1M throughput has to say which of the two it means.
8. **The unattributed 19.3% elementwise term.** In the profiled 128k decode window, 19.3% of
   device time was elementwise kernels (casts, dequant, softmax bookkeeping) that were never
   attributed to a specific operation. With the rotation gone this is a larger share of what
   remains, and it is the most obvious next optimisation.
9. **Two withdrawn claims are recorded rather than deleted**, because a reader of an earlier
   version needs to know they are gone: the "comm <= 1.2%" result (3b, E1) and the 4.4x n=1 pair
   (E16). A third, the "identical across seeds" reading, was that same configuration bug.

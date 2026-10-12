# VLLM_PLAN.md — the vLLM comparison: what it is, what it costs, and the GO/NO-GO

Written in Phase 5. **No 8x vLLM run was launched.** This document is the plan and
the decision, with the 1x rehearsal's measured result behind it.

---

## 1. What the comparison is, and what it is not

**The minimal useful comparison**, per the handoff: target-only greedy decode
throughput, same model (`gradientai/Llama-3-8B-Instruct-Gradient-1048k`), same
**aligned prompts** (the exact token ids the RASD target was fed, BOS included),
at 128k / 512k / 1M, 128 new tokens, on the same 8x A100 80GB SXM4 type as the
RASD rows, in vLLM 0.6.3 at TP=8.

It answers one question the paper currently cannot: **how far is RASD from a
production engine on the same prompts?** Every speedup in
`results/mlsys/RESULTS_OCT.md` is against RASD's own target-only path, which the
profile shows is itself ~28x above its own bandwidth floor. A reader cannot tell
whether "2.37x" is a good engine or a slow one without an external reference.

**What it is not.** It is one number, not a serving benchmark:

* **No speculation on the vLLM side.** RASD's number includes a 1B draft; vLLM's
  is plain target-only decode. The comparison is therefore of the *target-only*
  paths, which is the like-for-like part, and no acceptance is compared.
* **The precision differs and cannot be matched in 0.6.3.** RASD runs FP4 weights
  and an **NF4** KV cache (40 KiB/token, total); vLLM 0.6.3 has no NF4 KV path, so
  its KV is **bf16, 128 KiB/token — 3.2x RASD's**, and its weights are bf16
  (15.0 GiB against RASD's ~4.2 GiB NF4). It is therefore a *conservative*
  comparator for RASD's memory advantage and an *unmatched-arithmetic* comparator
  for speed. Both are recorded per row (`kv_dtype`, `weight_precision`).
* **No batching, no continuous batching, one sequence.** `max_num_seqs: 1`. It
  measures single-stream latency, which is what RASD's rows measure.
* It is not a RASD result and must never be quoted as one.

## 2. Memory per GPU at 128k / 512k / 1M with TP=8 (handoff item 2)

Llama-3-8B geometry (`n_layers` 32, `n_kv_heads` 8, `head_dim` 128, `hidden`
4096, `intermediate` 14336), bf16 weights and bf16 KV, 80 GB cards, TP=8.

| term | per GPU |
|---|---|
| weights, bf16, 8.03e9 params = 14.96 GiB sharded over TP=8 | **1.87 GiB** |
| KV, bf16, 128 KiB/token total = 16 KiB/token/GPU | 2.00 GiB @128k, **8.00** @512k, **16.00** @1M |
| vLLM overhead (CUDA graphs, NCCL/workspace, non-torch) | ~2-3 GiB |
| **subtotal, chunked prefill (16384 tokens/chunk)** | **6.9 / 12.9 / 20.9 GiB** |
| budget at `gpu_memory_utilization` 0.90 / 0.95 of 80 GB | **67.1 / 70.8 GiB** |
| **verdict** | **all three FIT**, with room |

**THE ONE THING THAT WOULD NOT FIT, AND IT USED TO BE WHAT THE LADDER ASKED
FOR.** With the prompt preprocessed in a single forward (i.e. no chunking), the
activation tensors scale with the whole context:

| context | hidden states | MLP intermediate (2 x 14336) | total with subtotal above |
|---|---|---|---|
| 128k | 1.00 GiB | 7.00 GiB | 14.9 GiB — fits |
| 512k | 4.00 GiB | 28.00 GiB | 40.9 GiB — fits |
| **1M** | **8.00 GiB** | **56.00 GiB** | **76.9 GiB — DOES NOT FIT** (budget 70.8) |

So: **128k and 512k fit either way; the 1M rung fits only with chunked prefill.**
The ladder's long-context attempts used to pass `max_num_batched_tokens = ctx`
while their note said "chunked prefill sized to ctx" — a chunk equal to the
context is no chunking — so the first 8x run would have been set up to OOM the one
rung the comparison most needs. Fixed in this phase: both attempts now set
`enable_chunked_prefill` with `CHUNKED_PREFILL_TOKENS = 16384`, which divides
128k, 512k and 1M exactly and keeps peak activations at ~1.0 GiB
(`tests/test_mlsys_vllm_firstrun_fixes.py`).

`max_num_seqs: 1` is equally load-bearing. vLLM allocates KV for
`max_num_seqs x max_model_len`; at the default 256 sequences and 1M tokens the
requirement is ~4 TiB, which vLLM refuses rather than allocates. With 1 sequence
it is 16 GiB — the number in the table.

**Also flagged:** at 512k and 1M the *prefill* is 523,263 / 1,047,551 tokens even
though only 128 tokens are decoded, so the wall time of a cell is dominated by
prefill, not by the thing being measured. That is true of the RASD rows too
(`ttft` 9.4 s at 128k on 8 ranks), which is why the headline comparison uses
**end-to-end** tok/s on both sides and decode-only is reported separately, only
where each engine can supply its own TTFT.

## 3. Exact commands

### 3.1 What has to be staged first — and why the committed sidecars are not enough

The stage takes its prompts from RASD token sidecars and demands
`engine_input_ids`: **the ids the target was actually fed, BOS included**.
`prompt_token_ids` is the same prompt *without* the BOS, and replaying it into
vLLM would replay a sequence the target never saw (one token at the front changes
every position's rotary phase).

`results/mlsys/sampling_session/tokens_slim/*.json` **deliberately do not carry
`engine_input_ids`** — that field is the whole prompt, 130k-1M ids per row, 169 MB
across the session — so pointing the stage at them yields **zero** usable cells.
That is checked, not assumed: it fails immediately at the CLI with
`--prompt-ids-from-sidecars: no usable sidecar in <dir> ... none carries
engine_input_ids` (asserted in `tests/test_mlsys_vllm_firstrun_fixes.py`), so the
discovery costs seconds here instead of a full 8x run.

**So the three target-only sidecars the comparison needs must be staged from the
campaign's session workspace, not from the repository:**

```bash
# On the Mac. These are the ORIGINAL sidecars (prompt_token_ids + engine_input_ids),
# one per rung, for the one book used as the aligned prompt. ~25 MB total, because
# the 1M prompt alone is 2 x 1,047,551 ids.
PULL="$HOME/.copilot/session-state/0f919471-cdca-48c0-a994-56a95eb1d6be/files/followup_samp2_pull/tokens"
mkdir -p /tmp/vllm_prompts
for r in SAMP_ALIGNED_128K_TARGET_pg19_train_0_s42 \
         SAMP_ALIGNED_512K_TARGET_pg19_train_0_s42 \
         SAMP_1M_TARGET_pg19_train_0_s42 ; do
  cp "$PULL/$r.json" /tmp/vllm_prompts/
done
du -sh /tmp/vllm_prompts           # expect ~25 MB
# then, after provisioning the 8x pod:
rsync -az --no-perms -e "ssh $SSH_OPTS" /tmp/vllm_prompts/ \
  ubuntu@$IP:~/RASD/results/mlsys/vllm_prompts/
```

### 3.2 On the pod (after `scripts/mlsys_setup_probe.sh` has provisioned it)

```bash
export HF_TOKEN=$(cat ~/.hf_token)              # gradientai/... and meta-llama/... are gated
bash scripts/mlsys_vllm_venv.sh                 # the ONLY sanctioned install path: torch 2.4.0+cu121,
                                                # transformers==4.46.3, vllm==0.6.3
VLLM_PY=~/RASD/.venv-vllm/bin/python

MLSYS_VLLM_PYTHON="$VLLM_PY" "$VLLM_PY" scripts/mlsys_vllm_baseline.py \
  --models gradientai/Llama-3-8B-Instruct-Gradient-1048k \
  --context-lengths 131072 524288 1048576 \
  --max-new-tokens 128 --matched-max-new-tokens 128 \
  --tensor-parallel-size 8 \
  --target-revision cd3069b65a8eb13da639d332a5f61b0fbb29fa73 \
  --prompt-ids-from-sidecars results/mlsys/vllm_prompts \
  --out results/mlsys/vllm_baseline.csv
```

Notes on the flags, each of which is load-bearing:

* `--target-revision` is now **passed to the load**, not merely recorded (fixed in
  this phase: it previously reached the CSV row and `_unit_match` but neither
  `LLM(...)` nor `AutoTokenizer.from_pretrained(...)`, so a row could claim a pin
  while the engine served `main`).
* **The install itself is two pins, not one, and the first rehearsal proved it.**
  vLLM 0.6.3 declares `transformers>=4.45.0` with **no upper bound**, so pip
  resolved **5.19.0** on 2026-10-12 — and the repo's own `requirements.txt` pins
  `transformers==5.5.4` for the analysis env, so "latest" is exactly the wrong
  major version here. transformers 5.x cannot see torch 2.4.0 ("PyTorch was not
  found. Models won't be available..."), which surfaced as a vLLM load error about
  torch. `mlsys_vllm_venv.sh` now installs `transformers==4.46.3` **before** vllm
  so the resolver cannot upgrade it, then asserts the installed versions and fails
  the install with the package named if either pin did not take. 4.46.3 is inside
  the GPU env's own range (`environment_gpu.yml`: `transformers>=4.46,<4.48`) and
  is the version the provisioned pod's campaign env actually reported.
* **The version gate:** `vllm.__version__` reads **`'dev'`** on that pod (vLLM's
  version module degrades when it cannot read its build; the same log carried
  "Failed to read commit hash"). The worker consequently refused a correctly
  pinned install on every attempt. It now verifies the pin against *every* identity
  the install exposes, preferring the distribution metadata, and records which
  source agreed in `vllm_version_source`. A wrong release is still refused.
* `--tensor-parallel-size 8` is required for `unit_matched=yes`; anything else is
  recorded but not certified.
* `rope`: `DEFAULT_ROPE` has **no entry** for Gradient-1048k, and that is correct —
  the model ships `rope_scaling: None` with `rope_theta` 3.58e9 and
  `max_position_embeddings` 1048576, so vLLM reads them from `config.json` and no
  override is applied. `hf_overrides` is only used for the Llama-2 YaRN arm.
* `--quantizations` is **not** passed: the first run is bf16. A
  `bitsandbytes` row would need `bitsandbytes` in the venv, which
  `scripts/mlsys_vllm_venv.sh` does not install — deliberately out of scope for
  the first run rather than an unnoticed gap. `kv_dtype` is bf16 because 0.6.3 has
  no other option.

**How decode tok/s is measured on the vLLM side**, because the comparison depends
on it: the worker times `llm.generate(...)` end to end (`end_to_end_wall_s`),
counts the emitted tokens (`output_tokens`) and derives

```
throughput_tps_end_to_end = output_tokens / end_to_end_wall_s
throughput_tps_decode_only = output_tokens / (end_to_end_wall_s - ttft_s)
```

where `ttft_s` is read from vLLM's own `RequestOutput.metrics.first_token_time`.
**In offline `generate()` that field is usually `None`**, and the worker records
`decode_only_wall_s` as empty rather than inventing a value — so the paired
comparison uses **end-to-end**, which is also what RASD's `time_sec` is
(`tokens_generated / time_sec`, prefill included). If a decode-only figure is
wanted for a specific row it has to come from a server-mode run, which is not part
of this comparison.

### 3.3 Driving it on the Mac

`scripts/mlsys_profile_1x.sh` is the driver for a paid single-box run and is
parameterised; `scripts/mlsys_vllm_remote.sh` is the pod side and is the SAME
script the rehearsal ran, with four env vars changed — that is the whole
difference between the rehearsal and the comparison:

```bash
cd /Users/amankesarwani/PycharmProjects/RASD
SESSION_TYPE=gpu_8x_a100_80gb_sxm4 SESSION_RATE=22.32 \
REMOTE_SCRIPT=scripts/mlsys_vllm_remote.sh \
PROFILE_OUT=results/mlsys/vllm_run PROFILE_TAG=vllm8x \
TIMER_MIN=120 \
REHEARSE_TP=8 \
REHEARSE_CTXS='131072 524288 1048576' \
REHEARSE_PROMPT_DIR=results/mlsys/vllm_prompts \
bash scripts/mlsys_profile_1x.sh
```

The driver refuses to launch while any instance exists, arms a **120-minute
detached dead-man timer and proves it by PPID 1 before any work starts**, writes
the HF token over stdin into a 600-mode file, launches the pod script detached
with its own `DONE` marker, and terminates with the instance count re-read from
the API. `REHEARSE_PROMPT_DIR` is staged into `~/RASD/results/mlsys/vllm_prompts`
by §3.1; a path with spaces is refused rather than mis-split.

## 4. Expected hours and cost

| stage of the run | estimate | basis |
|---|---|---|
| provisioning + repo/pool staging | 15 min | measured: 6.5 min on the Phase 3 profile, 3.5 min here |
| `mlsys_vllm_venv.sh` (torch 2.4.0+cu121 ~2.5 GB, vllm 0.6.3 + deps) | 10-20 min | one-time per instance |
| model load, 15 GiB bf16 from the Hub cache/network, per attempt | 3-6 min | 8B bf16 |
| 3 cells: prefill (131k / 523k / 1048k tokens, chunked 16384) + 128 decode tokens | 15-30 min | prefill-bound; see §2 |
| **total** | **1.0-1.5 h** | at $22.32/hr = **$22-34** |
| worst case, if the 1M rung burns all three ladder attempts | 2.0 h | **$45** |

**Budget check:** ~$328 spent of the $450 cap. This run at $22-34 (or $45 worst
case) plus the rehearsal's cost keeps the campaign inside the cap with margin;
**nothing else in this plan spends after it.**

## 5. The 1x rehearsal (handoff item 4) — PASSED, on the second attempt

**Attempt 1** (`gpu_1x_a100_sxm4`, us-east-1, 09:56-10:03Z, **$0.35**, 0 instances
afterward): installed fine, then produced **3 error rows and no measurement**. It
found the two blockers above — `transformers` 5.19.0 resolved by pip, and
`vllm.__version__ == 'dev'` tripping the pin gate. Both were fixed with tests, and
both attempts' CSVs are committed as evidence
(`results/mlsys/vllm_rehearsal/vllm_baseline_attempt1_failed.csv` — the failing one
— and `vllm_baseline.csv`). The per-attempt logs
(`rehearsal_attempt1.log`, `rehearsal.log`, `remote_stdout.log`) sit beside them in
the local working tree but are **not** committed: `.gitignore` excludes `*.log`
repo-wide, as it does for every other stage. Quoting the log is still legitimate —
it is the file the row's `log_path` column names — but it is not a committed blob.

**Attempt 2** (`gpu_1x_a100_sxm4`, asia-south-1, 10:10:41Z launch → 10:25:54Z DONE,
wall 0.31 h = **$0.61**, 90-minute dead-man timer confirmed PPID 1, **0 instances**
after terminate verified by the API, timer cancelled):

| what the rehearsal asked | what happened |
|---|---|
| install vLLM exactly as the 8x run would | **rc=0**: `torch 2.4.0+cu121`, `vllm 0.6.3`, `transformers 4.46.3` — the assertions the install now performs all passed |
| load Gradient-1048k | loaded at `max_model_len` 8192 on **the first ladder attempt**, no fallback needed; revision pinned to `cd3069b6...` |
| decode 128 tokens | **128 tokens in 2.5807 s end-to-end = 49.5996 tok/s** (`end_to_end_s` 2.5807, `output_tokens` 128) |
| ran cleanly? | `status=ok`, no error, `attempt=1`, `config_used=1-default-tp8` |
| `unit_matched` | **`no`, as designed** — it requires TP=8 and the RASD prompt ids, and this was one card with a synthetic prompt |

The wrapper's own verdict: `rows=1 ok=1`, `VLLM RUN DONE`. Note that the baseline
script's exit status was **1** (its stage-level rule records FAILED when no row is
unit-matched), which is exactly why the wrapper judges itself on the CSV instead.

**A third defect, found in the successful row itself and fixed immediately.**
Its `weight_precision` column read **`int4_bnb`** while `kv_dtype` on the same row
read `bfloat16` — which cannot both be true. Cause: `--quantizations` **defaults to
`["bfloat16"]`**, and the worker tested that label for *truthiness* when deciding
whether to report the un-introspectable bitsandbytes path. So every bf16 row was
labelled as 4-bit weights, in the column the cross-check prints beside RASD's `fp4`.
Fixed to key on `quant == "bitsandbytes"`, with the bnb path kept as the control
test. **The committed rehearsal CSV still shows the wrong label** — it is the
artifact the buggy version produced, and it is kept rather than edited; the row's
`kv_dtype=bfloat16`, its attempt log name and the load path all agree the weights
were bf16. Rows produced by the 8x run will carry the fix.

**What the rehearsal therefore establishes:** the install path, the tokenizer, the
`rope_theta` / `max_position_embeddings` handling, the model load and the decode
path all work on a real pod with the pinned versions. **What it does not**: anything
about 8-GPU behaviour, the `gradientai` prompt ids, 128k/512k/1M memory, or whether
the ladder's long-context attempts load — those need the 8x run.

The 49.60 tok/s is **not** a comparison number: 8k context, TP=1, a synthetic
prompt. It is one data point that the engine runs at the right order of magnitude.

## 5b. The first 8x attempt failed on the DRIVER, and the retry

**Attempt 1** (`gpu_8x_a100_80gb_sxm4`, us-east-1, launched 10:40:49Z, terminated
~10:55:30Z, **wall 0.245 h = $5.47**, 0 instances afterward, timer killed by hand
because the driver itself was reaped mid-trap): **zero rows at 131072, 524288 or
1048576.** The CSV holds exactly ONE row, at **context 8192** —
`max_model_len 8192`, `TP 1`, `prompt_source synthetic_same_paragraph_NOT_rasd_ids`,
`doc_id` empty, `status ok` at 54.92 tok/s, and `unit_matched=no` for
`prompt token ids are not the RASD ids (C2); tensor_parallel_size != 8 (C5); no
doc_id`. Log: `### running the baseline: ... @ [8192] tokens, TP=1, 128 new tokens`
and `prompt: synthetic, NOT the RASD ids`.

**WHY.** `REHEARSE_TP=8`, `REHEARSE_CTXS='131072 524288 1048576'` and
`REHEARSE_PROMPT_DIR=samp_prompts` were set on the Mac, but the driver built its
remote command as `env PROFILE_OUT=... PROFILE_TAG=... PROFILE_NPROC=...` and nothing
else crossed the boundary. The pod therefore started the wrapper with **its own
defaults**: one card, 8192 tokens, a synthetic prompt — on an 8-GPU box.

**WHY THE TESTS DID NOT CATCH IT.** `tests/test_mlsys_vllm_rehearsal.py` invokes the
WRAPPER directly. It therefore proved the wrapper forwards what it is GIVEN, and
never that the driver gave it anything: the gap was the boundary itself. This is the
campaign's own L12 ("testing the function is not calling it") expiring for a third
time, now at $5.47.

**THE FIX.** `REMOTE_ENV_NAMES` — an explicit, documented allowlist of variable names
the driver forwards into the remote `env` call, each single-quoted so a value with
spaces survives. Validated in the preflight, before the instance count and before
anything is launched: a named variable that is unset or EMPTY is refused (forwarding
`''` lets the pod fall back to its own default, which is the shape of this exact
mistake) and a value containing a single quote is refused as unsafe to quote.

**THE TEST NOW SPANS THE BOUNDARY.**
`tests/test_mlsys_profile_1x_remote_env.py` drives the REAL driver with stub `curl`,
`ssh` and `rsync` on PATH and asserts on the argv actually handed to ssh — the bytes
that would have gone to the pod, not the driver's own log line about them. Three
cases: the named vars are present; without the allowlist they are absent (that IS the
old behaviour, so it is the control); a named-but-unset variable is refused before
any ssh call. The driver also logs the forwarded names with values masked.

**An unrelated defect found in the same window, and now fixed:** the driver's three
result paths named `results/mlsys/session_a/` literally instead of `$SESSION_OUT`, so
the 1x n=5 rehearsal pulled only the committed Oct-10 leftovers that the repo rsync
had copied onto the pod, while its real stage CSVs stayed on the pod and died with
it. Fixed, with a test that asserts on the rsync argv; at $1.99/hr this was cheap to
find and at $22.32/hr it would have thrown away a ~$30 result.

**Attempt 2 — the retry, ONCE, on the 8x, 2-hour timer.** Same command with
`REMOTE_ENV_NAMES="REHEARSE_TP REHEARSE_CTXS REHEARSE_PROMPT_DIR"`,
`REHEARSE_TP=8`, `REHEARSE_CTXS='131072 524288 1048576'`,
`REHEARSE_PROMPT_DIR=samp_prompts`, `TIMER_MIN=120` ($44.64, inside the $60 ceiling).
Launched 11:13:44Z in us-midwest-1 with the 120-minute dead-man timer confirmed by
**PPID 1**; the payload staged 4/4 files and the driver logged the forwarded names, so
the boundary that failed last time is visibly crossed this time.

### 5c. The TTFT defect the retry exposed, and the fix (no re-run)
`_ttft_from_metrics()` returned `metrics.first_token_time` directly, on the
assumption that offline `generate()` exposes no TTFT at all. vLLM 0.6.3 does expose
`RequestMetrics`, but its fields are **absolute monotonic clock readings**, not
durations: at 1M `first_token_time - arrival_time` is the real prefill time, while
`first_token_time` alone is 1.79e9. `end_to_end - ttft` therefore came out as
-1.79e9 s and `throughput_tps_decode_only` as **1.27e11 tok/s** in a column that
reads as a measurement. A number that cannot be true is worse than a blank.

The fix makes the function return a *duration or nothing*: subtract `arrival_time`,
and accept the result only if it is non-negative and no longer than the request that
produced it (the caller now passes `end_to_end`); otherwise return None and let the
row write a blank. The warning text says which of the two happened.

**No re-run.** The affected columns are exactly the ones this campaign does not need:
the comparison quantity is end-to-end (`throughput_tps_end_to_end`), which is
unaffected and was never derived from the metric. Spending ~$20 to refill three
decorative columns would be the wrong trade, so the four rows keep their blanks —
and the CSV's three unusable values are named here rather than quietly quoted later.

### Verdict on the retry: **PASSED — 4/4 rows ok, at all three contexts.** (judged from the CSV)

Judged from `vllm_baseline.csv` (4 rows, `status=ok`, `unit_matched=yes` on every
row), never from rc. Pod timeline from `remote_stdout.log`: venv r=0 at 11:26:52Z,
baseline started 11:26:52Z, `DONE 2026-10-12T12:04:52Z 0`, terminated 12:08:37Z.

| context | doc | prompt_tokens | out | end-to-end s | **tok/s** | unit_matched |
|---|---|---|---|---|---|---|
| 131072 | pg19_train_0 | 130048 | 128 | 18.3210 | **6.9865** | yes |
| 524288 | pg19_train_0 | 523264 | 128 | 224.6151 | **0.5699** | yes |
| 1048576 | pg19_train_0 | 1047552 | 128 | 863.5764 | **0.1482** | yes |
| 1048576 | pg19_train_1 | 1047552 | 128 | 862.9615 | **0.1483** | yes |

- **The three conditions of the GO all held.** (1) `prompt_ids_from_engine=yes` and
  `prompt_ids_verified=yes` on all four rows, with the same sha256 the RASD sidecars
  carry (`eaf3abc1…` at 128k, `d22068eb…` at 512k, `c9b20fa4…`/`ec40d6c5…` at 1M) —
  the staged ORIGINAL sidecars with `engine_input_ids`, 4 cells found
  (`2 documents x 3 context(s)`). (2) TP=8 in every row, `1-default-tp8`, i.e.
  `max_num_seqs: 1` + chunked prefill (`enable_chunked_prefill`, 16384). (3) The 1M
  cell ran first among the long rungs and **succeeded on its first attempt**, ~900 s.
- **The unit matches.** Same target revision `cd3069b6…`, same 128 tokens, same
  doc_ids, `max_model_len == ctx`, `temperature 0.0`: these are the same prompts the
  RASD target-only rungs used, so the column in `RESULTS_OCT.md` is like for like.
- **Cost: wall 3285 s = 0.91 h = $20.37** (inside the $44.64 the 2-hour timer allowed,
  and half the estimated worst case). 0 instances afterward, timer cancelled by the
  driver. Cumulative campaign spend after this run: **~$341 of $450**.
- **A known defect in the CSV, recorded rather than hidden.** `ttft_s`,
  `decode_only_wall_s` and `throughput_tps_decode_only` are **unusable for this
  offline path**: vLLM 0.6.3's `RequestOutput.metrics` is NOT None (the docstring in
  `mlsys_vllm_baseline.py` assumed it would be), and `first_token_time` is an
  absolute monotonic clock reading, not a duration, so `end_to_end - ttft` came out
  as ≈ -1.79e9 and the decode rate as 1.27e11. Only the end-to-end columns are quoted
  above; the fix (treating the metric as `first_token_time - arrival_time`, and
  writing `<NA>` rather than a number when either is missing) is in §5c. No re-run is
  justified: the end-to-end number, which is the comparable one, is unaffected.

## 6. How it will be reported next to RASD

One table in `results/mlsys/RESULTS_OCT.md`, beside — never inside — the RASD
numbers. **The numbers below are the measured ones** (retry of 2026-10-12, §5b);
every tok/s is `tokens_generated / time_sec`, never the rounded `throughput_tps`.

### 6.1 Means over each arm's own rows (all arms on the same aligned window, TP=8)

| context | vLLM target-only (n) | RASD target-only (n) | RASD spec T=0.7 (n) | vLLM / RASD target-only | vLLM / RASD spec |
|---|---|---|---|---|---|
| 128k | **6.9865** (1) | 0.28310 (1) | 0.67350 (3) | **24.68x** | 10.37x |
| 512k | **0.56986** (1) | 0.09411 (1) | 0.24389 (3) | **6.06x** | 2.34x |
| 1M | **0.14828** (2) | 0.05103 (2) | 0.12077 (5) | **2.91x** | 1.23x |

The RASD-spec ratios are NOT like for like: the vLLM rows cover 1-2 books while the
spec means cover 3 (128k/512k) and 5 (1M). The matched-book column below is the one
to quote when a single document matters.

### 6.2 Matched book only (`pg19_train_0`, present in every arm)

| context | vLLM target-only | RASD target-only | RASD spec | vLLM / RASD target-only | vLLM / RASD spec | RASD spec / RASD target-only |
|---|---|---|---|---|---|---|
| 128k | 6.9865 | 0.28310 | 0.53084 | **24.68x** | 13.16x | 1.875x |
| 512k | 0.56986 | 0.09411 | 0.20607 | **6.06x** | 2.77x | 2.190x |
| 1M | 0.14822 | 0.05175 | 0.10801 | **2.86x** | 1.37x | 2.087x |

### 6.3 TTFT and peak memory

**vLLM TTFT: NOT AVAILABLE, and the column in the CSV must not be quoted.** The
retry's `ttft_s` holds `1.79e9` and `decode_only_wall_s` holds `-1.79e9` — see §5c:
vLLM 0.6.3 reports `RequestMetrics` timestamps as ABSOLUTE MONOTONIC CLOCK readings,
and the first version of the recorder subtracted one from a delta and wrote the
result down. So the honest statement is: **the offline `generate()` path used here
exposes no usable first-token time, and the run recorded none.** What the run does
support is the end-to-end wall per context, which includes the prefill of the whole
window. The one sound inference from it: vLLM's *total* wall is 2.1x (128k), 2.5x
(512k) and 2.5x (1M) longer than RASD's *prefill-only* `ttft_ms`, and vLLM's total
includes its own prefill plus 128 decode steps — so vLLM's prefill is the slower of
the two here, which is the opposite of the decode ordering.

| context | vLLM end-to-end wall | RASD `ttft_ms` (prefill only) | vLLM wall / RASD prefill |
|---|---|---|---|
| 128k | 18.321 s | 8.589 s | 2.13x |
| 512k | 224.615 s | 91.447 s | 2.46x |
| 1M | 863.576 s | 339.284 s | 2.55x |

**Peak memory per GPU.** vLLM recorded `peak_mem_mb` of **68798.7 / 68798.8 /
68799.0 MB (about 68.8 GB) at 128k / 512k / 1M** — flat across contexts, in the shape
a fixed `gpu_memory_utilization` preallocation gives, so it is the KV pool size and
NOT the working set at each window. RASD's `gpu_peak_mem_mb` on the same cells is
**8256.4 / 15781.2 / 25814.2 MB**: about 2.7x-8.3x smaller, and it *does* rise with
context. The two are not the same quantity (vLLM: preallocated pool from the driver
process at TP=8; RASD: peak of a 4-bit run incl. NF4 KV shards), so the table is
reported as-is and no ratio is claimed from it.

### 6.4 Everything a reviewer would flag about these two setups

1. **Weight precision — the biggest one.** vLLM ran **bf16** weights; RASD ran the
   **FP4** quantized target (`quantize_target: true`). The comparison is
   "production-precision engine" against "4-bit research path", not two engines at
   equal precision.
2. **KV precision.** vLLM **bf16** KV; RASD **NF4** KV (`kv_quant: true`,
   `kv_block_size 2048`). At 1M this is where RASD's 2.7x-lower memory peak and its
   flat decode cost come from.
3. **Decoding mode.** vLLM is **greedy** (`temperature 0.0`, `ignore_eos`, exactly
   128 tokens). Both RASD arms are **`temperature 0.7`, `top_p 1.0`, `ignore_eos`**,
   seed 42 (rows carry `temperature 0.7`). With EOS ignored this does not change how
   many tokens are produced, so the *throughput* comparison is unaffected; it does
   mean the vLLM row cannot be used for acceptance or for the acceptance-vs-context
   trend.
4. **No speculative decoding on the vLLM side.** The venv has no spec-decoding
   implementation, so vLLM is target-only. vLLM-vs-RASD-spec therefore compares a
   production target-only engine against a research spec path — it bounds one
   direction only and says nothing about RASD's spec path against a production spec
   path.
5. **Prompt ids: the same ids, one leading token apart.** vLLM was fed the sidecars'
   `engine_input_ids` — the ids the TARGET was actually fed, **including the leading
   BOS** — so its `prompt_tokens` reads 130048 / 523264 / 1047552 while RASD's CSV
   records 130047 / 523263 / 1047551 for the same window. The loader verifies the
   sidecar's `prompt_token_ids` hash (RASD's `prompt_sha256`,
   `dc21d9f3…` at 128k) before using it, and the one-token difference is the BOS, so
   the ids are the same prompt. vLLM's own `prompt_sha256` column hashes the 130048
   ids it fed (`eaf3abc1…`) and is therefore NOT comparable string-for-string with
   RASD's 130047-id hash — that is an artefact of hashing different objects, not a
   different prompt.
6. **Prefill strategy.** vLLM: chunked prefill (16384) + `max_num_seqs: 1` on **8
   ranks of one 8x A100 box**, no ring. RASD: ring attention across the same 8 ranks
   with `kv_block_size 2048`, `prefetch_depth 1`.
7. **`time_sec` includes prefill on both sides — verified, not assumed.** RASD's
   `time_sec` (2473.37 s at 1M) contains its `ttft_ms` (339.28 s), and
   `decode_tps = (tokens-1)/(time_sec - ttft)` reproduces the recorded 0.06, so its
   convention is the same end-to-end one vLLM's `end_to_end_wall_s` uses. The two
   numbers are therefore comparable *in convention*; they are not comparable in
   precision or in kernel set.
8. **n.** vLLM: 1 book at 128k/512k, 2 at 1M. RASD target-only: 1 book at 128k/512k,
   2 at 1M. RASD spec: 3 books at 128k/512k (the n=5 extension for these two rungs is
   running as this is written) and 5 at 1M (of 6 attempted — `pg19_train_1726` has
   only 765 continuation tokens at the 1M window and is excluded by data, §4 of
   `RESULTS_OCT.md`). No target-only rung has n>2, so the RASD target-only ratios rest
   on a single book at two of three contexts.
9. **Region/host.** The vLLM retry ran in us-midwest-1; the Oct 12 RASD session ran in
   us-east-1. Same 8x A100 80GB SXM4 type, different hosts.
10. **`unit_matched=yes` on all four vLLM rows** — target revision `cd3069b6…`, 128
    tokens, a real `doc_id`, TP=8, ids from the engine. The unit check is *within*
    vLLM's own ladder; it does not remove items 1-3.

The ratio must never be multiplied into, averaged with, or used to adjust the
2.37x/2.38x/2.59x headline speedups: those are RASD-vs-RASD and stand on their own.

### 6.5 Provenance of the vLLM rows (what code produced them)

The pod does not hold a git checkout: `scripts/mlsys_setup_probe.sh` rsyncs the Mac
**worktree** verbatim into `~/RASD`, so the executable revision is whatever the tree
was at the moment of provisioning. For the retry that was **11:24:44Z**, i.e.
**HEAD `d11fdbc`** ("Fix: the session driver pulled a hardcoded path, not
`$SESSION_OUT`", committed 11:24:21Z, 23 s earlier). Every file the vLLM path
executes — `scripts/mlsys_vllm_remote.sh`, `scripts/mlsys_vllm_baseline.py`,
`scripts/mlsys_vllm_venv.sh`, `scripts/mlsys_profile_decode.sh` — was **clean at
that commit**, so the four rows are reproducible from `d11fdbc` alone. Two
corroborating details in the pulled log: the 34b955d forwarding fix is visibly in
effect (`@ [131072 524288 1048576] tokens, TP=8`), and the parameter-contract block
that the wrapper carries today is **absent** from it (`parameters as received`
appears 0 times), which dates that block to after the run.

---

## 7. GO / NO-GO for running vLLM on 8x tomorrow (handoff item 5)

### **GO — with three conditions, and the run is NOT launched by this plan.**

**Why GO.** The question the comparison answers is the one the paper is missing:
every speedup in `RESULTS_OCT.md` is against RASD's own target-only path, which the
profile shows sits ~28x above its own bandwidth floor, and a reader cannot tell a
working engine from a slow one without an external reference. The machinery is now
proven on real hardware: the install, the pins, the revision, the load at 8k, the
decode and the DONE/terminate protocol all ran end-to-end for $0.96 total across
two attempts, on the second attempt with zero defects. The 8x run is the same
script with four env vars changed. Per-GPU memory at TP=8 fits all three contexts
(§2), and `bitsandbytes`, the only missing dependency, is not needed for a bf16 row.

**The three conditions, each of which is a way the run could still fail:**

1. **Stage the ORIGINAL sidecars** to `results/mlsys/vllm_prompts/` (§3.1). The
   committed `tokens_slim` set has no `engine_input_ids`, so pointing the run at it
   converts the whole comparison into synthetic prompts — it now fails in seconds
   at the CLI instead of after hours, but it would still fail.
2. **Keep `max_num_seqs: 1` and chunked prefill.** Without them the 1M rung does
   not fit (§2: 76.9 GiB against a 70.8 GiB budget) and the OOM would consume one
   of the three ladder attempts.
3. **Watch the 1M cell first.** It is the only rung that could not be rehearsed at
   its real size, and it is the one the comparison most needs. If its first attempt
   fails, the ladder's remaining two attempts are the fallback.

**Estimated hours and cost:** **1.0-1.5 h, $22-34**; worst case, if the 1M rung
burns all three attempts, **2.0 h, $45** (§4). Cumulative spend is **~$330 of the
$450 cap** (including the failing rehearsal's $0.35), so even the worst case leaves
~$75 of margin and this is the last planned spend.

**What is explicitly NOT part of this GO:**
* No 8x vLLM run is launched by this plan. It needs the operator's start, and the
  staging step in §3.1 to be executed first.
* No `src/` change, and no change to any RASD number. The vLLM rows are a new
  column beside the existing ones, never folded into them.
* No speculative-decoding comparison, because vLLM has no speculative decoding
  installed in this venv; the comparison is target-only against target-only.

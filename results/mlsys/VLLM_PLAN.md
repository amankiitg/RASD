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
bash scripts/mlsys_vllm_venv.sh                 # the ONLY sanctioned install path: torch 2.4.0+cu121, vllm==0.6.3
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
already parameterised:

```bash
SESSION_TYPE=gpu_8x_a100_80gb_sxm4 \
REMOTE_SCRIPT=scripts/mlsys_vllm_run_remote.sh \
PROFILE_OUT=results/mlsys/vllm_run PROFILE_TAG=vllm8x TIMER_MIN=120 \
RAW_CMD='bash scripts/mlsys_vllm_venv.sh && ...' \
bash scripts/mlsys_profile_1x.sh
```

(The 8x remote wrapper is the rehearsal's own script with `--tensor-parallel-size
8` and the staged prompt directory; it is written when the run is approved, not
before, so that nothing suggests this plan launched anything.)

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

## 5. How it will be reported next to RASD

One table in `results/mlsys/RESULTS_OCT.md`, beside — never inside — the RASD
numbers:

| context | RASD target-only tok/s (measured, this campaign) | vLLM 0.6.3 target-only tok/s (end-to-end, same prompt ids) | ratio |
|---|---|---|---|
| 128k | 0.28310 | *to be measured* | — |
| 512k | 0.09411 | *to be measured* | — |
| 1M | 0.05103 (mean of 2) | *to be measured* | — |

with, in the same table's notes: the precision asymmetry (NF4 KV/FP4 weights vs
bf16/bf16), `max_num_seqs: 1`, the exact `engine_input_ids` prompt hash per row,
`unit_matched` per row, and the statement that the ratio bounds **one direction
only** — it says how much slower RASD's target-only path is than a production
engine on the same prompt; it says nothing about RASD's speculative path against a
production speculative path, because vLLM has none installed here.

The ratio must never be multiplied into, averaged with, or used to adjust the
2.37x/2.38x/2.59x headline speedups: those are RASD-vs-RASD and stand on their own.

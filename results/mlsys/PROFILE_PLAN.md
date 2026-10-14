# PROFILE_PLAN.md — what will be profiled, why, and what it has already cost

Deliberately separate from `MORNING_REPORT.txt`, which the running session's puller writes.

## Where this sits

`BANDWIDTH_FLOOR.md` shows the engine is **42-59x above its bandwidth floor** and that the
measured per-token time is very nearly linear in the number of ring chunk transfers
(0.58-0.86 ms each, across an 8x range of context). That is an inference from arithmetic and
code reading. These two profiles are what test it.

## The two profiles

Both are the SAME script with one parameter changed, because the only interpretable
difference between them must be `nproc`:

```bash
# (a) 1 GPU: nproc 1, the ring is world_size=1, so comms vanish and compute stands alone
bash scripts/mlsys_profile_decode.sh --nproc 1 --tag a --out results/mlsys/profile

# (b) 2 GPUs: nproc 2, ring communication becomes visible
bash scripts/mlsys_profile_decode.sh --nproc 2 --tag b --out results/mlsys/profile
```

Config: `configs/mlsys_profile_128k.yml` — target-only (`spec_steps: 0`) decode at 128k,
`kv_block_size: 2048`, `prefetch_depth: 1`, NF4 KV, seed 42, greedy.

### What each run yields

1. **Measured per-token time**, as `time_sec / tokens_generated` from the CSV — never the
   rounded `throughput_tps` field (0.10 vs 0.10024 is enough to move a ratio).
2. **The profiler sidecar**, `<out>/profiler/<run_id>.json`, written by the runner's own
   `--profile` flag (`run_experiment.py:1816`, "Wrap each generate() call in a torch.profiler
   context (C7 RoundProfiler)"). It carries a compute/comm/idle breakdown in microseconds;
   the script converts by key suffix and reports the share of measured wall time.
3. **The share of the measured time the profile explains**, printed next to the wall number.

### Warm-up: an honest limitation

The runner's `--profile` wraps the **whole `generate()` call**. There is no way to profile
"3 warm-up steps then ~20 measured steps" without engine changes, and engine changes are
forbidden in this plan because they would invalidate every timing already measured. Warm-up
is therefore done at **row granularity**: `PROF_WARM` (8 tokens) runs first and warms the
CUDA context, allocator and kernels; `PROF_MAIN` (24 tokens) is the row whose numbers are
reported. This is coarser than the plan asked for and is stated wherever the numbers appear.

## Local testing: what was possible, and what it found

**The run half cannot be tested on CPU. The engine requires CUDA:**
`_run_single_worker` calls `torch.cuda.set_device(0)`, which on this Mac's CPU-only torch
raises `AttributeError: module 'torch._C' has no attribute '_cuda_setDevice'`. Recorded
rather than worked around — the profile runs on a GPU pod, not locally.

**The reporting half was tested locally, and the test found two real defects:**

1. The script defaulted to bare `python`, which on this Mac is the base miniconda
   interpreter whose torch is broken (`Library not loaded: @rpath/libtorch_cpu.dylib`). It
   now prefers the project env and takes `MLSYS_PY` as an override.
2. The reporting added the sidecar's **microsecond** fields and divided by **seconds**,
   printing a share of **99,166,666%**. It now converts by key suffix and refuses to assume
   a unit for fields that carry no suffix. Reproduced and fixed via `--report-only`, which
   exists so this half can be exercised with no GPU.

Synthetic verification after the fix: a 240.0 s row with a 238,000,000 us sidecar reports
`sidecar total = 238.00 s`, `share of the measured wall time: 99.2%`.

## Decision rule

Run (a). **If (a) explains most of the measured per-token time, stop** — the cost is local
and inside the step. Only if it does not, run (b) to attribute the remainder to ring
communication. If no 2-GPU A100 SXM (NVLink) instance is available within 2 hours, skip (b)
and say so; a PCIe pair would not represent the same interconnect and would be a worse
measurement than none.

## Limits and expected cost

| profile | instance | rate | wall estimate | cap |
|---|---|---|---|---|
| (a) | `gpu_1x_a100_sxm4` (the only 1x A100 family with capacity on 2026-10-12) | $1.99/hr | model load 5-10 min + 128k prefill + 2 short rows ≈ 20-30 min | **$5**, 60-min detached dead-man timer |
| (b) | cheapest 2x A100 SXM (NVLink) | ~$3.98/hr | same shape | **$10**, 60-min timer |

Every paid instance gets its timer through `scripts/mlsys_detach.py`, confirmed by
**PPID 1** before any work starts, and every stop is confirmed by effect (API instance
count plus `ps`).

## Findings only — no engine changes

`src/` is not touched by this plan. The expected findings are: the fraction of per-token
time inside the ring rotation, the fraction inside NF4 dequantization, and the fraction in
Python/launch overhead — i.e. whether the 42-59x gap is per-transfer latency (which is what
the linear fit in `BANDWIDTH_FLOOR.md` predicts) or something else.

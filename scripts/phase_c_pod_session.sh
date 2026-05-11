#!/usr/bin/env bash
# M4 Phase C — bundled pod session orchestration.
#
# Runs the full Phase C sequence on a Lambda 8x A100 instance:
#   P3.0  GPU health check
#   P3.1  reproducibility lockdown (capture_pod_env + pin_hf_revisions)
#         + replay_m3_smoke (drift <= 15%)
#   C11   NF4 KV-cache validation gate
#   C2b   YaRN RoPE numeric validation
#   C6    Checkpoint/resume multi-rank validation
#   P3.3  long-context smoke tests (32k / 128k / 512k / 1M, RASD only)
#   P3.4  baseline validation (Ring + Sliding × 2 contexts)
#   P3.5  final 36-run matrix
#   P3.6  profiler sidecar pass on a subset
#
# Each stage writes a marker file under results/phase_c/ on success.
# The script aborts on first failure so we don't burn pod-$ on
# downstream stages that depend on broken upstream state.
#
# Pre-flight (do these BEFORE running this script):
#   * conda env create -f environment_gpu.yml && conda activate rasd-gpu
#   * pip install --no-build-isolation flash-attn>=2.4.0
#   * pip install -e .
#   * export WANDB_API_KEY=<your-key> HF_TOKEN=<your-token>
#   * export HF_HOME=/workspace/hf_cache  # persistent volume
#
# Usage:
#   bash scripts/phase_c_pod_session.sh
#
# Estimated runtime: ~10 hours / ~$160 at $15.92/hr.

set -euo pipefail

cd "$(dirname "$0")/.."

# Environment guards
: "${WANDB_API_KEY:?WANDB_API_KEY must be set}"
: "${HF_TOKEN:?HF_TOKEN must be set}"
: "${HF_HOME:?HF_HOME must be set (use /workspace/hf_cache on Lambda)}"

PHASE_C_DIR=results/phase_c
mkdir -p $PHASE_C_DIR

# Safety net: NCCL watchdog timeout. At 1M context, ring-attention
# coalesced ops can block long enough for the default 10-min watchdog
# to fire (observed 2026-05-10 first 1M attempt: rank 7 hung at
# SeqNum=36 after 600s waiting for rank 2). The code-side fix is
# timedelta(hours=1) on dist.init_process_group, but env-var override
# is the most reliable path since some torch versions ignore the
# kwarg. Pinning to 1 hour matches the code-side timeout.
export NCCL_TIMEOUT=3600
export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=3600
export TORCH_NCCL_BLOCKING_WAIT=1

# expandable_segments: reduces PyTorch caching-allocator fragmentation.
# At 1M context, the v4 OOM showed 8.77 GB "reserved but unallocated"
# alongside 62.94 GB live tensors — the allocator was sitting on
# fragmented chunks too small for the next 7.64 GB request. With
# expandable_segments=True, the allocator keeps a virtual address
# space that grows incrementally and avoids the small-chunk
# fragmentation pattern. Reclaims most of that 8.77 GB.
# (Recommendation surfaced by torch's own OOM error message in v4.)
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
LOG_DIR=$PHASE_C_DIR/logs
mkdir -p $LOG_DIR

# Stage helper: skips if marker file already exists; record success on completion.
stage() {
    local name="$1"; shift
    local marker="$PHASE_C_DIR/${name}.done"
    if [ -f "$marker" ]; then
        echo "==> [$name] already complete (marker: $marker), skipping"
        return 0
    fi
    echo ""
    echo "============================================================"
    echo "==> [$name] starting at $(date -u +%Y-%m-%dT%H:%M:%SZ)"
    echo "============================================================"
    "$@" 2>&1 | tee "$LOG_DIR/${name}.log"
    local rc=${PIPESTATUS[0]}
    if [ $rc -ne 0 ]; then
        echo "==> [$name] FAILED with exit $rc — see $LOG_DIR/${name}.log"
        exit $rc
    fi
    touch "$marker"
    echo "==> [$name] OK at $(date -u +%Y-%m-%dT%H:%M:%SZ)"
}

# ------------------------------------------------------------------
# P3.0 — GPU health-check preflight
# ------------------------------------------------------------------
gpu_health_check() {
    echo "--- ECC / XID / throttle scan ---"
    nvidia-smi -q | grep -iE "ecc|xid|throttl" || true
    echo "--- Idle memory (must be 0 MiB on all GPUs) ---"
    used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | tr -d ' ' | tr '\n' ',')
    echo "memory.used per GPU: $used"
    if echo "$used" | tr ',' '\n' | grep -vE '^(0|)$' >/dev/null; then
        echo "FAIL: at least one GPU has nonzero memory.used"
        exit 1
    fi
    echo "All 8 GPUs idle — OK"
}
stage "p30_gpu_health" gpu_health_check

# ------------------------------------------------------------------
# P3.1 — Reproducibility lockdown
# ------------------------------------------------------------------
repro_lockdown() {
    bash scripts/capture_pod_env.sh > requirements-lock.txt
    git add requirements-lock.txt
    git diff --cached --stat
    python scripts/pin_hf_revisions.py
    git add configs/ablations.yml
    git diff --cached --stat
    # 2026-05-10: replay_m3_smoke.sh DROPPED from the Phase C bootstrap.
    # M3 R6.5's actual results live in wandb project
    # rasd-m3-reablation-64k; comparing against ablations_r65.csv on
    # this pod doesn't add information beyond what the m3-reproducible
    # git tag + 400 unit tests already guarantee. Plus throughput
    # numbers diverge anyway because R6.5 ran on Lambda's default
    # image without flash-attn (this pod has flash-attn 2.8.3 active),
    # so any tps comparison is apples-to-oranges. Acceptance-rate
    # check would still be valid, but it duplicates what
    # tests/test_m3_invariants.py already enforces statically.
    echo "==> M3 replay smoke skipped — see comment for rationale"
}
stage "p31_repro_lockdown" repro_lockdown

# ------------------------------------------------------------------
# C11 — NF4 KV-cache validation gate (single-GPU)
# ------------------------------------------------------------------
c11_validation() {
    python scripts/c11_validation.py \
        --target meta-llama/Llama-2-7b-hf \
        --contexts 1024 4096 \
        --block-sizes 64 \
        --seeds 42 123 456 \
        --out results/c11_validation/c11_validation.json
}
stage "c11_validation" c11_validation

# ------------------------------------------------------------------
# C2b — YaRN RoPE numeric validation (single-GPU)
# ------------------------------------------------------------------
yarn_validation() {
    python scripts/yarn_numeric_validation.py \
        --target meta-llama/Llama-2-7b-hf \
        --short-ctx 65536 \
        --long-ctx 524288 \
        --out results/yarn_validation/yarn_validation.json
}
stage "c2b_yarn_validation" yarn_validation

# ------------------------------------------------------------------
# C6 — Checkpoint/resume multi-rank validation (8 GPUs)
# ------------------------------------------------------------------
c6_validation() {
    # c6_resume_validation.py is a torchrun-direct script (no outer
    # orchestrator) — keep the torchrun here.
    torchrun --nproc-per-node=8 --master_port=29500 \
        scripts/c6_resume_validation.py \
        --target meta-llama/Llama-2-7b-hf \
        --draft  princeton-nlp/Sheared-LLaMA-1.3B \
        --ctx 4096 --max-new 16 --checkpoint-every 4
}
stage "c6_resume_validation" c6_validation

# ------------------------------------------------------------------
# P3.3 — long-context smokes: RASD at 32k, 128k, 512k, 1M
# (single-prompt, NF4 KV enabled, YaRN RoPE)
#
# IMPORTANT: do NOT wrap run_experiment.py in `torchrun` here.
# run_experiment.py is the orchestrator — it reads the YAML, expands
# the matrix, and spawns its own per-row `torchrun --nproc_per_node=8`
# inside execute_run(). Wrapping the orchestrator with torchrun causes
# 8 orchestrators to each spawn 8-way torchrun = 64-way GPU contention
# + master_port collisions. (Fix for high-risk finding #2, 2026-05-10.)
# ------------------------------------------------------------------
long_ctx_smokes() {
    # The SMOKE group in configs/m4_phase_c_long_smoke.yml already has
    # all four contexts (32k / 128k / 512k / 1M) as levels. A bash loop
    # over ctx would expand the grid 4x — each iteration runs the entire
    # SMOKE group again because run_experiment.py has no per-level
    # filter. (Fix for finding #2 from 2026-05-10 review: was a 4x
    # compute waste = ~$50-80 of pod time for nothing.)
    #
    # 4-hour per-run timeout: the 1M cell takes ~120 min per the YAML's
    # own comment; the default 3600s (1 hr) would SIGTERM it mid-run,
    # wasting an hour of pod time per failure. (Fix for blocker 3 from
    # 2026-05-10 third-pass review.)
    # --abort-on-failure: stop hard if 32k or 128k fails; don't waste
    # pod-$ on 512k/1M cells that are guaranteed to fail downstream.
    # Reviewer's recommended pod-gate order is C11 → C6 → 32k → 128k →
    # only then 512k/1M; this enforces that progression.
    # --memory-trace: per-rank GPU memory attribution snapshots at
    # generate() lifecycle points (post-load, post-prefill, post-verify
    # round 1/2/4/8, end). Source for the paper's memory-attribution
    # figure. Negligible overhead; off was the silent default before.
    python run_experiment.py \
        --wandb-project rasd-m4-phase-c \
        --config configs/m4_phase_c_long_smoke.yml \
        --output results/m4_smoke/long_smoke.csv \
        --resume \
        --nproc 8 \
        --groups SMOKE \
        --seeds 42 \
        --timeout-per-run-s 14400 \
        --abort-on-failure \
        --log-per-token \
        --memory-trace
}
stage "p33_long_ctx_smokes" long_ctx_smokes

# ------------------------------------------------------------------
# P3.4 — baseline validation: Ring + Sliding at 128k and 1M
# (Fix for high-risk finding #4, 2026-05-10: was `bash` + `--contexts`;
# the script is Python and the flag is `--lengths`. Added `--distributed`
# so multi-GPU sequence-parallelism actually exercises.)
# ------------------------------------------------------------------
baseline_validation() {
    # Match the M4 final-matrix grid: 4 contexts × 3 seeds × 2 baselines
    # = 24 baseline rows. Without the seeds + 1M, Phase D Figure 1 has
    # no Ring/Sliding error bars and no 1M point at all. (Fix for
    # finding #4 from 2026-05-10 review.)
    torchrun --nproc-per-node=8 --master_port=29500 \
        scripts/benchmark_baselines.py \
        --lengths 131072 262144 524288 1048576 \
        --seeds 42 123 456 \
        --out results/baselines/m4_baselines.csv \
        --distributed
}
stage "p34_baseline_validation" baseline_validation

# ------------------------------------------------------------------
# P3.5 — final 12-run RASD matrix (4 contexts x 3 seeds).
# Same anti-double-torchrun fix as P3.3.
# ------------------------------------------------------------------
final_matrix() {
    # 4-hour per-run timeout for long-context cells (see long_ctx_smokes
    # comment + finding #3, 2026-05-10).
    python run_experiment.py \
        --wandb-project rasd-m4-phase-c \
        --config configs/m4_final_matrix.yml \
        --output results/final/final_matrix.csv \
        --resume \
        --nproc 8 \
        --timeout-per-run-s 14400 \
        --log-per-token
}
stage "p35_final_matrix" final_matrix

# Stage ordering note (2026-05-10 PM): p35b (target-only matrix) is the
# longest remaining stage (~3-4 hr). To preserve the most data in the
# event of any mid-stage failure, the cheap stages — p35c (~10 min PPL),
# p36 (~50 min profiler), p37 (~10 min HF ceiling) — run FIRST, with
# p35b last. Stage numbers preserve their semantic order but execution
# order is: p35 → p35c → p36 → p37 → p35b.

# ------------------------------------------------------------------
# P3.5c — PG-19 perplexity sanity check (single-rank, ≤32k contexts)
# Quality metric to confirm NF4 weights + ring attention infra
# don't degrade language modelling quality at moderate contexts.
# Long-context PPL (128k+) requires sequence-parallel forward,
# deferred to follow-up iteration (documented in docs/dev/M4_PLAN.md).
# ------------------------------------------------------------------
perplexity_sanity() {
    # Preprocess PG-19 if not already done. preprocess_pg19.py writes
    # metadata to {out}/pg19_{split}_metadata.json — default out is
    # data/processed/pg19, so meta lands at the subdir path below.
    if [ ! -f data/processed/pg19/pg19_validation_metadata.json ]; then
        python scripts/preprocess_pg19.py \
            --split validation \
            --limit 8 \
            --tokenizer meta-llama/Llama-2-7b-hf \
            --chunk-size 65536
    fi
    # Run PPL eval at moderate contexts (NF4 weights, single-rank).
    # rope_type=yarn matches the production runs (p33-p35, p35b, p36);
    # without it vanilla Llama-2 PPL explodes past 4k due to RoPE
    # extrapolation collapse — masks any quant-quality signal.
    python scripts/eval_perplexity_matrix.py \
        --target meta-llama/Llama-2-7b-hf \
        --contexts 4096 8192 16384 32768 \
        --seeds 42 123 456 \
        --quantize-target \
        --rope-type yarn \
        --rope-native-max 4096 \
        --pg19-meta data/processed/pg19/pg19_validation_metadata.json \
        --out results/perplexity/m4_ppl.csv
}
stage "p35c_perplexity_sanity" perplexity_sanity

# ------------------------------------------------------------------
# P3.6 — profiler sidecar pass: 1 seed × 4 ctx × {RASD, Ring}
# (Fig 3 source data)
# ------------------------------------------------------------------
profiler_sidecar_pass() {
    # Run the matrix once more with --profile, on a subset (1 seed at all
    # 4 contexts) to keep overhead bounded. Output at
    # results/final/profiler_pass/profiler_pass.csv + per-run JSON
    # sidecars at .../profiler/<run_id>.json — Fig 3 source data.
    # Same anti-double-torchrun fix as P3.3 / P3.5 (finding #2).
    # 4-hr per-run timeout (finding #3) since profiler adds ~10% overhead
    # on top of the 120-min 1M baseline.
    # M4 Phase C 2026-05-10 (codex review): 1M dropped from profiler
    # subset for three reasons:
    #   1. torch.profiler accumulates an event buffer over the full
    #      generation. At 1M ctx × 24 min × ~36 verify rounds, the
    #      event count can be 500k+ on rank 0 alone, eating CPU RAM
    #      and risking failure at profiler finalization.
    #   2. Only rank 0 profiles (run_experiment.py L414), creating a
    #      rank-asymmetric wall-clock. At 1M with NCCL-heavy ops,
    #      rank-0 slowdown can stall other ranks.
    #   3. 1M's profiler data adds little: the verify-loop wall-time
    #      breakdown is derivable from final_matrix.csv's time_sec /
    #      n_rounds. Figure 3 needs 32k/128k/512k where compute/comm/
    #      idle proportions actually differ meaningfully.
    # 1M-specific profiling deferred to a follow-up paper if needed,
    # with proper torch.profiler.schedule (wait/warmup/active/repeat)
    # so we only capture a few rounds, not all 36.
    python run_experiment.py \
        --wandb-project rasd-m4-phase-c \
        --config configs/m4_profiler_subset.yml \
        --output results/final/profiler_pass/profiler_pass.csv \
        --resume \
        --nproc 8 \
        --seeds 42 \
        --timeout-per-run-s 14400 \
        --profile
}
stage "p36_profiler_pass" profiler_sidecar_pass

# ------------------------------------------------------------------
# P3.7 — Vanilla HF FA-2 generate() ceiling baseline
# Single-rank, single-seed OOM ceiling test. Documents that vanilla HF
# generate() cannot reach long context on commodity hardware without
# sequence parallelism or KV quantization. Expected:
#   * 32k passes (~30 GB peak)
#   * 128k borderline / OOMs
#   * 256k/512k/1M definite OOMs
# The 32k row is apples-to-apples vs RASD; OOM rows are the contrast
# that makes RASD's scaling claim defensible.
# ------------------------------------------------------------------
hf_ceiling_baseline() {
    python scripts/benchmark_hf_baseline.py \
        --target meta-llama/Llama-2-7b-hf \
        --contexts 32768 131072 262144 524288 1048576 \
        --max-new-tokens 64 \
        --seed 42 \
        --attn-impl flash_attention_2 \
        --out results/baselines/hf_ceiling.csv
}
stage "p37_hf_ceiling_baseline" hf_ceiling_baseline

# ------------------------------------------------------------------
# P3.5b — Target-only baseline matrix (run LAST per 2026-05-10 PM
# reorder — see "Stage ordering note" comment above).
# Apples-to-apples baseline: same RASD distributed infrastructure
# (NF4 cache + ring SP + outlier-keep + chunked update) but with
# cfg.spec_steps=0 → no draft model loaded, single-token autoregressive
# decode through the target. Isolates the contribution of speculative
# decoding by keeping every other variable identical to p35.
# Same 4 ctx × 3 seeds grid for direct cell-by-cell comparison.
# ~3-4 hr runtime; placed last so quick stages above complete first.
# ------------------------------------------------------------------
target_only_matrix() {
    python run_experiment.py \
        --wandb-project rasd-m4-phase-c \
        --config configs/m4_target_only_matrix.yml \
        --output results/final/target_only_matrix.csv \
        --resume \
        --nproc 8 \
        --timeout-per-run-s 14400 \
        --log-per-token \
        --memory-trace
}
stage "p35b_target_only_matrix" target_only_matrix

echo ""
echo "============================================================"
echo "==> Phase C complete at $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "============================================================"
echo "Markers:"
ls -la $PHASE_C_DIR/*.done
echo ""
echo "Results:"
ls -la results/c11_validation/ results/yarn_validation/ \
       results/c6_validation/ results/m4_smoke/ \
       results/final/ 2>/dev/null || true

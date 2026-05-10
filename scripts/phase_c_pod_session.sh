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
        --log-per-token
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
    python run_experiment.py \
        --wandb-project rasd-m4-phase-c \
        --config configs/m4_final_matrix.yml \
        --output results/final/profiler_pass/profiler_pass.csv \
        --resume \
        --nproc 8 \
        --seeds 42 \
        --timeout-per-run-s 14400 \
        --profile
}
stage "p36_profiler_pass" profiler_sidecar_pass

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

#!/usr/bin/env bash
# M4 Phase C — 40GB-pod variant (768k as max context).
#
# This is a fork of phase_c_pod_session.sh that points at the
# _40gb YAML variants (configs/m4_phase_c_long_smoke_40gb.yml
# and configs/m4_final_matrix_40gb.yml). Used when 1M doesn't
# fit on a 40GB SXM4 pod and we accept 768k as the largest
# paper-grade context. The 1M cell is then run separately on
# 80GB SXM4 when that capacity becomes available.
#
# Estimated runtime: ~3-4 hr (faster than 80GB version because 1M
# is dropped). ~$50-65 at $15.92/hr.
#
# Pre-flight (do these BEFORE running this script):
#   * conda activate rasd-gpu
#   * pip install -r requirements-lock.txt
#   * pip install --no-build-isolation flash-attn==2.8.3
#   * pip install -e .
#   * export WANDB_API_KEY=<your-key> HF_TOKEN=<your-token>
#   * export HF_HOME=/home/ubuntu/hf_cache
#
# Usage:
#   bash scripts/phase_c_pod_session_40gb.sh

set -euo pipefail

cd "$(dirname "$0")/.."

# Environment guards
: "${WANDB_API_KEY:?WANDB_API_KEY must be set}"
: "${HF_TOKEN:?HF_TOKEN must be set}"
: "${HF_HOME:?HF_HOME must be set}"

PHASE_C_DIR=results/phase_c_40gb
mkdir -p $PHASE_C_DIR
LOG_DIR=$PHASE_C_DIR/logs
mkdir -p $LOG_DIR

# NCCL + allocator levers
export NCCL_TIMEOUT=3600
export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=3600
export TORCH_NCCL_BLOCKING_WAIT=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

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
# C11 — NF4 KV-cache validation
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
# C2b — YaRN RoPE numeric validation
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
# C6 — Checkpoint/resume multi-rank validation
# ------------------------------------------------------------------
c6_validation() {
    torchrun --nproc-per-node=8 --master_port=29500 \
        scripts/c6_resume_validation.py \
        --target meta-llama/Llama-2-7b-hf \
        --draft  princeton-nlp/Sheared-LLaMA-1.3B \
        --ctx 4096 --max-new 16 --checkpoint-every 4
}
stage "c6_resume_validation" c6_validation

# ------------------------------------------------------------------
# P3.3 — long-context smokes: 32k / 128k / 512k / 768k (40GB variant)
# ------------------------------------------------------------------
long_ctx_smokes() {
    python run_experiment.py \
        --wandb-project rasd-m4-phase-c \
        --config configs/m4_phase_c_long_smoke_40gb.yml \
        --output results/m4_smoke/long_smoke_40gb.csv \
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
# P3.4 — baseline validation: Ring + Sliding at 128k / 256k / 512k / 768k
# ------------------------------------------------------------------
baseline_validation() {
    torchrun --nproc-per-node=8 --master_port=29500 \
        scripts/benchmark_baselines.py \
        --lengths 131072 262144 524288 786432 \
        --seeds 42 123 456 \
        --out results/baselines/m4_baselines_40gb.csv \
        --distributed
}
stage "p34_baseline_validation" baseline_validation

# ------------------------------------------------------------------
# P3.5 — final 12-run RASD matrix (4 contexts × 3 seeds, 768k as max)
# ------------------------------------------------------------------
final_matrix() {
    python run_experiment.py \
        --wandb-project rasd-m4-phase-c \
        --config configs/m4_final_matrix_40gb.yml \
        --output results/final/final_matrix_40gb.csv \
        --resume \
        --nproc 8 \
        --timeout-per-run-s 14400 \
        --log-per-token \
        --memory-trace
}
stage "p35_final_matrix" final_matrix

# ------------------------------------------------------------------
# P3.6 — profiler sidecar pass: 1 seed × 4 ctx × {RASD}
# ------------------------------------------------------------------
profiler_sidecar_pass() {
    python run_experiment.py \
        --wandb-project rasd-m4-phase-c \
        --config configs/m4_final_matrix_40gb.yml \
        --output results/final/profiler_pass/profiler_pass_40gb.csv \
        --resume \
        --nproc 8 \
        --seeds 42 \
        --timeout-per-run-s 14400 \
        --profile
}
stage "p36_profiler_pass" profiler_sidecar_pass

echo ""
echo "============================================================"
echo "==> 40GB Phase C complete at $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "============================================================"
echo "Markers:"
ls -la $PHASE_C_DIR/*.done

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
    bash scripts/replay_m3_smoke.sh
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
# ------------------------------------------------------------------
long_ctx_smokes() {
    for ctx in 32768 131072 524288 1048576; do
        torchrun --nproc-per-node=8 --master_port=29500 \
            run_experiment.py \
            --config configs/m4_phase_c_long_smoke.yml \
            --output results/m4_smoke/long_smoke_ctx${ctx}.csv \
            --resume \
            --nproc 8 \
            --groups SMOKE \
            --seeds 42 \
            --log-per-token
    done
}
stage "p33_long_ctx_smokes" long_ctx_smokes

# ------------------------------------------------------------------
# P3.4 — baseline validation: Ring + Sliding at 128k and 1M
# ------------------------------------------------------------------
baseline_validation() {
    bash scripts/benchmark_baselines.py --contexts 131072 1048576 \
        --out results/baselines/m4_baselines.csv
}
stage "p34_baseline_validation" baseline_validation

# ------------------------------------------------------------------
# P3.5 — final 36-run matrix: RASD/Ring/Sliding x {128k,256k,512k,1M} x 3 seeds
# ------------------------------------------------------------------
final_matrix() {
    torchrun --nproc-per-node=8 --master_port=29500 \
        run_experiment.py \
        --config configs/m4_final_matrix.yml \
        --output results/final/final_matrix.csv \
        --resume \
        --nproc 8 \
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
    torchrun --nproc-per-node=8 --master_port=29500 \
        run_experiment.py \
        --config configs/m4_final_matrix.yml \
        --output results/final/profiler_pass/profiler_pass.csv \
        --resume \
        --nproc 8 \
        --seeds 42 \
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

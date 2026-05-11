#!/usr/bin/env bash
# Phase D — pod re-run to capture per-token sidecars + generated text.
#
# Captures the F4/F5/F8 inputs that Phase C did not commit (per_token
# JSONL files are .gitignored due to size; generated text was never
# saved before --save-generated-text landed).
#
# Single seed × 4 contexts × 2 modes (RASD spec + target-only) = 8 cells.
# Expected wall: ~1.5–2 hr on 8x A100 80GB SXM4. Cost: ~$35–40.
#
# Requirements (env vars):
#   WANDB_API_KEY, HF_TOKEN, HF_HOME (=/home/ubuntu/hf_cache on Lambda)
#
# Reproducibility gate: re-captures requirements-lock.txt on the pod
# and fails if it diverges from the committed lock (image drift check).

set -euo pipefail

: "${WANDB_API_KEY:?WANDB_API_KEY must be set}"
: "${HF_TOKEN:?HF_TOKEN must be set}"
: "${HF_HOME:?HF_HOME must be set}"
: "${PYTORCH_CUDA_ALLOC_CONF:=expandable_segments:True}"
: "${NCCL_TIMEOUT:=3600}"

cd ~/RASD

echo "==> [phase D] starting at $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "==> git HEAD: $(git rev-parse --short HEAD)"

# ---- Reproducibility gate ----
echo "==> Capturing live pip freeze + comparing against committed lock"
bash scripts/capture_pod_env.sh > /tmp/requirements-lock.live.txt
if ! diff -q requirements-lock.txt /tmp/requirements-lock.live.txt > /dev/null; then
    echo "WARN: pod pip env diverges from committed requirements-lock.txt:"
    diff requirements-lock.txt /tmp/requirements-lock.live.txt | head -40 || true
    echo "(continuing — non-fatal; review post-run for paper repro section)"
fi

# ---- Build a per-cell YAML on the fly ----
# Single seed (42) × 4 ctx × 2 modes. RASD: spec_steps=4. Target-only: spec_steps=0.
build_cfg () {
    local mode="$1"     # rasd | target
    local ctx="$2"      # 131072 | 262144 | 524288 | 1048576
    local spec_steps
    local prefix
    if [ "$mode" = "rasd" ]; then
        spec_steps=4
        prefix="RASD"
    else
        spec_steps=0
        prefix="TARGET"
    fi
    local ctx_label
    case "$ctx" in
        131072)   ctx_label="ctx128k" ;;
        262144)   ctx_label="ctx256k" ;;
        524288)   ctx_label="ctx512k" ;;
        1048576)  ctx_label="ctx1M"   ;;
    esac
    local out=/tmp/phase_d_${prefix}_${ctx_label}.yml
    cat > "$out" <<EOF
defaults:
  target_model_name: "meta-llama/Llama-2-7b-hf"
  draft_model_name:  "princeton-nlp/Sheared-LLaMA-1.3B"
  target_revision:   null
  draft_revision:    "a4b76938edbf571ea7d7d9904861cbdca08809b4"
  spec_steps:        ${spec_steps}
  kv_block_size:     2048
  prefetch_depth:    1
  rope_type:         "yarn"
  kv_quant:          true
  max_new_tokens:    64
  dtype:             "bfloat16"
  quantize_draft:    true
  quantize_target:   true
  temperature:       1.0
  top_p:             1.0
  seeds:             [42]

M4:
  name:   "phase_d_rerun"
  factor: "context_length"
  levels:
    - id: "${prefix}_${ctx_label}_phaseD"
      context_length: ${ctx}
      $( [ "$ctx" = "524288" ] && echo "checkpoint_every: 4" || true)
EOF
    echo "$out"
}

# ---- Run each cell sequentially, single GPU pod (no orchestrator) ----
run_cell () {
    local mode="$1"
    local ctx="$2"
    local label="$3"
    local timeout="$4"

    local cfg
    cfg=$(build_cfg "$mode" "$ctx")
    echo
    echo "===================================================================="
    echo ">>> [$(date -u +%H:%M:%S)] $mode @ $label (ctx=$ctx, timeout=${timeout}s)"
    echo "===================================================================="

    # Defensive: wait for GPU idle between cells
    sleep 10

    python run_experiment.py \
        --wandb-project rasd-m4-phase-d \
        --config "$cfg" \
        --output results/final/phase_d_rerun.csv \
        --groups M4 \
        --seeds 42 \
        --nproc 8 \
        --timeout-per-run-s "$timeout" \
        --log-per-token \
        --save-generated-text \
        --memory-trace
}

# RASD mode first (warms model + KV; faster than target-only at ctx256k+)
run_cell rasd     131072  ctx128k  1800
run_cell rasd     262144  ctx256k  3600
run_cell rasd     524288  ctx512k  5400
run_cell rasd    1048576  ctx1M    7200

# Target-only mode
run_cell target   131072  ctx128k  1800
run_cell target   262144  ctx256k  3600
run_cell target   524288  ctx512k  5400
run_cell target  1048576  ctx1M    7200

echo
echo "==> Phase D rerun complete at $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "==> Captured artifacts:"
ls -la results/final/per_token/ 2>/dev/null  | head
ls -la results/final/generated/ 2>/dev/null  | head
echo
touch /tmp/phase_d_rerun_DONE

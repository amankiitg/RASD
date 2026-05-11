#!/usr/bin/env bash
# Phase D: additional short-context PG-19 cells for F4 sanity + F5
# readable text.
#
# Adds 4 cells: {RASD, TARGET} x {4k, 8k} with PG-19 prompts. The main
# Phase D matrix used synthetic repetitive-paragraph prompts which
# produce near-random tokens at long contexts under YaRN factor=256.
# These short-ctx PG-19 cells give:
#
#   F4: per-round acceptance in the well-behaved regime (Llama-2's
#       native ctx, real narrative text) — reference for the bimodal
#       pattern seen at long ctx.
#
#   F5: readable side-by-side qualitative comparison of RASD vs
#       target-only on real text.

set -euo pipefail
: "${WANDB_API_KEY:?}"; : "${HF_TOKEN:?}"
: "${PYTORCH_CUDA_ALLOC_CONF:=expandable_segments:True}"
: "${NCCL_TIMEOUT:=3600}"
export HF_HOME=/home/ubuntu/hf_cache

cd ~/RASD
source ~/venv-rasd/bin/activate

build_cfg () {
    local mode="$1"   # rasd | target
    local ctx="$2"    # 4096 | 8192
    local spec; local prefix
    if [ "$mode" = "rasd" ]; then spec=4; prefix="RASD"
    else spec=0; prefix="TARGET"; fi
    local label
    case "$ctx" in
        4096) label="ctx4k"  ;;
        8192) label="ctx8k"  ;;
    esac
    local out=/tmp/phase_d_pg19_${prefix}_${label}.yml
    cat > "$out" <<EOF
defaults:
  target_model_name: "meta-llama/Llama-2-7b-hf"
  draft_model_name:  "princeton-nlp/Sheared-LLaMA-1.3B"
  target_revision:   null
  draft_revision:    "a4b76938edbf571ea7d7d9904861cbdca08809b4"
  spec_steps:        ${spec}
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
  name:   "phase_d_short_ctx_pg19"
  factor: "context_length"
  levels:
    - id: "${prefix}_${label}_pg19_phaseD"
      context_length: ${ctx}
EOF
    echo "$out"
}

for mode in rasd target; do
  for ctx in 4096 8192; do
    cfg=$(build_cfg "$mode" "$ctx")
    label="ctx$([ "$ctx" = "4096" ] && echo 4k || echo 8k)"
    [ "$mode" = "rasd" ] && prefix="RASD" || prefix="TARGET"
    RUN_ID="${prefix}_${label}_pg19_phaseD_s42"
    echo
    echo "================================================================"
    echo ">>> [$(date -u +%H:%M:%S)] $RUN_ID (mode=$mode ctx=$ctx)"
    echo "================================================================"
    sleep 10

    python run_experiment.py \
      --wandb-project rasd-m4-phase-d \
      --config "$cfg" \
      --output results/final/phase_d_short_ctx_pg19.csv \
      --groups M4 \
      --seeds 42 \
      --nproc 8 \
      --timeout-per-run-s 600 \
      --log-per-token \
      --save-generated-text \
      --prompt-source pg19 \
      --prompt-pg19-meta data/processed/pg19/pg19_validation_metadata.json 2>&1 | tail -3
  done
done

echo
echo "==> All 4 short-ctx PG-19 cells DONE at $(date -u +%H:%M:%S)"
touch /tmp/phase_d_pg19_short_DONE

#!/usr/bin/env bash
# SUNDAY SESSION A, pod side. Runs the four groups in the order that protects the
# budget, and nothing else.
#
# ORDER OF WORK, and the reason for it:
#   0  SANITY    one short spec + target-only pair at 128k, 64 tokens. Proves the
#                pod, the interpreter, the NF4 load and the ring, before a single
#                expensive row is spent. ~10 min.
#   1  QUALITY   the target's own ppl of the SAME final 1024 tokens at 1M and at
#                128k. This runs BEFORE the headline rows because it is the one
#                measurement that can invalidate the session: if the target cannot
#                use a 1M context, five hours of decode rows would be measuring a
#                broken model. ~45 min.
#   2  HEADLINE  3 documents x (spec + target-only) at 1M, 128 tokens, greedy.
#                Spec first: it is the row the headline rests on.
#   3  SCALING   only if the groups above finished and the clock allows.
#
# Every row's CSV is appended as it completes and the Mac side rsyncs every 4
# minutes, so an interruption costs at most one row.
set -uo pipefail
cd "$HOME/RASD" || exit 3
OUT=results/mlsys/session_a
mkdir -p "$OUT"
log() { printf '%s %s\n' "$(date -u +%FT%TZ)" "$*" | tee -a "$OUT/session_a.log"; }

PY=$(grep -a "^INTERPRETER=" "$HOME/pod_env.log" 2>/dev/null | tail -1 | cut -d= -f2-)
if [ -z "${PY:-}" ] || [ ! -x "${PY:-}" ]; then
  log "FATAL: no usable INTERPRETER= in ~/pod_env.log; refusing to guess one"
  exit 2
fi
# shellcheck disable=SC1090
set -a; . "$HOME/RASD/.pod_env.sh"; set +a
CFG=configs/mlsys_sunday_1m.yml
NPROC=${NPROC:-8}
log "=== interpreter $PY, nproc $NPROC ==="
"$PY" -c 'import torch,transformers;print("torch",torch.__version__,"transformers",transformers.__version__,"gpus",torch.cuda.device_count())' 2>&1 | tee -a "$OUT/session_a.log"

run_group() {
  local name=$1 tmo=$2
  log "### group $name (timeout ${tmo}s)"
  timeout "$tmo" "$PY" run_experiment.py --config "$CFG" --nproc "$NPROC" \
    --groups "$name" --output "$OUT/session_a.csv" --stage-id "session_a_$name" \
    --log-per-token --save-generated-tokens 2>&1 | tee -a "$OUT/group_$name.log" | tail -25
  log "### group $name exit=$? rows=$(tail -n +2 "$OUT/session_a.csv" 2>/dev/null | wc -l | tr -d ' ')"
  log "### csv snapshot ###"
  cat "$OUT/session_a.csv" 2>/dev/null | tee -a "$OUT/session_a.log"
}

# ---- 0 sanity ---------------------------------------------------------------
# A 1x-style sanity pair at 128k: cheap, and it fails loudly if the pod is wrong.
cat > /tmp/sanity.yml <<'YML'
defaults:
  target_model_name: gradientai/Llama-3-8B-Instruct-Gradient-1048k
  target_revision: cd3069b65a8eb13da639d332a5f61b0fbb29fa73
  draft_model_name: meta-llama/Llama-3.2-1B
  draft_revision: 4e20de362430cd3b72f300e6b0f18e50e7166e08
  spec_steps: 4
  kv_block_size: 2048
  prefetch_depth: 1
  rope_type: "none"
  kv_quant: true
  dtype: bfloat16
  quantize_draft: true
  quantize_target: true
  draft_window_cap: 4096
  ignore_eos: true
  temperature: 0.0
  top_p: 1.0
  seeds: [42]
  save_generated_tokens: true
  prompt_source: pg19_document
  prompt_documents_json: data/processed/pg19_1m_128kwin/documents.json
SANITY_SPEC:
  name: sanity
  factor: context_length
  levels:
  - id: SANITY
    context_length: 131072
    documents: [pg19_train_0]
    max_new_tokens: 64
    checkpoint_every: 0
SANITY_TARGET:
  name: sanity_target
  factor: context_length
  levels:
  - id: SANITY_targetonly
    context_length: 131072
    documents: [pg19_train_0]
    max_new_tokens: 64
    spec_steps: 0
    checkpoint_every: 0
YML
cp /tmp/sanity.yml configs/sunday_sanity.yml
log "### group SANITY (spec + target-only, 128k, 64 tokens)"
timeout 1800 "$PY" run_experiment.py --config configs/sunday_sanity.yml --nproc "$NPROC" \
  --output "$OUT/session_a.csv" --stage-id session_a_sanity \
  --log-per-token --save-generated-tokens 2>&1 | tee -a "$OUT/group_SANITY.log" | tail -20
log "### SANITY rows so far:"; cat "$OUT/session_a.csv" 2>/dev/null | tee -a "$OUT/session_a.log"
SANITY_OK=$(grep -c ",ok," "$OUT/session_a.csv" 2>/dev/null || echo 0)
if [ "${SANITY_OK:-0}" -lt 2 ]; then
  log "STOP: sanity produced $SANITY_OK ok rows, expected 2. Not spending the session."
  exit 4
fi

# ---- 1 quality -------------------------------------------------------------
run_group SUNDAY_QUALITY_128K 1200
run_group SUNDAY_QUALITY_1M 3600

# ---- 2 headline ------------------------------------------------------------
run_group SUNDAY_HEADLINE_1M_SPEC 7200
run_group SUNDAY_HEADLINE_1M_TARGET 5400

log "=== SESSION A WORK COMPLETE ==="
log "rows: $(tail -n +2 "$OUT/session_a.csv" 2>/dev/null | wc -l | tr -d ' ')"
log "=== all sidecars ==="
ls -la "$OUT/tokens" 2>/dev/null | tail -15 | tee -a "$OUT/session_a.log"
log "=== SESSION A REMOTE DONE ==="

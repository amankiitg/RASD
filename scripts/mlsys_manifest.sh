#!/usr/bin/env bash
# Run the MLSys manifest stages in order, on the pod.
#
# Reads configs/mlsys_manifest.yml for stage order and outputs. Every stage:
#   * is preceded by a cost/ceiling check, and refuses to start if the projected
#     spend would cross the ceiling (writing a note instead of silently running)
#   * appends a row to the cost ledger
#   * writes an interim report line so results survive a cut-short run
#   * is skipped, not retried, if the coherence gate did not clear it
#
# Stages live server-side so a dropped SSH session does not kill the run.

set -uo pipefail
cd "$(dirname "$0")/.."
OUT=${MLSYS_OUT:-results/mlsys}
COST_LOG=$OUT/gpu_hours.csv
RATE=${NODE_RATE_PER_HOUR:-22.32}
MAX_COST=${MLSYS_MAX_COST_USD:-400}
mkdir -p "$OUT"
[ -f "$COST_LOG" ] || echo "stage,wall_seconds,nproc,gpu_hours,node_cost_usd" > "$COST_LOG"

spend() { awk -F, 'NR>1{c+=$5} END{printf "%.2f", c+0}' "$COST_LOG"; }

interim() { printf '%s %s\n' "$(date -u +%FT%TZ)" "$*" >> "$OUT/RUN_LOG.txt"; }

# Run a stage, recording wall time and cost. Never truncates an existing ledger.
stage() {   # $1=name  $2=timeout_s  $3..=cmd
  local name=$1 tmo=$2; shift 2
  local before after wall cost
  before=$(spend)
  if awk -v s="$before" -v m="$MAX_COST" 'BEGIN{exit !(s>=m)}'; then
    interim "SKIPPED name=$name reason=ceiling spend=\$$before"
    echo "SKIP $name (ceiling \$$MAX_COST reached at \$$before)"
    return 9
  fi
  local t0 t1
  t0=$(date -u +%s)
  echo "=== stage $name ==="
  # `if cmd; then rc=0; else rc=$?; fi` — NOT `cmd; rc=$?`, which under
  # `set -e` never reaches the guard because the shell exits first. The command
  # must be run EXACTLY ONCE here; running it a second time to capture rc would
  # execute the whole stage twice.
  local rc
  if timeout "$tmo" "$@"; then rc=0; else rc=$?; fi
  t1=$(date -u +%s)
  wall=$((t1-t0))
  cost=$(awk -v w="$wall" -v r="$RATE" 'BEGIN{printf "%.2f", w/3600.0*r}')
  echo "$name,$wall,8,$(awk -v w="$wall" 'BEGIN{printf "%.4f", w/3600.0*8}'),$cost" >> "$COST_LOG"
  if [ $rc -eq 0 ]; then
    interim "STAGE_OK name=$name wall=${wall}s cost=\$$cost cumulative=\$$(spend)"
  else
    interim "STAGE_FAILED name=$name rc=$rc wall=${wall}s cost=\$$cost"
  fi
  after=$(spend)
  echo "--- $name rc=$rc wall=${wall}s cost=\$$cost cumulative=\$$after"
  return 0        # a failed stage does not abort the ladder
}

gate_pass() {   # $1=candidate name -> 0 if the gate cleared it
  local c=$1
  python3 - "$OUT/coherence_gate.csv" "$c" <<'PY'
import csv, sys
try:
    for r in csv.DictReader(open(sys.argv[1])):
        if r["candidate"] == sys.argv[2]:
            sys.exit(0 if r["gate_pass"] == "True" else 1)
except FileNotFoundError:
    sys.exit(2)
sys.exit(1)
PY
}

interim "=== MANIFEST START rate=\$$RATE ceiling=\$$MAX_COST ==="
echo "spend before manifest: \$$(spend) of \$$MAX_COST"

# ---- stage 1: the gate (everything above 128k depends on it) --------------
stage coherence_gate 10800 python3 scripts/mlsys_coherence_gate.py \
  --candidates configs/mlsys_rope_candidates.json \
  --pg19-meta data/processed/pg19_llama3/pg19_validation_metadata.json \
  --out "$OUT/coherence_gate.csv" --gen-dir "$OUT/gate_generated"

if [ ! -s "$OUT/coherence_gate.csv" ]; then
  interim "ABORT: the coherence gate produced no verdicts; refusing to run any >128k stage"
  echo "ABORT: no gate verdicts"
  exit 1
fi
echo "gate verdicts:"
python3 - "$OUT/coherence_gate.csv" <<'PY'
import csv, sys
for r in csv.DictReader(open(sys.argv[1])):
    print(f"  {r['candidate']:<28} {'PASS' if r['gate_pass']=='True' else 'FAIL'}  {r['gate_reason'][:70]}")
PY

# ---- stage 2: natural-text f1 at the native 128k window -------------------
stage natural_f1_128k 21600 python3 run_experiment.py \
  --config configs/mlsys_natural_f1_128k.yml \
  --output "$OUT/natural_f1_128k.csv" --stage-id natural_f1_128k \
  --log-per-token --memory-trace --save-generated-text

# ---- stages 3-4: only configurations the gate cleared ---------------------
for cfg_stage in "natural_spec_gated:configs/mlsys_natural_gated.yml" \
                 "synthetic_spec_gated:configs/mlsys_synthetic_gated.yml"; do
  name=${cfg_stage%%:*}; cfg=${cfg_stage##*:}
  if [ ! -f "$cfg" ]; then
    interim "SKIPPED name=$name reason=config-missing path=$cfg"
    continue
  fi
  stage "$name" 28800 python3 run_experiment.py \
    --config "$cfg" --output "$OUT/${name}.csv" --stage-id "$name" \
    --log-per-token --memory-trace --save-generated-text
done

# ---- stage 5: vLLM baseline ----------------------------------------------
stage vllm_ladder 14400 python3 scripts/mlsys_vllm_baseline.py \
  --out "$OUT/vllm_baseline.csv" --attempts 3

# ---- reporting -----------------------------------------------------------
stage acceptance_report 1800 python3 scripts/mlsys_cluster_bootstrap.py \
  --traces "$OUT/per_token" --out "$OUT/acceptance_bootstrap.csv" --window 15

interim "=== MANIFEST END spend=\$$(spend) ==="
echo "total node cost: \$$(spend)"

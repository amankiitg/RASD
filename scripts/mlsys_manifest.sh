#!/usr/bin/env bash
# Run the MLSys manifest stages in order, on the pod.
#
# Reads configs/mlsys_manifest.yml for stage order, cost projections and
# outputs. Every stage:
#   * is preceded by a projected-cost check; a stage projected above
#     ask_before_stage_over_usd does NOT start unless its id is listed in
#     MLSYS_APPROVED_STAGES, and the refusal is written to the run log so the
#     operator sees what was declined and why
#   * appends a row to the cost ledger
#   * writes an interim report line so results survive a cut-short run
#   * is skipped, not retried, if the coherence gate did not clear it
#
# Stages live server-side so a dropped SSH session does not kill the run.

set -uo pipefail
cd "$(dirname "$0")/.."
OUT=${MLSYS_OUT:-results/mlsys}
MANIFEST=${MLSYS_MANIFEST:-configs/mlsys_manifest.yml}
COST_LOG=$OUT/gpu_hours.csv
RATE=${NODE_RATE_PER_HOUR:-22.32}
ASK_OVER=${MLSYS_ASK_OVER_USD:-300}
# Comma-separated stage ids the operator has explicitly approved to run while
# projected above ASK_OVER. Empty means "approve nothing": the run stops and
# reports rather than spending on a stage nobody signed off.
APPROVED=${MLSYS_APPROVED_STAGES:-}
mkdir -p "$OUT"
[ -f "$COST_LOG" ] || echo "stage,wall_seconds,nproc,gpu_hours,node_cost_usd" > "$COST_LOG"

spend() { awk -F, 'NR>1{c+=$5} END{printf "%.2f", c+0}' "$COST_LOG"; }

interim() { printf '%s %s\n' "$(date -u +%FT%TZ)" "$*" >> "$OUT/RUN_LOG.txt"; }

# Projected cost per stage, read from the manifest so the number the guard uses
# is the same number the plan quotes.
est_cost() {   # $1=stage id -> dollars
  python3 - "$MANIFEST" "$1" <<'PY'
import sys, yaml
m = yaml.safe_load(open(sys.argv[1]))
for s in m["stages"]:
    if s["id"] == sys.argv[2]:
        print(s.get("est_cost_usd", 0)); raise SystemExit
print(0)
PY
}

approved() { [ -n "$APPROVED" ] && [[ ",$APPROVED," == *",$1,"* ]]; }

# Run a stage, recording wall time and cost. Never truncates an existing ledger.
stage() {   # $1=name  $2=timeout_s  $3..=cmd
  local name=$1 tmo=$2; shift 2
  local est
  est=$(est_cost "$name")
  # `awk` prints 1 when est > ASK_OVER. The previous version tested
  # `[ 0 -eq <that> ]`, which is true for the CHEAP stages, so it refused cheap
  # stages and let the expensive ones run unapproved — the opposite of the
  # declared guard. Test the condition directly instead of through an integer
  # comparison nobody can read.
  if awk -v e="$est" -v a="$ASK_OVER" 'BEGIN{exit !(e > a)}' && ! approved "$name"; then
    # Above the approval threshold and not approved: refuse, and say so.
    interim "SKIPPED name=$name reason=needs_approval projected=\$$est over=\$$ASK_OVER"
    echo "SKIP $name (projected \$$est > \$$ASK_OVER, not in MLSYS_APPROVED_STAGES)"
    return 9
  fi
  local t0 t1 wall cost rc
  t0=$(date -u +%s)
  echo "=== stage $name (projected \$$est) ==="
  # `if cmd; then rc=0; else rc=$?; fi` — NOT `cmd; rc=$?`, which under
  # `set -e` never reaches the guard because the shell exits first. The command
  # must be run EXACTLY ONCE here; running it a second time to capture rc would
  # execute the whole stage twice.
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
  echo "--- $name rc=$rc wall=${wall}s cost=\$$cost cumulative=\$$(spend)"
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

# Acceptance accounting + document-level intervals + losslessness for a stage's
# output. Run after every spec stage, on the pod, so a failure is visible before
# the next stage starts rather than at report time.
report_spec_stage() {   # $1=stage csv path  $2=stage id
  local csv=$1 name=$2
  stage "${name}_losslessness" 3600 python3 scripts/mlsys_losslessness.py \
    --results "$csv" --tokens-dir "$(dirname "$csv")/tokens" \
    --out "$(dirname "$csv")/${name}_losslessness.csv"
  # Group by level_id, NOT context_length: the speculative and target-only arms
  # share a context_length, so grouping by it pooled them and averaged the
  # target-only acceptance (hard-coded 0.0) into the speculative number.
  stage "${name}_doc_intervals" 1800 python3 scripts/mlsys_document_bootstrap.py \
    --results "$csv" --group-by level_id \
    --out "$(dirname "$csv")/${name}_doc_intervals.csv"
  # --window: the plan compares rungs over a common window of rounds so cells
  # with different output lengths are on equal footing.
  stage "${name}_round_acceptance" 1800 python3 scripts/mlsys_cluster_bootstrap.py \
    --traces "$(dirname "$csv")/per_token" --window 15 \
    --out "$(dirname "$csv")/${name}_acceptance_bootstrap.csv"
}

DOCS=${MLSYS_DOCUMENTS_JSON:-data/processed/pg19_docs/documents.json}
interim "=== MANIFEST START rate=\$$RATE ask_over=\$$ASK_OVER approved='$APPROVED' ==="
echo "spend before manifest: \$$(spend)"

# ---- S0: calibrate the gate on real weights. MUST be first. ---------------
stage gate_calibration 14400 python3 scripts/mlsys_coherence_gate.py \
  --candidates configs/mlsys_gate_controls.json \
  --pg19-meta "$DOCS" \
  --out "$OUT/gate_calibration.csv" --gen-dir "$OUT/gate_calibration_generated"

if [ ! -s "$OUT/gate_calibration.csv" ]; then
  interim "ABORT: gate calibration produced no verdicts; the gate is uncalibrated"
  echo "ABORT: no calibration verdicts"
  exit 1
fi
# A positive control that fails means the GATE is wrong; a negative control that
# passes means the gate cannot detect a known-bad configuration. Either way no
# candidate may be gated until it is fixed.
if ! python3 scripts/mlsys_gate_calibration_check.py "$OUT/gate_calibration.csv"; then
  interim "ABORT: gate calibration failed its controls; refusing to gate candidates"
  echo "ABORT: calibration controls disagree"
  exit 1
fi

# ---- S1: implementation validation (RASD vs vLLM, native 128k) -----------
# The RASD/target-only sides of this cross-check come from the S4 128k stage's
# own rows and token sidecars; the vLLM side is measured here. Both are then
# compared by scripts/mlsys_losslessness.py at report time, so a disagreement
# between two implementations is visible rather than averaged away.
stage impl_validation 21600 python3 scripts/mlsys_vllm_baseline.py \
  --out "$OUT/impl_validation.csv" --attempts 3

# ---- S2: correction-note evidence ----------------------------------------
# Reuses the gate's measurement path with Llama-2 candidates, so the numbers the
# correction note quotes come from the same code the ladder's gate uses.
stage correction_note_evidence 14400 python3 scripts/mlsys_coherence_gate.py \
  --candidates configs/mlsys_correction_candidates.json \
  --pg19-meta "$DOCS" \
  --out "$OUT/correction_note_evidence.csv" \
  --gen-dir "$OUT/correction_note_generated"

# ---- S3: the gate, now calibrated ---------------------------------------
stage coherence_gate 21600 python3 scripts/mlsys_coherence_gate.py \
  --candidates configs/mlsys_rope_candidates.json \
  --pg19-meta "$DOCS" \
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

# ---- S4: natural-text f1 at the native 128k window ----------------------
for spec_stage in "natural_f1_128k:configs/mlsys_natural_f1_128k.yml" \
                  "natural_f1_128k_diverse:configs/mlsys_natural_f1_128k_diverse.yml"; do
  name=${spec_stage%%:*}; cfg=${spec_stage##*:}
  if [ ! -f "$cfg" ]; then
    interim "SKIPPED name=$name reason=config-missing path=$cfg"
    continue
  fi
  stage "$name" 43200 python3 run_experiment.py \
    --config "$cfg" --output "$OUT/${name}.csv" --stage-id "$name" \
    --log-per-token --memory-trace --save-generated-text \
    --save-generated-tokens
  report_spec_stage "$OUT/${name}.csv" "$name"
done

# ---- S5/S6: only configurations the gate cleared -------------------------
for spec_stage in "natural_spec_gated:configs/mlsys_natural_gated.yml" \
                  "synthetic_spec_gated:configs/mlsys_synthetic_gated.yml"; do
  name=${spec_stage%%:*}; cfg=${spec_stage##*:}
  if [ ! -f "$cfg" ]; then
    interim "SKIPPED name=$name reason=config-missing path=$cfg"
    continue
  fi
  # ENFORCE the gate. `require_gate_verdict: pass` was declared in the manifest
  # and never implemented: a gate that failed every candidate still let every
  # >128k stage run. The filter writes a config containing only the levels whose
  # configuration actually passed; a level with no matching verdict is dropped.
  filtered="$OUT/${name}.gated.yml"
  if ! python3 scripts/mlsys_gate_filter.py --gate "$OUT/coherence_gate.csv" \
        --config "$cfg" --out "$filtered" --allow-empty; then
    interim "SKIPPED name=$name reason=no_gate_passing_config"
    echo "SKIP $name (no configuration passed the gate)"
    continue
  fi
  stage "$name" 86400 python3 run_experiment.py \
    --config "$filtered" --output "$OUT/${name}.csv" --stage-id "$name" \
    --log-per-token --memory-trace --save-generated-text \
    --save-generated-tokens
  report_spec_stage "$OUT/${name}.csv" "$name"
done

# ---- S7: vLLM baseline ---------------------------------------------------
stage vllm_ladder 21600 python3 scripts/mlsys_vllm_baseline.py \
  --out "$OUT/vllm_baseline.csv" --attempts 3

interim "=== MANIFEST END spend=\$$(spend) ==="
echo "total node cost: \$$(spend)"

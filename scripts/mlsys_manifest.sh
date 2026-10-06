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
# Proof-of-start markers for stages that actually began. Kept OUT of results/:
# results/ is pulled, merged and published, and a scratch bookkeeping file has
# no business travelling with the data.
RAN_DIR=${MLSYS_RAN_DIR:-$(mktemp -d)}

# Interpreter used for every helper call in this file and for every stage. Bare
# `python3` is whatever is first on PATH, which on the pod is not necessarily the
# environment holding torch/transformers/PyYAML -- and a helper that dies because
# its import failed was previously indistinguishable from "the value is empty".
# Resolved once, then exported so the stages inherit it.
resolve_python() {
  local c
  if [ -n "${MLSYS_PYTHON:-}" ]; then printf '%s' "$MLSYS_PYTHON"; return; fi
  for c in "$HOME/miniconda3/envs/rasd/bin/python" \
           "$HOME/miniconda3/bin/python" /opt/conda/bin/python \
           "$(command -v python3 2>/dev/null)"; do
    if [ -n "$c" ] && [ -x "$c" ] && "$c" -c 'import yaml, torch' 2>/dev/null; then
      printf '%s' "$c"; return
    fi
  done
  for c in "$HOME/miniconda3/envs/rasd/bin/python" /opt/conda/bin/python \
           "$(command -v python3 2>/dev/null)"; do
    if [ -n "$c" ] && [ -x "$c" ]; then printf '%s' "$c"; return; fi
  done
  printf '%s' python3
}
PY=$(resolve_python)
export PY
export PATH="$(dirname "$PY"):$PATH"
RATE=${NODE_RATE_PER_HOUR:-22.32}
ASK_OVER=${MLSYS_ASK_OVER_USD:-300}
# Comma-separated stage ids the operator has explicitly approved to run while
# projected above ASK_OVER. Empty means "approve nothing": the run stops and
# reports rather than spending on a stage nobody signed off.
# Defaults to the operator's 2026-10-06 approval. Override with
# MLSYS_APPROVED_STAGES at launch; an empty value means "approve nothing".
APPROVED=${MLSYS_APPROVED_STAGES:-gate_calibration,engine_cap_smoke,coherence_gate,correction_note_evidence,natural_f1_128k,impl_validation,natural_spec_gated_256k,vllm_ladder}
# An EXPLICIT allowlist. MLSYS_APPROVED_STAGES alone cannot narrow a run: it is
# a cost guard, so it refuses a stage only when the projection exceeds
# ask_before_stage_over_usd. With two ids "approved", ten of the twelve stages --
# including the 256k rung at $290 -- are still under that threshold and would
# run. MLSYS_ONLY_STAGES is the authority, whatever a stage's projection.
#
# It DEFAULTS to the operator's approved set rather than to empty, and an empty
# value REFUSES EVERY STAGE. "Unset" must not be able to mean "run everything":
# a launch script that forgot the variable would otherwise run the whole
# campaign on the strength of an omission.
DEFAULT_ONLY=gate_calibration,engine_cap_smoke,coherence_gate,correction_note_evidence,natural_f1_128k,impl_validation,rope_intervention_128k,natural_spec_gated_256k,vllm_ladder
ONLY=${MLSYS_ONLY_STAGES-$DEFAULT_ONLY}
ONLY_EXPLICIT=0
[ -n "${MLSYS_ONLY_STAGES+set}" ] && ONLY_EXPLICIT=1

# Aggregate ceiling for this session. The per-stage guard bounds one stage; this
# bounds the session, which is the number the operator actually agreed to.
MAX_COST=${MLSYS_MAX_COST_USD:-850}
mkdir -p "$OUT"
[ -f "$COST_LOG" ] || echo "stage,wall_seconds,nproc,gpu_hours,node_cost_usd" > "$COST_LOG"

spend() { awk -F, 'NR>1{c+=$5} END{printf "%.2f", c+0}' "$COST_LOG"; }

# ---- wall-clock watchdog -------------------------------------------------
# Without this a run that hangs still bills until someone notices; the only
# mention of a watchdog before was a comment. The clock starts when the manifest
# does. A stage may declare a larger `watchdog_hours` (the 512k rung needs 40h
# and runs in its own session), which raises the limit from that stage onward.
_manifest_field() {   # $1=stage id (or __meta__)  $2=field
  "$PY" - "$MANIFEST" "$1" "$2" <<'PYW'
import sys, yaml
m = yaml.safe_load(open(sys.argv[1]))
want, field = sys.argv[2], sys.argv[3]
if want == "__meta__":
    print(m["meta"].get(field) or "")
    raise SystemExit
for s in m["stages"]:
    if s["id"] == want:
        v = s.get(field)
        print("" if v is None else v)
        raise SystemExit
print("")
PYW
}

START_EPOCH=$(date -u +%s)
WATCHDOG_SKIPS=0
COST_UNKNOWN=0
STAGE_FAILURES=0
BUDGET_SKIPS=0
STAGE_INVALID=0
ONLY_SKIPS=0
WATCHDOG=${MLSYS_MAX_HOURS:-$(_manifest_field __meta__ max_hours)}
WATCHDOG=${WATCHDOG:-20}
elapsed_hours() { awk -v n="$(date -u +%s)" -v s="$START_EPOCH" 'BEGIN{printf "%.3f", (n-s)/3600.0}'; }

interim() { printf '%s %s\n' "$(date -u +%FT%TZ)" "$*" >> "$OUT/RUN_LOG.txt"; }

# Projected cost per stage, read from the manifest so the number the guard uses
# is the same number the plan quotes.
# Projected cost per stage. FAIL CLOSED: if the manifest cannot be parsed, or the
# stage is not in it, or the estimate is missing or unreadable, this prints
# nothing and returns non-zero, and `stage` refuses to start. The earlier version
# printed `0` on every failure path, so a missing PyYAML or a typo in the stage
# id made every stage look FREE and the cost guard approved all of them.
est_cost() {   # $1=stage id -> dollars on stdout; non-zero if unknown
  "$PY" - "$MANIFEST" "$1" <<'PY'
import sys
try:
    import yaml
except ImportError:
    sys.exit(3)                      # no YAML parser: cannot cost it
try:
    m = yaml.safe_load(open(sys.argv[1]))
except Exception:
    sys.exit(4)                      # unparseable manifest
want = sys.argv[2]
for s in m.get("stages", []):
    if s["id"] == want:
        break
else:
    # Helper stages are priced in `helpers` and keyed either by their exact id
    # or by a suffix (`_losslessness` and friends, which are named per parent).
    for h in m.get("helpers", []):
        hid = h["id"]
        if want == hid or (hid.startswith("_") and want.endswith(hid)):
            break
    else:
        sys.exit(7)                  # stage id not in the manifest
    s = h
v = s.get("est_cost_usd")
if v is None or str(v).strip() == "":
    sys.exit(5)                      # no projection recorded
try:
    float(v)
except (TypeError, ValueError):
    sys.exit(6)                      # unreadable projection
print(v)
raise SystemExit(0)
PY
}

MANIFEST_STAGE_IDS=$("$PY" - "$MANIFEST" <<'PYIDS'
import sys, yaml
print(",".join(s["id"] for s in yaml.safe_load(open(sys.argv[1]))["stages"]))
PYIDS
)

# Derived sub-stages are named "<stage id>_<suffix>", so a prefix match has to be
# allowed for them -- but NOT for real stage ids, or approving `natural_f1_128k`
# would silently admit `natural_f1_128k_diverse`. Real ids therefore require an
# exact match; only names that are not manifest ids may match by prefix.
on_list() {   # $1=name  $2=comma list
  local name=$1 list=$2 e
  [ -z "$list" ] && return 1
  for e in ${list//,/ }; do
    [ "$name" = "$e" ] && return 0
    case ",$MANIFEST_STAGE_IDS," in
      *",$name,"*) ;;                                   # a real stage id: exact only
      *) case "$name" in "$e"_*) return 0;; esac ;;
    esac
  done
  return 1
}

approved() { on_list "$1" "$APPROVED"; }

# The parent of a helper stage, or "" if the name is not a helper. `_losslessness`
# and friends are named per parent, so a suffix match is how they are found.
helper_parent() {
  local want=$1
  "$PY" - "$MANIFEST" "$want" <<'PYH'
import sys, yaml
m = yaml.safe_load(open(sys.argv[1]))
want = sys.argv[2]
for h in m.get("helpers", []):
    hid = h["id"]
    if want == hid or (hid.startswith("_") and want.endswith(hid)):
        if hid.startswith("_"):
            print(want[: -len(hid)])          # "<parent><suffix>"
        else:
            print(h.get("parent") or "")
        raise SystemExit
print("")
PYH
}

# Some helpers run BEFORE their parent -- the rope-intervention gate is what
# decides whether that stage's arms may run at all. For those the requirement is
# that the parent is APPROVED, not that it has already run.
helper_runs_before_parent() {
  "$PY" - "$MANIFEST" "$1" <<'PYB'
import sys, yaml
m = yaml.safe_load(open(sys.argv[1]))
want = sys.argv[2]
for h in m.get("helpers", []):
    hid = h["id"]
    if want == hid or (hid.startswith("_") and want.endswith(hid)):
        print("1" if h.get("runs_before_parent") else "0")
        raise SystemExit
print("0")
PYB
}

# A helper may run only when its parent is on the allowlist AND actually ran.
helper_allowed() {
  local name=$1 parent
  parent=$(helper_parent "$name")
  [ -z "$parent" ] && return 0                # not a helper: no opinion
  if [ "$parent" != "*" ]; then
    on_list "$parent" "$ONLY" || return 1     # parent not approved
    if [ "$(helper_runs_before_parent "$name")" != "1" ]; then
      [ -f "$RAN_DIR/$parent" ] || return 2   # parent refused or never started
    fi
    # A helper that runs BEFORE its parent additionally needs the parent's own
    # prerequisites met, which the parent's checks handle at its own call site.
  fi
  return 0
}

# Run a stage, recording wall time and cost. Never truncates an existing ledger.
stage() {   # $1=name  $2=timeout_s  $3..=cmd
  local name=$1 tmo=$2; shift 2
  local est rc=0
  # FAIL CLOSED. If the projection cannot be read, refuse: an unreadable
  # estimate is not a zero estimate, and the guard below is the only thing
  # standing between an unapproved stage and the budget.
  est=$(est_cost "$name") || rc=$?
  if [ "$rc" -ne 0 ] || [ -z "$est" ]; then
    interim "REFUSED name=$name reason=no_cost_estimate rc=$rc"
    echo "REFUSE $name: no readable cost projection (rc=$rc); refusing to spend blind"
    COST_UNKNOWN=$((COST_UNKNOWN + 1))
    return 9
  fi
  # `awk` prints 1 when est > ASK_OVER. The previous version tested
  # `[ 0 -eq <that> ]`, which is true for the CHEAP stages, so it refused cheap
  # stages and let the expensive ones run unapproved — the opposite of the
  # declared guard. Test the condition directly instead of through an integer
  # comparison nobody can read.
  # A helper stage is gated by its PARENT, before anything else. Validating a
  # stage that was refused or never started reports a failure for something that
  # never happened, and does it in a way that reads like a real defect.
  helper_allowed "$name"; local hrc=$?
  if [ "$hrc" = "1" ]; then
    interim "SKIPPED_VALIDATION name=$name parent=$(helper_parent "$name") reason=parent_not_approved"
    echo "SKIP $name: parent $(helper_parent "$name") is not on the allowlist"
    return 9
  elif [ "$hrc" = "2" ]; then
    interim "SKIPPED_VALIDATION name=$name parent=$(helper_parent "$name") reason=parent_did_not_run"
    echo "SKIP $name: parent $(helper_parent "$name") did not run"
    return 9
  fi

  # The allowlist outranks the cost guard, and is checked first. An EMPTY
  # allowlist approves nothing: "unset" must not be able to mean "run
  # everything", or a launch that forgot the variable would run the campaign.
  if [ -z "$ONLY" ]; then
    interim "SKIPPED name=$name reason=empty_allowlist"
    echo "SKIP $name: MLSYS_ONLY_STAGES is empty, which approves nothing"
    ONLY_SKIPS=$((ONLY_SKIPS + 1))
    return 9
  fi
  # A helper is allowed by its PARENT's presence on the allowlist, never by its
  # own name. The allowlist is the operator's campaign plan and it names stages;
  # requiring an operator to also list `natural_f1_128k_losslessness` -- or
  # `rope_intervention_gate`, which is what decides whether that stage's arms may
  # run at all -- would mean the plan has to be edited every time a validation
  # step is added, and a forgotten entry refuses the check silently.
  #
  # The rope-intervention gate was refused this way: the stage id is a real
  # stage (not a `_suffix` helper), so the wildcard below did not apply, the
  # gate never ran, and every treated arm was then RECORDED as having failed a
  # gate that was never measured -- a false result in the results table.
  local gate_name=$name gate_parent
  gate_parent=$(helper_parent "$name")
  [ -n "$gate_parent" ] && [ "$gate_parent" != "*" ] && gate_name=$gate_parent
  if ! on_list "$gate_name" "$ONLY"; then
    if [ "$gate_name" != "$name" ]; then
      interim "SKIPPED name=$name reason=parent_not_in_allowlist parent=$gate_name"
      echo "SKIP $name: parent $gate_name is not in MLSYS_ONLY_STAGES"
    else
      interim "SKIPPED name=$name reason=needs_approval (not in MLSYS_ONLY_STAGES=$ONLY)"
      echo "SKIP $name (not in MLSYS_ONLY_STAGES; projected \$$est)"
    fi
    ONLY_SKIPS=$((ONLY_SKIPS + 1))
    return 9
  fi
  if awk -v e="$est" -v a="$ASK_OVER" 'BEGIN{exit !(e > a)}' && ! approved "$name"; then
    # Above the approval threshold and not approved: refuse, and say so.
    interim "SKIPPED name=$name reason=needs_approval projected=\$$est over=\$$ASK_OVER"
    echo "SKIP $name (projected \$$est > \$$ASK_OVER, not in MLSYS_APPROVED_STAGES)"
    return 9
  fi
  # Aggregate session ceiling. The per-stage guard bounds one stage; this bounds
  # the session, which is the number actually agreed to.
  local spent
  spent=$(spend)
  if awk -v s="$spent" -v e="$est" -v m="$MAX_COST" 'BEGIN{exit !(s + e > m)}'; then
    interim "BUDGET_REFUSED name=$name spent=\$$spent projected=\$$est cap=\$$MAX_COST"
    echo "REFUSE $name: \$$spent spent + \$$est projected > \$$MAX_COST session cap"
    BUDGET_SKIPS=$((BUDGET_SKIPS + 1))
    return 9
  fi

  # A stage may raise the watchdog for this session (own-session stages).
  local wd est_h el
  wd=$(_manifest_field "$name" watchdog_hours)
  if [ -n "$wd" ] && awk -v a="$wd" -v b="$WATCHDOG" 'BEGIN{exit !(a>b)}'; then
    WATCHDOG=$wd
    interim "WATCHDOG raised to ${wd}h for stage $name"
  fi

  # Finish-before-watchdog guard: refuse to START a stage whose projection would
  # run past the watchdog. Starting it and being killed mid-stage loses the whole
  # stage's wall time, which is the cost the guard exists to avoid.
  est_h=$(_manifest_field "$name" est_hours)
  el=$(elapsed_hours)
  if [ -n "$est_h" ] && awk -v e="$el" -v h="$est_h" -v w="$WATCHDOG" \
       'BEGIN{exit !(e + h > w)}'; then
    interim "WATCHDOG_REFUSED name=$name elapsed=${el}h projected=${est_h}h limit=${WATCHDOG}h"
    echo "REFUSE $name: ${el}h elapsed + ${est_h}h projected > ${WATCHDOG}h watchdog"
    # Skip this stage but let shorter ones still run; the ledger below turns the
    # skip into a non-zero exit so the run is never reported as clean.
    WATCHDOG_SKIPS=$((WATCHDOG_SKIPS + 1))
    return 9
  fi
  if awk -v e="$el" -v w="$WATCHDOG" 'BEGIN{exit !(e > w)}'; then
    interim "WATCHDOG_TRIPPED elapsed=${el}h limit=${WATCHDOG}h; stopping the ladder"
    echo "WATCHDOG: ${el}h elapsed exceeds ${WATCHDOG}h; stopping"
    exit 3
  fi

  local t0 t1 wall cost rc
  # Proof that this stage STARTED, so its helpers can tell "the parent ran" from
  # "the parent was refused". Written before the command, removed only if the
  # stage refuses.
  : > "$RAN_DIR/$name"
  t0=$(date -u +%s)
  echo "=== stage $name (projected \$$est, ${est_h}h; watchdog ${WATCHDOG}h) ==="
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
    # A stage that exits zero having written nothing usable is not a success:
    # rc only says the process did not crash. The row check below is what makes
    # "the stage passed" a statement about the data.
    interim "STAGE_FAILED name=$name rc=$rc wall=${wall}s cost=\$$cost"
    STAGE_FAILURES=$((STAGE_FAILURES + 1))
  fi
  echo "--- $name rc=$rc wall=${wall}s cost=\$$cost cumulative=\$$(spend)"
  return 0        # a failed stage does not abort the ladder
}

# Does the stage's CSV hold the rows it was supposed to produce, all ok?
# `--abort-on-failure` makes run_experiment stop at the first error row, but the
# row count is what says the stage ran to completion.
check_stage_rows() {   # $1=stage id  $2=csv  $3=expected rows (0 = skip)
  local name=$1 csv=$2 want=$3
  [ "$want" = "0" ] && return 0
  [ -f "$csv" ] || {
    interim "STAGE_INVALID name=$name reason=no_csv"
    echo "INVALID $name: no CSV at $csv"; return 1; }
  local got ok
  got=$("$PY" - "$csv" <<'PY'
import csv, sys
rows = list(csv.DictReader(open(sys.argv[1])))
print(len(rows))
PY
)
  ok=$("$PY" - "$csv" <<'PY'
import csv, sys
rows = list(csv.DictReader(open(sys.argv[1])))
bad = [r.get("run_id", "?") for r in rows
       if str(r.get("status", "")).strip() != "ok"]
print(",".join(bad))
PY
)
  if [ "$got" != "$want" ]; then
    interim "STAGE_INVALID name=$name reason=row_count got=$got want=$want"
    echo "INVALID $name: $got rows, expected $want"
    STAGE_INVALID=$((STAGE_INVALID + 1)); return 1
  fi
  if [ -n "$ok" ]; then
    interim "STAGE_INVALID name=$name reason=non_ok_rows ids=$ok"
    echo "INVALID $name: non-ok rows: $ok"
    STAGE_INVALID=$((STAGE_INVALID + 1)); return 1
  fi
  return 0
}

# Per-run timeout, >= 2x the longest projected single run at that rung. The
# measured target-only decode at 128k is ~4505 s for 1024 tokens, ~9000 s at
# 256k and ~18100 s at 512k, and the target-only arms are the longest runs in
# every stage. The default (3600 s) would kill them mid-run and record an error
# row, i.e. it would spend the whole stage and produce nothing.
per_run_timeout() {   # $1=context length
  case "$1" in
    131072) echo 9000 ;;
    262144) echo 18000 ;;
    524288) echo 36200 ;;
    *)      echo 36200 ;;   # gate stages carry 512k candidates
  esac
}

# How many rows the stage is supposed to write: ask the planner, do not guess.
# A planner failure means the count is unknown, which is a refusal (see the
# fail-closed note on est_cost) rather than a skipped check.
expected_rows() {   # $1=config  $2=groups
  "$PY" run_experiment.py --config "$1" --groups $2 --dry-run 2>/dev/null \
    | awk 'BEGIN{sep=0} /^-{10,}$/{sep=1; next} sep && NF>=3 {n++} END{print n+0}'
}

gate_pass() {   # $1=candidate name  $2=gate csv (default: the ladder's gate)
  local c=$1 csv=${2:-$OUT/coherence_gate.csv}
  "$PY" - "$csv" "$c" <<'PY'
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
  stage "${name}_losslessness" 3600 "$PY" scripts/mlsys_losslessness.py \
    --results "$csv" --tokens-dir "$(dirname "$csv")/tokens" \
    --out "$(dirname "$csv")/${name}_losslessness.csv"
  # Group by the stratum the PAIRED ARMS SHARE, which is the rung
  # (context_length). Grouping by level_id separated the speculative rows from
  # their target-only counterparts, so no pair could ever form and the file
  # contained no `paired_speedup` row at all -- while still looking successful.
  # The reason level_id was chosen (a single mean over both arms pools the
  # target-only structural zeros into the speculative number) is now handled
  # inside the script: the marginal means are computed PER ARM, so the stratum
  # can be shared for pairing without the estimates being mixtures.
  stage "${name}_doc_intervals" 1800 "$PY" scripts/mlsys_document_bootstrap.py \
    --results "$csv" --group-by context_length --ratio-metric decode_tps \
    --out "$(dirname "$csv")/${name}_doc_intervals.csv"
  # --window: the plan compares rungs over a common window of rounds so cells
  # with different output lengths are on equal footing.
  stage "${name}_round_acceptance" 1800 "$PY" scripts/mlsys_cluster_bootstrap.py \
    --traces "$(dirname "$csv")/per_token" --window 15 \
    --out "$(dirname "$csv")/${name}_acceptance_bootstrap.csv"
}

DOCS=${MLSYS_DOCUMENTS_JSON:-data/processed/pg19_docs/documents.json}
# The revisions a vLLM row is compared against. `--target-revision` (singular)
# cannot describe a ladder that runs two targets, and a row pinned to the WRONG
# model's revision is a comparison across models that looks pinned.
ALL_TARGET_REVS="meta-llama/Llama-3.1-8B=d04e592bb4f6aa9cfee91e2e20afa771667e1d4b,meta-llama/Llama-2-7b-hf=01c7f73d771dfac7d292323805ebc428287df4f9"
ALL_DRAFT_REVS="meta-llama/Llama-3.2-1B=4e20de362430cd3b72f300e6b0f18e50e7166e08"
TARGET_REVS_D3="meta-llama/Llama-3.1-8B=d04e592bb4f6aa9cfee91e2e20afa771667e1d4b"
interim "=== MANIFEST START rate=\$$RATE ask_over=\$$ASK_OVER watchdog=${WATCHDOG}h only='${ONLY:-<none>}' approved='$APPROVED' ==="
echo "spend before manifest: \$$(spend)"

# ---- S0: calibrate the gate on real weights. MUST be first. ---------------
stage gate_calibration 14400 "$PY" scripts/mlsys_coherence_gate.py \
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
if ! "$PY" scripts/mlsys_gate_calibration_check.py "$OUT/gate_calibration.csv"; then
  interim "ABORT: gate calibration failed its controls; refusing to gate candidates"
  echo "ABORT: calibration controls disagree"
  exit 1
fi

# ---- engine_cap_smoke: the FIRST speculative-dependent check ------------
# Runs BEFORE any speculative stage. The engine's generation cap has never
# executed on a GPU, and if it is wrong it fails QUIETLY: the generated length
# changes, which invalidates the losslessness comparison and makes the
# acceptance denominator a function of acceptance. So failure here STOPS the run
# rather than being recorded and skipped.
stage engine_cap_smoke 21600 "$PY" run_experiment.py \
  --config configs/mlsys_engine_cap_smoke.yml \
  --output "$OUT/engine_cap_smoke.csv" --stage-id engine_cap_smoke \
  --timeout-per-run-s "$(per_run_timeout 131072)" --abort-on-failure \
  --log-per-token --save-generated-tokens
if [ ! -s "$OUT/engine_cap_smoke.csv" ]; then
  interim "STOP: engine_cap_smoke produced no results; refusing to start speculative stages"
  echo "STOP: cap smoke produced nothing"
  exit 1
fi
check_stage_rows engine_cap_smoke "$OUT/engine_cap_smoke.csv" \
  "$(expected_rows configs/mlsys_engine_cap_smoke.yml "")" || {
    interim "STOP: engine_cap_smoke wrote invalid rows; refusing to start speculative stages"
    echo "STOP: the cap smoke's own rows are not all ok"
    exit 1; }
if ! "$PY" scripts/mlsys_cap_smoke_check.py --results "$OUT/engine_cap_smoke.csv"; then
  interim "STOP: engine_cap_smoke assertions FAILED; refusing to start natural_f1_128k"
  echo "STOP: the generation cap is not enforced; not starting speculative stages"
  exit 1
fi
interim "engine_cap_smoke PASSED: cap enforced and the pair is lossless"
# ---- S2: correction-note evidence ----------------------------------------
# Reuses the gate's measurement path with Llama-2 candidates, so the numbers the
# correction note quotes come from the same code the ladder's gate uses.
stage correction_note_evidence 14400 "$PY" scripts/mlsys_coherence_gate.py \
  --candidates configs/mlsys_correction_candidates.json \
  --pg19-meta "$DOCS" \
  --out "$OUT/correction_note_evidence.csv" \
  --gen-dir "$OUT/correction_note_generated"

# ---- S3: the gate, now calibrated ---------------------------------------
stage coherence_gate 21600 "$PY" scripts/mlsys_coherence_gate.py \
  --candidates configs/mlsys_rope_candidates.json \
  --pg19-meta "$DOCS" \
  --out "$OUT/coherence_gate.csv" --gen-dir "$OUT/gate_generated"

if [ ! -s "$OUT/coherence_gate.csv" ]; then
  interim "ABORT: the coherence gate produced no verdicts; refusing to run any >128k stage"
  echo "ABORT: no gate verdicts"
  exit 1
fi
echo "gate verdicts:"
"$PY" - "$OUT/coherence_gate.csv" <<'PY'
import csv, sys
for r in csv.DictReader(open(sys.argv[1])):
    print(f"  {r['candidate']:<28} {'PASS' if r['gate_pass']=='True' else 'FAIL'}  {r['gate_reason'][:70]}")
PY

# ---- S4: natural-text f1 at the native 128k window ----------------------
for spec_stage in "natural_f1_128k:configs/mlsys_natural_f1_128k.yml:" \
                  "natural_f1_128k_diverse:configs/mlsys_natural_f1_128k_diverse.yml:"; do
  name=${spec_stage%%:*}; rest=${spec_stage#*:}; cfg=${rest%%:*}; grp=${rest##*:}
  if [ ! -f "$cfg" ]; then
    interim "SKIPPED name=$name reason=config-missing path=$cfg"
    continue
  fi
  # --abort-on-failure: stop at the first error row rather than producing a CSV
  # that looks complete. --timeout-per-run-s: the target-only arms at this rung
  # take ~4505 s each, and the default 3600 s would kill every one of them.
  if [ -n "$grp" ]; then
    want=$(expected_rows "$cfg" "$grp")
    stage "$name" 43200 "$PY" run_experiment.py --config "$cfg" --groups $grp \
      --output "$OUT/${name}.csv" --stage-id "$name" \
      --timeout-per-run-s "$(per_run_timeout 131072)" --abort-on-failure \
      --log-per-token --memory-trace --save-generated-text --save-generated-tokens
    src_rc=$?
  else
    want=$(expected_rows "$cfg" "")
    stage "$name" 43200 "$PY" run_experiment.py --config "$cfg" \
      --output "$OUT/${name}.csv" --stage-id "$name" \
      --timeout-per-run-s "$(per_run_timeout 131072)" --abort-on-failure \
      --log-per-token --memory-trace --save-generated-text --save-generated-tokens
    src_rc=$?
  fi
  if [ "$src_rc" = "9" ]; then
    # Refused. Its validation and reporting are skipped too: checking rows that
    # were never produced reports a failure for something that did not happen.
    interim "SKIPPED_VALIDATION name=$name reason=stage_refused"
    continue
  fi
  check_stage_rows "$name" "$OUT/${name}.csv" "${want:-0}"
  report_spec_stage "$OUT/${name}.csv" "$name"
done

# ---- S1: implementation validation, vLLM-only against S4's own rows ------
# No RASD runs here: the comparison uses natural_f1_128k's rows and token
# sidecars, so this stage adds no RASD wall time.
# vLLM spec decoding on the SAME three documents as natural_f1_128k, given the
# EXACT prompt ids those RASD runs fed (from their token sidecars), greedy, at
# the matched 1024-token generation. Anything less and the row is not comparable
# and is flagged unit_matched=no rather than averaged into a ratio.
stage impl_validation 21600 "$PY" scripts/mlsys_vllm_baseline.py \
  --out "$OUT/impl_validation.csv" \
  --prompt-ids-from-sidecars "$OUT/tokens" \
  --documents pg19_train_0,pg19_train_1,pg19_train_115 \
  --context-lengths 131072 --max-new-tokens 1024 --matched-max-new-tokens 1024 \
  --models meta-llama/Llama-3.1-8B \
  --target-revisions "$TARGET_REVS_D3"

# ---- S5: the gated rungs, one stage per rung -----------------------------
# Each is severable so the 512k session can be approved and run on its own
# instance. 512k needs a 40h watchdog: at ~26h the default 20h would kill it.
for gated in "natural_spec_gated_256k:GATED_llama3_f16_256k:43200" \
             "natural_spec_gated_512k:GATED_llama3_f32_512k:144000"; do
  name=${gated%%:*}; rest=${gated#*:}; prefix=${rest%%:*}; tmo=${rest##*:}
  cfg=configs/mlsys_natural_gated.yml
  if [ ! -f "$cfg" ]; then
    interim "SKIPPED name=$name reason=config-missing path=$cfg"
    continue
  fi
  # ENFORCE the gate: keep only levels whose rope configuration passed.
  # NO --allow-empty: a gate that clears no candidate is a HARD FAILURE for this
  # stage, not a skip. Skipping was the silent path -- the ladder moved on and
  # the rung simply never happened, which reads in the report like a rung that
  # was never scheduled rather than one the gate rejected.
  filtered="$OUT/${name}.gated.yml"
  if ! "$PY" scripts/mlsys_gate_filter.py --gate "$OUT/coherence_gate.csv" \
        --config "$cfg" --out "$filtered"; then
    interim "STAGE_INVALID name=$name reason=no_gate_passing_config"
    echo "INVALID $name: the gate cleared no configuration for this rung"
    STAGE_INVALID=$((STAGE_INVALID + 1))
    continue
  fi
  want=$(expected_rows "$filtered" "${prefix}_SPEC ${prefix}_TARGET_FULL ${prefix}_TARGET_SHORT")
  stage "$name" "$tmo" "$PY" run_experiment.py --config "$filtered" \
    --groups ${prefix}_SPEC ${prefix}_TARGET_FULL ${prefix}_TARGET_SHORT \
    --output "$OUT/${name}.csv" --stage-id "$name" \
    --timeout-per-run-s "$(per_run_timeout "$(_manifest_field "$name" rung)")" \
    --abort-on-failure \
    --log-per-token --memory-trace --save-generated-text --save-generated-tokens
  if [ "$?" = "9" ]; then
    interim "SKIPPED_VALIDATION name=$name reason=stage_refused"
    continue
  fi
  check_stage_rows "$name" "$OUT/${name}.csv" "${want:-0}"
  report_spec_stage "$OUT/${name}.csv" "$name"
done

# ---- S6: synthetic arm, per rung, secondary ------------------------------
for syn in "synthetic_spec_gated_128k:NATIVE_synth_128k:43200" \
           "synthetic_spec_gated_256k:GATED_llama3_f16_256k_SYNTH:64800"; do
  name=${syn%%:*}; rest=${syn#*:}; prefix=${rest%%:*}; tmo=${rest##*:}
  cfg=configs/mlsys_synthetic_gated.yml
  [ -f "$cfg" ] || { interim "SKIPPED name=$name reason=config-missing"; continue; }
  filtered="$OUT/${name}.gated.yml"
  # NO --allow-empty here either: a gate that clears no candidate is a hard
  # failure, not a silent skip. Note synthetic_spec_gated_128k needs no verdict
  # (native, in-window), and the filter handles that by construction.
  if ! "$PY" scripts/mlsys_gate_filter.py --gate "$OUT/coherence_gate.csv" \
        --config "$cfg" --out "$filtered"; then
    interim "STAGE_INVALID name=$name reason=no_gate_passing_config"
    echo "INVALID $name: the gate cleared no configuration for this rung"
    STAGE_INVALID=$((STAGE_INVALID + 1))
    continue
  fi
  want=$(expected_rows "$filtered" "${prefix}_SPEC ${prefix}_TARGET_FULL ${prefix}_TARGET_SHORT")
  stage "$name" "$tmo" "$PY" run_experiment.py --config "$filtered" \
    --groups ${prefix}_SPEC ${prefix}_TARGET_FULL ${prefix}_TARGET_SHORT \
    --output "$OUT/${name}.csv" --stage-id "$name" \
    --timeout-per-run-s "$(per_run_timeout "$(_manifest_field "$name" rung)")" \
    --abort-on-failure \
    --log-per-token --memory-trace --save-generated-text --save-generated-tokens
  if [ "$?" = "9" ]; then
    interim "SKIPPED_VALIDATION name=$name reason=stage_refused"
    continue
  fi
  check_stage_rows "$name" "$OUT/${name}.csv" "${want:-0}"
  report_spec_stage "$OUT/${name}.csv" "$name"
done

# ---- S6b: rope_intervention_128k ----------------------------------------
# Pre-registered as C6 in the plan revisions. Context held at 128k; ONLY the
# target's rope moves, so this is the one comparison that isolates a rope
# intervention.
#
# PER-ARM GATING. Each treated arm clears the gate at 128k ON ITS OWN. A gate
# failure is recorded as THAT arm's result and the other arms still run: a
# factor that produces an incoherent target says nothing about acceptance, and
# inferring one factor's fate from another's is an assumption -- exactly the
# assumption the ARM4 f2 cell punished. The native arm is the control and always
# runs; without it there is no contrast to report.
if on_list rope_intervention_128k "$ONLY"; then
  stage rope_intervention_gate 14400 "$PY" scripts/mlsys_coherence_gate.py \
    --candidates configs/mlsys_rope_intervention_candidates.json \
    --pg19-meta "$DOCS" \
    --out "$OUT/rope_intervention_gate.csv" \
    --gen-dir "$OUT/rope_intervention_gate_generated"
  ri_gate_rc=$?
  RI_GATE_MEASURED=0
  if [ "$ri_gate_rc" = "0" ] && [ -s "$OUT/rope_intervention_gate.csv" ]; then
    RI_GATE_MEASURED=1
  else
    # A gate that produced no verdicts has not judged anything. Recording its
    # arms as "did not pass the gate" would turn a plumbing failure into a
    # scientific result -- and this is the exact shape of the ARM4 mistake,
    # where acceptance numbers were published for a target that was broken.
    interim "STAGE_INVALID name=rope_intervention_gate reason=no_verdicts rc=$ri_gate_rc"
    echo "INVALID rope_intervention_gate: no verdicts (rc=$ri_gate_rc); no arm's"
    echo "  fate is known, so none is recorded as a result."
    STAGE_INVALID=$((STAGE_INVALID + 1))
  fi

  RI_GROUPS="RI_native_128k_SPEC"
  for arm in "factor16:RI_llama3_f16_128k_SPEC:RI_llama3_f16_128k" \
             "factor32:RI_llama3_f32_128k_SPEC:RI_llama3_f32_128k"; do
    label=${arm%%:*}; rest=${arm#*:}; grp=${rest%%:*}; cand=${rest##*:}
    [ "$RI_GATE_MEASURED" = "1" ] || continue
    gate_pass "$cand" "$OUT/rope_intervention_gate.csv"
    gp_rc=$?
    if [ "$gp_rc" = "0" ]; then
      interim "GATE_PASS name=rope_intervention_128k arm=$label candidate=$cand"
      RI_GROUPS="$RI_GROUPS $grp"
    elif [ "$gp_rc" = "1" ]; then
      # That arm's RESULT, MEASURED and failed. Recorded, not retried, and the
      # round is not refused.
      interim "RESULT name=rope_intervention_128k arm=$label gate=FAIL candidate=$cand note=no_coherent_target_at_128k"
      echo "RESULT rope_intervention_128k arm=$label: did not pass the gate at 128k; that arm is not run"
    else
      # The gate ran but has no row for this candidate: a config/candidate
      # mismatch, not a verdict.
      interim "STAGE_INVALID name=rope_intervention_128k arm=$label reason=no_gate_row_for_$cand"
      echo "INVALID rope_intervention_128k arm=$label: the gate has no row for $cand"
      STAGE_INVALID=$((STAGE_INVALID + 1))
    fi
  done
  if [ "$RI_GATE_MEASURED" != "1" ]; then
    interim "SKIPPED name=rope_intervention_128k reason=gate_not_measured"
    echo "SKIP rope_intervention_128k: the gate was not measured, so the stage"
    echo "  would spend \$$(_manifest_field rope_intervention_128k est_usd) on arms"
    echo "  whose coherence is unknown."
  else

  want=$(expected_rows configs/mlsys_rope_intervention_128k.yml "$RI_GROUPS")
  stage rope_intervention_128k 43200 "$PY" run_experiment.py \
    --config configs/mlsys_rope_intervention_128k.yml \
    --groups $RI_GROUPS \
    --output "$OUT/rope_intervention_128k.csv" --stage-id rope_intervention_128k \
    --timeout-per-run-s 9000 --abort-on-failure \
    --log-per-token --memory-trace --save-generated-text --save-generated-tokens
  ri_rc=$?
  if [ "$ri_rc" = "9" ]; then
    interim "SKIPPED_VALIDATION name=rope_intervention_128k reason=stage_refused"
  elif [ "$ri_rc" = "0" ]; then
  check_stage_rows rope_intervention_128k "$OUT/rope_intervention_128k.csv" "${want:-0}"
  # Native vs each treated arm, paired by document, difference in alpha_round
  # with the pre-registered 0.05 equivalence margin. An arm whose gate failed
  # has no rows, so its contrast is simply absent -- which is the recorded
  # result for that arm, not a missing measurement to be filled in.
  stage rope_intervention_128k_comparison 1800 "$PY" \
    scripts/mlsys_rope_intervention.py \
    --results "$OUT/rope_intervention_128k.csv" \
    --out "$OUT/rope_intervention_128k_comparison.csv" \
    --metric acceptance_rate --margin 0.05
  fi
  fi
else
  interim "SKIPPED name=rope_intervention_128k reason=needs_approval"
fi

# ---- S7: vLLM baseline ---------------------------------------------------
# Plain (non-speculative) decode at every rung: the production-stack reference
# where the payoff boundary is claimed. impl_validation covers only 128k and
# only speculative decoding.
stage vllm_ladder 21600 "$PY" scripts/mlsys_vllm_baseline.py \
  --out "$OUT/vllm_baseline.csv" \
  --prompt-ids-from-sidecars "$OUT/tokens" \
  --context-lengths 131072 262144 524288 --max-new-tokens 1024 \
  --matched-max-new-tokens 1024 \
  --models meta-llama/Llama-3.1-8B meta-llama/Llama-2-7b-hf \
  --target-revisions "$ALL_TARGET_REVS" \
  --draft-revisions "$ALL_DRAFT_REVS"

if [ "$WATCHDOG_SKIPS" -gt 0 ] || [ "$COST_UNKNOWN" -gt 0 ] \
   || [ "$BUDGET_SKIPS" -gt 0 ] || [ "$STAGE_INVALID" -gt 0 ] \
   || [ "$STAGE_FAILURES" -gt 0 ]; then
  interim "=== MANIFEST INCOMPLETE watchdog_skips=$WATCHDOG_SKIPS cost_unknown=$COST_UNKNOWN budget_skips=$BUDGET_SKIPS stage_invalid=$STAGE_INVALID stage_failures=$STAGE_FAILURES spend=\$$(spend) ==="
  echo "MANIFEST INCOMPLETE:" \
       "watchdog=$WATCHDOG_SKIPS cost-unknown=$COST_UNKNOWN" \
       "budget=$BUDGET_SKIPS invalid=$STAGE_INVALID failed=$STAGE_FAILURES"
  echo "total node cost: \$$(spend)"
  exit 4
fi
# Stages refused by the allowlist are a SCOPE decision, not a failure: the run
# is complete with respect to what was approved. They are reported so the log
# says plainly what did not run and why.
interim "=== MANIFEST END spend=\$$(spend) only_skips=$ONLY_SKIPS approved='$ONLY' ==="
echo "total node cost: \$$(spend)"
[ "$ONLY_SKIPS" -gt 0 ] && echo "stages refused by the allowlist: $ONLY_SKIPS (not run; not approved)"
exit 0

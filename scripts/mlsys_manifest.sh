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
REPO=$PWD
OUT=${MLSYS_OUT:-results/mlsys}
MANIFEST=${MLSYS_MANIFEST:-configs/mlsys_manifest.yml}
COST_LOG=$OUT/gpu_hours.csv
# Proof-of-start markers for stages that actually began. Kept OUT of results/:
# results/ is pulled, merged and published, and a scratch bookkeeping file has
# no business travelling with the data.
# Per-invocation marker directory: a FRESH SUBDIRECTORY under whatever base the
# operator set. `MLSYS_RAN_DIR` names a location, and reusing a location reused
# the state -- a re-run inherited the previous invocation's `.rc` files, so
# `stage_ok` reported a stage as having succeeded in an attempt where it never
# ran.
RAN_BASE=${MLSYS_RAN_DIR:-$(mktemp -d)}
RAN_DIR="$RAN_BASE/attempt.$(date -u +%Y%m%dT%H%M%SZ).$$"
mkdir -p "$RAN_DIR"

# Interpreter used for every helper call in this file and for every stage. Bare
# `python3` is whatever is first on PATH, which on the pod is not necessarily the
# environment holding torch/transformers/PyYAML -- and a helper that dies because
# its import failed was previously indistinguishable from "the value is empty".
# Resolved once, then exported so the stages inherit it.
resolve_python() {
  local c
  # MLSYS_PYTHON first: the watcher resolved and VERIFIED an interpreter before
  # it started this manifest, and its answer beats a guess made here. The
  # 07:37Z run died on `No module named 'transformers'` because this function
  # re-derived the interpreter on its own and got it wrong.
  if [ -n "${MLSYS_PYTHON:-}" ]; then printf '%s' "$MLSYS_PYTHON"; return; fi
  # `rasd-gpu` is the env scripts/mlsys_pod_env.sh creates; `rasd` is this Mac's
  # name and exists only as a legacy fallback. Looking for `rasd` alone is how a
  # correctly provisioned pod ended up running on /usr/bin/python3.
  for c in "$HOME/miniconda3/envs/rasd-gpu/bin/python" \
           "$HOME/miniconda3/envs/rasd/bin/python" \
           "$HOME/miniconda3/bin/python" \
           /opt/conda/envs/rasd-gpu/bin/python /opt/conda/bin/python \
           "$(command -v python3 2>/dev/null)"; do
    if [ -n "$c" ] && [ -x "$c" ] && "$c" -c 'import yaml, torch' 2>/dev/null; then
      printf '%s' "$c"; return
    fi
  done
  for c in "$HOME/miniconda3/envs/rasd-gpu/bin/python" \
           "$HOME/miniconda3/envs/rasd/bin/python" /opt/conda/bin/python \
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

# ---------------------------------------------------------------------------
# Per-stage wall-clock header, in SECONDS. ONE place, deliberately.
#
# The timeout and the cost model (`est_hours` in configs/mlsys_manifest.yml) are
# two numbers describing the same run, so they have to agree: a header shorter
# than the projection kills the stage the model says fits -- after the money is
# spent, and as a timeout rather than as a wrong number. They used to be literals
# at twelve call sites, where nothing could check them: the 256k rung carried a
# 12h header against a 13.0h projection, so it would have been killed for being
# exactly as expensive as predicted.
#
# A plain list rather than `declare -A`: the rehearsal runs on this machine's
# bash 3.2, which has no associative arrays, and a timeout table that only works
# on the pod is a table nobody can rehearse. `stage()` refuses any stage whose
# header is missing or non-numeric, so "unset" cannot mean "run unbounded".
# ---------------------------------------------------------------------------
STAGE_TIMEOUTS='
gate_calibration 14400
engine_cap_smoke 21600
correction_note_evidence 14400
coherence_gate 21600
natural_f1_128k 43200
natural_f1_128k_diverse 43200
impl_validation 21600
natural_spec_gated_256k 64800
natural_spec_gated_512k 144000
rope_intervention_gate 14400
rope_intervention_128k 43200
rope_intervention_128k_comparison 1800
vllm_ladder 21600
synthetic_spec_gated_128k 43200
synthetic_spec_gated_256k 64800
natural_f1_128k_losslessness 3600
natural_f1_128k_doc_intervals 1800
natural_f1_128k_round_acceptance 1800
natural_f1_128k_diverse_losslessness 3600
natural_f1_128k_diverse_doc_intervals 1800
natural_f1_128k_diverse_round_acceptance 1800
natural_spec_gated_256k_losslessness 3600
natural_spec_gated_256k_doc_intervals 1800
natural_spec_gated_256k_round_acceptance 1800
natural_spec_gated_512k_losslessness 3600
natural_spec_gated_512k_doc_intervals 1800
natural_spec_gated_512k_round_acceptance 1800
rope_intervention_128k_losslessness 3600
rope_intervention_128k_doc_intervals 1800
rope_intervention_128k_round_acceptance 1800
'
# Projection in seconds, for the ratio the dry run asserts: the header must clear
# `est_hours` by 20%. Printed by `stage_timeout --check`.
STAGE_EST_HOURS='
gate_calibration 1.0
engine_cap_smoke 1.5
correction_note_evidence 2.1
coherence_gate 2.0
natural_f1_128k 6.5
natural_f1_128k_diverse 6.5
impl_validation 1.0
natural_spec_gated_256k 13.0
natural_spec_gated_512k 26.2
rope_intervention_gate 0.6
rope_intervention_128k 5.4
rope_intervention_128k_comparison 0.02
vllm_ladder 2.5
synthetic_spec_gated_128k 6.5
synthetic_spec_gated_256k 13.0
'

# The vLLM stages must run in the ISOLATED environment (R7): vLLM 0.6.3 pins its
# own torch, and this campaign's main environment is pinned to a torch/
# transformers pair every other number depends on. The interpreter is selected BY
# PATH and the stage REFUSES if it is missing -- running the baseline in the main
# environment would either break those pins or measure a different build, and
# both are worse than not running the stage.
vllm_python() {
  local py="${MLSYS_VLLM_PYTHON:-$REPO/.venv-vllm/bin/python}"
  if [ ! -x "$py" ]; then
    return 1
  fi
  echo "$py"
}

stage_timeout() {  # $1 = stage id; prints seconds, empty if undeclared
  printf '%s\n' "$STAGE_TIMEOUTS" | awk -v w="$1" '$1 == w { print $2; f = 1 } END { exit !f }'
}

stage_est_hours() {  # $1 = stage id; prints hours, empty if undeclared
  local key=$1
  # helpers share their parent's projection bucket
  case "$key" in
    *_losslessness|*_doc_intervals|*_round_acceptance) echo 0.05; return 0 ;;
  esac
  printf '%s\n' "$STAGE_EST_HOURS" | awk -v w="$key" '$1 == w { print $2; f = 1 } END { exit !f }'
}

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

# `depends_on` as a space-separated list. `_manifest_field` prints a Python list
# for a YAML list, which is unreadable in shell; this reads it properly.
prereqs_of() {   # $1=stage id -> space separated ids
  "$PY" - "$MANIFEST" "$1" <<'PYPR'
import sys, yaml
m = yaml.safe_load(open(sys.argv[1]))
for s in m["stages"]:
    if s["id"] == sys.argv[2]:
        print(" ".join(s.get("depends_on") or []))
        raise SystemExit
print("")
PYPR
}

START_EPOCH=$(date -u +%s)
WATCHDOG_SKIPS=0
COST_UNKNOWN=0
STAGE_FAILURES=0
BUDGET_SKIPS=0
STAGE_INVALID=0
TIMEOUT_UNKNOWN=0
VLLM_MISSING=0
ONLY_SKIPS=0
WATCHDOG=${MLSYS_MAX_HOURS:-$(_manifest_field __meta__ max_hours)}
WATCHDOG=${WATCHDOG:-20}
elapsed_hours() { awk -v n="$(date -u +%s)" -v s="$START_EPOCH" 'BEGIN{printf "%.3f", (n-s)/3600.0}'; }

interim() { printf '%s %s\n' "$(date -u +%FT%TZ)" "$*" | tee -a "$OUT/RUN_LOG.txt"; }

# --------------------------------------------------------------------------
# pod-side stall watchdog
# --------------------------------------------------------------------------
# Independent of the operator's machine: if the watcher dies, or the Mac sleeps,
# or nobody is watching at all, the POD still stops a run that has gone silent
# and frees the GPU. The two watchdogs read the SAME threshold file
# (configs/mlsys_stall_thresholds.json) so they cannot drift apart.
#
# "Progress" is a line in RUN_LOG.txt. interim() tees to stdout as well, so the
# operator's ~/manifest.log sees the same lines in the same order.
STALL_JSON="$REPO/configs/mlsys_stall_thresholds.json"
WATCHDOG_LOG="$HOME/manifest_watchdog.log"
STALL_DEFAULT_MIN=${MLSYS_STALL_MINUTES:-20}
# The manifest's own stdout, which is what the byte-level liveness check reads.
# Defaulted rather than assumed, so a caller that redirects elsewhere still
# gets a watched file instead of a path that never exists.
MANIFEST_LOG=${MLSYS_MANIFEST_LOG:-$HOME/manifest.log}
MLSYS_MANIFEST_LIVENESS_MIN=${MLSYS_MANIFEST_LIVENESS_MIN:-15}

# The threshold for a stage, in minutes. Falls back to the file's default, then
# to the environment default. A stage the file does not name is NOT exempt: it
# gets the default, because "unknown stage" is not a reason to watch less.
stall_minutes_for() {
  local name=$1 v
  v=$("$PY" - "$STALL_JSON" "$name" "$STALL_DEFAULT_MIN" <<'PYSTALL' 2>/dev/null
import json, sys
path, name, fallback = sys.argv[1], sys.argv[2], int(sys.argv[3])
try:
    d = json.load(open(path))
except Exception:
    print(fallback); raise SystemExit
print(int(d.get("stages", {}).get(name, d.get("default_minutes", fallback))))
PYSTALL
)
  case "$v" in ''|*[!0-9]*) v=$STALL_DEFAULT_MIN ;; esac
  printf '%s' "$v"
}

current_stage() {   # last STAGE_START in the progress log
  grep -oE 'STAGE_START name=[A-Za-z0-9_]+' "$OUT/RUN_LOG.txt" 2>/dev/null \
    | tail -1 | cut -d= -f2
}

# How long manifest.log may go without GROWING, in bytes, while a stage runs.
# Read from the same file as the per-stage thresholds, and deliberately not
# overridable per stage: this is the tight rule that catches a hang inside a
# stage, and a per-stage value would let exactly the stage that hangs opt out.
# The floor of 15 min is 2.4x the longest legitimate silence measured in the
# campaign's logs (a 1M-token prefill, 6.29 min).
liveness_minutes() {
  local v
  v=$("$PY" - "$STALL_JSON" "$MLSYS_MANIFEST_LIVENESS_MIN" <<'PYLIVE' 2>/dev/null
import json, sys
path, fallback = sys.argv[1], int(sys.argv[2])
try:
    d = json.load(open(path))
except Exception:
    print(fallback); raise SystemExit
print(int(d.get("liveness_minutes", fallback)))
PYLIVE
)
  case "$v" in ''|*[!0-9]*) v=$MLSYS_MANIFEST_LIVENESS_MIN ;; esac
  printf '%s' "$v"
}

# Every descendant of a pid, deepest first. TWO defects are baked into what this
# replaced, and both are worth stating because they were invisible until a test
# with a live child ran:
#   * `ps --ppid` is a GNU extension. macOS rejects it ("illegal option"), so the
#     first version silently returned nothing and killed nothing.
#   * A recursive walk that forks a subshell per node has no cycle protection.
#     PID reuse put a loop in the process table and it forked until the machine
#     reported "fork: Resource temporarily unavailable" -- a stop helper that
#     can take the box down is worse than the stall it exists to handle.
# So: ONE ps snapshot, one awk that walks each pid up to the root with a depth
# cap, then a numeric sort. No recursion, no fork per node, no cycle.
#
# $2 EXCLUDES a pid and its whole subtree, and that is essential rather than
# tidy: this watchdog is itself a child of the manifest, and the walk's own
# ps/awk/sort pipeline are its children. Without the exclusion the stop routine
# signals ITSELF -- which is why the escalation sometimes ran and sometimes did
# not, and why the selftest saw a child survive a stop it had already logged.
_descendants() {   # $1 = pid; $2 = pid to exclude (with its subtree)
  ps -eo pid=,ppid= 2>/dev/null | awk -v root="$1" -v skip="${2:-}" '
    { ppid[$1] = $2; seen[$1] = 1 }
    END {
      for (p in seen) {
        if (p == skip) continue
        q = p; d = 0; bad = 0
        while (q != "" && q != "0" && q != root && d < 64) {
          if (q == skip) { bad = 1; break }
          q = ppid[q]; d++
        }
        if (!bad && q == root && d > 0) print d, p
      }
    }' | sort -rn | awk '{ print $2 }'
}

# Stop the run, not just the shell around it.
#
# Signalling only $MANIFEST_PID does NOT stop anything: bash defers a trap until
# the current foreground command returns, and that command IS the run
# (`timeout ... run_experiment.py`). On 2026-10-09T04:02Z the watchdog logged,
# wrote STALL_ABORT and signalled, and the campaign then ran for another 70
# minutes -- the operator believed the stall was being handled and it was not.
# The children are signalled first so the foreground command returns at once and
# the manifest's own trap can run.
stop_run() {
  local why=$1
  local pids i n survivors
  # ORDER MATTERS, and the wrong order is a race I hit in the selftest. The
  # shell is signalled FIRST so its trap is already pending; the children are
  # signalled next so the foreground command returns and that trap can run. With
  # the children first the run died, the blocked `wait`/foreground command
  # returned, and the manifest ran on to its next statement BEFORE its own TERM
  # arrived -- exiting 0, which the watcher reads as success.
  #
  # The list is captured ONCE, BEFORE anybody is signalled, and every later step
  # uses that same list. Re-walking from $MANIFEST_PID after signalling finds
  # NOTHING: the manifest exits on its TERM within milliseconds (bash runs a
  # pending trap as soon as the foreground command returns), so its children are
  # reparented to init, are no longer descendants of a pid that is gone, and the
  # escalation then reads an empty survivor set. The selftest caught exactly
  # that: the term-ignoring child had to be killed by hand-applied logic while
  # stop_run logged no STALL_KILL.
  pids="$(_descendants "$MANIFEST_PID" "$(cat "$WATCHDOG_SELF_FILE" 2>/dev/null)")"
  kill -TERM "$MANIFEST_PID" 2>/dev/null
  for i in $pids; do kill -TERM "$i" 2>/dev/null; done
  # Escalate only while something is still alive. A python worker unwinds in
  # seconds; past that, this branch has failed at its only job.
  for n in 1 2 3; do
    sleep 2
    survivors=""
    for i in $pids; do
      kill -0 "$i" 2>/dev/null && survivors="$survivors $i"
    done
    [ -z "$survivors" ] && break
  done
  if [ -n "$survivors" ]; then
    printf '%s STALL_KILL reason=%s survivors=%s\n' \
      "$(date -u +%FT%TZ)" "$why" "$survivors" >> "$WATCHDOG_LOG"
    for i in $survivors; do
      kill -KILL "$i" 2>/dev/null
    done
    kill -KILL "$MANIFEST_PID" 2>/dev/null
  fi
}

stall_watchdog() {
  local name limit idle now mtime
  # Byte-level liveness, tracked independently of RUN_LOG.txt.
  #
  # The threshold above only moves when a STAGE_* line is written, so it cannot
  # see a hang INSIDE a stage: on 2026-10-09T00:00Z the probe deadlocked and
  # engine_cap_smoke's threshold (240 min) was not reached until 62 minutes of
  # silence had already been paid for. `manifest.log` is the manifest's own
  # stdout, so it grows continuously while a run is doing anything at all --
  # tqdm every second, a TRACE line at each phase boundary.
  #
  # The limit is `liveness_minutes` from the same thresholds file. It is NOT a
  # per-stage override and must not be turned into one: this check exists to be
  # tighter than every stage threshold, and the measured longest legitimate
  # silence in the campaign's logs is a 1M-token prefill at 6.29 min.
  local live_limit live_sig="" live_since=""
  live_limit=$(liveness_minutes)
  while true; do
    sleep 60
    # The manifest may have finished normally between iterations. Its PID would
    # then be reusable, and signalling a reused PID means killing an unrelated
    # process -- so never signal without confirming the parent is still there.
    if ! ps -p "$MANIFEST_PID" >/dev/null 2>&1; then
      # Also the path out of a stop: stop_run leaves this watchdog alive to
      # finish its escalation, and the manifest is gone by the time it returns.
      return 0
    fi
    now=$(date -u +%s)

    # --- (i) byte-level liveness on the manifest's own output ---------------
    # Only while a stage is running: before the first STAGE_START the manifest
    # is still provisioning (installing the venv writes to its own log), and
    # that is not a hang in a stage.
    if [ -f "$MANIFEST_LOG" ]; then
      local size
      size=$(stat -c %s "$MANIFEST_LOG" 2>/dev/null || echo "")
      name=$(current_stage)
      if [ -z "$live_since" ]; then
        live_sig="$size"; live_since=$now
      elif [ "$size" != "$live_sig" ]; then
        live_sig="$size"; live_since=$now
      elif [ -n "$name" ] && [ $(( now - live_since )) -gt $(( live_limit * 60 )) ]; then
        printf '%s LIVENESS name=%s limit=%smin idle=%ss bytes=%s reason=manifest.log-not-growing\n' \
          "$(date -u +%FT%TZ)" "$name" "$live_limit" \
          "$(( now - live_since ))" "$size" >> "$WATCHDOG_LOG"
        interim "STALL_ABORT name=$name limit=${live_limit}min idle=$(( now - live_since ))s reason=manifest.log-not-growing"
        echo "STALL: manifest.log has not grown for $(( now - live_since ))s (limit ${live_limit}min) in stage '${name}'"
        echo "STALL: stopping the run; the watcher pulls logs and terminates."
        stop_run "liveness:${name}"
        return 0
      fi
    fi

    # --- (ii) the per-stage bound, unchanged -------------------------------
    # This is the OUTER bound: it covers a manifest.log that keeps growing (so
    # liveness never fires) while no run completes, which is what the long
    # per-stage values are sized for.
    [ -f "$OUT/RUN_LOG.txt" ] || continue
    mtime=$(stat -c %Y "$OUT/RUN_LOG.txt" 2>/dev/null || echo "$now")
    idle=$(( now - mtime ))
    name=$(current_stage)
    limit=$(stall_minutes_for "$name")
    [ "$idle" -gt $(( limit * 60 )) ] || continue
    # Log, then stop the manifest. The message goes to the progress log too, so
    # whatever is watching from outside sees WHY the run ended rather than
    # inferring it from silence.
    printf '%s STALL name=%s limit=%smin idle=%ss\n' \
      "$(date -u +%FT%TZ)" "$name" "$limit" "$idle" >> "$WATCHDOG_LOG"
    interim "STALL_ABORT name=$name limit=${limit}min idle=${idle}s"
    echo "STALL: no progress for ${idle}s (limit ${limit}min for stage '${name}')"
    echo "STALL: stopping the run; the watcher pulls logs and terminates."
    stop_run "stage:${name}"
    return 0
  done
}

MANIFEST_PID=$$
# Where the watchdog can learn its own pid (see _descendants' exclusion).
WATCHDOG_SELF_FILE="${TMPDIR:-/tmp}/mlsys_watchdog_self.$$"
# A stall is not a crash: it must exit NON-ZERO and say so, so the watcher's
# fail-fast path picks it up instead of reading a clean finish.
# STALL_STOPPING tells the EXIT trap below that a stop is in progress, so the
# watchdog survives long enough to finish it. Without the flag the manifest took
# its own watchdog down on the way out and a run that ignored TERM was left
# alive -- the selftest caught exactly that.
trap 'STALL_STOPPING=1; interim "SIGNALLED name=$(current_stage) reason=stall-watchdog-or-operator"; echo "SIGNALLED: stopping"; exit 9' TERM INT
# Never leave the watchdog behind: it would outlive the manifest and, on a PID
# it no longer owns, be a loaded gun pointed at whatever inherited that number.
trap 'if [ -z "${STALL_STOPPING:-}" ]; then kill ${STALL_WATCHDOG_PID:-0} 2>/dev/null; fi' EXIT

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

# A helper may run only when its parent is on the allowlist, has the parent's
# COST approval where one is required, and actually ran.
#
# The cost half is the one that is easy to miss. `MLSYS_ONLY_STAGES` is the
# campaign plan; `MLSYS_APPROVED_STAGES` is the operator's explicit permission
# to spend more than `MLSYS_ASK_OVER_USD` on a named stage. A helper inherits its
# parent's approval, which is right -- the operator approved the stage, not its
# bookkeeping -- but inheriting only the ALLOWLIST would let a helper run as
# part of a parent that the cost guard was about to refuse, i.e. spend on the
# unapproved side of the boundary through a side door.
helper_allowed() {
  local name=$1 parent prc
  parent=$(helper_parent "$name")
  [ -z "$parent" ] && return 0                # not a helper: no opinion
  if [ "$parent" != "*" ]; then
    on_list "$parent" "$ONLY" || return 1     # parent not approved
    prc=$(est_cost "$parent" 2>/dev/null) || prc=""
    if [ -n "$prc" ] && awk -v e="$prc" -v a="$ASK_OVER" 'BEGIN{exit !(e > a)}' \
         && ! approved "$parent"; then
      return 3                                # parent over the threshold, unapproved
    fi
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
  # FAIL CLOSED on the wall-clock header. A stage whose header is missing or
  # non-numeric would run under `timeout ""`, i.e. unbounded -- and an unbounded
  # stage is exactly how a watchdog becomes the thing that finds the bug.
  case "$tmo" in
    ''|*[!0-9]*)
      interim "REFUSED name=$name reason=no_declared_timeout header='$tmo'"
      echo "REFUSE $name: wall-clock header '$tmo' is not a number of seconds" \
           "(add it to STAGE_TIMEOUTS)"
      TIMEOUT_UNKNOWN=$((TIMEOUT_UNKNOWN + 1)); return 9 ;;
  esac
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
  elif [ "$hrc" = "3" ]; then
    interim "SKIPPED name=$name parent=$(helper_parent "$name") reason=parent_over_threshold_not_cost_approved"
    echo "SKIP $name: its parent $(helper_parent "$name") projects above"\
         "\$$ASK_OVER and is not in MLSYS_APPROVED_STAGES"
    ONLY_SKIPS=$((ONLY_SKIPS + 1))
    return 9
  fi

  # DEPENDENCIES. A stage whose prerequisite did not succeed in THIS attempt is
  # refused, before any money is committed. Without this the dependents ran
  # anyway and were then checked against files from whenever the prerequisite
  # last worked -- the failure that looks like a normal result.
  local dep
  for dep in $(prereqs_of "$name"); do
    stage_ok "$dep" || {
      local drc=$?
      interim "SKIPPED name=$name reason=prerequisite_not_ok prereq=$dep rc=$drc"
      echo "SKIP $name: prerequisite $dep did not run in this attempt and"\
           "succeed (rc=$drc)"
      ONLY_SKIPS=$((ONLY_SKIPS + 1))
      return 9
    }
  done

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

  # ---- FRESH ATTEMPT ------------------------------------------------------
  # A stage's outputs must not outlive the attempt that wrote them. Without this
  # a re-run that crashed early left the previous attempt's CSV in place, and
  # everything downstream -- the prerequisite checks, the row count, the helpers
  # -- validated files from a run that was no longer the current one. The
  # failure mode is the worst kind: the numbers look complete and are simply not
  # from this attempt.
  #
  # So the files this stage owns are MOVED ASIDE (never deleted: an earlier
  # attempt is evidence) into attempts/<utc>/<name>/, and the marker written
  # below is what says "this stage started in the current attempt". Prerequisite
  # checks read the marker, not the filesystem: a refused or failed prerequisite
  # is a hard stop, never a reason to fall back on what a previous attempt left.
  archive_attempt "$name"
  : > "$RAN_DIR/$name"          # this stage STARTED in this attempt
  : > "$RAN_DIR/$name.attempt"  # ... and its outputs are from this attempt
  t0=$(date -u +%s)
  echo "=== stage $name (projected \$$est, ${est_h}h; watchdog ${WATCHDOG}h) ==="
  # The progress signal BOTH watchdogs read: a stage that started is a line, and
  # the line names the stage so each side can apply that stage's threshold.
  interim "STAGE_START name=$name timeout=${tmo}s"
  # `if cmd; then rc=0; else rc=$?; fi` — NOT `cmd; rc=$?`, which under
  # `set -e` never reaches the guard because the shell exits first. The command
  # must be run EXACTLY ONCE here; running it a second time to capture rc would
  # execute the whole stage twice.
  if timeout "$tmo" "$@"; then rc=0; else rc=$?; fi
  t1=$(date -u +%s)
  wall=$((t1-t0))
  cost=$(awk -v w="$wall" -v r="$RATE" 'BEGIN{printf "%.2f", w/3600.0*r}')
  echo "$name,$wall,8,$(awk -v w="$wall" 'BEGIN{printf "%.4f", w/3600.0*8}'),$cost" >> "$COST_LOG"
  # The exit code is recorded on disk as well as returned: a prerequisite check
  # runs in a different `if`, and reading the code from a file is the only way
  # it can know whether the stage it depends on actually succeeded in THIS
  # attempt rather than in a previous one.
  echo "$rc" > "$RAN_DIR/$name.rc"
  if [ $rc -eq 0 ]; then
    interim "STAGE_OK name=$name wall=${wall}s cost=\$$cost cumulative=\$$(spend)"
  else
    # A stage that exits zero having written nothing usable is not a success:
    # rc only says the process did not crash. The row check at the call site is
    # what makes "the stage passed" a statement about the data.
    interim "STAGE_FAILED name=$name rc=$rc wall=${wall}s cost=\$$cost"
    STAGE_FAILURES=$((STAGE_FAILURES + 1))
  fi
  echo "--- $name rc=$rc wall=${wall}s cost=\$$cost cumulative=\$$(spend)"
  # RETURN THE COMMAND'S CODE. Returning 0 unconditionally told every caller
  # that a failed stage had succeeded, so a caller that only knew how to check
  # for a refusal (9) went straight on to validate and report a stage that had
  # just failed. `stage_ok` is the sanctioned way to read this.
  return $rc
}

# The only sanctioned way to ask "may I validate and report this stage?".
#
#   0  the stage ran in THIS attempt and exited 0 -> validate its outputs
#   9  refused (allowlist, cost, watchdog, or its parent's approval)
#   *  it ran and failed: no validation, no reporting, and its dependents stop
stage_ok() {   # $1=stage name
  local name=$1 rc
  [ -f "$RAN_DIR/$name.attempt" ] || return 2   # did not run in this attempt
  rc=$(cat "$RAN_DIR/$name.rc" 2>/dev/null || echo 1)
  [ "$rc" = "0" ] && return 0
  [ "$rc" = "9" ] && return 9
  return 1
}

# Move a stage's own outputs aside, so nothing downstream can read a previous
# attempt's files. Archived, not deleted: an earlier attempt is evidence about
# what happened, and deleting it would be the one irreversible choice here.
#
# `attempts/` lives under the results dir so the operator can find it on the pod,
# and is EXCLUDED from the results pull (see the watcher): it is diagnostic, not
# a deliverable, and a previous attempt's CSVs in the delivered corpus would be
# indistinguishable from this run's.
archive_attempt() {   # $1=stage name
  local name=$1 dest stamp f moved=0
  stamp=$(date -u +%Y%m%dT%H%M%SZ)
  dest="$OUT/attempts/$stamp/$name"
  for f in "$OUT/$name.csv" "$OUT/$name".*.csv; do
    [ -e "$f" ] || continue
    [ "$moved" = "0" ] && { mkdir -p "$dest"; moved=1; }
    mv "$f" "$dest/" 2>/dev/null || true
  done
  # `<name>.gated.yml` is deliberately NOT touched here. It is DERIVED: the gate
  # filter OVERWRITES it immediately before the stage runs, in this attempt, and
  # a filter failure skips the stage outright -- so a stale copy cannot be used,
  # and moving or deleting it here (both tried) removes the very config the
  # stage is about to be launched with.
  # A stage that crashed mid-run leaves no CSV but may leave a partial file with
  # a different suffix; those are the ones that make a later validation look
  # plausible, so they move too.
  for f in "$OUT/$name"*.partial "$OUT/$name"*.tmp; do
    [ -e "$f" ] || continue
    [ "$moved" = "0" ] && { mkdir -p "$dest"; moved=1; }
    mv "$f" "$dest/" 2>/dev/null || true
  done
  [ "$moved" = "1" ] && interim "ATTEMPT_ARCHIVED name=$name dest=$dest"
  return 0
}

# Does the stage's CSV hold the rows it was supposed to produce, all ok?
# `--abort-on-failure` makes run_experiment stop at the first error row, but the
# row count is what says the stage ran to completion.
check_stage_rows() {   # $1=stage id  $2=csv  $3=expected rows [ $4=plan ids ]
  local name=$1 csv=$2 want=$3 plan=${4:-}
  # A stage whose planner says it has runs but whose plan counts zero is broken,
  # not empty. Treating 0 as "skip the check" was the silent path: every
  # assertion below was skipped and the stage was reported as fine.
  if [ -z "$want" ] || [ "$want" = "0" ]; then
    interim "STAGE_INVALID name=$name reason=expected_rows_zero"
    echo "INVALID $name: the planner reports 0 runs, so the row count cannot be"
    echo "  checked and the stage cannot be called complete (want='$want')"
    STAGE_INVALID=$((STAGE_INVALID + 1)); return 1
  fi
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
  # The SET of runs written must be the set that was planned, with no
  # duplicates. A count alone passes a stage that wrote 20 rows for the wrong 20
  # runs, and a duplicate run_id means a run was appended twice -- which is how
  # a stale row enters a file that is supposed to be from this attempt.
  if [ -n "$plan" ] && [ -s "$plan" ]; then
    if ! "$PY" - "$csv" "$plan" <<'PYID'
import csv, sys
rows = [r.get("run_id", "") for r in csv.DictReader(open(sys.argv[1]))]
planned = [l.strip() for l in open(sys.argv[2]) if l.strip()]
seen, dups = set(), []
for r in rows:
    if r in seen:
        dups.append(r)
    seen.add(r)
problems = []
if dups:
    problems.append("duplicate run_id(s): " + ", ".join(sorted(set(dups))[:5]))
missing = sorted(set(planned) - seen)
extra = sorted(seen - set(planned))
if missing:
    problems.append("planned but not written: " + ", ".join(missing[:5]))
if extra:
    problems.append("written but not planned: " + ", ".join(extra[:5]))
for p in problems:
    print("  " + p)
sys.exit(1 if problems else 0)
PYID
    then
      interim "STAGE_INVALID name=$name reason=run_id_set"
      echo "INVALID $name: the rows written are not the runs that were planned"
      STAGE_INVALID=$((STAGE_INVALID + 1)); return 1
    fi
  fi
  # Every row of every stage must satisfy the sequence identity
  # prompt_tokens + 1 BOS + tokens_generated == sequence_tokens. It is checked
  # here rather than in the cap smoke alone because it is what makes
  # `context_length` mean the same thing in every table: a row whose sequence
  # is not the sum of its parts is either mislabelled or measured against a
  # different prompt than the one recorded.
  local ident
  ident=$("$PY" scripts/mlsys_row_identity_check.py --results "$csv" 2>&1)
  if [ $? -ne 0 ]; then
    interim "STAGE_INVALID name=$name reason=sequence_identity"
    echo "INVALID $name: the sequence identity does not hold:"
    printf '%s\n' "$ident" | head -5 | sed 's/^/    /'
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
# $1=config $2=groups [$3=stage]. With a stage name, the planned run IDS are
# written to $RAN_DIR/<stage>.plan.ids as well as counted: a stage that wrote the
# right NUMBER of the wrong runs is not complete.
expected_rows() {   # $1=config  $2=groups [ $3=stage ]
  local n args=(--config "$1")
  # `--groups ""` is an argparse error ("expected at least one argument"), not
  # an empty selection, so an empty group list means the flag is omitted.
  [ -n "$2" ] && args+=(--groups $2)
  # Count the RUN LINES, not "every line after the separator". The trailer
  # ("10 runs total.") also has enough fields to look like a row, so the old
  # `NF>=3` counted the summary as a run and every stage's expected count was one
  # too high. A run line is `<run id>  <group>  <seed>  <config>`: at least four
  # fields, with the third being the numeric seed.
  n=$("$PY" run_experiment.py "${args[@]}" --dry-run 2>"$RAN_DIR/plan.err" \
      | awk 'BEGIN{sep=0} /^-{10,}$/{sep=1; next}
             sep && NF>=4 && $3 ~ /^[0-9]+$/ {n++} END{print n+0}')
  if [ -n "${3:-}" ]; then
    "$PY" run_experiment.py "${args[@]}" --dry-run 2>/dev/null \
      | awk 'BEGIN{sep=0} /^-{10,}$/{sep=1; next}
             sep && NF>=4 && $3 ~ /^[0-9]+$/ {print $1}' \
      | sort -u > "$RAN_DIR/$3.plan.ids"
  fi
  if [ -z "$n" ] || [ "$n" = "0" ]; then
    # Either the planner failed or the config/groups select nothing. Both mean
    # the stage's row count is unknowable, and an unknowable count is a failure
    # of the stage -- never a reason to skip its row check.
    echo "planner produced 0 rows for $1 --groups '$2'" >&2
    [ -s "$RAN_DIR/plan.err" ] && tail -3 "$RAN_DIR/plan.err" >&2
    return 1
  fi
  printf '%s' "$n"
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
# Validate and report a speculative stage's output. Runs ONLY for a stage that
# ran in this attempt and exited 0: validating a stage that was refused, or that
# failed, reports a defect for something that did not happen -- and reads files
# from whenever the last successful attempt was.
validate_and_report() {   # $1=stage id  $2=csv  $3=expected rows
  local name=$1 csv=$2 want=$3 rc
  stage_ok "$name"; rc=$?
  if [ "$rc" != "0" ]; then
    interim "SKIPPED_VALIDATION name=$name reason=stage_not_ok rc=$rc"
    echo "SKIP validation of $name: it did not run in this attempt and exit 0 (rc=$rc)"
    return 1
  fi
  check_stage_rows "$name" "$csv" "$want" "$RAN_DIR/$name.plan.ids" || return 1
  report_spec_stage "$csv" "$name"
  return 0
}

report_spec_stage() {   # $1=stage csv path  $2=stage id
  local csv=$1 name=$2
  stage "${name}_losslessness" "$(stage_timeout "${name}_losslessness")" "$PY" scripts/mlsys_losslessness.py \
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
  stage "${name}_doc_intervals" "$(stage_timeout "${name}_doc_intervals")" "$PY" scripts/mlsys_document_bootstrap.py \
    --results "$csv" --group-by context_length --ratio-metric decode_tps \
    --out "$(dirname "$csv")/${name}_doc_intervals.csv"
  # --window: the plan compares rungs over a common window of rounds so cells
  # with different output lengths are on equal footing.
  stage "${name}_round_acceptance" "$(stage_timeout "${name}_round_acceptance")" "$PY" scripts/mlsys_cluster_bootstrap.py \
    --traces "$(dirname "$csv")/per_token" --window 15 \
    --out "$(dirname "$csv")/${name}_acceptance_bootstrap.csv"
}

DOCS=${MLSYS_DOCUMENTS_JSON:-data/processed/pg19_docs/documents.json}
# The revision a vLLM row is compared against. Every vLLM cell in this campaign
# is Llama-3.1-8B at bf16 (plan revision, 2026-10-07): the ladder used to sweep
# Llama-2 as well, but the comparison it exists for is the production-stack
# THROUGHPUT reference for this campaign's target, and a second model at a
# second weight precision doubled the cells to answer a question nobody asked.
# A row pinned to another model's revision would be a comparison across models
# that looks pinned, so the pin is per-model and there is now exactly one.
TARGET_REVS_D3="meta-llama/Llama-3.1-8B=d04e592bb4f6aa9cfee91e2e20afa771667e1d4b"
interim "=== MANIFEST START rate=\$$RATE ask_over=\$$ASK_OVER watchdog=${WATCHDOG}h only='${ONLY:-<none>}' approved='$APPROVED' ==="
echo "spend before manifest: \$$(spend)"

# The pod's own stall watchdog, started before the first stage so a stall in
# stage 1 is covered. Detached from the stage list on purpose: it must keep
# running across stages, which is exactly when a per-stage timeout does not
# help.
stall_watchdog &
STALL_WATCHDOG_PID=$!
# Written AFTER the fork because that is when the pid exists; the watchdog reads
# this file, since a subshell cannot see a variable assigned after it started.
printf '%s\n' "$STALL_WATCHDOG_PID" > "$WATCHDOG_SELF_FILE" 2>/dev/null
interim "STALL_WATCHDOG started pid=$STALL_WATCHDOG_PID default=${STALL_DEFAULT_MIN}min thresholds=$STALL_JSON"

# ---- S0: calibrate the gate on real weights. MUST be first. ---------------
# CUDA_LAUNCH_BLOCKING=1 for THIS stage only. The 10:10Z run reported a
# device-side assert at `torch.cuda.empty_cache()`, which is a synchronising
# call rather than the failing one -- the actual illegal operation happened
# earlier and the async report pointed at the wrong place. Serialising kernels
# costs speed on a stage that has a 4h timeout and used 8 minutes, and it buys
# an error attributed to the candidate that caused it.
stage gate_calibration "$(stage_timeout gate_calibration)" \
  env CUDA_LAUNCH_BLOCKING=1 "$PY" scripts/mlsys_coherence_gate.py \
  --candidates configs/mlsys_gate_controls.json \
  --pg19-meta "$DOCS" \
  --out "$OUT/gate_calibration.csv" --gen-dir "$OUT/gate_calibration_generated"

# The prerequisite is the STAGE, not the file. An earlier attempt's CSV was
# moved aside before this stage ran, so its absence here means this attempt
# produced nothing -- but the reason has to be reported correctly, and a refused
# or failed calibration must stop the ladder rather than be re-checked against
# whatever was on disk.
cal_rc=$(cat "$RAN_DIR/gate_calibration.rc" 2>/dev/null || echo 1)
if [ "$cal_rc" != "0" ]; then
  interim "ABORT: gate calibration did not succeed in this attempt rc=$cal_rc"
  echo "ABORT: gate calibration rc=$cal_rc; the gate is uncalibrated and no"
  echo "  candidate may be gated from it."
  exit 1
fi
if [ ! -s "$OUT/gate_calibration.csv" ]; then
  interim "ABORT: gate calibration exited 0 but wrote no verdicts"
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
stage engine_cap_smoke "$(stage_timeout engine_cap_smoke)" "$PY" run_experiment.py \
  --config configs/mlsys_engine_cap_smoke.yml \
  --output "$OUT/engine_cap_smoke.csv" --stage-id engine_cap_smoke \
  --timeout-per-run-s "$(per_run_timeout 131072)" --abort-on-failure \
  --log-per-token --save-generated-tokens
cap_rc=$(cat "$RAN_DIR/engine_cap_smoke.rc" 2>/dev/null || echo 1)
if [ "$cap_rc" != "0" ]; then
  interim "STOP: engine_cap_smoke did not succeed in this attempt rc=$cap_rc"
  echo "STOP: the cap smoke rc=$cap_rc; a stage that failed is not a stage whose"
  echo "  cap arithmetic was verified, whatever is in its output directory."
  exit 1
fi
if [ ! -s "$OUT/engine_cap_smoke.csv" ]; then
  interim "STOP: engine_cap_smoke produced no results; refusing to start speculative stages"
  echo "STOP: cap smoke produced nothing"
  exit 1
fi
cap_want=$(expected_rows configs/mlsys_engine_cap_smoke.yml "" engine_cap_smoke) || {
  interim "STOP: the cap smoke's plan could not be counted"
  echo "STOP: the planner produced no rows for the cap smoke"; exit 1; }
check_stage_rows engine_cap_smoke "$OUT/engine_cap_smoke.csv" "$cap_want" || {
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
stage correction_note_evidence "$(stage_timeout correction_note_evidence)" "$PY" scripts/mlsys_coherence_gate.py \
  --candidates configs/mlsys_correction_candidates.json \
  --pg19-meta "$DOCS" \
  --out "$OUT/correction_note_evidence.csv" \
  --gen-dir "$OUT/correction_note_generated"

# ---- S3: the gate, now calibrated ---------------------------------------
stage coherence_gate "$(stage_timeout coherence_gate)" "$PY" scripts/mlsys_coherence_gate.py \
  --candidates configs/mlsys_rope_candidates.json \
  --pg19-meta "$DOCS" \
  --out "$OUT/coherence_gate.csv" --gen-dir "$OUT/gate_generated"

cg_rc=$(cat "$RAN_DIR/coherence_gate.rc" 2>/dev/null || echo 1)
if [ "$cg_rc" != "0" ]; then
  interim "ABORT: the coherence gate did not succeed in this attempt rc=$cg_rc"
  echo "ABORT: the coherence gate rc=$cg_rc; no >128k stage may run on verdicts"
  echo "  from an attempt that did not produce them."
  exit 1
fi
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
  # The row count is a PREREQUISITE, not an afterthought: an uncountable plan
  # means the stage cannot be called complete, so it never starts. Previously a
  # 0 here was passed along and check_stage_rows treated it as "skip the check".
  if [ -n "$grp" ]; then
    want=$(expected_rows "$cfg" "$grp" "$name") || {
      interim "STAGE_INVALID name=$name reason=planner_produced_no_rows"
      echo "INVALID $name: the planner produced 0 rows for groups '$grp'"
      STAGE_INVALID=$((STAGE_INVALID + 1)); continue; }
    stage "$name" "$(stage_timeout "$name")" "$PY" run_experiment.py --config "$cfg" --groups $grp \
      --output "$OUT/${name}.csv" --stage-id "$name" \
      --timeout-per-run-s "$(per_run_timeout 131072)" --abort-on-failure \
      --log-per-token --memory-trace --save-generated-text --save-generated-tokens
  else
    want=$(expected_rows "$cfg" "" "$name") || {
      interim "STAGE_INVALID name=$name reason=planner_produced_no_rows"
      echo "INVALID $name: the planner produced 0 rows"
      STAGE_INVALID=$((STAGE_INVALID + 1)); continue; }
    stage "$name" "$(stage_timeout "$name")" "$PY" run_experiment.py --config "$cfg" \
      --output "$OUT/${name}.csv" --stage-id "$name" \
      --timeout-per-run-s "$(per_run_timeout 131072)" --abort-on-failure \
      --log-per-token --memory-trace --save-generated-text --save-generated-tokens
  fi
  validate_and_report "$name" "$OUT/${name}.csv" "$want"
done

# ---- S1: implementation validation — a TARGET cross-check (R7) -----------
# No RASD runs here: the comparison uses natural_f1_128k's own target-only token
# sidecars, so this stage adds no RASD wall time.
#
# What it claims, after the SECOND 2026-10-07 plan revision: this is the
# production stack's THROUGHPUT at 128k on the same documents and the same
# engine_input_ids, under its own numerics. It claims nothing about token
# agreement as evidence of correctness: RASD runs FP4 weights and an NF4 KV
# cache and vLLM 0.6.3 has no NF4 KV path, so the two engines are not running
# the same arithmetic and their first divergence measures precision. Token
# agreement is reported descriptively (first divergence, agreement up to it,
# gaps there) under NOT_COMPARABLE_PRECISION and NEVER fails the stage; the
# stage fails only if no row is unit_matched for throughput.
#
# The interpreter is the ISOLATED venv (its own torch), selected by path; if it
# is missing the stage is refused rather than run against the main environment's
# pins.
if VLLM_PY=$(vllm_python); then
  stage impl_validation "$(stage_timeout impl_validation)" "$VLLM_PY" scripts/mlsys_vllm_baseline.py \
    --out "$OUT/impl_validation.csv" \
    --prompt-ids-from-sidecars "$OUT/tokens" \
    --rasd-target-sidecars "$OUT/tokens" \
    --compare-out "$OUT/impl_validation_target_crosscheck.csv" \
    --documents pg19_train_0,pg19_train_1,pg19_train_115 \
    --context-lengths 131072 --max-new-tokens 1024 --matched-max-new-tokens 1024 \
    --models meta-llama/Llama-3.1-8B \
    --target-revisions "$TARGET_REVS_D3"
else
  interim "REFUSED name=impl_validation reason=no_vllm_venv"
  echo "REFUSE impl_validation: no vLLM interpreter at"\
       "'\''${MLSYS_VLLM_PYTHON:-$REPO/.venv-vllm/bin/python}'\''; run"\
       "scripts/mlsys_vllm_venv.sh first. The main environment is NOT used for"\
       "the baseline (its torch/transformers pins are what every other number"\
       "was measured with)."
  VLLM_MISSING=$((VLLM_MISSING + 1))
fi

# ---- S5: the gated rungs, one stage per rung -----------------------------
# Each is severable so the 512k session can be approved and run on its own
# instance. 512k needs a 40h watchdog: at ~26h the default 20h would kill it.
for gated in "natural_spec_gated_256k:GATED_llama3_f16_256k" \
             "natural_spec_gated_512k:GATED_llama3_f32_512k"; do
  name=${gated%%:*}; rest=${gated#*:}; prefix=${rest%%:*}
  tmo=$(stage_timeout "$name")
  cfg=configs/mlsys_natural_gated.yml
  if [ ! -f "$cfg" ]; then
    interim "SKIPPED name=$name reason=config-missing path=$cfg"
    continue
  fi
  # ALLOWLIST FIRST. Filtering a config for a stage that may not run is wasted
  # work, and when the filter then found nothing to keep it recorded
  # STAGE_INVALID against a stage the operator had simply not approved -- an
  # invalid for a stage that was never scheduled, which is neither a defect nor
  # the operator's decision to make. Excluded is excluded, before any work.
  if ! on_list "$name" "$ONLY"; then
    interim "SKIPPED name=$name reason=needs_approval (before gate filter)"
    echo "SKIP $name (not in MLSYS_ONLY_STAGES; no gate filtering done)"
    ONLY_SKIPS=$((ONLY_SKIPS + 1))
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
  want=$(expected_rows "$filtered" "${prefix}_SPEC ${prefix}_TARGET_FULL ${prefix}_TARGET_SHORT" "$name") || {
    interim "STAGE_INVALID name=$name reason=planner_produced_no_rows"
    echo "INVALID $name: the gate filtered the rung down to nothing runnable"
    STAGE_INVALID=$((STAGE_INVALID + 1)); continue; }
  stage "$name" "$tmo" "$PY" run_experiment.py --config "$filtered" \
    --groups ${prefix}_SPEC ${prefix}_TARGET_FULL ${prefix}_TARGET_SHORT \
    --output "$OUT/${name}.csv" --stage-id "$name" \
    --timeout-per-run-s "$(per_run_timeout "$(_manifest_field "$name" rung)")" \
    --abort-on-failure \
    --log-per-token --memory-trace --save-generated-text --save-generated-tokens
  validate_and_report "$name" "$OUT/${name}.csv" "$want"
done

# ---- S6: synthetic arm, per rung, secondary ------------------------------
for syn in "synthetic_spec_gated_128k:NATIVE_synth_128k" \
           "synthetic_spec_gated_256k:GATED_llama3_f16_256k_SYNTH"; do
  name=${syn%%:*}; rest=${syn#*:}; prefix=${rest%%:*}
  tmo=$(stage_timeout "$name")
  cfg=configs/mlsys_synthetic_gated.yml
  [ -f "$cfg" ] || { interim "SKIPPED name=$name reason=config-missing"; continue; }
  # ALLOWLIST FIRST, same reason as the natural rungs: no filtering work, and no
  # invalid recorded, for a stage that was never approved.
  if ! on_list "$name" "$ONLY"; then
    interim "SKIPPED name=$name reason=needs_approval (before gate filter)"
    echo "SKIP $name (not in MLSYS_ONLY_STAGES; no gate filtering done)"
    ONLY_SKIPS=$((ONLY_SKIPS + 1))
    continue
  fi
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
  want=$(expected_rows "$filtered" "${prefix}_SPEC ${prefix}_TARGET_FULL ${prefix}_TARGET_SHORT" "$name") || {
    interim "STAGE_INVALID name=$name reason=planner_produced_no_rows"
    echo "INVALID $name: the gate filtered the rung down to nothing runnable"
    STAGE_INVALID=$((STAGE_INVALID + 1)); continue; }
  stage "$name" "$tmo" "$PY" run_experiment.py --config "$filtered" \
    --groups ${prefix}_SPEC ${prefix}_TARGET_FULL ${prefix}_TARGET_SHORT \
    --output "$OUT/${name}.csv" --stage-id "$name" \
    --timeout-per-run-s "$(per_run_timeout "$(_manifest_field "$name" rung)")" \
    --abort-on-failure \
    --log-per-token --memory-trace --save-generated-text --save-generated-tokens
  validate_and_report "$name" "$OUT/${name}.csv" "$want"
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
  stage rope_intervention_gate "$(stage_timeout rope_intervention_gate)" "$PY" scripts/mlsys_coherence_gate.py \
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

  want=$(expected_rows configs/mlsys_rope_intervention_128k.yml "$RI_GROUPS" rope_intervention_128k) || {
    interim "STAGE_INVALID name=rope_intervention_128k reason=planner_produced_no_rows"
    echo "INVALID rope_intervention_128k: the planner produced 0 rows for '$RI_GROUPS'"
    STAGE_INVALID=$((STAGE_INVALID + 1)); want=""; }
  if [ -n "$want" ]; then
  stage rope_intervention_128k "$(stage_timeout rope_intervention_128k)" "$PY" run_experiment.py \
    --config configs/mlsys_rope_intervention_128k.yml \
    --groups $RI_GROUPS \
    --output "$OUT/rope_intervention_128k.csv" --stage-id rope_intervention_128k \
    --timeout-per-run-s 9000 --abort-on-failure \
    --log-per-token --memory-trace --save-generated-text --save-generated-tokens
  # Only a stage that ran in THIS attempt and exited 0 has outputs worth
  # checking or comparing. A refusal and a failure are both "no result", and
  # neither may be turned into a comparison against whatever the last attempt
  # left behind.
  stage_ok rope_intervention_128k
  ri_rc=$?
  if [ "$ri_rc" != "0" ]; then
    interim "SKIPPED_VALIDATION name=rope_intervention_128k reason=stage_not_ok rc=$ri_rc"
    echo "SKIP validation/reporting of rope_intervention_128k (rc=$ri_rc)"
  else
  check_stage_rows rope_intervention_128k "$OUT/rope_intervention_128k.csv" "$want"
  # Native vs each treated arm, paired by document, difference in alpha_round
  # with the pre-registered 0.05 equivalence margin. An arm whose gate failed
  # has no rows, so its contrast is simply absent -- which is the recorded
  # result for that arm, not a missing measurement to be filled in.
  stage rope_intervention_128k_comparison "$(stage_timeout rope_intervention_128k_comparison)" "$PY" \
    scripts/mlsys_rope_intervention.py \
    --results "$OUT/rope_intervention_128k.csv" \
    --out "$OUT/rope_intervention_128k_comparison.csv" \
    --metric acceptance_rate --margin 0.05
  fi
  fi
  fi
else
  interim "SKIPPED name=rope_intervention_128k reason=needs_approval"
fi

# ---- S7: vLLM baseline ---------------------------------------------------
# Plain (non-speculative) decode at every rung: the production-stack throughput
# reference where the payoff boundary is claimed. impl_validation covers 128k;
# this covers the ladder. Same R7 contract as impl_validation: a throughput
# reference under vLLM's own numerics, with both engines' precision recorded per
# row, token agreement reported descriptively and never scored, and the stage
# failing only if no row is unit_matched for throughput.
if VLLM_PY=$(vllm_python); then
  stage vllm_ladder "$(stage_timeout vllm_ladder)" "$VLLM_PY" scripts/mlsys_vllm_baseline.py \
    --out "$OUT/vllm_baseline.csv" \
    --prompt-ids-from-sidecars "$OUT/tokens" \
    --rasd-target-sidecars "$OUT/tokens" \
    --compare-out "$OUT/vllm_ladder_target_crosscheck.csv" \
    --context-lengths 131072 262144 524288 --max-new-tokens 1024 \
    --matched-max-new-tokens 1024 \
    --models meta-llama/Llama-3.1-8B \
    --quantizations bfloat16 \
    --target-revisions "$TARGET_REVS_D3"
else
  interim "REFUSED name=vllm_ladder reason=no_vllm_venv"
  echo "REFUSE vllm_ladder: no vLLM interpreter (run scripts/mlsys_vllm_venv.sh)"
  VLLM_MISSING=$((VLLM_MISSING + 1))
fi

if [ "$WATCHDOG_SKIPS" -gt 0 ] || [ "$COST_UNKNOWN" -gt 0 ] \
   || [ "$BUDGET_SKIPS" -gt 0 ] || [ "$STAGE_INVALID" -gt 0 ] \
   || [ "$STAGE_FAILURES" -gt 0 ] || [ "$TIMEOUT_UNKNOWN" -gt 0 ] \
   || [ "$VLLM_MISSING" -gt 0 ]; then
  interim "=== MANIFEST INCOMPLETE watchdog_skips=$WATCHDOG_SKIPS cost_unknown=$COST_UNKNOWN budget_skips=$BUDGET_SKIPS stage_invalid=$STAGE_INVALID stage_failures=$STAGE_FAILURES timeout_unknown=$TIMEOUT_UNKNOWN vllm_missing=$VLLM_MISSING spend=\$$(spend) ==="
  echo "MANIFEST INCOMPLETE:" \
       "watchdog=$WATCHDOG_SKIPS cost-unknown=$COST_UNKNOWN" \
       "budget=$BUDGET_SKIPS invalid=$STAGE_INVALID failed=$STAGE_FAILURES" \
       "timeout-unknown=$TIMEOUT_UNKNOWN"
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

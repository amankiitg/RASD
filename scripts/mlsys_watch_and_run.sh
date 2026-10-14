#!/usr/bin/env bash
# Watch for an 8x A100 80GB instance, run the MLSys manifest on it, pull and
# verify the results, then terminate.
#
# Hard-won behaviours baked in (each one cost a real run to learn):
#
#   * A capacity REPORT is not capacity. The API will name a region and then
#     refuse every launch POST with `insufficient-capacity`. One phantom reading
#     must not end the wait, so the launch sits inside a deadline-bounded loop.
#   * The loop is gated on `count_instances() == 0`. If an instance exists we
#     must NOT keep retrying launches into it, or we orphan a second one.
#   * The deadline is an ABSOLUTE epoch, not "now + N hours". A relative
#     deadline silently drifts later on every restart.
#   * Termination is confirmed by polling instances down to ZERO. "terminating"
#     is not terminated, and a poll that suppresses stderr can print nothing
#     while looking like success.
#   * The results pull uses per-run staging and NEVER `--delete`: an rsync
#     --delete once removed 367 committed artifacts that the pod never held,
#     because the outbound sync excludes results/mlsys.
#
# Usage:
#   MLSYS_DEADLINE_EPOCH=<epoch> bash scripts/mlsys_watch_and_run.sh
# Env:
#   MLSYS_HOURS          hours from now if no explicit deadline (default 18)
#   MLSYS_INSTANCE_TYPE  default gpu_8x_a100_80gb_sxm4
#   MLSYS_ASK_OVER_USD   per-stage approval threshold (default 300)
#   MLSYS_APPROVED_STAGES  comma list of stage ids approved to run
#                        while projected above that threshold

set -uo pipefail
cd "$(dirname "$0")/.."
REPO=$PWD
SESSION_DIR=${MLSYS_SESSION_DIR:-$(mktemp -d)}
LOG=$SESSION_DIR/watcher.log
FOUND=$SESSION_DIR/CAPACITY_FOUND

INSTANCE_TYPE=${MLSYS_INSTANCE_TYPE:-gpu_8x_a100_80gb_sxm4}
ASK_OVER=${MLSYS_ASK_OVER_USD:-300}
# The allowlist is passed to the pod with the SAME default as the manifest's, and
# both are exported rather than left empty. An empty list means "approve
# nothing" on both sides -- "unset" must never be able to mean "run everything".
DEFAULT_ONLY=gate_calibration,engine_cap_smoke,coherence_gate,correction_note_evidence,natural_f1_128k,impl_validation,rope_intervention_128k,natural_spec_gated_256k,vllm_ladder
ONLY_STR=${MLSYS_ONLY_STAGES-$DEFAULT_ONLY}
[ -z "$ONLY_STR" ] && say "WARNING: MLSYS_ONLY_STAGES is empty; the pod will refuse every stage"
RATE=22.32
SSH_KEY=$HOME/.ssh/id_ed25519
SSH_USER=ubuntu
SSH_OPTS="-o StrictHostKeyChecking=no -o ConnectTimeout=25 -i $SSH_KEY"
# Interpreter to use ON THE POD, in two parts, because they are used differently
# and conflating them cost two 8xA100 launches on 2026-10-08.
#
#   RPY_RESOLVE  a COMMAND that prints the interpreter's path (no `$( )').
#   RPY          the resolved absolute path, used by every later remote command.
#
# The trap: RPY used to be a `$( ... )' EXPRESSION. That happens to work when it
# is interpolated into something that follows it with arguments --
# `$(...) -c "..."` runs the path it echoed -- so the data verification looked
# fine. But `ssh host "$RPY"' passes the expression as the WHOLE command, so the
# pod ran the echoed path with no arguments: python started, printed nothing,
# and exited. The resolution came back empty and the run refused to start.
#
# So resolution is a command that prints, and anything later is a plain path.
# The env is named `rasd-gpu': that is what scripts/auto_execute_phase_c.sh
# creates and what environment_gpu.yml declares. `rasd' is the local dev
# machine's name, kept as a fallback only.
RPY_RESOLVE='for c in "$HOME/miniconda3/envs/rasd-gpu/bin/python" \
                     "$HOME/miniconda3/envs/rasd/bin/python" \
                     "/opt/conda/envs/rasd-gpu/bin/python" \
                     "/opt/conda/envs/rasd/bin/python"; do
              [ -x "$c" ] && { echo "$c"; exit 0; }
            done
            exit 1'
RPY=""   # set by resolve_remote_python; empty means "not resolved yet"

# Local interpreter for this script's own helpers. Never bare `python3`.
PY=$(command -v python3)
for c in "$HOME/miniconda3/envs/rasd-gpu/bin/python" \
         "$HOME/miniconda3/envs/rasd/bin/python" /opt/conda/bin/python; do
  [ -x "$c" ] && PY="$c" && break
done

mkdir -p "$SESSION_DIR"
say() { printf '%s %s\n' "$(date -u +%FT%TZ)" "$*" | tee -a "$LOG"; }

# Look the key up by label, not by line number: an edit anywhere above it would
# otherwise hand us another variable's value and fail only as a 401 mid-wait.
KEY=$(grep -E '^[[:space:]]*LAMBDA_API_KEY[[:space:]]*=' runpod_creds.md 2>/dev/null \
      | tail -1 | sed -E 's/^[^=]*=[[:space:]]*//' | tr -d '`"' | tr -d '[:space:]')
if [ -z "${KEY:-}" ]; then
  say "FATAL: no LAMBDA_API_KEY line found in runpod_creds.md (looked up by label)"
  exit 2
fi

api_get() { curl -sS --max-time 60 -u "$KEY:" "https://cloud.lambda.ai/api/v1/$1"; }

# The HF token, required for the same reason and read the same way. Every target
# model in this campaign is a GATED repo (meta-llama/Llama-2-7b-hf,
# meta-llama/Llama-3.1-8B); without a token that can read them, the stages fail
# at model load. That failure used to be discovered hours in, so it is checked
# here, before anything is launched. NEVER echoed: not in `say`, not in a log.
HF_TOKEN_VALUE=$(grep -E '^[[:space:]]*HF_TOKEN[[:space:]]*=' runpod_creds.md 2>/dev/null \
      | tail -1 | sed -E 's/^[^=]*=[[:space:]]*//' | tr -d '`"' | tr -d '[:space:]')
if [ -z "${HF_TOKEN_VALUE:-}" ]; then
  say "FATAL: no HF_TOKEN line found in runpod_creds.md (looked up by label)."
  say "FATAL: the campaign's models are gated; without a token every arm fails"
  say "FATAL: at model load. Refusing to launch and bill for that."
  exit 2
fi
say "HF token: present ($(printf '%s' "$HF_TOKEN_VALUE" | wc -c | tr -d ' ') chars, not shown)"

count_instances() {
  local n
  n=$(api_get instances 2>/dev/null | "$PY" -c \
      "import json,sys; print(len(json.load(sys.stdin).get('data',[])))" 2>/dev/null)
  echo "${n:-unknown}"
}

# --------------------------------------------------------------------------
# deadline
# --------------------------------------------------------------------------
# Waiting for capacity has no campaign deadline: a slot may not appear for
# hours, and counting that wait against the campaign would silently shorten it.
# The wait is bounded separately (MLSYS_CAPACITY_WAIT_HOURS); the CAMPAIGN clock
# starts when an instance is actually acquired.
# Through awk, not $(( )): the hours can legitimately be fractional (a short
# rehearsal uses 0.05h), and bash arithmetic is integer-only -- it does not
# round a fractional operand, it fails the assignment and leaves the variable
# unset, which then trips `set -u` a few lines later with a message that points
# at the wrong place.
CAPACITY_WAIT_DEADLINE=$(awk -v now="$(date -u +%s)" \
  -v h="${MLSYS_CAPACITY_WAIT_HOURS:-72}" 'BEGIN{printf "%d", now + h * 3600}')
# The instance's own watchdog is the OUTER campaign clock, not a separate
# number. If they disagree, the inside guard can kill a stage the outside clock
# still considers in budget, or the outside clock can end the run while the
# inside guard is still happy to start a stage -- two answers to one question.
# MLSYS_MAX_HOURS, if set, must match; naming a different value is refused
# rather than silently preferred.
CAMPAIGN_HOURS=${MLSYS_CAMPAIGN_HOURS:-40}
if [ -n "${MLSYS_MAX_HOURS:-}" ] && [ "$MLSYS_MAX_HOURS" != "$CAMPAIGN_HOURS" ]; then
  say "FATAL: MLSYS_MAX_HOURS=$MLSYS_MAX_HOURS but the campaign clock is" \
      "${CAMPAIGN_HOURS}h. They must agree; setting them differently means the" \
      "inside watchdog and the outside deadline can kill each other's work."
  exit 7
fi
say "instance watchdog = campaign clock = ${CAMPAIGN_HOURS}h"
# Absolute override, when the operator names one explicitly.
DEADLINE=${MLSYS_DEADLINE_EPOCH:-0}
say "target=$INSTANCE_TYPE  per-stage approval threshold=\$$ASK_OVER @ \$$RATE/hr"
say "approved stages: ${MLSYS_APPROVED_STAGES:-<none>}"
say "allowlist: ${ONLY_STR:-<empty: nothing may run>}"
if [ "${DEADLINE:-0}" -gt 0 ] 2>/dev/null; then
  say "absolute deadline epoch=$DEADLINE ($(date -u -r "$DEADLINE" +%FT%TZ 2>/dev/null || date -u +%FT%TZ -d "@$DEADLINE"))"
else
  say "campaign deadline is set when the instance is acquired (+${MLSYS_CAMPAIGN_HOURS:-40}h)"
fi
say "capacity wait bounded until epoch=$CAPACITY_WAIT_DEADLINE (+${MLSYS_CAPACITY_WAIT_HOURS:-72}h)"

# --------------------------------------------------------------------------
# phase A: wait for capacity, then launch
# --------------------------------------------------------------------------
rm -f "$FOUND"
INSTANCE_ID=""
TERMINATED=0
# How many times to re-issue terminate and re-poll before declaring the
# termination unconfirmed. Each cycle is 40 polls x 15s.
TERMINATE_ATTEMPTS=${MLSYS_TERMINATE_ATTEMPTS:-3}

# Terminate and poll to ZERO. Idempotent and safe to call from a trap, so it can
# run on every exit path. Without the trap, Ctrl-C, a dropped session or any
# early exit left the instance running and billing with nobody watching it.
interruptible_sleep() {        # a plain `sleep` defers the trap until it ends
  # The child must not inherit stdout: an orphaned `sleep` holding the pipe open
  # would block anything reading our output until it expired on its own.
  sleep "$1" </dev/null >/dev/null 2>&1 &
  wait $! 2>/dev/null || true
}

terminate_and_confirm() {
  [ "$TERMINATED" = "1" ] && return 0
  [ -z "$INSTANCE_ID" ] && return 0
  say "TERMINATING $INSTANCE_ID"
  curl -sS --max-time 60 -u "$KEY:" -X POST \
    "https://cloud.lambda.ai/api/v1/instance-operations/terminate" \
    -H 'Content-Type: application/json' \
    -d "{\"instance_ids\":[\"$INSTANCE_ID\"]}" >>"$LOG" 2>&1
  say "confirming termination by polling to ZERO instances (stderr visible)"
  local confirmed=0 resp rc n attempt
  for attempt in $(seq 1 "$TERMINATE_ATTEMPTS"); do
    [ "$attempt" -gt 1 ] && {
      say "  re-issuing terminate (attempt $attempt)"
      curl -sS --max-time 60 -u "$KEY:" -X POST \
        "https://cloud.lambda.ai/api/v1/instance-operations/terminate" \
        -H 'Content-Type: application/json' \
        -d "{\"instance_ids\":[\"$INSTANCE_ID\"]}" >>"$LOG" 2>&1
    }
    for i in $(seq 1 40); do
      resp=$(api_get instances 2>&1); rc=$?
      if [ $rc -ne 0 ]; then
        say "  attempt $attempt.$i: api rc=$rc :: $(printf '%s' "$resp" | head -c 120)"
        interruptible_sleep 15; continue
      fi
      n=$(printf '%s' "$resp" | "$PY" -c \
        "import json,sys; print(len(json.load(sys.stdin).get('data',[])))" 2>&1)
      say "  attempt $attempt.$i: instances=$n"
      if [ "$n" = "0" ]; then confirmed=1; break; fi
      interruptible_sleep 15
    done
    [ "$confirmed" = "1" ] && break
    interruptible_sleep 30
  done
  # TERMINATED means CONFIRMED. Setting it after an unconfirmed attempt is how
  # a live instance survives the script that was supposed to end it: the flag
  # makes every later attempt a no-op, including the one from the exit trap.
  if [ "$confirmed" = "1" ]; then
    TERMINATED=1
    say "CONFIRMED TERMINATED (0 instances)"
    return 0
  fi
  say "!!! NOT CONFIRMED TERMINATED after $TERMINATE_ATTEMPTS x 40 polls"
  say "!!! the instance may still be billing: $INSTANCE_ID — CHECK THE DASHBOARD"
  return 1
}

# Pull BEFORE terminating on every signal path.
#
# WHY THIS EXISTS. `terminate_and_confirm` terminates the instance, and the pod's
# results go with it. Every other stop path in this script calls collect_incident
# first -- the stall paths, the dead-CUDA path, the deadline path. The signal path
# did not, which is exactly backwards: a signal is the case where nobody is
# watching, so the results are the only thing left.
#
# On 2026-10-10T00:37:04Z this path did the damage. The copilot runtime restarted
# (it came up at 00:37:15Z; the watcher and the monitor both died within 35s of
# each other) and SIGTERMed this script, which was 49 minutes into a healthy
# engine_cap_smoke. It logged "TERMINATING", terminated a working 8xA100, wrote
# NOTHING back, and exited 143 -- so that run's gate_calibration.csv (676s, $4.19)
# and its three completed engine_cap_smoke rows were destroyed with the pod. The
# only reason the campaign knows what they said is that the numbers were read out
# by hand while the pod was alive.
#
# The pull is bounded and best-effort: a stop must still stop. `timeout` caps it
# so an unreachable pod cannot hold the operator's Ctrl-C open, and a failure here
# is reported and stepped over rather than trapping the script.
pull_then_terminate() {
  [ "$TERMINATED" = "1" ] && return 0
  [ -z "$INSTANCE_ID" ] && return 0
  # The normal completion path below pulls through its own sha256-verified stage
  # and merges it. Pulling a second time there would re-merge the cost ledger.
  if [ -z "${RESULTS_HANDLED:-}" ] && [ -n "${IP:-}" ] \
     && [ "$(type -t collect_incident)" = "function" ]; then
    RESULTS_HANDLED=1
    say "STOP: pulling logs and results BEFORE terminating (signal or exit path)"
    if timeout 900 collect_incident "stop-signal-or-exit"; then
      say "STOP: the pull completed; terminating"
    else
      say "!!! STOP: the pull FAILED or timed out; terminating anyway (the"
      say "!!! instance is billing and a stop must stop)"
    fi
  fi
  terminate_and_confirm
}

# EXIT covers normal completion and explicit `exit`; INT/TERM cover Ctrl-C and a
# dropped session. TERMINATED makes the double-fire a no-op.
trap pull_then_terminate EXIT
trap 'pull_then_terminate; exit 130' INT
trap 'pull_then_terminate; exit 143' TERM

# --------------------------------------------------------------------------
# capacity wait: detect -> LAUNCH, in the same iteration
# --------------------------------------------------------------------------
# The failure this replaces: capacity for this GPU type appears and disappears
# within a couple of minutes (measured 2026-10-08: a window was visible at
# 00:49:14Z and gone by the 00:50:38Z poll). Anything that detects capacity in
# one tick and launches in the next is racing a window that is shorter than the
# tick, which is a bug in the cadence, not bad luck.
#
# So: the check and the launch are ONE step, the interval is sized to the window
# (~3 min), a human's "go now" is honoured within 10 s, and every launch attempt
# -- scheduled, detected or manual -- passes the same guard.
LAUNCH_NOW=$SESSION_DIR/LAUNCH_NOW
POLL_BASE=${MLSYS_POLL_BASE_S:-180}      # base cadence
POLL_JITTER=${MLSYS_POLL_JITTER_S:-45}   # +/- uniform, so 135-225s
BACKOFF_MAX=${MLSYS_BACKOFF_MAX_S:-900}  # ceiling for 429/5xx backoff
REGION_PREF=${MLSYS_REGION_PREF:-us-east-1,us-midwest-1,us-west-1,us-south-1,us-west-2}

# The ONLY path that starts an instance. Every caller routes through it, which
# is the only way the guard can be a property of the system rather than of each
# call site.
try_launch() {   # $1 = reason, $2 = region
  local reason=$1 region=$2 n resp
  n=$(count_instances)
  if [ "$n" != "0" ]; then
    say "LAUNCH_ABORT reason=$reason already_running=$n region=$region"
    say "  exactly one 8xA100 at a time; not launching a second"
    return 2
  fi
  if [ -z "$region" ]; then
    say "LAUNCH_ABORT reason=$reason no_region"
    return 2
  fi
  # Rehearsal path: exercises the whole decision chain -- guard, region choice,
  # override handling -- without creating a billable instance. It reports a
  # REFUSED outcome on purpose, so a dry run never deletes LAUNCH_NOW and never
  # sets INSTANCE_ID (the caller would otherwise believe it had a pod).
  if [ -n "${MLSYS_LAUNCH_DRY_RUN:-}" ]; then
    say "LAUNCH_DRY_RUN reason=$reason region=$region (no instance created)"
    return 1
  fi
  resp=$(curl -sS --max-time 120 -u "$KEY:" -X POST \
    "https://cloud.lambda.ai/api/v1/instance-operations/launch" \
    -H 'Content-Type: application/json' \
    -d "{\"region_name\":\"$region\",\"instance_type_name\":\"$INSTANCE_TYPE\",\"ssh_key_names\":[\"rasd-amank\"],\"name\":\"rasd-mlsys\",\"quantity\":1}" 2>&1)
  INSTANCE_ID=$(printf '%s' "$resp" | "$PY" -c "
import json,sys
try: print(json.loads(sys.stdin.read())['data']['instance_ids'][0])
except Exception: print('')" 2>/dev/null)
  if [ -n "$INSTANCE_ID" ]; then
    say "LAUNCHED $INSTANCE_ID in $region (reason=$reason)"
    echo "$region" > "$FOUND"
    return 0
  fi
  say "LAUNCH_REFUSED reason=$reason region=$region: $(printf '%s' "$resp" | head -c 160)"
  return 1
}

# Regions that list this instance type, in preference order, comma-separated.
regions_with_capacity() {
  api_get instance-types 2>/dev/null | "$PY" -c "
import json,sys
try: d=json.load(sys.stdin).get('data',{})
except Exception: d={}
t=d.get('$INSTANCE_TYPE',{})
print(','.join(r['name'] for r in t.get('regions_with_capacity_available',[])))" 2>/dev/null
}

# HTTP status of the capacity call, so 429/5xx can back off.
capacity_status() {
  curl -s -o /dev/null -w '%{http_code}' --max-time 60 -u "$KEY:" \
    "https://cloud.lambda.ai/api/v1/instance-types" 2>/dev/null
}

# Manual override: attempt a launch now, whatever the schedule says. Kept until
# a launch SUCCEEDS, so a refused attempt is retried rather than silently lost.
launch_now() {
  local avail region rc
  avail=$(regions_with_capacity)
  region=${avail%%,*}
  if [ -z "$region" ]; then
    # The advisory list is often empty when capacity is actually available (and
    # sometimes non-empty when it is not), so a human's go-signal is not
    # discarded just because the list is empty: try the preferred regions.
    for region in $(printf '%s' "$REGION_PREF" | tr ',' ' '); do
      rc=0; try_launch "launch_now" "$region" || rc=$?
      [ "$rc" = "0" ] && { rm -f "$LAUNCH_NOW"; return 0; }
      [ "$rc" = "2" ] && return 2                     # already running: stop
    done
    say "LAUNCH_NOW: all preferred regions refused; keeping the file and retrying"
    return 1
  fi
  rc=0; try_launch "launch_now" "$region" || rc=$?
  if [ "$rc" = "0" ]; then rm -f "$LAUNCH_NOW"; return 0; fi
  [ "$rc" = "2" ] && return 2
  say "LAUNCH_NOW: $region refused; keeping the file and retrying"
  return 1
}

# Sleep in 10 s slices so a LAUNCH_NOW written mid-sleep is honoured within 10 s
# rather than after the remaining nap. Returns 0 when the caller should exit its
# loop (launched, or another instance appeared).
interruptible_nap() {   # $1 = seconds
  local left=$1 slice=10
  while [ "$left" -gt 0 ]; do
    if [ -f "$LAUNCH_NOW" ]; then
      # The FILE is checked every 10s, as specified. The ATTEMPTS are spaced by
      # 30s so a standing override against a full fleet does not turn into a
      # launch request every 10s for hours -- a refusal and a rate limit look
      # the same from here, and only one of them is safe to generate.
      local tnow
      tnow=$(date -u +%s)
      if [ $(( tnow - ${LAST_NOW_ATTEMPT:-0} )) -ge 30 ]; then
        LAST_NOW_ATTEMPT=$tnow
        say "LAUNCH_NOW seen (${left}s left in this nap); attempting launch"
        launch_now
        case $? in
          0) return 0 ;;
          2) EXIT_REASON="other_instance"; return 0 ;;
        esac
      fi
      slice=10; [ "$left" -lt 10 ] && slice=$left
      interruptible_sleep $slice
      left=$((left - slice))
      continue
    fi
    # The last slice is shortened to what is left. Sleeping a flat 10 s made
    # every nap overshoot by up to 10 s -- invisible at a 3 min cadence, but it
    # also meant a 1 s backoff waited 10 s, i.e. the function did not honour its
    # argument.
    slice=10; [ "$left" -lt 10 ] && slice=$left
    interruptible_sleep $slice
    left=$((left - slice))
  done
  return 1
}

attempt=0
BACKOFF=0
LAST_NOW_ATTEMPT=0
# Why the wait ended when nothing was launched. The guard returning 2 is
# "somebody else's instance exists", which is NOT an expired wait: it needs
# the exit code and the message of the refusal path (5), not the deadline's (3).
EXIT_REASON="expired"
while [ "$(date -u +%s)" -lt "$CAPACITY_WAIT_DEADLINE" ]; do
  attempt=$((attempt+1))

  # A human saying "capacity is available" is an instruction to launch now, not
  # a hint to wait for the next tick. Checked before the API work so it cannot
  # be delayed by a slow call.
  if [ -f "$LAUNCH_NOW" ]; then
    # Record the attempt so the nap's own 10s check cannot immediately repeat
    # it. Without this, LAST_NOW_ATTEMPT is still 0 on the first pass and the
    # spacing guard is satisfied by the epoch itself, so the override fired
    # twice in the same second -- the release decided the attempt spacing.
    LAST_NOW_ATTEMPT=$(date -u +%s)
    launch_now
    case $? in
      0) break ;;
      2) INSTANCE_ID=""; EXIT_REASON="other_instance"; break ;;
    esac
  fi

  status=$(capacity_status)
  case "$status" in
    429|5*)
      BACKOFF=$(( BACKOFF == 0 ? POLL_BASE : BACKOFF * 2 ))
      [ "$BACKOFF" -gt "$BACKOFF_MAX" ] && BACKOFF=$BACKOFF_MAX
      nap=$BACKOFF
      say "attempt=$attempt regions=- capacity=unknown http=$status next_sleep=${nap}s (backoff)"
      interruptible_nap $nap && break
      continue ;;
  esac
  BACKOFF=0

  n=$(count_instances)
  case "$n" in
    ''|*[!0-9]*)
      # An unreadable count is NOT "an instance exists". Treating it as one
      # would end the watch -- permanently, and for the wrong reason -- on a
      # single transient API failure. Back off and re-read instead. (The launch
      # guard still fails CLOSED on an unreadable count: it refuses to launch.)
      BACKOFF=$(( BACKOFF == 0 ? POLL_BASE : BACKOFF * 2 ))
      [ "$BACKOFF" -gt "$BACKOFF_MAX" ] && BACKOFF=$BACKOFF_MAX
      say "attempt=$attempt regions=- capacity=unknown instance_count=$n next_sleep=${BACKOFF}s (retrying)"
      interruptible_nap "$BACKOFF" && break
      continue ;;
  esac
  if [ "$n" != "0" ]; then
    OTHER=$(api_get instances | "$PY" -c \
      "import json,sys;d=json.load(sys.stdin)['data'];print(','.join(x['id'] for x in d))")
    # REFUSE, with no adoption path at all. An adopted instance is one this
    # script did not launch, was not sized for this campaign, and would later
    # TERMINATE. An opt-in escape hatch for reusing a running instance existed
    # and is gone: the invariant "we only ever terminate the id we launched" is
    # worth more than the convenience of reusing a machine whose provenance we
    # do not know. No variable, flag or env var may reintroduce it, so this
    # block names none -- `mlsys_dry_run.sh` step 15 asserts that.
    say "FATAL: $n instance(s) already exist ($OTHER)."
    say "This watcher only runs on an instance IT launches, so that it only ever"
    say "terminates one it launched. Terminate them yourself, then re-run."
    exit 5
  fi

  avail=$(regions_with_capacity)
  nap=$(( POLL_BASE - POLL_JITTER + RANDOM % (2 * POLL_JITTER + 1) ))
  [ "$nap" -lt 30 ] && nap=30
  if [ -n "$avail" ]; then
    say "attempt=$attempt regions=$avail capacity=yes next_sleep=${nap}s"
    # LAUNCH IN THIS ITERATION. No handoff, no second loop, no waiting for the
    # next tick: the window can be shorter than the tick.
    for region in $(printf '%s' "$avail" | tr ',' ' '); do
      rc=0; try_launch "detected" "$region" || rc=$?
      [ "$rc" = "0" ] && break
      [ "$rc" = "2" ] && break
    done
    if [ -n "$INSTANCE_ID" ]; then break; fi
    say "  every listed region refused; re-polling rather than treating the"
    say "  advisory list as a reservation"
  else
    say "attempt=$attempt regions=none capacity=no next_sleep=${nap}s"
  fi
  interruptible_nap $nap && break
done

if [ -z "$INSTANCE_ID" ]; then
  if [ "$EXIT_REASON" = "other_instance" ]; then
    say "REFUSING TO ADOPT an instance this watcher did not launch."
    say "It only ever terminates the id it launched, so it will not run this"
    say "campaign on a machine it did not size or approve. Nothing was acquired"
    say "by this watcher, so nothing it will bill. Terminate it yourself, then re-run."
    exit 5
  fi
  say "CAPACITY WAIT EXPIRED after ${MLSYS_CAPACITY_WAIT_HOURS:-72}h with nothing launched."
  say "Nothing was acquired, so nothing is billing."
  exit 3
fi

# The instance is acquired: start the campaign clock, sized to the campaign.
if [ "${DEADLINE:-0}" -gt 0 ] 2>/dev/null; then
  say "campaign deadline from MLSYS_DEADLINE_EPOCH (absolute)"
else
  DEADLINE=$(( $(date -u +%s) + ${MLSYS_CAMPAIGN_HOURS:-40} * 3600 ))
  say "campaign clock starts NOW: $(date -u -r "$DEADLINE" +%FT%TZ) (+${MLSYS_CAMPAIGN_HOURS:-40}h, sized to the campaign)"
fi

# --------------------------------------------------------------------------
# wait for the instance, then boot it
# --------------------------------------------------------------------------
IP=""
for i in $(seq 1 60); do
  read -r status ip <<<"$(api_get instances | "$PY" -c "
import json,sys
d=json.load(sys.stdin).get('data',[])
for x in d:
    if x.get('id')=='$INSTANCE_ID':
        print(x.get('status'), x.get('ip') or '-'); break
else: print('gone','-')")"
  if [ "$status" = "active" ] && [ "$ip" != "-" ]; then IP=$ip; break; fi
  [ "$status" = "gone" ] && { say "instance disappeared during boot"; exit 4; }
  interruptible_sleep 20
done
[ -z "$IP" ] && { say "instance never became active"; terminate_and_confirm; exit 4; }
say "instance active at $IP"

for i in $(seq 1 30); do
  ssh $SSH_OPTS "$SSH_USER@$IP" true 2>/dev/null && break
  interruptible_sleep 10
done

say "staging repository + PG-19 data (metadata AND the chunks it names)"
ssh $SSH_OPTS "$SSH_USER@$IP" "mkdir -p ~/RASD/scripts ~/RASD/configs ~/RASD/results" 2>>"$LOG"
# A failed repo rsync used to be silent: the old form was
# `rsync ... && say "repo staged"`, so a non-zero exit merely skipped the
# message and the run continued against a pod with stale or missing code.
# Nothing downstream can recover from that -- every stage reads scripts/ and
# configs/ -- so it is fatal.
# results/mlsys is excluded outbound, EXCEPT the cumulative cost ledger. The
# comment further up has always claimed this exclusion existed; it did not. On
# 2026-10-08 that pushed 31 local artifacts onto the pod, including a stale
# RUN_LOG.txt from a 2026-10-04 macOS pre-flight run ("ABORTED AT PRECHECK 1 --
# NO CUDA GPU") which then appeared in the incident pull as if the pod had
# written it. Only the ledger travels, because the budget guard reads it to
# carry cumulative spend across sessions.
if ! rsync -az --no-perms --no-owner --no-group -e "ssh $SSH_OPTS" \
       --exclude '.git' --exclude 'results/final' --exclude 'manuscript' \
       --exclude '.venv*' --exclude '__pycache__' \
       --include 'results/mlsys/' \
       --include 'results/mlsys/gpu_hours.csv' \
       --exclude 'results/mlsys/*' \
       "$REPO/" "$SSH_USER@$IP:~/RASD/" >>"$LOG" 2>&1; then
  say "FATAL: staging the repository to the pod failed; refusing to run"
  say "FATAL: against code we cannot confirm is the code under test"
  terminate_and_confirm
  exit 6
fi
say "  repo staged"

# --------------------------------------------------------------------------
# provision the CAMPAIGN environment, before anything is measured
# --------------------------------------------------------------------------
# In the FOREGROUND, unlike the vLLM venv below, because nothing can run
# without it: every stage imports transformers through this interpreter. The
# install is multi-GB and flash-attn compiles, so it costs real minutes of paid
# GPU time -- but the alternative is what happened on 2026-10-08: the run
# started, gate_calibration died in two seconds, and the instance billed for two
# hours because the completion check could not fire either.
#
# Provisioning FIRST also means the data-verification commands below run under
# the campaign interpreter rather than a stock `python3'. They only import the
# standard library, so with the old fallback they passed happily on a pod that
# had no transformers at all -- a green check that meant nothing.
say "provisioning the campaign environment on the pod (setup log: ~/pod_env.log)"
# The token goes to the PROVISIONING too, not only to the manifest. pod_env.sh
# requires it (it proves gated-model access as part of verification), and on
# 2026-10-08 this command omitted it: the script failed closed, the watcher
# terminated the instance ten minutes after launching it, and the campaign never
# started. Wiring the token into one of the two places that need it is exactly
# the kind of half-fix that costs an 8x launch.
if ! ssh $SSH_OPTS "$SSH_USER@$IP" \
      "cd ~/RASD && HF_TOKEN='$HF_TOKEN_VALUE' bash scripts/mlsys_pod_env.sh > ~/pod_env.log 2>&1"; then
  say "FATAL: the campaign environment could not be provisioned."
  say "FATAL: ~/pod_env.log tail:"
  ssh $SSH_OPTS "$SSH_USER@$IP" 'tail -25 ~/pod_env.log' 2>&1 | sed 's/^/    /' | tee -a "$LOG"
  say "FATAL: refusing to start a campaign that cannot import transformers."
  terminate_and_confirm
  exit 6
fi
ssh $SSH_OPTS "$SSH_USER@$IP" 'grep -E "^    " ~/pod_env.log' 2>&1 | sed 's/^/  /' | tee -a "$LOG"

# Resolve the interpreter NOW, once the environment exists, and hold it for the
# rest of the run. Failing here is free; failing inside a stage is not.
#
# Two sources, because the provisioning script already knows the answer: it
# prints INTERPRETER=<path> as its last line. Reading that is primary; the
# RPY_RESOLVE command is the fallback for a pod provisioned by something else.
# Either way the result must be an absolute path, so a stray message on stdout
# cannot be mistaken for one.
PY_REMOTE=$(ssh $SSH_OPTS "$SSH_USER@$IP" \
  'grep -a "^INTERPRETER=" ~/pod_env.log 2>/dev/null | tail -1 | cut -d= -f2-' 2>/dev/null)
if [ -z "$PY_REMOTE" ]; then
  say "  (no INTERPRETER= line in ~/pod_env.log; resolving it directly)"
  PY_REMOTE=$(ssh $SSH_OPTS "$SSH_USER@$IP" "$RPY_RESOLVE" 2>/dev/null)
fi
case "$PY_REMOTE" in
  /*) ;;
  "") say "FATAL: no campaign interpreter on the pod (looked for the rasd-gpu env)."
      say "FATAL: the environment was provisioned, so either it is not where the"
      say "FATAL: resolution looks or the resolution itself is broken."
      say "FATAL: not starting a run that would die in its first stage."
      terminate_and_confirm
      exit 6 ;;
  *)  say "FATAL: interpreter resolution returned '$(printf '%s' "$PY_REMOTE" | head -c 80)',"
      say "FATAL: which is not an absolute path. Refusing to use it."
      terminate_and_confirm
      exit 6 ;;
esac
if ! ssh $SSH_OPTS "$SSH_USER@$IP" "test -x '$PY_REMOTE'" 2>/dev/null; then
  say "FATAL: the resolved interpreter '$PY_REMOTE' is not executable on the pod."
  terminate_and_confirm
  exit 6
fi
say "  interpreter: $PY_REMOTE"
# One resolution, checked once, used by every subsequent command as a plain path.
RPY=$PY_REMOTE

# The green check that is actually green: prove the interpreter the STAGES will
# use can import what the stages import, before the manifest is started.
if ! ssh $SSH_OPTS "$SSH_USER@$IP" \
      "$RPY -c \"import torch, transformers, bitsandbytes, flash_attn, diptest; \
print('  torch', torch.__version__, '| transformers', transformers.__version__, \
'| cuda', torch.cuda.is_available(), torch.cuda.device_count(), 'devices')\"" \
      2>&1 | tee -a "$LOG" | grep -q "torch"; then
  say "FATAL: the campaign interpreter cannot import the stage dependencies."
  say "FATAL: not starting a run that would die in its first stage."
  terminate_and_confirm
  exit 6
fi

# The metadata names relative paths; the memmap files must travel with it or the
# stage dies exactly like pg19_short_target did. Verify AFTER the copy, from the
# run directory, for both metadata shapes (per-book `documents` and the older
# concatenated `chunks`). pg19_docs is the primary pool; pg19_docs_diverse backs
# the diverse-pool arm; the two older dirs still back legacy cells.
for d in data/processed/pg19_docs data/processed/pg19_docs_diverse \
         data/processed/pg19_llama3 data/processed/pg19; do
  [ -d "$d" ] || continue
  if ! rsync -az --no-perms --no-owner --no-group -e "ssh $SSH_OPTS" \
         "$d/" "$SSH_USER@$IP:~/RASD/$d/" >>"$LOG" 2>&1; then
    say "FATAL: staging $d to the pod failed; refusing to run on incomplete data"
    terminate_and_confirm
    exit 6
  fi
  # The pod's own interpreter, preferring the environment that has torch and
  # transformers. Bare `python3` on the pod may be a stock system python.
  ssh $SSH_OPTS "$SSH_USER@$IP" "cd ~/RASD && $RPY -c \"
import json,pathlib,sys
m=json.load(open('$d/documents.json')) if pathlib.Path('$d/documents.json').exists() else json.load(open('$d/pg19_validation_metadata.json'))
items=m.get('documents') or m.get('chunks') or []
missing=[c['file'] for c in items if not pathlib.Path(c['file']).exists()]
print('  $d: %d files, %d unresolved' % (len(items), len(missing)))
sys.exit(1 if missing else 0)\"" >>"$LOG" 2>&1 \
    && say "  $d verified" || {
      say "FATAL: $d has unresolved metadata paths on the pod; the stage would"
      say "FATAL: die mid-run on a paid instance. Staging is incomplete."
      terminate_and_confirm
      exit 6
    }
done

# --------------------------------------------------------------------------
# provision the ISOLATED vLLM environment (R7)
# --------------------------------------------------------------------------
# The vLLM stages run in their own venv with their own torch, selected by path;
# if it is missing they REFUSE rather than run against the campaign's pins. So it
# is provisioned here, on the pod, as part of the setup.
#
# Started in the BACKGROUND, not blocking: the install is a multi-GB download
# and impl_validation is the sixth stage, hours behind gate_calibration and the
# 128k natural-text rung. Blocking would idle a paid 8xA100 for the length of a
# pip install; running it concurrently costs nothing and the pin makes a
# mid-install state impossible to observe (pip writes the venv's python last).
#
# A failure is NOT fatal to the campaign: the vLLM stages then refuse and the
# manifest records that honestly as MANIFEST INCOMPLETE, which is the correct
# outcome for "we could not build the reference". Killing the 128k stages over a
# pip failure would be the wrong trade.
ssh $SSH_OPTS "$SSH_USER@$IP" \
  "cd ~/RASD && nohup bash scripts/mlsys_vllm_venv.sh > ~/vllm_venv.log 2>&1 & echo started" \
  >>"$LOG" 2>&1 && say "  vLLM venv provisioning started (log: ~/vllm_venv.log)" \
  || say "  WARNING: could not start the vLLM venv provisioning (the vLLM stages will refuse)"

# --------------------------------------------------------------------------
# run the manifest
# --------------------------------------------------------------------------
say "starting the manifest"
# The manifest runs behind a wrapper that writes its exit code to ~/manifest.rc.
# Liveness via `pgrep -f mlsys_manifest.sh' could not work: the remote shell
# that runs the check has that pattern in ITS OWN argv, so pgrep matched the
# checker and returned 0 forever. On 2026-10-08 that meant the loop below could
# never break -- the manifest had aborted after two seconds, and the instance
# would have billed until the 40h deadline. A marker file is unambiguous, and it
# carries the exit code, so "finished" and "finished badly" stop looking alike.
# The marker is removed first: a stale one from an earlier attempt would report
# a completion that has not happened.
#
# THE WHOLE CHAIN IS GROUPED AND REDIRECTED, and that grouping is the fix, not
# decoration. `&' has lower precedence than `&&', so this
#
#     cd ~/RASD && . env.sh && nohup manifest > ~/manifest.log 2>&1 & echo started
#
# backgrounds the ENTIRE `&&' chain, not just the manifest. The backgrounded
# subshell keeps the channel as its stdout (only the innermost command's output
# was redirected), so ssh does not see EOF and does not return until the manifest
# EXITS. Measured 2026-10-09 on a 1x A100 against a 12s stub: 12.5s with the
# `cd', 0.4s without it. That is what pinned the watcher's ssh open for 1h42m on
# 2026-10-09 -- it never reached its monitoring loop, so nothing reacted to the
# stall its own watchdog had already detected, and the run continued 70 minutes.
#
# `</dev/null' was the earlier hypothesis (stdin, not stdout) and it does NOT fix
# this: an A/B on the pod measured 12.6s and 12.5s with it and without it. It is
# kept because a manifest that cannot read the channel cannot hold it either.
# The parenthesis + redirects are asserted by the tests, because this is exactly
# the kind of invisible quoting detail that regresses silently.
ssh $SSH_OPTS "$SSH_USER@$IP" \
  "( cd ~/RASD && rm -f ~/manifest.rc && set -a && . ~/RASD/.pod_env.sh && set +a && \
   HF_TOKEN='$HF_TOKEN_VALUE' \
   MLSYS_PYTHON='$PY_REMOTE' \
   NODE_RATE_PER_HOUR=$RATE \
   MLSYS_STALL_MINUTES=${MLSYS_STALL_MINUTES:-20} \
   MLSYS_ASK_OVER_USD=${MLSYS_ASK_OVER_USD:-300} \
   MLSYS_MAX_HOURS=${MLSYS_CAMPAIGN_HOURS:-40} \
   MLSYS_MAX_COST_USD=${MLSYS_MAX_COST_USD:-850} \
   MLSYS_ONLY_STAGES='${ONLY_STR}' \
   MLSYS_APPROVED_STAGES='${ONLY_STR}' \
   nohup bash -c 'bash scripts/mlsys_manifest.sh; echo \$? > ~/manifest.rc' ) \
     </dev/null > ~/manifest.log 2>&1 & echo started" >>"$LOG" 2>&1

# A failed ssh says NOTHING about whether the manifest is still running. Treating
# it as "finished" ends the run and, worse, moves on to terminate an instance
# whose work is still in progress. The marker file is only read after an ssh
# that exited 0, so an unreachable pod is retried, never mistaken for "done".
# --------------------------------------------------------------------------
# incident collection: logs only, then stop the meter
# --------------------------------------------------------------------------
# Used by every unexpected ending: an aborted manifest (rc != 0), a manifest
# process that vanished without a marker, and a stall. The point is to preserve
# the evidence and then STOP, rather than sitting on a paid GPU working out what
# happened.
#
# Logs AND results. The original version of this function pulled logs only, on
# the reasoning that the normal path already pulls results and that pulling
# again would delay termination. The 2026-10-08T14:23Z gate run disproved it:
# the gate measured all nine controls, wrote `gate_calibration.csv` and the nine
# generated continuations, was refused by its own controls, and the watcher
# captured the logs, terminated, and lost the CSV -- the pod stopped answering
# ssh within about 30 seconds. The logs carry the REASON a run failed; they do
# not carry its ROWS, and the rows are what the next attempt has to be planned
# against. That cost a whole re-run of the gate to learn nothing new.
#
# What the pull must NOT do is overwrite good local work with a partial one.
# So: everything lands in the incident directory first, and merging into
# `results/` is conservative per file type --
#   * the cost ledger ACCUMULATES, so it is merged row-by-row, never replaced
#     (scripts/mlsys_merge_cost_ledger.py);
#   * every other CSV and generated-text directory is copied in only where
#     nothing is there yet; an existing local file wins and the pod's copy stays
#     in the incident directory as the evidence.
collect_incident() {   # $1 = reason (short, no newlines)
  local reason=$1 INC f
  INC="$REPO/results/mlsys/incident_$(date -u +%Y%m%dT%H%M%SZ)"
  mkdir -p "$INC"
  say "INCIDENT ($reason): logs + results -> $INC"
  {
    echo "reason : $reason"
    echo "utc    : $(date -u +%FT%TZ)"
    echo "instance: ${INSTANCE_ID:-none}"
    echo "region : $(cat "$FOUND" 2>/dev/null || echo unknown)"
    echo "manifest_rc: ${MANIFEST_RC:-<none>}"
    echo "host   : $(hostname)"
  } > "$INC/incident.txt"
  for f in manifest.log manifest.rc pod_env.log vllm_venv.log manifest_watchdog.log; do
    if ssh $SSH_OPTS "$SSH_USER@$IP" "test -f ~/$f" 2>/dev/null; then
      ssh $SSH_OPTS "$SSH_USER@$IP" "cat ~/$f" > "$INC/$f" 2>/dev/null \
        && say "  pulled ~/$f ($(wc -l < "$INC/$f" | tr -d ' ') lines)" \
        || say "  ~/$f exists but could not be read"
    else
      say "  ~/$f not present"
    fi
  done
  if ssh $SSH_OPTS "$SSH_USER@$IP" 'test -f ~/RASD/results/mlsys/RUN_LOG.txt' 2>/dev/null; then
    ssh $SSH_OPTS "$SSH_USER@$IP" 'cat ~/RASD/results/mlsys/RUN_LOG.txt' \
      > "$INC/RUN_LOG.txt" 2>/dev/null && say "  pulled RUN_LOG.txt"
  fi

  # --- results, before we terminate anything -------------------------------
  local POD_RESULTS="$INC/pod_results" n_csv n_gen copied kept rel src
  mkdir -p "$POD_RESULTS"
  # `--timeout` bounds the I/O so a wedged transfer cannot hold termination
  # open; a failure here is reported and does not stop the shutdown.
  if rsync -az --no-perms --no-owner --no-group -e "ssh $SSH_OPTS" --timeout=120 \
         --exclude 'checkpoints/' --exclude 'attempts/' \
         "$SSH_USER@$IP:~/RASD/results/mlsys/" "$POD_RESULTS/" >>"$LOG" 2>&1; then
    n_csv=$(find "$POD_RESULTS" -name '*.csv' | wc -l | tr -d ' ')
    n_gen=$(find "$POD_RESULTS" -type d -name '*_generated' | wc -l | tr -d ' ')
    say "  pulled results/mlsys ($n_csv csv, $n_gen generated-text dir(s))"
    # 1. the ledger: merged, never replaced
    if [ -f "$POD_RESULTS/gpu_hours.csv" ]; then
      if "$PY" "$REPO/scripts/mlsys_merge_cost_ledger.py" \
           --local "$REPO/results/mlsys/gpu_hours.csv" \
           --pod "$POD_RESULTS/gpu_hours.csv" > "$INC/ledger_merge.json" \
           2>>"$LOG"; then
        say "  ledger merge: $(cat "$INC/ledger_merge.json")"
      else
        say "  !!! ledger merge REFUSED: $(cat "$INC/ledger_merge.json" 2>/dev/null)"
        say "  !!! the pod's ledger is kept at $POD_RESULTS/gpu_hours.csv"
      fi
    fi
    # 2. everything else: copy only where the local path does not exist
    copied=0
    kept=0
    while IFS= read -r src; do
      [ -n "$src" ] || continue
      rel=${src#"$POD_RESULTS/"}
      case "$rel" in gpu_hours.csv) continue ;; esac
      if [ -e "$REPO/results/mlsys/$rel" ]; then
        kept=$((kept + 1))
      else
        mkdir -p "$(dirname "$REPO/results/mlsys/$rel")"
        cp -R "$src" "$REPO/results/mlsys/$rel"
        copied=$((copied + 1))
      fi
    done <<PODLIST
$(find "$POD_RESULTS" \( -name '*.csv' -o -type d -name '*_generated' \) | sort)
PODLIST
    say "  $copied path(s) newly copied into results/mlsys, $kept already present locally (pod copy kept in $POD_RESULTS)"
  else
    say "  !!! results rsync FAILED; the incident holds logs only"
  fi

  cp "$LOG" "$INC/watcher.log" 2>/dev/null || true
  say "INCIDENT captured at $INC"
}

MANIFEST_RC=""
SSH_UNKNOWN=0
STALL_MIN=${MLSYS_STALL_MINUTES:-20}
STALL_S=$(( STALL_MIN * 60 ))
PROGRESS_SIG=""
PROGRESS_SINCE=$(date -u +%s)
MANIFEST_POLL=${MLSYS_MANIFEST_POLL_S:-60}

# The threshold the operator's side applies, per stage, read from the SAME file
# the pod watchdog reads so the two cannot disagree about what "quiet" means.
STALL_JSON="$REPO/configs/mlsys_stall_thresholds.json"
stall_minutes_for() {   # $1 = stage name (may be empty)
  local name=$1 v
  v=$("$PY" - "$STALL_JSON" "$name" "$STALL_MIN" <<'PYSTALL' 2>/dev/null
import json, sys
path, name, fallback = sys.argv[1], sys.argv[2], int(sys.argv[3])
try:
    d = json.load(open(path))
except Exception:
    print(fallback); raise SystemExit
print(int(d.get("stages", {}).get(name, d.get("default_minutes", fallback))))
PYSTALL
)
  case "$v" in ''|*[!0-9]*) v=$STALL_MIN ;; esac
  printf '%s' "$v"
}

# The byte-level liveness threshold: how long the pod's ~/manifest.log may go
# without growing while a stage runs. Read from the SAME file the pod watchdog
# reads, so the two cannot disagree -- and deliberately without the per-stage
# override, because the whole point is to be tighter than every stage value.
liveness_minutes() {
  local v
  v=$("$PY" - "$STALL_JSON" "${MLSYS_MANIFEST_LIVENESS_MIN:-15}" <<'PYLIVE' 2>/dev/null
import json, sys
path, fallback = sys.argv[1], int(sys.argv[2])
try:
    d = json.load(open(path))
except Exception:
    print(fallback); raise SystemExit
print(int(d.get("liveness_minutes", fallback)))
PYLIVE
)
  case "$v" in ''|*[!0-9]*) v=15 ;; esac
  printf '%s' "$v"
}
LIVE_MIN=$(liveness_minutes)
LIVE_SIG=""
LIVE_SINCE=""

while true; do
  # One round trip for the marker, the progress signature and the current stage.
  probe=$(ssh $SSH_OPTS "$SSH_USER@$IP" '
    f=~/manifest.log; r=~/RASD/results/mlsys/RUN_LOG.txt
    if [ -f ~/manifest.rc ]; then printf "rc=%s\n" "$(cat ~/manifest.rc)"; else echo "rc=RUNNING"; fi
    if [ -f "$f" ]; then
      printf "logbytes=%s\n" "$(stat -c %s "$f" 2>/dev/null || echo 0)"
    else
      printf "logbytes=\n"
    fi
    if [ -f "$r" ]; then
      printf "lines=%s\n" "$(wc -l < "$r" | tr -d " ")"
      printf "mtime=%s\n" "$(stat -c %Y "$r" 2>/dev/null || echo 0)"
      printf "stage=%s\n" "$(grep -oE "STAGE_START name=[A-Za-z0-9_]+" "$r" 2>/dev/null | tail -1 | cut -d= -f2)"
    else
      printf "lines=0\nmtime=0\nstage=\n"
    fi
    if pgrep -f "[b]ash scripts/mlsys_manifest.sh" >/dev/null; then echo "alive=1"; else echo "alive=0"; fi
  ' 2>/dev/null)
  rc=$?
  marker=$(printf '%s\n' "$probe" | sed -n 's/^rc=//p')
  if [ "$rc" -ne 0 ]; then
    SSH_UNKNOWN=$((SSH_UNKNOWN + 1))
    say "ssh check failed (rc=$rc) — state UNKNOWN, retry $SSH_UNKNOWN"
    if [ "$SSH_UNKNOWN" -ge "${MLSYS_SSH_UNKNOWN_LIMIT:-30}" ]; then
      say "!!! $SSH_UNKNOWN consecutive ssh failures: cannot tell whether the"
      say "!!! manifest is running. Continuing to wait rather than guessing."
    fi
  elif printf '%s' "$marker" | grep -qE '^[0-9]+$'; then
    MANIFEST_RC=$marker
    if [ "$MANIFEST_RC" = "0" ]; then
      say "manifest finished rc=0"
      break
    fi
    # FAIL FAST. An aborted manifest is an incident: capture the logs, stop the
    # meter, and do not run the normal verification path over a run that has
    # already declared itself broken.
    say "manifest finished rc=$MANIFEST_RC — ABORTED"
    collect_incident "manifest aborted rc=$MANIFEST_RC"
    MANIFEST_ABORTED=1
    break
  elif [ "$marker" != "RUNNING" ]; then
    # "ssh ok, marker unreadable" is its own state: treating it as finished
    # would pull and terminate mid-run; treating it as running hides a broken
    # pod.
    SSH_UNKNOWN=$((SSH_UNKNOWN + 1))
    say "completion marker unreadable ('$(printf '%s' "$marker" | head -c 60)') — state UNKNOWN, retry $SSH_UNKNOWN"
  else
    SSH_UNKNOWN=0
    alive=$(printf '%s\n' "$probe" | sed -n 's/^alive=//p')
    sig=$(printf '%s\n' "$probe" | sed -n 's/^lines=//p')/$(printf '%s\n' "$probe" | sed -n 's/^mtime=//p')
    stage=$(printf '%s\n' "$probe" | sed -n 's/^stage=//p')
    now=$(date -u +%s)
    if [ "$sig" != "$PROGRESS_SIG" ]; then
      PROGRESS_SIG=$sig
      PROGRESS_SINCE=$now
    fi
    # (a) the process is gone and no marker was written: it crashed or was
    #     killed without recording an exit code.
    if [ "$alive" = "0" ]; then
      say "manifest process is gone with no completion marker — crashed"
      collect_incident "manifest process gone, no marker"
      MANIFEST_ABORTED=1
      break
    fi
    # (b) no new progress line. The threshold is per stage, because some stages
    #     run a single long job and legitimately print nothing between runs.
    limit=$(stall_minutes_for "$stage")
    idle=$(( now - PROGRESS_SINCE ))
    if [ "$idle" -gt $(( limit * 60 )) ]; then
      say "STALL: no progress for ${idle}s in stage '${stage}' (limit ${limit}min)"
      collect_incident "stall in ${stage:-unknown} (${idle}s idle, limit ${limit}min)"
      MANIFEST_ABORTED=1
      break
    fi
    # (c) byte-level liveness on the pod's own manifest.log. The per-stage bound
    #     above only moves when a STAGE_* line lands, so a hang INSIDE a stage is
    #     invisible to it: on 2026-10-09T00:00Z the probe deadlocked and 62 min
    #     of 8xA100-80GB (~$43) were billed against a run that could never
    #     finish, because engine_cap_smoke's own limit is 240 min. manifest.log
    #     grows continuously while a run does anything -- tqdm every second, a
    #     TRACE line per phase -- so no growth means nothing is happening.
    #     Checked only while a stage is running: before the first STAGE_START the
    #     pod is still provisioning, which is not a hang in a stage.
    logbytes=$(printf '%s\n' "$probe" | sed -n 's/^logbytes=//p')
    if [ -z "$LIVE_SIG" ]; then
      LIVE_SIG="$logbytes"; LIVE_SINCE=$now
    elif [ "$logbytes" != "$LIVE_SIG" ]; then
      LIVE_SIG="$logbytes"; LIVE_SINCE=$now
    elif [ -n "$stage" ] && [ $(( now - LIVE_SINCE )) -gt $(( LIVE_MIN * 60 )) ]; then
      say "STALL: the pod's manifest.log has not grown for $(( now - LIVE_SINCE ))s"
      say "       in stage '${stage}' (liveness limit ${LIVE_MIN}min), holding at ${logbytes} bytes"
      collect_incident "liveness stall in ${stage} ($(( now - LIVE_SINCE ))s without a byte written, limit ${LIVE_MIN}min)"
      MANIFEST_ABORTED=1
      break
    fi
  fi
  if [ "${DEADLINE:-0}" -gt 0 ] && [ "$(date -u +%s)" -gt "$DEADLINE" ]; then
    say "campaign deadline hit mid-manifest"; break
  fi
  interruptible_sleep "$MANIFEST_POLL"
done

# An incident is a stop, not a pause: terminate through the verified path and
# leave with a code that says the campaign did not complete.
if [ "${MANIFEST_ABORTED:-0}" = "1" ]; then
  say "terminating the instance and stopping: the campaign did not complete"
  terminate_and_confirm
  exit 7
fi

# --------------------------------------------------------------------------
# per-run staged pull, sha256-verified, NEVER --delete
# --------------------------------------------------------------------------
STAGE=$SESSION_DIR/pull_$(date -u +%Y%m%dT%H%M%SZ)
mkdir -p "$STAGE"
PULL_FAILED=0
MERGE_OK=0
say "pulling results to $STAGE (no --delete anywhere)"
if ! rsync -az --no-perms --no-owner --no-group -e "ssh $SSH_OPTS" \
       --exclude 'checkpoints/' --exclude 'attempts/' \
       "$SSH_USER@$IP:~/RASD/results/mlsys/" "$STAGE/" >>"$LOG" 2>&1; then
  # A partial pull merged into results/ is indistinguishable from a complete
  # one. Stop before merging anything.
  say "!!! rsync FAILED; not merging a partial pull. Staged files kept at $STAGE"
  PULL_FAILED=1
fi

if [ "$PULL_FAILED" = "0" ]; then
  say "  rsync ok"

  # --- the sha256 comparison the comment always claimed --------------------
  # The remote computes a manifest and the local side recomputes it. rsync
  # catches a size change, but not a transfer that was silently truncated and
  # padded, and not a file the remote never wrote at all.
  say "verifying per-file sha256 against the remote"
  ssh $SSH_OPTS "$SSH_USER@$IP" \
    'cd ~/RASD/results/mlsys 2>/dev/null && find . -type f -not -path "*/checkpoints/*" -print0 | sort -z | xargs -0 shasum -a 256' \
    > "$SESSION_DIR/pull_remote.sha256" 2>>"$LOG"
  if [ ! -s "$SESSION_DIR/pull_remote.sha256" ]; then
    say "!!! could not obtain the remote sha256 manifest; verification NOT done"
    PULL_FAILED=1
  else
    ( cd "$STAGE" && find . -type f -print0 | sort -z | xargs -0 shasum -a 256 ) \
      > "$SESSION_DIR/pull_local.sha256" 2>>"$LOG"
    if diff -u "$SESSION_DIR/pull_remote.sha256" \
                "$SESSION_DIR/pull_local.sha256" >>"$LOG" 2>&1; then
      say "  sha256 OK: every file matches the remote ($(wc -l < "$SESSION_DIR/pull_local.sha256" | tr -d ' ') files)"
    else
      say "!!! sha256 MISMATCH between remote and staged pull; see $LOG"
      diff "$SESSION_DIR/pull_remote.sha256" "$SESSION_DIR/pull_local.sha256" \
        | head -20 >>"$LOG" 2>&1
      PULL_FAILED=1
    fi
  fi
fi

# The merge is GATED on a fully verified pull. It previously ran
# unconditionally -- the python block was not inside any `if`, so PULL_FAILED
# only changed the wording of a message while an unverified or truncated pull
# was copied into results/ regardless. That is the worst shape for this bug: a
# merged short CSV is indistinguishable from a short run, and the failure would
# have been discovered in the analysis, not here.
# The merge rule lives in scripts/mlsys_pull_merge.sh so it can be exercised
# without a GPU and an SSH session: the rehearsal feeds it a verified pull, a
# pull with one injected hash mismatch, and an empty pull, and asserts that only
# the first one lands. It was inline here before, with the copy OUTSIDE the
# guard, so an unverified pull was merged while the message said otherwise.
if [ "$PULL_FAILED" = "0" ]; then
  if bash scripts/mlsys_pull_merge.sh \
       --stage-dir "$STAGE" \
       --remote-sha "$SESSION_DIR/pull_remote.sha256" \
       --local-sha "$SESSION_DIR/pull_local.sha256" \
       --dest results/mlsys --log "$LOG"; then
    MERGE_OK=1
  else
    say "!!! NOT MERGED: the pull did not verify (see above). results/ is"
    say "!!! untouched and the staged copy is preserved at $STAGE."
    MERGE_OK=0
  fi
else
  say "!!! NOT MERGED: rsync or the remote manifest failed. results/ is"
  say "!!! untouched and the staged copy is preserved at $STAGE."
  MERGE_OK=0
fi

# --------------------------------------------------------------------------
# 60-minute grace, then terminate and CONFIRM
# --------------------------------------------------------------------------
say "waiting ${MLSYS_GRACE_MINUTES:-60} min for operator input before terminating"
interruptible_sleep $(( ${MLSYS_GRACE_MINUTES:-60} * 60 ))

uptime_s=$(( $(date -u +%s) - DEADLINE ))
say "manifest finished (reason=manifest-finished)"
# The staged pull above already brought the results home and merged them, so the
# EXIT trap must not pull a second time (it would re-merge the cost ledger).
RESULTS_HANDLED=1
terminate_and_confirm

say "results in $STAGE; session dir $SESSION_DIR"

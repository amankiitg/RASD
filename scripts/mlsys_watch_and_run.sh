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
# Interpreter to use ON THE POD. Resolved there, because the local PATH says
# nothing about the remote machine.
RPY='$( if [ -x "$HOME/miniconda3/envs/rasd/bin/python" ]; then echo "$HOME/miniconda3/envs/rasd/bin/python"; else command -v python3; fi )'

# Local interpreter for this script's own helpers. Never bare `python3`.
PY=$(command -v python3)
for c in "$HOME/miniconda3/envs/rasd/bin/python" /opt/conda/bin/python; do
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

# EXIT covers normal completion and explicit `exit`; INT/TERM cover Ctrl-C and a
# dropped session. TERMINATED makes the double-fire a no-op.
trap terminate_and_confirm EXIT
trap 'terminate_and_confirm; exit 130' INT
trap 'terminate_and_confirm; exit 143' TERM

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
if ! rsync -az --no-perms --no-owner --no-group -e "ssh $SSH_OPTS" \
       --exclude '.git' --exclude 'results/final' --exclude 'manuscript' \
       --exclude '.venv*' --exclude '__pycache__' \
       "$REPO/" "$SSH_USER@$IP:~/RASD/" >>"$LOG" 2>&1; then
  say "FATAL: staging the repository to the pod failed; refusing to run"
  say "FATAL: against code we cannot confirm is the code under test"
  terminate_and_confirm
  exit 6
fi
say "  repo staged"

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
  ssh $SSH_OPTS "$SSH_USER@$IP" "cd ~/RASD && ${RPY:-python3} -c \"
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
ssh $SSH_OPTS "$SSH_USER@$IP" \
  "cd ~/RASD && NODE_RATE_PER_HOUR=$RATE \
   MLSYS_ASK_OVER_USD=${MLSYS_ASK_OVER_USD:-300} \
   MLSYS_MAX_HOURS=${MLSYS_CAMPAIGN_HOURS:-40} \
   MLSYS_MAX_COST_USD=${MLSYS_MAX_COST_USD:-850} \
   MLSYS_ONLY_STAGES='${ONLY_STR}' \
   MLSYS_APPROVED_STAGES='${ONLY_STR}' \
   nohup bash scripts/mlsys_manifest.sh > ~/manifest.log 2>&1 & echo started" >>"$LOG" 2>&1

# A failed ssh says NOTHING about whether the manifest is still running. Treating
# it as "finished" ends the run and, worse, moves on to terminate an instance
# whose work is still in progress. Only exit code 1 from `pgrep` (no match, on a
# successful connection) means finished; 255 means the connection failed and is
# retried.
SSH_UNKNOWN=0
while true; do
  ssh $SSH_OPTS "$SSH_USER@$IP" 'pgrep -f mlsys_manifest.sh >/dev/null' 2>/dev/null
  rc=$?
  if [ "$rc" -eq 1 ]; then
    say "manifest finished (ssh ok, no manifest process)"
    break
  elif [ "$rc" -eq 0 ]; then
    SSH_UNKNOWN=0
  else
    SSH_UNKNOWN=$((SSH_UNKNOWN + 1))
    say "ssh check failed (rc=$rc) — state UNKNOWN, retry $SSH_UNKNOWN"
    if [ "$SSH_UNKNOWN" -ge "${MLSYS_SSH_UNKNOWN_LIMIT:-30}" ]; then
      say "!!! $SSH_UNKNOWN consecutive ssh failures: cannot tell whether the"
      say "!!! manifest is running. Continuing to wait rather than guessing."
    fi
  fi
  if [ "${DEADLINE:-0}" -gt 0 ] && [ "$(date -u +%s)" -gt "$DEADLINE" ]; then
    say "campaign deadline hit mid-manifest"; break
  fi
  interruptible_sleep 120
done

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
terminate_and_confirm

say "results in $STAGE; session dir $SESSION_DIR"

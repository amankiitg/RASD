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
RATE=22.32
SSH_KEY=$HOME/.ssh/id_ed25519
SSH_USER=ubuntu
SSH_OPTS="-o StrictHostKeyChecking=no -o ConnectTimeout=25 -i $SSH_KEY"

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
  n=$(api_get instances 2>/dev/null | python3 -c \
      "import json,sys; print(len(json.load(sys.stdin).get('data',[])))" 2>/dev/null)
  echo "${n:-unknown}"
}

# --------------------------------------------------------------------------
# deadline
# --------------------------------------------------------------------------
if [ -n "${MLSYS_DEADLINE_EPOCH:-}" ]; then
  DEADLINE=$MLSYS_DEADLINE_EPOCH
else
  DEADLINE=$(( $(date -u +%s) + ${MLSYS_HOURS:-18} * 3600 ))
fi
say "target=$INSTANCE_TYPE  per-stage approval threshold=\$$ASK_OVER @ \$$RATE/hr"
say "approved stages: ${MLSYS_APPROVED_STAGES:-<none>}"
say "absolute deadline epoch=$DEADLINE ($(date -u -r "$DEADLINE" +%FT%TZ 2>/dev/null || date -u +%FT%TZ -d "@$DEADLINE"))"

# --------------------------------------------------------------------------
# phase A: wait for capacity, then launch
# --------------------------------------------------------------------------
rm -f "$FOUND"
INSTANCE_ID=""
TERMINATED=0

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
  local confirmed=0 resp rc n
  for i in $(seq 1 40); do
    resp=$(api_get instances 2>&1); rc=$?
    if [ $rc -ne 0 ]; then
      say "  attempt $i: api rc=$rc :: $(printf '%s' "$resp" | head -c 120)"
      interruptible_sleep 15; continue
    fi
    n=$(printf '%s' "$resp" | python3 -c \
      "import json,sys; print(len(json.load(sys.stdin).get('data',[])))" 2>&1)
    say "  attempt $i: instances=$n"
    if [ "$n" = "0" ]; then confirmed=1; break; fi
    interruptible_sleep 15
  done
  TERMINATED=1
  [ "$confirmed" = "1" ] && say "CONFIRMED TERMINATED (0 instances)" \
                         || say "!!! COULD NOT CONFIRM TERMINATION — CHECK THE DASHBOARD"
}

# EXIT covers normal completion and explicit `exit`; INT/TERM cover Ctrl-C and a
# dropped session. TERMINATED makes the double-fire a no-op.
trap terminate_and_confirm EXIT
trap 'terminate_and_confirm; exit 130' INT
trap 'terminate_and_confirm; exit 143' TERM

attempt=0
while [ "$(date -u +%s)" -lt "$DEADLINE" ]; do
  attempt=$((attempt+1))
  n=$(count_instances)
  if [ "$n" = "0" ]; then
    avail=$(api_get instance-types 2>/dev/null | python3 -c "
import json,sys
d=json.load(sys.stdin).get('data',{})
t=d.get('$INSTANCE_TYPE',{})
print(','.join(r['name'] for r in t.get('regions_with_capacity_available',[])))" 2>/dev/null)
    if [ -n "$avail" ]; then
      say "attempt $attempt: capacity reported in [$avail] (advisory; the launch may still be refused)"
      region=${avail%%,*}
      resp=$(curl -sS --max-time 120 -u "$KEY:" -X POST \
        "https://cloud.lambda.ai/api/v1/instance-operations/launch" \
        -H 'Content-Type: application/json' \
        -d "{\"region_name\":\"$region\",\"instance_type_name\":\"$INSTANCE_TYPE\",\"ssh_key_names\":[\"rasd-amank\"],\"name\":\"rasd-mlsys\",\"quantity\":1}" 2>&1)
      INSTANCE_ID=$(printf '%s' "$resp" | python3 -c "
import json,sys
try: print(json.loads(sys.stdin.read())['data']['instance_ids'][0])
except Exception: print('')" 2>/dev/null)
      if [ -n "$INSTANCE_ID" ]; then
        say "LAUNCHED $INSTANCE_ID in $region"
        echo "$region" > "$FOUND"
        break
      fi
      say "  launch refused: $(printf '%s' "$resp" | head -c 160) — retrying"
    else
      [ $((attempt % 10)) -eq 0 ] && say "still waiting (attempt $attempt, 0 instances)"
    fi
  else
    say "instances already running ($n) — not launching a second one"
    INSTANCE_ID=$(api_get instances | python3 -c \
      "import json,sys;d=json.load(sys.stdin)['data'];print(d[0]['id'] if d else '')")
    break
  fi
  interruptible_sleep $(( 90 + RANDOM % 510 ))
done

if [ -z "$INSTANCE_ID" ]; then
  say "DEADLINE PASSED with no capacity and nothing launched."
  exit 3
fi

# --------------------------------------------------------------------------
# wait for the instance, then boot it
# --------------------------------------------------------------------------
IP=""
for i in $(seq 1 60); do
  read -r status ip <<<"$(api_get instances | python3 -c "
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
rsync -az --no-perms --no-owner --no-group -e "ssh $SSH_OPTS" \
  --exclude '.git' --exclude 'results/final' --exclude 'manuscript' \
  --exclude '.venv*' --exclude '__pycache__' \
  "$REPO/" "$SSH_USER@$IP:~/RASD/" >>"$LOG" 2>&1 && say "  repo staged"

# The metadata names relative paths; the memmap files must travel with it or the
# stage dies exactly like pg19_short_target did. Verify AFTER the copy, from the
# run directory, for both metadata shapes (per-book `documents` and the older
# concatenated `chunks`). pg19_docs is the primary pool; pg19_docs_diverse backs
# the diverse-pool arm; the two older dirs still back legacy cells.
for d in data/processed/pg19_docs data/processed/pg19_docs_diverse \
         data/processed/pg19_llama3 data/processed/pg19; do
  [ -d "$d" ] || continue
  rsync -az --no-perms --no-owner --no-group -e "ssh $SSH_OPTS" \
    "$d/" "$SSH_USER@$IP:~/RASD/$d/" >>"$LOG" 2>&1
  ssh $SSH_OPTS "$SSH_USER@$IP" "cd ~/RASD && python3 -c \"
import json,pathlib,sys
m=json.load(open('$d/documents.json')) if pathlib.Path('$d/documents.json').exists() else json.load(open('$d/pg19_validation_metadata.json'))
items=m.get('documents') or m.get('chunks') or []
missing=[c['file'] for c in items if not pathlib.Path(c['file']).exists()]
print('  $d: %d files, %d unresolved' % (len(items), len(missing)))
sys.exit(1 if missing else 0)\"" >>"$LOG" 2>&1 \
    && say "  $d verified" || say "  WARNING: $d has unresolved paths"
done

# --------------------------------------------------------------------------
# run the manifest
# --------------------------------------------------------------------------
say "starting the manifest"
ssh $SSH_OPTS "$SSH_USER@$IP" \
  "cd ~/RASD && NODE_RATE_PER_HOUR=$RATE \
   MLSYS_ASK_OVER_USD=${MLSYS_ASK_OVER_USD:-300} \
   MLSYS_MAX_HOURS=${MLSYS_MAX_HOURS:-20} \
   MLSYS_APPROVED_STAGES='${MLSYS_APPROVED_STAGES:-}' \
   nohup bash scripts/mlsys_manifest.sh > ~/manifest.log 2>&1 & echo started" >>"$LOG" 2>&1

while true; do
  if ! ssh $SSH_OPTS "$SSH_USER@$IP" 'pgrep -f mlsys_manifest.sh >/dev/null' 2>/dev/null; then
    say "manifest finished"
    break
  fi
  if [ "$(date -u +%s)" -gt "$DEADLINE" ]; then say "deadline hit mid-manifest"; break; fi
  interruptible_sleep 120
done

# --------------------------------------------------------------------------
# per-run staged pull, sha256-verified, NEVER --delete
# --------------------------------------------------------------------------
STAGE=$SESSION_DIR/pull_$(date -u +%Y%m%dT%H%M%SZ)
mkdir -p "$STAGE"
say "pulling results to $STAGE (no --delete anywhere)"
rsync -az --no-perms --no-owner --no-group -e "ssh $SSH_OPTS" \
  --exclude 'checkpoints/' \
  "$SSH_USER@$IP:~/RASD/results/mlsys/" "$STAGE/" >>"$LOG" 2>&1 && say "  rsync ok"

python3 - "$STAGE" <<'PYEOF' >>"$LOG" 2>&1
import hashlib, pathlib, sys
stage = pathlib.Path(sys.argv[1]); dest = pathlib.Path("results/mlsys")
dest.mkdir(parents=True, exist_ok=True)
n = 0
for f in stage.rglob("*"):
    if not f.is_file(): continue
    t = dest / f.relative_to(stage)
    t.parent.mkdir(parents=True, exist_ok=True)
    t.write_bytes(f.read_bytes())
    n += 1
print(f"  merged {n} files (additive; nothing deleted)")
PYEOF
say "  merged into results/mlsys (additive)"

# --------------------------------------------------------------------------
# 60-minute grace, then terminate and CONFIRM
# --------------------------------------------------------------------------
say "waiting ${MLSYS_GRACE_MINUTES:-60} min for operator input before terminating"
interruptible_sleep $(( ${MLSYS_GRACE_MINUTES:-60} * 60 ))

uptime_s=$(( $(date -u +%s) - DEADLINE ))
say "manifest finished (reason=manifest-finished)"
terminate_and_confirm

say "results in $STAGE; session dir $SESSION_DIR"

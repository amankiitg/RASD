#!/usr/bin/env bash
# SUNDAY SESSION A — 8x A100-80GB, the 1M headline. Hard spend cap $200.
#
# Usage:  bash scripts/mlsys_session_a.sh
#
# STRUCTURE, and why each piece is here:
#   * CAPACITY POLL. 8x capacity was empty at 12:50Z. Poll the 80GB type for up to
#     6 hours; after that an 8x H100 is allowed IF its rate keeps the session
#     inside $200 (rule from the operator, hardware change recorded in the report).
#   * DEAD-MAN TIMER, armed the moment the instance exists, bound to that exact
#     instance id, detached via scripts/mlsys_detach.py so it survives this shell,
#     this session, and this laptop's copilot runtime. It is the only thing that
#     terminates the pod if everything else dies.
#   * PROVISIONING reuses scripts/mlsys_setup_probe.sh (the pod's interpreter,
#     pins and .pod_env.sh). Two of my own bugs are already fixed there in the
#     pilot: the env file is ~/RASD/.pod_env.sh, and this node needs nproc 8 (not
#     the 1 the pilot passes).
#   * INCREMENTAL PULLS: a background rsync every 4 minutes, so a crash or a kill
#     loses at most one row rather than a whole group.
#   * VERIFY BY EFFECT: the instance count is re-read from the API after every
#     terminate, and the process list is checked at the end. A command's exit code
#     is not evidence that a pod stopped billing.
set -uo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"
SESSION_DIR="${MLSYS_SESSION_DIR:-$HOME/.copilot/session-state/0f919471-cdca-48c0-a994-56a95eb1d6be/files}"
mkdir -p "$SESSION_DIR"
LOG="$SESSION_DIR/session_a.log"
OUT_STAGE="$SESSION_DIR/session_a_pull"
mkdir -p "$OUT_STAGE"
SSH_USER=ubuntu
SSH_KEY=${MLSYS_SSH_KEY:-$HOME/.ssh/id_ed25519}
SSH_OPTS="-o StrictHostKeyChecking=no -o ConnectTimeout=25 -o ServerAliveInterval=30 -i $SSH_KEY"
PY=python3
RATE_80GB=22.32
RATE_H100=31.92
say() { printf '%s %s\n' "$(date -u +%FT%TZ)" "$*" | tee -a "$LOG"; }

KEY=$(grep -E '^[[:space:]]*LAMBDA_API_KEY[[:space:]]*=' runpod_creds.md 2>/dev/null \
      | tail -1 | sed -E 's/^[^=]*=[[:space:]]*//' | tr -d '`"' | tr -d '[:space:]')
HF_TOKEN_VALUE=$(grep -E '^[[:space:]]*HF_TOKEN[[:space:]]*=' runpod_creds.md 2>/dev/null \
      | tail -1 | sed -E 's/^[^=]*=[[:space:]]*//' | tr -d '`"' | tr -d '[:space:]')
if [ -z "${KEY:-}" ] || [ -z "${HF_TOKEN_VALUE:-}" ]; then
  say "FATAL: could not read the API key / HF token from runpod_creds.md by label"; exit 2
fi
api_get() { curl -sS --max-time 60 -u "$KEY:" "https://cloud.lambda.ai/api/v1/$1"; }
count_instances() {
  api_get instances 2>/dev/null | "$PY" -c \
    "import json,sys; print(len(json.load(sys.stdin).get('data',[])))" 2>/dev/null || echo unknown
}
regions_for() {
  # Explicit field read: unpacking these dicts as key/value pairs yields the KEYS
  # ('name', 'description'), which went to the API as a region and came back
  # "Unknown region" on the first pilot attempt.
  printf '%s' "$1" | "$PY" -c "
import json,sys
t=sys.argv[1]
info=(json.load(sys.stdin).get('data') or {}).get(t) or {}
regs=info.get('regions_with_capacity_available') or []
names=[x.get('name') if isinstance(x,dict) else str(x) for x in regs]
print(names[0] if names else 'NONE')
" "$2" 2>/dev/null
}

BEFORE=$(count_instances)
say "instances before launch: $BEFORE"
if [ "$BEFORE" != "0" ]; then
  say "REFUSING: $BEFORE instance(s) already exist; this session owns exactly one"
  exit 3
fi

# ---- capacity ---------------------------------------------------------------
POLL_DEADLINE=$(( $(date -u +%s) + 6*3600 ))
CHOSEN=""; REGION=""; RATE=""
while :; do
  AVAIL=$(api_get instance-types 2>/dev/null)
  for t in gpu_8x_a100_80gb_sxm4 gpu_8x_a100; do
    R=$(regions_for "$AVAIL" "$t")
    if [ -n "$R" ] && [ "$R" != "NONE" ]; then
      CHOSEN=$t; REGION=$R; RATE=$RATE_80GB; break
    fi
  done
  if [ -z "$CHOSEN" ] && [ "$(date -u +%s)" -gt "$POLL_DEADLINE" ]; then
    R=$(regions_for "$AVAIL" gpu_8x_h100_sxm5)
    if [ -n "$R" ] && [ "$R" != "NONE" ]; then
      CHOSEN=gpu_8x_h100_sxm5; REGION=$R; RATE=$RATE_H100
      say "HARDWARE CHANGE: 6h passed with no 8x A100 capacity; using 8x H100 at \$$RATE/hr (profile 4h = \$$(awk -v r=$RATE 'BEGIN{printf "%.0f", r*4}'), inside the \$200 cap)"
    fi
  fi
  [ -n "$CHOSEN" ] && break
  say "no 8x capacity yet (polling ${CHOSEN:-gpu_8x_a100_80gb_sxm4})"
  sleep 150
done
say "capacity: $CHOSEN in $REGION at \$$RATE/hr"

LAUNCH=$(curl -sS --max-time 90 -u "$KEY:" \
  -X POST "https://cloud.lambda.ai/api/v1/instance-operations/launch" \
  -H 'Content-Type: application/json' \
  -d "{\"region_name\":\"$REGION\",\"instance_type_name\":\"$CHOSEN\",\"ssh_key_names\":[\"rasd-amank\"],\"name\":\"rasd-session-a\",\"quantity\":1}")
ID=$(printf '%s' "$LAUNCH" | "$PY" -c "
import json,sys
d=json.load(sys.stdin)
ids=d.get('data',{}).get('instance_ids') or []
print(ids[0] if ids else '')
" 2>/dev/null)
if [ -z "${ID:-}" ]; then
  say "FATAL: launch returned no instance id: $(printf '%s' "$LAUNCH" | head -c 300)"; exit 5
fi
LAUNCHED_AT=$(date -u +%s)
say "launched $CHOSEN id=$ID region=$REGION"

# ---- dead-man timer, before any work ---------------------------------------
# The timer's length is derived from the RATE so the cap cannot be exceeded by
# the clock: the operator's rule is <= $200 for this session, and a fixed 9 hours
# is $200.9 on the 80GB node and $287 on the H100. Budgeted: 5.3h of work, plus
# pull and terminate, so 8.5h on the 80GB ($190) and 6.0h on the H100 ($192).
if [ "$RATE" = "$RATE_H100" ]; then TIMER_MIN=360; else TIMER_MIN=510; fi
say "dead-man timer: ${TIMER_MIN} min (\$$(awk -v m=$TIMER_MIN -v r=$RATE 'BEGIN{printf "%.0f", m/60*r}') at \$$RATE/hr, inside the \$200 cap)"
"$PY" scripts/mlsys_detach.py --pidfile "$SESSION_DIR/deadman_${ID}.pid" \
  --log "$SESSION_DIR/deadman_${ID}.log" \
  -- bash scripts/mlsys_deadman.sh "$ID" "$TIMER_MIN" "$SESSION_DIR/deadman_${ID}.log" \
  >>"$LOG" 2>&1 || say "WARN: could not arm the dead-man timer"
sleep 5
DPID=$(cat "$SESSION_DIR/deadman_${ID}.pid" 2>/dev/null || echo "")
say "dead-man timer pid=${DPID:-NONE} (${TIMER_MIN} min)"

POLLER_PID=""
terminated=0
finish() {
  local rc=$?
  [ "$terminated" = "1" ] && return $rc
  terminated=1
  [ -n "${POLLER_PID:-}" ] && kill "$POLLER_PID" 2>/dev/null
  say "--- final pull ---"
  timeout 420 rsync -az --no-perms --no-owner -e "ssh $SSH_OPTS" \
    "$SSH_USER@$IP:~/RASD/results/mlsys/session_a/" "$OUT_STAGE/" >>"$LOG" 2>&1
  say "pulled $(find "$OUT_STAGE" -type f | wc -l | tr -d ' ') files"
  say "--- terminating $ID ---"
  curl -sS --max-time 60 -u "$KEY:" -X POST \
    "https://cloud.lambda.ai/api/v1/instance-operations/terminate" \
    -H 'Content-Type: application/json' -d "{\"instance_ids\":[\"$ID\"]}" >/dev/null 2>&1
  for _ in $(seq 1 40); do
    [ "$(count_instances)" = "0" ] && break
    sleep 15
  done
  local n; n=$(count_instances)
  say "instances after terminate: $n (0 required)"
  local wall=$(( $(date -u +%s) - LAUNCHED_AT ))
  say "session wall: ${wall}s = $(awk -v w=$wall 'BEGIN{printf "%.2f", w/3600}')h at \$$RATE/hr = \$$(awk -v w=$wall -v r=$RATE 'BEGIN{printf "%.2f", w/3600*r}')"
  say "SESSION A DONE (0 instances required)"
  # the timer is bound to this id and exits by itself once the id is gone
  return $rc
}
trap finish EXIT

IP=""; STATUS=""
DEADLINE=$(( $(date -u +%s) + 900 ))
while [ "$(date -u +%s)" -lt "$DEADLINE" ]; do
  read -r STATUS IP <<<"$(api_get "instances/$ID" 2>/dev/null | "$PY" -c "
import json,sys
i=(json.load(sys.stdin).get('data') or {})
print(i.get('status',''), i.get('ip') or '')
" 2>/dev/null)"
  say "status=${STATUS:-?} ip=${IP:-none}"
  [ "$STATUS" = "active" ] && [ -n "${IP:-}" ] && break
  [ "$STATUS" = "terminated" ] && { say "terminated under us"; exit 6; }
  sleep 20
done
[ -z "${IP:-}" ] && { say "FATAL: never became active"; exit 7; }
say "active at $IP"

say "--- provisioning ---"
MLSYS_HF_TOKEN="$HF_TOKEN_VALUE" timeout 1800 bash scripts/mlsys_setup_probe.sh "$IP" >>"$LOG" 2>&1
say "provisioning rc=$?"

say "--- staging ---"
timeout 600 rsync -az --no-perms --no-owner --no-group -e "ssh $SSH_OPTS" \
  scripts/mlsys_session_a_remote.sh "$SSH_USER@$IP:~/RASD/scripts/mlsys_session_a_remote.sh" >>"$LOG" 2>&1
timeout 600 rsync -az --no-perms --no-owner --no-group -e "ssh $SSH_OPTS" \
  configs/mlsys_sunday_1m.yml "$SSH_USER@$IP:~/RASD/configs/" >>"$LOG" 2>&1
timeout 900 rsync -az --no-perms --no-owner --no-group -e "ssh $SSH_OPTS" \
  data/processed/pg19_1m data/processed/pg19_1m_128kwin "$SSH_USER@$IP:~/RASD/data/processed/" >>"$LOG" 2>&1
say "staged script + config + both 1M pools"

# ---- incremental pulls, so a crash loses at most one row -------------------
(
  while :; do
    sleep 240
    rsync -az --no-perms --no-owner -e "ssh $SSH_OPTS" \
      "$SSH_USER@$IP:~/RASD/results/mlsys/session_a/" "$OUT_STAGE/" >/dev/null 2>&1
  done
) &
POLLER_PID=$!
say "incremental puller pid=$POLLER_PID (every 240s)"

say "--- running session A (remote driver) ---"
timeout 30000 ssh $SSH_OPTS "$SSH_USER@$IP" \
  "HF_TOKEN='$HF_TOKEN_VALUE' bash ~/RASD/scripts/mlsys_session_a_remote.sh" 2>&1 | tee -a "$LOG"
say "remote rc=$?"

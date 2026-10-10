#!/usr/bin/env bash
# Launch ONE 1x A100, provision it, run scripts/mlsys_pilot_remote.sh, pull, and
# TERMINATE. Bounded at 55 minutes wall clock by construction.
#
# Usage:  bash scripts/mlsys_pilot_1x.sh
#
# WHY A SEPARATE SCRIPT. The campaign's watcher/manifest machinery is built for a
# multi-hour, multi-stage 8x run: a capacity poller, a detached watcher, a
# manifest with per-stage timeouts, a monitor. None of that is needed to run two
# configs on one card for half an hour, and standing it up would cost more
# attention than the pilot. What IS reused, deliberately, is the part that was
# expensive to get right: `mlsys_setup_probe.sh` (the pod's interpreter, pins and
# `.pod_env.sh`).
#
# SAFETY. An unconditional trap terminates on ANY exit path -- success, error,
# SIGINT, timeout -- and then re-reads the instance list from the API and says so.
# A pilot that leaves a card billing is worse than one that fails.
set -uo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"
SESSION_DIR="${MLSYS_SESSION_DIR:-$HOME/.copilot/session-state/0f919471-cdca-48c0-a994-56a95eb1d6be/files}"
mkdir -p "$SESSION_DIR"
LOG="$SESSION_DIR/pilot_1x.log"
SSH_USER=ubuntu
SSH_KEY=${MLSYS_SSH_KEY:-$HOME/.ssh/id_ed25519}
SSH_OPTS="-o StrictHostKeyChecking=no -o ConnectTimeout=25 -o ServerAliveInterval=30 -i $SSH_KEY"
PY=python3
say() { printf '%s %s\n' "$(date -u +%FT%TZ)" "$*" | tee -a "$LOG"; }

# The key by LABEL, never echoed.
KEY=$(grep -E '^[[:space:]]*LAMBDA_API_KEY[[:space:]]*=' runpod_creds.md 2>/dev/null \
      | tail -1 | sed -E 's/^[^=]*=[[:space:]]*//' | tr -d '`"' | tr -d '[:space:]')
HF_TOKEN_VALUE=$(grep -E '^[[:space:]]*HF_TOKEN[[:space:]]*=' runpod_creds.md 2>/dev/null \
      | tail -1 | sed -E 's/^[^=]*=[[:space:]]*//' | tr -d '`"' | tr -d '[:space:]')
if [ -z "${KEY:-}" ] || [ -z "${HF_TOKEN_VALUE:-}" ]; then
  say "FATAL: could not read LAMBDA_API_KEY / HF_TOKEN from runpod_creds.md by label"
  exit 2
fi
api_get() { curl -sS --max-time 60 -u "$KEY:" "https://cloud.lambda.ai/api/v1/$1"; }

count_instances() {
  api_get instances 2>/dev/null | "$PY" -c \
    "import json,sys; print(len(json.load(sys.stdin).get('data',[])))" 2>/dev/null || echo unknown
}

BEFORE=$(count_instances)
say "instances before launch: $BEFORE"
if [ "$BEFORE" != "0" ]; then
  say "REFUSING: $BEFORE instance(s) already exist. This pilot owns exactly one."
  exit 3
fi

# Preference order: the cheapest single 80GB A100 first, then the 40GB. The
# 1M-model job downloads a 16 GB bf16 target, so the 80GB card removes any doubt
# about the load; the 40GB card is the fallback if that is all that is free.
TYPES="gpu_1x_a100 gpu_1x_a100_sxm4"
AVAIL=$(api_get instance-types 2>/dev/null)
CHOSEN=""
REGION=""
for t in $TYPES; do
  # `regions_with_capacity_available` is a list of {"name","description"} dicts.
  # Unpacking those with `for r, v in ...` yields the KEYS ('name',
  # 'description'), which then went to the API as a region and came back
  # "Unknown region" -- so read the field explicitly.
  picked=$(printf '%s' "$AVAIL" | "$PY" -c "
import json,sys
t=sys.argv[1]
info=(json.load(sys.stdin).get('data') or {}).get(t) or {}
regs=info.get('regions_with_capacity_available') or []
names=[x.get('name') if isinstance(x,dict) else str(x) for x in regs]
print((names[0] if names else 'NONE') + '|' + t)
" "$t" 2>/dev/null)
  REGION=${picked%%|*}
  CHOSEN=${picked##*|}
  if [ -n "$REGION" ] && [ "$REGION" != "NONE" ] && [ "$REGION" != "$picked" ]; then
    say "capacity: $CHOSEN in $REGION"; break
  fi
  REGION=""
done
if [ -z "${REGION:-}" ]; then
  say "NO CAPACITY for $TYPES right now; not waiting (this pilot is not a campaign)."
  exit 4
fi

LAUNCH=$(curl -sS --max-time 90 -u "$KEY:" \
  -X POST "https://cloud.lambda.ai/api/v1/instance-operations/launch" \
  -H 'Content-Type: application/json' \
  -d "{\"region_name\":\"$REGION\",\"instance_type_name\":\"$CHOSEN\",\"ssh_key_names\":[\"rasd-amank\"],\"name\":\"rasd-pilot-1x\",\"quantity\":1}")
ID=$(printf '%s' "$LAUNCH" | "$PY" -c "
import json,sys
d=json.load(sys.stdin)
ids=d.get('data',{}).get('instance_ids') or []
print(ids[0] if ids else '')
" 2>/dev/null)
if [ -z "${ID:-}" ]; then
  say "FATAL: launch returned no instance id: $(printf '%s' "$LAUNCH" | head -c 300)"
  exit 5
fi
say "launched $CHOSEN id=$ID region=$REGION"

IP=""
terminated=0
terminate() {
  local rc=$?
  if [ "$terminated" = "1" ]; then return $rc; fi
  terminated=1
  say "--- terminating ---"
  if [ -n "${ID:-}" ]; then
    curl -sS --max-time 60 -u "$KEY:" -X POST \
      "https://cloud.lambda.ai/api/v1/instance-operations/terminate" \
      -H 'Content-Type: application/json' -d "{\"instance_ids\":[\"$ID\"]}" \
      2>&1 | head -c 200 | tee -a "$LOG"
    echo | tee -a "$LOG"
    for _ in $(seq 1 30); do
      if [ "$(count_instances)" = "0" ]; then break; fi
      sleep 10
    done
  fi
  say "instances after terminate: $(count_instances) (0 required)"
  return $rc
}
trap terminate EXIT

# Wait for the instance to be active (bounded: 12 min of capacity paperwork).
DEADLINE=$(( $(date -u +%s) + 720 ))
while [ "$(date -u +%s)" -lt "$DEADLINE" ]; do
  read -r STATUS IP <<<"$(api_get "instances/$ID" 2>/dev/null | "$PY" -c "
import json,sys
i=(json.load(sys.stdin).get('data') or {})
print(i.get('status',''), i.get('ip') or '')
" 2>/dev/null)"
  say "status=$STATUS ip=${IP:-none}"
  [ "$STATUS" = "active" ] && [ -n "${IP:-}" ] && break
  [ "$STATUS" = "terminated" ] && { say "instance terminated under us"; exit 6; }
  sleep 20
done
if [ -z "${IP:-}" ]; then say "FATAL: never became active"; exit 7; fi

# The pilot's own clock. Raised from 55 to 80 minutes on the operator's
# instruction after the first attempt lost 10 minutes to a setup bug: the 1x
# A100-40GB is $1.99/hr, so a full 80-minute session is ~$2.65 against the $10
# pilot budget, and the pull and the termination must fit inside it.
PILOT_DEADLINE=$(( $(date -u +%s) + ${PILOT_MINUTES:-80}*60 ))
say "active at $IP; pilot deadline $(date -u -r $PILOT_DEADLINE +%FT%TZ)"

say "--- provisioning (the watcher's own pod setup) ---"
MLSYS_HF_TOKEN="$HF_TOKEN_VALUE" timeout 1500 bash scripts/mlsys_setup_probe.sh "$IP" \
  >>"$LOG" 2>&1
prov=$?
say "provisioning rc=$prov"
if [ "$prov" != "0" ]; then
  say "provisioning failed; pulling only the log"
  timeout 120 rsync -az --no-perms -e "ssh $SSH_OPTS" \
    "$SSH_USER@$IP:~/RASD/results/mlsys/pod_setup_log*" "$SESSION_DIR/" 2>/dev/null
  exit 8
fi

# Stage the remote driver itself (it is inside the repo, but the repo was staged
# BEFORE this file existed in a fresh clone -- rsync it explicitly).
timeout 300 rsync -az --no-perms --no-owner --no-group -e "ssh $SSH_OPTS" \
  scripts/mlsys_pilot_remote.sh "$SSH_USER@$IP:~/RASD/scripts/mlsys_pilot_remote.sh" \
  >>"$LOG" 2>&1
timeout 300 rsync -az --no-perms --no-owner --no-group -e "ssh $SSH_OPTS" \
  data/processed/pg19_1m "$SSH_USER@$IP:~/RASD/data/processed/" >>"$LOG" 2>&1
timeout 300 rsync -az --no-perms --no-owner --no-group -e "ssh $SSH_OPTS" \
  configs/mlsys_pilot_neartie_32k.yml configs/mlsys_pilot_1m_32k.yml \
  "$SSH_USER@$IP:~/RASD/configs/" >>"$LOG" 2>&1
say "pilot scripts + 1M data staged"

REMAINING=$(( PILOT_DEADLINE - $(date -u +%s) ))
[ "$REMAINING" -lt 120 ] && { say "no time left for the runs"; exit 9; }
say "--- running the pilot jobs (${REMAINING}s budget) ---"
# HF_TOKEN has to travel: every model in both configs is a gated repo, and the
# pod env does not carry it. Passed on the ssh command line as the existing
# probe runner does. It is never echoed to any log.
timeout "$REMAINING" ssh $SSH_OPTS "$SSH_USER@$IP" \
  "HF_TOKEN='$HF_TOKEN_VALUE' PILOT_JOBS='${PILOT_JOBS:-AB}' bash ~/RASD/scripts/mlsys_pilot_remote.sh" 2>&1 | tee -a "$LOG"
say "remote rc=$?"

say "--- pulling results ---"
STAGE="$SESSION_DIR/pilot_pull_$(date -u +%Y%m%dT%H%M%SZ)"
mkdir -p "$STAGE"
timeout 300 rsync -az --no-perms --no-owner -e "ssh $SSH_OPTS" \
  "$SSH_USER@$IP:~/RASD/results/mlsys/pilot/" "$STAGE/" >>"$LOG" 2>&1
say "pulled: $(find "$STAGE" -type f | wc -l | tr -d ' ') files, $(du -sh "$STAGE" | cut -f1)"
ls -la "$STAGE" | head -20 | tee -a "$LOG"
echo "$STAGE" > "$SESSION_DIR/PILOT_PULL_DIR"
say "PILOT FINISHED"

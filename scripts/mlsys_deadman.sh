#!/usr/bin/env bash
# Terminate ONE instance id at a hard deadline, no matter what else dies.
#
# WHY THIS IS SEPARATE FROM ANY DRIVER. Every driver in this project has a trap
# that terminates, and every one of those traps is inside the process that could
# be killed. The 2026-10-10 session-restart incident is the evidence: the
# copilot runtime restarted, the trap's parent died, and a healthy 8xA100 kept
# billing with nothing left alive that knew its id. This script is launched
# DETACHED (scripts/mlsys_detach.py, setsid) and knows only an id and a clock, so
# it survives whatever happens to the session that started it.
#
# It is deliberately dumb: sleep until the deadline, then terminate THAT id, then
# read the API back and say whether it worked. It never touches any other
# instance, so two of them can never race each other.
#
# Usage: bash scripts/mlsys_deadman.sh <instance-id> <minutes-from-now> <logfile>
set -uo pipefail
ID=${1:?usage: mlsys_deadman.sh <instance-id> <minutes> <logfile>}
MINUTES=${2:?usage: mlsys_deadman.sh <instance-id> <minutes> <logfile>}
LOG=${3:-/tmp/deadman_$ID.log}
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"
say() { printf '%s %s\n' "$(date -u +%FT%TZ)" "$*" >>"$LOG"; }

KEY=$(grep -E '^[[:space:]]*LAMBDA_API_KEY[[:space:]]*=' runpod_creds.md 2>/dev/null \
      | tail -1 | sed -E 's/^[^=]*=[[:space:]]*//' | tr -d '`"' | tr -d '[:space:]')
[ -z "${KEY:-}" ] && { say "FATAL: no API key"; exit 2; }

DEADLINE=$(python3 -c "import time;print(int(time.time()+$MINUTES*60))")
say "armed for instance $ID at $(date -u -r "$DEADLINE" +%FT%TZ) (in ${MINUTES}m)"
while [ "$(date -u +%s)" -lt "$DEADLINE" ]; do
  if ! curl -sS --max-time 30 -u "$KEY:" "https://cloud.lambda.ai/api/v1/instances/$ID" \
       2>/dev/null | grep -q '"id"'; then
    say "instance $ID no longer exists; timer exits without action"
    exit 0
  fi
  sleep 30
done
say "DEADLINE REACHED: terminating $ID"
curl -sS --max-time 60 -u "$KEY:" -X POST \
  "https://cloud.lambda.ai/api/v1/instance-operations/terminate" \
  -H 'Content-Type: application/json' -d "{\"instance_ids\":[\"$ID\"]}" \
  >>"$LOG" 2>&1 || true
for _ in $(seq 1 40); do
  N=$(curl -sS --max-time 30 -u "$KEY:" "https://cloud.lambda.ai/api/v1/instances" 2>/dev/null \
      | python3 -c "import json,sys;print(len(json.load(sys.stdin).get('data',[])))" 2>/dev/null)
  say "instances after terminate: ${N:-unknown}"
  [ "${N:-1}" = "0" ] && break
  sleep 15
done
say "DEADMAN DONE"

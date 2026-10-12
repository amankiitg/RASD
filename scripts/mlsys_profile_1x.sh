#!/usr/bin/env bash
# Phase 3 driver: launch ONE 1x A100, arm a DETACHED dead-man timer, provision, run
# scripts/mlsys_profile_remote.sh (which runs scripts/mlsys_profile_decode.sh), pull,
# terminate, and verify by effect.
#
# WHY A SEPARATE DRIVER. The campaign driver (scripts/mlsys_session_a.sh) is built
# around a stage chain driven by a config; the profile is one script on one card. What
# this REUSES is every part that was expensive to get right, because each one has
# already cost money: the detached dead-man timer, the token over stdin into a
# 600-mode file, the detached remote run with its own DONE marker, and API-verified
# termination. Nothing new is invented here except the wiring.
#
# THE TIMER IS ARMED BEFORE ANY WORK. Lesson L4: launch everything long-lived through
# scripts/mlsys_detach.py and verify by EFFECT -- for the timer that means PPID 1, and
# for the instance it means the API count. A command's exit code proves neither.
#
# BUDGET. The handoff caps profile (a) at $5 and mandates a 60-minute timer. At
# $1.99/hr (gpu_1x_a100_sxm4) a full 60 minutes is $1.99. The internal work deadline
# is the timer less 5 minutes so the timer never races the driver.
#
# Usage:
#   bash scripts/mlsys_profile_1x.sh                    # profile (a), nproc 1
#   PROFILE_TAG=b PROFILE_NPROC=2 SESSION_TYPE=gpu_2x_a100_sxm4 \
#     TIMER_MIN=60 bash scripts/mlsys_profile_1x.sh     # profile (b), if needed
set -uo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"
SESSION_DIR="${MLSYS_SESSION_DIR:-$HOME/.copilot/session-state/577ce8e6-e608-4421-bb55-1637cb0c95a0/files}"
mkdir -p "$SESSION_DIR"

PROFILE_TAG=${PROFILE_TAG:-a}
PROFILE_NPROC=${PROFILE_NPROC:-1}
TIMER_MIN=${TIMER_MIN:-60}
# One instance type, overridable only by the caller of this script (profile (b)).
SESSION_TYPE=${SESSION_TYPE:-gpu_1x_a100_sxm4}
# The plan's preference order: the type that actually has capacity this week first.
# L10: an exact-match poll on a type with no capacity waits until the cutoff for no
# reason, so the fallback is a small explicit list, not a wait.
TYPE_FALLBACKS=${TYPE_FALLBACKS:-gpu_1x_a100_sxm4 gpu_1x_a100}
PROFILE_OUT=${PROFILE_OUT:-results/mlsys/profile}
# WHICH POD-SIDE SCRIPT TO RUN. Defaults to the Phase 3 profile, so the profile's
# validated behaviour is unchanged; Phase 5's vLLM rehearsal points it at its own
# wrapper. The contract a remote script must honour is the one both already do:
# read PROFILE_OUT, write its artifacts there, and write $OUT/DONE with its own
# exit code from an EXIT trap. That contract is what makes the Mac side able to
# treat an ssh return as meaningless (L9) and an artifact as the result (L14).
REMOTE_SCRIPT=${REMOTE_SCRIPT:-scripts/mlsys_profile_remote.sh}
LOG="$SESSION_DIR/profile_${PROFILE_TAG}.log"
PULL="$SESSION_DIR/profile_${PROFILE_TAG}_pull"
rm -rf "$PULL"; mkdir -p "$PULL"

SSH_USER=ubuntu
SSH_KEY=${MLSYS_SSH_KEY:-$HOME/.ssh/id_ed25519}
SSH_OPTS="-o StrictHostKeyChecking=no -o ConnectTimeout=25 -o ServerAliveInterval=30 -i $SSH_KEY"
PY=python3
RATE=${SESSION_RATE:-1.99}
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
  # ('name','description'), which the API rejects as an unknown region.
  printf '%s' "$1" | "$PY" -c "
import json,sys
t=sys.argv[1]
info=(json.load(sys.stdin).get('data') or {}).get(t) or {}
regs=info.get('regions_with_capacity_available') or []
names=[x.get('name') if isinstance(x,dict) else str(x) for x in regs]
print(names[0] if names else 'NONE')
" "$2" 2>/dev/null
}

say "profile=$PROFILE_TAG nproc=$PROFILE_NPROC timer=${TIMER_MIN}min type=$SESSION_TYPE rate=\$$RATE"
BEFORE=$(count_instances)
say "instances before launch: $BEFORE"
if [ "$BEFORE" != "0" ]; then
  say "REFUSING: $BEFORE instance(s) already exist; this profile owns exactly one"
  exit 3
fi

AVAIL=$(api_get instance-types 2>/dev/null)
# CANDIDATES. With the default type (the 1x SXM4 the plan names), try the small
# fallback list; with an explicit SESSION_TYPE (profile (b), say), try only that.
if [ "$SESSION_TYPE" = "gpu_1x_a100_sxm4" ]; then CANDIDATES=$TYPE_FALLBACKS; else CANDIDATES=$SESSION_TYPE; fi
CHOSEN=""; REGION=""
for t in $CANDIDATES; do
  REGION=$(regions_for "$AVAIL" "$t")
  if [ -n "$REGION" ] && [ "$REGION" != "NONE" ]; then CHOSEN=$t; break; fi
  REGION=""
done
if [ -z "${REGION:-}" ]; then
  say "NO CAPACITY for any of '$CANDIDATES' right now; not waiting (a profile is not a campaign)"
  exit 4
fi
say "capacity: $CHOSEN in $REGION"

LAUNCH=$(curl -sS --max-time 90 -u "$KEY:" \
  -X POST "https://cloud.lambda.ai/api/v1/instance-operations/launch" \
  -H 'Content-Type: application/json' \
  -d "{\"region_name\":\"$REGION\",\"instance_type_name\":\"$CHOSEN\",\"ssh_key_names\":[\"rasd-amank\"],\"name\":\"rasd-profile-${PROFILE_TAG}\",\"quantity\":1}")
ID=$(printf '%s' "$LAUNCH" | "$PY" -c "
import json,sys
ids=(json.load(sys.stdin).get('data') or {}).get('instance_ids') or []
print(ids[0] if ids else '')
" 2>/dev/null)
if [ -z "${ID:-}" ]; then
  say "FATAL: launch returned no instance id: $(printf '%s' "$LAUNCH" | head -c 300)"; exit 5
fi
LAUNCHED_AT=$(date -u +%s)
say "launched $CHOSEN id=$ID region=$REGION"

# ---- dead-man timer, BEFORE any work ---------------------------------------
say "dead-man timer: ${TIMER_MIN} min at \$$RATE/hr = \$$(awk -v m=$TIMER_MIN -v r=$RATE 'BEGIN{printf "%.2f", m/60*r}')"
"$PY" scripts/mlsys_detach.py --pidfile "$SESSION_DIR/deadman_${ID}.pid" \
  --log "$SESSION_DIR/deadman_${ID}.log" \
  -- bash scripts/mlsys_deadman.sh "$ID" "$TIMER_MIN" "$SESSION_DIR/deadman_${ID}.log" \
  >>"$LOG" 2>&1 || say "WARN: could not arm the dead-man timer"
sleep 5
DPID=$(cat "$SESSION_DIR/deadman_${ID}.pid" 2>/dev/null || echo "")
if [ -n "${DPID:-}" ]; then
  DPPID=$(ps -o ppid= -p "$DPID" 2>/dev/null | tr -d ' ')
  say "dead-man timer pid=$DPID ppid=${DPPID:-?} for $ID"
  [ "${DPPID:-0}" = "1" ] || say "WARN: the timer is NOT reparented to pid 1; it can die with this shell"
else
  say "WARN: no dead-man timer pid file; the pod would have no independent backstop"
fi

IP=""
terminated=0
finish() {
  local rc=$?
  [ "$terminated" = "1" ] && return $rc
  terminated=1
  if [ -n "${IP:-}" ]; then
    say "--- final pull ---"
    timeout 420 rsync -az --no-perms --no-owner -e "ssh $SSH_OPTS" \
      "$SSH_USER@$IP:~/RASD/$PROFILE_OUT/" "$PULL/" >>"$LOG" 2>&1
    say "pulled $(find "$PULL" -type f | wc -l | tr -d ' ') files"
  fi
  say "--- terminating $ID ---"
  curl -sS --max-time 60 -u "$KEY:" -X POST \
    "https://cloud.lambda.ai/api/v1/instance-operations/terminate" \
    -H 'Content-Type: application/json' -d "{\"instance_ids\":[\"$ID\"]}" >/dev/null 2>&1
  for _ in $(seq 1 40); do
    [ "$(count_instances)" = "0" ] && break
    sleep 15
  done
  say "instances after terminate: $(count_instances) (0 required)"
  local wall=$(( $(date -u +%s) - LAUNCHED_AT ))
  say "profile wall: ${wall}s = $(awk -v w=$wall 'BEGIN{printf "%.2f", w/3600}')h at \$$RATE/hr = \$$(awk -v w=$wall -v r=$RATE 'BEGIN{printf "%.2f", w/3600*r}')"
  TPID=$(cat "$SESSION_DIR/deadman_${ID}.pid" 2>/dev/null || echo "")
  if [ -n "${TPID:-}" ]; then
    kill "$TPID" 2>/dev/null && say "dead-man timer $TPID cancelled" || say "timer $TPID already gone"
  fi
  say "PROFILE SESSION DONE (0 instances required)"
  return $rc
}
trap finish EXIT

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

say "--- waiting for ssh ---"
SSH_READY=0
for _ in $(seq 1 40); do
  if ssh $SSH_OPTS -o BatchMode=yes "$SSH_USER@$IP" true >/dev/null 2>&1; then
    SSH_READY=1; break
  fi
  sleep 15
done
say "ssh ready: $SSH_READY"
[ "$SSH_READY" = "1" ] || { say "FATAL: ssh never became usable; pulling nothing and terminating"; exit 8; }

# ---- provisioning: the probe stages the repo, the 600-mode token and the interpreter
say "--- provisioning (scripts/mlsys_setup_probe.sh) ---"
MLSYS_HF_TOKEN="$HF_TOKEN_VALUE" timeout 1800 bash scripts/mlsys_setup_probe.sh "$IP" >>"$LOG" 2>&1
prov=$?
say "provisioning rc=$prov"
[ "$prov" = "0" ] || { say "FATAL: provisioning failed; terminating rather than running against an unverified interpreter"; exit 8; }

# ---- the instance type and the shape travel WITH the results ----------------
"$PY" - "$ID" "$CHOSEN" "$REGION" "$RATE" "$LAUNCHED_AT" "$PROFILE_TAG" "$PROFILE_NPROC" <<'META' > "$PULL/profile_meta.json"
import datetime, json, sys
iid, itype, region, rate, launched, tag, nproc = sys.argv[1:8]
print(json.dumps({
    "session": "phase3-profile",
    "profile": tag,
    "nproc": int(nproc),
    "instance_id": iid,
    "instance_type": itype,
    "region": region,
    "rate_usd_per_hour": float(rate),
    "launched_utc": datetime.datetime.fromtimestamp(
        int(launched), datetime.timezone.utc).isoformat(),
    "config": "configs/mlsys_profile_128k.yml",
    "what_it_measures": "target-only decode at 128k, 24 measured tokens, torch.profiler "
                        "compute/comm/idle sidecar; target-only means spec_steps 0",
}, indent=2))
META
say "profile_meta.json written (remote=$REMOTE_SCRIPT)"

# ---- the remote run, DETACHED, with its own DONE marker ---------------------
PODOUT="RASD/$PROFILE_OUT"
say "--- launching the profile DETACHED on the pod ---"
START_CMD="mkdir -p ~/RASD/$PROFILE_OUT && cd ~/RASD && rm -f $PROFILE_OUT/DONE && setsid nohup env PROFILE_OUT='$PROFILE_OUT' PROFILE_TAG='$PROFILE_TAG' PROFILE_NPROC='$PROFILE_NPROC' bash $REMOTE_SCRIPT >> $PROFILE_OUT/remote_stdout.log 2>&1 < /dev/null & echo STARTED:\$!"
timeout 300 ssh $SSH_OPTS "$SSH_USER@$IP" "$START_CMD" 2>&1 | tee -a "$LOG"
say "remote launched; polling for the DONE marker, not for this shell"

POLL_DEADLINE=$(( LAUNCHED_AT + TIMER_MIN*60 - 300 ))
say "poll deadline $(date -u -r "$POLL_DEADLINE" +%FT%TZ) (dead-man deadline less 5 min)"
DONE_SEEN=""
while :; do
  if [ "$(date -u +%s)" -ge "$POLL_DEADLINE" ]; then
    say "POLL DEADLINE reached; taking over (the dead-man timer remains the backstop)"
    break
  fi
  POLL_OUT=$(timeout 90 ssh $SSH_OPTS -o ConnectTimeout=20 "$SSH_USER@$IP" \
    "if [ -s $PODOUT/DONE ]; then echo MARKER \$(cat $PODOUT/DONE | tr '\n' ' '); \
     elif pgrep -f $(basename $REMOTE_SCRIPT) >/dev/null; then echo RUNNING; \
     elif [ -s $PODOUT/remote_stdout.log ]; then echo DIED; else echo NOPROC; fi" 2>/dev/null)
  PRC=$?
  case "$POLL_OUT" in
    MARKER*) DONE_SEEN="${POLL_OUT#MARKER }"; say "DONE marker: $DONE_SEEN"; break ;;
    RUNNING) say "profile running ($(date -u +%FT%TZ))" ;;
    DIED)    say "WARN: remote process gone but no DONE marker; waiting one more cycle" ;;
    NOPROC)  NOPROC=$(( ${NOPROC:-0} + 1 ))
             say "no remote process yet ($NOPROC/3)"
             [ "$NOPROC" -ge 3 ] && { say "LAUNCH FAILED: nothing running on the pod after 3 polls"; break; } ;;
    *)       say "POLL SSH FAILED (rc=$PRC). The pod keeps working and we keep waiting." ;;
  esac
  sleep 60
done
say "profile remote ended: DONE='${DONE_SEEN:-none}'"

# ---- judge from the artifact, not the marker's rc (L1, L14) ----------------
say "--- pulling the profiler sidecars and the CSV explicitly ---"
timeout 300 rsync -az --no-perms --no-owner -e "ssh $SSH_OPTS" \
  "$SSH_USER@$IP:~/RASD/$PROFILE_OUT/" "$PULL/" >>"$LOG" 2>&1
say "pulled $(find "$PULL" -type f | wc -l | tr -d ' ') files"
"$PY" - "$PULL" <<'PYSUM'
import csv, glob, json, os, sys
d = sys.argv[1]
for f in sorted(glob.glob(os.path.join(d, "*.csv"))):
    rows = list(csv.DictReader(open(f)))
    ok = [r for r in rows if r.get("status") == "ok"]
    print(f"  {os.path.basename(f)}: rows={len(rows)} ok={len(ok)}")
    for r in ok:
        print(f"    {r['run_id']}: {r['tokens_generated']} tokens / {float(r['time_sec']):.1f} s "
              f"= {int(r['tokens_generated'])/float(r['time_sec']):.4f} tok/s")
side = sorted(glob.glob(os.path.join(d, "profiler", "*.json")))
if side:
    print("  sidecars:", [os.path.basename(p) for p in side])
PYSUM
say "PROFILE PULL COMPLETE"

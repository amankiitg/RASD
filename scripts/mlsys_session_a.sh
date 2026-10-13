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
# SESSION_TAG separates a rehearsal's LOCAL artifacts from the real run's. The pod-side
# path can stay fixed because every instance starts with an empty results dir, but the
# Mac-side pull dir is reused across runs and rsync here does NOT use --delete -- so a
# file the real run never produces (because a stage was killed) keeps the REHEARSAL's
# version sitting there looking like a result. That is exactly what happened on
# 2026-10-11: session_a_SUNDAY_SCALING_128K_TARGET.csv held one rehearsal row with
# context_length 4096, which reads as a 0.18x "128k speedup" if you trust the filename.
SESSION_TAG=${SESSION_TAG:-session_a}
LOG="$SESSION_DIR/$SESSION_TAG.log"
OUT_STAGE="$SESSION_DIR/${SESSION_TAG}_pull"
# FRESH EVERY SESSION. rsync here does NOT use --delete, so a file the real run never
# produces keeps whatever an earlier run (or a rehearsal) left under that name -- which is
# exactly how a rehearsal row with context_length 4096 sat in the real run's directory and
# read as a 0.18x "128k speedup" on 2026-10-11. Clearing the directory makes that
# impossible rather than merely detectable.
rm -rf "$OUT_STAGE"
mkdir -p "$OUT_STAGE"
SSH_USER=ubuntu
SSH_KEY=${MLSYS_SSH_KEY:-$HOME/.ssh/id_ed25519}
SSH_OPTS="-o StrictHostKeyChecking=no -o ConnectTimeout=25 -o ServerAliveInterval=30 -i $SSH_KEY"
PY=python3
RATE_80GB=22.32
# THE COMMIT THE POD RUNS (2026-10-13). The pod's ~/RASD is an rsync of the working tree and
# carries no `.git`, so a pod-side `git rev-parse HEAD` returns nothing and every row recorded
# `git_commit: unknown` -- which voids a rerun whose whole point is to compare two ENGINES
# (stop condition S2). The DRIVER is the authority here, because the driver is what stages the
# tree, so it names the commit for every remote script rather than leaving each one to guess.
# `run_experiment.py` and the df arms driver both prefer this variable; the parameter contract
# asserts it arrived.
CODE_COMMIT=${CODE_COMMIT:-$(git rev-parse HEAD 2>/dev/null || echo unknown)}
export CODE_COMMIT
echo "CODE_COMMIT=$CODE_COMMIT"
# ONE instance type, and it is REQUIRED -- there is deliberately no default.
#
# WHY NO DEFAULT. A default is a silent fallback: on 2026-10-12 the 1x rehearsal
# picker exported SESSION_TYPE before validating its own selection, and when that
# selection came back empty the driver quietly substituted the 8x type and would have
# rented eight A100s (about $22/hr) to run a five-minute rehearsal that is supposed to
# cost under $3. It was stopped by hand before it launched anything. An empty
# SESSION_TYPE is never a value to be defaulted; it is a bug upstream, and the only
# safe response is to stop before any instance is created.
#
# The allowed list is exactly the types this campaign has sanctioned: the 8x A100 80GB
# SXM4 main session, the 1x and 2x A100 types used by the rehearsals and the profiler,
# and the five 1x fallbacks the operator approved for the quant-ablation rehearsal.
ALLOWED_TYPES="gpu_8x_a100_80gb_sxm4 gpu_2x_a100_sxm4 gpu_2x_a100 \
gpu_1x_a100_sxm4 gpu_1x_a100 gpu_1x_h100_pcie gpu_1x_a6000 gpu_1x_a10"
if [ -z "${SESSION_TYPE:-}" ]; then
  echo "FATAL: SESSION_TYPE is empty. It is required and has no default; refusing to" >&2
  echo "FATAL: guess an instance type and risk renting the wrong one. Set it to one of:" >&2
  echo "FATAL:   $ALLOWED_TYPES" >&2
  exit 2
fi
case " $ALLOWED_TYPES " in
  *" $SESSION_TYPE "*) : ;;
  *)
    echo "FATAL: SESSION_TYPE='$SESSION_TYPE' is not in the allowed list; refusing to run." >&2
    echo "FATAL: allowed: $ALLOWED_TYPES" >&2
    exit 2
    ;;
esac
RATE=${SESSION_RATE:-22.32}
TIMER_MIN=${TIMER_MIN:-1200}          # 20 h (rehearsal: 60)
SESSION_CAP=${SESSION_CAP:-500}       # safety net, never used to shorten a step
SESSION_CFG=${SESSION_CFG:-configs/mlsys_sunday_1m.yml}
SESSION_SANITY_CFG=${SESSION_SANITY_CFG:-configs/mlsys_sunday_sanity.yml}
NPROC=${NPROC:-8}
# Which pod-side script this session runs. Defaulted, so every pre-existing launch is
# unchanged; the decode diagnosis sets it to scripts/mlsys_ops_remote.sh.
REMOTE_SCRIPT=${REMOTE_SCRIPT:-scripts/mlsys_session_a_remote.sh}
[ -f "$REMOTE_SCRIPT" ] || { echo "FATAL: REMOTE_SCRIPT='$REMOTE_SCRIPT' does not exist" >&2; exit 2; }
# A CUSTOM POD-SIDE SCRIPT MUST BE TOLD WHAT TO DO, EXPLICITLY. The ops profiler is
# env-driven, and its vars have in-script defaults -- exactly the silent fallback this
# project has been bitten by. When a custom script is selected they are therefore
# REQUIRED here: unset means the launch stops before an instance exists, rather than a
# profile quietly running at the wrong context or on the wrong book.
OPS_ENV=""
# SCOPED TO THE OPS SCRIPT (2026-10-13). This block required its parameter set from ANY
# custom REMOTE_SCRIPT, which made it impossible to add a second one: a script that does not
# use OPS_* would be rejected for not setting them. The generic fail-fast forwarding below
# (FORWARD_ENV) is the mechanism for custom scripts now; this block keeps the ops script's
# behaviour byte-identical, including the OPS_RANKS default.
if [ "$REMOTE_SCRIPT" = "scripts/mlsys_ops_remote.sh" ]; then
  for v in PROFILE_OUT OPS_CTX OPS_TOKENS OPS_BOOK OPS_QUANT OPS_TOP OPS_RANKS; do
    eval "[ -n "\${$v:-}" ]" || {
      echo "FATAL: $v must be set when REMOTE_SCRIPT=$REMOTE_SCRIPT" >&2; exit 2; }
  done
  OPS_ENV="PROFILE_OUT='$PROFILE_OUT' OPS_CTX='$OPS_CTX' OPS_TOKENS='$OPS_TOKENS' OPS_BOOK='$OPS_BOOK' OPS_QUANT='$OPS_QUANT' OPS_TOP='$OPS_TOP' OPS_RANKS='${OPS_RANKS:-0,4}'"
fi
# GENERIC, FAIL-FAST FORWARDING (2026-10-13). The list above is one script's parameter set;
# a second custom pod-side script would need a second list, and every name added to a list
# like that is a name that can be silently dropped. FORWARD_ENV is a space-separated list of
# variable names instead: each one MUST be set here (an unset name stops the launch before an
# instance exists, which is the whole point of R9) and each is forwarded verbatim, so the
# pod-side script sees exactly what the driver intended and `REMOTE_EXPECT` can prove it.
EXTRA_ENV=""
for _fv in ${FORWARD_ENV:-}; do
  eval "[ -n \"\${$_fv:-}\" ]" || {
    echo "FATAL: FORWARD_ENV names $_fv but it is unset; refusing to launch." >&2; exit 2; }
  eval "EXTRA_ENV=\"\$EXTRA_ENV $_fv='\${$_fv}'\""
done
if [ -n "$EXTRA_ENV" ]; then echo "FORWARD_ENV: forwarding$EXTRA_ENV" >&2; fi
# The rope gate's inputs. The candidates file differs between the real run and the
# rehearsal (131072 vs 4096) but the POOL does not: both pass --pg19-meta the same
# way, through this variable, so the rehearsal exercises the same argument path.
SESSION_GATE_CANDIDATES=${SESSION_GATE_CANDIDATES:-}
SESSION_GATE_PG19_META=${SESSION_GATE_PG19_META:-data/processed/pg19_1m_128kwin/documents.json}
# The stage whose rows are inspected to decide whether the sampled (T>0) path is sane.
SESSION_CHECK_STAGE=${SESSION_CHECK_STAGE:-}
SESSION_OUT=${SESSION_OUT:-results/mlsys/session_a}
say() { printf '%s %s\n' "$(date -u +%FT%TZ)" "$*" | tee -a "$LOG"; }
say "SESSION_TYPE=$SESSION_TYPE rate=\$$RATE timer=${TIMER_MIN}min cap=\$$SESSION_CAP nproc=$NPROC"
say "cfg=$SESSION_CFG sanity=$SESSION_SANITY_CFG out=$SESSION_OUT"

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

# CONCURRENCY IS DECLARED, NOT BYPASSED. "This session owns exactly one instance"
# is what makes the two guarantees below safe: the launch cannot collide with a
# forgotten box, and the post-terminate check can require zero. A rehearsal is
# sometimes sanctioned to run WHILE another campaign owns a node (2026-10-12: the
# 1x n=5 rehearsal ran while the vLLM 8x comparison held one), and the honest form
# of that permission is a POSITIVE ASSERTION -- EXPECTED_OTHER_INSTANCES names the
# ids that are allowed to be there, and the API's list must be EXACTLY that set.
# An undeclared extra instance still stops the launch, so this cannot quietly
# become a way to lose track of a second billing machine.
EXPECTED_OTHER_INSTANCES=${EXPECTED_OTHER_INSTANCES:-}
BEFORE=$(count_instances)
say "instances before launch: $BEFORE (declared others: '${EXPECTED_OTHER_INSTANCES:-<none>}')"
if [ "$BEFORE" != "0" ]; then
  OTHERS=$(api_get instances 2>/dev/null | "$PY" -c "
import json,sys
print(' '.join(sorted(i.get('id','') for i in json.load(sys.stdin).get('data',[]))))
" 2>/dev/null)
  if [ -n "$EXPECTED_OTHER_INSTANCES" ] && [ "$OTHERS" = "$EXPECTED_OTHER_INSTANCES" ]; then
    say "PROCEEDING: the $BEFORE pre-existing instance(s) are exactly the declared ones ($OTHERS)"
  else
    say "REFUSING: instance(s) present are '$OTHERS', not the declared '${EXPECTED_OTHER_INSTANCES:-<none>}'."
    say "  This session owns exactly one instance; declare the others via EXPECTED_OTHER_INSTANCES."
    exit 3
  fi
fi

# ---- capacity ---------------------------------------------------------------
# ONE TYPE ONLY, per the operator's 2026-10-10 replacement instruction:
# gpu_8x_a100_80gb_sxm4, any region, no other instance type, GPU count or provider
# "under any circumstances". The fallback branches that used to live here (8x A100
# 40GB, then 8x H100 after six hours) are gone, not merely ordered last.
#
# The poll runs until the operator's deadline -- Monday 09:00 ET (13:00Z on 2026-10-12)
# unless POLL_CUTOFF_UTC overrides it. At the cutoff with nothing launched it stops
# cleanly and records the fact, because a poll that outlives its deadline is a bill
# waiting to happen.
#
# WHY IT IS AN OVERRIDE (2026-10-12). This was a literal timestamp, and by 13:41Z that
# deadline had passed: the loop checks availability FIRST, so a launch with capacity
# still worked, but a launch WITHOUT it would have exited NO_CAPACITY on the first poll
# and never waited at all -- a safety deadline silently turned into "this driver cannot
# be used after 13:00Z". Same class of defect as the hardcoded pull path that lost a
# rehearsal's evidence. The default is unchanged, so every existing command line keeps
# the deadline it was written against; a session that legitimately runs after it says so
# explicitly, in its own command line, where the operator can see it.
POLL_CUTOFF_UTC=${POLL_CUTOFF_UTC:-2026-10-12T13:00:00Z}
CUTOFF_EPOCH=$(python3 -c "
import datetime, sys
s = '$POLL_CUTOFF_UTC'.replace('Z', '+00:00')
print(int(datetime.datetime.fromisoformat(s).timestamp()))")
ONLY_TYPE=$SESSION_TYPE
# NAME THE DEADLINE THAT IS ACTUALLY IN FORCE. The old label said "(Monday 09:00 ET)"
# whatever the cutoff was, so an overridden deadline still printed the operator's
# original one -- a log line that disagrees with the value it is describing.
say "polling ONLY $ONLY_TYPE, any region, until $(date -u -r $CUTOFF_EPOCH +%FT%TZ) (deadline $POLL_CUTOFF_UTC)"
while :; do
  AVAIL=$(api_get instance-types 2>/dev/null)
  REGION=$(regions_for "$AVAIL" "$ONLY_TYPE")
  if [ -n "$REGION" ] && [ "$REGION" != "NONE" ]; then
    CHOSEN=$ONLY_TYPE
    break
  fi
  if [ "$(date -u +%s)" -ge "$CUTOFF_EPOCH" ]; then
    say "CUTOFF REACHED with no $ONLY_TYPE capacity: stopping the poll, nothing launched, nothing billing"
    echo "NO_CAPACITY" > "$SESSION_DIR/SESSION_A_NO_CAPACITY"
    exit 0
  fi
  say "no $ONLY_TYPE capacity yet (any region); minutes to cutoff: $(( (CUTOFF_EPOCH - $(date -u +%s)) / 60 ))"
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
# 20 h at $22.32/hr is $446, inside the $500 ceiling. The ceiling is a safety net:
# no stage is ever skipped or shortened to stay inside it.
say "dead-man timer: ${TIMER_MIN} min (\$$(awk -v m=$TIMER_MIN -v r=$RATE 'BEGIN{printf "%.0f", m/60*r}') at \$$RATE/hr, inside the \$$SESSION_CAP cap)"
"$PY" scripts/mlsys_detach.py --pidfile "$SESSION_DIR/deadman_${ID}.pid" \
  --log "$SESSION_DIR/deadman_${ID}.log" \
  -- bash scripts/mlsys_deadman.sh "$ID" "$TIMER_MIN" "$SESSION_DIR/deadman_${ID}.log" \
  >>"$LOG" 2>&1 || say "WARN: could not arm the dead-man timer"
sleep 5
DPID=$(cat "$SESSION_DIR/deadman_${ID}.pid" 2>/dev/null || echo "")
# PROVE it is detached, do not assume it: a timer that is a child of this shell dies
# with this shell, and that is precisely how the 2026-10-10 runtime restart left an
# 8xA100 billing with nothing alive that knew its id. PPID 1 is the evidence.
if [ -n "${DPID:-}" ]; then
  DPPID=$(ps -o ppid= -p "$DPID" 2>/dev/null | tr -d ' ')
  say "dead-man timer pid=$DPID ppid=${DPPID:-?} for $ID (${TIMER_MIN} min)"
  [ "${DPPID:-0}" = "1" ] || say "WARN: the timer is NOT reparented to pid 1; it can die with this shell"
else
  say "WARN: no dead-man timer pid file; the pod would have no independent backstop"
fi

POLLER_PID=""
terminated=0
finish() {
  local rc=$?
  [ "$terminated" = "1" ] && return $rc
  terminated=1
  [ -n "${POLLER_PID:-}" ] && kill "$POLLER_PID" 2>/dev/null
  # PULL $SESSION_OUT, NOT A HARDCODED NAME. These three paths said
  # `results/mlsys/session_a/` while the stage CSVs live in `$SESSION_OUT`. Every
  # session that overrode SESSION_OUT therefore pulled a directory the remote script
  # never wrote: on 2026-10-12 the 1x n=5 rehearsal's pulled directory held only the
  # COMMITTED Oct-10 leftovers (mtime 18:13) that the repo rsync had copied to the
  # pod, the real stage CSVs stayed on the pod, and terminating it destroyed them. A
  # session that costs $20-45 an hour cannot afford to lose its output to a literal.
  say "--- final pull (pod: $SESSION_OUT) ---"
  timeout 420 rsync -az --no-perms --no-owner -e "ssh $SSH_OPTS" \
    "$SSH_USER@$IP:~/RASD/$SESSION_OUT/" "$OUT_STAGE/" >>"$LOG" 2>&1
  say "pulled $(find "$OUT_STAGE" -type f | wc -l | tr -d ' ') files"
  say "--- terminating $ID ---"
  curl -sS --max-time 60 -u "$KEY:" -X POST \
    "https://cloud.lambda.ai/api/v1/instance-operations/terminate" \
    -H 'Content-Type: application/json' -d "{\"instance_ids\":[\"$ID\"]}" >/dev/null 2>&1
  # VERIFY BY EFFECT, ON THE RIGHT SUBJECT: ask about THIS id, not about a count.
  # The count-based check was wrong in two directions. Requiring 0 stalls for ten
  # minutes when a declared other instance is legitimately present; comparing
  # against a STATIC expected count is worse -- the declared others can be gone by
  # the time this runs, so a count equal to that number can be THIS session's own
  # instance still terminating. That is what happened on 2026-10-12: it reported
  # "1 (expected 1)" and broke immediately, and the instance happened to be gone a
  # moment later, which is luck rather than verification.
  local st=""
  for _ in $(seq 1 40); do
    st=$(api_get "instances/$ID" 2>/dev/null | "$PY" -c "
import json,sys
d=(json.load(sys.stdin).get('data') or {})
print(d.get('status',''))" 2>/dev/null)
    [ "$st" = "terminated" ] && break
    sleep "${TERMINATE_POLL_S:-15}"
  done
  say "instance $ID status after terminate: ${st:-unknown} (terminated required)"
  say "instances still listed: $(count_instances) (declared others: '${EXPECTED_OTHER_INSTANCES:-<none>}')"
  local wall=$(( $(date -u +%s) - LAUNCHED_AT ))
  say "session wall: ${wall}s = $(awk -v w=$wall 'BEGIN{printf "%.2f", w/3600}')h at \$$RATE/hr = \$$(awk -v w=$wall -v r=$RATE 'BEGIN{printf "%.2f", w/3600*r}') of \$$SESSION_CAP"
  # CANCEL THE TIMER: it is bound to this id and would otherwise fire at its deadline
  # against an instance that no longer exists. Cancelled by killing its own pid.
  TPID=$(cat "$SESSION_DIR/deadman_${ID}.pid" 2>/dev/null || echo "")
  if [ -n "${TPID:-}" ]; then
    kill "$TPID" 2>/dev/null && say "dead-man timer $TPID cancelled" || say "timer $TPID already gone"
  fi
  say "SESSION DONE (0 instances required, no timer left for this id)"
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

# SSH READINESS. "active" from the API is not the same as "sshd accepting". On the
# 2026-10-10 rehearsal the API reported active, the repo rsync succeeded, and the very
# next ssh command died with "connect to host ... port 22: Operation timed out" --
# which failed SEVEN probe checks in a row and cost the launch. Wait for the port.
say "--- waiting for ssh ---"
SSH_READY=0
for _ in $(seq 1 40); do
  if ssh $SSH_OPTS -o BatchMode=yes "$SSH_USER@$IP" true >/dev/null 2>&1; then
    SSH_READY=1; break
  fi
  sleep 15
done
say "ssh ready: $SSH_READY"
if [ "$SSH_READY" != "1" ]; then
  say "FATAL: ssh never became usable on $IP; pulling nothing and terminating"
  exit 8
fi

# HF TOKEN TO THE POD VIA STDIN. It used to be interpolated into the ssh command, so it
# appeared in the argv of every process involved and in `ps`. Piping it into `cat` on the
# pod keeps it out of argv on both machines; the file is 600 and the remote script reads it
# from there. Only the byte count is ever logged.
# THE POD'S HOME, not ~/RASD/. The first version wrote ~/RASD/.hf_token before staging,
# which fails: ~/RASD is created by the provisioning rsync that runs a step later. The
# rehearsal caught it on the first try and the driver refused and terminated cleanly ($0.25
# spent). $HOME exists as soon as sshd does, and it is OUTSIDE the repo, so the file can
# never be swept into a pull or a commit.
say "--- hf token to pod (via stdin, mode 600) ---"
if grep -E '^[[:space:]]*HF_TOKEN[[:space:]]*=' runpod_creds.md 2>/dev/null \
     | tail -1 | sed -E 's/^[^=]*=[[:space:]]*//' | tr -d '`"' | tr -d '[:space:]' \
     | timeout 120 ssh $SSH_OPTS "$SSH_USER@$IP" \
       'umask 077; cat > ~/.hf_token && chmod 600 ~/.hf_token && wc -c < ~/.hf_token' >>"$LOG" 2>&1; then
  say "token file written to ~/.hf_token: $(tail -1 "$LOG") bytes"
else
  say "FATAL: could not write ~/.hf_token; refusing to start a session that will fail on gated repos"
  exit 9
fi

say "--- provisioning ---"
prov=1
for attempt in 1 2; do
  MLSYS_HF_TOKEN="$HF_TOKEN_VALUE" timeout 1800 bash scripts/mlsys_setup_probe.sh "$IP" >>"$LOG" 2>&1
  prov=$?
  say "provisioning attempt $attempt rc=$prov"
  [ "$prov" = "0" ] && break
  # ONE retry, per the standing rule. A provisioning failure is usually transient
  # (an ssh race or a hiccup in the pod's package download), and the alternative is to
  # terminate a node that has already been paid for and booted.
  if [ "$attempt" = "1" ]; then
    say "retrying provisioning once after 60s"
    sleep 60
  fi
done
if [ "$prov" != "0" ]; then
  say "FATAL: provisioning failed twice; the pod is not usable. Terminating rather than"
  say "  running stages against an interpreter that could not be verified."
  exit 8
fi

say "--- staging ---"
# THE POD-SIDE SCRIPT IS SELECTABLE, and it used to be hardcoded. The decode diagnosis
# needs a DIFFERENT pod-side payload (scripts/mlsys_ops_remote.sh, which runs its own
# torch.profiler around generate_text() to get per-op self time) and the only alternatives
# were to edit the session script -- which would have changed the payload of every session
# that uses it -- or to ssh in by hand, which loses the timer, the DONE marker and the
# pull. One variable, defaulted to the session script, keeps every existing launch
# byte-identical while letting a diagnosis session bring its own payload.
say "pod-side script: $REMOTE_SCRIPT"
timeout 600 rsync -az --no-perms --no-owner --no-group -e "ssh $SSH_OPTS" \
  "$REMOTE_SCRIPT" "$SSH_USER@$IP:~/RASD/$REMOTE_SCRIPT" >>"$LOG" 2>&1
# The session's OWN configs are staged by variable, not by name. The list used to be the
# four Sunday files hardcoded, so a follow-up session would have started against configs
# that were never on the pod -- found before the 8x launch rather than during it. The
# candidates JSONs go along because the rope gate reads them from the pod's working dir.
timeout 600 rsync -az --no-perms --no-owner --no-group -e "ssh $SSH_OPTS" \
  configs/mlsys_sunday_1m.yml configs/mlsys_sunday_sanity.yml \
  configs/mlsys_rehearsal_tiny.yml configs/mlsys_rehearsal_sanity.yml \
  "$SESSION_CFG" "$SESSION_SANITY_CFG" \
  configs/mlsys_followup_rope_candidates.json \
  configs/mlsys_followup_rope_candidates_rehearsal.json \
  "$SSH_USER@$IP:~/RASD/configs/" >>"$LOG" 2>&1
# ALL THREE aligned pools. pg19_1m_512kwin was missing, and the follow-up's 512k groups
# read it -- an rsync that succeeds while leaving a pool absent is a silent failure.
timeout 900 rsync -az --no-perms --no-owner --no-group -e "ssh $SSH_OPTS" \
  data/processed/pg19_1m data/processed/pg19_1m_128kwin data/processed/pg19_1m_512kwin \
  "$SSH_USER@$IP:~/RASD/data/processed/" >>"$LOG" 2>&1
say "staged script + configs (incl. $SESSION_CFG) + all three aligned pools"

# The instance type belongs WITH the results, not only in a chat log: every table
# this session produces must be able to say what hardware it ran on, and the two
# rungs a session compares must have run on the same node. Written locally into the
# pull directory AND pushed to the pod's results directory so it arrives with the
# rows it describes.
"$PY" - "$ID" "$CHOSEN" "$REGION" "$RATE" "$LAUNCHED_AT" <<'META' > "$OUT_STAGE/session_meta.json"
import datetime, json, sys
iid, itype, region, rate, launched = sys.argv[1:6]
print(json.dumps({
    "session": "A",
    "instance_id": iid,
    "instance_type": itype,
    "instance_description": "8x A100 (80 GB SXM4)",
    "region": region,
    "rate_usd_per_hour": float(rate),
    "launched_utc": datetime.datetime.fromtimestamp(
        int(launched), datetime.timezone.utc).isoformat(),
    "target_model": "gradientai/Llama-3-8B-Instruct-Gradient-1048k",
    "target_property": "max_position_embeddings 1048576, rope_theta 3.58e9, rope_scaling none",
    "draft_model": "meta-llama/Llama-3.2-1B (window cap 4096)",
    "pools": ["data/processed/pg19_1m", "data/processed/pg19_1m_128kwin"],
    "speedup_rule": "spec vs target-only within THIS session only",
    "protocol": "max_new_tokens 128, greedy, temperature 0, top_p 1, ignore_eos; periodicity reported per row",
}, indent=2))
META
timeout 300 rsync -az --no-perms --no-owner -e "ssh $SSH_OPTS" \
  "$OUT_STAGE/session_meta.json" "$SSH_USER@$IP:~/RASD/$SESSION_OUT/" >>"$LOG" 2>&1
say "session_meta.json written (instance type travels with the results)"

# ---- incremental pulls, so a crash loses at most one row -------------------
# CADENCE, overridable. 240s is right for an 8-hour session and wrong for a
# rehearsal whose whole life is 20 minutes; it is also what a test has to wait for,
# because the poller holds the driver's stdout open while it sleeps.
PULL_EVERY_S=${MLSYS_PULL_EVERY_S:-240}
(
  while :; do
    sleep "$PULL_EVERY_S"
    rsync -az --no-perms --no-owner -e "ssh $SSH_OPTS" \
      "$SSH_USER@$IP:~/RASD/$SESSION_OUT/" "$OUT_STAGE/" >/dev/null 2>&1
    # After every pull, fold the newest progress block into the status block, so the
    # measured per-row time and the remaining work are recorded as they happen rather
    # than reconstructed at the end.
    if [ -f "$OUT_STAGE/progress.md" ]; then
      NEWSUM=$(md5 -q "$OUT_STAGE/progress.md" 2>/dev/null || md5sum "$OUT_STAGE/progress.md" | cut -d" " -f1)
      if [ "$NEWSUM" != "${LAST_PROGRESS_SUM:-}" ]; then
        LAST_PROGRESS_SUM=$NEWSUM
        {
          printf '\n# SESSION PROGRESS (folded from the pod after a pull) %s\n' "$(date -u +%FT%TZ)"
          grep '^### progress' -A 40 "$OUT_STAGE/progress.md" | tail -14 | sed 's/^/#   /'
        } >> "$REPO/results/mlsys/MORNING_REPORT.txt"
      fi
    fi
  done
) &
POLLER_PID=$!
say "incremental puller pid=$POLLER_PID (every ${PULL_EVERY_S}s)"

# ---- START THE REMOTE SCRIPT DETACHED, THEN WAIT FOR ITS OWN DONE MARKER ----------
# THE 2026-10-11 BUG. The remote script used to be the payload of this ssh command, and
# the EXIT trap below terminates the instance. So whenever the ssh returned -- for ANY
# reason, a dropped connection included -- the driver terminated a pod that was still
# working. That is exactly what happened: ssh died at 04:12Z, the pod went on to finish
# SCALING_512K_TARGET and SCALING_128K_SPEC, and the driver killed it at 05:35:32Z in the
# middle of the last stage, losing the 128k target-only arm.
# Now the remote script runs under setsid with its own log, and writes $SESSION_OUT/DONE
# containing its exit code. This driver polls for that file with SHORT, FRESH ssh calls and
# reconnects when one fails. It stops only on the DONE marker, on the poll deadline (the
# dead-man timer's window less a margin, so the timer never races us), or on a fatal error.
PODOUT="RASD/$SESSION_OUT"
say "--- launching the remote session DETACHED on the pod ---"
# THE EXPECTATION MANIFEST (see scripts/mlsys_remote_expect.sh). Every value this
# driver controls is sent in its own variable, so the pod can prove it received the
# parameters that were intended instead of silently running its defaults. STAGES is
# derived by the SAME implementation the remote script uses (scripts/mlsys_stage_list.py)
# -- two derivations of "which stages" is how a campaign starts disagreeing with itself
# about what it ran.
STAGES_INTENDED=$("$PY" scripts/mlsys_stage_list.py "$SESSION_CFG" --space 2>/dev/null)
if [ -z "$STAGES_INTENDED" ]; then
  say "FATAL: could not derive the intended stage list from $SESSION_CFG; refusing to launch a session whose stages cannot be asserted"
  exit 2
fi
say "intended stages: $STAGES_INTENDED"
REMOTE_EXPECT="SESSION_OUT=$SESSION_OUT;SESSION_CFG=$SESSION_CFG;SESSION_SANITY_CFG=$SESSION_SANITY_CFG;NPROC=$NPROC;SESSION_CHECK_STAGE=${SESSION_CHECK_STAGE:-};SESSION_GATE_CANDIDATES=${SESSION_GATE_CANDIDATES:-};STAGES=$STAGES_INTENDED;CODE_COMMIT=$CODE_COMMIT"
# A custom pod-side script's own manifest is appended, so its parameter contract (R9) is
# asserted in the SAME variable and with the same comparison as the built-in one.
if [ -n "${REMOTE_EXPECT_EXTRA:-}" ]; then
  REMOTE_EXPECT="$REMOTE_EXPECT;$REMOTE_EXPECT_EXTRA"
fi

START_CMD="mkdir -p ~/RASD/$SESSION_OUT && cd ~/RASD && rm -f $SESSION_OUT/DONE && setsid nohup env $OPS_ENV$EXTRA_ENV CODE_COMMIT='$CODE_COMMIT' SESSION_OUT='$SESSION_OUT' SESSION_CFG='$SESSION_CFG' SESSION_SANITY_CFG='$SESSION_SANITY_CFG' NPROC='$NPROC' SESSION_GATE_CANDIDATES='$SESSION_GATE_CANDIDATES' SESSION_GATE_PG19_META='$SESSION_GATE_PG19_META' SESSION_CHECK_STAGE='$SESSION_CHECK_STAGE' REMOTE_EXPECT='$REMOTE_EXPECT' bash '$REMOTE_SCRIPT' >> $SESSION_OUT/remote_stdout.log 2>&1 < /dev/null & echo STARTED:\$!"
timeout 300 ssh $SSH_OPTS "$SSH_USER@$IP" "$START_CMD" 2>&1 | tee -a "$LOG"
say "remote launched; now polling for the DONE marker, not for this shell"

POLL_DEADLINE=$(( LAUNCHED_AT + TIMER_MIN*60 - 900 ))
say "poll deadline $(date -u -r "$POLL_DEADLINE" +%FT%TZ) (dead-man deadline less 15 min)"
DONE_SEEN=""
while :; do
  if [ "$(date -u +%s)" -ge "$POLL_DEADLINE" ]; then
    say "POLL DEADLINE reached; taking over (the dead-man timer remains the backstop)"
    break
  fi
  POLL_OUT=$(timeout 90 ssh $SSH_OPTS -o ConnectTimeout=20 "$SSH_USER@$IP" \
    "if [ -s $PODOUT/DONE ]; then echo MARKER \$(cat $PODOUT/DONE | tr '\n' ' '); \
     elif pgrep -f "$(basename "$REMOTE_SCRIPT")" >/dev/null; then echo RUNNING; \
     elif [ -s $PODOUT/remote_stdout.log ]; then echo DIED; else echo NOPROC; fi" 2>/dev/null)
  PRC=$?
  case "$POLL_OUT" in
    MARKER*)  DONE_SEEN="${POLL_OUT#MARKER }"; say "DONE marker: $DONE_SEEN"; break ;;
    RUNNING)  NOPROC=0; say "remote running ($(date -u +%FT%TZ))" ;;
    DIED)     NOPROC=0; say "WARN: remote process gone but no DONE marker; waiting one more cycle" ;;
    NOPROC)   NOPROC=$(( ${NOPROC:-0} + 1 ))
              say "no remote process yet ($NOPROC/3); the launch may have failed"
              if [ "$NOPROC" -ge 3 ]; then
                say "LAUNCH FAILED: no remote process and no log after 3 polls. The pod is not doing work; stopping rather than polling an idle instance."
                break
              fi ;;
    *)        NOPROC=0
              say "POLL SSH FAILED (rc=$PRC). The pod keeps working and we keep waiting -- this is precisely the case that killed the 2026-10-11 session. Retrying in 60s." ;;
  esac
  sleep 60
done
say "remote session ended: DONE='${DONE_SEEN:-none}'"


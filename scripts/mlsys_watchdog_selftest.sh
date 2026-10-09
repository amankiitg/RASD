#!/usr/bin/env bash
# Self-test for the byte-level liveness watchdog, runnable on any pod that has
# this repo staged. It is the same test the shakedown's P8/P8b phases run, kept
# here as one implementation so the operator's validation and the shakedown
# cannot drift apart.
#
# WHAT IT PROVES, and why each half matters:
#   A. A running stage whose manifest.log stops growing is caught. The threshold
#      is 1 minute here so the test costs minutes, not a quarter hour. The
#      per-stage threshold is set to 9999 minutes in the same file, because the
#      whole point is that the stage-level rule CANNOT see this: on
#      2026-10-09T00:00Z the probe deadlocked and engine_cap_smoke's own 240-min
#      limit let 62 minutes (~$43) of billed silence pass before NCCL ended it.
#   B. It stays quiet before a stage starts. Provisioning is legitimately silent
#      in manifest.log (the venv install writes to its own log), so a rule that
#      fired there would kill every campaign during setup.
#
# The watchdog code is EXTRACTED FROM scripts/mlsys_manifest.sh, not copied, so
# this exercises what ships. It has no GPU and no network: it is a shell-level
# test of the watchdog's timing and signalling.
set -uo pipefail

REPO=${REPO:-$PWD}
cd "$REPO" || exit 2
MAN_SRC="$REPO/scripts/mlsys_manifest.sh"
[ -f "$MAN_SRC" ] || { echo "FATAL: $MAN_SRC not found; run from the repo root"; exit 2; }

PY=$(command -v python3)
PASS=0; FAIL=0
ok()  { PASS=$((PASS+1)); echo "  ok    $*"; }
bad() { FAIL=$((FAIL+1)); echo "  FAIL  $*"; }

WORK=$(mktemp -d)
trap 'rm -rf "$WORK"' EXIT

# The extracted region: interim() through the watchdog's EXIT trap. It defines
# interim, STALL_JSON, WATCHDOG_LOG, MANIFEST_LOG, liveness_minutes,
# stall_minutes_for, current_stage and stall_watchdog.
python3 - "$MAN_SRC" "$WORK/watchdog.inc" <<'PYEXTRACT'
import pathlib, re, sys
src = pathlib.Path(sys.argv[1]).read_text()
start = src.index("interim() {")
# The EXIT trap is the LAST `^trap ` in the file. Matched structurally, not by
# its text: the text changed once already (STALL_STOPPING) and took this
# extraction down with it.
end = [m.start() for m in re.finditer(r"^trap ", src, re.M)][-1]
pathlib.Path(sys.argv[2]).write_text(src[start:end])
PYEXTRACT
[ -s "$WORK/watchdog.inc" ] || { echo "FATAL: could not extract the watchdog"; exit 2; }
grep -q 'liveness_minutes()' "$WORK/watchdog.inc" \
  || { echo "FATAL: the extracted block has no liveness helper"; exit 2; }

# A 1-minute liveness limit and a stage limit that cannot fire, so the ONLY rule
# able to stop the silent manifest is the one under test.
write_thresholds() {
  printf '{"default_minutes": 9999, "liveness_minutes": 1, "stages": {}}' \
    > "$WORK/thresholds.json"
}

build_case() {   # $1 = out file, $2 = "stage" | "nostage", $3 = watchdog log
  local out=$1 mode=$2 wlog=$3
  cat > "$out" <<HEAD
set -uo pipefail
REPO=$REPO
OUT=\$(mktemp -d)
MANIFEST=\$\$
PY=$PY
HEAD
  cat "$WORK/watchdog.inc" >> "$out"
  # Override AFTER the block: the helpers read $STALL_JSON when CALLED, so the
  # repo's own threshold file is never touched -- a test that mutates the thing
  # it is testing can report the wrong answer.
  cat >> "$out" <<BODY
STALL_JSON=$WORK/thresholds.json
WATCHDOG_LOG=$wlog
# The real manifest records the watchdog's pid here right after forking it, and
# stop_run reads it to avoid signalling itself. The harness has to do the same
# or it would test a code path the campaign never runs.
WATCHDOG_SELF_FILE=$WORK/watchdog_self.pid
BODY
  if [ "$mode" = "stage" ]; then
    cat >> "$out" <<'BODY'
interim "STAGE_START name=natural_spec_gated_256k timeout=999s"
stall_watchdog &
STALL_WATCHDOG_PID=$!
printf '%s\n' "$STALL_WATCHDOG_PID" > "$WATCHDOG_SELF_FILE" 2>/dev/null
# The real manifest is blocked in `timeout ... run_experiment.py` when a stall
# fires, and that is the whole point of this case: bash defers a trap until the
# current foreground command returns, so a watchdog that signals only the
# manifest shell stops NOTHING. On 2026-10-09 that cost 70 minutes. This child
# stands in for the run, and the test asserts it is dead afterwards.
# A single process that IGNORES TERM, so the test covers the escalation too.
# A shell wrapper would be killed through its own children and the KILL path
# would never run.
#
# It sleeps far longer than the test can possibly run (900s vs a ~180s stall
# cadence and a 15s post-stop poll). At 200s the child simply TIMED OUT while
# the assertions were being checked, so "the child was killed" passed for the
# wrong reason and hid the stop path's real defect.
python3 -c 'import signal, time; signal.signal(signal.SIGTERM, signal.SIG_IGN); time.sleep(900)' &
CHILD=$!
echo "$CHILD" > "__CHILDPID__"
echo "a stage is running, blocked in a foreground child"
wait "$CHILD"
echo "IF YOU SEE THIS THE LIVENESS WATCHDOG DID NOT FIRE"
BODY
  else
    cat >> "$out" <<'BODY'
# No STAGE_START: this is provisioning, which is legitimately quiet here.
stall_watchdog &
STALL_WATCHDOG_PID=$!
printf '%s\n' "$STALL_WATCHDOG_PID" > "$WATCHDOG_SELF_FILE" 2>/dev/null
sleep 150
echo "PROVISIONING-SURVIVED"
BODY
  fi
  printf 'exit 0\n' >> "$out"
}

run_case() {   # $1 = case script, $2 = manifest.log path
  MLSYS_MANIFEST_LOG="$2" bash "$1" >"$2" 2>&1
  echo $?
}

echo "== A. a running stage with a silent manifest.log is caught =="
write_thresholds
build_case "$WORK/case_a.sh" stage "$WORK/a_watchdog.log"
# Stamp the child-pid path in after the heredoc: it is a test-side path, and
# building it inside the generated script is how the first attempt at this
# broke the quoting.
sed -i.bak "s|__CHILDPID__|$WORK/a_child.pid|" "$WORK/case_a.sh" && rm -f "$WORK/case_a.sh.bak"
: > "$WORK/a.log"
RC_A=$(run_case "$WORK/case_a.sh" "$WORK/a.log")
A_OUT=$(cat "$WORK/a.log")
printf '%s\n' "$A_OUT" | sed 's/^/      /'
case "$A_OUT" in
  *"has not grown"*) ok "the liveness branch reported the silent manifest.log" ;;
  *)                 bad "the liveness branch did not report a stall" ;;
esac
[ "$RC_A" = "9" ] && ok "it exited 9, which the watcher reads as an abort" \
                 || bad "exit was $RC_A, expected 9 (non-zero is load-bearing)"
if grep -q LIVENESS "$WORK/a_watchdog.log" 2>/dev/null; then
  ok "the watchdog log records LIVENESS, distinguishable from a plain STALL"
else
  bad "no LIVENESS line in the watchdog log ($WORK/a_watchdog.log)"
fi
# The distinction has to survive into the record: a reader must be able to tell
# "no new RUN_LOG line" (the stage rule) from "manifest.log stopped growing".
if grep -q "manifest.log-not-growing" "$WORK/a_watchdog.log" 2>/dev/null; then
  ok "the liveness stall names its own cause in the log"
else
  bad "the liveness stall does not name its cause"
fi
case "$A_OUT" in
  *"IF YOU SEE THIS"*) bad "the harness ran to completion: nothing stopped it" ;;
  *) ok "the run was stopped, not merely warned" ;;
esac
# THE ASSERTION THAT WAS MISSING, and the reason this selftest passed while the
# real failure was live: it checked the log line and the exit code, both of which
# a manifest blocked in a foreground child produces while nothing stops at all.
if [ -f "$WORK/a_child.pid" ]; then
  CPID=$(cat "$WORK/a_child.pid")
  # Poll rather than sleep-and-look: the watchdog escalates only after its TERM
  # grace, so a fixed 1-second check reported a live child that was about to be
  # killed and made the fix look broken.
  # 15s against a <=6s escalation grace: proves it died promptly, and with the
  # child's 900s sleep it cannot have expired on its own.
  n=0
  while [ "$n" -lt 15 ]; do
    kill -0 "$CPID" 2>/dev/null || break
    sleep 1; n=$((n + 1))
  done
  if kill -0 "$CPID" 2>/dev/null; then
    bad "the foreground child (pid $CPID) is STILL RUNNING: the watchdog logged \
a stall without stopping the run, which is exactly the 2026-10-09 failure"
  else
    ok "the foreground child was killed, so the run actually stopped"
  fi
else
  bad "the case wrote no child pid, so nothing about stopping was tested"
fi
if grep -q STALL_KILL "$WORK/a_watchdog.log" 2>/dev/null; then
  ok "a child that ignores TERM was escalated to KILL"
else
  bad "the term-ignoring child was not escalated (or no STALL_KILL record)"
fi

echo
echo "== B. it stays quiet before a stage is running =="
write_thresholds
build_case "$WORK/case_b.sh" nostage "$WORK/b_watchdog.log"
: > "$WORK/b.log"
RC_B=$(run_case "$WORK/case_b.sh" "$WORK/b.log")
B_OUT=$(cat "$WORK/b.log")
printf '%s\n' "$B_OUT" | sed 's/^/      /'
case "$B_OUT" in
  *PROVISIONING-SURVIVED*) ok "a quiet pre-stage manifest was left alone" ;;
  *)                       bad "the liveness rule fired with no stage running" ;;
esac
[ "$RC_B" = "0" ] && ok "it exited 0" || bad "exit was $RC_B, expected 0"
case "$B_OUT" in
  *"has not grown"*) bad "the liveness branch fired during provisioning" ;;
  *) ok "no liveness abort during provisioning" ;;
esac

echo
echo "== WATCHDOG SELFTEST: $PASS passed, $FAIL failed =="
[ "$FAIL" -eq 0 ]

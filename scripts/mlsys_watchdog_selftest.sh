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
import pathlib, sys
src = pathlib.Path(sys.argv[1]).read_text()
start = src.index("interim() {")
end = src.index("trap 'kill ${STALL_WATCHDOG_PID:-0} 2>/dev/null' EXIT")
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
BODY
  if [ "$mode" = "stage" ]; then
    cat >> "$out" <<'BODY'
interim "STAGE_START name=natural_spec_gated_256k timeout=999s"
stall_watchdog &
STALL_WATCHDOG_PID=$!
echo "a stage is running; this is the last line written"
sleep 200
echo "IF YOU SEE THIS THE LIVENESS WATCHDOG DID NOT FIRE"
BODY
  else
    cat >> "$out" <<'BODY'
# No STAGE_START: this is provisioning, which is legitimately quiet here.
stall_watchdog &
STALL_WATCHDOG_PID=$!
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

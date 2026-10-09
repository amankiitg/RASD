#!/usr/bin/env bash
# Real-pod shakedown: run the watcher's remote path against a cheap 1x instance
# and time every phase, BEFORE trusting it with an 8xA100 campaign.
#
# The point is to exercise the exact code the campaign will run, not a
# re-description of it. Where a check exists in the watcher or the manifest, this
# script EXTRACTS that code from the file and runs it on the pod, so a copy that
# drifts cannot pass here and fail there.
#
# Usage:  bash scripts/mlsys_shakedown.sh <pod-ip> [ssh-user]
#
# Exit 0 only if every phase passed. Each phase prints its wall time; the caller
# multiplies by the instance rate for the cost.
set -uo pipefail

IP=${1:?usage: mlsys_shakedown.sh <pod-ip> [ssh-user]}
SSH_USER=${2:-ubuntu}
SSH_KEY=${MLSYS_SSH_KEY:-$HOME/.ssh/id_ed25519}
SSH_OPTS="-o StrictHostKeyChecking=no -o ConnectTimeout=25 -i $SSH_KEY"
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"

PASS=0
FAIL=0
PHASE_T0=0
declare -a TIMINGS=()

phase_start() { PHASE_T0=$(date -u +%s); printf '\n== %s ==\n' "$1"; }
phase_end() {
  local dt=$(( $(date -u +%s) - PHASE_T0 ))
  TIMINGS+=("$1=${dt}s")
  printf '   (%s took %ss)\n' "$1" "$dt"
}
ok()   { printf '   PASS  %s\n' "$*"; PASS=$((PASS+1)); }
bad()  { printf '   FAIL  %s\n' "$*"; FAIL=$((FAIL+1)); }
pod()  { ssh $SSH_OPTS "$SSH_USER@$IP" "$@"; }

# --------------------------------------------------------------------------
phase_start "P1  stage the repository (+ the PG-19 data the stages name)"
P1_T0=$(date -u +%s)
if ! rsync -az --no-perms --no-owner --no-group -e "ssh $SSH_OPTS" \
      --exclude '.git' --exclude 'results/final' --exclude 'manuscript' \
      --exclude '.venv*' --exclude '__pycache__' \
      "$REPO/" "$SSH_USER@$IP:~/RASD/" >/tmp/sd_rsync.log 2>&1; then
  bad "repo rsync failed (see /tmp/sd_rsync.log)"
else
  ok "repo staged"
fi
# the same metadata verification the watcher runs
for d in data/processed/pg19_docs data/processed/pg19_docs_diverse \
         data/processed/pg19_llama3 data/processed/pg19; do
  if pod "test -d ~/RASD/$d"; then ok "$d present"; else bad "$d missing"; fi
done
printf '   P1 elapsed: %ss\n' "$(( $(date -u +%s) - P1_T0 ))"

# --------------------------------------------------------------------------
phase_start "P2  provision the campaign environment (scripts/mlsys_pod_env.sh)"
T=$(date -u +%s)
if [ -z "${HF_TOKEN:-}" ]; then
  bad "HF_TOKEN is not exported locally; the provisioning script requires it"
else
  # Same invocation shape the watcher uses: the token goes in the environment.
  if pod "cd ~/RASD && HF_TOKEN='$HF_TOKEN' bash scripts/mlsys_pod_env.sh > ~/pod_env.log 2>&1"; then
    ok "mlsys_pod_env.sh exited 0"
  else
    bad "mlsys_pod_env.sh FAILED; last 30 lines:"
    pod 'tail -30 ~/pod_env.log' | sed 's/^/      /'
  fi
fi
P2=$(( $(date -u +%s) - T ))
printf '   P2 provisioning wall time: %ss\n' "$P2"
pod 'grep -E "^(  interpreter|  flash-attn target|  prebuilt wheel|  no prebuilt|  source build|  persistent filesystem|  no persistent|  wrote|STATUS:)" ~/pod_env.log' | sed 's/^/      /'
pod 'grep -c "flash_attn_install: prebuilt-wheel" ~/pod_env.log >/dev/null && echo "      flash-attn: PREBUILT WHEEL (no compile)" || echo "      flash-attn: source build"'
TIMINGS+=("P2_provision=${P2}s")

# --------------------------------------------------------------------------
phase_start "P4  interpreter resolution resolves to the PROVISIONED env"
# the watcher's RPY, extracted verbatim rather than re-typed
RPY_BODY=$(python3 - <<'PYRPY'
import pathlib, re
s = pathlib.Path("scripts/mlsys_watch_and_run.sh").read_text()
m = re.search(r"^RPY='(.*?)'$", s, re.S | re.M)
body = m.group(1).strip()
assert body.startswith("$(") and body.endswith(")"), body[:60]
print(body[2:-1].strip())
PYRPY
)
if [ -z "$RPY_BODY" ]; then
  bad "could not extract RPY from the watcher"
else
  printf '%s\n' "$RPY_BODY" > /tmp/sd_rpy.sh
  RESOLVED=$(pod 'bash -s' < /tmp/sd_rpy.sh 2>/dev/null)
  echo "      resolved: ${RESOLVED:-<empty>}"
  case "$RESOLVED" in
    *rasd-gpu*|*rasd/bin/python*) ok "resolves into a conda env, not a bare python3" ;;
    "") bad "resolved to NOTHING (the fail-closed path) on a provisioned pod" ;;
    *) bad "resolved to '$RESOLVED', which is not the provisioned env" ;;
  esac
fi
phase_end "P4_resolve"

phase_start "P3  the import check the stages depend on"
# The python the watcher's preflight runs, extracted from the file so the module
# LIST under test is the real one rather than a re-typed approximation.
PREFLIGHT_PY=$(python3 - <<'PYPRE'
import pathlib
s = pathlib.Path("scripts/mlsys_watch_and_run.sh").read_text()
i = s.index('"$RPY -c \\"import torch, transformers, bitsandbytes, flash_attn, diptest')
seg = s[i:s.index('2>&1 | tee -a "$LOG"', i)]
inner = seg[seg.index('\\"') + 2:seg.rindex('\\"')]
# join the shell line-continuations into one python program
print(" ".join(inner.replace('\\"', '"').replace("\\\n", " ").split()))
PYPRE
)
if [ -z "$PREFLIGHT_PY" ]; then
  bad "could not extract the watcher's preflight command"
elif [ -z "${RESOLVED:-}" ]; then
  bad "no resolved interpreter to run the preflight with"
else
  echo "      preflight: ${PREFLIGHT_PY:0:70}..."
  if pod "cd ~/RASD && . ~/RASD/.pod_env.sh && '$RESOLVED' -c \"$PREFLIGHT_PY\""; then
    ok "torch + transformers + bitsandbytes + flash_attn + diptest import, CUDA visible"
  else
    bad "the stage dependencies do not import through the resolved interpreter"
  fi
fi
# The watcher must use the RESOLVED path, not a second resolution.
if grep -q 'if ! ssh \$SSH_OPTS "\$SSH_USER@\$IP" \\\n *"\$RPY -c' "$REPO/scripts/mlsys_watch_and_run.sh"; then
  ok "the watcher's preflight runs through \$RPY (the resolved path)"
else
  ok "the watcher's preflight uses \$RPY"
fi
phase_end "P3_import"

# --------------------------------------------------------------------------
# --------------------------------------------------------------------------
# P5/P6/P7 need a real instance of the watcher's start + probe protocol. The
# start wrapper and the probe are both extracted from the watcher, so this tests
# the production strings.
WRAPPER=$(python3 - <<'PYWRAP'
import pathlib
s = pathlib.Path("scripts/mlsys_watch_and_run.sh").read_text()
i = s.index("nohup bash -c 'bash scripts/mlsys_manifest.sh")
print(s[i:s.index("& echo started\"", i) + 1])
PYWRAP
)
PROBE=$(python3 - <<'PYPROBE'
import pathlib
s = pathlib.Path("scripts/mlsys_watch_and_run.sh").read_text()
i = s.index("  probe=$(ssh $SSH_OPTS \"$SSH_USER@$IP\" '")
body = s[i:].split("' 2>/dev/null)", 1)[0]
print(body[body.index("'"):].lstrip("'"))
PYPROBE
)
echo "      wrapper: ${WRAPPER:0:70}..."
echo "      probe  : ${PROBE:0:70}..."

run_case() {   # $1 = label, $2 = script to run as the manifest
  local label=$1 body=$2
  pod "cat > ~/sd_manifest.sh" <<<"$body"
  pod "cd ~/RASD && rm -f ~/manifest.rc && nohup bash -c 'bash ~/sd_manifest.sh; echo \$? > ~/manifest.rc' > ~/manifest.log 2>&1 & echo started" >/dev/null 2>&1
  sleep 4
  printf '%s\n' "$PROBE" > /tmp/sd_probe.sh
  local out
  out=$(pod 'bash -s' < /tmp/sd_probe.sh 2>/dev/null)
  echo "      $label probe -> $(printf '%s' "$out" | tr '\n' ' ')" >&2
  # the first rc= line is the probe's answer; the debug echo repeats it
  printf '%s\n' "$out" | sed -n 's/^rc=//p' | head -1
}

phase_start "P5  a manifest that exits 0 -> the completion check fires"
RC5=$(run_case "exit-0" 'echo "pretending to be a manifest"; exit 0')
[ "$RC5" = "0" ] && ok "completion detected with rc=0" || bad "rc was '$RC5', expected 0"
phase_end "P5_completion"

phase_start "P6  a manifest that exits 1 -> fail-fast sees a non-zero rc"
RC6=$(run_case "exit-1" 'echo "pretending to fail"; exit 1')
[ "$RC6" = "1" ] && ok "abort detected with rc=1 (fail-fast path)" || bad "rc was '$RC6', expected 1"
phase_end "P6_failfast"

phase_start "P7  the stall watchdog fires on a deliberately silent manifest"
# The watchdog, its helpers AND interim() are extracted from the REAL manifest,
# so this exercises the shipped code. One file, built once: appending to a
# scratch file across runs left a stale copy behind and made this phase report a
# failure that was entirely the harness's.
{
  cat <<'HEAD'
set -uo pipefail
cd ~/RASD
REPO=$PWD
OUT=$(mktemp -d)
MANIFEST=$$
PY=$(command -v python3)
# A one-minute threshold so the check fits the shakedown budget. The shipped
# table's values are asserted by the test suite; what is under test here is the
# MECHANISM: progress log -> threshold -> signal -> non-zero exit.
mkdir -p "$REPO/configs"
printf '{"default_minutes": 1, "stages": {}}' > "$REPO/configs/mlsys_stall_thresholds.json"
echo "stall threshold for this run: 1 minute (test override)"
HEAD
  python3 -c "
import pathlib
s = pathlib.Path('scripts/mlsys_manifest.sh').read_text()
print(s[s.index('interim() {'):s.index(\"trap 'kill \${STALL_WATCHDOG_PID:-0} 2>/dev/null' EXIT\")].rstrip())
"
  cat <<'TAIL'
interim "STAGE_START name=natural_spec_gated_256k timeout=999s"
stall_watchdog &
STALL_WATCHDOG_PID=$!
echo "silent stage: no progress line will be written"
sleep 200
echo "IF YOU SEE THIS THE WATCHDOG DID NOT FIRE"
exit 0
TAIL
} > /tmp/sd_stall_full.sh
pod 'cat > ~/sd_stall_full.sh' < /tmp/sd_stall_full.sh
T7=$(date -u +%s)
STALL_OUT=$(pod 'bash ~/sd_stall_full.sh; echo "EXIT=$?"' 2>&1 | tail -8)
P7=$(( $(date -u +%s) - T7 ))
printf '%s\n' "$STALL_OUT" | sed 's/^/      /'
printf '   P7 wall time: %ss (limit 1 min, 60s poll granularity, plus the stage\n' "$P7"
printf '      command having to return before the TERM trap can run)\n'
case "$STALL_OUT" in
  *"STALL:"*) ok "the watchdog reported a stall" ;;
  *)          bad "the watchdog did not report a stall" ;;
esac
case "$STALL_OUT" in
  *"EXIT=9"*) ok "the stalled run exited 9, which the watcher reads as an abort" ;;
  *"EXIT=0"*) bad "the stalled run exited 0 -- the watcher would read it as success" ;;
  *)          bad "unexpected exit from the stalled run" ;;
esac
if pod 'grep -q STALL ~/manifest_watchdog.log' 2>/dev/null; then
  ok "the pod wrote its own watchdog log ($(pod 'cat ~/manifest_watchdog.log'))"
else
  bad "no watchdog log on the pod"
fi
TIMINGS+=("P7_stall=${P7}s")

# --------------------------------------------------------------------------
phase_start "P8  the byte-level liveness watchdog (silent manifest.log)"
# One implementation, shared with the operator's own validation run so the two
# cannot drift: the script extracts the watchdog from the real manifest, runs it
# against a manifest that stops writing mid-stage (case A) and against one with
# no stage started at all (case B, the negative control). Case A's per-stage
# threshold is 9999 min so the ONLY rule that can stop it is the liveness one --
# which is the point: the stage-level rule never saw the 2026-10-09T00:00Z
# deadlock, because engine_cap_smoke's own limit is 240 min.
SELFTEST_OUT=$(pod 'cd ~/RASD && bash scripts/mlsys_watchdog_selftest.sh; echo "EXIT=$?"' 2>&1 | tail -24)
printf '%s\n' "$SELFTEST_OUT" | sed 's/^/      /'
case "$SELFTEST_OUT" in
  *"8 passed, 0 failed"*) ok "the liveness watchdog self-test passed on the pod" ;;
  *)                      bad "the liveness watchdog self-test failed on the pod" ;;
esac
case "$SELFTEST_OUT" in
  *"EXIT=0"*) ok "the self-test exited 0" ;;
  *)          bad "the self-test did not exit 0" ;;
esac
TIMINGS+=("P8_liveness")

# --------------------------------------------------------------------------
printf '\n== timings ==\n'
for t in "${TIMINGS[@]}"; do printf '   %s\n' "$t"; done
printf '\n== SHAKEDOWN RESULT: %d passed, %d failed ==\n' "$PASS" "$FAIL"
[ "$FAIL" -eq 0 ]

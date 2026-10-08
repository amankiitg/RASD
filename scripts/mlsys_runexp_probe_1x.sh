#!/usr/bin/env bash
# ONE real run_experiment.py invocation on a single GPU, in the same command form
# the manifest uses, on a pod the setup probe has already provisioned.
#
# Usage:  bash scripts/mlsys_runexp_probe_1x.sh <pod-ip>
#
# WHERE THIS FITS
#   1. launch gpu_1x_a100_sxm4
#   2. MLSYS_HF_TOKEN=<token> bash scripts/mlsys_setup_probe.sh <ip>
#      -> stages the repo and runs the WATCHER'S OWN provisioning, so the pod
#         gets the real interpreter, the real pins and the real .pod_env.sh
#   3. bash scripts/mlsys_runexp_probe_1x.sh <ip>      <-- this script
#   4. pull the CSV and the log home, then terminate
#
# WHY IT REFUSES INSTEAD OF FALLING BACK
# The campaign lost a launch to an interpreter resolver that returned nothing and
# silently fell back to a bare python3 with no transformers. If .pod_env.sh is
# not on the pod, this script says so and stops; it never guesses an interpreter.
#
# WHAT IS ASSERTED, because "the process exited 0" is not the claim under test:
#   * the wandb failure that stopped engine_cap_smoke does not appear at all --
#     WANDB_MODE=disabled is in .pod_env.sh and init_wandb degrades gracefully,
#     and this is the first real check that either of them works;
#   * every planned row is present with status=ok;
#   * tokens were actually generated and at least one round ran, i.e. a forward
#     pass happened rather than a plan being printed.
set -uo pipefail

IP=${1:?usage: mlsys_runexp_probe_1x.sh <pod-ip>}
SSH_USER=${2:-ubuntu}
SSH_KEY=${MLSYS_SSH_KEY:-$HOME/.ssh/id_ed25519}
SSH_OPTS="-o StrictHostKeyChecking=no -o ConnectTimeout=25 -i $SSH_KEY"
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"

CFG=configs/mlsys_runexp_probe_1x.yml
# The LOCAL python used to read the pulled CSV. Not the pod's: the pod
# interpreter is resolved separately below and must be the campaign env.
MLSYS_LOCAL_PY=${MLSYS_LOCAL_PY:-python3}
# Quoted so the tilde survives to the REMOTE shell. Unquoted, bash expands it
# here and the pod is asked to mkdir /Users/<this laptop>/RASD, which it cannot
# do -- and the failure reads as a pod problem.
POD_OUT='~/RASD/results/mlsys/probe_1x'
POD_CSV=$POD_OUT/runexp_probe.csv
POD_LOG=$POD_OUT/runexp_probe.log
# NOT `GROUPS`. That name is a special READONLY bash array holding the caller's
# group ids, so `GROUPS="PROBE_SPEC PROBE_TARGET"` is silently ignored and
# expands to a number -- the run then plans zero rows for a group called "20".
PROBE_GROUPS="PROBE_SPEC PROBE_TARGET"

# THE CAMPAIGN'S ENVIRONMENT, not just its command form.
# The watcher starts the manifest with HF_TOKEN in its environment and the
# manifest's stages inherit it; every target model in this campaign is a gated
# repo, so without it the first model load 401s. A probe that ran the right
# COMMAND with the wrong ENVIRONMENT reported a GatedRepoError that the campaign
# would never have hit -- a false finding, which is worse than no probe. Read by
# label, never printed.
HF_TOKEN_VALUE=$(grep -E '^[[:space:]]*HF_TOKEN[[:space:]]*=' runpod_creds.md 2>/dev/null \
                 | head -1 | sed -E 's/^[^=]*=[[:space:]]*//' | tr -d '"'"'"' \r')
if [ -z "$HF_TOKEN_VALUE" ]; then
  echo "FATAL: no HF_TOKEN line found in runpod_creds.md (looked up by label)." >&2
  exit 2
fi

PASS=0; FAIL=0
ok()  { printf '  PASS  %s\n' "$*"; PASS=$((PASS+1)); }
bad() { printf '  FAIL  %s\n' "$*"; FAIL=$((FAIL+1)); }
pod() { ssh $SSH_OPTS "$SSH_USER@$IP" "$@"; }

printf '\n== 0. stage THIS working tree, and prove the pod is running it ==\n'
# A probe tests the code it is given. The first version of this script assumed
# the setup probe's earlier rsync was current, so a fix made afterwards was
# never exercised and the probe reported the SAME failure twice -- a false
# negative that looked like the fix not working.
if rsync -az --no-perms --no-owner --no-group -e "ssh $SSH_OPTS" \
     --exclude '.git' --exclude 'results/final' --exclude 'manuscript' \
     --exclude '.venv*' --exclude '__pycache__' \
     "$REPO/" "$SSH_USER@$IP:~/RASD/" >/tmp/probe_rsync.log 2>&1; then
  ok "working tree staged to the pod"
else
  bad "rsync failed"; tail -3 /tmp/probe_rsync.log
fi
LOCAL_SHA=$(shasum -a 256 run_experiment.py | awk '{print $1}')
POD_SHA=$(pod "sha256sum ~/RASD/run_experiment.py 2>/dev/null | cut -d' ' -f1" 2>/dev/null)
if [ "$LOCAL_SHA" = "$POD_SHA" ]; then
  ok "the pod's run_experiment.py is byte-identical to this working tree"
else
  bad "the pod is running a DIFFERENT run_experiment.py (local $LOCAL_SHA, pod ${POD_SHA:-unreadable}): this probe would be testing old code"
  exit 2
fi

printf '\n== 1. the pod holds the campaign environment (no fallback) ==\n'
if pod 'test -f ~/RASD/.pod_env.sh'; then
  ok "~/RASD/.pod_env.sh present"
else
  bad "~/RASD/.pod_env.sh is absent: run scripts/mlsys_setup_probe.sh first. Refusing to guess an interpreter."
  exit 2
fi
if pod 'grep -q "^export WANDB_MODE=disabled$" ~/RASD/.pod_env.sh'; then
  ok ".pod_env.sh disables wandb (the setting that stopped engine_cap_smoke)"
else
  bad ".pod_env.sh does not set WANDB_MODE=disabled; wandb will ask for a key"
fi
# Resolve the interpreter the way the CAMPAIGN does. `.pod_env.sh` does not
# activate conda -- it exports cache, NCCL and PYTHONPATH settings -- so
# `command -v python` inside it finds /usr/bin/python, which is the silent
# fallback this project has already paid for twice. provisioning writes the
# resolved path to ~/pod_env.log as INTERPRETER=, and the manifest reads it from
# there; this reads the same line.
PY_REMOTE=$(pod 'grep -a "^INTERPRETER=" ~/pod_env.log 2>/dev/null | tail -1 | cut -d= -f2-' 2>/dev/null)
case "$PY_REMOTE" in
  */envs/rasd-gpu/bin/python) ok "resolved the campaign interpreter: $PY_REMOTE" ;;
  "") bad "no INTERPRETER= line in ~/pod_env.log; run scripts/mlsys_setup_probe.sh first"; exit 2 ;;
  *)  bad "resolved '$PY_REMOTE', which is not the campaign env"; exit 2 ;;
esac
if pod "test -x $PY_REMOTE"; then
  ok "the resolved interpreter exists and is executable"
else
  bad "$PY_REMOTE is not an executable on the pod"; exit 2
fi

printf '\n== 2. the command, in the manifest form (plus --nproc 1) ==\n'
printf '  HF token: present (%s chars, not shown)\n' "$(printf '%s' "$HF_TOKEN_VALUE" | wc -c | tr -d ' ')"
# The manifest's form, verbatim in its arguments; see the config header for why
# --nproc 1 is the single unavoidable difference.
RUNNER="cd ~/RASD && set -a && . ~/RASD/.pod_env.sh && set +a && \
HF_TOKEN='$HF_TOKEN_VALUE' MLSYS_PYTHON='$PY_REMOTE' mkdir -p $POD_OUT && \
# The previous attempt's CSV is deleted from the POD only, and only
# because it was already pulled home (step 3 archives it first). A stage
# refuses to write into a file another stage's rows are in -- the same
# guard the manifest honours by archiving an attempt before a retry --
# so leaving it would make a re-run fail for a reason that is not the
# code under test.
rm -f $POD_CSV && \
HF_TOKEN='$HF_TOKEN_VALUE' MLSYS_PYTHON='$PY_REMOTE' \
$PY_REMOTE run_experiment.py --config $CFG --groups $PROBE_GROUPS \
  --output $POD_CSV --stage-id probe_1x --nproc 1 \
  --timeout-per-run-s 900 --abort-on-failure \
  --log-per-token --memory-trace --save-generated-text --save-generated-tokens \
  > $POD_LOG 2>&1"
echo "  $ $PY_REMOTE run_experiment.py --config $CFG --groups $PROBE_GROUPS --nproc 1 ..."
T=$(date -u +%s)
pod "$RUNNER"
RC=$?
WALL=$(( $(date -u +%s) - T ))
if [ "$RC" = "0" ]; then
  ok "run_experiment exited 0 (${WALL}s wall)"
else
  bad "run_experiment exited $RC after ${WALL}s"
fi

printf '\n== 3. pull the artifacts home and assert on them LOCALLY ==\n'
# Pulled rather than inspected over ssh: a multi-line heredoc inside an ssh
# argument is exactly the kind of nested quoting that has already produced three
# false probe results in this project, and the CSV has to come home anyway.
LOCAL_OUT=${MLSYS_PROBE_LOCAL_OUT:-results/mlsys/probe_1x}
mkdir -p "$LOCAL_OUT"
# Archive rather than overwrite: a failed probe attempt is evidence about what
# the pod did, and the manifest's own rule is that an earlier attempt is moved
# aside instead of deleted.
if [ -f "$LOCAL_OUT/runexp_probe.csv" ]; then
  mkdir -p "$LOCAL_OUT/attempts"
  STAMP=$(date -u +%Y%m%dT%H%M%SZ)
  mv "$LOCAL_OUT/runexp_probe.csv" "$LOCAL_OUT/attempts/runexp_probe.$STAMP.csv"
  [ -f "$LOCAL_OUT/runexp_probe.log" ] && \
    mv "$LOCAL_OUT/runexp_probe.log" "$LOCAL_OUT/attempts/runexp_probe.$STAMP.log"
  ok "archived the previous attempt to attempts/runexp_probe.$STAMP.*"
fi
if scp -q $SSH_OPTS "$SSH_USER@$IP:$POD_CSV" "$LOCAL_OUT/runexp_probe.csv" 2>/dev/null; then
  ok "pulled the CSV to $LOCAL_OUT/runexp_probe.csv"
else
  bad "could not pull $POD_CSV"
fi
if scp -q $SSH_OPTS "$SSH_USER@$IP:$POD_LOG" "$LOCAL_OUT/runexp_probe.log" 2>/dev/null; then
  ok "pulled the log to $LOCAL_OUT/runexp_probe.log"
else
  bad "could not pull $POD_LOG"
fi

printf '\n== 4. the wandb failure that stopped engine_cap_smoke ==\n'
LOG="$LOCAL_OUT/runexp_probe.log"
if [ -f "$LOG" ]; then
  if grep -q 'No API key configured' "$LOG"; then
    bad "the credential error is BACK: $(grep -m1 'No API key configured' "$LOG")"
  else
    ok "no 'No API key configured' in the log"
  fi
  if grep -q 'wandb init failed' "$LOG"; then
    printf '  note  wandb still failed to initialise, and the run continued:\n'
    grep -m2 'wandb init failed' "$LOG" | sed 's/^/        /'
  else
    ok "wandb initialised with no failure at all (WANDB_MODE=disabled)"
  fi
  printf '  --- last 6 lines ---\n'
  tail -6 "$LOG" | sed 's/^/        /'
else
  bad "no log pulled, so the wandb question is unanswered"
fi

printf '\n== 5. the CSV row the stage exists to write ==\n'
CSV="$LOCAL_OUT/runexp_probe.csv"
if [ -s "$CSV" ]; then
  SUMMARY=$("$MLSYS_LOCAL_PY" - "$CSV" <<'PYCSV'
import csv, sys
rows = list(csv.DictReader(open(sys.argv[1])))
n = lambda k: sum(int(float(r.get(k) or 0)) for r in rows)
print("rows=%d" % len(rows))
print("ok=%d" % sum(1 for r in rows if r.get("status") == "ok"))
print("tok=%d" % n("tokens_generated"))
print("rounds=%d" % n("n_rounds"))
for r in rows:
    print("row %s status=%s tokens=%s rounds=%s acc=%s" % (
        r.get("run_id"), r.get("status"), r.get("tokens_generated"),
        r.get("n_rounds"), r.get("acceptance_rate")))
PYCSV
)
  printf '%s\n' "$SUMMARY" | sed 's/^/        /'
  rows=$(printf '%s\n' "$SUMMARY" | sed -n 's/^rows=//p')
  oks=$(printf '%s\n' "$SUMMARY" | sed -n 's/^ok=//p')
  tok=$(printf '%s\n' "$SUMMARY" | sed -n 's/^tok=//p')
  rnd=$(printf '%s\n' "$SUMMARY" | sed -n 's/^rounds=//p')
  [ "${rows:-0}" -ge 3 ] && ok "the canary plus both planned rows are present" \
                         || bad "expected 3 rows (canary + spec + target-only), got ${rows:-0}"
  [ "${oks:-0}" = "${rows:-1}" ] && ok "every row has status=ok" \
                                 || bad "${oks:-0} of ${rows:-0} rows have status=ok"
  [ "${tok:-0}" -gt 0 ] && ok "tokens were generated (${tok} total): a forward pass happened" \
                        || bad "no tokens generated: nothing was decoded"
  [ "${rnd:-0}" -gt 0 ] && ok "decode rounds ran (${rnd} total)" \
                        || bad "no rounds ran"
else
  bad "no CSV pulled, so no row was written"
fi

printf '\n== 5. the rows, for the record ==\n'
printf '  CSV on the pod: %s\n' "$POD_CSV"
printf '  log on the pod: %s\n' "$POD_LOG"

printf '\n== PROBE RESULT: %d passed, %d failed ==\n' "$PASS" "$FAIL"
[ "$FAIL" -eq 0 ] || exit 1
echo "run_experiment works end to end on one GPU through the real provisioning."

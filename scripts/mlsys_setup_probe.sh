#!/usr/bin/env bash
# Run the WATCHER'S OWN setup commands against a pod, without the watcher.
#
# Why this exists: the full shakedown re-implemented the setup steps, so it
# passed while the watcher's versions of those steps were broken. Two 8xA100
# launches were lost that way (a missing HF_TOKEN in the provisioning command,
# then an interpreter resolver that returned nothing). This probe extracts the
# commands FROM scripts/mlsys_watch_and_run.sh and runs them, so what is tested
# is what will run.
#
# Usage:  MLSYS_HF_TOKEN=<token> bash scripts/mlsys_setup_probe.sh <pod-ip>
set -uo pipefail

IP=${1:?usage: mlsys_setup_probe.sh <pod-ip>}
SSH_USER=${2:-ubuntu}
SSH_KEY=${MLSYS_SSH_KEY:-$HOME/.ssh/id_ed25519}
SSH_OPTS="-o StrictHostKeyChecking=no -o ConnectTimeout=25 -i $SSH_KEY"
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"
WATCH=scripts/mlsys_watch_and_run.sh
HFT=${MLSYS_HF_TOKEN:?MLSYS_HF_TOKEN must be set}

PASS=0; FAIL=0
ok()  { printf '  PASS  %s\n' "$*"; PASS=$((PASS+1)); }
bad() { printf '  FAIL  %s\n' "$*"; FAIL=$((FAIL+1)); }
pod() { ssh $SSH_OPTS "$SSH_USER@$IP" "$@"; }

# --- 1. the staging command, verbatim -------------------------------------
printf '\n== staging (the watcher rsync verbatim) ==\n'
if rsync -az --no-perms --no-owner --no-group -e "ssh $SSH_OPTS" \
     --exclude '.git' --exclude 'results/final' --exclude 'manuscript' \
     --exclude '.venv*' --exclude '__pycache__' \
     "$REPO/" "$SSH_USER@$IP:~/RASD/" >/tmp/probe_rsync.log 2>&1; then
  ok "repo staged"
else
  bad "rsync failed"; tail -3 /tmp/probe_rsync.log
fi

# --- 2. the provisioning command, EXTRACTED from the watcher ---------------
printf '\n== provisioning (extracted from the watcher, token included) ==\n'
PROV_INNER=$(python3 - <<'PYPROV'
import pathlib
s = pathlib.Path("scripts/mlsys_watch_and_run.sh").read_text()
i = s.index('"cd ~/RASD && HF_TOKEN=')
# Include the redirect: slicing UP TO it drops where the log goes, so the
# provisioning runs with its output on stdout and ~/pod_env.log is never
# created -- which the resolution step then reads. Third harness bug of this
# shape; the extractors are now asserted, not eyeballed.
_end = s.index('> ~/pod_env.log 2>&1', i) + len('> ~/pod_env.log 2>&1')
seg = s[i:_end]
# the literal text the watcher sends, with the token placeholder substituted
print(seg[1:].replace("$HF_TOKEN_VALUE", "HFT_PLACEHOLDER"))
PYPROV
)
if [ -z "$PROV_INNER" ]; then
  bad "could not extract the provisioning command"
else
  echo "  extracted: ${PROV_INNER:0:80}..."
  case "$PROV_INNER" in
    *HFT_PLACEHOLDER*) echo "  token placeholder present: yes" ;;
    *)                 echo "  token placeholder present: NO (the token is not wired in)" ;;
  esac
  # The extractors are the weak link: three bugs of that shape in one session, each
# of which made the probe report a watcher failure that did not exist (or hide a
# real one). So the extraction is asserted here, not eyeballed.
case "$PROV_INNER" in
  *mlsys_pod_env.sh*) ;;
  *) bad "the extracted provisioning command does not run mlsys_pod_env.sh"; ;;
esac
case "$PROV_INNER" in
  *pod_env.log*) ;;
  *) bad "the extracted provisioning command has no log redirect" ;;
esac
case "$PROV_INNER" in
  *HF_TOKEN=*) ;;
  *) bad "the extracted provisioning command has no token" ;;
esac
INNER=${PROV_INNER//HFT_PLACEHOLDER/$HFT}
  T=$(date -u +%s)
  if pod "cd ~/RASD && $INNER"; then
    ok "provisioning exited 0 ($(($(date -u +%s)-T))s wall)"
  else
    bad "provisioning FAILED"
    pod 'tail -20 ~/pod_env.log' | sed 's/^/        /'
  fi
fi

# --- 3. the resolution command, EXTRACTED ---------------------------------
printf '\n== interpreter resolution (extracted from the watcher) ==\n'
RESOLVE_PRIMARY=$(python3 - <<'PYRES'
import pathlib
s = pathlib.Path("scripts/mlsys_watch_and_run.sh").read_text()
i = s.index('PY_REMOTE=$(ssh $SSH_OPTS "$SSH_USER@$IP" \\')
# The terminator is quote-space-2>/dev/null), NOT the first "2>/dev/null)" --
# the quoted command contains that string itself, and stopping there leaves a
# dangling quote that makes the remote shell fail. Which is exactly what the
# first run of this probe reported.
seg = s[i:s.index("' 2>/dev/null)", i)]
inner = seg[seg.index("'") + 1:]
assert not inner.endswith("'"), "extraction left a dangling quote"
print(inner)
PYRES
)
echo "  extracted: ${RESOLVE_PRIMARY:0:80}..."
PY_REMOTE=$(pod "$RESOLVE_PRIMARY" 2>/dev/null)
case "$PY_REMOTE" in
  /*) ok "resolved an absolute path: $PY_REMOTE" ;;
  "") bad "resolution returned NOTHING (this is what cost a launch)" ;;
  *)  bad "resolved '$PY_REMOTE', which is not an absolute path" ;;
esac
if [ -n "$PY_REMOTE" ] && pod "test -x '$PY_REMOTE'"; then
  ok "the resolved interpreter exists and is executable"
else
  bad "the resolved interpreter is not executable on the pod"
fi

# --- 4. the preflight, EXTRACTED -----------------------------------------
printf '\n== stage-dependency preflight (extracted from the watcher) ==\n'
PREFLIGHT_PY=$(python3 - <<'PYFLY'
import pathlib
s = pathlib.Path("scripts/mlsys_watch_and_run.sh").read_text()
i = s.index('"$RPY -c \\"import torch, transformers, bitsandbytes, flash_attn, diptest')
seg = s[i:s.index('2>&1 | tee -a "$LOG"', i)]
inner = seg[seg.index('\\"') + 2:seg.rindex('\\"')]
print(" ".join(inner.replace('\\"', '"').replace("\\\n", " ").split()))
PYFLY
)
if [ -z "$PY_REMOTE" ]; then
  bad "no interpreter to preflight with"
else
  if pod "cd ~/RASD && '$PY_REMOTE' -c \"$PREFLIGHT_PY\""; then
    ok "torch/transformers/bitsandbytes/flash_attn/diptest import; CUDA visible"
  else
    bad "the stage dependencies do not import"
  fi
fi

# --- 5. the manifest environment ----------------------------------------
printf '\n== the manifest environment (what the manifest run inherits) ==\n'
if pod 'test -f ~/RASD/.pod_env.sh' && pod 'grep -q HF_HOME ~/RASD/.pod_env.sh'; then
  ok "~/.pod_env.sh written with the cache/env settings"
else
  bad "~/.pod_env.sh is missing or incomplete"
fi

printf '\n== SETUP PROBE RESULT: %d passed, %d failed ==\n' "$PASS" "$FAIL"
[ "$FAIL" -eq 0 ]

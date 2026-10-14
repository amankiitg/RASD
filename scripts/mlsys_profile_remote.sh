#!/usr/bin/env bash
# POD side of the Phase 3 profile. Started DETACHED by scripts/mlsys_profile_1x.sh
# (setsid, own log), and it writes its OWN DONE marker so the Mac side never has to
# treat an ssh return as the end of the work (lesson L9: an ssh return is not
# evidence that remote work ended -- on 2026-10-11 it terminated a pod mid-stage).
#
# WHY A WRAPPER AND NOT THE DECODE SCRIPT DIRECTLY. Two reasons, both about the rc:
#   1. `mlsys_profile_decode.sh` ends with `echo`, so its exit status is echo's. A
#      session's honest rc has to be derived from the ARTIFACT: the CSV must exist and
#      must hold at least one ok row (lesson L14: verify the blob, not the intention).
#   2. The Mac side needs a marker whose WRITE proves the trap fired on every path,
#      including the early refusals below.
#
# Usage, from the driver (all three are env vars, so the exact command line can be
# rehearsed with tiny settings):
#   PROFILE_TAG=a PROFILE_NPROC=1 PROFILE_OUT=results/mlsys/profile \
#     bash scripts/mlsys_profile_remote.sh
set -uo pipefail
cd "$HOME/RASD" || exit 3
OUT=${PROFILE_OUT:-results/mlsys/profile}
TAG=${PROFILE_TAG:-a}
NPROC=${PROFILE_NPROC:-1}
mkdir -p "$OUT"
PODDONE="$OUT/DONE"

remote_finish() {
  local rc=$?
  printf '%s %s\n' "$(date -u +%FT%TZ)" "$rc" > "$PODDONE" 2>/dev/null
  printf '%s %s\n' "$(date -u +%FT%TZ)" "=== PROFILE REMOTE EXIT rc=$rc; DONE marker written ===" \
    >> "$OUT/profile.log" 2>/dev/null
}
trap remote_finish EXIT

log() { printf '%s %s\n' "$(date -u +%FT%TZ)" "$*" | tee -a "$OUT/profile.log"; }

# HF TOKEN FROM A FILE, never from a command line (it used to travel in argv and
# therefore in `ps` on both machines).
if [ -s "$HOME/.hf_token" ]; then
  HF_TOKEN=$(cat "$HOME/.hf_token")
  export HF_TOKEN
  log "HF token: read from ~/.hf_token ($(wc -c < "$HOME/.hf_token" | tr -d ' ') bytes)"
else
  log "WARN: no ~/.hf_token; model downloads may fail if the repos are gated"
fi

PY=$(grep -a "^INTERPRETER=" "$HOME/pod_env.log" 2>/dev/null | tail -1 | cut -d= -f2-)
if [ -z "${PY:-}" ] || [ ! -x "${PY:-}" ]; then
  log "FATAL: no usable INTERPRETER= in ~/pod_env.log; refusing to guess one"
  exit 2
fi
# shellcheck disable=SC1090
set -a; . "$HOME/RASD/.pod_env.sh"; set +a
log "=== profile tag=$TAG nproc=$NPROC out=$OUT interpreter=$PY ==="
"$PY" -c 'import torch,transformers;print("torch",torch.__version__,"transformers",transformers.__version__,"gpus",torch.cuda.device_count())' 2>&1 | tee -a "$OUT/profile.log"

# ---- THE PARAMETER CONTRACT, BEFORE THE MEASUREMENT -------------------------
# Same check as the other pod-side scripts: what arrived must be what the driver
# intended, verified before the measurement rather than inferred from it afterwards.
# KEYS ARE THE DRIVER'S VARIABLE NAMES on both sides, and each falls back to the
# literal `<absent>` when it never crossed the boundary, so a dropped setting reads
# as "never received" rather than as the default this script silently fell back to.
# shellcheck disable=SC1090
. scripts/mlsys_remote_expect.sh
mlsys_expect "profile-remote" "${REMOTE_EXPECT:-}" \
  "PROFILE_OUT=${PROFILE_OUT-<absent>};PROFILE_TAG=${PROFILE_TAG-<absent>};PROFILE_NPROC=${PROFILE_NPROC-<absent>}" || exit 2

# THE MEASUREMENT. One script, one parameter changed between profile (a) and (b).
bash scripts/mlsys_profile_decode.sh --nproc "$NPROC" --tag "$TAG" --out "$OUT" >> "$OUT/decode_${TAG}_nproc${NPROC}.log" 2>&1
drc=$?
log "### decode script rc=$drc (see decode_${TAG}_nproc${NPROC}.log)"

# THE ARTIFACT DECIDES. rc alone is not a result signal.
ok=$("$PY" - "$OUT/profile_${TAG}_nproc${NPROC}.csv" <<'PYOK' 2>/dev/null
import csv, os, sys
p = sys.argv[1]
if not os.path.exists(p):
    print(0); raise SystemExit
print(sum(1 for r in csv.DictReader(open(p)) if r.get('status') == 'ok'))
PYOK
)
sides=$(ls "$OUT"/profiler/*.json 2>/dev/null | wc -l | tr -d ' ')
log "### profile ok rows=$ok profiler sidecars=$sides"
if [ "${ok:-0}" -lt 1 ]; then
  log "FAILED: no ok profile row; this profile produced no measurement"
  exit 1
fi
if [ "${sides:-0}" -lt 1 ]; then
  log "FAILED: no profiler sidecar; --profile did not write one, so there is nothing to attribute time with"
  exit 1
fi
log "=== PROFILE REMOTE DONE (single device: profile is (a) if nproc=1, (b) if nproc=2) ==="

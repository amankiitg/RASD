#!/usr/bin/env bash
# POD side of every paid vLLM run, rehearsal and the 8x comparison alike. Started
# through scripts/mlsys_profile_1x.sh (the driver the Phase 3 profile validated,
# with REMOTE_SCRIPT pointed here), which supplies the detached dead-man timer, the
# 600-mode token file, the DONE-marker protocol and API-verified termination.
#
# ONE SCRIPT, ONE PARAMETER, the same reason the profile has one script for both
# nproc values: the rehearsal and the real run must differ ONLY in the things that
# are named here, or "the rehearsal proves the 8x run works" stops being true.
#
# DEFAULTS = THE 1x REHEARSAL, which is what the handoff asked for and what ran on
# 2026-10-12: install vLLM exactly as the 8x run would
# (`scripts/mlsys_vllm_venv.sh`, the campaign interpreter as the venv base), load
# `gradientai/Llama-3-8B-Instruct-Gradient-1048k` at 8k, decode 128 tokens greedily,
# report the tok/s. TP=1 there, so the row is deliberately NOT unit_matched -- that
# requires 8 GPUs and the RASD prompt ids, which is the 8x run's job.
#
# THE 8x COMPARISON IS THE SAME SCRIPT WITH FOUR ENV VARS:
#   REHEARSE_TP=8
#   REHEARSE_CTXS='131072 524288 1048576'
#   REHEARSE_PROMPT_DIR=results/mlsys/vllm_prompts   (original sidecars, see the plan)
#   out: PROFILE_OUT=results/mlsys/vllm_run
#
# THE RC COMES FROM THE ARTIFACT: a CSV that exists and holds at least one ok row.
# `unit_matched=no` is EXPECTED here and must not fail the rehearsal, so the
# script is explicit about which of the two it is checking.
set -uo pipefail
cd "$HOME/RASD" || exit 3
OUT=${PROFILE_OUT:-results/mlsys/vllm_rehearsal}
# PROFILE_TAG is reported in the parameter contract below; it was read nowhere before,
# so `set -u` made the contract itself the first thing to notice it was unset.
TAG=${PROFILE_TAG:-}
NPROC=${PROFILE_NPROC:-1}
mkdir -p "$OUT"
PODDONE="$OUT/DONE"

remote_finish() {
  local rc=$?
  printf '%s %s\n' "$(date -u +%FT%TZ)" "$rc" > "$PODDONE" 2>/dev/null
  printf '%s %s\n' "$(date -u +%FT%TZ)" "=== VLLM RUN EXIT rc=$rc; DONE marker written ===" \
    >> "$OUT/rehearsal.log" 2>/dev/null
}
trap remote_finish EXIT

log() { printf '%s %s\n' "$(date -u +%FT%TZ)" "$*" | tee -a "$OUT/rehearsal.log"; }

if [ -s "$HOME/.hf_token" ]; then
  HF_TOKEN=$(cat "$HOME/.hf_token")
  export HF_TOKEN
  log "HF token: read from ~/.hf_token ($(wc -c < "$HOME/.hf_token" | tr -d ' ') bytes)"
else
  log "FATAL: no ~/.hf_token; gradientai/... and meta-llama/... are gated"
  exit 2
fi

PY=$(grep -a "^INTERPRETER=" "$HOME/pod_env.log" 2>/dev/null | tail -1 | cut -d= -f2-)
if [ -z "${PY:-}" ] || [ ! -x "${PY:-}" ]; then
  log "FATAL: no usable INTERPRETER= in ~/pod_env.log"
  exit 2
fi
log "=== vLLM rehearsal: campaign interpreter $PY, out $OUT ==="
"$PY" -c 'import torch,transformers;print("campaign env: torch",torch.__version__,"transformers",transformers.__version__)' 2>&1 | tee -a "$OUT/rehearsal.log"

MODEL=${REHEARSE_MODEL:-gradientai/Llama-3-8B-Instruct-Gradient-1048k}
REV=${REHEARSE_REV:-cd3069b65a8eb13da639d332a5f61b0fbb29fa73}
CTX=${REHEARSE_CTX:-8192}
CTXS=${REHEARSE_CTXS:-$CTX}
NEW=${REHEARSE_NEW:-128}
TP=${REHEARSE_TP:-1}
PROMPT_DIR=${REHEARSE_PROMPT_DIR:-}
CSV="$OUT/vllm_baseline.csv"

# ---- 1. the install, EXACTLY as the 8x run would ----------------------------
# The venv is built from the campaign's own interpreter, which is the pin
# mlsys_vllm_venv.sh documents: the two then share a Python ABI, and `python3 -m
# venv`'s dependence on the distro's python3-venv package (absent on many images)
# never enters the picture.
# ---- THE PARAMETER CONTRACT, BEFORE THE HEAVY WORK --------------------------
# This is the check that would have caught the 2026-10-12 8x run: it installed vLLM,
# loaded the model and decoded at 8192 tokens on ONE card because the driver never
# forwarded REHEARSE_TP/CTXS/PROMPT_DIR, and nothing here could tell that the pod was
# running defaults. It runs before the venv install and before any model load.
# shellcheck disable=SC1090
. scripts/mlsys_remote_expect.sh
# KEYS ARE THE DRIVER'S VARIABLE NAMES on both sides. The driver forwards
# PROFILE_OUT/PROFILE_TAG/PROFILE_NPROC and whatever REMOTE_ENV_NAMES names, and that
# is exactly what it can assert; reporting the same names here means a key present in
# the manifest but absent from "received" is a variable that did not survive the
# boundary, which is the failure being checked for. The wrapper's own defaults
# (REHEARSE_MODEL, REHEARSE_REV, REHEARSE_CTX, REHEARSE_NEW) are printed but only
# asserted when the driver declares them. A setting that never arrived is reported as
# `<absent>` rather than as the wrapper default this script fell back to: on
# 2026-10-12 the wrapper's defaults WERE the problem (TP=1, ctx=8192), so printing
# them as "received" would have hidden the very thing being verified.
mlsys_expect "vllm-remote" "${REMOTE_EXPECT:-}" \
  "PROFILE_OUT=${PROFILE_OUT-<absent>};PROFILE_TAG=${PROFILE_TAG-<absent>};PROFILE_NPROC=${PROFILE_NPROC-<absent>};REHEARSE_TP=${REHEARSE_TP-<absent>};REHEARSE_CTXS=${REHEARSE_CTXS-<absent>};REHEARSE_PROMPT_DIR=${REHEARSE_PROMPT_DIR-<absent>};REHEARSE_MODEL=${REHEARSE_MODEL-<absent>};REHEARSE_REV=${REHEARSE_REV-<absent>};REHEARSE_CTX=${REHEARSE_CTX-<absent>};REHEARSE_NEW=${REHEARSE_NEW-<absent>}" || exit 2

log "### installing the vLLM venv (this is the 8x run's own install path)"
MLSYS_VLLM_BASE_PYTHON="$PY" bash scripts/mlsys_vllm_venv.sh 2>&1 | tee -a "$OUT/rehearsal.log"
venv_rc=${PIPESTATUS[0]}
log "### venv rc=$venv_rc"
VENV="${MLSYS_VLLM_VENV:-$HOME/RASD/.venv-vllm}"
# A NON-ZERO INSTALL RC IS FATAL EVEN IF AN INTERPRETER EXISTS. The installer
# returns 2 (no base interpreter), 3 (venv creation failed), 4 (torch install
# failed) and 5 (vllm install failed); in the last two a python binary IS present
# and only the packages are missing, so continuing would surface the failure much
# later as an unrelated `import vllm` error. The rehearsal exists to report "the
# install path works", so it must stop where the install stopped.
if [ "$venv_rc" != "0" ]; then
  log "FAILED: the vLLM install returned rc=$venv_rc (2=no base interpreter, 3=venv creation, 4=torch, 5=vllm)"
  exit 1
fi
if [ ! -x "$VENV/bin/python" ]; then
  log "FAILED: no interpreter at $VENV/bin/python; the install did not complete"
  exit 1
fi

# ---- 2. load + decode, one row ---------------------------------------------
# THE PROMPT IDS. Without a directory the stage builds a SYNTHETIC prompt of the
# right length and the row can never be unit_matched. `load_rasd_cells` refuses a
# directory that yields no cell (the committed slim sidecars are exactly that
# case), so a wrong path fails here rather than after the comparison's wall time.
#
# WHY THIS IS A SCALAR AND NOT AN ARRAY. macOS ships bash 3.2, where
# `"${arr[@]}"` on an EMPTY array under `set -u` is an unbound-variable error --
# this script failed there on the first local run, and every other campaign script
# avoids arrays for the same reason. The path is therefore checked for spaces
# rather than silently relied on not to have any.
PROMPT_ARG=""
if [ -n "$PROMPT_DIR" ]; then
  case "$PROMPT_DIR" in
    *" "*) log "FATAL: REHEARSE_PROMPT_DIR must not contain spaces (bash 3.2 cannot expand an optional array)"; exit 2 ;;
  esac
  PROMPT_ARG="--prompt-ids-from-sidecars $PROMPT_DIR"
  log "### prompt ids from $PROMPT_DIR (original RASD sidecars, engine_input_ids required)"
fi
log "### running the baseline: $MODEL @ [$CTXS] tokens, TP=$TP, $NEW new tokens, revision $REV"
env MLSYS_VLLM_PYTHON="$VENV/bin/python" \
  "$VENV/bin/python" scripts/mlsys_vllm_baseline.py \
    --models "$MODEL" \
    --context-lengths $CTXS \
    --max-new-tokens "$NEW" \
    --matched-max-new-tokens "$NEW" \
    --tensor-parallel-size "$TP" \
    --target-revision "$REV" \
    $PROMPT_ARG \
    --out "$CSV" 2>&1 | tee -a "$OUT/rehearsal.log"
run_rc=${PIPESTATUS[0]}
log "### baseline rc=$run_rc"

# ---- 3. the verdict, from the artifact --------------------------------------
if [ ! -s "$CSV" ]; then
  log "FAILED: no CSV at $CSV; the run produced no row at all"
  exit 1
fi
"$PY" - "$CSV" <<'PYSUM' 2>&1 | tee -a "$OUT/rehearsal.log"
import csv, sys
rows = list(csv.DictReader(open(sys.argv[1])))
ok = [r for r in rows if r.get("status") == "ok"]
print(f"### rows={len(rows)} ok={len(ok)}")
for r in rows:
    print(f"    {r.get('model')} @ {r.get('context_length')} max_model_len={r.get('max_model_len')} "
          f"TP={r.get('tensor_parallel_size')} attempt={r.get('attempt')} "
          f"config={r.get('config_used')} status={r.get('status')} "
          f"end_to_end_s={r.get('end_to_end_wall_s')} out_tokens={r.get('output_tokens')} "
          f"tps_e2e={r.get('throughput_tps_end_to_end')} "
          f"tps_decode={r.get('throughput_tps_decode_only')} "
          f"kv_dtype={r.get('kv_dtype')} weights={r.get('weight_precision')} "
          f"unit_matched={r.get('unit_matched')}")
    if r.get("status") != "ok":
        print(f"      error_class={r.get('error_class')} error={str(r.get('error'))[:400]}")
print("NOTE: at TP=1 and/or without a prompt-ids directory, unit_matched=no is "
      "EXPECTED and is not a failure -- it requires TP=8 and the RASD ids.")
sys.exit(0 if ok else 1)
PYSUM
sum_rc=${PIPESTATUS[0]}
if [ "$sum_rc" != "0" ]; then
  log "FAILED: no ok row in $CSV"
  exit 1
fi
log "=== VLLM RUN DONE: the engine installed, loaded and decoded ==="

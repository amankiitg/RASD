#!/usr/bin/env bash
# POD side of the Phase 5 vLLM REHEARSAL. Runs on ONE 1x A100 through
# scripts/mlsys_profile_1x.sh (the same driver the Phase 3 profile used, with
# REMOTE_SCRIPT pointed here), which supplies the detached dead-man timer, the
# 600-mode token file, the DONE-marker protocol and API-verified termination.
#
# WHAT THIS REHEARSES, per the handoff: install vLLM EXACTLY as the 8x run would
# (`scripts/mlsys_vllm_venv.sh`, the campaign's own interpreter as the venv base),
# load `gradientai/Llama-3-8B-Instruct-Gradient-1048k` at 8k context, decode 128
# tokens greedily, and report the tok/s and whether it ran cleanly. TP=1 on one
# card, so the row is deliberately NOT unit_matched -- that is the 8x run's job and
# requires 8 GPUs. The rehearsal's question is "does the install, the load, the
# rope_theta, the tokenizer and the decode path work at all", not "is it fair".
#
# THE RC COMES FROM THE ARTIFACT: a CSV that exists and holds at least one ok row.
# `unit_matched=no` is EXPECTED here and must not fail the rehearsal, so the
# script is explicit about which of the two it is checking.
set -uo pipefail
cd "$HOME/RASD" || exit 3
OUT=${PROFILE_OUT:-results/mlsys/vllm_rehearsal}
mkdir -p "$OUT"
PODDONE="$OUT/DONE"

remote_finish() {
  local rc=$?
  printf '%s %s\n' "$(date -u +%FT%TZ)" "$rc" > "$PODDONE" 2>/dev/null
  printf '%s %s\n' "$(date -u +%FT%TZ)" "=== VLLM REHEARSAL EXIT rc=$rc; DONE marker written ===" \
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
NEW=${REHEARSE_NEW:-128}
CSV="$OUT/vllm_baseline.csv"

# ---- 1. the install, EXACTLY as the 8x run would ----------------------------
# The venv is built from the campaign's own interpreter, which is the pin
# mlsys_vllm_venv.sh documents: the two then share a Python ABI, and `python3 -m
# venv`'s dependence on the distro's python3-venv package (absent on many images)
# never enters the picture.
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
log "### running the baseline: $MODEL @ $CTX tokens, TP=1, $NEW new tokens, revision $REV"
env MLSYS_VLLM_PYTHON="$VENV/bin/python" \
  "$VENV/bin/python" scripts/mlsys_vllm_baseline.py \
    --models "$MODEL" \
    --context-lengths "$CTX" \
    --max-new-tokens "$NEW" \
    --matched-max-new-tokens "$NEW" \
    --tensor-parallel-size 1 \
    --target-revision "$REV" \
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
print("NOTE: unit_matched=no is EXPECTED here -- it requires TP=8 and the RASD "
      "prompt ids, and this rehearsal is one card by design.")
sys.exit(0 if ok else 1)
PYSUM
sum_rc=${PIPESTATUS[0]}
if [ "$sum_rc" != "0" ]; then
  log "FAILED: no ok row in $CSV"
  exit 1
fi
log "=== VLLM REHEARSAL DONE: the engine installed, loaded and decoded ==="

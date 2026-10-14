#!/usr/bin/env bash
# POD side of the DECODE-DIAGNOSIS FOLLOW-UP (Part 1).
#
# THE QUESTION. A 128k target-only decode step on 8x A100 costs 3.3-3.7 s. The Phase C
# profile puts 73.4% of the decode window in NCCL SendRecv over 256 rotation rounds, i.e.
# ~9.3 ms per round for a 266 MB payload -> ~29 GB/s. That is 5-10x below A100 NVLink line
# rate, and it has two possible causes with opposite fixes:
#
#   (1) TRANSPORT — NCCL picked a slow path (SHM, PCIe, or NET) instead of P2P/IPC over
#       NVLink. An env var fixes it, and the microbenchmark shows the fix.
#   (2) PER-ROUND OVERHEAD / RANK SKEW — the bytes are not the problem; the round costs
#       ~9.3 ms whatever it carries. Then only removing rounds (split-KV) or shrinking
#       per-round work helps.
#
# So this script measures the primitive on its own (stage 2) and the engine on the same box
# (stages 3-4), and compares them.
#
# STAGES, AND WHY THIS ORDER. Every stage is independent: a failure is logged and the next
# stage still runs. That is deliberate -- the Phase D session lost its config-only arm
# because a CUDA fault in an earlier stage made the chain-stop skip everything after it.
# Only two gates exist, both recorded as marker files: the split-KV arm is not attempted at
# 128k unless its 8k smoke passed, and ppl is not attempted unless split-KV ran at 128k.
#
#   0 topo    nvidia-smi topo -m and the GPU inventory
#   1 bench   the bare batch_isend_irecv ring (no model), 256 rounds x 5 payloads
#   2 smoke   split-KV at 8k under CUDA_LAUNCH_BLOCKING=1 -- a synchronous traceback, or a row
#   3 arms    the engine: 4 arms x {target-only, spec} at 128k, ONE model load, token streams
#   4 ppl     1024-token target ppl per arm (prefill path -- see the caveat in the log)
#
# Usage, from the driver (all env vars, so the exact command line can be rehearsed):
#   DF_OUT=results/mlsys/decode_df DF_CTX_ARMS=131072 DF_CTX_SMOKE=8192 DF_BOOK=pg19_train_0 \
#   DF_ARMS=ring2048,ring32768,ring32768x,splitkv DF_SEED=42 DF_BENCH_ROUNDS=256 \
#   DF_TARGET_TOKENS=16 DF_SPEC_TOKENS=32 NPROC=8 bash scripts/mlsys_df_remote.sh
set -uo pipefail
cd "$HOME/RASD" || exit 3

OUT=${DF_OUT:-results/mlsys/decode_df}
CTX_ARMS=${DF_CTX_ARMS:-131072}
CTX_SMOKE=${DF_CTX_SMOKE:-8192}
BOOK=${DF_BOOK:-pg19_train_0}
ARMS=${DF_ARMS:-ring2048,ring32768,ring32768x,splitkv}
SEED=${DF_SEED:-42}
ROUNDS=${DF_BENCH_ROUNDS:-256}
TARGET_TOKENS=${DF_TARGET_TOKENS:-16}
SPEC_TOKENS=${DF_SPEC_TOKENS:-32}
NPROC=${NPROC:-8}
# A 1x PLUMBING REHEARSAL ONLY. `install_ring_attention` is a no-op at world_size<=1, so
# without this the arms script refuses to run (correctly: every arm would be the same run).
# Explicit and off by default, and it is in the parameter contract, so a real 8x session can
# never be silently downgraded to a one-rank run whose rows look plausible.
SINGLE_RANK_OK=${DF_SINGLE_RANK_OK:-0}
SR_FLAG=""
[ "$SINGLE_RANK_OK" = "1" ] && SR_FLAG="--single-rank-ok"
# The greedy pair is the decode-path correctness gate for a non-bit-identical arm: with T=0
# there is no sampling, so two correct engines must agree token-for-token and a divergence
# is structural rather than a near-tie. On by default for the measured run; the rehearsal
# sets it off to keep its own runtime down.
GREEDY_FLAG="--greedy-check"
[ "${DF_GREEDY_CHECK:-1}" = "0" ] && GREEDY_FLAG=""
mkdir -p "$OUT" "$OUT/ring_bench"
PODDONE="$OUT/DONE"

log() { printf '%s %s\n' "$(date -u +%FT%TZ)" "$*" | tee -a "$OUT/df.log"; }
remote_finish() {
  local rc=$?
  printf '%s %s\n' "$(date -u +%FT%TZ)" "$rc" > "$PODDONE" 2>/dev/null
  log "=== DF REMOTE EXIT rc=$rc; marker written ==="
}
trap remote_finish EXIT

if [ -s "$HOME/.hf_token" ]; then
  HF_TOKEN=$(cat "$HOME/.hf_token"); export HF_TOKEN
  log "HF token: read from ~/.hf_token ($(wc -c < "$HOME/.hf_token" | tr -d ' ') bytes)"
else
  log "WARN: no ~/.hf_token; gated repos will fail"
fi
PY=$(grep -a "^INTERPRETER=" "$HOME/pod_env.log" 2>/dev/null | tail -1 | cut -d= -f2-)
if [ -z "${PY:-}" ] || [ ! -x "${PY:-}" ]; then
  log "FATAL: no usable INTERPRETER= in ~/pod_env.log"; exit 2
fi
# shellcheck disable=SC1090
set -a; . "$HOME/RASD/.pod_env.sh"; set +a
log "=== df remote: interpreter $PY nproc=$NPROC ctx_arms=$CTX_ARMS ctx_smoke=$CTX_SMOKE book=$BOOK arms=$ARMS seed=$SEED out=$OUT ==="

# ---- THE PARAMETER CONTRACT, BEFORE ANY HEAVY WORK --------------------------
# shellcheck disable=SC1090
. scripts/mlsys_remote_expect.sh
# THE RECEIVED STRING MUST NAME EVERY KEY THE DRIVER'S MANIFEST NAMES. The first version of
# this call listed only the DF_* settings, so the contract correctly refused to start over
# six variables it had no report for -- the pod had received them, it just had not said so.
# A contract that cannot see an arrival is as bad as a missing arrival: it stops a correct
# run. The keys are the driver's variable names on both sides.
#
# STAGES is DERIVED here from the config with the same script the driver uses
# (mlsys_stage_list.py), rather than passed: two derivations of "which stages" is how a
# campaign starts disagreeing with itself about what it ran.
STEPS_ONELINE=""
if [ -n "${SESSION_CFG:-}" ] && [ -f "${SESSION_CFG:-}" ]; then
  STEPS_ONELINE=$("$PY" scripts/mlsys_stage_list.py "$SESSION_CFG" --space 2>/dev/null \
                  | sed -E 's/ +$//')
fi
log "stages from $SESSION_CFG: $STEPS_ONELINE"
mlsys_expect "df-remote" "${REMOTE_EXPECT:-}" \
  "SESSION_OUT=${SESSION_OUT-<absent>};SESSION_CFG=${SESSION_CFG-<absent>};SESSION_SANITY_CFG=${SESSION_SANITY_CFG-<absent>};NPROC=${NPROC-<absent>};SESSION_CHECK_STAGE=${SESSION_CHECK_STAGE-};SESSION_GATE_CANDIDATES=${SESSION_GATE_CANDIDATES-};STAGES=$STEPS_ONELINE;CODE_COMMIT=${CODE_COMMIT-<absent>};DF_OUT=${DF_OUT-<absent>};DF_CTX_ARMS=${DF_CTX_ARMS-<absent>};DF_CTX_SMOKE=${DF_CTX_SMOKE-<absent>};DF_BOOK=${DF_BOOK-<absent>};DF_ARMS=${DF_ARMS-<absent>};DF_SEED=${DF_SEED-<absent>};DF_BENCH_ROUNDS=${DF_BENCH_ROUNDS-<absent>};DF_TARGET_TOKENS=${DF_TARGET_TOKENS-<absent>};DF_SPEC_TOKENS=${DF_SPEC_TOKENS-<absent>};DF_SINGLE_RANK_OK=${DF_SINGLE_RANK_OK-<absent>}" || exit 2

# DF_OUT must be the session directory: the driver pulls ~/RASD/$SESSION_OUT, so anything
# written outside it is lost to the pull (2026-10-12: an entire rehearsal's evidence).
if [ "${DF_OUT:-}" != "${SESSION_OUT:-}" ]; then
  log "FATAL: DF_OUT='${DF_OUT:-<unset>}' but SESSION_OUT='${SESSION_OUT:-<unset>}'. The driver pulls ~/RASD/\$SESSION_OUT, so evidence written elsewhere never leaves the pod."
  exit 2
fi
# The commit the staged tree came from. The pod has no .git (its ~/RASD is an rsync of the
# working tree), so without this every row records "unknown" for the engine provenance --
# which is what the Part 1 rows did until this line existed.
log "CODE_COMMIT=$CODE_COMMIT (the tree this pod is running)"

"$PY" -c 'import torch;print("torch",torch.__version__,"cuda",torch.cuda.device_count(),
                                torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu")' \
  2>&1 | tee -a "$OUT/df.log"

# =============================== STAGE 0: TOPO ===============================
{
  echo "### nvidia-smi topo -m"
  nvidia-smi topo -m 2>&1 || echo "nvidia-smi topo failed"
  echo
  echo "### gpus"
  nvidia-smi --query-gpu=index,name,memory.total,pcie.link.gen.current,pcie.link.width.current \
             --format=csv 2>&1 || true
  echo
  echo "### nvlink"
  nvidia-smi nvlink --status 2>&1 | head -40 || echo "nvlink query failed"
} > "$OUT/topo.txt" 2>&1
log "### stage0 topo written ($(wc -l < "$OUT/topo.txt") lines)"

# ============================= STAGE 1: MICROBENCH ===========================
# The bare primitive. 256 rounds is exactly one decode token's worth of rotation rounds, so
# `ms_per_round x 256` is directly comparable to the profile's decode-window NCCL time.
log "### stage1 bench: $ROUNDS rounds x 5 payloads, $NPROC ranks"
"$PY" -m torch.distributed.run --nproc_per_node="$NPROC" --master_port=29511 \
  scripts/mlsys_ring_microbench.py --rounds "$ROUNDS" --warmup 10 \
  --out "$OUT/ring_bench" --tag ringbench >> "$OUT/bench.log" 2>&1
bench_rc=$?
log "### stage1 bench rc=$bench_rc"
grep -h "^\[ringbench\]" "$OUT/bench.log" 2>/dev/null | tee -a "$OUT/df.log" || true

# ============================== STAGE 2: SMOKE ===============================
# CUDA_LAUNCH_BLOCKING=1 makes the fault synchronous, so the traceback names the kernel that
# faulted instead of the next API call (which is how the Phase D traceback pointed at an NF4
# dequant that could not have been responsible). NCCL_DEBUG=INFO with SUBSYS=INIT,P2P is on
# here as well because this is the process that initialises the process group: it is where
# the transport choice is made and logged, and it costs nothing to capture on a run we are
# doing anyway.
if [ -f "$OUT/smoke/cmd_rc" ]; then rm -f "$OUT/smoke/cmd_rc"; fi
mkdir -p "$OUT/smoke"
log "### stage2 smoke: split-KV @ $CTX_SMOKE, CUDA_LAUNCH_BLOCKING=1, NCCL_DEBUG=INIT,P2P"
CUDA_LAUNCH_BLOCKING=1 NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=INIT,P2P \
"$PY" -m torch.distributed.run --nproc_per_node="$NPROC" --master_port=29512 \
  scripts/mlsys_df_arms.py --mode smoke --ctx "$CTX_SMOKE" \
  --arms ring2048,splitkv --book "$BOOK" --seed "$SEED" \
  --target-tokens 8 $SR_FLAG --out "$OUT/smoke" >> "$OUT/smoke.log" 2>&1
smoke_rc=$?
echo "$smoke_rc" > "$OUT/smoke/cmd_rc"
log "### stage2 smoke rc=$smoke_rc"
grep -h "^\[df\]" "$OUT/smoke.log" 2>/dev/null | tee -a "$OUT/df.log" || true

# The transport evidence, extracted from THIS process's NCCL debug stream.
# WRITTEN VIA A TEMP FILE: the first version appended a block that began by `cat`-ting the
# output file into itself, which is an error and left the first section empty.
NCCL_RAW="$OUT/nccl_transport_raw.txt"
grep -aE "NCCL INFO (Channel|P2P|NET|SHM|Setting affinity|NVLS|Bootstrap|Connected|via )" \
     "$OUT/smoke.log" 2>/dev/null | sort -u | head -80 > "$NCCL_RAW" || true
{
  echo "### distinct transport/init lines (NCCL_DEBUG=INIT,P2P), from the smoke process"
  if [ -s "$NCCL_RAW" ]; then cat "$NCCL_RAW"; else echo "(none -- see smoke.log)"; fi
  echo
  echo "### every distinct 'via ' phrase with its count"
  grep -aoE "via [A-Za-z0-9/_. :-]*" "$OUT/smoke.log" 2>/dev/null | sort | uniq -c | sort -rn | head -20 || true
  echo
  echo "### transport keyword counts in smoke.log"
  for pat in NVLink "P2P/IPC" SHM "NET/" "shared memory" nvidia_peermem; do
    n=$(grep -ac -- "$pat" "$OUT/smoke.log" 2>/dev/null || true)
    printf '%-16s %s\n' "$pat" "${n:-0}"
  done
} > "$OUT/nccl_transport.txt" 2>&1
log "### stage2 transport evidence written to $OUT/nccl_transport.txt ($(wc -l < "$OUT/nccl_transport.txt") lines)"

# ============================== STAGE 3: ARMS ================================
# The measurement. One model load, all arms, target-only + spec, unprofiled.
ARM_ARMS="$ARMS"
if [ ! -f "$OUT/smoke/splitkv_ok.marker" ]; then
  ARM_ARMS=$(printf '%s' "$ARMS" | tr ',' '\n' | grep -v '^splitkv$' | paste -sd, -)
  log "### stage3: split-KV smoke did NOT pass -> dropping splitkv from the arms ($ARM_ARMS)"
fi
log "### stage3 arms @ $CTX_ARMS: $ARM_ARMS"
"$PY" -m torch.distributed.run --nproc_per_node="$NPROC" --master_port=29513 \
  scripts/mlsys_df_arms.py --mode arms --ctx "$CTX_ARMS" \
  --arms "$ARM_ARMS" --book "$BOOK" --seed "$SEED" \
  --target-tokens "$TARGET_TOKENS" --spec-tokens "$SPEC_TOKENS" --out "$OUT/arms" \
  $SR_FLAG $GREEDY_FLAG >> "$OUT/arms.log" 2>&1
arms_rc=$?
log "### stage3 arms rc=$arms_rc"
grep -h "^\[df\]" "$OUT/arms.log" 2>/dev/null | tee -a "$OUT/df.log" || true

# =============================== STAGE 4: PPL ================================
# Only if split-KV actually produced a 128k row: for a bit-identical arm this check is
# redundant, and it costs a prefill pass per arm.
split_ok=0
if [ -f "$OUT/arms/df_arms.csv" ]; then
  split_ok=$("$PY" -c "
import csv,sys
try:
    rows=list(csv.DictReader(open('$OUT/arms/df_arms.csv')))
except Exception:
    print(0); sys.exit()
print(sum(1 for r in rows if r.get('arm')=='splitkv' and r.get('status')=='ok'))
")
fi
if [ "$split_ok" -gt 0 ]; then
  log "### stage4 ppl: split-KV has $split_ok ok row(s) -> running the ppl check for every arm"
  "$PY" -m torch.distributed.run --nproc_per_node="$NPROC" --master_port=29514 \
    scripts/mlsys_df_arms.py --mode ppl --ctx "$CTX_ARMS" \
    --arms "$ARM_ARMS" --book "$BOOK" --seed "$SEED" --out "$OUT/ppl" \
    $SR_FLAG >> "$OUT/ppl.log" 2>&1
  log "### stage4 ppl rc=$?"
  grep -h "^\[df\]" "$OUT/ppl.log" 2>/dev/null | tee -a "$OUT/df.log" || true
else
  log "### stage4 ppl SKIPPED: split-KV produced no ok row at 128k (split_ok=$split_ok)"
fi

log "### df remote done; evidence under $OUT"
exit 0

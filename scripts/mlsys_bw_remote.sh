#!/usr/bin/env bash
# POD side of the TORCH_NCCL_BLOCKING_WAIT diagnostic (results/lcfm/BLOCKING_WAIT_DIAG.md).
#
# THE QUESTION. Every campaign run exported TORCH_NCCL_BLOCKING_WAIT=1, and in torch 2.5.1
# that flag makes WorkNCCL::synchronizeInternal poll with a fixed 10 ms sleep
# (kSynchronizeBusyWaitMillis) until the work's CUDA event reports completion. A bare ring
# round measured ~10.31 ms for 4 KB and ~10.35 ms for 266 MB. That flatness is the shape a
# fixed poll produces and is NOT the shape NVLink produces. This session measures both arms
# on one box and lets the rows decide.
#
# TWO ARMS, ONE SESSION, AND WHY THEY MUST BE TWO PROCESS LAUNCHES. ProcessGroupNCCL reads
# blockingWait_ ONCE, in its constructor (line 826) and copies it into every Work, so no
# in-process switch exists and a stage cannot change arms halfway. Each stage below is its
# own `torch.distributed.run`, and the arm is applied with `env` / `env -u` AFTER
# .pod_env.sh is sourced, so the sourced file cannot override the arm (that `set -a`
# override is exactly why this variable was not per-run controllable before).
#
# STAGES, IN PRIORITY ORDER (a failure is logged and the next stage still runs):
#   A0/A1 topo + bench   the bare ring, both timings, both arms
#   B0/B1 arms @128k     ring2048, ring32768, splitkv x {target-only, spec}, both arms
#   C0/C1 1M target-only one row per arm, if the budget allows (BW_1M_OK=1)
#
# WHY THE BENCH RUNS TWICE PER ARM. Under the async arm, `r.wait()` on a point-to-point work
# returns after ENQUEUEING a stream dependency and does not wait for the transfer at all
# (ProcessGroupNCCL::synchronizeInternal does nothing CPU-side when blockingWait_ is false and
# the work has no barrier tensor). Timing only `wait` would therefore show the async arm as
# ~100x faster for a reason that is an artefact of the instrument. `--timing both` emits
# `wait` (what the engine's loop pays) and `sync` (the transfer itself, whatever the mode).
#
# Usage, from the driver (all env vars, so the exact command line can be rehearsed):
#   BW_OUT=results/mlsys/bw_diag BW_BOOK=pg19_train_0 BW_SEED=42 BW_ROUNDS=256 \
#   BW_CTX=131072 BW_CTX_1M=1048576 BW_TARGET_TOKENS=16 BW_SPEC_TOKENS=32 BW_1M_OK=0 \
#   BW_ENV_ARMS="blocking async" BW_SINGLE_RANK_OK=0 \
#   BW_ARMS=ring2048,ring32768,splitkv NPROC=8 bash scripts/mlsys_bw_remote.sh
set -uo pipefail
cd "$HOME/RASD" || exit 3

OUT=${BW_OUT:-results/mlsys/bw_diag}
BOOK=${BW_BOOK:-pg19_train_0}
SEED=${BW_SEED:-42}
ROUNDS=${BW_ROUNDS:-256}
CTX=${BW_CTX:-131072}
CTX_1M=${BW_CTX_1M:-1048576}
ARMS=${BW_ARMS:-ring2048,ring32768,splitkv}
TARGET_TOKENS=${BW_TARGET_TOKENS:-16}
SPEC_TOKENS=${BW_SPEC_TOKENS:-32}
ONE_M_OK=${BW_1M_OK:-0}
# A 1x PLUMBING REHEARSAL ONLY. `install_ring_attention` is a no-op at world_size <= 1, so
# without this the arms driver refuses to run -- correctly, because every arm would be the
# same run and the rows would look plausible while measuring nothing. Explicit, off by
# default, and in the parameter contract, so a real 8x session can never be silently
# downgraded to a one-rank run.
SINGLE_RANK_OK=${BW_SINGLE_RANK_OK:-0}
SR_FLAG=""
[ "$SINGLE_RANK_OK" = "1" ] && SR_FLAG="--single-rank-ok"
NPROC=${NPROC:-8}
# The arms, in order. `blocking` is the historical configuration under investigation;
# `async` is the replacement. Both names appear in every row's env_arm column.
ENV_ARMS=${BW_ENV_ARMS:-"blocking async"}

PODDONE="$OUT/DONE"

log() { printf '%s %s\n' "$(date -u +%FT%TZ)" "$*" | tee -a "$OUT/bw.log"; }
remote_finish() {
  local rc=$?
  printf '%s %s\n' "$(date -u +%FT%TZ)" "$rc" > "$PODDONE" 2>/dev/null
  log "=== BW REMOTE EXIT rc=$rc; marker written ==="
}
trap remote_finish EXIT

mkdir -p "$OUT" "$OUT/bench" "$OUT/arms"

if [ -s "$HOME/.hf_token" ]; then
  HF_TOKEN=$(cat "$HOME/.hf_token"); export HF_TOKEN
  log "HF token: read from ~/.hf_token ($(wc -c < "$HOME/.hf_token" | tr -d ' ') bytes)"
else
  log "FATAL: no ~/.hf_token; the models are gated and every arm would fail at load"
  exit 2
fi

PY=$(grep -a "^INTERPRETER=" "$HOME/pod_env.log" 2>/dev/null | tail -1 | cut -d= -f2-)
if [ -z "${PY:-}" ] || [ ! -x "${PY:-}" ]; then
  log "FATAL: no usable INTERPRETER= in ~/pod_env.log; refusing to guess one"
  exit 2
fi
# shellcheck disable=SC1090
set -a; . "$HOME/RASD/.pod_env.sh"; set +a
log "=== bw remote: interpreter $PY nproc=$NPROC arms='$ARMS' env_arms='$ENV_ARMS' ctx=$CTX book=$BOOK seed=$SEED rounds=$ROUNDS out=$OUT ==="
log "=== .pod_env.sh as sourced: $(grep -E 'TORCH_NCCL|NCCL_TIMEOUT|MLSYS_NCCL_WAIT_MODE' "$HOME/RASD/.pod_env.sh" | tr '\n' ' ')"

# ---- THE PARAMETER CONTRACT, BEFORE ANY HEAVY WORK --------------------------
# shellcheck disable=SC1090
. scripts/mlsys_remote_expect.sh
STEPS_ONELINE=""
if [ -n "${SESSION_CFG:-}" ] && [ -f "${SESSION_CFG:-}" ]; then
  STEPS_ONELINE=$("$PY" scripts/mlsys_stage_list.py "$SESSION_CFG" --space 2>/dev/null \
                  | sed -E 's/ +$//')
fi
mlsys_expect "bw-remote" "${REMOTE_EXPECT:-}" \
  "SESSION_OUT=${SESSION_OUT-<absent>};SESSION_CFG=${SESSION_CFG-<absent>};SESSION_SANITY_CFG=${SESSION_SANITY_CFG-<absent>};NPROC=${NPROC-<absent>};SESSION_CHECK_STAGE=${SESSION_CHECK_STAGE-};SESSION_GATE_CANDIDATES=${SESSION_GATE_CANDIDATES-};STAGES=$STEPS_ONELINE;CODE_COMMIT=${CODE_COMMIT-<absent>};BW_OUT=${BW_OUT-<absent>};BW_BOOK=${BW_BOOK-<absent>};BW_SEED=${BW_SEED-<absent>};BW_ROUNDS=${BW_ROUNDS-<absent>};BW_CTX=${BW_CTX-<absent>};BW_CTX_1M=${BW_CTX_1M-<absent>};BW_ARMS=${BW_ARMS-<absent>};BW_TARGET_TOKENS=${BW_TARGET_TOKENS-<absent>};BW_SPEC_TOKENS=${BW_SPEC_TOKENS-<absent>};BW_1M_OK=${BW_1M_OK-<absent>};BW_ENV_ARMS=${BW_ENV_ARMS-<absent>};BW_SINGLE_RANK_OK=${BW_SINGLE_RANK_OK-<absent>}" \
  || exit 2

if [ "${BW_OUT:-}" != "${SESSION_OUT:-}" ]; then
  log "FATAL: BW_OUT='${BW_OUT:-<unset>}' but SESSION_OUT='${SESSION_OUT:-<unset>}'. The driver pulls ~/RASD/\$SESSION_OUT, so evidence written elsewhere never leaves the pod."
  exit 2
fi

# ---- THE ARM ENVIRONMENT, PROVEN RATHER THAN INTENDED -----------------------
# `env_arm_env <arm>` echoes the `env ...` prefix for that arm. TWO properties matter:
#   * `blocking` must SET the flag, and `async` must UNSET it (`env -u`), because the flag
#     may still be present from .pod_env.sh on a pod provisioned before today;
#   * both arms keep the 1-hour timeouts, since a 1M prefill legitimately runs for tens of
#     minutes and they are the hang budget, not the thing under test.
env_arm_env() {
  case "$1" in
    blocking)
      # `env -u TORCH_NCCL_ASYNC_ERROR_HANDLING` as well: with both set torch uses blocking
      # and silently disables async handling, so leaving it set would give the same behaviour
      # but a row that is ambiguous at a glance. One mode knob per arm.
      printf '%s' "env -u TORCH_NCCL_ASYNC_ERROR_HANDLING TORCH_NCCL_BLOCKING_WAIT=1 MLSYS_NCCL_WAIT_MODE=blocking" ;;
    async)
      printf '%s' "env -u TORCH_NCCL_BLOCKING_WAIT TORCH_NCCL_ASYNC_ERROR_HANDLING=1 MLSYS_NCCL_WAIT_MODE=async" ;;
    *)
      log "FATAL: unknown arm '$1'"; exit 2 ;;
  esac
}

# ============================= STAGE 0: TOPO =================================
log "### stage0 topo"
{
  echo "### nvidia-smi topo -m"
  nvidia-smi topo -m 2>&1 || true
  echo
  echo "### gpus"
  nvidia-smi --query-gpu=index,name,memory.total --format=csv 2>&1 || true
  echo
  echo "### the wait mode this session was provisioned with"
  grep -E 'TORCH_NCCL|NCCL_TIMEOUT|MLSYS_NCCL_WAIT_MODE' "$HOME/RASD/.pod_env.sh" || true
  echo
  echo "### the torch that will be imported, and the poll constant it carries"
  "$PY" - <<'PYEOF' 2>&1 || true
import torch
print("torch", torch.__version__)
import torch.distributed.distributed_c10d as d
print("blocking-wait env name:", getattr(d, "TORCH_NCCL_BLOCKING_WAIT", "n/a"))
PYEOF
} > "$OUT/topo.txt" 2>&1
log "### stage0 topo written ($(wc -l < "$OUT/topo.txt") lines)"

# ====================== STAGES A0/A1 AND B0/B1: TWO ARMS =====================
for ARM in $ENV_ARMS; do
  ARM_ENV=$(env_arm_env "$ARM")
  log "=============================================================="
  log "### ARM $ARM: $ARM_ENV"
  log "=============================================================="

  # ---- stage A: the bare ring, both timings --------------------------------
  log "### stage A[$ARM] bench: $ROUNDS rounds, --timing both"
  # shellcheck disable=SC2086
  $ARM_ENV "$PY" -m torch.distributed.run --nproc_per_node="$NPROC" --master_port=29611 \
    scripts/mlsys_ring_microbench.py --rounds "$ROUNDS" --warmup 10 --timing both \
    --env-arm "$ARM" --out "$OUT/bench" --tag "ringbench_$ARM" >> "$OUT/bench_$ARM.log" 2>&1
  log "### stage A[$ARM] bench rc=$?"
  grep -h "^\[ringbench\]" "$OUT/bench_$ARM.log" 2>/dev/null | tee -a "$OUT/bw.log" || true

  # ---- stage B: the engine, 128k, all arms ---------------------------------
  # ring2048 is the configuration the campaign actually shipped; ring32768 is the chunk
  # variant; splitkv is the rotation-free engine. All three under BOTH wait modes, because
  # the arm that matters for the paper's F4 claim is ring2048 and the arm that matters for
  # F5 is splitkv.
  mkdir -p "$OUT/arms/$ARM"
  log "### stage B[$ARM] arms @ $CTX: $ARMS (target-only + spec)"
  # shellcheck disable=SC2086
  # shellcheck disable=SC2086
  $ARM_ENV "$PY" -m torch.distributed.run --nproc_per_node="$NPROC" --master_port=29612 \
    scripts/mlsys_df_arms.py --mode arms --ctx "$CTX" \
    --arms "$ARMS" --book "$BOOK" --seed "$SEED" \
    --target-tokens "$TARGET_TOKENS" --spec-tokens "$SPEC_TOKENS" \
    --env-arm "$ARM" $SR_FLAG --out "$OUT/arms/$ARM" >> "$OUT/arms_$ARM.log" 2>&1
  log "### stage B[$ARM] arms rc=$?"
  grep -h "^\[df\]" "$OUT/arms_$ARM.log" 2>/dev/null | tail -20 | tee -a "$OUT/bw.log" || true
  # The token digests, logged HERE so the arm-to-arm comparison is visible in the session log
  # and not only in a file that a later reader has to go looking for.
  if [ -f "$OUT/arms/$ARM/token_digests.json" ]; then
    log "### stage B[$ARM] token digests: $(tr -d '\n ' < "$OUT/arms/$ARM/token_digests.json")"
  fi

  # ---- stage C: one 1M target-only row -------------------------------------
  if [ "$ONE_M_OK" = "1" ]; then
    mkdir -p "$OUT/onem/$ARM"
    log "### stage C[$ARM] 1M target-only @ $CTX_1M, $TARGET_TOKENS tokens"
    # shellcheck disable=SC2086
    $ARM_ENV "$PY" -m torch.distributed.run --nproc_per_node="$NPROC" --master_port=29613 \
      scripts/mlsys_df_arms.py --mode arms --ctx "$CTX_1M" \
      --arms ring2048 --book "$BOOK" --seed "$SEED" \
      --target-tokens "$TARGET_TOKENS" --spec-tokens "$SPEC_TOKENS" \
      --env-arm "$ARM" $SR_FLAG --out "$OUT/onem/$ARM" >> "$OUT/onem_$ARM.log" 2>&1
    log "### stage C[$ARM] 1M rc=$?"
    grep -h "^\[df\]" "$OUT/onem_$ARM.log" 2>/dev/null | tail -5 | tee -a "$OUT/bw.log" || true
  else
    log "### stage C[$ARM] SKIPPED (BW_1M_OK=$ONE_M_OK)"
  fi
done

log "=== all arms done ==="
ls -la "$OUT"/arms/*/session* 2>/dev/null | tee -a "$OUT/bw.log" || true

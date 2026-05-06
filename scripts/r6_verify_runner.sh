#!/bin/bash
# Post-launch runner for R6 verification.
# Run on a Lambda 8x A100 instance after rsync + dep install.
#
# Validates two things:
#   1. Option B fix (no draft RoPE scaling): ctx=64k × W=8 NF4 sync should
#      now fit in 40GB (was 35.7 GB OOM before, expected ~25 GB after).
#   2. A4 async characterization: prefetch_depth=1 across max_new ∈ {16,32,64}
#      × seed ∈ {42,123} with full stderr captured to disk per run.
#
# Usage (on the instance):
#   bash scripts/r6_verify_runner.sh /home/ubuntu/RASD/results/r6_verify
set -u
OUT_DIR="${1:-/home/ubuntu/RASD/results/r6_verify}"
mkdir -p "$OUT_DIR"
cd /home/ubuntu/RASD

# HF_TOKEN must already be exported by the caller (do NOT hardcode secrets).
# Set it inline in the SSH command that invokes this runner, e.g.:
#   ssh ... "export HF_TOKEN=<token>; bash scripts/r6_verify_runner.sh"
if [ -z "${HF_TOKEN:-}" ]; then
    echo "ERROR: HF_TOKEN not set. Export it before invoking this runner." >&2
    exit 2
fi
export PYTHONPATH=/home/ubuntu/RASD
export TORCH_NCCL_BLOCKING_WAIT=1

LOG="$OUT_DIR/runner.log"
echo "=== R6 verify runner starting $(date) ===" | tee "$LOG"

run_smoke() {
    local NAME="$1"; shift
    local LOGFILE="$OUT_DIR/${NAME}.log"
    echo "" | tee -a "$LOG"
    echo "--- ${NAME} starting $(date) ---" | tee -a "$LOG"
    echo "  cmd: torchrun ... $@" | tee -a "$LOG"
    # Capture FULL output (no grep filter) to per-run log so we don't lose stderr
    timeout 600 torchrun --nproc_per_node=8 --master_port=29530 \
        scripts/r6_smoke.py "$@" \
        --out "$OUT_DIR/${NAME}.json" > "$LOGFILE" 2>&1
    local RC=$?
    if [ $RC -eq 0 ]; then
        echo "  ${NAME}: PASS (exit 0)" | tee -a "$LOG"
        grep -E "alpha|tokens|tps|peak memory" "$LOGFILE" | head -10 | tee -a "$LOG"
    else
        echo "  ${NAME}: FAIL (exit $RC)" | tee -a "$LOG"
        echo "  last 25 lines of stderr:" | tee -a "$LOG"
        tail -25 "$LOGFILE" | tee -a "$LOG"
    fi
}

# === 1. Option B verification: ctx=64k × W=8 NF4 sync ===
# This was OOM in R6.4. Per Option B fix, draft KV at 64k drops from
# 12 GB → 770 MB; per-rank total should drop from 35.7 GB → ~25 GB.
run_smoke "optionB_64k_w8_sync" \
    --context-length 65536 --max-new-tokens 8 --spec-steps 4 \
    --target-quant nf4 --draft-quant nf4 \
    --prefetch-depth 0 --kv-block-size 999999 --seed 42

# === 2. A4 characterization: async (prefetch_depth=1) at multiple max_new × seeds ===
# Sync was solid at all configs. Async is the question. Capture full stderr
# per run to a separate log so we can see actual errors if any fire.
for SEED in 42 123; do
    for MAXNEW in 16 32 64; do
        run_smoke "a4_async_w8_max${MAXNEW}_s${SEED}" \
            --context-length 8192 --max-new-tokens "$MAXNEW" --spec-steps 4 \
            --target-quant nf4 --draft-quant nf4 \
            --prefetch-depth 1 --kv-block-size 999999 --seed "$SEED"
    done
done

echo "" | tee -a "$LOG"
echo "=== R6 verify runner done $(date) ===" | tee -a "$LOG"
echo "Per-run logs in $OUT_DIR/*.log"
echo "Per-run JSON metrics in $OUT_DIR/*.json"
ls -la "$OUT_DIR"

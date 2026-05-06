#!/bin/bash
# Diagnostic runner for the async-ring hang at max_new ≥ 64 W=8.
#
# Pre-condition: code patched on the instance with RING_TIMING=1
# instrumentation in src/models/ring_attention_kernel.py (per-step
# wall-clock prints).
#
# Run on the instance after dep install:
#   bash scripts/r6_async_debug.sh /home/ubuntu/RASD/results/r6_debug
#
# Outputs per-run logs with timing data so we can see where time is sucked.
set -u
OUT_DIR="${1:-/home/ubuntu/RASD/results/r6_debug}"
mkdir -p "$OUT_DIR"
cd /home/ubuntu/RASD

if [ -z "${HF_TOKEN:-}" ]; then
    echo "ERROR: HF_TOKEN not set" >&2
    exit 2
fi
export PYTHONPATH=/home/ubuntu/RASD
export TORCH_NCCL_BLOCKING_WAIT=1

run_with_timing() {
    local NAME="$1"; shift
    local LOG="$OUT_DIR/${NAME}.log"
    echo "" | tee -a "$OUT_DIR/runner.log"
    echo "--- $NAME starting $(date +%H:%M:%S) ---" | tee -a "$OUT_DIR/runner.log"
    timeout 900 \
        env RING_TIMING=1 \
        torchrun --nproc_per_node=8 --master_port=29540 \
        scripts/r6_smoke.py "$@" \
        --out "$OUT_DIR/${NAME}.json" > "$LOG" 2>&1
    local RC=$?
    if [ $RC -eq 0 ]; then
        echo "  $NAME: PASS (exit 0)" | tee -a "$OUT_DIR/runner.log"
    else
        echo "  $NAME: FAIL (exit $RC)" | tee -a "$OUT_DIR/runner.log"
    fi
    # Extract timing summary regardless of pass/fail
    echo "  --- ring step timing (from this run, last 30 timing lines): ---" | tee -a "$OUT_DIR/runner.log"
    grep -E "ring rank=0 step=|ring rank=0 entering|ring rank=0 loop done" "$LOG" | tail -30 | tee -a "$OUT_DIR/runner.log"
}

# Same async config that hung: NF4 8k W=8 prefetch=1
# Test at max_new=8, 16, 32, 48, 64 to find exact failure threshold
for MN in 8 16 32 48 64; do
    run_with_timing "async_w8_max${MN}" \
        --context-length 8192 --max-new-tokens "$MN" --spec-steps 4 \
        --target-quant nf4 --draft-quant nf4 \
        --prefetch-depth 1 --kv-block-size 999999 --seed 42
done

# Compare: same max_new=64 in sync mode (known good)
run_with_timing "sync_w8_max64" \
    --context-length 8192 --max-new-tokens 64 --spec-steps 4 \
    --target-quant nf4 --draft-quant nf4 \
    --prefetch-depth 0 --kv-block-size 999999 --seed 42

echo ""
echo "=== Done ==="
ls -la "$OUT_DIR"

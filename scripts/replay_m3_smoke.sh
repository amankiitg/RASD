#!/usr/bin/env bash
# M3 replay smoke test: runs a representative row from each ablation
# group at the M3 R6.5 production setting (64k ctx, Lambda 40GB SXM4)
# and compares against the **R6.5 committed numbers** in
# results/ablations/ablations_r65.csv — NOT the original RunPod 80GB
# 8k-context CSV (results/ablations/ablations.csv) which was
# invalidated by the 2026-04-16 audit. Replaying against an
# invalidated golden was a real bug discovered 2026-05-10.
#
# Usage (on pod):
#   bash scripts/replay_m3_smoke.sh
#
# Rows replayed (one per group, seed=42):
#   default_canary_s42, A1_sheared_1b_s42, A2_k4_s42,
#   A3_block512_s42, A4_async1_s42, A5_llama2_7b_s42
#
# The script asserts that throughput_tps is within 15% of the
# committed value (tolerance accounts for pod-to-pod hardware variance,
# NOT semantic change).
set -euo pipefail

cd "$(dirname "$0")/.."

SMOKE_CSV="results/ablations/m3_replay_smoke.csv"
GOLDEN="results/ablations/ablations_r65.csv"  # M3 R6.5 (Lambda 40GB) — the latest, validated M3
# 2026-05-10: switched from throughput_tps (15%) to acceptance_rate (±0.05).
# R6.5 ran on Lambda's default image which had no flash-attn → SDPA
# fallback. M4 pods install flash-attn explicitly (~2-3× speedup at
# long ctx) so throughput legitimately differs by 100%+ between R6.5
# and current runs. Acceptance rate is the semantic guardrail —
# hardware-independent, deterministic given same seed + config — so
# regressions in the verify-loop math (Option B / Fix2 / Fix3 / Fix4)
# would still be caught here.
ACC_TOLERANCE=0.05  # absolute tolerance on acceptance_rate

echo "==> Replaying M3 smoke rows to $SMOKE_CSV"
python run_experiment.py \
    --config configs/ablations.yml \
    --output "$SMOKE_CSV" \
    --nproc 8 \
    --seeds 42 \
    --groups A1 A2 A3 A4 A5

echo ""
echo "==> Comparing acceptance_rate against $GOLDEN (±${ACC_TOLERANCE} absolute)"
echo "    (throughput_tps NOT compared — varies with flash-attn / hardware)"
python - <<PY
import csv, sys
golden = {r['run_id']: r for r in csv.DictReader(open('$GOLDEN'))}
smoke  = {r['run_id']: r for r in csv.DictReader(open('$SMOKE_CSV'))}
fail = 0
for rid, s in smoke.items():
    if s.get('status') != 'ok':
        print(f"  SMOKE FAIL {rid}: status={s.get('status')}")
        fail += 1; continue
    g = golden.get(rid)
    if not g:
        print(f"  SKIP {rid}: not in golden")
        continue
    ga, sa = float(g['acceptance_rate']), float(s['acceptance_rate'])
    abs_delta = abs(ga - sa)
    # Throughput shown as informational only (NOT a pass/fail signal)
    gt, st = float(g['throughput_tps']), float(s['throughput_tps'])
    tag = "OK" if abs_delta <= $ACC_TOLERANCE else "REGRESSION"
    if abs_delta > $ACC_TOLERANCE:
        fail += 1
    print(
        f"  {tag:10s} {rid:26s}  α: golden={ga:.3f}  smoke={sa:.3f}  Δ={abs_delta:.3f}"
        f"   |   tps (info only): golden={gt:.2f}  smoke={st:.2f}"
    )
sys.exit(1 if fail else 0)
PY

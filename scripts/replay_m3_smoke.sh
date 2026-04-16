#!/usr/bin/env bash
# M3 replay smoke test: runs a single representative row from each ablation group
# and compares against the committed numbers in results/ablations/ablations.csv.
# If M4 code changes silently break M3 semantics, this catches it fast.
#
# Usage (on pod):
#   bash scripts/replay_m3_smoke.sh
#
# Rows replayed (one per group, seed=42):
#   default_canary_s42, A1_sheared_1b_s42, A2_k4_s42,
#   A3_block512_s42, A4_async1_s42, A5_llama2_7b_s42
#
# The script asserts that throughput_tps is within 15% of the committed value
# (tolerance accounts for pod-to-pod hardware variance, NOT semantic change).
set -euo pipefail

cd "$(dirname "$0")/.."

SMOKE_CSV="results/ablations/m3_replay_smoke.csv"
GOLDEN="results/ablations/ablations.csv"
TOLERANCE=0.15  # 15% throughput tolerance

echo "==> Replaying M3 smoke rows to $SMOKE_CSV"
python run_experiment.py \
    --config configs/ablations.yml \
    --output "$SMOKE_CSV" \
    --nproc 8 \
    --seeds 42 \
    --groups A1 A2 A3 A4 A5

echo ""
echo "==> Comparing against $GOLDEN (tolerance ${TOLERANCE})"
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
    gt, st = float(g['throughput_tps']), float(s['throughput_tps'])
    delta = abs(gt - st) / max(gt, 1e-6)
    tag = "OK" if delta <= $TOLERANCE else "REGRESSION"
    if delta > $TOLERANCE:
        fail += 1
    print(f"  {tag:10s} {rid:26s}  golden={gt:7.2f}  smoke={st:7.2f}  delta={delta*100:.1f}%")
sys.exit(1 if fail else 0)
PY

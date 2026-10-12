#!/usr/bin/env bash
# ONE script for BOTH profiles, parameterised by --nproc.
#
# WHY ONE SCRIPT AND NOT TWO. The 1-GPU and 2-GPU profiles must differ in EXACTLY one
# thing (nproc) or the comparison between them is not interpretable. Two hand-maintained
# scripts drift; one script with a parameter cannot. PROFILE_PLAN.md records the two exact
# invocations.
#
# WHAT IT DOES. Runs a single target-only decode row at the given context with the
# runner's EXISTING --profile flag (run_experiment.py:1816, "Wrap each generate() call in a
# torch.profiler context (C7 RoundProfiler)"), which writes a per-run compute/comm/idle
# sidecar to <output_dir>/profiler/<run_id>.json. It then reports, in one place:
#   - the measured per-token time from the CSV, as time_sec/tokens_generated (never the
#     rounded throughput_tps field);
#   - the profiler sidecar, verbatim keys;
#   - the share of the measured per-token time that the profiled CUDA time explains.
#
# NO ENGINE CHANGES. src/ is untouched by design: any change there invalidates every
# timing already measured.
#
# Usage:
#   bash scripts/mlsys_profile_decode.sh --nproc 1 --tag a
#   bash scripts/mlsys_profile_decode.sh --nproc 2 --tag b
set -uo pipefail
NPROC=1; TAG=a; CFG=configs/mlsys_profile_128k.yml; OUT=results/mlsys/profile; REPORT_ONLY=0
while [ $# -gt 0 ]; do
  case "$1" in
    --nproc) NPROC="$2"; shift 2 ;;
    --tag)   TAG="$2";   shift 2 ;;
    --cfg)   CFG="$2";   shift 2 ;;
    --out)   OUT="$2";   shift 2 ;;
    --report-only) REPORT_ONLY=1; shift 1 ;;
    *) echo "unknown arg: $1"; exit 2 ;;
  esac
done
# INTERPRETER. The first version defaulted to bare `python`, which on this Mac is the BASE
# miniconda interpreter whose torch is broken -- `ImportError: Library not loaded:
# @rpath/libtorch_cpu.dylib`. The local dry run failed on that and nothing else, which is
# exactly what a local test is for. Prefer the project env, and let MLSYS_PY override
# (on the pod the loop still comes from ~/pod_env.log).
PY=${MLSYS_PY:-}
if [ -z "$PY" ]; then
  for _c in "$HOME/miniconda3/envs/rasd/bin/python" "$HOME/miniconda3/envs/rasd-gpu/bin/python"; do
    [ -x "$_c" ] && PY="$_c" && break
  done
fi
PY=${PY:-python}
echo "interpreter: $PY"
mkdir -p "$OUT"
CSV="$OUT/profile_${TAG}_nproc${NPROC}.csv"
echo "=== profile $TAG: nproc=$NPROC cfg=$CFG csv=$CSV ==="
if [ "$REPORT_ONLY" != "1" ]; then
echo "--- 1. the profiled run (the runner's own --profile path) ---"
timeout 5400 "$PY" run_experiment.py --config "$CFG" --nproc "$NPROC" \
  --output "$CSV" --stage-id "profile_${TAG}_nproc${NPROC}" --profile 2>&1 | tail -25
fi
echo "--- 2. the profiler sidecar ---"
for f in "$OUT"/profiler/*.json; do
  [ -f "$f" ] || { echo "  NO SIDECAR at $OUT/profiler/ -- the --profile flag did not write one"; break; }
  echo "  $f"; "$PY" -c "
import json,sys
d=json.load(open(sys.argv[1]))
for k in sorted(d): print(f'    {k} = {d[k]}')
" "$f"
done
echo "--- 3. measured vs profiled, per token ---"
"$PY" - "$CSV" "$OUT" <<'PYEOF'
import csv, glob, json, sys, os
csvp, out = sys.argv[1], sys.argv[2]
if not os.path.exists(csvp):
    print("  no CSV; the run did not produce a row"); raise SystemExit
rows = [r for r in csv.DictReader(open(csvp)) if r.get('status') == 'ok']
if not rows:
    print("  no ok rows"); raise SystemExit
for r in rows:
    tps = int(r['tokens_generated']) / float(r['time_sec'])
    print(f"  {r['run_id']}: tokens={r['tokens_generated']} time={float(r['time_sec']):.1f}s "
          f"-> {tps:.4f} tok/s = {1/tps*1000:.1f} ms/token (exact, not the rounded field)")
    side = os.path.join(out, 'profiler', r['run_id'] + '.json')
    if os.path.exists(side):
        # UNITS. The sidecar's fields are microseconds (`compute_us`, `comm_us`,
        # `idle_us`); time_sec is seconds. The first version added the sidecar's numbers and
        # divided by seconds, and reported a 99,166,666% share of wall time. The synthetic
        # test above caught it precisely because it runs with no GPU. Convert by key suffix
        # and say what was assumed, rather than printing a number with no unit.
        d = json.load(open(side))
        sec, unknown = 0.0, []
        for k, v in d.items():
            if not isinstance(v, (int, float)):
                continue
            if k.endswith('_us'):
                sec += v / 1e6
            elif k.endswith('_ms'):
                sec += v / 1e3
            elif k.endswith('_s') or k.endswith('_sec'):
                sec += v
            else:
                unknown.append(k)
        wall = float(r['time_sec'])
        print(f"    sidecar total = {sec:.2f} s (converted from *_us/*_ms/*_s by key suffix)")
        if unknown:
            print(f"    NOT COUNTED (no unit suffix, so no unit was assumed): {unknown}")
        if wall:
            print(f"    share of the measured wall time: {sec/wall*100:.1f}%")
    else:
        print(f"    no sidecar at {side}")
PYEOF
echo "=== done: $TAG ==="

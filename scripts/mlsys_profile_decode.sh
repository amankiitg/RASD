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
import csv, json, os, sys
csvp, out = sys.argv[1], sys.argv[2]
if not os.path.exists(csvp):
    print("  no CSV; the run did not produce a row"); raise SystemExit
rows = [r for r in csv.DictReader(open(csvp)) if r.get('status') == 'ok']
if not rows:
    print("  no ok rows"); raise SystemExit

# THE CANONICAL SIDECAR SCHEMA, from src/analysis/profiler.py:214-226:
#   compute_us, comm_us, idle_us, other_us   -- the four buckets
#   total_us = compute+comm+other            -- a SUBTOTAL of the three above
#   wall_us                                  -- the profiler context's wall clock
#   overlap_us, *_pct                        -- derived
# The FIRST version of this report added up EVERY *_us field. That double counts:
# total_us duplicates compute+comm+other, and wall_us is the denominator rather than
# a component, so the "sidecar total" it printed was never a share of anything. The
# buckets and their shares of wall_us are the honest form, so they are read by NAME
# here; a suffix-sum fallback is kept only for an unknown schema, and it says so.
BUCKETS = ("compute_us", "comm_us", "idle_us", "other_us")

def num(d, k):
    v = d.get(k)
    return float(v) if isinstance(v, (int, float)) else None

for r in rows:
    tps = int(r['tokens_generated']) / float(r['time_sec'])
    print(f"  {r['run_id']}: tokens={r['tokens_generated']} time={float(r['time_sec']):.1f}s "
          f"-> {tps:.4f} tok/s = {1/tps*1000:.1f} ms/token (exact, not the rounded field)")
    side = os.path.join(out, 'profiler', r['run_id'] + '.json')
    if not os.path.exists(side):
        print(f"    no sidecar at {side}")
        continue
    d = json.load(open(side))
    row_s = float(r['time_sec'])
    wall_us = num(d, 'wall_us')
    if all(num(d, k) is not None for k in BUCKETS) and wall_us:
        print(f"    profiler wall = {wall_us/1e6:.2f} s vs the row's measured {row_s:.2f} s "
              f"({wall_us/1e6/row_s*100:.1f}% of it; the profiler wraps generate(), which "
              f"includes prefill)")
        for k in BUCKETS:
            v = num(d, k)
            print(f"    {k:10s} = {v/1e6:9.2f} s   {v/wall_us*100:6.1f}% of the profiler wall")
        cc = num(d, 'compute_us') + num(d, 'comm_us') + num(d, 'other_us')
        tot = num(d, 'total_us')
        print(f"    compute+comm+other = {cc/1e6:.2f} s"
              + (f"  (the sidecar's own total_us = {tot/1e6:.2f} s, the same three buckets "
                 f"-- which is why they must not be summed again)" if tot is not None else ""))
        ov = num(d, 'overlap_us')
        if ov is not None:
            pct = num(d, 'overlap_pct')
            print(f"    overlap    = {ov/1e6:9.2f} s"
                  + (f"   {pct*100:6.1f}%" if pct is not None else "")
                  + " -- CUDA time EXCEEDING the wall clock, i.e. kernels running"
                    " concurrently; when this is nonzero the buckets are scaled, not additive")
    else:
        # Unknown schema: fall back to a suffix sum, and say that is what happened.
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
        print(f"    UNKNOWN SIDECAR SCHEMA ({sorted(d)}); fell back to summing by suffix "
              f"= {sec:.2f} s, which may double count")
        if unknown:
            print(f"    NOT COUNTED (no unit suffix): {unknown}")
PYEOF
echo "=== done: $TAG ==="

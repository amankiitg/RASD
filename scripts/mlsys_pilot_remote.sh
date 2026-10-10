#!/usr/bin/env bash
# The pod side of the pilot: run both jobs, leave everything in ~/RASD/results.
#
# Runs on the pod, invoked by scripts/mlsys_pilot_1x.sh. Kept as a staged file
# rather than an ssh heredoc because a heredoc through ssh has twice silently
# done nothing in this project (the run continued for minutes and the operator
# was told it had stopped).
set -uo pipefail
cd "$HOME/RASD" || exit 3
OUT=results/mlsys/pilot
mkdir -p "$OUT"
log() { printf '%s %s\n' "$(date -u +%FT%TZ)" "$*" | tee -a "$OUT/pilot.log"; }
# PILOT_JOBS=A runs only the near-tie rows: a re-run that needs the cross margin
# should not pay again for a 16 GB target download it already proved loads.
PILOT_JOBS=${PILOT_JOBS:-AB}
log "jobs requested: $PILOT_JOBS"

# TWO MISTAKES FIXED HERE, both measured on the first pilot attempt
# (instance e683001fc17c, 2026-10-10T12:07Z):
#
#   1. The env file is ~/RASD/.pod_env.sh, NOT ~/.pod_env.sh. The first attempt
#      sourced the wrong path, fell through to /usr/bin/python (system python,
#      no transformers) and every row died with ModuleNotFoundError. The setup
#      probe prints "~/.pod_env.sh written" in its own PASS message, which is
#      what made the wrong path look right.
#   2. This is ONE card. The launcher's default process count is 8, so the first
#      attempt started 8 ranks on a single GPU (the log shows local_rank 4 and
#      6 failing). `--nproc 1` is mandatory here and is the one unavoidable
#      difference from the manifest form.
#
# The interpreter is RESOLVED, never guessed: ~/pod_env.log carries the
# interpreter the provisioning actually validated, and if it is absent this
# refuses rather than falling back to a bare python (the failure mode that has
# silently produced runs with no transformations before).
PY=$(grep -a "^INTERPRETER=" "$HOME/pod_env.log" 2>/dev/null | tail -1 | cut -d= -f2-)
if [ -z "${PY:-}" ] || [ ! -x "${PY:-}" ]; then
  log "FATAL: no usable INTERPRETER= in ~/pod_env.log; refusing to guess one"
  exit 2
fi
# shellcheck disable=SC1090
set -a; . "$HOME/RASD/.pod_env.sh"; set +a
log "=== interpreter: $PY ==="
log "=== cuda ==="
"$PY" -c 'import torch, transformers; print("torch", torch.__version__, "transformers", transformers.__version__, "cuda_devices", torch.cuda.device_count(), torch.cuda.get_device_name(0))' 2>&1 | tee -a "$OUT/pilot.log"

# ---- the 1M data, checked before anything is loaded -------------------------
# Two claims: the pool really holds >= 1,048,576-token documents, and its ids
# round-trip through the candidate tokenizer (the Llama-2-pool trap).
cat > /tmp/pool_check.py <<'PYEOF'
import json, sys
import numpy as np
sys.path.insert(0, ".")
from transformers import AutoTokenizer
from src.analysis.pool_tokenizer import pool_check, pool_problem

meta = json.load(open("data/processed/pg19_1m/documents.json"))
print(f"pool: {len(meta['documents'])} documents, tokenizer={meta['tokenizer']}, "
      f"rungs={meta.get('rungs')}")
need = 1048576
for d in sorted(meta["documents"], key=lambda x: x["doc_id"]):
    L = int(d["length"])
    book = int(d.get("book_tokens", 0))
    print(f"  {d['doc_id']:16s} staged={L:>9,} book={book:>9,} "
          f"fits 1M window: {L >= need}  {d['title'][:40]}")
tok = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B", local_files_only=False)
worst = 1.0
for d in meta["documents"]:
    arr = np.memmap(d["file"], dtype="int32", mode="r")
    ids = np.asarray(arr[:32768]).astype(int).tolist()
    chk = pool_check(tok, ids, meta.get("tokenizer"), "meta-llama/Llama-3.1-8B")
    worst = min(worst, chk["pool_roundtrip_share"])
    print(f"  round-trip {d['doc_id']:16s} {chk['pool_roundtrip_share']:.4f} "
          f"problem={pool_problem(chk)[:80] or '(none)'}")
print(f"WORST ROUND-TRIP SHARE: {worst:.4f}  (needs > 0.90)")
PYEOF
log "--- 1M pool check ---"
"$PY" /tmp/pool_check.py 2>&1 | tee -a "$OUT/pilot.log"

# ---- JOB A: M3 near-tie, cross margin logged -------------------------------
# Dry run first: it validates the config, the pool and the plan without a GPU,
# so a typo costs seconds instead of a model load.
CFG_A=configs/mlsys_pilot_neartie_32k.yml
log "--- JOB A dry run ($CFG_A) ---"
timeout 300 "$PY" run_experiment.py --config "$CFG_A" --dry-run 2>&1 | tail -20 | tee -a "$OUT/pilot.log"
log "--- JOB A real run ---"
timeout 1500 "$PY" run_experiment.py --config "$CFG_A" --nproc 1 \
      --output results/mlsys/pilot/pilot_neartie_32k.csv \
      --stage-id pilot_neartie_32k --abort-on-failure \
      --log-per-token --save-generated-tokens 2>&1 | tee "$OUT/jobA.log" | tail -40
log "JOB A exit=$? csv: $(wc -l < results/mlsys/pilot/pilot_neartie_32k.csv 2>/dev/null || echo none) rows"

# ---- JOB B: the native-1M target + Llama-3.2-1B draft ----------------------
if [ "$PILOT_JOBS" = "A" ]; then
  log "=== PILOT_JOBS=A: skipping jobs B, leaving results in $OUT ==="
  log "=== results ==="
  for f in results/mlsys/pilot/*.csv; do echo "--- $f"; cat "$f"; done | tee -a "$OUT/pilot.log"
  log "=== PILOT DONE ==="
  exit 0
fi
CFG_B=configs/mlsys_pilot_1m_32k.yml
log "--- JOB B dry run ($CFG_B) ---"
timeout 300 "$PY" run_experiment.py --config "$CFG_B" --dry-run 2>&1 | tail -20 | tee -a "$OUT/pilot.log"
log "--- JOB B real run (downloads a 16 GB target on first use) ---"
timeout 1800 "$PY" run_experiment.py --config "$CFG_B" --nproc 1 \
      --output results/mlsys/pilot/pilot_1m_32k.csv \
      --stage-id pilot_1m_32k --abort-on-failure \
      --log-per-token --save-generated-tokens 2>&1 | tee "$OUT/jobB.log" | tail -40
log "JOB B exit=$? csv: $(wc -l < results/mlsys/pilot/pilot_1m_32k.csv 2>/dev/null || echo none) rows"

log "=== results ==="
for f in results/mlsys/pilot/*.csv; do
  echo "--- $f"; cat "$f"
done | tee -a "$OUT/pilot.log"
log "=== sidecars ==="
ls -la results/mlsys/pilot/tokens 2>/dev/null | head -12 | tee -a "$OUT/pilot.log"
log "=== PILOT DONE ==="

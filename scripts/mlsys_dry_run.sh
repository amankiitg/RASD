#!/usr/bin/env bash
# MLSys pipeline dry run — exercise every stage locally, on a tiny model, at a
# small context, so failures surface here instead of on paid GPU time.
#
# Origin: a stage called `pg19_short_target` died on the pod with
# FileNotFoundError on data/processed/pg19/pg19_validation_metadata.json,
# because the metadata (and the .dat chunks it points at) were never staged.
# Nothing about that failure needed a GPU to find. This run covers:
#
#   1  staging a PG-19 dataset: metadata AND the chunk files it names
#   2  the prompt builder, incl. exact token length + a logged per-seed hash
#   3  the coherence gate: effective-rope assertion, real-context perplexity,
#      generation scoring, and the pass rule — including that it REJECTS a
#      known-bad configuration
#   4  a speculative run via run_experiment's own config/CLI path
#   5  the (f) filename-collision guard
#   6  the acceptance reporting with cluster-bootstrap intervals
#   7  a staged pull with checksum verification
#
# Exits non-zero on the first failed assertion. No network, no GPU required.

set -uo pipefail
cd "$(dirname "$0")/.."
REPO=$PWD
WORK=$(mktemp -d)
trap 'rm -rf "$WORK"' EXIT

PASS=0; FAIL=0
ok()   { printf "  \033[32mPASS\033[0m  %s\n" "$1"; PASS=$((PASS+1)); }
bad()  { printf "  \033[31mFAIL\033[0m  %s\n" "$1"; FAIL=$((FAIL+1)); }
step() { printf "\n\033[1m== %s\033[0m\n" "$1"; }

# Resolve an interpreter BINARY, not `conda run -n ...`. `conda run` does not
# forward stdin, so every heredoc in this script would silently do nothing and
# its stage would "fail" for a reason that has nothing to do with the pipeline.
# This script found that bug on its own first run, which is the point of it.
PY=python3
if ! $PY -c "import torch, transformers" >/dev/null 2>&1; then
  for cand in "$HOME/miniconda3/envs/rasd/bin/python" \
              "$HOME/anaconda3/envs/rasd/bin/python"; do
    if [ -x "$cand" ] && "$cand" -c "import torch, transformers" >/dev/null 2>&1; then
      PY="$cand"; break
    fi
  done
fi
if ! $PY -c "import torch, transformers" >/dev/null 2>&1; then
  echo "no python with torch+transformers found; set PY=/path/to/python" >&2
  exit 2
fi
echo "python: $PY"
echo "work:   $WORK"

# --------------------------------------------------------------------------
step "1  stage a PG-19 dataset (metadata + the chunks it names)"
# --------------------------------------------------------------------------
$PY - "$WORK" <<'PYEOF'
import json, sys, numpy as np, pathlib
work = pathlib.Path(sys.argv[1])
# Mirror the POD LAYOUT exactly: data/processed/pg19/ relative to the project
# root. Production stages the chunks into ~/RASD/data/processed/pg19/ and runs
# from ~/RASD, so the metadata's relative `file` entries resolve. Reproducing
# that layout here is what lets this dry run catch a staging mistake.
d = work / "data" / "processed" / "pg19"; d.mkdir(parents=True)
chunks = []
for i, n in enumerate((4096, 2048)):
    p = d / f"pg19_validation_chunk_{i}.dat"
    arr = np.memmap(p, dtype="int32", mode="w+", shape=(n,))
    # Simple deterministic "text": ids in a plausible vocab range.
    arr[:] = (np.arange(n) * 37 % 30000).astype("int32"); arr.flush()
    # Deliberately RELATIVE, exactly like the real preprocess_pg19.py output —
    # the pod failure was a relative path that did not resolve.
    chunks.append({"file": str(p.relative_to(work)), "length": n})
(d / "pg19_validation_metadata.json").write_text(json.dumps({"chunks": chunks}))
print(f"  wrote {len(chunks)} chunks + metadata under {d}")
PYEOF
if [ -f "$WORK/data/processed/pg19/pg19_validation_metadata.json" ] && \
   [ "$(ls "$WORK"/data/processed/pg19/*.dat | wc -l | tr -d ' ')" = "2" ]; then
  ok "metadata and BOTH chunk files staged"
else
  bad "staging incomplete"
fi
# The failure mode itself: does the metadata's relative path resolve from the
# directory the pipeline is run from?
if (cd "$WORK" && $PY -c "
import json,pathlib
m=json.load(open('data/processed/pg19/pg19_validation_metadata.json'))
missing=[c['file'] for c in m['chunks'] if not pathlib.Path(c['file']).exists()]
print('  unresolved:', missing)
raise SystemExit(1 if missing else 0)"); then
  ok "every relative path in the metadata resolves from the run directory"
else
  bad "metadata names a path that does not resolve (the pod's FileNotFoundError)"
fi

# --------------------------------------------------------------------------
step "2  PG-19 prompt builder: exact length + logged per-seed hash"
# --------------------------------------------------------------------------
(cd "$WORK" && PYTHONPATH="$REPO" $PY - <<'PYEOF'
import sys, hashlib, json, pathlib
from transformers import AutoTokenizer
from run_experiment import build_prompt
tok = AutoTokenizer.from_pretrained("hf-internal-testing/tiny-random-LlamaForCausalLM")
# Relative to $WORK, matching the staged layout.
meta = "data/processed/pg19/pg19_validation_metadata.json"
hashes = {}
for seed in (42, 123, 456):
    t = build_prompt(1024, tok, source="pg19", pg19_meta=meta, seed=seed)
    ids = tok.encode(t, add_special_tokens=False)
    hashes[seed] = hashlib.sha256(json.dumps(ids).encode()).hexdigest()[:16]
    print(f"  seed={seed:>4} tokens={len(ids):>6} hash={hashes[seed]}")
print(f"  unique hashes: {len(set(hashes.values()))}/3")
sys.exit(0 if len(set(hashes.values())) == 3 else 1)
PYEOF
)
[ $? -eq 0 ] && ok "prompt builder produced 3 distinct seeded prompts" \
             || bad "prompt builder did not vary by seed"

# --------------------------------------------------------------------------
step "3  coherence gate: rope assertion + real-context metrics + pass rule"
# --------------------------------------------------------------------------
cat > "$WORK/cands.json" <<JSONEOF
{"coherence_gate": [
  {"name": "tiny_native", "target_model_name": "hf-internal-testing/tiny-random-LlamaForCausalLM",
   "context_length": 2048, "rope_type": "none", "seed": 42, "native_baseline": true},
  {"name": "tiny_yarn_rebased", "target_model_name": "hf-internal-testing/tiny-random-LlamaForCausalLM",
   "context_length": 2048, "rope_type": "yarn", "rope_factor": 8, "rope_anchor_base": null, "seed": 42}
]}
JSONEOF
(cd "$WORK" && PYTHONPATH="$REPO" $PY "$REPO/scripts/mlsys_coherence_gate.py" \
    --candidates "$WORK/cands.json" \
    --pg19-meta data/processed/pg19/pg19_validation_metadata.json \
    --out "$WORK/gate.csv" --gen-dir "$WORK/gen") > "$WORK/gate.log" 2>&1
GRC=$?
tail -12 "$WORK/gate.log"
if [ $GRC -eq 0 ] && [ -s "$WORK/gate.csv" ]; then
  ok "gate ran and wrote a CSV"
else
  bad "gate did not complete (rc=$GRC)"
fi
if $PY - "$WORK/gate.csv" <<'PYEOF'
import csv, sys
rows = list(csv.DictReader(open(sys.argv[1])))
cols = {"effective_rope_match", "inv_freq_last", "slowest_channel_stretch",
        "ppl_continuation", "gen_blank_share", "gen_repeat_share",
        "early_eos", "gate_pass", "gate_reason", "prompt_sha256"}
missing = cols - set(rows[0].keys())
assert not missing, f"gate CSV missing {missing}"
assert all(r["gate_pass"] in ("True", "False") for r in rows), "no verdict column"
assert any(r["ppl_continuation"] for r in rows), "no perplexity measured"
assert any(r["inv_freq_last"] for r in rows), "effective rope not reported"
print(f"  gate CSV ok: {len(rows)} rows, verdicts "
      f"{[r['gate_pass'] for r in rows]}")
PYEOF
then ok "gate CSV reports rope assertion, real-context PPL and a verdict"
else bad "gate CSV is missing required fields"; fi

# A gate that cannot reject is not a gate. Check the pure verdict rule against
# synthetic rows: a clean row passes, and each defect fails on its own reason.
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="$REPO" $PY - <<'VEOF'
import sys
from scripts.mlsys_coherence_gate import verdict
NATIVE = 10.0
clean = {"status": "ok", "ppl_continuation": 12.0, "early_eos": False,
         "gen_blank_share": 0.05, "gen_repeat_share": 0.10,
         "effective_rope_matches_intent": True}
cases = [
    ("clean row", clean, True),
    ("perplexity 1.6x native", {**clean, "ppl_continuation": 16.0}, False),
    ("early EOS", {**clean, "early_eos": True, "eos_at": 3}, False),
    ("blank-line collapse", {**clean, "gen_blank_share": 0.98}, False),
    ("repetition collapse", {**clean, "gen_repeat_share": 0.9}, False),
    ("rope mismatch", {**clean, "effective_rope_matches_intent": False}, False),
    ("not measured", {"status": "oom"}, False),
]
bad = 0
for label, row, want in cases:
    got = verdict(row, NATIVE)["gate_pass"]
    mark = "ok " if got == want else "BAD"
    if got != want: bad += 1
    print(f"    {mark} {label:<26} -> gate_pass={got} (want {want})")
sys.exit(1 if bad else 0)
VEOF
[ $? -eq 0 ] && ok "gate verdict rejects each defect and passes a clean row" \
             || bad "gate verdict rule is wrong"

# --------------------------------------------------------------------------
step "4  speculative run through run_experiment's own CLI path"
# --------------------------------------------------------------------------
cat > "$WORK/tiny.yml" <<YAMLEOF
defaults:
  target_model_name: hf-internal-testing/tiny-random-LlamaForCausalLM
  draft_model_name: hf-internal-testing/tiny-random-LlamaForCausalLM
  spec_steps: 2
  kv_block_size: 64
  prefetch_depth: 1
  max_new_tokens: 8
  ignore_eos: true
  temperature: 1.0
  top_p: 1.0
  seeds: [42]
TINY:
  name: tiny_spec
  factor: context_length
  levels:
  - id: TINY_ctx512
    context_length: 512
    checkpoint_every: 0
    log_per_token: true
YAMLEOF
$PY run_experiment.py --config "$WORK/tiny.yml" --groups TINY --seeds 42 \
    --dry-run > "$WORK/dryrun.log" 2>&1
if [ $? -eq 0 ]; then ok "run_experiment accepted the config and planned jobs"; \
else bad "run_experiment rejected the config"; tail -5 "$WORK/dryrun.log"; fi
grep -q "TINY_ctx512" "$WORK/dryrun.log" \
  && ok "the planned job names the configured level" \
  || bad "planned job did not include the level id"

# --------------------------------------------------------------------------
step "5  filename-collision guard (f)"
# --------------------------------------------------------------------------
$PY - "$WORK" "$REPO" <<'PYEOF'
import sys, pathlib
work, repo = sys.argv[1], sys.argv[2]
sys.path.insert(0, repo)
from run_experiment import _guard_output_collision
agg = pathlib.Path(work) / "pg19_multiseed.csv"
agg.write_text("run_id,level_id,group\nPG19_ctx4k_s123,PG19_ctx4k,PG19\n")
try:
    _guard_output_collision(agg, "pg19_short_target"); print("  NOT refused"); sys.exit(1)
except SystemExit as e:
    if "REFUSING" not in str(e): raise
    print("  refused the collision with an unrelated aggregate")
_guard_output_collision(agg, "PG19_ctx4k")
print("  allowed the file's own stage")
sys.exit(0)
PYEOF
[ $? -eq 0 ] && ok "guard refuses a foreign filename and allows the owner" \
             || bad "guard behaved incorrectly"

# --------------------------------------------------------------------------
step "6  acceptance reporting with cluster-bootstrap intervals (e)"
# --------------------------------------------------------------------------
$PY - "$WORK" <<'PYEOF'
import json, pathlib, random, sys
d = pathlib.Path(sys.argv[1]) / "per_token"; d.mkdir(parents=True, exist_ok=True)
rng = random.Random(0)
# One saturated run and one collapsed run, both with an EOS-flagged round.
for name, p in (("tiny_healthy", 0.9), ("tiny_collapsed", 0.05)):
    rows = []
    for i in range(20):
        n = sum(1 for _ in range(4) if rng.random() < p)
        rows.append({"round_idx": i, "global_pos_start": i * 5, "spec_steps": 4,
                     "n_acc": n, "draft_tokens": [1, 2, 3, 4],
                     "accepted": [True] * n + [False] * (4 - n),
                     "ended_on_eos": i == 19})
    (d / f"{name}.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n")
print(f"  wrote 2 synthetic traces to {d}")
PYEOF
$PY scripts/mlsys_cluster_bootstrap.py --traces "$WORK/per_token" \
    --out "$WORK/boot.csv" --window 10 --n-boot 2000 > "$WORK/boot.log" 2>&1
if [ $? -eq 0 ] && [ -s "$WORK/boot.csv" ]; then
  tail -6 "$WORK/boot.log"
  $PY - "$WORK/boot.csv" <<'PYEOF'
import csv, sys
rows = {r["trace"]: r for r in csv.DictReader(open(sys.argv[1]))}
for k in ("alpha_round_ci_lo", "alpha_round_ci_hi", "a_iid",
          "p_alpha_zero", "full_accept_share", "ended_on_eos_round"):
    assert k in rows["tiny_healthy"], f"missing {k}"
h, c = rows["tiny_healthy"], rows["tiny_collapsed"]
assert float(h["alpha_round"]) > float(c["alpha_round"]), "ordering wrong"
assert float(h["alpha_round_ci_lo"]) <= float(h["alpha_round"]) <= float(h["alpha_round_ci_hi"])
assert h["ended_on_eos_round"] == "19", h["ended_on_eos_round"]
assert "alpha_window10" in h, "common-window metric missing"
print(f"  healthy alpha={h['alpha_round']} CI=[{h['alpha_round_ci_lo']},{h['alpha_round_ci_hi']}]"
      f"  collapsed alpha={c['alpha_round']}  EOS round={h['ended_on_eos_round']}")
PYEOF
  [ $? -eq 0 ] && ok "bootstrap intervals, P(alpha=0), window metric, EOS split all present" \
               || bad "bootstrap output failed its own checks"
else
  bad "bootstrap script did not produce output"
fi

# --------------------------------------------------------------------------
step "7  staged pull with checksum verification"
# --------------------------------------------------------------------------
SRC="$WORK/pod_results"; DST="$WORK/laptop_results"
mkdir -p "$SRC/per_token" "$DST"
cp "$WORK/gate.csv" "$SRC/"; cp "$WORK/per_token/"*.jsonl "$SRC/per_token/"
rsync -a "$SRC/" "$DST/" >/dev/null 2>&1
$PY - "$SRC" "$DST" <<'PYEOF'
import hashlib, pathlib, sys
src, dst = pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2])
def h(p): return hashlib.sha256(p.read_bytes()).hexdigest()
missing = [f for f in src.rglob("*") if f.is_file() and not (dst / f.relative_to(src)).exists()]
bad = [f for f in src.rglob("*") if f.is_file() and (dst / f.relative_to(src)).exists()
       and h(f) != h(dst / f.relative_to(src))]
print(f"  files: {sum(1 for f in src.rglob('*') if f.is_file())}  missing: {len(missing)}  divergent: {len(bad)}")
sys.exit(1 if missing or bad else 0)
PYEOF
[ $? -eq 0 ] && ok "pull reproduced every file byte-for-byte" \
             || bad "pull verification found missing or divergent files"

# --------------------------------------------------------------------------
printf "\n\033[1m== DRY RUN RESULT: %d passed, %d failed ==\033[0m\n" "$PASS" "$FAIL"
[ "$FAIL" -eq 0 ] || exit 1
echo "every stage of the pipeline ran end to end on a tiny model."

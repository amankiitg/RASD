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
#   8  documents: one book per document, contiguous prompt windows, and the
#      per-document run expansion
#   9  losslessness: a token-level match is LOSSLESS, a divergence is a
#      MISMATCH with its first position, and a mismatched request is refused
#  10  target quality: sharded NLL tiles the sequence once and equals the
#      unsharded total
#  11  document-level intervals and the payoff verdict (which must NOT round an
#      interval that contains 1.0 toward the favourable side)
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
# The generation shares are judged against the PAIRED BASELINE ROW (2026-10-08
# revision), so every case passes the row it would have been paired with.
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="$REPO" $PY - <<'VEOF'
import sys
from scripts.mlsys_coherence_gate import verdict
BASE = {"status": "ok", "ppl_continuation": 10.0, "gen_blank_share": 0.05,
        "gen_repeat_share": 0.10}
clean = {"status": "ok", "ppl_continuation": 12.0, "early_eos": False,
         "gen_blank_share": 0.05, "gen_repeat_share": 0.10,
         "effective_rope_matches_intent": True}
cases = [
    ("clean row", clean, BASE, True),
    ("perplexity 1.6x native", {**clean, "ppl_continuation": 16.0}, BASE, False),
    ("degenerate EOS", {**clean, "early_eos": True, "eos_at": 3}, BASE, False),
    ("blank-line collapse", {**clean, "gen_blank_share": 0.98}, BASE, False),
    # The relative rule, in both directions: 2.4x the baseline's own share is
    # rejected although it is nowhere near the absolute ceiling, and 1.8x is
    # not. Without these two the suite would pass a rule that had quietly
    # reverted to an absolute threshold.
    ("blank share 2.4x baseline", {**clean, "gen_blank_share": 0.12}, BASE, False),
    ("blank share 1.8x baseline", {**clean, "gen_blank_share": 0.09}, BASE, True),
    # A zero baseline share leaves the ratio undefined, so only the ceiling
    # applies. Documented hole, asserted so it stays a decision.
    ("zero baseline share", {**clean, "gen_blank_share": 0.5},
     {**BASE, "gen_blank_share": 0.0}, True),
    ("repetition collapse", {**clean, "gen_repeat_share": 0.9}, BASE, False),
    ("rope mismatch", {**clean, "effective_rope_matches_intent": False}, BASE, False),
    ("not measured", {"status": "oom"}, BASE, False),
]
bad = 0
for label, row, base, want in cases:
    got = verdict(row, base)["gate_pass"]
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
# Ownership is the stage's OWN planned level ids, not a pattern in the stage
# name. The pattern version could never own the campaign's pairs (stage
# `engine_cap_smoke`, levels `CAPS_prefix64`) and, called from the worker with
# no --stage-id, it refused the SECOND ROW of every stage.
try:
    _guard_output_collision(agg, "pg19_short_target",
                            planned_level_ids={"pg19_short"})
    print("  NOT refused"); sys.exit(1)
except SystemExit as e:
    if "REFUSING" not in str(e): raise
    print("  refused the collision with an unrelated aggregate")
_guard_output_collision(agg, "PG19_ctx4k", planned_level_ids={"PG19_ctx4k"})
print("  allowed the file's own stage")
# ...and the same stage coming back for its second row, which is what failed on
# the pod: the file now holds a row this stage wrote.
_guard_output_collision(agg, "PG19_ctx4k",
                        planned_level_ids={"PG19_ctx4k", "PG19_ctx1M"})
print("  allowed a later row of the same stage")
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
for k in ("alpha_round_ci_lo", "alpha_round_ci_hi", "alpha_total_ratio",
          "alpha_iid",
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
step "8  documents: one book per document, windows, run expansion"
# --------------------------------------------------------------------------
(cd "$WORK" && PYTHONPATH="$REPO" $PY - <<'PYEOF'
import json, sys, pathlib, numpy as np
from run_experiment import _build_pg19_document_prompt, _exact_token_window

ctx, gen = 2048, 1024            # > CONTINUATION_TOKENS so a prompt fits
docs_dir = pathlib.Path("data/processed/pg19_docs"); docs_dir.mkdir(parents=True, exist_ok=True)
entries = []
for slot in range(3):
    ids = list(range(1000 + slot * 4096, 1000 + slot * 4096 + ctx + 64))
    p = docs_dir / f"doc_{slot:03d}.dat"
    arr = np.asarray(ids, dtype=np.int32)
    mm = np.memmap(p, dtype="int32", mode="w+", shape=arr.shape); mm[:] = arr[:]; mm.flush()
    entries.append({"doc_id": f"pg19_train_{slot}", "slot": slot, "file": str(p),
                    "length": len(ids)})
(docs_dir / "documents.json").write_text(json.dumps({"documents": entries}))


class StableTok:
    """Round-trips exactly, like Llama-3.1's byte-level BPE."""
    def decode(self, ids): return " ".join(f"t{i}" for i in ids)
    def encode(self, text, add_special_tokens=False):
        return [int(t[1:]) for t in text.split()]


class DriftTok:
    """Never stabilises: the encoded length is independent of the input length.

    A tokenizer whose round-trip merely shifts by a constant IS resolvable by
    adjusting the source length, so the unresolvable case needs a length that
    does not track the input at all.
    """
    def decode(self, ids): return " ".join(f"t{i}" for i in ids)
    def encode(self, text, add_special_tokens=False):
        return [0] * 5


tok = StableTok()
hashes = set()
for e in entries:
    text, cont, prov = _build_pg19_document_prompt(
        str(docs_dir / "documents.json"), ctx, e["doc_id"], tok, gen_tokens=gen)
    ids = list(range(1000 + e["slot"] * 4096, 1000 + e["slot"] * 4096 + ctx + 64))
    # The continuation must be contiguous with the prompt, and the sequence the
    # engine builds (prompt + leading BOS + generation) must fit the rung.
    assert len(cont) == gen, (len(cont), gen)
    assert prov["prompt_tokens"] == ctx - gen - 1, prov["prompt_tokens"]
    assert prov["sequence_tokens"] == ctx, prov["sequence_tokens"]
    assert cont[0] == ids[prov["prompt_tokens"]], "continuation not contiguous"
    hashes.add(prov["prompt_sha256"])
print(f"  documents: {len(entries)}  unique prompt hashes: {len(hashes)}/{len(entries)}")
print(f"  prompt={prov['prompt_tokens']} + 1 BOS + gen={gen} = {prov['sequence_tokens']} == ctx={ctx}")

# A tokenizer whose decode->encode is not length-preserving must NOT be allowed
# to run: the sequence would silently be a different context than the rung.
try:
    _exact_token_window(DriftTok(), list(range(100)), 100)
except RuntimeError as exc:
    print(f"  unresolvable tokenizer refused: {str(exc)[:64]}...")
else:
    print("  ERROR: a drifting tokenizer was accepted")
    sys.exit(1)

sys.exit(0 if len(hashes) == len(entries) else 1)
PYEOF
)
[ $? -eq 0 ] && ok "document windows contiguous, rung exact, drift refused" \
             || bad "document prompt windows were wrong"

cat > "$WORK/docs.yml" <<'YAMLEOF'
defaults:
  target_model_name: hf-internal-testing/tiny-random-LlamaForCausalLM
  draft_model_name: hf-internal-testing/tiny-random-LlamaForCausalLM
  spec_steps: 2
  kv_block_size: 64
  prefetch_depth: 1
  max_new_tokens: 8
  ignore_eos: true
  prompt_source: pg19_document
  prompt_documents_json: data/processed/pg19_docs/documents.json
  seeds: [42]
DOCS:
  name: docs
  factor: context_length
  levels:
  - id: DOC_ctx2048
    context_length: 2048
    documents: [pg19_train_0, pg19_train_1]
YAMLEOF
(cd "$WORK" && PYTHONPATH="$REPO" $PY "$REPO/run_experiment.py" --config "$WORK/docs.yml" \
    --groups DOCS --seeds 42 --dry-run > "$WORK/docs_dryrun.log" 2>&1)
if [ $? -eq 0 ]; then
  grep -q "pg19_train_0" "$WORK/docs_dryrun.log" && grep -q "pg19_train_1" "$WORK/docs_dryrun.log" \
    && ok "document expansion produced one planned run per document" \
    || bad "document expansion did not name both documents"
else bad "run_experiment rejected the document config"; tail -5 "$WORK/docs_dryrun.log"; fi

# --------------------------------------------------------------------------
step "9  losslessness: token-level agreement, and a mismatch is caught"
# --------------------------------------------------------------------------
$PY - "$WORK" <<'PYEOF'
import csv, json, pathlib, sys
work = pathlib.Path(sys.argv[1])
tw = work / "lossless" / "tokens"; tw.mkdir(parents=True, exist_ok=True)
# Three documents x (spec, target-only), exercising all three verdicts:
#   d0  full-length partner, identical                    -> LOSSLESS
#   d1  SHORT partner (128-token baseline), identical     -> LOSSLESS_PREFIX_128
#   d2  divergence at token 3                             -> MISMATCH
rows, toks = [], {}
FULL, SHORT = 1024, 128
for doc in ("d0", "d1", "d2"):
    body = list(range(100, 100 + (FULL if doc != "d1" else FULL)))
    spec = body if doc != "d2" else body[:3] + [999] + body[4:]
    tgt = body if doc != "d1" else body[:SHORT]
    for arm, ss, ids in (("spec", 4, spec), ("tgt", 0, tgt)):
        rid = f"{doc}_{arm}"
        toks[rid] = ids
        rows.append({"run_id": rid, "doc_id": doc, "context_length": 2048,
                     "max_new_tokens": FULL if doc != "d1" or arm == "spec" else SHORT,
                     "spec_steps": ss, "status": "ok",
                     # Shared pairing key per rung+document, as run_experiment
                     # writes it, plus the role that names the intended partner.
                     "pair_id": f"2048:{doc}",
                     "arm_role": ("spec" if arm == "spec"
                                  else ("target_short" if doc == "d1"
                                        else "target_full")),
                     "prompt_sha256": "h" + doc, "prompt_tokens": 1024,
                     "acceptance_rate": 0.9,
                     # decode_tps is the pre-registered primary ratio metric;
                     # the end-to-end rate is reported beside it.
                     "decode_tps": 10.0, "throughput_tps": 8.0,
                     "full_accept_share": 0.75})
        (tw / f"{rid}.json").write_text(json.dumps(
            {"run_id": rid, "generated_token_ids": ids,
             "doc_id": doc, "prompt_sha256": "h" + doc,
             "context_length": 2048, "max_new_tokens": 5}))
with (work / "lossless.csv").open("w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
PYEOF
$PY scripts/mlsys_losslessness.py --results "$WORK/lossless.csv" \
    --tokens-dir "$WORK/lossless/tokens" --out "$WORK/losslessness.csv" \
    > "$WORK/lossless.log" 2>&1
$PY - "$WORK" <<'PYEOF'
import csv, pathlib, sys
rows = {r["spec_run_id"]: r for r in
        csv.DictReader((pathlib.Path(sys.argv[1]) / "losslessness.csv").open())}
want = {"d0_spec": "LOSSLESS", "d1_spec": "LOSSLESS_PREFIX_128"}
bad = []
for rid, exp in want.items():
    got = rows[rid]["verdict"]
    print(f"  {rid} verdict={got} prefix={rows[rid]['verified_prefix']} (want {exp})")
    if got != exp:
        bad.append(rid)
d2 = rows["d2_spec"]
print(f"  d2_spec verdict={d2['verdict']} first_mismatch_position="
      f"{d2['first_mismatch_position']} (want MISMATCH at 3)")
if d2["verdict"] != "MISMATCH" or d2["first_mismatch_position"] != "3":
    bad.append("d2_spec")
sys.exit(1 if bad else 0)
PYEOF
[ $? -eq 0 ] && ok "full match, 128-token prefix and divergence each got the right verdict" \
             || bad "losslessness verdicts were wrong"
# A pair whose requests disagree must be refused, not ticked green. Two ways to
# disagree: a field inside the pair key (no pair is found at all) and a field
# outside it (the pair is found but rejected). Both must refuse.
$PY - "$WORK" <<'PYEOF'
import csv, pathlib, sys
work = pathlib.Path(sys.argv[1])
p = work / "lossless.csv"
rows = list(csv.DictReader(p.open()))
for r in rows:
    if r["run_id"] == "d0_tgt":
        r["prompt_tokens"] = "9999"      # outside the pair key -> BAD_PAIR
    if r["run_id"] == "d2_tgt":
        r["prompt_sha256"] = "DIFFERENT"  # inside the pair key -> NO_PAIR
with p.open("w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
PYEOF
$PY scripts/mlsys_losslessness.py --results "$WORK/lossless.csv" \
    --tokens-dir "$WORK/lossless/tokens" --out "$WORK/losslessness2.csv" \
    > "$WORK/lossless2.log" 2>&1
$PY - "$WORK" <<'PYEOF'
import csv, pathlib, sys
rows = {r["spec_run_id"]: r for r in
        csv.DictReader((pathlib.Path(sys.argv[1]) / "losslessness2.csv").open())}
a, b = rows["d0_spec"], rows["d2_spec"]
print(f"  d0_spec verdict={a['verdict']}  d2_spec verdict={b['verdict']}")
sys.exit(0 if a["verdict"] == "BAD_PAIR" and b["verdict"] == "NO_PAIR" else 1)
PYEOF
[ $? -eq 0 ] && ok "a mismatched request was refused, not ticked" \
             || bad "mismatched request was compared anyway"

# --------------------------------------------------------------------------
step "10  target quality: sharded perplexity equals unsharded"
# --------------------------------------------------------------------------
(cd "$WORK" && PYTHONPATH="$REPO" $PY - <<'PYEOF'
import sys, torch
from src.analysis.target_quality import continuation_nll, perplexity_from_sums

D, V, L, SCORE_FROM = 8, 16, 64, 40
torch.manual_seed(0)
emb = torch.nn.Embedding(V, D)
head = torch.nn.Linear(D, V, bias=False)

class Inner(torch.nn.Module):
    # Hidden at a position depends only on that token, so the sharded and
    # unsharded paths must produce the SAME total NLL. A mismatch would mean
    # the shard bounds lose or double-count positions, not that the model
    # quality differs.
    def forward(self, input_ids, position_ids=None, use_cache=False,
                past_key_values=None):
        return type("O", (), {"last_hidden_state": emb(input_ids)})()

class Mini(torch.nn.Module):
    def __init__(self):
        super().__init__(); self.model = Inner(); self.lm_head = head

m = Mini()
ids = torch.tensor([[i % V for i in range(L)]], dtype=torch.long)

t1, n1, _ = continuation_nll(m, ids, SCORE_FROM,
                             forward=lambda li, ap: m.model(input_ids=li,
                                                           position_ids=ap).last_hidden_state,
                             rank=0, world_size=1)
tot, cnt = 0.0, 0
for r in range(4):
    t, n, _ = continuation_nll(m, ids, SCORE_FROM,
                               forward=lambda li, ap: m.model(input_ids=li,
                                                             position_ids=ap).last_hidden_state,
                               rank=r, world_size=4)
    tot += t; cnt += n
print(f"  unsharded nll={t1:.6f} n={n1}   sharded nll={tot:.6f} n={cnt}")
# The COUNT must match exactly -- that is the tiling property. The sums are
# allowed to differ in the last bits because four shards accumulate their
# cross-entropy in a different order than one does.
assert n1 == cnt == L - SCORE_FROM, (n1, cnt, L - SCORE_FROM)
assert abs(t1 - tot) <= 1e-5 * max(1.0, abs(t1)), (t1, tot)
print(f"  ppl={perplexity_from_sums(t1, n1):.6f} over {n1} scored tokens")
sys.exit(0)
PYEOF
)
[ $? -eq 0 ] && ok "sharded NLL tiles the sequence exactly once and matches unsharded" \
             || bad "sharded target-quality NLL disagrees with unsharded"

# --------------------------------------------------------------------------
step "11  document-level intervals and the payoff verdict"
# --------------------------------------------------------------------------
$PY scripts/mlsys_document_bootstrap.py --results "$WORK/lossless.csv" \
    --group-by context_length --out "$WORK/doc_intervals.csv" \
    > "$WORK/doc_boot.log" 2>&1
if [ $? -eq 0 ]; then
  $PY - "$WORK" <<'PYEOF'
import csv, pathlib, sys
rows = list(csv.DictReader((pathlib.Path(sys.argv[1]) / "doc_intervals.csv").open()))
sp = [r for r in rows if r["estimate"] == "paired_speedup"]
assert sp, "no paired speedup row produced"
# Paired BY ID: the row must say which target arm each pair used, which is only
# recorded when the pairing went through pair_id/arm_role rather than position.
# The column aggregates over the documents in the group, so a group holding both
# a full-baseline document and a short-baseline one reports both roles.
for r in sp:
    roles = (r.get("paired_arm_roles") or "").split(",")
    assert roles and all(roles), (
        f"paired row does not record which target arm it used: {r}")
    assert set(roles) <= {"target_full", "target_short",
                          "target_1024", "target_128"}, roles
# The fixture pairs one document with the SHORT baseline and the rest with the
# full one, so a positional pairing would have shown a single role for all three.
assert set().union(*[set((r["paired_arm_roles"] or "").split(",")) for r in sp]) \
    == {"target_full", "target_short"}, [r["paired_arm_roles"] for r in sp]
r = sp[0]
print(f"  paired speedup point={r['point']} ci=[{r['ci_lo']}, {r['ci_hi']}] "
      f"n={r['n_documents']} verdict={r['verdict']}")
# The spec arm is 1.0x the target arm here, so the interval must contain 1.0
# and the verdict must be inconclusive rather than rounded to the favourable
# side.
assert abs(float(r["point"]) - 1.0) < 1e-9, r["point"]
assert r["verdict"] == "inconclusive", r["verdict"]
for m in rows:
    assert m["ci_lo"] and m["ci_hi"] and m["n_documents"], m
sys.exit(0)
PYEOF
  [ $? -eq 0 ] && ok "intervals computed per document and the verdict did not round" \
               || bad "interval output was wrong"
else bad "document bootstrap script failed"; tail -5 "$WORK/doc_boot.log"; fi

# --------------------------------------------------------------------------
step "12  the run guards: watchdog, ledger and terminate-on-every-exit"
# These guards live in the shell around the pipeline, so the dry run checks the
# wiring structurally AND exercises the arithmetic on the manifest's real values.
# A guard that silently no-ops is exactly the failure mode worth catching here.
"$PY" - scripts/mlsys_manifest.sh scripts/mlsys_watch_and_run.sh <<'PYDRY'
import sys, yaml
man = open(sys.argv[1]).read()
watch = open(sys.argv[2]).read()
m = yaml.safe_load(open("configs/mlsys_manifest.yml"))
stages = {s["id"]: s for s in m["stages"]}
fails = []

# --- the watchdog must stop the ladder, not merely log ------------------------
if "_manifest_field __meta__ max_hours" not in man:
    fails.append("the watchdog never reads meta.max_hours")
if "WATCHDOG_SKIPS=$((WATCHDOG_SKIPS + 1))" not in man:
    fails.append("a watchdog refusal does not reach the ledger")
if "MANIFEST INCOMPLETE" not in man or "exit 4" not in man:
    fails.append("a refused stage cannot produce a non-zero exit")
if "WATCHDOG_TRIPPED" not in man or "exit 3" not in man:
    fails.append("an elapsed-time trip does not stop the ladder")
if not stages["natural_spec_gated_512k"].get("watchdog_hours"):
    fails.append("the 512k stage declares no watchdog override")

# --- the finish-before-watchdog arithmetic, on the manifest's real numbers ----
def refuse(elapsed, stage, limit):
    return elapsed + stages[stage]["est_hours"] > limit

if not refuse(3.0, "natural_spec_gated_512k", 20):
    fails.append("512k was not refused under a 20h watchdog")
if refuse(0.0, "natural_spec_gated_512k", 40):
    fails.append("512k was refused even with its own 40h watchdog")
if not refuse(19.0, "natural_f1_128k", 20):
    fails.append("a stage projecting past the watchdog was allowed to start")
if refuse(1.0, "gate_calibration", 20):
    fails.append("a short early stage was refused")

# --- every stage's timeout must clear its own projection by >=20% ------------
# The header and the projection are two numbers describing the same run, and a
# header below the projection kills the stage the model says fits. They live in
# one bash table and one YAML section, so the relation is checked rather than
# assumed. The 256k rung carried 12h against a 13.0h projection before this.
import re as _re
man = open("scripts/mlsys_manifest.sh").read()


def _table(name):
    """Parse a `NAME='\n id value\n ...'` block out of the manifest."""
    block = man.split(f"{name}='", 1)[1].split("\n'", 1)[0]
    out = {}
    for line in block.strip().splitlines():
        parts = line.split()
        if len(parts) == 2:
            out[parts[0]] = float(parts[1])
    return out


headers = _table("STAGE_TIMEOUTS")
proj = _table("STAGE_EST_HOURS")
if not headers:
    fails.append("no stage timeouts are declared in STAGE_TIMEOUTS")
if not proj:
    fails.append("no cost projections are declared in STAGE_EST_HOURS")
checked = 0
for name, tmo_s in sorted(headers.items()):
    est = proj.get(name)
    if est is None and any(name.endswith("_" + suf) for suf in (
            "losslessness", "doc_intervals", "round_acceptance")):
        est = 0.05          # helpers share the parent's helper bucket
    if est is None:
        fails.append(f"{name}: header declared but no projection to check it "
                     f"against")
        continue
    checked += 1
    if tmo_s < est * 3600 * 1.2:
        fails.append(f"{name}: header {tmo_s/3600:.1f}h does not clear its "
                     f"{est:.1f}h projection by 20% "
                     f"(needs {est*1.2:.2f}h)")
if checked < 15:
    fails.append(f"only {checked} headers were checked against a projection")
# The call sites must all resolve: `stage()` refuses a header that is missing or
# non-numeric, so a launcher that names a stage nobody priced stops the run --
# this is the check that the two lists have not drifted apart.
called = set(_re.findall(r'stage_timeout\s+([a-z_0-9]+)', man))
# Stages launched through a loop pass "$name"; only the loops that actually call
# `stage_timeout "$name"` bind such a header, and their lists are right there.
for chunk in man.split("\nfor ")[1:]:
    head, _, body = chunk.partition("do")
    if 'stage_timeout "$name"' not in body[:400]:
        continue
    called.update(m.group(1) for m in _re.finditer(r'"(\w+):', head))
undeclared = sorted(c for c in called if c not in headers)
if undeclared:
    fails.append(f"the manifest launches stages with no declared timeout: "
                 f"{undeclared}")
if "natural_spec_gated_256k 64800" not in man:
    fails.append("the 256k stage header is not 64800 (18h)")
if "no_declared_timeout" not in man:
    fails.append("stage() does not fail closed on a missing header")
print(f"  | headers checked against the cost model: {checked} stages")

# --- every exit path must terminate the instance ------------------------------
if "trap terminate_and_confirm EXIT" not in watch:
    fails.append("no EXIT trap: an early exit would orphan the instance")
if "exit 130" not in watch or "exit 143" not in watch:
    fails.append("SIGINT/SIGTERM would not terminate the instance")
if "terminate_and_confirm; exit 4" not in watch:
    fails.append("the never-became-active path does not terminate")
# a plain sleep defers the trap until it expires, so none may remain
for line in watch.splitlines():
    t = line.strip()
    if t.startswith("sleep ") and "</dev/null" not in t:
        fails.append("plain sleep would defer the trap: " + t)

for f in fails:
    print("  check failed: " + f)
sys.exit(1 if fails else 0)
PYDRY
  [ $? -eq 0 ] && ok "watchdog, ledger and terminate-on-every-exit are wired" \
               || bad "a run guard is missing or its arithmetic is wrong"

# The allowlist must outrank the cost guard. MLSYS_APPROVED_STAGES alone is a COST
# guard, so a stage cheap enough to sit under the threshold still runs unapproved
# -- with two ids "approved", ten of twelve stages would have run. Reuse the
# manifest's own guard functions rather than restating the rule here.
sed -n '/^MANIFEST_STAGE_IDS=/,/^approved() { on_list/p' scripts/mlsys_manifest.sh > "$WORK/guards.sh"
if [ -s "$WORK/guards.sh" ]; then
  ( MANIFEST=configs/mlsys_manifest.yml
    . "$WORK/guards.sh"
    ONLY="gate_calibration,engine_cap_smoke"; APPROVED="$ONLY"; ASK_OVER=300
    on_list gate_calibration "$ONLY" || { echo "  allowed stage refused"; exit 1; }
    on_list engine_cap_smoke  "$ONLY" || { echo "  allowed stage refused"; exit 1; }
    # cheap but unapproved: the cost guard would let this through
    on_list natural_f1_128k "$ONLY" && { echo "  unapproved stage allowed"; exit 1; }
    on_list vllm_ladder "$ONLY" && { echo "  unapproved stage allowed"; exit 1; }
    # a real stage id must match exactly, so approving the parent must not admit
    # the _diverse sibling
    on_list natural_f1_128k_diverse "natural_f1_128k" && { echo "  prefix leaked"; exit 1; }
    # but a derived sub-stage of an approved parent must follow it
    on_list natural_f1_128k_losslessness "natural_f1_128k" || { echo "  sub-stage blocked"; exit 1; }
    exit 0 ) 2>/dev/null
  [ $? -eq 0 ] && ok "allowlist outranks the cost guard; unapproved stages are refused" \
               || bad "the allowlist let an unapproved stage through (or blocked an approved one)"
else bad "could not extract the guard functions from the manifest"; fi

# --------------------------------------------------------------------------
step "13  plan/code contracts: tolerance, pre-registration and control baselines"
"$PY" - <<'PYPLAN'
import json, re, sys, pathlib
plan = pathlib.Path("docs/mlsys_analysis_plan.md").read_text()
gate = pathlib.Path("scripts/mlsys_coherence_gate.py").read_text()
fails = []

# --- the tolerance is fixed in ONE place, and plan and code agree ------------
m = re.search(r"^PPL_TOLERANCE\s*=\s*([0-9.]+)", gate, re.M)
if not m:
    fails.append("the gate declares no PPL_TOLERANCE")
else:
    tol = float(m.group(1))
    if abs(tol - 1.5) > 1e-9:
        fails.append(f"PPL_TOLERANCE is {tol}, but the plan revision fixes 1.5")
    if "FIXED at 1.5" not in plan:
        fails.append("the plan has no revision fixing the tolerance at 1.5")
    # The revision quotes the withdrawn rule verbatim, so look only at the
    # normative list item that stated the threshold was unset.
    for line in plan.splitlines():
        if line.strip().startswith("- Gate threshold:"):
            if "1.5" not in line:
                fails.append("the plan's gate-threshold item does not fix 1.5: "
                             + line.strip()[:70])
            break
    else:
        fails.append("the plan has no 'Gate threshold:' item")

# --- every gate control must have a declared baseline at its own context -----
spec = json.loads(pathlib.Path("configs/mlsys_gate_controls.json").read_text())
cands = spec["candidates"] if isinstance(spec, dict) else spec
baselines = [(c["context_length"], c["target_model_name"]) for c in cands
             if c.get("native_baseline")]
for c in cands:
    key = (c["context_length"], c["target_model_name"])
    if key not in baselines:
        fails.append(
            f"control {c['name']} has no declared native baseline at its own "
            f"context {c['context_length']} for {c['target_model_name']}")

# --- the new stage is pre-registered before it is implemented ----------------
if "rope_intervention_128k" in pathlib.Path("configs/mlsys_manifest.yml").read_text():
    if "rope_intervention_128k" not in plan:
        fails.append("rope_intervention_128k is in the manifest but not "
                     "pre-registered in the plan (the revision must precede it)")
for f in fails:
    print("  check failed: " + f)
sys.exit(1 if fails else 0)
PYPLAN
  [ $? -eq 0 ] && ok "tolerance fixed in one place, every control has a baseline, new stage pre-registered" \
               || bad "a plan/code contract is broken"

# --------------------------------------------------------------------------
step "14  launch blockers: fail-closed guards and the watcher's lifecycle"
MAN=scripts/mlsys_manifest.sh
WATCH=scripts/mlsys_watch_and_run.sh
"$PY" - "$MAN" "$WATCH" <<'PYBLK'
import sys, re, yaml
man = open(sys.argv[1]).read()
watch = open(sys.argv[2]).read()
m = yaml.safe_load(open("configs/mlsys_manifest.yml"))
fails = []

# 1. the allowlist defaults to the approved set on BOTH sides
approv = ("gate_calibration,engine_cap_smoke,coherence_gate,"
          "correction_note_evidence,natural_f1_128k,impl_validation,"
          "rope_intervention_128k,natural_spec_gated_256k,vllm_ladder")
if f"DEFAULT_ONLY={approv}" not in man:
    fails.append("the manifest's allowlist default is not the approved set")
# empty must refuse, never admit
if 'if [ -z "$ONLY" ]; then' not in man:
    fails.append("an empty allowlist does not refuse stages")
if 'ONLY=${MLSYS_ONLY_STAGES-$DEFAULT_ONLY}' not in man:
    fails.append("the allowlist does not default to the approved set")

# 2. fail closed; conda python; session cap
if "sys.exit(3)" not in man or "sys.exit(7)" not in man:
    fails.append("est_cost does not fail closed on an unreadable projection")
if "no readable cost projection" not in man:
    fails.append("the runner does not refuse a stage with no cost estimate")
if re.search(r"(?m)^\s*python3 - ", man) or re.search(r"(?m)\bpython3 scripts/", man):
    fails.append("the manifest still calls bare python3")
if re.search(r"(?m)^\s*python3 ", open("scripts/mlsys_dry_run.sh").read()):
    fails.append("the dry run itself calls bare python3")
if m["meta"].get("max_cost_usd") != 850:
    fails.append("no $850 session cap in the manifest")
if "MAX_COST" not in man:
    fails.append("the session cap is not enforced")
if "MLSYS_MAX_COST_USD:-850" not in man:
    fails.append("the manifest's default session cap is not $850")
if "MLSYS_MAX_COST_USD=${MLSYS_MAX_COST_USD:-850}" not in watch:
    fails.append("the watcher does not pass the $850 ceiling")

# 3/4. control baselines and the fixed tolerance are checked in step 13

# 5. the rope reference anchor (checked by the pytest suite as well)
if "original_max_position_embeddings" not in open(
        "scripts/mlsys_coherence_gate.py").read():
    fails.append("the gate no longer reads the shipped anchor")

# 6. a gate that clears nothing must be a hard failure. Look at the CODE, not
# the comments: the comments name the flag to explain why it is gone.
man_code = "\n".join(l for l in man.splitlines()
                     if not l.strip().startswith("#"))
if "--allow-empty" in man_code:
    fails.append("a gated stage can still silently skip when the gate clears "
                 "no candidate")
if "reason=no_gate_passing_config" not in man:
    fails.append("no hard failure when the gate clears no candidate")

# 7. per-rung per-run timeouts
for rung in ("131072", "262144", "524288"):
    if rung not in man:
        fails.append(f"no per-run timeout for rung {rung}")
if "--timeout-per-run-s" not in man:
    fails.append("no --timeout-per-run-s is passed")
if man.count("--timeout-per-run-s") < 4:
    fails.append("not every stage passes --timeout-per-run-s")
smoke = man[man.index("stage engine_cap_smoke"):]
if "--timeout-per-run-s" not in smoke[:400]:
    fails.append("engine_cap_smoke passes no per-run timeout")

# 8. a stage passes only on valid rows
if "--abort-on-failure" not in man or man.count("--abort-on-failure") < 3:
    fails.append("--abort-on-failure is not passed to every runner stage")
if "check_stage_rows" not in man:
    fails.append("no row-count check on stage output")
if "STAGE_INVALID" not in man:
    fails.append("an invalid stage does not reach the ledger")

for f in fails:
    print("  check failed: " + f)
sys.exit(1 if fails else 0)
PYBLK
  [ $? -eq 0 ] && ok "launch blockers: allowlist, fail-closed cost, cap, timeouts, rows, watcher lifecycle" \
               || bad "a launch blocker is unfixed"

# Behavioural, not textual: est_cost must FAIL on a stage that is not in the
# manifest, rather than reporting 0 (which made every stage look free).
( MANIFEST=configs/mlsys_manifest.yml; PY="$PY"
  eval "$(sed -n '/^est_cost() {/,/^}/p' scripts/mlsys_manifest.sh)"
  if est_cost no_such_stage >/dev/null 2>&1; then
    echo "  est_cost returned success for an unknown stage"
    exit 1
  fi
  est_cost gate_calibration >/dev/null 2>&1 || {
    echo "  est_cost failed for a real stage"; exit 1; } ) 2>/dev/null
[ $? -eq 0 ] && ok "est_cost fails closed on an unknown stage and succeeds on a real one" \
             || bad "est_cost does not fail closed"

# Behavioural: an empty allowlist must refuse EVERY stage.
( MANIFEST=configs/mlsys_manifest.yml; PY="$PY"; ONLY=""
  eval "$(sed -n '/^MANIFEST_STAGE_IDS=/,/^approved() { on_list/p' scripts/mlsys_manifest.sh)"
  for s in gate_calibration engine_cap_smoke natural_f1_128k; do
    if [ -n "$ONLY" ] || on_list "$s" "$ONLY"; then
      echo "  empty allowlist admitted $s"; exit 1
    fi
  done; exit 0 ) 2>/dev/null
[ $? -eq 0 ] && ok "an empty allowlist admits no stage" \
             || bad "an empty allowlist still admits stages"

# --------------------------------------------------------------------------
step "15  the watcher's lifecycle: refuse, verify, terminate-to-zero"
"$PY" - scripts/mlsys_watch_and_run.sh <<'PYWBLK'
import re, sys
watch = open(sys.argv[1]).read()
# The pod side of the stall watchdog lives in the manifest; the two halves are
# asserted together because a watchdog that exists on only one side is not the
# guarantee the plan claims.
man = open("scripts/mlsys_manifest.sh").read()
fails = []
approv = ("gate_calibration,engine_cap_smoke,coherence_gate,"
          "correction_note_evidence,natural_f1_128k,impl_validation,"
          "rope_intervention_128k,natural_spec_gated_256k,vllm_ladder")

# the allowlist and the interpreter, on the watcher side
if "MLSYS_ONLY_STAGES" not in watch:
    fails.append("the watcher never passes MLSYS_ONLY_STAGES to the pod")
if approv not in watch:
    fails.append("the watcher's allowlist default is not the approved set")
if re.search(r"(?m)\bpython3 -c", watch):
    fails.append("the watcher still calls bare python3")

# refuse rather than adopt. The opt-in escape hatch is gone ENTIRELY, so the
# check is now the absence of any adoption path, not the presence of a gate:
# asserting the string existed was satisfied by a comment, which is how the
# previous version of this check passed while proving nothing.
if "already exist" not in watch or "exit 5" not in watch:
    fails.append("the watcher does not refuse to start when an instance exists")
if "MLSYS_ADOPT_EXISTING" in watch:
    fails.append("an adoption opt-in is back; the watcher would terminate an "
                 "instance it did not launch")
if re.search(r'INSTANCE_ID=\$\(api_get instances', watch):
    fails.append("an existing instance is adopted into INSTANCE_ID")

# TERMINATED means confirmed
term = watch[watch.index("terminate_and_confirm()"):]
term = term[:term.index("\n}") + 2]
if "TERMINATED=1" not in term:
    fails.append("terminate_and_confirm never sets TERMINATED")
elif 'if [ "$confirmed" = "1" ]; then' not in term:
    fails.append("TERMINATED=1 is not guarded by the API confirmation")
if "NOT CONFIRMED TERMINATED" not in term:
    fails.append("an unconfirmed termination is not reported")
if "re-issuing terminate" not in term:
    fails.append("termination is not retried")

# ssh failure is UNKNOWN. Both failure shapes must be distinguished from a
# completion: an unreachable pod (ssh exits non-zero) and a readable pod whose
# marker cannot be read. Neither may break the wait, and neither may be read as
# "the manifest finished" -- that would pull, and terminate, mid-run.
loop = watch[watch.index("MANIFEST_RC="):]
loop = loop[:loop.index('interruptible_sleep "$MANIFEST_POLL"')]
if not re.search(r'if \[ "\$rc" -ne 0 \]; then', loop):
    fails.append("an ssh failure is not routed to the UNKNOWN state")
if "state UNKNOWN" not in loop:
    fails.append("an ssh failure is still treated as the manifest finishing")
if not re.search(r"elif printf '%s' \"\$marker\" \| grep -qE '\^\[0-9\]\+\$'", loop):
    fails.append("a completion is not required to be a numeric exit code")
if "break" not in loop[loop.index("MANIFEST_RC=$marker"):]:
    fails.append("a completion does not end the wait")
if "MANIFEST_RC=$marker" not in loop:
    fails.append("the manifest's exit code is not carried past the wait")
# the marker must be read only under an ssh that succeeded
if loop.index('if [ "$rc" -ne 0 ]') > loop.index("MANIFEST_RC=$marker"):
    fails.append("the marker is believed before the connection is checked")
# an ABORT (non-zero rc) must be an incident: logs collected, run stopped, and
# no normal verify-and-merge over a run that already declared itself broken
if "collect_incident" not in loop:
    fails.append("an aborted manifest does not collect its logs")
if "MANIFEST_ABORTED=1" not in loop:
    fails.append("an aborted manifest is not flagged as an incident")
after = watch[watch.index('if [ "${MANIFEST_ABORTED:-0}" = "1" ];'):]
after = after[:after.index("\nfi\n") + 4]
if "terminate_and_confirm" not in after or "exit 7" not in after:
    fails.append("an incident does not terminate and exit non-zero")
if watch.index('if [ "${MANIFEST_ABORTED:-0}" = "1" ];') > watch.index("per-run staged pull"):
    fails.append("the incident path runs after the normal pull, so it is not fail-fast")
# the stall watchdog: no new progress line, or the process gone with no marker
if "PROGRESS_SINCE" not in loop or "stall_minutes_for" not in loop:
    fails.append("the operator side has no stall detection")
if "no completion marker" not in loop:
    fails.append("a manifest that vanished without writing rc is not detected")
if loop.count("collect_incident") < 2:
    fails.append("a stall or a vanished manifest does not collect its logs")
if 'pgrep -f "[b]ash scripts/mlsys_manifest.sh"' not in loop:
    fails.append("the liveness pattern can match the shell that runs it")
if "mlsys_stall_thresholds.json" not in watch or "mlsys_stall_thresholds.json" not in man:
    fails.append("the two watchdogs do not share one threshold table")
if "stall_watchdog &" not in man or 'interim "STAGE_START name=' not in man:
    fails.append("the pod has no self-enforced stall watchdog")
# The byte-level liveness rule, on BOTH sides. The per-stage thresholds only
# move at stage boundaries, so a hang inside a stage is invisible to them: that
# is how the 2026-10-09T00:00Z probe deadlock cost 62 minutes (~$43) unnoticed.
if '"liveness_minutes"' not in open("configs/mlsys_stall_thresholds.json").read():
    fails.append("no liveness_minutes in the shared threshold table")
if "MANIFEST_LOG" not in man or 'stat -c %s "$MANIFEST_LOG"' not in man:
    fails.append("the pod does not measure byte-level liveness on manifest.log")
if "LIVENESS name=" not in man:
    fails.append("a liveness stall is not distinguishable from a stage stall")
if 'stat -c %s "$f"' not in watch or "logbytes=" not in watch:
    fails.append("the operator's probe does not carry the pod's manifest.log size")
if "LIVE_MIN * 60" not in loop:
    fails.append("the operator side does not apply the liveness rule")

# pull integrity: the merge is delegated to a script that refuses to copy an
# unverified pull, and the watcher's failure paths mark the result clearly.
if "rsync FAILED" not in watch or "NOT MERGED" not in watch:
    fails.append("an rsync failure does not stop the merge")
if "scripts/mlsys_pull_merge.sh" not in watch:
    fails.append("the merge is not delegated to the verified-pull script, so "
                 "the rule cannot be exercised without a GPU and an SSH session")
if 'if [ "$PULL_FAILED" = "0" ]; then' not in watch:
    fails.append("the merge is not gated on PULL_FAILED")
if "shasum -a 256" not in watch or "sha256 MISMATCH" not in watch:
    fails.append("the sha256 pull-verify the comment claims is not implemented")

# campaign clock
if "CAPACITY_WAIT_DEADLINE" not in watch:
    fails.append("the deadline is not separated from the capacity wait")
if "campaign clock starts NOW" not in watch:
    fails.append("the campaign deadline does not start at acquisition")
if "MLSYS_CAMPAIGN_HOURS:-40" not in watch:
    fails.append("the campaign deadline is not sized to the campaign (~40h)")

for f in fails:
    print("  check failed: " + f)
sys.exit(1 if fails else 0)
PYWBLK
  [ $? -eq 0 ] && ok "the watcher refuses, verifies and terminates to zero" \
               || bad "a watcher lifecycle guarantee is missing"

# --------------------------------------------------------------------------
step "16  rope_intervention_128k: pre-registered, gated, plannable"
"$PY" - <<'PYRI'
import json, sys, yaml, pathlib
fails = []
plan = pathlib.Path("docs/mlsys_analysis_plan.md").read_text()
man = pathlib.Path("configs/mlsys_manifest.yml").read_text()
shell = pathlib.Path("scripts/mlsys_manifest.sh").read_text()

# pre-registered BEFORE it appears in the manifest (it does)
if "rope_intervention_128k" not in plan:
    fails.append("the stage is not pre-registered in the plan")
if "equivalence margin: 0.05" not in plan.lower():
    fails.append("the pre-registered equivalence margin of 0.05 is missing")
if "document-bootstrap" not in plan:
    fails.append("the comparison is not the pre-registered document bootstrap")

cfg = yaml.safe_load(open("configs/mlsys_rope_intervention_128k.yml"))
groups = [k for k in cfg if k != "defaults"]
if groups != ["RI_native_128k_SPEC", "RI_llama3_f16_128k_SPEC",
              "RI_llama3_f32_128k_SPEC"]:
    fails.append(f"unexpected arms: {groups}")
lv = {g: cfg[g]["levels"][0] for g in groups}
native = lv["RI_native_128k_SPEC"]
treated_arms = [lv["RI_llama3_f16_128k_SPEC"], lv["RI_llama3_f32_128k_SPEC"]]

if "rope_type" in native:
    fails.append("the control arm declares a rope, so it is not the shipped one")
for t in treated_arms:
    # everything that is not the rope must be identical to the control
    for key in set(native) | set(t):
        if key in ("rope_type", "rope_factor", "rope_anchor_base") or \
           key in ("id", "rope_arm", "name", "notes"):
            continue
        if native.get(key) != t.get(key):
            fails.append(f"a treated arm differs from the control in {key}, "
                         f"so the rope is not isolated")
    if t.get("rope_type") != "llama3":
        fails.append("a treated arm is not a llama3 configuration")
    if int(t.get("rope_anchor_base", 0)) != 8192:
        fails.append("a treated arm is not anchored where the model ships "
                     "(8192), so the factor is not the only change")
    if len(t.get("documents", [])) != 10:
        fails.append("a treated arm does not use the 10 core documents")
factors = sorted(int(t["rope_factor"]) for t in treated_arms)
if factors != [16, 32]:
    fails.append(f"expected factors 16 and 32, got {factors}")
# every arm is labelled, or the analysis cannot tell the speculative arms apart
for g, lv0 in lv.items():
    if not lv0.get("rope_arm"):
        fails.append(f"{g} carries no rope_arm label")
if "rope_arm" not in open("run_experiment.py").read():
    fails.append("rope_arm never reaches the CSV")
if int(cfg["defaults"].get("temperature", -1)) != 0.0:
    fails.append("the stage is not greedy")
if cfg["defaults"].get("ignore_eos") is not True:
    fails.append("the stage does not fix the EOS policy")

# no target-only arm and no losslessness requirement: nothing here is a ratio
for g in groups:
    if "TARGET" in g:
        fails.append(f"a target-only arm is present: {g}")

# the gate runs first, EACH treated arm is gated on its own, and the native arm
# always runs
i_gate = shell.find("rope_intervention_gate")
i_stage = shell.find("stage rope_intervention_128k ")
if i_gate < 0 or i_stage < 0:
    fails.append("the stage or its gate invocation is missing")
elif i_gate > i_stage:
    fails.append("the arms run before the gate, so an incoherent target can run")
if "for arm in" not in shell or "gate_pass \"$cand\"" not in shell:
    fails.append("the treated arms are not gated independently")
if 'RI_GROUPS="RI_native_128k_SPEC"' not in shell:
    fails.append("the native control is not unconditionally included")
if "RESULT name=rope_intervention_128k arm=$label gate=FAIL" not in shell:
    fails.append("a gate failure is not recorded as THAT arm's result")
# the comparison is the pre-registered paired difference, not a ratio
if "mlsys_rope_intervention.py" not in shell:
    fails.append("the native-vs-each-factor comparison stage is missing")
if "--margin 0.05" not in shell:
    fails.append("the comparison does not apply the pre-registered 0.05 margin")

# the gate candidates file declares a baseline at the candidate's own context
j = json.load(open("configs/mlsys_rope_intervention_candidates.json"))
base = {(c["context_length"], c["target_model_name"])
        for c in j["candidates"] if c.get("native_baseline")}
for c in j["candidates"]:
    if (c["context_length"], c["target_model_name"]) not in base:
        fails.append(f"{c['name']} has no baseline at its own context")

# approved on both sides
if "rope_intervention_128k" not in man:
    fails.append("the stage has no manifest entry")
for f in ("scripts/mlsys_manifest.sh", "scripts/mlsys_watch_and_run.sh"):
    if "rope_intervention_128k" not in open(f).read():
        fails.append(f"{f} does not list the stage as approved")

for f in fails:
    print("  check failed: " + f)
sys.exit(1 if fails else 0)
PYRI
  [ $? -eq 0 ] && ok "rope_intervention_128k is pre-registered, isolated, gated first and approved" \
               || bad "the new stage is not as pre-registered"

# --------------------------------------------------------------------------
step "17  the Codex round: helper pricing, gate references, grouping, rehearsal"
# --------------------------------------------------------------------------
"$PY" - scripts/mlsys_manifest.sh scripts/mlsys_watch_and_run.sh <<'PYBLK'
import json, re, sys, yaml
man = open(sys.argv[1]).read()
watch = open(sys.argv[2]).read()
m = yaml.safe_load(open("configs/mlsys_manifest.yml"))
fails = []

# --- helpers are priced, and priced under their parent ---------------------
helper_ids = [h["id"] for h in m.get("helpers", [])]
if not helper_ids:
    fails.append("the manifest declares no helper stages")
for h in m.get("helpers", []):
    has_cost = (h.get("est_cost_usd") is not None
                or h.get("est_usd") is not None)
    parent_cost = any(s.get("id") == h.get("parent") and s.get("est_cost_usd")
                      for s in m.get("stages", []))
    if not (has_cost or parent_cost):
        fails.append(f"helper {h['id']} has no cost and no priced parent, so it "
                     f"cannot be refused-or-approved on cost")
# a helper whose parent was refused must skip its own validation AND not count
# as an invalid stage: the refusal path returns 9 before any STAGE_INVALID
# bookkeeping, and the call sites `continue` on 9.
if 'SKIPPED_VALIDATION name=$name' not in man:
    fails.append("a refused stage does not skip its validation")
# Every spec stage's validation must be gated on the stage having RUN and
# SUCCEEDED in this attempt. The earlier shape only recognised a refusal (exit
# 9), so a stage that failed -- any other non-zero code -- went straight on to
# be validated and reported.
if "validate_and_report() {" not in man:
    fails.append("there is no validate_and_report() gate for spec stages")
if man.count("  validate_and_report ") < 3:
    fails.append("not every spec stage routes its validation through the gate")
if "stage_ok rope_intervention_128k" not in man:
    fails.append("the rope-intervention stage is not gated on stage_ok")
for var in ('"$src_rc" = "9"', '"$ri_rc" = "9"', '"$?" = "9"'):
    if var in man:
        fails.append(f"a call site still recognises only the refusal code "
                     f"({var}), so a failed stage would be validated anyway")
if 'stage_ok "$name"' not in man:
    fails.append("stage_ok is not used by the validation gate")

# --- the gate reference policy --------------------------------------------
for f in ("configs/mlsys_rope_candidates.json", "configs/mlsys_gate_controls.json",
          "configs/mlsys_rope_intervention_candidates.json",
          "configs/mlsys_correction_candidates.json"):
    d = json.load(open(f))
    key = "candidates" if "candidates" in d else "coherence_gate"
    for c in d[key]:
        if not c.get("target_revision"):
            fails.append(f"{f}: {c['name']} has no pinned target_revision")
    if "rope_candidates" in f or "correction" in f:
        for c in d[key]:
            if int(c["context_length"]) > 131072 and c.get("reference_context") != 131072 \
                    and not c.get("native_baseline"):
                fails.append(f"{f}: {c['name']} is an extension with no declared "
                             f"reference_context, so it would be judged against "
                             f"its own context")
    if "correction" in f:
        if not any(c.get("native_baseline") for c in d[key]):
            fails.append(f"{f}: no declared baseline, so the stage can produce "
                         f"no ratio at all")
# controls keep same-context baselines
d = json.load(open("configs/mlsys_gate_controls.json"))
bases = {(c["context_length"], c["target_model_name"])
         for c in d["candidates"] if c.get("native_baseline")}
for c in d["candidates"]:
    if c.get("expect") and c.get("role") != "baseline" \
            and (c["context_length"], c["target_model_name"]) not in bases:
        fails.append(f"control {c['name']} has no baseline at its own context")

# --- the campaign clock is one number -------------------------------------
if "MLSYS_MAX_HOURS=${MLSYS_CAMPAIGN_HOURS:-40}" not in watch:
    fails.append("MLSYS_MAX_HOURS is not the campaign clock")
if 'MLSYS_CAMPAIGN_HOURS:-40' not in watch:
    fails.append("the campaign clock default is not 40h")

# --- the paired bootstrap groups by the SHARED stratum ---------------------
if "--group-by level_id" in man:
    fails.append("the bootstrap groups by level_id, which separates the arms "
                 "so no pair can form")
if "--group-by context_length" not in man:
    fails.append("the bootstrap does not group by the shared stratum")
boot = open("scripts/mlsys_document_bootstrap.py").read()
if "paired_speedup" not in boot or "no paired" not in boot.lower():
    fails.append("the bootstrap does not fail when no pair formed")

# --- vLLM rows are pinned per model, and the prompt ids are ids ------------
vllm = open("scripts/mlsys_vllm_baseline.py").read()
if "--target-revisions" not in man:
    fails.append("the manifest does not pin a revision per model for vLLM")
if '"prompt_token_ids"' not in vllm:
    fails.append("the vLLM worker does not pass prompt_token_ids directly")
if "prompt_ids_verified" not in vllm:
    fails.append("the vLLM row does not record whether it consumed the ids given")

# --- the rehearsal exists and is wired as a dry-run step ------------------
import os
for f in ("scripts/mlsys_rehearsal.sh",
          "scripts/rehearsal/stub_run_experiment.py",
          "scripts/rehearsal/stub_coherence_gate.py",
          "scripts/rehearsal/stub_vllm_baseline.py",
          "scripts/mlsys_pull_merge.sh"):
    if not os.path.exists(f):
        fails.append(f"{f} is missing")
if os.path.exists("scripts/mlsys_rehearsal.sh") and \
        not os.access("scripts/mlsys_rehearsal.sh", os.X_OK):
    fails.append("the rehearsal is not executable")

for f in fails:
    print("  check failed: " + f)
sys.exit(1 if fails else 0)
PYBLK
  [ $? -eq 0 ] && ok "helpers priced, gate references declared, grouping shared, rehearsal present" \
               || bad "the Codex-round contracts are not in place"

echo "  running the end-to-end rehearsal (stubbed GPU, default allowlist)..."
if bash scripts/mlsys_rehearsal.sh > "$WORK/rehearsal.log" 2>&1; then
  ok "the rehearsal passed: $(grep -c '^  ok' "$WORK/rehearsal.log") checks, every approved stage and helper ran"
else
  bad "the rehearsal FAILED"
  grep '^  FAIL' "$WORK/rehearsal.log" | head -10 | sed 's/^/  | /'
  tail -5 "$WORK/rehearsal.log" | sed 's/^/  | /'
fi

# --------------------------------------------------------------------------
step "18  freshness, the row identity and the reference labels"
# --------------------------------------------------------------------------
"$PY" - scripts/mlsys_manifest.sh scripts/mlsys_watch_and_run.sh \
       run_experiment.py <<'PYBLK'
import json, os, re, sys, yaml


def read(path: str) -> str:
    """Read a file that the fix is supposed to have created.

    A missing file is a FAILURE of this check, not a crash: the point of the
    step is to say what is not in place, and a traceback says nothing.
    """
    try:
        return open(path).read()
    except OSError:
        globals().setdefault("_missing", []).append(path)
        return ""


man = read(sys.argv[1])
watch = read(sys.argv[2])
rexp = read(sys.argv[3])
fails = list(f"{p} does not exist" for p in globals().get("_missing", []))


def idx(text: str, needle: str) -> int:
    """Position, or -1. `str.index` raises, and a check script that crashes on
    the pre-fix tree reports nothing about what is missing."""
    return text.find(needle)

# --- 1. a stage's exit code reaches its callers, and freshness is tracked ----
if "  return $rc" not in man:
    fails.append("stage() does not return the command's exit code")
if "stage_ok() {" not in man or ".attempt" not in man:
    fails.append("there is no stage_ok() gate on the per-attempt marker")
if "archive_attempt() {" not in man:
    fails.append("stage outputs are not archived per attempt")
if "  archive_attempt \"$name\"" not in man:
    fails.append("stage() does not archive before running")
i_arch = idx(man, "  archive_attempt \"$name\"")
i_cmd = idx(man, '  if timeout "$tmo" "$@"; then rc=0; else rc=$?; fi')
if i_arch != -1 and i_cmd != -1 and i_arch > i_cmd:
    fails.append("the archive happens AFTER the stage runs, which is too late")
if "--exclude 'attempts/'" not in watch:
    fails.append("attempts/ is not excluded from the results pull, so a previous "
                 "attempt's CSVs would enter the delivered corpus")
# prerequisites read the recorded code, not just the file
for st in ("gate_calibration", "engine_cap_smoke", "coherence_gate"):
    if f'cat "$RAN_DIR/{st}.rc"' not in man:
        fails.append(f"the {st} prerequisite check does not read its exit code")
# dependencies
if "prereqs_of() {" not in man or "prerequisite_not_ok" not in man:
    fails.append("a stage does not refuse when its prerequisite did not succeed")

# --- 2. row counts ---------------------------------------------------------
if "reason=planner_produced_no_rows" not in man:
    fails.append("a planner that produces 0 rows is not a stage failure")
if '[ "$want" = "0" ] && return 0' in man:
    fails.append("check_stage_rows still treats an expected 0 as 'skip'")
if "NF>=4 && $3 ~ /^[0-9]+$/" not in man:
    fails.append("the run-line counter does not exclude the 'N runs total.' "
                 "trailer, so every expected count is one too high")
if "mlsys_row_identity_check.py" not in man:
    fails.append("the sequence identity is not checked on every stage's rows")
if 'seq = metrics.get("sequence_tokens")' not in rexp:
    fails.append("the row does not take the ENGINE's measurement of the "
                 "sequence length")
if 'row["sequence_tokens"] = (int(row["prompt_tokens"]) + 1' in rexp:
    fails.append("the row computes sequence_tokens from the fields the identity "
                 "check compares it against, which makes the check a tautology")
engine_src = read("src/models/rasd_inference.py")
if engine_src.count("sequence_len = int(generated_ids.shape[1])") != 2:
    fails.append("the engine does not measure the sequence length off the final "
                 "sequence tensor in both generation paths")
if "from the sidecar" not in read("scripts/mlsys_row_identity_check.py") \
        and "no token sidecar" not in read("scripts/mlsys_row_identity_check.py"):
    fails.append("the identity check does not take the emitted count from the "
                 "sidecar ids")

# --- 3. the allowlist is consulted before any gate filtering ---------------
for tag in ("natural rungs", "synthetic rungs"):
    pass
if man.count("reason=needs_approval (before gate filter)") != 2:
    fails.append("the gated loops do not check the allowlist before gate_filter, "
                 "so an excluded stage does filtering work and records an invalid")
i_gated = idx(man, "for gated in ")
i_filter = idx(man, "mlsys_gate_filter.py", ) if i_gated == -1 else \
    idx(man[i_gated:], "mlsys_gate_filter.py")
i_allow = -1 if i_gated == -1 else idx(man[i_gated:], 'on_list "$name" "$ONLY"')
if i_gated != -1 and i_filter != -1 and i_allow != -1 and i_allow > i_filter:
    fails.append("the 512k/synthetic allowlist check comes after the gate filter")

# --- 4. helpers need the parent's cost approval ---------------------------
if "return 3                                # parent over the threshold" not in man:
    fails.append("a helper does not require its parent's cost approval")
if "reason=parent_over_threshold_not_cost_approved" not in man:
    fails.append("the helper cost refusal is not reported")

# --- 5. reference labels --------------------------------------------------
gate = read("scripts/mlsys_coherence_gate.py")
if "FIELDS = [" not in gate:
    fails.append("the gate script has no FIELDS declaration")
else:
    fields = gate.split("FIELDS = [")[1].split("]")[0]
    for col in ("reference_role", "baseline_role", "role", "expect"):
        if f'"{col}"' not in fields:
            fails.append(f"the gate CSV has no {col} column")
try:
    corr = json.load(open("configs/mlsys_correction_candidates.json"))
except OSError:
    corr = {"candidates": []}
    fails.append("configs/mlsys_correction_candidates.json is missing")
names = {c["name"]: c for c in corr["candidates"]}
for n in ("B_llama2_unscaled_ood_32k", "B_llama2_unscaled_ood_128k"):
    if n not in names:
        fails.append(f"{n} is missing: the 32k/128k references must be labelled "
                     f"as unscaled-OOD references")
    elif names[n].get("reference_role") != "unscaled_ood_reference":
        fails.append(f"{n} is not labelled unscaled_ood_reference")
if "B_llama2_native_4096" not in names:
    fails.append("the in-distribution reference (Llama-2 native at 4096) is "
                 "missing")
elif names["B_llama2_native_4096"].get("reference_role") != \
        "in_distribution_reference":
    fails.append("the 4096 row is not labelled in_distribution_reference")
for stale in ("B_llama2_native_32k", "B_llama2_native_128k"):
    if stale in names:
        fails.append(f"{stale} still calls an out-of-distribution reference "
                     f"'native'")
plan = read("docs/mlsys_analysis_plan.md")
if "unscaled_ood_reference" not in plan:
    fails.append("the plan does not state the reference labels")

# --- 5b. the BOS in the vLLM comparison, and the correction pairing --------
vllm_src = read("scripts/mlsys_vllm_baseline.py")
if "engine_input_ids" not in vllm_src:
    fails.append("the vLLM comparison does not use the engine's input ids, so it "
                 "replays a prompt without the BOS")
if "prompt_ids_from_engine" not in vllm_src:
    fails.append("a vLLM row does not record whether its ids came from the engine")
if "prompt_tokens + 1" not in vllm_src and "int(ptok) + 1" not in vllm_src:
    fails.append("the vLLM loader does not check the BOS accounting of the "
                 "engine ids against prompt_tokens")
if "engine_input_ids" not in rexp:
    fails.append("run_experiment does not put engine_input_ids in the sidecar")
if "engine_input_ids" not in read("src/models/rasd_inference.py"):
    fails.append("the engine does not report the ids it actually fed")
if "engine_input_ids" not in read("scripts/rehearsal/stub_run_experiment.py"):
    fails.append("the stub sidecars lack engine_input_ids, so the rehearsal "
                 "cannot exercise the BOS rule")

gate_src = read("scripts/mlsys_coherence_gate.py")
for needle, why in (
        ("def pairing_verdict", "the gate has no pairing verdict"),
        ("continuation_sha256", "the gate does not hash the continuation"),
        ("UNPAIRED", "an unpaired reference does not refuse the ratio"),
        ("cross_context", "an extension is not labelled cross-context"),
        ("def gate_sample", "the gate has no single window call path"),
        ("bos_id=getattr(tok, ", "the gate's window call does not pass the BOS"),
):
    if needle not in gate_src:
        fails.append(why)

# --- 5c. freshness markers and the run-id set -----------------------------
if "attempt.$(date -u +%Y%m%dT%H%M%SZ).$$" not in man:
    fails.append("the marker directory is not per-invocation, so a re-run "
                 "inherits the previous attempt's .rc files")
if "plan.ids" not in man:
    fails.append("the planned run ids are not written, so only the COUNT is "
                 "checked")
if "reason=run_id_set" not in man:
    fails.append("check_stage_rows does not compare the run_id set")

# --- 6. the stub shares the real code paths -------------------------------
stub = read("scripts/rehearsal/stub_run_experiment.py")
if "_round_commit_plan" not in stub:
    fails.append("the stub does not build traces through the engine's planner")
if "real.format_dry_run" not in stub:
    fails.append("the stub prints its own dry-run table instead of the real one")
if "real.apply_cli_to_runs" not in stub:
    fails.append("the stub carries its own copy of the CLI propagation")
if "def _propagate" in stub:
    fails.append("the stub still has its own propagation copy")
if "--abort-on-failure" in stub and "abort_on_failure" not in stub:
    fails.append("the stub ignores --abort-on-failure")
if "MLSYS_REHEARSAL_FAIL_RUN" not in stub:
    fails.append("the stub cannot inject a failing run, so the abort path is "
                 "never exercised")
vstub = read("scripts/rehearsal/stub_vllm_baseline.py")
if "MLSYS_REHEARSAL_VLLM_IDS" not in vstub:
    fails.append("the vLLM stub cannot produce a mismatched or unavailable id "
                 "report, so unit_matched is never exercised")

# --- 7. the rehearsal keeps the cases that prove the above ---------------
reh = read("scripts/mlsys_rehearsal.sh")
for want, why in (
        ("per-attempt freshness", "the freshness case is missing"),
        ("the vLLM worker's reported ids", "the vLLM id cases are missing"),
        ("the row identity holds on every stage's rows",
         "the row-identity case is missing"),
        ("MLSYS_REHEARSAL_FAIL_RUN=", "no injected-failure case"),
        ("build_row set prompt_ids_from_engine",
         "the rehearsal does not assert that the PARENT sets the engine marker"),
):
    if want not in reh:
        fails.append(why)

for f in fails:
    print("  check failed: " + f)
sys.exit(1 if fails else 0)
PYBLK
  [ $? -eq 0 ] && ok "stage freshness, strict row counts, reference labels and the shared stub paths are all in place" \
               || bad "the second Codex round's contracts are not in place"

# --- 8. the parent owns the engine marker, and the ladder actually retries --
# The worker cannot report `prompt_ids_from_engine` (it never sees the RASD
# sidecar), so the parent must set it. A parent that merely copied the worker's
# fields left the marker empty on EVERY row and no vLLM row could ever be
# unit-matched. And a ladder whose execution sits outside the `for` loop builds
# every spec and runs only the last one: the 64k fallback, reported as the 128k
# rung.
"$PY" - <<'PYBLK'
import ast, pathlib, sys
fails = []
vsrc = pathlib.Path("scripts/mlsys_vllm_baseline.py").read_text()
tree = ast.parse(vsrc)
fn = next((n for n in tree.body if isinstance(n, ast.FunctionDef)
           and n.name == "build_row"), None)
if fn is None:
    fails.append("the vLLM parent has no build_row; the row path is inlined "
                 "again and the marker can only come from the worker")
else:
    body = ast.get_source_segment(vsrc, fn) or ""
    if 'row["prompt_ids_from_engine"]' not in body:
        fails.append("build_row does not set prompt_ids_from_engine from the cell")
main = next((n for n in tree.body if isinstance(n, ast.FunctionDef)
             and n.name == "main"), None)
keys = {k.value for n in ast.walk(main) if isinstance(n, ast.Dict)
        for k in n.keys if isinstance(k, ast.Constant)}
if "prompt_ids_from_engine" in keys:
    fails.append("the worker spec still carries prompt_ids_from_engine, which "
                 "the worker cannot know: it is a fact about the sidecar")
if "no row is unit_matched=yes" not in vsrc:
    fails.append("the parent exits 0 with zero unit-matched rows, so a stage "
                 "with no usable baseline is recorded ok")
if "shorter-context fallback" not in vsrc:
    fails.append("a 64k ladder fallback can be certified as unit_matched=yes "
                 "at 128k, putting a 64k throughput in the 128k column")
if '"max_model_len"' not in vsrc:
    fails.append("the row does not record the model length it ran at, so a "
                 "fallback is indistinguishable from the rung")
lines = vsrc.splitlines()
ind = lambda s: len(s) - len(s.lstrip())          # noqa: E731
i_loop = next((i for i, l in enumerate(lines)
               if "for idx, att in enumerate(ATTEMPT_LADDER" in l), None)
i_run = next((i for i, l in enumerate(lines)
              if "run_attempt(spec, log_path" in l), None)
if i_loop is None or i_run is None:
    fails.append("the attempt ladder is gone from the vLLM parent")
elif ind(lines[i_run]) <= ind(lines[i_loop]):
    fails.append("the attempt execution is outside the ladder loop: only the "
                 "last attempt (the 64k fallback) would ever run")
vstub = pathlib.Path("scripts/rehearsal/stub_vllm_baseline.py").read_text()
if "vb.build_row" not in vstub:
    fails.append("the vLLM stub does not use the production row path, so the "
                 "rehearsal cannot catch a parent that never sets the marker")
if '"prompt_ids_from_engine"' in vstub:
    fails.append("the vLLM stub still sets prompt_ids_from_engine itself, which "
                 "is what hid the parent's bug")
gc = pathlib.Path("configs/mlsys_gate_controls.json").read_text()
if "DIFFERENT seed" in gc:
    fails.append("the gate controls still claim their baselines use a different "
                 "seed; the references share their candidate's sample")
for f in fails:
    print("  check failed: " + f)
sys.exit(1 if fails else 0)
PYBLK
  [ $? -eq 0 ] && ok "the parent sets the engine marker and the attempt ladder retries" \
               || bad "the parent/ladder contracts are not in place"

# --- 19. this round's contracts: window, tie rule, KV truth, venv ------------
"$PY" - <<'PYX'
import sys
fails = []

def read(p):
    try:
        return open(p).read()
    except OSError as e:
        fails.append(f"cannot read {p}: {e}")
        return ""

eng = read("src/models/rasd_inference.py")
# B1: ONE place applies the draft window, and it keeps the recent tokens.
if "_draft_window(" not in eng:
    fails.append("the draft window is not a single named rule")
# The removed construct, specifically: the tokenizer call that kept the FIRST
# `draft_max_len` tokens. (A bare "truncation=True" also appears in the
# docstring explaining why it went, so it cannot be the needle.)
if "max_length=self.draft_max_len" in eng:
    fails.append("a second, right-truncating draft input is back: the draft "
                 "would be conditioned on the OPENING of the prompt")
if "_draft_window(raw_draft_ids" not in eng:
    fails.append("generate() does not apply the single draft-window rule")
if "draft_input_ids=None" not in eng:
    fails.append("generate_text still builds its own draft input")
# B4: the gap is measured, in both arms, and reaches the sidecar.
if "_top1_top2_gap" not in eng or eng.count("_top1_top2_gap(") < 3:
    fails.append("the target's top1-top2 gap is not recorded in both arms")
if 'metrics["token_gaps"]' not in eng:
    fails.append("the gaps never leave the engine")
rexp = read("run_experiment.py")
if '"token_gaps"' not in rexp:
    fails.append("the token sidecar does not carry the gaps")
loss = read("src/analysis/losslessness.py")
if "NUMERIC_TIE" not in loss or "TIE_GAP" not in loss:
    fails.append("the losslessness verdict has no tie rule")
if "_gap_at" not in loss:
    fails.append("the tie rule does not read the gaps")
if "NUMERIC_TIE" not in read("scripts/mlsys_losslessness.py"):
    fails.append("the losslessness stage does not report ties")
# R1: the tail fold, and R3: the KV length read back from the cache.
cache = read("src/models/nf4_dynamic_cache.py")
if "_append_nf4_chunk" not in cache or "tail_merge_below" not in cache:
    fails.append("the NF4 tail fold is missing: per-step cost grows with steps")
if "_kv_seq_len(" not in eng:
    fails.append("kv_len_after is not read back from the cache")
if 'rec["kv_len_after"] = int(prior_target_len + committed)' in eng:
    fails.append("kv_len_after is again the arithmetic the truncation was asked "
                 "to perform")
# R4: the consensus broadcast includes the draft tokens.
if "_broadcast_round_state" not in eng:
    fails.append("the round-state broadcast is not a single named rule")
# R7: the baseline runs in its own interpreter, selected by path.
man = read("scripts/mlsys_manifest.sh")
if "vllm_python()" not in man:
    fails.append("the vLLM interpreter is not selected by path")
if 'stage impl_validation' not in man or "VLLM_PY" not in man:
    fails.append("impl_validation does not use the isolated interpreter")
if man.count("vllm_python; do") + man.count("$(vllm_python)") < 2:
    fails.append("only one vLLM stage selects the isolated interpreter")
if "no_vllm_venv" not in man:
    fails.append("a missing vLLM venv does not refuse the stage")
vsrc = read("scripts/mlsys_vllm_baseline.py")
if "compare_targets" not in vsrc or "--rasd-target-sidecars" not in vsrc:
    fails.append("impl_validation is not the target cross-check")
if "NOT_COMPARABLE_PRECISION" not in vsrc:
    fails.append("the cross-check scores token agreement between engines that "
                 "do not share numerics (FP4 weights + NF4 KV vs bf16)")
for col in ("weight_precision", "kv_dtype"):
    if f'"{col}"' not in vsrc:
        fails.append(f"the cross-check does not record {col}, so a divergence "
                     f"between the two engines has no stated cause")
if 'usable = [x for x in out_rows if x.get("unit_matched") == "yes"]' not in vsrc:
    fails.append("the cross-check's exit code is not decided by the throughput "
                 "unit match")
if "RASD requires CUDA." not in eng:
    fails.append("the engine no longer states its CUDA requirement, so the "
                 "cap smoke's GPU-side alignment check has no stated reason")
smoke = read("scripts/mlsys_cap_smoke_check.py")
if "len(gaps) != len(ids)" not in smoke:
    fails.append("the cap smoke does not assert gap/ids alignment on every row")
if 'spec_gaps=a.get("token_gaps")' not in smoke or \
        'target_gaps=b.get("token_gaps")' not in smoke:
    fails.append("the cap smoke does not pass both arms' gaps into the verdict")
if "_step_gap(" not in eng or "seed_gap" not in eng:
    fails.append("the seed token's gap is not recorded")
if "round_gaps" not in eng:
    fails.append("the spec arm does not record gaps on every round")
if "logprobs=2" not in vsrc:
    fails.append("the vLLM side does not record its own top-2 gap, so the tie "
                 "rule cannot be applied to its output")
if "_position_gaps" not in vsrc:
    fails.append("vLLM's per-position gaps are not extracted")
if "def compare_targets" in vsrc and "load_rasd_target_sidecars" not in vsrc:
    fails.append("the cross-check cannot find the RASD target-only sidecars")
if "spec_steps" not in vsrc.split("def load_rasd_target_sidecars")[1][:900]:
    fails.append("the cross-check does not restrict itself to target-only "
                 "sidecars (spec_steps == 0)")
if "seen_cells" not in vsrc:
    fails.append("the prompt-sidecar loader does not dedup a document, so the "
                 "spec and target-only sidecars launch it twice")
vvenv = read("scripts/mlsys_vllm_venv.sh")
if "vllm==$VLLM_PIN" not in vvenv and "vllm==${VLLM_PIN}" not in vvenv:
    fails.append("the provisioning script does not pin vLLM")
if "torch" not in vvenv:
    fails.append("the provisioning script does not install its own torch")
reh = read("scripts/mlsys_rehearsal.sh")
for want, why in (
        ("the vLLM reference: throughput decides",
         "the vLLM reference case is missing"),
        ("MLSYS_REHEARSAL_VLLM_DIVERGE", "no divergence knob"),
        ("NOT_COMPARABLE_PRECISION",
         "the precision verdict is not rehearsed"),
        ("no unit-matched row -> the stage fails",
         "the no-usable-reference case is not rehearsed"),
):
    if want not in reh:
        fails.append(why)
for f in fails:
    print("  check failed: " + f)
sys.exit(1 if fails else 0)
PYX
  [ $? -eq 0 ] && ok "window, tie rule, KV truth, tail fold and the isolated venv are all in place" \
               || bad "this round's contracts are not in place"


# --- 20. measured numerics, the tie rule on the smoke, resume, the ladder ----
"$PY" - <<'PYX'
import re, sys, yaml, pathlib
fails = []

def read(p):
    try:
        return open(p).read()
    except OSError as e:
        fails.append(f"cannot read {p}: {e}")
        return ""

eng = read("src/models/rasd_inference.py")
rexp = read("run_experiment.py")
smoke = read("scripts/mlsys_cap_smoke_check.py")
vs = read("scripts/mlsys_vllm_baseline.py")
man = open("scripts/mlsys_manifest.sh").read()
plan = read("docs/mlsys_analysis_plan.md")

# 1. the precision columns come from the loaded state, not the config
for fn in ("def detect_weight_precision", "def detect_kv_precision",
           "def numerics_report", "def normalize_dtype_name"):
    if fn not in eng:
        fails.append(f"{fn} missing: the precision columns have no measured source")
if "is_loaded_in_4bit" not in eng:
    fails.append("the weight label does not read the model's 4-bit state")
if "bnb_4bit_quant_type" not in eng:
    fails.append("the weight label cannot distinguish fp4 from nf4 storage")
if 'metrics["numerics"] = numerics_report(' not in eng:
    fails.append("the engine does not measure the numerics of a run")
if eng.count('metrics["numerics"] = numerics_report(') != 2:
    fails.append("the numerics are not measured in BOTH arms")
if re.search(r'"weight_precision":\s*\(run\.get', rexp):
    fails.append("run_experiment derives weight_precision from the config again")
if re.search(r'"kv_dtype":\s*\("nf4" if run\.get', rexp):
    fails.append("run_experiment derives kv_dtype from the config again")
if "numerics = metrics.pop(\"numerics\"" not in rexp:
    fails.append("run_experiment does not take the MEASURED numerics")
if "**numerics," not in rexp:
    fails.append("the measured numerics never reach the sidecar")
for soft in ("bf16", "bfloat16"):
    if soft not in eng:
        fails.append(f"dtype normalization does not handle {soft!r}")

# the rehearsal must go through the same production detector
stub = read("scripts/rehearsal/stub_run_experiment.py")
if "numerics_report(" not in stub:
    fails.append("the rehearsal stub does not use the production numerics "
                 "detector, so it could pass while the detector was broken")
if '"_real_run_experiment"' in stub and "real.numerics_report" in stub:
    fails.append("the stub calls numerics_report on the wrong module")

# 2. the cap smoke asserts the STRUCTURAL contract and reports the teacher-forced
#    numbers without gating on them (analysis plan 6.1f). The probe gate was
#    withdrawn, not re-thresholded: its 8-rank numbers came from a forward the
#    engine never runs (the probe fed the full prompt to a model that is fed the
#    rank's slice), and a gate on a wrong input is not repaired by a bigger
#    tolerance. Token identity stays REPORTED for the same reason it always was:
#    the two arms run the target in different forward shapes.
for needle, why in (
        ("TOL_BF16 = 1.875",
         "the reference tolerance is missing; the reported numbers lose their "
         "scale"),
        ('DECLARED_KV = ("nf4", "bfloat16")',
         "the declared KV set is missing"),
        ('if kvd not in DECLARED_KV', "the KV dtype is not asserted per row"),
        ('if wp != "fp4"', "the weight precision is not asserted per row"),
        ("REPORT ONLY",
         "the teacher-forced numbers are not marked as a report"),
        ("no measured precision", "a row with no measured precision is not a "
         "failure"),
):
    if needle not in smoke:
        fails.append(f"the cap smoke: {why} (missing {needle!r})")
# REPORT ONLY means nothing about the probe can fail the stage. Assert the
# absence of the machinery that did, because restoring it would re-arm the
# campaign on numbers measured through the wrong input.
for needle, why in (
        ("GATED_KV", "the gate constant is back; the probe would stop stages "
         "again"),
        ("teacher-forced MISMATCH", "the probe can fail the stage again"),
        ("UNVERIFIED, which is not a pass",
         "a missing probe would fail the stage again, and with the probe "
         "disabled EVERY row would fail"),
):
    if needle in smoke:
        fails.append(f"the cap smoke: {why} (found {needle!r})")
# ... and no cap-smoke row may request the probe, or the stage would pay its wall
# clock for numbers nothing can act on.
if "teacher_forced_check" in read("configs/mlsys_engine_cap_smoke.yml"):
    fails.append("an engine_cap_smoke row still requests the disabled probe")
    if needle not in smoke:
        fails.append(f"the cap smoke: {why} (missing {needle!r})")
if "no measured precision" not in smoke:
    fails.append("a row with no measured precision is not a failure")
if 'spec_gaps=a.get("token_gaps")' not in smoke:
    fails.append("the cap smoke no longer passes both arms' gaps into the "
                 "reported identity verdict")

# 3. the ladder is Llama-3.1-8B at bf16 only, at the recomputed cost
if "--models meta-llama/Llama-3.1-8B meta-llama/Llama-2-7b-hf" in man:
    fails.append("the ladder still sweeps Llama-2")
mv = read("configs/mlsys_manifest.yml")
try:
    d = yaml.safe_load(mv)
    lad = next(s for s in d["stages"] if s["id"] == "vllm_ladder")
except Exception as e:                                   # noqa: BLE001
    fails.append(f"cannot read the ladder's cost entry: {e}")
    lad = {}
if lad.get("models") != ["meta-llama/Llama-3.1-8B"]:
    fails.append(f"the ladder's models are {lad.get('models')!r}")
if lad.get("quantizations") != ["bfloat16"]:
    fails.append(f"the ladder's quantizations are {lad.get('quantizations')!r}")
if int(lad.get("est_cost_usd") or 0) != 28:
    fails.append(f"the ladder's cost is {lad.get('est_cost_usd')!r}, not 28")
if float(lad.get("est_hours") or 0) != 1.25:
    fails.append(f"the ladder's projection is {lad.get('est_hours')!r}, not 1.25")
if "NARROWED (plan revision, 2026-10-07)" not in mv:
    fails.append("the narrowing is not stated in the manifest's cost entry")
if "vllm_ladder is Llama-3.1-8B at bf16" not in plan:
    fails.append("the plan does not carry the dated narrowing revision")

# 4. the alignment raise is deferred past the last collective
i_check = rexp.index("alignment_error = \"\"")
i_ppl = rexp.rindex("_measure_target_ppl(")   # the CALL, not the definition
i_raise = rexp.index("if alignment_error:\n            raise RuntimeError")
if not (i_check < i_ppl < i_raise):
    fails.append("the gap-alignment raise is not after the perplexity "
                 "collective: rank 0 would strand the other ranks")
if "LAST COLLECTIVE IS DONE" not in rexp:
    fails.append("the deferral is not explained where it happens")
if "tok_sidecar = None" not in rexp or "if not alignment_error:" not in rexp:
    fails.append("a misaligned gap array would still be written to a sidecar")

# 5. resume is refused, and no campaign config can reach it
if "resume refused" not in eng:
    fails.append("generate does not refuse a found checkpoint")
if "ckpt.past_kv" in eng or "ckpt.n_rounds" in eng:
    fails.append("an unreachable resume-restore branch is still present")
if "if ckpt is None:" in eng:
    fails.append("the prefill is still guarded as if a resume path existed")
for p in sorted(pathlib.Path("configs").glob("mlsys_*.yml")):
    try:
        doc = yaml.safe_load(p.read_text())
    except Exception as e:                               # noqa: BLE001
        fails.append(f"{p.name}: {e}")
        continue
    vals = []
    def walk(x):
        if isinstance(x, dict):
            for k, v in x.items():
                if k == "checkpoint_every":
                    vals.append(v)
                walk(v)
        elif isinstance(x, list):
            for v in x:
                walk(v)
    walk(doc)
    nz = [v for v in vals if int(v or 0) != 0]
    if nz:
        fails.append(f"{p.name} sets checkpoint_every={nz}; a leftover "
                     f"checkpoint would make the stage refuse")

for f in fails:
    print("  check failed: " + f)
sys.exit(1 if fails else 0)
PYX
  [ $? -eq 0 ] && ok "measured numerics, the smoke's tie rule, the narrowed ladder, deferred abort and the resume refusal" \
               || bad "this round's contracts are not in place"


# --- 21. the remote setup provisions the isolated vLLM environment ----------
"$PY" - <<'PYX'
import sys
fails = []
watch = open("scripts/mlsys_watch_and_run.sh").read()
venv = open("scripts/mlsys_vllm_venv.sh").read()
man = open("scripts/mlsys_manifest.sh").read()
podenv = open("scripts/mlsys_pod_env.sh").read()

if "mlsys_vllm_venv.sh" not in watch:
    fails.append("the watcher does not provision the vLLM venv, so the vLLM "
                 "stages would refuse on a freshly booted pod")
if watch.index("mlsys_vllm_venv.sh") > watch.index("MLSYS_ONLY_STAGES="):
    fails.append("the venv is provisioned after the manifest starts")
if "nohup bash scripts/mlsys_vllm_venv.sh" not in watch:
    fails.append("the venv provisioning does not run detached, so a multi-GB "
                 "install would block the campaign clock")
# The path the venv script creates must be the path the manifest looks for.
if "$REPO/.venv-vllm/bin/python" not in man:
    fails.append("the manifest does not look for $REPO/.venv-vllm/bin/python")
if "{MLSYS_VLLM_VENV:-$REPO/.venv-vllm}" not in venv:
    fails.append("the venv script's default path differs from the manifest's")
for want, why in (("vllm==$VLLM_PIN", "the pin is not applied"),
                  ("torch==", "no torch is installed into the venv"),
                  ("python3-venv", "a venv failure gives no actionable hint")):
    if want not in venv:
        fails.append(f"mlsys_vllm_venv.sh: {why}")

# --- the CAMPAIGN environment -------------------------------------------
# The 2026-10-08 run died in gate_calibration because the pod had no conda and
# the interpreter resolution fell back to a bare python3. Both halves are pinned.
if "mlsys_pod_env.sh" not in watch:
    fails.append("the watcher never provisions the campaign environment, so a "
                 "pod without one fails in its first stage")
elif watch.index("mlsys_pod_env.sh") > watch.index('say "starting the manifest"'):
    fails.append("the campaign environment is provisioned after the manifest starts")
else:
    prov = watch[watch.index("mlsys_pod_env.sh"):watch.index('say "starting the manifest"')]
    if "terminate_and_confirm" not in prov or "exit 6" not in prov:
        fails.append("a failed campaign provisioning leaves a paid instance running")
    if "import torch, transformers, bitsandbytes, flash_attn, diptest" not in prov:
        fails.append("the stage dependencies are never proven to import")
import re as _re
# The pod interpreter: a resolver that PRINTS a path (not a `$( ... )'
# expression, which ssh would execute as the whole command and which returns
# nothing), looked up under the env name the bootstrap actually creates.
_rpy = watch[watch.index("RPY_RESOLVE='"):]
_rpy = _rpy[:_rpy.index("exit 1") + len("exit 1")]
if _re.match(r"RPY_RESOLVE='\$\(", _rpy) or "$(" in _rpy:
    fails.append("the interpreter resolver is a command substitution, so passing "
                 "it to ssh as a command runs the path with no arguments")
if "command -v python3" in _rpy:
    fails.append("the pod interpreter still falls back to a bare python3")
if "rasd-gpu" not in _rpy:
    fails.append("the interpreter resolution does not look for the env the "
                 "bootstrap creates")
if 'PY_REMOTE=$(ssh $SSH_OPTS "$SSH_USER@$IP" "$RPY_RESOLVE"' not in watch:
    fails.append("the resolver is not run as a command that prints")
if "INTERPRETER=" not in watch or "INTERPRETER=" not in podenv:
    fails.append("the provisioning script does not report the interpreter it built")
if "RPY=$PY_REMOTE" not in watch:
    fails.append("the resolved interpreter is not reused as a plain path")
if _re.search(r"pgrep -f mlsys_manifest", "\n".join(
        l for l in watch.splitlines() if not l.lstrip().startswith("#"))):
    fails.append("the completion check is still a self-matching pgrep")
# The interpreter the stages use must be decided in ONE place. The 07:37Z run
# passed every setup check and then died on `No module named 'transformers'`
# because the manifest re-derived its own answer.
if "MLSYS_PYTHON" not in man:
    fails.append("the manifest ignores MLSYS_PYTHON")
else:
    _i = man.index("resolve_python()")
    _body = man[_i:man.index("\n}", _i)]
    if _body.index("MLSYS_PYTHON") > _body.index("miniconda3/envs"):
        fails.append("the manifest guesses an interpreter before it checks MLSYS_PYTHON")
    if "rasd-gpu" not in _body:
        fails.append("the manifest looks only for `rasd`, this Mac's env name")
if "MLSYS_PYTHON='$PY_REMOTE'" not in watch:
    fails.append("the watcher does not pass the interpreter it verified")
# results/mlsys must not travel outbound: 575 local artifacts used to land on
# the pod, including a stale RUN_LOG.txt that came back looking like pod output
_r = watch.index('rsync -az --no-perms --no-owner --no-group -e "ssh $SSH_OPTS"')
_rb = watch[_r:watch.index('"$REPO/" "$SSH_USER@$IP:~/RASD/"', _r)]
if "--exclude 'results/mlsys/*'" not in _rb:
    fails.append("local results/mlsys is still pushed to the pod")
if "--include 'results/mlsys/gpu_hours.csv'" not in _rb:
    fails.append("the cumulative cost ledger no longer travels")
if "echo \\$? > ~/manifest.rc" not in watch:
    fails.append("the manifest's exit code is not recorded")
for want, why in (("conda create -n", "the env is never created"),
                  ("requirements-lock.txt", "the pins are not the locked ones"),
                  ("--no-build-isolation", "flash-attn cannot build"),
                  ("diptest", "diptest is not in the lock file"),
                  ("torch.cuda.is_available", "CUDA is never proven"),
                  # No W&B credential reaches the pod, so wandb must be told not
                  # to ask for one. Its absence killed engine_cap_smoke 24 s in
                  # on 2026-10-08, before any GPU work, and would have killed
                  # every stage that runs run_experiment.
                  ("WANDB_MODE=disabled",
                   "wandb will ask for a credential the pod does not have")):
    if want not in podenv:
        fails.append(f"mlsys_pod_env.sh: {why}")
for f in fails:
    print("  check failed: " + f)
sys.exit(1 if fails else 0)
PYX
  [ $? -eq 0 ] && ok "the remote setup provisions the campaign env and the isolated vLLM venv, and fails closed" \
               || bad "the remote setup does not provision the environments it needs"


printf "\n\033[1m== DRY RUN RESULT: %d passed, %d failed ==\033[0m\n" "$PASS" "$FAIL"
[ "$FAIL" -eq 0 ] || exit 1
echo "every stage of the pipeline ran end to end on a tiny model."

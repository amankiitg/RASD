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
python3 - scripts/mlsys_manifest.sh scripts/mlsys_watch_and_run.sh <<'PYDRY'
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
          "natural_spec_gated_256k,vllm_ladder")
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
if m["meta"].get("max_cost_usd") != 700:
    fails.append("no $700 session cap in the manifest")
if "MAX_COST" not in man:
    fails.append("the session cap is not enforced")

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
fails = []
approv = ("gate_calibration,engine_cap_smoke,coherence_gate,"
          "correction_note_evidence,natural_f1_128k,impl_validation,"
          "natural_spec_gated_256k,vllm_ladder")

# the allowlist and the interpreter, on the watcher side
if "MLSYS_ONLY_STAGES" not in watch:
    fails.append("the watcher never passes MLSYS_ONLY_STAGES to the pod")
if approv not in watch:
    fails.append("the watcher's allowlist default is not the approved set")
if re.search(r"(?m)\bpython3 -c", watch):
    fails.append("the watcher still calls bare python3")

# refuse rather than adopt
if "already exist" not in watch or "exit 5" not in watch:
    fails.append("the watcher does not refuse to start when an instance exists")
if "MLSYS_ADOPT_EXISTING" not in watch:
    fails.append("adopting an existing instance is not gated on an explicit opt-in")

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

# ssh failure is UNKNOWN
if '"$rc" -eq 1' not in watch:
    fails.append("the manifest-running check does not distinguish ssh failure")
if "state UNKNOWN" not in watch:
    fails.append("an ssh failure is still treated as the manifest finishing")

# pull integrity
if "rsync FAILED" not in watch or "NOT merged" not in watch:
    fails.append("an rsync failure does not stop the merge")
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

printf "\n\033[1m== DRY RUN RESULT: %d passed, %d failed ==\033[0m\n" "$PASS" "$FAIL"
[ "$FAIL" -eq 0 ] || exit 1
echo "every stage of the pipeline ran end to end on a tiny model."

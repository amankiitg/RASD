#!/usr/bin/env bash
# One-box REPRODUCTION of the gate's CUDA abort. NOT part of the campaign.
#
# This script exists so the 1x repro is separable from the 8x campaign in the
# strongest way available: the manifest must not read it, and a test asserts
# that (tests/test_mlsys_gate_tokenizer_vocab.py::test_the_manifest_never_reads
# _the_1x_repro). Nothing here may become a fitting decision for the campaign --
# no context lengths, no sharding, no device maps, no dtypes.
#
# What it reproduces, measured 2026-10-08 on gpu_8x_a100_80gb_sxm4 and then on
# gpu_1x_a100_sxm4:
#
#   ../aten/src/ATen/native/cuda/Indexing.cu:1308: indexSelectLargeIndex:
#     Assertion `srcIndex < srcSelectDimSize` failed.
#   reported at torch.cuda.empty_cache()
#
#   failing op: modeling_llama.py:891  inputs_embeds = self.embed_tokens(input_ids)
#
# The gate's prompt came from data/processed/pg19_docs/documents.json, whose own
# metadata records "tokenizer": "meta-llama/Llama-3.1-8B". Candidate
# B_llama2_yarn8_32k_correct is Llama-2-7B, vocab_size 32000: 2391 of its 31744
# prompt ids (7.5%) were >= 32000, first at index 3, max 127146.
#
# Usage, on a pod provisioned by scripts/mlsys_pod_env.sh:
#   bash scripts/mlsys_gate_repro_1x.sh B_llama2_yarn8_32k_correct
#   bash scripts/mlsys_gate_repro_1x.sh B_llama2_yarn8_32k_correct N2_llama2_misanchored
#
# With no arguments it runs the two candidates that were involved, plus the two
# baselines that make their ratios computable. A candidate with no declared
# baseline at its own context is deliberately UNJUDGEABLE, so the gate refuses
# to write a CSV for a lone non-baseline row ("refusing to compute ratios
# against a baseline nobody declared") -- that refusal is by design, not a bug.
#
# The gate is invoked with the campaign's OWN candidate file, read-only, and
# its own default single-device map. The only thing set here is
# CUDA_LAUNCH_BLOCKING, which changes when an error is reported and never what
# is computed.
set -uo pipefail

PY=${MLSYS_PYTHON:-python}
META=${MLSYS_REPRO_PG19_META:-data/processed/pg19_docs/documents.json}
CONTROLS=${MLSYS_REPRO_CONTROLS:-configs/mlsys_gate_controls.json}
OUT=${MLSYS_REPRO_OUT:-/tmp/mlsys_gate_repro}

mkdir -p "$OUT"

# The candidates to run. Defaults: the two that were involved, plus the two
# baselines their ratios need. Every name must exist in $CONTROLS.
NAMES=("$@")
if [ ${#NAMES[@]} -eq 0 ]; then
  NAMES=(B_llama2_yarn8_32k_correct N2_llama2_misanchored
         B_native_128k N1_arm4_f2_construction)
fi

CAND_JSON="$OUT/candidates.json"
"$PY" - "$CONTROLS" "$CAND_JSON" "${NAMES[@]}" <<'PYGEN' || exit 2
import json, sys
controls, out, names = sys.argv[1], sys.argv[2], sys.argv[3:]
c = json.load(open(controls))["candidates"]
by = {x["name"]: x for x in c}
missing = [n for n in names if n not in by]
if missing:
    sys.exit(f"not in {controls}: {missing}\navailable: {sorted(by)}")
json.dump({"candidates": [by[n] for n in names]}, open(out, "w"), indent=1)
print("candidates:", ", ".join(names))
PYGEN

echo
echo "=== running the gate (CUDA_LAUNCH_BLOCKING=1) ==="
CUDA_LAUNCH_BLOCKING=1 "$PY" scripts/mlsys_coherence_gate.py \
  --candidates "$CAND_JSON" \
  --pg19-meta "$META" \
  --out "$OUT/gate.csv" \
  --gen-dir "$OUT/generated"
rc=$?
echo "gate rc=$rc"

if [ -s "$OUT/gate.csv" ]; then
  echo
  echo "=== rows (the fields that decide the abort) ==="
  "$PY" - "$OUT/gate.csv" <<'PYSHOW'
import csv, sys
cols = ("candidate", "context_length", "pool_tokenizer", "candidate_tokenizer",
        "prompt_ids_out_of_vocab", "prompt_first_out_of_vocab_index",
        "prompt_first_out_of_vocab_id", "prompt_max_id", "prompt_vocab_size",
        "ppl_continuation", "non_finite_logits", "early_eos", "gate_pass",
        "status")
for r in csv.DictReader(open(sys.argv[1])):
    print("  " + " | ".join(f"{c}={r.get(c, '')}" for c in cols))
PYSHOW
else
  echo
  echo "NO CSV. An empty row set means the run died before any row was built;"
  echo "the fields above are what would have said which candidate and why."
fi
exit "$rc"

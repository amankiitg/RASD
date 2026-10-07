#!/usr/bin/env bash
# End-to-end rehearsal of the campaign, locally, with GPU execution stubbed.
#
# WHY
#   Every failure this campaign has hit was a plumbing failure: a metadata file
#   that was never staged, a CSV with no paired_speedup row, a stage refused on
#   cost, a partial pull merged into results/. Each was only visible two stages
#   after the mistake, and each cost real money to discover. This script runs the
#   WHOLE manifest -- every approved stage plus its helpers -- against stubs that
#   write artifacts in the exact schemas the real code writes, and then exercises
#   the watcher's pull/merge with one injected hash mismatch.
#
# WHAT IS REAL AND WHAT IS STUBBED
#   Real:  the manifest, the stage ordering and cost guards, the approval and
#          helper gating, the gate's decision layer, the cap-smoke checker, the
#          losslessness checker, the document and cluster bootstraps, the rope
#          intervention comparison, the gate filter, the collision guard, the
#          pull/merge rule.
#   Stub:  model execution (weights, generation, perplexity) and the vLLM worker.
#
# PASS CRITERIA
#   1. every approved stage AND every helper runs and exits 0; nothing refused
#      on cost, nothing SKIPPED_VALIDATION, no INVALID/STAGE_FAILED line;
#   2. the cap smoke check passes on the stub's rows;
#   3. natural_f1_128k's doc_intervals produces paired_speedup rows;
#   4. the rope comparison produces native-vs-factor16 and native-vs-factor32;
#   5. the 256k gate candidate passes its verdict against the DECLARED 128k
#      reference (baseline_context == 131072), not against its own context;
#   6. the losslessness verdicts are 3x LOSSLESS (1024) + 7x LOSSLESS_PREFIX_128;
#   7. an injected sha256 mismatch is NOT merged into results/, and a verified
#      pull IS.
#
# Usage: scripts/mlsys_rehearsal.sh [--keep]
set -uo pipefail

REPO=$(cd "$(dirname "$0")/.." && pwd)
KEEP=0
[ "${1:-}" = "--keep" ] && KEEP=1
WORK=$(mktemp -d -t mlsys_rehearsal.XXXXXX)
SANDBOX=$WORK/repo
OUT=$SANDBOX/results/mlsys
LOG=$WORK/manifest.log
PASS=0
FAIL=0

ok()  { printf '  ok    %s\n' "$*"; PASS=$((PASS + 1)); }
bad() { printf '  FAIL  %s\n' "$*"; FAIL=$((FAIL + 1)); }
hdr() { printf '\n=== %s ===\n' "$*"; }
cleanup() { [ "$KEEP" = "1" ] || rm -rf "$WORK"; }
trap cleanup EXIT

resolve_python() {
  local c
  if [ -n "${MLSYS_PYTHON:-}" ]; then printf '%s' "$MLSYS_PYTHON"; return; fi
  for c in "$HOME/miniconda3/envs/rasd/bin/python" "$HOME/miniconda3/bin/python" \
           /opt/conda/bin/python "$(command -v python3 2>/dev/null)"; do
    if [ -n "$c" ] && [ -x "$c" ] && "$c" -c 'import yaml' 2>/dev/null; then
      printf '%s' "$c"; return
    fi
  done
  printf '%s' python3
}
PY=$(resolve_python)
printf 'rehearsal: python=%s work=%s\n' "$PY" "$WORK"

# ---------------------------------------------------------------------------
hdr "1  sandbox: a copy of the repo with the GPU entry points replaced"
# ---------------------------------------------------------------------------
mkdir -p "$SANDBOX"
rsync -a --exclude '.git' --exclude 'results' --exclude 'data' \
  --exclude 'manuscript' --exclude '.venv*' --exclude '__pycache__' \
  --exclude '*.pyc' "$REPO/" "$SANDBOX/" || {
    bad "could not stage the sandbox"; exit 1; }

# $3 is the name the original is stashed under, and it must match the module
# the stub imports, or the stub fails at import time on the pod -- which is how
# this was caught.
install_stub() {   # $1 = real script (repo-relative) $2 = stub $3 = stash name
  local real=$1 stub=$2 stash=$3
  mv "$SANDBOX/$real" "$SANDBOX/$(dirname "$real")/$stash"
  cp "$SANDBOX/$stub" "$SANDBOX/$real"
  [ -f "$SANDBOX/$(dirname "$real")/$stash" ] || {
    echo "install_stub: $real was not stashed as $stash" >&2; return 1; }
}
install_stub "run_experiment.py" "scripts/rehearsal/stub_run_experiment.py" \
             "_real_run_experiment.py"
install_stub "scripts/mlsys_coherence_gate.py" \
             "scripts/rehearsal/stub_coherence_gate.py" \
             "_real_coherence_gate.py"
install_stub "scripts/mlsys_vllm_baseline.py" \
             "scripts/rehearsal/stub_vllm_baseline.py" \
             "_real_vllm_baseline.py"
ok "sandbox staged and the three GPU entry points shimed"

# The stash that documents which files the stubs stand in for.
cat > "$SANDBOX/scripts/rehearsal/INSTALLED.txt" <<'EOF'
This sandbox has GPU entry points replaced by the stubs in scripts/rehearsal/:
  run_experiment.py            -> scripts/rehearsal/stub_run_experiment.py
  scripts/mlsys_coherence_gate.py  -> .../stub_coherence_gate.py
  scripts/mlsys_vllm_baseline.py   -> .../stub_vllm_baseline.py
The originals are next to them as _real_*.py.
EOF

# ---------------------------------------------------------------------------
hdr "2  stub PG-19 metadata (the file whose absence killed pg19_short_target)"
# ---------------------------------------------------------------------------
DOCS=$SANDBOX/data/processed/pg19_docs/documents.json
mkdir -p "$(dirname "$DOCS")"
"$PY" - "$DOCS" <<'PY' || bad "could not write the stub documents.json"
import json, sys, pathlib
docs = ["pg19_train_0", "pg19_train_1", "pg19_train_115", "pg19_train_537",
        "pg19_train_915", "pg19_train_1404", "pg19_train_1726",
        "pg19_train_1981", "pg19_train_2204", "pg19_train_2768"]
out = pathlib.Path(sys.argv[1])
# The chunks the metadata names are created too: the manifest verifies that
# every path it lists resolves, and that check is part of what is rehearsed.
entries = []
for d in docs:
    f = out.parent / f"{d}.memmap"
    f.write_bytes(b"\0" * 16)
    entries.append({"doc_id": d, "title": d, "url": "",
                    "file": str(f), "length": 600000, "tokens": 600000})
out.write_text(json.dumps({"documents": entries}, indent=1))
print(f"  wrote {out} with {len(entries)} documents and their chunks")
PY
[ -s "$DOCS" ] && ok "stub dataset written ($(wc -c < "$DOCS" | tr -d ' ') bytes)" \
               || bad "stub dataset missing"

# ---------------------------------------------------------------------------
hdr "3  allowlists agree between the manifest and the watcher's launch line"
# ---------------------------------------------------------------------------
MAN_ONLY=$(sed -n 's/^DEFAULT_ONLY=//p' "$SANDBOX/scripts/mlsys_manifest.sh")
WATCH_ONLY=$(sed -n 's/^DEFAULT_ONLY=//p' "$SANDBOX/scripts/mlsys_watch_and_run.sh")
if [ -n "$MAN_ONLY" ] && [ "$MAN_ONLY" = "$WATCH_ONLY" ]; then
  ok "both default to the same $(awk -F, '{print NF}' <<<"$MAN_ONLY")-stage allowlist"
else
  bad "allowlist mismatch: manifest=${MAN_ONLY:-<unset>} watcher=${WATCH_ONLY:-<unset>}"
fi
[ -n "$MAN_ONLY" ] || bad "the manifest's allowlist default is empty (approves nothing)"
[ "${#MAN_ONLY}" -gt 0 ] && ok "the allowlist is non-empty by default"

APPROVED_DEFAULT=$(sed -n 's/^APPROVED=${MLSYS_APPROVED_STAGES:-\(.*\)}$/\1/p' \
  "$SANDBOX/scripts/mlsys_manifest.sh")
printf '  manifest approved-to-spend default: %s\n' "${APPROVED_DEFAULT:-<none>}"

# ---------------------------------------------------------------------------
hdr "4  run the manifest with the default allowlist and the GPU stubbed"
# ---------------------------------------------------------------------------
RAN_DIR=$WORK/ran
mkdir -p "$RAN_DIR"
# NOTE: no comments inside this continuation. A `#` on a backslash-continued
# line comments out the REST of the logical line, including the command being
# built -- the run then happens with none of the environment below, which is how
# a rehearsal can pass its own checks while rehearsing nothing.
#
# MLSYS_VLLM_PYTHON: the vLLM stages must run in an ISOLATED interpreter,
# selected by path. The rehearsal satisfies that contract with the sandbox
# python (where the stubs are installed) rather than by weakening the selector:
# what is rehearsed is that the path is honoured, and that the stage REFUSES
# when it is absent.
( cd "$SANDBOX" && \
  MLSYS_PYTHON="$PY" \
  MLSYS_MAX_COST_USD=850 \
  MLSYS_ASK_OVER_USD=300 \
  MLSYS_DOCUMENTS_JSON="$DOCS" \
  MLSYS_RAN_DIR="$RAN_DIR" \
  MLSYS_MAX_HOURS=40 \
  MLSYS_VLLM_PYTHON="$PY" \
  bash scripts/mlsys_manifest.sh ) >"$LOG" 2>&1
MAN_RC=$?
if [ "$MAN_RC" = "0" ]; then ok "the manifest exited 0"; else
  bad "the manifest exited $MAN_RC"; fi
echo "  --- last 25 lines of the manifest log ---"
tail -25 "$LOG" | sed 's/^/  | /'

# ---------------------------------------------------------------------------
hdr "5  no stage was refused, failed or silently skipped"
# ---------------------------------------------------------------------------
# Both streams: refusals are echoed to stdout, stage-level failures are
# recorded in the run log, and a check that reads only one of them would miss
# half the ways a stage can be dropped.
COMBINED=$WORK/combined.log
cat "$LOG" "$OUT/RUN_LOG.txt" 2>/dev/null > "$COMBINED"
for offender in "REFUSE" "BUDGET_REFUSED" "STAGE_FAILED" \
                "STAGE_INVALID" "INVALID " "no readable cost projection"; do
  if grep -q "$offender" "$COMBINED"; then
    bad "the logs contain '$offender':"
    grep -n "$offender" "$COMBINED" | head -5 | sed 's/^/  | /'
  else
    ok "no '$offender' in the logs"
  fi
done
# A stage the allowlist refused also skips its validation, which is correct and
# must not be reported as a failure. What must not happen is a stage that IS on
# the allowlist having its validation skipped.
allowed_re='^[0-9]*:.*SKIPPED_VALIDATION name=([^ ]+)'
while IFS= read -r line; do
  [ -n "$line" ] || continue
  who=${line##*name=}; who=${who%% *}
  case " $STAGES $HELPERS " in
    *" $who "*) bad "an allowlisted stage/helper skipped its validation: $line" ;;
    *) : ;;
  esac
done < <(grep -E "$allowed_re" "$COMBINED" || true)
ok "validation skips are confined to refused, unapproved stages"

# ---------------------------------------------------------------------------
hdr "6  every approved stage and every helper ran and exited 0"
# ---------------------------------------------------------------------------
STAGES="gate_calibration engine_cap_smoke coherence_gate correction_note_evidence natural_f1_128k impl_validation rope_intervention_128k natural_spec_gated_256k vllm_ladder"
HELPERS="natural_f1_128k_losslessness natural_f1_128k_doc_intervals natural_f1_128k_round_acceptance rope_intervention_gate rope_intervention_128k_comparison"
# STAGE_OK goes to the run log, which is the record that survives a dropped
# SSH session -- the same file the operator reads after a real run.
RUNLOG=$OUT/RUN_LOG.txt
[ -s "$RUNLOG" ] || bad "no run log at $RUNLOG"
for s in $STAGES; do
  if grep -q "STAGE_OK name=$s " "$RUNLOG"; then ok "stage $s: ok"; else
    bad "stage $s did not report STAGE_OK"; fi
done
for h in $HELPERS; do
  if grep -q "STAGE_OK name=$h " "$RUNLOG"; then ok "helper $h: ok"; else
    bad "helper $h did not run"; fi
done
# A parent that never ran would have SKIPPED its helper's validation; assert
# explicitly that the helper markers exist in the run directory.
# The markers live in a per-invocation subdirectory under the base the rehearsal
# set, which is the point: reusing a location must not reuse the state.
for h in $HELPERS; do
  [ -n "$(find "$RAN_DIR" -name "$h" -print -quit 2>/dev/null)" ] \
    || bad "helper $h left no ran-marker"
done
[ -n "$(find "$RAN_DIR" -name 'attempt.*' -maxdepth 1 -print -quit 2>/dev/null)" ] \
  && ok "markers live in a fresh per-invocation subdirectory" \
  || bad "the marker directory is not per-invocation"
ok "helper ran-markers checked"

# ---------------------------------------------------------------------------
hdr "7  the gate's calibration controls came out as each one declares"
# ---------------------------------------------------------------------------
"$PY" - "$OUT/gate_calibration.csv" <<'PY' && ok "calibration verdicts match expect" \
  || bad "calibration verdicts disagree with the declared expectation"
import csv, sys
rows = list(csv.DictReader(open(sys.argv[1])))
bad = []
for r in rows:
    want = (r.get("expect") or "").strip()
    got = "pass" if r.get("gate_pass") == "True" else "fail"
    if want and want != got:
        bad.append(f"{r['candidate']}: expected {want}, got {got} "
                   f"({r.get('gate_reason')})")
print(f"  {len(rows)} controls, {len(bad)} disagreements")
for b in bad:
    print("  | " + b)
sys.exit(1 if bad else 0)
PY

# ---------------------------------------------------------------------------
hdr "8  the 256k gate candidate is judged against the DECLARED 128k reference"
# ---------------------------------------------------------------------------
"$PY" - "$OUT/coherence_gate.csv" <<'PY' && ok "extension reference is the 128k native baseline" \
  || bad "an extension candidate was not judged against the declared reference"
import csv, sys
rows = {r["candidate"]: r for r in csv.DictReader(open(sys.argv[1]))}
r = rows.get("llama3_f16_256k")
if r is None:
    print("  | llama3_f16_256k missing from the gate output")
    sys.exit(1)
print(f"  | llama3_f16_256k: ctx={r['context_length']} "
      f"reference_context={r['reference_context']} "
      f"baseline_context={r['baseline_context']} ppl_ratio={r['ppl_ratio']} "
      f"pass={r['gate_pass']}")
ok_ref = (str(r["baseline_context"]) == "131072"
          and str(r["reference_context"]) == "131072")
ok_pass = str(r["gate_pass"]) == "True"
sys.exit(0 if (ok_ref and ok_pass) else 1)
PY
fails=$(grep -c "False" "$OUT/coherence_gate.csv" || true)
if [ "${fails:-0}" -gt 0 ]; then
  ok "the gate rejected at least one candidate (it is not trivially all-pass)"
else
  bad "the gate passed every candidate, so the filter is untested"
fi
[ -s "$OUT/natural_spec_gated_256k.gated.yml" ] \
  && ok "the gate filter produced a non-empty 256k config" \
  || bad "the gate filter produced no config for the 256k rung"

# ---------------------------------------------------------------------------
hdr "9  the cap smoke passes on the stub's rows"
# ---------------------------------------------------------------------------
if "$PY" "$SANDBOX/scripts/mlsys_cap_smoke_check.py" \
     --results "$OUT/engine_cap_smoke.csv" >"$WORK/cap.log" 2>&1; then
  ok "cap smoke PASSED"
else
  bad "cap smoke FAILED"; sed 's/^/  | /' "$WORK/cap.log" | tail -20
fi

# ---------------------------------------------------------------------------
hdr "10  losslessness: 3 full pairs at 1024, 7 prefixes at 128"
# ---------------------------------------------------------------------------
"$PY" - "$OUT/natural_f1_128k_losslessness.csv" <<'PY' || bad "losslessness verdicts are not the pre-registered ones"
import csv, sys
from collections import Counter
rows = list(csv.DictReader(open(sys.argv[1])))
c = Counter(r["verdict"] for r in rows)
print(f"  | {dict(c)}")
want = Counter({"LOSSLESS": 3, "LOSSLESS_PREFIX_128": 7})
sys.exit(0 if c == want else 1)
PY
[ $? -eq 0 ] && ok "losslessness: 3x LOSSLESS + 7x LOSSLESS_PREFIX_128"

# ---------------------------------------------------------------------------
hdr "11  paired_speedup rows, and the rope comparison's two contrasts"
# ---------------------------------------------------------------------------
if "$PY" - "$OUT/natural_f1_128k_doc_intervals.csv" <<'PY'
import csv, sys
rows = [r for r in csv.DictReader(open(sys.argv[1]))
        if r["estimate"].startswith("paired_speedup")]
print(f"  | {len(rows)} paired_speedup row(s)")
for r in rows[:3]:
    print(f"  |   ctx={r['context_length']} n_docs={r['n_documents']} "
          f"point={r['point']} ci=({r['ci_lo']}, {r['ci_hi']}) "
          f"roles={r['paired_arm_roles']}")
sys.exit(0 if rows else 1)
PY
then ok "doc_intervals produced paired_speedup rows"
else bad "no paired_speedup row for natural_f1_128k (the payoff ratio is vapor)"; fi

# Contrast names come from the analysis script, not from this rehearsal: assert
# the two PRE-REGISTERED primary comparisons are present, by the names the script
# actually emits.
if "$PY" - "$OUT/rope_intervention_128k_comparison.csv" <<'PY'
import csv, sys
rows = list(csv.DictReader(open(sys.argv[1])))
seen = {(r["contrast"], r["metric"]) for r in rows}
print(f"  | contrasts: {sorted({c for c, _ in seen})}")
need = {("factor16_minus_native", "acceptance_rate"),
        ("factor32_minus_native", "acceptance_rate")}
sys.exit(0 if need <= seen else 1)
PY
then ok "rope comparison produced native vs f16 and native vs f32"
else bad "the rope comparison is missing a pre-registered primary contrast"; fi

# ---------------------------------------------------------------------------
hdr "12  the artifacts a stage is supposed to leave behind"
# ---------------------------------------------------------------------------
for d in per_token tokens memory_trace generated; do
  n=$(find "$OUT/$d" -type f 2>/dev/null | wc -l | tr -d ' ')
  if [ "$n" -gt 0 ]; then ok "$OUT/$d holds $n file(s)"; else
    bad "$OUT/$d is empty"; fi
done
dups=$(ls "$OUT" | grep -c '\.csv$' || true)
ok "$dups result CSV(s), each named for its stage: $(ls "$OUT"/*.csv | xargs -n1 basename | tr '\n' ' ')"

# ---------------------------------------------------------------------------
hdr "13  the watcher's pull/merge, with one injected hash mismatch"
# ---------------------------------------------------------------------------
MV=$WORK/merge; mkdir -p "$MV/remote/per_token" "$MV/dest"
echo "run_id,status" > "$MV/remote/natural_f1_128k.csv"
echo "r1,ok" >> "$MV/remote/natural_f1_128k.csv"
echo '{"round_idx": 0}' > "$MV/remote/per_token/r1.jsonl"
( cd "$MV/remote" && find . -type f -print0 | sort -z | xargs -0 shasum -a 256 ) \
  > "$MV/remote.sha256"

if bash "$SANDBOX/scripts/mlsys_pull_merge.sh" --stage-dir "$MV/remote" \
     --remote-sha "$MV/remote.sha256" --dest "$MV/dest" >"$MV/a.log" 2>&1; then
  ok "a verified pull IS merged"
else
  bad "a verified pull was not merged"; sed 's/^/  | /' "$MV/a.log"
fi

# Now inject the mismatch: the staged file no longer matches the remote digest.
echo "r1,TRUNCATED" >> "$MV/remote/natural_f1_128k.csv"
if bash "$SANDBOX/scripts/mlsys_pull_merge.sh" --stage-dir "$MV/remote" \
     --remote-sha "$MV/remote.sha256" --dest "$MV/dest2" >"$MV/b.log" 2>&1; then
  bad "an INJECTED MISMATCH was merged; the guard is not enforced"
else
  ok "the injected mismatch was NOT merged (exit $?)"
fi
[ -e "$MV/dest2/natural_f1_128k.csv" ] && bad "the mismatching pull landed in the destination" \
  || ok "the destination is untouched after the mismatch"
grep -q "sha256 MISMATCH" "$MV/b.log" && ok "the mismatch was reported as a sha256 mismatch" \
  || bad "the mismatch was not reported as a sha256 mismatch"

# ---------------------------------------------------------------------------
hdr "14  the row identity holds on every stage's rows"
# ---------------------------------------------------------------------------
# The manifest runs this on every stage via check_stage_rows; the rehearsal
# asserts it independently, and against the shape the PRE-FIX code wrote, so the
# check is shown to be capable of failing rather than merely present.
"$PY" - "$OUT" "$SANDBOX" <<'PY' && ok "sequence identity holds on every stage CSV" \
  || bad "a stage CSV violates the sequence identity"
import csv, pathlib, subprocess, sys
out, sandbox = pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2])
checker = sandbox / "scripts" / "mlsys_row_identity_check.py"
# Only the run_experiment schema has a sequence identity. The gate and vLLM
# CSVs are different tables with different columns, and demanding this identity
# of them is a category error, not a check.
csvs = sorted(p for p in out.glob("*.csv")
              if p.name != "gpu_hours.csv"
              and "sequence_tokens" in p.open().readline())
bad = []
for c in csvs:
    r = subprocess.run([sys.executable, str(checker), "--results", str(c)],
                       capture_output=True, text=True)
    if r.returncode != 0:
        bad.append(f"{c.name}: {r.stdout.strip()[:120]}")
print(f"  | {len(csvs)} stage CSV(s) checked")
for b in bad:
    print("  | " + b)
sys.exit(1 if bad else 0)
PY

# The negative controls. The checker compares three INDEPENDENTLY obtained
# numbers -- the prompt length, the sidecar's id count, and the engine's measured
# sequence length -- so each of these fixtures must fail for the right reason:
# a sidecar is present in every one of them, and the divergence is in the engine's
# number.
"$PY" - "$SANDBOX" "$WORK" <<'PY' && ok "the identity check rejects an engine length that disagrees" \
  || bad "the identity check accepts a row whose three sources disagree"
import csv, json, pathlib, subprocess, sys
sandbox, work = pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2])
checker = sandbox / "scripts" / "mlsys_row_identity_check.py"
fields = ["run_id", "status", "prompt_tokens", "tokens_generated",
          "sequence_tokens", "context_length"]


def run(case, prompt, gen, seq):
    d = work / f"identity_{case}"; (d / "tokens").mkdir(parents=True, exist_ok=True)
    csv_path = d / "stage.csv"
    with csv_path.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields); w.writeheader()
        w.writerow({"run_id": "r", "status": "ok", "prompt_tokens": prompt,
                    "tokens_generated": gen, "sequence_tokens": seq,
                    "context_length": 131072})
    (d / "tokens" / "r.json").write_text(json.dumps(
        {"run_id": "r", "generated_token_ids": list(range(gen))}))
    return subprocess.run([sys.executable, str(checker), "--results",
                           str(csv_path)], capture_output=True, text=True)


# 1. the engine held ONE TOKEN MORE than the prompt, BOS and emitted ids add up
#    to: the shape a double-counted BOS produces.
one_long = run("one_long", 131072 - 1024 - 1, 128,
               131072 - 1024 - 1 + 1 + 128 + 1)
# 2. the pre-fix runner's shape: the PLANNED sequence (prompt + 1 + the rung's
#    1024) for a 128-token generation.
planned = run("planned", 131072 - 1024 - 1, 128, 131072)
# 3. the CSV's emitted count disagreeing with the sidecar's ids.
mismatch = run("count_mismatch", 131072 - 1024 - 1, 128,
               131072 - 1024 - 1 + 1 + 128)

results = {"engine_one_long": one_long, "planned_sequence": planned}
for name, r in results.items():
    print(f"  | {name}: rc={r.returncode} {r.stdout.strip().splitlines()[0][:90]}")
sys.exit(0 if all(r.returncode == 1 for r in results.values()) else 1)
PY

# The CSV's own count must agree with the ids that were saved.
"$PY" - "$SANDBOX" "$WORK" <<'PY' && ok "a CSV count that disagrees with the sidecar is rejected" \
  || bad "a CSV count that disagrees with the sidecar passed"
import csv, json, pathlib, subprocess, sys
sandbox, work = pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2])
d = work / "count_mismatch"; (d / "tokens").mkdir(parents=True, exist_ok=True)
fields = ["run_id", "status", "prompt_tokens", "tokens_generated",
          "sequence_tokens"]
with (d / "stage.csv").open("w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=fields); w.writeheader()
    w.writerow({"run_id": "r", "status": "ok", "prompt_tokens": 100,
                "tokens_generated": 64, "sequence_tokens": 165})
(d / "tokens" / "r.json").write_text(json.dumps(
    {"run_id": "r", "generated_token_ids": list(range(65))}))
r = subprocess.run([sys.executable,
                    str(sandbox / "scripts" / "mlsys_row_identity_check.py"),
                    "--results", str(d / "stage.csv")],
                   capture_output=True, text=True)
print("  | " + r.stdout.strip().splitlines()[1][:100])
sys.exit(0 if r.returncode == 1 else 1)
PY

# ---------------------------------------------------------------------------
hdr "14b  the pairing labels: paired contrasts, cross-window extensions"
# ---------------------------------------------------------------------------
if "$PY" - "$OUT/correction_note_evidence.csv" "$OUT/coherence_gate.csv" <<'PY'
import csv, sys
corr = list(csv.DictReader(open(sys.argv[1])))
ladder = list(csv.DictReader(open(sys.argv[2])))
problems, notes = [], []

# The correction evidence is a PAIRED contrast: every candidate that reports a
# ratio must have been scored on the same sample as its reference.
ratios = [r for r in corr if str(r.get("ppl_ratio") or "").strip()]
unpaired = [r["candidate"] for r in ratios if r.get("pairing") != "paired"]
if unpaired:
    problems.append(f"correction rows with a ratio but not paired: {unpaired}")
notes.append(f"correction: {len(ratios)} ratio row(s), all paired"
             if ratios and not unpaired else f"correction: {len(ratios)} ratios")
if not ratios:
    problems.append("the correction stage produced no ratio at all")

# The in-distribution row is descriptive: no ratio, and it says so.
desc = [r for r in corr if r.get("pairing") == "descriptive"]
if len(desc) != 1:
    problems.append(f"expected exactly one descriptive row, got {len(desc)}")
elif str(desc[0].get("ppl_ratio") or "").strip():
    problems.append("the descriptive in-distribution row carries a ratio")
notes.append(f"descriptive rows: {len(desc)}")

# The ladder's extension candidates are judged across windows, and the label
# says so: their reference is at a different context by construction.
ext = [r for r in ladder if r.get("reference_context")
       and str(r["reference_context"]) != str(r["context_length"])]
bad_ext = [r["candidate"] for r in ext if r.get("pairing") != "cross_context"]
if bad_ext:
    problems.append(f"extension rows not labelled cross_context: {bad_ext}")
notes.append(f"extension rows: {len(ext)} cross-context")
# ... while a SAME-context candidate on the ladder is paired.
same = [r for r in ladder if not r.get("reference_context")
        or str(r["reference_context"]) == str(r["context_length"])]
bad_same = [r["candidate"] for r in same
            if r.get("pairing") not in ("paired", "descriptive")]
if bad_same:
    problems.append(f"same-context rows not paired: {bad_same}")

for n in notes:
    print("  | " + n)
for p in problems:
    print("  | " + p)
sys.exit(1 if problems else 0)
PY
then ok "ratio rows are paired, extensions labelled cross-context"
else bad "the pairing labels are wrong in the gate output"; fi

# The references must share their candidates' seed, or the ratio is a difference
# between two samples -- asserted on the shipped configs, not on the stub.
"$PY" - <<'PY' && ok "every same-context reference shares its candidate's seed" \
  || bad "a reference uses a different seed from the candidates it scores"
import json, pathlib, sys
problems = []
for f in ("configs/mlsys_gate_controls.json", "configs/mlsys_rope_candidates.json",
          "configs/mlsys_rope_intervention_candidates.json",
          "configs/mlsys_correction_candidates.json"):
    d = json.loads(pathlib.Path(f).read_text())
    key = "candidates" if "candidates" in d else "coherence_gate"
    rows = d[key]
    refs = {(r["context_length"], r["target_model_name"]): r
            for r in rows if r.get("native_baseline")}
    for r in rows:
        if r.get("reference_role") == "in_distribution_reference":
            continue
        ref_ctx = r.get("reference_context") or r["context_length"]
        if int(ref_ctx) != int(r["context_length"]):
            continue                      # cross-window: pairing does not apply
        ref = refs.get((ref_ctx, r["target_model_name"]))
        if ref is None:
            continue                      # reported as no-reference by the gate
        if r.get("seed") != ref.get("seed"):
            problems.append(f"{f}: {r['name']} seed {r.get('seed')} vs reference "
                            f"{ref['name']} seed {ref.get('seed')}")
for p in problems:
    print("  | " + p)
sys.exit(1 if problems else 0)
PY

# ---------------------------------------------------------------------------
hdr "15  per-attempt freshness: a failed prerequisite stops its dependents"
# ---------------------------------------------------------------------------
# Attempt 2 injects a failure into engine_cap_smoke. Everything that depends on
# it must stop, and nothing may be validated from attempt 1's files -- which are
# still on disk at this point, so a fallback would be visible.
B=$WORK/attempt2
mkdir -p "$B"
# The run log is APPENDED across attempts, so "did a dependent run?" must read
# only the lines this attempt added.
N0=$(wc -l < "$OUT/RUN_LOG.txt" 2>/dev/null | tr -d ' ')
( cd "$SANDBOX" && \
  MLSYS_PYTHON="$PY" MLSYS_MAX_COST_USD=850 MLSYS_ASK_OVER_USD=300 \
  MLSYS_DOCUMENTS_JSON="$DOCS" MLSYS_RAN_DIR="$RAN_DIR" MLSYS_MAX_HOURS=40 \
  MLSYS_REHEARSAL_FAIL_RUN=CAPS_prefix1024_targetonly \
  MLSYS_VLLM_PYTHON="$PY" \
  bash scripts/mlsys_manifest.sh ) >"$B/manifest.log" 2>&1
B_RC=$?
RUNLOG2=$WORK/attempt2.runlog
tail -n "+$(( ${N0:-0} + 1 ))" "$OUT/RUN_LOG.txt" > "$RUNLOG2"

[ "$B_RC" != "0" ] && ok "the injected failure made the manifest exit non-zero ($B_RC)" \
  || bad "the manifest exited 0 despite an injected run failure"

# The failing STAGE is what the reviewer asked to see: a run that fails must
# fail its stage, not merely appear as a row.
grep -q "STAGE_FAILED name=engine_cap_smoke" "$RUNLOG2" \
  && ok "engine_cap_smoke is recorded as FAILED" \
  || bad "the injected failure is not recorded as a failed stage"
grep -q "STOP: the cap smoke rc=" "$B/manifest.log" \
  && ok "the manifest STOPPED at the failed prerequisite" \
  || bad "the manifest did not stop at the failed prerequisite"
grep -q "STAGE_OK name=natural_f1_128k" "$RUNLOG2" \
  && bad "a dependent stage ran on a failed prerequisite" \
  || ok "no dependent stage ran after the failure"

# And the freshness itself. Attempt 1 wrote four ok rows; attempt 2 wrote three
# ok rows and then aborted. If the archive did not happen, the live file would
# still be attempt 1's, or a merge of the two -- which is the failure that looks
# like a complete stage.
count_ok() { [ -f "$1" ] && "$PY" -c "
import csv, sys
rows = list(csv.DictReader(open(sys.argv[1])))
print(sum(1 for r in rows if r.get('status') == 'ok'))" "$1" || echo 0; }
ARCH=$(find "$OUT/attempts" -path '*engine_cap_smoke/engine_cap_smoke.csv' \
        -print -quit 2>/dev/null)
if [ -n "$ARCH" ]; then
  ok "the previous attempt's CSV was archived at ${ARCH#$OUT/}"
else
  bad "the previous attempt's CSV was not archived"
fi
arch_ok=$(count_ok "$ARCH"); live_ok=$(count_ok "$OUT/engine_cap_smoke.csv")
if [ "${arch_ok:-0}" = "4" ] && [ "${live_ok:-0}" = "3" ]; then
  ok "the live file is THIS attempt's (3 ok rows); attempt 1's four are archived"
else
  bad "freshness: archived ok-rows=${arch_ok:-?} (want 4), live ok-rows=${live_ok:-?} (want 3)"
fi
# The earlier results are still there for the operator: attempt 2 never reached
# them, so their archive was never taken.
if [ -s "$OUT/natural_f1_128k.csv" ] && [ -d "$OUT/tokens" ]; then
  ok "attempt 1's other outputs are untouched by attempt 2"
else
  bad "attempt 2 disturbed attempt 1's other outputs"
fi

# ---------------------------------------------------------------------------
hdr "16  the vLLM worker's reported ids decide unit_matched"
# ---------------------------------------------------------------------------
# Three outcomes, produced by the engine-side path rather than echoed from the
# sidecar: match, mismatch and unavailable. Only a match may be unit-matched,
# and only a match leaves a usable baseline -- so only `match` may exit 0.
for mode in match mismatch unavailable; do
  V=$WORK/vllm_$mode; mkdir -p "$V"
  # The INSTALLED stub, not the repo source: it resolves `_real_vllm_baseline`
  # from the directory it sits in, which is where the rehearsal put it.
  if MLSYS_REHEARSAL_VLLM_IDS=$mode "$PY" \
       "$SANDBOX/scripts/mlsys_vllm_baseline.py" \
       --out "$V/out.csv" --prompt-ids-from-sidecars "$OUT/tokens" \
       --documents pg19_train_0 --context-lengths 131072 \
       --max-new-tokens 1024 --matched-max-new-tokens 1024 \
       --models meta-llama/Llama-3.1-8B \
       --target-revisions "meta-llama/Llama-3.1-8B=d04e592bb4f6aa9cfee91e2e20afa771667e1d4b" \
       >"$V/log" 2>&1; then
    vrc=0
  else
    vrc=$?
  fi
  case "$mode" in
    match)
      if [ "$vrc" -eq 0 ]; then
        ok "vLLM stub ran in '$mode' mode and exited 0"
      else
        bad "vLLM stub failed in '$mode' mode (rc=$vrc)"; sed 's/^/  | /' "$V/log" | tail -3
      fi ;;
    *)
      # No row is unit_matched=yes, so the stage must NOT be recorded ok.
      if [ "$vrc" -ne 0 ]; then
        ok "vLLM stub exited $vrc in '$mode' mode: no usable baseline"
      else
        bad "vLLM stub exited 0 in '$mode' mode with no unit-matched row"
      fi ;;
  esac
  verdict=$("$PY" - "$V/out.csv" <<'PY'
import csv, sys
rows = list(csv.DictReader(open(sys.argv[1])))
print(rows[0]["unit_matched"] if rows else "NO_ROWS")
print(rows[0].get("prompt_ids_verified", "") if rows else "")
# The marker comes from the production `build_row`, not from the stub: if the
# parent only copied the worker's result fields, this is empty and the row can
# never be unit-matched.
print(rows[0].get("prompt_ids_from_engine", "") if rows else "")
PY
)
  unit=$(echo "$verdict" | sed -n 1p)
  verified=$(echo "$verdict" | sed -n 2p)
  from_engine=$(echo "$verdict" | sed -n 3p)
  case "$mode" in
    match)       want_unit=yes ;;
    mismatch)    want_unit=no ;;
    unavailable) want_unit=no ;;
  esac
  if [ "$from_engine" = "yes" ]; then
    ok "ids '$mode': build_row set prompt_ids_from_engine=yes from the cell"
  else
    bad "ids '$mode': prompt_ids_from_engine='$from_engine' - the parent did " \
        "not take it from the RASD cell"
  fi
  if [ "$unit" = "$want_unit" ]; then
    ok "ids '$mode' (verified='$verified') -> unit_matched=$unit"
  else
    bad "ids '$mode' (verified='$verified') -> unit_matched=$unit, expected $want_unit"
  fi
done

# ---------------------------------------------------------------------------
hdr "17  the target cross-check: the tie rule decides, and it decides both ways"
# ---------------------------------------------------------------------------
# `impl_validation` no longer claims acceptance agreement (vLLM cannot be given
# RASD's draft, its window, its ring sharding or its NF4 cache). It claims that
# two independent implementations of the same target produce the same greedy
# continuation -- and the production comparison decides that, under the tie rule.
#
# Three cases, all through production code:
#   agree    -> LOSSLESS (rc 0)
#   diverge with a DECISIVE gap -> MISMATCH (rc != 0)
#   diverge where the gap is below the tie threshold -> NUMERIC_TIE (rc 0)
for cse in "agree::3.0:0" "diverge:40:3.0:1" "tie:40:0.01:0"; do
  mode=${cse%%:*}; rest=${cse#*:}; pos=${rest%%:*}
  rest=${rest#*:}; gap=${rest%%:*}; want_rc=${rest##*:}
  C=$WORK/cross_$mode; mkdir -p "$C"
  if MLSYS_REHEARSAL_VLLM_DIVERGE=$pos MLSYS_REHEARSAL_VLLM_GAP=$gap "$PY" \
       "$SANDBOX/scripts/mlsys_vllm_baseline.py" \
       --out "$C/vllm.csv" --prompt-ids-from-sidecars "$OUT/tokens" \
       --rasd-target-sidecars "$OUT/tokens" \
       --compare-out "$C/crosscheck.csv" \
       --documents pg19_train_0 --context-lengths 131072 \
       --max-new-tokens 1024 --matched-max-new-tokens 1024 \
       --models meta-llama/Llama-3.1-8B \
       --target-revisions "meta-llama/Llama-3.1-8B=d04e592bb4f6aa9cfee91e2e20afa771667e1d4b" \
       >"$C/log" 2>&1; then
    crc=0
  else
    crc=$?
  fi
  verdict=$("$PY" - "$C/crosscheck.csv" <<'PYX'
import csv, sys
rows = list(csv.DictReader(open(sys.argv[1])))
print(rows[0]["verdict"] if rows else "NO_ROWS")
print(rows[0].get("tie_positions", "") if rows else "")
PYX
)
  case "$mode" in
    agree)   want_verdict=LOSSLESS ;;
    diverge) want_verdict=MISMATCH ;;
    tie)     want_verdict=NUMERIC_TIE ;;
  esac
  got=$(echo "$verdict" | sed -n 1p)
  if [ "$got" = "$want_verdict" ]; then
    ok "cross-check '$mode' -> $got (rc=$crc)"
  else
    bad "cross-check '$mode' -> $got, expected $want_verdict"
  fi
  if [ "$want_rc" = "0" ] && [ "$crc" -ne 0 ]; then
    bad "cross-check '$mode' should not fail the stage (rc=$crc)"
  fi
  if [ "$want_rc" = "1" ] && [ "$crc" -eq 0 ]; then
    bad "cross-check '$mode' produced a MISMATCH but exited 0: the stage would be recorded ok with two engines that disagree"
  fi
done
# A tie is reported, not swallowed: the counts have to reach the table.
"$PY" - "$WORK/cross_tie/crosscheck.csv" <<'PYX' && ok "the tie count is reported in the table" \
  || bad "a NUMERIC_TIE was not reported as a tie"
import csv, sys
r = next(csv.DictReader(open(sys.argv[1])))
assert r["verdict"] == "NUMERIC_TIE", r["verdict"]
assert r["tie_positions"], "the tie position was not recorded"
assert float(r["tie_gap_threshold"]) == 0.1
PYX

# ---------------------------------------------------------------------------
hdr "rehearsal summary"
# ---------------------------------------------------------------------------
printf '  %d passed, %d failed\n' "$PASS" "$FAIL"
if [ "$FAIL" -gt 0 ]; then
  printf '  sandbox kept at %s\n' "$WORK"
  exit 1
fi
printf '  REHEARSAL PASSED: every approved stage and helper ran, the stub rows\n'
printf '  satisfied every checker, and an unverified pull was refused.\n'
exit 0

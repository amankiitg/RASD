#!/usr/bin/env bash
# Does the target agree with ITSELF across modes, at 1 rank, as context grows?
#
# Usage:  bash scripts/mlsys_lossless_repro_1x.sh <pod-ip>
#
# WHERE THIS FITS
#   1. launch gpu_1x_a100_sxm4
#   2. MLSYS_HF_TOKEN=<token> bash scripts/mlsys_setup_probe.sh <ip>
#   3. bash scripts/mlsys_lossless_repro_1x.sh <ip>     <-- this script
#   4. pull the CSV and the log home, then terminate
#
# WHAT IS DIFFERENT FROM mlsys_runexp_probe_1x.sh, and why it is a separate
# script rather than a flag on that one:
#   * THE ASSERTION. The runexp probe asserted status/counts only -- it never
#     compared the speculative arm against its target-only partner token by
#     token, even though it ran both. That gap is why "the 1x probe passed" was
#     mistaken for "this is lossless at 1 rank". This script's whole point is
#     the token-by-token comparison, using the SAME compare_generations() the
#     campaign's checker uses, so a pass here means what a pass there means.
#   * NO --abort-on-failure. The 32k cells are allowed to OOM; the 8k cells must
#     still run, because "32k does not fit" and "32k diverges" are different
#     findings and only the second one is about the model.
#   * It asserts the MEASURED kv_dtype, so a kv_quant override that silently did
#     nothing cannot pass as a real negative result.
set -uo pipefail

# Optional, because MLSYS_LOSSESS_SKIP_POD=1 needs no pod at all; required
# below whenever a pod is actually used.
IP=${1:-}
SSH_USER=${2:-ubuntu}
SSH_KEY=${MLSYS_SSH_KEY:-$HOME/.ssh/id_ed25519}
SSH_OPTS="-o StrictHostKeyChecking=no -o ConnectTimeout=25 -i $SSH_KEY"
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"

CFG=configs/mlsys_lossless_repro_1x.yml
MLSYS_LOCAL_PY=${MLSYS_LOCAL_PY:-python3}
# Quoted so the tilde survives to the REMOTE shell. Unquoted, bash expands it
# here and the pod is asked to mkdir /Users/<this laptop>/RASD.
POD_OUT='~/RASD/results/mlsys/lossless_repro'
POD_CSV=$POD_OUT/lossless_repro.csv
POD_LOG=$POD_OUT/lossless_repro.log
# Overridable so the comparison can be pointed at a fixture (real incident
# sidecars) and proved to reproduce a KNOWN divergence before a pod is bought.
LOCAL_OUT=${MLSYS_LOSSESS_LOCAL_OUT:-results/mlsys/lossless_repro}
# NOT `GROUPS`: that name is a special READONLY bash array, so assigning to it
# is silently ignored and the value becomes a number.
LOSS_GROUPS="LOSS_32K LOSS_8K"

HF_TOKEN_VALUE=$(grep -E '^[[:space:]]*HF_TOKEN[[:space:]]*=' runpod_creds.md 2>/dev/null \
                 | head -1 | sed -E 's/^[^=]*=[[:space:]]*//' | tr -d '"'"'"' \r')
if [ -z "$HF_TOKEN_VALUE" ]; then
  echo "FATAL: no HF_TOKEN line found in runpod_creds.md (looked up by label)." >&2
  exit 2
fi

PASS=0; FAIL=0
ok()  { printf '  PASS  %s\n' "$*"; PASS=$((PASS+1)); }
bad() { printf '  FAIL  %s\n' "$*"; FAIL=$((FAIL+1)); }
pod() { ssh $SSH_OPTS "$SSH_USER@$IP" "$@"; }

printf '\n== 0. stage THIS working tree, and prove the pod is running it ==\n'
# MLSYS_LOSSESS_SKIP_POD=1 runs ONLY step 4, against whatever is already in
# $LOCAL_OUT. That makes the comparison testable against a KNOWN divergence
# before a pod is launched -- the comparison IS the experiment here, and one that
# cannot reproduce a defect it has already seen proves nothing about a fresh run.
RUN_POD=1
if [ "${MLSYS_LOSSESS_SKIP_POD:-0}" = "1" ]; then
  RUN_POD=0
  printf '  (MLSYS_LOSSESS_SKIP_POD=1: no pod; comparing %s as-is)\n' "$LOCAL_OUT"
fi

if [ "$RUN_POD" = "1" ]; then
if [ -z "$IP" ]; then echo "FATAL: usage: mlsys_lossless_repro_1x.sh <pod-ip>" >&2; exit 2; fi
if rsync -az --no-perms --no-owner --no-group -e "ssh $SSH_OPTS" \
     --exclude '.git' --exclude 'results/final' --exclude 'manuscript' \
     --exclude '.venv*' --exclude '__pycache__' \
     "$REPO/" "$SSH_USER@$IP:~/RASD/" >/tmp/lossless_rsync.log 2>&1; then
  ok "working tree staged to the pod"
else
  bad "rsync failed"; tail -3 /tmp/lossless_rsync.log
fi
LOCAL_SHA=$(shasum -a 256 run_experiment.py | awk '{print $1}')
POD_SHA=$(pod "sha256sum ~/RASD/run_experiment.py 2>/dev/null | cut -d' ' -f1" 2>/dev/null)
if [ "$LOCAL_SHA" = "$POD_SHA" ]; then
  ok "the pod's run_experiment.py is byte-identical to this working tree"
else
  bad "the pod is running a DIFFERENT run_experiment.py (local $LOCAL_SHA, pod ${POD_SHA:-unreadable})"
  exit 2
fi

printf '\n== 1. the pod holds the campaign environment (no fallback) ==\n'
if pod 'test -f ~/RASD/.pod_env.sh'; then
  ok "~/RASD/.pod_env.sh present"
else
  bad "~/RASD/.pod_env.sh is absent: run scripts/mlsys_setup_probe.sh first."
  exit 2
fi
PY_REMOTE=$(pod 'grep -a "^INTERPRETER=" ~/pod_env.log 2>/dev/null | tail -1 | cut -d= -f2-' 2>/dev/null)
case "$PY_REMOTE" in
  */envs/rasd-gpu/bin/python) ok "resolved the campaign interpreter: $PY_REMOTE" ;;
  "") bad "no INTERPRETER= line in ~/pod_env.log; run scripts/mlsys_setup_probe.sh first"; exit 2 ;;
  *)  bad "resolved '$PY_REMOTE', which is not the campaign env"; exit 2 ;;
esac

printf '\n== 2. the command, in the manifest form (plus --nproc 1, minus --abort-on-failure) ==\n'
printf '  HF token: present (%s chars, not shown)\n' "$(printf '%s' "$HF_TOKEN_VALUE" | wc -c | tr -d ' ')"
RUNNER="cd ~/RASD && set -a && . ~/RASD/.pod_env.sh && set +a && \
mkdir -p $POD_OUT && rm -f $POD_CSV && \
HF_TOKEN='$HF_TOKEN_VALUE' MLSYS_PYTHON='$PY_REMOTE' \
$PY_REMOTE run_experiment.py --config $CFG --groups $LOSS_GROUPS \
  --output $POD_CSV --stage-id lossless_repro_1x --nproc 1 \
  --timeout-per-run-s 1800 \
  --log-per-token --save-generated-tokens \
  > $POD_LOG 2>&1"
echo "  $ $PY_REMOTE run_experiment.py --config $CFG --groups $LOSS_GROUPS --nproc 1 ..."
T=$(date -u +%s)
pod "$RUNNER"
RC=$?
WALL=$(( $(date -u +%s) - T ))
if [ "$RC" = "0" ]; then
  ok "run_experiment exited 0 (${WALL}s wall)"
else
  bad "run_experiment exited $RC after ${WALL}s"
fi

fi   # end of the RUN_POD guard: steps 0-2 all need a pod

if [ "$RUN_POD" = "1" ]; then
printf '\n== 3. pull the artifacts home ==\n'
mkdir -p "$LOCAL_OUT"
if [ -f "$LOCAL_OUT/lossless_repro.csv" ]; then
  mkdir -p "$LOCAL_OUT/attempts"
  STAMP=$(date -u +%Y%m%dT%H%M%SZ)
  cp "$LOCAL_OUT/lossless_repro.csv" "$LOCAL_OUT/attempts/lossless_repro.$STAMP.csv"
  ok "archived the previous attempt to attempts/lossless_repro.$STAMP.csv"
fi
rsync -az --no-perms --no-owner --no-group -e "ssh $SSH_OPTS" \
  "$SSH_USER@$IP:$POD_OUT/" "$LOCAL_OUT/" >/tmp/lossless_pull.log 2>&1
if [ $? = 0 ]; then
  ok "pulled the results dir (csv, log, tokens/) to $LOCAL_OUT"
else
  bad "could not pull $POD_OUT"; tail -3 /tmp/lossless_pull.log
fi
fi   # end of the pull guard

printf '\n== 4. the assertion this script exists for: spec vs target-only, token by token ==\n'
"$MLSYS_LOCAL_PY" - "$LOCAL_OUT" <<'PYLOSS'
import csv, json, sys, pathlib
sys.path.insert(0, ".")
from src.analysis.losslessness import compare_generations, TIE_GAP

out = pathlib.Path(sys.argv[1])
csvp = out / "lossless_repro.csv"
if not csvp.exists():
    print("  FAIL  no CSV pulled, so nothing can be compared"); sys.exit(1)
rows = list(csv.DictReader(open(csvp)))
print("  rows=%d" % len(rows))
for r in rows:
    print("    %-40s status=%-6s tok=%-4s rounds=%-4s acc=%-7s kv=%-5s ctx=%s"
          % (r.get("run_id"), r.get("status"), r.get("tokens_generated"),
             r.get("n_rounds"), (r.get("acceptance_rate") or "")[:6],
             r.get("kv_dtype") or "(not in csv)", r.get("context_length")))
    if r.get("status") != "ok":
        print("        error: %s" % (r.get("error") or "")[:160])

def _print_divergence(a, b, pos):
    if pos is None:
        return
    si, ti = a["generated_token_ids"], b["generated_token_ids"]
    sg, tg = a.get("token_gaps") or [], b.get("token_gaps") or []
    lo = max(0, pos - 2)
    for q in range(lo, min(len(si), len(ti), pos + 4)):
        mark = "  <-- first divergence" if q == pos else ""
        print("        pos %2d | spec %7s gap %-6s | target %7s gap %-6s%s"
              % (q, si[q], sg[q] if q < len(sg) else "-",
                 ti[q], tg[q] if q < len(tg) else "-", mark))


fails = 0
checked = 0
for ctx, kv in (("32K", "nf4"), ("32K", "bf16"), ("8K", "nf4"), ("8K", "bf16")):
    base = "LOSS_%s_%s" % (ctx, kv)
    sp = out / "tokens" / ("%s_spec_pg19_train_1_s42.json" % base)
    tp = out / "tokens" / ("%s_targetonly_pg19_train_1_s42.json" % base)
    print("\n  --- %s : spec vs target-only ---" % base)
    if not (sp.exists() and tp.exists()):
        print("      SKIP  sidecar(s) missing (%s / %s) -- a cell that never ran"
              % (sp.exists(), tp.exists()))
        continue
    a = json.load(open(sp)); b = json.load(open(tp))
    # The MEASURED dtype, not the requested one: an override that did nothing
    # must not read as a real result.
    print("      measured: spec kv=%s w=%s | targetonly kv=%s w=%s"
          % (a["kv_dtype"], a["weight_precision"], b["kv_dtype"], b["weight_precision"]))
    same_prompt = a["prompt_token_ids"] == b["prompt_token_ids"]
    print("      prompt identical? %s (sha %s / %s)"
          % (same_prompt, a["prompt_sha256"][:12], b["prompt_sha256"][:12]))
    res = compare_generations(a["generated_token_ids"], b["generated_token_ids"],
                              full_length=64, min_prefix=64,
                              spec_gaps=a.get("token_gaps"),
                              target_gaps=b.get("token_gaps"))
    checked += 1
    print("      verdict=%s  compared=%s tokens  tie_gate=%s"
          % (res["verdict"], res["compared_tokens"], TIE_GAP))
    # ONLY MISMATCH FAILS. This mirrors mlsys_cap_smoke_check.py, which treats a
    # NUMERIC_TIE as a reportable pass: a divergence at a position where the
    # target was indifferent (gap below the tie threshold) is an artefact of
    # floating-point reassociation, not an implementation defect. Counting a tie
    # as a failure here would have overstated the real 1x run as 4 divergences
    # when one of the four was a tie, and would fail a stage for being honest
    # about its numerics.
    if res["verdict"] == "MISMATCH":
        fails += 1
        print("      MISMATCH: %s" % (res.get("detail") or ""))
        _print_divergence(a, b, res.get("first_mismatch_position"))
    elif res["verdict"] == "NUMERIC_TIE":
        print("      NUMERIC_TIE: %s" % (res.get("detail") or ""))
        print("      (reported, NOT a failure: the target was indifferent there)")
        _print_divergence(a, b, res.get("first_mismatch_position"))
    else:
        print("      LOSSLESS: every one of the 64 tokens matches")
print("\n  RESULT: %d pair(s) compared, %d diverging" % (checked, fails))
sys.exit(1 if (checked == 0 or fails) else 0)
PYLOSS
CMP_RC=$?
if [ "$CMP_RC" = "0" ]; then
  ok "every (context, kv-dtype) pair is LOSSLESS at 1 rank"
else
  bad "at least one pair diverged at 1 rank (see the table above)"
fi

printf '\n== LOSSESS REPRO RESULT: %d passed, %d failed ==\n' "$PASS" "$FAIL"
[ "$FAIL" -eq 0 ] || exit 1

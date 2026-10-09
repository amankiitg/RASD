#!/usr/bin/env bash
# Does the teacher-forced probe actually work with a RING, on 2 ranks?
#
# Usage:  bash scripts/mlsys_probe_2rank_check.sh <pod-ip>
#
# WHERE THIS FITS
#   1. launch gpu_2x_a100
#   2. MLSYS_HF_TOKEN=<token> bash scripts/mlsys_setup_probe.sh <ip>
#   3. bash scripts/mlsys_probe_2rank_check.sh <ip>     <-- this script
#   4. pull, then terminate
#
# WHY IT IS NOT RUN ON THE 8x. The bug it validates was a rank-participation bug:
# rank 0 entered the probe alone because `gen_ids` is rank-0-only. Two ranks is
# enough to reproduce that class of failure and to prove the fix, and it costs
# 1/4 of the 8x price. If this passes and the 8x then fails, that is itself
# information about the ring at 8 -- which the campaign's own gate will report.
#
# IT ALSO RUNS THE LIVENESS WATCHDOG SELF-TEST, on the same instance, because that
# rule is what catches a future hang of exactly the kind this bug produced, and
# both need the same pod.
set -uo pipefail

IP=${1:-}
# The GPU count of whatever instance this is pointed at. It becomes --nproc and
# it is what the probe's own world_size must report, because the whole point of
# this run is RANK PARTICIPATION: a probe that silently ran on one rank would
# otherwise look exactly like a success. Defaults to 2, the cheapest thing that
# can expose the bug.
NPROC=${3:-${MLSYS_PROBE2_NPROC:-2}}
SSH_USER=${2:-ubuntu}
SSH_KEY=${MLSYS_SSH_KEY:-$HOME/.ssh/id_ed25519}
SSH_OPTS="-o StrictHostKeyChecking=no -o ConnectTimeout=25 -i $SSH_KEY"
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"

CFG=configs/mlsys_probe_2rank.yml
GROUP=PROBE_2RANK
MLSYS_LOCAL_PY=${MLSYS_LOCAL_PY:-python3}
POD_OUT='~/RASD/results/mlsys/probe_2rank'
LOCAL_OUT=${MLSYS_PROBE2_LOCAL_OUT:-results/mlsys/probe_2rank}

# The 1-rank reference this run has to be comparable to. Measured on real
# hardware at 1 rank, 8k, bf16, 64 tokens, with the engine's own probe.
REF_SHORTFALL=0.375
REF_NON_ARGMAX=2
REF_POSITIONS=63
TOL_BF16=1.875            # must match mlsys_cap_smoke_check.TOL_BF16

HF_TOKEN_VALUE=$(grep -E '^[[:space:]]*HF_TOKEN[[:space:]]*=' runpod_creds.md 2>/dev/null \
                 | head -1 | sed -E 's/^[^=]*=[[:space:]]*//' | tr -d '"'"'"' \r')
if [ -z "$HF_TOKEN_VALUE" ]; then
  echo "FATAL: no HF_TOKEN line found in runpod_creds.md (looked up by label)." >&2
  exit 2
fi
if [ -z "$IP" ]; then echo "FATAL: usage: mlsys_probe_2rank_check.sh <pod-ip> [user] [nproc]" >&2; exit 2; fi

PASS=0; FAIL=0
ok()  { printf '  PASS  %s\n' "$*"; PASS=$((PASS+1)); }
bad() { printf '  FAIL  %s\n' "$*"; FAIL=$((FAIL+1)); }
pod() { ssh $SSH_OPTS "$SSH_USER@$IP" "$@"; }

printf '\n== 0. stage THIS working tree, and prove the pod is running it ==\n'
if rsync -az --no-perms --no-owner --no-group -e "ssh $SSH_OPTS" \
     --exclude '.git' --exclude 'results/final' --exclude 'manuscript' \
     --exclude '.venv*' --exclude '__pycache__' \
     "$REPO/" "$SSH_USER@$IP:~/RASD/" >/tmp/probe2_rsync.log 2>&1; then
  ok "working tree staged to the pod"
else
  bad "rsync failed"; tail -3 /tmp/probe2_rsync.log; exit 2
fi
# Both files that carry the fix. run_experiment.py holds the broadcast and the
# agreement; rasd_inference.py holds the budget and the exit criteria.
for f in run_experiment.py src/models/rasd_inference.py; do
  L=$(shasum -a 256 "$f" | awk '{print $1}')
  P=$(pod "sha256sum ~/RASD/$f 2>/dev/null | cut -d' ' -f1" 2>/dev/null)
  if [ "$L" = "$P" ]; then ok "$f is byte-identical on the pod"
  else bad "$f differs (local ${L:0:12}, pod ${P:0:12})"; exit 2; fi
done

printf '\n== 1. the pod holds the campaign environment ==\n'
pod 'test -f ~/RASD/.pod_env.sh' \
  && ok "~/RASD/.pod_env.sh present" \
  || { bad "~/RASD/.pod_env.sh absent: run scripts/mlsys_setup_probe.sh first"; exit 2; }
PY_REMOTE=$(pod 'grep -a "^INTERPRETER=" ~/pod_env.log 2>/dev/null | tail -1 | cut -d= -f2-' 2>/dev/null)
case "$PY_REMOTE" in
  */envs/rasd-gpu/bin/python) ok "campaign interpreter: $PY_REMOTE" ;;
  *) bad "unexpected interpreter '$PY_REMOTE'"; exit 2 ;;
esac

printf '\n== 2. the probe on %d ranks (8k, bf16, 64 tokens) ==\n' "$NPROC"
printf '  (nproc %d = the GPU count of this instance; the probe must report world_size=%d)\n' "$NPROC" "$NPROC"
RUNNER="cd ~/RASD && set -a && . ~/RASD/.pod_env.sh && set +a && \
mkdir -p $POD_OUT && rm -f $POD_OUT/probe_2rank.csv && \
HF_TOKEN='$HF_TOKEN_VALUE' MLSYS_PYTHON='$PY_REMOTE' \
$PY_REMOTE run_experiment.py --config $CFG --groups $GROUP \
  --output $POD_OUT/probe_2rank.csv --stage-id probe_2rank --nproc $NPROC \
  --timeout-per-run-s 3600 \
  --log-per-token --save-generated-tokens \
  > $POD_OUT/probe_2rank.log 2>&1"
T=$(date -u +%s)
pod "$RUNNER"
RC=$?
WALL=$(( $(date -u +%s) - T ))
if [ "$RC" = "0" ]; then ok "run_experiment exited 0 (${WALL}s wall)"
else bad "run_experiment exited $RC after ${WALL}s"; fi

printf '\n== 3. pull the artifacts home ==\n'
mkdir -p "$LOCAL_OUT"
rsync -az --no-perms --no-owner --no-group -e "ssh $SSH_OPTS" \
  "$SSH_USER@$IP:$POD_OUT/" "$LOCAL_OUT/" >/tmp/probe2_pull.log 2>&1 \
  && ok "pulled $POD_OUT to $LOCAL_OUT" \
  || { bad "could not pull $POD_OUT"; tail -3 /tmp/probe2_pull.log; }

printf '\n== 4. did EVERY rank produce numbers, and are they comparable? ==\n'
"$MLSYS_LOCAL_PY" - "$LOCAL_OUT" "$REF_SHORTFALL" "$REF_NON_ARGMAX" "$REF_POSITIONS" "$TOL_BF16" "$NPROC" <<'PYP2'
import csv, json, sys, pathlib

out = pathlib.Path(sys.argv[1])
ref_shortfall, ref_nonargmax, ref_positions, tol = (
    float(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4]), float(sys.argv[5]))
want_ranks = int(sys.argv[6])

csvp = out / "probe_2rank.csv"
if not csvp.exists():
    print("  FAIL  no CSV pulled"); sys.exit(1)
rows = list(csv.DictReader(open(csvp)))
print("  rows=%d (expected 2)" % len(rows))
problems = []
for r in rows:
    print("    %-26s status=%-6s tok=%-4s rounds=%-4s acc=%-7s kv=%s"
          % (r.get("run_id"), r.get("status"), r.get("tokens_generated"),
             r.get("n_rounds"), (r.get("acceptance_rate") or "")[:6],
             r.get("kv_dtype") or "(unset)"))
    if r.get("status") != "ok":
        problems.append("%s: status=%s error=%s"
                        % (r.get("run_id"), r.get("status"),
                           (r.get("error") or "")[:160]))

print()
for r in rows:
    rid = r.get("run_id")
    p = out / "tokens" / ("%s.tflossless.json" % rid)
    print("  --- %s" % rid)
    if not p.exists():
        problems.append("%s: NO probe sidecar -- the probe did not run" % rid)
        print("      MISSING sidecar"); continue
    d = json.load(open(p))
    if d.get("error"):
        problems.append("%s: probe ERROR %s" % (rid, d["error"][:200]))
        print("      ERROR: %s" % d["error"][:200]); continue
    nf = d.get("noise_floor") or {}
    print("      world_size=%s measured_kv=%s tokens=%s positions=%s"
          % (d.get("world_size"), d.get("measured_kv"), d.get("tokens"),
             d.get("positions")))
    print("      max_shortfall=%s worst_position=%s non_argmax=%s/%s (%.1f%%)"
          % (d.get("max_shortfall"), d.get("worst_position"), d.get("non_argmax"),
             d.get("positions"), 100 * float(d.get("non_argmax_fraction") or 0)))
    print("      noise_floor max|delta|=%s control=%s"
          % (nf.get("max_abs_delta"), nf.get("control_max_abs_delta")))
    print("      budget_s=%s elapsed_s=%s" % (d.get("budget_s"), d.get("elapsed_s")))
    # The two properties that make the numbers usable at all.
    if int(d.get("world_size") or 0) != want_ranks:
        problems.append("%s: world_size=%s, expected %d -- a probe that ran on "
                        "fewer ranks looks identical to a success"
                        % (rid, d.get("world_size"), want_ranks))
    if str(d.get("measured_kv")) != "bfloat16":
        problems.append("%s: measured_kv=%s, expected bfloat16 -- a kv_quant "
                        "override that did nothing would read as a real result"
                        % (rid, d.get("measured_kv")))
    ms = float(d.get("max_shortfall") or 0.0)
    if ms > tol:
        problems.append("%s: max shortfall %.4f > TOL_bf16 %.3f" % (rid, ms, tol))
    # Determinism: the control is the same computation twice and must be zero.
    if float(nf.get("control_max_abs_delta") or 0.0) != 0.0:
        problems.append("%s: control_max_abs_delta=%s, expected 0.0 -- the run is "
                        "not deterministic, so nothing else is interpretable"
                        % (rid, nf.get("control_max_abs_delta")))

print()
print("  COMPARISON TO THE 1-RANK REFERENCE (same doc, 8k, bf16, 64 tokens)")
print("  %d rank(s) here. The ring reorders the floating-point sums, so a"
      % want_ranks)
print("  difference is a result to report, not automatically a failure.")
print("    %-28s %-12s %s" % ("quantity", "1 rank", "2 ranks"))
for r in rows:
    p = out / "tokens" / ("%s.tflossless.json" % r.get("run_id"))
    if not p.exists(): continue
    d = json.load(open(p))
    if d.get("error"): continue
    print("    %-28s %-12s %s" % ("max shortfall", ref_shortfall,
                                  d.get("max_shortfall")))
    print("    %-28s %-12s %s" % ("non-argmax positions", "%d/%d" % (ref_nonargmax, ref_positions),
                                  "%s/%s" % (d.get("non_argmax"), d.get("positions"))))
    nf = d.get("noise_floor") or {}
    print("    %-28s %-12s %s" % ("noise floor max|delta|", "0.688",
                                  nf.get("max_abs_delta")))
    print("    %-28s %-12s %s" % ("rank count", 1, d.get("world_size")))

print()
if problems:
    for x in problems:
        print("  PROBLEM: %s" % x)
    sys.exit(1)
print("  every ranked row produced numbers, all within TOL_bf16 %.3f" % tol)
sys.exit(0)
PYP2
PROBE_RC=$?
[ "$PROBE_RC" = "0" ] && ok "the probe ran on every rank and its numbers are usable" \
                      || bad "the probe did not produce usable numbers (see above)"

printf '\n== 5. the byte-level liveness watchdog self-test, same pod ==\n'
LIVE_OUT=$(pod 'cd ~/RASD && bash scripts/mlsys_watchdog_selftest.sh; echo "EXIT=$?"' 2>&1 | tail -22)
printf '%s\n' "$LIVE_OUT" | sed 's/^/      /'
case "$LIVE_OUT" in
  *"8 passed, 0 failed"*) ok "the liveness watchdog self-test passed on the pod" ;;
  *)                      bad "the liveness watchdog self-test failed on the pod" ;;
esac

printf '\n== RESULT: %d passed, %d failed ==\n' "$PASS" "$FAIL"
[ "$FAIL" -eq 0 ]

#!/usr/bin/env bash
# COMBINED SESSION, pod side. One instance, one session, a to e, GPU never released.
#
# ORDER AND GATING: each stage runs ONLY if the previous one succeeded. A failed stage
# stops the chain, and the RERUN pass at the end retries that stage's failed rows once.
# The headline (c) has priority: the cheap quality check runs before it because a bad
# 1M context would invalidate every decode row that followed.
#
# NO "skip if projected over" logic exists here. The cost ceiling is a safety net, and
# the operator's instruction is that it must never shorten a planned step. The only
# thing that removes a stage is the failure of the one before it.
#
# PARAMETERS, overridable so the SAME script can be rehearsed with tiny settings
# without changing one command line, flag, output path or stage id:
#   SESSION_OUT         results/mlsys/session_a
#   SESSION_CFG         configs/mlsys_sunday_1m.yml
#   SESSION_SANITY_CFG  configs/mlsys_sunday_sanity.yml
#   NPROC               8
set -uo pipefail
cd "$HOME/RASD" || exit 3
OUT=${SESSION_OUT:-results/mlsys/session_a}
CFG=${SESSION_CFG:-configs/mlsys_sunday_1m.yml}
SANITY_CFG=${SESSION_SANITY_CFG:-configs/mlsys_sunday_sanity.yml}
NPROC=${NPROC:-8}
mkdir -p "$OUT"

# DONE MARKER: this script is started DETACHED on the pod (setsid), and the Mac side
# waits for THIS FILE rather than for the ssh command to return. That is the fix for
# 2026-10-11: the Mac->pod ssh dropped at 04:12Z, the driver treated the ssh return as
# the end of the work and terminated the instance at 05:35:32Z, and a healthy pod was
# killed mid-stage. A remote script OUTLIVES the ssh session that started it, so its own
# exit status is the only honest signal that it is finished. The trap fires on every exit
# path in this file, including the early `exit 2/3/4` refusals.
PODDONE="$OUT/DONE"
remote_finish() {
  local rc=$?
  printf '%s %s\n' "$(date -u +%FT%TZ)" "$rc" > "$PODDONE" 2>/dev/null
  printf '%s %s\n' "$(date -u +%FT%TZ)" "=== REMOTE EXIT rc=$rc; DONE marker written ===" \
    >> "$OUT/session_a.log" 2>/dev/null
}
trap remote_finish EXIT

log() { printf '%s %s\n' "$(date -u +%FT%TZ)" "$*" | tee -a "$OUT/session_a.log"; }

# HF TOKEN FROM A FILE, never from a command line. The token used to travel as
# HF_TOKEN=... inside the ssh command, which put it in argv and therefore in `ps` on both
# machines. The driver writes ~/RASD/.hf_token with mode 600 and this reads it.
if [ -s "$HOME/.hf_token" ]; then
  HF_TOKEN=$(cat "$HOME/.hf_token")
  export HF_TOKEN
  log "HF token: read from ~/.hf_token ($(wc -c < "$HOME/.hf_token" | tr -d ' ') bytes, mode $(stat -c '%a' "$HOME/.hf_token" 2>/dev/null))"
else
  log "WARN: no ~/.hf_token; model downloads may fail if the repos are gated"
fi

PY=$(grep -a "^INTERPRETER=" "$HOME/pod_env.log" 2>/dev/null | tail -1 | cut -d= -f2-)
if [ -z "${PY:-}" ] || [ ! -x "${PY:-}" ]; then
  log "FATAL: no usable INTERPRETER= in ~/pod_env.log; refusing to guess one"
  exit 2
fi
# shellcheck disable=SC1090
set -a; . "$HOME/RASD/.pod_env.sh"; set +a
log "=== interpreter $PY nproc $NPROC cfg $CFG sanity $SANITY_CFG out $OUT ==="
"$PY" -c 'import torch,transformers;print("torch",torch.__version__,"transformers",transformers.__version__,"gpus",torch.cuda.device_count())' 2>&1 | tee -a "$OUT/session_a.log"

ok_rows()  { if [ -f "${1:-}" ]; then tail -n +2 "$1" | grep -c ",ok,"; else echo 0; fi; }
bad_rows() { if [ -f "${1:-}" ]; then tail -n +2 "$1" | grep -vc ",ok," ; else echo 0; fi; }

# PROGRESS: per-stage measured row times, written for the Mac side to pull every 4
# minutes and fold into the status block, so "measured time per row and the projection
# for the remaining steps" is a number rather than a feeling.
progress() {
  "$PY" - "$OUT" >> "$OUT/progress.md" <<'PYEOF'
import csv, glob, os, sys, time
out = sys.argv[1]
rows = []
for f in sorted(glob.glob(os.path.join(out, 'session_a_*.csv'))):
    stage = os.path.basename(f)[len('session_a_'):-4]
    try:
        for r in csv.DictReader(open(f)):
            if r.get('run_id'):
                rows.append((stage, r.get('context_length', ''), r.get('status', ''),
                             float(r.get('time_sec') or 0),
                             float(r.get('throughput_tps') or 0),
                             float(r.get('acceptance_rate') or 0)))
    except Exception as e:                                   # noqa: BLE001
        print(f"- {stage}: unreadable ({e})")
print(f"### progress {time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}")
for s in sorted({t[0] for t in rows}):
    sel = [t for t in rows if t[0] == s]
    ok = [t for t in sel if t[2] == 'ok']
    ts = sorted(t[3] for t in ok)
    bits = [f"- {s}: rows={len(sel)} ok={len(ok)}"]
    if ts:
        bits += [f"median_row_s={ts[len(ts)//2]:.1f}", f"max_row_s={ts[-1]:.1f}",
                 f"tps={sum(t[4] for t in ok)/len(ok):.3f}",
                 f"accept={sum(t[5] for t in ok)/len(ok):.4f}"]
    print(" ".join(bits))
tot = sum(t[3] for t in rows if t[2] == 'ok')
print(f"- measured row time total {tot:.1f}s = {tot/3600:.2f}h over "
      f"{sum(1 for t in rows if t[2]=='ok')} rows")
PYEOF
}

run_stage() {   # name timeout
  local name="$1" tmo="$2" csv="$OUT/session_a_$1.csv" rc n
  progress
  log "### stage $name (timeout ${tmo}s)"
  timeout "$tmo" "$PY" run_experiment.py --config "$CFG" --nproc "$NPROC" \
    --groups "$name" --output "$csv" --stage-id "session_a_$name" \
    --log-per-token --save-generated-tokens 2>&1 | tee -a "$OUT/group_$name.log" | tail -20
  rc=$?
  n=$(ok_rows "$csv")
  log "### stage $name rc=$rc ok_rows=$n bad_rows=$(bad_rows "$csv")"
  progress
  [ "$n" -gt 0 ] && return 0 || return 1
}

# ---------------------- THE STAGE LIST AND THE PARAMETER CONTRACT, FIRST OF ALL --
# ORDER MATTERS HERE, AND MOVING THIS WAS THE FIX (2026-10-12, found while judging the
# n=5 run). This block used to sit AFTER the sanity pair, because the stage list is
# derived from the session config while the sanity pair uses a different one. That made
# the claim in its own comment false: a pod that had received the wrong parameter set
# still loaded two models and generated two rows before refusing -- and the sanity pair
# is not a cheap check, it is a model load. "Before any heavy work" has to mean before
# the sanity pair too, so the derivation and the check now run first.
#
# THE STEP LIST COMES FROM THE CONFIG, in file order. A hardcoded list meant a
# different config (the quality-only redo) could not be run by this same script without
# editing it -- and this script is the one thing a running session's pod is reading.
# Deriving the list keeps ONE tested script for every session shape.
# THE STAGE LIST HAS ONE IMPLEMENTATION, shared with the drivers that assert it
# (scripts/mlsys_stage_list.py). An inline copy here and another in a driver is how a
# campaign comes to disagree with itself about what it ran.
STEPS=$("$PY" scripts/mlsys_stage_list.py "$CFG")
steps_rc=$?
if [ "$steps_rc" != "0" ] || [ -z "$STEPS" ]; then
  log "FATAL: could not derive a stage list from $CFG (rc=$steps_rc); refusing to record stages without knowing which ones"
  exit 2
fi
log "steps from $CFG: $(echo $STEPS | tr '\n' ' ')"

# ---- THE PARAMETER CONTRACT, BEFORE THE SANITY PAIR AND BEFORE ANY STAGE ----
# The stage ids are derived just above, and the driver sends the list it intended; if
# they disagree, or if the config/output/nproc the pod received is not what the driver
# sent, this stops BEFORE the sanity pair rather than after it. On 2026-10-12 an 8x run
# spent ~$5.5 executing a different parameter set than its driver had set, and nothing
# on the pod could notice.
# shellcheck disable=SC1090
. scripts/mlsys_remote_expect.sh
# ONE SPACE BETWEEN, NONE AT THE END. `echo $STEPS | tr '\n' ' '` leaves a TRAILING
# space, and the driver's manifest (mlsys_stage_list.py --space) does not -- so the
# contract would have refused every session over a whitespace difference. Caught by
# the boundary test before any 8x money was spent on it.
STEPS_ONELINE=$(printf '%s' "$STEPS" | tr '\n' ' ' | sed -E 's/ +$//')
# KEYS ARE THE DRIVER'S VARIABLE NAMES on both sides (see the vllm wrapper for why).
mlsys_expect "session-a-remote" "${REMOTE_EXPECT:-}" \
  "SESSION_OUT=$OUT;SESSION_CFG=$CFG;SESSION_SANITY_CFG=$SANITY_CFG;NPROC=$NPROC;SESSION_CHECK_STAGE=${SESSION_CHECK_STAGE:-};SESSION_GATE_CANDIDATES=${SESSION_GATE_CANDIDATES:-};STAGES=$STEPS_ONELINE;CODE_COMMIT=${CODE_COMMIT-<absent>}" || exit 2

# ------------------------------------------------------------------ (a) sanity --
log "### STEP a: sanity pair (both arms, one invocation)"
timeout 3600 "$PY" run_experiment.py --config "$SANITY_CFG" --nproc "$NPROC" \
  --output "$OUT/session_a_sanity.csv" --stage-id session_a_sanity \
  --log-per-token --save-generated-tokens 2>&1 | tee -a "$OUT/group_SANITY.log" | tail -20
SAN_RC=$?
SAN=$(ok_rows "$OUT/session_a_sanity.csv")
log "### STEP a rc=$SAN_RC ok_rows=$SAN (need 2)"
progress
if [ "${SAN:-0}" -lt 2 ]; then
  log "STOP: the sanity pair produced $SAN ok rows, expected 2. Not spending the session."
  exit 4
fi

# ------------------------------------- (b) quality, (c) headline, (d) scaling ---
prev=0
for name in $STEPS; do
  tmo=21600
  # THE ROPE GATE, placed immediately before the first rope group so it can never delay
  # the quality or scaling stages. Its verdict is REPORTED, never enforced: an arm that
  # fails still gets its rows, and its acceptance is then marked as coming from an
  # incoherent model. Driven by SESSION_GATE_CANDIDATES so the stage list stays config-
  # driven and no other session pays for it.
  if [ "${GATE_DONE:-0}" != "1" ] && [ -n "${SESSION_GATE_CANDIDATES:-}" ]; then
    case "$name" in ROPE_*)
      GATE_DONE=1
      log "### STEP gate: coherence gate on the rope arms"
      timeout 14400 "$PY" scripts/mlsys_coherence_gate.py \
        --candidates "$SESSION_GATE_CANDIDATES" \
        --pg19-meta "${SESSION_GATE_PG19_META:-data/processed/pg19_docs/documents.json}" \
        --out "$OUT/session_a_rope_fix_gate.csv" \
        --gen-dir "$OUT/rope_fix_gate_generated" 2>&1 | tee -a "$OUT/group_ROPE_GATE.log" | tail -20
      log "### STEP gate rc=$? (reported, never enforced)"
      progress ;;
    esac
  fi
  if [ "$prev" != "0" ]; then
    log "SKIP $name: the previous stage failed, so the chain stops here"
    continue
  fi
  run_stage "$name" "$tmo"
  prev=$?

  # STEP c GATE: the 1M sampling check. If the first sampled 1M row is not sane -- no ok
  # row, acceptance outside (0,1], zero spec steps, no tokens -- the session STOPS HERE,
  # before the other T>0 stages run, so a broken sampling path costs one row instead of
  # five hours. Driven by SESSION_CHECK_STAGE so the same code path is rehearsed.
  #
  # NO `prev` GUARD, AND THAT IS THE POINT (closed 2026-10-12). `run_stage` returns
  # non-zero exactly when the stage produced ZERO ok rows, so an empty check stage set
  # prev=1 and the `[ "$prev" = "0" ]` that used to sit here SKIPPED the one inspection
  # that was meant to catch exactly that. The stages after it logged SKIP, the rerun pass
  # retried the empty row WITHOUT re-verifying it, and the script exited 0 -- a "clean"
  # session that had run none of d/e/f/g, with the DONE marker's rc agreeing.
  # An empty check stage is a failure of the check, so it must reach this check and exit
  # 5. See tests/test_mlsys_session_a_check_stage.py.
  #
  # WHAT THIS DOES NOT CLOSE. If a stage BEFORE the check fails, the SKIP guard above
  # `continue`s to the next name, so the check stage is never reached and its check never
  # runs. That path is the designed chain-stop (and the rerun pass below still retries the
  # failed row once), but it too ends in rc=0. It is loud rather than silent -- the stage
  # and its SKIP lines are in the log -- so judge THAT case from the stage log lines, as
  # L1 says. Only an empty check stage that was actually reached exits 5.
  if [ -n "${SESSION_CHECK_STAGE:-}" ] && [ "$SESSION_CHECK_STAGE" = "$name" ]; then
    log "### STEP c check: inspecting $name"
    if "$PY" - "$OUT/session_a_$name.csv" <<'PYCHK'
import csv, os, sys
if not os.path.exists(sys.argv[1]):
    print(f"no CSV at {sys.argv[1]}: the stage produced no row at all"); sys.exit(1)
rows = [r for r in csv.DictReader(open(sys.argv[1])) if r.get('status') == 'ok']
if not rows:
    print("no ok rows"); sys.exit(1)
bad = []
for r in rows:
    a = float(r.get('acceptance_rate') or 0)
    if not (0.0 < a <= 1.0):
        bad.append(f"{r['run_id']}: acceptance {a} outside (0,1]")
    if int(r.get('spec_steps') or 0) < 1:
        bad.append(f"{r['run_id']}: spec_steps 0, this is not a spec row")
    if int(r.get('tokens_generated') or 0) < 1:
        bad.append(f"{r['run_id']}: no tokens generated")
print(f"{len(rows)} ok row(s); " + ("acceptance sane, spec steps set, tokens present"
                                    if not bad else "; ".join(bad)))
sys.exit(1 if bad else 0)
PYCHK
    then
      # NAMES THE STAGE, not "the 1M path": SESSION_CHECK_STAGE can be any spec stage
      # (the seeds/ablations session gates on a 128k one), and a message that names the
      # wrong one is how a reader concludes the wrong thing was checked.
      log "### STEP c check PASSED: $name is sane (acceptance inside (0,1], spec steps set, tokens present); continuing"
    else
      log "### STEP c check FAILED: STOPPING. Nothing else runs at T>0; the quality rows that already landed are kept."
      exit 5
    fi
  fi
done

# ------------------------------------------------------------------ (e) reruns --
for f in "$OUT"/session_a_*.csv; do
  [ -f "$f" ] || continue
  base=$(basename "$f" .csv); name=${base#session_a_}
  bad=$(bad_rows "$f")
  if [ "${bad:-0}" -gt 0 ]; then
    log "### STEP e: retrying $name once ($bad non-ok rows)"
    if [ "$name" = "sanity" ]; then
      timeout 3600 "$PY" run_experiment.py --config "$SANITY_CFG" --nproc "$NPROC" \
        --output "$f" --stage-id session_a_sanity --resume \
        --log-per-token --save-generated-tokens 2>&1 | tee -a "$OUT/group_${name}_retry.log" | tail -20
    else
      timeout 21600 "$PY" run_experiment.py --config "$CFG" --nproc "$NPROC" \
        --groups "$name" --output "$f" --stage-id "session_a_$name" --resume \
        --log-per-token --save-generated-tokens 2>&1 | tee -a "$OUT/group_${name}_retry.log" | tail -20
    fi
    log "### STEP e $name retry rc=$? ok_rows=$(ok_rows "$f") bad_rows=$(bad_rows "$f")"
    progress
  fi
done

progress
log "=== COMBINED SESSION WORK COMPLETE ==="
log "stage CSVs: $(ls "$OUT"/session_a_*.csv 2>/dev/null | wc -l | tr -d ' '), ok rows: $(cat "$OUT"/session_a_*.csv 2>/dev/null | grep -c ',ok,')"
log "=== COMBINED SESSION REMOTE DONE ==="

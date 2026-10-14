#!/usr/bin/env bash
# POD side of the OPERATOR-LEVEL profile: which ops actually consume the decode step.
#
# WHY THIS EXISTS. Stage b measured a 128k decode step as 3.79 s of which <=1.7% is ring
# transfer and ~87-98% is local work -- while prefilling all 128k tokens costs 8.8 s. A
# per-step path that costs a third of a full prefill is doing far too much work, and the
# three-bucket sidecar (compute/comm/idle/other) cannot say WHICH ops. This writes the
# top-N ops by SELF DEVICE TIME, with call counts, next to the row it profiled.
#
# WHY NOT `--profile`. The runner's flag drives `src/analysis/profiler.py`, which bins
# events into four buckets and writes only their totals. Getting per-op time through that
# path means editing `src/`, which the campaign forbids for anything that could touch a
# timing it has already published. So this wrapper does its own `torch.profiler.profile`
# around `generate_text()` and calls `key_averages()` itself: no `src/` change, and the
# engine is built from the same config keys every other row uses.
#
# WHAT IT IS NOT. Not a throughput row: 24 tokens and a profiler, so its own tok/s is not
# comparable to anything. It is a denominator for "where does the step go".
#
# Usage, from the driver (all env vars, so the exact command line can be rehearsed):
#   PROFILE_OUT=results/mlsys/ops_profile OPS_CTX=131072 OPS_TOKENS=24 \
#   OPS_BOOK=pg19_train_0 OPS_QUANT=headline bash scripts/mlsys_ops_remote.sh
set -uo pipefail
cd "$HOME/RASD" || exit 3
OUT=${PROFILE_OUT:-results/mlsys/ops_profile}
CTX=${OPS_CTX:-131072}
TOKENS=${OPS_TOKENS:-24}
BOOK=${OPS_BOOK:-pg19_train_0}
QUANT=${OPS_QUANT:-headline}          # headline = FP4 weights + NF4 KV | bf16 = neither
TOP=${OPS_TOP:-20}
mkdir -p "$OUT"
PODDONE="$OUT/DONE"

log() { printf '%s %s\n' "$(date -u +%FT%TZ)" "$*" | tee -a "$OUT/ops.log"; }
remote_finish() {
  local rc=$?
  printf '%s %s\n' "$(date -u +%FT%TZ)" "$rc" > "$PODDONE" 2>/dev/null
  log "=== OPS PROFILE EXIT rc=$rc; DONE marker written ==="
}
trap remote_finish EXIT

if [ -s "$HOME/.hf_token" ]; then
  HF_TOKEN=$(cat "$HOME/.hf_token"); export HF_TOKEN
  log "HF token: read from ~/.hf_token ($(wc -c < "$HOME/.hf_token" | tr -d ' ') bytes)"
else
  log "WARN: no ~/.hf_token; gated repos will fail"
fi
PY=$(grep -a "^INTERPRETER=" "$HOME/pod_env.log" 2>/dev/null | tail -1 | cut -d= -f2-)
if [ -z "${PY:-}" ] || [ ! -x "${PY:-}" ]; then
  log "FATAL: no usable INTERPRETER= in ~/pod_env.log"; exit 2
fi
# shellcheck disable=SC1090
set -a; . "$HOME/RASD/.pod_env.sh"; set +a
log "=== ops profile: interpreter $PY ctx=$CTX tokens=$TOKENS book=$BOOK quant=$QUANT out=$OUT ==="
"$PY" -c 'import torch;print("torch",torch.__version__,"cuda",torch.cuda.device_count())' 2>&1 | tee -a "$OUT/ops.log"

# ---- THE PARAMETER CONTRACT, BEFORE THE MODEL LOAD --------------------------
# shellcheck disable=SC1090
. scripts/mlsys_remote_expect.sh
mlsys_expect "ops-remote" "${REMOTE_EXPECT:-}" \
  "PROFILE_OUT=${PROFILE_OUT-<absent>};OPS_CTX=${OPS_CTX-<absent>};OPS_TOKENS=${OPS_TOKENS-<absent>};OPS_BOOK=${OPS_BOOK-<absent>};OPS_QUANT=${OPS_QUANT-<absent>}" || exit 2

"$PY" - "$OUT" "$CTX" "$TOKENS" "$BOOK" "$QUANT" "$TOP" <<'PYOPS'
import csv, json, os, sys, time
sys.path.insert(0, os.getcwd())
out, ctx, tokens, book, quant, top = (sys.argv[1], int(sys.argv[2]), int(sys.argv[3]),
                                      sys.argv[4], sys.argv[5], int(sys.argv[6]))
import torch
from torch.profiler import ProfilerActivity, profile

# The runner's own prompt builder: the same aligned 128k window, the same exact-window
# search, so this row is the same prompt the published 128k rows used. Imported, not
# copied -- a second implementation of the prompt rule is how two rows stop being the
# same cell.
import run_experiment as R
from src.models.rasd_inference import RASDConfig, RASDInference
from transformers import AutoTokenizer

TARGET = "gradientai/Llama-3-8B-Instruct-Gradient-1048k"
TARGET_REV = "cd3069b65a8eb13da639d332a5f61b0fbb29fa73"
DRAFT = "meta-llama/Llama-3.2-1B"
DRAFT_REV = "4e20de362430cd3b72f300e6b0f18e50e7166e08"
kw = dict(
    target_model_name=TARGET, target_revision=TARGET_REV,
    draft_model_name=DRAFT, draft_revision=DRAFT_REV,
    spec_steps=0,                      # target-only: the decode step under investigation
    kv_block_size=2048, prefetch_depth=1, rope_type="none",
    dtype="bfloat16", top_p=1.0, ignore_eos=True, temperature=0.7,
    max_new_tokens=tokens, draft_window_cap=4096, seed=42,
    log_per_token=False, save_generated_tokens=False, quantize_draft=True,
    quantize_target=(quant != "bf16"),
    kv_quant=(quant != "bf16"),
)
tok = AutoTokenizer.from_pretrained(TARGET, revision=TARGET_REV)
prompt_text, cont, prov = R._build_pg19_document_prompt(
    "data/processed/pg19_1m_128kwin/documents.json", ctx, book, tok, gen_tokens=1024)
print(f"[ops] prompt built for {book} at {ctx}: {len(prompt_text)} chars, "
      f"continuation {len(cont)} tokens", flush=True)
engine = RASDInference(RASDConfig(**kw))

t0 = time.perf_counter()
with profile(activities=[ProfilerActivity.CUDA, ProfilerActivity.CPU],
             record_shapes=False, profile_memory=False, with_stack=False) as prof:
    text, metrics = engine.generate_text(prompt_text)
wall = time.perf_counter() - t0

# SELF DEVICE TIME, by the name torch 2.5 uses, falling back to the older spelling.
rows, total_us, field = [], 0.0, "self_device_time_total"
for e in prof.key_averages():
    v = getattr(e, field, None)
    if v is None:
        field = "self_cuda_time_total"
        v = getattr(e, field, 0.0) or 0.0
    rows.append((float(v), int(getattr(e, "count", 0) or 0), str(e.key)))
total_us = sum(r[0] for r in rows)
rows.sort(reverse=True)
ops = [{"op": n, "self_device_us": round(us, 1), "self_device_s": round(us / 1e6, 4),
        "calls": c, "pct_of_self_device": round(us / total_us * 100, 2) if total_us else 0.0}
       for us, c, n in rows[:top]]
json.dump({"row": {"run_id": f"OPS_{book}_ctx{ctx}_tok{tokens}_{quant}",
                   "context_length": ctx, "book": book, "quant": quant,
                   "profiled_tokens": tokens,
                   "tokens_generated": metrics.get("tokens_generated"),
                   "time_sec": metrics.get("time_sec"), "ttft_ms": metrics.get("ttft_ms"),
                   "gpu_peak_mem_mb": metrics.get("gpu_peak_mem_mb"),
                   "profiler_wall_s": round(wall, 4),
                   "quantize_target": kw["quantize_target"], "kv_quant": kw["kv_quant"]},
           "self_device_field": field,
           "self_device_total_s": round(total_us / 1e6, 4),
           "ops_above": [r[2] for r in rows[top:top + 10]],
           "ops": ops},
          open(os.path.join(out, "ops_top%d.json" % top), "w"), indent=1)

with open(os.path.join(out, "ops_profile.csv"), "w", newline="") as fh:
    w = csv.writer(fh)
    w.writerow(["run_id", "book", "context_length", "quant", "tokens_generated",
                "time_sec", "ttft_ms", "gpu_peak_mem_mb", "profiler_wall_s",
                "self_device_total_s", "status"])
    w.writerow([f"OPS_{book}_ctx{ctx}_tok{tokens}_{quant}", book, ctx, quant,
                metrics.get("tokens_generated"), metrics.get("time_sec"),
                metrics.get("ttft_ms"), metrics.get("gpu_peak_mem_mb"), round(wall, 4),
                round(total_us / 1e6, 4), "ok"])
print(f"[ops] row: {metrics.get('tokens_generated')} tokens / {metrics.get('time_sec'):.3f} s, "
      f"ttft {metrics.get('ttft_ms'):.1f} ms, profiler wall {wall:.1f} s", flush=True)
for o in ops[:10]:
    print(f"[ops]   {o['self_device_s']:9.3f} s  {o['pct_of_self_device']:5.1f}%  "
          f"x{o['calls']:<7d} {o['op'][:70]}", flush=True)
print(f"[ops] self device total {total_us/1e6:.1f} s over {len(rows)} distinct ops", flush=True)
PYOPS
prc=$?
log "### ops profiler rc=$prc"
[ -s "$OUT/ops_top${TOP}.json" ] || { log "FAILED: no ops JSON was written"; exit 1; }
log "### wrote $OUT/ops_top${TOP}.json and $OUT/ops_profile.csv"

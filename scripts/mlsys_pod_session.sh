#!/usr/bin/env bash
# MLSys experiment program — bundled single-command pod session.
#
# Runs the whole MLSys program end to end on the 8x A100-80GB node and
# records GPU-hours + cost per phase. Every stage is resumable via a
# marker file, so a crash or an interrupted shell resumes rather than
# restarting.
#
#   PHASE 0  env report / tokenizer identity / ring binding / transformers pin
#   PHASE 1  3-arm native-vs-YaRN experiment (the headline)
#   PHASE 2  multi-seed PG-19 dose-response + 3-seed per-round traces
#   PHASE 3  bf16-draft mechanism isolation @64k
#   PHASE 4  dip test + acceptance accounting + bootstrap CIs
#   PHASE 5  vLLM production-stack baseline @128k
#   REPORT   master table + GPU-hours / cost breakdown
#
# PRE-FLIGHT (once, before this script):
#   conda env create -f environment_gpu.yml && conda activate rasd-gpu
#   pip install --no-build-isolation flash-attn>=2.4.0
#   pip install -e .
#   export WANDB_API_KEY=... HF_TOKEN=... HF_HOME=/workspace/hf_cache
#   # vLLM needs a SEPARATE venv (it conflicts with the pinned transformers):
#   python -m venv ~/venv-vllm && ~/venv-vllm/bin/pip install vllm
#
# Usage:
#   NPROC=8 bash scripts/mlsys_pod_session.sh
#
# Env knobs:
#   NPROC               GPUs per run (default 8)
#   NODE_RATE_PER_HOUR  $/hr for the whole node, for cost reporting
#                       (default 15.92, the 8xA100 rate used for M4)
#   SKIP_VLLM=1         skip Phase 5
#   TIMEOUT_LONG_S      per-run timeout for >=128k cells (default 14400)
#
# Read this before interpreting results:
#   * configs assume 8x A100 80GB. On 40GB use the *_40gb.yml variants and
#     expect arm 3 (native draft window) to OOM — that OOM is a result.
#   * Phase 5's vLLM throughput must be compared at the SAME unit as
#     RASD's throughput_tps (end-to-end generation tok/s). The script
#     prints a warning for any row it could not unit-match.

set -euo pipefail

cd "$(dirname "$0")/.."

: "${WANDB_API_KEY:?WANDB_API_KEY must be set}"
: "${HF_TOKEN:?HF_TOKEN must be set}"
: "${HF_HOME:?HF_HOME must be set (use /workspace/hf_cache on Lambda)}"

NPROC="${NPROC:-8}"
NODE_RATE_PER_HOUR="${NODE_RATE_PER_HOUR:-15.92}"
# Hard budget guard. Before each stage the cumulative cost is recomputed
# from gpu_hours.csv; once it reaches this figure no further stage is
# launched, so an unattended session cannot run past the ceiling. The
# final `report` stage is exempt (we always want the cost breakdown).
MAX_COST_USD="${MAX_COST_USD:-600}"
TIMEOUT_LONG_S="${TIMEOUT_LONG_S:-14400}"
SKIP_VLLM="${SKIP_VLLM:-0}"
SKIP_LLAMA3="${SKIP_LLAMA3:-0}"   # set 1 if the tokenizer check fails

OUT=results/mlsys
LOG_DIR=$OUT/logs
mkdir -p "$OUT" "$LOG_DIR" "$OUT/per_token"

export NCCL_TIMEOUT=3600
export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=3600
export TORCH_NCCL_BLOCKING_WAIT=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

COST_LOG=$OUT/gpu_hours.csv
if [ ! -f "$COST_LOG" ]; then
    echo "stage,wall_seconds,nproc,gpu_hours,node_cost_usd" > "$COST_LOG"
fi

# Cumulative spend recorded so far (column 5 of the cost log).
cumulative_cost() {
    if [ -f "$COST_LOG" ]; then
        awk -F, 'NR>1 { c += $5 } END { printf "%.2f", c+0 }' "$COST_LOG"
    else
        echo "0.00"
    fi
}

# --- stage helper: budget guard + resume marker + timing + accounting -----
stage() {
    local name="$1"; shift
    local marker="$OUT/.markers/${name}.done"
    mkdir -p "$OUT/.markers"
    if [ -f "$marker" ]; then
        echo "==> [$name] already complete, skipping"
        return 0
    fi
    # Budget guard: refuse to START a new stage once the ceiling is hit.
    # `report` is exempt so the cost breakdown always gets written.
    if [ "$name" != "report" ]; then
        local spent
        spent=$(cumulative_cost)
        if awk -v s="$spent" -v m="$MAX_COST_USD" 'BEGIN{exit !(s>=m)}'; then
            echo ""
            echo "!!! BUDGET GUARD TRIPPED: cumulative \$$spent >= \$$MAX_COST_USD"
            echo "!!! Not launching [$name]. Stopping so the instance can be terminated."
            echo "$(date -u +%FT%TZ) BUDGET_GUARD_TRIPPED spent=$spent limit=$MAX_COST_USD at=$name" \
                >> "$OUT/RUN_LOG.txt"
            exit 99
        fi
    fi
    echo ""
    echo "============================================================"
    echo "==> [$name] starting $(date -u +%Y-%m-%dT%H:%M:%SZ)  (spent so far: \$$(cumulative_cost))"
    echo "============================================================"
    local t0 t1 rc
    t0=$(date +%s)
    "$@" 2>&1 | tee "$LOG_DIR/${name}.log"
    rc=${PIPESTATUS[0]}
    t1=$(date +%s)
    local wall=$((t1 - t0))
    # GPU-hours = wall-clock hours x GPUs held for the whole stage.
    awk -v s="$name" -v w="$wall" -v n="$NPROC" -v r="$NODE_RATE_PER_HOUR" \
        'BEGIN { gh = (w/3600.0)*n; cost = (w/3600.0)*r;
                 printf "%s,%d,%d,%.4f,%.2f\n", s, w, n, gh, cost }' \
        >> "$COST_LOG"
    if [ "$rc" -ne 0 ]; then
        echo "!!! [$name] FAILED (rc=$rc). Log: $LOG_DIR/${name}.log"
        echo "!!! Markers are per-stage; fix and re-run to resume."
        exit "$rc"
    fi
    touch "$marker"
    echo "==> [$name] done in ${wall}s"
}

run_grid() {  # $1 = config, $2 = output csv, $3... = extra runner flags
    local cfg="$1"; local out="$2"; shift 2
    python run_experiment.py \
        --config "$cfg" \
        --output "$out" \
        --nproc "$NPROC" \
        --resume \
        --timeout-per-run-s "$TIMEOUT_LONG_S" \
        "$@"
}

# =========================================================================
# PHASE 0 — setup & compatibility
# =========================================================================

p0_env() {
    echo "--- GPU inventory ---"
    nvidia-smi
    echo
    echo "--- leak check (any GPU with >0 MiB used = dirty node) ---"
    nvidia-smi --query-gpu=index,memory.used --format=csv,noheader
    echo
    echo "--- non-NVIDIA / driver detail ---"
    nvidia-smi --query-gpu=index,name,memory.total,driver_version \
        --format=csv,noheader
    echo
    echo "--- python env ---"
    python -c "import torch, transformers; \
print('torch', torch.__version__, 'cuda', torch.cuda.is_available()); \
print('transformers', transformers.__version__)"
    nproc
    free -g || true
}

p0_transformers_pin() {
    # Gate on the CAPABILITY the ring patch needs, not just a version
    # string. 4.48.0 keeps a 4.x version while removing
    # self.num_key_value_heads and moving rotary_emb — so a ">=4.46 and
    # <5" check passed on a stack that cannot run ring attention at all.
    python - <<'PY'
import sys
try:
    import transformers
    v = transformers.__version__
    print(f"transformers {v}")
except Exception as e:
    print("FAIL: transformers not importable:", e); sys.exit(1)

try:
    try:
        from transformers.models.llama.configuration_llama import LlamaConfig
    except Exception:
        from transformers import LlamaConfig
    try:
        from transformers.models.llama.modeling_llama import LlamaForCausalLM
    except Exception:
        from transformers import LlamaForCausalLM
    cfg = LlamaConfig(vocab_size=256, hidden_size=128, intermediate_size=256,
                      num_hidden_layers=1, num_attention_heads=8,
                      num_key_value_heads=2, max_position_embeddings=131072)
    attn = LlamaForCausalLM(cfg).model.layers[0].self_attn
    need = ["q_proj", "k_proj", "v_proj", "o_proj",
            "num_key_value_heads", "rotary_emb"]
    missing = [a for a in need if not hasattr(attn, a)]
except Exception as e:
    print("FAIL: could not probe LlamaAttention:", type(e).__name__, e)
    sys.exit(1)

if missing:
    print("FAIL: LlamaAttention is missing", missing)
    print("  The ring patch reads self.num_key_value_heads and")
    print("  self.rotary_emb. transformers 4.48.0 removed/moved them, and")
    print("  5.x did the same. Required range: >=4.46,<4.48")
    sys.exit(1)
print("PASS: LlamaAttention has the surface the ring patch needs")
PY
}

p0_tokenizer() {
    # The tokenizer check exits 1 (divergent) or 2 (no gated access).
    # NEITHER is a reason to abort: per the run instructions, no access
    # means "skip the native arms and run everything else". Making this
    # stage fatal aborted two whole runs at rc=2.
    # NB: the check MUST run in a condition context. Under `set -e` a bare
    # failing command aborts the entire script *before* `local rc=$?` is
    # ever reached — that is precisely what killed the 8xA100 run at
    # Phase 0.2 (the check's own PASS/FAIL lines printed, then the session
    # exited 1 with no stage banner). `if ...; then` suppresses set -e.
    local rc=0
    if python scripts/mlsys_check_tokenizer_llama3.py; then
        rc=0
    else
        rc=$?
    fi
    if [ "$rc" -eq 0 ]; then
        echo "[tokenizer] identity verified — native arms enabled"
    else
        echo "[tokenizer] rc=$rc => no gated-model access;"
        echo "[tokenizer] arms 2/3 will be recorded as 'skipped: no gated-model access'"
        mkdir -p "$OUT/.markers"
        touch "$OUT/.markers/SKIP_NATIVE_ARMS"
    fi
    return 0
}

# True when the native (Llama-3) arms must be skipped. Checks BOTH the
# env var and a marker file, because `stage` runs its function in a
# pipeline subshell where `export` would not reach the arm stages.
skip_native_arms() {
    [ "${SKIP_LLAMA3:-0}" = "1" ] || [ -f "$OUT/.markers/SKIP_NATIVE_ARMS" ]
}

p0_ring_binding() {
    # The check verifies the ring patch binds to Llama-3.1's attention
    # surface. When the native arms are already disabled there is nothing
    # to verify, and a failure here must NOT kill a run whose remaining
    # work (Llama-2 + Sheared) is the production config.
    if [ "${SKIP_LLAMA3:-0}" = "1" ]; then
        echo "SKIPPED: native arms disabled (no gated-model access) — the"
        echo "         Llama-3.1 ring-binding check is not applicable."
        echo "         Arm 1 / Phase 2 / Phase 3 use the production Llama-2"
        echo "         ring path, exercised by every M3/M4 run."
        return 0
    fi
    python scripts/mlsys_ring_binding_check.py
}

# =========================================================================
# PHASE 1 — 3-arm native-vs-YaRN experiment
# =========================================================================

p1_arm1() {
    run_grid configs/mlsys_arm1_llama2_yarn.yml \
        "$OUT/arm1_llama2_yarn.csv" --log-per-token --memory-trace
}

p1_arm2() {
    if skip_native_arms; then
        echo "SKIP arm 2 (native target, capped draft): no gated-model access"
        echo "skipped: no gated-model access" >> "$OUT/RUN_LOG.txt"
        return 0
    fi
    run_grid configs/mlsys_arm2_native_cappeddraft.yml \
        "$OUT/arm2_native_cappeddraft.csv" --log-per-token --memory-trace
}

p1_arm3() {
    if skip_native_arms; then
        echo "SKIP arm 3 (native draft window): no gated-model access"
        echo "skipped: no gated-model access" >> "$OUT/RUN_LOG.txt"
        return 0
    fi
    run_grid configs/mlsys_arm3_native_nativedraft.yml \
        "$OUT/arm3_native_nativedraft.csv" --log-per-token --memory-trace
}

# =========================================================================
# PHASE 2 — multi-seed analyses
# =========================================================================

p2_pg19_spec() {
    run_grid configs/mlsys_pg19_multiseed.yml \
        "$OUT/pg19_multiseed.csv" \
        --log-per-token --memory-trace \
        --prompt-source pg19 \
        --prompt-pg19-meta data/processed/pg19/pg19_validation_metadata.json
}

p2_pg19_1m_target_only() {
    run_grid configs/mlsys_pg19_1m_targetonly.yml \
        "$OUT/pg19_multiseed.csv" \
        --log-per-token --memory-trace \
        --prompt-source pg19 \
        --prompt-pg19-meta data/processed/pg19/pg19_validation_metadata.json
}

p2_matrix_traces() {
    # Seed-42 traces already exist from Phase D; bring them alongside the
    # new ones so all three seeds live in one directory for the dip test.
    for f in results/final/per_token/RASD_ctx128k_phaseD_s42.jsonl \
             results/final/per_token/RASD_ctx256k_phaseD_s42.jsonl \
             results/final/per_token/RASD_ctx512k_phaseD_s42.jsonl \
             results/final/per_token/RASD_ctx1M_phaseD_s42.jsonl; do
        if [ -f "$f" ] && [ ! -f "$OUT/per_token/$(basename "$f")" ]; then
            cp "$f" "$OUT/per_token/"
            echo "seeded $(basename "$f") from results/final/per_token/"
        fi
    done
    # Seeds 123/456 -> sidecars land in $OUT/per_token/ because the output
    # csv lives in $OUT (see _per_token_sidecar_path in run_experiment.py).
    run_grid configs/mlsys_llama2_matrix_multiseed.yml \
        "$OUT/llama2_matrix_multiseed.csv" --log-per-token
}

# =========================================================================
# PHASE 3 — mechanism isolation
# =========================================================================

p3_bf16_draft() {
    run_grid configs/mlsys_bf16_draft_isolation.yml \
        "$OUT/bf16_draft_isolation.csv" --log-per-token --memory-trace
}

# =========================================================================
# PHASE 4 — statistics
# =========================================================================

p4_analysis() {
    python scripts/mlsys_analysis.py \
        --results-dir "$OUT" --trace-dir "$OUT/per_token"
}

# =========================================================================
# PHASE 5 — vLLM production-stack baseline
# =========================================================================

p5_vllm() {
    if [ "$SKIP_VLLM" = "1" ]; then
        echo "SKIP_VLLM=1 — skipping Phase 5"
        return 0
    fi
    if [ ! -x "$HOME/venv-vllm/bin/python" ]; then
        echo "no ~/venv-vllm found — creating it (vLLM conflicts with the"
        echo "pinned transformers, so it must NOT go in the main env)"
        python -m venv "$HOME/venv-vllm"
        "$HOME/venv-vllm/bin/pip" install -q vllm
    fi
    # vLLM measures its own wall clock; the node-rate accounting for this
    # stage is still the node rate x wall time, recorded by `stage`.
    "$HOME/venv-vllm/bin/python" scripts/mlsys_vllm_baseline.py \
        --models meta-llama/Llama-2-7b-hf meta-llama/Llama-3.1-8B \
        --context-lengths 131072 \
        --max-new-tokens 64 \
        --out "$OUT/vllm_baseline.csv" || true
    # A missing/failed vLLM run must not abort the session: its absence is
    # itself a reportable outcome, so swallow the exit code.
}

# =========================================================================
# REPORT
# =========================================================================

report() {
    echo ""
    echo "============================================================"
    echo "==> MLSys session cost accounting"
    echo "============================================================"
    column -s, -t "$COST_LOG" || cat "$COST_LOG"
    echo
    awk -F, -v r="$NODE_RATE_PER_HOUR" \
        'NR>1 { gh += $4; cost += $5 } END {
            printf "TOTAL GPU-hours: %.2f\n", gh;
            printf "TOTAL node cost: $%.2f  (at $%.2f/hr/node)\n", cost, r;
        }' "$COST_LOG"
    echo
    echo "Artefacts:"
    ls -la "$OUT"/*.csv 2>/dev/null || true
    echo
    echo "Next: fill results/mlsys/MASTER_TABLE.txt from these CSVs."
}

# =========================================================================
# RUN
# =========================================================================
stage p0_env              p0_env
stage p0_transformers     p0_transformers_pin
stage p0_ring_binding     p0_ring_binding
stage p0_tokenizer        p0_tokenizer

stage p1_arm1             p1_arm1
stage p1_arm2             p1_arm2
stage p1_arm3             p1_arm3

stage p2_pg19_spec        p2_pg19_spec
stage p2_pg19_1m_target   p2_pg19_1m_target_only
stage p2_matrix_traces    p2_matrix_traces

stage p3_bf16_draft       p3_bf16_draft

stage p4_analysis         p4_analysis

stage p5_vllm             p5_vllm

stage p4_analysis_final   p4_analysis
stage report              report

echo ""
echo "==> MLSys session complete at $(date -u +%Y-%m-%dT%H:%M:%SZ)"

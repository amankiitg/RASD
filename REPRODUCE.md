# Reproducing RASD Results

A cold-start guide: hardware → env → command → expected outputs. This
file is the canonical entry point for anyone trying to reproduce or
extend the M3 ablation study.

## TL;DR

Two environments, two purposes:

| Goal | Environment | File | Hardware |
|---|---|---|---|
| Re-run experiments (M3 ablation, M4 final matrix) | `rasd-gpu` conda | `environment_gpu.yml` | 8× A100-SXM4-40 GB |
| Regenerate figures, run tests, develop offline | `.venv_analysis` venv | `requirements.txt` | any CPU (Python 3.10+) |

The two are intentionally separate. The pod env pins
`transformers==4.44.2 + accelerate==0.33.0 + bitsandbytes>=0.49.0` —
those exact versions produced every M3 row. The local env tracks newer
libraries appropriate for analysis but is not used for inference.

## Pinned versions reference

- **Pod (canonical for results):** `environment_gpu.yml`
  - `python=3.10` / `pytorch=2.1.0` / `pytorch-cuda=12.1`
  - `transformers==4.44.2` (4.38 lacks dynamic RoPE; 5.x breaks tuple `past_kv`)
  - `accelerate==0.33.0` (≥1.0 breaks bitsandbytes `.to()` on quantized models)
  - `bitsandbytes>=0.49.0` (<0.45 missing CUDA 12.9 .so; 0.44 has triton import error)
  - `flash-attn>=2.4.0` (install with `--no-build-isolation`)
- **Local (analysis + tests + figures):** `requirements.txt`
  - `python>=3.10` (tested on 3.14)
  - `torch==2.11.0` CPU build, `transformers==5.5.4`, `pytest==9.0.3`
  - `numpy / pandas / scipy / matplotlib / seaborn / tabulate`

## 1. Reproducing the M3 ablation (R6.5, 49 rows, 64k context)

### Hardware

Lambda Cloud `gpu_8x_a100` SKU = 8× A100-SXM4-40 GB, $15.92/hr in
`europe-central-1` (most reliably available region for this SKU as of
2026-05). Region-locked filesystems are not used by R6.5 — results scp
back to the dev machine each session.

### Environment setup (on pod)

```bash
# After SSH'ing into a fresh Lambda 8×A100 instance:
conda env create -f environment_gpu.yml
conda activate rasd-gpu

# flash-attn must be installed separately (build_isolation strips torch):
pip install --no-build-isolation flash-attn>=2.4.0

# Capture exact transitive dep versions for this pod (REPRODUCE-d):
bash scripts/capture_pod_env.sh > requirements-lock.txt
git add requirements-lock.txt && git commit -m "Pod env lock"
```

### Pin Hugging Face model revisions (REPRODUCE-e)

Llama-2 is gated; the rest are public. On pod with `HF_TOKEN` set:

```bash
export HF_TOKEN=<your token>
python scripts/pin_hf_revisions.py
# updates configs/ablations.yml with model commit hashes for:
#   - meta-llama/Llama-2-7b-hf
#   - meta-llama/Llama-2-13b-hf
#   - princeton-nlp/Sheared-LLaMA-1.3B (already pinned)
#   - TinyLlama/TinyLlama-1.1B-step-2T (already pinned)
git add configs/ablations.yml && git commit -m "Pin Llama-2 revisions"
```

### Sanity-check the pod first

```bash
# 1. GPU leak check — must show 0 MiB on all 8 GPUs:
nvidia-smi --query-gpu=memory.used --format=csv

# 2. Replay smoke (one seed × each group) — asserts throughput within
#    15% of golden CSV. Catches semantic regressions before burning $80
#    on the full grid.
bash scripts/replay_m3_smoke.sh
```

### Run the full ablation

```bash
# Inline env vars (NEVER export — operational guide in configs/ablations.yml):
WANDB_API_KEY=<key> HF_TOKEN=<token> HF_HOME=/workspace/hf_cache \
torchrun --nproc-per-node=8 run_experiment.py \
    --config configs/ablations.yml \
    --output results/ablations/ablations_r65_repro.csv \
    --resume \
    --nproc 8
```

`--resume` skips rows with `status=ok` already in the CSV. Total
runtime: ~4.6 hr for 48 production rows + canary at $15.92/hr ≈ **$80**.

### Expected outputs

- `results/ablations/ablations_r65_repro.csv` — should match
  `results/ablations/ablations_r65.csv` to within seed-level numeric
  tolerances. 46/49 rows succeed; 3 OOM (A5_llama2_13b × 3 seeds —
  predicted 40 GB ceiling failure).
- α range: 0.105 (A2 k=12) → 0.424 (A2 k=2)
- Per-rank memory: ~25 GB / 40 GB across all 46 successful rows
- Wandb runs under your project of choice

Compare against the canonical R6.5 rows in
[`results/ablations/r65_audit.md`](results/ablations/r65_audit.md). If
your reproduction shows >5% absolute α drift, see the four M3 fixes in
the README and verify your branch contains commits `eb9297a`, `e875f6d`,
`45b2b40`, `ad2bf5e`.

### Tagged reference point

```bash
git checkout m3-reablation
```

This tag points at the exact commit that produced `ablations_r65.csv`.

## 2. Reproducing figures + tables locally (no GPU)

Set up the local analysis env once:

```bash
python3 -m venv .venv_analysis
source .venv_analysis/bin/activate
pip install -r requirements.txt
pip install -e .            # makes `from src.models...` importable

# Confirm the env:
pytest tests/                # all 91 tests should pass in ~40s
```

Then regenerate the M3 ablation figures + tables from the canonical CSV:

```bash
# Bootstrap CIs, Figure 2 ablation summary, LaTeX tables:
python scripts/compute_ablation_cis.py \
    --csv results/ablations/ablations_r65.csv
python scripts/plot_figure2.py \
    --csv results/ablations/ablations_r65.csv \
    --out figures/fig2_ablation_heatmap.pdf
python scripts/emit_ablation_table.py \
    --csv results/ablations/ablations_r65.csv \
    --out tables/ablation_summary.tex
```

Outputs land in `figures/` and `tables/`.

## 3. Repository tour for reviewers

| Path | What it is |
|---|---|
| [src/models/rasd_inference.py](src/models/rasd_inference.py) | RASD engine: prefill → ring-attention forward → speculative verify |
| [src/models/ring_attention_kernel.py](src/models/ring_attention_kernel.py) | Layout-agnostic ring kernel with online-softmax merge, FA-2/SDPA dispatch |
| [src/models/ring_llama_attention.py](src/models/ring_llama_attention.py) | LlamaAttention monkey-patch installer |
| [run_experiment.py](run_experiment.py) | Top-level orchestrator: expands ablation grid → torchrun subprocess per row → CSV append |
| [tests/](tests/) | 91 unit tests including [tests/test_m3_invariants.py](tests/test_m3_invariants.py) (locks in M3 fixes) |
| [configs/ablations.yml](configs/ablations.yml) | Ablation grid + RunPod operational guide |
| [M3_RING_INTEGRATION_PLAN.md](docs/dev/M3_RING_INTEGRATION_PLAN.md) | M3 redesign chronicle (audit findings → R6.5 results) |
| [docs/M3_mentor_summary.md](docs/M3_mentor_summary.md) | One-page M3 closure summary |
| [M4_PLAN.md](docs/dev/M4_PLAN.md) | M4 plan (1M context, NF4 KV, 36-run final matrix) |

## 4. Known gotchas

- **NEVER `pkill -9` CUDA processes.** Orphans driver-level VRAM that
  can't be reclaimed without a pod restart. Use SIGTERM only — see
  `configs/ablations.yml` "Clean-kill procedure" section.
- **HF_HOME must be `/workspace/hf_cache`** on Lambda pods (persistent
  volume). Default `~/.cache/huggingface` does not survive pod
  restarts.
- **80 GB SXM4 ≠ 40 GB SXM4.** R6.5 ran at 40 GB; pushing past 64k
  context on 7B target needs NF4 KV-cache (M4 work item C11). 13B at
  64k×W=8 NF4 OOMs at ~40 GB.
- **NCCL v2.26+ required** for the cross-rank logits broadcast (Fix2)
  to work without bf16-drift desync hangs.
- **Subprocess timeout patch**: `run_experiment.py` defaults to 120s
  per-row; long-context runs need 600s. See
  `scripts/r6_verify_runner.sh` for the `sed` patch.

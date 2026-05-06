# RASD — Ring Attention with Speculative Decoding

Distributed long-context LLM inference that combines FlashAttention-blocked
**ring attention** (KV cache partitioned across ranks, rotated via ring P2P)
with **self-speculative decoding** (small draft model proposes tokens, target
model verifies in parallel). The result: single-GPU-budget memory at
long-context scale, with the wall-clock speedup of speculation.

## Repository layout

```
configs/           YAML ablation + experiment specs (see ablations.yml header
                   for the full RunPod operational guide)
src/
├── models/        RASDInference engine, FA-2-blocked ring attention
├── baselines/     Vanilla ring-attention + sliding-window baselines
├── analysis/      (M4) metrics, bootstrap CIs, figures, tables
└── utils/         device selection, helpers
scripts/           standalone benchmarks (baselines, flash-memory validation,
                   PG-19 preprocessing)
run_experiment.py  top-level orchestrator: expands ablation grid, launches
                   per-run torchrun subprocesses, appends rows to CSV
results/
├── ablations/     ablations.csv — 49 rows from M3 ablation grid
├── baselines/     baselines.csv, flash_memory_validation.csv
└── final/         (M4) final_results.json, per-seed final runs
tests/             unit tests for RASD components + ring protocol
figures/, tables/, manuscript/  (M4) paper deliverables
environment_gpu.yml  pinned conda env (transformers==4.44.2,
                     accelerate==0.33.0, bitsandbytes>=0.49.0, flash-attn>=2.4)
```

## Current status

- **Milestone 1** — literature review (`literature_review/`)
- **Milestone 2** — baselines (ring attention, sliding window) + data pipeline
- **Milestone 3** — ablation study **complete (after re-ablation)**.
  Original M3 results invalidated 2026-04-16 by post-analysis audit;
  architecture rebuilt R0–R3.5 + R5 (May 2026), 4 fixes landed, R6.5
  re-ablation produced 46/49 valid rows at **ctx=64k × 8× A100-SXM4-40GB**
  in `results/ablations/ablations_r65.csv`. Tagged
  [`m3-reablation`](../../tree/m3-reablation).
- **Milestone 4** — evaluation & analysis at 1M context (in progress)

### Key findings from M3 (R6.5 re-ablation, see [`results/ablations/r65_audit.md`](results/ablations/r65_audit.md))

| axis | finding | note |
|---|---|---|
| **A2** spec_steps k | α decreases monotonically with k | k=2: α=0.38; k=12: α=0.11. tps roughly flat (more tokens/round offset lower α/token). Sweet spot k∈{4,8}. |
| **A3** chunk_size (ring P2P) | larger chunks = higher tps, monotonic | 256 → 2048 yields tps 0.63 → 1.23 (~94% gain). α invariant. Default 512 too conservative; production should use 1024-2048. |
| **A4** prefetch_depth | sync == async-1 == async-2 | **identical to logged precision at every seed.** Modern NCCL (v2.26+) handles compute/comm overlap on its internal stream — explicit Python prefetch adds zero benefit at this scale. Useful negative result. |
| **A1** draft size | TinyLlama-1.1B ≈ Sheared-LLaMA-1.3B | both within seed variance; 18% extra draft params don't decisively help α. |
| **A5** target | Llama-2-7B works (α=0.253, tps=0.87); 13B OOMs at ctx=64k×W=8 NF4 on 40 GB | motivates M4 NF4 KV-cache work to push 13B and 1M context. |

α range across 46 successful rows: **0.105 (A2 k=12 floor) to 0.424 (A2 k=2 best)** — 3-7× higher than the original M3 baseline (0.06-0.11) at floor and ceiling. Memory rock-stable at ~25 GB/rank across all rows, identical across all 8 ranks every row.

Wandb project: [`rasd-m3-reablation-64k`](https://wandb.ai/amank-iitg-uc-berkeley-electrical-engineering-computer-s/rasd-m3-reablation-64k)

### Critical engineering fixes (R6 session 2026-05-06)

The four fixes that took the architecture from "structurally broken" to "produces clean, deterministic, paper-grade ablation data":

| Fix | Commit | What it does |
|---|---|---|
| **Option B** | [`eb9297a`](../../commit/eb9297a) | Don't RoPE-scale the draft model — caps draft at native 4k context. Saves ~11 GB/rank at ctx=64k by shrinking replicated draft KV from ~12 GB → ~770 MB. |
| **Fix2** | [`e875f6d`](../../commit/e875f6d) | Broadcast `target_logits_v` and `draft_logits` from rank 0 before accept/reject. Eliminates cross-rank divergence from bf16-noise drift in ring online-softmax (root cause of NCCL coalesced-op timeouts at high iteration counts). |
| **Fix3** | [`45b2b40`](../../commit/45b2b40) | Auto-truncate prompt to multiple of `world_size` in `_prefill`. Tokenizers regularly return off-by-a-few token counts; the divisibility assertion was crashing rank 0. |
| **Fix4** | [`ad2bf5e`](../../commit/ad2bf5e) | Remove legacy `_ring_peer_loop` master/slave pattern from `run_experiment.py`. After R3 deleted the prefetcher, all ranks must run the full pipeline in lockstep. |

Full chronicle in [`M3_RING_INTEGRATION_PLAN.md`](M3_RING_INTEGRATION_PLAN.md) (fix log + live R6.5 findings at the bottom). Mentor summary: [`docs/M3_mentor_summary.md`](docs/M3_mentor_summary.md).

## Quick start

### Local smoke test (1 GPU or CPU)

```bash
conda env create -f environment_gpu.yml
conda activate rasd-gpu
pytest tests/
python run_experiment.py --config configs/ablations.yml --dry-run
```

### Full ablation grid (8×A100 node)

See the **"RunPod Operational Guide"** header in
[`configs/ablations.yml`](configs/ablations.yml) — it has the full
required sequence:

1. GPU leak check (`nvidia-smi` must show 0 MiB on all 8 GPUs)
2. Dependency install (pinned versions)
3. Subprocess timeout patch (`sed` in `run_experiment.py`)
4. Inline env-var launch (`WANDB_API_KEY`, `HF_TOKEN`, `HF_HOME`)
5. Clean-kill procedure (SIGTERM only — never `pkill -9` CUDA procs, it
   orphans driver-level VRAM that can't be reclaimed without pod restart)

```bash
python run_experiment.py \
  --config configs/ablations.yml \
  --output results/ablations/ablations.csv \
  --resume --nproc 8
```

`--resume` skips any row with `status=ok` already in the CSV.

## Experiment artifacts

- **CSV rows** go to the per-group subdirectory under `results/`
- **wandb logs** under project `rasd-ablations` (free tier OK)
- **Models cached** on `/workspace/hf_cache` (persistent volume, *not* `/root`)

## Hardware notes

- **R6.5 re-ablation (May 2026)** ran on **Lambda 8× A100-SXM4-40GB**
  (`gpu_8x_a100`, $15.92/hr) in europe-central-1. Per-rank memory at
  ctx=64k×W=8 NF4 stays at ~25 GB / 40 GB stably across all 46
  successful ablation rows. Lambda operational guide in
  [`runpod_creds.md`](runpod_creds.md) (gitignored — contains creds).
- **Original M3 (April 2026)** ran on RunPod 8× A100-SXM 80GB. Pre-audit
  results in `results/ablations/ablations.csv` (invalidated; superseded
  by `ablations_r65.csv`).
- 4-bit NF4 quantization (via `bitsandbytes`) on both draft and target.
- Empirical 40 GB SXM4 ceiling at **ctx≈64k for the 7B target**.
  ctx=128k OOMs at ~38 GB/rank even with `expandable_segments`.
  Pushing past 64k on 40 GB hardware needs NF4 KV-cache quantization
  (queued for M4 as `C11`).

## Acknowledgements

This work uses **Lambda Labs research compute credits ($2,000)** for the
post-audit R6 verification + R6.5 re-ablation runs (May 2026). Earlier
M3 ablation runs (April 2026) used RunPod credits.

## License & attribution

Research code for a course-project scope. See `literature_review/` for
prior work this builds on.

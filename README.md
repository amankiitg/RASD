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
- **Milestone 3** — ablation study **complete**: 49/49 runs valid in
  `results/ablations/ablations.csv`, 8k context on 8×A100 SXM
- **Milestone 4** — evaluation & analysis at 1M context (in progress)

### Key findings from M3 (see `results/ablations/ablations.csv`)

| axis (level_id) | winner | note |
|---|---|---|
| Target (A5) | Llama-2-7B | 7B beats 13B on both tps and acceptance at fixed draft |
| Draft (A1) | Sheared-LLaMA-1.3B | small edge over TinyLlama-1.1B |
| Spec steps (A2) | k = 4 | k=2 under-speculates; k≥8 falls back to near-greedy |
| KV block (A3) | block = 1024 | acceptance rises (0.06→0.13) with block size; 1024 is the practical sweet spot at 8k |
| Overlap (A4) | any | sync / async-1 / async-2 within 3% on 8×A100 SXM (interconnect fast enough that async has little latency to hide at 8k) |

### Critical engineering fix

**NCCL deadlock at `kv_block_size=2048`** (`src/models/rasd_inference.py:554-568`,
commit [`dc14915`](../../commit/dc14915)): rank 0 was sending its per-round
tick *after* `batch_isend_irecv`. In NCCL eager mode, unbatched `dist.send`
serializes behind the batch — but the batch's `irecv` waits on the peer's
`isend`, and peers were blocked on the tick → circular wait.
Fix: send the tick **before** the P2P batch.

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

- Tested on **RunPod 8×A100 SXM 80GB**. Orphaned-VRAM incidents were the
  primary source of lost compute in M3 — the 5-step clean-kill sequence in
  `configs/ablations.yml` exists specifically to prevent them.
- 4-bit NF4 quantization (via bitsandbytes) on both draft and target lets
  Llama-2-13B + Sheared-LLaMA-1.3B fit comfortably within 80 GB × 8.

## License & attribution

Research code for a course-project scope. See `literature_review/` for
prior work this builds on.

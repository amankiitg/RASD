# M4 Phase C — Pod Session Runbook

Single-page operator checklist for the bundled M4 pod session.
**Estimated runtime: ~10 hr** at $15.92/hr ≈ **$160** on Lambda
8× A100-SXM4-40 GB (`gpu_8x_a100`). Within the $2000 Lambda research
credit; ~$480 total project spend after this session.

After Phase C, only **Phase D** (post-pod paper deliverables, all local)
remains in M4.

## What this session produces

1. `requirements-lock.txt` (committed) — pinned transitive pod env
2. `configs/ablations.yml` (committed) — Llama-2 model revision hashes
3. `results/c11_validation/c11_validation.json` — NF4 codec gate
4. `results/yarn_validation/yarn_validation.json` — YaRN at long ctx
5. `results/c6_validation/c6_validation_rank{0..7}.json` — resume sanity
6. `results/m4_smoke/long_smoke_ctx*.csv` — RASD at 32k / 128k / 512k / 1M
7. `results/baselines/m4_baselines.csv` — Ring + Sliding at 128k / 1M
8. `results/final/final_matrix.csv` — **the 36-row M4 matrix**
9. Per-row per-position trace `.jsonl` sidecars under `results/final/per_token/`
10. (TBD) `results/final/profiler/*.json` — Fig 3 source data

## Hardware + region

- Lambda Cloud: `gpu_8x_a100` SKU (A100-SXM4-40GB), $15.92/hr
- Region: **europe-central-1** (most reliable availability for this SKU
  per the May 2026 R6.5 session). No `rasd-fs` filesystem there — `scp`
  results back to dev machine each session.
- Tag at Phase B start: [`m4-phase-b-complete`](../../tree/m4-phase-b-complete)
  — pod can `git checkout m4-phase-b-complete` to start from the exact
  Phase B exit state.

## Pre-flight (do these BEFORE running the master script)

```bash
# Inside the Lambda 8×A100 instance:
git clone https://github.com/amankiitg/RASD.git && cd RASD
git checkout m4-phase-b-complete   # or main if more recent

conda env create -f environment_gpu.yml
conda activate rasd-gpu

# flash-attn build strips torch from the isolated env — disable isolation:
pip install --no-build-isolation flash-attn>=2.4.0

# Editable install so `from src.models...` resolves:
pip install -e .

# Confirm tests pass on the pod:
pytest tests/

# Inline env vars (NEVER export — operational guide in configs/ablations.yml):
export WANDB_API_KEY=<your-key>
export HF_TOKEN=<your-token>
export HF_HOME=/workspace/hf_cache       # persistent volume on Lambda
```

## The bundled session

```bash
bash scripts/phase_c_pod_session.sh
```

The script runs every stage in sequence. Each writes a marker file
under `results/phase_c/<stage>.done`. Re-runs skip already-completed
stages, so a crashed pod can `git pull` and re-run to pick up.

### Stage sequence

| Stage | What | Stop on FAIL? |
|---|---|---|
| **p30_gpu_health** | nvidia-smi ECC/XID scan; confirm 0 MiB idle on all 8 GPUs | yes — bad hardware aborts the session |
| **p31_repro_lockdown** | `capture_pod_env.sh` → `requirements-lock.txt`; `pin_hf_revisions.py` (Llama-2 hashes); `replay_m3_smoke.sh` (drift ≤15%) | yes — semantic regression aborts |
| **c11_validation** | NF4 codec gate at ctx ∈ {1024, 4096} × seeds {42, 123, 456}. Pass: rel_err ≤ 6%, compression ≥ 3x | yes |
| **c2b_yarn_validation** | YaRN at ctx=512k (factor=128) — no NaN/inf; PPL within 2× of linear baseline at ctx=64k | yes |
| **c6_resume_validation** | Multi-rank checkpoint+resume produces same final tokens as one-shot | yes |
| **p33_long_ctx_smokes** | RASD single-prompt at 32k / 128k / 512k / 1M (1 seed each) | yes — memory ceiling check |
| **p34_baseline_validation** | Ring + Sliding × {128k, 1M} | yes |
| **p35_final_matrix** | 36-run matrix: {RASD, Ring, Sliding} × {128k, 256k, 512k, 1M} × 3 seeds, `--log-per-token` | no — `--resume` lets us pick up after partial failure |
| **p36_profiler_pass** | Profiler sidecar pass on a subset (Fig 3 source data) | no |

### Reviewer's recommended pod-gate order (2026-05-10, third pass)

> "Run the pod in this order and stop hard on failure: C11 integration
> gate → C6 resume gate → 32k smoke → 128k smoke → only then 512k/1M."

The master script's stage sequence already matches this. Three things
make the "stop hard" enforceable in code:

1. The master script aborts on first non-zero exit between stages
   (so `c11_validation` failure halts everything before c6 / smokes).
2. `long_ctx_smokes` passes `--abort-on-failure` to `run_experiment.py`,
   so a 128k OOM stops the script before 512k/1M attempts. Use
   `--resume` on a re-run to pick up after the fix.
3. `final_matrix` does NOT pass `--abort-on-failure` — we WANT to see
   how the matrix behaves at all 12 cells even if one OOMs. Phase D
   reads from a partially-failed CSV correctly.

### Host RAM watch (first 512k checkpoint)

NF4 cache at ctx=512k × W=8 holds ~10 GB / rank in NF4-packed bytes.
At checkpoint save time (`checkpoint_every: 4` in the YAML), each
rank does `to_serializable() + .cpu()` which copies ~10 GB to host
RAM, then `torch.save()` writes it to disk. Total host RAM peak
during a save: 8 ranks × ~10 GB = ~80 GB. At 1M, ~17 GB / rank ×
8 = ~136 GB.

Lambda's 8x A100 SKU has ~1-2 TB of host RAM so this is fine, but
**watch `/proc/meminfo` during the first 512k checkpoint** to
confirm. If host RAM gets tight, drop checkpoint frequency
(`checkpoint_every: 8` instead of `4`) — recovery loses 1-2 more
spec rounds (~5-15 min) but halves checkpoint disk + memory.

### Operational rules

- **NEVER `pkill -9` CUDA processes.** Orphans driver-level VRAM that
  can't be reclaimed without pod restart. Use SIGTERM only (`pkill -15`
  or `kill -TERM`).
- **HF_HOME must be on `/workspace`.** Default `~/.cache/huggingface`
  doesn't survive pod restarts.
- **Subprocess timeout**: at 1M context, individual rows can take
  20+ minutes. The runner's per-row timeout in
  `scripts/r6_verify_runner.sh` defaults to 600s — bump for long runs
  if needed.
- **Watch GPU memory**: `watch -n 5 nvidia-smi`. If any rank exceeds
  ~38 GB consistently at 1M context, kill the run and investigate
  (NF4 KV is supposed to keep us at ~30 GB).

  Note (2026-05-10): the `kv_quant=true` flag in the M4 YAMLs now
  exercises **true NF4 storage** via `NF4DynamicCache` (commits
  `2585822` + `c0c1205`). Earlier doc revisions referred to a
  round-trip-only path that didn't actually save memory — that has
  been removed. Per-rank K/V at 1M × W=8 should land at ~17 GB,
  with total per-rank memory ~30 GB / 40 GB.

## Post-session checklist

When the master script finishes (or you decide to stop early):

```bash
# Sanity-check what completed
ls -la results/phase_c/*.done

# scp results back to dev machine
scp -r ubuntu@<pod-ip>:~/RASD/results/c11_validation/    ./
scp -r ubuntu@<pod-ip>:~/RASD/results/yarn_validation/   ./
scp -r ubuntu@<pod-ip>:~/RASD/results/c6_validation/     ./
scp -r ubuntu@<pod-ip>:~/RASD/results/m4_smoke/          ./
scp -r ubuntu@<pod-ip>:~/RASD/results/baselines/         ./
scp -r ubuntu@<pod-ip>:~/RASD/results/final/             ./
scp -r ubuntu@<pod-ip>:~/RASD/results/phase_c/           ./
scp     ubuntu@<pod-ip>:~/RASD/requirements-lock.txt     ./

# Commit the pod-side artifacts to the repo:
git add requirements-lock.txt configs/ablations.yml results/
git commit -m "M4 Phase C — bundled pod session results"
git tag -a m4-phase-c-complete -m "M4 final matrix landed"
git push origin main m4-phase-c-complete
```

## Then: terminate the pod

Lambda charges per-second. Don't leave it idle.

## Next: Phase D (local, no pod)

Phase D is the **last** M4 phase. Generates Fig 1, Fig 3, Fig 4, Fig 5
from the matrix CSV + sidecars; produces `final_results.json` and LaTeX
tables; drafts manuscript sections. ~3-5 days local.

Fig 2 (heatmap) and `analysis/error_analysis.md` (R6.5 portion) are
already done from Phase A3.

## Phase C blocker fixes (2026-05-10)

External code review surfaced 5 high-risk findings; all 5 are now
fixed in commits before this runbook's referenced m4-phase-b-complete
tag. Summary for the operator:

| # | Issue | Fix commit |
|---|---|---|
| 1 | `kv_quant=True` round-trip only, not real NF4 storage | `2585822` (NF4DynamicCache class) + `c0c1205` (wired into generate) |
| 2 | Double torchrun on orchestrator stages | `b993f67` |
| 3 | `build_run_configs` filtered groups by `A*` prefix only | `b993f67` |
| 4 | Baselines stage used `bash` + `--contexts` | `b993f67` |
| 5 | RNG state never populated → resume divergence | `b993f67` |

If you're rolling back to a tag before any of these commits, the
issues above re-apply. Recommended starting point: tag whatever
includes `c0c1205` or later.

## Risks during this session

- **Llama-2 13B OOM at 64k×W=8 NF4** (predicted in M3). With C11 NF4
  KV implemented in the kernel, 13B should fit at ~22 GB/rank — but
  the 13B kernel-integration testing is pod-side work bundled into
  P3.5. If it OOMs, the M4 matrix uses 7B only (which is the documented
  M3→M4 inheritance choice anyway).
- **YaRN at factor=256** (1M ctx) may have unforeseen numerics issues.
  c2b_yarn_validation tests factor=128 first; if that passes we're
  reasonably confident factor=256 works.
- **NCCL coalesced timeouts** — historical bugbear. Fix2 (e875f6d)
  closed the M3-era root cause; if a new variant appears, fall back
  to sync ring (prefetch_depth=0).
- **Capacity volatility on Lambda** — the 8× SKU has had stretches
  with zero availability. If the pod terminates mid-session, `git pull`
  on a fresh pod and re-run `bash scripts/phase_c_pod_session.sh` —
  marker files let it pick up from the last completed stage.

# M4 Phase C — Pod Session Runbook

Single-page operator checklist for the bundled M4 pod session.
**Estimated runtime: ~10 hr** at $15.92/hr ≈ **$160** on Lambda
8× A100-SXM4-40 GB (`gpu_8x_a100`). Within the $2000 Lambda research
credit; ~$495 total project spend after this session.

**Cost-of-debugging note (2026-05-10):** the first Phase C launch
went through 4 failed bootstrap attempts (~$15 total) before reaching
c11_validation. Root causes were all environmental drift since R6.5
(Anaconda's 2024 TOS rollout, conda+pip torch conflicts, bitsandbytes
0.49.2 requiring torch>=2.4, orphaned `nvidia-*-cu13` packages
shadowing torch's bundled cu12 NCCL). Now stabilised via
`requirements-lock.txt` (captured from a healthy pod, committed at
`600fafe`). Future Phase C re-launches should bootstrap in 10-15 min.

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

**This pre-flight is for manual operator runs.** The
`scripts/auto_execute_phase_c.sh` script runs all of this
automatically when capacity hits. Operator path is documented here
in case you need to bring up a pod manually.

```bash
# Inside the Lambda 8×A100 instance:
git clone https://github.com/amankiitg/RASD.git && cd RASD
git checkout main      # latest, includes 2026-05-10 bootstrap fixes

# Bare conda env (NOT `conda env create -f environment_gpu.yml` —
# that path has known issues with conda+pip torch conflicts that
# took ~$15 of debug time on 2026-05-10. See M4_PLAN.md "Pod-side
# debugging chronicle" for the full story.):
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main || true
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r || true
conda create -n rasd-gpu python=3.10 -y
conda activate rasd-gpu

# Install pinned versions from the captured lock file. This pins
# torch 2.5.1+cu124, bitsandbytes 0.49.2, transformers 4.44.2,
# numpy<2.0 (accelerate compat), and ~130 transitive deps to the
# exact versions known to import + run on Lambda's CUDA driver 12.8.
pip install -r requirements-lock.txt

# flash-attn (must be --no-build-isolation since its setup.py
# imports torch from the env). Lambda's default image does NOT
# include flash-attn — without this step, ring attention silently
# falls back to PyTorch SDPA (this is what happened in R6.5; see
# M3_RING_INTEGRATION_PLAN.md Postscript).
pip install --no-build-isolation "flash-attn>=2.4.0"

# Confirm flash-attn is active in the kernel:
python -c "from src.models.ring_attention_kernel import _FLASH_AVAILABLE; assert _FLASH_AVAILABLE, 'flash-attn not active'"

# Editable install so `from src.models...` resolves:
pip install -e .

# Confirm tests pass on the pod (~2 min):
pip install pytest
pytest tests/

# Inline env vars (NEVER export — operational guide in configs/ablations.yml):
export WANDB_API_KEY=<your-key>
export HF_TOKEN=<your-token>
export HF_HOME=/home/ubuntu/hf_cache     # Lambda's filesystems aren't mounted in
                                          # asia-northeast-2 / europe-central-1 so
                                          # this is per-instance scratch
```

## The bundled session

```bash
bash scripts/phase_c_pod_session.sh
```

The script runs every stage in sequence. Each writes a marker file
under `results/phase_c/<stage>.done`. Re-runs skip already-completed
stages, so a crashed pod can `git pull` and re-run to pick up.

### Stage sequence (current as of 2026-05-10)

| Stage | What | Wandb project | Stop on FAIL? |
|---|---|---|---|
| **p30_gpu_health** | nvidia-smi ECC/XID scan; confirm 0 MiB idle on all 8 GPUs | — | yes — bad hardware aborts |
| **p31_repro_lockdown** | `capture_pod_env.sh` → `requirements-lock.txt`; `pin_hf_revisions.py` (Llama-2 hashes). **`replay_m3_smoke.sh` removed 2026-05-10** — M3 is already pinned via `m3-reproducible` git tag + 13 invariant tests; the replay duplicated work + cost ~25 min/launch + would fail throughput tolerance because M4 has flash-attn but R6.5 didn't | — | yes |
| **c11_validation** | NF4 codec gate at ctx ∈ {1024, 4096} × seeds {42, 123, 456}, plus end-to-end production gate (assert NF4DynamicCache survives prefill + `_truncate_kv`). Pass: rel_err ≤ 15%, compression ≥ 3× | — | yes |
| **c2b_yarn_validation** | YaRN at ctx=512k (factor=128) — no NaN/inf; PPL within 2× of linear baseline at ctx=64k | — | yes |
| **c6_resume_validation** | Multi-rank checkpoint+resume produces same final tokens as one-shot | — | yes |
| **p33_long_ctx_smokes** | RASD single-prompt at 32k / 128k / 512k / 1M (1 seed each), `--abort-on-failure` so 128k OOM doesn't burn pod-$ on doomed 512k/1M | **`rasd-m4-phase-c`** | yes (smoke runner has `--abort-on-failure`) |
| **p34_baseline_validation** | Ring + Sliding × {128k, 256k, 512k, 1M} × 3 seeds. CSV column is `forward_tps` — **NOT directly comparable to RASD's generation `throughput_tps`** (different metric semantics; Phase D Figure 1 caption must clarify) | — (raw CSV) | yes |
| **p35_final_matrix** | 12-run RASD matrix: 4 contexts × 3 seeds (Ring/Sliding baselines come from p34, not this YAML), `--log-per-token` | **`rasd-m4-phase-c`** | no — `--resume` lets us pick up after partial failure |
| **p36_profiler_pass** | Profiler sidecar pass: 4 contexts × seed 42 with `--profile`. Source data for Fig 3 (compute/comm/idle stacked breakdown) | **`rasd-m4-phase-c`** | no |

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

## Bootstrap pitfalls (2026-05-10 chronicle)

The first Phase C launch hit 4 distinct failures before bootstrap
even finished. Documenting so the next session can side-step them.
Each item below is followed by the symptom, root cause, and how the
current `scripts/auto_execute_phase_c.sh` + `requirements-lock.txt`
prevent it.

### 1. Anaconda TOS rollout (~$6.50 burned)
- **Symptom:** `conda env create -f environment_gpu.yml` exits with
  `CondaToSNonInteractiveError: Terms of Service have not been accepted
  for the following channels: pkgs/main, pkgs/r`.
- **Cause:** Anaconda's 2024 TOS enforcement reached the Lambda image
  in early May 2026.
- **Prevention:** the bootstrap now runs
  `conda tos accept --override-channels --channel <each>` before any
  conda command. Plus we no longer use `conda env create` at all
  (see #3); we do `conda create -n rasd-gpu python=3.10` which doesn't
  need the default channels.

### 2. flash-attn build can't see torch
- **Symptom:** `conda env create` proceeds, then fails inside the
  pip section: `ModuleNotFoundError: No module named torch` while
  building flash-attn from source.
- **Cause:** Conda's pip subcall doesn't pass `--no-build-isolation`,
  so flash-attn's `setup.py` runs in an isolated env that doesn't
  see the conda-installed torch.
- **Prevention:** flash-attn was removed from `environment_gpu.yml`'s
  pip section. Bootstrap installs it explicitly via
  `pip install --no-build-isolation flash-attn>=2.4.0` AFTER the
  base env is set up.

### 3. conda+pip torch upgrade chain
- **Symptom:** conda installs torch 2.1.0+cu121 from `pytorch::pytorch=2.1.0`,
  then pip section's transitive deps (deepspeed, accelerate) re-install
  torch as 2.11.0+cu130. Lambda's CUDA driver is 12.8 → cu130 doesn't
  load → `dist.init_process_group` fails with "CUDA driver version is
  insufficient for CUDA runtime version".
- **Cause:** No version pin on torch in the pip section; conda+pip
  resolver mismatch.
- **Prevention:** environment_gpu.yml's conda+pip approach is replaced
  with `requirements-lock.txt`, captured from a healthy pod. All ~130
  versions are pinned including `torch==2.5.1+cu124` (which is new
  enough for bitsandbytes>=0.49 but compatible with driver 12.8).

### 4. NCCL cu13 shadowing (~$3.70 burned)
- **Symptom:** `dist.init_process_group(backend="nccl")` fails with
  `Cuda failure 'CUDA driver version is insufficient'`.
- **Cause:** During the brief torch 2.11+cu130 install (before we
  noticed and rolled back), pip pulled `nvidia-nccl-cu13`,
  `nvidia-cudnn-cu13`, etc. Even after rolling back torch to
  2.5.1+cu124, those cu13 NCCL libs remained on disk and were
  loaded ahead of torch's bundled cu12 NCCL.
- **Prevention:** the captured `requirements-lock.txt` includes only
  cu12 nvidia packages; cu13 leftovers don't reappear. If future
  drift somehow re-introduces them, `pip uninstall nvidia-*-cu13`
  on the pod fixes it (then `pip install --force-reinstall --no-deps
  nvidia-cudnn-cu12==9.1.0.70`).

### 5. replay_m3_smoke false-fail on flash-attn speedup
- **Symptom:** `bash scripts/replay_m3_smoke.sh` reports REGRESSION
  with the new run 2-3× FASTER than R6.5 golden numbers.
- **Cause:** R6.5 ran on Lambda's default image without flash-attn
  (kernel silently fell back to PyTorch SDPA). M4 explicitly installs
  flash-attn, so attention is 2-3× faster at long ctx. The 15%
  throughput tolerance can't accommodate this.
- **Prevention:** `replay_m3_smoke.sh` removed from `phase_c_pod_session.sh`'s
  bootstrap (commit `b9b51f3`). M3 is pinned via the `m3-reproducible`
  git tag + 13 invariant tests in `tests/test_m3_invariants.py` —
  the replay was redundant. Documented in
  [`M3_RING_INTEGRATION_PLAN.md`](M3_RING_INTEGRATION_PLAN.md) Postscript.

### Total cost of these failures: ~$15
This document + the audit table in [`M4_PLAN.md`](M4_PLAN.md) is the
preventative sum. If you see Phase C bootstrap take more than
**~15 minutes**, something has drifted — check `requirements-lock.txt`
against the local copy and look for new cu13 packages or upgrades.

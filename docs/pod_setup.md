# Pod setup (Lambda Cloud, 8xA100-80GB campaign)

Everything here was previously only in `runpod_creds.md`, mixed in with secrets.
That file now holds secrets and nothing else; the operational instructions live
here, and the executable ones live in `scripts/mlsys_pod_env.sh`.

Provenance: `runpod_creds.md` claimed these were "learned from session
2026-04-02" (RunPod) and "learned 2026-05-05, R6.1 session" (Lambda). The RunPod
section describes a provider and an image we no longer use and is kept only so
the divergence is not rediscovered; **Lambda is the current path.**

## Secrets stay in `runpod_creds.md`

| Variable | Who reads it |
|---|---|
| `LAMBDA_API_KEY` | `scripts/mlsys_watch_and_run.sh` (by label), local only |
| `HF_TOKEN` | `scripts/mlsys_watch_and_run.sh` (by label) → forwarded to the pod |
| `WANDB_API_KEY` | nothing automated; export it by hand if you want W&B |
| `RUNPOD_API_KEY` | nothing; RunPod is not used |
| `SSH_PUBLIC_KEY` / `SSH_KEY_PATH` | reference only |

## Instance type and region

- The campaign type is **`gpu_8x_a100_80gb_sxm4`**. 80 GB per GPU.
- `gpu_8x_a100` is the **40 GB** SKU. It is not a substitute: the 1M/256k
  configurations were sized for 80 GB, and swapping down would also invalidate
  the approved cost model. The watcher only ever asks for the 80 GB type.
- There is **no** `gpu_1x_a100_80gb_sxm4` SKU. Single-GPU 80 GB work needs
  `gpu_1x_h100_sxm5` (H100 80 GB) or `gpu_1x_gh200`. For 1x shakedowns,
  `gpu_1x_a100_sxm4` is the 40 GB card and is what the shakedown uses.
- Multi-GPU capacity is chronically zero; 1x capacity flickers in and out within
  minutes. Measured 2026-10-08: an 8x window was visible at 00:49:14Z and gone by
  00:50:38Z (~84s), which is why the watcher polls every 60 ±20s and launches in
  the same iteration that it detects capacity.

## Launch payload

```json
{
  "region_name": "<region with capacity>",
  "instance_type_name": "gpu_8x_a100_80gb_sxm4",
  "ssh_key_names": ["rasd-amank"],
  "name": "rasd-mlsys",
  "quantity": 1
}
```

- SSH user is **`ubuntu`**, not `root` (RunPod used root).
- The SSH key is the same `~/.ssh/id_ed25519` as before, uploaded to Lambda as
  key name `rasd-amank`.
- **`file_system_names` is deliberately NOT sent.** See below.
- A capacity *report* is not capacity: the launch is frequently refused with
  `instance-operations/launch/insufficient-capacity` even after the type is
  listed. Retry rather than treating the list as a reservation.

## The persistent filesystem, and why the campaign does not attach it

`rasd-fs` (id `2294b1c020e64be8a17b1c29fd47b76b`) lives in **`us-west-2`** and
mounts at `/lambda/nfs/rasd-fs/`. It holds `hf_cache/`, `pip_cache/` and
`results/`.

It is **region-locked**, and `gpu_8x_a100_80gb_sxm4` almost never appears in
`us-west-2` (the one window we caught was `us-midwest-1`). Attaching it would
mean never launching. So the campaign runs on **ephemeral instance-local disk**
and accepts re-downloading the models each launch — the trade-off the credentials
file itself records as option (a).

Consequences to expect on every 8x launch:

- model weights download again (Llama-2-7B ≈ 13 GB, Llama-3.1-8B ≈ 16 GB,
  Llama-3.2-1B ≈ 2.5 GB, Sheared-1.3B ≈ 2.6 GB), which is why the first stage
  pays a download.
- nothing on the pod survives termination; results come back through the
  watcher's sha256-verified pull.
- `scripts/mlsys_pod_env.sh` therefore points the HF cache at the filesystem
  **only if it is actually mounted**, and otherwise at a local directory. It
  never points at a path that does not exist, because a nonexistent `HF_HOME`
  is worse than no `HF_HOME`.

Filesystem operations are dashboard-only: `POST /api/v1/file-systems` returns
HTTP 405, and deletion is not exposed by the API either. It costs $0.20/GB/month
whether or not an instance is attached.

## Environment variables every experiment command needs

The credentials file is emphatic about this, having once watched a dashboard show
0 GB used after a smoke that should have downloaded ~13 GB:

```bash
export HF_HOME=/lambda/nfs/rasd-fs/hf_cache
export HF_HUB_CACHE=/lambda/nfs/rasd-fs/hf_cache/hub
export TRANSFORMERS_CACHE=/lambda/nfs/rasd-fs/hf_cache
export PIP_CACHE_DIR=/lambda/nfs/rasd-fs/pip_cache
export PYTHONPATH=/home/ubuntu/RASD
export HF_TOKEN=<secret>
mkdir -p /lambda/nfs/rasd-fs/hf_cache/hub
```

`HF_HOME` alone is documented as having misbehaved; all three HF variables are
set together, and the `hub/` subdirectory is pre-created, because without it HF
may silently fall back to `~/.cache/huggingface/hub/` (instance-local, lost on
termination). `scripts/mlsys_pod_env.sh` writes these to `~/RASD/.pod_env.sh`
and the watcher sources that file for the manifest run.

## The NCCL / allocator settings the campaign runs with

From `scripts/mlsys_pod_session.sh`, which is the validated long-run flow:

```bash
export NCCL_TIMEOUT=3600
export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=3600
export TORCH_NCCL_BLOCKING_WAIT=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
```

## Setup sequence on a fresh instance

Executable form: `scripts/mlsys_pod_env.sh` (idempotent; it is what the watcher
runs). In order:

1. **SSH in** as `ubuntu`.
2. **Verify a clean GPU**: `nvidia-smi --query-gpu=index,memory.used --format=csv,noheader`
   should show 0 MiB on every GPU. Orphaned memory means terminate and relaunch.
3. **conda.** Lambda images *used to* ship miniconda at `~/miniconda3`. The image
   we landed on 2026-10-08 had **no conda anywhere**, which is what made the
   environment a required provisioning step rather than an assumption.
4. **Create env `rasd-gpu`** (python 3.10) and
   `pip install -r requirements-lock.txt`.
5. **flash-attn**, from a prebuilt wheel where one exists (see below).
6. **`pip install -e .`** and, separately, **`diptest`** — it is in
   `environment_gpu.yml` but is **not** in `requirements-lock.txt`, which was
   captured before the dip test existed.
7. **`touch` the `__init__.py` files.** rsync sometimes drops zero-byte files,
   and the resulting `ModuleNotFoundError` looks like a missing package:
   `src/__init__.py`, `src/models/__init__.py`, `src/baselines/__init__.py`.
8. **Verify** by importing what the stages import.

The environment is named **`rasd-gpu`**. Anything that looks for `rasd` is
looking for the local dev machine's env and will silently miss the pod's.

## flash-attn: prebuilt wheel, not a source build

The credentials file says only `pip install --no-build-isolation flash-attn`,
which compiles for tens of minutes on a paid GPU. FlashAttention publishes
prebuilt wheels whose filename encodes **four** things that must all match the
interpreter:

```
flash_attn-2.8.3+cu12torch2.5cxx11abiFALSE-cp310-cp310-linux_x86_64.whl
             ^^^^    ^^^^^^^ ^^^^^^^^^^^^ ^^^^^
             CUDA    torch    C++11 ABI    python
```

With the locked `torch==2.5.1+cu124` on python 3.10, the tag is
`cu12torch2.5`/`cp310`; the ABI component depends on how the installed torch was
built, so `mlsys_pod_env.sh` reads it from the interpreter itself:

```bash
python -c 'import torch; print("TRUE" if torch._C._GLIBCXX_USE_CXX11_ABI else "FALSE")'
```

and falls back to the source build only if no wheel exists at that tag. Both
`cxx11abiFALSE` and `cxx11abiTRUE` cp310 wheels for `v2.8.3` were verified
present (HTTP 200) on 2026-10-08.

The ring kernel falls back to PyTorch SDPA when `flash_attn` is not importable,
so a missing flash-attn degrades rather than crashes — which is exactly why the
import check is explicit instead of relying on a stage to notice.

## Known gotchas

- **PyTorch version vs RNG drift.** The same seed does not produce identical CUDA
  samples across torch versions, so per-prompt acceptance can shift while
  aggregates stay in range. Expect "no regression direction", not byte-identical
  reproduction of pre-2026 numbers.
- **Lambda's `8.0E available` in `df`** is the NFS virtual size, not a quota.
- **bitsandbytes 0.49.2** (resolved from the `>=0.49.0` pin) works against
  Lambda's CUDA 12.x with no special steps.
- **rsync from local** must exclude `.venv*`: a 1.6 GB accident the first time.
  The watcher's exclude list also drops `.git`, `manuscript`, `results/final` and
  `data/processed`'s bulk, and pushes `data/processed/**` separately because the
  metadata names those paths.

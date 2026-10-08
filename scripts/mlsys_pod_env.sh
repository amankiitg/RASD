#!/usr/bin/env bash
# Provision the CAMPAIGN environment on a pod, and prove it works.
#
# Why this exists: the watcher used to *prefer* a conda env and silently fall
# back to `command -v python3` when it was absent. On 2026-10-08 that fallback
# happened -- the image had no conda at all -- and gate_calibration died two
# seconds into a paid 8xA100 run with `No module named 'transformers'`, after
# which the run billed for another two hours because the completion check could
# not fire either. Two lessons are baked in here:
#
#   * the environment is PROVISIONED, not assumed. The recipe is the one from
#     scripts/auto_execute_phase_c.sh, which is the flow that produced the M3
#     numbers; it already handles a missing conda by installing Miniconda.
#   * a missing environment is a FAILURE, loudly, before any stage runs. The
#     last line is a real import check of the packages the stages need, so
#     "pip exited 0" cannot stand in for "the campaign can run".
#
# Idempotent: re-running it on a provisioned pod only re-verifies.
#
# Everything applied here comes from the operator notes that used to live in
# runpod_creds.md; the prose version is docs/pod_setup.md, which is where to look
# for the reasoning (why the persistent filesystem is not attached, why the env
# is called rasd-gpu, why three HF cache variables are set rather than one).
#
# Usage (on the pod, before the manifest):
#     bash scripts/mlsys_pod_env.sh
set -uo pipefail

# NCCL / allocator settings from the validated long-run flow
# (scripts/mlsys_pod_session.sh). Written into the env file so the stages
# inherit them.
NCCL_SETTINGS='NCCL_TIMEOUT=3600
TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=3600
TORCH_NCCL_BLOCKING_WAIT=1
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True'

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/.." && pwd)"
CONDA_DIR="${MLSYS_CONDA_DIR:-$HOME/miniconda3}"
ENV_NAME="${MLSYS_CONDA_ENV:-rasd-gpu}"
FLASH_PIN="${MLSYS_FLASH_ATTN_PIN:-2.8.3}"
LOCK="${MLSYS_LOCK_FILE:-$REPO/requirements-lock.txt}"

echo "=== campaign environment ==="
echo "  repo        : $REPO"
echo "  conda       : $CONDA_DIR"
echo "  env         : $ENV_NAME"
echo "  lock        : $LOCK"
echo "  flash-attn  : $FLASH_PIN"

# --------------------------------------------------------------------------
# 1. conda itself. Lambda images used to ship miniconda; this one does not, so
#    the "preinstalled" assumption is checked rather than believed.
# --------------------------------------------------------------------------
if [ ! -f "$CONDA_DIR/etc/profile.d/conda.sh" ]; then
  echo "--> no conda at $CONDA_DIR; installing Miniconda"
  MC=/tmp/miniconda_installer.sh
  if ! curl -fsSL --max-time 600 \
       https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -o "$MC"; then
    echo "FATAL: could not download the Miniconda installer" >&2
    exit 2
  fi
  if ! bash "$MC" -b -p "$CONDA_DIR" >/tmp/miniconda_install.log 2>&1; then
    echo "FATAL: Miniconda install failed; see /tmp/miniconda_install.log" >&2
    tail -20 /tmp/miniconda_install.log >&2
    exit 2
  fi
  rm -f "$MC"
fi

# shellcheck disable=SC1091
source "$CONDA_DIR/etc/profile.d/conda.sh" || {
  echo "FATAL: conda.sh exists but could not be sourced" >&2; exit 2; }

# --------------------------------------------------------------------------
# 2. the environment. Bare python, then pip: conda's own pip subcall does not
#    pass --no-build-isolation, which is what flash-attn's setup.py needs.
# --------------------------------------------------------------------------
if ! conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
  echo "--> creating conda env $ENV_NAME (python 3.10)"
  # Accept the Anaconda channel TOS. Non-fatal: older condas have no `tos`.
  conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main >/dev/null 2>&1 || true
  conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r >/dev/null 2>&1 || true
  if ! conda create -n "$ENV_NAME" python=3.10 -y >/tmp/conda_create.log 2>&1; then
    echo "FATAL: conda create failed; see /tmp/conda_create.log" >&2
    tail -20 /tmp/conda_create.log >&2
    exit 2
  fi
fi
conda activate "$ENV_NAME" || { echo "FATAL: cannot activate $ENV_NAME" >&2; exit 2; }

PY="$(command -v python)"
echo "  interpreter : $PY ($("$PY" -V 2>&1))"
PYBIN="$PY"
export PYTHONPATH="$REPO"

# --------------------------------------------------------------------------
# 2b. cache locations and the token, written where the MANIFEST can source them
# --------------------------------------------------------------------------
# The operator notes are emphatic that HF_HOME alone misbehaved once (R6.1
# session: the filesystem showed 0 GB used after a smoke that should have pulled
# ~13 GB), so all three HF variables are set together and the `hub/` directory is
# pre-created -- without it HF can silently fall back to instance-local
# ~/.cache/huggingface/hub/, which is lost on termination.
#
# The cache is only placed on the persistent filesystem if that filesystem is
# actually MOUNTED. Pointing HF_HOME at an unmounted /lambda/nfs path is worse
# than having no HF_HOME at all: downloads fail outright instead of landing
# locally.
FS_ROOT="${MLSYS_FS_ROOT:-/lambda/nfs/rasd-fs}"
if [ -d "$FS_ROOT" ]; then
  CACHE_ROOT="$FS_ROOT"
  echo "  persistent filesystem mounted at $FS_ROOT"
else
  CACHE_ROOT="$HOME/.cache/rasd"
  echo "  no persistent filesystem at $FS_ROOT (ephemeral disk this run);"
  echo "  model weights download again and are lost on termination"
fi
export HF_HOME="$CACHE_ROOT/hf_cache"
export HF_HUB_CACHE="$HF_HOME/hub"
export TRANSFORMERS_CACHE="$HF_HOME"
export PIP_CACHE_DIR="$CACHE_ROOT/pip_cache"
mkdir -p "$HF_HUB_CACHE" "$PIP_CACHE_DIR" || true

# The token is forwarded by the watcher and is REQUIRED: every target model in
# this campaign (meta-llama/Llama-2-7b-hf, meta-llama/Llama-3.1-8B) is a gated
# repo. Without it the download 401s and the stage dies hours into a paid run,
# so absence is rejected here and access is proven below.
if [ -z "${HF_TOKEN:-}" ]; then
  echo "FATAL: HF_TOKEN is not set. The campaign's models are gated; without a" >&2
  echo "FATAL: token every arm fails at model load." >&2
  echo "STATUS: env-failed" > "$REPO/.mlsys_env_status"
  exit 2
fi

{
  echo "# Written by scripts/mlsys_pod_env.sh on $(date -u +%FT%TZ). Source this"
  echo "# before any command that loads a model or writes a cache."
  echo "export HF_HOME=$HF_HOME"
  echo "export HF_HUB_CACHE=$HF_HUB_CACHE"
  echo "export TRANSFORMERS_CACHE=$TRANSFORMERS_CACHE"
  echo "export PIP_CACHE_DIR=$PIP_CACHE_DIR"
  echo "export PYTHONPATH=$PYTHONPATH"
  printf '%s\n' "$NCCL_SETTINGS" | sed 's/^/export /'
} > "$REPO/.pod_env.sh"
echo "  wrote $REPO/.pod_env.sh (cache, NCCL and allocator settings)"

# --------------------------------------------------------------------------
# 3. the pins the campaign was measured under.
# --------------------------------------------------------------------------
if [ ! -f "$LOCK" ]; then
  echo "FATAL: lock file '$LOCK' is missing; refusing to install unpinned deps" >&2
  exit 2
fi
echo "--> pip install -r requirements-lock.txt"
if ! pip install -q -r "$LOCK"; then
  echo "FATAL: pip install -r requirements-lock.txt failed" >&2
  exit 2
fi

# flash-attn is deliberately NOT in the lock file: its setup.py imports torch at
# build time and pip builds wheels in an isolated env without runtime deps. The
# operator notes say only `pip install --no-build-isolation flash-attn`, which
# COMPILES: tens of minutes of paid GPU time on every fresh pod.
#
# FlashAttention publishes prebuilt wheels whose filename encodes four things
# that must all agree with the installed interpreter:
#
#   flash_attn-2.8.3+cu12torch2.5cxx11abiFALSE-cp310-cp310-linux_x86_64.whl
#                    ^^^^    ^^^^^^^ ^^^^^^^^^^^^ ^^^^^
#                    CUDA    torch    C++11 ABI    python
#
# The ABI part is not guessable -- it depends on how the installed torch was
# built -- so it is read from the interpreter. The source build remains the
# fallback for a tag with no published wheel.
FLASH_WHEEL_URL=""
install_flash_attn() {
  local abi torchtag pytag cudatag url
  abi=$("$PYBIN" -c 'import torch; print("TRUE" if torch._C._GLIBCXX_USE_CXX11_ABI else "FALSE")' 2>/dev/null)
  torchtag=$("$PYBIN" -c 'import torch; v=torch.__version__.split("+")[0].split("."); print(v[0]+"."+v[1])' 2>/dev/null)
  cudatag=$("$PYBIN" -c 'import torch; print("cu"+torch.version.cuda.split(".")[0])' 2>/dev/null)
  pytag=$("$PYBIN" -c 'import sys; print("cp"+str(sys.version_info[0])+str(sys.version_info[1]))' 2>/dev/null)
  echo "  flash-attn target: $cudatag / torch$torchtag / cxx11abi$abi / $pytag"
  url="https://github.com/Dao-AILab/flash-attention/releases/download/v$FLASH_PIN/flash_attn-$FLASH_PIN+${cudatag}torch${torchtag}cxx11abi${abi}-${pytag}-${pytag}-linux_x86_64.whl"
  FLASH_WHEEL_URL="$url"
  if curl -fsIL --max-time 60 "$url" -o /dev/null 2>/dev/null; then
    echo "  prebuilt wheel found; installing it (no compile)"
    if pip install -q "$url"; then
      return 0
    fi
    echo "  prebuilt wheel failed to install; falling back to a source build"
  else
    echo "  no prebuilt wheel at that tag; falling back to a source build"
  fi
  echo "  source build: compiles, expect tens of minutes"
  FLASH_WHEEL_URL=""
  pip install --no-build-isolation "flash-attn==$FLASH_PIN"
}

if ! "$PYBIN" -c "import flash_attn" >/dev/null 2>&1; then
  if ! install_flash_attn; then
    echo "FATAL: flash-attn $FLASH_PIN could not be installed" >&2
    echo "STATUS: env-failed" > "$REPO/.mlsys_env_status"
    exit 2
  fi
fi

# diptest (the Phase 4 Hartigan dip test) is in environment_gpu.yml but NOT in
# requirements-lock.txt, which was captured before the MLSys work. Installed
# explicitly rather than assumed, because the failure mode is a stage dying
# hours in on a paid box. 0.9.0 is the last release with a cp310 manylinux
# wheel -- the pod's python is 3.10, and a newer pin would build from source.
DIPTEST_PIN="${MLSYS_DIPTEST_PIN:-diptest==0.9.0}"
if ! "$PYBIN" -c "import diptest" >/dev/null 2>&1; then
  echo "--> pip install $DIPTEST_PIN"
  if ! pip install -q "$DIPTEST_PIN"; then
    echo "FATAL: $DIPTEST_PIN failed to install (check for a cp310 wheel)" >&2
    echo "STATUS: env-failed" > "$REPO/.mlsys_env_status"
    exit 2
  fi
fi

echo "--> pip install -e ."
if ! pip install -q -e "$REPO"; then
  echo "FATAL: 'pip install -e .' failed" >&2
  echo "STATUS: env-failed" > "$REPO/.mlsys_env_status"
  exit 2
fi

# rsync sometimes drops ZERO-BYTE files, and the resulting ModuleNotFoundError
# looks like a missing package rather than a missing empty file. The operator
# notes call this out; the fix is cheap and the failure is not.
for f in src/__init__.py src/models/__init__.py src/baselines/__init__.py; do
  if [ ! -f "$REPO/$f" ]; then
    touch "$REPO/$f"
    echo "  recreated missing $f (rsync drops zero-byte files)"
  fi
done

# --------------------------------------------------------------------------
# 4. THE CHECK THAT MATTERS. Every failure above could in principle exit 0 and
#    still leave an environment the stages cannot use, so the last word is an
#    import of exactly what the stages import.
# --------------------------------------------------------------------------
echo "=== verifying the environment can actually run the campaign ==="
if ! "$PY" - <<'PYVERIFY'; then
import importlib, sys
missing, versions = [], {}
for mod, attr in (("torch", "__version__"), ("transformers", "__version__"),
                  ("tokenizers", "__version__"), ("accelerate", "__version__"),
                  ("bitsandbytes", "__version__"), ("datasets", "__version__"),
                  ("numpy", "__version__"), ("scipy", "__version__"),
                  ("jsonlines", None), ("yaml", None), ("diptest", None)):
    try:
        m = importlib.import_module(mod)
        versions[mod] = getattr(m, attr, None) or "ok"
    except Exception as e:                      # noqa: BLE001 - report, don't crash
        missing.append(f"{mod} ({type(e).__name__}: {e})")
for mod in ("flash_attn", "rasd"):
    try:
        importlib.import_module(mod)
        versions[mod] = "ok"
    except Exception as e:                      # noqa: BLE001
        missing.append(f"{mod} ({type(e).__name__}: {e})")

# Every target model in this campaign is a GATED repo. A token that exists but
# lacks access to it fails exactly like no token at all -- except hours later,
# inside a stage. Metadata only: this costs one request and no download.
try:
    from huggingface_hub import HfApi
    api = HfApi()
    for repo in ("meta-llama/Llama-2-7b-hf", "meta-llama/Llama-3.1-8B"):
        try:
            api.model_info(repo)
            versions[f"hf:{repo.split('/')[-1]}"] = "accessible"
        except Exception as e:                  # noqa: BLE001
            missing.append(f"gated access to {repo} ({type(e).__name__}: {e})")
except Exception as e:                          # noqa: BLE001
    missing.append(f"huggingface_hub ({type(e).__name__}: {e})")

for k in sorted(versions):
    print(f"    {k:20s} {versions[k]}")

import torch
print(f"    cuda available   {torch.cuda.is_available()} ({torch.cuda.device_count()} devices)")
if not torch.cuda.is_available():
    missing.append("torch.cuda.is_available() is False")
if torch.cuda.is_available() and torch.cuda.device_count() < 1:
    missing.append("no CUDA devices visible")

if missing:
    print("  MISSING: " + "; ".join(missing))
    sys.exit(1)
print("  environment verified")
PYVERIFY
  echo "FATAL: the campaign environment is unusable -- the stages would fail." >&2
  echo "STATUS: env-failed" > "$REPO/.mlsys_env_status"
  exit 2
fi

# Orphaned GPU memory from a previous container on the same host would corrupt
# every measurement. The operator notes say to terminate and relaunch rather
# than try to clear it.
if command -v nvidia-smi >/dev/null 2>&1; then
  echo "--- GPU state (expect 0 MiB used on every device) ---"
  nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv,noheader | sed 's/^/    /'
  inuse=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null \
          | awk '$1 > 512 {c++} END {print c+0}')
  if [ "${inuse:-0}" -gt 0 ]; then
    echo "FATAL: $inuse device(s) already hold more than 512 MiB." >&2
    echo "FATAL: refusing to measure on a dirty GPU; terminate and relaunch." >&2
    echo "STATUS: env-failed" > "$REPO/.mlsys_env_status"
    exit 2
  fi
fi

echo "STATUS: ok" > "$REPO/.mlsys_env_status"
if [ -n "$FLASH_WHEEL_URL" ]; then
  echo "flash_attn_install: prebuilt-wheel"
else
  echo "flash_attn_install: source-build"
fi
echo "done."

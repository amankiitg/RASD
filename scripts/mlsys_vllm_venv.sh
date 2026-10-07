#!/usr/bin/env bash
# Create the ISOLATED environment the vLLM baseline runs in (R7).
#
# Why a separate environment at all: vLLM 0.6.3 pins its own torch, and the
# campaign's main environment is pinned to a torch/transformers pair that the
# rest of the pipeline depends on (`transformers==4.47.1` for rope_anchor_base,
# and the bitsandbytes NF4 path). Installing vLLM into it would either break
# those pins or silently upgrade them, and every number in the campaign is
# measured through them. So vLLM gets its own interpreter, which the manifest
# invokes BY PATH; the main environment is never touched.
#
# The venv is created from the SAME interpreter that runs the campaign
# (`=== the interpreter the campaign uses ===` below), so the two share a Python
# ABI and the stage's rows can be compared without a version caveat.
#
# Usage (on the pod, before the vLLM stages):
#     bash scripts/mlsys_vllm_venv.sh
#     # then the manifest finds it via MLSYS_VLLM_PYTHON or the default path
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/.." && pwd)"
VENV="${MLSYS_VLLM_VENV:-$REPO/.venv-vllm}"
PY_BASE="${MLSYS_VLLM_BASE_PYTHON:-$(command -v python3)}"
VLLM_PIN="${MLSYS_VLLM_PIN:-0.6.3}"

echo "=== vLLM environment ==="
echo "  venv        : $VENV"
echo "  base python : $PY_BASE"
echo "  vllm pin    : $VLLM_PIN"

if [ ! -x "$PY_BASE" ]; then
  echo "FATAL: base interpreter '$PY_BASE' not found" >&2
  exit 2
fi

if [ ! -x "$VENV/bin/python" ]; then
  echo "--> creating $VENV"
  if ! "$PY_BASE" -m venv "$VENV" 2>"$VENV.venv-err"; then
    # `python3 -m venv` needs the distro's python3-venv/ensurepip on Debian
    # images. Say WHICH dependency is missing rather than reporting "venv
    # creation failed" and leaving the operator to guess on a paid instance.
    echo "FATAL: venv creation failed for $VENV" >&2
    sed 's/^/  | /' "$VENV.venv-err" 2>/dev/null | tail -5 >&2
    echo "  hint: 'python3 -m venv' needs the python3-venv package" >&2
    rm -f "$VENV.venv-err"
    exit 3
  fi
  rm -f "$VENV.venv-err"
fi

"$VENV/bin/python" -m pip install --quiet --upgrade pip

# torch first, so vLLM's resolver does not pick a different build for itself:
# the CUDA wheel is what the A100s need, and letting pip choose is how a CPU
# wheel gets installed and then fails at load time with an unrelated error.
TORCH_PINS="${MLSYS_VLLM_TORCH_PINS:---extra-index-url https://download.pytorch.org/whl/cu121 torch==2.4.0}"
# shellcheck disable=SC2086
"$VENV/bin/python" -m pip install --quiet $TORCH_PINS || {
  echo "FATAL: torch install into the vLLM venv failed" >&2; exit 4; }

"$VENV/bin/python" -m pip install --quiet "vllm==$VLLM_PIN" || {
  echo "FATAL: vllm==$VLLM_PIN install failed" >&2; exit 5; }

# Report what actually landed. The manifest records this in every row and refuses
# to unit-match a row whose version is not the pin, so a resolver that quietly
# installed 0.6.4 must be visible here rather than at analysis time.
"$VENV/bin/python" - <<'PY'
import importlib.metadata as md
import sys

def _v(name):
    try:
        return md.version(name)
    except Exception:
        return "absent"

print(f"  python      : {sys.version.split()[0]}")
for pkg in ("torch", "vllm", "transformers"):
    print(f"  {pkg:<12}: {_v(pkg)}")
PY
echo "done. The campaign's own environment was not modified."

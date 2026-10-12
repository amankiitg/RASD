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

# TRANSFORMERS MUST BE PINNED BELOW 5.x, AND THE FIRST REHEARSAL PROVED IT.
# vLLM 0.6.3 declares `transformers>=4.45.0` with NO upper bound, so pip resolved
# **5.19.0** on 2026-10-12 -- and the repo's own requirements.txt pins 5.5.4 for
# the local analysis env, so a resolver reaching for "latest" lands on exactly the
# major version that breaks this. transformers 5.x cannot see torch 2.4.0 and says
# so ("PyTorch was not found. Models won't be available..."), which then surfaces
# as a vLLM load failure about torch. 4.46.3 satisfies `>=4.45.0`, keeps the
# tokenizer API vLLM 0.6.3 expects, and is inside the range the GPU environment
# already uses (`environment_gpu.yml`: `transformers>=4.46,<4.48`) -- in fact it is
# the exact version the provisioned pod's campaign env reported.
TRANSFORMERS_PIN="${MLSYS_VLLM_TRANSFORMERS_PIN:-4.46.3}"
"$VENV/bin/python" -m pip install --quiet "transformers==$TRANSFORMERS_PIN" || {
  echo "FATAL: transformers==$TRANSFORMERS_PIN install into the vLLM venv failed" >&2; exit 6; }

"$VENV/bin/python" -m pip install --quiet "vllm==$VLLM_PIN" || {
  echo "FATAL: vllm==$VLLM_PIN install failed" >&2; exit 5; }

# Report what actually landed. The manifest records this in every row and refuses
# to unit-match a row whose version is not the pin, so a resolver that quietly
# installed 0.6.4 must be visible here rather than at analysis time.
"$VENV/bin/python" - "$VLLM_PIN" <<'PY'
import importlib.metadata as md
import sys

def _v(name):
    try:
        return md.version(name)
    except Exception:
        return "absent"

pin = sys.argv[1]
versions = {p: _v(p) for p in ("torch", "vllm", "transformers")}
print(f"  python      : {sys.version.split()[0]}")
for pkg, v in versions.items():
    print(f"  {pkg:<12}: {v}")

# ASSERT, DO NOT MERELY REPORT. The first rehearsal printed exactly this table,
# with `transformers: 5.19.0`, and went on to run -- the failure surfaced later as
# a vLLM load error about torch. An install that cannot work must fail HERE, where
# the message names the package.
bad = []
if versions["vllm"] != pin:
    bad.append(f"vllm is {versions['vllm']!r}, pin is {pin!r}")
try:
    if int(versions["transformers"].split(".")[0]) >= 5:
        bad.append(f"transformers {versions['transformers']} is 5.x, which cannot "
                   f"see torch 2.4.0 (pin it below 5)")
except (ValueError, IndexError):
    bad.append(f"transformers version unreadable: {versions['transformers']!r}")
if bad:
    print("FATAL: " + "; ".join(bad), file=sys.stderr)
    sys.exit(7)
PY
report_rc=$?
if [ "$report_rc" != "0" ]; then
  echo "FATAL: the vLLM venv does not satisfy its own pins (rc=$report_rc)" >&2
  exit 7
fi
echo "done. The campaign's own environment was not modified."

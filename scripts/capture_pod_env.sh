#!/usr/bin/env bash
# Capture the exact pip environment from a working pod into requirements-lock.txt
# so future pods can reproduce M3/M4 numbers bit-for-bit.
#
# Run this once on a pod AFTER the ablation grid has completed successfully:
#   bash scripts/capture_pod_env.sh > requirements-lock.txt
#   git add requirements-lock.txt && git commit -m "Lock pod env for M3"
#
# To replay on a new pod:
#   pip install -r requirements-lock.txt
set -euo pipefail

echo "# Captured $(date -u +%Y-%m-%dT%H:%M:%SZ) from pod $(hostname)"
echo "# Python $(python --version 2>&1)"
echo "# CUDA  $(nvcc --version 2>/dev/null | tail -1 || echo 'nvcc not found')"
echo "# Torch $(python -c 'import torch; print(torch.__version__, torch.version.cuda)')"
echo "#"
pip freeze

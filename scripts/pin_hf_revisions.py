#!/usr/bin/env python3
"""Fetch current HF commit hashes for models used in the ablation grid.

Run this on a pod (or any machine with HF_TOKEN set) to capture revisions for
gated models like Llama-2 that require auth. Prints YAML snippets to paste
into `configs/ablations.yml` under `defaults:` and per-level entries.

Usage:
    export HF_TOKEN=hf_xxx
    python scripts/pin_hf_revisions.py
"""
import os
import sys

try:
    from huggingface_hub import HfApi
except ImportError:
    sys.exit("huggingface_hub not installed. `pip install huggingface_hub`.")

MODELS = [
    "meta-llama/Llama-2-7b-hf",
    "meta-llama/Llama-2-13b-hf",
    "princeton-nlp/Sheared-LLaMA-1.3B",
    "TinyLlama/TinyLlama_v1.1",
]

def main():
    api = HfApi(token=os.environ.get("HF_TOKEN"))
    print("# HF revisions (paste into configs/ablations.yml)\n")
    for repo in MODELS:
        try:
            info = api.model_info(repo)
            print(f"# {repo}")
            print(f"#   revision: {info.sha}")
        except Exception as exc:
            print(f"# {repo}  FAILED: {exc}", file=sys.stderr)

if __name__ == "__main__":
    main()

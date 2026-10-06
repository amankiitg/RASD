#!/usr/bin/env python
"""Dump the EXACT prompt token ids RASD used, for the vLLM baseline (C2).

C2 exists because "a prompt of the right length built from the same text" is
not the same prompt. This exporter calls RASD's own ``build_prompt`` with
RASD's own tokenizer and seed, so the ids handed to vLLM are the ids RASD
actually generated with, and the resulting sha256 in vllm_baseline.csv can be
compared against the one in the RASD run's own CSV row.

The prompt length is ``window - max_new_tokens`` because RASD sizes the prompt
so that prompt + generation exactly fills the context window:

  Llama-3.1-8B @131072, 128 new tokens -> 130944 prompt tokens  (ARM4-f1)
  Llama-2-7b-hf @131072,  64 new tokens -> 131008 prompt tokens  (Arm1)

The two targets need DIFFERENT prompt lengths, which is why this writes a map
rather than one list. Keying by ``<model>@<ctx>`` lets the wrapper pick the
right one per cell and refuse to guess if a key is missing.

Usage:
    python scripts/mlsys_dump_prompt_ids.py --out /tmp/rasd_prompt_ids.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

# (model, context window, max_new_tokens) — max_new must equal the matched
# RASD cell's, otherwise prompt+output would not fill the window the same way.
TARGETS = [
    ("meta-llama/Llama-3.1-8B", 131072, 128),
    ("meta-llama/Llama-2-7b-hf", 131072, 64),
]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="/tmp/rasd_prompt_ids.json")
    ap.add_argument("--seed", type=int, default=42,
                    help="seed-42 is the cell both targets have in common; "
                         "ARM4-f1 seed 42 is the Arm2 replication")
    args = ap.parse_args()

    from transformers import AutoTokenizer
    from run_experiment import build_prompt

    out: dict[str, list[int]] = {}
    for model, window, max_new in TARGETS:
        prompt_len = window - max_new
        tok = AutoTokenizer.from_pretrained(model)
        text = build_prompt(prompt_len, tok, source="synthetic", seed=args.seed)
        ids = tok(text, add_special_tokens=False)["input_ids"]
        out[f"{model}@{window}"] = list(ids)
        print(f"{model}: prompt={len(ids)} tokens "
              f"(target {prompt_len}), max_new={max_new}")

    dest = Path(args.out)
    dest.write_text(json.dumps(out))
    print(f"[write] {dest}  ({len(out)} cells)")
    return 0


if __name__ == "__main__":
    sys.exit(main())

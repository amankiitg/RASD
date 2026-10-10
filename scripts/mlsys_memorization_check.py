#!/usr/bin/env python3
"""Continuation perplexity of one model on several pools, through the GATE's code.

WHY THIS EXISTS. On 2026-10-10 the coherence gate reported ppl ~1.03 for
Llama-3.1-8B on PG-19 at 32k -- about 97% per-token confidence. A read-only audit
excluded copying (0 of 975 continuation 50-grams appear in any prompt, at every
rung), degenerate text (all 1009 16-grams distinct, top token " the" at 7.2%) and
scoring misalignment (hidden[q-1] -> ids[q], n=1024). The remaining explanation
is that the scored books are RECALLED rather than modelled: the pool is
`emozilla/pg19` with `"split": "train"`, i.e. public-domain Gutenberg text that
is very likely in Llama-3.1's pretraining mixture.

So this measures the same quantity, with the same functions, on three pools:
  * pg19_train  -- data/processed/pg19_docs (the gate's own pool)
  * pg19_valid  -- data/processed/pg19_llama3 (held-out split, Llama-3 tokenized)
  * postcutoff  -- 2025 US Federal Register final rules (US Government work,
                   public domain, published well after the training cut-off)

SAME CODE PATH, deliberately: `mlsys_coherence_gate.load_pg19_window` builds the
window and `mlsys_coherence_gate.continuation_perplexity` scores it, so the only
thing that differs between rows is the data. The model is built the way the gate
builds it (RASDInference._build_hf_config with the native rope, bf16, single
device), so the comparison is against the gate's own reported numbers.

TOKENIZER HAZARD, checked before running and worth stating: data/processed/pg19
is NOT Llama-3.1 tokenized (its ids decode to garbage and peak at 29968), so it
is unusable for this model -- and the gate would not have complained, because
every Llama-2-range id is inside Llama-3.1's 128k vocabulary. The
Llama-3-compatible validation split is data/processed/pg19_llama3. Using the
wrong directory here would have produced a huge ppl and a false "memorized"
verdict.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def load_gate(repo: Path):
    """Import the gate as a module so its OWN window/scoring code is used."""
    spec = importlib.util.spec_from_file_location(
        "_gate_for_memcheck", repo / "scripts" / "mlsys_coherence_gate.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default="/home/ubuntu/RASD")
    ap.add_argument("--model", default="meta-llama/Llama-3.1-8B")
    ap.add_argument("--revision", default="d04e592bb4f6aa9cfee91e2e20afa771667e1d4b")
    ap.add_argument("--context-length", type=int, default=32768)
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 123, 456])
    ap.add_argument("--out", default="/home/ubuntu/memcheck.json")
    args = ap.parse_args()

    repo = Path(args.repo)
    sys.path.insert(0, str(repo))
    gate = load_gate(repo)
    from src.models.rasd_inference import RASDInference

    pools = {
        # label: (metadata, note)
        "pg19_train": (repo / "data/processed/pg19_docs/documents.json",
                       "gate pool, emozilla/pg19 split=train"),
        "pg19_valid": (repo / "data/processed/pg19_llama3/pg19_validation_metadata.json",
                       "held-out split, Llama-3.1 tokenized"),
        "postcutoff": (repo / "data/processed/pg19_postcutoff/pg19_validation_metadata.json",
                       "US Federal Register 2025 final rules, public domain"),
    }

    tok = AutoTokenizer.from_pretrained(args.model, revision=args.revision)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    # Built the way run_candidate builds it: native rope for this model, bf16,
    # single device -- so the numbers are comparable to the gate's own.
    hf_cfg = RASDInference._build_hf_config(
        None, args.model, args.revision, args.context_length, label="memcheck",
        apply_rope_scaling=False, rope_type="linear", rope_factor=None,
        rope_anchor_base=None)
    t0 = time.time()
    model = AutoModelForCausalLM.from_pretrained(
        args.model, config=hf_cfg, revision=args.revision,
        torch_dtype=(torch.bfloat16 if torch.cuda.is_available() else torch.float32),
        device_map={"": 0} if torch.cuda.is_available() else {"": "cpu"},
    ).eval()
    print(f"model loaded in {time.time() - t0:.0f}s "
          f"(cuda={torch.cuda.is_available()})", flush=True)

    rows = []
    for label, (meta, note) in pools.items():
        if not Path(meta).exists():
            print(f"[SKIP] {label}: {meta} not present", flush=True)
            continue
        for seed in args.seeds:
            try:
                t1 = time.time()
                prompt, cont = gate.load_pg19_window(
                    str(meta), args.context_length, seed,
                    bos_id=getattr(tok, "bos_token_id", None), tokenizer=tok)
                ppl, bad = gate.continuation_perplexity(model, prompt, cont)
                row = {"pool": label, "seed": seed, "note": note,
                       "prompt_tokens": len(prompt), "continuation_tokens": len(cont),
                       "ppl": round(float(ppl), 4), "non_finite": bad is not None,
                       "seconds": round(time.time() - t1, 1)}
                print(f"[{label}] seed={seed} prompt={len(prompt)} "
                      f"cont={len(cont)} PPL={ppl:.4f} ({row['seconds']}s)",
                      flush=True)
            except Exception as e:                       # noqa: BLE001
                row = {"pool": label, "seed": seed, "note": note,
                       "error": f"{type(e).__name__}: {e}"}
                print(f"[{label}] seed={seed} FAILED: {row['error']}", flush=True)
            rows.append(row)

    Path(args.out).write_text(json.dumps(rows, indent=1))
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Check 4 of the M3 post-analysis: fp16-vs-NF4 quantization mini-ablation.

Runs a reduced-scope 4-cell × 2-seed study on a single A100 to
characterize how much α the bitsandbytes NF4 quantization costs with
the (now-patched) speculative-verify code. This is *not* a full M3
rerun.

Cells (two binary knobs: target dtype and draft dtype)

    cell id  | target quant | draft quant
    ---------+--------------+------------
    fp_fp    | bfloat16     | bfloat16
    fp_nf    | bfloat16     | NF4
    nf_fp    | NF4          | bfloat16
    nf_nf    | NF4          | NF4  (matches M3 A2_k4 regime)

Hardware: 1× A100 80GB. No ring attention — single GPU, no torchrun.
Context length 2048, spec_steps=4, max_new_tokens=256, temperature=1.0.

Usage
-----
    # single cell, single seed (what the pod driver actually calls)
    python scripts/run_quant_ablation.py \\
        --target-quant nf4 --draft-quant nf4 --seed 42

    # full sweep (all 4 cells × 2 seeds)
    python scripts/run_quant_ablation.py --sweep

Output
------
One JSON per (cell, seed) at
`results/quant_mini/cell_{target}_{draft}_s{seed}.json` with per-prompt
metrics + aggregate. `--sweep` additionally writes a summary CSV at
`results/quant_mini/summary.csv`.

Budget: ~5 min/run × 8 runs = ~40 min wall clock on one A100.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from dataclasses import asdict
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))


# ---- Auto-load HF_TOKEN from runpod_creds.md if present (same as Check 1) ----
import re
if "HF_TOKEN" not in os.environ:
    creds = REPO / "runpod_creds.md"
    if creds.exists():
        m = re.search(r"HF_TOKEN\s*=\s*(\S+)", creds.read_text())
        if m:
            os.environ["HF_TOKEN"] = m.group(1)
            os.environ["HUGGINGFACE_HUB_TOKEN"] = m.group(1)


# ---------------------------------------------------------------------------
# Benchmark prompts — 10 diverse long-ish prompts, same style as run_experiment
# ---------------------------------------------------------------------------

PROMPT_SEEDS = [
    "The following is a detailed technical analysis of distributed "
    "machine learning systems. Ring attention enables long-context "
    "inference by sharding the sequence across GPUs. Speculative "
    "decoding accelerates generation by using a smaller draft model.",

    "The theory of plate tectonics explains the gradual movement of "
    "the Earth's lithospheric plates over geological time. Major "
    "boundaries include divergent, convergent, and transform faults.",

    "In the field of macroeconomics, the output gap measures the "
    "difference between actual and potential GDP. A positive gap "
    "indicates overheating, while a negative gap suggests slack.",

    "A binary search tree maintains the invariant that each node's "
    "left subtree contains only keys less than the node's key, and "
    "the right subtree contains only keys greater.",

    "Climate feedback mechanisms amplify or dampen the initial "
    "forcing. The ice-albedo feedback is positive because retreating "
    "ice exposes darker ocean, increasing heat absorption.",

    "Transformer language models use self-attention to compute "
    "representations of input sequences. Each token attends to every "
    "other token via scaled dot-product attention.",

    "The Krebs cycle is a central pathway of aerobic respiration in "
    "the mitochondrial matrix. Acetyl-CoA enters the cycle and is "
    "oxidised to carbon dioxide, producing ATP, NADH, and FADH2.",

    "Modern cryptography is built on computational hardness "
    "assumptions such as integer factorization and discrete "
    "logarithms. Public-key schemes enable secure communication "
    "without a shared secret.",

    "Renaissance humanism emphasized the study of classical texts and "
    "the dignity of the individual. Petrarch, Erasmus, and More "
    "shaped its intellectual trajectory across Italy and northern Europe.",

    "Relativity fundamentally reshaped 20th-century physics. Special "
    "relativity showed that simultaneity is observer-dependent; "
    "general relativity reinterpreted gravity as spacetime curvature.",
]


def build_prompt(seed_text: str, context_length: int, tokenizer) -> str:
    """Pad a seed snippet to ~context_length tokens by repetition."""
    tokens = tokenizer.encode(seed_text)
    repeats = max(1, context_length // max(1, len(tokens)))
    full_text = (seed_text + " ") * repeats
    full_tokens = tokenizer.encode(full_text)[:context_length]
    return tokenizer.decode(full_tokens)


# ---------------------------------------------------------------------------
# Cell runner
# ---------------------------------------------------------------------------

CELL_ORDER = ["fp_fp", "fp_nf", "nf_fp", "nf_nf"]

def _cell_id(target_quant: str, draft_quant: str) -> str:
    return f"{'fp' if target_quant == 'fp16' else 'nf'}_" \
           f"{'fp' if draft_quant == 'fp16' else 'nf'}"


def run_cell(target_quant: str,
             draft_quant:  str,
             seed: int,
             n_prompts: int = 10,
             context_length: int = 2048,
             spec_steps: int = 4,
             max_new_tokens: int = 256,
             temperature: float = 1.0,
             top_p: float = 1.0,
             out_dir: Path = REPO / "results" / "quant_mini") -> dict:
    """Run one cell: n_prompts × one seed. Returns aggregate dict and
    writes per-cell JSON."""
    import torch
    from src.models.rasd_inference import RASDConfig, RASDInference

    torch.cuda.set_device(0)

    cfg = RASDConfig(
        target_model_name = "meta-llama/Llama-2-7b-hf",
        draft_model_name  = "princeton-nlp/Sheared-LLaMA-1.3B",
        spec_steps        = spec_steps,
        kv_block_size     = 512,                # unused at world_size=1
        prefetch_depth    = 0,                  # unused at world_size=1
        max_new_tokens    = max_new_tokens,
        dtype             = "bfloat16",         # same as M3 for clean comparison
        quantize_target   = (target_quant == "nf4"),
        quantize_draft    = (draft_quant == "nf4"),
        temperature       = temperature,
        top_p             = top_p,
        seed              = seed,
        debug             = False,
    )

    cell_id = _cell_id(target_quant, draft_quant)
    print(f"\n{'='*72}\nCell {cell_id}  seed={seed}\n"
          f"  target={target_quant}  draft={draft_quant}\n"
          f"  ctx={context_length}  k={spec_steps}  new_tokens={max_new_tokens}\n"
          f"{'='*72}", flush=True)

    t_load = time.perf_counter()
    engine = RASDInference(cfg)
    load_sec = time.perf_counter() - t_load
    print(f"  [load] {load_sec:.1f}s", flush=True)

    per_prompt = []
    for i, seed_text in enumerate(PROMPT_SEEDS[:n_prompts]):
        prompt = build_prompt(seed_text, context_length, engine.tokenizer)
        t0 = time.perf_counter()
        _, metrics = engine.generate_text(prompt)
        wall = time.perf_counter() - t0
        entry = {
            "prompt_idx":       i,
            "tokens_generated": metrics["tokens_generated"],
            "wall_sec":         round(wall, 3),
            "throughput_tps":   round(metrics["throughput_tps"], 2),
            "acceptance_rate":  round(metrics["acceptance_rate"], 4),
            "mean_latency_ms":  round(metrics["mean_latency_ms"], 3),
            "gpu_peak_mem_mb":  round(metrics["gpu_peak_mem_mb"], 1),
            "n_rounds":         metrics["n_rounds"],
        }
        per_prompt.append(entry)
        print(f"  [prompt {i}] tok={entry['tokens_generated']:>3}  "
              f"tps={entry['throughput_tps']:>6.2f}  "
              f"α={entry['acceptance_rate']:.3f}  "
              f"rounds={entry['n_rounds']}", flush=True)

    # Aggregate (mean across prompts with tokens_generated >= 20, matching M3)
    valid = [p for p in per_prompt if p["tokens_generated"] >= 20]
    def _mean(key):
        return (sum(p[key] for p in valid) / len(valid)) if valid else float("nan")
    agg = {
        "cell_id":         cell_id,
        "target_quant":    target_quant,
        "draft_quant":     draft_quant,
        "seed":            seed,
        "n_prompts":       n_prompts,
        "n_valid":         len(valid),
        "alpha_mean":      round(_mean("acceptance_rate"), 4),
        "throughput_mean": round(_mean("throughput_tps"),  2),
        "latency_mean_ms": round(_mean("mean_latency_ms"), 3),
        "peak_mem_mb":     round(max((p["gpu_peak_mem_mb"] for p in per_prompt), default=0.0), 1),
        "load_sec":        round(load_sec, 1),
        "per_prompt":      per_prompt,
        "config":          asdict(cfg),
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"cell_{cell_id}_s{seed}.json"
    out_path.write_text(json.dumps(agg, indent=2, default=str))
    print(f"  [write] {out_path}", flush=True)
    print(f"  [summary] α={agg['alpha_mean']:.3f}  "
          f"tps={agg['throughput_mean']:.2f}  "
          f"n_valid={agg['n_valid']}/{n_prompts}", flush=True)
    return agg


# ---------------------------------------------------------------------------
# Sweep driver
# ---------------------------------------------------------------------------

def sweep(seeds=(42, 123), **kw) -> list[dict]:
    """Run all 4 cells × len(seeds) seeds. Writes summary CSV."""
    results = []
    for cell in CELL_ORDER:
        target_quant = "fp16" if cell.startswith("fp") else "nf4"
        draft_quant  = "fp16" if cell.endswith("fp")   else "nf4"
        for s in seeds:
            agg = run_cell(target_quant=target_quant,
                           draft_quant=draft_quant,
                           seed=s, **kw)
            results.append(agg)

    out_dir = kw.get("out_dir", REPO / "results" / "quant_mini")
    csv_path = out_dir / "summary.csv"
    with csv_path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["cell_id", "target_quant", "draft_quant", "seed",
                    "n_valid", "alpha_mean", "throughput_mean",
                    "latency_mean_ms", "peak_mem_mb", "load_sec"])
        for r in results:
            w.writerow([r[k] for k in ["cell_id", "target_quant", "draft_quant",
                                       "seed", "n_valid", "alpha_mean",
                                       "throughput_mean", "latency_mean_ms",
                                       "peak_mem_mb", "load_sec"]])
    print(f"\n[sweep done] summary → {csv_path}")
    return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--target-quant", choices=["fp16", "nf4"], default="nf4")
    p.add_argument("--draft-quant",  choices=["fp16", "nf4"], default="nf4")
    p.add_argument("--seed",         type=int, default=42)
    p.add_argument("--n-prompts",    type=int, default=10)
    p.add_argument("--context-length", type=int, default=2048)
    p.add_argument("--spec-steps",   type=int, default=4)
    p.add_argument("--max-new-tokens", type=int, default=256)
    p.add_argument("--temperature",  type=float, default=1.0)
    p.add_argument("--top-p",        type=float, default=1.0)
    p.add_argument("--sweep",        action="store_true",
                   help="Run all 4 cells × 2 seeds")
    args = p.parse_args(argv)

    common = dict(
        n_prompts=args.n_prompts,
        context_length=args.context_length,
        spec_steps=args.spec_steps,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
    )

    if args.sweep:
        sweep(**common)
    else:
        run_cell(args.target_quant, args.draft_quant, args.seed, **common)
    return 0


if __name__ == "__main__":
    sys.exit(main())

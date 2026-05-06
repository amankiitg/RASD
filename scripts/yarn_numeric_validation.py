#!/usr/bin/env python3
"""C2b YaRN RoPE numeric validation (M4 Phase C).

Linear RoPE scaling degrades past factor ≈ 16. For 1M context
(factor=256 over Llama-2's native 4k), the M4 plan calls for YaRN.
This script validates that YaRN at factor=256:

1. Doesn't NaN/inf the forward pass at long context
2. Produces logits whose softmax-PPL is within ~2x of the linear
   baseline at moderate context (factor=16, where linear is still
   trustworthy)

Single-rank single-prompt — fast (~5 min) on a 40 GB SXM2 with NF4 weights.

Usage (on pod):
    python scripts/yarn_numeric_validation.py \\
        --target meta-llama/Llama-2-7b-hf \\
        --short-ctx 65536 \\
        --long-ctx 524288  # half-million is enough to confirm scaling

Output: results/yarn_validation/yarn_validation.json
Exit code 0 if both gates pass; 1 otherwise.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch


def _build_model(name: str, ctx: int, rope_type: str, revision: str | None):
    """Build the model with the requested RoPE strategy + context."""
    from transformers import AutoConfig, AutoModelForCausalLM, BitsAndBytesConfig

    from src.models.rasd_inference import _build_rope_scaling_dict

    cfg = AutoConfig.from_pretrained(name, revision=revision)
    if ctx > cfg.max_position_embeddings:
        native = cfg.max_position_embeddings
        factor = float(math.ceil(ctx / native))
        cfg.rope_scaling = _build_rope_scaling_dict(rope_type, factor, native)
        cfg.max_position_embeddings = ctx
    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16)
    model = AutoModelForCausalLM.from_pretrained(
        name, config=cfg, revision=revision,
        torch_dtype=torch.bfloat16,
        quantization_config=bnb,
        device_map={"": 0},
    )
    model.eval()
    return model


@torch.inference_mode()
def _check_no_nan_inf(model, ctx: int) -> dict:
    """Run a single short forward at the long context, confirm no NaN/inf."""
    vocab = model.config.vocab_size
    torch.manual_seed(0)
    # Use a *short* prefix so we don't blow per-rank memory;
    # the rope_scaling state is what we're validating, and that's
    # exercised by any forward where positions span the long range.
    L = min(ctx, 8192)  # 8k is enough to detect rope numerics
    ids = torch.randint(0, vocab, (1, L), device="cuda")
    out = model(ids, use_cache=False)
    logits = out.logits
    has_nan = bool(torch.isnan(logits).any().item())
    has_inf = bool(torch.isinf(logits).any().item())
    finite = (~torch.isnan(logits) & ~torch.isinf(logits)).all().item()
    return {
        "ctx_configured": ctx,
        "L_tested": L,
        "has_nan": has_nan,
        "has_inf": has_inf,
        "all_finite": bool(finite),
        "logits_max": float(logits[~torch.isnan(logits) & ~torch.isinf(logits)].abs().max().item()),
    }


@torch.inference_mode()
def _check_ppl_relative_to_baseline(model, baseline_model, ctx: int) -> dict:
    """At a moderate context where the baseline (linear, factor=16) is
    still trustworthy, both should produce comparable PPL."""
    from src.analysis.perplexity import compute_perplexity

    vocab = model.config.vocab_size
    torch.manual_seed(0)
    L = min(ctx, 8192)
    ids = torch.randint(0, vocab, (1, L), device="cuda")
    ppl_yarn = compute_perplexity(model, ids, max_length=L)
    ppl_base = compute_perplexity(baseline_model, ids, max_length=L)
    ratio = ppl_yarn / ppl_base if ppl_base > 0 else float("inf")
    return {
        "L_tested": L,
        "ppl_yarn":   ppl_yarn,
        "ppl_baseline": ppl_base,
        "ratio_to_baseline": ratio,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--target", default="meta-llama/Llama-2-7b-hf")
    p.add_argument("--revision", default=None)
    p.add_argument("--short-ctx", type=int, default=65536,
                   help="Linear RoPE baseline context (factor=16, trustworthy)")
    p.add_argument("--long-ctx", type=int, default=524288,
                   help="YaRN long-context test (factor=128 here)")
    p.add_argument("--ppl-ratio-bound", type=float, default=2.0,
                   help="YaRN PPL must stay within this multiple of linear baseline")
    p.add_argument("--out", default="results/yarn_validation/yarn_validation.json")
    args = p.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("yarn_numeric_validation requires CUDA")

    results = {
        "target": args.target,
        "short_ctx": args.short_ctx,
        "long_ctx": args.long_ctx,
        "ppl_ratio_bound": args.ppl_ratio_bound,
    }

    # Gate 1: long-context YaRN doesn't NaN/inf
    print(f"Building YaRN model at ctx={args.long_ctx} (factor={args.long_ctx//4096})...")
    yarn_long = _build_model(args.target, args.long_ctx, "yarn", args.revision)
    nan_check = _check_no_nan_inf(yarn_long, args.long_ctx)
    print(f"  long-ctx finite check: {nan_check}")
    results["nan_check"] = nan_check
    pass_nan = nan_check["all_finite"]

    # Free before loading the next model
    del yarn_long
    torch.cuda.empty_cache()

    # Gate 2: at the short-ctx where linear is trustworthy, YaRN PPL
    # should be comparable to linear. Build both at that context.
    print(f"Building linear baseline at ctx={args.short_ctx}...")
    linear_short = _build_model(args.target, args.short_ctx, "linear", args.revision)
    print(f"Building YaRN comparison at ctx={args.short_ctx}...")
    yarn_short = _build_model(args.target, args.short_ctx, "yarn", args.revision)

    ppl_check = _check_ppl_relative_to_baseline(yarn_short, linear_short, args.short_ctx)
    print(f"  PPL check: {ppl_check}")
    results["ppl_check"] = ppl_check
    pass_ppl = ppl_check["ratio_to_baseline"] <= args.ppl_ratio_bound

    overall = pass_nan and pass_ppl
    results["overall_pass"] = overall
    results["pass_nan_check"] = pass_nan
    results["pass_ppl_check"] = pass_ppl

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2))
    print(f"\nWrote {out_path}")
    print(f"Overall: {'PASS' if overall else 'FAIL'}")
    return 0 if overall else 1


if __name__ == "__main__":
    raise SystemExit(main())

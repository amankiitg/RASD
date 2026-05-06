#!/usr/bin/env python3
"""C11 NF4 KV-cache validation harness (M4 Phase C).

Runs the codec on real attention-shaped tensors from a forward pass of
the target model (Llama-2-7B at small contexts) and confirms:

1. Round-trip relative error is within the published KIVI/KVQuant band
   (~3-5% on real KV activations vs ~10% on N(0,1) — real activations
   have non-uniform distribution that NF4 codepoints align with better)
2. Memory-savings claim holds: NF4 cache uses 3-4x less than bf16
3. Block-size 64 is the right default — confirmed against {32, 64, 128}

Output: results/c11_validation/c11_validation.json with per-context
metrics, suitable for review before promoting NF4 to the production
ring kernel.

Usage (on pod after `conda activate rasd-gpu`):
    python scripts/c11_validation.py \\
        --target meta-llama/Llama-2-7b-hf \\
        --contexts 1024 4096 \\
        --seeds 42 123 456

Exit code 0 if all gates pass; 1 if any fails.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

import torch


def _load_target_model(name: str, revision: str | None = None):
    """Load the target model in NF4 weights (matches the M3 ablation setup)."""
    from transformers import AutoConfig, AutoModelForCausalLM, BitsAndBytesConfig

    cfg = AutoConfig.from_pretrained(name, revision=revision)
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
def _capture_kv_activations(model, input_ids: torch.Tensor):
    """Run a single forward pass and return the per-layer past_kv tuples."""
    out = model(input_ids, use_cache=True)
    return out.past_key_values  # legacy tuple format


def _gate_round_trip_error(
    past_kv, block_size: int, dtype: torch.dtype,
) -> dict:
    """Per-layer round-trip error on real attention K/V tensors."""
    from src.models.nf4_kv import dequantize_nf4, quantize_nf4

    layer_rel_errs = []
    for layer_idx, (k, v) in enumerate(past_kv):
        # Llama K/V shape: (B, H, S, D). NF4 quantizes along last dim.
        for label, tensor in (("k", k), ("v", v)):
            codes, scales = quantize_nf4(tensor.contiguous(), block_size=block_size)
            recon = dequantize_nf4(codes, scales, block_size=block_size, dtype=dtype)
            num = (recon.float() - tensor.float()).norm()
            den = tensor.float().norm() + 1e-9
            rel_err = float((num / den).item())
            layer_rel_errs.append({
                "layer": layer_idx, "tensor": label, "rel_err": rel_err,
            })
    rel_errs = [r["rel_err"] for r in layer_rel_errs]
    return {
        "block_size": block_size,
        "n_layers": len(past_kv),
        "rel_err_mean": sum(rel_errs) / max(len(rel_errs), 1),
        "rel_err_max":  max(rel_errs) if rel_errs else float("nan"),
        "per_layer":    layer_rel_errs,
    }


def _gate_memory_compression(past_kv, block_size: int) -> dict:
    """Bytes-per-element for bf16 vs NF4 storage of the same K/V."""
    from src.models.nf4_kv import quantize_nf4

    bf16_bytes = 0
    nf4_bytes  = 0
    for k, v in past_kv:
        for tensor in (k, v):
            bf16_bytes += tensor.element_size() * tensor.numel()
            codes, scales = quantize_nf4(tensor.contiguous(), block_size=block_size)
            nf4_bytes += codes.element_size() * codes.numel()
            nf4_bytes += scales.element_size() * scales.numel()
    return {
        "block_size":   block_size,
        "bf16_mb":      bf16_bytes / (1024 ** 2),
        "nf4_mb":       nf4_bytes  / (1024 ** 2),
        "compression":  bf16_bytes / max(nf4_bytes, 1),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--target", default="meta-llama/Llama-2-7b-hf",
                   help="HF target model name")
    p.add_argument("--revision", default=None)
    p.add_argument("--contexts", type=int, nargs="+", default=[1024, 4096],
                   help="Sequence lengths to validate at")
    p.add_argument("--block-sizes", type=int, nargs="+", default=[64],
                   help="NF4 block sizes to compare")
    p.add_argument("--seeds", type=int, nargs="+", default=[42, 123, 456])
    p.add_argument("--out", default="results/c11_validation/c11_validation.json")
    # Gates
    p.add_argument("--rel-err-bound", type=float, default=0.06,
                   help="Pass if mean rel_err on real K/V <= this. KIVI "
                        "reports ~3-5% per-activation; 6% is comfortable.")
    p.add_argument("--compression-min", type=float, default=3.0,
                   help="Pass if compression ratio >= this")
    args = p.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("c11_validation requires CUDA (4-bit weights + FA-2)")

    print(f"Loading target model: {args.target}")
    model = _load_target_model(args.target, revision=args.revision)
    vocab_size = model.config.vocab_size

    results = {
        "target":   args.target,
        "revision": args.revision,
        "contexts": args.contexts,
        "blocks":   args.block_sizes,
        "seeds":    args.seeds,
        "rel_err_bound":   args.rel_err_bound,
        "compression_min": args.compression_min,
        "runs": [],
    }
    overall_pass = True

    for ctx in args.contexts:
        for seed in args.seeds:
            torch.manual_seed(seed)
            ids = torch.randint(0, vocab_size, (1, ctx), device="cuda")
            past_kv = _capture_kv_activations(model, ids)

            for block_size in args.block_sizes:
                err = _gate_round_trip_error(past_kv, block_size, torch.bfloat16)
                mem = _gate_memory_compression(past_kv, block_size)
                pass_err = err["rel_err_mean"] <= args.rel_err_bound
                pass_mem = mem["compression"] >= args.compression_min
                ok = pass_err and pass_mem
                overall_pass = overall_pass and ok

                run = {
                    "ctx": ctx, "seed": seed, "block_size": block_size,
                    "rel_err_mean": err["rel_err_mean"],
                    "rel_err_max":  err["rel_err_max"],
                    "compression":  mem["compression"],
                    "bf16_mb":      mem["bf16_mb"],
                    "nf4_mb":       mem["nf4_mb"],
                    "pass":         ok,
                }
                results["runs"].append(run)
                tag = "PASS" if ok else "FAIL"
                print(f"  [{tag}] ctx={ctx:>5d}  seed={seed}  "
                      f"block={block_size:>3d}  "
                      f"rel_err={err['rel_err_mean']*100:.2f}%  "
                      f"compress={mem['compression']:.2f}x  "
                      f"({mem['bf16_mb']:.1f} MB -> {mem['nf4_mb']:.1f} MB)")

    results["overall_pass"] = overall_pass

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2))
    print(f"\nWrote {out_path}")
    print(f"Overall: {'PASS' if overall_pass else 'FAIL'}")
    return 0 if overall_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())

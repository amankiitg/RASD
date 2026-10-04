#!/usr/bin/env python3
"""MLSys Phase 0.3 — does install_ring_attention bind to Llama-3.1's
attention module?

`src/models/ring_llama_attention.install_ring_attention` was written
against Llama-2's `LlamaAttention`. Arms 2 and 3 point it at Llama-3.1-8B,
so before burning GPU time we confirm the patch actually lands on the
same module surface.

This builds a Llama-3.1-SHAPED model from a hand-specified config (random
weights, tiny dims) rather than loading the real 8B checkpoint:

  * it exercises the code path under test (layer.self_attn discovery,
    attribute surface, forward monkey-patching), which is what can
    actually break when the architecture changes;
  * loading 8B weights needs a gated token download and ~16 GB, which
    tells us nothing extra about binding.

Attributes required by the ring forward (see rasd_inference and the
kernel): q_proj, k_proj, v_proj, o_proj, num_key_value_heads, rotary_emb.

NOTE: this checks *binding*, not numerics. Executing the ring forward
needs the pod's pinned transformers (the repo notes 5.x broke tuple
past_key_values, which is the cache layout the dual-cache design relies
on). Run this on the pod too as part of the env smoke check.

Exit: 0 = binding OK, 1 = binding broken (patch needed), 2 = inconclusive.
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

REQUIRED_ATTRS = ["q_proj", "k_proj", "v_proj", "o_proj",
                  "num_key_value_heads", "rotary_emb"]

# Llama-3.1-8B's real architecture, shrunk so construction is cheap.
# The things that matter for binding are preserved: GQA (num_key_value_heads
# < num_attention_heads), head_dim=128, rope_theta=500000, and the
# 131072 native context.
LLAMA31_SHAPE = dict(
    vocab_size=128256,
    hidden_size=512,
    intermediate_size=1024,
    num_hidden_layers=3,
    num_attention_heads=8,
    num_key_value_heads=2,          # GQA — the Llama-2 patch must cope
    head_dim=128,
    max_position_embeddings=131072,
    rope_theta=500000.0,
    rms_norm_eps=1e-5,
)


def _load_llama_classes():
    """Import LlamaConfig / LlamaForCausalLM across transformers layouts.

    Top-level `from transformers import LlamaForCausalLM` is not
    guaranteed: it failed on the Lambda pod (transformers 4.4x) even
    though the package imported fine. Transformers lazy-loads model
    classes via `_LazyModule`, so a lazy-attr miss is possible while
    `import transformers` succeeds. Fall back to the submodule path.

    Returns (LlamaConfig, LlamaForCausalLM, note) or raises RuntimeError
    with every attempt's error attached.
    """
    import transformers
    errors = []

    def _try(label, fn):
        try:
            return fn()
        except Exception as e:  # noqa: BLE001
            errors.append(f"{label}: {type(e).__name__}: {e}")
            return None

    LlamaConfig = _try("transformers.LlamaConfig",
                       lambda: __import__("transformers",
                                          fromlist=["LlamaConfig"]).LlamaConfig)
    LlamaForCausalLM = _try(
        "transformers.LlamaForCausalLM",
        lambda: __import__("transformers",
                           fromlist=["LlamaForCausalLM"]).LlamaForCausalLM)
    if LlamaForCausalLM is None:
        LlamaForCausalLM = _try(
            "transformers.models.llama.modeling_llama.LlamaForCausalLM",
            lambda: __import__("transformers.models.llama.modeling_llama",
                               fromlist=["LlamaForCausalLM"]).LlamaForCausalLM)
    if LlamaConfig is None:
        LlamaConfig = _try(
            "transformers.models.llama.configuration_llama.LlamaConfig",
            lambda: __import__("transformers.models.llama.configuration_llama",
                               fromlist=["LlamaConfig"]).LlamaConfig)

    note = f"transformers {transformers.__version__}"
    if LlamaConfig is None or LlamaForCausalLM is None:
        raise RuntimeError(note + " | " + " | ".join(errors))
    return LlamaConfig, LlamaForCausalLM, note


def main() -> int:
    try:
        LlamaConfig, LlamaForCausalLM, note = _load_llama_classes()
        print(f"  loaded Llama classes via fallback chain ({note})")
    except Exception as e:  # noqa: BLE001
        print(f"  ERROR: cannot import transformers Llama classes: {e}")
        return 2

    from src.models.ring_llama_attention import install_ring_attention

    print("=" * 72)
    print("Phase 0.3 — ring attention binding to a Llama-3.1-shaped model")
    print("=" * 72)

    try:
        cfg = LlamaConfig(**LLAMA31_SHAPE)
        # keep the tiny model single-dtype and cheap
        model = LlamaForCausalLM(cfg)
    except Exception as e:  # noqa: BLE001
        print(f"  ERROR: could not build the Llama-3.1-shaped model: {e}")
        return 2

    n_layers = len(model.model.layers)
    print(f"  built Llama-3.1-shaped model: {n_layers} layers, "
          f"GQA {cfg.num_attention_heads}q/{cfg.num_key_value_heads}kv, "
          f"head_dim={getattr(cfg, 'head_dim', 'n/a')}")
    try:
        import transformers
        print(f"  transformers {transformers.__version__}")
    except Exception:  # noqa: BLE001
        pass

    attn0 = model.model.layers[0].self_attn
    missing = [a for a in REQUIRED_ATTRS if not hasattr(attn0, a)]
    if missing:
        # Distinguish "the patch needs fixing" from "this transformers
        # major version moved the API". 5.x replaced num_key_value_heads
        # with num_key_value_groups and hoisted rotary_emb to the model.
        is_v5 = (hasattr(attn0, "num_key_value_groups")
                 or hasattr(attn0, "rotary_fn"))
        print(f"  [FAIL] attention module is missing {missing}")
        if is_v5:
            print()
            print("  CAUSE: this looks like the transformers 5.x LlamaAttention,")
            print("         which moved `rotary_emb` to the model and replaced")
            print("         `num_key_value_heads` with `num_key_value_groups`.")
            print("         The ring forward reads self.rotary_emb and")
            print("         self.num_key_value_heads, so it cannot run here.")
            print("         PIN transformers>=4.46,<5 (see environment_gpu.yml).")
            print("         This is a version mismatch, not a Llama-3.1 issue.")
            return 2
        print("         install_ring_attention would patch something that")
        print("         cannot execute the ring forward — patch required.")
        return 1
    print(f"  [PASS] all required attributes present: {', '.join(REQUIRED_ATTRS)}")
    print(f"         num_key_value_heads = {attn0.num_key_value_heads}")

    # world_size=1 must be a strict no-op (preserves single-rank exactness)
    original_forward = attn0.forward
    n = install_ring_attention(model, world_size=1, rank=0)
    if n != 0 or model.model.layers[0].self_attn.forward is not original_forward:
        print(f"  [FAIL] world_size=1 was not a no-op (patched {n} layers)")
        return 1
    print("  [PASS] world_size=1 is a no-op (single-rank path unchanged)")

    # world_size>1 must patch every layer and install the ring state
    n = install_ring_attention(model, world_size=4, rank=2, chunk_size=2048,
                               prefetch_depth=1, kv_quant=True)
    if n != n_layers:
        print(f"  [FAIL] patched {n} layers, expected {n_layers}")
        return 1
    print(f"  [PASS] patched {n}/{n_layers} LlamaAttention layers")

    bad = []
    for i, layer in enumerate(model.model.layers):
        a = layer.self_attn
        if getattr(a, "forward", None).__func__.__name__ != "_ring_llama_attention_forward":
            bad.append((i, "forward not patched"))
        for attr, want in (("_ring_rank", 2), ("_ring_world_size", 4),
                           ("_ring_chunk_size", 2048), ("_ring_prefetch_depth", 1),
                           ("_ring_kv_quant", True), ("_ring_prefill_len", 0)):
            if getattr(a, attr, "<missing>") != want:
                bad.append((i, f"{attr}={getattr(a, attr, '<missing>')} != {want}"))
    if bad:
        print(f"  [FAIL] {len(bad)} binding problems, first 3: {bad[:3]}")
        return 1
    print("  [PASS] ring state installed on every layer "
          "(rank/world_size/chunk/prefetch/kv_quant/prefill_len)")

    print()
    print("  VERDICT: install_ring_attention binds cleanly to Llama-3.1's")
    print("           attention surface. No patch required.")
    print("           (Binding only — numerical execution still needs the")
    print("            pod's pinned transformers + CUDA.)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""
Plumbing tests for the LlamaAttention ring-attention patch
(src/models/ring_llama_attention.py).

R2 of M3_RING_INTEGRATION_PLAN.md.

These tests verify the *installer* (monkey-patch lifecycle) using mock
attention modules. Multi-rank correctness on real Llama models is gated
on R6 (pod) since transformers==4.44.2 is not installed locally.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest


def _make_mock_target_model(num_layers: int = 4):
    """Return an object with .model.layers[i].self_attn — the Llama structural subset."""
    layers = []
    for i in range(num_layers):
        attn = SimpleNamespace()
        attn.layer_idx = i
        attn.head_dim = 32
        attn.num_heads = 4
        attn.num_key_value_heads = 4
        attn.num_key_value_groups = 1

        # A minimal "forward" stand-in we can spot in tests
        def original_forward(*args, _i=i, **kwargs):
            return f"original-forward-layer-{_i}"

        attn.forward = original_forward
        layers.append(SimpleNamespace(self_attn=attn))

    inner = SimpleNamespace(layers=layers)
    return SimpleNamespace(model=inner)


class TestInstaller:
    def test_no_op_world_size_1(self):
        from src.models.ring_llama_attention import install_ring_attention
        model = _make_mock_target_model(4)
        n = install_ring_attention(model, world_size=1, rank=0)
        assert n == 0
        # Layers must be untouched
        for layer in model.model.layers:
            assert not hasattr(layer.self_attn, "_ring_world_size")
            assert layer.self_attn.forward("x") == f"original-forward-layer-{layer.self_attn.layer_idx}"

    def test_no_op_world_size_none(self):
        from src.models.ring_llama_attention import install_ring_attention
        model = _make_mock_target_model(2)
        n = install_ring_attention(model, world_size=None, rank=0)
        assert n == 0

    def test_patches_all_layers(self):
        from src.models.ring_llama_attention import install_ring_attention
        model = _make_mock_target_model(8)
        n = install_ring_attention(model, world_size=4, rank=2)
        assert n == 8
        for layer in model.model.layers:
            attn = layer.self_attn
            assert attn._ring_rank == 2
            assert attn._ring_world_size == 4
            # Original forward stashed
            assert callable(attn._ring_original_forward)
            # Forward replaced with our patched version (bound method)
            assert attn.forward.__func__.__name__ == "_ring_llama_attention_forward"

    def test_patched_forward_falls_through_at_world_size_1(self):
        """Even after install, if _ring_world_size becomes 1 the patch hands back to original."""
        from src.models.ring_llama_attention import install_ring_attention
        model = _make_mock_target_model(2)
        install_ring_attention(model, world_size=4, rank=0)
        attn = model.model.layers[0].self_attn
        # Force fall-through by reverting world_size
        attn._ring_world_size = 1
        # The patched forward should hit the world_size==1 short-circuit
        result = attn.forward("dummy_hidden_states")
        assert result == "original-forward-layer-0"

    def test_raises_for_non_llama_structure(self):
        from src.models.ring_llama_attention import install_ring_attention
        bad_model = SimpleNamespace()  # no .model.layers
        with pytest.raises(RuntimeError, match="layers"):
            install_ring_attention(bad_model, world_size=2, rank=0)


class TestCacheHelpers:
    """Verify the cache-handling helpers across DynamicCache and tuple formats."""

    def test_cache_is_empty_none(self):
        from src.models.ring_llama_attention import _cache_is_empty
        assert _cache_is_empty(None, layer_idx=0)

    def test_cache_is_empty_tuple_short(self):
        from src.models.ring_llama_attention import _cache_is_empty
        # past_key_value as legacy tuple of length 2, asking for layer 5
        assert _cache_is_empty([(None, None), (None, None)], layer_idx=5)

    def test_cache_is_empty_dynamic(self):
        from src.models.ring_llama_attention import _cache_is_empty

        class FakeCache:
            key_cache = []  # nothing yet

        assert _cache_is_empty(FakeCache(), layer_idx=0)

    def test_cache_read_tuple(self):
        import torch
        from src.models.ring_llama_attention import _cache_read
        k = torch.randn(1, 4, 8, 16)
        v = torch.randn(1, 4, 8, 16)
        cache = [(k, v)]
        rk, rv = _cache_read(cache, 0)
        assert rk is k and rv is v

    def test_cache_read_dynamic(self):
        import torch
        from src.models.ring_llama_attention import _cache_read

        class FakeCache:
            key_cache: list
            value_cache: list

        c = FakeCache()
        c.key_cache = [torch.zeros(1)]
        c.value_cache = [torch.ones(1)]
        rk, rv = _cache_read(c, 0)
        assert torch.equal(rk, torch.zeros(1))
        assert torch.equal(rv, torch.ones(1))

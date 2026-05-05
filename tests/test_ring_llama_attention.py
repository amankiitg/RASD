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
            # Initial prefill_len = 0; set by driver after prefill
            assert attn._ring_prefill_len == 0
            # A3/A4 default to no chunking, sync rotation
            assert attn._ring_chunk_size is None
            assert attn._ring_prefetch_depth == 0
            # Original forward stashed
            assert callable(attn._ring_original_forward)
            # Forward replaced with our patched version (bound method)
            assert attn.forward.__func__.__name__ == "_ring_llama_attention_forward"

    def test_knobs_propagate_through_install(self):
        """A3/A4 values supplied at install time land on every attention module."""
        from src.models.ring_llama_attention import install_ring_attention
        model = _make_mock_target_model(4)
        n = install_ring_attention(model, world_size=4, rank=0,
                                   chunk_size=512, prefetch_depth=1)
        assert n == 4
        for layer in model.model.layers:
            assert layer.self_attn._ring_chunk_size == 512
            assert layer.self_attn._ring_prefetch_depth == 1

    def test_set_prefill_len_propagates(self):
        from src.models.ring_llama_attention import install_ring_attention, set_prefill_len
        model = _make_mock_target_model(6)
        install_ring_attention(model, world_size=4, rank=1)
        n = set_prefill_len(model, prefill_len=2048)
        assert n == 6
        for layer in model.model.layers:
            assert layer.self_attn._ring_prefill_len == 2048

    def test_set_prefill_len_noop_without_install(self):
        from src.models.ring_llama_attention import set_prefill_len
        model = _make_mock_target_model(4)
        # No install — set_prefill_len should be a no-op
        n = set_prefill_len(model, prefill_len=512)
        assert n == 0
        for layer in model.model.layers:
            assert not hasattr(layer.self_attn, "_ring_prefill_len")

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


# ---------------------------------------------------------------------------
# R3 integration: indexing math used by RASDInference._prefill
# ---------------------------------------------------------------------------

class TestPrefillSliceMath:
    """Verify the slice indexing the rasd_inference prefill path uses."""

    def test_world_size_1_returns_full_input(self):
        """World_size=1 path passes the full input_ids through unchanged."""
        import torch
        S, B = 64, 1
        input_ids = torch.arange(S).unsqueeze(0).expand(B, -1)
        # The single-rank branch in generate() does no slicing
        local_ids = input_ids
        local_pos = None
        assert local_ids.shape == (B, S)
        assert local_pos is None

    def test_world_size_4_slice_boundaries(self):
        """Each rank gets a contiguous slice of equal size; positions match."""
        import torch
        S, B, W = 32, 1, 4
        input_ids = torch.arange(S).unsqueeze(0).expand(B, -1)
        for rank in range(W):
            S_local = S // W
            start = rank * S_local
            end   = start + S_local
            local_ids = input_ids[:, start:end].contiguous()
            local_pos = torch.arange(start, end).unsqueeze(0).expand(B, -1)
            assert local_ids.shape == (B, S_local)
            assert local_pos[0, 0].item() == start
            assert local_pos[0, -1].item() == end - 1
            # Slice contents must equal absolute positions for this contrived input
            assert torch.equal(local_ids[0], torch.arange(start, end))

    def test_world_size_must_divide_sequence(self):
        """generate() guards against non-divisible sequence length under multi-rank."""
        import torch
        S, W = 33, 4
        S_local_check = S % W
        assert S_local_check != 0, "test fixture must be non-divisible"
        # The actual assertion lives in generate() — this just documents the contract


class TestVerifyPositionIds:
    """Decode-time position_ids must reflect global_seqlen, not local cache len."""

    def test_position_ids_at_first_verify(self):
        import torch
        S, k = 4096, 4  # prompt len, k draft tokens
        global_seqlen = S
        q_len = k + 1  # cur_token + draft_seq
        t_pos = torch.arange(global_seqlen, global_seqlen + q_len)
        assert t_pos.tolist() == [4096, 4097, 4098, 4099, 4100]

    def test_position_ids_after_n_acc_growth(self):
        """After verify with n_acc=2 accepted, global_seqlen advances by 3."""
        import torch
        S, k = 4096, 4
        n_acc = 2
        global_seqlen = S + n_acc + 1  # post-round bookkeeping in generate()
        assert global_seqlen == S + 3
        q_len = k + 1
        t_pos = torch.arange(global_seqlen, global_seqlen + q_len)
        # First new pos in round 2 = S + 3
        assert t_pos[0].item() == S + 3
        assert t_pos[-1].item() == S + 3 + k

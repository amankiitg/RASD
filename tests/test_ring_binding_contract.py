"""Regression tests for the ring-attention binding contract.

These lock in the behaviour that scripts/mlsys_ring_binding_check.py
verifies, plus the specific defect that aborted an 8xA100 run:

The check compared bound methods with `is`:

    original_forward = attn.forward
    if ... attn.forward is not original_forward:   # ALWAYS True

`obj.forward` constructs a NEW bound-method object on every attribute
access, so an identity comparison on bound methods is always False and
the check reported "world_size=1 was not a no-op" even though
install_ring_attention had legitimately patched nothing. The correct
test compares `__func__`.
"""

from types import SimpleNamespace as NS

import pytest

from src.models.ring_llama_attention import install_ring_attention


def _make_model(n_layers: int = 3):
    """Minimal Llama-shaped stand-in: obj.model.layers[i].self_attn."""
    class _Attn:
        def forward(self, *args, **kwargs):  # pragma: no cover - never called
            return "original"

    layers = [NS(self_attn=_Attn()) for _ in range(n_layers)]
    return NS(model=NS(layers=layers))


def test_bound_method_identity_is_unreliable():
    """Documents why the old check was broken.

    If this ever starts failing, Python's method-binding semantics
    changed and the `__func__` comparison can be revisited.
    """
    attn = _make_model(1).model.layers[0].self_attn
    assert attn.forward is not attn.forward
    assert attn.forward.__func__ is attn.forward.__func__


def test_world_size_one_is_a_strict_noop():
    model = _make_model(3)
    before = model.model.layers[0].self_attn.forward.__func__

    n = install_ring_attention(model, world_size=1, rank=0)

    assert n == 0
    after = model.model.layers[0].self_attn.forward.__func__
    assert after is before, "world_size=1 must not touch forward"


@pytest.mark.parametrize("world_size", [0, 1, None])
def test_non_distributed_world_sizes_are_noops(world_size):
    model = _make_model(2)
    assert install_ring_attention(model, world_size=world_size, rank=0) == 0


def test_world_size_gt_one_patches_every_layer():
    model = _make_model(4)

    n = install_ring_attention(model, world_size=4, rank=2,
                              chunk_size=2048, prefetch_depth=1,
                              kv_quant=True)

    assert n == 4
    for layer in model.model.layers:
        attn = layer.self_attn
        assert attn.forward.__func__.__name__ == "_ring_llama_attention_forward"
        assert attn._ring_rank == 2
        assert attn._ring_world_size == 4
        assert attn._ring_chunk_size == 2048
        assert attn._ring_prefetch_depth == 1
        assert attn._ring_kv_quant is True
        assert attn._ring_prefill_len == 0


def test_original_forward_is_preserved_for_fallback():
    """The single-rank fallback relies on _ring_original_forward."""
    model = _make_model(2)
    original = model.model.layers[0].self_attn.forward.__func__

    install_ring_attention(model, world_size=4, rank=1)

    stored = model.model.layers[0].self_attn._ring_original_forward
    assert getattr(stored, "__func__", stored) is original

"""`_reseed` must return the RNG to the state a fresh process starts in.

The one-load arm driver generates every arm from ONE engine, and `RASDInference.__init__`
seeds the RNG once (rasd_inference.py:1015) rather than per generation. Without a reseed
each arm inherits the state its predecessors left behind, so an arm-to-arm token-stream
difference can be an artefact of ARM ORDER instead of the engine change under test. The 1x
rehearsal showed exactly that: at world_size==1 all four arms are the same code, yet three
agreed and one differed.

This is also the test that would have caught the `NameError: name 'torch' is not defined`
that the second rehearsal died on -- `_reseed` lives at module level while `torch` is a
local import inside `main()`, so calling it is the only way to find out.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
pytest.importorskip("torch")

_spec = importlib.util.spec_from_file_location("mlsys_df_arms",
                                               REPO / "scripts" / "mlsys_df_arms.py")
df_arms = importlib.util.module_from_spec(_spec)
sys.modules["mlsys_df_arms"] = df_arms
_spec.loader.exec_module(df_arms)


def test_reseed_is_callable_at_module_level():
    """The regression: a module-level helper that reaches for a name only main() imports."""
    df_arms._reseed(42)          # must not raise NameError


def test_reseed_makes_sampling_reproducible():
    import torch
    torch.manual_seed(0)
    torch.randn(5)               # advance the stream, as a previous arm would have
    df_arms._reseed(7)
    first = torch.randn(4).tolist()
    torch.randn(100)             # pretend another arm ran in between
    df_arms._reseed(7)
    second = torch.randn(4).tolist()
    assert first == second, "a reseeded stream must not depend on what ran before it"


def test_reseed_differs_between_seeds():
    import torch
    df_arms._reseed(1)
    a = torch.randn(4).tolist()
    df_arms._reseed(2)
    b = torch.randn(4).tolist()
    assert a != b


def test_the_arm_table_names_only_known_arms():
    for name in df_arms.ARMS:
        spec = df_arms.ARMS[name]
        assert set(spec) == {"kv_block_size", "ring_decode_mode",
                            "ring_expand_after_rotation"}, name
    assert df_arms.ARMS["ring32768x"]["ring_expand_after_rotation"] is True
    assert df_arms.ARMS["ring2048"]["ring_expand_after_rotation"] is False

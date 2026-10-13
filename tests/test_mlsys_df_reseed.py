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


# THE DIGESTS FROM THE THIRD 1x REHEARSAL (2026-10-13), verbatim. At world_size==1 no ring
# knob is installed, so all four arms are the same code and must produce identical streams --
# and after the reseed fix they do. Before it, `splitkv`'s two streams were exactly exchanged.
_REHEARSAL_DIGESTS_AFTER_FIX = {
    "ring2048|target": "1c4dbd04a4f04c4c", "ring2048|spec": "c57d95008d435eb2",
    "ring2048|greedy": "1c4dbd04a4f04c4c",
    "ring32768|target": "1c4dbd04a4f04c4c", "ring32768|spec": "c57d95008d435eb2",
    "ring32768|greedy": "1c4dbd04a4f04c4c",
    "ring32768x|target": "1c4dbd04a4f04c4c", "ring32768x|spec": "c57d95008d435eb2",
    "ring32768x|greedy": "1c4dbd04a4f04c4c",
    "splitkv|target": "1c4dbd04a4f04c4c", "splitkv|spec": "c57d95008d435eb2",
    "splitkv|greedy": "1c4dbd04a4f04c4c",
}

_REHEARSAL_DIGESTS_BEFORE_FIX = {
    "ring2048|target": "1c4dbd04a4f04c4c", "ring2048|spec": "c57d95008d435eb2",
    "ring32768|target": "1c4dbd04a4f04c4c", "ring32768|spec": "c57d95008d435eb2",
    "ring32768x|target": "1c4dbd04a4f04c4c", "ring32768x|spec": "c57d95008d435eb2",
    "splitkv|target": "c57d95008d435eb2", "splitkv|spec": "1c4dbd04a4f04c4c",
}


def test_the_self_test_accepts_the_arms_after_the_reseed_fix():
    got = dict((m, (n, k)) for m, n, k in df_arms._single_rank_selftest(_REHEARSAL_DIGESTS_AFTER_FIX))
    assert got == {"greedy": (1, 4), "spec": (1, 4), "target": (1, 4)}, got


def test_the_self_test_rejects_the_arms_before_the_reseed_fix():
    """The check has to FAIL on the state the rehearsal actually caught, or it proves nothing."""
    got = dict((m, (n, k)) for m, n, k in df_arms._single_rank_selftest(_REHEARSAL_DIGESTS_BEFORE_FIX))
    assert got["spec"] == (2, 4), got
    assert got["target"] == (2, 4), got


def test_the_self_test_does_not_unpack_string_keys():
    """The regression: keys are "<arm>|<mode>" strings, not tuples."""
    df_arms._single_rank_selftest({"a|target": "x", "b|target": "y"})   # must not raise

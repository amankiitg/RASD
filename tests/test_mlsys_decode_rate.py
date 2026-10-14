"""The decode-only rate must not count the prefill token.

`decode_tps` is the primary paired ratio: the payoff boundary is claimed on it,
and the plan revision justifies that choice by saying the prefill is identical
across arms and under 5% of the decode wall. If the numerator counted the token
that prefill produced, the ratio would be pulled toward 1.0 by exactly one
token's worth of the shorter arm's time -- a bias in the direction of the null
hypothesis, which is the direction that cannot be caught by looking at the
result.

The comment already said `tokens_generated - 1`; the code divided the full
count. These tests pin the formula and the degenerate cases.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent


def _load():
    spec = importlib.util.spec_from_file_location(
        "runexp_rate", REPO / "run_experiment.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["runexp_rate"] = mod
    try:
        spec.loader.exec_module(mod)
    except Exception as exc:          # CUDA-only imports are unavailable here
        pytest.skip(f"run_experiment not importable: {type(exc).__name__}")
    return mod


@pytest.mark.skipif(not (REPO / "run_experiment.py").exists(),
                    reason="repo layout changed")
def test_decode_rate_excludes_the_prefill_token():
    rate = _load().decode_rate_tps
    # 101 tokens, 1.0s to first token, 11.0s total -> 100 tokens over 10.0s.
    assert rate(101, 11.0, 1000.0) == pytest.approx(10.0, abs=1e-9)
    # The pre-fix formula would have said 101/10.0 = 10.1. Assert the two
    # differ, or this test could pass against the old code too.
    assert rate(101, 11.0, 1000.0) != pytest.approx(101 / 10.0, abs=1e-9)


@pytest.mark.skipif(not (REPO / "run_experiment.py").exists(),
                    reason="repo layout changed")
def test_decode_rate_degenerate_cases_do_not_go_negative_or_divide_by_zero():
    rate = _load().decode_rate_tps
    # One token is entirely prefill: no decode product, so the rate is 0 rather
    # than -1/wall (which a naive n-1 would produce).
    assert rate(1, 5.0, 1000.0) == 0.0
    assert rate(0, 5.0, 1000.0) == 0.0
    # ttft absent, or ttft == the whole wall, must not raise or go infinite.
    assert rate(11, 10.0, None) == pytest.approx(1.0, abs=1e-9)
    assert rate(11, 10.0, 10_000.0) >= 0.0


@pytest.mark.skipif(not (REPO / "run_experiment.py").exists(),
                    reason="repo layout changed")
def test_the_row_builder_uses_the_helper_for_both_arms():
    """One call site builds every row, spec and target-only.

    If a second site existed that computed the rate inline, one arm could keep
    the old formula and the paired ratio would carry the difference.
    """
    src = (REPO / "run_experiment.py").read_text()
    assert '"decode_tps"' in src
    assert "decode_rate_tps(" in src
    # no leftover inline division for the decode rate
    assert 'round(metrics["tokens_generated"] / _decode_wall' not in src
    assert src.count("decode_rate_tps(") >= 2      # the def plus the call site


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))

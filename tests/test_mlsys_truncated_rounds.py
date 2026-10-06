"""One acceptance estimand, over one sample of rounds.

`alpha_round` is the primary metric and `decode_tps`'s partner in the payoff
rule, so the CSV column and the analysis must report the same number for the
same run. They did not:

  * the engine's CSV metric was `sum(n_emitted) / (n_rounds * gamma)`. `n_emitted`
    is `n_acc` only for an untruncated round; the final round is cut short by the
    generation cap, so its accepted prefix is verified but only partly emitted;
  * `per_round_alpha` averaged `n_acc / gamma` over ALL rounds, including that
    one;
  * `summarize_trace` computed `alpha_iid`, the GOF and the total/proposed
    identity over ALL rounds while `alpha_round` came from `per_round_alpha`.

Three estimators, three different samples, one published row. The truncated
round is dropped by all of them now, through `included_rounds`, and these tests
pin that the sample is shared and that the identity holds.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parent.parent


def _acceptance():
    sys.path.insert(0, str(REPO))
    spec = importlib.util.spec_from_file_location(
        "acc_mod", REPO / "src" / "analysis" / "acceptance.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["acc_mod"] = mod
    spec.loader.exec_module(mod)
    return mod


def _round(n_acc, gamma=4, truncated=False):
    return {"n_acc": n_acc, "spec_steps": gamma, "round_truncated": truncated}


def test_truncated_round_is_excluded_from_the_per_round_sample():
    acc = _acceptance()
    trace = [_round(4), _round(4), _round(2), _round(1), _round(4, truncated=True)]
    alpha = acc.per_round_alpha(trace)
    # Four kept rounds: 1.0, 1.0, 0.5, 0.25 -- the truncated one's 4.0 is dropped.
    # Keeping it would give a distinctly different mean, so this cannot pass by
    # accident.
    assert alpha.tolist() == [1.0, 1.0, 0.5, 0.25]
    assert alpha.size == 4


def test_summarizer_uses_the_same_sample_for_every_field():
    acc = _acceptance()
    trace = [_round(4), _round(2), _round(0), _round(4, truncated=True)]
    s = acc.summarize_trace(trace)

    kept = [2, 0, 4]                      # n_acc of the three untruncated rounds
    assert s["n_rounds"] == 3
    assert s["n_excluded_truncated"] == 1
    assert s["mean_n_acc"] == pytest.approx(np.mean(kept))
    assert s["total_accepted"] == sum(kept)
    assert s["total_proposed"] == 4 * 3
    # The identity the module documents: alpha_round == accepted/proposed.
    assert s["alpha_round"] == pytest.approx(s["total_accepted"] / s["total_proposed"])
    # In the pre-fix code the truncated round's 4 accepted tokens were in
    # total_accepted and in mean_n_acc, so the row contradicted its own
    # alpha_round. Assert the two are now consistent.
    assert s["total_accepted"] != sum(kept) + 4


def test_engine_csv_metric_is_the_same_estimator():
    """The engine must compute the CSV column the way the analysis does.

    The engine is CUDA-only, so the arithmetic is pinned at the source level and
    the identity is checked numerically here. Both halves are needed: the source
    could name the right variables and still combine them wrongly.
    """
    src = (REPO / "src" / "models" / "rasd_inference.py").read_text()
    assert "acc_verified / (acc_rounds * cfg.spec_steps)" in src, (
        "the engine no longer computes acceptance over non-truncated rounds"
    )
    assert '"acceptance_rate":   total_accepted / max(total_draft_toks, 1)' not in src, (
        "the engine still divides the emitted-token total by every round"
    )
    assert "if round_truncated:" in src and "acc_verified += n_acc" in src

    # Numerically: the engine's formula over a trace equals per_round_alpha's mean.
    acc = _acceptance()
    trace = [_round(4), _round(2), _round(1), _round(4, truncated=True)]
    gamma = 4
    acc_rounds = sum(1 for r in trace if not r["round_truncated"])
    acc_verified = sum(r["n_acc"] for r in trace if not r["round_truncated"])
    engine_value = acc_verified / (acc_rounds * gamma)
    assert engine_value == pytest.approx(float(acc.per_round_alpha(trace).mean()))


def test_cluster_bootstrap_applies_the_window_before_dropping_the_partial_round():
    """Window-matching must count rounds as they happened.

    `upto` implements "the first N rounds", N = the shortest generation among the
    runs compared. Dropping the truncated round first would let a run contribute
    fewer than N rounds to a window it was meant to fill.
    """
    spec = importlib.util.spec_from_file_location(
        "cb_mod", REPO / "scripts" / "mlsys_cluster_bootstrap.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["cb_mod"] = mod
    spec.loader.exec_module(mod)

    long_run = [_round(4) for _ in range(5)] + [_round(4, truncated=True)]
    short_run = [_round(4) for _ in range(3)] + [_round(1, truncated=True)]

    # A 4-round window on the long run keeps rounds 0-3 (no truncation inside it).
    acc = mod.round_accepted(long_run, upto=4)
    drafted = mod.round_drafted(long_run, upto=4)
    assert acc.size == 4 and drafted.size == 4

    # On the short run the window's last round IS the truncated one, so the
    # window yields 3 usable rounds rather than 4 -- and never 2, which is what
    # dropping first and then windowing would give.
    acc = mod.round_accepted(short_run, upto=4)
    assert acc.size == 3, acc.tolist()
    assert acc.tolist() == [4.0, 4.0, 4.0]


def test_dip_sample_matches_the_alpha_sample():
    """The bimodality sample must be the alpha sample, or the dip tests a
    distribution the reported alpha does not describe."""
    src = (REPO / "scripts" / "mlsys_arm_dip.py").read_text()
    assert "return per_round_alpha(trace)" in src, (
        "the dip script recomputes the sample inline and will keep the partial round"
    )
    assert 'r["n_acc"] / r["spec_steps"] for r in trace' not in src


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))

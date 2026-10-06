"""Document-level intervals: resampling the document, and the payoff verdict."""
from __future__ import annotations

import pytest

from src.analysis.document_bootstrap import (
    clears, document_mean_bootstrap, paired_bootstrap, t_cluster_interval,
)


def test_document_bootstrap_is_paired_and_the_verdict_never_rounds():
    # 10 spec/throughput pairs where the spec arm is uniformly faster.
    spec = [10.0, 12.0, 9.0, 11.0, 13.0, 10.5, 9.5, 12.5, 11.5, 10.2]
    targ = [5.0, 6.0, 4.5, 5.5, 6.5, 5.2, 4.8, 6.2, 5.7, 5.1]
    r = paired_bootstrap(spec, targ, kind="ratio", seed=1)
    assert r["point"] == pytest.approx(sum(spec) / sum(targ), rel=1e-12)
    assert r["lo"] > 1.0 and r["n_documents"] == 10
    assert clears(r, 1.0) == "above"

    # A payoff interval that straddles 1.0 is inconclusive, NOT "below". The
    # plan forbids rounding it toward the favourable side.
    assert clears({"lo": 0.7, "hi": 1.4}, 1.0) == "inconclusive"
    assert clears({"lo": 0.8, "hi": 0.99}, 1.0) == "below"
    assert clears({"lo": float("nan"), "hi": 1.2}, 1.0) == "inconclusive"

    # Pairing is enforced: unaligned arms raise rather than silently comparing
    # a 9-document mean against a 10-document mean.
    try:
        paired_bootstrap(spec[:9], targ, kind="ratio")
    except ValueError as e:
        assert "aligned" in str(e)
    else:
        raise AssertionError("unaligned paired arms were accepted")

    # Identical arms give a zero difference, and the t interval is reported
    # beside the bootstrap so a disagreement is visible rather than hidden.
    d = paired_bootstrap(spec, spec, kind="difference", seed=1)
    assert d["point"] == 0.0 and d["lo"] == 0.0 and d["hi"] == 0.0
    boot = document_mean_bootstrap(spec, seed=1)
    t_ci = t_cluster_interval(spec)
    assert t_ci["n_documents"] == boot["n_documents"] == 10
    assert t_ci["lo"] <= t_ci["mean"] <= t_ci["hi"]


if __name__ == "__main__":
    raise SystemExit(__import__("pytest").main([__file__, "-q"]))

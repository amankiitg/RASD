"""A divergence at an indifferent target is a TIE, not a losslessness failure.

Under greedy decoding two runs that see the same prompt must agree token for
token -- but a bf16 target whose top-1 and top-2 logits are within rounding
distance can pick either for arithmetic reasons (reduction order; the ring's
non-associative online-softmax merge, which the engine already has to broadcast
logits to neutralise). Reporting that as an implementation defect is the failure
mode this module guards: an inert rule looks exactly like a strict one until a
real divergence appears and is waved through.

So the distinguishing evidence is the target's own top-1 minus top-2 logit gap at
the divergence, in BOTH arms, recorded per emitted position by the engine.
"""
from __future__ import annotations

import pytest
import torch

from src.analysis.losslessness import (
    TIE_GAP, _gap_at, compare_generations, stage_requirement,
)
from src.models.rasd_inference import _top1_top2_gap

SPEC = [10, 20, 30, 40, 50]
TARGET_SAME = [10, 20, 30, 40, 50]


def _logits(top1_top2: list[tuple[float, float]]) -> torch.Tensor:
    """(1, S, vocab) where each position's top two are the given values."""
    rows = []
    for a, b in top1_top2:
        row = [0.0] * 8
        row[0], row[1] = a, b
        rows.append(row)
    return torch.tensor([rows])


def test_the_gap_measures_top1_minus_top2_per_position():
    gaps = _top1_top2_gap(_logits([(5.0, 4.0), (2.0, 1.5), (9.0, 1.0)]))
    assert gaps == pytest.approx([1.0, 0.5, 8.0])


def test_an_empty_or_missing_logit_tensor_has_no_gaps():
    assert _top1_top2_gap(None) == []
    assert _top1_top2_gap(torch.zeros(1, 0, 8)) == []


def test_identical_runs_are_lossless_regardless_of_gaps():
    res = compare_generations(SPEC, TARGET_SAME, full_length=5,
                              spec_gaps=[0.0] * 5, target_gaps=[0.0] * 5)
    assert res["verdict"] == "LOSSLESS"
    assert res["numeric_tie"] is False


def test_a_divergence_at_an_indifferent_target_is_a_tie():
    spec = [10, 20, 31, 40, 50]
    res = compare_generations(spec, TARGET_SAME, full_length=5,
                              spec_gaps=[9.0, 9.0, 0.02, 9.0, 9.0],
                              target_gaps=[9.0, 9.0, 5.0, 9.0, 9.0])
    assert res["verdict"] == "NUMERIC_TIE"
    assert res["tie_positions"] == [2]
    assert res["numeric_tie"] is True
    assert res["gap_at_divergence_spec"] == pytest.approx(0.02)
    assert "below the" in res["detail"]


def test_a_tie_is_recognised_from_either_arm():
    """The target's gap is a property of the position, not of one arm's run."""
    spec = [10, 20, 31, 40, 50]
    res = compare_generations(spec, TARGET_SAME, full_length=5,
                              spec_gaps=[9.0] * 5,
                              target_gaps=[9.0, 9.0, 0.05, 9.0, 9.0])
    assert res["verdict"] == "NUMERIC_TIE"


def test_a_divergence_at_a_decided_target_is_a_mismatch():
    spec = [10, 20, 31, 40, 50]
    res = compare_generations(spec, TARGET_SAME, full_length=5,
                              spec_gaps=[9.0, 9.0, 6.0, 9.0, 9.0],
                              target_gaps=[9.0, 9.0, 7.0, 9.0, 9.0])
    assert res["verdict"] == "MISMATCH"
    assert res["first_mismatch_position"] == 2
    assert res["numeric_tie"] is False
    assert res["tie_positions"] == []


def test_a_missing_gap_is_not_a_tie():
    """Absence of evidence is not indifference."""
    spec = [10, 20, 31, 40, 50]
    for sg, tg in ((None, None), ([], [9.0] * 5), ([9.0] * 5, None),
                   ([9.0] * 2, [9.0] * 5)):
        res = compare_generations(spec, TARGET_SAME, full_length=5,
                                  spec_gaps=sg, target_gaps=tg)
        assert res["verdict"] == "MISMATCH", (
            f"gaps {sg!r}/{tg!r} were treated as a tie without evidence")


def test_the_threshold_is_the_documented_one():
    """0.1 on the raw logit scale, and it is the boundary that decides."""
    spec = [10, 20, 31, 40, 50]
    at = compare_generations(spec, TARGET_SAME, full_length=5,
                             spec_gaps=[9.0, 9.0, TIE_GAP, 9.0, 9.0],
                             target_gaps=[9.0] * 5)
    just_under = compare_generations(spec, TARGET_SAME, full_length=5,
                                     spec_gaps=[9.0, 9.0, TIE_GAP - 1e-6, 9.0,
                                                9.0],
                                     target_gaps=[9.0] * 5)
    assert just_under["verdict"] == "NUMERIC_TIE"
    assert at["verdict"] == "MISMATCH", (
        "a gap of exactly the threshold is NOT below it")


def test_gap_lookup_is_bounds_safe():
    assert _gap_at([1.0, 2.0], 0) == 1.0
    assert _gap_at([1.0, 2.0], 5) is None
    assert _gap_at([1.0, 2.0], -1) is None
    assert _gap_at(None, 0) is None
    assert _gap_at(["x"], 0) is None


def test_a_stage_does_not_fail_on_ties_but_reports_them():
    rows = [
        {"spec_run_id": "a", "verdict": "LOSSLESS", "verified_prefix": 1024,
         "target_tokens": 1024},
        {"spec_run_id": "b", "verdict": "NUMERIC_TIE",
         "verified_prefix": 1024, "target_tokens": 1024},
    ]
    req = stage_requirement(rows, full_length=1024)
    assert req["ok"] is True, req["failures"]
    assert req["numeric_ties"] == 1, "ties must be counted, not swallowed"


def test_a_stage_fails_on_a_mismatch():
    rows = [
        {"spec_run_id": "a", "verdict": "LOSSLESS", "verified_prefix": 1024,
         "target_tokens": 1024},
        {"spec_run_id": "b", "verdict": "MISMATCH",
         "first_mismatch_position": 7, "verified_prefix": 7,
         "target_tokens": 1024},
    ]
    req = stage_requirement(rows, full_length=1024)
    assert req["ok"] is False
    assert req["mismatches"] == 1
    assert any("MISMATCH" in f for f in req["failures"])

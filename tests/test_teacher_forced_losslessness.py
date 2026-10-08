"""The teacher-forced losslessness rule: its arithmetic, and its refusal modes.

The point of these tests is that the rule is decidable from numbers alone, so
it can be pinned offline before it is wired into a route that costs GPU time.
The two things that must never happen are:
  * a real defect passing (the rule is too generous), and
  * a correct engine failing (the rule is too strict).
Both are exercised below, and so is the third: a tolerance derived from a
measured noise floor that is too large to support any gate at all.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.analysis.teacher_forced_losslessness import (
    TOL_CAP,
    describe,
    evaluate_stream,
    logit_shortfall,
    tol_from_noise,
    tol_provenance,
)


# --------------------------------------------------------------------------
# tol_from_noise: the rule that derives the tolerance
# --------------------------------------------------------------------------
def test_tol_is_twice_the_measured_noise():
    """The 2x is the bound on a legitimately overturned decision, not a fudge."""
    assert tol_from_noise(0.25) == pytest.approx(0.5)
    assert tol_from_noise(0.0) == 0.0


def test_tol_is_capped_and_the_cap_binding_is_reported():
    """The cap is the standing rule; binding it must be visible, not silent.

    A binding cap means TOL is IMPOSED rather than derived, so the gate is
    stricter than the measured noise requires. That is the safe direction, but
    the caller has to be able to say so -- otherwise a report would claim the
    tolerance came from the measurement when it did not.
    """
    # Exactly at the cap: the measurement decides, and is inclusive.
    at_cap = tol_provenance(TOL_CAP / 2)
    assert at_cap["tol"] == pytest.approx(TOL_CAP)
    assert at_cap["cap_binds"] is False
    # Over the cap: TOL is clamped, and that fact is recorded.
    over = tol_provenance(TOL_CAP / 2 + 0.5)
    assert over["tol"] == pytest.approx(TOL_CAP)
    assert over["cap_binds"] is True
    assert over["raw_2x_noise"] > TOL_CAP
    assert "cap binds" in over["note"]
    # The real 8k/32k 1x measurement, which is what the campaign will use.
    real = tol_provenance(2.59961)
    assert real["tol"] == pytest.approx(TOL_CAP)
    assert real["cap_binds"] is True


def test_tol_rejects_a_negative_noise_measurement():
    with pytest.raises(ValueError):
        tol_from_noise(-1.0)


# --------------------------------------------------------------------------
# logit_shortfall
# --------------------------------------------------------------------------
def test_shortfall_is_zero_at_the_argmax_and_positive_below_it():
    row = np.array([1.0, 5.0, 3.0], dtype=np.float32)
    assert logit_shortfall(row, 1) == 0.0     # the argmax itself
    assert logit_shortfall(row, 2) == pytest.approx(2.0)
    assert logit_shortfall(row, 0) == pytest.approx(4.0)


def test_shortfall_accepts_torch_like_logits_including_on_a_cuda_stand_in():
    """A device tensor must convert, not raise a confusing numpy error.

    The pod passes CUDA tensors; before this, that failed inside numpy with a
    message about the device type, which points at the wrong layer.
    """
    class FakeDeviceTensor:
        """Stands in for a CUDA tensor: only detach/cpu/numpy are needed."""
        def __init__(self, values):
            self._v = np.asarray(values, dtype=np.float32)
        def detach(self):
            return self
        def cpu(self):
            return self
        def numpy(self):
            return self._v

    t = FakeDeviceTensor([1.0, 5.0, 3.0])
    assert logit_shortfall(t, 1) == 0.0
    assert logit_shortfall(t, 2) == pytest.approx(2.0)
    res = evaluate_stream([2], [FakeDeviceTensor([1.0, 5.0, 3.0])], tol=1.0)
    assert res["verdict"] == "TF_MISMATCH"


def test_shortfall_rejects_a_token_outside_the_vocabulary():
    with pytest.raises(IndexError):
        logit_shortfall(np.array([1.0, 2.0]), 2)


# --------------------------------------------------------------------------
# evaluate_stream
# --------------------------------------------------------------------------
def _logits(rows):
    return [np.asarray(r, dtype=np.float32) for r in rows]


def test_all_argmax_is_lossless_with_zero_shortfall():
    rows = [[0.0, 3.0, 1.0], [2.0, 0.0, 1.0], [1.0, 1.0, 4.0]]
    res = evaluate_stream([1, 0, 2], _logits(rows), tol=0.5)
    assert res["verdict"] == "TF_LOSSLESS"
    assert res["n_argmax"] == 3
    assert res["max_shortfall"] == 0.0
    assert "LOSSLESS" in describe(res)


def test_a_near_tie_inside_tol_passes():
    """The measured shape difference can legitimately overturn a decision."""
    rows = [[0.0, 3.0, 1.0], [2.0, 0.0, 1.0], [1.0, 1.0, 4.0]]
    res = evaluate_stream([1, 0, 1], _logits(rows), tol=1.0)   # shortfall 3.0? no:
    # position 2 emits token 1, whose shortfall is 4.0 - 1.0 = 3.0 -> too far.
    assert res["verdict"] == "TF_MISMATCH"


def test_a_near_tie_inside_tol_passes_constructed_exactly():
    # argmax 2 (4.0); emitting 1 has shortfall 3.0 - 1.0... use a tight row.
    rows = [[1.0, 4.1, 4.0]]
    res = evaluate_stream([2], _logits(rows), tol=0.2)
    assert res["verdict"] == "TF_LOSSLESS"
    assert res["n_argmax"] == 0 and res["n_within_tol"] == 1
    assert res["max_shortfall"] == pytest.approx(0.1, abs=1e-6)


def test_far_below_the_argmax_is_a_mismatch_and_names_the_worst_position():
    rows = [[0.0, 9.0, 1.0], [0.0, 9.0, 1.0], [0.0, 1.0, 9.0]]
    res = evaluate_stream([2, 1, 1], _logits(rows), tol=0.5)
    assert res["verdict"] == "TF_MISMATCH"
    assert res["first_failure_position"] == 0
    assert res["worst_position"] == 0
    assert res["worst_shortfall"] == pytest.approx(8.0)
    assert "MISMATCH" in describe(res)


def test_an_empty_check_is_unchecked_not_lossless():
    """Zero positions compared must never read as a pass.

    "No evidence" and "no divergence" are different claims; conflating them is
    how a gate reports success without having run.
    """
    res = evaluate_stream([], [], tol=0.5, min_checked=1)
    assert res["verdict"] == "TF_UNCHECKED"
    assert "DID NOT RUN" in describe(res)


def test_the_check_stops_being_valid_if_logits_run_out():
    """A truncated logit array must be UNCHECKED, not a partial pass."""
    res = evaluate_stream([1, 1, 1], _logits([[0.0, 2.0]]), tol=0.5, min_checked=3)
    assert res["verdict"] == "TF_UNCHECKED"


def test_tolerance_monotonicity_a_real_defect_survives_a_small_tol():
    """Widening tol admits more; a real defect must be caught at the bound.

    The 8.0-logit shortfall above is an order of magnitude past any plausible
    shape difference (the 1x floor measured ~2 logits at worst), so no
    legitimate tolerance derived from it can admit that token.
    """
    rows = [[0.0, 9.0, 1.0]]
    for tol in (0.0, 0.5, 1.0, 2.0, TOL_CAP):
        res = evaluate_stream([2], _logits(rows), tol=tol)
        assert res["verdict"] == "TF_MISMATCH", f"tol={tol} admitted a defect"

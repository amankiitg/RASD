"""`token_gaps` must be aligned, element-for-element, with the emitted ids.

The losslessness tie rule reads the gap **at the position where two arms first
disagree**. A list that is one entry short -- the seed token, or a round after the
first -- shifts every later lookup by one, and the failure is invisible: it
reports the *neighbouring* token's indifference, i.e. it excuses a real mismatch
as a numerics tie. That is worse than having no gap array at all.

`RASDInference.generate` cannot be executed here: `_setup_streams` raises
"RASD requires CUDA." without a GPU. So these tests drive the engine's own
accumulation code -- `_step_gap`, `_round_emitted_gaps`, and the loop's append
order as `_round_commit_plan` defines it -- with a hand-built logits tensor in
place of the forward. That covers the alignment logic; the first GPU stage then
proves the wiring end to end, because the cap smoke asserts the length
relationship on every row of both arms (asserted structurally at the bottom).
"""
from __future__ import annotations

from pathlib import Path

import pytest
import torch

from src.models.rasd_inference import (
    _round_commit_plan, _round_emitted_gaps, _step_gap, _top1_top2_gap,
)

REPO = Path(__file__).resolve().parent.parent


def _logits(spec: list[tuple[float, float]]) -> torch.Tensor:
    """(1, S, vocab) whose top two at each position are the given values."""
    rows = []
    for a, b in spec:
        row = [0.0] * 8
        row[0], row[1] = a, b
        rows.append(row)
    return torch.tensor([rows])


def _gap(a: float, b: float) -> float:
    return a - b


class _FakeEngine:
    """The append order the loop uses, without CUDA.

    `generated` and `emitted_gaps` are appended together: a token is emitted and
    its gap is recorded in the same step, so the two lists are the same length
    after every round. `_round_commit_plan` decides how many tokens a round emits
    and whether a bonus follows, so the truncation arithmetic under test is the
    engine's, not this file's.
    """

    def __init__(self, logits_by_round: list[torch.Tensor], gamma: int,
                 budget: int, seed_gap: list):
        self.logits = logits_by_round
        self.gamma = gamma
        self.budget = budget
        self.generated: list[int] = [0]              # the seed token
        self.gaps: list[float] = list(seed_gap)
        self.expected: list[float] = list(seed_gap)

    def run(self, n_accs: list[int]) -> None:
        for rnd, (logits, n_acc) in enumerate(zip(self.logits, n_accs)):
            budget = self.budget - len(self.generated)
            if budget < 1:
                break
            n_emit, committed, with_bonus, _tr = _round_commit_plan(
                budget, n_acc, self.gamma)
            ids = list(range(100 * (rnd + 1), 100 * (rnd + 1) + committed))
            self.generated.extend(ids)
            self.gaps.extend(_round_emitted_gaps(logits, n_emit, n_acc,
                                                 with_bonus))
            # Independently: the gap of each emitted position, read straight off
            # the logits tensor in the order the tokens are appended.
            self.expected.extend(
                _top1_top2_gap(logits[:, :n_emit, :])
                + (_top1_top2_gap(logits[:, n_acc:n_acc + 1, :])
                   if with_bonus else []))


def test_a_step_gap_is_exactly_one_position():
    gaps = _step_gap(_logits([(5.0, 4.5)])[:, 0, :])
    assert gaps == pytest.approx([0.5])
    assert _step_gap(None) == []


def test_a_round_reports_one_gap_per_emitted_token():
    logits = _logits([(9.0, 8.0), (7.0, 6.5), (5.0, 1.0), (4.0, 3.9), (3.0, 2.0)])

    full = _round_emitted_gaps(logits, n_emit=4, n_acc=4, with_bonus=True)
    assert full == pytest.approx([1.0, 0.5, 4.0, 0.1, 1.0]), (
        "4 accepted draft positions + the bonus position 4")

    truncated = _round_emitted_gaps(logits, n_emit=2, n_acc=4, with_bonus=False)
    assert truncated == pytest.approx([1.0, 0.5]), (
        "a budget-truncated round emits its prefix and no bonus")


def test_no_acceptance_still_reports_the_bonus_position():
    logits = _logits([(9.0, 8.0), (2.0, 1.0)])
    gaps = _round_emitted_gaps(logits, n_emit=0, n_acc=0, with_bonus=True)
    assert gaps == pytest.approx([1.0]), (
        "with n_acc=0 the only emitted token is the bonus, read at position 0")


def test_the_seed_token_is_the_first_gap():
    """The loop's first emitted position is the token sampled from prefill."""
    seed = _step_gap(_logits([(6.0, 5.0)])[:, 0, :])
    assert len(seed) == 1

    eng = _FakeEngine([_logits([(3.0, 1.0)])], gamma=2, budget=8,
                      seed_gap=seed)
    eng.run([0])                                  # n_acc=0 -> 1 bonus token
    assert eng.gaps == pytest.approx(eng.expected)
    assert len(eng.gaps) == len(eng.generated) == 2


def test_gaps_stay_aligned_across_many_rounds_including_the_truncated_one():
    """The case that broke: gaps for round 1 and later, and a short final round."""
    logits = [_logits([(4.0, 3.0), (2.0, 1.9), (1.5, 1.0), (1.0, 0.5)])] * 4
    seed = _step_gap(_logits([(7.0, 6.0)])[:, 0, :])
    eng = _FakeEngine(logits, gamma=3, budget=6, seed_gap=seed)

    eng.run([3, 3, 3, 3])          # 1 + 4 + 4 + (budget clamps the rest)

    assert len(eng.generated) == 6, "the budget must be respected exactly"
    assert len(eng.gaps) == len(eng.generated), (
        f"{len(eng.gaps)} gaps for {len(eng.generated)} emitted tokens: the tie "
        f"rule would read the wrong position")
    assert eng.gaps == pytest.approx(eng.expected)


def test_a_round_with_no_room_emits_nothing_and_records_nothing():
    seed = _step_gap(_logits([(7.0, 6.0)])[:, 0, :])
    eng = _FakeEngine([_logits([(4.0, 3.0)])], gamma=3, budget=1, seed_gap=seed)
    before = list(eng.gaps)

    eng.run([3])                    # budget exhausted by the seed token

    assert eng.gaps == before
    assert len(eng.gaps) == len(eng.generated)


def test_the_engine_cannot_run_on_cpu_so_the_cap_smoke_proves_the_wiring():
    """Stated as a check, not as prose: the GPU-side assertion must exist.

    `generate()` requires CUDA streams (rasd_inference._setup_streams raises
    without a GPU), so the alignment above is proven on the accumulation code
    and on real hardware by the cap smoke's per-row length check. If that check
    is ever removed, the alignment has no end-to-end proof at all -- which is
    what this test prevents.
    """
    eng = (REPO / "src" / "models" / "rasd_inference.py").read_text()
    assert "RASD requires CUDA." in eng

    check = (REPO / "scripts" / "mlsys_cap_smoke_check.py").read_text()
    assert "token_gaps" in check and "len(gaps) != len(ids)" in check, (
        "the cap smoke no longer asserts that the gap array is as long as the "
        "emitted ids, so nothing proves the alignment on a GPU")
    assert "spec_gaps=a.get(\"token_gaps\")" in check, (
        "the cap smoke does not pass the speculative arm's gaps into the verdict")
    assert "target_gaps=b.get(\"token_gaps\")" in check, (
        "the cap smoke does not pass the target arm's gaps into the verdict")

    rexp = (REPO / "run_experiment.py").read_text()
    assert "len(token_gaps) != len(gen_ids)" in rexp, (
        "run_experiment no longer refuses to write a misaligned gap array")

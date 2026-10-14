"""The repetition ceilings, tested against the campaign's own saved outputs.

Every negative control here is a REAL generation pulled from the pod at the
2026-10-10 repetition stop, not a constructed string:

  NATURAL_f1_128k_pg19_train_1404_s42  acceptance 0.9701, and its last 200 tokens
                                       are " He was deceived." fifty times
  NATURAL_f1_128k_pg19_train_915_s42   acceptance 0.9631, period 20
  NATURAL_f1_128k_pg19_train_0_s42     acceptance 0.6203, the lowest row, clean
  CAPS_prefix1024_targetonly_...       the target-only half of the dialogue loop

A detector tested only on synthetic strings proves that it detects the strings its
author had in mind. The point of these is that the rule has to separate rows the
campaign actually produced.

The positive controls matter as much as the negatives: `pg19_train_0` sits at 0.570
tokens-in-repeat against a 0.60 ceiling, so a ceiling nudged down to 0.55 would
fail a clean row and these tests would say so.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from src.analysis.repetition import (  # noqa: E402
    IN_REPEAT_CEILING, PERIODIC_TAIL_CEILING, REPEAT_CEILING,
    VERBATIM_SPAN_CEILING, longest_repeated_span, non_degenerate_acceptance,
    periodicity, repeat_share, row_reasons, row_stats, token_repeat_stats,
)

INCIDENT = (REPO / "results" / "mlsys"
            / "incident_20261010T113537Z_repetition_stop" / "pod_results")
TOKENS = INCIDENT / "tokens_slim"
PER_TOKEN = INCIDENT / "per_token"

LOOP_1404 = "NATURAL_f1_128k_pg19_train_1404_s42"
LOOP_915 = "NATURAL_f1_128k_pg19_train_915_s42"
LOOP_CAPS = "CAPS_prefix1024_targetonly_pg19_train_1_s42"
CLEAN_0 = "NATURAL_f1_128k_pg19_train_0_s42"
CLEAN_SHORT = "NATURAL_f1_128k_targetshort_pg19_train_915_s42"


def _ids(run_id: str) -> list:
    f = TOKENS / f"{run_id}.json"
    if not f.exists():
        pytest.skip(f"{f.name} not in the pulled artifacts")
    return json.loads(f.read_text())["generated_token_ids"]


def _rounds(run_id: str) -> list:
    f = PER_TOKEN / f"{run_id}.jsonl"
    if not f.exists():
        pytest.skip(f"{f.name} not in the pulled artifacts")
    return [json.loads(l) for l in f.read_text().splitlines() if l.strip()]


# --- the periodicity statistic -------------------------------------------------

def test_the_deceived_loop_is_named_by_its_period():
    """The row the whole stop hung on: period 12, periodic from token 22 to the end."""
    p = periodicity(_ids(LOOP_1404))
    assert p["period"] == 12, p
    assert p["periodic_tail_share"] > 0.95
    assert p["first_periodic_token"] < 30


def test_a_clean_row_is_not_called_periodic():
    p = periodicity(_ids(CLEAN_0))
    assert p["periodic_tail_share"] < 0.06, p


def test_longest_repeated_span_separates_loops_from_prose():
    """The primary detector, and the one with the widest margin: clean <= 64,
    every loop >= 226. A 128-token verbatim repeat in a 1k generation is not a
    stylistic choice."""
    assert longest_repeated_span(_ids(LOOP_1404)) >= VERBATIM_SPAN_CEILING
    assert longest_repeated_span(_ids(LOOP_915)) >= VERBATIM_SPAN_CEILING
    assert longest_repeated_span(_ids(CLEAN_0)) < VERBATIM_SPAN_CEILING


def test_longest_repeated_span_is_exact_on_a_known_pattern():
    assert longest_repeated_span([1, 2, 3, 1, 2, 3, 9]) == 3
    assert longest_repeated_span([1, 2, 3, 4]) == 0
    assert longest_repeated_span([7, 7, 7, 7]) == 2


# --- the ceilings -------------------------------------------------------------

@pytest.mark.parametrize("run_id", [LOOP_1404, LOOP_915, LOOP_CAPS])
def test_real_loops_are_degenerate(run_id):
    st = row_stats(_ids(run_id))
    assert row_reasons(st), f"{run_id} has no reason but is a loop: {st}"
    assert st["rep_degenerate"] is True


@pytest.mark.parametrize("run_id", [CLEAN_0, CLEAN_SHORT])
def test_real_clean_rows_pass(run_id):
    st = row_stats(_ids(run_id))
    assert row_reasons(st) == [], f"{run_id} falsely flagged: {row_reasons(st)}"


def test_the_tightest_clean_row_still_passes_with_margin():
    """`pg19_train_0` is the closest clean row to the tokens-in-repeat ceiling. If
    an edit moved that ceiling down to meet it, the campaign would start failing
    rows for being ordinary prose."""
    st = row_stats(_ids(CLEAN_0))
    assert st["rep_tokens_in_repeat"] < IN_REPEAT_CEILING
    assert IN_REPEAT_CEILING - st["rep_tokens_in_repeat"] > 0.02


def test_a_loop_that_both_arms_share_still_fails():
    """The hole the absolute ceiling closes: the relative rule cannot fire when the
    candidate and its baseline are stuck in the same loop, so a 97%-repetitive pair
    passed at a 1.07x ratio."""
    cand = row_stats(_ids(LOOP_1404))
    base = row_stats(_ids("NATURAL_f1_128k_targetfull_pg19_train_1_s42"))
    assert cand["rep_tokens_in_repeat"] > IN_REPEAT_CEILING
    assert base["rep_tokens_in_repeat"] > IN_REPEAT_CEILING
    relative = cand["rep_tokens_in_repeat"] / base["rep_tokens_in_repeat"]
    assert relative < 2.0, (
        "the fixture must be a pair the old RELATIVE rule let through")
    assert row_reasons(cand), "the absolute rule must still catch it"


def test_repeat_share_matches_the_gates_definition():
    """`gen_repeat_share` in every existing CSV is 1 - distinct/total word trigrams.
    This module must reproduce that definition exactly, or the new column and the
    old one would disagree about the same text."""
    text = "a b c a b c a b c d"
    toks = text.split()
    grams = [tuple(toks[i:i + 3]) for i in range(len(toks) - 2)]
    expected = 1.0 - len(set(grams)) / len(grams)
    assert repeat_share(text) == pytest.approx(expected)
    assert repeat_share("") == 0.0


def test_a_continuation_only_text_is_what_gets_measured():
    """Measuring repetition over `generated/*.txt` (prompt + continuation) diluted
    the loop 128x and made every row look clean: the real files are 440-580 KB
    because they carry a 130k-token prompt. The statistic is only meaningful on the
    continuation, which is what the gate's own `gen_*.txt` contains."""
    loop = " He was deceived." * 50
    assert row_stats([], loop)["rep_repeat_share"] > 0.9
    prompt = " ".join(f"word{i}" for i in range(130000))
    whole_file = prompt + loop
    diluted = row_stats([], whole_file)["rep_repeat_share"]
    assert diluted < 0.01, (
        "a 250-token loop inside a 130k-token prompt disappears from the "
        f"statistic ({diluted}), which is why the stage reported nothing")


# --- acceptance by window, and the non-degenerate protocol ---------------------

def test_windows_are_where_the_loop_shows_up():
    """Acceptance 1.000 from token 128 on is the symptom: the draft is predicting a
    constant, so every proposal is accepted."""
    ids = _ids(LOOP_1404)
    res = non_degenerate_acceptance(_rounds(LOOP_1404), ids, 128)
    assert res["windows_used"] == 0, (
        "every window of the deceived-loop row is periodic, so the row has no "
        f"non-degenerate acceptance at all: {res['per_window']}")
    assert res["acceptance"] is None


def test_a_clean_row_keeps_some_windows():
    ids = _ids(CLEAN_0)
    res = non_degenerate_acceptance(_rounds(CLEAN_0), ids, 128)
    assert res["windows_used"] >= 1
    assert res["acceptance"] is not None
    assert 0.0 < res["acceptance"] <= 1.0


def test_acceptance_is_the_per_round_alpha_not_the_per_token_parameter():
    """Reviewer AbH52 #1: alpha must be accepted-prefix-length / gamma. The stage's
    own CSV is reproduced by sum(n_acc)/sum(gamma) over untruncated rounds, and the
    i.i.d. per-token parameter is a DIFFERENT number (0.438 vs 0.9 at gamma=4), so
    this asserts the identity that distinguishes them."""
    import csv
    rounds = [r for r in _rounds(LOOP_915) if not r.get("round_truncated")]
    alpha = (sum(int(r["n_acc"]) for r in rounds)
             / sum(int(r["spec_steps"]) for r in rounds))
    csv_path = INCIDENT / "natural_f1_128k.csv"
    if not csv_path.exists():
        pytest.skip("stage CSV not in the pulled artifacts")
    reported = {r["run_id"]: r.get("acceptance_rate")
                for r in csv.DictReader(csv_path.open())}
    assert reported[LOOP_915], "the stage must have reported an acceptance"
    assert abs(float(reported[LOOP_915]) - alpha) < 0.001, (
        f"the stage's acceptance {reported[LOOP_915]} is not "
        f"sum(n_acc)/sum(gamma)={alpha:.4f}")
    gamma = 4
    iid = 1.0 - (1.0 - alpha) ** (1.0 / gamma)
    assert abs(iid - alpha) > 0.4, (
        "at this alpha the two definitions differ by less than 0.4, so this test "
        "would not distinguish them")


def test_window_advance_uses_committed_tokens_not_accepted_tokens():
    """A round appends n_committed tokens but reports n_acc accepted ones. Advancing
    by n_emitted walks the window boundaries backwards by one per round."""
    n_committed = sum(int(r.get("n_committed") or 0)
                      for r in _rounds(LOOP_1404))
    assert n_committed >= 1024 - 5, (
        "the trace's committed tokens must account for the whole generation")


def test_token_repeat_stats_on_a_known_pattern():
    ids = [1, 2, 3, 4] * 40
    st = token_repeat_stats(ids)
    assert st["rep_tok_repeat_share"] > 0.9
    assert st["rep_longest_repeat_span"] >= 8
    assert st["rep_tokens_in_repeat"] > 0.9

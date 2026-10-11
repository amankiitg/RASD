"""The gate refuses to run with zero arms.

`load_candidates` accepted an empty candidate list and returned it. The gate
then ran nothing, wrote a CSV with a header and no rows, and exited 0 -- so a
caller that checks only the exit status reads "the gate passed". On
2026-10-10 I wrote a candidates file whose entries used `candidate` as the key
instead of `name`; that particular file would NOT have been caught by this
check (it resolved to three entries, not zero), which is why the tests below
also pin what this guard does and does not cover. See the module docstring
limit test.

These tests pin the empty case for every documented top-level shape, and pin
that the real configs on disk still load, so the check cannot start rejecting
the files it exists to protect.
"""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
GATE = REPO / "scripts" / "mlsys_coherence_gate.py"

# Every candidate file the project actually runs. If the new check rejected one
# of these, it would be breaking the thing it protects.
REAL_FILES = [
    REPO / "configs" / "mlsys_gate_controls.json",
    REPO / "configs" / "mlsys_rope_candidates.json",
    REPO / "configs" / "mlsys_rope_intervention_candidates.json",
    REPO / "configs" / "mlsys_correction_candidates.json",
    REPO / "configs" / "mlsys_followup_rope_candidates.json",
    REPO / "configs" / "mlsys_followup_rope_candidates_rehearsal.json",
]


def _load_gate():
    spec = importlib.util.spec_from_file_location("_gate_under_test", GATE)
    mod = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(mod)
    except Exception as e:                      # noqa: BLE001
        pytest.skip(f"gate not importable here: {e}")
    return mod


# --------------------------------------------------------------------------
# the empty list is refused, under every documented shape
# --------------------------------------------------------------------------

@pytest.mark.parametrize("key", ["coherence_gate", "candidates", "controls"])
def test_empty_list_under_a_documented_key_is_refused(key):
    gate = _load_gate()
    with pytest.raises(SystemExit) as e:
        gate.load_candidates({key: []})
    msg = str(e.value)
    assert "ZERO candidates" in msg, msg
    assert key in msg, f"the error must name the key it found empty: {msg}"


def test_empty_bare_list_is_refused():
    gate = _load_gate()
    with pytest.raises(SystemExit) as e:
        gate.load_candidates([])
    assert "ZERO candidates" in str(e.value)


def test_a_misspelled_top_level_key_still_reports_the_keys_it_saw():
    """The wrong-key case keeps its own, older message.

    It is a different mistake from an empty list and the message should not
    change under it, or the two get debugged as if they were one.
    """
    gate = _load_gate()
    with pytest.raises(SystemExit) as e:
        gate.load_candidates({"rope_arms": [{"name": "x"}]})
    msg = str(e.value)
    assert "no candidate list" in msg, msg
    assert "rope_arms" in msg, f"the error must name the keys present: {msg}"


def test_a_nonempty_list_is_returned_unchanged():
    gate = _load_gate()
    arms = [{"name": "a"}, {"name": "b"}]
    assert gate.load_candidates({"candidates": arms}) == arms
    assert gate.load_candidates(arms) == arms


# --------------------------------------------------------------------------
# the check must not reject the files it protects
# --------------------------------------------------------------------------

@pytest.mark.parametrize("path", REAL_FILES, ids=lambda p: p.name)
def test_every_real_candidate_file_still_loads_non_empty(path):
    gate = _load_gate()
    if not path.exists():
        pytest.skip(f"{path.name} not present")
    loaded = gate.load_candidates(json.loads(path.read_text()))
    assert loaded, f"{path.name} loaded zero candidates; the guard would refuse the real run"


def test_the_rehearsal_twin_loads_the_same_number_of_arms_as_the_real_file():
    """The rehearsal only rehearses the gate if it arms the same number of arms."""
    gate = _load_gate()
    real = REPO / "configs" / "mlsys_followup_rope_candidates.json"
    twin = REPO / "configs" / "mlsys_followup_rope_candidates_rehearsal.json"
    if not (real.exists() and twin.exists()):
        pytest.skip("follow-up candidates files not present")
    a = gate.load_candidates(json.loads(real.read_text()))
    b = gate.load_candidates(json.loads(twin.read_text()))
    assert len(a) == len(b) == 3
    assert [x["name"] for x in a] == [x["name"] for x in b]


# --------------------------------------------------------------------------
# the limit of this guard, recorded as a test so it is not mistaken for
# coverage of the bug that prompted it
# --------------------------------------------------------------------------

def test_the_empty_check_does_not_catch_a_wrong_PER_ENTRY_key():
    """KNOWN GAP, pinned deliberately.

    My 2026-10-10 file was {"candidates": [{"candidate": "ROFIX_F16"}, ...]}.
    That resolves to three entries -- so this guard passes it, and the gate then
    reads name=None on every arm. The empty check cannot see this; only a
    per-entry required-field check could, and the project's own candidate files
    are not consistent enough for one (they use `expect` and `expected`, and
    roles drawn from candidate/baseline/positive/negative/absent), so adding one
    is a decision, not a fix. This test exists so the gap is visible rather than
    assumed covered.
    """
    gate = _load_gate()
    wrong_key_entries = {"candidates": [{"candidate": "ROFIX_F16"},
                                        {"candidate": "ROFIX_F32"}]}
    loaded = gate.load_candidates(wrong_key_entries)
    assert len(loaded) == 2, "this guard is about the empty case, not this one"
    assert all("name" not in e for e in loaded), (
        "if a per-entry check is added later, this test should be flipped, not deleted"
    )

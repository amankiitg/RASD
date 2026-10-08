"""The cost ledger accumulates, so it must be merged and never replaced.

`results/mlsys/gpu_hours.csv` is the campaign's running total and the file the
watcher's `spend()` reads against the hard cap. The pod's copy is a SUPERSET of
it — the pod appends a row as it spends — so a plain copy in either direction
loses rows. Before `scripts/mlsys_merge_cost_ledger.py` existed, an aborted 8x
run lost its spend entirely: the row only ever reached the pod, and the incident
path terminated the instance without pulling the ledger.

The rules asserted here:

  * a pod row is appended when the local file does not already hold that row;
  * the body is a MULTISET union, so two runs that produce byte-identical rows
    are both kept and re-running the merge is a no-op;
  * local rows are never dropped or reordered, and the header is never rewritten;
  * a header mismatch refuses instead of appending rows into a file whose
    columns it does not know — the caller keeps the pod's copy as the evidence.
"""
from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
MERGE = REPO / "scripts" / "mlsys_merge_cost_ledger.py"

HEADER = "stage,wall_seconds,nproc,gpu_hours,node_cost_usd"
LOCAL = HEADER + "\nARM1,600,8,1.3,29.0\n"
POD = HEADER + "\nARM1,600,8,1.3,29.0\nARM2,900,8,2.0,44.64\n"


def _load():
    spec = importlib.util.spec_from_file_location("_ledger_merge", MERGE)
    mod = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(mod)
    except Exception as e:                      # noqa: BLE001
        pytest.skip(f"merge module not importable here: {e}")
    return mod


def test_a_new_pod_row_is_appended():
    m = _load().merge(LOCAL, POD)
    assert m["status"] == "merged"
    assert m["appended"] == 1
    assert m["text"].splitlines() == LOCAL.splitlines() + ["ARM2,900,8,2.0,44.64"]


def test_merging_twice_changes_nothing():
    mod = _load()
    once = mod.merge(LOCAL, POD)["text"]
    twice = mod.merge(once, POD)
    assert twice["status"] == "unchanged"
    assert twice["appended"] == 0
    assert twice["text"] == once


def test_identical_rows_are_counted_not_deduplicated():
    """Two genuine runs may write byte-identical rows; both must survive."""
    mod = _load()
    local = HEADER + "\nARM1,600,8,1.3,29.0\n"
    pod = HEADER + "\nARM1,600,8,1.3,29.0\nARM1,600,8,1.3,29.0\n"
    m = mod.merge(local, pod)
    assert m["appended"] == 1, m
    assert m["text"].count("ARM1,600,8,1.3,29.0") == 2
    # ...and merging again must NOT add a third copy of a row the pod only has
    # twice.
    again = mod.merge(m["text"], pod)
    assert again["appended"] == 0
    assert again["text"].count("ARM1,600,8,1.3,29.0") == 2


def test_local_rows_are_neither_dropped_nor_reordered():
    m = _load().merge(LOCAL, POD)
    lines = m["text"].splitlines()
    assert lines[0] == HEADER
    assert lines[1] == "ARM1,600,8,1.3,29.0"


def test_a_pod_that_knows_nothing_new_leaves_the_local_file_alone():
    m = _load().merge(LOCAL, HEADER + "\nARM1,600,8,1.3,29.0\n")
    assert m["status"] == "unchanged"
    assert m["text"] == LOCAL


def test_a_missing_local_ledger_is_created_from_the_pod():
    m = _load().merge("", POD)
    assert m["status"] == "created"
    assert m["text"] == POD


def test_an_empty_pod_ledger_is_not_an_empty_local_ledger():
    m = _load().merge(LOCAL, "")
    assert m["status"] == "no_pod_rows"
    assert m["text"] == LOCAL


def test_a_header_mismatch_refuses_rather_than_appending():
    other = "stage,seconds,gpus,cost" + "\nARM2,900,8,44.64\n"
    m = _load().merge(LOCAL, other)
    assert m["status"] == "header_mismatch"
    assert m["text"] == LOCAL, "a refused merge must not change the local file"
    assert m["appended"] == 0


def test_the_cli_refuses_with_a_nonzero_exit_and_keeps_the_file(tmp_path):
    local = tmp_path / "gpu_hours.csv"
    local.write_text(LOCAL)
    pod = tmp_path / "pod.csv"
    pod.write_text("junk,columns\n1,2\n")
    proc = subprocess.run(
        [sys.executable, str(MERGE), "--local", str(local), "--pod", str(pod)],
        capture_output=True, text=True)
    assert proc.returncode == 2, proc.stdout + proc.stderr
    assert json.loads(proc.stdout)["status"] == "header_mismatch"
    assert local.read_text() == LOCAL


def test_the_cli_merges_in_place_and_reports_what_it_did(tmp_path):
    local = tmp_path / "gpu_hours.csv"
    local.write_text(LOCAL)
    pod = tmp_path / "pod.csv"
    pod.write_text(POD)
    proc = subprocess.run(
        [sys.executable, str(MERGE), "--local", str(local), "--pod", str(pod)],
        capture_output=True, text=True)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert json.loads(proc.stdout) == {"appended": 1, "local_rows": 1,
                                       "status": "merged", "total_rows": 2}
    assert local.read_text() == POD

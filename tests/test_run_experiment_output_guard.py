"""The output-collision guard must not refuse a stage's own file.

The guard exists because the ARM4 driver pointed a stage named
`pg19_short_target` at `results/mlsys/pg19_multiseed.csv`, the filename of an
unrelated aggregate, and would have overwritten it.

The first version of it decided ownership by pattern-matching the stage id's
first token against the recorded level ids. That cannot be true for this
campaign -- the stage is `engine_cap_smoke` while its levels are
`CAPS_prefix64` -- and it was called from the WORKER, whose argv never carried
`--stage-id`, so the stage it compared was the literal string "unnamed". The
worker does not even write the CSV: it writes a per-run `.tmp` that the parent
reads and appends.

Consequence: the second row of every stage died with "subprocess produced no
output". Every measurement stage in the campaign. Found by the 1x
run_experiment probe on 2026-10-08, before it could cost another 8xA100 launch.

These tests hold both ends: the real campaign pair must be ALLOWED, and a
genuinely foreign file must still be REFUSED.
"""
from __future__ import annotations

import csv
import importlib.util
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
RUN_EXPERIMENT = REPO / "run_experiment.py"


def _load():
    spec = importlib.util.spec_from_file_location("runexp_guard",
                                                  RUN_EXPERIMENT)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["runexp_guard"] = mod
    try:
        spec.loader.exec_module(mod)
    except Exception as e:                      # noqa: BLE001
        pytest.skip(f"run_experiment not importable here: {e}")
    return mod


def _csv_with_levels(path: Path, levels) -> Path:
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["run_id", "group", "level_id",
                                          "seed", "status"])
        w.writeheader()
        for i, lv in enumerate(levels):
            w.writerow({"run_id": f"{lv}_s42", "group": "G", "level_id": lv,
                        "seed": 42, "status": "ok"})
    return path


# --------------------------------------------------------------------------
# the regression: a stage's own rows, appended by its own later rows
# --------------------------------------------------------------------------

def test_a_stage_may_append_to_a_file_holding_its_own_plans_rows(tmp_path):
    """Row 2 of every stage. This is what failed."""
    mod = _load()
    out = _csv_with_levels(tmp_path / "engine_cap_smoke.csv", ["CAPS_prefix64"])
    mod._guard_output_collision(out, "engine_cap_smoke",
                                planned_level_ids={"CAPS_prefix64"})
    # Reached here: no SystemExit.


def test_the_real_campaign_pair_is_allowed_despite_matching_no_name_pattern(tmp_path):
    """The exact pair that made the old heuristic refuse: the stage id shares no
    token with the level id."""
    mod = _load()
    assert "engine" not in "CAPS_prefix64"
    out = _csv_with_levels(tmp_path / "s.csv", ["CAPS_prefix64"])
    mod._guard_output_collision(out, "engine_cap_smoke",
                                planned_level_ids={"CAPS_prefix64"})


def test_the_canarys_own_row_does_not_trip_the_guard(tmp_path):
    """The canary appends to the same file before the grid runs."""
    mod = _load()
    out = _csv_with_levels(tmp_path / "s.csv",
                           ["probe_1x_canary_s42"])
    mod._guard_output_collision(
        out, "probe_1x",
        planned_level_ids={"probe_1x_canary_s42", "PROBE_ctx4k_spec"})


def test_a_previous_attempt_of_the_same_stage_is_allowed(tmp_path):
    """A retry re-runs the same levels into the same file."""
    mod = _load()
    out = _csv_with_levels(tmp_path / "s.csv", ["ARM2_a", "ARM2_b"])
    mod._guard_output_collision(out, "arm2",
                                planned_level_ids={"ARM2_a", "ARM2_b"})


# --------------------------------------------------------------------------
# the protection must survive
# --------------------------------------------------------------------------

def test_a_foreign_file_is_still_refused(tmp_path):
    """The ARM4 disaster: a stage pointed at an unrelated aggregate."""
    mod = _load()
    out = _csv_with_levels(tmp_path / "pg19_multiseed.csv",
                           ["pg19_1M_seed42", "pg19_1M_seed123"])
    with pytest.raises(SystemExit) as e:
        mod._guard_output_collision(out, "pg19_short_target",
                                    planned_level_ids={"pg19_short"})
    msg = str(e.value)
    assert "REFUSING to write" in msg
    assert "pg19_1M_seed42" in msg, "the message must name the foreign level"
    assert "pg19_short_target" in msg


def test_a_partly_foreign_file_is_refused(tmp_path):
    """One foreign level id is enough: the file is not this stage's."""
    mod = _load()
    out = _csv_with_levels(tmp_path / "s.csv", ["ARM2_a", "OTHER_b"])
    with pytest.raises(SystemExit):
        mod._guard_output_collision(out, "arm2",
                                planned_level_ids={"ARM2_a"})


def test_an_empty_plan_cannot_own_a_file_with_rows(tmp_path):
    mod = _load()
    out = _csv_with_levels(tmp_path / "s.csv", ["ARM2_a"])
    with pytest.raises(SystemExit):
        mod._guard_output_collision(out, "arm2", planned_level_ids=set())


def test_a_missing_file_is_not_checked(tmp_path):
    mod = _load()
    mod._guard_output_collision(tmp_path / "absent.csv", "s",
                                planned_level_ids={"a"})


def test_a_file_with_no_level_id_column_is_not_ours_to_check(tmp_path):
    mod = _load()
    out = tmp_path / "other.csv"
    out.write_text("a,b\n1,2\n")
    mod._guard_output_collision(out, "s", planned_level_ids=set())


def test_overwrite_stage_still_overrides(tmp_path):
    mod = _load()
    out = _csv_with_levels(tmp_path / "s.csv", ["foreign"])
    mod._guard_output_collision(out, "s", planned_level_ids={"mine"},
                                overwrite=True)


# --------------------------------------------------------------------------
# it must live in the process that writes the CSV
# --------------------------------------------------------------------------

def test_the_guard_runs_in_the_parent_and_not_in_the_worker():
    """The worker writes a per-run `.tmp`; the parent appends the CSV. Guarding
    in the worker was guarding a file it does not write, with a stage id it was
    never given."""
    src = RUN_EXPERIMENT.read_text()
    worker = src[src.index("if args._worker:"):]
    worker = worker[:worker.index("return", worker.index("_run_single_worker"))]
    assert "_guard_output_collision" not in worker, (
        "the worker guards again; it has no --stage-id and does not write the CSV")
    assert "planned_level_ids" in src, (
        "the parent does not compute the plan it owns the file by")
    assert "worker_args = [\"--_worker\"" in src
    # and the worker argv must not carry --stage-id, because it is not the
    # guard's business any more; if that changes, this test should be revisited.
    wa = src[src.index("worker_args = ["):]
    wa = wa[:wa.index("]")]
    assert "--stage-id" not in wa

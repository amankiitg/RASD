"""RoundProfiler's op tables and traces: what they write, and what they must NOT change.

The decode diagnosis needs per-op self time, which `key_averages()` already had and the
four-bucket summary discarded. These tests pin the three properties the campaign depends on:

1. With `ops_path`/`trace_path` unset, nothing new is written and the summary is what it
   always was -- the diagnosis must not alter a single number any other session reads.
2. With them set, a JSON with the measured field name, the per-op rows and both totals
   appears, plus a gzip file whose contents are the chrome trace (not a plain file wearing a
   .gz name).
3. The rank travels into the sidecar, because two ranks' tables are otherwise identical in
   shape and indistinguishable to a reader.
"""
from __future__ import annotations

import gzip
import json
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

torch = pytest.importorskip("torch")
from src.analysis.profiler import RoundProfiler  # noqa: E402


def _work():
    a = torch.randn(256, 256)
    for _ in range(3):
        a = a @ torch.randn(256, 256)
    return a


def test_ops_path_unset_writes_only_the_summary(tmp_path):
    out = tmp_path / "summary.json"
    with RoundProfiler(enabled=True, output_path=out) as prof:
        _work()
    assert out.exists(), "the pre-existing summary must still be written"
    assert prof.summary, "and must still be populated"
    assert not (tmp_path / "profiler_ops").exists()
    payload = json.loads(out.read_text())
    assert set(payload) == set(prof.summary), \
        "the summary schema must not gain or lose keys when the new options are unused"


def test_ops_and_trace_are_written_when_asked(tmp_path):
    ops = tmp_path / "profiler_ops" / "RUN.rank4.json"
    trace = tmp_path / "profiler_traces" / "RUN.rank4.json.gz"
    with RoundProfiler(enabled=True, rank=4, ops_path=ops, trace_path=trace) as prof:
        _work()
    assert prof.summary, "the summary is still produced alongside the tables"

    payload = json.loads(ops.read_text())
    assert payload["rank"] == 4
    assert payload["device_field"] in ("self_device_time_total", "self_cuda_time_total")
    assert payload["distinct_ops"] > 0
    # On a CPU-only dev machine the CUDA column is all zeros by construction; the CPU
    # column is the one that must carry rows, and `primary_table` says which was used.
    primary = payload["top_device"] if payload["primary_table"] == "device" else payload["top_cpu"]
    assert primary, "a table with no rows would prove nothing"
    for row in primary:
        assert set(row) >= {"op", "us", "s", "calls", "pct", "input_shapes"}
        assert row["calls"] >= 1
    # Percentages are shares of the measured column's total, so they can never exceed 100.
    assert sum(r["pct"] for r in primary) <= 100.01

    with gzip.open(trace, "rb") as fh:
        trace_payload = json.loads(fh.read().decode())
    assert "traceEvents" in trace_payload, "the .gz must contain a real chrome trace"


def test_record_shapes_is_optional_and_off_by_default(tmp_path):
    """record_shapes costs memory, so it is opt-in; the table must work either way."""
    ops = tmp_path / "ops.json"
    with RoundProfiler(enabled=True, ops_path=ops, record_shapes=False) as prof:
        _work()
    assert prof.summary
    payload = json.loads(ops.read_text())
    primary = payload["top_device"] if payload["primary_table"] == "device" else payload["top_cpu"]
    assert primary, "rows must exist without shapes too"

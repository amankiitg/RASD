"""Tests for src/analysis/profiler.py — the Fig 3 profiler wrapper.

CPU-only smoke tests; the same code runs on the pod with CUDA events
flowing in. Categorization is a pure substring match so it's identical
in both environments.
"""
from __future__ import annotations

import json
from pathlib import Path

import torch

from src.analysis.profiler import (
    COMM_PATTERNS,
    COMPUTE_PATTERNS,
    RoundProfiler,
    aggregate_events,
    categorize_event,
)


# ---------------------------------------------------------------------------
# categorize_event — pure substring rules
# ---------------------------------------------------------------------------

class TestCategorize:
    def test_compute_aten_matmul(self):
        assert categorize_event("aten::matmul") == "compute"

    def test_compute_aten_softmax(self):
        assert categorize_event("aten::softmax") == "compute"
        assert categorize_event("aten::_softmax") == "compute"

    def test_compute_aten_linear(self):
        assert categorize_event("aten::linear") == "compute"

    def test_compute_attention(self):
        assert categorize_event("aten::scaled_dot_product_attention") == "compute"

    def test_compute_layer_norm(self):
        assert categorize_event("aten::layer_norm") == "compute"
        assert categorize_event("aten::rms_norm") == "compute"

    def test_compute_silu_gelu(self):
        assert categorize_event("aten::silu") == "compute"
        assert categorize_event("aten::gelu") == "compute"

    def test_compute_embedding_rope(self):
        assert categorize_event("aten::embedding") == "compute"
        assert categorize_event("aten::rotary") == "compute"

    def test_comm_nccl(self):
        assert categorize_event("nccl:broadcast") == "comm"
        assert categorize_event("ncclKernel_AllReduce") == "comm"

    def test_comm_c10d(self):
        assert categorize_event("c10d::broadcast") == "comm"
        assert categorize_event("c10d::all_reduce") == "comm"

    def test_comm_isend_irecv(self):
        assert categorize_event("c10d::isend") == "comm"
        assert categorize_event("c10d::irecv") == "comm"
        assert categorize_event("aten::batch_isend_irecv") == "comm"

    def test_other_default(self):
        assert categorize_event("aten::randn") == "other"
        assert categorize_event("aten::empty") == "other"
        assert categorize_event("aten::contiguous") == "other"
        assert categorize_event("CudaStreamSynchronize") == "other"

    def test_lowercased_match(self):
        """Mixed-case event names should still classify correctly."""
        assert categorize_event("ATEN::MATMUL") == "compute"
        assert categorize_event("NCCL:Broadcast") == "comm"

    def test_comm_takes_precedence_over_compute(self):
        """If an event has both nccl + matmul-like substrings (rare —
        typically only in fused all-reduce-with-bias-grad kernels), it
        should still classify as comm. Order matters in the impl."""
        # Construct a synthetic event name that contains both
        ev = "ncclKernel_matmul_fused_allreduce"
        # Both substrings are present; comm patterns are checked first
        assert categorize_event(ev) == "comm"

    def test_pattern_lists_nonempty(self):
        """Defensive: future refactors must not empty the pattern lists."""
        assert len(COMPUTE_PATTERNS) >= 5
        assert len(COMM_PATTERNS) >= 5


# ---------------------------------------------------------------------------
# aggregate_events — bucket totals + percentages
# ---------------------------------------------------------------------------

class _StubEvent:
    """Minimal stand-in for torch.profiler FunctionEventAvg."""

    def __init__(self, key: str, cpu_us: float = 0.0, cuda_us: float = 0.0):
        self.key = key
        self.self_cpu_time_total = cpu_us
        self.self_cuda_time_total = cuda_us


class TestAggregate:
    def test_summary_keys(self):
        """Lock the schema Fig 3 reads from."""
        events = [_StubEvent("aten::matmul", cpu_us=100)]
        summary = aggregate_events(events, wall_time_us=200)
        expected = {
            "compute_us", "comm_us", "idle_us", "other_us",
            "total_us", "wall_us",
            "compute_pct", "comm_pct", "idle_pct", "other_pct",
        }
        assert set(summary.keys()) == expected

    def test_compute_only(self):
        events = [
            _StubEvent("aten::matmul",  cpu_us=100),
            _StubEvent("aten::softmax", cpu_us=50),
        ]
        s = aggregate_events(events, wall_time_us=200)
        assert s["compute_us"] == 150
        assert s["comm_us"] == 0
        assert s["other_us"] == 0
        assert s["idle_us"] == 50  # wall(200) - events(150)
        assert abs(s["compute_pct"] - 0.75) < 1e-9

    def test_mix_compute_comm_other(self):
        events = [
            _StubEvent("aten::matmul",      cpu_us=100),
            _StubEvent("c10d::broadcast",   cpu_us=20),
            _StubEvent("aten::contiguous",  cpu_us=10),
        ]
        s = aggregate_events(events, wall_time_us=200)
        assert s["compute_us"] == 100
        assert s["comm_us"]    == 20
        assert s["other_us"]   == 10
        assert s["idle_us"]    == 70

    def test_overlap_clamps_idle_to_zero(self):
        """When multi-stream overlap makes sum(events) > wall, idle = 0."""
        events = [
            _StubEvent("aten::matmul", cpu_us=300),  # > wall on purpose
        ]
        s = aggregate_events(events, wall_time_us=100)
        assert s["idle_us"] == 0

    def test_zero_wall_no_div_zero(self):
        events = [_StubEvent("aten::matmul", cpu_us=100)]
        s = aggregate_events(events, wall_time_us=0)
        # Should not raise; pcts should be sane
        assert s["compute_pct"] == 100  # falls back to denom=1
        assert s["wall_us"] == 0

    def test_cpu_plus_cuda_summed(self):
        """cpu + cuda time both contribute to bucket totals."""
        events = [_StubEvent("aten::matmul", cpu_us=10, cuda_us=90)]
        s = aggregate_events(events, wall_time_us=200)
        assert s["compute_us"] == 100  # 10 + 90


# ---------------------------------------------------------------------------
# RoundProfiler — context manager
# ---------------------------------------------------------------------------

class TestRoundProfilerDisabled:
    def test_disabled_default(self):
        prof = RoundProfiler()
        assert prof.enabled is False

    def test_disabled_no_summary(self):
        with RoundProfiler() as p:
            x = torch.matmul(torch.randn(8, 8), torch.randn(8, 8))
            assert p.summary is None
        assert p.summary is None

    def test_disabled_round_marker_is_noop(self):
        prof = RoundProfiler(enabled=False)
        with prof.round_marker("verify_round_0"):
            pass  # no exception, no torch.cuda calls

    def test_disabled_doesnt_create_output_file(self, tmp_path):
        out = tmp_path / "should_not_exist.json"
        with RoundProfiler(enabled=False, output_path=out):
            pass
        assert not out.exists()


class TestRoundProfilerEnabled:
    def test_enabled_collects_compute_events(self):
        with RoundProfiler(enabled=True) as p:
            for _ in range(3):
                a = torch.randn(64, 64)
                b = torch.randn(64, 64)
                _ = torch.matmul(a, b)
        assert p.summary is not None
        # We did real matmuls; compute bucket should be > 0
        assert p.summary["compute_us"] > 0
        # No NCCL on CPU; comm bucket should be 0
        assert p.summary["comm_us"] == 0

    def test_enabled_writes_json(self, tmp_path):
        out = tmp_path / "prof.json"
        with RoundProfiler(enabled=True, output_path=out) as p:
            _ = torch.matmul(torch.randn(32, 32), torch.randn(32, 32))
        assert out.exists()
        loaded = json.loads(out.read_text())
        assert loaded == p.summary
        assert loaded["wall_us"] > 0

    def test_summary_schema_locked(self):
        """Schema must match the Fig 3 contract — same keys as
        aggregate_events returns directly."""
        with RoundProfiler(enabled=True) as p:
            _ = torch.matmul(torch.randn(16, 16), torch.randn(16, 16))
        keys = set(p.summary.keys())
        assert keys == {
            "compute_us", "comm_us", "idle_us", "other_us",
            "total_us", "wall_us",
            "compute_pct", "comm_pct", "idle_pct", "other_pct",
        }

    def test_exception_in_body_does_not_leak_profiler(self):
        """If user code raises, the profiler context still exits cleanly."""
        prof = RoundProfiler(enabled=True)
        try:
            with prof:
                raise RuntimeError("intentional")
        except RuntimeError:
            pass
        # Internal handles released regardless
        assert prof._prof is None

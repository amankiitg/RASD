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

    def test_other_explicit_noise(self):
        """M4 Phase C 2026-05-10: categorize_event was inverted to
        default-to-compute (not default-to-other). Only explicitly
        listed SKIP_PATTERNS produce 'other'."""
        assert categorize_event("aten::empty")   == "other"
        assert categorize_event("aten::zeros")   == "other"
        assert categorize_event("cudaLaunchKernel") == "other"
        assert categorize_event("ProfilerStep")  == "other"

    def test_default_unknown_op_is_compute(self):
        """Unknown ops default to 'compute' (the codex fix — previously
        defaulted to 'other' and left 54% of wall in that bucket on
        the PROFILED_ctx128k_s42 run)."""
        assert categorize_event("aten::randn")    == "compute"
        assert categorize_event("aten::some_new_kernel") == "compute"

    def test_compute_includes_flash_attn_and_tensor_ops(self):
        """The expanded compute taxonomy covers flash-attn and tensor
        manipulation ops that p36's first run mis-bucketed as 'other'."""
        assert categorize_event("flash_attn::flash_attn_func") == "compute"
        assert categorize_event("_flash_attention_forward") == "compute"
        assert categorize_event("aten::contiguous") == "compute"
        assert categorize_event("aten::clone")      == "compute"
        assert categorize_event("aten::view")       == "compute"
        assert categorize_event("aten::permute")    == "compute"

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


class TestNVTXPhaseMarkers:
    """generate() must wrap each major phase with NVTX ranges so
    torch.profiler captures per-phase wall-time as user annotations.
    Codex review 2026-05-10: round_marker() was defined in
    profiler.py but never called from generate(). These tests guard
    against regressing that fix."""

    def _src(self):
        from pathlib import Path
        return (Path(__file__).resolve().parent.parent
                / "src" / "models" / "rasd_inference.py").read_text()

    def test_nvtx_helpers_defined(self):
        src = self._src()
        assert "def _nvtx_push" in src
        assert "def _nvtx_pop" in src

    def test_prefill_target_wrapped(self):
        src = self._src()
        assert 'phase:prefill_target' in src

    def test_prefill_draft_wrapped(self):
        src = self._src()
        assert 'phase:prefill_draft' in src

    def test_verify_rounds_wrapped(self):
        src = self._src()
        # f-string usage so we match the format
        assert 'phase:verify_round_' in src

    def test_target_only_steps_wrapped(self):
        src = self._src()
        assert 'phase:autoregressive_step_' in src


class TestAggregate:
    def test_summary_keys(self):
        """Lock the schema Fig 3 reads from."""
        events = [_StubEvent("aten::matmul", cpu_us=100)]
        summary = aggregate_events(events, wall_time_us=200)
        expected = {
            "compute_us", "comm_us", "idle_us", "other_us",
            "total_us", "wall_us",
            # M4 Phase C 2026-05-10 (codex review): overlap diagnostic
            # for multi-stream concurrency.
            "overlap_us",
            "compute_pct", "comm_pct", "idle_pct", "other_pct",
            "overlap_pct",
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
        # Use aten::empty (in SKIP_PATTERNS) for the "other" bucket since
        # the 2026-05-10 inversion moved aten::contiguous → "compute".
        events = [
            _StubEvent("aten::matmul",     cpu_us=100),
            _StubEvent("c10d::broadcast",  cpu_us=20),
            _StubEvent("aten::empty",      cpu_us=10),
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
        """When wall_time_us=0 (degenerate case), aggregate_events
        falls back to wall=1us and treats event time as 'all compute,
        massive overlap' rather than dividing by zero. M4 Phase C
        2026-05-10 (codex normalization fix): pcts are now bounded
        fractions in [0, 1], not raw ratios that can exceed 100.
        compute_pct here is 1.0 (100% of wall is compute) because
        the overlap normalization scales event time down to wall."""
        events = [_StubEvent("aten::matmul", cpu_us=100)]
        s = aggregate_events(events, wall_time_us=0)
        assert s["wall_us"] == 1.0  # defensive fallback
        # Overlap branch: 100 us of events vs 1 us wall.
        assert s["compute_pct"] == 1.0   # 100% of wall is compute
        assert s["idle_pct"] == 0.0       # no idle when overlap
        assert s["overlap_us"] > 0        # 100 - 1 = 99 us overlap
        # Sum of compute+comm+other+idle pct must equal exactly 1.0
        assert abs(s["compute_pct"] + s["comm_pct"] + s["other_pct"] + s["idle_pct"] - 1.0) < 1e-9

    def test_normalization_invariant_no_overlap(self):
        """When total_event_us <= wall_us, the four pcts must sum
        to exactly 1.0 (compute + comm + other + idle = 100% of wall).
        Closes the codex finding 'compute_pct + comm_pct + other_pct
        could exceed 100%'."""
        events = [
            _StubEvent("aten::matmul",     cpu_us=40),
            _StubEvent("c10d::broadcast",  cpu_us=20),
            _StubEvent("aten::empty",      cpu_us=10),
        ]
        s = aggregate_events(events, wall_time_us=200)
        # No overlap: events sum to 70, wall is 200, idle = 130
        assert s["overlap_us"] == 0
        assert abs(s["compute_pct"] + s["comm_pct"] + s["other_pct"] + s["idle_pct"] - 1.0) < 1e-9
        assert s["idle_us"] == 130

    def test_normalization_invariant_with_overlap(self):
        """When total_event_us > wall_us (multi-stream overlap),
        the four pcts must STILL sum to exactly 1.0 — idle goes to 0,
        compute/comm/other are scaled down, overlap_pct captures the
        excess. Closes the codex finding about > 100% sums in stacked
        bar plotting."""
        events = [
            _StubEvent("aten::matmul",      cpu_us=180),
            _StubEvent("nccl::all_reduce",  cpu_us=120),
        ]
        s = aggregate_events(events, wall_time_us=200)
        # 300us events overlap 200us wall
        assert s["overlap_us"] == 100
        assert s["idle_pct"] == 0.0
        # Sum of compute+comm+other+idle must equal 1.0
        assert abs(s["compute_pct"] + s["comm_pct"] + s["other_pct"] + s["idle_pct"] - 1.0) < 1e-9
        # Verify scaling preserved bucket ratios (compute:comm = 180:120 = 3:2)
        assert abs(s["compute_pct"] / s["comm_pct"] - 1.5) < 1e-9
        # overlap_pct = 100/200 = 0.5
        assert abs(s["overlap_pct"] - 0.5) < 1e-9

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
        aggregate_events returns directly. Updated 2026-05-10 to
        include overlap_us / overlap_pct from the codex
        normalization fix."""
        with RoundProfiler(enabled=True) as p:
            _ = torch.matmul(torch.randn(16, 16), torch.randn(16, 16))
        keys = set(p.summary.keys())
        assert keys == {
            "compute_us", "comm_us", "idle_us", "other_us",
            "total_us", "wall_us",
            "overlap_us",
            "compute_pct", "comm_pct", "idle_pct", "other_pct",
            "overlap_pct",
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

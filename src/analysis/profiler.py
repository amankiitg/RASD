"""torch.profiler wrapper for Fig 3 (mentor M4 deliverable).

Wraps a region of generate() with `torch.profiler.profile`, then bins
each captured event into one of three buckets — **compute**, **comm**,
**idle** — so the resulting JSON can drive the stacked-bar figure
described in the mentor M4 brief:

    Figure 3: A stacked bar chart showing the time breakdown
              (computation, communication, idle) for Ring Attention
              vs. RASD, demonstrating the effect of overlap.

Design contract:
- Default OFF. When `enabled=False` the context manager is a no-op
  and adds zero overhead; M3 replay byte-identical when not used.
- Categorization is a pure substring match on the event key
  (lowercased). Easy to test on CPU with synthetic ops; robust to
  minor torch version churn.
- Output schema (`RoundProfiler.summary`) is the contract Fig 3
  reads — locked by tests/test_profiler.py.

Wiring into `generate()` and the production pod run lives in M4
Phase 3.6 — this module just provides the primitive.
"""
from __future__ import annotations

import contextlib
import json
import time
from pathlib import Path
from typing import Dict, Iterable, Optional

# torch is installed in both the local dev env (.venv_analysis) and
# the pod env. Profiler is part of core torch since 1.8.
import torch
from torch.profiler import ProfilerActivity, profile

# ---------------------------------------------------------------------------
# Categorization
# ---------------------------------------------------------------------------

# Substrings that mark an event as model compute (GEMMs, attention,
# normalization, activations, embeddings, RoPE). Lowercase compare.
COMPUTE_PATTERNS: tuple[str, ...] = (
    "matmul", "::mm", "::bmm", "::addmm", "linear",
    "softmax",
    "scaled_dot_product",
    "layer_norm", "rmsnorm", "rms_norm",
    "silu", "gelu", "relu",
    "embedding", "rotary", "rope",
    "attention",
)

# Substrings that mark an event as cross-rank communication. NCCL and
# c10d primitives + the named collective ops we use under multi-rank
# (Fix2 broadcasts, ring batch_isend_irecv, etc.).
COMM_PATTERNS: tuple[str, ...] = (
    "nccl", "c10d",
    "all_reduce", "all_gather", "reduce_scatter",
    "broadcast",
    "::send", "::recv",
    "isend", "irecv",
    "batch_isend_irecv",
)


def categorize_event(name: str) -> str:
    """Return one of {"compute", "comm", "other"}.

    Pure function; covered by tests/test_profiler.py.
    """
    key = name.lower()
    if any(p in key for p in COMM_PATTERNS):
        return "comm"
    if any(p in key for p in COMPUTE_PATTERNS):
        return "compute"
    return "other"


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

def _event_time_us(e) -> float:
    """Sum of CPU + CUDA self-time for a profiler event, in microseconds.

    Events from torch.profiler.key_averages() expose `self_cpu_time_total`
    + `self_cuda_time_total` in microseconds. Older field names are also
    handled for forward-compat with the small range of torch versions
    we support locally vs on pod.
    """
    cpu = getattr(e, "self_cpu_time_total", None)
    if cpu is None:
        cpu = getattr(e, "cpu_time_total", 0.0)
    cuda = getattr(e, "self_cuda_time_total", None)
    if cuda is None:
        cuda = getattr(e, "cuda_time_total", 0.0)
    return float(cpu) + float(cuda)


def aggregate_events(events: Iterable, wall_time_us: float) -> Dict[str, float]:
    """Bin events into compute/comm/idle and return a stacked-bar-ready dict.

    Args:
        events       : iterable of FunctionEventAvg from prof.key_averages()
        wall_time_us : total wall-clock time spent in the profiler context,
                       in microseconds. Used to derive the idle bucket
                       (= wall - compute - comm), so wall must be > 0.

    Returns dict with keys: compute_us, comm_us, idle_us, other_us,
    total_us, wall_us — all in microseconds, plus a fraction view
    `compute_pct / comm_pct / idle_pct / other_pct` for direct plot use.
    """
    compute_us = 0.0
    comm_us    = 0.0
    other_us   = 0.0
    for e in events:
        t = _event_time_us(e)
        bucket = categorize_event(e.key)
        if bucket == "compute":
            compute_us += t
        elif bucket == "comm":
            comm_us += t
        else:
            other_us += t

    total_event_us = compute_us + comm_us + other_us
    # idle = wall - (sum of categorized event times). Clamped to 0
    # because under multi-stream concurrency, sum of self-times can
    # exceed wall (kernels overlap). When that happens we report
    # idle=0 and call the overlap "negative idle in our model" — the
    # caller can detect this by checking total_event_us > wall_us.
    idle_us = max(0.0, wall_time_us - total_event_us)

    summary = {
        "compute_us": compute_us,
        "comm_us":    comm_us,
        "idle_us":    idle_us,
        "other_us":   other_us,
        "total_us":   total_event_us,
        "wall_us":    wall_time_us,
    }
    # Fractions for convenience in figure code; protect against /0
    denom = wall_time_us if wall_time_us > 0 else 1.0
    summary.update({
        "compute_pct": compute_us / denom,
        "comm_pct":    comm_us    / denom,
        "idle_pct":    idle_us    / denom,
        "other_pct":   other_us   / denom,
    })
    return summary


# ---------------------------------------------------------------------------
# Context manager
# ---------------------------------------------------------------------------

class RoundProfiler:
    """Context-manager wrapper around torch.profiler.profile for Fig 3.

    Usage:
        with RoundProfiler(enabled=True, output_path="prof.json") as prof:
            engine.generate(...)
        # prof.summary now holds {compute_us, comm_us, idle_us, ...}

    When `enabled=False` (the default) this is a true no-op — no
    profiler started, no overhead added, summary stays None. This is
    important: M3 replay must be byte-identical when the flag is off.
    """

    def __init__(
        self,
        enabled: bool = False,
        output_path: Optional[str | Path] = None,
        activities: Optional[list[ProfilerActivity]] = None,
    ):
        self.enabled = enabled
        self.output_path = Path(output_path) if output_path else None
        self.summary: Optional[Dict[str, float]] = None
        self._prof: Optional[profile] = None
        self._t_start: Optional[float] = None
        if activities is None:
            # Sensible defaults: CPU always; CUDA only when available
            # (CPU-only test environment shouldn't pay the CUDA profiler
            # init cost for nothing).
            activities = [ProfilerActivity.CPU]
            if torch.cuda.is_available():
                activities.append(ProfilerActivity.CUDA)
        self._activities = activities

    def __enter__(self) -> "RoundProfiler":
        if not self.enabled:
            return self
        self._t_start = time.perf_counter()
        self._prof = profile(
            activities=self._activities,
            record_shapes=False,
            with_stack=False,
        )
        self._prof.__enter__()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if not self.enabled:
            return False
        try:
            self._prof.__exit__(exc_type, exc_val, exc_tb)
            wall_us = (time.perf_counter() - self._t_start) * 1e6
            self.summary = aggregate_events(
                self._prof.key_averages(), wall_us
            )
            if self.output_path is not None:
                self.output_path.parent.mkdir(parents=True, exist_ok=True)
                with self.output_path.open("w") as f:
                    json.dump(self.summary, f, indent=2)
        finally:
            self._prof = None
            self._t_start = None
        return False

    def round_marker(self, label: str):
        """NVTX range marker at a verify-loop boundary.

        Usage inside generate():
            with profiler.round_marker(f"verify_round_{n}"):
                target_out = self.target_model(...)

        No-op when CUDA NVTX is unavailable (CPU-only smoke tests, MPS).
        """
        if not self.enabled:
            return contextlib.nullcontext()
        if torch.cuda.is_available():
            try:
                return torch.cuda.nvtx.range(label)
            except AttributeError:
                # Older torch versions: range() context not exposed;
                # fall back to push/pop. We wrap into a tiny context.
                return _NVTXLegacy(label)
        return contextlib.nullcontext()


class _NVTXLegacy:
    """Fallback NVTX context for older torch versions without nvtx.range()."""

    def __init__(self, label: str):
        self._label = label

    def __enter__(self):
        torch.cuda.nvtx.range_push(self._label)
        return self

    def __exit__(self, *exc):
        torch.cuda.nvtx.range_pop()
        return False

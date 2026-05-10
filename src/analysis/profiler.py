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
    "p2p", "process_group",
)

# Substrings that mark an event as noise / non-compute overhead. These
# don't contribute meaningfully to "what is the GPU doing" — they're
# allocator, dispatch, profiler self-overhead, data loading, or sync
# waits that aren't really comm. Bucketed as "other".
SKIP_PATTERNS: tuple[str, ...] = (
    "cudalaunchkernel", "cudaeventrecord", "cudaeventquery",
    "cudahostalloc", "cudafree", "cudamemcpy",
    "::empty", "::zeros", "::ones",
    "profilerstep", "profiler_step",
    "dataloader",
)

# Optional explicit compute hints kept for backward-compat / docs.
# The categorization no longer DEPENDS on this list — anything that's
# not comm and not noise gets bucketed as compute. Listed here so
# readers know what we EXPECT to dominate compute.
COMPUTE_PATTERNS: tuple[str, ...] = (
    # GEMM / matmul / linear
    "matmul", "::mm", "::bmm", "::addmm", "linear",
    # Attention kernels
    "softmax",
    "scaled_dot_product", "sdpa",
    "flash_attn", "fa_fwd", "fa_bwd",
    "_flash_attention_forward",
    "attention",
    # Norms
    "layer_norm", "rmsnorm", "rms_norm", "native_layer_norm",
    # Activations
    "silu", "gelu", "relu", "swiglu", "tanh", "sigmoid",
    # Embedding / positional
    "embedding", "rotary", "rope",
    # Tensor manipulation that runs on GPU as part of model forward
    "::clone", "::contiguous", "::view", "::reshape",
    "::cat", "::stack", "::permute", "::transpose",
    "::pad", "::slice",
    # Elementwise math (narrow patterns to avoid spurious matches)
    "::add_", "::mul_", "::sub_", "::div_", "::pow_",
    "::add.", "::mul.", "::sub.", "::div.",
)


def categorize_event(name: str) -> str:
    """Return one of {"compute", "comm", "other"}.

    Logic inverted M4 Phase C 2026-05-10 (codex-flagged: the previous
    forward-match approach left ~54% of wall-clock in "other" because
    flash_attn, tensor-manipulation kernels, and generic aten::* ops
    weren't in COMPUTE_PATTERNS):

      1. If event matches a COMM_PATTERNS substring → "comm"
      2. Elif event matches a SKIP_PATTERNS substring → "other"
      3. Else → "compute"   (default; covers anything actually doing
                             math on tensors, even ops we didn't
                             explicitly list)

    For the Figure 3 stacked bar this is more honest: "compute" =
    "useful GPU work that's not comm", "comm" = "cross-rank moves",
    "other" = "explicitly identified noise / allocator / dispatch",
    "idle" = "wall − all of the above".

    Pure function; covered by tests/test_profiler.py.
    """
    key = name.lower()
    if any(p in key for p in COMM_PATTERNS):
        return "comm"
    if any(p in key for p in SKIP_PATTERNS):
        return "other"
    return "compute"


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

    Returns dict with the following keys (all _us values in microseconds):

      Raw event time per bucket (CPU + CUDA self-time, can exceed wall
      due to multi-stream concurrency):
        compute_us, comm_us, other_us, total_us, wall_us

      Figure-3-ready normalized fractions guaranteed to sum to 1.0
      exactly (so a stacked bar plots without gaps):
        compute_pct, comm_pct, other_pct, idle_pct

      Diagnostic for multi-stream overlap (>0 means async helped;
      this is the amount that compute+comm+other exceeded wall):
        overlap_us, overlap_pct

    Normalization fix (codex review 2026-05-10): the pre-fix code
    computed compute_pct = compute_us / wall_us etc., which under
    multi-stream concurrency could give sums > 1.0 (kernels overlap
    in real time but their summed self-time counts each). The fix:
    when total_event_us > wall_us, scale all three buckets down by
    wall_us/total_event_us so they sum to exactly 1.0 with idle=0
    and overlap_pct captures the excess; when total_event_us <
    wall_us, keep the raw fractions and assign the remainder to idle.
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

    if wall_time_us <= 0:
        wall_time_us = 1.0  # defensive

    overlap_us = max(0.0, total_event_us - wall_time_us)
    if total_event_us > wall_time_us:
        # Multi-stream concurrency: scale event buckets so they sum
        # to wall_us, idle=0, overlap_us captures the excess.
        scale = wall_time_us / total_event_us
        compute_pct = (compute_us * scale) / wall_time_us
        comm_pct    = (comm_us    * scale) / wall_time_us
        other_pct   = (other_us   * scale) / wall_time_us
        idle_pct    = 0.0
        idle_us     = 0.0
    else:
        # Standard case: idle = wall - total_event
        compute_pct = compute_us / wall_time_us
        comm_pct    = comm_us    / wall_time_us
        other_pct   = other_us   / wall_time_us
        idle_us     = wall_time_us - total_event_us
        idle_pct    = idle_us / wall_time_us

    return {
        "compute_us": compute_us,
        "comm_us":    comm_us,
        "idle_us":    idle_us,
        "other_us":   other_us,
        "total_us":   total_event_us,
        "wall_us":    wall_time_us,
        "overlap_us": overlap_us,
        # Figure-3-ready fractions sum to 1.0 exactly
        "compute_pct": compute_pct,
        "comm_pct":    comm_pct,
        "idle_pct":    idle_pct,
        "other_pct":   other_pct,
        "overlap_pct": overlap_us / wall_time_us,
    }


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

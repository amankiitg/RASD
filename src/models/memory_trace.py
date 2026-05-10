"""Per-rank GPU memory trace for paper Figure 3 / memory attribution.

Snapshots torch.cuda.memory_allocated and max_memory_allocated at
lifecycle points during generate(), so that

  delta(stage_i, stage_i+1) = memory cost added by stage_i+1

attributes the final peak to its components (weights, KV, activations,
verify-round working). Off by default; gated by RASDConfig.memory_trace.

Usage:
    tracer = MemoryTracer(rank=0, run_id="SMOKE_ctx128k_s42",
                          out_dir="results/m4_smoke/memory_trace/")
    tracer.snapshot("post_load")
    ...prefill...
    tracer.snapshot("post_prefill")
    ...verify rounds...
    tracer.snapshot("end")
    tracer.write()  # JSON sidecar at results/.../memory_trace/<run_id>.rank0.json

Why not torch.cuda.memory_summary(): that's human-readable text,
multi-KB per snapshot, not machine-parseable. We emit the smaller set
of fields that matter for attribution.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Dict, List, Optional

import torch


class MemoryTracer:
    """Lightweight per-rank GPU memory tracer.

    Each `snapshot(label)` call records:
      * allocated_mb   — current live tensors (torch.cuda.memory_allocated)
      * reserved_mb    — current reserved by allocator
      * max_alloc_mb   — peak allocated since last reset
      * max_reserved_mb — peak reserved since last reset

    On `write()`, dump the full sequence as JSON. Diff between
    consecutive snapshots = the stage's cost.

    INCREMENTAL FLUSH (M4 Phase C 2026-05-10 fix): when
    `flush_each_snapshot=True` (default), each snapshot() call
    immediately rewrites the JSON sidecar. This means an OOM mid-
    forward still leaves attribution data on disk — without this,
    the v4 1M OOM produced no JSON at all because write() was only
    called at end of generate(), never reached. The cost is ~1 ms
    per snapshot for a small JSON write; negligible vs the
    ~100ms-100s computation between snapshots.

    Cheap: each snapshot is 4 calls to torch.cuda; no synchronisation.
    Adds ~10us per snapshot. Safe to call dozens of times during a
    run.
    """

    def __init__(
        self,
        rank: int,
        run_id: str,
        out_dir: str | os.PathLike,
        device: torch.device | int = 0,
        reset_max_at_start: bool = True,
        flush_each_snapshot: bool = True,
    ):
        self.rank = rank
        self.run_id = run_id
        self.out_dir = Path(out_dir)
        self.device = (
            torch.device(device) if not isinstance(device, torch.device)
            else device
        )
        self.flush_each_snapshot = flush_each_snapshot
        self._snapshots: List[Dict] = []
        if reset_max_at_start and torch.cuda.is_available():
            # torch.cuda.reset_peak_memory_stats can raise on some
            # torch builds when the calling process hasn't initialized
            # CUDA yet (pytest workers in particular). Treat as
            # informational since it's just a counter reset.
            try:
                torch.cuda.reset_peak_memory_stats(self.device)
            except RuntimeError:
                pass

    def snapshot(self, label: str, **extra) -> None:
        """Record a labelled memory point. `extra` is merged in for
        run-specific tags (layer_idx, round_idx, etc).

        If self.flush_each_snapshot is True (default), immediately
        rewrites the JSON sidecar so an OOM after this snapshot still
        leaves attribution data on disk.
        """
        if not torch.cuda.is_available():
            return
        rec = {
            "label":           label,
            "allocated_mb":    torch.cuda.memory_allocated(self.device)     / (1024 ** 2),
            "reserved_mb":     torch.cuda.memory_reserved(self.device)      / (1024 ** 2),
            "max_alloc_mb":    torch.cuda.max_memory_allocated(self.device) / (1024 ** 2),
            "max_reserved_mb": torch.cuda.max_memory_reserved(self.device)  / (1024 ** 2),
        }
        rec.update(extra)
        self._snapshots.append(rec)
        if self.flush_each_snapshot:
            # Best-effort incremental flush. Wrap in try/except so a
            # filesystem hiccup doesn't kill the run.
            try:
                self.write()
            except Exception:
                pass

    def write(self) -> Optional[Path]:
        """Dump all snapshots to a JSON sidecar.

        Path: <out_dir>/<run_id>.rank<rank>.json. Returns the path on
        success, or None if there was nothing to write.
        """
        if not self._snapshots:
            return None
        self.out_dir.mkdir(parents=True, exist_ok=True)
        path = self.out_dir / f"{self.run_id}.rank{self.rank}.json"
        payload = {
            "run_id":    self.run_id,
            "rank":      self.rank,
            "device":    str(self.device),
            "snapshots": self._snapshots,
        }
        path.write_text(json.dumps(payload, indent=2))
        return path

    def attribution_summary(self) -> Dict[str, float]:
        """Return the diff between consecutive snapshots (in MB) keyed
        by f'{prev_label}->{curr_label}'. Useful for printing on rank 0
        at end of run."""
        out: Dict[str, float] = {}
        for i in range(1, len(self._snapshots)):
            prev = self._snapshots[i - 1]
            curr = self._snapshots[i]
            key = f"{prev['label']}->{curr['label']}"
            out[key] = curr["allocated_mb"] - prev["allocated_mb"]
        if self._snapshots:
            out["__final_max_alloc_mb"] = self._snapshots[-1]["max_alloc_mb"]
        return out

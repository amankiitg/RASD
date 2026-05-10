"""Unit tests for MemoryTracer (M4 paper memory attribution)."""
from __future__ import annotations

import json
from pathlib import Path

import pytest


def test_no_op_when_cuda_unavailable(tmp_path, monkeypatch):
    """On CPU-only test runners, snapshot() must return without error
    and write() must produce no file (nothing to attribute)."""
    import torch
    from src.models.memory_trace import MemoryTracer

    # Force the cuda-availability check inside snapshot() to take the
    # no-op path even on a GPU-equipped dev box (so this test is
    # deterministic in CI).
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    tracer = MemoryTracer(rank=0, run_id="t0", out_dir=tmp_path)
    tracer.snapshot("post_load")
    tracer.snapshot("post_prefill")
    out = tracer.write()
    assert out is None
    # No JSON file should be created
    assert not list(tmp_path.glob("*.json"))


def test_attribution_summary_diffs_consecutive_snapshots():
    """attribution_summary() returns delta between consecutive labels
    plus the final max_alloc."""
    from src.models.memory_trace import MemoryTracer

    tracer = MemoryTracer(rank=0, run_id="t1", out_dir="/tmp")
    # Bypass snapshot() (which requires CUDA) and inject directly
    tracer._snapshots = [
        {"label": "post_load",     "allocated_mb": 5000.0, "max_alloc_mb": 5000.0},
        {"label": "post_prefill",  "allocated_mb": 7000.0, "max_alloc_mb": 7100.0},
        {"label": "end",           "allocated_mb": 6800.0, "max_alloc_mb": 9200.0},
    ]
    summary = tracer.attribution_summary()
    assert summary["post_load->post_prefill"] == pytest.approx(2000.0)
    assert summary["post_prefill->end"] == pytest.approx(-200.0)
    assert summary["__final_max_alloc_mb"] == pytest.approx(9200.0)


def test_write_emits_json_with_expected_schema(tmp_path):
    """When snapshots exist, write() emits a JSON sidecar with the
    documented shape: {run_id, rank, device, snapshots: [...]}."""
    from src.models.memory_trace import MemoryTracer

    tracer = MemoryTracer(rank=3, run_id="SMOKE_ctx128k_s42", out_dir=tmp_path)
    # Bypass snapshot() (CUDA-required) and inject directly
    tracer._snapshots = [
        {"label": "post_load",
         "allocated_mb": 5000.0, "reserved_mb": 5500.0,
         "max_alloc_mb": 5000.0, "max_reserved_mb": 5500.0},
        {"label": "end",
         "allocated_mb": 8000.0, "reserved_mb": 9000.0,
         "max_alloc_mb": 9100.0, "max_reserved_mb": 10000.0,
         "n_rounds": 17},
    ]
    out = tracer.write()
    assert out is not None
    assert out.name == "SMOKE_ctx128k_s42.rank3.json"

    payload = json.loads(out.read_text())
    assert payload["run_id"] == "SMOKE_ctx128k_s42"
    assert payload["rank"] == 3
    assert len(payload["snapshots"]) == 2
    # Extra kwargs survive the round-trip
    assert payload["snapshots"][1]["n_rounds"] == 17


def test_rasd_config_default_is_off():
    """RASDConfig.memory_trace must default to False so existing
    matrix runs (M3 replay) are byte-identical."""
    from src.models.rasd_inference import RASDConfig
    cfg = RASDConfig(
        target_model_name="x", draft_model_name="y",
        spec_steps=4, kv_block_size=2048, prefetch_depth=1,
        seed=42,
    )
    assert cfg.memory_trace is False
    assert cfg.memory_trace_dir is None


def test_run_experiment_propagates_memory_trace_flag():
    """run_experiment.py must propagate --memory-trace onto each run
    dict so the worker subprocess picks it up."""
    import re
    src = Path(__file__).resolve().parent.parent / "run_experiment.py"
    text = src.read_text()
    # CLI flag exists
    assert '--memory-trace' in text
    # Propagation onto run dict exists
    assert re.search(
        r'if args\.memory_trace:[\s\S]{0,200}r\[[\"\']memory_trace[\"\']\]\s*=\s*True',
        text,
    ), "args.memory_trace should set run['memory_trace']=True for each run"
    # Worker passes it through to RASDConfig
    assert re.search(
        r'memory_trace\s*=\s*bool\(run\.get\([\"\']memory_trace[\"\']',
        text,
    ), "RASDConfig kwarg memory_trace not wired in worker"

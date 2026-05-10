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


def test_flush_each_snapshot_writes_after_every_snapshot(tmp_path, monkeypatch):
    """When flush_each_snapshot=True (default), the JSON sidecar must
    be rewritten after EVERY snapshot. This is the OOM-survival
    property: an OOM after snapshot N still leaves snapshots [0..N]
    on disk.

    Caught by the M4 v4 1M OOM: the previous design only flushed at
    end of generate(); when OOM killed the worker mid-prefill, no
    JSON was ever written and we had no attribution data.
    """
    import torch
    from src.models.memory_trace import MemoryTracer

    # Stub the cuda counters so this runs on CPU-only test machines too.
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "memory_allocated", lambda d=None: 1234 * 1024 ** 2)
    monkeypatch.setattr(torch.cuda, "memory_reserved",  lambda d=None: 2345 * 1024 ** 2)
    monkeypatch.setattr(torch.cuda, "max_memory_allocated", lambda d=None: 3456 * 1024 ** 2)
    monkeypatch.setattr(torch.cuda, "max_memory_reserved",  lambda d=None: 4567 * 1024 ** 2)
    monkeypatch.setattr(torch.cuda, "reset_peak_memory_stats", lambda d=None: None)

    tracer = MemoryTracer(rank=0, run_id="t_flush", out_dir=tmp_path)
    expected = tmp_path / "t_flush.rank0.json"

    tracer.snapshot("a")
    assert expected.exists(), "snapshot should flush JSON immediately"
    payload_after_a = json.loads(expected.read_text())
    assert len(payload_after_a["snapshots"]) == 1

    tracer.snapshot("b")
    payload_after_b = json.loads(expected.read_text())
    assert len(payload_after_b["snapshots"]) == 2

    tracer.snapshot("c")
    payload_after_c = json.loads(expected.read_text())
    assert len(payload_after_c["snapshots"]) == 3
    assert [s["label"] for s in payload_after_c["snapshots"]] == ["a", "b", "c"]


def test_flush_disabled_writes_only_at_end(tmp_path, monkeypatch):
    """When flush_each_snapshot=False, snapshot() must NOT write the
    JSON; only write() does. This is the legacy / batched-write mode
    in case anyone needs lower I/O for trivially short runs."""
    import torch
    from src.models.memory_trace import MemoryTracer

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "memory_allocated", lambda d=None: 1024 ** 2)
    monkeypatch.setattr(torch.cuda, "memory_reserved",  lambda d=None: 1024 ** 2)
    monkeypatch.setattr(torch.cuda, "max_memory_allocated", lambda d=None: 1024 ** 2)
    monkeypatch.setattr(torch.cuda, "max_memory_reserved",  lambda d=None: 1024 ** 2)
    monkeypatch.setattr(torch.cuda, "reset_peak_memory_stats", lambda d=None: None)

    tracer = MemoryTracer(
        rank=0, run_id="t_no_flush", out_dir=tmp_path,
        flush_each_snapshot=False,
    )
    expected = tmp_path / "t_no_flush.rank0.json"
    tracer.snapshot("a")
    tracer.snapshot("b")
    assert not expected.exists(), "flush_each_snapshot=False shouldn't write"
    tracer.write()
    assert expected.exists()
    assert len(json.loads(expected.read_text())["snapshots"]) == 2


def test_per_layer_prefill_hooks_registered_under_memory_trace():
    """rasd_inference.generate() must register per-LlamaDecoderLayer
    forward hooks ONLY when mem_tracer is active, and remove them
    before any decode/verify forward. Without this, the OOM
    attribution data wouldn't surface which layer pushed past the
    memory limit (caught by v4 1M OOM giving zero traces)."""
    import re
    src = Path(__file__).resolve().parent.parent / "src" / "models" / "rasd_inference.py"
    text = src.read_text()
    # Hook registration is gated on mem_tracer being non-None
    assert re.search(
        r"if\s+mem_tracer\s+is\s+not\s+None:[\s\S]{0,800}register_forward_hook",
        text,
    ), "Per-layer hook registration must be gated on mem_tracer is not None"
    # Hooks are removed after prefill
    assert "h.remove()" in text, "Hooks must be removed before decode loop"


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

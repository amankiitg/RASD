"""A rerun row must be able to prove which ENGINE produced it.

The Part 3 rerun compares two engines rather than two settings of one engine, so the
provenance columns are load-bearing. Two failure modes are worth a test each:

  * a knob recorded from the CONFIG rather than from the engine. `install_ring_attention`
    sets these attributes from the config, but if a key stops being wired through, the
    engine keeps its default while the config still claims the new value, and the row looks
    correct. `_ring_engine_report` reads the patched modules instead, and is tested here
    against a stub engine whose modules are set by hand.
  * a per-level YAML key that does not survive the merge into a run dict. This campaign has
    already lost time to one of those (the `seeds` overwrite), so the merge is asserted
    rather than assumed, and `RASDConfig` is asserted to accept the keys the runner passes
    to it -- a config key with no matching field is a TypeError at load time, on the pod,
    after the model download.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

pytest.importorskip("torch")
run_experiment = pytest.importorskip("run_experiment")


class _StubAttn:
    """Stands in for a patched LlamaAttention: only the ring attributes matter."""


class _StubLayer:
    def __init__(self):
        self.self_attn = _StubAttn()


class _StubEngine:
    def __init__(self, n_layers=3):
        self.target_model = type("M", (), {})()
        self.target_model.model = type("Inner", (), {})()
        self.target_model.model.layers = [_StubLayer() for _ in range(n_layers)]


def _patch(engine, chunk=32768, mode="split_kv", expand=True, only=None):
    for i, layer in enumerate(engine.target_model.model.layers):
        if only is not None and i not in only:
            continue
        a = layer.self_attn
        a._ring_chunk_size = chunk
        a._ring_decode_mode = mode
        a._ring_expand_after_rotation = expand


def test_engine_report_reads_the_live_modules():
    e = _StubEngine(3)
    _patch(e)
    got = run_experiment._ring_engine_report(e)
    assert got["ring_decode_mode"] == "split_kv"
    assert got["ring_chunk_size"] == 32768
    assert got["ring_expand_after_rotation"] is True
    assert got["ring_layers_patched"] == 3


def test_engine_report_marks_an_unpatched_engine_as_zero_layers():
    """world_size<=1 makes `install_ring_attention` a no-op. That has to be VISIBLE in the
    row: two identical engines compared as if they differed is the worst outcome."""
    got = run_experiment._ring_engine_report(_StubEngine(3))
    assert got["ring_layers_patched"] == 0, "an unpatched engine must report 0 layers"
    assert got["ring_decode_mode"] == ""


def test_engine_report_flags_a_half_applied_arm():
    """A module left at the old value must show up as MIXED, not be averaged into a claim."""
    e = _StubEngine(3)
    _patch(e)
    e.target_model.model.layers[1].self_attn._ring_decode_mode = "ring"
    got = run_experiment._ring_engine_report(e)
    assert got["ring_decode_mode"].startswith("MIXED(")
    assert "ring" in got["ring_decode_mode"] and "split_kv" in got["ring_decode_mode"]


def test_engine_provenance_columns_are_in_the_csv_contract():
    for col in ("ring_decode_mode", "ring_chunk_size", "ring_expand_after_rotation",
                "ring_layers_patched", "git_commit", "nccl_env"):
        assert col in run_experiment.CSV_FIELDS, f"{col} missing from CSV_FIELDS"


def test_per_level_ring_keys_survive_the_merge():
    cfg = {
        "defaults": {"seeds": [42], "kv_block_size": 2048, "temperature": 0.7,
                     "ring_decode_mode": "ring", "ring_expand_after_rotation": False},
        "STAGE": {"name": "stage", "factor": "context_length", "levels": [
            {"id": "STAGE_A", "context_length": 131072, "documents": ["pg19_train_0"],
             "ring_decode_mode": "split_kv", "ring_expand_after_rotation": True,
             "kv_block_size": 32768},
        ]},
    }
    runs = run_experiment.build_run_configs(cfg, None, debug=False)
    assert len(runs) == 1
    r = runs[0]
    assert r["ring_decode_mode"] == "split_kv", "a per-level key must beat the default"
    assert r["ring_expand_after_rotation"] is True
    assert r["kv_block_size"] == 32768
    # The default must NOT be mutated for stages that do not override it.
    assert cfg["defaults"]["ring_decode_mode"] == "ring"


def test_rasdconfig_accepts_the_ring_keys_the_runner_passes():
    """A config key with no matching RASDConfig field fails only on the pod, after the
    download. This is the cheap place to find that out."""
    from src.models.rasd_inference import RASDConfig
    fields = getattr(RASDConfig, "__dataclass_fields__", {})
    for key in ("ring_decode_mode", "ring_expand_after_rotation", "kv_block_size",
                "prefetch_depth", "kv_quant", "quantize_target"):
        assert key in fields, f"RASDConfig has no field '{key}'"
    c = RASDConfig(ring_decode_mode="split_kv", ring_expand_after_rotation=True)
    assert c.ring_decode_mode == "split_kv" and c.ring_expand_after_rotation is True
    assert RASDConfig().ring_expand_after_rotation is False, "the default must stay off"


def test_nccl_env_report_lists_only_what_is_set(monkeypatch):
    """The GENERAL knobs are recorded only when set. The wait-mode block is the exception:
    it is always emitted, because "unset" is a value there (it selects the async arm) and a
    reader must be able to tell "unset" from "not looked at"."""
    for k in run_experiment._NCCL_KEYS + run_experiment._NCCL_WAIT_KEYS:
        monkeypatch.delenv(k, raising=False)
    got = run_experiment._nccl_env_report()
    expected = "wait_mode=none;" + ";".join(
        f"{k}=<unset>" for k in run_experiment._NCCL_WAIT_KEYS)
    assert got == expected, got
    monkeypatch.setenv("NCCL_P2P_LEVEL", "NVL")
    monkeypatch.setenv("NCCL_DEBUG", "INFO")
    got = run_experiment._nccl_env_report()
    assert "NCCL_P2P_LEVEL=NVL" in got and "NCCL_DEBUG=INFO" in got
    assert "NCCL_SHM_DISABLE" not in got, "unset knobs must not appear"


def test_the_wait_mode_is_always_recorded_including_when_unset(monkeypatch):
    """Stop-condition parity with the engine provenance rule: a row that cannot say which
    wait-mode arm produced it is void, so the mode is derived from the environment the process
    actually has and appears in `nccl_env` on every row.

    The two arms are distinguished by which of the two knobs is SET, and blocking wins if both
    are, because that is what ProcessGroupNCCL does (it reads BLOCKING_WAIT first and then
    disables async handling)."""
    for k in run_experiment._NCCL_KEYS + run_experiment._NCCL_WAIT_KEYS:
        monkeypatch.delenv(k, raising=False)
    assert run_experiment._nccl_wait_mode() == "none"
    assert "TORCH_NCCL_BLOCKING_WAIT=<unset>" in run_experiment._nccl_env_report()

    monkeypatch.setenv("TORCH_NCCL_BLOCKING_WAIT", "1")
    assert run_experiment._nccl_wait_mode() == "blocking"
    assert "wait_mode=blocking" in run_experiment._nccl_env_report()
    assert "TORCH_NCCL_BLOCKING_WAIT=1" in run_experiment._nccl_env_report()

    monkeypatch.delenv("TORCH_NCCL_BLOCKING_WAIT")
    monkeypatch.setenv("TORCH_NCCL_ASYNC_ERROR_HANDLING", "1")
    assert run_experiment._nccl_wait_mode() == "async"
    got = run_experiment._nccl_env_report()
    assert "wait_mode=async" in got and "TORCH_NCCL_ASYNC_ERROR_HANDLING=1" in got
    assert "TORCH_NCCL_BLOCKING_WAIT=<unset>" in got, "the async arm must say blocking is unset"

    # Both set: ProcessGroupNCCL uses blocking and turns async handling off, so the row must
    # say so rather than leaving a reader to assume the async arm took effect.
    monkeypatch.setenv("TORCH_NCCL_BLOCKING_WAIT", "1")
    assert run_experiment._nccl_wait_mode() == "blocking"

    # `0` is not a mode: it means "not this arm".
    monkeypatch.setenv("TORCH_NCCL_BLOCKING_WAIT", "0")
    monkeypatch.setenv("TORCH_NCCL_ASYNC_ERROR_HANDLING", "0")
    assert run_experiment._nccl_wait_mode() == "none"


def test_code_commit_wins_over_a_pod_side_rev_parse(monkeypatch):
    """The pod has no .git, so `git rev-parse HEAD` there returns nothing and every row
    recorded `git_commit: unknown` -- which voids a rerun whose whole purpose is to compare two
    engines (stop condition S2). The DRIVER stages the tree, so the driver names the commit,
    and the runner must prefer it."""
    import run_experiment as R
    monkeypatch.setattr(R, "_GIT_COMMIT_CACHE", None)
    monkeypatch.setenv("CODE_COMMIT", "deadbeefcafe1234")
    assert R._git_commit() == "deadbeefcafe1234"
    # ... and it is cached, so a per-row call does not shell out 40 times.
    monkeypatch.delenv("CODE_COMMIT", raising=False)
    assert R._git_commit() == "deadbeefcafe1234"


def test_git_commit_falls_back_when_the_variable_is_absent(monkeypatch):
    import run_experiment as R
    monkeypatch.setattr(R, "_GIT_COMMIT_CACHE", None)
    monkeypatch.delenv("CODE_COMMIT", raising=False)
    got = R._git_commit()
    assert got and got != "unknown", "in the repo the local rev-parse must still work"


def test_an_empty_code_commit_does_not_shadow_the_fallback(monkeypatch):
    import run_experiment as R
    monkeypatch.setattr(R, "_GIT_COMMIT_CACHE", None)
    monkeypatch.setenv("CODE_COMMIT", "   ")
    assert R._git_commit() != "   "

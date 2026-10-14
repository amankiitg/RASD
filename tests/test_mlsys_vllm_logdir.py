"""The vLLM stages must survive a log directory that does not exist yet.

MEASURED FAILURE, 2026-10-10T11:32:02Z. `impl_validation` ran for THREE SECONDS and
failed with rc=1:

    FileNotFoundError: [Errno 2] No such file or directory:
      '/home/ubuntu/RASD/results/mlsys/logs/vllm_meta-llama_Llama-3.1-8B_bfloat16_attempt1.spec.json'

`run_attempt` wrote the spec JSON to LOGDIR before creating LOGDIR: the
`spec_path.write_text(...)` call sat above the `log_path.parent.mkdir(...)` that
would have made it valid. On a fresh pod results/mlsys/logs/ does not exist, so
both vLLM stages -- impl_validation and vllm_ladder -- died before starting a
worker, which is why neither had EVER produced a row in any arming. The stage's own
`write_rows` created the CSV's directory and `LOGDIR` had no creator at all.

`tokens_slim` note: these tests do not need vLLM installed. `run_attempt` spawns
`sys.executable -m` the worker script, so what is asserted here is the PARENT's file
handling with a command that exits immediately.
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
VLLM = REPO / "scripts" / "mlsys_vllm_baseline.py"


def _load():
    spec = importlib.util.spec_from_file_location("_vllm_logdir_under_test", VLLM)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_run_attempt_creates_a_missing_log_directory(tmp_path, monkeypatch):
    """The regression itself: a LOGDIR that does not exist must be created."""
    vllm = _load()
    # A path whose parent chain does not exist yet, mirroring results/mlsys/logs/
    # on a fresh pod.
    log_path = tmp_path / "results" / "mlsys" / "logs" / "vllm_m_bfloat16_attempt1.log"
    assert not log_path.parent.exists(), "the fixture must start with no directory"

    # Replace the actual vLLM worker with a no-op that writes a result file, so the
    # test exercises the PARENT's file handling rather than needing vLLM.
    real_popen = vllm.subprocess.Popen

    class FakeProc:
        returncode = 0
        def __init__(self, cmd, **kw):
            self.stdout = iter([])
            result = Path(cmd[-1]).with_suffix(".result.json")
            result.write_text(json.dumps({"status": "ok"}))
        def kill(self): pass
        def wait(self, timeout=None): return 0

    monkeypatch.setattr(vllm.subprocess, "Popen", FakeProc)
    rc, log = vllm.run_attempt({"model": "m", "env": {}, "context_length": 8},
                               log_path, timeout_s=30)

    assert log_path.parent.exists(), "run_attempt did not create its log directory"
    assert log_path.exists(), "the attempt log was not written"
    assert (log_path.with_suffix(".spec.json")).exists(), \
        "the spec file was not written, so the worker could not have read it"


def test_the_spec_write_sits_after_the_mkdir_in_the_source():
    """Guards the ORDER, not just the effect: a later edit could restore the bug
    and still pass the test above if something else created the directory."""
    src = VLLM.read_text()
    body = src[src.index("def run_attempt("):]
    body = body[:body.index("\ndef ", 1)]
    i_mkdir = body.index("parent.mkdir(")
    i_spec = body.index("spec_path.write_text(")
    assert i_mkdir < i_spec, (
        "the log directory is created AFTER the spec is written; on a fresh pod "
        "that is a FileNotFoundError three seconds into the stage")

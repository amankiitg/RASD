"""A wandb failure must never kill a stage that has no wandb dependency.

On 2026-10-08 `engine_cap_smoke` failed 24 seconds in having done no GPU work at
all:

    run_experiment.py:1673 main -> :993 _run_single_worker -> :523 init_wandb
    wandb.errors.errors.UsageError: No API key configured. Use `wandb login`.
    -> torch.distributed.elastic...ChildFailedError

The pod was running transformers 4.46.3, torch 2.5.1+cu124 and eight A100s; the
only thing missing was a W&B credential. `init_wandb` caught `ImportError` — the
"wandb is not installed" case — and that was enough while the pod had no wandb at
all. An INSTALLED wandb with no key raises `UsageError`, which is not an
`ImportError`, so it propagated out of rank 0 and torchrun labelled the stage
failed. Six of the nine approved stages run `run_experiment`, so all six would
have failed identically; the two gates and the vLLM ladder do not use it.

These tests hold the line: whatever wandb does, `init_wandb` and `log_wandb`
return or warn, and the measurement continues.
"""
from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
RUN_EXPERIMENT = REPO / "run_experiment.py"


def _load():
    spec = importlib.util.spec_from_file_location("runexp_wandb",
                                                  RUN_EXPERIMENT)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["runexp_wandb"] = mod
    try:
        spec.loader.exec_module(mod)
    except Exception as e:                      # noqa: BLE001
        pytest.skip(f"run_experiment not importable here: {e}")
    return mod


def _fake_wandb(init_error=None):
    """A wandb stand-in with the REAL nesting the traceback showed.

    `wandb.errors.errors.UsageError` — not a bare `Exception` — so a test that
    passes here is a statement about the exception the pod actually raised.
    """
    errors_errors = types.ModuleType("wandb.errors.errors")

    class UsageError(Exception):
        pass

    errors_errors.UsageError = UsageError
    errors = types.ModuleType("wandb.errors")
    errors.errors = errors_errors
    wandb = types.ModuleType("wandb")
    wandb.errors = errors

    def init(**_kw):
        if init_error is not None:
            raise init_error
        return types.SimpleNamespace(name="WB_RUN")

    wandb.init = init
    return wandb, UsageError


@pytest.fixture
def captured(monkeypatch):
    """Collect the warnings instead of printing them."""
    seen = []

    def fake_warning(msg, *a):
        seen.append(msg % a if a else msg)

    mod = _load()
    monkeypatch.setattr(mod.log, "warning", fake_warning)
    return mod, seen


def test_an_installed_wandb_without_a_key_does_not_raise(captured, monkeypatch):
    """The exact failure: UsageError out of wandb.init."""
    mod, seen = captured
    wandb, UsageError = _fake_wandb()

    def raise_usage_error(**_kw):
        raise UsageError("No API key configured. "
                         "Use `wandb login` to log in.")

    wandb.init = raise_usage_error
    monkeypatch.setitem(sys.modules, "wandb", wandb)

    assert mod.init_wandb({"run_id": "r1"}, "proj") is None
    assert any("wandb init failed" in s and "UsageError" in s for s in seen), \
        f"the reason was not recorded: {seen}"
    assert any("No API key configured" in s for s in seen), \
        "the message the operator needs to see was dropped"


def test_a_missing_wandb_is_still_tolerated(captured, monkeypatch):
    mod, seen = captured
    monkeypatch.setitem(sys.modules, "wandb", None)   # makes `import wandb` fail
    assert mod.init_wandb({"run_id": "r1"}, "proj") is None
    assert any("not installed" in s for s in seen)


@pytest.mark.parametrize("exc", [OSError("network unreachable"),
                                 RuntimeError("unexpected protocol version"),
                                 ValueError("config not serialisable")])
def test_no_wandb_failure_mode_propagates(captured, monkeypatch, exc):
    """A logging side-channel has open-ended failure modes and none of them is
    a fact about the model."""
    mod, _ = captured
    wandb, _U = _fake_wandb(exc)
    monkeypatch.setitem(sys.modules, "wandb", wandb)
    assert mod.init_wandb({"run_id": "r1"}, "proj") is None


def test_a_working_wandb_still_returns_its_run(captured, monkeypatch):
    """The hardening must not disable logging when logging works."""
    mod, _ = captured
    wandb, _U = _fake_wandb()
    monkeypatch.setitem(sys.modules, "wandb", wandb)
    got = mod.init_wandb({"run_id": "r1"}, "proj")
    assert got is not None and got.name == "WB_RUN"


def test_a_failing_wandb_log_does_not_discard_the_row(captured, monkeypatch):
    """log_wandb runs after the metrics exist. A raise here would throw away a
    run that has already been measured."""
    mod, seen = captured
    wandb, _U = _fake_wandb()
    monkeypatch.setitem(sys.modules, "wandb", wandb)

    class ExplodingRun:
        def log(self, _m):
            raise RuntimeError("wandb buffer drop")

        def finish(self, **_kw):
            raise RuntimeError("wandb buffer drop")

    mod.log_wandb(ExplodingRun(), {"acceptance_rate": 0.5})   # must not raise
    assert any("wandb log failed" in s for s in seen), seen


def test_log_wandb_is_a_noop_without_a_run(captured):
    mod, _ = captured
    assert mod.log_wandb(None, {"a": 1}) is None


def test_the_error_paths_finish_call_cannot_mask_the_real_error():
    """Structural, because reaching it needs a distributed failure: the
    `wb_run.finish(exit_code=1)` at the end of the worker's `except` block must
    be wrapped, or a wandb failure there replaces the original exception AND
    skips the CSV row written just below."""
    src = RUN_EXPERIMENT.read_text()
    i = src.index("wb_run.finish(exit_code=1)")
    window = src[i - 260:i]
    assert "try:" in window, \
        "the error-path finish() is unwrapped; a wandb failure would replace " \
        "the exception being handled"


def test_the_pod_env_file_disables_wandb_before_anything_runs():
    """The second layer. run_experiment now degrades a wandb failure to a
    warning, and this says the same thing one layer earlier so the pod never
    tries to reach W&B at all. Asserted on the writer, because the file itself
    is only ever produced on the pod.

    (scripts/mlsys_dry_run.sh asserts the same line as part of its provisioning
    contract; this keeps the reason next to the wandb reasoning.)"""
    src = (REPO / "scripts" / "mlsys_pod_env.sh").read_text()
    assert "WANDB_MODE=disabled" in src, (
        "the pod env file no longer disables wandb; a pod with no W&B "
        "credential would fail every run_experiment stage")
    # It has to be EXPORTED into the file the stage sources, not merely set in
    # the provisioning shell, or the manifest's python never sees it.
    assert 'echo "export WANDB_MODE=disabled"' in src

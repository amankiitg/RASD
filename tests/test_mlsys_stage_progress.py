"""Stages whose manifest.log goes quiet for longer than the liveness rule.

The byte-liveness rule stops a stage after `liveness_minutes` (15) of a
`manifest.log` that does not grow, and it cannot tell a working stage from a hung
one without output. Two stages could have been silent for longer than that with
nothing wrong at all:

  * `coherence_gate` built ALL of its rows before printing anything. Its seven
    rope candidates are one 128k, four 256k and two 512k, each of which loads a
    model, forwards the prompt and generates 200 tokens on ONE GPU
    (`_device_map()` is {"": 0}), so the stage could sit for 25-40 minutes with
    an empty log.
  * `vllm_ladder` prints per cell and per attempt, but a single attempt runs the
    worker as a SUBPROCESS whose output is teed to the attempt's own log file.
    From the manifest's point of view the attempt is silent from `attempt N` to
    its outcome, and it may legitimately run for MAX_ATTEMPT_WALL_S = 20 minutes
    before the timer kills it.

Both are now covered by host-side progress lines. Host-side is not decoration:
these run inside the measurement, so nothing here may touch the device -- no
`.item()`, no `.cpu()`, no `synchronize`, no collective, no barrier. That is
asserted on the AST rather than trusted.
"""
from __future__ import annotations

import ast
import importlib.util
import io
import sys
import threading
import time
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
GATE = REPO / "scripts" / "mlsys_coherence_gate.py"
VLLM = REPO / "scripts" / "mlsys_vllm_baseline.py"


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _tree(path: Path) -> ast.Module:
    return ast.parse(path.read_text())


def _prints_in(node: ast.AST) -> list[ast.Call]:
    return [n for n in ast.walk(node)
            if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
            and n.func.id == "print"]


def _printed_text(call: ast.Call) -> str:
    """Every string literal in a print call, joined."""
    return " ".join(n.value for n in ast.walk(call)
                    if isinstance(n, ast.Constant) and isinstance(n.value, str))


def _flushes(call: ast.Call) -> bool:
    return any(k.arg == "flush" and getattr(k.value, "value", None) is True
               for k in call.keywords)


# ---------------------------------------------------------------------------
# coherence_gate
# ---------------------------------------------------------------------------

def test_the_gate_announces_a_candidate_before_it_runs_it(tmp_path):
    """The interleaving, not merely the presence of the strings.

    A line printed after all candidates have run would leave the long silence in
    place, so what matters is that the announcement lands BEFORE the work. The
    stub appends to the same stream the gate prints to, so the order in the
    captured text is the real order.
    """
    gate = _load(GATE, "_gate_progress_under_test")
    out = io.StringIO()
    seen: list[str] = []

    class FakeTok:
        pad_token = None
        eos_token = "<eos>"

    def fake_tok(model_name):
        return FakeTok()

    def fake_candidate(cand, tok, meta_path, out_dir):
        out.write(f"    <working on {cand['name']}>\n")
        seen.append(cand["name"])
        return {"candidate": cand["name"], "context_length": cand["context_length"],
                "status": "ok", "ppl_continuation": 1.0, "native_baseline": True,
                "role": "in_distribution_reference"}

    cands = [{"name": "c_one", "target_model_name": "m", "context_length": 131072,
              "native_baseline": True, "role": "in_distribution_reference"},
             {"name": "c_two", "target_model_name": "m", "context_length": 262144,
              "native_baseline": True, "role": "in_distribution_reference"}]
    gate.run_candidate = fake_candidate
    gate.load_candidates = lambda spec: cands
    gate.AutoTokenizer = type("T", (), {"from_pretrained": staticmethod(fake_tok)})

    # main() reads the candidates file itself before consulting load_candidates,
    # so the path has to be a real (if trivial) JSON file.
    spec_path = tmp_path / "cands.json"
    spec_path.write_text("{}")
    old_argv, old_stdout = sys.argv, sys.stdout
    sys.argv = ["gate", "--candidates", str(spec_path),
                "--pg19-meta", str(tmp_path / "meta.json"),
                "--out", str(tmp_path / "out.csv"),
                "--gen-dir", str(tmp_path / "gen")]
    sys.stdout = out
    try:
        gate.main()
    finally:
        sys.argv, sys.stdout = old_argv, old_stdout

    text = out.getvalue()
    i_start = text.index("[GATE] candidate 1/2 c_one ctx=131072 start")
    i_work = text.index("<working on c_one>")
    i_done = text.index("[GATE] candidate 1/2 c_one done")
    assert i_start < i_work < i_done, (
        "the announcement must bracket the work, not follow it:\n" + text)
    assert "[GATE] candidate 2/2 c_two ctx=262144 start" in text
    assert seen == ["c_one", "c_two"]


def test_the_gate_stamps_every_phase_that_can_take_minutes():
    """Inside one candidate, the phases are minutes apart on a 512k row.

    Loading an 8B model, forwarding a 512k prompt and generating from it are each
    measured in minutes, so the longest silence has to be one PHASE, not one
    candidate. Asserted on the AST so a print that loses `flush=True` (and then
    sits in a pipe buffer, which is the same as not printing) is caught too.
    """
    tree = _tree(GATE)
    run_candidate = next(n for n in ast.walk(tree)
                         if isinstance(n, ast.FunctionDef)
                         and n.name == "run_candidate")
    texts = [(_printed_text(c), c) for c in _prints_in(run_candidate)]
    for want in ("model loaded", "ppl forward", "ppl done", "generation done"):
        hits = [(t, c) for t, c in texts if want in t]
        assert hits, f"no progress line for phase '{want}' inside run_candidate"
        assert all(_flushes(c) for _, c in hits), \
            f"the '{want}' line does not flush, so it can sit in a buffer"


def test_the_gate_progress_path_cannot_sync_a_device():
    """It runs before and between measurements, so it must not move one."""
    tree = _tree(GATE)
    run_candidate = next(n for n in ast.walk(tree)
                         if isinstance(n, ast.FunctionDef)
                         and n.name == "run_candidate")
    banned = {"item", "cpu", "numpy", "tolist", "synchronize", "cuda"}
    for call in _prints_in(run_candidate):
        for node in ast.walk(call):
            if isinstance(node, ast.Attribute) and node.attr in banned:
                raise AssertionError(
                    f"a progress line calls .{node.attr}, which can sync the device")


# ---------------------------------------------------------------------------
# vllm_ladder
# ---------------------------------------------------------------------------

def test_the_attempt_heartbeat_is_tighter_than_the_liveness_rule():
    vllm = _load(VLLM, "_vllm_progress_under_test")
    assert vllm.ATTEMPT_HEARTBEAT_S <= 120, (
        "a heartbeat at this cadence leaves the manifest's log quiet for long "
        "enough that the liveness rule is doing the pacing")
    assert vllm.ATTEMPT_HEARTBEAT_S * 4 < 15 * 60, (
        "the heartbeat must land several times inside the 15-minute rule")


def test_the_heartbeat_runs_on_a_thread_not_in_the_read_loop():
    """The read loop blocks on the child's pipe, so a silent hang never reaches it.

    A worker that writes nothing and hangs is precisely the case this covers; an
    in-loop heartbeat would print nothing in exactly that case.
    """
    tree = _tree(VLLM)
    run_attempt = next(n for n in ast.walk(tree)
                       if isinstance(n, ast.FunctionDef)
                       and n.name == "run_attempt")
    threads = [n for n in ast.walk(run_attempt)
               if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
               and n.func.attr == "Thread"]
    assert threads, "the heartbeat is not on a thread"
    assert any(k.arg == "daemon" for t in threads for k in t.keywords), \
        "a non-daemon heartbeat can keep a finished run alive"
    text = " ".join(_printed_text(c) for c in _prints_in(run_attempt))
    assert "still running" in text and "elapsed" in text


def test_a_silent_attempt_prints_a_heartbeat(monkeypatch, tmp_path, capsys):
    """Long enough to matter, and silent, is the case that matters."""
    vllm = _load(VLLM, "_vllm_heartbeat_under_test")
    monkeypatch.setattr(vllm, "ATTEMPT_HEARTBEAT_S", 0.05)

    class FakeProc:
        def __init__(self):
            self.returncode = 0
            self.killed = False
            self.stdout = self._lines()

        def _lines(self):
            # 0.3s of silence, then one line: the heartbeat must already have
            # printed by the time the child speaks.
            time.sleep(0.3)
            yield "# vLLM started\n"

        def kill(self):          # pragma: no cover - the timeout must not fire
            self.killed = True

        def wait(self, timeout=None):
            return 0

    monkeypatch.setattr(vllm.subprocess, "Popen", lambda *a, **k: FakeProc())
    spec = {"model": "m", "env": {}, "context_length": 8}
    rc, log = vllm.run_attempt(spec, tmp_path / "attempt.log", timeout_s=60)
    out = capsys.readouterr().out
    assert rc == 0
    assert "[VLLM] still running" in out, (
        "an attempt that says nothing for its first third of a second produced "
        "no heartbeat:\n" + out)

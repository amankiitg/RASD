"""Path A: an empty check stage must stop the chain, not exit 0.

WHAT WENT WRONG (lesson L1 of `results/mlsys/HANDOFF.md`). In
`scripts/mlsys_session_a_remote.sh` the chain runs one stage after another, and the
STEP c check -- the inspection that decides whether the sampled (T>0) path is sane --
was guarded by `[ "$prev" = "0" ]`. `run_stage` returns non-zero exactly when the
stage produced **zero ok rows**, so an empty check stage set `prev=1` and the guard
then SKIPPED the check. The rest of the stages logged `SKIP`, never ran, the rerun
pass retried the empty row without re-verifying it, and the script exited **0** --
a "clean" session that had run none of d/e/f/g. The DONE marker's rc is therefore
not a success signal.

THE FIX. Drop the `prev` guard, so the check runs whenever the named stage is
reached, empty or not.

WHY THIS TEST RUNS THE SCRIPT rather than reading it. Lesson L13/L14: assert on the
artifact (the DONE marker's rc, the CSVs that do or do not exist), never on the
intention. A source-level assertion would have passed for the buggy version. The
script is therefore executed end to end in a sandbox HOME with a stub
`run_experiment.py`, a stub `timeout`, and a config whose stage order is the one
under test.
"""
from __future__ import annotations

import os
import shutil
import stat
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
REMOTE = REPO / "scripts" / "mlsys_session_a_remote.sh"

CSV_HEADER = ("run_id,context_length,status,time_sec,throughput_tps,"
              "acceptance_rate,spec_steps,tokens_generated")

# The stage order the script derives from the config's key order. PRECHECK exists so
# the "a stage BEFORE the check failed" case is reachable; the check stage must be
# named by SESSION_CHECK_STAGE.
STUB = '''\
import csv, os, sys

CSV_HEADER = ("run_id,context_length,status,time_sec,throughput_tps,"
              "acceptance_rate,spec_steps,tokens_generated")

argv = sys.argv[1:]
opt = {}
i = 0
while i < len(argv):
    a = argv[i]
    if a.startswith("--"):
        key = a[2:]
        val = argv[i + 1] if i + 1 < len(argv) and not argv[i + 1].startswith("--") else "1"
        opt[key] = val
        i += 2
    else:
        i += 1

out = opt.get("output", "out.csv")
groups = opt.get("groups")
mode = os.environ

def rows_for(name):
    """(status, acceptance) per stage. `ok` rows carry ',ok,' in the raw line."""
    if name == "CANARY":
        return [("ok", 0.40)] * 2
    if name == "PRECHECK":
        return [] if mode.get("FAKE_PRECHECK") == "fail" else [("ok", 0.40)]
    if name == "CHECK":
        m = mode.get("FAKE_CHECK", "ok")
        if m == "empty":
            return [("error", 0.0)]
        if m == "badvalue":
            return [("ok", 1.7)]
        return [("ok", 0.40)]
    return [("ok", 0.40)]

if groups is None:                     # step (a): the sanity pair, separate config
    data = [("ok", 0.40)] * 2
else:
    data = rows_for(groups)

with open(out, "w", newline="") as fh:
    fh.write(CSV_HEADER + "\\n")
    for n, (status, accept) in enumerate(data):
        fh.write(",".join([
            f"{status}_{n}", "1048576", status, "100.0", "0.01",
            f"{accept}", "4", "128",
        ]) + "\\n")
sys.exit(0)
'''


def _sandbox(tmp_path: Path, stage_order: list[str]) -> tuple[Path, dict]:
    """A HOME with the real remote script, a stub runner, and a stub `timeout`."""
    home = tmp_path / "home"
    rasd = home / "RASD"
    (rasd / "scripts").mkdir(parents=True)
    (rasd / "configs").mkdir()
    (tmp_path / "bin").mkdir()

    shutil.copy(REMOTE, rasd / "scripts" / "mlsys_session_a_remote.sh")
    # The remote script sources the parameter contract and derives its stage list with
    # the shared implementation; both are real files it must find on the pod.
    for helper in ("mlsys_remote_expect.sh", "mlsys_stage_list.py"):
        shutil.copy(REPO / "scripts" / helper, rasd / "scripts" / helper)
    (rasd / "run_experiment.py").write_text(STUB)
    (rasd / ".pod_env.sh").write_text("")
    (home / "pod_env.log").write_text(f"INTERPRETER={sys.executable}\n")
    (home / ".hf_token").write_text("stub\n")

    # `timeout` is not in POSIX and this runs on macOS; stub it so the test does not
    # depend on coreutils being installed.
    to = tmp_path / "bin" / "timeout"
    to.write_text('#!/bin/sh\nshift\nexec "$@"\n')
    to.chmod(to.stat().st_mode | stat.S_IEXEC)

    body = "\n".join(
        f"{name}:\n  name: {name.lower()}\n  factor: context_length\n  levels:\n"
        f"  - id: {name}\n    context_length: 1048576\n    documents: [d0]\n"
        f"    max_new_tokens: 8\n"
        for name in stage_order)
    (rasd / "configs" / "stages.yml").write_text("defaults:\n  seeds: [42]\n" + body)
    (rasd / "configs" / "sanity.yml").write_text(
        "defaults:\n  seeds: [42]\nSANITY:\n  name: sanity\n  factor: context_length\n"
        "  levels:\n  - id: SANITY\n    context_length: 1048576\n    documents: [d0]\n"
        "    max_new_tokens: 8\n")

    # THE PARAMETER CONTRACT. The remote script refuses to start without a manifest
    # (see scripts/mlsys_remote_expect.sh), so the sandbox must send one exactly as the
    # driver does -- and the stage list is derived with the SHARED implementation, which
    # is what keeps the two sides from disagreeing about which stages are running.
    import subprocess as _sp
    stages = _sp.run([sys.executable, str(REPO / "scripts" / "mlsys_stage_list.py"),
                      str(rasd / "configs" / "stages.yml"), "--space"],
                     capture_output=True, text=True).stdout.strip()
    env = dict(os.environ)
    env.update({
        "HOME": str(home),
        "PATH": f"{tmp_path / 'bin'}{os.pathsep}{os.environ['PATH']}",
        "SESSION_OUT": "out",
        "SESSION_CFG": "configs/stages.yml",
        "SESSION_SANITY_CFG": "configs/sanity.yml",
        "SESSION_CHECK_STAGE": "CHECK",
        "SESSION_GATE_CANDIDATES": "",
        "NPROC": "2",
        "REMOTE_EXPECT": ("SESSION_OUT=out;SESSION_CFG=configs/stages.yml;"
                          "SESSION_SANITY_CFG=configs/sanity.yml;NPROC=2;"
                          "SESSION_CHECK_STAGE=CHECK;SESSION_GATE_CANDIDATES=;"
                          f"STAGES={stages}"),
    })
    return home, env


def _run(tmp_path: Path, stage_order: list[str], check: str,
         precheck: str = "ok", expect: str | None = None
         ) -> tuple[int, str, Path, str]:
    """Run the real script; return (DONE rc, session log, output dir, stdout).

    The rc is read from the DONE marker the script's own EXIT trap writes -- the
    artifact, not the intention (L14).
    """
    home, env = _sandbox(tmp_path, stage_order)
    env["FAKE_CHECK"] = check
    env["FAKE_PRECHECK"] = precheck
    if expect is not None:
        env["REMOTE_EXPECT"] = expect
    proc = subprocess.run(
        ["bash", str(home / "RASD" / "scripts" / "mlsys_session_a_remote.sh")],
        cwd=str(home / "RASD"), env=env, capture_output=True, text=True, timeout=300)
    out = home / "RASD" / "out"
    marker = out / "DONE"
    assert marker.exists(), (
        f"no DONE marker (the EXIT trap wrote nothing)\n"
        f"stdout:\n{proc.stdout[-3000:]}\nstderr:\n{proc.stderr[-3000:]}")
    rc = int(marker.read_text().split()[1])
    log = (out / "session_a.log").read_text() if (out / "session_a.log").exists() else ""
    # stdout AND stderr: on the pod both are redirected into remote_stdout.log, so the
    # script's own FATAL lines and its progress lines end up in one place there.
    return rc, log, out, proc.stdout + proc.stderr


STAGES = ["CANARY", "PRECHECK", "CHECK", "LATER1", "LATER2"]


def test_an_empty_check_stage_exits_five(tmp_path):
    """THE regression. Zero ok rows in the check stage must exit 5.

    The buggy version exited 0 here, because `prev=1` skipped the very check that
    was supposed to catch it.
    """
    rc, log, out, _ = _run(tmp_path, STAGES, check="empty")
    assert rc == 5, f"an empty check stage must exit 5, got {rc}\n{log[-2000:]}"
    assert "STEP c check FAILED" in log
    # and the stop is PROMPT: the stages after it never ran. They did run under the
    # bug, where `prev` had already been zeroed for them by the earlier skips.
    for later in ("LATER1", "LATER2"):
        assert not (out / f"session_a_{later}.csv").exists(), \
            f"{later} ran even though the check stage failed"


def test_a_stage_before_the_check_failing_is_loud_but_still_exits_zero(tmp_path):
    """The case the `prev` guard change does NOT close, pinned as it really behaves.

    If a stage BEFORE the check fails, the SKIP guard `continue`s before the check is
    reached, so the check never runs; the chain stops loudly (SKIP lines) and the script
    exits 0 after the retry pass. That is the designed chain-stop -- making it exit 5
    would abort the retry pass that can still recover a transient failure -- but it means
    an rc=0 session can still have run none of d/e/f/g, which is exactly why L1 says to
    judge from the stage log lines and the CSVs rather than from the rc.
    """
    rc, log, out, _ = _run(tmp_path, STAGES, check="ok", precheck="fail")
    assert "SKIP CHECK" in log, "the chain must stop loudly, with a SKIP line"
    assert "STEP c check PASSED" not in log, \
        "the check must never be reported as passed when it never ran"
    for later in ("LATER1", "LATER2"):
        assert f"SKIP {later}" in log
        assert not (out / f"session_a_{later}.csv").exists()
    assert rc == 0, f"unchanged by this fix; got {rc}"


def test_a_check_row_with_impossible_acceptance_exits_five(tmp_path):
    """Pre-existing behaviour, pinned: a present but insane row is a failure too.

    The reason line goes to the script's stdout, which the pod keeps in
    `remote_stdout.log`; the verdict goes to the stage log.
    """
    rc, log, _, stdout = _run(tmp_path, STAGES, check="badvalue")
    assert rc == 5, f"acceptance outside (0,1] must exit 5, got {rc}\n{log[-2000:]}"
    assert "STEP c check FAILED" in log
    assert "acceptance 1.7 outside" in stdout


def test_a_healthy_check_stage_runs_the_whole_chain(tmp_path):
    """The control. Without it, "exits 5" could pass for a script that always does."""
    rc, log, out, _ = _run(tmp_path, STAGES, check="ok")
    assert rc == 0, f"a healthy session must exit 0, got {rc}\n{log[-2000:]}"
    assert "STEP c check PASSED" in log
    for later in ("LATER1", "LATER2"):
        assert (out / f"session_a_{later}.csv").exists(), f"{later} did not run"
    assert "SKIP LATER1" not in log


# ---------------------------------------------------------------------------
# the parameter contract (scripts/mlsys_remote_expect.sh), from the pod's side
# ---------------------------------------------------------------------------

def test_a_wrong_intended_value_stops_the_session_before_any_stage(tmp_path):
    """The 2026-10-12 failure in its general form: the pod's parameters differ from
    what the driver intended, and the pod must refuse rather than run."""
    rc, log, out, stdout = _run(tmp_path, STAGES, check="ok",
                                expect="SESSION_OUT=out;SESSION_CFG=configs/stages.yml;"
                                       "SESSION_SANITY_CFG=configs/sanity.yml;NPROC=99;"
                                       "SESSION_CHECK_STAGE=CHECK;"
                                       "SESSION_GATE_CANDIDATES=;"
                                       "STAGES=" + " ".join(STAGES))
    assert rc == 2, f"a wrong NPROC must stop the session, got {rc}\n{log[-1500:]}"
    assert "NPROC" in stdout and "intended" in stdout
    assert "### STEP a" in log, "the refusal must come after the cheap sanity, not instead of it"


def test_a_parameter_the_pod_never_received_is_named(tmp_path):
    """A key in the manifest that the pod's own set does not carry means the pod is
    using a DEFAULT for it -- the exact shape of the $5.5 mistake."""
    rc, log, out, stdout = _run(tmp_path, STAGES, check="ok",
                                expect="SESSION_OUT=out;SESSION_CFG=configs/stages.yml;"
                                       "SESSION_SANITY_CFG=configs/sanity.yml;NPROC=2;"
                                       "SESSION_CHECK_STAGE=CHECK;"
                                       "SESSION_GATE_CANDIDATES=;"
                                       "SESSION_TAG=never_sent;"
                                       "STAGES=" + " ".join(STAGES))
    assert rc == 2, f"an unverifiable parameter must stop the session, got {rc}\n{log[-1500:]}"
    assert "SESSION_TAG" in stdout
    assert "never received that setting" in stdout

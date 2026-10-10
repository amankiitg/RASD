"""The two defects that destroyed a healthy run on 2026-10-10T00:37Z.

WHAT HAPPENED. The armed watcher was 49 minutes into a healthy
`engine_cap_smoke` when it received a SIGTERM. The copilot runtime it had been
launched under restarted 11 seconds later (00:37:15Z), and the separately
launched launch monitor stopped heartbeating at 00:36:31Z -- both died together.

Two things were wrong, and they are independent:

1. THE SIGNAL PATH DID NOT PULL. Every in-loop stop path calls `collect_incident`
   first, which pulls the pod's logs and results and merges the cost ledger. The
   TERM/INT/EXIT traps called `terminate_and_confirm` alone, so the trap
   terminated a working 8xA100 and destroyed that run's `gate_calibration.csv`
   (676s, $4.19) and its three completed `engine_cap_smoke` rows. A signal is
   precisely the case where the results are the only thing left, so this was
   backwards.

2. `nohup ... & disown` DOES NOT DETACH. `nohup` ignores SIGHUP and `disown`
   clears the job table, but neither moves the process out of the parent's
   session, so a session-scoped signal still reaches it. `setsid(1)` is not on
   macOS; `os.setsid()` in a double-forked child is.
"""
from __future__ import annotations

import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
WATCHER = REPO / "scripts" / "mlsys_watch_and_run.sh"
DETACH = REPO / "scripts" / "mlsys_detach.py"
SRC = WATCHER.read_text()


# ---------------------------------------------------------------------------
# 1. the signal path pulls before it terminates
# ---------------------------------------------------------------------------

def test_every_trap_pulls_before_terminating():
    """A trap that only terminates throws the results away with the pod."""
    for line in SRC.splitlines():
        if line.startswith("trap ") and "terminate_and_confirm" in line:
            raise AssertionError(
                f"a trap still terminates without pulling: {line.strip()}")
    for want in ("trap pull_then_terminate EXIT",
                 "trap 'pull_then_terminate; exit 130' INT",
                 "trap 'pull_then_terminate; exit 143' TERM"):
        assert want in SRC, f"missing: {want}"


def test_the_pull_happens_before_the_terminate_inside_the_function():
    """Order inside the function is the whole point."""
    body = SRC[SRC.index("pull_then_terminate() {"):]
    body = body[:body.index("\n}\n")]
    i_pull = body.index("collect_incident")
    i_term = body.index("terminate_and_confirm")
    assert i_pull < i_term, "collect_incident must run BEFORE terminate_and_confirm"
    assert "timeout 900" in body, (
        "the pull is unbounded, so an unreachable pod could hold a Ctrl-C open; "
        "a stop must still stop")


def test_the_pull_is_skipped_only_where_it_is_already_done():
    """Two skips, both deliberate: no instance, and results already merged."""
    body = SRC[SRC.index("pull_then_terminate() {"):]
    body = body[:body.index("\n}\n")]
    assert '[ -z "$INSTANCE_ID" ] && return 0' in body, \
        "without an instance there is nothing to pull"
    assert '${RESULTS_HANDLED:-}' in body, \
        "the normal completion path pulls and merges; a second pull would " \
        "re-merge the cost ledger"
    # and the normal path must set that flag
    tail = SRC[SRC.index('say "manifest finished (reason=manifest-finished)"'):]
    assert "RESULTS_HANDLED=1" in tail[:400], \
        "the normal completion path does not mark the results as handled"


def test_the_trap_path_is_the_only_one_that_changed():
    """The in-loop stop paths already pulled; they must keep doing so."""
    for reason in ("liveness stall in", "STALL:"):
        assert reason in SRC
    assert SRC.count("collect_incident ") >= 5, \
        "the in-loop abort paths call collect_incident; they still must"


# ---------------------------------------------------------------------------
# 2. the detacher really detaches
# ---------------------------------------------------------------------------

def test_the_detacher_returns_immediately(tmp_path):
    """The caller must get its shell back; the child keeps running."""
    cmd = [sys.executable, str(DETACH), "--pidfile", str(tmp_path / "p"),
           "--log", str(tmp_path / "l"), "--", "sleep", "30"]
    t0 = time.time()
    rc = subprocess.run(cmd, timeout=20).returncode
    elapsed = time.time() - t0
    assert rc == 0
    assert elapsed < 5, f"the detacher blocked for {elapsed:.1f}s"
    pid = int((tmp_path / "p").read_text().strip())
    assert pid > 0
    time.sleep(0.3)
    os.kill(pid, 0)                      # still alive
    os.kill(pid, signal.SIGKILL)


def test_the_detached_child_is_in_its_own_process_group(tmp_path):
    """THE property `nohup`+`disown` does not provide.

    A child in the launcher's group dies when that group is signalled, which is
    what took the watcher down at 2026-10-10T00:37:04Z.
    """
    wrapper = os.fork()
    if wrapper == 0:                              # pragma: no cover - child
        os.setpgid(0, 0)
        subprocess.run([sys.executable, str(DETACH),
                        "--pidfile", str(tmp_path / "p"),
                        "--log", str(tmp_path / "l"), "--", "sleep", "30"])
        time.sleep(30)
        os._exit(0)
    try:
        time.sleep(2)
        group = os.getpgid(wrapper)
        child = int((tmp_path / "p").read_text().strip())
        os.kill(child, 0)
        assert os.getpgid(child) != group, \
            "the child shares the launcher's process group: a group signal kills it"
        # the experiment that matters: kill the launcher's whole group
        os.killpg(group, signal.SIGTERM)
        time.sleep(1.5)
        os.kill(child, 0)                         # raises if it died
        os.kill(child, signal.SIGKILL)
    finally:
        try:
            os.killpg(os.getpgid(wrapper), signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            pass


def test_a_child_in_the_launchers_group_does_die(tmp_path):
    """The control, so the test above cannot pass for a trivial reason.

    Without this, `os.kill(child, 0)` succeeding would prove nothing about the
    group signal -- it would also succeed if the signal had never been sent.
    """
    wrapper = os.fork()
    if wrapper == 0:                              # pragma: no cover - child
        os.setpgid(0, 0)
        p = subprocess.Popen(["sleep", "30"])     # inherits the group
        (tmp_path / "p").write_text(str(p.pid))
        time.sleep(30)
        os._exit(0)
    try:
        time.sleep(2)
        group = os.getpgid(wrapper)
        child = int((tmp_path / "p").read_text().strip())
        os.killpg(group, signal.SIGTERM)
        time.sleep(1.5)
        with pytest.raises(OSError):
            os.kill(child, 0)
    finally:
        try:
            os.killpg(os.getpgid(wrapper), signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            pass


def test_the_detacher_redirects_stdin_and_captures_output(tmp_path):
    cmd = [sys.executable, str(DETACH), "--pidfile", str(tmp_path / "p"),
           "--log", str(tmp_path / "l"), "--",
           "bash", "-c", "echo from-the-child; read -t 1 line; echo stdin=$?"]
    subprocess.run(cmd, timeout=20)
    time.sleep(1.5)
    log = (tmp_path / "l").read_text()
    assert "from-the-child" in log, "the child's stdout is not captured"
    assert "stdin=" in log, "the child did not finish; it may be waiting on a tty"
    pid = int((tmp_path / "p").read_text().strip())
    try:
        os.kill(pid, signal.SIGKILL)
    except OSError:
        pass

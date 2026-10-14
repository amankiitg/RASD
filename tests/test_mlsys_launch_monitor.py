"""The monitor must report a gate result even when the pod is already gone.

On 2026-10-08T15:17Z both gates finished and neither was reported:

    15:29:57Z  STAGE_OK     name=gate_calibration   (the first gate ever to
                                                     calibrate, in 8x dollars)
    15:30:16Z  CAPS_prefix64_pg19_train_1_s42 FAILED (wandb, no GPU work)
    15:30:22Z  STAGE_FAILED name=engine_cap_smoke
    15:30:31Z  TERMINATING a651b11d...
    15:32:37Z  CONFIRMED TERMINATED (0 instances)

The monitor's only source of gate results was `ssh ubuntu@<ip> grep ~/manifest.log`
once a minute, and the instance was terminated 34 seconds after the first gate
finished. Two results existed, both were in the local incident directory (the new
`collect_incident` pull had already brought them home), and neither reached
ARM_STATUS.

So the local `results/mlsys/incident_*/manifest.log` is now read as well, both as
soon as it appears and again when the watcher exits. These tests run the real
monitor against a fake session directory: no pod, no network, no instance.

The other half is the stale-line defect: armings append to the same stdout file,
so phase A matched the PREVIOUS arming's `LAUNCHED` line and reported it as this
one's. The arming now writes a marker line first and the monitor reads only what
follows the last marker.
"""
from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
MONITOR = Path(os.environ.get(
    "RASD_MONITOR_PATH",
    "/Users/amankesarwani/.copilot/session-state/"
    "0f919471-cdca-48c0-a994-56a95eb1d6be/files/mlsys_launch_monitor.sh"))

MARKER = "===== ARMING 2026-10-08T15:04:08Z =====\n"

CAL_OK_LOG = """\
2026-10-08T15:17:43Z === MANIFEST START rate=$22.32 ===
2026-10-08T15:17:44Z STAGE_START name=gate_calibration timeout=14400s
  candidate                    ctx     anchor        ppl   ratio   blank    rep   eos  gate
  P1_native_32k              32768   declared     1.0325     1.0  0.1667 0.0075    no  PASS
2026-10-08T15:29:57Z STAGE_OK name=gate_calibration wall=733s cost=$4.54 cumulative=$39.40
2026-10-08T15:29:58Z STAGE_START name=engine_cap_smoke timeout=21600s
2026-10-08T15:30:22Z STAGE_FAILED name=engine_cap_smoke rc=1 wall=24s cost=$0.15
STOP: the cap smoke rc=1
"""


def _sandbox(tmp_path, stdout_text, manifest_log=None, gates_dir=None):
    """A session dir and a repo with an incident, and no pod in sight."""
    sess = tmp_path / "sess"
    sess.mkdir()
    (sess / "watcher.stdout").write_text(stdout_text)
    results = tmp_path / "repo" / "results" / "mlsys"
    if manifest_log is not None:
        inc = results / (gates_dir or "incident_20261008T153023Z")
        inc.mkdir(parents=True)
        (inc / "manifest.log").write_text(manifest_log)
        (inc / "incident.txt").write_text("reason : manifest aborted rc=1\n")
    else:
        results.mkdir(parents=True)
    return sess


def _run(tmp_path, sess, seconds=3, poll=1):
    env = dict(os.environ,
               MLSYS_MONITOR_SESSION=str(sess),
               MLSYS_MONITOR_STDOUT=str(sess / "watcher.stdout"),
               MLSYS_MONITOR_REPO=str(tmp_path / "repo"),
               MLSYS_MONITOR_SECONDS=str(seconds),
               MLSYS_MONITOR_POLL_A=str(poll),
               MLSYS_MONITOR_POLL_B=str(poll))
    proc = subprocess.run(["bash", str(MONITOR)], capture_output=True, text=True,
                          timeout=90, env=env)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    return (sess / "ARM_STATUS").read_text()


# --------------------------------------------------------------------------
# the result that was lost
# --------------------------------------------------------------------------

def test_a_gate_result_is_reported_from_the_local_incident(tmp_path):
    """The pod is gone; the incident on disk is enough."""
    sess = _sandbox(
        tmp_path,
        MARKER + "2026-10-08T15:04:15Z LAUNCHED a651b11d in us-midwest-1\n"
                 "2026-10-08T15:30:31Z terminating the instance and stopping\n"
                 "2026-10-08T15:32:37Z CONFIRMED TERMINATED (0 instances)\n",
        manifest_log=CAL_OK_LOG)
    status = _run(tmp_path, sess)

    assert "EVENT=LAUNCHED" in status
    assert "EVENT=TERMINAL" in status, "the watcher's exit was not reported"
    assert "EVENT=GATE_RESULT" in status, (
        "the gate result was not reported: this is the 2026-10-08T15:17Z defect")
    assert "gate=gate_calibration verdict=OK" in status
    assert "source=local:" in status, "the result did not come from the incident"


def test_a_failed_gate_is_reported_and_flagged_as_a_stop(tmp_path):
    sess = _sandbox(
        tmp_path,
        MARKER + "2026-10-08T15:04:15Z LAUNCHED a651b11d in us-midwest-1\n"
                 "2026-10-08T15:32:37Z CONFIRMED TERMINATED (0 instances)\n",
        manifest_log=CAL_OK_LOG)
    status = _run(tmp_path, sess)

    assert "gate=engine_cap_smoke verdict=FAILED" in status, (
        "the failing gate was not reported")
    assert "gate=gate_calibration verdict=OK" in status, (
        "a later failure suppressed the earlier pass")
    assert "do not retry" in status.lower()
    assert "gates reported: gate_calibration engine_cap_smoke" in status


def test_a_gate_that_never_ran_is_not_reported_as_anything(tmp_path):
    """Only the two gates that appear in the log, and no invented third."""
    sess = _sandbox(
        tmp_path,
        MARKER + "2026-10-08T15:04:15Z LAUNCHED a651b11d in us-midwest-1\n"
                 "2026-10-08T15:32:37Z CONFIRMED TERMINATED (0 instances)\n",
        manifest_log=CAL_OK_LOG)
    status = _run(tmp_path, sess)
    assert "gate=coherence_gate" not in status
    assert status.count("EVENT=GATE_RESULT") == 2


def test_a_stale_launched_line_from_a_previous_arming_is_ignored(tmp_path):
    """The other defect: phase A matched the whole file, so the last arming's
    LAUNCHED line was reported again as this one's."""
    sess = _sandbox(
        tmp_path,
        "2026-10-08T14:10:22Z LAUNCHED 238e8d3534b740f7b2bdc2f1bba550b5 in "
        "us-east-1 (reason=detected)\n"
        "2026-10-08T14:17:50Z instance active at 132.145.192.145\n"
        + MARKER +
        "2026-10-08T15:04:09Z attempt=1 regions=us-midwest-1 capacity=yes\n")
    status = _run(tmp_path, sess, seconds=2)
    assert "EVENT=LAUNCHED" not in status, (
        "a previous arming's LAUNCHED line was reported as this arming's")
    assert "EVENT=TERMINAL" not in status


def test_this_armings_launch_is_reported_exactly_once(tmp_path):
    sess = _sandbox(
        tmp_path,
        "2026-10-08T14:10:22Z LAUNCHED 238e8d3534b740f7b2bdc2f1bba550b5 in "
        "us-east-1\n"
        + MARKER +
        "2026-10-08T15:04:15Z LAUNCHED a651b11d in us-midwest-1 (reason=detected)\n"
                 "2026-10-08T15:32:37Z CONFIRMED TERMINATED (0 instances)\n")
    status = _run(tmp_path, sess, seconds=3)
    assert status.count("EVENT=LAUNCHED") == 1, status
    assert "a651b11d" in status, "the current launch was not the one reported"
    assert "238e8d3534" not in status, "the stale launch was reported"


def test_a_watcher_that_stops_before_launching_is_terminal(tmp_path):
    """Phase A's job: the 72h capacity window closing, or a fail-closed exit."""
    sess = _sandbox(
        tmp_path,
        MARKER + "2026-10-08T15:04:09Z attempt=1 regions=[] capacity=no\n"
                 "2026-10-08T15:04:20Z CAPACITY WAIT EXPIRED after 72h\n")
    status = _run(tmp_path, sess, seconds=2)
    assert "EVENT=TERMINAL" in status
    assert "CAPACITY WAIT EXPIRED" in status
    assert "EVENT=LAUNCHED" not in status


def test_the_incident_source_is_read_before_the_pod_is_touched(tmp_path):
    """Structural: the local read must come first in phase B, so a dead pod can
    never suppress a result that is already on disk."""
    src = MONITOR.read_text()
    body = src[src.index("phase B"):]
    local = body.index("(1) the LOCAL incident")
    stop = body.index("(2) the watcher has finished")
    pod = body.index("(3) while the pod is alive")
    assert local < stop < pod, (
        "the order changed: the local incident must be read before the watcher's "
        "exit and before any SSH attempt")


@pytest.mark.skipif(not MONITOR.exists(), reason="monitor not installed")
def test_the_monitor_never_terminates_or_launches_anything(tmp_path):
    """It is a notification channel. It must not be able to spend money or stop a
    run. Comments are stripped first: the watcher function it waits on is named
    in the prose that explains why the monitor waits for it."""
    code = "\n".join(l for l in MONITOR.read_text().splitlines()
                     if not l.lstrip().startswith("#"))
    for forbidden in ("terminate_and_confirm", "launch_instance",
                      "lambdalabs.com", "curl "):
        assert forbidden not in code, (
            f"the monitor CALLS {forbidden!r}; it must stay read-only")


# --------------------------------------------------------------------------
# an incident from a PREVIOUS arming is not this arming's result
#
# On 2026-10-08T17:36Z the monitor started while the previous run's incident was
# still the newest on disk and reported ITS gates at 17:37:12Z -- before this
# run's gates existed. It then marked them SEEN, so the real results were never
# reported at all. The incident directory's name carries its UTC timestamp and
# the arming marker carries the arming's, so the cutoff is a string comparison
# of the same 14 digits.
# --------------------------------------------------------------------------

STALE_LOG = """\
2026-10-08T15:29:57Z STAGE_OK name=gate_calibration wall=733s cost=$4.54
2026-10-08T15:30:22Z STAGE_FAILED name=engine_cap_smoke rc=1 wall=24s
"""

FRESH_LOG = """\
2026-10-08T18:02:44Z STAGE_OK name=gate_calibration wall=753s cost=$4.67
2026-10-08T18:11:00Z STAGE_START name=engine_cap_smoke timeout=21600s
"""

MARKER_NEWER = "===== ARMING 2026-10-08T16:24:55Z =====\n"


def test_an_incident_from_before_this_arming_is_not_reported(tmp_path):
    """Today's case exactly: the previous run's incident is on disk at launch,
    with gate_calibration OK and engine_cap_smoke FAILED, and it must NOT be
    reported -- that FAILED belongs to a different run."""
    sess = _sandbox(
        tmp_path,
        MARKER_NEWER +
        "2026-10-08T17:36:51Z LAUNCHED 5d1f0f61 in us-midwest-1\n"
        "2026-10-08T18:40:00Z CONFIRMED TERMINATED (0 instances)\n",
        manifest_log=STALE_LOG, gates_dir="incident_20261008T153023Z")
    status = _run(tmp_path, sess, seconds=3)

    assert "EVENT=LAUNCHED" in status
    assert "EVENT=TERMINAL" in status
    assert "gate=gate_calibration" not in status, (
        "the PREVIOUS arming's gate_calibration was reported as this one's")
    assert "gate=engine_cap_smoke" not in status, (
        "the PREVIOUS arming's engine_cap_smoke FAILED was reported; that is "
        "the failure this test exists for")
    assert "ignoring incidents older than" in status


def test_this_armings_incident_is_still_reported(tmp_path):
    """The cutoff must not swallow the current run's own incident."""
    sess = _sandbox(
        tmp_path,
        MARKER_NEWER +
        "2026-10-08T17:36:51Z LAUNCHED 5d1f0f61 in us-midwest-1\n"
        "2026-10-08T18:40:00Z CONFIRMED TERMINATED (0 instances)\n",
        manifest_log=FRESH_LOG, gates_dir="incident_20261008T183500Z")
    status = _run(tmp_path, sess, seconds=3)
    assert "gate=gate_calibration verdict=OK" in status
    assert "source=local:" in status


def test_an_old_incident_does_not_shadow_a_new_one(tmp_path):
    """Both on disk: the newest is read, and the old one's FAILED never appears
    even though its manifest.log has one."""
    sess = _sandbox(
        tmp_path,
        MARKER_NEWER +
        "2026-10-08T17:36:51Z LAUNCHED 5d1f0f61 in us-midwest-1\n"
        "2026-10-08T18:40:00Z CONFIRMED TERMINATED (0 instances)\n",
        manifest_log=FRESH_LOG, gates_dir="incident_20261008T183500Z")
    # the stale one, written after so it is also the newest by mtime
    stale = tmp_path / "repo" / "results" / "mlsys" / "incident_20261008T153023Z"
    stale.mkdir(parents=True)
    (stale / "manifest.log").write_text(STALE_LOG)
    status = _run(tmp_path, sess, seconds=3)
    assert "incident_20261008T183500Z" in status, \
        "the current run's incident was not the one read"
    assert "gate=engine_cap_smoke" not in status, (
        "the stale incident's engine_cap_smoke FAILED was reported")


def test_the_fallback_cutoff_is_the_launch_time_not_zero(tmp_path):
    """With no marker (an older-style stdout), the cutoff is this arming's
    LAUNCHED time; an incident from before it is still excluded. Zero would
    admit every old incident."""
    sess = _sandbox(
        tmp_path,
        "2026-10-08T17:36:51Z LAUNCHED 5d1f0f61 in us-midwest-1\n"
        "2026-10-08T18:40:00Z CONFIRMED TERMINATED (0 instances)\n",
        manifest_log=STALE_LOG, gates_dir="incident_20261008T153023Z")
    status = _run(tmp_path, sess, seconds=3)
    assert "EVENT=LAUNCHED" in status
    assert "gate=engine_cap_smoke" not in status
    assert "017361" in status or "ignoring incidents older than" in status

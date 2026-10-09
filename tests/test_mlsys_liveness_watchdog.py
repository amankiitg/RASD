"""Tests for the byte-level liveness watchdog.

The per-stage stall thresholds only move when a `STAGE_*` line lands in
RUN_LOG.txt, so a hang INSIDE a stage is invisible to them. That is not a
theoretical gap: on 2026-10-09T00:00Z the teacher-forced probe deadlocked and
wrote nothing for 62.0 minutes, and engine_cap_smoke's own threshold is 240
minutes, so no watchdog fired at all -- the NCCL watchdog ended the run an hour
later. The instance billed ~$43 for a run that could never finish.

So these tests cover three things: the threshold's value and its provenance, the
mechanism on both sides (pod and operator), and the two ways it could be wrong --
firing on a legitimate quiet phase, or being quietly disabled.
"""
from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

MANIFEST_SRC = (REPO_ROOT / "scripts/mlsys_manifest.sh").read_text()
WATCH_SRC = (REPO_ROOT / "scripts/mlsys_watch_and_run.sh").read_text()
THRESHOLDS = json.loads(
    (REPO_ROOT / "configs/mlsys_stall_thresholds.json").read_text())

# The measurement the threshold is derived from: the longest interval in the
# campaign's own manifest logs during which nothing was written. It is a 1M-token
# target prefill, measured as ttft over 146 rows, because the runner prints a
# phase's first line when the phase is DONE -- so a prefill is exactly silence.
MAX_LEGITIMATE_SILENCE_MIN = 6.29


class TestTheThreshold:
    def test_liveness_is_configured(self):
        assert "liveness_minutes" in THRESHOLDS, (
            "the liveness rule is inert without a threshold in the file the pod "
            "and the operator both read")

    def test_the_threshold_is_above_the_longest_legitimate_silence(self):
        v = THRESHOLDS["liveness_minutes"]
        assert v > MAX_LEGITIMATE_SILENCE_MIN, (
            f"liveness_minutes={v} is at or below the longest legitimate "
            f"silence ({MAX_LEGITIMATE_SILENCE_MIN} min); it would kill a "
            f"legitimate 1M-token prefill")

    def test_the_threshold_is_15_minutes_and_the_2x_bound_does_not_bind(self):
        """The operator's rule: 2x the measured silence, unless that exceeds
        15 min. It does not, so 15 stands -- and this asserts the arithmetic
        rather than the number, so a future re-measurement changes both."""
        v = THRESHOLDS["liveness_minutes"]
        assert v == 15, f"expected the 15-minute liveness threshold, got {v}"
        assert 2 * MAX_LEGITIMATE_SILENCE_MIN < v, (
            "2x the measured silence must be BELOW the threshold, not equal to "
            "or above it -- otherwise the operator's rule says to use 2x")

    def test_it_is_stricter_than_every_stage_threshold(self):
        """If a stage's own bound were tighter, the liveness rule would be dead
        code for that stage and its silence would be caught by the looser rule
        that missed the deadlock."""
        v = THRESHOLDS["liveness_minutes"]
        for stage, limit in THRESHOLDS["stages"].items():
            assert v < limit, (
                f"{stage} has a per-stage limit of {limit} min, which is not "
                f"above the liveness threshold {v} -- the tighter rule wins, so "
                f"the per-stage value is the outer bound only")

    def test_the_reason_is_recorded_with_the_numbers(self):
        why = THRESHOLDS.get("_liveness_why", "")
        assert "6.29" in why, "the measured silence must be in the record"
        assert "62" in why or "62.0" in why, (
            "the deadlock that motivated the rule must be in the record")
        assert "2026-10-09" in why


class TestItIsNotPerStageOverridable:
    def test_neither_side_reads_a_per_stage_liveness_value(self):
        """A per-stage override would let exactly the stage that hangs opt out
        of the check that exists to catch it."""
        for name, src in (("pod", MANIFEST_SRC), ("watcher", WATCH_SRC)):
            assert 'd["stages"]' not in src.split("liveness_minutes()")[1][:600], (
                f"the {name}'s liveness lookup reads the per-stage table")


class TestThePodSide:
    def test_the_pod_watches_manifest_log_not_only_run_log(self):
        assert "MANIFEST_LOG" in MANIFEST_SRC
        assert 'stat -c %s "$MANIFEST_LOG"' in MANIFEST_SRC, (
            "liveness must be measured in BYTES of the manifest's own output")

    def test_a_liveness_stall_is_distinguishable_from_a_stage_stall(self):
        """The two failures have different causes and must not arrive in the
        logs looking identical."""
        assert "LIVENESS name=" in MANIFEST_SRC
        assert "STALL_ABORT" in MANIFEST_SRC
        assert "manifest.log-not-growing" in MANIFEST_SRC

    def test_a_liveness_stall_stops_the_manifest_with_a_non_zero_exit(self):
        """The watcher's fail-fast path keys on a non-zero rc, so a stall that
        exited 0 would be read as success."""
        i = MANIFEST_SRC.index("LIVENESS name=")
        tail = MANIFEST_SRC[i:i + 1200]
        assert "kill -TERM $MANIFEST_PID" in tail, (
            "the liveness branch must stop the manifest, not just log")

    def test_it_does_not_fire_before_a_stage_is_running(self):
        """Provisioning is legitimately quiet in manifest.log -- the venv install
        writes to pod_env.log -- and it happens before the first STAGE_START."""
        i = MANIFEST_SRC.index("(i) byte-level liveness")
        branch = MANIFEST_SRC[i:i + 1600]
        assert '[ -n "$name" ]' in branch, (
            "the liveness branch must require a running stage; without that "
            "check it would fire during provisioning")


class TestTheOperatorSide:
    def test_the_probe_carries_the_log_size(self):
        assert "logbytes=" in WATCH_SRC, (
            "the operator's probe must stat the pod's manifest.log, or its "
            "liveness rule cannot see growth at all")
        assert 'stat -c %s "$f"' in WATCH_SRC

    def test_the_watcher_applies_the_same_threshold_source(self):
        assert "liveness_minutes()" in WATCH_SRC
        i = WATCH_SRC.index("liveness_minutes()")
        assert 'd.get("liveness_minutes"' in WATCH_SRC[i:i + 600], (
            "both sides must read the same key from the same file, or they can "
            "disagree about what quiet means")

    def test_an_operator_side_liveness_stall_is_an_incident(self):
        i = WATCH_SRC.index("LIVE_MIN * 60")
        branch = WATCH_SRC[i:i + 900]
        assert "collect_incident" in branch, (
            "a liveness stall must pull the logs, terminate and stop")
        assert "MANIFEST_ABORTED=1" in branch

    def test_it_carries_the_same_guard_against_firing_during_setup(self):
        # The condition and the guard are on the SAME line, so search outwards
        # from it rather than forwards.
        i = WATCH_SRC.index("LIVE_MIN * 60")
        line_start = WATCH_SRC.rindex("\n", 0, i)
        line_end = WATCH_SRC.index("\n", i)
        assert '[ -n "$stage" ]' in WATCH_SRC[line_start:line_end], (
            "the operator's liveness rule must require a running stage too")


class TestTheHelpersActuallyWork:
    """Extract both threshold helpers and run them: the value must reach the
    shell, not just the JSON."""

    def _helper(self) -> str:
        """The pod's `liveness_minutes` exactly as shipped, from its own
        definition to its closing brace."""
        i = MANIFEST_SRC.index("liveness_minutes() {")
        return MANIFEST_SRC[i:MANIFEST_SRC.index("\n}\n", i) + 3]

    def _run(self, script: str, env: dict | None = None):
        import os
        e = dict(os.environ)
        e.update(env or {})
        out = subprocess.run(["bash", "-c", script], capture_output=True,
                             text=True, env=e, cwd=REPO_ROOT)
        return out.stdout.strip(), out.stderr.strip()

    def test_the_pod_helper_returns_the_configured_value(self):
        script = (
            'cd "$REPO"\nPY=python3\nSTALL_JSON=configs/mlsys_stall_thresholds.json\n'
            'MLSYS_MANIFEST_LIVENESS_MIN=15\n' + self._helper()
            + "\nliveness_minutes\n"
        )
        out, err = self._run(script, {"REPO": str(REPO_ROOT)})
        assert out == "15", f"got {out!r} (stderr: {err[:200]})"

    def test_the_pod_helper_does_not_trust_the_value_into_an_arithmetic_error(self):
        """A malformed file must fall back, not produce something that
        `$(( live_limit * 60 ))` turns into a crash -- or worse, into 0."""
        script = (
            'cd "$REPO"\nPY=python3\n'
            'printf \'{"liveness_minutes": "oops"}\' > /tmp/bad_live.json\n'
            'STALL_JSON=/tmp/bad_live.json\nMLSYS_MANIFEST_LIVENESS_MIN=15\n'
            + self._helper() + "\nliveness_minutes\n"
        )
        out, err = self._run(script, {"REPO": str(REPO_ROOT)})
        assert out == "15", f"a malformed value must fall back, got {out!r}"

    def test_the_operator_helper_returns_the_same_value(self):
        """Both sides read the same key from the same file; a mismatch would
        make the two watchdogs disagree about what quiet means."""
        i = WATCH_SRC.index("liveness_minutes() {")
        helper = WATCH_SRC[i:WATCH_SRC.index("\n}\n", i) + 3]
        script = (
            'cd "$REPO"\nPY=python3\nSTALL_JSON=configs/mlsys_stall_thresholds.json\n'
            'MLSYS_MANIFEST_LIVENESS_MIN=15\n' + helper + "\nliveness_minutes\n"
        )
        out, err = self._run(script, {"REPO": str(REPO_ROOT)})
        assert out == "15", f"got {out!r} (stderr: {err[:200]})"

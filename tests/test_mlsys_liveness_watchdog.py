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
        """The record must carry the measurement AND the failure that forced the
        re-derivation, or the next reader will re-derive it from the same wrong
        assumption (ttft) that made the first version unsafe."""
        why = THRESHOLDS.get("_liveness_why", "")
        assert "6.29" in why, "the measured prefill silence must be in the record"
        assert "2026-10-09" in why, "the incident date must be in the record"
        assert "973" in why or "16.2" in why, (
            "the silent row that tripped the rule must be in the record")
        assert "70 minutes" in why or "70 min" in why, (
            "how long the stopped campaign kept running must be in the record")
        assert "PROGRESS" in why, (
            "the fix (in-row progress) must be recorded as the answer, not a "
            "widened threshold")


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

    def test_a_liveness_stall_stops_the_RUN_with_a_non_zero_exit(self):
        """The watcher's fail-fast path keys on a non-zero rc, so a stall that
        exited 0 would be read as success -- and signalling the manifest shell
        alone does not stop anything, because bash defers the trap until the
        foreground command (the run) returns. Both halves are asserted: the
        branch calls stop_run, and stop_run signals the descendants FIRST and
        escalates to KILL if they survive."""
        i = MANIFEST_SRC.index("LIVENESS name=")
        tail = MANIFEST_SRC[i:i + 1200]
        assert "stop_run" in tail, (
            "the liveness branch must stop the run, not just log")
        fn = MANIFEST_SRC[MANIFEST_SRC.index("stop_run() {"):]
        fn = fn[:fn.index("\n}\n")]
        assert "_descendants" in fn, (
            "stop_run must walk the process tree; the run is a grandchild of "
            "the manifest, so signalling the shell alone stops nothing")
        assert fn.index("_descendants") < fn.index('kill -TERM "$MANIFEST_PID"'), (
            "the children must be signalled BEFORE the shell, or the shell's "
            "deferred trap cannot run until the row finishes")
        assert "kill -KILL" in fn, (
            "a run that ignores TERM must be escalated, not waited on")

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


# ---------------------------------------------------------------------------
# In-row progress (the fix for the 2026-10-09 false stall)
#
# The liveness rule cannot be both tight and safe unless the runner keeps
# talking, and it did not: a whole ROW can be silent. These tests pin the two
# properties that make the fix usable -- it must not perturb the measurements it
# runs beside, and it must actually bound the silence.
# ---------------------------------------------------------------------------

class TestTheProgressLineIsMeasurementNeutral:
    """It runs inside every campaign row's decode loop, so it must not touch the
    device, the collectives, or the clock in a way that changes latency."""

    def _src(self):
        import inspect
        sys.path.insert(0, str(REPO_ROOT))
        from src.models import rasd_inference as ri
        return inspect.getsource(ri._row_heartbeat)

    @pytest.mark.parametrize("attr,why", [
        ("item", "reading a tensor value forces a device sync, which would "
                 "change the per-round latency the campaign measures"),
        ("cpu", "a host copy syncs the device"),
        ("numpy", "a host copy syncs the device"),
        ("tolist", "a host copy syncs the device"),
        ("synchronize", "an explicit device sync stalls the decode loop"),
        ("cuda", "any device call is unnecessary here"),
    ])
    def test_the_progress_path_calls_no_device_method(self, attr, why):
        """Checked on the AST, not the text: the docstring SAYS it does none of
        these things, and a text search would match its own prose."""
        import ast
        tree = ast.parse(self._src())
        for node in ast.walk(tree):
            if isinstance(node, ast.Attribute):
                assert node.attr != attr, f"the progress path calls .{attr}(): {why}"

    @pytest.mark.parametrize("mod,why", [
        ("dist", "a collective inside the decode loop would be a new "
                 "synchronisation point between ranks"),
        ("torch", "any device interaction is unnecessary here"),
    ])
    def test_the_progress_path_imports_no_collective_module(self, mod, why):
        import ast
        tree = ast.parse(self._src())
        for node in ast.walk(tree):
            if isinstance(node, ast.Name):
                assert node.id != mod, f"the progress path references {mod}: {why}"

    def test_the_progress_path_has_no_call_to_a_collective_helper(self):
        import ast
        tree = ast.parse(self._src())
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                assert node.func.id not in (
                    "barrier", "all_reduce", "broadcast", "all_gather_object",
                    "all_gather", "synchronize"), (
                    f"the progress path calls {node.func.id}()")

    def test_the_call_sites_are_gated_on_the_round_count(self):
        """The gating is what keeps the clock read off most rounds. If it were
        removed the helper would still work, so only a source assertion can
        notice, and the cost would then be paid on every round of every row."""
        sys.path.insert(0, str(REPO_ROOT))
        from src.models import rasd_inference as ri
        import inspect
        src = inspect.getsource(ri.RASDInference.generate)
        gated = src.count("if n_rounds % HEARTBEAT_EVERY_ROUNDS == 0:")
        assert gated == 2, (
            f"expected both decode loops to gate the heartbeat, found {gated}")

    def test_it_reads_only_host_side_counters(self):
        import ast
        src = self._src()
        assert "time.perf_counter()" in src
        tree = ast.parse(src)
        args = {a.arg for a in tree.body[0].args.args}
        assert {"rank", "n_rounds", "tokens", "t_start"} <= args, (
            "the helper should receive counts, not tensors: the token count is "
            "computed at the call site, where it is already gated")
        for node in ast.walk(tree):
            if isinstance(node, ast.Attribute):
                assert node.attr != "shape", (
                    "the helper must not touch tensors at all")


class TestTheProgressLineBehaviour:
    def _call(self, *a, **kw):
        import io, contextlib
        sys.path.insert(0, str(REPO_ROOT))
        from src.models import rasd_inference as ri
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            ri._row_heartbeat(*a, **kw)
        return buf.getvalue()

    def test_rank_zero_only(self):
        import types
        cfg = types.SimpleNamespace(run_id="r")
        state = {}
        out = self._call(cfg, 1, 32, 5, 0.0, state, every_s=0.0, every_rounds=1)
        assert out == "", "a non-zero rank printed a progress line"
        assert state == {}, "a non-zero rank mutated the cadence state"

    def test_it_prints_once_the_cadence_has_elapsed(self):
        import time, types
        cfg = types.SimpleNamespace(run_id="r")
        state = {}
        first = self._call(cfg, 0, 16, 5, time.perf_counter(), state,
                           every_s=1e9, every_rounds=1)
        assert first == "", "printed before the cadence elapsed"
        second = self._call(cfg, 0, 32, 5, time.perf_counter(), state,
                            every_s=0.0, every_rounds=1)
        assert "[PROGRESS run=r round=32 tokens=5" in second, second

    def test_a_non_multiple_round_does_not_even_read_the_clock(self):
        """The modulo gate is the difference between ~0.07 us per round and
        ~1 us per round, and it is also what the source test asserts. Behaviour
        that proves the gate is live: with a zero cadence (always due), a round
        that is not a multiple of the cadence still prints nothing."""
        import types
        cfg = types.SimpleNamespace(run_id="r")
        state = {}
        assert self._call(cfg, 0, 17, 5, 0.0, state, every_s=0.0,
                          every_rounds=16) == "", "the modulo gate is not live"
        assert self._call(cfg, 0, 32, 5, 0.0, state, every_s=0.0,
                          every_rounds=16) != "", "the gate blocked a due round"

    def test_the_state_it_returns_is_the_one_it_was_given(self):
        import types
        cfg = types.SimpleNamespace(run_id="r")
        state = {}
        sys.path.insert(0, str(REPO_ROOT))
        from src.models import rasd_inference as ri
        returned = ri._row_heartbeat(cfg, 0, 16, 5, 0.0, state, every_s=0.0)
        assert returned is state


class TestTheCadenceBoundsTheSilence:
    def test_the_worst_case_silence_is_inside_the_liveness_threshold(self):
        """The two constants and the threshold are one design. If a future
        change raises HEARTBEAT_EVERY_ROUNDS or lowers liveness_minutes without
        the other, a legitimate row can go silent for longer than the watchdog
        waits -- which is precisely how the 2026-10-09 false stall happened."""
        import json
        sys.path.insert(0, str(REPO_ROOT))
        from src.models import rasd_inference as ri
        thr_min = json.loads(
            (REPO_ROOT / "configs/mlsys_stall_thresholds.json").read_text()
        )["liveness_minutes"]
        # Slowest legitimate round in the campaign: a 1M-context row, measured
        # at ~32 s. The armed stages top out at 256k, far below this.
        slowest_round_s = 32.0
        worst = ((ri.HEARTBEAT_EVERY_ROUNDS - 1) * slowest_round_s
                 + ri.HEARTBEAT_EVERY_S)
        assert worst < thr_min * 60, (
            f"worst-case silence {worst:.0f}s is not inside the {thr_min}-minute "
            f"liveness threshold")

    def test_the_round_gate_is_at_least_16(self):
        """The operator's floor: a cadence tighter than every 16 rounds buys
        nothing a 60 s clock check does not already buy, and costs more."""
        sys.path.insert(0, str(REPO_ROOT))
        from src.models import rasd_inference as ri
        assert ri.HEARTBEAT_EVERY_ROUNDS >= 16

"""An aborted run's ROWS must come home, not just the reason it aborted.

The 2026-10-08T14:23Z gate run measured all nine controls, wrote
`gate_calibration.csv` and nine generated continuations, was refused by its own
controls, and the watcher captured the logs, terminated the instance and lost
the CSV — the pod stopped answering ssh within about 30 seconds. The logs carried
the reason; the rows were gone. Planning the next attempt against a reason
without the numbers cost a whole re-run of the gate.

Two halves are tested here:

  * `collect_incident` in `scripts/mlsys_watch_and_run.sh` actually pulls
    `results/mlsys/` before any termination, copies CSVs and generated-text
    directories into `results/` where nothing is there yet, never overwrites an
    existing local result, and merges the one accumulating file (the cost
    ledger) instead of replacing it;
  * the merged output is checked by running the function itself, with `ssh` and
    `rsync` stubbed, rather than by asserting on its shape.
"""
from __future__ import annotations

import os
import shutil
import stat
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
WATCH = REPO / "scripts" / "mlsys_watch_and_run.sh"
MERGE = REPO / "scripts" / "mlsys_merge_cost_ledger.py"

HEADER = "stage,wall_seconds,nproc,gpu_hours,node_cost_usd"


def _body() -> str:
    """The text of `collect_incident`, from its definition to its closing brace."""
    text = WATCH.read_text()
    start = text.index("collect_incident() {")
    # +2 keeps the "\n}" that closes the function; dropping it yields a body
    # whose brace is unbalanced, which is a syntax error, not a test failure.
    end = text.index("\n}\n", start) + 2
    return text[start:end]


# --------------------------------------------------------------------------
# structural: the pull exists, and the ordering that makes it matter
# --------------------------------------------------------------------------

def test_collect_incident_pulls_results_not_only_logs():
    body = _body()
    assert "results/mlsys/" in body, (
        "collect_incident does not pull results/mlsys at all; an aborted run's "
        "rows would be lost with the instance, which is what happened on "
        "2026-10-08T14:23Z")
    assert "pod_results" in body
    assert "-e \"ssh $SSH_OPTS\"" in body, "the results pull is not over ssh"


def test_the_pull_takes_every_csv_and_the_generated_text():
    body = _body()
    assert "-name '*.csv'" in body, "the pull does not select the CSVs"
    assert "-name '*_generated'" in body, (
        "the pull does not select the gate's generated continuations; the CSV "
        "alone does not say what was generated")


def test_the_pull_is_time_bounded_and_skips_the_bulky_directories():
    body = _body()
    # Termination is what this path exists for; a wedged transfer must not hold
    # it open.
    assert "--timeout=" in body
    assert "--exclude 'checkpoints/'" in body
    assert "--exclude 'attempts/'" in body


def test_the_cost_ledger_is_merged_never_replaced():
    body = _body()
    assert "mlsys_merge_cost_ledger.py" in body, (
        "the pod's cost ledger would be replaced or dropped; it accumulates, so "
        "it has to be merged")
    assert "--local" in body and "--pod" in body
    assert 'case "$rel" in gpu_hours.csv) continue' in body, (
        "the ledger is also going through the copy-if-absent branch, which "
        "would either skip it or clobber it")


def test_no_pull_can_overwrite_an_existing_local_result():
    body = _body()
    assert '[ -e "$REPO/results/mlsys/$rel" ]' in body, (
        "the merge into results/ is not guarded by an existence test; a failed "
        "attempt could overwrite a good local result")
    # The copy must be in the `else` branch: guarded, not merely attempted.
    guard = body.index('[ -e "$REPO/results/mlsys/$rel" ]')
    copy = body.index('cp -R "$src" "$REPO/results/mlsys/$rel"')
    assert copy > guard


def test_every_incident_collects_before_it_terminates():
    """Ordering, not presence: a pull after the instance is gone is worthless."""
    text = WATCH.read_text()
    calls = [i for i in range(len(text))
             if text.startswith("collect_incident \"", i)]
    assert calls, "the watcher never calls collect_incident"
    terminate = text.index("terminate_and_confirm\n  exit 7")
    for call in calls:
        assert call < terminate, (
            "a collect_incident call site sits after the abort path's "
            "terminate_and_confirm; it would pull from a dead instance")


# --------------------------------------------------------------------------
# executable: run the real function, with ssh and rsync stubbed
# --------------------------------------------------------------------------

def _stub_bin(tmp: Path, rsync_rc: int = 0) -> Path:
    """A PATH directory holding `ssh` and `rsync` stand-ins."""
    b = tmp / "bin"
    b.mkdir(parents=True, exist_ok=True)
    (b / "ssh").write_text(
        '#!/bin/bash\n'
        'for a in "$@"; do last="$a"; done\n'
        'case "$last" in\n'
        '  "test -f ~/"*) f="$FAKE_HOME/${last#test -f ~/}"\n'
        '                 [ -f "$f" ] && exit 0 || exit 1 ;;\n'
        '  "cat ~/"*)     cat "$FAKE_HOME/${last#cat ~/}" 2>/dev/null; exit 0 ;;\n'
        'esac\n'
        'exit 0\n')
    (b / "rsync").write_text(
        '#!/bin/bash\n'
        f'[ {rsync_rc} -ne 0 ] && exit {rsync_rc}\n'
        'for a in "$@"; do last="$a"; done\n'
        'mkdir -p "$last"\n'
        'cp -R "$POD_SRC/." "$last/"\n'
        'exit 0\n')
    for name in ("ssh", "rsync"):
        (b / name).chmod((b / name).stat().st_mode | stat.S_IEXEC)
    return b


def _run_collect_incident(tmp: Path, rsync_rc: int = 0) -> Path:
    """Execute the production `collect_incident` in a throwaway repo."""
    repo = tmp / "repo"
    (repo / "results" / "mlsys").mkdir(parents=True)

    # What the pod holds.
    pod = tmp / "pod_results"
    (pod / "gate_calibration_generated").mkdir(parents=True)
    (pod / "gate_calibration.csv").write_text(HEADER.replace("stage", "cand")
                                              + "\nP1,2,1,0.5,1.0\n")
    (pod / "gpu_hours.csv").write_text(
        HEADER + "\nARM2,900,8,2.0,44.64\n")
    (pod / "coherence_gate.csv").write_text("from,the,pod\n")
    (pod / "gate_calibration_generated" / "gen_P1.txt").write_text("pod text\n")

    # What is already local: an older result that must survive, and a ledger.
    (repo / "results" / "mlsys" / "coherence_gate.csv").write_text("local,rows\n")
    (repo / "results" / "mlsys" / "gpu_hours.csv").write_text(
        HEADER + "\nARM1,600,8,1.3,29.0\n")

    home = tmp / "home"
    home.mkdir()
    (home / "manifest.log").write_text("gate table here\n")
    (home / "manifest.rc").write_text("1\n")

    runner = tmp / "runner.sh"
    runner.write_text(
        '#!/bin/bash\n'
        'set -u\n'
        f'REPO="{repo}"\n'
        f'LOG="{tmp}/watcher.log"\n'
        'touch "$LOG"\n'
        f'FOUND="{tmp}/region"\n'
        'echo us-west-2 > "$FOUND"\n'
        'SSH_USER=ubuntu\n'
        'SSH_OPTS="-o StrictHostKeyChecking=no"\n'
        'IP=1.2.3.4\n'
        f'PY="{sys.executable}"\n'
        'INSTANCE_ID=i-123\n'
        'MANIFEST_RC=1\n'
        f'export FAKE_HOME="{home}"\n'
        f'export POD_SRC="{pod}"\n'
        f'export PATH="{tmp}/bin:$PATH"\n'
        'say() { echo "  $*"; }\n'
        'mkdir -p "$REPO/scripts"\n'
        f'cp "{MERGE}" "$REPO/scripts/mlsys_merge_cost_ledger.py"\n'
        '\n'
        + _body() + '\n\n'
        'collect_incident "manifest aborted rc=1"\n'
        'echo "INCIDENT_DIR=$(ls -d "$REPO"/results/mlsys/incident_* | head -1)"\n'
        'echo "DONE"\n')
    runner.chmod(runner.stat().st_mode | stat.S_IEXEC)

    _stub_bin(tmp, rsync_rc)
    proc = subprocess.run(["bash", str(runner)], capture_output=True, text=True,
                          timeout=120)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "DONE" in proc.stdout, proc.stdout + proc.stderr
    inc = next((repo / "results" / "mlsys").glob("incident_*"))
    return inc


def test_a_failed_runs_rows_come_home(tmp_path):
    inc = _run_collect_incident(tmp_path)
    local = inc.parent
    assert (inc / "pod_results" / "gate_calibration.csv").exists(), (
        "the gate CSV was not pulled even into the incident directory")
    assert (local / "gate_calibration.csv").read_text().startswith("cand,"), (
        "the failed run's gate CSV did not reach results/mlsys")
    assert (local / "gate_calibration_generated" / "gen_P1.txt").read_text() \
        == "pod text\n", "the generated continuations did not reach results/mlsys"
    assert (inc / "pod_results" / "gate_calibration_generated"
            / "gen_P1.txt").exists()


def test_an_existing_local_result_is_not_overwritten(tmp_path):
    inc = _run_collect_incident(tmp_path)
    local = inc.parent
    assert (local / "coherence_gate.csv").read_text() == "local,rows\n", (
        "the pull overwrote a local result with the pod's copy of the same name")
    assert (inc / "pod_results" / "coherence_gate.csv").read_text() \
        == "from,the,pod\n", "the pod's copy was dropped instead of kept"


def test_the_ledger_gains_the_pods_rows_and_keeps_its_own(tmp_path):
    inc = _run_collect_incident(tmp_path)
    ledger = (inc.parent / "gpu_hours.csv").read_text()
    assert "ARM1,600,8,1.3,29.0" in ledger, "the local ledger row was lost"
    assert "ARM2,900,8,2.0,44.64" in ledger, (
        "the aborted run's spend was not merged into the ledger")
    assert ledger.splitlines()[0] == HEADER


def test_a_failed_pull_says_so_and_still_finishes(tmp_path):
    """A dead pod must not stop the incident path from completing."""
    inc = _run_collect_incident(tmp_path, rsync_rc=1)
    assert (inc / "incident.txt").exists()
    assert not (inc.parent / "gate_calibration.csv").exists()

"""The session driver must pull the directory the session actually wrote.

WHAT THIS PREVENTS. All three of the driver's result paths named
`results/mlsys/session_a/` literally while the stage CSVs live in `$SESSION_OUT`.
Every session that overrode SESSION_OUT therefore pulled a directory the remote
script never wrote. On 2026-10-12 that was the 1x n=5 rehearsal: its pulled
directory held only the COMMITTED Oct-10 leftovers (mtime 18:13) that the repo rsync
had copied onto the pod, the real stage CSVs stayed on the pod, and terminating the
instance destroyed them. The rehearsal had to be run again -- cheap at $1.99/hr,
which is exactly why this was caught on a 1x box and not on the 8x one at $22.32/hr.

The assertion is on the argv handed to rsync, captured by a stub on PATH, so it is
about the command that ran rather than about the driver's own log line.
"""
from __future__ import annotations

import os
import shutil
import stat
import subprocess
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
DRIVER = REPO / "scripts" / "mlsys_session_a.sh"

CURL_STUB = '''\
#!/bin/sh
url=""
for a in "$@"; do case "$a" in http*) url="$a" ;; esac; done
case "$url" in
  *"/instance-types") echo '{"data": {"gpu_1x_a100_sxm4": {"regions_with_capacity_available": [{"name": "us-east-1"}]}}}' ;;
  *"/launch")         echo '{"data": {"instance_ids": ["stub-session-0001"]}}' ;;
  *"/instances/stub-session-0001")
      if [ -f "$CURL_STATE" ]; then
        echo '{"data": {"id": "stub-session-0001", "status": "terminated"}}'
      else
        echo '{"data": {"id": "stub-session-0001", "status": "active", "ip": "10.0.0.7"}}'
      fi ;;
  *"/terminate")      : > "$CURL_STATE"; echo '{}' ;;
  *"/instances")      echo '{"data": []}' ;;
  *) echo "STUB CURL: unexpected URL $url" >&2; exit 9 ;;
esac
'''

SSH_STUB = '''\
#!/bin/sh
for a in "$@"; do last="$a"; done
case "$last" in
  true) ;;
  *"rds/DONE"*) echo "MARKER 2026-01-01T00:00:00Z 0" ;;
  *) : ;;
esac
exit 0
'''

RSYNC_STUB = '''\
#!/bin/sh
printf '%s\\n' "$*" >> "$RSYNC_ARGV_LOG"
exit 0
'''


def _sandbox(tmp_path: Path) -> tuple[Path, dict, Path]:
    repo = tmp_path / "repo"
    (repo / "scripts").mkdir(parents=True)
    (repo / "configs").mkdir()
    shutil.copy(DRIVER, repo / "scripts" / "mlsys_session_a.sh")
    probe = repo / "scripts" / "mlsys_setup_probe.sh"
    probe.write_text("#!/usr/bin/env bash\nexit 0\n")
    probe.chmod(probe.stat().st_mode | stat.S_IEXEC)
    (repo / "scripts" / "mlsys_deadman.sh").write_text("#!/bin/sh\nexit 0\n")
    # Existence-checked by the driver before any instance is created.
    (repo / "scripts" / "mlsys_session_a_remote.sh").write_text("#!/bin/sh\nexit 0\n")
    (repo / "scripts" / "mlsys_detach.py").write_text(
        'import sys\np = sys.argv[sys.argv.index("--pidfile") + 1]\n'
        'open(p, "w").write("1")\n')
    (repo / "runpod_creds.md").write_text(
        "LAMBDA_API_KEY = `fake-key-preflight-only`\nHF_TOKEN = `fake-token-preflight-only`\n")
    (tmp_path / "bin").mkdir()
    for name, body in (("curl", CURL_STUB), ("ssh", SSH_STUB), ("rsync", RSYNC_STUB)):
        f = tmp_path / "bin" / name
        f.write_text(body)
        f.chmod(f.stat().st_mode | stat.S_IEXEC)
    log = tmp_path / "rsync_argv.log"
    env = dict(os.environ)
    env["PATH"] = f"{tmp_path / 'bin'}{os.pathsep}{os.environ['PATH']}"
    env["MLSYS_SESSION_DIR"] = str(tmp_path / "sess")
    env["RSYNC_ARGV_LOG"] = str(log)
    env["CURL_STATE"] = str(tmp_path / "terminate_called")
    env["TERMINATE_POLL_S"] = "0.1"
    # The incremental puller would otherwise hold the driver's stdout open for 240s.
    env["MLSYS_PULL_EVERY_S"] = "1"
    env["SESSION_TYPE"] = "gpu_1x_a100_sxm4"
    env["SESSION_RATE"] = "1.99"
    env["TIMER_MIN"] = "5"
    env["SESSION_TAG"] = "unit_tag"
    return repo, env, log


def test_the_pull_targets_the_session_output_directory(tmp_path):
    repo, env, log = _sandbox(tmp_path)
    env["SESSION_OUT"] = "results/mlsys/n5_rehearsal"
    proc = subprocess.run(["bash", str(repo / "scripts" / "mlsys_session_a.sh")],
                          cwd=str(repo), env=env, capture_output=True, text=True,
                          timeout=600)
    argv = log.read_text() if log.exists() else ""
    assert "results/mlsys/n5_rehearsal/" in argv, (
        "the driver pulled something else; the session's own output would be lost "
        f"on terminate:\n{argv}")
    assert "results/mlsys/session_a/" not in argv, (
        f"a hardcoded session_a path survived:\n{argv}")
    assert "final pull (pod: results/mlsys/n5_rehearsal)" in proc.stdout, proc.stdout


def test_the_default_session_output_still_works(tmp_path):
    """The control: the default value must be pulled exactly as before."""
    repo, env, log = _sandbox(tmp_path)
    env.pop("SESSION_OUT", None)
    subprocess.run(["bash", str(repo / "scripts" / "mlsys_session_a.sh")],
                   cwd=str(repo), env=env, capture_output=True, text=True, timeout=600)
    argv = log.read_text() if log.exists() else ""
    assert "results/mlsys/session_a/" in argv, argv

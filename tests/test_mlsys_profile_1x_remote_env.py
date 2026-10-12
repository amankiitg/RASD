"""The driver-to-pod env contract: what is actually in the remote command.

WHAT THIS PREVENTS, AND WHY THE EXISTING TESTS DID NOT. On 2026-10-12 the vLLM 8x
run was launched with `REHEARSE_TP=8`, `REHEARSE_CTXS='131072 524288 1048576'` and
`REHEARSE_PROMPT_DIR=samp_prompts`, but the driver built its remote command with only
`PROFILE_OUT`/`PROFILE_TAG`/`PROFILE_NPROC`. The pod therefore started the wrapper
with ITS OWN DEFAULTS -- one card, 8192 tokens, a synthetic prompt -- on an 8-GPU
box, and burned ~$5.5 before the CSV made it obvious.

tests/test_mlsys_vllm_rehearsal.py could not see it: it invokes the wrapper DIRECTLY,
so it proves the wrapper forwards what it is GIVEN, never that the driver gave it
anything (L12 -- testing the function is not calling it). The gap was the boundary
itself, so the test has to span the boundary: this drives the real driver with stub
`curl`, `ssh` and `rsync` on PATH, then reads the command the driver actually handed
to ssh.

The stub `ssh` writes its argv to a file, so the assertion is on the bytes that would
have gone to the pod, not on the driver's own log line about them.
"""
from __future__ import annotations

import json
import os
import shutil
import stat
import subprocess
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
DRIVER = REPO / "scripts" / "mlsys_profile_1x.sh"

CURL_STUB = '''\
#!/bin/sh
# The URL is not the last argv: a POST carries -d '{...}' after it. Scan for it.
url=""
for a in "$@"; do case "$a" in http*) url="$a" ;; esac; done
case "$url" in
  *"/instances")
      # The preflight must see zero instances; after the terminate, also zero.
      echo '{"data": []}' ;;
  *"/instance-types")
      echo '{"data": {"gpu_1x_a100_sxm4": {"regions_with_capacity_available": [{"name": "us-east-1"}]}}}' ;;
  *"/launch")
      echo '{"data": {"instance_ids": ["stub-instance-0001"]}}' ;;
  *"/instances/stub-instance-0001")
      if [ -f "$CURL_STATE" ]; then
        echo '{"data": {"id": "stub-instance-0001", "status": "terminated"}}'
      else
        echo '{"data": {"id": "stub-instance-0001", "status": "active", "ip": "10.0.0.9"}}'
      fi ;;
  *"/terminate")
      : > "$CURL_STATE"; echo '{}' ;;
  *) echo "STUB CURL: unexpected URL $url" >&2; exit 9 ;;
esac
'''

# Logs argv, then answers the way each call site expects. The poll call must return a
# MARKER so the driver finishes in seconds instead of polling for 145 minutes.
SSH_STUB = '''\
#!/bin/sh
printf '%s\\n' "$*" >> "$SSH_ARGV_LOG"
for a in "$@"; do last="$a"; done
case "$last" in
  true) ;;                                             # ssh readiness probe
  *"$PODOUT/DONE"*|*"pgrep -f"*) echo "MARKER $(date -u +%FT%TZ) 0" ;;
  *) : ;;
esac
exit 0
'''

RSYNC_STUB = '#!/bin/sh\nexit 0\n'


def _sandbox(tmp_path: Path) -> tuple[Path, dict]:
    repo = tmp_path / "repo"
    (repo / "scripts").mkdir(parents=True)
    (repo / "configs").mkdir()
    shutil.copy(DRIVER, repo / "scripts" / "mlsys_profile_1x.sh")
    shutil.copy(REPO / "scripts" / "mlsys_vllm_remote.sh",
                repo / "scripts" / "mlsys_vllm_remote.sh")
    # Provisioning is stubbed: it rsyncs and runs the pod's installer, which is not
    # what this test is about and would need a real host.
    probe = repo / "scripts" / "mlsys_setup_probe.sh"
    probe.write_text("#!/usr/bin/env bash\nexit 0\n")
    probe.chmod(probe.stat().st_mode | stat.S_IEXEC)
    # The timer is stubbed out: this test is about the remote command, and the real
    # dead-man would poll the API for an hour.
    (repo / "scripts" / "mlsys_deadman.sh").write_text("#!/bin/sh\nexit 0\n")
    (repo / "scripts" / "mlsys_detach.py").write_text(
        'import sys\n'
        'p = sys.argv[sys.argv.index("--pidfile") + 1]\n'
        'open(p, "w").write("1")\n')
    (repo / "runpod_creds.md").write_text(
        "LAMBDA_API_KEY = `fake-key-preflight-only`\nHF_TOKEN = `fake-token-preflight-only`\n")
    (tmp_path / "bin").mkdir()
    for name, body in (("curl", CURL_STUB), ("ssh", SSH_STUB), ("rsync", RSYNC_STUB)):
        f = tmp_path / "bin" / name
        f.write_text(body)
        f.chmod(f.stat().st_mode | stat.S_IEXEC)
    env = dict(os.environ)
    env["PATH"] = f"{tmp_path / 'bin'}{os.pathsep}{os.environ['PATH']}"
    env["MLSYS_SESSION_DIR"] = str(tmp_path / "sess")
    env["SSH_ARGV_LOG"] = str(tmp_path / "ssh_argv.log")
    env["PODOUT"] = "RASD/results/mlsys/stub_out"
    env["CURL_STATE"] = str(tmp_path / "terminate_called")
    env["TERMINATE_POLL_S"] = "0.1"
    env["PROFILE_OUT"] = "results/mlsys/stub_out"
    env["PROFILE_TAG"] = "stub"
    env["TIMER_MIN"] = "5"
    return repo, env


def _run_driver(tmp_path: Path, extra_env: dict):
    repo, env = _sandbox(tmp_path)
    env.update(extra_env)
    proc = subprocess.run(["bash", str(repo / "scripts" / "mlsys_profile_1x.sh")],
                          cwd=str(repo), env=env, capture_output=True, text=True,
                          timeout=600)
    argv_log = Path(env["SSH_ARGV_LOG"])
    return proc, (argv_log.read_text() if argv_log.exists() else "")


def _start_cmd(argv_log: str) -> str:
    """The one ssh argv that carries the remote launch."""
    for line in argv_log.splitlines():
        if "setsid nohup" in line:
            return line
    raise AssertionError(f"no remote-launch command was sent:\n{argv_log}")


def test_named_remote_env_vars_reach_the_pod_command(tmp_path):
    """THE REGRESSION. These three are the vLLM comparison's whole difference from
    the rehearsal, and they were silently dropped."""
    proc, argv_log = _run_driver(tmp_path, {
        "REMOTE_SCRIPT": "scripts/mlsys_vllm_remote.sh",
        "REMOTE_ENV_NAMES": "REHEARSE_TP REHEARSE_CTXS REHEARSE_PROMPT_DIR",
        "REHEARSE_TP": "8",
        "REHEARSE_CTXS": "131072 524288 1048576",
        "REHEARSE_PROMPT_DIR": "samp_prompts",
    })
    cmd = _start_cmd(argv_log)
    # The terminate check asks about THIS id and must see it terminated.
    assert "status after terminate: terminated (terminated required)" in proc.stdout,         proc.stdout[-2000:]
    assert "REHEARSE_TP='8'" in cmd, cmd
    assert "REHEARSE_CTXS='131072 524288 1048576'" in cmd, cmd
    assert "REHEARSE_PROMPT_DIR='samp_prompts'" in cmd, cmd
    assert "bash scripts/mlsys_vllm_remote.sh" in cmd, cmd
    assert proc.returncode == 0, proc.stdout


def test_without_the_allowlist_nothing_extra_is_sent(tmp_path):
    """The control, and the description of the old behaviour: no list, no vars.

    A run that forgets REMOTE_ENV_NAMES now fails loudly on the pod side instead of
    quietly running the wrapper's defaults only because the wrapper is written to
    require the same names -- which is why the wrapper also refuses nothing silently.
    """
    proc, argv_log = _run_driver(tmp_path, {
        "REMOTE_SCRIPT": "scripts/mlsys_vllm_remote.sh",
        "REHEARSE_TP": "8",
    })
    cmd = _start_cmd(argv_log)
    assert "REHEARSE_TP" not in cmd, cmd
    assert proc.returncode == 0, proc.stdout


def test_a_named_but_unset_variable_is_refused_before_anything_is_launched(tmp_path):
    """Forwarding '' would let the pod fall back to its own default -- exactly the
    silent shape of the mistake this allowlist exists for."""
    proc, argv_log = _run_driver(tmp_path, {
        "REMOTE_ENV_NAMES": "REHEARSE_TP REHEARSE_NOT_SET",
        "REHEARSE_TP": "8",
    })
    assert proc.returncode == 2, proc.stdout
    assert "REHEARSE_NOT_SET but it is unset or empty" in proc.stdout
    assert argv_log == "", "the refusal must happen before any ssh call"

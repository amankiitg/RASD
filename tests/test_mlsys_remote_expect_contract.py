"""The parameter contract, tested ACROSS the driver-to-pod boundary.

WHY THIS TEST LOOKS LIKE THIS. Two cheaper tests already existed and neither could
have caught the 2026-10-12 failure ($5.47, an 8-GPU box running one card at 8192
tokens with a synthetic prompt):

  * a wrapper test invokes the remote script DIRECTLY, so it proves the remote honours
    the variables it is given and nothing about what the driver gives it;
  * a unit test of the comparison helper proves the comparison works and nothing about
    where its inputs come from.

The failure was in the seam. So this test spans it: it runs the REAL driver with stub
`curl`/`ssh`/`rsync` on PATH, captures the remote command the driver actually sent,
extracts the environment assignments from that command, and then runs the REAL remote
script with exactly those assignments -- once complete, and once with one variable
removed on purpose. The removed-variable case is the one that must refuse.
"""
from __future__ import annotations

import os
import re
import shutil
import stat
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
DRIVER = REPO / "scripts" / "mlsys_profile_1x.sh"
REMOTE = REPO / "scripts" / "mlsys_vllm_remote.sh"
PROFILE_REMOTE = REPO / "scripts" / "mlsys_profile_remote.sh"
EXPECT = REPO / "scripts" / "mlsys_remote_expect.sh"

CURL_STUB = '''\
#!/bin/sh
url=""
for a in "$@"; do case "$a" in http*) url="$a" ;; esac; done
case "$url" in
  *"/instance-types") echo '{"data": {"gpu_1x_a100_sxm4": {"regions_with_capacity_available": [{"name": "us-east-1"}]}}}' ;;
  *"/launch")         echo '{"data": {"instance_ids": ["stub-contract-1"]}}' ;;
  *"/instances/stub-contract-1")
      if [ -f "$CURL_STATE" ]; then
        echo '{"data": {"id": "stub-contract-1", "status": "terminated"}}'
      else
        echo '{"data": {"id": "stub-contract-1", "status": "active", "ip": "10.0.0.5"}}'
      fi ;;
  *"/terminate")      : > "$CURL_STATE"; echo '{}' ;;
  *"/instances")      echo '{"data": []}' ;;
  *) exit 9 ;;
esac
'''
SSH_STUB = '''\
#!/bin/sh
printf '%s\\n' "$*" >> "$SSH_ARGV_LOG"
for a in "$@"; do last="$a"; done
case "$last" in
  true) ;;
  *"DONE"*) echo "MARKER 2026-01-01T00:00:00Z 0" ;;
  *) : ;;
esac
exit 0
'''
RSYNC_STUB = '#!/bin/sh\nexit 0\n'


def _driver_env(tmp_path: Path, remote_script: str = "scripts/mlsys_vllm_remote.sh",
                env_names: str = "REHEARSE_TP REHEARSE_CTXS REHEARSE_PROMPT_DIR",
                extra_env: dict | None = None) -> tuple[Path, dict]:
    repo = tmp_path / "repo"
    (repo / "scripts").mkdir(parents=True)
    shutil.copy(DRIVER, repo / "scripts" / "mlsys_profile_1x.sh")
    shutil.copy(REMOTE, repo / "scripts" / "mlsys_vllm_remote.sh")
    shutil.copy(PROFILE_REMOTE, repo / "scripts" / "mlsys_profile_remote.sh")
    shutil.copy(EXPECT, repo / "scripts" / "mlsys_remote_expect.sh")
    probe = repo / "scripts" / "mlsys_setup_probe.sh"
    probe.write_text("#!/usr/bin/env bash\nexit 0\n")
    probe.chmod(probe.stat().st_mode | stat.S_IEXEC)
    (repo / "scripts" / "mlsys_deadman.sh").write_text("#!/bin/sh\nexit 0\n")
    (repo / "scripts" / "mlsys_detach.py").write_text(
        'import sys\np = sys.argv[sys.argv.index("--pidfile") + 1]\n'
        'open(p, "w").write("1")\n')
    (repo / "runpod_creds.md").write_text(
        "LAMBDA_API_KEY = `fake-key-contract-only`\nHF_TOKEN = `fake-token-contract-only`\n")
    (tmp_path / "bin").mkdir()
    for name, body in (("curl", CURL_STUB), ("ssh", SSH_STUB), ("rsync", RSYNC_STUB)):
        f = tmp_path / "bin" / name
        f.write_text(body)
        f.chmod(f.stat().st_mode | stat.S_IEXEC)
    env = dict(os.environ)
    env["PATH"] = f"{tmp_path / 'bin'}{os.pathsep}{os.environ['PATH']}"
    env["MLSYS_SESSION_DIR"] = str(tmp_path / "sess")
    env["SSH_ARGV_LOG"] = str(tmp_path / "ssh_argv.log")
    env["CURL_STATE"] = str(tmp_path / "terminated")
    env["TERMINATE_POLL_S"] = "0.1"
    env["PROFILE_OUT"] = "results/mlsys/contract_out"
    env["PROFILE_TAG"] = "contract"
    env["TIMER_MIN"] = "5"
    env["REMOTE_SCRIPT"] = remote_script
    env["REMOTE_ENV_NAMES"] = env_names
    env["REHEARSE_TP"] = "8"
    env["REHEARSE_CTXS"] = "131072 524288 1048576"
    env["REHEARSE_PROMPT_DIR"] = "samp_prompts"
    env.update(extra_env or {})
    return repo, env


def _sent_env(tmp_path: Path, **kwargs) -> dict:
    """Run the real driver and return the env assignments it put in the ssh command."""
    repo, env = _driver_env(tmp_path, **kwargs)
    subprocess.run(["bash", str(repo / "scripts" / "mlsys_profile_1x.sh")],
                   cwd=str(repo), env=env, capture_output=True, text=True, timeout=600)
    argv = Path(env["SSH_ARGV_LOG"]).read_text()
    cmd = next(l for l in argv.splitlines() if "setsid nohup" in l)
    # `env NAME='value' ...` -- values never contain a single quote, and the driver
    # refuses to forward one that does.
    return dict(re.findall(r"(?:^| )([A-Z_][A-Z0-9_]*)='([^']*)'", cmd))


def _pod_sandbox(tmp_path: Path) -> tuple[Path, dict]:
    home = tmp_path / "pod"
    rasd = home / "RASD"
    (rasd / "scripts").mkdir(parents=True)
    shutil.copy(REMOTE, rasd / "scripts" / "mlsys_vllm_remote.sh")
    shutil.copy(EXPECT, rasd / "scripts" / "mlsys_remote_expect.sh")
    venv = rasd / "scripts" / "mlsys_vllm_venv.sh"
    venv.write_text("#!/usr/bin/env bash\necho 'STUB VENV ok'\nexit 0\n")
    venv.chmod(venv.stat().st_mode | stat.S_IEXEC)
    (rasd / "scripts" / "mlsys_vllm_baseline.py").write_text('''
import csv, os, sys
argv = sys.argv[1:]
out = argv[argv.index("--out") + 1]
with open(out, "w", newline="") as fh:
    w = csv.writer(fh)
    w.writerow(["model", "context_length", "max_model_len", "tensor_parallel_size",
                "attempt", "config_used", "status", "end_to_end_wall_s",
                "output_tokens", "throughput_tps_end_to_end",
                "throughput_tps_decode_only", "kv_dtype", "weight_precision",
                "unit_matched", "error_class", "error"])
    w.writerow(["m", 131072, 131072, 8, 1, "1-default-tp8", "ok", 2.0, 128, 64.0,
                "", "bfloat16", "bfloat16", "no", "", ""])
''')
    (home / "pod_env.log").write_text(f"INTERPRETER={sys.executable}\n")
    (home / ".hf_token").write_text("stub\n")
    (rasd / ".venv-vllm" / "bin").mkdir(parents=True)
    base = rasd / ".venv-vllm" / "bin" / "python"
    base.write_text(f'#!/bin/sh\nexec "{sys.executable}" "$@"\n')
    base.chmod(base.stat().st_mode | stat.S_IEXEC)
    env = dict(os.environ)
    env["HOME"] = str(home)
    return home, env


def _run_remote(tmp_path: Path, sent: dict, drop: str = "") -> subprocess.CompletedProcess:
    home, env = _pod_sandbox(tmp_path)
    for k, v in sent.items():
        if k == drop:
            continue
        env[k] = v
    return subprocess.run(
        ["bash", str(home / "RASD" / "scripts" / "mlsys_vllm_remote.sh")],
        cwd=str(home / "RASD"), env=env, capture_output=True, text=True, timeout=600)


def test_the_drivers_own_command_verifies_on_the_pod(tmp_path):
    sent = _sent_env(tmp_path)
    # The three settings that went missing on 2026-10-12 are IN the command.
    assert sent["REHEARSE_TP"] == "8"
    assert sent["REHEARSE_CTXS"] == "131072 524288 1048576"
    assert sent["REHEARSE_PROMPT_DIR"] == "samp_prompts"
    proc = _run_remote(tmp_path, sent)
    out = proc.stdout + proc.stderr
    assert "expected setting(s) verified" in out, out[-3000:]
    assert "refusing to start" not in out


def test_a_variable_missing_from_the_command_is_refused_on_the_pod(tmp_path):
    """THE CASE THAT MATTERS: the driver sent it, the pod never got it, and the pod
    must say so BEFORE loading a model or installing anything."""
    sent = _sent_env(tmp_path)
    assert "REHEARSE_TP" in sent, "the driver must be sending it for this test to mean anything"
    proc = _run_remote(tmp_path, sent, drop="REHEARSE_TP")
    out = proc.stdout + proc.stderr
    assert proc.returncode == 2, out[-3000:]
    assert "REHEARSE_TP" in out, "the refusal must name the variable"
    assert "never received that setting" in out, out[-3000:]
    assert "STUB VENV ok" not in out, \
        "the refusal must come BEFORE the venv install (that is the heavy work)"
    # THE OUTPUT DIRECTORY AND THE MARKER ARE EXPECTED. The Mac side polls for DONE,
    # so a refusal that wrote none would look like a hang (L9) -- the wrapper creates
    # the directory and arms its trap before it verifies anything. What must not exist
    # is an artifact of the heavy work.
    root = tmp_path / "pod" / "RASD" / "results"
    done = next(root.rglob("DONE"), None)
    assert done is not None, "the refusal must still write its DONE marker"
    assert done.read_text().split()[1] == "2", done.read_text()
    assert not list(root.rglob("vllm_baseline.csv")), \
        "no baseline CSV may exist after a refusal"


def test_a_manifest_that_is_absent_altogether_is_refused(tmp_path):
    """No manifest means the pod cannot verify anything, which is the state that cost
    the money -- so it refuses rather than guessing."""
    sent = _sent_env(tmp_path)
    sent.pop("REMOTE_EXPECT", None)
    proc = _run_remote(tmp_path, sent)
    out = proc.stdout + proc.stderr
    assert proc.returncode == 2, out[-2000:]
    assert "no expectation manifest arrived" in out


# ---- the SAME contract on the other pod-side script -------------------------
# The decode profiler is the other wrapper the driver can start (REMOTE_SCRIPT), and
# it used to report its settings under different key names (OUT/TAG/NPROC) than the
# driver's variables -- so the driver could not have declared an expectation for it
# even if it had tried. One vocabulary now, and this asserts the seam holds there too.
def _profile_pod_sandbox(tmp_path: Path) -> tuple[Path, dict]:
    home = tmp_path / "podprof"
    rasd = home / "RASD"
    (rasd / "scripts").mkdir(parents=True)
    shutil.copy(PROFILE_REMOTE, rasd / "scripts" / "mlsys_profile_remote.sh")
    shutil.copy(EXPECT, rasd / "scripts" / "mlsys_remote_expect.sh")
    # THE HEAVY WORK, stubbed to leave a marker. "Refused before any heavy work" is
    # only worth asserting if the heavy work would have left a trace.
    (rasd / "scripts" / "mlsys_profile_decode.sh").write_text(
        "#!/usr/bin/env bash\ntouch \"$DECODE_MARKER\"\nexit 0\n")
    (rasd / ".pod_env.sh").write_text("")
    (home / "pod_env.log").write_text(f"INTERPRETER={sys.executable}\n")
    (home / ".hf_token").write_text("stub\n")
    env = dict(os.environ)
    env["HOME"] = str(home)
    env["DECODE_MARKER"] = str(tmp_path / "decode_ran")
    return home, env


def _run_profile_remote(tmp_path: Path, sent: dict, drop: str = "") -> subprocess.CompletedProcess:
    home, env = _profile_pod_sandbox(tmp_path)
    for k, v in sent.items():
        if k == drop:
            continue
        env[k] = v
    return subprocess.run(
        ["bash", str(home / "RASD" / "scripts" / "mlsys_profile_remote.sh")],
        cwd=str(home / "RASD"), env=env, capture_output=True, text=True, timeout=600)


def test_the_profile_remote_verifies_the_drivers_command(tmp_path):
    # No extra forwarded names here: the profile wrapper reads only PROFILE_*. (If the
    # allowlist names settings the chosen REMOTE_SCRIPT never reads, the contract
    # refuses -- forwarding a setting nothing consumes is a config error worth
    # surfacing, not a harmless extra.)
    sent = _sent_env(tmp_path, remote_script="scripts/mlsys_profile_remote.sh", env_names="")
    assert sent["PROFILE_OUT"] == "results/mlsys/contract_out"
    assert sent["PROFILE_NPROC"] == "1"
    assert "REMOTE_EXPECT" in sent, "the driver must send a manifest for this to mean anything"
    proc = _run_profile_remote(tmp_path, sent)
    out = proc.stdout + proc.stderr
    assert "expected setting(s) verified" in out, out[-3000:]
    assert (tmp_path / "decode_ran").exists(), \
        "with every setting verified the profiler must go on to do the work"


def test_a_profile_setting_missing_from_the_command_is_refused(tmp_path):
    sent = _sent_env(tmp_path, remote_script="scripts/mlsys_profile_remote.sh", env_names="")
    proc = _run_profile_remote(tmp_path, sent, drop="PROFILE_TAG")
    out = proc.stdout + proc.stderr
    assert proc.returncode == 2, out[-3000:]
    assert "PROFILE_TAG" in out, "the refusal must name the variable"
    assert "never received that setting" in out, out[-3000:]
    assert not (tmp_path / "decode_ran").exists(), \
        "the refusal must come BEFORE the profiler's heavy work (L14/L16: verify by effect)"

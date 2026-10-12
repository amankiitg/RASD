"""The driver must not launch a second billing instance by accident -- and must
still be usable when a second one is sanctioned.

`scripts/mlsys_session_a.sh` owns exactly one instance: it refuses to launch when any
exists, and after terminating it requires the API to read zero. Both guarantees are
what stop a forgotten box from being billed alongside this one.

But a rehearsal is sometimes sanctioned to run WHILE another campaign owns a node: on
2026-10-12 the 1x n=5 rehearsal ran while the vLLM 8x comparison held one, and a
rehearsal that has to wait for the comparison to finish is not a rehearsal that runs
concurrently. The permission is therefore a POSITIVE ASSERTION rather than a bypass:
`EXPECTED_OTHER_INSTANCES` names the instance ids allowed to be present, the API's
list must be EXACTLY that set, and the post-terminate expectation becomes that same
count instead of 0.

Driven against a stub `curl` whose use is PROVED by a marker file: with the real curl
the fake key would 401 and the same exit codes would appear for the wrong reason.
"""
from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
DRIVER = REPO / "scripts" / "mlsys_session_a.sh"

CURL_STUB = '''\
#!/bin/sh
: > "$CURL_STUB_MARKER"
url_last=""
for a in "$@"; do url_last="$a"; done
case "$url_last" in
  *"/instances")      echo "{\\"data\\": $INSTANCES_JSON}" ;;
  *"/instance-types") echo "{\\"data\\": $TYPES_JSON}" ;;
  *"/launch")         echo '{}' ;;
  *) echo "STUB CURL: unexpected URL $url_last" >&2; exit 9 ;;
esac
'''

# One capacity-bearing type, so the capacity poll succeeds and the launch is reached.
TYPES = '{"gpu_1x_a100_sxm4": {"regions_with_capacity_available": [{"name": "us-east-1"}]}}'


def _sandbox(tmp_path: Path) -> tuple[Path, dict]:
    repo = tmp_path / "repo"
    (repo / "scripts").mkdir(parents=True)
    shutil.copy(DRIVER, repo / "scripts" / "mlsys_session_a.sh")
    shutil.copy(REPO / "scripts" / "mlsys_detach.py",
                repo / "scripts" / "mlsys_detach.py")
    shutil.copy(REPO / "scripts" / "mlsys_deadman.sh",
                repo / "scripts" / "mlsys_deadman.sh")
    (repo / "runpod_creds.md").write_text(
        "LAMBDA_API_KEY = `fake-key-preflight-only`\n"
        "HF_TOKEN = `fake-token-preflight-only`\n")
    (tmp_path / "bin").mkdir()
    stub = tmp_path / "bin" / "curl"
    stub.write_text(CURL_STUB)
    stub.chmod(0o755)
    env = dict(os.environ)
    env["PATH"] = f"{tmp_path / 'bin'}{os.pathsep}{os.environ['PATH']}"
    env["MLSYS_SESSION_DIR"] = str(tmp_path / "sess")
    env["TYPES_JSON"] = TYPES
    env["CURL_STUB_MARKER"] = str(tmp_path / "curl_was_the_stub")
    env["SESSION_TYPE"] = "gpu_1x_a100_sxm4"
    env["SESSION_RATE"] = "1.99"
    env["TIMER_MIN"] = "60"
    return repo, env


def _run(tmp_path: Path, instances: str, expected: str = ""):
    repo, env = _sandbox(tmp_path)
    env["INSTANCES_JSON"] = instances
    if expected:
        env["EXPECTED_OTHER_INSTANCES"] = expected
    proc = subprocess.run(["bash", str(repo / "scripts" / "mlsys_session_a.sh")],
                          cwd=str(repo), env=env, capture_output=True, text=True,
                          timeout=300)
    assert (tmp_path / "curl_was_the_stub").exists(), \
        "the real curl answered, so the preflight was never exercised"
    return proc


def test_an_undeclared_instance_still_stops_the_launch(tmp_path):
    proc = _run(tmp_path, '[{"id": "someone-elses-box"}]')
    assert proc.returncode == 3, proc.stdout
    assert "REFUSING" in proc.stdout


def test_a_declared_instance_is_allowed_through_and_then_normal_rules_apply(tmp_path):
    """Declared correctly the launch proceeds; the stub then returns no instance id,
    so the driver stops exactly there. Reaching that stop is the proof it passed the
    guard rather than the guard being unreachable."""
    proc = _run(tmp_path, '[{"id": "the-vllm-8x-run"}]',
                expected="the-vllm-8x-run")
    assert "PROCEEDING: the 1 pre-existing instance(s) are exactly the declared ones" \
        in proc.stdout
    assert "REFUSING" not in proc.stdout
    assert proc.returncode == 5, proc.stdout


def test_a_mismatched_declaration_is_refused(tmp_path):
    """Declaring a different id must not open the door."""
    proc = _run(tmp_path, '[{"id": "some-other-box"}]', expected="the-vllm-8x-run")
    assert proc.returncode == 3, proc.stdout
    assert "REFUSING" in proc.stdout


def test_two_instances_need_two_declarations(tmp_path):
    proc = _run(tmp_path, '[{"id": "a"}, {"id": "b"}]', expected="a")
    assert proc.returncode == 3, proc.stdout
    assert "not the declared" in proc.stdout

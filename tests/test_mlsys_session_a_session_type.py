"""`SESSION_TYPE` is required, and the driver stops before renting anything without it.

The bug this guards against happened for real. On 2026-10-12 the 1x rehearsal picker
exported `SESSION_TYPE` before validating its own selection; when the selection came
back empty, `SESSION_TYPE=${SESSION_TYPE:-gpu_8x_a100_80gb_sxm4}` silently substituted
the 8x type, and a rehearsal budgeted at "<= $3" would have rented eight A100s at
$22.32/hr. It was caught by hand, but nothing in the code made it impossible.

An empty instance type is never a value to be defaulted: it is evidence of a bug
upstream, and the only safe response is to stop before any instance exists. So the
default is gone, an unknown type is refused by name, and the tests below prove the
stop happens BEFORE the first API call -- not merely that some non-zero code appears.
The stub `curl` writes a marker file, so "no API call" is observed rather than assumed;
with the real curl a fake key would 401 and the same exit code would appear for the
wrong reason.
"""
from __future__ import annotations

import os
import re
import shutil
import subprocess
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
DRIVER = REPO / "scripts" / "mlsys_session_a.sh"
PICKER = REPO / "scripts" / "mlsys_rehearsal_pick_1x.sh"

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

# No type has capacity, so every run that reaches the poll ends by itself at the
# deadline instead of waiting. Reaching that end is the proof the gate let it through.
TYPES = '{"gpu_1x_a100_sxm4": {"regions_with_capacity_available": []}}'


def _allowed_types() -> list[str]:
    text = DRIVER.read_text()
    block = re.search(r'ALLOWED_TYPES="(.*?)"', text, re.S)
    assert block, "the driver no longer declares ALLOWED_TYPES"
    # The value spans lines with backslash continuations; those are separators, not types.
    return block.group(1).replace("\\", " ").split()


def _sandbox(tmp_path: Path, session_type: str | None) -> tuple[Path, dict]:
    repo = tmp_path / "repo"
    (repo / "scripts").mkdir(parents=True)
    for name in ("mlsys_session_a.sh", "mlsys_detach.py", "mlsys_deadman.sh"):
        shutil.copy(REPO / "scripts" / name, repo / "scripts" / name)
    (repo / "runpod_creds.md").write_text(
        "LAMBDA_API_KEY = `fake-key-gate-test`\n"
        "HF_TOKEN = `fake-token-gate-test`\n")
    (tmp_path / "bin").mkdir()
    stub = tmp_path / "bin" / "curl"
    stub.write_text(CURL_STUB)
    stub.chmod(0o755)
    env = dict(os.environ)
    env["PATH"] = f"{tmp_path / 'bin'}{os.pathsep}{os.environ['PATH']}"
    env["MLSYS_SESSION_DIR"] = str(tmp_path / "sess")
    env["INSTANCES_JSON"] = "[]"
    env["TYPES_JSON"] = TYPES
    env["CURL_STUB_MARKER"] = str(tmp_path / "curl_was_the_stub")
    env["SESSION_RATE"] = "1.99"
    env["TIMER_MIN"] = "60"
    # A deadline already in the past: no capacity is available above, so the poll ends
    # on its first pass instead of waiting for a card that will never come.
    env["POLL_CUTOFF_UTC"] = "2020-01-01T00:00:00Z"
    env.pop("SESSION_TYPE", None)
    if session_type is not None:
        env["SESSION_TYPE"] = session_type
    return repo, env


def _run(tmp_path: Path, session_type: str | None):
    repo, env = _sandbox(tmp_path, session_type)
    return subprocess.run(["bash", str(repo / "scripts" / "mlsys_session_a.sh")],
                          cwd=str(repo), env=env, capture_output=True, text=True,
                          timeout=300), tmp_path / "curl_was_the_stub"


def test_an_unset_session_type_stops_before_the_first_api_call(tmp_path):
    proc, marker = _run(tmp_path, None)
    assert proc.returncode == 2, (proc.stdout, proc.stderr)
    assert "FATAL: SESSION_TYPE is empty" in proc.stderr
    assert not marker.exists(), "the driver reached the API without an instance type"


def test_an_explicitly_empty_session_type_is_rejected_the_same_way(tmp_path):
    """`SESSION_TYPE=` and an unset variable are the same mistake, and both are the
    shape the picker produced. Neither may fall back to the 8x type."""
    proc, marker = _run(tmp_path, "")
    assert proc.returncode == 2, (proc.stdout, proc.stderr)
    assert "FATAL: SESSION_TYPE is empty" in proc.stderr
    assert not marker.exists(), "the driver reached the API without an instance type"


def test_an_unknown_instance_type_is_refused_by_name(tmp_path):
    proc, marker = _run(tmp_path, "gpu_99x_bogus")
    assert proc.returncode == 2, (proc.stdout, proc.stderr)
    assert "gpu_99x_bogus" in proc.stderr
    assert "not in the allowed list" in proc.stderr
    assert not marker.exists(), "the driver reached the API with an unknown type"


@pytest.mark.parametrize("session_type", _allowed_types())
def test_every_allowed_type_passes_the_gate_and_reaches_the_capacity_poll(
        tmp_path, session_type):
    """The other direction: the guard must not be a wall. Each sanctioned type gets
    through, which is shown by the run ending at the capacity deadline with the
    driver's own NO_CAPACITY marker -- a stop that is only reachable past the gate."""
    proc, marker = _run(tmp_path, session_type)
    assert "FATAL: SESSION_TYPE" not in proc.stderr, proc.stderr
    assert proc.returncode == 0, (proc.stdout, proc.stderr)
    assert (tmp_path / "sess" / "SESSION_A_NO_CAPACITY").exists(), proc.stdout
    assert marker.exists(), "the poll never called the API, so this proves nothing"


def test_the_rehearsal_picker_never_offers_a_type_the_driver_refuses(tmp_path):
    """The two lists are edited in different files, and the failure mode is exactly the
    silent disagreement this whole guard exists for: the picker choosing a type the
    driver then rejects (or worse, would have defaulted past)."""
    picker = PICKER.read_text()
    offered = re.search(r'TYPES=\$\{REHEARSAL_TYPES:-"(.*?)"\}', picker, re.S)
    assert offered, "the picker no longer declares its type list"
    allowed = set(_allowed_types())
    missing = [t for t in offered.group(1).split() if t not in allowed]
    assert not missing, f"the picker can select types the driver refuses: {missing}"

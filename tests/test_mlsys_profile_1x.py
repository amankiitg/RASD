"""Local rehearsal for the Phase 3 profile scripts.

THE RULE (lesson L11): rehearse on the cheap box with the EXACT command lines before
spending on a paid one; the 2026-10-11 rehearsal caught five real defects that way.
`mlsys_profile_decode.sh --report-only` covers the reporting half with no GPU, and
`tests` there found a wrong default interpreter and a microsecond/second share. What
it cannot cover is the WIRING added for Phase 3:

  * `scripts/mlsys_profile_remote.sh` -- the pod-side wrapper. It must write a DONE
    marker on every exit path, and its rc must come from the ARTIFACT (a CSV with at
    least one ok row AND a profiler sidecar), not from the decode script, whose exit
    status is its last `echo`'s (lesson L14). Both failure shapes are asserted here.
  * `scripts/mlsys_profile_1x.sh` -- the driver. Its preflight is what stands between
    this and a second billing instance, so it is tested behaviourally against a stub
    `curl`: refuse when an instance already exists, and refuse rather than wait when
    the type has no capacity. No network call is made.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
REMOTE = REPO / "scripts" / "mlsys_profile_remote.sh"
DECODE = REPO / "scripts" / "mlsys_profile_decode.sh"
DRIVER = REPO / "scripts" / "mlsys_profile_1x.sh"

# A stub runner that writes the two artifacts the wrapper judges on, controlled by
# env so the three cases (ok, no ok rows, no sidecar) share one file.
RUN_STUB = '''\
import csv, json, os, sys

argv = sys.argv[1:]
opt = {}
i = 0
while i < len(argv):
    a = argv[i]
    if a.startswith("--"):
        key = a[2:]
        val = argv[i + 1] if i + 1 < len(argv) and not argv[i + 1].startswith("--") else "1"
        opt[key] = val
        i += 2
    else:
        i += 1

out = opt.get("output", "out.csv")
mode = os.environ.get("FAKE_PROFILE", "ok")
os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
header = ("run_id,context_length,status,time_sec,throughput_tps,acceptance_rate,"
          "spec_steps,tokens_generated,max_new_tokens,temperature")
status = "error" if mode == "no_ok_rows" else "ok"
with open(out, "w", newline="") as fh:
    fh.write(header + "\\n")
    fh.write(f"PROF_MAIN,131072,{status},240.0,0.1,0.0,0,24,24,0.0\\n")
if mode != "no_sidecar":
    d = os.path.join(os.path.dirname(os.path.abspath(out)), "profiler")
    os.makedirs(d, exist_ok=True)
    json.dump({"compute_us": 180e6, "comm_us": 5e6, "idle_us": 55e6, "other_us": 0.0,
               "total_us": 185e6, "wall_us": 240e6, "overlap_us": 0.0,
               "compute_pct": 0.75, "comm_pct": 0.02, "idle_pct": 0.23,
               "other_pct": 0.0, "overlap_pct": 0.0},
              open(os.path.join(d, "PROF_MAIN.json"), "w"), indent=1)
'''


def _pod_sandbox(tmp_path: Path) -> tuple[Path, dict]:
    """A pod-shaped HOME: the real decode+wrapper scripts, a stub runner, a stub timeout."""
    home = tmp_path / "home"
    rasd = home / "RASD"
    (rasd / "scripts").mkdir(parents=True)
    (rasd / "configs").mkdir()
    (tmp_path / "bin").mkdir()

    shutil.copy(DECODE, rasd / "scripts" / "mlsys_profile_decode.sh")
    shutil.copy(REMOTE, rasd / "scripts" / "mlsys_profile_remote.sh")
    (rasd / "configs" / "mlsys_profile_128k.yml").write_text(
        "defaults:\n  seeds: [42]\nPROFILES:\n  name: p\n  factor: context_length\n"
        "  levels:\n  - id: PROF_MAIN\n    context_length: 131072\n    documents: [d0]\n"
        "    max_new_tokens: 24\n")
    (rasd / "run_experiment.py").write_text(RUN_STUB)
    (rasd / ".pod_env.sh").write_text("")
    (home / "pod_env.log").write_text(f"INTERPRETER={sys.executable}\n")
    (home / ".hf_token").write_text("stub\n")

    to = tmp_path / "bin" / "timeout"
    to.write_text('#!/bin/sh\nshift\nexec "$@"\n')
    to.chmod(0o755)

    env = dict(os.environ)
    env.update({
        "HOME": str(home),
        "PATH": f"{tmp_path / 'bin'}{os.pathsep}{os.environ['PATH']}",
        "PROFILE_OUT": "results/mlsys/profile",
        "PROFILE_TAG": "a",
        "PROFILE_NPROC": "1",
        "MLSYS_PY": sys.executable,
    })
    return home, env


def _run_remote(tmp_path: Path, mode: str) -> tuple[int, str, Path]:
    home, env = _pod_sandbox(tmp_path)
    env["FAKE_PROFILE"] = mode
    proc = subprocess.run(
        ["bash", str(home / "RASD" / "scripts" / "mlsys_profile_remote.sh")],
        cwd=str(home / "RASD"), env=env, capture_output=True, text=True, timeout=300)
    out = home / "RASD" / "results" / "mlsys" / "profile"
    marker = out / "DONE"
    assert marker.exists(), f"no DONE marker\nstdout:{proc.stdout[-2000:]}\nstderr:{proc.stderr[-2000:]}"
    rc = int(marker.read_text().split()[1])
    log = (out / "profile.log").read_text() if (out / "profile.log").exists() else ""
    return rc, log, out


def test_the_wrapper_reports_success_from_the_artifact(tmp_path):
    rc, log, out = _run_remote(tmp_path, "ok")
    assert rc == 0, f"a clean profile must exit 0, got {rc}\n{log[-2000:]}"
    assert "profile ok rows=1" in log, "the ok-row count must be logged, not assumed"
    assert (out / "profile_a_nproc1.csv").exists()
    assert (out / "profiler" / "PROF_MAIN.json").exists()


def test_the_wrapper_fails_when_there_is_no_ok_row(tmp_path):
    """The decode script ends with `echo`, so its rc is 0 whatever happened. Taking
    the rc as the result is exactly the silent-success bug of lesson L1."""
    rc, log, _ = _run_remote(tmp_path, "no_ok_rows")
    assert rc == 1, f"no ok row must not read as success, got {rc}\n{log[-2000:]}"
    assert "FAILED: no ok profile row" in log


def test_the_wrapper_fails_when_the_profile_wrote_no_sidecar(tmp_path):
    """An ok row with no sidecar is a measurement with nothing to attribute time with."""
    rc, log, _ = _run_remote(tmp_path, "no_sidecar")
    assert rc == 1, f"a missing sidecar must not read as success, got {rc}\n{log[-2000:]}"
    assert "FAILED: no profiler sidecar" in log


# ---------------------------------------------------------------------------
# the driver's preflight, against a stub curl
# ---------------------------------------------------------------------------

CURL_STUB = '''\
#!/bin/sh
# Answers the two preflight endpoints only. Anything else is a test bug, so fail loud.
# Touch a marker so the test can PROVE this stub, not the real curl, answered: with a
# real curl the fake key would 401 and the driver would still exit 3/4 for the wrong
# reason, so the assertions would pass without exercising the preflight at all.
: > "$CURL_STUB_MARKER"
for a in "$@"; do url="$a"; done
case "$url" in
  *"/instances") echo "{\\"data\\": $INSTANCES_JSON}" ;;
  *"/instance-types") echo "{\\"data\\": $TYPES_JSON}" ;;
  *) echo "STUB CURL: unexpected URL $url" >&2; exit 9 ;;
esac
'''


def _driver_sandbox(tmp_path: Path) -> tuple[Path, dict]:
    """A repo-shaped sandbox for the driver: fake creds, stub curl, copied real scripts."""
    repo = tmp_path / "repo"
    (repo / "scripts").mkdir(parents=True)
    shutil.copy(DRIVER, repo / "scripts" / "mlsys_profile_1x.sh")
    # The driver reads these two by name; their contents do not matter for preflight.
    shutil.copy(REPO / "scripts" / "mlsys_detach.py", repo / "scripts" / "mlsys_detach.py")
    shutil.copy(REPO / "scripts" / "mlsys_deadman.sh", repo / "scripts" / "mlsys_deadman.sh")
    (repo / "runpod_creds.md").write_text(
        "LAMBDA_API_KEY = `fake-key-for-preflight-only`\n"
        "HF_TOKEN = `fake-token-for-preflight-only`\n")
    (tmp_path / "bin").mkdir()
    stub = tmp_path / "bin" / "curl"
    stub.write_text(CURL_STUB)
    stub.chmod(0o755)

    env = dict(os.environ)
    env["PATH"] = f"{tmp_path / 'bin'}{os.pathsep}{os.environ['PATH']}"
    env["MLSYS_SESSION_DIR"] = str(tmp_path / "sess")
    env["CURL_STUB_MARKER"] = str(tmp_path / "curl_was_the_stub")
    return repo, env


def _assert_stub_was_used(tmp_path: Path) -> None:
    assert (tmp_path / "curl_was_the_stub").exists(), \
        "the real curl answered, so the preflight was never exercised"


def test_the_driver_refuses_when_an_instance_already_exists(tmp_path):
    repo, env = _driver_sandbox(tmp_path)
    env["INSTANCES_JSON"] = '[{"id":"someone-elses-instance"}]'
    env["TYPES_JSON"] = '{}'
    proc = subprocess.run(["bash", str(repo / "scripts" / "mlsys_profile_1x.sh")],
                          cwd=str(repo), env=env, capture_output=True, text=True, timeout=120)
    _assert_stub_was_used(tmp_path)
    assert proc.returncode == 3, f"expected the refusal, got {proc.returncode}\n{proc.stdout}"
    assert "REFUSING" in proc.stdout


def test_the_driver_refuses_rather_than_waiting_with_no_capacity(tmp_path):
    """L10: an exact-match poll on a type with no capacity waits until the cutoff for
    no reason. A profile is not a campaign, so it exits instead."""
    repo, env = _driver_sandbox(tmp_path)
    env["INSTANCES_JSON"] = "[]"
    env["TYPES_JSON"] = '{"gpu_1x_a100_sxm4": {"regions_with_capacity_available": []}}'
    proc = subprocess.run(["bash", str(repo / "scripts" / "mlsys_profile_1x.sh")],
                          cwd=str(repo), env=env, capture_output=True, text=True, timeout=120)
    _assert_stub_was_used(tmp_path)
    assert proc.returncode == 4, f"expected the no-capacity exit, got {proc.returncode}\n{proc.stdout}"
    assert "NO CAPACITY" in proc.stdout


# ---------------------------------------------------------------------------
# STAGE_EXTRA: the optional payload a run needs on the pod but cannot keep in
# the repository (Phase 5's 29 MB of original RASD token sidecars).
# ---------------------------------------------------------------------------

def _driver_preflight(tmp_path, extra: dict) -> subprocess.CompletedProcess:
    repo, env = _driver_sandbox(tmp_path)
    # The stub must not be consulted at all: these are refused before the API.
    env["INSTANCES_JSON"] = "[]"
    env["TYPES_JSON"] = "{}"
    env.update(extra)
    return subprocess.run(["bash", str(repo / "scripts" / "mlsys_profile_1x.sh")],
                          cwd=str(repo), env=env, capture_output=True, text=True,
                          timeout=120)


def test_a_payload_without_a_destination_is_refused(tmp_path):
    """A half-specified payload must cost nothing rather than half-arrive."""
    payload = tmp_path / "payload"
    payload.mkdir()
    (payload / "one.json").write_text("{}")
    proc = _driver_preflight(tmp_path, {"STAGE_EXTRA": str(payload)})
    assert proc.returncode == 2, proc.stdout
    assert "needs STAGE_EXTRA_DEST" in proc.stdout
    assert not (tmp_path / "curl_was_the_stub").exists(), \
        "the refusal must happen before any API call"


def test_an_empty_payload_is_refused(tmp_path):
    """An empty directory would rsync successfully and stage nothing -- the
    silent no-op this check exists for."""
    payload = tmp_path / "payload"
    payload.mkdir()
    proc = _driver_preflight(tmp_path, {"STAGE_EXTRA": str(payload),
                                       "STAGE_EXTRA_DEST": "samp_prompts"})
    assert proc.returncode == 2, proc.stdout
    assert "is empty" in proc.stdout


def test_a_missing_payload_directory_is_refused(tmp_path):
    proc = _driver_preflight(tmp_path, {"STAGE_EXTRA": str(tmp_path / "nope"),
                                       "STAGE_EXTRA_DEST": "samp_prompts"})
    assert proc.returncode == 2, proc.stdout
    assert "is not a directory" in proc.stdout

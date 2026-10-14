"""The Phase 5 rehearsal wrapper must judge itself from the CSV, not the rc.

The rehearsal's whole purpose is to find out whether vLLM installs, loads
`gradientai/Llama-3-8B-Instruct-Gradient-1048k` at 8k and decodes -- on ONE card,
where `unit_matched` is necessarily `no` because the fairness rule requires TP=8
and the RASD prompt ids. So the wrapper has to distinguish "the engine worked but
the row is not a fair 8-GPU comparison" (success) from "the engine produced no ok
row at all" (failure), and it has to do it from the artifact: the baseline script's
own exit status is unreliable (its stage is recorded FAILED when no row is
unit-matched, which is exactly the expected case here).

Driven in a sandbox with a stub venv builder and a stub baseline, so no GPU and no
network are needed.
"""
from __future__ import annotations

import json
import os
import stat
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
REMOTE = REPO / "scripts" / "mlsys_vllm_remote.sh"
# The parameter-contract helper every pod-side wrapper sources before it does any work.
EXPECT = REPO / "scripts" / "mlsys_remote_expect.sh"

VENV_STUB = '''\
#!/usr/bin/env bash
set -uo pipefail
echo "STUB VENV: creating $MLSYS_VLLM_VENV (base $MLSYS_VLLM_BASE_PYTHON)"
mkdir -p "$MLSYS_VLLM_VENV/bin"
cat > "$MLSYS_VLLM_VENV/bin/python" <<EOF
#!/bin/sh
exec "$MLSYS_VLLM_BASE_PYTHON" "\\$@"
EOF
chmod +x "$MLSYS_VLLM_VENV/bin/python"
exit ${FAKE_VENV_RC:-0}
'''

BASELINE_STUB = '''\
import csv, json, os, sys

argv = sys.argv[1:]
# Record the exact command line, so the 8x path can be asserted on what the
# wrapper actually passed rather than on the wrapper's own log line.
if os.environ.get("FAKE_ARGV_OUT"):
    open(os.environ["FAKE_ARGV_OUT"], "w").write(json.dumps(argv))
# One line per INVOCATION, so a test can assert on how many passes the wrapper ran and
# on each pass's flags separately (the two-pass prefill run needs exactly that).
if os.environ.get("FAKE_ARGV_LOG"):
    with open(os.environ["FAKE_ARGV_LOG"], "a") as fh:
        fh.write(json.dumps(argv))
        fh.write(chr(10))
out = "out.csv"
for i, a in enumerate(argv):
    if a == "--out" and i + 1 < len(argv):
        out = argv[i + 1]
mode = __import__("os").environ.get("FAKE_VLLM_MODE", "ok")
status = "error" if mode == "no_ok_rows" else "ok"
with open(out, "w", newline="") as fh:
    w = csv.writer(fh)
    w.writerow(["model", "context_length", "max_model_len", "tensor_parallel_size",
                "attempt", "config_used", "status", "end_to_end_wall_s",
                "output_tokens", "throughput_tps_end_to_end",
                "throughput_tps_decode_only", "kv_dtype", "weight_precision",
                "unit_matched", "error_class", "error"])
    w.writerow(["gradientai/Llama-3-8B-Instruct-Gradient-1048k", 8192, 8192, 1, 1,
                "1-default-tp8", status, 12.5, 128, 10.24, "", "bfloat16",
                "bfloat16", "no", "", ""])
'''


def _sandbox(tmp_path: Path, venv_rc: int = 0) -> tuple[Path, dict]:
    home = tmp_path / "home"
    rasd = home / "RASD"
    (rasd / "scripts").mkdir(parents=True)
    (rasd / ".pod_env.sh").write_text("")
    (home / "pod_env.log").write_text(f"INTERPRETER={sys.executable}\n")
    (home / ".hf_token").write_text("stub\n")
    # THE REAL SCRIPT UNDER TEST, and stubs for the two things it drives.
    import shutil
    shutil.copy(REMOTE, rasd / "scripts" / "mlsys_vllm_remote.sh")
    # The wrapper verifies its parameters against the driver's manifest before it
    # installs anything, and it sources this helper to do it -- so the sandbox must
    # carry it, or the wrapper dies at the source line with 127/2 instead of testing
    # anything (found 2026-10-12: four tests failed exactly this way).
    shutil.copy(EXPECT, rasd / "scripts" / "mlsys_remote_expect.sh")
    vp = rasd / "scripts" / "mlsys_vllm_venv.sh"
    vp.write_text(VENV_STUB)
    vp.chmod(vp.stat().st_mode | stat.S_IEXEC)
    (rasd / "scripts" / "mlsys_vllm_baseline.py").write_text(BASELINE_STUB)
    env = dict(os.environ)
    env.update({
        "HOME": str(home),
        "PROFILE_OUT": "results/mlsys/vllm_rehearsal",
        "PROFILE_NPROC": "1",
        # THE MANIFEST THE DRIVER SENDS. The wrapper verifies what it received against
        # it before installing anything, and refuses when there is no manifest at all,
        # so a direct invocation must bring one -- built here from the same settings
        # this test passes, which is what the driver does.
        "REMOTE_EXPECT": "PROFILE_OUT=results/mlsys/vllm_rehearsal;PROFILE_NPROC=1",
        "MLSYS_VLLM_VENV": str(rasd / ".venv-vllm"),
        "MLSYS_VLLM_BASE_PYTHON": sys.executable,
        "FAKE_VENV_RC": str(venv_rc),
    })
    return home, env


def _run(tmp_path: Path, venv_rc: int = 0, mode: str = "ok"):
    home, env = _sandbox(tmp_path, venv_rc)
    env["FAKE_VLLM_MODE"] = mode
    proc = subprocess.run(
        ["bash", str(home / "RASD" / "scripts" / "mlsys_vllm_remote.sh")],
        cwd=str(home / "RASD"), env=env, capture_output=True, text=True, timeout=300)
    out = home / "RASD" / "results" / "mlsys" / "vllm_rehearsal"
    marker = out / "DONE"
    assert marker.exists(), f"no DONE marker\nstdout:{proc.stdout[-2000:]}"
    return int(marker.read_text().split()[1]), (out / "rehearsal.log").read_text(), out


def test_a_clean_rehearsal_succeeds_even_though_unit_matched_is_no(tmp_path):
    """The expected 1x case: the engine works, the row is not a fair comparison."""
    rc, log, out = _run(tmp_path)
    assert rc == 0, f"a clean rehearsal must exit 0, got {rc}\n{log[-2000:]}"
    assert "VLLM RUN DONE" in log
    assert "rows=1 ok=1" in log
    assert "unit_matched=no is EXPECTED" in log
    assert (out / "vllm_baseline.csv").exists()


def test_the_8x_parameters_reach_the_engine(tmp_path):
    """The rehearsal and the real run are ONE script with named differences.

    If these flags did not travel, "the rehearsal proves the 8x run works" would be
    a claim about a different command line.
    """
    home, env = _sandbox(tmp_path)
    argv_path = tmp_path / "argv.json"
    env.update({
        "FAKE_VLLM_MODE": "ok",
        "FAKE_ARGV_OUT": str(argv_path),
        "REHEARSE_TP": "8",
        "REHEARSE_CTXS": "131072 524288 1048576",
        "REHEARSE_PROMPT_DIR": "results/mlsys/vllm_prompts",
        # The manifest grows with the settings: these three are the ones that were
        # dropped in transit on 2026-10-12, so they are exactly what must be declared.
        "REMOTE_EXPECT": ("PROFILE_OUT=results/mlsys/vllm_rehearsal;PROFILE_NPROC=1;"
                          "REHEARSE_TP=8;REHEARSE_CTXS=131072 524288 1048576;"
                          "REHEARSE_PROMPT_DIR=results/mlsys/vllm_prompts"),
    })
    subprocess.run(
        ["bash", str(home / "RASD" / "scripts" / "mlsys_vllm_remote.sh")],
        cwd=str(home / "RASD"), env=env, capture_output=True, text=True, timeout=300)
    argv = json.loads(argv_path.read_text())
    assert argv[argv.index("--tensor-parallel-size") + 1] == "8"
    ctxs = argv[argv.index("--context-lengths") + 1:argv.index("--context-lengths") + 4]
    assert ctxs == ["131072", "524288", "1048576"], ctxs
    assert argv[argv.index("--prompt-ids-from-sidecars") + 1] == "results/mlsys/vllm_prompts"


def test_a_run_with_no_ok_row_fails(tmp_path):
    rc, log, _ = _run(tmp_path, mode="no_ok_rows")
    assert rc == 1, f"no ok row must not read as success, got {rc}\n{log[-2000:]}"
    assert "FAILED: no ok row" in log


def test_a_failed_venv_install_fails_instead_of_running(tmp_path):
    """The install is the thing being rehearsed; if it fails, say so before running."""
    rc, log, _ = _run(tmp_path, venv_rc=4)
    assert rc == 1, f"a failed install must fail the rehearsal, got {rc}\n{log[-2000:]}"
    assert "the vLLM install returned rc=4" in log


def _invocations(tmp_path, extra_env):
    """Every command line the wrapper handed the baseline, in order."""
    home, env = _sandbox(tmp_path)
    log = tmp_path / "argv.jsonl"
    env.update({"FAKE_VLLM_MODE": "ok", "FAKE_ARGV_LOG": str(log)})
    env.update(extra_env)
    subprocess.run(
        ["bash", str(home / "RASD" / "scripts" / "mlsys_vllm_remote.sh")],
        cwd=str(home / "RASD"), env=env, capture_output=True, text=True, timeout=300)
    if not log.exists():
        return []
    return [json.loads(l) for l in log.read_text().splitlines() if l.strip()]


def test_the_prefill_pass_is_a_second_one_token_pass_in_its_own_csv(tmp_path):
    """THE TTFT THE OFFLINE PATH CANNOT BE TRUSTED TO REPORT, measured instead.

    The retry's `ttft_s` was 1.79e9 because vLLM's stamps are absolute. The recorder is
    fixed, but a fix can only turn a bad number into a blank -- so the prefill is also
    MEASURED, by a second pass of the same prompts with max_new_tokens=1, whose wall is
    the prefill plus one step. This asserts the wrapper really runs that pass, with the
    flags that keep it a probe rather than a measurement: no `--matched-max-new-tokens`
    (so its rows are labelled unit_matched=no for the honest reason), one book, and its
    own CSV.
    """
    invocations = _invocations(tmp_path, {"VLLM_PREFILL_PASS": "1",
                                          "REHEARSE_CTXS": "131072 524288 1048576"})
    assert len(invocations) == 2, f"expected a main pass and a prefill pass, got {invocations}"
    main, pre = invocations
    assert main[main.index("--max-new-tokens") + 1] == "128"
    assert main[main.index("--matched-max-new-tokens") + 1] == "128"
    assert pre[pre.index("--max-new-tokens") + 1] == "1"
    assert "--matched-max-new-tokens" not in pre, (
        "a 1-token probe must not declare itself the matched cell of a 128-token RASD row")
    assert pre[pre.index("--documents") + 1] == "pg19_train_0", (
        "one book is enough: the prefill cost is a function of the token count")
    assert pre[pre.index("--out") + 1].endswith("vllm_prefill.csv"), (
        "its own CSV, so it can never be averaged into the comparison")
    assert pre[pre.index("--context-lengths") + 1:pre.index("--context-lengths") + 4] == \
        ["131072", "524288", "1048576"], "the same contexts as the main pass"


def test_without_the_flag_the_wrapper_runs_one_pass(tmp_path):
    """The control: the 8x retry's behaviour must be unchanged by default."""
    invocations = _invocations(tmp_path, {"REHEARSE_CTXS": "131072"})
    assert len(invocations) == 1, invocations
    assert "--documents" not in invocations[0]

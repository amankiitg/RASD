"""The wait-mode arm switch: every row must prove which NCCL wait mode produced it.

WHAT THIS IS GUARDING. Every campaign run before 2026-10-14 ran with
`TORCH_NCCL_BLOCKING_WAIT=1`, written into the pod's `.pod_env.sh` by
`scripts/mlsys_pod_env.sh` and sourced with `set -a` by both remote scripts. In torch 2.5.1
that flag makes `ProcessGroupNCCL::WorkNCCL::synchronizeInternal` poll with a fixed 10 ms
sleep (`kSynchronizeBusyWaitMillis`), which inflates EVERY synchronous collective by up to
10 ms -- 224 of them per decode token in the ring engine. Two properties follow and both are
tested here:

  * the mode is read ONCE per process group, so the two arms must be two launches, and the
    arm has to be applied with `env` / `env -u` AFTER `.pod_env.sh` is sourced (`set -a` means
    the file overrides a forwarded value, which is exactly how the flag became uncontrollable);
  * a row that cannot say which arm it came from is void, so `env_arm` labels the intent and
    `nccl_env` records the effect, with `<unset>` spelled out.

The shell tests run the REAL scripts with a stub interpreter and a stub `torchrun`, because a
source-level assertion would pass for a script that exports the variables in the wrong order
(the lesson from tests/test_mlsys_session_a_check_stage.py).
"""
from __future__ import annotations

import os
import re
import shutil
import stat
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
REMOTE = REPO / "scripts" / "mlsys_bw_remote.sh"
POD_ENV = REPO / "scripts" / "mlsys_pod_env.sh"


def _pod_env_lines(mode: str, tmp_path: Path) -> str:
    """Run the real mode resolver and writer out of mlsys_pod_env.sh.

    The whole script is not run (its job is to install a conda env and pinned wheels, which
    needs a pod), so the two pieces under test are extracted verbatim: the resolver that turns
    MLSYS_NCCL_WAIT_MODE into NCCL lines, and the writer that puts them in the env file. The
    assertion is on the lines that would ship, not on the intent.
    """
    src = POD_ENV.read_text()
    start = src.index('MLSYS_NCCL_WAIT_MODE="${MLSYS_NCCL_WAIT_MODE:-async}"')
    resolver = src[start:src.index("esac\n", start) + len("esac\n")]
    # The NCCL literal, taken from the file so a future edit to it is covered too.
    lit_start = src.index("NCCL_SETTINGS='")
    lit = src[lit_start:src.index("'\n", lit_start) + 1]
    # The writer block: anchored on the redirect and searched BACKWARDS for the brace that
    # opens it. Searching forwards from the "Written by" echo finds a brace in a later
    # function instead, because the echo is itself inside the block it names.
    w_end = src.index('} > "$REPO/.pod_env.sh"') + len('} > "$REPO/.pod_env.sh"')
    w_start = src.rindex("{", 0, w_end)
    writer = src[w_start:w_end]

    harness = "\n".join([
        'REPO="%s"' % tmp_path,
        'HF_HOME=/h HF_HUB_CACHE=/h/h TRANSFORMERS_CACHE=/h/t PIP_CACHE_DIR=/p '
        'PYTHONPATH=/y',
        resolver,
        lit,
        writer,
        'cat "$REPO/.pod_env.sh"',
    ])
    env = dict(os.environ)
    env["MLSYS_NCCL_WAIT_MODE"] = mode
    proc = subprocess.run(["bash", "-c", harness], env=env, capture_output=True, text=True)
    return proc.stdout + proc.stderr


def test_the_async_mode_never_exports_blocking_wait(tmp_path):
    """The replacement. If this line is ever written again, the 10 ms poll is back and every
    collective in the session pays it."""
    out = _pod_env_lines("async", tmp_path)
    assert "TORCH_NCCL_ASYNC_ERROR_HANDLING=1" in out
    assert "TORCH_NCCL_BLOCKING_WAIT" not in out, out
    # The hang budget must survive the change: it is why the flag was added in the first place.
    assert "NCCL_TIMEOUT=3600" in out and "TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=3600" in out


def test_the_blocking_mode_is_kept_and_selects_only_blocking_wait(tmp_path):
    """Kept for the diagnostic's Arm A. Exactly one wait knob, so a row is unambiguous."""
    out = _pod_env_lines("blocking", tmp_path)
    assert "TORCH_NCCL_BLOCKING_WAIT=1" in out
    assert "TORCH_NCCL_ASYNC_ERROR_HANDLING" not in out, out
    assert "NCCL_TIMEOUT=3600" in out


def test_an_unknown_mode_is_refused_rather_than_defaulted(tmp_path):
    """A typo must not silently select a mode: it is the difference between the two arms."""
    out = _pod_env_lines("bogus", tmp_path)
    assert "FATAL" in out and "not one of async|blocking" in out


def test_the_mode_is_recorded_as_a_comment_not_an_export():
    """`set -a` means the file overrides a forwarded value. Exporting the mode would make it
    uncontrollable again, which is the defect being fixed."""
    src = POD_ENV.read_text()
    assert '"export MLSYS_NCCL_WAIT_MODE=' not in src
    assert '"# MLSYS_NCCL_WAIT_MODE=$MLSYS_NCCL_WAIT_MODE' in src


def _sandbox(tmp_path: Path) -> tuple[Path, dict]:
    """A HOME with the real remote script, a stub interpreter and a stub torchrun."""
    home = tmp_path / "home"
    rasd = home / "RASD"
    (rasd / "scripts").mkdir(parents=True)
    (tmp_path / "bin").mkdir()

    shutil.copy(REMOTE, rasd / "scripts" / "mlsys_bw_remote.sh")
    for helper in ("mlsys_remote_expect.sh", "mlsys_stage_list.py"):
        shutil.copy(REPO / "scripts" / helper, rasd / "scripts" / helper)
    (rasd / "configs").mkdir()
    (rasd / "configs" / "bw.yml").write_text(
        "defaults:\n  seeds: [42]\nBW_TOP0:\n  name: bw_topo\n  factor: context_length\n"
        "  levels:\n  - id: BW_TOP0\n    context_length: 8192\n    documents: [d0]\n")

    # The pod's env file, written the way mlsys_pod_env.sh writes it in the async default.
    (rasd / ".pod_env.sh").write_text(
        "export NCCL_TIMEOUT=3600\n"
        "export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=3600\n"
        "# MLSYS_NCCL_WAIT_MODE=async (provisioning-time default; a run may override per stage)\n"
        "export TORCH_NCCL_ASYNC_ERROR_HANDLING=1\n")

    # A stub interpreter: every invocation records the NCCL environment it was handed, which
    # is the effect under test. `-c` and `-m` forms both land here.
    # A stub interpreter that RECORDS the NCCL environment it was handed (the effect under
    # test) and otherwise delegates to the real interpreter, so the stage-list helper and the
    # topo stage still work. Delegation matters: a stub that answers every call with a
    # constant string makes the parameter contract fail for the wrong reason, which would
    # hide a real contract bug behind a plausible-looking pass.
    real_py = sys.executable
    stub = tmp_path / "bin" / "stubpy"
    stub.write_text(
        '#!/bin/sh\n'
        'case "$*" in\n'
        '  *"torch.distributed.run"*)\n'
        '     # An arm launch. Record the NCCL environment it was handed and do no work.\n'
        '     echo "PYENV block=${TORCH_NCCL_BLOCKING_WAIT-<unset>}'
        ' async=${TORCH_NCCL_ASYNC_ERROR_HANDLING-<unset>}'
        ' mode=${MLSYS_NCCL_WAIT_MODE-<unset>}"\n'
        '     exit 0 ;;\n'
        'esac\n'
        'exec ' + real_py + ' "$@"\n')
    stub.chmod(stub.stat().st_mode | stat.S_IEXEC)
    (home / "pod_env.log").write_text(f"INTERPRETER={stub}\n")
    (home / ".hf_token").write_text("stub\n")
    # The bench and arms write into $OUT; the remote script creates those directories itself.
    (rasd / "topo_probe").mkdir(exist_ok=True)

    env = dict(os.environ)
    env.update({
        "HOME": str(home),
        "PATH": f"{tmp_path / 'bin'}{os.pathsep}{os.environ['PATH']}",
        "SESSION_OUT": "out",
        "SESSION_CFG": "configs/bw.yml",
        "NPROC": "1",
        "SESSION_SANITY_CFG": "configs/bw.yml",
        "CODE_COMMIT": "deadbeef",
        "BW_OUT": "out",
        "BW_BOOK": "pg19_train_0",
        "BW_SEED": "42",
        "BW_ARMS": "ring2048",
        "BW_ROUNDS": "4",
        "BW_CTX": "8192",
        "BW_CTX_1M": "1048576",
        "BW_ENV_ARMS": "blocking async",
        "BW_SINGLE_RANK_OK": "1",
        "BW_1M_OK": "0",
        "BW_TARGET_TOKENS": "8",
        "BW_SPEC_TOKENS": "8",
        "REMOTE_EXPECT": (
            "SESSION_OUT=out;SESSION_CFG=configs/bw.yml;SESSION_SANITY_CFG=configs/bw.yml;NPROC=1;"
            "SESSION_CHECK_STAGE=;SESSION_GATE_CANDIDATES=;STAGES=BW_TOP0;"
            "CODE_COMMIT=deadbeef;BW_OUT=out;BW_BOOK=pg19_train_0;BW_SEED=42;BW_ROUNDS=4;"
            "BW_CTX=8192;BW_CTX_1M=1048576;BW_ARMS=ring2048;BW_TARGET_TOKENS=8;"
            "BW_SPEC_TOKENS=8;BW_1M_OK=0;BW_ENV_ARMS=blocking async;BW_SINGLE_RANK_OK=1"),
    })
    return home, env


def test_each_arm_launch_gets_its_own_environment(tmp_path):
    """THE regression. The two arms must reach the interpreter with the right NCCL
    environment, applied AFTER `.pod_env.sh` is sourced. The pod's file is provisioned in the
    async default, so the blocking arm has to actively SET its flag and the async arm has to
    actively UNSET it: relying on the file would make the arms depend on how the pod happened
    to be provisioned, which is how this defect survived for months.
    """
    home, env = _sandbox(tmp_path)
    proc = subprocess.run(["bash", str(home / "RASD" / "scripts" / "mlsys_bw_remote.sh")],
                          cwd=str(home / "RASD"), env=env, capture_output=True, text=True,
                          timeout=300)
    # The recorded environment lands in the per-arm stage logs (the session log carries only
    # the greppable summary lines), so read every log the script wrote.
    out_dir = home / "RASD" / "out"
    text = proc.stdout + proc.stderr
    for f in sorted(out_dir.glob("*.log")):
        text += f.read_text()

    # The arms are distinguished by BLOCKING_WAIT alone. ASYNC is asserted PRESENT, not
    # absent: torchrun's elastic agent injects it into every worker (defaulting to 1), which
    # the 1x rehearsal proved, so the blocking arm cannot clear it and a test that demanded
    # `async=<unset>` would be testing a fiction.
    blocking = [l for l in text.split("\n") if "block=1" in l]
    asynchronous = [l for l in text.split("\n") if "block=<unset>" in l and "async=1" in l]
    assert blocking, f"the blocking arm never ran with its flag set:\n{text[-3000:]}"
    assert asynchronous, f"the async arm never ran with blocking unset:\n{text[-3000:]}"
    assert "mode=blocking" in blocking[0], blocking[0]
    assert "mode=async" in asynchronous[0], asynchronous[0]
    assert "async=1" in blocking[0], (
        "torchrun injects TORCH_NCCL_ASYNC_ERROR_HANDLING=1 into every worker; the blocking "
        "arm is expected to show it alongside its flag, and torch resolves that by using "
        "blocking wait. If this ever reads <unset>, the injection has changed and the "
        "diagnostic's premise needs re-checking.")


def test_both_arms_run_and_the_launches_are_separate(tmp_path):
    """Two process launches, because the mode is read once per process group. If a future
    edit folds them into one, the second arm would silently measure the first arm's mode."""
    home, env = _sandbox(tmp_path)
    subprocess.run(["bash", str(home / "RASD" / "scripts" / "mlsys_bw_remote.sh")],
                   cwd=str(home / "RASD"), env=env, capture_output=True, text=True, timeout=300)
    log = (home / "RASD" / "out" / "bw.log").read_text()
    assert log.count("### ARM blocking") == 1 and log.count("### ARM async") == 1
    for arm in ("blocking", "async"):
        assert f"### stage A[{arm}] bench rc=" in log
        assert f"### stage B[{arm}] arms rc=" in log
    # and the rows are separated by arm on disk, not merged into one directory
    assert (home / "RASD" / "out" / "arms" / "blocking").is_dir()
    assert (home / "RASD" / "out" / "arms" / "async").is_dir()


def test_a_missing_manifest_stops_before_any_work(tmp_path):
    """R9: an unverifiable parameter set must refuse rather than guess. The contract is the
    thing that makes a forwarded value provable, so its absence has to be fatal."""
    home, env = _sandbox(tmp_path)
    env.pop("REMOTE_EXPECT")
    proc = subprocess.run(["bash", str(home / "RASD" / "scripts" / "mlsys_bw_remote.sh")],
                          cwd=str(home / "RASD"), env=env, capture_output=True, text=True,
                          timeout=300)
    text = proc.stdout + proc.stderr + \
        ((home / "RASD" / "out" / "bw.log").read_text()
         if (home / "RASD" / "out" / "bw.log").exists() else "")
    assert "### ARM blocking" not in text, "work ran without a verifiable manifest"
    marker = home / "RASD" / "out" / "DONE"
    assert marker.exists(), "the DONE marker must still be written so rc is judgeable"

"""The driver->pod parameter contract, tested across the boundary it actually runs on.

WHY A SUBPROCESS TEST. `mlsys_remote_expect.sh` is sourced by a bash script on the pod; the
thing that can break is the STRING the driver builds versus the string the pod reports, and
that is a shell-level question -- a Python-level test of the comparison function would not
see a key that one side spells differently or omits. So this builds the driver's manifest
with the driver's own derivation (`mlsys_stage_list.py`, the same script the driver calls)
and runs the comparison in bash.

The failure this covers has happened twice:
  * 2026-10-12, ~$5.5: the driver forwarded a subset of its settings and the pod silently ran
    its defaults.
  * 2026-10-13: the pod's script reported only the DF_* subset as received and the contract
    correctly refused to start over six variables it had no report for -- a contract that
    cannot see an arrival stops a CORRECT run.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
CONFIG = "configs/mlsys_df_followup.yml"

MANIFEST_TAIL = ("DF_OUT=results/mlsys/decode_df_reh;DF_CTX_ARMS=8192;DF_CTX_SMOKE=8192;"
                 "DF_BOOK=pg19_train_0;DF_ARMS=ring2048,splitkv;DF_SEED=42;"
                 "DF_BENCH_ROUNDS=4;DF_TARGET_TOKENS=4;DF_SPEC_TOKENS=4;"
                 "DF_SINGLE_RANK_OK=1")

RECEIVED_HEAD = ("SESSION_OUT=results/mlsys/decode_df_reh;SESSION_CFG={cfg};"
                 "SESSION_SANITY_CFG={cfg};NPROC=1;SESSION_CHECK_STAGE=;"
                 "SESSION_GATE_CANDIDATES=;STAGES={stages}")


def _stages() -> str:
    out = subprocess.run([sys.executable, "scripts/mlsys_stage_list.py", CONFIG, "--space"],
                         cwd=REPO, capture_output=True, text=True, check=True)
    return out.stdout.rstrip()


def _run(manifest: str, received: str) -> subprocess.CompletedProcess:
    script = (
        "set -uo pipefail\n"
        f'cd "{REPO}"\n'
        ". scripts/mlsys_remote_expect.sh\n"
        f"mlsys_expect t '{manifest}' '{received}'\n"
    )
    return subprocess.run(["bash", "-c", script], cwd=REPO, capture_output=True, text=True)


def _manifest() -> str:
    return (f"SESSION_OUT=results/mlsys/decode_df_reh;SESSION_CFG={CONFIG};"
            f"SESSION_SANITY_CFG={CONFIG};NPROC=1;SESSION_CHECK_STAGE=;"
            f"SESSION_GATE_CANDIDATES=;STAGES={_stages()};" + MANIFEST_TAIL)


def _received(**overrides) -> str:
    """The pod's report, with optional values replaced (keys are the driver's upper-case names)."""
    import re
    head = RECEIVED_HEAD.format(cfg=CONFIG, stages=_stages())
    body = MANIFEST_TAIL
    for key, val in overrides.items():
        body = re.sub(rf"(?<=[;]){key.upper()}=[^;]*", f"{key.upper()}={val}", body)
    return f"{head};{body}"


def test_a_complete_report_starts():
    r = _run(_manifest(), _received())
    assert r.returncode == 0, f"a complete report must not stop the run: {r.stderr[-800:]}"
    assert "verified" in r.stdout


def test_a_dropped_setting_stops_the_run():
    recv = ";".join(p for p in _received().split(";") if not p.startswith("DF_CTX_ARMS="))
    r = _run(_manifest(), recv)
    assert r.returncode == 2
    assert "DF_CTX_ARMS" in r.stderr and "never received" in r.stderr


def test_a_wrong_value_stops_the_run():
    r = _run(_manifest(), _received(df_ctx_arms="131072"))
    assert r.returncode == 2
    assert "131072" in r.stderr and "intended" in r.stderr


def test_a_missing_manifest_stops_the_run():
    r = _run("", _received())
    assert r.returncode == 2
    assert "no expectation manifest" in r.stderr


def test_the_stage_list_is_derived_the_same_way_on_both_sides():
    """The pod derives STAGES from the config; the driver derives it with the same script.
    A trailing-space difference once refused every session, so it is asserted here."""
    stages = _stages()
    assert stages and not stages.endswith(" ")
    assert stages.split() == [k for k in ("DF_TOP0", "DF_BENCH_RING", "DF_SMOKE_SPLIT_8K",
                                          "DF_ARMS_128K", "DF_PPL_128K")], stages
    r = _run(_manifest(), _received())
    assert r.returncode == 0

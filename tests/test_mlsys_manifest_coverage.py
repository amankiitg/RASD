"""Every key the driver's manifest names must be REPORTED by the pod-side script that reads it.

WHY A STRUCTURAL TEST. This has now failed twice in one day, in the same way and with
opposite causes, and neither failure is visible in a unit test of the comparison function:

  * 2026-10-13, the df script reported only its own DF_* subset, so the contract refused to
    start over six variables that HAD arrived. A contract that cannot see an arrival stops a
    correct run.
  * 2026-10-13, adding CODE_COMMIT to the driver's manifest without adding it to the standard
    remote script's report did the same thing to the rerun's 1x rehearsal (DONE rc=2).

Both are one missing `KEY=${KEY-<absent>}` in a string. The strings are long, the keys are
added in one place and needed in another, and nothing local notices -- so the relationship is
asserted here instead of remembered.

The check is a regex over the two files: the keys of the driver's REMOTE_EXPECT and the keys of
each remote script's `mlsys_expect` "received" argument. It is deliberately dumb: it compares
NAMES, not values, because a name that is missing is the failure mode.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
DRIVER = REPO / "scripts" / "mlsys_session_a.sh"
REMOTE_SCRIPTS = [REPO / "scripts" / "mlsys_session_a_remote.sh",
                  REPO / "scripts" / "mlsys_df_remote.sh"]

KEY = r"([A-Z][A-Z0-9_]*)\s*=\s*[$'\"]?[\{\$]?"


def _driver_keys() -> set[str]:
    src = DRIVER.read_text()
    # The manifest is built with REMOTE_EXPECT="A=..;B=..;..." (and may be extended by
    # REMOTE_EXPECT_EXTRA, whose keys a custom script supplies itself).
    blocks = re.findall(r'REMOTE_EXPECT="([^"]*)"', src)
    assert blocks, "no REMOTE_EXPECT assignment found in the driver"
    keys: set[str] = set()
    for b in blocks:
        for pair in b.split(";"):
            m = re.match(KEY, pair.strip())
            if m:
                keys.add(m.group(1))
    assert keys, "the driver's manifest parsed to no keys"
    return keys


def _received_keys(path: Path) -> set[str]:
    src = path.read_text()
    # The "received" argument is the 3rd argument of the mlsys_expect call: a quoted string of
    # KEY=value pairs. It is the longest quoted "K=V;K=V" literal in the file.
    cands = re.findall(r'"((?:[A-Z][A-Z0-9_]*=[^";]*;)*[A-Z][A-Z0-9_]*=[^";]*)"', src)
    assert cands, f"no received-parameter string found in {path.name}"
    longest = max(cands, key=len)
    keys = {m.group(1) for m in re.finditer(KEY, longest)}
    assert keys, f"the received string in {path.name} parsed to no keys"
    return keys


@pytest.mark.parametrize("remote", REMOTE_SCRIPTS, ids=lambda p: p.name)
def test_the_pod_reports_every_key_the_driver_sends(remote):
    missing = _driver_keys() - _received_keys(remote)
    assert not missing, (
        f"{remote.name} does not REPORT {sorted(missing)} to mlsys_expect. The values arrive -- "
        f"the pod-side script just does not say so -- and the contract then refuses to start a "
        f"correct run (2026-10-13, twice). Add them to the 'received' string.")


def test_the_driver_sends_the_commit():
    """CODE_COMMIT is what makes a rerun's rows able to prove which engine produced them
    (stop condition S2). If it ever leaves the manifest, the rows silently go back to
    `git_commit: unknown`."""
    assert "CODE_COMMIT" in _driver_keys()
    driver_src = DRIVER.read_text()
    assert re.search(r"CODE_COMMIT=\'?\$CODE_COMMIT", driver_src), \
        "the driver must put CODE_COMMIT in the pod's environment, not only in the manifest"

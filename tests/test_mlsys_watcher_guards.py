"""The watcher's pull/merge must be *unable* to import an unverified pull.

Three specific failures are guarded here, all of them "silent success" shapes
where the script keeps going and looks fine:

  * the repo rsync used the form `rsync ... && say "repo staged"`, so a non-zero
    exit merely skipped the message and the run continued against a pod with
    stale code;
  * an unresolved metadata path was downgraded to a WARNING, so a stage that
    needs the PG-19 chunks would die mid-run on a paid instance;
  * the results merge ran unconditionally -- `PULL_FAILED` only reworded a
    message -- so a truncated or unverified pull was copied into results/
    anyway, where a short CSV is indistinguishable from a short run.

These are asserted structurally because the alternative is executing a
cloud-provisioning script: the rehearsal in scripts/mlsys_rehearsal.sh
exercises the merge behaviour end to end, and this pins the shape that makes it
possible.
"""
from __future__ import annotations

from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
WATCHER = REPO / "scripts" / "mlsys_watch_and_run.sh"
SRC = WATCHER.read_text() if WATCHER.exists() else ""


MERGE_CALL = "scripts/mlsys_pull_merge.sh"


def _merge_call_region():
    """(index of the guard, index of the merge call, index of the closing fi)."""
    guard = SRC.index('if [ "$PULL_FAILED" = "0" ]; then\n  if bash '
                      + MERGE_CALL)
    call = SRC.index(MERGE_CALL, guard)
    tail = SRC.index("MERGE_OK=0", call)
    return guard, call, tail


@pytest.mark.skipif(not SRC, reason="watcher missing")
def test_the_merge_runs_only_after_a_verified_pull():
    """The merge used to sit OUTSIDE the PULL_FAILED guard, so the guard only
    chose the wording of a message while an unverified pull was copied into
    results/ regardless."""
    guard, call, tail = _merge_call_region()
    assert guard < call < tail
    invocations = [l for l in SRC.splitlines()
                   if MERGE_CALL in l and not l.lstrip().startswith("#")]
    assert len(invocations) == 1, (
        f"the merge is invoked from {len(invocations)} places; a second path "
        f"would bypass the guard")


@pytest.mark.skipif(not SRC, reason="watcher missing")
def test_no_second_inline_merge_path_exists():
    """A leftover inline `write_bytes` copy would bypass the guard entirely."""
    assert "t.write_bytes(f.read_bytes())" not in SRC
    assert SRC.count("dest.write_bytes") == 0


@pytest.mark.skipif(not SRC, reason="watcher missing")
def test_the_merge_verdict_is_visible_in_the_exit_state():
    assert "MERGE_OK=0" in SRC and "MERGE_OK=1" in SRC


@pytest.mark.skipif(not SRC, reason="watcher missing")
def test_repo_rsync_failure_is_fatal():
    i = SRC.index("staging the repository to the pod failed")
    # an `if ! rsync` guard, not `rsync && say`
    head = SRC[:i]
    assert "if ! rsync" in head[head.rindex("staging repository"):], \
        "the repo rsync is still of the `rsync ... && say` form"
    assert "exit 6" in SRC[i:i + 400]
    assert "terminate_and_confirm" in SRC[i:i + 400]


@pytest.mark.skipif(not SRC, reason="watcher missing")
def test_unresolved_metadata_paths_are_fatal():
    assert "WARNING: $d has unresolved paths" not in SRC, \
        "an unresolved metadata path is still a warning, not a failure"
    i = SRC.index("has unresolved metadata paths on the pod")
    assert "exit 6" in SRC[i:i + 400]


@pytest.mark.skipif(not SRC, reason="watcher missing")
def test_nothing_merges_before_every_sha256_matches():
    """A single diff of remote vs local manifests, and PULL_FAILED on mismatch."""
    assert "pull_remote.sha256" in SRC and "pull_local.sha256" in SRC
    i = SRC.index("sha256 MISMATCH")
    assert "PULL_FAILED=1" in SRC[i:i + 300]
    # an unobtainable remote manifest must also block the merge
    j = SRC.index("could not obtain the remote sha256 manifest")
    assert "PULL_FAILED=1" in SRC[j:j + 300]


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-q"]))


# ---------------------------------------------------------------------------
# The merge rule itself, exercised rather than read
# ---------------------------------------------------------------------------

MERGE = REPO / "scripts" / "mlsys_pull_merge.sh"


def _sha_manifest(root: Path) -> str:
    import subprocess
    out = subprocess.run(
        ["bash", "-c",
         'find . -type f -print0 | sort -z | xargs -0 shasum -a 256'],
        cwd=root, capture_output=True, text=True, check=True)
    return out.stdout


def _merge(stage: Path, remote_sha: Path, dest: Path):
    import subprocess
    return subprocess.run(
        ["bash", str(MERGE), "--stage-dir", str(stage),
         "--remote-sha", str(remote_sha), "--dest", str(dest)],
        capture_output=True, text=True)


@pytest.mark.skipif(not MERGE.exists(), reason="merge script missing")
def test_a_verified_pull_is_merged(tmp_path):
    stage = tmp_path / "stage"
    (stage / "per_token").mkdir(parents=True)
    (stage / "natural_f1_128k.csv").write_text("run_id,status\nr1,ok\n")
    (stage / "per_token" / "r1.jsonl").write_text('{"round_idx": 0}\n')
    remote = tmp_path / "remote.sha256"
    remote.write_text(_sha_manifest(stage))
    dest = tmp_path / "dest"

    r = _merge(stage, remote, dest)
    assert r.returncode == 0, r.stdout + r.stderr
    assert (dest / "natural_f1_128k.csv").read_text().startswith("run_id")
    assert (dest / "per_token" / "r1.jsonl").exists()


@pytest.mark.skipif(not MERGE.exists(), reason="merge script missing")
def test_one_injected_hash_mismatch_merges_nothing(tmp_path):
    """The rehearsal's injected failure: a file changed after the pull."""
    stage = tmp_path / "stage"
    stage.mkdir()
    (stage / "natural_f1_128k.csv").write_text("run_id,status\nr1,ok\n")
    (stage / "other.csv").write_text("run_id,status\nr2,ok\n")
    remote = tmp_path / "remote.sha256"
    remote.write_text(_sha_manifest(stage))

    # the pull is now stale for exactly one file
    (stage / "other.csv").write_text("run_id,status\nr2,TRUNCATED\n")
    dest = tmp_path / "dest"
    dest.mkdir()
    (dest / "already_here.csv").write_text("earlier stage\n")

    r = _merge(stage, remote, dest)
    assert r.returncode != 0
    assert "NOT MERGED" in r.stdout
    assert not (dest / "natural_f1_128k.csv").exists(), (
        "a mismatching pull was partially merged")
    assert (dest / "already_here.csv").exists(), "results/ was disturbed"


@pytest.mark.skipif(not MERGE.exists(), reason="merge script missing")
def test_an_empty_pull_merges_nothing(tmp_path):
    dest = tmp_path / "dest"
    remote = tmp_path / "remote.sha256"
    remote.write_text("x\n")
    r = _merge(tmp_path / "never_created", remote, dest)
    assert r.returncode == 1
    assert "NOT MERGED" in r.stdout


@pytest.mark.skipif(not MERGE.exists(), reason="merge script missing")
def test_an_unobtainable_remote_manifest_is_not_verified(tmp_path):
    """An absent manifest means unverifiable, which is not verified."""
    stage = tmp_path / "stage"
    stage.mkdir()
    (stage / "a.csv").write_text("x\n")
    r = _merge(stage, tmp_path / "missing.sha256", tmp_path / "dest")
    assert r.returncode == 2
    assert "NOT MERGED" in r.stdout

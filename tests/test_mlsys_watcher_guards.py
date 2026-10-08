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


# ---------------------------------------------------------------------------
# The capacity wait: detect and launch are ONE step
# ---------------------------------------------------------------------------
#
# The failure being guarded: capacity for this GPU type appears and disappears
# inside ~90s (measured 2026-10-08: visible at 00:49:14Z, gone by the 00:50:38Z
# poll). A watcher that checks in one tick and launches in the next is racing a
# window shorter than its own tick, so the launch has to happen in the same
# iteration as the detection, and a human's "go now" has to interrupt the sleep.

POLL_LOOP_ANCHOR = 'avail=$(regions_with_capacity)'


def _script_with_stubs(tmp_path: Path, extra: dict[str, str]) -> tuple[Path, Path]:
    """A copy of the watcher with the network functions stubbed out.

    The script resolves the repo from its own location, so the copy goes in a
    sandbox with a dummy credentials file -- no real key is read and no request
    leaves the machine.
    """
    import re
    sandbox = tmp_path
    (sandbox / "sub").mkdir(parents=True, exist_ok=True)
    (sandbox / "runpod_creds.md").write_text("LAMBDA_API_KEY = dummy\n")

    src = SRC
    for name, body in extra.items():
        i = src.index(f"{name}() {{")
        j = src.index("\n}\n", i) + 3
        # The `;` before the closing brace is load-bearing: `{ echo 0 }` is a
        # syntax error, because after a complete command the `}` is read as an
        # argument and the group never closes.
        src = src[:i] + f"{name}() {{ {body}; }}\n" + src[j:]
    p = sandbox / "sub" / "watcher.sh"
    p.write_text(src)
    import subprocess
    check = subprocess.run(["bash", "-n", str(p)], capture_output=True, text=True)
    assert check.returncode == 0, f"the stubbed watcher does not parse: {check.stderr}"
    assert re.search(r"^count_instances\(\) \{ echo", src, re.M), "stub not applied"
    return p, sandbox


def _run_watcher(script: Path, sandbox: Path,
                 observe_seconds: float | None = None, **env):
    """Run the stubbed watcher, optionally only long enough to see the first
    iteration.

    The loop's own deadline is in whole seconds (awk truncates the hours), so a
    sub-second value skips the body entirely -- `observe_seconds` stops the
    process from outside instead, and every `say` is its own `tee` that has
    already flushed by the time it returns, so the lines logged so far survive
    the signal.
    """
    import subprocess
    e = {"PATH": "/usr/bin:/bin:/usr/sbin:/sbin",
         "HOME": str(sandbox),
         "MLSYS_SESSION_DIR": str(sandbox / "sess"),
         "MLSYS_PYTHON": "/usr/bin/python3",
         "MLSYS_CAPACITY_WAIT_HOURS": "0.01",
         "MLSYS_POLL_BASE_S": "30", "MLSYS_POLL_JITTER_S": "5"}
    e.update(env)
    proc = subprocess.Popen(["bash", str(script)], stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE, text=True, env=e)
    if observe_seconds is None:
        out, err = proc.communicate(timeout=120)
    else:
        try:
            out, err = proc.communicate(timeout=observe_seconds)
        except subprocess.TimeoutExpired:
            proc.terminate()
            out, err = proc.communicate(timeout=20)
    return subprocess.CompletedProcess(proc.args, proc.returncode, out, err)


NO_NETWORK = {"count_instances": "echo 0",
              "regions_with_capacity": "true",
              "capacity_status": "echo 200"}


def test_the_launch_sits_in_the_detection_iteration(tmp_path):
    """Not a handoff to a second loop: the launch call is inside the same
    iteration that read the capacity report."""
    i = SRC.index(POLL_LOOP_ANCHOR)
    body = SRC[i:SRC.index("\ndone\n", i)]
    assert 'try_launch "detected"' in body, (
        "the launch left the iteration that detected capacity")
    assert body.index(POLL_LOOP_ANCHOR) < body.index('try_launch "detected"')


def test_the_guard_is_the_only_path_that_launches(tmp_path):
    """A second launch site would be a second place for the guard to be
    forgotten -- the shape that orphans a second instance."""
    posts = [l for l in SRC.splitlines()
             if "instance-operations/launch" in l
             and not l.lstrip().startswith("#")]
    assert len(posts) == 1, f"{len(posts)} launch POST sites; expected 1"
    i = SRC.index(posts[0])
    guard = SRC.rindex("already_running", 0, i)
    assert SRC.index("try_launch() {") < guard < i, (
        "the instance-count check is not between the entry point and the POST")


def test_a_human_override_is_checked_before_the_api_work(tmp_path):
    """LAUNCH_NOW must not wait behind a slow capacity call."""
    loop = SRC.index('while [ "$(date -u +%s)" -lt "$CAPACITY_WAIT_DEADLINE" ]')
    override = SRC.index('[ -f "$LAUNCH_NOW" ]; then', loop)
    first_api = SRC.index("status=$(capacity_status)", loop)
    assert override < first_api
    assert "interruptible_sleep 10" in SRC, "the nap is not sliced for an override"


def test_the_backoff_covers_rate_limits_and_server_errors(tmp_path):
    assert "429|5*)" in SRC
    assert "BACKOFF_MAX" in SRC and "MLSYS_BACKOFF_MAX_S" in SRC
    assert "BACKOFF=0" in SRC, "the backoff never resets after a good poll"


def test_a_foreign_instance_is_refused_not_adopted(tmp_path):
    """Behaviour, not shape: an instance this watcher did not launch must stop
    it -- with the refusal's exit code (5), not the deadline's (3) -- even when
    a human has just asked it to launch."""
    script, sandbox = _script_with_stubs(
        tmp_path, dict(NO_NETWORK, count_instances="echo 1"))
    (sandbox / "sess").mkdir()
    (sandbox / "sess" / "LAUNCH_NOW").write_text("")
    r = _run_watcher(script, sandbox, MLSYS_LAUNCH_DRY_RUN="1")
    assert r.returncode == 5, r.stdout + r.stderr
    assert "REFUSING TO ADOPT" in r.stdout
    assert "CAPACITY WAIT EXPIRED" not in r.stdout


def test_a_detected_window_launches_in_the_same_poll_line(tmp_path):
    """The whole point: one iteration produces both the detection and the
    attempt. In dry-run mode the attempt cannot create an instance, so the
    evidence is that both lines carry the same attempt number."""
    script, sandbox = _script_with_stubs(
        tmp_path, dict(NO_NETWORK,
                       regions_with_capacity="echo us-east-1",
                       count_instances="echo 0"))
    r = _run_watcher(script, sandbox, observe_seconds=8,
                     MLSYS_LAUNCH_DRY_RUN="1")
    out = r.stdout
    assert "capacity=yes" in out, out
    lines = out.splitlines()
    detect = [l for l in lines if "capacity=yes" in l][0]
    assert "regions=us-east-1" in detect
    attempt = detect.split("attempt=")[1].split()[0]
    launched = [l for l in lines if "LAUNCH_DRY_RUN reason=detected" in l]
    assert launched, out
    assert "attempt={}".format(attempt) in detect
    assert "region=us-east-1" in launched[0]
    # same iteration: nothing was polled between the report and the attempt
    assert lines.index(detect) < lines.index(launched[0])
    assert not [l for l in lines[lines.index(detect) + 1:lines.index(launched[0])
                     ] if "attempt=" in l]


def test_a_refused_launch_keeps_the_override_and_bills_nothing(tmp_path):
    """A refusal is not a success: the file stays so the attempt is retried,
    and no instance id is set (so nothing is terminated later by mistake)."""
    script, sandbox = _script_with_stubs(tmp_path, dict(NO_NETWORK))
    (sandbox / "sess").mkdir()
    now = sandbox / "sess" / "LAUNCH_NOW"
    now.write_text("")
    r = _run_watcher(script, sandbox, observe_seconds=8,
                     MLSYS_LAUNCH_DRY_RUN="1")
    assert now.exists(), "a refused launch discarded the override"
    assert "LAUNCH_DRY_RUN reason=launch_now" in r.stdout
    assert "TERMINATING" not in r.stdout, "a dry run must not terminate anything"


def test_an_unreadable_instance_count_does_not_end_the_wait(tmp_path):
    """`count_instances` failing is not evidence an instance exists; exiting on
    it would end a 72h watch on one transient API error. The guard still fails
    closed, because it refuses to launch on an unknown count."""
    script, sandbox = _script_with_stubs(
        tmp_path, dict(NO_NETWORK, count_instances="echo unknown"))
    # a 1s retry cadence and a 2s horizon so the test sees two full iterations
    r = _run_watcher(script, sandbox, MLSYS_POLL_BASE_S="1",
                     MLSYS_CAPACITY_WAIT_HOURS="0.0006")
    assert r.returncode == 3, r.stdout + r.stderr
    assert "FATAL" not in r.stdout
    assert "instance_count=unknown" in r.stdout
    assert r.stdout.count("next_sleep=1s (retrying)") >= 2, r.stdout


def test_every_fallback_attempt_names_both_its_reason_and_region(tmp_path):
    """Each attempt in `launch_now`'s region-fallback loop must report BOTH
    fields, and must report *the region it is trying*.

    The failure this pins is a call whose arguments do not match the callee's
    signature: the attempt still happens and the line still looks like a launch
    attempt, so nothing downstream notices, but the log becomes useless for
    telling "we tried us-midwest-1" apart from "we were told the reason was
    us-midwest-1" -- and `reason=` is what the campaign's reporting keys on.
    """
    import re
    script, sandbox = _script_with_stubs(
        tmp_path, dict(NO_NETWORK, regions_with_capacity="true"))
    (sandbox / "sess").mkdir()
    (sandbox / "sess" / "LAUNCH_NOW").write_text("")
    pref = "us-east-1,us-midwest-1,us-west-1,us-south-1,us-west-2"

    r = _run_watcher(script, sandbox, observe_seconds=8,
                     MLSYS_LAUNCH_DRY_RUN="1", MLSYS_REGION_PREF=pref)
    attempts = [l for l in r.stdout.splitlines() if "LAUNCH_DRY_RUN" in l]

    assert len(attempts) == 5, r.stdout
    for wanted, line in zip(pref.split(","), attempts):
        assert "reason=launch_now" in line, f"reason field is wrong: {line}"
        assert f"region={wanted}" in line, f"region field is wrong: {line}"
        # and nothing that is a region may have landed in the reason field
        assert not re.search(r"reason=us-", line), \
            f"a region name is in the reason field: {line}"
        got = re.search(r"\bregion=(\S+)", line)
        assert got and got.group(1) == wanted, f"wrong region: {line}"

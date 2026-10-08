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

# A token value that must never appear in any log, output or file the watcher
# writes. Runtime-checked rather than grepped for, because "does this line print
# the value" is not a question a regex answers reliably.
HF_TOKEN_SENTINEL = "hf_SENTINEL_do_not_leak_2851b1d1f"


def _script_with_stubs(tmp_path: Path, extra: dict[str, str]) -> tuple[Path, Path]:
    """A copy of the watcher with the network functions stubbed out.

    The script resolves the repo from its own location, so the copy goes in a
    sandbox with a dummy credentials file -- no real key is read and no request
    leaves the machine.
    """
    import re
    sandbox = tmp_path
    (sandbox / "sub").mkdir(parents=True, exist_ok=True)
    (sandbox / "runpod_creds.md").write_text(
        "LAMBDA_API_KEY = dummy-lambda-key\n"
        f"HF_TOKEN = {HF_TOKEN_SENTINEL}\n")

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


# ---------------------------------------------------------------------------
# The 2026-10-08 run: three ways the setup lied about being ready
# ---------------------------------------------------------------------------
#
# The run reached a paid 8xA100, staged everything, reported every dataset
# "verified", started the manifest, and died two seconds later with
# `No module named 'transformers'`. Then it billed for two more hours, because
# the completion check could not fire either. Each of the three causes is
# pinned here.

POD_ENV = REPO / "scripts" / "mlsys_pod_env.sh"


def test_the_interpreter_resolution_fails_closed():
    """The old form ended in `else command -v python3`, so a pod with no conda
    silently ran the campaign on a stock system interpreter."""
    i = SRC.index("RPY_RESOLVE='")
    rpy = SRC[i:SRC.index("'", SRC.index("exit 1", i))]
    assert "command -v python3" not in rpy, \
        "the pod interpreter still falls back to a bare python3"
    assert "exit 1" in rpy, "the resolution cannot report absence"
    assert "rasd-gpu" in rpy, (
        "the resolution does not look for the env the bootstrap creates; "
        "looking only for `rasd` is how a provisioned pod went unseen")
    assert '${RPY:-python3}' not in SRC, \
        "a remote command still falls back to python3"


def test_the_interpreter_is_never_a_command_substitution():
    """The bug that cost a second 8xA100 launch.

    RPY used to be a `$( ... )' EXPRESSION. Interpolated into a command that
    follows it with arguments (`$(...) -c "..."`) it happens to work, which is
    why the data verification looked fine. Passed to ssh as the WHOLE command it
    does not: the pod runs the echoed path with no arguments, python starts and
    prints nothing, and the resolution comes back empty.
    """
    i = SRC.index("RPY_RESOLVE='")
    rpy = SRC[i:SRC.index("\n", SRC.index("exit 1", i))]
    assert not rpy.lstrip("RPY_RESOLVE='").startswith("$("), \
        "the resolver is a command substitution again"
    assert "$(" not in rpy, "the resolver contains a command substitution"

    # resolution prints the path, and the result is validated before use
    assert 'PY_REMOTE=$(ssh $SSH_OPTS "$SSH_USER@$IP" "$RPY_RESOLVE"' in SRC, \
        "the resolver is not run as a command that prints"
    assert "INTERPRETER=" in SRC, \
        "the watcher does not read the path the provisioning script printed"
    i2 = SRC.index("case \"$PY_REMOTE\" in")
    assert "/*)" in SRC[i2:i2 + 300], \
        "the resolved interpreter is not required to be an absolute path"
    assert "test -x" in SRC, "the resolved interpreter is never checked for existence"
    # and it is used as a plain path afterwards
    assert "RPY=$PY_REMOTE" in SRC


def _code_lines(src: str) -> str:
    """The script with comment lines removed.

    Assertions about what the script DOES have to look at code: a comment that
    names a removed hazard would otherwise keep a "the hazard is gone" check
    green, which is how the old MLSYS_ADOPT_EXISTING check passed on a comment.
    """
    return "\n".join(l for l in src.splitlines() if not l.lstrip().startswith("#"))


def test_the_completion_check_cannot_match_itself():
    """The check must not embed a process pattern its own command line carries.

    `ssh host 'pgrep -f mlsys_manifest.sh'` runs a remote shell whose argv holds
    that pattern, so pgrep matched the checker and returned 0 forever -- verified
    on the pod on 2026-10-08: rc=0 with no manifest running, and therefore no
    reachable "finished" state. This is asserted structurally rather than by
    running pgrep, because the semantics being relied on are the pod's procps,
    not macOS's.
    """
    # the COMPLETION decision must not involve a process pattern at all
    i = SRC.index('if [ -f ~/manifest.rc ]; then printf "rc=%s')
    cmd = SRC[i:SRC.index("else echo \"rc=RUNNING\"", i)]
    assert "pgrep" not in cmd, "the completion check is a process match again"
    assert "mlsys_manifest" not in cmd, (
        "the completion check embeds the manifest's own command string, which is "
        "exactly what made the pattern match the checker")
    assert "pgrep -f mlsys_manifest" not in _code_lines(SRC), \
        "the self-matching liveness check is back in the code"
    # the LIVENESS check may use a pattern, but only a self-excluding one
    live = _code_lines(SRC).splitlines()
    live = [l for l in live if "pgrep" in l]
    assert live, "there is no liveness check at all"
    for l in live:
        assert "[b]ash scripts/mlsys_manifest.sh" in l, (
            f"a liveness pattern can match its own command line: {l.strip()}")


def test_completion_is_read_from_a_marker_not_from_a_process_list():
    # the wrapper records the manifest's own exit code
    assert "echo \\$? > ~/manifest.rc" in SRC, \
        "the manifest is not wrapped, so its exit code is lost"
    # a stale marker from an earlier attempt must not report a completion
    assert "rm -f ~/manifest.rc" in SRC, \
        "a stale completion marker can report a run that never started"
    # read only after a successful ssh, and only a numeric value counts
    assert 'if [ -f ~/manifest.rc ]; then printf "rc=%s' in SRC, \
        "the marker is not read"
    assert 'echo "rc=RUNNING"' in SRC, \
        "a running manifest is not distinguishable from a finished one"
    assert "grep -qE '^[0-9]+$'" in SRC, (
        "a non-numeric marker is not distinguished from a completion")
    assert "MANIFEST_RC=" in SRC, "the manifest's exit code is not kept"
    # the ssh status is checked BEFORE the marker is believed
    loop = _wait_loop()
    assert loop.index('if [ "$rc" -ne 0 ]') < loop.index("MANIFEST_RC=$marker"), \
        "an ssh failure can be read as a completion"
    # an aborted manifest is an incident, not a clean finish
    assert "ABORTED" in loop


def test_the_remote_setup_provisions_the_campaign_environment():
    """The watcher must BUILD the environment, not hope for one. The image this
    ran on had no conda at all."""
    assert POD_ENV.exists(), "scripts/mlsys_pod_env.sh is missing"
    env = POD_ENV.read_text()
    assert "mlsys_pod_env.sh" in SRC, \
        "the watcher never provisions the campaign environment"
    assert SRC.index("mlsys_pod_env.sh") < SRC.index('say "starting the manifest"'), \
        "the environment is provisioned after the manifest starts"
    # a failure is fatal, and it terminates first
    tail = SRC[SRC.index("mlsys_pod_env.sh"):]
    tail = tail[:tail.index('say "starting the manifest"')]
    assert "terminate_and_confirm" in tail and "exit 6" in tail, (
        "a failed provisioning leaves a paid instance running")
    # the provisioning script's own prerequisites
    for want, why in (
            ("conda create -n", "the env is never created"),
            ("requirements-lock.txt", "the pins are not the locked ones"),
            ("--no-build-isolation", "flash-attn cannot build without it"),
            ("diptest", "diptest is missing from the lock file but the dip test needs it"),
            ("torch.cuda.is_available", "the check does not prove CUDA works"),
            ("exit 2", "a failure does not exit non-zero")):
        assert want in env, f"mlsys_pod_env.sh: {why}"


def test_the_environment_is_proven_before_a_single_stage_runs():
    """`pip` exiting 0 is not "the campaign can run". The last word is an import
    of what the stages import, through the interpreter they will use."""
    before_manifest = SRC[:SRC.index('say "starting the manifest"')]
    assert "import torch, transformers, bitsandbytes, flash_attn, diptest" in before_manifest, \
        "nothing proves the stage dependencies import before the run starts"
    check = before_manifest[before_manifest.index("import torch, transformers"):]
    assert "terminate_and_confirm" in check, (
        "an unusable environment does not stop the run before it bills")


# ---------------------------------------------------------------------------
# The 2026-10-08 run, part two: stopping the meter
# ---------------------------------------------------------------------------

POD_ENV_SRC = (REPO / "scripts" / "mlsys_pod_env.sh").read_text()
MANIFEST_SRC = (REPO / "scripts" / "mlsys_manifest.sh").read_text()
STALL_JSON_PATH = REPO / "configs" / "mlsys_stall_thresholds.json"


def _incident_block() -> str:
    """The watcher's post-wait handling of an aborted manifest."""
    i = SRC.index("collect_incident() {")
    return SRC[i:SRC.index("MANIFEST_RC=\"\"", i)]


def _wait_loop() -> str:
    i = SRC.index("MANIFEST_RC=\"\"")
    return SRC[i:SRC.index("interruptible_sleep \"$MANIFEST_POLL\"", i)]


def test_an_aborted_manifest_pulls_logs_and_terminates():
    """rc != 0 is an incident, not a result: capture the logs, stop the meter,
    and do NOT run the normal verify-and-merge path over a broken run."""
    loop = _wait_loop()
    assert 'MANIFEST_ABORTED=1' in loop, "an aborted manifest is not flagged"
    # the aborting branch must be the numeric-marker branch, not the ssh one
    aborted = loop[loop.index("ABORTED"):]
    assert "collect_incident" in aborted, "no logs are collected on an abort"

    after = SRC[SRC.index("if [ \"${MANIFEST_ABORTED:-0}\" = \"1\" ];"):]
    head = after[:after.index("\nfi\n") + 4]
    assert "terminate_and_confirm" in head, "an incident does not terminate"
    assert "exit 7" in head, "an incident does not exit non-zero"
    # and it stops BEFORE the pull/merge section
    assert SRC.index("if [ \"${MANIFEST_ABORTED:-0}\" = \"1\" ];") \
        < SRC.index("per-run staged pull"), \
        "the incident path runs after the normal pull, so it is not fail-fast"


def test_the_incident_directory_is_named_and_populated_as_the_plan_says():
    blk = _incident_block()
    assert 'results/mlsys/incident_$(date -u +%Y%m%dT%H%M%SZ)' in blk, \
        "incidents do not land in results/mlsys/incident_<UTC>/"
    assert "mkdir -p" in blk
    for f in ("manifest.log", "manifest.rc", "pod_env.log", "RUN_LOG.txt"):
        assert f in blk, f"the incident does not capture {f}"
    assert "incident.txt" in blk, "the incident does not state its own reason"


def test_a_stall_or_a_vanished_manifest_stops_the_run():
    loop = _wait_loop()
    # (a) process gone with no marker
    assert 'alive=0' in loop or '[ "$alive" = "0" ]' in loop
    assert "no completion marker" in loop, \
        "a manifest that vanished without writing rc is not detected"
    # (b) no progress for the stage's limit
    assert "stall_minutes_for" in loop, "the stall limit is not read per stage"
    assert "PROGRESS_SINCE" in loop and "idle" in loop
    assert "STALL:" in loop, "a stall is not reported"
    # both paths are incidents, and both set the flag that terminates
    assert loop.count("collect_incident") >= 2
    assert loop.count("MANIFEST_ABORTED=1") >= 2
    # liveness must not self-match: the bracket form cannot match its own argv
    assert 'pgrep -f "[b]ash scripts/mlsys_manifest.sh"' in loop, \
        "the liveness pattern can match the shell running it"


def test_both_watchdogs_read_one_threshold_table():
    """Two watchdogs that disagree about 'quiet' are worse than one: the pod
    would kill a run the operator's side still considers healthy, or vice versa."""
    assert STALL_JSON_PATH.exists(), "the shared threshold file is missing"
    import json
    d = json.loads(STALL_JSON_PATH.read_text())
    assert isinstance(d.get("default_minutes"), int) and d["default_minutes"] > 0
    assert d.get("stages"), "no per-stage thresholds"
    # the watcher reads it, the manifest reads it, from the same path
    assert "mlsys_stall_thresholds.json" in SRC
    assert "mlsys_stall_thresholds.json" in MANIFEST_SRC
    # a stage that is not in the table is not exempt
    assert 'd.get("stages", {}).get(name, d.get("default_minutes"' in SRC
    assert 'd.get("stages", {}).get(name, d.get("default_minutes"' in MANIFEST_SRC


def test_every_stage_override_is_justified():
    """An override with no reason is how a watchdog quietly stops watching."""
    import json
    d = json.loads(STALL_JSON_PATH.read_text())
    just = d.get("_justification", {})
    for stage, minutes in d["stages"].items():
        assert stage in just, f"stage '{stage}' has a threshold with no reason"
        assert isinstance(minutes, int) and minutes > 0
    # the long stages are the ones with raised limits, and they are raised
    assert d["stages"]["natural_spec_gated_256k"] > d["default_minutes"]


def test_the_pod_watchdog_is_self_enforcing_and_cannot_kill_a_reused_pid():
    assert "stall_watchdog()" in MANIFEST_SRC, \
        "the pod has no stall watchdog of its own"
    assert "stall_watchdog &" in MANIFEST_SRC, "the pod watchdog is never started"
    # it must be started before the first stage, or stage 1 is unwatched
    assert MANIFEST_SRC.index("stall_watchdog &") < MANIFEST_SRC.index("stage gate_calibration")
    # the progress signal both sides read
    assert 'interim "STAGE_START name=$name' in MANIFEST_SRC, \
        "no stage-start line, so a long stage looks like a stall"
    # interim must reach stdout as well as RUN_LOG.txt
    assert "| tee -a \"$OUT/RUN_LOG.txt\"" in MANIFEST_SRC
    # a stall exits NON-ZERO so the watcher's fail-fast path sees it
    assert "exit 9" in MANIFEST_SRC and "SIGNALLED" in MANIFEST_SRC
    # and the watchdog never signals a PID it no longer owns
    assert 'ps -p "$MANIFEST_PID" >/dev/null 2>&1' in MANIFEST_SRC, \
        "the watchdog can signal a reused PID"
    assert "STALL_WATCHDOG_PID" in MANIFEST_SRC, \
        "the watchdog is not cleaned up when the manifest exits"


def test_the_hf_token_is_required():
    """Every target model is a gated repo (meta-llama/Llama-2-7b-hf and
    meta-llama/Llama-3.1-8B both return 200 only with a token). Missing must stop
    the run before it launches and bills."""
    assert "HF_TOKEN_VALUE" in SRC, "the HF token is not read at all"
    i = SRC.index("HF_TOKEN_VALUE=$(grep")
    block = SRC[i:SRC.index("say \"HF token: present", i) + 60]
    assert "exit 2" in block, "a missing HF token does not stop the run"
    assert "HF_TOKEN='$HF_TOKEN_VALUE'" in SRC, "the token is not forwarded"
    # and the run really does refuse without it
    assert "gated" in block


def test_a_missing_hf_token_stops_the_watcher(tmp_path):
    """Behavioural: the value is deleted from the sandbox credentials and the
    watcher must refuse before doing anything else."""
    script, sandbox = _script_with_stubs(tmp_path, dict(NO_NETWORK))
    (sandbox / "runpod_creds.md").write_text("LAMBDA_API_KEY = dummy\n")
    r = _run_watcher(script, sandbox, observe_seconds=8)
    assert r.returncode == 2, r.stdout + r.stderr
    assert "no HF_TOKEN line" in r.stdout


def test_the_hf_token_value_never_reaches_a_log(tmp_path):
    """Behavioural, with a sentinel: whatever the watcher writes -- stdout, the
    watcher log, the incident dir -- must not contain the token."""
    script, sandbox = _script_with_stubs(
        tmp_path, dict(NO_NETWORK, regions_with_capacity="echo us-east-1"))
    (sandbox / "sess").mkdir()
    r = _run_watcher(script, sandbox, observe_seconds=8,
                     MLSYS_LAUNCH_DRY_RUN="1")
    leaked = []
    if HF_TOKEN_SENTINEL in (r.stdout + r.stderr):
        leaked.append("stdout/stderr")
    for f in sandbox.rglob("*"):
        # the credentials file IS the input; it is supposed to contain the token.
        # Everything else is output the watcher produced.
        if f.is_file() and f.name != "runpod_creds.md":
            try:
                if HF_TOKEN_SENTINEL in f.read_text(errors="ignore"):
                    leaked.append(str(f.relative_to(sandbox)))
            except Exception:                    # noqa: BLE001
                pass
    assert not leaked, f"the HF token leaked into: {leaked}"
    assert "not shown" in r.stdout, "the log should record presence only"


def test_the_manifest_run_sources_the_pod_env_and_the_token():
    start = SRC[SRC.index('say "starting the manifest"'):]
    start = start[:start.index("manifest.rc' > ~/manifest.log")] 
    assert ". ~/RASD/.pod_env.sh" in start, \
        "the manifest does not inherit the HF cache / NCCL env"
    assert "HF_TOKEN=" in start, "the token is not in the manifest's environment"
    assert "MLSYS_STALL_MINUTES" in start, \
        "the stall limit is not passed to the pod watchdog"


def test_pod_env_applies_the_operator_notes():
    """Everything that used to live only in runpod_creds.md."""
    for want, why in (
            ("HF_HOME", "the HF cache root is not set"),
            ("HF_HUB_CACHE", "HF_HUB_CACHE is not set"),
            ("TRANSFORMERS_CACHE", "TRANSFORMERS_CACHE is not set"),
            ("PIP_CACHE_DIR", "the pip cache is not set"),
            ("mkdir -p \"$HF_HUB_CACHE\"", "the hub/ dir is not pre-created"),
            ("NCCL_TIMEOUT", "the NCCL settings from the validated flow are missing"),
            ("PYTORCH_CUDA_ALLOC_CONF", "the allocator setting is missing"),
            ("src/__init__.py", "zero-byte __init__.py files are not restored"),
            ("flash_attn-", "no prebuilt flash-attn wheel is constructed"),
            ("_GLIBCXX_USE_CXX11_ABI", "the wheel ABI is guessed, not probed"),
            ("--no-build-isolation", "the source-build fallback is gone"),
            ("model_info", "gated model access is not proven"),
            ("nvidia-smi", "a dirty GPU is not detected")):
        assert want in POD_ENV_SRC, f"mlsys_pod_env.sh: {why}"
    # a mounted filesystem is used only if it IS mounted
    assert "if [ -d \"$FS_ROOT\" ]" in POD_ENV_SRC, \
        "the cache points at a path that may not exist"
    # the env file is what the manifest sources
    assert ".pod_env.sh" in POD_ENV_SRC


def test_the_token_reaches_every_remote_command_that_needs_it():
    """The 2026-10-08 near-miss: HF_TOKEN was added to the manifest command but
    not to the provisioning command, so pod_env.sh failed closed on the pod and
    the watcher terminated an 8x instance ten minutes after launching it.

    Anything that runs on the pod and touches a gated model needs the token. The
    assert is on the commands THEMSELVES -- extracted from the script -- rather
    than on the presence of the string somewhere in the file, because "the token
    appears once" is what was true when this failed."""
    # the provisioning command
    i = SRC.index('"cd ~/RASD && HF_TOKEN=')
    prov = SRC[i:SRC.index('> ~/pod_env.log 2>&1', i)]
    assert "mlsys_pod_env.sh" in prov, "this is not the provisioning command"
    assert "HF_TOKEN='$HF_TOKEN_VALUE'" in prov, \
        "the provisioning command does not carry the token"

    # the manifest command
    j = SRC.index('"cd ~/RASD && rm -f ~/manifest.rc')
    man = SRC[j:SRC.index("echo started\"", j)]
    assert "HF_TOKEN='$HF_TOKEN_VALUE'" in man, \
        "the manifest command does not carry the token"
    assert ". ~/RASD/.pod_env.sh" in man, \
        "the manifest does not inherit the pod environment"

    # and nothing else on the pod needs it: the remaining ssh calls read files,
    # resolve an interpreter, or check a marker
    for line in _code_lines(SRC).splitlines():
        if 'ssh $SSH_OPTS "$SSH_USER@$IP"' in line and "pod_env.sh" in line:
            assert "HF_TOKEN" in line, line

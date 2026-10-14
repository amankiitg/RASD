"""The 1x run_experiment probe must exercise the campaign path, not a shortcut.

engine_cap_smoke died on 2026-10-08T15:30Z on
`wandb.errors.errors.UsageError: No API key configured` -- before any GPU work,
on a code path that had never executed on a pod. That cost an 8xA100 launch to
find. This probe exists so the NEXT such assumption is found on one GPU for
about a dollar, which only works if the probe really takes the campaign's path.

The two failure modes it has to avoid are opposites:

  * the probe drifts into a convenient subset (no quantisation, a smaller dtype,
    no draft window cap), and then passes while the campaign's path is broken;
  * the probe leaks 1x-shaped settings back into the campaign, which is what the
    1x gate repro's isolation test exists to prevent.

So the config must MIRROR the campaign's defaults on every lever that changes
what the engine does, and nothing the campaign reads may reference it.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import pytest
import yaml

REPO = Path(__file__).resolve().parent.parent
PROBE_CFG = REPO / "configs" / "mlsys_runexp_probe_1x.yml"
PROBE_SH = REPO / "scripts" / "mlsys_runexp_probe_1x.sh"
CAMPAIGN_CFG = REPO / "configs" / "mlsys_arm2_native_cappeddraft.yml"
MANIFEST = REPO / "scripts" / "mlsys_manifest.sh"
WATCHER = REPO / "scripts" / "mlsys_watch_and_run.sh"


def _probe() -> dict:
    return yaml.safe_load(PROBE_CFG.read_text())


# --------------------------------------------------------------------------
# it must not be reachable from the campaign
# --------------------------------------------------------------------------

def test_no_campaign_path_reads_the_1x_probe():
    assert PROBE_CFG.exists() and PROBE_SH.exists()
    for rel in ("scripts/mlsys_manifest.sh",
                "scripts/mlsys_watch_and_run.sh",
                "scripts/mlsys_pod_env.sh"):
        assert "runexp_probe_1x" not in (REPO / rel).read_text(), (
            f"{rel} references the 1x probe; the campaign would run a "
            f"single-GPU-shaped invocation")
    for p in sorted((REPO / "configs").glob("mlsys_*")):
        if p.name == PROBE_CFG.name:
            continue
        assert "runexp_probe_1x" not in p.read_text(), \
            f"{p.name} references the 1x probe"


def test_the_probe_is_not_an_approved_stage():
    """It must not become a campaign stage by accident: the allowlist is the
    thing that decides what may run on a paid instance."""
    text = WATCHER.read_text() + MANIFEST.read_text()
    stage_names = set(re.findall(r'\b([a-z_]+probe[a-z_]*)\b', text))
    assert "runexp_probe_1x" not in stage_names
    assert "probe_1x" not in MANIFEST.read_text(), \
        "the manifest mentions probe_1x; it is a manual repro, not a stage"


# --------------------------------------------------------------------------
# it must MIRROR the campaign, or it proves nothing about the campaign
# --------------------------------------------------------------------------

@pytest.mark.parametrize("key", ["dtype", "kv_block_size", "prefetch_depth",
                                 "rope_type", "kv_quant", "quantize_draft",
                                 "quantize_target", "draft_window_cap",
                                 "temperature", "top_p"])
def test_the_probe_mirrors_the_campaign_defaults(key):
    """Every lever that changes what the engine computes, held identical: the
    probe's value is only meaningful if it is the campaign's value."""
    probe = _probe()["defaults"][key]
    campaign = yaml.safe_load(CAMPAIGN_CFG.read_text())["defaults"][key]
    assert probe == campaign, (
        f"{key}: the probe uses {probe!r} but the campaign uses {campaign!r}; "
        f"the probe would not be exercising the campaign's path")


def test_the_probe_has_both_a_speculative_run_and_its_target_only_baseline():
    """spec_steps=0 is a different code path (no draft, no verification), and it
    is the one the matched baselines use."""
    groups = {k: v for k, v in _probe().items()
              if k not in ("defaults", "canary")}
    steps = set()
    for g in groups.values():
        for level in g["levels"]:
            steps.add(level.get("spec_steps", _probe()["defaults"]["spec_steps"]))
    assert {0, 4} <= steps, f"the probe does not cover both paths: {steps}"
    for g in groups.values():
        for level in g["levels"]:
            assert level.get("checkpoint_every") == 0, (
                "a checkpoint cannot be resumed (the per-token gaps are not "
                "stored), so the campaign asserts 0 everywhere")


def test_the_probe_is_small_enough_to_be_cheap():
    """A few tokens at 4k on a 1B pair: the point is a forward pass, not a
    measurement."""
    d = _probe()["defaults"]
    assert d["max_new_tokens"] <= 16
    for g in (v for k, v in _probe().items() if k not in ("defaults", "canary")):
        for level in g["levels"]:
            assert level["context_length"] <= 8192


# --------------------------------------------------------------------------
# the invocation must be the manifest's form
# --------------------------------------------------------------------------

def test_the_probe_uses_the_manifests_command_form():
    """Same arguments as a campaign stage, so a flag the manifest passes and the
    probe forgets cannot hide a failure."""
    sh = PROBE_SH.read_text()
    for arg in ("--config", "--groups", "--output", "--stage-id",
                "--timeout-per-run-s", "--abort-on-failure", "--log-per-token",
                "--memory-trace", "--save-generated-text",
                "--save-generated-tokens"):
        assert arg in sh, f"the probe does not pass {arg}"
        assert arg in MANIFEST.read_text(), (
            f"{arg} is not part of the manifest's form; the probe should not "
            f"invent its own")


def test_the_one_deviation_from_the_manifest_form_is_nproc():
    """The manifest passes no --nproc and so runs 8; a 1x cannot. The probe must
    pass 1 explicitly, and must say why, because that is the one thing this
    probe cannot cover."""
    sh = PROBE_SH.read_text()
    assert "--nproc 1" in sh
    assert "nproc" in PROBE_CFG.read_text(), (
        "the config does not record that the probe runs single-GPU")


def _probe_code() -> str:
    """The probe with its comments stripped.

    Assertions about what a script does must not match the prose that explains
    it: the comment saying `command -v python` is the wrong way to resolve an
    interpreter contains that exact string.
    """
    return "\n".join(l for l in PROBE_SH.read_text().splitlines()
                     if not l.lstrip().startswith("#"))


def test_the_probe_refuses_rather_than_guessing_an_interpreter():
    """A fallback interpreter is the failure this project has paid for twice."""
    sh = _probe_code()
    assert "test -f ~/RASD/.pod_env.sh" in sh
    assert "Refusing to guess an interpreter" in sh
    assert "envs/rasd-gpu/bin/python" in sh, (
        "the probe does not require the CAMPAIGN interpreter; any python would "
        "do, and then a bare python3 with no transformers would pass")
    assert 'grep -a "^INTERPRETER=" ~/pod_env.log' in sh, (
        "the probe must resolve the interpreter the way the CAMPAIGN does, from "
        "the INTERPRETER= line provisioning writes")
    assert "command -v python" not in sh, (
        "the probe resolves an interpreter by name; .pod_env.sh does not "
        "activate conda, so that finds /usr/bin/python -- the silent fallback "
        "this project has already paid for twice")


def test_the_probe_asserts_the_wandb_setting_it_depends_on():
    sh = PROBE_SH.read_text()
    assert "WANDB_MODE=disabled" in sh, (
        "the probe does not check the setting that stopped engine_cap_smoke, so "
        "it cannot distinguish 'fixed' from 'not exercised'")
    assert "No API key configured" in sh, \
        "the probe does not look for the exact error it exists to rule out"


def test_the_probe_pulls_its_evidence_home():
    """The artifacts are the record; inspecting them over ssh is how three false
    probe results were produced in this project."""
    sh = PROBE_SH.read_text()
    assert "scp " in sh and "results/mlsys/probe_1x" in sh


def test_the_probe_does_not_assign_to_GREOUPS_special_variables():
    """A bash trap that cost a probe run.

    `GROUPS` is a special READONLY array in bash holding the caller's group ids.
    `GROUPS="PROBE_SPEC PROBE_TARGET"` is silently ignored, and the name then
    expands to a number -- the first run passed `--groups 20` and planned zero
    rows for a group called "20", which reads as a pod problem rather than a
    shell one. `POD_OUT=~/...` is the same family: unquoted, the tilde is
    expanded on THIS machine and the pod is asked to mkdir a laptop path.

    Comments are stripped: the comment explaining the GROUPS trap contains the
    exact assignment it warns against.
    """
    sh = _probe_code()
    assert "GROUPS=" not in sh.replace("PROBE_GROUPS=", ""), (
        "the probe assigns to the readonly GROUPS array")
    assert "PROBE_GROUPS=" in sh
    assert "POD_OUT=~/RASD" not in sh, (
        "the remote path uses an unquoted tilde, which expands locally")
    assert "POD_OUT='~/RASD" in sh, "the remote tilde is not quoted"


def test_the_probe_passes_the_campaigns_environment_not_just_its_arguments():
    """A probe with the right command and the wrong environment lies.

    The first run of this probe reported `GatedRepoError: 401` on
    meta-llama/Llama-3.2-1B, which looked like a campaign blocker. It was not:
    the watcher starts the manifest with HF_TOKEN in its environment
    (scripts/mlsys_watch_and_run.sh:677) and the stages inherit it, while the
    probe's plain `ssh host "..."` had no token at all. Reporting an invented
    blocker is worse than reporting none, because it costs a diagnosis.
    """
    sh = _probe_code()
    assert "HF_TOKEN=" in sh, (
        "the probe does not pass HF_TOKEN; every target model in this campaign "
        "is a gated repo and the load would 401")
    assert "runpod_creds.md" in sh, \
        "the token is not read by label, so it would have to be inline"
    assert "set -a" in sh and ". ~/RASD/.pod_env.sh" in sh, (
        "the manifest sources .pod_env.sh with `set -a` so its exports reach "
        "the stages; the probe must do the same")


def test_the_probe_never_prints_the_token():
    sh = PROBE_SH.read_text()
    assert "not shown" in sh, "the probe does not state that it withholds the token"
    # Any echo/printf of the bare variable would leak it into the log.
    for line in sh.splitlines():
        if "HF_TOKEN_VALUE" in line and ("echo" in line or "printf" in line):
            assert "wc -c" in line or "not shown" in line, (
                f"this line prints the token: {line.strip()}")


def test_the_probe_clears_the_pods_stale_output_before_running():
    """A stage refuses to write into a file that holds another stage's rows.

    The second probe run failed on exactly that guard -- the previous attempt's
    CSV was still on the pod -- which is the guard behaving correctly and the
    probe not. The pod's copy is deleted only after the local archive (step 3)
    has it.
    """
    sh = _probe_code()
    assert "rm -f $POD_CSV" in sh, (
        "the probe does not clear the pod's previous attempt, so the staleness "
        "guard will refuse the re-run for a reason that is not the code")
    assert "attempts/" in sh, (
        "the probe overwrites its previous artifacts instead of archiving them")


def test_the_probe_stages_the_working_tree_and_proves_it():
    """A probe that tests yesterday's code produces yesterday's answer.

    The first version assumed the setup probe's rsync was still current. A fix
    made after that rsync was therefore never exercised, and the probe reported
    the identical failure twice -- which reads as "the fix did not work".
    """
    sh = _probe_code()
    assert "rsync " in sh, "the probe does not stage the working tree"
    assert "sha256sum ~/RASD/run_experiment.py" in sh, (
        "the probe does not verify that the pod is running the code under test")
    assert "byte-identical to this working tree" in sh

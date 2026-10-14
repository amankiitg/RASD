"""Guards for the 1x losslessness repro: isolation, and a comparison that works.

Two things are worth pinning here, and neither is about the engine:

1. ISOLATION. configs/mlsys_lossless_repro_1x.yml and
   configs/mlsys_lossless_determinism_1x.yml are diagnostics. If the manifest or
   the watcher ever referenced either, a diagnosis config would silently become
   part of the campaign -- the failure mode the runexp probe's own test already
   guards against for its config.

2. THE COMPARISON. The repro's entire value is the token-by-token verdict, so
   the driver must be shown to FAIL on a known divergence and PASS on a known
   match. It was validated against the real 2026-10-08T19:33Z incident sidecars
   before the pod was bought; these tests pin the same behaviour with small
   synthetic sidecars so they run offline and forever.

MLSYS_LOSSESS_SKIP_POD=1 runs only step 4 of the driver, against whatever
directory MLSYS_LOSSESS_LOCAL_OUT names, which is what makes (2) testable.
"""

from __future__ import annotations

import json
import os
import pathlib
import subprocess
import sys

import pytest

REPO = pathlib.Path(__file__).resolve().parents[1]
DRIVER = REPO / "scripts" / "mlsys_lossless_repro_1x.sh"
REPRO_CFG = REPO / "configs" / "mlsys_lossless_repro_1x.yml"
DET_CFG = REPO / "configs" / "mlsys_lossless_determinism_1x.yml"


# --------------------------------------------------------------------------
# 1. isolation
# --------------------------------------------------------------------------
@pytest.mark.parametrize("cfg_name", [
    "mlsys_lossless_repro_1x.yml",
    "mlsys_lossless_determinism_1x.yml",
])
def test_diagnostic_config_is_not_part_of_the_campaign(cfg_name):
    """No campaign script may reference a 1x diagnostic config."""
    hits = []
    for path in (REPO / "scripts").glob("*.sh"):
        if path.name.startswith("mlsys_lossless"):
            continue
        text = path.read_text(errors="replace")
        if cfg_name in text:
            hits.append(str(path.relative_to(REPO)))
    for path in (REPO / "configs").glob("mlsys_manifest.yml"):
        if cfg_name in path.read_text(errors="replace"):
            hits.append(str(path.relative_to(REPO)))
    assert hits == [], f"{cfg_name} leaked into the campaign: {hits}"


def test_repro_config_runs_at_the_smallest_contexts_and_both_kv_dtypes():
    """The two knobs the experiment turns must actually be in the file."""
    text = REPRO_CFG.read_text()
    for token in ("context_length: 32768", "context_length: 8192"):
        assert token in text, f"missing {token!r}"
    # One NF4 arm and one bf16 arm per context, each with a matched partner.
    # Counted at LEVEL indentation: `defaults` also sets kv_quant, so a bare
    # substring count would be off by the defaults block.
    assert text.count("    kv_quant: true") == 4, "2 NF4 arms per context x 2"
    assert text.count("    kv_quant: false") == 4, "2 bf16 arms per context x 2"
    assert text.count("    spec_steps: 0") == 4, "every arm needs a target-only partner"


def test_determinism_config_runs_each_mode_twice():
    """The probe is only meaningful if each mode appears twice, identically."""
    text = DET_CFG.read_text()
    assert text.count("    spec_steps: 4") == 2, "two speculative attempts"
    assert text.count("    spec_steps: 0") == 2, "two target-only attempts"
    assert "DET_8K_specA" in text and "DET_8K_specB" in text


# --------------------------------------------------------------------------
# 2. the comparison
# --------------------------------------------------------------------------
def _sidecar(path: pathlib.Path, ids: list[int], gaps: list[float],
             prompt_ids: list[int], kv_dtype: str = "nf4") -> None:
    path.write_text(json.dumps({
        "run_id": path.stem,
        "generated_token_ids": ids,
        "token_gaps": gaps,
        "prompt_token_ids": prompt_ids,
        "prompt_sha256": "0" * 64,
        "kv_dtype": kv_dtype,
        "weight_precision": "fp4",
    }))


def _fixture(root: pathlib.Path, *, spec: list[int], target: list[int],
             spec_gaps: list[float], target_gaps: list[float]) -> pathlib.Path:
    (root / "tokens").mkdir(parents=True, exist_ok=True)
    prompt = list(range(100))
    _sidecar(root / "tokens" / "LOSS_8K_nf4_spec_pg19_train_1_s42.json",
             spec, spec_gaps, prompt)
    _sidecar(root / "tokens" / "LOSS_8K_nf4_targetonly_pg19_train_1_s42.json",
             target, target_gaps, prompt)
    with open(root / "lossless_repro.csv", "w", newline="") as fh:
        fh.write("run_id,status,tokens_generated,n_rounds,acceptance_rate,"
                 "kv_dtype,context_length,error\n")
        fh.write("LOSS_8K_nf4_spec_pg19_train_1_s42,ok,64,39,0.16,,8192,\n")
        fh.write("LOSS_8K_nf4_targetonly_pg19_train_1_s42,ok,64,63,0.0,,8192,\n")
    return root


def _run_driver(fixture: pathlib.Path) -> subprocess.CompletedProcess:
    env = dict(os.environ)
    env["MLSYS_LOSSESS_SKIP_POD"] = "1"
    env["MLSYS_LOSSESS_LOCAL_OUT"] = str(fixture)
    env["MLSYS_LOCAL_PY"] = sys.executable
    return subprocess.run(["bash", str(DRIVER)], cwd=REPO, env=env,
                          capture_output=True, text=True, timeout=180)


def test_driver_fails_on_a_known_divergence(tmp_path):
    """A divergence at a decided position must fail, and be named exactly."""
    spec = [10, 20, 30, 627] + [1] * 60
    target = [10, 20, 30, 26] + [1] * 60
    fx = _fixture(tmp_path, spec=spec, target=target,
                  spec_gaps=[1.0] * 4, target_gaps=[1.0] * 4)
    res = _run_driver(fx)
    assert res.returncode == 1, res.stdout + res.stderr
    assert "verdict=MISMATCH" in res.stdout
    assert "<-- first divergence" in res.stdout
    assert "1 diverging" in res.stdout


def test_driver_passes_on_identical_tokens(tmp_path):
    """The same trajectory must pass, or the test above proves nothing."""
    ids = [10, 20, 30, 40] + [1] * 60
    fx = _fixture(tmp_path, spec=ids, target=list(ids),
                  spec_gaps=[1.0] * 64, target_gaps=[1.0] * 64)
    res = _run_driver(fx)
    assert res.returncode == 0, res.stdout + res.stderr
    assert "verdict=LOSSLESS" in res.stdout
    assert "every (context, kv-dtype) pair is LOSSLESS" in res.stdout


def test_driver_reports_a_below_tie_gate_divergence_as_a_tie(tmp_path):
    """A near-indifferent target is a NUMERIC_TIE, which does NOT fail.

    This is the branch that let one of the four real 1x cells (32k, bf16 KV)
    pass, so it must keep working: the tie allowance is the difference between
    "the engine is broken" and "the target was flipping a coin".
    """
    spec = [10, 20, 30, 627] + [1] * 60
    target = [10, 20, 30, 26] + [1] * 60
    spec_gaps = [1.0, 1.0, 1.0, 0.0] + [1.0] * 60   # exactly indifferent
    fx = _fixture(tmp_path, spec=spec, target=target,
                  spec_gaps=spec_gaps, target_gaps=[1.0] * 64)
    res = _run_driver(fx)
    assert res.returncode == 0, res.stdout + res.stderr
    assert "verdict=NUMERIC_TIE" in res.stdout


def test_driver_never_passes_without_comparing_anything(tmp_path):
    """An empty fixture must FAIL, not vacuously pass.

    The whole reason this repro exists is that the earlier 1x probe "passed"
    while never comparing a single token; a comparison that can silently check
    zero pairs would repeat that mistake in a new place.
    """
    (tmp_path / "lossless_repro.csv").write_text(
        "run_id,status,tokens_generated,n_rounds,acceptance_rate,kv_dtype,"
        "context_length,error\n")
    res = _run_driver(tmp_path)
    assert res.returncode == 1, res.stdout + res.stderr
    assert "compared, 0 diverging" in res.stdout or "0 pair(s) compared" in res.stdout

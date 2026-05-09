"""Regression tests for the four Phase C blocker fixes (2026-05-10).

External code-review surfaced four real bugs that would have made the
bundled Phase C pod session produce zero useful data. These tests lock
in the fixes so a later refactor can't silently re-introduce them.

Fixes covered:
  #2  drop outer torchrun from phase_c_pod_session.sh — orchestrator
      run_experiment.py spawns its own per-row torchrun; wrapping it
      again caused 64-way GPU contention + master_port collisions.
  #3  build_run_configs filtered groups by `A*` prefix; M4 YAMLs use
      `SMOKE` and `M4` as group keys → expanded to zero runs. Fix
      switches to NON_GROUP_KEYS = {defaults, canary} exclusion.
  #4  benchmark_baselines stage: was `bash` (file is python) +
      `--contexts` (actual flag is `--lengths`). Fix invokes via
      `torchrun ... python script.py --lengths ... --distributed`.
  #5  GenerationCheckpoint had `rng_state` field but
      _maybe_save_checkpoint never populated it; resume never
      restored it. Under temperature > 0 (default 1.0), resumed
      runs would diverge from uninterrupted ones. Fix populates +
      restores torch.get_rng_state().

Pure source-inspection tests except where the fix has a runtime
component (Fix #3 is exercised end-to-end via build_run_configs).
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
PHASE_C_SH = (REPO_ROOT / "scripts" / "phase_c_pod_session.sh").read_text()
RUN_EXP_SRC = (REPO_ROOT / "run_experiment.py").read_text()
RASD_INF_SRC = (REPO_ROOT / "src" / "models" / "rasd_inference.py").read_text()


# ---------------------------------------------------------------------------
# Fix #3 — groups filter (NON_GROUP_KEYS exclusion, not A* prefix)
# ---------------------------------------------------------------------------

class TestFix3GroupsFilter:
    def test_no_a_prefix_filter_in_source(self):
        """The buggy `[k for k in cfg if k.startswith("A")]` must be gone."""
        assert 'k.startswith("A")' not in RUN_EXP_SRC, (
            "Fix #3 regression: build_run_configs still filters by A* prefix; "
            "M4 YAMLs (SMOKE, M4) would expand to zero runs"
        )

    def test_non_group_keys_constant_present(self):
        assert "NON_GROUP_KEYS" in RUN_EXP_SRC, (
            "Fix #3 regression: NON_GROUP_KEYS exclusion list missing"
        )

    def test_m3_yaml_still_expands(self):
        """M3 YAML uses A1..A5 — must still expand correctly under the new filter."""
        from run_experiment import build_run_configs
        m3_cfg = {
            "defaults": {"seeds": [42], "spec_steps": 4, "kv_block_size": 512,
                         "prefetch_depth": 1, "target_model_name": "x",
                         "draft_model_name": "y"},
            "A1": {"name": "draft", "factor": "draft_model_name",
                   "levels": [{"id": "A1_lvl1", "draft_model_name": "z"}]},
            "A2": {"name": "k", "factor": "spec_steps",
                   "levels": [{"id": "A2_lvl1", "spec_steps": 2}]},
        }
        runs = build_run_configs(m3_cfg, groups=None, debug=False)
        assert len(runs) == 2
        groups_in_runs = {r["group"] for r in runs}
        assert groups_in_runs == {"A1", "A2"}

    def test_m4_smoke_yaml_expands(self):
        """M4 phase-C smoke YAML uses `SMOKE` group — was previously dropped."""
        from run_experiment import build_run_configs
        m4_smoke_cfg = {
            "defaults": {"seeds": [42], "spec_steps": 4, "kv_block_size": 2048,
                         "prefetch_depth": 1, "target_model_name": "x",
                         "draft_model_name": "y"},
            "SMOKE": {"name": "long_smoke", "factor": "context_length",
                      "levels": [
                          {"id": "SMOKE_ctx32k", "context_length": 32768},
                          {"id": "SMOKE_ctx128k", "context_length": 131072},
                      ]},
        }
        runs = build_run_configs(m4_smoke_cfg, groups=None, debug=False)
        assert len(runs) == 2, (
            "Fix #3 regression: SMOKE group did not expand to runs"
        )
        assert all(r["group"] == "SMOKE" for r in runs)

    def test_m4_final_matrix_yaml_expands(self):
        """M4 final-matrix YAML uses `M4` group — was previously dropped."""
        from run_experiment import build_run_configs
        m4_cfg = {
            "defaults": {"seeds": [42, 123, 456], "spec_steps": 4,
                         "kv_block_size": 2048, "prefetch_depth": 1,
                         "target_model_name": "x", "draft_model_name": "y"},
            "M4": {"name": "matrix", "factor": "context_length",
                   "levels": [{"id": "M4_ctx128k", "context_length": 131072}]},
        }
        runs = build_run_configs(m4_cfg, groups=None, debug=False)
        assert len(runs) == 3  # 1 level * 3 seeds

    def test_canary_and_defaults_still_excluded(self):
        """`canary` and `defaults` keys must NOT be treated as groups."""
        from run_experiment import build_run_configs
        cfg = {
            "defaults": {"seeds": [42], "target_model_name": "x",
                         "draft_model_name": "y", "spec_steps": 4,
                         "kv_block_size": 512, "prefetch_depth": 1},
            "canary": {"id": "canary_s42", "seed": 42},
            "A1": {"name": "x", "factor": "y",
                   "levels": [{"id": "A1_lvl1"}]},
        }
        runs = build_run_configs(cfg, groups=None, debug=False)
        # Only A1's one level x one seed = 1 run; canary+defaults excluded
        assert len(runs) == 1
        assert runs[0]["group"] == "A1"

    def test_groups_filter_argument_still_works(self):
        """Explicit --groups SMOKE selects only that group."""
        from run_experiment import build_run_configs
        cfg = {
            "defaults": {"seeds": [42], "target_model_name": "x",
                         "draft_model_name": "y", "spec_steps": 4,
                         "kv_block_size": 512, "prefetch_depth": 1},
            "SMOKE": {"name": "x", "factor": "y",
                      "levels": [{"id": "SMOKE_ctx32k", "context_length": 32768}]},
            "M4":    {"name": "x", "factor": "y",
                      "levels": [{"id": "M4_ctx128k", "context_length": 131072}]},
        }
        runs = build_run_configs(cfg, groups=["SMOKE"], debug=False)
        assert len(runs) == 1
        assert runs[0]["group"] == "SMOKE"


# ---------------------------------------------------------------------------
# Fix #2 — no outer torchrun on orchestrator stages
# ---------------------------------------------------------------------------

class TestFix2DropOuterTorchrun:
    @pytest.mark.parametrize("stage_name", [
        "long_ctx_smokes", "final_matrix", "profiler_sidecar_pass",
    ])
    def test_orchestrator_stage_does_not_use_outer_torchrun(self, stage_name):
        """Stages that invoke run_experiment.py (the orchestrator) must
        use plain `python`, not `torchrun`. run_experiment.py spawns its
        own per-row torchrun internally."""
        m = re.search(
            rf"^{stage_name}\(\) \{{(.*?)^\}}",
            PHASE_C_SH, re.DOTALL | re.MULTILINE,
        )
        assert m is not None, f"stage function {stage_name}() not found"
        body = m.group(1)
        # Strip comments from the body before checking for command-level
        # torchrun. A bare word "torchrun" inside a comment line is fine
        # (e.g. anti-regression note); only an actual command wrapping
        # run_experiment.py is the bug.
        non_comment_lines = [
            line for line in body.splitlines()
            if not line.lstrip().startswith("#")
        ]
        non_comment_body = "\n".join(non_comment_lines)
        assert not re.search(
            r"torchrun[^|]*?run_experiment\.py", non_comment_body
        ), (
            f"Fix #2 regression: {stage_name} wraps run_experiment.py "
            f"in torchrun — would cause double-torchrun GPU contention"
        )
        # Sanity: the body still invokes run_experiment.py via python
        assert "run_experiment.py" in non_comment_body, (
            f"{stage_name} no longer invokes run_experiment.py at all"
        )

    def test_c6_validation_stage_keeps_torchrun(self):
        """c6_resume_validation.py is a torchrun-direct script (not the
        orchestrator) — keep the torchrun wrapping for that one."""
        m = re.search(
            r"^c6_validation\(\) \{(.*?)^\}",
            PHASE_C_SH, re.DOTALL | re.MULTILINE,
        )
        assert m is not None
        body = m.group(1)
        assert "torchrun" in body, (
            "c6_validation must keep its torchrun (script is direct, "
            "not the orchestrator)"
        )


# ---------------------------------------------------------------------------
# Fix #4 — baselines stage uses python + --lengths + --distributed
# ---------------------------------------------------------------------------

class TestFix4BaselinesInvocation:
    def test_baselines_uses_python_not_bash(self):
        """benchmark_baselines.py is a Python file — must be invoked via
        python (or torchrun, since the script supports --distributed)."""
        m = re.search(
            r"^baseline_validation\(\) \{(.*?)^\}",
            PHASE_C_SH, re.DOTALL | re.MULTILINE,
        )
        assert m is not None
        body = m.group(1)
        assert "bash scripts/benchmark_baselines.py" not in body, (
            "Fix #4 regression: baseline stage still invokes Python file "
            "via bash — would fail to run"
        )
        # Either `python ...benchmark_baselines.py` or `torchrun ...benchmark_baselines.py`
        assert re.search(r"(python|torchrun)[^|]*?benchmark_baselines\.py", body), (
            "Fix #4 regression: baseline stage doesn't invoke Python script "
            "via python/torchrun"
        )

    def test_baselines_uses_lengths_not_contexts(self):
        m = re.search(
            r"^baseline_validation\(\) \{(.*?)^\}",
            PHASE_C_SH, re.DOTALL | re.MULTILINE,
        )
        body = m.group(1)
        assert "--contexts" not in body, (
            "Fix #4 regression: baseline stage uses --contexts; "
            "actual flag is --lengths"
        )
        assert "--lengths" in body, (
            "Fix #4 regression: --lengths flag missing"
        )

    def test_baselines_uses_distributed(self):
        """At long context (1M), single-process won't fit — need ring."""
        m = re.search(
            r"^baseline_validation\(\) \{(.*?)^\}",
            PHASE_C_SH, re.DOTALL | re.MULTILINE,
        )
        body = m.group(1)
        assert "--distributed" in body, (
            "Fix #4 regression: --distributed missing; would single-process "
            "the 1M ctx run and OOM"
        )


# ---------------------------------------------------------------------------
# Fix #5 — RNG state populated on save + restored on resume
# ---------------------------------------------------------------------------

class TestFix5RngState:
    def test_save_populates_rng_state(self):
        """_maybe_save_checkpoint must call torch.get_rng_state() and
        pass it to GenerationCheckpoint."""
        assert "rng_state=torch.get_rng_state()" in RASD_INF_SRC, (
            "Fix #5 regression: _maybe_save_checkpoint not populating "
            "rng_state — resumed runs diverge under temperature > 0"
        )

    def test_resume_restores_rng_state(self):
        """The resume branch in generate() must call torch.set_rng_state()."""
        assert "torch.set_rng_state(ckpt.rng_state)" in RASD_INF_SRC, (
            "Fix #5 regression: resume branch not restoring rng_state"
        )

    def test_restore_guarded_against_none(self):
        """Older checkpoints (saved before this fix) had rng_state=None.
        Restore must guard against the None case so old checkpoints
        don't crash a resume."""
        # Look for the guarded form
        assert re.search(
            r"if ckpt\.rng_state is not None:[\s\S]{0,80}torch\.set_rng_state\(ckpt\.rng_state\)",
            RASD_INF_SRC,
        ), (
            "Fix #5 regression: rng_state restore not guarded by None-check; "
            "old pre-fix checkpoints would crash on resume"
        )

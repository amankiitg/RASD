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


# ---------------------------------------------------------------------------
# 2026-05-10 second-pass review fixes
# ---------------------------------------------------------------------------

class TestSecondPassFix1CacheSubclass:
    """The 2nd-pass review found that NF4DynamicCache was duck-typed as
    DynamicCache but did NOT subclass transformers.cache_utils.Cache.
    LlamaModel.forward in transformers 4.44.2 explicitly checks
    `not isinstance(past_key_values, Cache)` and replaces non-Cache
    objects via DynamicCache.from_legacy_cache(...) → our NF4 storage
    would have been silently swapped out for bf16. CRITICAL fix."""

    def test_isinstance_cache(self):
        from transformers.cache_utils import Cache
        from src.models.nf4_dynamic_cache import NF4DynamicCache
        assert isinstance(NF4DynamicCache(), Cache), (
            "2nd-pass #1 regression: NF4DynamicCache no longer subclasses "
            "transformers.cache_utils.Cache. LlamaModel.forward will "
            "replace it before attention sees it; kv_quant=True will "
            "silently store bf16; 1M context will OOM at 40 GB."
        )


class TestSecondPassFix2NoOuterLoop:
    """P3.3 long_ctx_smokes had a `for ctx in 32k 128k 512k 1M; do
    python ... --groups SMOKE; done` loop. Each iteration runs the
    entire SMOKE group (which itself enumerates all 4 contexts via
    its level entries), so 4×4 = 16 cells instead of 4. ~$50-80 of
    pod time wasted."""

    def test_long_ctx_smokes_has_no_outer_for_loop(self):
        """The long_ctx_smokes function body must not contain a
        `for ctx in ...; do ... done` style loop; the SMOKE YAML
        already enumerates all 4 contexts."""
        m = re.search(
            r"^long_ctx_smokes\(\) \{(.*?)^\}",
            PHASE_C_SH, re.DOTALL | re.MULTILINE,
        )
        assert m is not None, "long_ctx_smokes function not found"
        body = m.group(1)
        # Strip comments before searching for the loop
        non_comment = "\n".join(
            line for line in body.splitlines()
            if not line.lstrip().startswith("#")
        )
        assert not re.search(r"\bfor\b\s+ctx\b", non_comment), (
            "2nd-pass #2 regression: long_ctx_smokes has an outer "
            "`for ctx in ...` loop that runs the SMOKE group 4× — "
            "wastes ~$50-80 of pod time"
        )


class TestSecondPassFix3CudaRngState:
    """The original Fix #5 saved `torch.get_rng_state()` (CPU only).
    Sampling runs on CUDA tensors (torch.multinomial / torch.rand_like
    on the verify forward's accept_prob tensor), which consumes the
    CUDA generator state — independent from CPU. Without saving
    cuda_rng_state, resumed runs still diverge."""

    def test_checkpoint_dataclass_has_cuda_rng_field(self):
        from src.models.checkpoint import GenerationCheckpoint
        import dataclasses
        field_names = {f.name for f in dataclasses.fields(GenerationCheckpoint)}
        assert "cuda_rng_state" in field_names, (
            "2nd-pass #3 regression: GenerationCheckpoint missing "
            "cuda_rng_state field; resumed runs still diverge under "
            "GPU sampling"
        )

    def test_save_populates_cuda_rng(self):
        """_maybe_save_checkpoint must set cuda_rng_state from
        torch.cuda.get_rng_state(self._device) when CUDA available."""
        assert "cuda_rng_state=" in RASD_INF_SRC
        assert "torch.cuda.get_rng_state(self._device)" in RASD_INF_SRC, (
            "2nd-pass #3 regression: cuda_rng_state not populated from "
            "torch.cuda.get_rng_state(self._device)"
        )

    def test_resume_restores_cuda_rng(self):
        """Resume branch in generate() must call torch.cuda.set_rng_state."""
        assert "torch.cuda.set_rng_state(ckpt.cuda_rng_state, self._device)" in RASD_INF_SRC, (
            "2nd-pass #3 regression: resume branch not restoring cuda_rng_state"
        )

    def test_cuda_restore_guarded_against_none(self):
        """Old checkpoints had cuda_rng_state=None. Restore must guard."""
        assert re.search(
            r"if\s*\(?\s*ckpt\.cuda_rng_state is not None",
            RASD_INF_SRC,
        ), (
            "2nd-pass #3 regression: cuda_rng_state restore not None-guarded"
        )

    def test_payload_round_trips_cuda_rng(self):
        """save() / load() preserve cuda_rng_state through the on-disk
        format."""
        import torch
        from src.models.checkpoint import GenerationCheckpoint
        ckpt = GenerationCheckpoint(
            n_rounds=1, global_seqlen=10, total_accepted=1,
            total_draft_toks=2, prefill_len=8,
            cur_token=torch.tensor([[42]]),
            generated=[torch.tensor([[1]])],
            past_kv=tuple([(torch.zeros(1, 2, 4, 4), torch.zeros(1, 2, 4, 4))]),
            draft_past_kv=tuple([(torch.zeros(1, 2, 4, 4), torch.zeros(1, 2, 4, 4))]),
            cuda_rng_state=torch.tensor([1, 2, 3, 4], dtype=torch.uint8),
        )
        import tempfile, pathlib
        with tempfile.TemporaryDirectory() as tmp:
            p = pathlib.Path(tmp) / "ckpt.pt"
            ckpt.save(p)
            loaded = GenerationCheckpoint.load(p)
        assert loaded.cuda_rng_state is not None
        assert torch.equal(loaded.cuda_rng_state, ckpt.cuda_rng_state)


class TestThirdPassBlocker3Timeout:
    """3rd-pass review: 3600s hard timeout will SIGTERM 1M cells.
    The smoke YAML's own comment says ctx=1M ~120 min. We need a
    configurable per-run timeout, raised for the long-context stages."""

    def test_no_hardcoded_3600_timeout(self):
        """The bare 3600 magic number must be gone — replaced with a
        parametrized timeout argument."""
        # Find the proc.wait() call; its argument should be a variable
        # name, not the literal 3600
        m = re.search(
            r"proc\.wait\(timeout=(\w+)\)",
            (REPO_ROOT / "run_experiment.py").read_text(),
        )
        assert m is not None
        timeout_arg = m.group(1)
        assert timeout_arg != "3600", (
            "3rd-pass blocker 3 regression: hard-coded 60-min timeout "
            "will SIGTERM 1M runs (~120 min expected)"
        )

    def test_timeout_per_run_s_cli_flag(self):
        rep = (REPO_ROOT / "run_experiment.py").read_text()
        assert "--timeout-per-run-s" in rep, (
            "3rd-pass blocker 3 regression: --timeout-per-run-s flag missing"
        )

    def test_long_ctx_stages_use_4hr_timeout(self):
        """phase_c_pod_session.sh stages that hit ctx >= 128k must pass
        --timeout-per-run-s with a value >= 14400 (4 hr) — anything less
        risks killing 1M cells mid-run."""
        text = PHASE_C_SH
        for stage_func in ("long_ctx_smokes", "final_matrix",
                           "profiler_sidecar_pass"):
            m = re.search(
                rf"^{stage_func}\(\) \{{(.*?)^\}}",
                text, re.DOTALL | re.MULTILINE,
            )
            assert m is not None
            body = m.group(1)
            # Look for --timeout-per-run-s with value >= 14400
            t = re.search(r"--timeout-per-run-s\s+(\d+)", body)
            assert t is not None, (
                f"3rd-pass blocker 3 regression: {stage_func} doesn't "
                "set --timeout-per-run-s; default 3600 will kill 1M cells"
            )
            seconds = int(t.group(1))
            assert seconds >= 14400, (
                f"3rd-pass blocker 3 regression: {stage_func} uses "
                f"timeout {seconds}s; need >= 14400 for 1M (~120 min)"
            )


class TestThirdPassBlocker2CheckpointPlumbing:
    """3rd-pass review: RASDConfig had checkpoint_every / checkpoint_dir
    / run_id but run_experiment.py never plumbed them through. Phase C
    1M runs ran with checkpointing disabled — every crash on a 120-min
    cell would have lost the full run. The mentor M4 risk-mitigation
    plan explicitly named checkpointing as required."""

    def test_rasd_config_call_passes_checkpoint_every(self):
        """The RASDConfig(...) call in _run_single_worker must read
        checkpoint_every from the run dict so YAML overrides take
        effect."""
        rep = (REPO_ROOT / "run_experiment.py").read_text()
        assert re.search(
            r'checkpoint_every\s*=\s*int\(run\.get\([\"\']checkpoint_every[\"\'],\s*0\)\)',
            rep,
        ), (
            "3rd-pass blocker 2 regression: RASDConfig() not reading "
            "checkpoint_every from run dict"
        )

    def test_rasd_config_call_passes_checkpoint_dir(self):
        rep = (REPO_ROOT / "run_experiment.py").read_text()
        assert "checkpoint_dir" in rep
        # Must derive a sensible default from output_csv path
        assert re.search(
            r'checkpoint_dir\s*=\s*\([\s\S]{0,200}Path\(output_csv\)',
            rep,
        ), "3rd-pass blocker 2 regression: checkpoint_dir not derived from output_csv"

    def test_rasd_config_call_passes_run_id(self):
        rep = (REPO_ROOT / "run_experiment.py").read_text()
        assert re.search(
            r'run_id\s*=\s*run\[[\"\']run_id[\"\']\]',
            rep,
        ), "3rd-pass blocker 2 regression: run_id not passed to RASDConfig"

    def test_checkpoint_every_cli_flag_present(self):
        rep = (REPO_ROOT / "run_experiment.py").read_text()
        assert "--checkpoint-every" in rep, (
            "3rd-pass blocker 2 regression: --checkpoint-every CLI flag missing"
        )

    def test_long_ctx_yaml_levels_set_checkpoint_every(self):
        """The 512k and 1M cells in both M4 YAMLs must declare a non-zero
        checkpoint_every. Without this, the headline 1M cells run with
        checkpointing disabled and a crash kills the run."""
        for yaml_path in ("configs/m4_phase_c_long_smoke.yml",
                          "configs/m4_final_matrix.yml"):
            text = (REPO_ROOT / yaml_path).read_text()
            # Find the ctx512k and ctx1M level blocks; each must contain
            # a `checkpoint_every: <int>` line within ~10 lines of the id
            for level_id_substring in ("ctx512k", "ctx1M"):
                m = re.search(
                    rf'id:\s*"\w+_{level_id_substring}"[\s\S]{{0,400}}?'
                    r"checkpoint_every:\s*(\d+)",
                    text,
                )
                assert m is not None, (
                    f"3rd-pass blocker 2 regression: {yaml_path} level "
                    f"{level_id_substring!r} missing checkpoint_every"
                )
                ce = int(m.group(1))
                assert ce >= 1, (
                    f"3rd-pass blocker 2 regression: {yaml_path} "
                    f"{level_id_substring} has checkpoint_every={ce}"
                )


class TestSecondPassFix4BaselinesCoverage:
    """P3.4 baselines stage previously ran only at 128k+1M, no seeds.
    The M4 final matrix grid is 4 contexts × 3 seeds — Phase D Figure
    1 needs matching baseline rows for error bars + 1M comparison.
    Also: the script reported per-shard throughput in distributed
    mode (seq_len / time where seq_len = local_len), under-reporting
    by world_size×."""

    def test_phase_c_baselines_invokes_4_contexts(self):
        m = re.search(
            r"^baseline_validation\(\) \{(.*?)^\}",
            PHASE_C_SH, re.DOTALL | re.MULTILINE,
        )
        body = m.group(1)
        # All 4 contexts of the M4 final matrix must be on the command line
        for length in (131072, 262144, 524288, 1048576):
            assert str(length) in body, (
                f"2nd-pass #4 regression: baseline stage missing ctx={length}"
            )

    def test_phase_c_baselines_invokes_3_seeds(self):
        m = re.search(
            r"^baseline_validation\(\) \{(.*?)^\}",
            PHASE_C_SH, re.DOTALL | re.MULTILINE,
        )
        body = m.group(1)
        for seed in (42, 123, 456):
            assert str(seed) in body, (
                f"2nd-pass #4 regression: baseline stage missing seed={seed}"
            )

    def test_benchmark_baselines_has_seeds_arg(self):
        from pathlib import Path
        src = (Path(__file__).resolve().parent.parent /
               "scripts" / "benchmark_baselines.py").read_text()
        assert "--seeds" in src, (
            "2nd-pass #4 regression: benchmark_baselines.py missing --seeds"
        )

    def test_benchmark_baselines_csv_has_seed_column(self):
        from pathlib import Path
        src = (Path(__file__).resolve().parent.parent /
               "scripts" / "benchmark_baselines.py").read_text()
        # CSV_HEADER should include "seed"
        assert re.search(r'CSV_HEADER\s*=\s*\[[^\]]*"seed"', src), (
            "2nd-pass #4 regression: CSV_HEADER missing 'seed' column"
        )

    def test_benchmark_module_takes_total_len_for_throughput(self):
        """Throughput must be reported at total_len, not local shard.
        Otherwise distributed runs underreport tps by world_size×."""
        from pathlib import Path
        src = (Path(__file__).resolve().parent.parent /
               "scripts" / "benchmark_baselines.py").read_text()
        # The fix routes total_len through to the function and uses it
        # to compute throughput
        assert "total_len: Optional[int]" in src or "total_len:" in src
        assert "measured_len" in src, (
            "2nd-pass #4 regression: benchmark_module not using total_len "
            "for throughput; would underreport in distributed mode"
        )

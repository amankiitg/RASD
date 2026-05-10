"""Tests for the C5 wiring in run_experiment.py — TTFT CSV column +
per-position trace sidecar.

Most of run_experiment.py needs an actual GPU + subprocess machinery
to exercise. C5 itself is mostly plumbing (CSV columns, sidecar path,
cfg propagation), so we test:

1. The pure helpers `_per_token_sidecar_path` and
   `write_per_token_sidecar` directly with synthetic input.
2. CSV_FIELDS includes the new ttft_ms column.
3. Source-inspection guards on the integration points so a future
   refactor that drops the wiring fails loudly.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
import run_experiment  # noqa: E402

RUN_EXP_SRC = (REPO_ROOT / "run_experiment.py").read_text()


# ---------------------------------------------------------------------------
# CSV schema
# ---------------------------------------------------------------------------

class TestCsvFields:
    def test_ttft_ms_in_fields(self):
        assert "ttft_ms" in run_experiment.CSV_FIELDS, (
            "C5 regression: ttft_ms missing from CSV_FIELDS"
        )

    def test_ttft_ms_after_mean_latency(self):
        """Display order matters for human-readable CSVs — keep ttft_ms
        next to mean_latency_ms."""
        fields = run_experiment.CSV_FIELDS
        assert fields.index("ttft_ms") == fields.index("mean_latency_ms") + 1

    def test_existing_columns_preserved(self):
        """C5 must be additive — must not drop any pre-existing column."""
        for needed in ("run_id", "group", "level_id", "seed",
                       "target_model_name", "draft_model_name",
                       "spec_steps", "kv_block_size", "prefetch_depth",
                       "context_length", "dtype",
                       "tokens_generated", "time_sec", "throughput_tps",
                       "acceptance_rate", "mean_latency_ms",
                       "gpu_peak_mem_mb", "n_rounds", "status", "error"):
            assert needed in run_experiment.CSV_FIELDS, (
                f"C5 regression: lost CSV column {needed!r}"
            )


# ---------------------------------------------------------------------------
# Sidecar path helper
# ---------------------------------------------------------------------------

class TestSidecarPath:
    def test_sits_under_per_token_subdir(self, tmp_path):
        csv = tmp_path / "results.csv"
        path = run_experiment._per_token_sidecar_path(csv, "A2_k4_s42")
        assert path.parent.name == "per_token"
        assert path.parent.parent == tmp_path
        assert path.name == "A2_k4_s42.jsonl"

    def test_path_consistent_across_runs(self, tmp_path):
        csv = tmp_path / "out.csv"
        a = run_experiment._per_token_sidecar_path(csv, "A1_sheared_1b_s42")
        b = run_experiment._per_token_sidecar_path(csv, "A1_sheared_1b_s42")
        assert a == b

    def test_distinct_run_ids_distinct_paths(self, tmp_path):
        csv = tmp_path / "out.csv"
        a = run_experiment._per_token_sidecar_path(csv, "X_s42")
        b = run_experiment._per_token_sidecar_path(csv, "Y_s42")
        assert a != b


# ---------------------------------------------------------------------------
# Sidecar writer
# ---------------------------------------------------------------------------

class TestWriteSidecar:
    def test_skips_when_trace_is_none(self, tmp_path):
        csv = tmp_path / "out.csv"
        result = run_experiment.write_per_token_sidecar(None, csv, "X_s42")
        assert result is None
        assert not (tmp_path / "per_token").exists()

    def test_skips_when_trace_is_empty(self, tmp_path):
        csv = tmp_path / "out.csv"
        result = run_experiment.write_per_token_sidecar([], csv, "X_s42")
        assert result is None
        assert not (tmp_path / "per_token").exists()

    def test_writes_one_line_per_record(self, tmp_path):
        csv = tmp_path / "out.csv"
        trace = [
            {"round_idx": 0, "global_pos_start": 100, "spec_steps": 4,
             "n_acc": 2, "draft_tokens": [1, 2, 3, 4],
             "accepted": [True, True, False, False]},
            {"round_idx": 1, "global_pos_start": 103, "spec_steps": 4,
             "n_acc": 4, "draft_tokens": [5, 6, 7, 8],
             "accepted": [True, True, True, True]},
        ]
        path = run_experiment.write_per_token_sidecar(trace, csv, "X_s42")
        assert path is not None
        assert path.exists()
        lines = path.read_text().strip().split("\n")
        assert len(lines) == 2
        assert json.loads(lines[0]) == trace[0]
        assert json.loads(lines[1]) == trace[1]

    def test_creates_parent_dir(self, tmp_path):
        """`per_token/` subdir must be created on demand — pod-side
        first-run failure if missing."""
        csv = tmp_path / "out.csv"
        assert not (tmp_path / "per_token").exists()
        trace = [{"round_idx": 0, "global_pos_start": 0,
                  "spec_steps": 1, "n_acc": 1,
                  "draft_tokens": [42], "accepted": [True]}]
        path = run_experiment.write_per_token_sidecar(trace, csv, "X_s42")
        assert (tmp_path / "per_token").exists()
        assert path.parent.name == "per_token"


# ---------------------------------------------------------------------------
# Integration points (source inspection)
# ---------------------------------------------------------------------------

class TestIntegrationPoints:
    def test_log_per_token_passed_to_rasd_config(self):
        """RASDConfig must receive log_per_token from the run dict so
        the sidecar can be enabled per-row from the launcher."""
        assert re.search(
            r"log_per_token\s*=\s*bool\(run\.get\([\"']log_per_token[\"'],\s*False\)\)",
            RUN_EXP_SRC,
        ), (
            "C5 regression: log_per_token not propagated to RASDConfig "
            "in _run_single_worker"
        )

    def test_per_token_trace_popped_before_wandb(self):
        """The trace is a list-of-dicts and would break wandb.log; it
        must be popped from metrics before log_wandb()."""
        # Look for the .pop("per_token_trace", ...) call
        assert 'metrics.pop("per_token_trace"' in RUN_EXP_SRC, (
            "C5 regression: per_token_trace not popped from metrics — "
            "wandb.log() will fail on the list payload"
        )

    def test_ttft_ms_added_to_row(self):
        """Row update must include rounded ttft_ms so it lands in the CSV."""
        assert re.search(
            r'"ttft_ms"\s*:\s*round\(metrics\["ttft_ms"\],\s*\d+\)',
            RUN_EXP_SRC,
        ), "C5 regression: ttft_ms not added to row.update() in _run_single_worker"

    def test_log_per_token_cli_flag_present(self):
        assert "--log-per-token" in RUN_EXP_SRC, (
            "C5 regression: --log-per-token CLI flag missing from main()"
        )

    def test_log_per_token_default_off_when_not_passed(self):
        """Default-off invariant: M3 replay must be byte-identical when
        --log-per-token isn't passed. Verify the propagation is gated
        by `if args.log_per_token:`."""
        assert re.search(
            r"if args\.log_per_token:[\s\S]{0,200}r\[[\"']log_per_token[\"']\]\s*=\s*True",
            RUN_EXP_SRC,
        ), (
            "C5 regression: --log-per-token propagation doesn't gate on "
            "the flag — would unconditionally enable trace logging"
        )

    def test_sidecar_written_only_on_rank_zero(self):
        """Only rank 0 should write the sidecar file; otherwise we'd
        get 8x duplicate writes per run."""
        assert re.search(
            r"if local_rank == 0:[\s\S]{0,200}write_per_token_sidecar\(",
            RUN_EXP_SRC,
        ), (
            "C5 regression: write_per_token_sidecar not gated by "
            "`if local_rank == 0:` — would duplicate writes 8x"
        )


class TestProfileFlag:
    """C7 profiler --profile CLI flag wiring."""

    def test_profile_cli_flag_present(self):
        assert "--profile" in RUN_EXP_SRC, (
            "--profile flag missing from run_experiment.py"
        )

    def test_profile_propagates_to_run_dict(self):
        """`if args.profile: r['profile'] = True` must be present so
        the subprocess worker picks it up via the JSON-serialized run."""
        assert re.search(
            r"if args\.profile:[\s\S]{0,200}r\[[\"\']profile[\"\']\]\s*=\s*True",
            RUN_EXP_SRC,
        ), (
            "C7 regression: --profile not propagated to run dict in main()"
        )

    def test_profiler_wraps_generate(self):
        """RoundProfiler must wrap the engine.generate_text call when on."""
        assert "from src.analysis.profiler import RoundProfiler" in RUN_EXP_SRC, (
            "C7 regression: RoundProfiler not imported in worker"
        )
        assert re.search(
            r"with RoundProfiler\(enabled=True\)[\s\S]{0,200}engine\.generate_text",
            RUN_EXP_SRC,
        ), (
            "C7 regression: RoundProfiler not wrapping generate_text call"
        )

    def test_profile_disabled_path_uses_existing_call(self):
        """When --profile is off, the existing un-wrapped generate_text
        path must still be hit — no overhead, no behavior change."""
        # Look for the else branch with the bare generate_text call
        assert re.search(
            r"else:\s*\n\s+_, metrics = engine\.generate_text\(prompt\)",
            RUN_EXP_SRC,
        ), (
            "C7 regression: profile-disabled path is missing or wrapped"
        )

    def test_profiler_summary_popped_before_wandb(self):
        """The summary dict shouldn't go to wandb.log as raw payload."""
        assert 'metrics.pop("_profiler_summary"' in RUN_EXP_SRC, (
            "C7 regression: _profiler_summary not popped before wandb"
        )

    def test_profiler_sidecar_written_only_on_rank_zero(self):
        """Profiler sidecar gated by `if local_rank == 0:` like per-token trace."""
        assert re.search(
            r"if local_rank == 0:[\s\S]{0,400}write_profiler_sidecar\(",
            RUN_EXP_SRC,
        ), (
            "C7 regression: write_profiler_sidecar not gated by rank-0"
        )

    def test_profiler_sidecar_path_under_profiler_subdir(self):
        from run_experiment import _profiler_sidecar_path
        from pathlib import Path
        p = _profiler_sidecar_path(Path("/tmp/results.csv"), "M4_ctx128k_s42")
        assert p.name == "M4_ctx128k_s42.json"
        assert p.parent.name == "profiler"

    def test_canary_inherits_profile_flag(self):
        """2026-05-10 regression: canary_run is built independently of
        all_runs, so the loop that sets r['profile']=True from args.profile
        doesn't reach it. The canary then runs without profiling, producing
        no profile JSON — and we can't catch JSON-write bugs before the
        full matrix starts. The fix: explicitly propagate args.profile
        onto canary_run too."""
        assert re.search(
            r"if args\.profile:[\s\S]{0,300}canary_run\[[\"\']profile[\"\']\]\s*=\s*True",
            RUN_EXP_SRC,
        ), (
            "Canary regression: --profile not propagated to canary_run; "
            "canary will silently skip profiling and miss bugs the matrix hits."
        )

    def test_canary_inherits_log_per_token_flag(self):
        """Same shape as profile: canary must mirror --log-per-token too."""
        assert re.search(
            r"if args\.log_per_token:[\s\S]{0,300}canary_run\[[\"\']log_per_token[\"\']\]\s*=\s*True",
            RUN_EXP_SRC,
        ), (
            "Canary regression: --log-per-token not propagated to canary_run."
        )

    def test_canary_inherits_memory_trace_flag(self):
        """Same shape as profile: canary must mirror --memory-trace too."""
        assert re.search(
            r"if args\.memory_trace:[\s\S]{0,400}canary_run\[[\"\']memory_trace[\"\']\]\s*=\s*True",
            RUN_EXP_SRC,
        ), (
            "Canary regression: --memory-trace not propagated to canary_run."
        )

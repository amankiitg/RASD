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
        path must still be hit — no overhead, no behavior change.
        Pattern updated 2026-05-11 (p38): worker now captures
        generated_text instead of discarding with `_` so RULER niah can
        score model output."""
        # Look for the else branch with a bare generate_text call (no
        # RoundProfiler wrapping). The variable name is now
        # generated_text (was `_`).
        assert re.search(
            r"else:\s*\n\s+\w+,\s*metrics\s*=\s*engine\.generate_text\(prompt\)",
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


# ---------------------------------------------------------------------------
# p35d prompt-source flag (PG-19 vs synthetic)
# ---------------------------------------------------------------------------

class TestPromptSource:
    """p35d 2026-05-11: --prompt-source pg19 lets RASD use real narrative
    text instead of the repeated synthetic paragraph. Acceptance ablation
    that addresses the 1M low-acceptance finding from p35b."""

    def test_build_prompt_default_synthetic(self):
        """Default source stays 'synthetic' so M3 + earlier p35/p35b
        runs are byte-identical."""
        from run_experiment import build_prompt
        import inspect
        sig = inspect.signature(build_prompt)
        assert sig.parameters["source"].default == "synthetic"

    def test_build_prompt_synthetic_unchanged(self):
        """Synthetic path produces the same repeated-paragraph prompt
        regardless of seed/source kwargs."""
        from run_experiment import build_prompt

        class _StubTok:
            def encode(self, s):
                return list(range(len(s.split())))
            def decode(self, ids):
                return " ".join(f"tok{i}" for i in ids)

        tok = _StubTok()
        p1 = build_prompt(64, tok)
        p2 = build_prompt(64, tok, source="synthetic", seed=42)
        p3 = build_prompt(64, tok, source="synthetic", seed=999)
        assert p1 == p2 == p3, "synthetic prompt must be seed-independent"

    def test_build_prompt_pg19_requires_meta(self):
        """source='pg19' without a meta_path must raise — silently
        falling back to synthetic would corrupt the acceptance ablation."""
        from run_experiment import build_prompt
        try:
            build_prompt(64, tokenizer=None, source="pg19", pg19_meta=None)
        except ValueError as e:
            assert "pg19" in str(e).lower() or "meta" in str(e).lower()
            return
        raise AssertionError("build_prompt(source='pg19', meta=None) should ValueError")

    def test_build_prompt_pg19_loads_chunk(self, tmp_path):
        """source='pg19' reads from a preprocess_pg19.py metadata.json
        and slices to the requested context_length."""
        import json, numpy as np
        from run_experiment import build_prompt

        # Build a synthetic PG-19 chunk on disk
        chunk_path = tmp_path / "pg19_test_chunk_0.dat"
        ids = np.arange(2048, dtype=np.int32)
        mm = np.memmap(chunk_path, dtype="int32", mode="w+", shape=(2048,))
        mm[:] = ids; mm.flush()
        meta_path = tmp_path / "pg19_test_metadata.json"
        meta_path.write_text(json.dumps({
            "chunks": [{"file": str(chunk_path), "length": 2048}]
        }))

        class _StubTok:
            def decode(self, ids):
                # Return a deterministic string from the slice so we can
                # assert build_prompt produced *something* from these ids.
                return f"PG19[{ids[0]}..{ids[-1]}]"

        out = build_prompt(64, _StubTok(), source="pg19",
                           pg19_meta=str(meta_path), seed=42)
        assert out.startswith("PG19[")
        # Must have produced 64 tokens from the chunk
        assert ".." in out

    def test_build_prompt_pg19_seed_reproducible(self, tmp_path):
        """Same seed → same PG-19 chunk slice. Required so a run_id can
        be reproduced byte-identical across re-runs."""
        import json, numpy as np
        from run_experiment import build_prompt

        chunk_path = tmp_path / "pg19_test_chunk_0.dat"
        mm = np.memmap(chunk_path, dtype="int32", mode="w+", shape=(2048,))
        mm[:] = np.arange(2048, dtype=np.int32); mm.flush()
        meta_path = tmp_path / "pg19_test_metadata.json"
        meta_path.write_text(json.dumps({
            "chunks": [{"file": str(chunk_path), "length": 2048}]
        }))

        class _StubTok:
            def decode(self, ids):
                return ",".join(str(i) for i in ids)

        a = build_prompt(64, _StubTok(), source="pg19",
                         pg19_meta=str(meta_path), seed=42)
        b = build_prompt(64, _StubTok(), source="pg19",
                         pg19_meta=str(meta_path), seed=42)
        assert a == b, "PG-19 chunk pick must be seed-deterministic"
        # Different seed → different slice (high probability with 2048 - 64 + 1
        # = 1985 possible start positions)
        c = build_prompt(64, _StubTok(), source="pg19",
                         pg19_meta=str(meta_path), seed=999)
        assert a != c, "different seed should pick a different slice"

    def test_cli_prompt_source_flag_present(self):
        """--prompt-source flag must exist and accept {synthetic,pg19}."""
        assert "--prompt-source" in RUN_EXP_SRC
        assert re.search(
            r"--prompt-source[\s\S]{0,200}choices=\[[\"\']synthetic[\"\'],\s*[\"\']pg19[\"\']",
            RUN_EXP_SRC,
        ), "--prompt-source must constrain choices to synthetic/pg19"

    def test_cli_prompt_source_propagates_to_run_dict(self):
        """Like --profile/--log-per-token, the flag must land on each
        run dict so the worker subprocess sees it through the --_worker
        JSON payload."""
        assert re.search(
            r"if args\.prompt_source\s*!=\s*[\"\']synthetic[\"\'][\s\S]{0,400}"
            r"r\[[\"\']prompt_source[\"\']\]\s*=\s*args\.prompt_source",
            RUN_EXP_SRC,
        ), (
            "p35d regression: --prompt-source not propagated to all_runs; "
            "worker subprocesses will fall back to synthetic prompts."
        )

    def test_cli_prompt_source_propagates_to_canary(self):
        """Same canary-inheritance pattern as profile/log_per_token/memory_trace."""
        assert re.search(
            r"if args\.prompt_source\s*!=\s*[\"\']synthetic[\"\'][\s\S]{0,400}"
            r"canary_run\[[\"\']prompt_source[\"\']\]\s*=\s*args\.prompt_source",
            RUN_EXP_SRC,
        ), (
            "p35d canary regression: --prompt-source not propagated to canary_run; "
            "canary will run on synthetic prompts and mismatch the matrix."
        )

    def test_cli_pg19_requires_meta_flag(self):
        """--prompt-source=pg19 without --prompt-pg19-meta must SystemExit
        at CLI-parse time, not silently fall back to synthetic."""
        assert re.search(
            r"prompt_source\s*==\s*[\"\']pg19[\"\'][\s\S]{0,200}"
            r"args\.prompt_pg19_meta",
            RUN_EXP_SRC,
        ), (
            "p35d safety: --prompt-source=pg19 without --prompt-pg19-meta "
            "must error at CLI parse, not corrupt the run."
        )

    def test_worker_reads_prompt_source_from_run_dict(self):
        """The worker (_run_single_worker) must read prompt_source from
        the run dict and pass it to build_prompt — the missing link
        between CLI flag and the actual prompt content."""
        assert re.search(
            r"build_prompt\([\s\S]{0,400}source=run\.get\([\"\']prompt_source[\"\']",
            RUN_EXP_SRC,
        ), (
            "p35d wiring gap: worker doesn't pass run['prompt_source'] "
            "into build_prompt(); CLI flag has no effect on actual run."
        )


# ---------------------------------------------------------------------------
# p38 RULER niah prompt source + sidecar + scorer
# ---------------------------------------------------------------------------

class TestRulerNiahPrompt:
    """p38 2026-05-11: --prompt-source ruler_niah builds a needle-in-
    haystack prompt with a seed-derived magic number, writes needle
    metadata to a sidecar, and the scorer marks accuracy by string
    match in the generated text."""

    def _stub_tokenizer(self):
        """Minimal tokenizer: encode by char count (cheap, reproducible)."""
        class _T:
            def encode(self, s, add_special_tokens=True):
                return list(range(len(s)))  # 1 token per char
            def decode(self, ids):
                # Encode the id list back to a marker we can inspect
                return f"PROMPT[{ids[0]}..{ids[-1]}|len={len(ids)}]"
        return _T()

    def test_ruler_choice_in_cli_flag(self):
        assert re.search(
            r"--prompt-source[\s\S]{0,400}choices=\[[\"\']synthetic[\"\'],"
            r"\s*[\"\']pg19[\"\'],\s*[\"\']ruler_niah[\"\']",
            RUN_EXP_SRC,
        ), "ruler_niah must be in --prompt-source choices"

    def test_ruler_sidecar_dir_cli_flag_present(self):
        assert "--ruler-sidecar-dir" in RUN_EXP_SRC, (
            "--ruler-sidecar-dir flag must exist so the post-hoc scorer "
            "can find the needle metadata."
        )

    def test_build_prompt_ruler_writes_sidecar(self, tmp_path):
        from run_experiment import build_prompt
        out = build_prompt(
            128, self._stub_tokenizer(), source="ruler_niah",
            seed=42, ruler_sidecar_dir=str(tmp_path), run_id="TEST_run",
        )
        sidecar = tmp_path / "TEST_run.ruler_niah.json"
        assert sidecar.exists(), "ruler_niah must write a sidecar JSON"
        import json
        meta = json.loads(sidecar.read_text())
        for k in ("run_id", "seed", "context_length", "magic_number",
                  "needle_position_frac", "needle_text"):
            assert k in meta, f"sidecar missing key: {k}"
        assert meta["run_id"] == "TEST_run"
        assert meta["seed"] == 42
        assert 10_000 <= meta["magic_number"] <= 99_999
        assert 0.05 <= meta["needle_position_frac"] <= 0.95

    def test_build_prompt_ruler_seed_reproducible(self, tmp_path):
        """Same seed -> same magic number + same position. Required so
        the scorer can verify by seed alone if needed."""
        from run_experiment import build_prompt
        import json
        d1 = tmp_path / "a"
        d2 = tmp_path / "b"
        build_prompt(256, self._stub_tokenizer(), source="ruler_niah",
                     seed=42, ruler_sidecar_dir=str(d1), run_id="R1")
        build_prompt(256, self._stub_tokenizer(), source="ruler_niah",
                     seed=42, ruler_sidecar_dir=str(d2), run_id="R2")
        m1 = json.loads((d1 / "R1.ruler_niah.json").read_text())
        m2 = json.loads((d2 / "R2.ruler_niah.json").read_text())
        assert m1["magic_number"] == m2["magic_number"]
        assert m1["needle_position_frac"] == m2["needle_position_frac"]

    def test_build_prompt_ruler_distinct_seeds(self, tmp_path):
        """Different seeds -> different magic numbers (very high probability
        across 90000 candidates)."""
        from run_experiment import build_prompt
        import json
        d = tmp_path
        build_prompt(256, self._stub_tokenizer(), source="ruler_niah",
                     seed=42, ruler_sidecar_dir=str(d), run_id="A")
        build_prompt(256, self._stub_tokenizer(), source="ruler_niah",
                     seed=999, ruler_sidecar_dir=str(d), run_id="B")
        m1 = json.loads((d / "A.ruler_niah.json").read_text())
        m2 = json.loads((d / "B.ruler_niah.json").read_text())
        assert m1["magic_number"] != m2["magic_number"]

    def test_worker_captures_generated_text_for_ruler(self):
        """When prompt_source=ruler_niah, worker must capture
        generated_text from generate_text() and write to
        <ruler_sidecar_dir>/<run_id>.generated.txt — NOT discard with `_`."""
        assert re.search(
            r"generated_text,\s*metrics\s*=\s*engine\.generate_text",
            RUN_EXP_SRC,
        ), "Worker must capture generated_text from generate_text(), not discard"
        assert re.search(
            r"run\.get\([\"\']prompt_source[\"\']\)\s*==\s*[\"\']ruler_niah[\"\'][\s\S]{0,400}"
            r"\.generated\.txt",
            RUN_EXP_SRC,
        ), "Worker must write generated.txt sidecar for ruler_niah runs"


class TestRulerScorer:
    """The post-hoc scorer reads paired sidecars and emits accuracy."""

    def test_score_one_found(self, tmp_path):
        import json, sys
        sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))
        # Use importlib because filename starts with a digit-free word but is a script
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "score_ruler_niah",
            Path(__file__).parent.parent / "scripts" / "score_ruler_niah.py"
        )
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)

        # Build paired sidecars: needle metadata + generated text
        needle_meta = {
            "run_id": "TEST_r1",
            "seed": 42,
            "context_length": 4096,
            "magic_number": 12345,
            "needle_position_frac": 0.5,
            "needle_position_tokens": 2048,
            "needle_text": "The magic number is 12345.",
            "question_text": "What is the magic number?",
        }
        (tmp_path / "TEST_r1.ruler_niah.json").write_text(json.dumps(needle_meta))
        (tmp_path / "TEST_r1.generated.txt").write_text(
            "The magic number is 12345 according to the prompt."
        )

        result = mod.score_one(tmp_path / "TEST_r1.ruler_niah.json")
        assert result["found"] == 1
        assert result["magic_number"] == "12345"

    def test_score_one_not_found(self, tmp_path):
        import json, importlib.util
        spec = importlib.util.spec_from_file_location(
            "score_ruler_niah",
            Path(__file__).parent.parent / "scripts" / "score_ruler_niah.py"
        )
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)

        needle_meta = {
            "run_id": "TEST_r2",
            "seed": 42,
            "context_length": 4096,
            "magic_number": 12345,
            "needle_position_frac": 0.5,
            "needle_position_tokens": 2048,
            "needle_text": "The magic number is 12345.",
            "question_text": "What is the magic number?",
        }
        (tmp_path / "TEST_r2.ruler_niah.json").write_text(json.dumps(needle_meta))
        (tmp_path / "TEST_r2.generated.txt").write_text(
            "The model said something unrelated."
        )
        result = mod.score_one(tmp_path / "TEST_r2.ruler_niah.json")
        assert result["found"] == 0

    def test_score_one_substring_safety(self, tmp_path):
        """'42' must NOT match '420' or '142' — needs whole-number boundary."""
        import json, importlib.util
        spec = importlib.util.spec_from_file_location(
            "score_ruler_niah",
            Path(__file__).parent.parent / "scripts" / "score_ruler_niah.py"
        )
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)

        needle_meta = {
            "run_id": "TEST_r3",
            "seed": 42,
            "context_length": 4096,
            "magic_number": 42,
            "needle_position_frac": 0.5,
            "needle_position_tokens": 2048,
            "needle_text": "The magic number is 42.",
            "question_text": "What is the magic number?",
        }
        (tmp_path / "TEST_r3.ruler_niah.json").write_text(json.dumps(needle_meta))
        (tmp_path / "TEST_r3.generated.txt").write_text(
            "The number is 420 or maybe 142."  # contains '42' as substring
        )
        result = mod.score_one(tmp_path / "TEST_r3.ruler_niah.json")
        assert result["found"] == 0, "'42' must not match inside '420' or '142'"

    def test_score_one_missing_generated(self, tmp_path):
        """When generated.txt is missing (run crashed), score = -1 not 0."""
        import json, importlib.util
        spec = importlib.util.spec_from_file_location(
            "score_ruler_niah",
            Path(__file__).parent.parent / "scripts" / "score_ruler_niah.py"
        )
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        needle_meta = {
            "run_id": "TEST_r4",
            "seed": 42, "context_length": 4096, "magic_number": 99999,
            "needle_position_frac": 0.5, "needle_position_tokens": 2048,
            "needle_text": "n", "question_text": "q",
        }
        (tmp_path / "TEST_r4.ruler_niah.json").write_text(json.dumps(needle_meta))
        result = mod.score_one(tmp_path / "TEST_r4.ruler_niah.json")
        assert result["found"] == -1

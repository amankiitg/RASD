"""Tests for the vLLM attempt ladder and its traceback extraction.

The first two Phase-5 attempts recorded only vLLM's one-line summary
("Engine core initialization failed. See root cause above.") as the `error`
field, which is undiagnosable. These tests pin the behaviour that fixes it:
the LAST Python exception is mined out of the tee'd worker log, including the
awkward real-world case where vLLM's useless one-liner is the FINAL line.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent


def _load_module():
    """Import scripts/mlsys_vllm_baseline.py (scripts/ is not a package)."""
    spec = importlib.util.spec_from_file_location(
        "mlsys_vllm_baseline", REPO / "scripts" / "mlsys_vllm_baseline.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["mlsys_vllm_baseline"] = mod
    spec.loader.exec_module(mod)
    return mod


vllm_mod = _load_module()


class TestExtractLastException:
    def test_real_cause_wins_over_vllm_one_liner(self):
        """vLLM prints the traceback, THEN 'Engine core initialization
        failed' as the last line. The traceback is the useful part."""
        log = (
            "INFO loading\n"
            "Traceback (most recent call last):\n"
            '  File "/vllm/engine/arg_utils.py", line 900, in create_engine\n'
            "    engine = EngineCore(...)\n"
            "ValueError: User-specified max_model_len (131136) is greater "
            "than the derived max_model_len\n"
            "ERROR Engine core initialization failed. See root cause above.\n"
        )
        cls, err = vllm_mod.extract_last_exception(log)
        assert cls == "ValueError"
        assert "Traceback (most recent call last)" in err
        assert "131136" in err
        assert err.splitlines()[-1].startswith("ValueError:")

    def test_takes_the_last_exception_not_the_first(self):
        log = (
            "Traceback (most recent call last):\n"
            '  File "a.py", line 1\n'
            "ValueError: first problem\n"
            "INFO retrying\n"
            "Traceback (most recent call last):\n"
            '  File "b.py", line 2\n'
            "RuntimeError: second problem\n"
        )
        cls, err = vllm_mod.extract_last_exception(log)
        assert cls == "RuntimeError"
        assert "second problem" in err
        assert "first problem" not in err

    def test_no_python_exception_falls_back_to_tail(self):
        """A CUDA abort / OOM kill has no Python exception; the row must
        still carry evidence rather than an empty error."""
        log = "INFO starting\nCUDA error: an illegal memory access\nKilled\n"
        cls, err = vllm_mod.extract_last_exception(log)
        assert cls == ""
        assert "illegal memory access" in err
        assert err.strip() != ""

    def test_empty_log_is_empty_not_an_exception(self):
        cls, err = vllm_mod.extract_last_exception("")
        assert cls == ""
        assert err == ""

    @pytest.mark.parametrize("line,expected", [
        ("ValueError: bad", "ValueError"),
        ("RuntimeError: bad", "RuntimeError"),
        ("torch.cuda.OutOfMemoryError: bad", "torch.cuda.OutOfMemoryError"),
        ("ModuleNotFoundError: No module named 'vllm'", "ModuleNotFoundError"),
        ("some random output", ""),
    ])
    def test_error_class_detection(self, line, expected):
        cls, _ = vllm_mod.extract_last_exception(line + "\n")
        assert cls == expected

    def test_error_is_truncated_for_csv_safety(self):
        log = "Traceback (most recent call last):\n" + ("x" * 50000) + \
              "\nValueError: huge\n"
        _, err = vllm_mod.extract_last_exception(log)
        assert len(err) <= 4000


class TestAttemptLadder:
    def test_three_distinct_attempts(self):
        """Each rung must be a DISTINCT fix, otherwise the ladder is just
        the same failure three times."""
        ladder = vllm_mod.ATTEMPT_LADDER
        assert len(ladder) == 3
        names = [a["name"] for a in ladder]
        assert len(set(names)) == 3

    def test_attempt1_uses_exactly_the_context(self):
        assert vllm_mod.ATTEMPT_LADDER[0]["max_model_len"] == "ctx"

    def test_attempt2_escalates_memory_and_serialises(self):
        a2 = vllm_mod.ATTEMPT_LADDER[1]
        assert a2["kwargs"]["gpu_memory_utilization"] == 0.95
        assert a2["kwargs"]["enforce_eager"] is True
        assert a2["kwargs"]["max_num_seqs"] == 1
        assert a2["env"].get("VLLM_ALLOW_LONG_MAX_MODEL_LEN") == "1"

    def test_attempt3_is_a_valid_64k_fallback(self):
        """The last resort must be a legitimate reference point, not a
        silently different experiment."""
        a3 = vllm_mod.ATTEMPT_LADDER[2]
        assert a3["max_model_len"] == 65536
        assert a3["max_model_len"] != "ctx"

    def test_attempt_timeout_is_bounded(self):
        assert 0 < vllm_mod.MAX_ATTEMPT_WALL_S <= 20 * 60


class TestUnitMatching:
    def test_end_to_end_field_exists_and_is_the_comparable_one(self):
        """RASD's throughput_tps includes prefill, so the comparable vLLM
        number is throughput_tps_end_to_end — not the decode-only figure."""
        assert "throughput_tps_end_to_end" in vllm_mod.CSV_FIELDS
        assert "throughput_tps_decode_only" in vllm_mod.CSV_FIELDS
        assert "unit_matched" in vllm_mod.CSV_FIELDS

    def test_failure_provenance_columns_exist(self):
        for col in ("attempt", "config_used", "error_class", "log_path"):
            assert col in vllm_mod.CSV_FIELDS

    def test_llama2_rope_matches_the_rasd_arm1_treatment(self):
        rope = vllm_mod.DEFAULT_ROPE["meta-llama/Llama-2-7b-hf"]
        assert rope["type"] == "yarn"
        assert rope["factor"] == 32.0
        assert rope["original_max_position_embeddings"] == 4096

    def test_llama31_is_left_native(self):
        """At 128k Llama-3.1 is inside its own window; overriding its rope
        would make the cell incomparable to RASD arms 2/3."""
        assert vllm_mod.DEFAULT_ROPE["meta-llama/Llama-3.1-8B"] is None

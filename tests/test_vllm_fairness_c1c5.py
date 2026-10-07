"""C1-C5 vLLM fairness rules + B4 PG-19 target-only config.

The point of the vLLM baseline is an apples-to-apples reference for the RASD
speedup. A baseline that ran is not the same as a baseline that is
*comparable*, so these tests pin the four conditions that make a row
eligible, and the config that gives the 4k/8k PG-19 points a denominator at
their missing seeds.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest
import yaml

REPO = Path(__file__).resolve().parent.parent
_WRAPPER = REPO / "scripts" / "mlsys_vllm_baseline.py"


def _load_wrapper():
    spec = importlib.util.spec_from_file_location("mlsys_vllm_baseline", _WRAPPER)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def wrapper():
    return _load_wrapper()


REV = "d04e592bb4f6aa9cfee91e2e20afa771667e1d4b"


class _Args:
    matched_max_new_tokens = 128
    target_revision = REV
    draft_revision = REV

    def __init__(self, mn: int = 128) -> None:
        self.matched_max_new_tokens = mn


def _ok_row(w, **overrides):
    """A row that satisfies EVERY fairness condition, C1-C5.

    `max_new_tokens` must equal the matched RASD cell's, which the caller sets
    on _Args; the hash must be the one _ids_sha produces for the ids the test
    passes in, and the row must name its document so it can be paired. A
    fixture missing any of these is not an "ok" row any more -- the contract
    was widened to include the sampling rule, the pin and the prompt's
    provenance, so the fixture follows.
    """
    row = {
        "eos_policy": w.EOS_POLICY,
        "tensor_parallel_size": 8,
        "max_new_tokens": 128,
        "temperature": 0.0,                 # the RASD cells are greedy
        "vllm_version": w.VLLM_PIN,         # the pinned release
        "doc_id": "pg19_train_0",           # so the pair can be formed
        # vLLM must be shown to have consumed the ids it was given, and the
        # revisions pin WHICH model the comparison is against.
        "prompt_ids_verified": "yes",
        # The ids must be the ENGINE's input ids (BOS included), not a rebuilt
        # prompt: that is now one of the fairness conditions.
        "prompt_ids_from_engine": "yes",
        "target_revision": REV,
        "draft_revision": REV,
    }
    row.update(overrides)
    return row


class TestC1VersionPin:
    def test_pin_is_an_explicit_release(self, wrapper):
        # A ratio against "whatever pip installed" cannot be reproduced or
        # cited, so the pin must be a concrete version string.
        assert isinstance(wrapper.VLLM_PIN, str)
        assert wrapper.VLLM_PIN.count(".") >= 1
        assert "latest" not in wrapper.VLLM_PIN.lower()

    def test_version_is_recorded_in_csv_fields(self, wrapper):
        # Recording the ACTUAL version matters more than the pin: if the
        # install resolved differently, the row must show it.
        assert "vllm_version" in wrapper.CSV_FIELDS


class TestC2PromptIds:
    def test_missing_prompt_ids_disqualifies_the_row(self, wrapper):
        ok, why = wrapper._unit_match_verdict(_ok_row(wrapper), None, _Args())
        assert ok is False
        assert "C2" in why

    def test_exact_prompt_ids_qualify(self, wrapper):
        ids = [1, 2, 3]
        row = _ok_row(wrapper, prompt_sha256=wrapper._ids_sha(ids))
        ok, why = wrapper._unit_match_verdict(row, ids, _Args())
        assert ok is True, why

    def test_lookup_accepts_several_key_spellings(self, wrapper):
        want = [7, 7, 7]
        for key in ("m@131072", "m_131072", "131072"):
            assert wrapper._lookup_prompt_ids({key: want}, "m", 131072) == want

    def test_lookup_supports_a_shared_prompt(self, wrapper):
        assert wrapper._lookup_prompt_ids({"*": [5]}, "any", 42) == [5]

    def test_lookup_miss_returns_none(self, wrapper):
        # A miss must be None, not a fabricated prompt: None is what makes
        # the row ineligible instead of silently wrong.
        assert wrapper._lookup_prompt_ids({"other@1": [1]}, "m", 131072) is None
        assert wrapper._lookup_prompt_ids({}, "m", 131072) is None


class TestC3OneEosPolicy:
    def test_policy_ignores_eos_and_fixes_the_token_count(self, wrapper):
        assert "ignore_eos" in wrapper.EOS_POLICY

    def test_a_row_with_a_different_eos_policy_is_rejected(self, wrapper):
        row = dict(_ok_row(wrapper), eos_policy="stop_on_eos")
        ok, why = wrapper._unit_match_verdict(row, [1, 2], _Args())
        assert ok is False
        assert "C3" in why

    def test_eos_policy_is_recorded_in_csv_fields(self, wrapper):
        assert "eos_policy" in wrapper.CSV_FIELDS


class TestC4BothPrecisions:
    def test_both_bitsandbytes_and_bf16_are_default(self, wrapper):
        # The draft story rests on 4-bit bitsandbytes weights, so the bf16 row
        # alone would not answer the reviewer; the 4-bit row alone would make
        # the quantization difference invisible. Both, as separate rows.
        import inspect

        src = inspect.getsource(wrapper.main)
        assert '"bitsandbytes"' in src
        assert '"bfloat16"' in src

    def test_quantization_is_recorded_in_csv_fields(self, wrapper):
        assert "quantization" in wrapper.CSV_FIELDS


class TestC5UnitMatching:
    def test_all_four_conditions_together(self, wrapper):
        ids = [1]
        row = _ok_row(wrapper, prompt_sha256=wrapper._ids_sha(ids))
        ok, why = wrapper._unit_match_verdict(row, ids, _Args())
        assert ok is True, why

    def test_tensor_parallelism_must_be_8(self, wrapper):
        row = dict(_ok_row(wrapper), tensor_parallel_size=1)
        ok, why = wrapper._unit_match_verdict(row, [1], _Args())
        assert ok is False
        assert "tensor_parallel" in why

    def test_max_new_tokens_must_match_the_rasd_cell(self, wrapper):
        row = dict(_ok_row(wrapper), max_new_tokens=64)
        ok, why = wrapper._unit_match_verdict(row, [1], _Args())
        assert ok is False
        assert "max_new_tokens" in why

    def test_end_to_end_and_decode_only_are_separate_columns(self, wrapper):
        # C5 requires both be reported, separately, so a reader can see that
        # end-to-end includes prefill and is the one matching throughput_tps.
        assert "throughput_tps_end_to_end" in wrapper.CSV_FIELDS
        assert "throughput_tps_decode_only" in wrapper.CSV_FIELDS


class TestB4PG19TargetOnlyConfig:
    CRITICAL_REV = "a4b76938edbf571ea7d7d9904861cbdca08809b4"

    @pytest.fixture(scope="class")
    def cfg(self):
        path = REPO / "configs" / "mlsys_pg19_short_targetonly.yml"
        return yaml.safe_load(path.read_text())

    def test_target_only(self, cfg):
        # spec_steps=0 is what makes this a baseline rather than a spec cell.
        assert cfg["defaults"]["spec_steps"] == 0

    def test_covers_exactly_the_missing_seeds(self, cfg):
        # Seed 42 already has 4k/8k target rows in results/final; adding it
        # again would create a duplicate, and omitting 123/456 would leave the
        # speedup without a denominator.
        assert cfg["defaults"]["seeds"] == [123, 456]

    def test_covers_4k_and_8k_only(self, cfg):
        levels = cfg["PG19_SHORT_TARGET_ONLY"]["levels"]
        assert [lv["context_length"] for lv in levels] == [4096, 8192]

    def test_matches_seed_42_generation_settings(self, cfg):
        # These must equal the seed-42 rows they pair with, or the ratio
        # compares two different configurations.
        d = cfg["defaults"]
        assert d["target_model_name"] == "meta-llama/Llama-2-7b-hf"
        assert d["draft_model_name"] == "princeton-nlp/Sheared-LLaMA-1.3B"
        assert d["draft_revision"] == self.CRITICAL_REV
        assert d["kv_block_size"] == 2048
        assert d["prefetch_depth"] == 1
        assert d["max_new_tokens"] == 64
        assert d["rope_type"] == "yarn"
        assert d["dtype"] == "bfloat16"

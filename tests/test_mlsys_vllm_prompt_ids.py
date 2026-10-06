"""vLLM must be handed the RASD prompt as TOKEN IDS -- and must prove it took them.

The baseline's whole claim is "same prompt as the RASD cell". The previous
version recovered a text prompt with `tokenizer.decode(ids)` and let vLLM
re-encode it. That turns the claim into a property of the tokenizer instead of a
property of the run: decode is not injective, so the sequence vLLM conditions on
can differ from the one RASD generated, and nothing in the row would show it.
vLLM accepts `prompt_token_ids` directly and reports the ids it consumed, so the
claim can be *checked* rather than asserted.

These tests drive the real `worker_main` with a deliberately lossy tokenizer, so
a regression to the decode/re-encode path is a failure here, not a silent
downgrade of every throughput ratio in the paper.
"""
from __future__ import annotations

import importlib.util
import json
import sys
import types
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
SCRIPT = REPO / "scripts" / "mlsys_vllm_baseline.py"

RASD_IDS = [128000, 9906, 1917, 527, 264, 13, 42, 7]
TARGET_REV = "d04e592bb4f6aa9cfee91e2e20afa771667e1d4b"
DRAFT_REV = "4e8bd4cd9b6f8f6b0c1e1a4a0f0f7f5c9c0e0d0a"


def _load_module():
    spec = importlib.util.spec_from_file_location("_vllm_baseline_under_test",
                                                  SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class _LossyTokenizer:
    """decode() throws information away, so decode->encode is not the identity.

    A round trip through this tokenizer cannot give back the ids it was given,
    which is exactly the failure the real (non-injective) decode can hide.
    """

    decode_calls: list = []
    encode_calls: list = []

    def __init__(self):
        self.decode_calls = []
        self.encode_calls = []
        _LossyTokenizer.decode_calls = self.decode_calls
        _LossyTokenizer.encode_calls = self.encode_calls

    def decode(self, ids, **kw):                       # pragma: no cover
        self.decode_calls.append(list(ids))
        return "TEXT-THAT-REENCODES-TO-DIFFERENT-IDS"

    def __call__(self, text, **kw):
        self.encode_calls.append(text)
        return {"input_ids": [1, 2, 3]}


class _FakeLLM:
    last_prompt = None
    last_params = None
    report_ids = None            # None -> echo what was given

    def __init__(self, **kw):
        self.kw = kw

    def generate(self, prompts, params, use_tqdm=False):
        type(self).last_prompt = prompts[0]
        type(self).last_params = params
        ids = (prompts[0]["prompt_token_ids"]
               if isinstance(prompts[0], dict) else None)
        reported = type(self).report_ids if type(self).report_ids is not None \
            else (list(ids) if ids is not None else None)
        out = types.SimpleNamespace(prompt_token_ids=reported, metrics=None)
        out.outputs = [types.SimpleNamespace(token_ids=[5, 6, 7, 8])]
        return [out]


def _install_fakes(monkeypatch, mod):
    torch = types.ModuleType("torch")

    class _OOM(RuntimeError):
        pass

    torch.cuda = types.SimpleNamespace(
        is_available=lambda: False,
        reset_peak_memory_stats=lambda *a, **k: None,
        max_memory_allocated=lambda: 0,
        empty_cache=lambda: None,
        OutOfMemoryError=_OOM,
    )
    transformers = types.ModuleType("transformers")
    transformers.AutoTokenizer = types.SimpleNamespace(
        from_pretrained=lambda model, **kw: _LossyTokenizer())
    vllm = types.ModuleType("vllm")
    vllm.__version__ = mod.VLLM_PIN
    vllm.LLM = _FakeLLM
    vllm.SamplingParams = lambda **kw: kw
    for name, m in (("torch", torch), ("transformers", transformers),
                    ("vllm", vllm)):
        monkeypatch.setitem(sys.modules, name, m)
    _FakeLLM.last_prompt = None
    _FakeLLM.last_params = None
    _FakeLLM.report_ids = None
    return torch


def _spec(tmp_path, **over):
    spec = {
        "result_path": str(tmp_path / "r.json"),
        "model": "meta-llama/Llama-3.1-8B-Instruct",
        "context_length": 131072,
        "max_new_tokens": 1024,
        "max_model_len": 131072,
        "rope": None,
        "quantization": None,
        "tensor_parallel_size": 8,
        "kwargs": {},
        "prompt_ids": list(RASD_IDS),
        "prompt_source": "rasd_token_ids",
        "target_revision": TARGET_REV,
        "draft_revision": DRAFT_REV,
    }
    spec.update(over)
    return spec


def _run(tmp_path, monkeypatch, spec, mod=None):
    mod = mod or _load_module()
    _install_fakes(monkeypatch, mod)
    p = tmp_path / "spec.json"
    p.write_text(json.dumps(spec))
    rc = mod.worker_main(p)
    result = json.loads(Path(spec["result_path"]).read_text())
    return rc, result, mod


# ---------------------------------------------------------------------------

def test_prompt_is_passed_as_ids_not_reencoded_text(tmp_path, monkeypatch):
    """The regression test. Before the fix this received a decoded string."""
    _, result, _ = _run(tmp_path, monkeypatch, _spec(tmp_path))

    assert isinstance(_FakeLLM.last_prompt, dict), (
        "vLLM was given text, not token ids; 'same prompt' is then an "
        "assumption about the tokenizer rather than a fact about the run")
    assert _FakeLLM.last_prompt["prompt_token_ids"] == RASD_IDS


def test_the_ids_are_never_round_tripped_through_decode(tmp_path, monkeypatch):
    _run(tmp_path, monkeypatch, _spec(tmp_path))

    assert _LossyTokenizer.decode_calls == [], (
        "decode() was called on the RASD ids; a lossy decode silently changes "
        "the sequence vLLM conditions on")
    assert not any("TEXT-THAT-REENCODES" in str(c)
                   for c in _LossyTokenizer.encode_calls)


def test_worker_records_that_vllm_consumed_the_supplied_ids(tmp_path, monkeypatch):
    _, result, _ = _run(tmp_path, monkeypatch, _spec(tmp_path))

    assert result["status"] == "ok"
    assert result["prompt_ids_verified"] == "yes"
    assert result["prompt_ids_used_sha256"] == result["prompt_ids_used_sha256"]
    assert result["prompt_ids_used_sha256"], "the ids vLLM took are unrecorded"


def test_a_vllm_prompt_id_mismatch_is_reported_not_hidden(tmp_path, monkeypatch):
    """If vLLM consumes different ids, the row must say so."""
    mod = _load_module()
    _install_fakes(monkeypatch, mod)
    _FakeLLM.report_ids = [1, 2, 3]

    spec = _spec(tmp_path)
    p = tmp_path / "spec.json"
    p.write_text(json.dumps(spec))
    mod.worker_main(p)
    result = json.loads(Path(spec["result_path"]).read_text())

    assert result["prompt_ids_verified"] == "no"
    assert result["prompt_ids_used_sha256"]


def test_greedy_and_ignore_eos_match_the_rasd_cells(tmp_path, monkeypatch):
    _run(tmp_path, monkeypatch, _spec(tmp_path))

    params = _FakeLLM.last_params
    assert params["temperature"] == 0.0
    assert params["max_tokens"] == 1024
    assert params["ignore_eos"] is True


def test_revisions_travel_from_spec_to_row(tmp_path, monkeypatch):
    _, result, _ = _run(tmp_path, monkeypatch, _spec(tmp_path))

    assert result["target_revision"] == TARGET_REV
    assert result["draft_revision"] == DRAFT_REV


# --- the verdict itself -----------------------------------------------------

def _verdict_args(**over):
    args = types.SimpleNamespace(matched_max_new_tokens=1024,
                                 target_revision=TARGET_REV,
                                 draft_revision=DRAFT_REV)
    for k, v in over.items():
        setattr(args, k, v)
    return args


def _good_row(mod, **over):
    row = {"eos_policy": mod.EOS_POLICY, "tensor_parallel_size": 8,
           "max_new_tokens": 1024, "temperature": 0.0,
           "vllm_version": mod.VLLM_PIN, "doc_id": "pg19_train_0",
           "prompt_sha256": mod._ids_sha(RASD_IDS),
           "prompt_ids_verified": "yes",
           "target_revision": TARGET_REV, "draft_revision": DRAFT_REV}
    row.update(over)
    return row


def test_unverified_prompt_ids_are_not_a_unit_match():
    mod = _load_module()
    ok, why = mod._unit_match_verdict(_good_row(mod, prompt_ids_verified=""),
                                      RASD_IDS, _verdict_args())
    assert not ok and "unverified" in why


def test_mismatched_target_revision_is_not_a_unit_match():
    mod = _load_module()
    ok, why = mod._unit_match_verdict(_good_row(mod, target_revision="deadbeef"),
                                      RASD_IDS, _verdict_args())
    assert not ok and "target_revision" in why


def test_missing_revision_on_a_pinned_row_is_not_a_unit_match():
    mod = _load_module()
    ok, why = mod._unit_match_verdict(_good_row(mod, draft_revision=""),
                                      RASD_IDS, _verdict_args())
    assert not ok and "draft_revision" in why


def test_a_fully_pinned_row_is_a_unit_match():
    mod = _load_module()
    ok, why = mod._unit_match_verdict(_good_row(mod), RASD_IDS, _verdict_args())
    assert ok, why


def test_an_unpinned_comparison_does_not_require_revisions():
    """Older calls without --target-revision stay usable (weaker, not broken)."""
    mod = _load_module()
    args = _verdict_args(target_revision="", draft_revision="")
    ok, why = mod._unit_match_verdict(_good_row(mod, target_revision="",
                                                draft_revision=""),
                                      RASD_IDS, args)
    assert ok, why


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))


# --- per-model revision pinning (the ladder runs two targets) --------------

def test_a_bare_revision_in_a_map_is_refused():
    """`--target-revisions 01c7f7...` cannot say which model it pins."""
    mod = _load_module()
    with pytest.raises(SystemExit) as e:
        mod._parse_revision_map("01c7f73d771dfac7d292323805ebc428287df4f9")
    assert "model=revision" in str(e.value)


def test_a_revision_map_parses_per_model():
    mod = _load_module()
    m = mod._parse_revision_map("meta-llama/Llama-3.1-8B=aaa,"
                                "meta-llama/Llama-2-7b-hf=bbb")
    assert m == {"meta-llama/Llama-3.1-8B": "aaa",
                 "meta-llama/Llama-2-7b-hf": "bbb"}


def test_a_row_pinned_to_the_other_model_is_not_unit_matched():
    """The failure this guards: a ladder row pinned to the wrong target's rev."""
    mod = _load_module()
    args = _verdict_args(target_revision="", draft_revision="")
    ok, why = mod._unit_match_verdict(
        _good_row(mod, target_revision="01c7f73d771dfac7d292323805ebc428287df4f9"),
        RASD_IDS, args, target_revision=TARGET_REV, draft_revision=DRAFT_REV)
    assert not ok and "target_revision" in why


def test_an_explicit_per_model_revision_overrides_the_singular_flag():
    mod = _load_module()
    args = _verdict_args(target_revision="WRONG", draft_revision="")
    ok, why = mod._unit_match_verdict(_good_row(mod), RASD_IDS, args,
                                      target_revision=TARGET_REV,
                                      draft_revision="")
    assert ok, why

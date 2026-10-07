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

import csv
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
           "prompt_ids_verified": "yes", "prompt_ids_from_engine": "yes",
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


# --- the BOS: vLLM must be given the ids the TARGET was fed ----------------

BOS = 128000
PROMPT_N0_BOS = [9906, 1917, 527, 264, 13, 42, 7]
ENGINE_IDS = [BOS] + PROMPT_N0_BOS


def _sidecar(tmp_path, ids, prompt_tokens, sha=None):
    import hashlib
    d = tmp_path / "tokens"
    d.mkdir(exist_ok=True)
    payload = {"run_id": "r1", "doc_id": "pg19_train_0",
               "context_length": 131072,
               "prompt_tokens": prompt_tokens,
               "prompt_token_ids": PROMPT_N0_BOS,
               "prompt_sha256": hashlib.sha256(
                   ",".join(str(i) for i in PROMPT_N0_BOS).encode()).hexdigest()}
    if ids is not None:
        payload["engine_input_ids"] = ids
    (d / "r1.json").write_text(json.dumps(payload))
    return d


def test_the_loader_uses_the_engine_input_ids_with_the_bos(tmp_path):
    mod = _load_module()
    mod.load_rasd_cells.__globals__.setdefault("__file__", "")
    cells = mod.load_rasd_cells(_sidecar(tmp_path, ENGINE_IDS, len(PROMPT_N0_BOS)))
    assert len(cells) == 1
    assert cells[0]["prompt_ids"] == ENGINE_IDS, (
        "the prompt handed to vLLM must be the ids the target was fed")
    assert cells[0]["prompt_ids"][0] == BOS
    assert cells[0]["engine_input_ids"] is True


def test_a_sidecar_without_engine_ids_yields_no_prompt(tmp_path):
    """The BOS-carrying field is what makes the prompt the target's.

    Without it the document has no usable prompt ids, so the stage refuses
    rather than comparing vLLM against a prompt the target never saw.
    """
    mod = _load_module()
    with pytest.raises(SystemExit) as e:
        mod.load_rasd_cells(_sidecar(tmp_path, None, len(PROMPT_N0_BOS)))
    assert "engine_input_ids" in str(e.value)


def test_a_sidecar_missing_the_bos_cannot_be_unit_matched(tmp_path):
    """`len(engine_input_ids) == prompt_tokens + 1` is the BOS accounting.

    A sidecar whose engine ids are the prompt WITHOUT the BOS is one token short
    of the sequence the target conditioned on, so it is not the target's prompt
    and the row must not be unit-matched.
    """
    mod = _load_module()
    # engine_input_ids without the BOS, and prompt_tokens claiming the full count
    with pytest.raises(SystemExit):
        mod.load_rasd_cells(_sidecar(tmp_path, PROMPT_N0_BOS,
                                     len(PROMPT_N0_BOS)))
    # ... and the row-level guard refuses too, for a cell that arrived some
    # other way (a rebuilt prompt, an older sidecar read through --prompt-ids).
    args = _verdict_args(target_revision="", draft_revision="")
    ok, why = mod._unit_match_verdict(
        _good_row(mod, prompt_ids_from_engine="", prompt_sha256=None),
        PROMPT_N0_BOS, args)
    assert not ok, (
        "a row whose prompt ids are not the engine's input ids was unit-matched")


def test_the_row_records_that_the_ids_are_the_engines():
    mod = _load_module()
    args = _verdict_args(target_revision="", draft_revision="")
    ok, why = mod._unit_match_verdict(
        _good_row(mod, prompt_ids_from_engine="",
                  prompt_sha256=mod._ids_sha(ENGINE_IDS)), ENGINE_IDS, args)
    assert not ok and "engine's input ids" in why
    ok2, why2 = mod._unit_match_verdict(
        _good_row(mod, prompt_ids_from_engine="yes",
                  prompt_sha256=mod._ids_sha(ENGINE_IDS)), ENGINE_IDS, args)
    assert ok2, why2


# --- the PARENT's row and CSV path -----------------------------------------
# The worker cannot report `prompt_ids_from_engine`: whether the ids handed to
# vLLM are the ids the target engine was fed is a fact about the RASD sidecar,
# which the worker never sees. An earlier parent copied only the worker's result
# fields, so the marker was empty on EVERY row and no vLLM row could ever be
# unit-matched -- the stage would have "passed" while producing a comparison
# table with zero comparable rows. These tests drive the production parent path
# (`build_row` + `write_rows`, and `main` end to end with a stubbed worker) with
# a fake worker result that does NOT contain the marker.

def _parent_args(**over):
    args = types.SimpleNamespace(max_new_tokens=1024, tensor_parallel_size=8,
                                 matched_max_new_tokens=1024,
                                 target_revision=TARGET_REV,
                                 draft_revision=DRAFT_REV)
    for k, v in over.items():
        setattr(args, k, v)
    return args


def _worker_result(mod, prompt_ids, **over):
    """A result shaped exactly like the real worker's `result.update({...})`."""
    result = {
        "status": "ok", "prompt_tokens": len(prompt_ids),
        "vllm_version": mod.VLLM_PIN, "quantization": "bfloat16",
        "eos_policy": mod.EOS_POLICY, "prompt_source": "rasd_token_ids",
        "prompt_sha256": mod._ids_sha(prompt_ids),
        "prompt_ids_used_sha256": mod._ids_sha(prompt_ids),
        "prompt_ids_verified": "yes",
        "target_revision": TARGET_REV, "draft_revision": DRAFT_REV,
        "output_tokens": 1024, "end_to_end_wall_s": 40.0,
        "decode_only_wall_s": 39.0, "ttft_s": 1.0,
        "throughput_tps_end_to_end": 25.6,
        "throughput_tps_decode_only": 26.25,
        "peak_mem_mb": 78000.0, "error": "", "error_class": "",
    }
    # If the worker ever reported these, this test would be checking the test
    # rather than the parent: the marker is the parent's to set. `doc_id` and
    # `temperature` are likewise the parent's (the worker is not told which
    # document its prompt came from, and does not know the RASD cells are
    # greedy).
    for forbidden in ("prompt_ids_from_engine", "doc_id", "temperature"):
        assert forbidden not in result
        assert forbidden not in over, f"{forbidden} must not come from the worker"
    result.update(over)
    return result


def _build(mod, cell, prompt_ids, **over):
    return mod.build_row(
        cell, "meta-llama/Llama-3.1-8B", 131072, prompt_ids,
        _worker_result(mod, prompt_ids, **over),
        attempt_idx=1, config_name="1-default-tp8",
        log_path="results/mlsys/logs/vllm_stub_attempt1.log", rope=None,
        args=_parent_args(), max_model_len=131072, target_revision=TARGET_REV,
        draft_revision=DRAFT_REV)


def test_a_shorter_context_fallback_is_not_a_unit_match(tmp_path):
    """The ladder's 64k fallback must not be certified as the 128k rung.

    It is a legitimate row to report -- "vLLM could not load 128k" is a result
    -- but pairing it with a 128k RASD cell would put a 64k throughput in the
    128k speedup column, which no reader could detect from the other columns.
    """
    mod = _load_module()
    cell = mod.load_rasd_cells(
        _sidecar(tmp_path, ENGINE_IDS, len(PROMPT_N0_BOS)))[0]

    fallen_back = mod.build_row(
        cell, "meta-llama/Llama-3.1-8B", 131072, ENGINE_IDS,
        _worker_result(mod, ENGINE_IDS),
        attempt_idx=3, config_name="3-fallback-64k", log_path="logs/x.log",
        rope=None, args=_parent_args(), max_model_len=65536,
        target_revision=TARGET_REV, draft_revision=DRAFT_REV)

    assert fallen_back["unit_matched"] == "no"
    assert "shorter-context fallback" in fallen_back["error"]
    assert fallen_back["max_model_len"] == 65536


def test_the_parent_sets_the_engine_marker_from_the_cell(tmp_path):
    """A cell whose sidecar carries `engine_input_ids` must yield yes."""
    mod = _load_module()
    cell = mod.load_rasd_cells(
        _sidecar(tmp_path, ENGINE_IDS, len(PROMPT_N0_BOS)))[0]

    row = _build(mod, cell, ENGINE_IDS)
    assert row["prompt_ids_from_engine"] == "yes"
    assert row["unit_matched"] == "yes", row["error"]

    out = tmp_path / "out.csv"
    mod.write_rows(out, [row])
    with out.open() as fh:
        got = list(csv.DictReader(fh))
    assert len(got) == 1
    assert got[0]["unit_matched"] == "yes", got[0]["error"]
    assert got[0]["prompt_ids_from_engine"] == "yes"


def test_a_cell_without_engine_ids_yields_no_unit_match(tmp_path):
    """No `engine_input_ids` -> no proof the ids are the target's -> no."""
    mod = _load_module()
    cell = {"doc_id": "pg19_train_0", "prompt_ids": ENGINE_IDS,
            "prompt_sha256": mod._ids_sha(ENGINE_IDS), "sidecar": "r1.json"}

    row = _build(mod, cell, ENGINE_IDS)
    assert row["prompt_ids_from_engine"] == ""
    assert row["unit_matched"] == "no"
    assert "engine's input ids" in row["error"]

    out = tmp_path / "out.csv"
    mod.write_rows(out, [row])
    with out.open() as fh:
        got = list(csv.DictReader(fh))
    assert got[0]["unit_matched"] == "no"


def _parent_argv(sidecar_dir, out, **over):
    argv = ["mlsys_vllm_baseline.py", "--out", str(out),
            "--prompt-ids-from-sidecars", str(sidecar_dir),
            "--context-lengths", "131072", "--max-new-tokens", "1024",
            "--matched-max-new-tokens", "1024",
            "--quantizations", "bfloat16",
            "--models", "meta-llama/Llama-3.1-8B",
            "--target-revisions", f"meta-llama/Llama-3.1-8B={TARGET_REV}",
            "--draft-revision", DRAFT_REV]
    for k, v in over.items():
        argv += [f"--{k.replace('_', '-')}", str(v)]
    return argv


def _drive_main(mod, tmp_path, monkeypatch, runner=None, **result_over):
    """Run the real main() with a stubbed worker subprocess."""
    if runner is None:
        def runner(spec):                                   # noqa: E731
            Path(spec["result_path"]).write_text(json.dumps(
                _worker_result(mod, spec["prompt_ids"], **result_over)))
            return 0, "stub worker log"

    def fake_run_attempt(spec, log_path, timeout_s):
        return runner(spec)

    monkeypatch.setattr(mod, "run_attempt", fake_run_attempt)
    out = tmp_path / "parent.csv"
    argv = _parent_argv(tmp_path / "tokens", out)
    monkeypatch.setattr(sys, "argv", argv)
    return mod.main(), out


def test_the_attempt_ladder_actually_retries(tmp_path, monkeypatch):
    """Regression: the execution used to sit OUTSIDE the ladder loop.

    The `for idx, att` body ended at the spec dict, so the loop built all three
    specs and then ran only the LAST one -- the 64k fallback -- labelling it
    attempt 3. Every vLLM row would have come from a 64k run while the row said
    128k, and `attempt`/`config_used` would have lied about which ladder fix
    produced it. Found by driving the parent end to end; the rehearsal could not
    see it because its stub replaces the whole module.
    """
    mod = _load_module()
    _sidecar(tmp_path, ENGINE_IDS, len(PROMPT_N0_BOS))
    calls: list = []

    def runner(spec):
        calls.append(spec["max_model_len"])
        if len(calls) < 3:
            return 1, "engine core initialization failed"
        Path(spec["result_path"]).write_text(json.dumps(
            _worker_result(mod, spec["prompt_ids"])))
        return 0, "ok"

    rc, out = _drive_main(mod, tmp_path, monkeypatch, runner=runner)

    assert calls == [131072, 131072, 65536], (
        "each ladder attempt must be tried in order until one succeeds")
    with out.open() as fh:
        rows = list(csv.DictReader(fh))
    assert [r["config_used"] for r in rows] == [
        "1-default-tp8", "2-eager-mem95-seq1", "3-fallback-64k"]
    # Only attempt 3 ran, and it ran at 64k. Its row is REPORTED -- "128k did
    # not load" is a result -- but it is not certified as the 128k rung, so the
    # stage has no usable baseline and must not be recorded ok.
    fallback = [r for r in rows if r["status"] == "ok"]
    assert len(fallback) == 1
    assert fallback[0]["max_model_len"] == "65536"
    assert fallback[0]["unit_matched"] == "no"
    assert "shorter-context fallback" in fallback[0]["error"]
    assert rc != 0, "zero unit-matched rows is a failed stage"


def test_the_ladder_stops_at_the_first_success(tmp_path, monkeypatch):
    mod = _load_module()
    _sidecar(tmp_path, ENGINE_IDS, len(PROMPT_N0_BOS))
    calls: list = []

    def runner(spec):
        calls.append(spec["max_model_len"])
        Path(spec["result_path"]).write_text(json.dumps(
            _worker_result(mod, spec["prompt_ids"])))
        return 0, "ok"

    rc, out = _drive_main(mod, tmp_path, monkeypatch, runner=runner)

    assert calls == [131072], "a successful attempt must not be repeated"
    assert rc == 0
    with out.open() as fh:
        rows = list(csv.DictReader(fh))
    assert len(rows) == 1
    assert rows[0]["config_used"] == "1-default-tp8"


def test_main_writes_a_unit_matched_row_end_to_end(tmp_path, monkeypatch):
    mod = _load_module()
    _sidecar(tmp_path, ENGINE_IDS, len(PROMPT_N0_BOS))

    rc, out = _drive_main(mod, tmp_path, monkeypatch)

    assert rc == 0, "a unit-matched row must leave the stage ok"
    with out.open() as fh:
        rows = list(csv.DictReader(fh))
    assert len(rows) == 1
    assert rows[0]["unit_matched"] == "yes", rows[0]["error"]
    assert rows[0]["prompt_ids_from_engine"] == "yes"


def test_main_fails_the_stage_when_no_row_is_unit_matched(tmp_path,
                                                        monkeypatch):
    """Zero comparable rows is a FAILED stage, not an ok one.

    Otherwise the campaign records a stage ok and a speedup table is built on
    an empty lookup while every individual number looks correct.
    """
    mod = _load_module()
    _sidecar(tmp_path, ENGINE_IDS, len(PROMPT_N0_BOS))

    # vLLM reported consuming ids that are not the ones it was given: the run
    # happened, the row is not comparable.
    rc, out = _drive_main(mod, tmp_path, monkeypatch, prompt_ids_verified="no")

    assert rc != 0, "a stage with no usable baseline must not exit 0"
    with out.open() as fh:
        rows = list(csv.DictReader(fh))
    assert rows and all(r["unit_matched"] == "no" for r in rows)

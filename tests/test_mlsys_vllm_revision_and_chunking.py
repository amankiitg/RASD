"""Two first-run blockers in the vLLM baseline, fixed before spending on the 8x.

1. THE DECLARED REVISION WAS NEVER PASSED TO THE LOAD. `target_revision` travelled
   from the CLI into the spec, into the CSV row, and into `_unit_match`, which
   refuses to certify a row whose revision disagrees with the RASD cell's. But
   neither `LLM(...)` nor `AutoTokenizer.from_pretrained(...)` ever received it, so
   the engine served whatever `main` resolved to at load time while the row
   claimed a pin. Nothing downstream could detect the disagreement, which is the
   "verify the blob, not the intention" failure (L14) with a Hub repository in
   place of a file.

2. `max_num_batched_tokens = ctx` IS NOT CHUNKED PREFILL. The ladder's long-context
   attempt set the batch size to the whole context while its note read "chunked
   prefill sized to ctx"; a chunk equal to the context means no chunking, so the
   scheduler batches the entire prompt into one forward. At 1M the activation
   tensors alone (8.6 GB for the hidden states, 46.2 GB for the MLP intermediate)
   would not fit beside 17.2 GB of bf16 KV per rank in a 0.95 x 80 GB budget.

These tests run the real worker with stub `torch`/`transformers`/`vllm` modules, so
they assert on what the code does with a spec, not on the source text.
"""
from __future__ import annotations

import importlib.util
import json
import sys
import types
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
SCRIPT = REPO / "scripts" / "mlsys_vllm_baseline.py"

TARGET_REV = "cd3069b65a8eb13da639d332a5f61b0fbb29fa73"
RASD_IDS = [128000, 9906, 1917, 11]


def _load_module():
    spec = importlib.util.spec_from_file_location("_vllm_rev_under_test", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class _FakeLLM:
    last_kw = None
    last_tok_rev = None

    def __init__(self, **kw):
        type(self).last_kw = kw
        self.kw = kw

    def generate(self, prompts, params, use_tqdm=False):
        ids = prompts[0]["prompt_token_ids"] if isinstance(prompts[0], dict) else None
        out = types.SimpleNamespace(prompt_token_ids=list(ids) if ids else None,
                                    metrics=None)
        out.outputs = [types.SimpleNamespace(token_ids=[5, 6, 7, 8])]
        return [out]


class _RecordingTokenizer:
    def __init__(self, **kw):
        type(self).last_kw = kw

    def __call__(self, text, **kw):            # pragma: no cover - unused here
        return {"input_ids": [1, 2, 3]}


def _install_fakes(monkeypatch, mod):
    torch = types.ModuleType("torch")

    class _OOM(RuntimeError):
        pass

    torch.cuda = types.SimpleNamespace(is_available=lambda: False,
                                       reset_peak_memory_stats=lambda *a, **k: None,
                                       max_memory_allocated=lambda: 0,
                                       empty_cache=lambda: None,
                                       OutOfMemoryError=_OOM)
    transformers = types.ModuleType("transformers")
    transformers.AutoTokenizer = types.SimpleNamespace(
        from_pretrained=lambda model, **kw: _RecordingTokenizer(**kw))
    vllm = types.ModuleType("vllm")
    vllm.__version__ = mod.VLLM_PIN
    vllm.LLM = _FakeLLM
    vllm.SamplingParams = lambda **kw: kw
    for name, m in (("torch", torch), ("transformers", transformers), ("vllm", vllm)):
        monkeypatch.setitem(sys.modules, name, m)
    _FakeLLM.last_kw = None
    _RecordingTokenizer.last_kw = None


def _spec(tmp_path, **over):
    spec = {
        "result_path": str(tmp_path / "r.json"),
        "model": "gradientai/Llama-3-8B-Instruct-Gradient-1048k",
        "context_length": 131072,
        "max_new_tokens": 128,
        "max_model_len": 131072,
        "rope": None,
        "quantization": None,
        "tensor_parallel_size": 8,
        "kwargs": {},
        "prompt_ids": list(RASD_IDS),
        "prompt_source": "rasd_token_ids",
        "target_revision": TARGET_REV,
        "draft_revision": "",
    }
    spec.update(over)
    return spec


def _run(tmp_path, monkeypatch, spec):
    mod = _load_module()
    _install_fakes(monkeypatch, mod)
    p = tmp_path / "spec.json"
    p.write_text(json.dumps(spec))
    rc = mod.worker_main(p)
    return rc, json.loads(Path(spec["result_path"]).read_text())


def test_the_declared_revision_reaches_both_loads(tmp_path, monkeypatch):
    _run(tmp_path, monkeypatch, _spec(tmp_path))
    assert _FakeLLM.last_kw is not None, "the engine was never constructed"
    assert _FakeLLM.last_kw.get("revision") == TARGET_REV, (
        "LLM() was not given the declared revision, so the row's pin is a claim "
        f"about the Hub, not about the run: {_FakeLLM.last_kw.get('revision')!r}")
    assert _FakeLLM.last_kw.get("tokenizer_revision") == TARGET_REV
    assert _RecordingTokenizer.last_kw.get("revision") == TARGET_REV, (
        "AutoTokenizer loaded an unpinned revision")


def test_an_unpinned_spec_passes_no_revision(tmp_path, monkeypatch):
    """The control: without a declared revision the load must not invent one."""
    _run(tmp_path, monkeypatch, _spec(tmp_path, target_revision=""))
    assert "revision" not in _FakeLLM.last_kw
    assert "tokenizer_revision" not in _FakeLLM.last_kw


def test_no_ladder_attempt_asks_for_an_unchunked_long_context():
    """The config rule, asserted on the ladder itself."""
    mod = _load_module()
    long_attempts = [a for a in mod.ATTEMPT_LADDER
                     if a["max_model_len"] == "ctx"]
    assert long_attempts, "the ladder lost its long-context attempts"
    for a in long_attempts:
        mnbt = a.get("max_num_batched_tokens")
        if mnbt is None:
            continue
        assert mnbt != "ctx", f"{a['name']} sets the batch to the whole context"
        assert isinstance(mnbt, int) and mnbt <= mod.CHUNKED_PREFILL_TOKENS, \
            f"{a['name']} batches {mnbt} tokens in one forward"
        assert a["kwargs"].get("enable_chunked_prefill") is True, \
            f"{a['name']} bounds the batch but never asks for chunked prefill"

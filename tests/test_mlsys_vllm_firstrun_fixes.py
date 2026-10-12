"""Three first-run blockers in the vLLM baseline, fixed before spending on the 8x.

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

3. THE SIDECAR SET THE STAGE WILL BE POINTED AT. The stage takes its prompts from
   RASD token sidecars, and the set committed for this session
   (`results/mlsys/sampling_session/tokens_slim/`) deliberately DROPS
   `engine_input_ids` -- it is the whole prompt, 130k-1M ids per row. This was
   checked before the 8x run so that the discovery would not happen on a paid
   instance. The check found that `load_rasd_cells` already refuses an empty
   result by itself, at the CLI, before any worker starts: that behaviour is
   asserted here rather than re-implemented.
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


# ---------------------------------------------------------------------------
# 3. zero RASD cells must fail before the stage spends hours
# ---------------------------------------------------------------------------

def test_a_sidecar_set_without_engine_ids_fails_before_running(tmp_path):
    """The committed SLIM sidecars are exactly this case, and it already fails fast.

    Asserted from the CLI, because that is where the refusal has to happen: once
    a cell loop starts, every row would run a synthetic prompt and the stage
    would be recorded FAILED hours later.
    """
    import subprocess
    d = tmp_path / "slim_tokens"
    d.mkdir()
    (d / "SAMP_1M_TARGET_pg19_train_0_s42.json").write_text(json.dumps({
        "run_id": "SAMP_1M_TARGET_pg19_train_0_s42",
        "doc_id": "pg19_train_0",
        "context_length": 1048576,
        "spec_steps": 0,
        "generated_token_ids": [1, 2, 3],
        "prompt_tokens": 1047551,
        "prompt_source_len": 1047551,
        # engine_input_ids and prompt_token_ids: absent, as in the slim set.
    }))
    proc = subprocess.run(
        [sys.executable, str(SCRIPT),
         "--models", "gradientai/Llama-3-8B-Instruct-Gradient-1048k",
         "--prompt-ids-from-sidecars", str(d),
         "--out", str(tmp_path / "out.csv")],
        cwd=str(REPO), capture_output=True, text=True, timeout=300)
    out = proc.stdout + proc.stderr
    assert proc.returncode != 0, "zero cells must not proceed to run"
    assert "no usable sidecar" in out
    assert "engine_input_ids" in out, "the refusal must name the missing field"


# ---------------------------------------------------------------------------
# 4. the version gate must not refuse a correctly pinned install
# ---------------------------------------------------------------------------

def test_a_dev_version_attribute_does_not_refuse_a_pinned_distribution(
        tmp_path, monkeypatch):
    """THE REHEARSAL'S FAILURE, 2026-10-12.

    `vllm==0.6.3` from PyPI reported `vllm.__version__ == 'dev'` on that pod, and
    the worker compared that single value against the pin and raised SystemExit --
    for ALL THREE ladder attempts. Three "wrong version" rows, no measurement.
    The distribution metadata is the citable identity and says 0.6.3.
    """
    import importlib.metadata as md
    real = md.version
    monkeypatch.setattr(md, "version",
                        lambda n: "0.6.3" if n == "vllm" else real(n))
    mod = _load_module()
    _install_fakes(monkeypatch, mod)
    # The fake module reports the value the real one reported on the pod.
    monkeypatch.setattr(sys.modules["vllm"], "__version__", "dev")
    p = tmp_path / "spec.json"
    p.write_text(json.dumps(_spec(tmp_path)))
    rc = mod.worker_main(p)
    result = json.loads(Path(_spec(tmp_path)["result_path"]).read_text())
    assert rc == 0, f"a correctly pinned install must not be refused: {result}"
    assert result["vllm_version"] == "0.6.3"
    assert result["vllm_version_source"] == "importlib.metadata", \
        "the row must say WHICH identity agreed"


def test_a_wrong_version_is_still_refused(tmp_path, monkeypatch):
    """The control: the gate must still refuse a genuinely wrong release."""
    import importlib.metadata as md
    real = md.version
    monkeypatch.setattr(md, "version",
                        lambda n: "0.6.4" if n == "vllm" else real(n))
    mod = _load_module()
    _install_fakes(monkeypatch, mod)
    monkeypatch.setattr(sys.modules["vllm"], "__version__", "0.6.4")
    p = tmp_path / "spec.json"
    p.write_text(json.dumps(_spec(tmp_path)))
    import pytest
    with pytest.raises(SystemExit) as e:
        mod.worker_main(p)
    assert "0.6.3" in str(e.value) and "0.6.4" in str(e.value)


def test_the_venv_pins_transformers_below_five():
    """The other half of the rehearsal's failure.

    vLLM 0.6.3 declares `transformers>=4.45.0` with no cap, so pip resolved
    5.19.0; transformers 5.x cannot see torch 2.4.0. The pin must be installed
    BEFORE vllm (so the resolver never upgrades it) and the script must ASSERT the
    result rather than print it.
    """
    src = (REPO / "scripts" / "mlsys_vllm_venv.sh").read_text()
    i_tf = src.index('pip install --quiet "transformers==$TRANSFORMERS_PIN"')
    i_vllm = src.index('pip install --quiet "vllm==$VLLM_PIN"')
    assert i_tf < i_vllm, "transformers must be pinned before vllm is installed"
    assert "MLSYS_VLLM_TRANSFORMERS_PIN:-4.46.3" in src
    assert "is 5.x, which cannot" in src and "sys.exit(7)" in src, \
        "the install must fail when the pin did not take"


# ---------------------------------------------------------------------------
# 5. the bf16 label must not be reported as 4-bit weights
# ---------------------------------------------------------------------------

def test_a_bf16_row_is_not_labelled_int4_bnb(tmp_path, monkeypatch):
    """FOUND ON THE REHEARSAL ROW, 2026-10-12.

    `--quantizations` DEFAULTS TO ["bfloat16"], and the worker tested that label for
    truthiness, so every bf16 row carried `weight_precision=int4_bnb` -- beside its
    own `kv_dtype=bfloat16`, which cannot both be true, and in the column the
    cross-check puts next to RASD's `fp4`. The rehearsal CSV in
    `results/mlsys/vllm_rehearsal/` still shows the wrong label; the fix is here.
    """
    mod = _load_module()
    _install_fakes(monkeypatch, mod)
    p = tmp_path / "spec.json"
    p.write_text(json.dumps(_spec(tmp_path, quantization="bfloat16")))
    rc = mod.worker_main(p)
    result = json.loads(Path(_spec(tmp_path)["result_path"]).read_text())
    assert rc == 0
    assert result["weight_precision"] != "int4_bnb", (
        "a bf16 row was labelled as 4-bit weights")
    assert result["weight_precision"] == result["kv_dtype"]


def test_the_bitsandbytes_label_is_still_int4_bnb(tmp_path, monkeypatch):
    """The control: the bnb path is the one that IS un-introspectable."""
    mod = _load_module()
    _install_fakes(monkeypatch, mod)
    p = tmp_path / "spec.json"
    p.write_text(json.dumps(_spec(tmp_path, quantization="bitsandbytes")))
    rc = mod.worker_main(p)
    result = json.loads(Path(_spec(tmp_path)["result_path"]).read_text())
    assert rc == 0
    assert result["weight_precision"] == "int4_bnb"


class _Metrics:
    def __init__(self, **kw):
        self.__dict__.update(kw)


class _Out:
    def __init__(self, metrics):
        self.metrics = metrics


def test_ttft_is_a_delta_and_not_an_absolute_clock_reading():
    """THE 2026-10-12 BUG, as a test.

    vLLM 0.6.3's RequestMetrics carries ABSOLUTE MONOTONIC timestamps. The first
    version of this helper returned `first_token_time` on its own, so the 8x run's CSV
    recorded ttft_s = 1.79e9, decode_only_wall_s = -1.79e9 and a decode rate of
    1.27e11 tok/s. The helper must return the DIFFERENCE, or nothing at all.
    """
    mod = _load_module()
    # The stamp the real run produced, with a prefill of 300 s inside it.
    out = _Out(_Metrics(arrival_time=1791804235.0333, first_token_time=1791804535.0333))
    assert mod._ttft_from_metrics(out, end_to_end=863.5764) == 300.0
    assert mod._ttft_from_metrics(out, end_to_end=863.5764) != 1791804535.0333


def test_an_implausible_ttft_is_refused_rather_than_recorded():
    """A delta longer than the request that produced it cannot be a TTFT."""
    mod = _load_module()
    out = _Out(_Metrics(arrival_time=0.0, first_token_time=900.0))
    assert mod._ttft_from_metrics(out, end_to_end=18.3) is None


def test_a_negative_delta_is_refused():
    mod = _load_module()
    out = _Out(_Metrics(arrival_time=100.0, first_token_time=50.0))
    assert mod._ttft_from_metrics(out, end_to_end=18.3) is None


def test_missing_metrics_or_missing_fields_yield_no_ttft():
    mod = _load_module()
    assert mod._ttft_from_metrics(_Out(None), end_to_end=18.3) is None
    assert mod._ttft_from_metrics(_Out(_Metrics(first_token_time=5.0)),
                                  end_to_end=18.3) is None
    assert mod._ttft_from_metrics(_Out(_Metrics(arrival_time=1.0)),
                                  end_to_end=18.3) is None

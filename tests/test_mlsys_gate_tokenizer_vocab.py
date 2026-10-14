"""The gate must never hand a model another model's token ids.

Measured 2026-10-08 on the 8xA100 campaign: `gate_calibration` ran six
candidates and then aborted with

    ../aten/src/ATen/native/cuda/Indexing.cu:1308: indexSelectLargeIndex:
      Assertion `srcIndex < srcSelectDimSize` failed.

reported at `torch.cuda.empty_cache()`. The failing op was
`modeling_llama.py:891  inputs_embeds = self.embed_tokens(input_ids)`, reached
with `B_llama2_yarn8_32k_correct` (Llama-2-7B, vocab_size 32000): the prompt
came from `data/processed/pg19_docs/documents.json`, whose own metadata records
`"tokenizer": "meta-llama/Llama-3.1-8B"`. 2391 of that candidate's 31744 prompt
ids (7.5%) are >= 32000, the first at index 3, and the max is 127146.

The stage wrote NO CSV, so the incident had no rows to diagnose. Four things are
pinned here:

  * an out-of-vocab prompt is RECORDED as the candidate's outcome and the
    forward is never attempted -- the whole context dies otherwise, taking every
    later candidate with it;
  * the window is re-tokenized with the candidate's tokenizer when the pool's
    differs, which is what makes the Llama-2 controls measurable at all;
  * when the tokenizers AGREE the ids are byte-for-byte what they were before
    the fix, so the Llama-3.1 controls' recorded numbers stay comparable;
  * non-finite logits are recorded with their first position rather than
    crashed on, because the f2 construction is a known-broken control and that
    behaviour is a finding.
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parent.parent
GATE = REPO / "scripts" / "mlsys_coherence_gate.py"


def _load_gate():
    spec = importlib.util.spec_from_file_location("_gate_vocab_under_test", GATE)
    mod = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(mod)
    except Exception as e:                      # noqa: BLE001
        pytest.skip(f"gate not importable here: {e}")
    return mod


# ---------------------------------------------------------------------------
# the accounting itself
# ---------------------------------------------------------------------------

def test_out_of_vocab_ids_are_counted_with_their_first_position():
    gate = _load_gate()
    ids = [5, 6, 40132, 7, 127146, 8]
    d = gate.assert_prompt_in_vocab(ids, 32000)
    assert d["prompt_ids_out_of_vocab"] == 2
    assert d["prompt_first_out_of_vocab_index"] == 2
    assert d["prompt_first_out_of_vocab_id"] == 40132
    assert d["prompt_max_id"] == 127146
    assert d["prompt_vocab_size"] == 32000


def test_negative_ids_are_out_of_vocab_too():
    """< 0 is the other half of the same assert.

    `srcIndex < srcSelectDimSize` is unsigned, so a negative index wraps and
    trips it just as a large one does. Checking only the upper bound would leave
    the same abort reachable by a different route.
    """
    gate = _load_gate()
    d = gate.assert_prompt_in_vocab([1, -3, 2], 32000)
    assert d["prompt_ids_out_of_vocab"] == 1
    assert d["prompt_first_out_of_vocab_index"] == 1
    assert d["prompt_first_out_of_vocab_id"] == -3


def test_an_in_range_prompt_reports_zero():
    gate = _load_gate()
    d = gate.assert_prompt_in_vocab([0, 1, 31999], 32000)
    assert d["prompt_ids_out_of_vocab"] == 0
    assert d["prompt_first_out_of_vocab_index"] == ""
    assert d["prompt_first_out_of_vocab_id"] == ""


# ---------------------------------------------------------------------------
# run_candidate: record, and do not run the forward
# ---------------------------------------------------------------------------

class _Boom(Exception):
    pass


class _FakeInner:
    """Stands in for `model.model`. Records that it was called."""

    def __init__(self, log):
        self.log = log

    def __call__(self, *args, **kwargs):
        self.log.append("inner-forward")
        raise _Boom("the forward must not be attempted")


class _FakeModel:
    def __init__(self, vocab_size, log):
        self.config = type("C", (), {"vocab_size": vocab_size})()
        self.model = _FakeInner(log)
        self.device = "cpu"

    def eval(self):
        return self


def _drive_run_candidate(monkeypatch, tmp_path, prompt_ids, vocab_size,
                         log):
    """Run the REAL `run_candidate` with everything except the guard faked."""
    gate = _load_gate()

    class _Cfg:
        rope_scaling = None
        max_position_embeddings = 4096

    class _RASD:
        @staticmethod
        def _build_hf_config(*a, **k):
            return _Cfg()

    import src.models.rasd_inference as ri
    monkeypatch.setattr(ri, "RASDInference", _RASD, raising=False)

    class _AM:
        @staticmethod
        def from_pretrained(*a, **k):
            return _FakeModel(vocab_size, log)

    monkeypatch.setattr(gate, "AutoModelForCausalLM", _AM)
    monkeypatch.setattr(gate, "assert_effective_rope", lambda *a, **k: {})
    monkeypatch.setattr(gate, "gate_sample",
                        lambda meta, ctx, seed, tok: (list(prompt_ids), [1, 2, 3]))

    meta = tmp_path / "documents.json"
    meta.write_text(json.dumps({
        "tokenizer": "meta-llama/Llama-3.1-8B",
        "documents": [{"file": "x.dat", "length": 999999, "doc_id": "d"}],
    }))

    class Tok:
        name_or_path = "meta-llama/Llama-2-7b-hf"
        bos_token_id = 1
        pad_token_id = 0
        eos_token_id = 2

    cand = {"name": "B_llama2_yarn8_32k_correct",
            "target_model_name": "meta-llama/Llama-2-7b-hf",
            "rope_type": "yarn", "rope_factor": 8, "rope_anchor_base": 4096,
            "context_length": 32768, "seed": 42, "native_baseline": True}
    return gate.run_candidate(cand, Tok(), str(meta), tmp_path), gate


def test_an_out_of_vocab_prompt_is_recorded_and_the_forward_is_skipped(
        monkeypatch, tmp_path):
    """The exact 2026-10-08 case, driven through the real row-building code.

    Before the fix this candidate reached `embed_tokens` with id 40132, the
    context died, and the `finally`'s `empty_cache()` raised -- so the stage
    produced no CSV and the incident had nothing to diagnose. The assertion is
    on the ROW, not on an exception, because "recorded" is the whole contract.
    """
    log = []
    prompt = [1] + [100, 200, 40132] + [7] * 20   # one id past Llama-2's vocab
    row, gate = _drive_run_candidate(monkeypatch, tmp_path, prompt, 32000, log)

    assert "inner-forward" not in log, (
        "the forward ran on a prompt the embedding cannot index; on CUDA that "
        "aborts the context and takes every later candidate with it")
    assert row["status"] == "invalid_prompt_tokens"
    assert row["prompt_ids_out_of_vocab"] == 1
    assert row["prompt_first_out_of_vocab_index"] == 3
    assert row["prompt_first_out_of_vocab_id"] == 40132
    assert row["prompt_vocab_size"] == 32000
    assert row["pool_tokenizer"] == "meta-llama/Llama-3.1-8B"
    assert row["candidate_tokenizer"] == "meta-llama/Llama-2-7b-hf"
    # Not measured is not zero. A 0.0 would be read as a measurement.
    assert row["ppl_continuation"] == ""
    assert "indexSelectLargeIndex" in row["error"]


def test_an_in_vocab_prompt_is_measured(monkeypatch, tmp_path):
    """The guard must not block the candidates that are fine.

    A check that reported out-of-vocab ids for an in-range prompt would turn
    every Llama-3.1 candidate into an unmeasured row -- i.e. it would disable
    the gate while looking like it was protecting it.
    """
    log = []
    row, gate = _drive_run_candidate(monkeypatch, tmp_path, [1, 100, 200],
                                     32000, log)
    assert row.get("prompt_ids_out_of_vocab") == 0
    assert row.get("status") != "invalid_prompt_tokens"
    assert log == ["inner-forward"], (
        "an in-vocab prompt must reach the forward")


# ---------------------------------------------------------------------------
# re-tokenization: does the model see its OWN ids?
# ---------------------------------------------------------------------------

def _write_pool(tmp_path, ids, tokenizer_name):
    meta = tmp_path / "documents.json"
    dat = tmp_path / "doc.dat"
    np.asarray(ids, dtype="int32").tofile(dat)
    meta.write_text(json.dumps({
        "tokenizer": tokenizer_name,
        "documents": [{"file": str(dat), "length": len(ids), "doc_id": "d"}],
    }))
    return str(meta)


def test_the_window_is_unchanged_when_the_tokenizers_agree(tmp_path):
    """The Llama-3.1 path must be byte-identical to before the fix.

    Every control already recorded for Llama-3.1 was measured on these ids. A
    'fix' that silently re-tokenized them too would invalidate those numbers
    while looking like a no-op, so the claim is exactly "same result as the
    path that had no tokenizer argument at all".
    """
    gate = _load_gate()
    ids = [i % 50000 + 1 for i in range(6000)]     # varied, all in range
    meta = _write_pool(tmp_path, ids, "meta-llama/Llama-3.1-8B")

    class SameTok:
        name_or_path = "meta-llama/Llama-3.1-8B"

        def decode(self, ids):                     # pragma: no cover
            raise AssertionError(
                "the pool's own tokenizer must not be re-encoded: these ids "
                "were already produced by it, and re-encoding them would move "
                "the numbers the Llama-3.1 controls were measured on")

    before = gate.load_pg19_window(meta, 2048, 42)
    after = gate.load_pg19_window(meta, 2048, 42, tokenizer=SameTok())
    assert before == after, (
        "matching tokenizers changed the window; the previously recorded "
        "Llama-3.1 control numbers would no longer be comparable")


def test_a_different_tokenizer_re_tokenizes_the_window(tmp_path):
    """The candidate's ids, not the pool's, reach the candidate.

    This is the fix for the abort: a pool tokenized by Llama-3.1 contains ids
    Llama-2-7B's 32000-row embedding cannot index.
    """
    gate = _load_gate()
    # EVERY id is past Llama-2-7B's 32000-row embedding, so the fixture
    # exercises the abort at whatever offset the seed happens to pick.
    pool_ids = [i % 90000 + 32000 for i in range(6000)]
    meta = _write_pool(tmp_path, pool_ids, "meta-llama/Llama-3.1-8B")
    raw = gate.load_pg19_window(meta, 2048, 42)          # what the old code did
    assert max(raw[0]) >= 32000, (
        "the fixture must actually exercise the failure: the pool ids have to "
        "exceed Llama-2-7B's embedding")

    class PoolTok:
        name_or_path = "meta-llama/Llama-3.1-8B"

        def decode(self, ids):
            return " " + " ".join(str(i) for i in ids)

    class CandTok:
        name_or_path = "meta-llama/Llama-2-7b-hf"

        def decode(self, ids):                        # pragma: no cover
            raise AssertionError("decoding must use the POOL tokenizer")

        def encode(self, text, add_special_tokens=False):
            return [min(int(t), 31999) for t in text.split()[1:]]

    real = gate.AutoTokenizer
    gate.AutoTokenizer = type("A", (), {
        "from_pretrained": staticmethod(lambda n: PoolTok())})
    try:
        prompt, cont = gate.load_pg19_window(
            meta, 2048, 42, tokenizer=CandTok())
    finally:
        gate.AutoTokenizer = real

    assert max(prompt) < 32000 and max(cont) < 32000, (
        "ids outside the candidate's embedding reached the window; on CUDA "
        "these abort the context in indexSelectLargeIndex")
    # Budget: prompt_len = ctx - max(CONTINUATION_TOKENS, GENERATE_TOKENS),
    # and no BOS was requested, so 2048 - 1024.
    assert len(prompt) == 1024
    assert len(cont) == 1024


def test_the_re_tokenization_converges_across_a_token_ratio_beyond_4096(
        tmp_path):
    """A candidate needing a much LARGER token count must still converge.

    Llama-2's 32k vocabulary needs ~1.13x more tokens than Llama-3.1's 128k one
    for the same text, so reaching a 32k-token Llama-2 window from a Llama-3.1
    pool needs a shift of thousands of ids. Two shapes of failure are pinned
    here, both measured on the real pod:

      * an absolute cap of 4096 on the correction rejected exactly the case this
        function exists for and silently returned None -- which the caller
        treats as "keep the pool ids", i.e. it puts the abort straight back;
      * correcting one estimate at a time does not converge at this ratio. The
        map `L <- L + span - f(L)` has contraction factor -0.13, so it
        oscillates: measured 32767 -> 28403 -> 29001 -> 28928 -> ... with no
        exact hit in eight probes. Bisection has no such failure mode.
    """
    gate = _load_gate()
    span = 8192
    pool_ids = [i % 90000 + 32000 for i in range(40000)]
    meta = _write_pool(tmp_path, pool_ids, "meta-llama/Llama-3.1-8B")

    class PoolTok:
        name_or_path = "meta-llama/Llama-3.1-8B"

        def decode(self, ids):
            return " " + " ".join(str(i) for i in ids)

    class RatioTok:
        """~1.125 candidate tokens per pool id, like Llama-2 vs Llama-3.1."""

        name_or_path = "meta-llama/Llama-2-7b-hf"

        def encode(self, text, add_special_tokens=False):
            out = []
            for i, t in enumerate(text.split()[1:]):
                out.append(int(t))
                if i % 8 == 0:
                    out.append(int(t))
            return [min(x, 31999) for x in out]

    arr = np.asarray(pool_ids, dtype="int32")
    res = gate._tokenize_window_for(RatioTok(), PoolTok(), arr, 0, span)
    assert res is not None, (
        "the re-tokenization gave up, so the pool's out-of-range ids would be "
        "used and the device context would abort")
    assert len(res) == span, (
        "the window must be EXACTLY the token budget; a longer one measures "
        "extrapolation past the candidate's own context")
    assert max(res) < 32000, "ids outside the candidate's embedding survived"


# ---------------------------------------------------------------------------
# telling a dead context apart from a row that merely failed
# ---------------------------------------------------------------------------

def test_a_generation_oom_does_not_discard_the_measured_perplexity(
        monkeypatch, tmp_path):
    """The perplexity is measured BEFORE generation, and must survive it.

    On a 40GB card the generation phase runs out of memory where the forward
    succeeded (measured: `CUDA OOM ... 223 MiB free` at 32k). Overwriting the
    row's perplexity with the generation failure would throw away the only
    number the stage exists to produce.
    """
    gate = _load_gate()
    import torch
    import src.analysis.target_quality as tq

    log = []

    class Model:
        device = "cpu"
        config = type("C", (), {"vocab_size": 32000})()

        class _Inner:
            def __call__(self, *a, **k):
                log.append("forward")
                return type("H", (), {"last_hidden_state": torch.zeros(1, 5, 4)})()

        model = _Inner()

        def eval(self):
            return self

        def lm_head(self, x):
            return torch.zeros(x.shape[0], x.shape[1], 8)

        def generate(self, *a, **k):
            log.append("generate")
            raise torch.cuda.OutOfMemoryError("CUDA out of memory")

    monkeypatch.setattr(tq, "continuation_nll",
                        lambda *a, **k: (2.0, 1, torch.zeros(1, 5, 4)))
    monkeypatch.setattr(gate, "continuation_perplexity",
                        lambda m, p, c: (7.0155, None))

    class _RASD:
        @staticmethod
        def _build_hf_config(*a, **k):
            return type("Cfg", (), {
                "rope_scaling": None, "max_position_embeddings": 4096})()

    import src.models.rasd_inference as ri
    monkeypatch.setattr(ri, "RASDInference", _RASD, raising=False)

    class _AM:
        @staticmethod
        def from_pretrained(*a, **k):
            return Model()

    monkeypatch.setattr(gate, "AutoModelForCausalLM", _AM)
    monkeypatch.setattr(gate, "assert_effective_rope", lambda *a, **k: {})
    monkeypatch.setattr(gate, "gate_sample",
                        lambda meta, ctx, seed, tok: ([1, 2, 3], [4, 5, 6]))

    meta = tmp_path / "documents.json"
    meta.write_text(json.dumps({
        "tokenizer": "meta-llama/Llama-2-7b-hf",
        "documents": [{"file": "x.dat", "length": 999999, "doc_id": "d"}]}))

    class Tok:
        name_or_path = "meta-llama/Llama-2-7b-hf"
        bos_token_id = 1
        pad_token_id = 0
        eos_token_id = 2

    row = gate.run_candidate(
        {"name": "B_llama2_yarn8_32k_correct",
         "target_model_name": "meta-llama/Llama-2-7b-hf",
         "context_length": 32768, "seed": 42}, Tok(), str(meta), tmp_path)

    assert row["status"] == "oom", f"got {row['status']}: {row.get('error')}"
    assert row["ppl_continuation"] == 7.0155, (
        "the measured perplexity was discarded by the generation failure")
    assert not gate._cuda_died(row), (
        "a per-row OOM is not a dead context")


def test_a_per_row_oom_is_not_a_dead_context():
    """`CUDA OOM` is a resource limit on ONE row, not a poisoned context.

    The first version matched the bare substring "CUDA", so every OOM row was
    reported as "the context is gone, no later row means anything" -- which
    would discard a run whose remaining candidates were perfectly measurable.
    """
    gate = _load_gate()
    assert gate._cuda_died(
        {"status": "oom", "error": "CUDA OOM: tried to allocate 12 GiB"}) is False


def test_the_guards_own_message_is_not_a_dead_context():
    """An out-of-vocab prompt is a RECORDED outcome, and the context is alive.

    The guard's message explains that a forward would abort the device context;
    matching the word "CUDA" in it made the gate announce that the context had
    died, on the very path that exists to prevent the death.
    """
    gate = _load_gate()
    row = {"prompt_ids_out_of_vocab": 2391,
           "error": ("2391 of 31744 prompt ids are outside "
                     "meta-llama/Llama-2-7b-hf's embedding (vocab_size=32000); "
                     "the forward is not attempted: it would abort the device "
                     "context in indexSelectLargeIndex.")}
    assert gate._cuda_died(row) is False


def test_a_device_side_assert_is_a_dead_context():
    gate = _load_gate()
    assert gate._cuda_died(
        {"error": "RuntimeError: CUDA error: device-side assert triggered"}) is True
    assert gate._cuda_died({"cuda_poisoned": True}) is True


# ---------------------------------------------------------------------------
# the 1x repro must not be able to reach the 8x campaign
# ---------------------------------------------------------------------------

def test_no_campaign_path_reads_the_1x_repro():
    """The repro and the campaign must not be connected.

    The campaign fits 8 GPUs. A repro that quietly carried a 1x-specific setting
    -- a smaller context, one device, a different dtype -- into the campaign
    would make its numbers unrepresentative of the machine it runs on, and the
    damage would be invisible because it would look like a repro convenience.
    The only shared change permitted is the root-cause fix, which is
    device-count independent; this pins that the repro is a dead end.
    """
    repro = REPO / "scripts" / "mlsys_gate_repro_1x.sh"
    assert repro.exists(), "the 1x repro script is missing"

    for rel in ("scripts/mlsys_manifest.sh", "scripts/mlsys_watch_and_run.sh"):
        assert "mlsys_gate_repro_1x" not in (REPO / rel).read_text(), (
            f"{rel} references the 1x repro; the campaign would inherit "
            f"whatever fitting decision it carries")

    for p in sorted((REPO / "configs").glob("mlsys_*")):
        assert "repro_1x" not in p.read_text(), (
            f"{p.name} references the 1x repro")


def test_the_gate_still_places_every_candidate_on_one_device():
    """The gate's device map must not have acquired a 1x-shaped special case.

    `_device_map()` returning `{"": 0}` (or cpu) is what makes a gate row fit a
    single rank. A change here -- sharding the gate, or keying the placement on
    a device count -- would silently alter what the gate measures.
    """
    gate = _load_gate()
    src = GATE.read_text()
    assert 'return {"": 0} if torch.cuda.is_available() else {"": "cpu"}' in src, (
        "the gate's device map changed; the campaign fits 8 GPUs and the gate "
        "is a single-device measurement")


# ---------------------------------------------------------------------------
# non-finite logits
# ---------------------------------------------------------------------------

def test_non_finite_logits_are_recorded_with_their_first_position(monkeypatch):
    """A NaN logit is the configuration's finding, not a gate malfunction.

    The f2 construction is a KNOWN-BROKEN negative control, so "the logits went
    non-finite" is evidence about it. Crashing would lose the evidence and take
    the later candidates down too.
    """
    gate = _load_gate()
    import torch

    class Model:
        hidden = 4
        vocab = 8
        device = "cpu"

        def lm_head(self, x):
            out = torch.zeros(x.shape[0], x.shape[1], self.vocab)
            out[:, 2, 0] = float("nan")     # non-finite at the 3rd position
            return out

    import src.analysis.target_quality as tq
    monkeypatch.setattr(tq, "continuation_nll",
                        lambda *a, **k: (1.0, 1, torch.zeros(1, 5, 4)))
    ppl, bad = gate.continuation_perplexity(Model(), [1], [2, 3])
    assert bad == 2, f"expected the non-finite position 2, got {bad}"
    assert ppl == pytest.approx(float(torch.e ** torch.tensor(1.0, dtype=torch.float64)))


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))

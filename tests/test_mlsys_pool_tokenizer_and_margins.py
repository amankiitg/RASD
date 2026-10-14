"""The tokenizer-vs-pool assertion, and the M3 cross-margin logging.

FIXTURES ARE REAL. `data/processed/pg19_docs` was built with the Llama-3.1
tokenizer and `data/processed/pg19` with Llama-2's, so the two are a genuine
positive and negative control pair for the trap. The engine tests use a stand-in
for `torch.topk` so they run without CUDA or a model.

The trap, restated because it is the reason these exist: pointing a Llama-3.1
candidate at the Llama-2 pool does NOT raise. Every Llama-2 id is inside the
Llama-3.1 vocabulary, so the range check passes, the forward runs, and a perplexity
comes back for text that is not what the model reads. The memorization control would
have returned a FALSE "memorized" verdict in the direction that flatters the story.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

# Set BEFORE transformers is imported, and for a measured reason: this module
# loads real tokenizers, and tokenizers initialises a thread pool on first use.
# A later test that forks then inherits that pool and logs "the current process
# just got forked, after parallelism has already been used"; in a FULL suite run
# that coincided with one flaky failure in tests/test_mlsys_watcher_guards.py (it
# passes in isolation and in every pairing tried). Disabling parallelism here
# costs this module nothing -- it tokenizes a few thousand ids.
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import pytest

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from src.analysis.pool_tokenizer import (  # noqa: E402
    ROUNDTRIP_MIN, pool_check, pool_problem, tokenizer_family,
)

PG19_DOCS = REPO / "data" / "processed" / "pg19_docs" / "documents.json"
PG19_LLAMA2 = (REPO / "data" / "processed" / "pg19"
               / "pg19_validation_metadata.json")


def _tokenizer(name="meta-llama/Llama-3.1-8B"):
    transformers = pytest.importorskip("transformers")
    try:
        return transformers.AutoTokenizer.from_pretrained(
            name, local_files_only=True)
    except Exception as e:                                  # noqa: BLE001
        pytest.skip(f"tokenizer {name} not cached locally: {e}")


def _pool_ids(meta_path: Path, n: int = 4096):
    import numpy as np
    meta = json.loads(meta_path.read_text())
    docs = meta.get("documents") or meta.get("chunks")
    if not docs:
        pytest.skip(f"{meta_path} has no documents")
    d = sorted(docs, key=lambda x: -x["length"])[0]
    arr = np.memmap(d["file"], dtype="int32", mode="r")
    return np.asarray(arr[:min(n, int(d["length"]))]).astype(int).tolist()


# --- the family helper --------------------------------------------------------

@pytest.mark.parametrize("name,expected", [
    ("meta-llama/Llama-3.1-8B", "llama3"),
    ("meta-llama/Llama-3.2-1B", "llama3"),
    ("meta-llama/Llama-2-7b-hf", "llama2"),
    ("princeton-nlp/Sheared-LLaMA-1.3B", "llama1"),
])
def test_family_matches_the_pairing_the_arms_need(name, expected):
    assert tokenizer_family(name) == expected


def test_the_two_native_models_are_the_same_family():
    """Phase 0 item 2: the native arms depend on Llama-3.1-8B and Llama-3.2-1B
    being interchangeable, so the family check must not separate them."""
    assert (tokenizer_family("meta-llama/Llama-3.1-8B")
            == tokenizer_family("meta-llama/Llama-3.2-1B"))


def test_the_two_pg19_pools_are_different_families():
    """If this ever stops holding, the trap has changed shape and the detector
    below is testing nothing."""
    assert (tokenizer_family("meta-llama/Llama-3.1-8B")
            != tokenizer_family("meta-llama/Llama-2-7b-hf"))


# --- the round trip -----------------------------------------------------------

def test_a_matching_pool_round_trips_exactly():
    if not PG19_DOCS.exists():
        pytest.skip("pg19_docs is not staged")
    tok = _tokenizer()
    chk = pool_check(tok, _pool_ids(PG19_DOCS), "meta-llama/Llama-3.1-8B",
                     "meta-llama/Llama-3.1-8B")
    assert chk["pool_roundtrip_share"] > 0.99, chk
    assert chk["pool_tokenizer_match"] is True
    assert pool_problem(chk) == ""


def test_every_1m_document_round_trips():
    """THE POSITIVE CONTROL THAT CATCHES THE BUG THE FIRST ONE MISSED.

    `decode()` defaults to clean_up_tokenization_spaces=True, which rewrites
    punctuation spacing; re-encoding THAT text re-segments it and the check
    reported a tokenizer mismatch on pools tokenized perfectly. Measured
    2026-10-10 on data/processed/pg19_1m: with the default, four of these six
    documents "failed" (1.3%, 3.6%, 7.9%, 73.2% match) and with it off all six
    match 100.0%. A single-document control passes either way for the Bible,
    which is exactly how such a bug survives a test suite.
    """
    meta_path = REPO / "data" / "processed" / "pg19_1m" / "documents.json"
    if not meta_path.exists():
        pytest.skip("the 1M pool is not built")
    tok = _tokenizer()
    meta = json.loads(meta_path.read_text())
    worst, worst_id = 1.0, ""
    for d in meta["documents"]:
        import numpy as np
        arr = np.memmap(d["file"], dtype="int32", mode="r")
        ids = np.asarray(arr[:16384]).astype(int).tolist()
        chk = pool_check(tok, ids, meta.get("tokenizer"),
                         "meta-llama/Llama-3.1-8B")
        if chk["pool_roundtrip_share"] < worst:
            worst, worst_id = chk["pool_roundtrip_share"], d["doc_id"]
        assert chk["pool_roundtrip_share"] > 0.99, (
            f"{d['doc_id']} round-trips at {chk['pool_roundtrip_share']} with a "
            f"matching tokenizer: {pool_problem(chk)}")
    assert worst > 0.99, f"worst {worst_id} at {worst}"


def test_the_llama2_pool_is_caught_by_the_round_trip():
    """The negative control: a Llama-2-tokenized pool read by a Llama-3.1
    tokenizer. Its metadata carries NO tokenizer name, so only the round trip can
    catch it -- which is the whole reason the check is on the data."""
    if not PG19_LLAMA2.exists():
        pytest.skip("the llama2 pool is not staged")
    tok = _tokenizer()
    chk = pool_check(tok, _pool_ids(PG19_LLAMA2), None, "meta-llama/Llama-3.1-8B")
    assert chk["pool_roundtrip_share"] < 0.5, chk
    assert chk["pool_tokenizer_match"] == ""
    problem = pool_problem(chk)
    assert "round-trip" in problem and "different tokenizer" in problem


def test_a_name_mismatch_is_caught_even_when_the_data_looks_fine():
    """Two independent signals: the metadata name, and the data itself. A pool that
    declares Llama-2 but round-trips is still refused, because the declared
    tokenizer is what the pairing story rests on."""
    if not PG19_DOCS.exists():
        pytest.skip("pg19_docs is not staged")
    tok = _tokenizer()
    chk = pool_check(tok, _pool_ids(PG19_DOCS), "meta-llama/Llama-2-7b-hf",
                     "meta-llama/Llama-3.1-8B")
    assert chk["pool_roundtrip_share"] > 0.99
    assert chk["pool_tokenizer_match"] is False
    assert "different family" in pool_problem(chk)


def test_the_threshold_sits_between_the_two_measured_populations():
    assert 0.5 < ROUNDTRIP_MIN < 0.99


# --- M3 cross margins ---------------------------------------------------------

def test_cross_margin_classifies_a_decided_divergence():
    """The case the existing scalar gaps cannot classify: `spec=972 target=1114`
    with each arm favouring its own token by 6.875 and 7.5. fp4 quantises logits in
    0.0625 steps, so this is not a rounding artefact."""
    from src.analysis.losslessness import cross_margins
    spec_topk = [[(972, 20.0), (1114, 13.125)]]
    target_topk = [[(1114, 21.0), (972, 13.5)]]
    out = cross_margins(spec_topk, target_topk, [972], [1114], 0)
    assert out["divergence_class"] == "decided", out
    assert out["cross_margin_spec"] == pytest.approx(6.875)
    assert out["cross_margin_target"] == pytest.approx(7.5)


def test_cross_margin_classifies_a_tie():
    from src.analysis.losslessness import cross_margins
    spec_topk = [[(1, 10.0), (2, 9.95)]]
    target_topk = [[(2, 10.0), (1, 9.96)]]
    out = cross_margins(spec_topk, target_topk, [1], [2], 0)
    assert out["divergence_class"] == "tie", out
    assert abs(out["cross_margin_spec"]) < 0.1


def test_cross_margin_is_undetermined_without_topk():
    """The honest answer when the sidecar predates --dump-logits-topk. Returning a
    classification here would be inventing evidence."""
    from src.analysis.losslessness import cross_margins
    out = cross_margins(None, None, [1], [2], 0)
    assert out["divergence_class"] == "undetermined"
    assert out["cross_margin_spec"] == ""


def test_cross_margin_reports_when_the_competitor_is_outside_topk():
    from src.analysis.losslessness import cross_margins
    spec_topk = [[(1, 10.0), (3, 9.0)]]
    target_topk = [[(2, 10.0), (3, 9.0)]]
    out = cross_margins(spec_topk, target_topk, [1], [2], 0)
    assert out["divergence_class"] == "undetermined", out


def test_the_engine_records_topk_aligned_with_the_ids():
    """The dump has to be a parallel array to `generated`, for the same reason
    `token_gaps` does: the divergence lookup is positional."""
    import torch
    from src.models.rasd_inference import _round_emitted_topk, _step_topk
    logits = torch.zeros(1, 6, 10)
    logits[0, :, 3] = 5.0
    logits[0, :, 7] = 4.0
    step = _step_topk(logits[:, 0, :], 2)
    assert step[0][0][0] == 3 and step[0][1][0] == 7, step
    # n_emit=2 accepted then a bonus drawn from position n_acc=2
    out = _round_emitted_topk(logits, n_emit=2, n_acc=2, with_bonus=True, k=2)
    assert len(out) == 3, "one entry per emitted token"
    numpy_less = _round_emitted_topk(logits, n_emit=2, n_acc=2,
                                     with_bonus=False, k=2)
    assert len(numpy_less) == 2, "a truncated round emits no bonus"


def test_topk_dump_is_off_by_default():
    """`dump_logits_topk` defaults to 0 so the existing sidecar format and the M3
    replay are unchanged unless a run asks for the extra data."""
    from src.models.rasd_inference import RASDConfig
    assert RASDConfig(seed=1).dump_logits_topk == 0


def test_the_sidecar_schema_carries_the_topk_dump():
    """The engine computing a metric is not the same as the sidecar carrying it.

    `write_generated_tokens_sidecar` is handed an explicit dict literal, so a
    metric the engine adds and that literal omits is dropped in silence. On the
    2026-10-10 pilot the run was configured with dump_logits_topk=6 and the
    sidecar came back with no `token_topk`, so the cross margin read
    "undetermined" for the very divergence the run existed to explain -- and the
    field was absent from the CSV too, so nothing looked broken.
    """
    src = (REPO / "run_experiment.py").read_text()
    # The CALL, not the definition: the definition lives in another module and a
    # first-occurrence search silently inspected the wrong text.
    i = src.index('write_generated_tokens_sidecar(\n                    output_csv')
    block = src[i:i + 3000]
    # `gen_ids` is the third POSITIONAL argument, so it is not a quoted key --
    # an assertion on the string "generated_token_ids" here failed against
    # correct code. The dict keys are the ones that have to be checked.
    assert "gen_ids" in block.split("{")[0], "the call must pass the token ids"
    for field in ("token_gaps", "token_topk"):
        assert f'"{field}"' in block, (
            f"{field} is not in the sidecar schema; the engine can compute it "
            f"and it will still never reach the file")

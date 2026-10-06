"""The sequence identity, and the prompt every arm of a rung must share.

Two properties, both of which were silently false:

  * `sequence_tokens` must be the sequence the engine ACTUALLY built --
    `prompt_tokens + 1 (BOS) + tokens_generated` -- for every arm. It was
    recorded from the document plan (`len(prompt) + 1 + the rung's generation
    length`), so a short baseline claimed a sequence length it never produced and
    an early-stopping run claimed the full one.

  * the arms of a rung must be given the SAME prompt, or the losslessness prefix
    check compares two different contexts and `require_same_request` rightly
    refuses every pair. The prompt window comes from `prompt_gen_tokens` (the
    rung), not from the arm's own cap.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

import run_experiment as rex          # noqa: E402

CHECKER = REPO / "scripts" / "mlsys_row_identity_check.py"


def _load_checker():
    spec = importlib.util.spec_from_file_location("_row_identity", CHECKER)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ---------------------------------------------------------------------------
# the checker that runs on every stage's rows
# ---------------------------------------------------------------------------

def _csv(tmp_path, rows) -> Path:
    import csv
    p = tmp_path / "stage.csv"
    fields = ["run_id", "status", "prompt_tokens", "tokens_generated",
              "sequence_tokens"]
    with p.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    return p


def _row(rid="r1", prompt=131007, gen=64, seq=None, status="ok"):
    return {"run_id": rid, "status": status, "prompt_tokens": prompt,
            "tokens_generated": gen,
            "sequence_tokens": prompt + 1 + gen if seq is None else seq}


def test_a_row_that_satisfies_the_identity_passes(tmp_path):
    mod = _load_checker()
    problems, checked = mod.check(_csv(tmp_path, [_row()]))
    assert problems == [] and checked == 1


def test_a_row_short_by_the_seed_token_is_caught(tmp_path):
    """The old off-by-one: `generated + bonuses == cap`, missing the seed."""
    mod = _load_checker()
    problems, _ = mod.check(_csv(tmp_path, [_row(seq=131007 + 64)]))
    assert problems and "sequence_tokens" in problems[0]


def test_a_short_arm_must_report_what_it_actually_built(tmp_path):
    """A 128-token generation into the rung's prompt is NOT the rung length."""
    mod = _load_checker()
    # what the document plan used to claim: prompt + 1 + the rung's 1024
    wrong = _row(prompt=131072 - 1024 - 1, gen=128,
                 seq=131072 - 1024 - 1 + 1 + 1024)
    problems, _ = mod.check(_csv(tmp_path, [wrong]))
    assert problems, "a short arm claiming the rung's sequence length passed"
    right = _row(prompt=131072 - 1024 - 1, gen=128)
    assert mod.check(_csv(tmp_path, [right]))[0] == []


def test_a_row_missing_the_fields_cannot_be_checked(tmp_path):
    mod = _load_checker()
    problems, _ = mod.check(_csv(tmp_path, [{"run_id": "r", "status": "ok",
                                             "prompt_tokens": "",
                                             "tokens_generated": "5",
                                             "sequence_tokens": "6"}]))
    assert problems and "cannot check" in problems[0]


def test_error_rows_are_not_checked_but_ok_rows_are(tmp_path):
    mod = _load_checker()
    problems, checked = mod.check(_csv(tmp_path, [
        _row("bad", status="error", seq=1), _row("good")]))
    assert problems == [] and checked == 1


# ---------------------------------------------------------------------------
# the prompt every arm of a rung shares
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def tiny_tokenizer():
    pytest.importorskip("transformers")
    from transformers import AutoTokenizer
    try:
        tok = AutoTokenizer.from_pretrained(
            "hf-internal-testing/tiny-random-LlamaForCausalLM")
    except Exception as e:                      # noqa: BLE001
        pytest.skip(f"no tokenizer available offline: {e}")
    return tok


@pytest.fixture()
def pg19_doc(tmp_path):
    """One synthetic 'book' of 200k tokens plus its metadata."""
    rng = np.random.default_rng(0)
    ids = rng.integers(100, 30000, size=200_000).astype("int32")
    mem = tmp_path / "doc0.memmap"
    ids.tofile(mem)
    meta = tmp_path / "documents.json"
    meta.write_text(
        '{"documents": [{"doc_id": "doc0", "title": "t", "url": "", '
        f'"file": "{mem}", "length": {ids.size}, "tokens": {ids.size}}}]}}')
    return str(meta)


def _arm(doc_id, ctx, max_new, prompt_gen=None):
    run = {"doc_id": doc_id, "context_length": ctx, "max_new_tokens": max_new,
           "seed": 42, "run_id": f"{doc_id}_{ctx}_{max_new}"}
    if prompt_gen is not None:
        run["prompt_gen_tokens"] = prompt_gen
    return run


def test_spec_and_short_baseline_share_the_prompt(tiny_tokenizer, pg19_doc):
    """The property the losslessness prefix check depends on."""
    spec = _arm("doc0", 32768, 1024)
    short = _arm("doc0", 32768, 128, prompt_gen=1024)
    _, _, spec_prov = rex._build_pg19_document_prompt(
        pg19_doc, 32768, "doc0", tiny_tokenizer,
        gen_tokens=rex._prompt_gen_tokens(spec))
    _, _, short_prov = rex._build_pg19_document_prompt(
        pg19_doc, 32768, "doc0", tiny_tokenizer,
        gen_tokens=rex._prompt_gen_tokens(short))
    assert short_prov["prompt_tokens"] == spec_prov["prompt_tokens"]
    assert short_prov["prompt_sha256"] == spec_prov["prompt_sha256"]
    # ... and the short arm's own sequence is SHORTER than the rung, which is
    # what the row-identity check must accept.
    assert (short_prov["prompt_tokens"] + 1 + 128
            < short_prov["sequence_tokens"])


def test_without_the_rung_length_the_prompts_diverge(tiny_tokenizer, pg19_doc):
    """The pre-fix behaviour, asserted so the fix cannot silently revert."""
    spec = _arm("doc0", 32768, 1024)
    wrong = _arm("doc0", 32768, 128)            # no prompt_gen_tokens
    _, _, a = rex._build_pg19_document_prompt(
        pg19_doc, 32768, "doc0", tiny_tokenizer,
        gen_tokens=rex._prompt_gen_tokens(spec))
    _, _, b = rex._build_pg19_document_prompt(
        pg19_doc, 32768, "doc0", tiny_tokenizer,
        gen_tokens=rex._prompt_gen_tokens(wrong))
    assert a["prompt_sha256"] != b["prompt_sha256"], (
        "without prompt_gen_tokens the two arms agree, so this test proves "
        "nothing about the fix")


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))

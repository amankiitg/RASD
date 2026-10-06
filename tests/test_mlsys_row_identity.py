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

def _csv(tmp_path, rows, sidecars: dict | None = None) -> Path:
    """A stage CSV plus the token sidecars the identity check reads.

    The emitted count is taken from the SIDECAR, so a fixture without one cannot
    be checked at all -- which is itself one of the cases below.
    """
    import csv, json
    p = tmp_path / "stage.csv"
    fields = ["run_id", "status", "prompt_tokens", "tokens_generated",
              "sequence_tokens"]
    with p.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    for rid, ids in (sidecars or {}).items():
        d = tmp_path / "tokens"
        d.mkdir(exist_ok=True)
        (d / f"{rid}.json").write_text(json.dumps(
            {"run_id": rid, "generated_token_ids": list(ids)}))
    return p


def _row(rid="r1", prompt=131007, gen=64, seq=None, status="ok"):
    return {"run_id": rid, "status": status, "prompt_tokens": prompt,
            "tokens_generated": gen,
            "sequence_tokens": prompt + 1 + gen if seq is None else seq}


def _side(rid="r1", gen=64, extra=0):
    return {rid: list(range(gen + extra))}


def test_a_row_that_satisfies_the_identity_passes(tmp_path):
    mod = _load_checker()
    problems, checked = mod.check(
        _csv(tmp_path, [_row()], _side()), tmp_path / "tokens")
    assert problems == [] and checked == 1


def test_an_engine_sequence_one_token_longer_is_rejected(tmp_path):
    """The engine's own count is what makes this a check rather than a sum.

    With `sequence_tokens` recorded as `prompt + 1 + generated` by the runner,
    and asserted here as the same expression, nothing could ever fail. The engine
    now measures it off the tensor it holds, so a generation loop that kept one
    position too many -- a BOS counted twice, a round that committed one token
    past the cap -- shows up as exactly this divergence.
    """
    mod = _load_checker()
    problems, _ = mod.check(
        _csv(tmp_path, [_row(seq=131007 + 1 + 64 + 1)], _side()),
        tmp_path / "tokens")
    assert problems, "a sequence one token longer than prompt + BOS + emitted passed"
    assert "sequence_tokens" in problems[0] and "+1" in problems[0]


def test_a_row_short_by_the_seed_token_is_caught(tmp_path):
    """The old off-by-one: `generated + bonuses == cap`, missing the seed."""
    mod = _load_checker()
    problems, _ = mod.check(_csv(tmp_path, [_row(seq=131007 + 64)], _side()),
                            tmp_path / "tokens")
    assert problems and "sequence_tokens" in problems[0]


def test_a_short_arm_must_report_what_it_actually_built(tmp_path):
    """A 128-token generation into the rung's prompt is NOT the rung length."""
    mod = _load_checker()
    # what the document plan used to claim: prompt + 1 + the rung's 1024
    wrong = _row(prompt=131072 - 1024 - 1, gen=128,
                 seq=131072 - 1024 - 1 + 1 + 1024)
    problems, _ = mod.check(_csv(tmp_path, [wrong], _side(gen=128)),
                            tmp_path / "tokens")
    assert problems, "a short arm claiming the rung's sequence length passed"
    right = _row(prompt=131072 - 1024 - 1, gen=128)
    assert mod.check(_csv(tmp_path, [right], _side(gen=128)),
                     tmp_path / "tokens")[0] == []


def test_the_csv_count_must_agree_with_the_sidecar(tmp_path):
    """`tokens_generated` is checked against the ids that were actually saved."""
    mod = _load_checker()
    problems, _ = mod.check(
        _csv(tmp_path, [_row(gen=64)], _side(gen=65)), tmp_path / "tokens")
    assert problems and "sidecar holds 65" in problems[0]


def test_a_row_without_a_sidecar_cannot_be_checked(tmp_path):
    mod = _load_checker()
    problems, _ = mod.check(_csv(tmp_path, [_row()]), tmp_path / "tokens")
    assert problems and "no token sidecar" in problems[0]


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
        _row("bad", status="error", seq=1), _row("good"),
    ], {"good": list(range(64))}), tmp_path / "tokens")
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


# ---------------------------------------------------------------------------
# where the number comes from
# ---------------------------------------------------------------------------

RUNNER = (REPO / "run_experiment.py").read_text()
ENGINE = (REPO / "src" / "models" / "rasd_inference.py").read_text()
STUB = (REPO / "scripts" / "rehearsal" / "stub_run_experiment.py").read_text()


def test_the_engine_measures_the_sequence_off_the_tensor():
    """`int(generated_ids.shape[1])` in BOTH generation paths.

    The sequence length is the length of the final sequence the loop holds; if
    it is reconstructed from the row's other fields the identity check becomes a
    restatement and loses all power.
    """
    assert ENGINE.count('sequence_len = int(generated_ids.shape[1])') == 2, (
        "the spec path and the target-only path must both measure it")
    assert ENGINE.count('"sequence_tokens":   sequence_len,') == 2


def test_the_runner_takes_the_engine_measurement_not_a_sum():
    assert 'seq = metrics.get("sequence_tokens")' in RUNNER, (
        "the row does not take the engine's measurement")
    assert 'row["sequence_tokens"] = int(seq)' in RUNNER
    assert 'row["sequence_tokens"] = (int(row["prompt_tokens"]) + 1' not in RUNNER, (
        "the row is back to computing the sequence it is supposed to be "
        "checked against, which makes mlsys_row_identity_check a tautology")


def test_the_stub_measures_through_the_engine_side_object():
    """The stub must measure an object, not add the fields up.

    If it added them up, the rehearsal would agree with itself while the real
    runner diverged -- and this is not hypothetical: a first attempt used the
    trace's final `kv_len_after`, which is one LESS than the emitted count
    because the last token's keys and values are computed by a forward that
    never happens. The identity check caught it."""
    assert "def engine_held_sequence(" in STUB, (
        "the stub must build the sequence object the engine-side loop holds")
    assert 'row["sequence_tokens"] = len(held)' in STUB, (
        "the stub must MEASURE the held object's length")
    assert 'row["sequence_tokens"] = len(prompt_ids) + 1 + cap' not in STUB
    assert "emitted_ids = held[len(engine_input_ids(prompt_ids)):]" in STUB, (
        "the sidecar must be the engine's own slice of the held object")
    assert "def engine_input_ids(" in STUB, (
        "the engine's prompt tensor must be built with the BOS, so the identity "
        "compares two genuinely different measurements")

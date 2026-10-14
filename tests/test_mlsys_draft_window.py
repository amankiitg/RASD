"""The draft must be conditioned on the RECENT context, not the document's start.

`RASDInference.generate_text` used to build its own draft input with
`tokenizer(prompt, max_length=self.draft_max_len, truncation=True)`. HF
truncation defaults to `truncation_side="right"`, so the tokens it KEPT were the
opening of the prompt. At any context longer than the draft window (Sheared
LLaMA-1.3B = 4096, Llama-3.2-1B capped at 4096) the draft was therefore
conditioned on the first 4k tokens of a 128k document while the target was
conditioned on its last ones -- and `manuscript/workshop/main.tex` says the
opposite ("it attends over only the most recent 4,096 tokens of the sequence").

These tests use the pinned tokenizers offline (they are in the local HF cache)
and the real `generate_text`, so a regression to a second, separately truncated
draft input is a failure here rather than a silent change of the draft's
conditioning.
"""
from __future__ import annotations

import pytest
import torch

from src.models.rasd_inference import RASDInference, _draft_window

TARGET_TOK = "meta-llama/Llama-2-7b-hf"        # ARM1's target
DRAFT_TOK = "princeton-nlp/Sheared-LLaMA-1.3B"  # ARM1's draft; shares the tokenizer
WINDOW = 4096                                   # Sheared-LLaMA-1.3B native window


def _long_text() -> str:
    return ("the quick brown fox jumps over the lazy dog. " * 600).strip()


@pytest.fixture(scope="module")
def tok():
    transformers = pytest.importorskip("transformers")
    return transformers.AutoTokenizer.from_pretrained(DRAFT_TOK)


def _engine(tokenizer, captured: dict, window: int = WINDOW):
    """A RASDInference carrying only what generate_text() needs.

    Loading 7B of weights to test the shape of one tokenizer call is not
    necessary: generate_text's whole job is to tokenize and delegate.
    """
    eng = RASDInference.__new__(RASDInference)
    eng.tokenizer = tokenizer
    eng.draft_max_len = window
    eng._device = torch.device("cpu")

    def _generate(input_ids, attention_mask=None, draft_input_ids=None,
                  **kwargs):
        captured["input_ids"] = input_ids
        captured["draft_input_ids"] = draft_input_ids
        captured["attention_mask"] = attention_mask
        return input_ids, {}

    eng.generate = _generate
    return eng


def test_generate_text_leaves_the_draft_input_to_generate(tok):
    """No second, separately truncated draft input.

    The window is enforced in `generate` (BOS + most recent tokens). A caller
    that pre-truncates here can only disagree with it, and did.
    """
    captured: dict = {}
    prompt = _long_text()
    n_prompt = len(tok(prompt)["input_ids"])
    assert n_prompt > WINDOW, "the fixture prompt must exceed the draft window"

    _engine(tok, captured).generate_text(prompt)

    assert captured["draft_input_ids"] is None, (
        "generate_text built its own draft input again; the draft window would "
        "then have two definitions, and this one keeps the FIRST tokens")
    assert captured["input_ids"].shape[1] == n_prompt, (
        "the target's input must not be truncated")


def test_the_draft_window_keeps_the_bos_and_the_recent_tokens(tok):
    ids = tok(_long_text())["input_ids"]
    assert len(ids) > WINDOW

    out = _draft_window(torch.tensor([ids]), WINDOW)

    assert out.shape[1] == WINDOW
    assert out[0, 0].item() == tok.bos_token_id, "the leading BOS is kept"
    assert out[0, 1:].tolist() == ids[-(WINDOW - 1):], (
        "the draft's ids must END with the prompt's final tokens")
    assert out[0].tolist() != ids[:WINDOW], (
        "the draft was given the OPENING of the prompt, which is what the "
        "removed tokenizer-side right-truncation did")


def test_a_prompt_that_fits_is_untouched(tok):
    """Short-context runs must stay bit-identical to before."""
    ids = tok("a short prompt")["input_ids"]
    out = _draft_window(torch.tensor([ids]), WINDOW)
    assert out[0].tolist() == ids


def test_a_window_of_one_token_is_just_the_bos(tok):
    ids = tok(_long_text())["input_ids"]
    out = _draft_window(torch.tensor([ids]), 1)
    assert out.tolist() == [[ids[0]]]

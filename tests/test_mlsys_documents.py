"""Documents: one book per document, and the prompt windows a rung consumes."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from run_experiment import _build_pg19_document_prompt, build_run_configs
from scripts.mlsys_build_documents import eligible


class _StubTok:
    """Identity codec: one token per character is not needed, just round-trip."""

    def __init__(self):
        self.vocab = {i: f"t{i}" for i in range(100000)}

    def decode(self, ids):
        return " ".join(self.vocab.get(int(i), "t0") for i in ids)

    def encode(self, text, add_special_tokens=False):
        return [int(t[1:]) for t in text.split()]


def test_document_selection_and_prompt_windows(tmp_path: Path):
    # Selection: only books that can serve the top rung, longest first, and the
    # rule is the plan's (10 longest) so it is reproducible without an RNG.
    # A book must cover the rung PLUS one generation: the engine re-tokenises
    # the prompt string, so the source slice can be longer than the prompt it
    # produces and the continuation must still fit after it.
    books = [{"index": i, "tokens": t} for i, t in
             enumerate([1000, 60000, 525312, 700000, 900000, 525313, 530000])]
    # `eligible` is the sorted pool; the plan's "10 longest" rule is the
    # caller's slice, so the pool is asserted and then sliced here.
    pool = eligible(books, max_rung=524288, need_docs=3)
    assert [b["tokens"] for b in pool] == [900000, 700000, 530000, 525313, 525312]
    assert [b["tokens"] for b in pool[:3]] == [900000, 700000, 530000]
    with pytest.raises(RuntimeError, match="not feasible"):
        eligible(books, max_rung=524288, need_docs=6)

    # A document is one memmap, so the prompt and the scored continuation are
    # contiguous slices of one book and the sequence is exactly the rung.
    ctx, gen = 64, 8
    ids = list(range(1000, 1000 + ctx))
    dat = tmp_path / "doc_000.dat"
    np.memmap(dat, dtype="int32", mode="w+", shape=(len(ids),))[:] = ids
    meta = {"documents": [{"doc_id": "pg19_train_0", "file": str(dat),
                           "length": len(ids), "title": "t", "url": "u"}]}
    dj = tmp_path / "documents.json"
    dj.write_text(json.dumps(meta))

    text, cont, prov = _build_pg19_document_prompt(
        str(dj), ctx, "pg19_train_0", _StubTok(), gen_tokens=gen)

    # prompt is ctx - gen - 1: the - 1 is the engine's leading BOS, without
    # which the model would see ctx + 1 positions, one past a native window.
    assert prov["prompt_tokens"] == ctx - gen - 1
    assert len(cont) == gen
    assert cont == ids[ctx - gen - 1:ctx - 1]
    # contiguous with the prompt: the token after the prompt is the
    # continuation's first, so one forward scores both at the right offsets
    assert cont[0] == ids[prov["prompt_tokens"]]

    with pytest.raises(ValueError, match="unknown doc_id"):
        _build_pg19_document_prompt(str(dj), ctx, "nope", _StubTok(), gen_tokens=gen)

    # Document expansion: a level naming documents produces one run each, with
    # the document in the run id, and seeds stay a no-op axis as the plan says.
    cfg = {
        "defaults": {"seeds": [42], "context_length": ctx},
        "G1": {"levels": [{"id": "r1", "documents": ["d_a", "d_b"]}]},
    }
    runs = build_run_configs(cfg, None, debug=False)
    assert [r["run_id"] for r in runs] == ["r1_d_a_s42", "r1_d_b_s42"]
    assert [r["doc_id"] for r in runs] == ["d_a", "d_b"]


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))

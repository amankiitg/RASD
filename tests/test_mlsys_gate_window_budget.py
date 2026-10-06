"""The gate must not score a candidate outside the window it is calibrating."""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parent.parent


def _load_gate():
    spec = importlib.util.spec_from_file_location(
        "gate_mod", REPO / "scripts" / "mlsys_coherence_gate.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["gate_mod"] = mod
    spec.loader.exec_module(mod)
    return mod


def test_gate_window_fits_inside_the_context_it_calibrates(tmp_path: Path):
    gate = _load_gate()
    # ctx must exceed CONTINUATION_TOKENS (1024) or there is no room for a
    # prompt at all and the loader is supposed to raise.
    ctx = 2048
    ids = list(range(5000, 5000 + ctx + 64))
    dat = tmp_path / "doc_000.dat"
    np.memmap(dat, dtype="int32", mode="w+", shape=(len(ids),))[:] = ids
    # The new one-book-per-document shape...
    (tmp_path / "documents.json").write_text(json.dumps(
        {"documents": [{"doc_id": "d0", "file": str(dat), "length": len(ids)}]}))
    # ...and the older concatenated-chunk shape must both still load.
    (tmp_path / "chunks.json").write_text(json.dumps(
        {"chunks": [{"file": str(dat), "length": len(ids)}]}))

    for name in ("documents.json", "chunks.json"):
        prompt, cont = gate.load_pg19_window(str(tmp_path / name), ctx, seed=7)
        # prompt + continuation == context: an earlier version used a
        # full-length prompt PLUS a continuation, i.e. ctx + 1024 positions,
        # which scores a candidate past its own window and would flatter every
        # candidate against the native positive control.
        assert len(prompt) + len(cont) == ctx
        assert len(cont) == gate.CONTINUATION_TOKENS
        # The generation must fit too, not just the perplexity pass.
        assert len(prompt) + gate.GENERATE_TOKENS <= ctx

    # Too short a document fails loudly rather than being silently shortened.
    (tmp_path / "short.json").write_text(json.dumps(
        {"documents": [{"doc_id": "s", "file": str(dat), "length": 10}]}))
    with pytest.raises(RuntimeError, match="cannot run at"):
        gate.load_pg19_window(str(tmp_path / "short.json"), ctx, seed=1)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))

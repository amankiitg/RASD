"""Losslessness: a spec run must match its target-only run token for token."""
from __future__ import annotations

import pytest

from src.analysis.losslessness import (
    compare_generations, first_mismatch, require_same_request,
)


def test_losslessness_comparison_and_pairing_guard():
    # Identical streams over the full generation are lossless.
    ok = compare_generations([1, 2, 3, 4], [1, 2, 3, 4], requested_tokens=4)
    assert ok["lossless"] is True
    assert ok["first_mismatch_position"] == ""
    assert ok["compared_tokens"] == 4

    # The first differing position is reported, not just a boolean. This is the
    # value a reviewer asks for when a cell fails.
    bad = compare_generations([1, 2, 9, 4], [1, 2, 3, 4], requested_tokens=4)
    assert bad["lossless"] is False
    assert bad["first_mismatch_position"] == 2
    assert "spec=9" in bad["detail"] and "target=3" in bad["detail"]

    # A short run is a divergence even though it agrees with the prefix: both
    # runs were asked for the same length, so stopping early changed the output.
    short = compare_generations([1, 2], [1, 2, 3, 4], requested_tokens=4)
    assert short["lossless"] is False
    assert short["first_mismatch_position"] == 2

    assert first_mismatch([], []) is None
    assert first_mismatch([1], [1, 2]) == 1

    # Pairing is checked rather than assumed: a mismatched request would still
    # produce a green tick and make the whole check worthless.
    spec = {"prompt_sha256": "a", "prompt_tokens": 10, "context_length": 131072,
            "max_new_tokens": 1024, "spec_steps": 4}
    tgt = dict(spec, spec_steps=0)
    assert require_same_request(spec, tgt) == []
    assert require_same_request(spec, dict(tgt, doc_id="x", prompt_sha256="b")) == \
        ["prompt_sha256: spec='a' target='b'"]
    assert require_same_request(spec, spec) == [
        f"target row has spec_steps={spec['spec_steps']}"
    ]
    assert require_same_request(dict(spec, spec_steps=0), tgt) == [
        "spec row has spec_steps=0 (it is a target-only run)"
    ]


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))

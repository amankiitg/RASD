"""Losslessness: verdicts, the prefix case, and the pairing guard.

The campaign runs three target-only partners at the full 1024 tokens and seven at
128 tokens, because a throughput RATE needs a steady state while losslessness
needs an equal length. The verdict must therefore say which kind of agreement it
verified rather than reporting a short partner as no agreement at all.
"""
from __future__ import annotations

import pytest

from src.analysis.losslessness import (
    compare_generations, first_mismatch, require_same_request,
    stage_requirement,
)


def test_prefix_verdict_distinguishes_full_match_from_short_partner():
    """A 128-token partner verifying the first 128 of 1024 is not a failure."""
    spec = list(range(1000, 2024))          # 1024 generated tokens

    # Full-length partner: everything matches, verdict LOSSLESS.
    full = compare_generations(spec, spec[:1024], full_length=1024,
                               min_prefix=128)
    assert full["verdict"] == "LOSSLESS"
    assert full["lossless"] is True
    assert full["verified_prefix"] == 1024
    assert full["meets_min_prefix"] is True

    # Short partner: the first 128 match, verdict names that prefix.
    short = compare_generations(spec, spec[:128], full_length=1024,
                                min_prefix=128)
    assert short["verdict"] == "LOSSLESS_PREFIX_128"
    assert short["lossless"] is False          # not a full-length check
    assert short["lossless_full_prefix"] is True
    assert short["verified_prefix"] == 128
    assert short["meets_min_prefix"] is True

    # A prefix shorter than the minimum does NOT satisfy the requirement, even
    # though every token it compared matched.
    tiny = compare_generations(spec, spec[:64], full_length=1024, min_prefix=128)
    assert tiny["verdict"] == "LOSSLESS_PREFIX_64"
    assert tiny["meets_min_prefix"] is False

    # A divergence inside the prefix is a MISMATCH with its position, and it must
    # outrank any prefix verdict.
    bad = list(spec[:128])
    bad[7] = 4242
    mism = compare_generations(spec, bad, full_length=1024, min_prefix=128)
    assert mism["verdict"] == "MISMATCH"
    assert mism["first_mismatch_position"] == 7
    assert mism["meets_min_prefix"] is False
    assert "spec=1007" in mism["detail"] and "target=4242" in mism["detail"]


def test_stage_requirement_needs_a_full_length_check_and_a_min_prefix():
    """A stage cannot pass on short partners alone."""
    full_ok = {"spec_run_id": "a", "verdict": "LOSSLESS", "verified_prefix": 1024,
               "target_tokens": 1024}
    short_ok = {"spec_run_id": "b", "verdict": "LOSSLESS_PREFIX_128",
                "verified_prefix": 128, "target_tokens": 128}
    short_bad = {"spec_run_id": "c", "verdict": "MISMATCH",
                 "verified_prefix": 128, "target_tokens": 128,
                 "first_mismatch_position": 3}
    too_short = {"spec_run_id": "d", "verdict": "LOSSLESS_PREFIX_64",
                 "verified_prefix": 64, "target_tokens": 64}

    assert stage_requirement([full_ok, short_ok], 1024)["ok"] is True
    assert stage_requirement([short_ok], 1024)["ok"] is False            # no full check
    assert stage_requirement([full_ok, short_bad], 1024)["ok"] is False  # divergence
    assert stage_requirement([full_ok, too_short], 1024)["ok"] is False  # prefix < min


def test_pair_guard_allows_a_shorter_partner_but_not_a_longer_one():
    base = {"prompt_sha256": "h", "prompt_tokens": 10, "context_length": 131072,
            "temperature": 0.0, "top_p": 1.0, "ignore_eos": True,
            "rope_type": "none", "rope_factor": "", "rope_anchor_base": "",
            "target_revision": "r", "draft_revision": "r"}
    spec = dict(base, spec_steps=4, max_new_tokens=1024, _n_tokens=1024)
    short = dict(base, spec_steps=0, max_new_tokens=128, _n_tokens=128)
    full = dict(base, spec_steps=0, max_new_tokens=1024, _n_tokens=1024)

    # max_new_tokens is deliberately NOT compared: a 128-token partner against a
    # 1024-token run is the supported case.
    assert require_same_request(spec, short) == []
    assert require_same_request(spec, full) == []

    # A partner LONGER than the run it checks cannot be a prefix comparison.
    longer = dict(base, spec_steps=0, max_new_tokens=2048, _n_tokens=2048)
    assert any("more tokens than the run it checks" in p
               for p in require_same_request(spec, longer))

    # The other contract fields still have to agree.
    assert require_same_request(spec, dict(short, temperature=1.0)) == [
        "temperature: spec=0.0 target=1.0"
    ]
    assert require_same_request(spec, dict(short, prompt_sha256="x")) == [
        "prompt_sha256: spec='h' target='x'"
    ]
    assert first_mismatch([], []) is None
    assert first_mismatch([1], [1, 2]) == 1


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))

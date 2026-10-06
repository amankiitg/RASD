"""A vLLM row may only count as comparable when every fairness condition holds.

The vLLM baseline is the production-stack reference the speedup claims are read
against, so a row that is merely "ok" is not enough: it has to be the same
prompt, the same generation length, the same EOS policy and the same sampling
rule, on the pinned release. Three of those were not enforced:

  * the sampling temperature was 1.0 while the RASD cells are greedy;
  * the pin was a fallback rather than a check, so an unreadable version was
    reported AS the pin;
  * the prompt-id hash used a different spelling from the sidecars', which made
    the "is this the RASD prompt" test impossible to satisfy.

That last one is the dangerous shape: an inert check looks like a strict one.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

REPO = Path(__file__).resolve().parent.parent
SRC = (REPO / "scripts" / "mlsys_vllm_baseline.py").read_text()


def _load_helpers():
    """Exec just the pure helpers, since the module imports torch and vllm."""
    import re
    ns = {"hashlib": hashlib, "json": json, "re": re, "Path": Path,
          "SystemExit": SystemExit, "print": print,
          "float": float, "int": int, "str": str, "list": list}
    start = SRC.index("VLLM_PIN = ")
    end = SRC.index("def load_rasd_cells")
    exec(SRC[start:end], ns)                     # constants + _ids_sha
    start = SRC.index("def rasd_decode_rate")
    end = SRC.index("def load_rasd_cells")
    exec(SRC[start:end], ns)
    start = SRC.index("def _unit_match_verdict")
    end = SRC.index("def main(")
    exec(SRC[start:end], ns)
    return ns


def _args():
    return SimpleNamespace(matched_max_new_tokens=1024)


def _row(**kw):
    import importlib
    ns = _load_helpers()
    sha = ns["_ids_sha"]([128000, 9906, 1917])
    row = {"eos_policy": ns["EOS_POLICY"], "tensor_parallel_size": 8,
           "max_new_tokens": 1024, "temperature": 0.0,
           "vllm_version": ns["VLLM_PIN"], "doc_id": "pg19_train_0",
           "prompt_sha256": sha}
    row.update(kw)
    return ns, row


@pytest.mark.skipif(not SRC, reason="script missing")
def test_prompt_hash_matches_the_rasd_sidecar_convention():
    ns = _load_helpers()
    ids = [128000, 9906, 1917, 42]
    rasd = hashlib.sha256(",".join(str(i) for i in ids).encode()).hexdigest()
    assert ns["_ids_sha"](ids) == rasd, (
        "the prompt-id hash disagrees with run_experiment's, so the "
        "'is this the RASD prompt' check can never pass"
    )
    # and the OLD spelling really is different, so the test is not vacuous
    assert hashlib.sha256(json.dumps(ids).encode()).hexdigest()[:16] != rasd


@pytest.mark.skipif(not SRC, reason="script missing")
def test_unit_match_requires_greedy_pinned_and_the_rasd_prompt():
    ns, row = _row()
    ok, why = ns["_unit_match_verdict"](row, [128000, 9906, 1917], _args())
    assert ok, why

    for field, bad, needle in (
        ("temperature", 1.0, "temperature"),
        ("vllm_version", "0.6.2", "vllm_version"),
        ("max_new_tokens", 128, "max_new_tokens"),
        ("eos_policy", "stop_on_eos", "eos policy"),
        ("tensor_parallel_size", 4, "tensor_parallel_size"),
        ("doc_id", "", "doc_id"),
    ):
        _, r = _row(**{field: bad})
        ok, why = ns["_unit_match_verdict"](r, [128000, 9906, 1917], _args())
        assert not ok, f"{field}={bad!r} still counted as unit-matched"
        assert needle in why, why


@pytest.mark.skipif(not SRC, reason="script missing")
def test_a_row_whose_prompt_hash_disagrees_is_not_comparable():
    ns, row = _row(prompt_sha256="0" * 64)
    ok, why = ns["_unit_match_verdict"](row, [128000, 9906, 1917], _args())
    assert not ok and "prompt sha256" in why, why

    # no prompt ids at all is not comparable either
    ok, why = ns["_unit_match_verdict"](row, None, _args())
    assert not ok and "prompt token ids" in why, why


@pytest.mark.skipif(not SRC, reason="script missing")
def test_pin_is_enforced_not_defaulted():
    assert "getattr(_vllm, \"__version__\", \"\") or VLLM_PIN" not in SRC, (
        "the version still falls back to the pin, so a row can claim the "
        "pinned release while running anything"
    )
    assert "vLLM version is" in SRC and "SystemExit" in SRC


@pytest.mark.skipif(not SRC, reason="script missing")
def test_sampling_is_greedy_and_the_decode_rate_matches_rasd():
    assert "SamplingParams(temperature=0.0" in SRC, (
        "vLLM still samples while RASD runs greedy"
    )
    ns = _load_helpers()
    # (101 - 1) / 10.0, the same convention as RASD's decode_tps
    assert ns["rasd_decode_rate"](101, 10.0) == pytest.approx(10.0)
    assert ns["rasd_decode_rate"](0, 10.0) == 0.0


@pytest.mark.skipif(not SRC, reason="script missing")
def test_sidecars_carry_the_exact_prompt_ids_the_engine_fed():
    """Without this the vLLM arm has nothing to be given."""
    src = (REPO / "run_experiment.py").read_text()
    assert '"prompt_token_ids"' in src, (
        "the token sidecar does not record the prompt ids"
    )


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))

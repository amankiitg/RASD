"""Every arm of a rung must be handed the SAME prompt.

The plan's losslessness check compares token ids between a speculative run and
its target-only partners, and `require_same_request` refuses the pair unless both
share a `prompt_sha256`. The PG-19 prompt window is `[0, C - gen_tokens - 1)`,
where `gen_tokens` used to be the arm's OWN `max_new_tokens`.

So the seven 128-token baselines were given a prompt 896 tokens longer than the
1024-token speculative run they exist to check: a different context, over which
token equality says nothing. Every one of those cells would have come back
BAD_PAIR from the real checker, and a stage declaring `losslessness: required`
would have failed on a defect introduced by the prompt builder -- with no GPU
needed to see it, and no run to blame.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

yaml = pytest.importorskip("yaml")
rexp = pytest.importorskip("run_experiment")

STAGE_CONFIGS = [
    "configs/mlsys_natural_f1_128k.yml",
    "configs/mlsys_natural_f1_128k_diverse.yml",
    "configs/mlsys_natural_gated.yml",
    "configs/mlsys_synthetic_gated.yml",
]


def _config(f: str) -> dict:
    return yaml.safe_load((REPO / f).read_text())


def test_the_rung_prompt_length_overrides_the_arms_own_cap():
    short = {"max_new_tokens": 128, "prompt_gen_tokens": 1024}
    assert rexp._prompt_gen_tokens(short) == 1024
    # and an arm that declares nothing keeps the old behaviour
    assert rexp._prompt_gen_tokens({"max_new_tokens": 128}) == 128
    assert rexp._prompt_gen_tokens({}) == 1024


@pytest.mark.parametrize("f", STAGE_CONFIGS)
def test_short_baselines_pin_the_rung_prompt(f):
    cfg = _config(f)
    spec_gens = {int(lvl.get("max_new_tokens", 1024))
                 for k, v in cfg.items() if k not in ("defaults", "canary")
                 and "TARGET" not in k
                 for lvl in v.get("levels", [])}
    assert spec_gens, f"{f}: no speculative group found"
    for k, v in cfg.items():
        if k in ("defaults", "canary") or "TARGET_SHORT" not in k:
            continue
        for lvl in v.get("levels", []):
            assert int(lvl.get("prompt_gen_tokens") or 0) in spec_gens, (
                f"{f}:{lvl['id']} generates {lvl.get('max_new_tokens')} tokens "
                f"but does not pin prompt_gen_tokens to the rung's "
                f"{sorted(spec_gens)}; its prompt would not match the "
                f"speculative row it is paired with")


@pytest.mark.parametrize("f", STAGE_CONFIGS)
def test_all_arms_of_one_rung_share_a_prompt_window(f):
    """The property that actually matters, checked on the planner's output."""
    runs = rexp.build_run_configs(_config(f), None, False, seed_filter=None)
    by_rung: dict = {}
    for r in runs:
        key = (r["level_id"].split("_")[-1], str(r.get("context_length")),
               str(r.get("doc_id", "")))
        gen = rexp._prompt_gen_tokens(r)
        prompt_len = int(r["context_length"]) - gen - 1
        by_rung.setdefault(key, set()).add(prompt_len)
    mixed = {k: sorted(v) for k, v in by_rung.items() if len(v) > 1}
    assert not mixed, (
        f"{f}: these rungs give different prompt lengths to different arms, so "
        f"the losslessness pairing is over different contexts: {mixed}")


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))

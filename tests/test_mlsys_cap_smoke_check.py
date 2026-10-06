"""The cap-smoke checker must fail on the failures it exists to detect.

The engine's generation cap cannot be exercised without a GPU, so the safest
available verification is on the checker itself: feed it a pass case and each
failure mode and require the right verdict.
"""
from __future__ import annotations

import csv
import importlib.util
import json
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent


def _load():
    spec = importlib.util.spec_from_file_location(
        "cap_smoke", REPO / "scripts" / "mlsys_cap_smoke_check.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["cap_smoke"] = mod
    spec.loader.exec_module(mod)
    return mod


FIELDS = ["run_id", "doc_id", "context_length", "max_new_tokens", "spec_steps",
          "status", "prompt_tokens", "prompt_sha256", "sequence_tokens",
          "tokens_generated", "temperature", "top_p", "ignore_eos",
          "rope_type", "rope_factor", "rope_anchor_base",
          "target_revision", "draft_revision"]


def _write(tmp: Path, cap=64, spec_gen=None, trace=None, partner=True,
           lossless=True):
    rows, toks = [], {}
    (tmp / "per_token").mkdir(parents=True, exist_ok=True)
    (tmp / "tokens").mkdir(parents=True, exist_ok=True)
    spec_ids = list(range(100, 100 + cap))
    tgt_ids = list(spec_ids) if lossless else [999] + spec_ids[1:]
    for arm, ss, ids, gen in (("spec", 3, spec_ids, spec_gen or cap),
                              ("tgt", 0, tgt_ids, cap)):
        if arm == "tgt" and not partner:
            continue
        rid = f"CAP_{arm}"
        toks[rid] = ids
        rows.append({"run_id": rid, "doc_id": "d0", "context_length": 2048,
                     "max_new_tokens": cap, "spec_steps": ss, "status": "ok",
                     # prompt + 1 BOS + generated == context, by construction;
                     # a fixture that breaks its own identity would make the
                     # checker look wrong when it is right.
                     "prompt_tokens": 2048 - 1 - gen, "prompt_sha256": "h",
                     "sequence_tokens": 2048, "tokens_generated": gen,
                     "temperature": 0.0, "top_p": 1.0, "ignore_eos": True,
                     "rope_type": "none", "rope_factor": "", "rope_anchor_base": "",
                     "target_revision": "r", "draft_revision": "r"})
        (tmp / "tokens" / f"{rid}.json").write_text(json.dumps(
            {"run_id": rid, "generated_token_ids": ids}))
    with (tmp / "res.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)
    # A consistent default trace: with gamma=3 each round yields 3 accepted + 1
    # bonus = 4 tokens, so cap/4 full rounds account for the cap exactly.
    full = {"n_acc": 3, "n_emitted": 3, "round_truncated": False}
    tr = trace if trace is not None else [dict(full) for _ in range(cap // 4)]
    for rid in (f"CAP_{a}" for a in ("spec", "tgt")):
        if toks.get(rid) is None:
            continue
        (tmp / "per_token" / f"{rid}.jsonl").write_text(
            "\n".join(json.dumps(x) for x in tr) + "\n")
    return tmp / "res.csv"


def test_cap_smoke_checker_accepts_a_clean_run_and_rejects_each_defect(tmp_path):
    mod = _load()

    # A clean run passes.
    problems, notes = mod.check(_write(tmp_path / "a"), tmp_path / "a" / "tokens")
    assert problems == [], problems
    assert any("generated exactly 64" in n for n in notes)

    # The defect the stage exists for: the cap is not enforced.
    problems, _ = mod.check(_write(tmp_path / "b", spec_gen=67),
                            tmp_path / "b" / "tokens")
    assert any("cap is not enforced" in p for p in problems), problems

    # A divergence is not lossless, and must be reported as such rather than
    # passing because the lengths agree.
    problems, _ = mod.check(_write(tmp_path / "c", lossless=False),
                            tmp_path / "c" / "tokens")
    assert any("MISMATCH" in p for p in problems), problems

    # No target-only partner means losslessness is unverified, which is not the
    # same as verified.
    problems, _ = mod.check(_write(tmp_path / "d", partner=False),
                            tmp_path / "d" / "tokens")
    assert any("no target-only partner" in p for p in problems), problems

    # Trace bookkeeping that does not sum to the cap is a failure: the cap being
    # enforced in the CSV but not in the round accounting would mean the two
    # disagree about what was generated.
    problems, _ = mod.check(
        _write(tmp_path / "e", trace=[{"n_acc": 3, "n_emitted": 3,
                                       "round_truncated": False}]),
        tmp_path / "e" / "tokens")
    assert any("trace accounts for" in p for p in problems), problems

    # Only the final round may be cut short.
    problems, _ = mod.check(
        _write(tmp_path / "f", trace=[{"n_acc": 3, "n_emitted": 3,
                                       "round_truncated": False},
                                      {"n_acc": 3, "n_emitted": 1,
                                       "round_truncated": True},
                                      {"n_acc": 3, "n_emitted": 3,
                                       "round_truncated": False}]),
        tmp_path / "f" / "tokens")
    assert any("only the FINAL" in p for p in problems), problems


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))

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
    # bonus = 4 tokens, so cap/4 full rounds account for the cap exactly. The KV
    # chain is included because the checker asserts it: a round must leave the KV
    # covering exactly what it emitted.
    tr = trace if trace is not None else _kv_trace(cap // 4, base=1984)
    for rid in (f"CAP_{a}" for a in ("spec", "tgt")):
        if toks.get(rid) is None:
            continue
        (tmp / "per_token" / f"{rid}.jsonl").write_text(
            "\n".join(json.dumps(x) for x in tr) + "\n")
    return tmp / "res.csv"


def _kv_trace(n_rounds, base, gamma=3):
    """`n_rounds` full rounds with a consistent KV chain."""
    out, kv = [], base
    for _ in range(n_rounds):
        committed = gamma + 1
        out.append({"n_acc": gamma, "n_emitted": gamma, "n_committed": committed,
                    "round_truncated": False,
                    "kv_len_before": kv, "kv_len_after": kv + committed})
        kv += committed
    return out


def _kv_trace_truncated(n_full, base, gamma=3, partial=2):
    """Full rounds then one round cut short by the budget (no bonus)."""
    out, kv = [], base
    for _ in range(n_full):
        out.append({"n_acc": gamma, "n_emitted": gamma, "n_committed": gamma + 1,
                    "round_truncated": False,
                    "kv_len_before": kv, "kv_len_after": kv + gamma + 1})
        kv += gamma + 1
    # The target VERIFIED `gamma` but only `partial` fit in the budget, so the
    # accepted prefix is emitted in part and no bonus is available.
    out.append({"n_acc": gamma, "n_emitted": partial, "n_committed": partial,
                "round_truncated": True,
                "kv_len_before": kv, "kv_len_after": kv + partial})
    return out


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


def test_cap_smoke_checks_the_truncated_round_kv_geometry(tmp_path):
    """The KV after a round must cover exactly what the round emitted.

    The truncated final round is where this can go wrong, and the engine needs
    CUDA, so this checker is the only place the arithmetic is ever exercised. Two
    failure modes, both silent on a GPU:

      * keeping the verified-but-unemitted tail (KV too long) -- the next round's
        cur_token then sits at the wrong positional offset and every later round
        is corrupted;
      * committing a bonus the budget had no room for (KV too short, and one
        token too many emitted).
    """
    mod = _load()

    # Clean: 15 full rounds of 3 accepted + 1 bonus, then a round cut short at 2.
    cap = 62
    good = _kv_trace_truncated(15, base=1986, partial=2)
    csv_path = _write(tmp_path / "ok", cap=cap, trace=good)
    problems, _ = mod.check(csv_path)
    assert problems == [], problems

    # KV kept the whole verified prefix instead of only the emitted part.
    kept_tail = [dict(x) for x in good]
    trunc = kept_tail[-1]
    trunc["kv_len_after"] = trunc["kv_len_before"] + 4     # 3 verified + bonus
    csv_path = _write(tmp_path / "tail", cap=cap, trace=kept_tail)
    problems, _ = mod.check(csv_path)
    assert any("verified emitted prefix" in p or "committed" in p
               for p in problems), problems

    # A truncated round must not emit a bonus.
    with_bonus = [dict(x) for x in good]
    trunc = with_bonus[-1]
    trunc["n_committed"] = trunc["n_emitted"] + 1
    trunc["kv_len_after"] = trunc["kv_len_before"] + trunc["n_committed"]
    csv_path = _write(tmp_path / "bonus", cap=cap, trace=with_bonus)
    problems, _ = mod.check(csv_path)
    assert any("committed" in p for p in problems), problems

    # A gap in the KV chain: the next round starts where the last one did not end.
    gap = [dict(x) for x in good]
    gap[-1] = dict(gap[-1], kv_len_before=gap[-1]["kv_len_before"] - 1)
    csv_path = _write(tmp_path / "gap", cap=cap, trace=gap)
    problems, _ = mod.check(csv_path)
    assert problems, "a broken KV chain was accepted"

    # The KV length must be recorded at all, or it cannot be verified.
    silent = [dict(x) for x in good]
    silent[-1] = {k: v for k, v in silent[-1].items() if k != "kv_len_after"}
    csv_path = _write(tmp_path / "silent", cap=cap, trace=silent)
    problems, _ = mod.check(csv_path)
    assert any("no kv_len_after" in p for p in problems), problems

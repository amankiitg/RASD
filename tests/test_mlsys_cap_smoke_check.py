"""The cap smoke must check the shapes the real code actually writes.

Two things this file is careful about, because getting either wrong makes the
checker look right while it is wrong:

* The rows use `run_experiment.CSV_FIELDS` itself, not a hand-written column
  list. A fixture with its own columns tests the fixture.
* A target-only row has NO verify rounds, so it has no per-round trace, and
  DEMANDING one of it asked the generator to have produced something it does not
  have. It is checked on `tokens_generated == max_new_tokens` and its token
  sidecar instead.

The arithmetic also has to count the seed token. `generated` holds one token
before the verify loop starts, so the identity is

    1 + accepted_emitted + bonuses == max_new_tokens

and the previous `accepted + bonuses == cap` was short by exactly that token.
"""
from __future__ import annotations

import csv
import importlib.util
import json
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent


def _load_checker():
    spec = importlib.util.spec_from_file_location(
        "cap_mod", REPO / "scripts" / "mlsys_cap_smoke_check.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["cap_mod"] = mod
    spec.loader.exec_module(mod)
    return mod


def _run_experiment_fields():
    """The column set run_experiment writes, straight from the source."""
    spec = importlib.util.spec_from_file_location(
        "runexp_fields", REPO / "run_experiment.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["runexp_fields"] = mod
    spec.loader.exec_module(mod)
    return list(mod.CSV_FIELDS)


FIELDS = _run_experiment_fields()

CONTEXT = 2048
GAMMA = 3


def _row(run_id, spec_steps, cap, *, status="ok", doc="d0"):
    """A row with the real column set, so the fixture cannot drift from it."""
    row = {f: "" for f in FIELDS}
    row.update({
        "run_id": run_id, "doc_id": doc, "context_length": CONTEXT,
        "max_new_tokens": cap, "spec_steps": spec_steps, "status": status,
        "prompt_tokens": CONTEXT - 1 - cap,
        "sequence_tokens": CONTEXT, "tokens_generated": cap,
        "prompt_sha256": "h" + doc, "temperature": 0.0, "top_p": 1.0,
        "ignore_eos": True, "pair_id": f"{CONTEXT}:{doc}",
        "arm_role": "spec" if spec_steps else "target_full",
        "acceptance_rate": 0.75 if spec_steps else 0.0,
        "decode_tps": 10.0, "throughput_tps": 8.0,
    })
    return row


def _round(i, base, n_acc, *, truncated=False, spec_steps=GAMMA, kv=None):
    """One per-round trace entry, with the keys the engine writes."""
    n_emit = n_acc if not truncated else min(n_acc, GAMMA)
    committed = n_emit if truncated else n_emit + 1
    kb = base if kv is None else kv
    return {
        "round_idx": i, "global_pos_start": kb, "spec_steps": spec_steps,
        "n_acc": n_acc,
        "draft_tokens": list(range(spec_steps)), "accepted": [True] * n_acc,
        "ended_on_eos": False,
        "n_emitted": n_emit, "round_truncated": truncated,
        "n_committed": committed, "kv_len_before": kb,
        "kv_len_after": kb + committed,
    }


def _spec_trace(cap, base=1984):
    """Rounds that account for the cap exactly, final round truncated."""
    full, rest = divmod(cap - 1, GAMMA + 1)
    tr, kv = [], base
    for i in range(full):
        tr.append(_round(i, base, GAMMA, kv=kv))
        kv = tr[-1]["kv_len_after"]
    if rest:
        tr.append(_round(full, base, GAMMA, truncated=True, kv=kv))
    return tr


def _write(tmp: Path, cap=64, *, trace=None, partner=True, sidecar_ids=None,
           spec_status="ok", target_status="ok", lossless=True):
    rows = []
    spec_ids = list(range(100, 100 + cap))
    tgt_ids = list(spec_ids) if lossless else [999] + spec_ids[1:]
    (tmp / "per_token").mkdir(parents=True, exist_ok=True)
    (tmp / "tokens").mkdir(parents=True, exist_ok=True)

    rows.append(_row("CAP_spec", GAMMA, cap, status=spec_status))
    tr = _spec_trace(cap) if trace is None else trace
    (tmp / "per_token" / "CAP_spec.jsonl").write_text(
        "\n".join(json.dumps(x) for x in tr) + ("\n" if tr else ""))
    (tmp / "tokens" / "CAP_spec.json").write_text(json.dumps(
        {"run_id": "CAP_spec", "generated_token_ids": spec_ids,
         "prompt_token_ids": [1, 2, 3], "doc_id": "d0",
         "context_length": CONTEXT, "max_new_tokens": cap}))

    if partner:
        rows.append(_row("CAP_tgt", 0, cap, status=target_status))
        (tmp / "tokens" / "CAP_tgt.json").write_text(json.dumps(
            {"run_id": "CAP_tgt",
             "generated_token_ids": sidecar_ids if sidecar_ids else tgt_ids,
             "prompt_token_ids": [1, 2, 3], "doc_id": "d0",
             "context_length": CONTEXT, "max_new_tokens": cap}))

    with (tmp / "res.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)
    return tmp / "res.csv"


def test_a_clean_pair_passes_with_the_real_arithmetic(tmp_path):
    mod = _load_checker()
    problems, notes = mod.check(_write(tmp_path, cap=64))
    assert problems == [], problems
    assert any("1 seed" in n for n in notes), notes


def test_a_target_only_row_needs_no_per_round_trace(tmp_path):
    """The absence of a verify loop is not a missing artifact."""
    mod = _load_checker()
    csv_path = _write(tmp_path, cap=64)
    assert not (tmp_path / "per_token" / "CAP_tgt.jsonl").exists()
    problems, _ = mod.check(csv_path)
    assert problems == [], problems


def test_a_spec_row_without_a_trace_is_still_a_failure(tmp_path):
    mod = _load_checker()
    csv_path = _write(tmp_path, cap=64)
    (tmp_path / "per_token" / "CAP_spec.jsonl").unlink()
    problems, _ = mod.check(csv_path)
    assert any("no per-round trace" in p for p in problems), problems


def test_the_seed_token_is_counted(tmp_path):
    """`emitted + bonuses == cap` is short by the seed; this is the regression."""
    mod = _load_checker()
    cap = 64
    full = cap // (GAMMA + 1)
    tr, kv = [], 1984
    for i in range(full):
        tr.append(_round(i, 1984, GAMMA, kv=kv))
        kv = tr[-1]["kv_len_after"]
    csv_path = _write(tmp_path, cap=cap, trace=tr)
    problems, _ = mod.check(csv_path)
    assert any("seed" in p and "expected 64" in p for p in problems), problems


def test_the_cap_itself_must_hold_in_the_sidecar(tmp_path):
    mod = _load_checker()
    csv_path = _write(tmp_path, cap=64, sidecar_ids=[1, 2, 3])
    problems, _ = mod.check(csv_path)
    assert any("token sidecar holds 3 ids" in p for p in problems), problems


def test_a_non_ok_row_is_ignored_rather_than_counted_as_a_pair(tmp_path):
    mod = _load_checker()
    csv_path = _write(tmp_path, cap=64, target_status="error")
    problems, _ = mod.check(csv_path)
    assert any("no target-only partner" in p for p in problems), problems


def test_each_cap_defect_is_caught(tmp_path):
    """The KV geometry, and the two ways a round can commit wrongly."""
    mod = _load_checker()
    cap = 64
    good = _spec_trace(cap)

    kept = [dict(x) for x in good]
    kept[-1]["kv_len_after"] = kept[-1]["kv_len_before"] + GAMMA + 1
    p, _ = mod.check(_write(tmp_path / "tail", cap=cap, trace=kept))
    assert any("committed" in x or "verified emitted prefix" in x for x in p), p

    bonus = [dict(x) for x in good]
    bonus[-1]["n_committed"] = bonus[-1]["n_emitted"] + 1
    bonus[-1]["kv_len_after"] = bonus[-1]["kv_len_before"] + bonus[-1]["n_committed"]
    p, _ = mod.check(_write(tmp_path / "bonus", cap=cap, trace=bonus))
    assert any("committed" in x for x in p), p

    gap = [dict(x) for x in good]
    gap[-1]["kv_len_before"] = gap[-1]["kv_len_before"] - 1
    p, _ = mod.check(_write(tmp_path / "gap", cap=cap, trace=gap))
    assert p, "a broken KV chain was accepted"

    silent = [dict(x) for x in good]
    silent[-1] = {k: v for k, v in silent[-1].items() if k != "kv_len_after"}
    p, _ = mod.check(_write(tmp_path / "silent", cap=cap, trace=silent))
    assert any("no kv_len_after" in x for x in p), p


def test_losslessness_uses_the_sidecars_of_both_arms(tmp_path):
    mod = _load_checker()
    problems, notes = mod.check(_write(tmp_path, cap=64, lossless=True))
    assert problems == [], problems
    assert any("losslessness OK" in n for n in notes), notes

    problems, _ = mod.check(_write(tmp_path / "bad", cap=64, lossless=False))
    assert any("losslessness" in p.lower() or "MISMATCH" in p for p in problems)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))

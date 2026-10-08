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


def _add_gated_pair(rows, tmp: Path, cap, *, kv="bfloat16",
                    spec_shortfall=0.375, with_probe=True, probe_error=None,
                    identity=True, positions=63):
    """Append a GATED pair (its KV dtype is in the checker's GATED_KV).

    The gate is the teacher-forced probe on the speculative arm, so a fixture
    without a probe is not "clean" -- it is UNVERIFIED, which the checker must
    refuse. Tests that want the clean case therefore need this pair present, and
    tests that want to exercise the refusal override it.
    """
    spec_ids = list(range(500, 500 + cap))
    tgt_ids = list(spec_ids) if identity else spec_ids[:5] + [999] + spec_ids[6:]
    rows.append(_row("CAP_bf16_spec", GAMMA, cap))
    rows.append(_row("CAP_bf16_tgt", 0, cap))
    (tmp / "per_token" / "CAP_bf16_spec.jsonl").write_text(
        "\n".join(json.dumps(x) for x in _spec_trace(cap)) + "\n")
    for rid, ids in (("CAP_bf16_spec", spec_ids), ("CAP_bf16_tgt", tgt_ids)):
        (tmp / "tokens" / f"{rid}.json").write_text(json.dumps({
            "run_id": rid, "generated_token_ids": ids,
            "token_gaps": [3.0] * len(ids),
            "weight_precision": "fp4", "kv_dtype": kv,
            "prompt_token_ids": [1, 2, 3], "doc_id": "d0",
            "context_length": CONTEXT, "max_new_tokens": cap}))
    if with_probe or probe_error:
        payload = ({"error": probe_error, "world_size": 8} if probe_error else {
            "world_size": 8, "measured_kv": kv, "tokens": positions + 1,
            "positions": positions, "max_shortfall": spec_shortfall,
            "worst_position": 1, "non_argmax": 0, "non_argmax_fraction": 0.0,
            "first_non_argmax_position": None, "failures": [],
            "noise_floor": {"max_abs_delta": 0.9, "control_max_abs_delta": 0.0,
                            "positions": positions}})
        (tmp / "tokens" / "CAP_bf16_spec.tflossless.json").write_text(
            json.dumps(payload))


def _write(tmp: Path, cap=64, *, trace=None, partner=True, sidecar_ids=None,
           spec_status="ok", target_status="ok", lossless=True, gated=True,
           gated_kw=None):
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
         # One gap per emitted id, in the shape the engine writes: the cap smoke
         # asserts this length relationship, so a fixture without it is
         # incomplete rather than exempt.
         "token_gaps": [3.0] * len(spec_ids),
         # The precision the engine MEASURED; the smoke asserts the campaign's
         # fp4 weights + nf4 cache on every row.
         "weight_precision": "fp4", "kv_dtype": "nf4",
         "prompt_token_ids": [1, 2, 3], "doc_id": "d0",
         "context_length": CONTEXT, "max_new_tokens": cap}))

    if partner:
        rows.append(_row("CAP_tgt", 0, cap, status=target_status))
        _tgt_ids = sidecar_ids if sidecar_ids else tgt_ids
        (tmp / "tokens" / "CAP_tgt.json").write_text(json.dumps(
            {"run_id": "CAP_tgt", "generated_token_ids": _tgt_ids,
             "token_gaps": [3.0] * len(_tgt_ids),
             "weight_precision": "fp4", "kv_dtype": "nf4",
             "prompt_token_ids": [1, 2, 3], "doc_id": "d0",
             "context_length": CONTEXT, "max_new_tokens": cap}))

    if gated:
        _add_gated_pair(rows, tmp, cap, **(gated_kw or {}))

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


def test_token_identity_between_the_arms_is_reported_but_does_not_gate(tmp_path):
    """PLAN RULE (6.1c): identity is evidence, not the gate.

    The two arms run the target in different forward SHAPES -- a packed
    (gamma+1)-token verify versus one token per step -- and in bf16 those shapes
    do not agree bit-for-bit, which is measured, not assumed. So a pair whose
    trajectories differ is not by itself a failure: the gate is whether each
    token the spec arm emitted is the target's own argmax given the stream's own
    prefix, which is what the probe measures.
    """
    mod = _load_checker()
    problems, notes = mod.check(_write(tmp_path, cap=64, lossless=True))
    assert problems == [], problems
    assert any("token identity" in n and "REPORTED" in n for n in notes), notes

    # A diverged pair whose tokens are all the target's argmax must PASS.
    problems, notes = mod.check(
        _write(tmp_path / "diverged", cap=64, lossless=False))
    assert problems == [], problems
    assert any("MISMATCH" in n and "REPORTED" in n for n in notes), notes


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))


# ---------------------------------------------------------------------------
# The plan's rule for ties, and the measured numerics, on the cap-smoke path
# ---------------------------------------------------------------------------

def _write_tie(tmp: Path, *, position: int, gap: float, oth_gap: float = 3.0):
    """A spec/target pair that diverges at `position`, with chosen gaps."""
    cap = 16
    spec_ids = list(range(100, 100 + cap))
    tgt_ids = list(spec_ids)
    tgt_ids[position] = 999
    (tmp / "per_token").mkdir(parents=True, exist_ok=True)
    (tmp / "tokens").mkdir(parents=True, exist_ok=True)
    rows = [_row("CAP_spec", GAMMA, cap), _row("CAP_tgt", 0, cap)]
    # The gated pair has to be present here too: without it the stage has no
    # pair whose KV dtype is gated, and the anti-vanishing guard (correctly)
    # refuses the whole run, which would mask what these tests are asserting.
    _add_gated_pair(rows, tmp, cap)
    tr = _spec_trace(cap)
    (tmp / "per_token" / "CAP_spec.jsonl").write_text(
        "\n".join(json.dumps(x) for x in tr) + "\n")
    sg = [oth_gap] * cap
    sg[position] = gap
    for rid, ids, gaps in (("CAP_spec", spec_ids, sg),
                           ("CAP_tgt", tgt_ids, [oth_gap] * cap)):
        (tmp / "tokens" / f"{rid}.json").write_text(json.dumps({
            "run_id": rid, "generated_token_ids": ids, "token_gaps": gaps,
            "weight_precision": "fp4", "kv_dtype": "nf4",
            "prompt_token_ids": [1, 2, 3], "doc_id": "d0",
            "context_length": CONTEXT, "max_new_tokens": cap}))
    with (tmp / "res.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)
    return tmp


def test_a_numeric_tie_passes_and_is_reported(tmp_path):
    """PLAN RULE: only MISMATCH fails a stage; a tie is reported."""
    mod = _load_checker()
    _write_tie(tmp_path, position=5, gap=0.01)
    problems, notes = mod.check(tmp_path / "res.csv")

    assert problems == [], problems
    joined = " ".join(notes)
    # The position must be named. It is reported inside the verdict detail
    # ("token 5: ...") rather than as a bare "[5]", so the assertion follows the
    # detail text; what matters is that the reader can see WHERE the tie was.
    assert "NUMERIC_TIE" in joined and "token 5" in joined, (
        f"the tie and its position must be reported: {notes}")
    assert "not a gate" in joined


def test_a_decided_identity_divergence_is_reported_not_failed(tmp_path):
    """The old rule failed this; the pre-registered rule reports it.

    Kept as a named test because it is the exact expectation that changed, so a
    future reader can see that it was deliberate rather than a regression.
    """
    mod = _load_checker()
    _write_tie(tmp_path, position=5, gap=3.0)
    problems, notes = mod.check(tmp_path / "res.csv")

    assert not any("MISMATCH" in p for p in problems), problems
    assert any("token identity" in n for n in notes), notes


def test_a_tie_is_recorded_from_either_arm(tmp_path):
    """The target's indifference is a property of the position."""
    mod = _load_checker()
    _write_tie(tmp_path, position=5, gap=3.0, oth_gap=3.0)
    # ... now make the TARGET arm indifferent there instead.
    p = tmp_path / "tokens" / "CAP_tgt.json"
    d = json.loads(p.read_text())
    d["token_gaps"][5] = 0.02
    p.write_text(json.dumps(d))

    problems, notes = mod.check(tmp_path / "res.csv")

    assert problems == [], problems
    assert any("NUMERIC_TIE" in n for n in notes)


def test_a_row_without_measured_numerics_fails(tmp_path):
    """The columns are load-bearing; an unstated precision is a failure."""
    mod = _load_checker()
    _write(tmp_path)
    p = tmp_path / "tokens" / "CAP_spec.json"
    d = json.loads(p.read_text())
    d.pop("weight_precision")
    d.pop("kv_dtype")
    p.write_text(json.dumps(d))

    problems, _notes = mod.check(tmp_path / "res.csv")

    assert any("no measured precision" in p for p in problems), problems


def test_an_unmeasured_or_undeclared_precision_fails(tmp_path):
    """The guard is "the declared configuration was actually built", not "nf4".

    bf16 KV is now a DECLARED configuration (the gated pair), so it is no longer
    a failure on its own -- but a dtype outside the declared set still is, and
    non-fp4 weights still are. Widening the allowlist this far keeps the original
    purpose: a row whose config claimed 4-bit while the loader skipped it, or a
    dtype nobody declared, must not pass silently.
    """
    mod = _load_checker()
    _write(tmp_path)
    p = tmp_path / "tokens" / "CAP_tgt.json"
    d = json.loads(p.read_text())
    d["weight_precision"] = "bfloat16"
    d["kv_dtype"] = "float8"
    p.write_text(json.dumps(d))

    problems, _notes = mod.check(tmp_path / "res.csv")

    assert any("weight_precision='bfloat16'" in q for q in problems), problems
    assert any("kv_dtype='float8'" in q for q in problems), problems


def test_the_gate_cannot_vanish_if_no_pair_is_gated(tmp_path):
    """Every pair report-only must FAIL: it would mean nothing was gated.

    The silent way this happens is a bf16 level whose kv_quant override was
    inert, so it ran as NF4 and was classified report-only. The stage would then
    "pass" having tested nothing -- the same failure class the measured-dtype
    assertion exists for.
    """
    mod = _load_checker()
    _write(tmp_path, gated_kw={"kv": "nf4"})
    problems, _notes = mod.check(tmp_path / "res.csv")
    assert any("GATED_KV" in q for q in problems), problems


# ---------------------------------------------------------------------------
# The teacher-forced gate (analysis plan 6.1c)
#
# The numbers in these fixtures are the MEASURED ones from
# results/mlsys/teacher_forced/, not invented values, so the tests document the
# controls they were built from:
#
#   bf16 positive (spec arm)          max shortfall 0.375  (8k) / 0.188 (32k)
#   NEG off-by-one KV, 8k bf16                              16.125
#   NEG unverified draft, 8k bf16                            6.750
#
# TOL_bf16 = 1.875 = 2 x 0.9375, the measured bf16 noise floor over 126
# positions. The gate sits between the positives (inside by 5x-10x) and the
# negatives (outside by 3.6x-8.6x).
# ---------------------------------------------------------------------------

def test_the_gate_accepts_the_measured_bf16_positive(tmp_path):
    mod = _load_checker()
    _write(tmp_path, gated_kw={"spec_shortfall": 0.375})
    problems, notes = mod.check(tmp_path / "res.csv")
    assert problems == [], problems
    joined = " ".join(notes)
    assert "teacher-forced max shortfall 0.3750 <= TOL_bf16" in joined, notes


def test_the_off_by_one_negative_control_fails_the_gate(tmp_path):
    """A KV/position misalignment must be caught, not tolerated.

    EVERY token is the target's argmax from one position earlier. The worst
    shortfall measured for this defect is 16.125 logits at 8k bf16, which is
    8.6x TOL_bf16.
    """
    mod = _load_checker()
    _write(tmp_path, gated_kw={"spec_shortfall": 16.125})
    problems, _notes = mod.check(tmp_path / "res.csv")
    assert any("teacher-forced MISMATCH" in p for p in problems), problems
    assert any("16.1250 > TOL_bf16 1.875" in p for p in problems), problems


def test_the_unverified_draft_negative_control_fails_the_gate(tmp_path):
    """Accepting the draft without verifying it must be caught.

    This is the tightest control: a GOOD draft is right most of the time, so the
    defect is rare but large. Its best case is the 6.75-logit unverified draft at
    8k bf16 -- still 3.6x TOL_bf16, which is the margin the whole rule rests on.
    """
    mod = _load_checker()
    _write(tmp_path, gated_kw={"spec_shortfall": 6.75})
    problems, _notes = mod.check(tmp_path / "res.csv")
    assert any("teacher-forced MISMATCH" in p for p in problems), problems
    assert any("6.7500 > TOL_bf16 1.875" in p for p in problems), problems


def test_a_gated_pair_without_a_probe_is_unverified_not_a_pass(tmp_path):
    """A missing probe must FAIL. "Not measured" is not "measured clean"."""
    mod = _load_checker()
    _write(tmp_path, gated_kw={"with_probe": False})
    problems, _notes = mod.check(tmp_path / "res.csv")
    assert any("UNVERIFIED, which is not a pass" in p for p in problems), problems


def test_a_gated_pair_whose_probe_errored_is_unverified(tmp_path):
    mod = _load_checker()
    _write(tmp_path, gated_kw={"probe_error": "CUDA out of memory"})
    problems, _notes = mod.check(tmp_path / "res.csv")
    assert any("CUDA out of memory" in p and "UNVERIFIED" in p
               for p in problems), problems


def test_the_nf4_pair_is_reported_and_never_gated(tmp_path):
    """NF4 is measured and printed, but a large shortfall there does NOT fail.

    NF4 KV noise alone flips 20% of argmaxes at this context, so gating it would
    fail legitimate runs; its separation from real defects is only 1.31x. The
    bf16 pair is still present, so the gate itself is intact.
    """
    mod = _load_checker()
    _write(tmp_path)
    (tmp_path / "tokens" / "CAP_spec.tflossless.json").write_text(json.dumps({
        "world_size": 8, "measured_kv": "nf4", "tokens": 64, "positions": 63,
        "max_shortfall": 20.44, "worst_position": 33, "non_argmax": 39,
        "non_argmax_fraction": 39 / 63.0, "first_non_argmax_position": 1,
        "failures": [],
        "noise_floor": {"max_abs_delta": 7.75, "control_max_abs_delta": 0.0,
                        "positions": 63}}))
    problems, notes = mod.check(tmp_path / "res.csv")
    assert problems == [], problems
    joined = " ".join(notes)
    assert "kv=nf4 [REPORT ONLY, not gated]" in joined, notes
    assert "20.4400" in joined and "max|delta|=7.75" in joined, notes


def test_the_pairing_key_separates_kv_precisions(tmp_path):
    """A bf16 spec arm must not be paired with an NF4 target-only arm.

    The key originally omitted the KV dtype, so at the same doc/context/cap a
    bf16 spec row would have been compared against the NF4 baseline -- measuring
    the numerics difference this stage exists to quantify instead of testing
    losslessness.
    """
    mod = _load_checker()
    _write(tmp_path)
    # Remove the bf16 partner, leaving only the NF4 target at that cap.
    (tmp_path / "tokens" / "CAP_bf16_tgt.json").unlink()
    import csv as _csv
    rows = [r for r in _csv.DictReader(open(tmp_path / "res.csv"))
            if r["run_id"] != "CAP_bf16_tgt"]
    with (tmp_path / "res.csv").open("w", newline="") as fh:
        w = _csv.DictWriter(fh, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)

    problems, _notes = mod.check(tmp_path / "res.csv")
    assert any("no target-only partner" in p and "kv=bfloat16" in p
               for p in problems), problems


def test_an_empty_kv_dtype_is_a_failure_not_an_unquantised_run(tmp_path):
    """The engine NAMES a bf16 run, so an empty kv_dtype means the measurement
    is missing -- and a missing measurement is not a clean one.

    Checked against the real artifacts: every sidecar in
    results/mlsys/lossless_repro/tokens/ carries a dtype, `bfloat16` for the
    non-quantised cells and `nf4` for the rest, so `""` is never what a healthy
    run writes. (An earlier shape of this change let the checker fall back to the
    teacher-forced probe's `measured_kv`; that was reverted once the evidence
    showed the engine already reports it, because two sources for one fact is a
    second thing to get wrong -- and the rehearsal stub, not the engine, turned
    out to be the component that reported `""`.)
    """
    mod = _load_checker()
    _write(tmp_path, gated_kw={"kv": "bfloat16"})
    for rid in ("CAP_bf16_spec", "CAP_bf16_tgt"):
        p = tmp_path / "tokens" / f"{rid}.json"
        d = json.loads(p.read_text())
        d["kv_dtype"] = ""
        p.write_text(json.dumps(d))

    problems, _notes = mod.check(tmp_path / "res.csv")
    assert any("no measured precision" in q for q in problems), problems

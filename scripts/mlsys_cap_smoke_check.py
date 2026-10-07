#!/usr/bin/env python3
"""Validate the generation cap on real output before the campaign depends on it.

The engine change that stops a verify round overshooting `max_new_tokens` has
never executed on a GPU. If it is wrong it fails QUIETLY: the generated length
changes, which breaks the `prompt + BOS + generated == context` identity,
invalidates the losslessness comparison (the target-only arm stops exactly on the
cap), and makes the acceptance denominator a function of acceptance.

This script is the gate on that change. It asserts, for every cell of
`configs/mlsys_engine_cap_smoke.yml`:

  1. `tokens_generated == max_new_tokens` EXACTLY, in both arms, at both caps.
  2. the final per-round trace record's `n_emitted` / `round_truncated` are
     consistent with the cap: the emitted tokens across all rounds sum to the
     cap, and a round is flagged truncated only when the budget cut it short.
  3. the speculative/target-only pair is token-identical (losslessness), which
     is the property the cap exists to keep checkable.

Exit non-zero if any assertion fails. The manifest stops before the
speculative stages on a non-zero exit.

Usage:
    python scripts/mlsys_cap_smoke_check.py --results results/mlsys/engine_cap_smoke.csv
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from src.analysis.losslessness import compare_generations, require_same_request


def _read_csv(path: Path) -> list[dict]:
    with path.open() as fh:
        return list(csv.DictReader(fh))


def _trace(results_dir: Path, run_id: str) -> list[dict]:
    p = results_dir / "per_token" / f"{run_id}.jsonl"
    if not p.exists():
        return []
    return [json.loads(line) for line in p.read_text().splitlines() if line.strip()]


# `generated` holds one token before the verify loop starts: the seed
# `cur_token` that the first round verifies. Every token-count identity in this
# file has to include it.
INITIAL_TOKEN = 1


def _sidecar(tokens_dir: Path, run_id: str) -> dict | None:
    """The token sidecar for a run, or None. Malformed counts as missing."""
    p = tokens_dir / f"{run_id}.json"
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text())
    except Exception:                                  # noqa: BLE001
        return None


def check(results_csv: Path, tokens_dir: Path | None = None) -> tuple[list[str], list[str]]:
    """Returns (problems, notes)."""
    problems: list[str] = []
    notes: list[str] = []
    rows = _read_csv(results_csv)
    results_dir = results_csv.resolve().parent
    tokens_dir = tokens_dir or (results_dir / "tokens")

    ok_rows = [r for r in rows if r.get("status") == "ok"]
    if not ok_rows:
        return ([f"no row completed successfully in {results_csv}"], notes)

    for r in ok_rows:
        rid, cap = r["run_id"], int(r["max_new_tokens"])
        gen = int(r["tokens_generated"])
        # A row with no verify rounds is a target-only run: there is nothing for
        # a per-round trace to record, so it is checked on the cap and on its
        # token sidecar instead. Demanding a trace of it asked the generator to
        # have produced a verify loop it does not have.
        is_spec = str(r.get("spec_steps", "")).strip() not in ("", "0")

        if gen != cap:
            problems.append(
                f"{rid}: tokens_generated={gen} but max_new_tokens={cap} "
                f"({gen - cap:+d}); the cap is not enforced"
            )
        else:
            notes.append(f"{rid}: generated exactly {cap}")

        # GAP ALIGNMENT, on every row of BOTH arms. `token_gaps` is a parallel
        # array to the emitted ids and the losslessness tie rule reads it by
        # position: a list one short reports the NEXT token's indifference, which
        # is how a real mismatch gets excused as a numerics tie. The engine
        # cannot be run on CPU (it requires CUDA streams), so this is the check
        # that proves the alignment on real hardware -- it runs on the first GPU
        # stage, before any expensive one.
        sidecar = _sidecar(tokens_dir, rid)
        if sidecar is None:
            problems.append(f"{rid}: no token sidecar, so the gap array cannot "
                            f"be checked for alignment")
        else:
            ids = sidecar.get("generated_token_ids") or []
            gaps = sidecar.get("token_gaps")
            if gaps is None:
                problems.append(f"{rid}: the sidecar carries no token_gaps; the "
                                f"tie rule would treat every divergence as a "
                                f"MISMATCH with no evidence either way")
            elif len(gaps) != len(ids):
                problems.append(
                    f"{rid}: token_gaps has {len(gaps)} entries for {len(ids)} "
                    f"emitted ids; the tie rule would read the wrong position"
                )
            else:
                notes.append(f"{rid}: {len(gaps)} gaps aligned with "
                             f"{len(ids)} emitted ids")

            # NUMERICS, measured by the engine off the loaded models and the
            # live cache. The campaign's stages run FP4 weights and an NF4 KV
            # cache, and every comparison in the plan rests on that: the vLLM
            # cross-check reports token agreement instead of scoring it BECAUSE
            # vLLM cannot reproduce this cache. A row whose config claimed 4-bit
            # while the loader skipped it would invalidate that reasoning
            # silently, so the values are asserted rather than assumed.
            wp = sidecar.get("weight_precision")
            kvd = sidecar.get("kv_dtype")
            if not wp or not kvd:
                problems.append(
                    f"{rid}: the sidecar carries no measured precision "
                    f"(weight_precision={wp!r} kv_dtype={kvd!r}); the numerics "
                    f"every comparison depends on are unstated")
            else:
                if wp != "fp4":
                    problems.append(
                        f"{rid}: weight_precision={wp!r}, expected 'fp4' — the "
                        f"campaign's cells are FP4 and a different precision "
                        f"makes them incomparable with the published numbers")
                if kvd != "nf4":
                    problems.append(
                        f"{rid}: kv_dtype={kvd!r}, expected 'nf4' — the KV cache "
                        f"is what the vLLM comparison cannot reproduce")
                if wp == "fp4" and kvd == "nf4":
                    notes.append(f"{rid}: numerics fp4 weights + nf4 KV "
                                 f"(measured)")

        # The sequence the engine built must be the rung.
        pt, seq = r.get("prompt_tokens"), r.get("sequence_tokens")
        if pt and seq and int(seq) != int(r["context_length"]):
            problems.append(
                f"{rid}: sequence_tokens={seq} != context_length="
                f"{r['context_length']}"
            )
        elif pt and seq:
            if int(pt) + 1 + gen != int(seq):
                problems.append(
                    f"{rid}: {pt} prompt + 1 BOS + {gen} generated != {seq}"
                )

        # The token sidecar, for BOTH arms: it is the record of what was
        # actually emitted, and for a target-only row it is the only evidence
        # available.
        side = _sidecar(tokens_dir, rid)
        if side is None:
            problems.append(f"{rid}: no token sidecar, so the emitted tokens "
                            f"cannot be checked")
        else:
            ids = side.get("generated_token_ids") or []
            if len(ids) != cap:
                problems.append(
                    f"{rid}: token sidecar holds {len(ids)} ids, expected the "
                    f"cap ({cap})"
                )
            if not is_spec:
                notes.append(
                    f"{rid}: target-only, {gen} generated == cap, sidecar has "
                    f"{len(ids)} ids"
                )
                continue

        tr = _trace(results_dir, rid)
        if not tr:
            problems.append(f"{rid}: no per-round trace, so the cap "
                            f"bookkeeping cannot be checked")
            continue

        # `generated` already holds ONE token before the loop starts (the seed
        # `cur_token` the first round verifies). So the cap is
        #   1 + (sum of accepted tokens emitted) + (one bonus per untruncated
        #       round)
        # and NOT `emitted + bonuses`, which was short by exactly that first
        # token.
        emitted = sum(int(x.get("n_emitted", x["n_acc"])) for x in tr)
        # The bonus token: one per round unless the budget cut the round short
        # (see _round_commit_plan). The truncated final round has no room for it.
        bonuses = sum(1 for x in tr if not x.get("round_truncated"))
        if emitted + bonuses + INITIAL_TOKEN != cap:
            problems.append(
                f"{rid}: trace accounts for {emitted} accepted + {bonuses} bonus "
                f"+ {INITIAL_TOKEN} seed = {emitted + bonuses + INITIAL_TOKEN} "
                f"tokens, expected {cap}"
            )
        truncated = [i for i, x in enumerate(tr) if x.get("round_truncated")]
        if truncated and truncated != [len(tr) - 1]:
            problems.append(
                f"{rid}: rounds {truncated} flagged truncated; only the FINAL "
                f"round can be cut short by the budget"
            )
        last = tr[-1]
        if int(last.get("n_emitted", last["n_acc"])) + (
                0 if last.get("round_truncated") else 1) < 1:
            problems.append(f"{rid}: final round committed nothing")

        # KV geometry. The context the next round reads must cover exactly the
        # tokens this round emitted -- no less (the next cur_token would then sit
        # at the wrong positional offset, silently corrupting every later round)
        # and no more (it would keep the verified-but-unemitted tail). A
        # truncated round is where this can go wrong, which is why it is
        # asserted here rather than assumed: the engine needs CUDA, so this is
        # the only place the arithmetic is ever checked end to end.
        kv_expected_prev = None
        for i, x in enumerate(tr):
            if "kv_len_after" not in x:
                problems.append(
                    f"{rid}: round {i} records no kv_len_after, so the KV "
                    f"length cannot be verified"
                )
                break
            kb, ka = int(x.get("kv_len_before", -1)), int(x["kv_len_after"])
            committed = int(x.get("n_committed", -1))
            emitted_i = int(x.get("n_emitted", x["n_acc"]))
            truncated_i = bool(x.get("round_truncated"))
            if committed < 0:
                problems.append(f"{rid}: round {i} records no n_committed")
                continue
            if ka != kb + committed:
                problems.append(
                    f"{rid}: round {i} KV {kb} -> {ka} but committed "
                    f"{committed} tokens"
                )
            # A truncated round commits nothing for the unemitted tail and gets
            # no bonus; an untruncated round commits the emitted prefix plus one
            # bonus token.
            want = emitted_i if truncated_i else emitted_i + 1
            if committed != want:
                problems.append(
                    f"{rid}: round {i} committed {committed}, expected {want} "
                    f"(emitted {emitted_i}, truncated={truncated_i})"
                )
            if truncated_i and ka - kb != emitted_i:
                problems.append(
                    f"{rid}: truncated round {i} covers {ka - kb} positions but "
                    f"emitted {emitted_i}; the KV must cover exactly the "
                    f"verified emitted prefix"
                )
            if kv_expected_prev is not None and kb != kv_expected_prev:
                problems.append(
                    f"{rid}: round {i} starts at KV {kb} but the previous round "
                    f"left {kv_expected_prev}"
                )
            kv_expected_prev = ka
        notes.append(
            f"{rid}: spec, {len(tr)} rounds, 1 seed + {emitted} accepted + "
            f"{bonuses} bonus = {cap}, final round n_acc={last['n_acc']} "
            f"n_emitted={last.get('n_emitted')} "
            f"truncated={last.get('round_truncated')} "
            f"kv {tr[0].get('kv_len_before')}->{kv_expected_prev}"
        )

    # Losslessness between each spec row and its target-only partner.
    by_key = {}
    for r in ok_rows:
        if str(r.get("spec_steps", "")).strip() in ("", "0"):
            by_key[(r.get("doc_id"), r.get("context_length"),
                    r.get("max_new_tokens"))] = r
    seen = 0
    for r in ok_rows:
        if str(r.get("spec_steps", "")).strip() in ("", "0"):
            continue
        partner = by_key.get((r.get("doc_id"), r.get("context_length"),
                              r.get("max_new_tokens")))
        if partner is None:
            problems.append(f"{r['run_id']}: no target-only partner at the same "
                            f"cap, so losslessness is unverified")
            continue
        bad = require_same_request(r, partner)
        if bad:
            problems.append(f"{r['run_id']}: pair refused: {'; '.join(bad)}")
            continue
        a = json.loads((tokens_dir / f"{r['run_id']}.json").read_text()) \
            if (tokens_dir / f"{r['run_id']}.json").exists() else None
        b = json.loads((tokens_dir / f"{partner['run_id']}.json").read_text()) \
            if (tokens_dir / f"{partner['run_id']}.json").exists() else None
        if a is None or b is None:
            problems.append(f"{r['run_id']}: missing token sidecar for a pair "
                            f"member")
            continue
        # Both arms of the smoke use the SAME cap, so the verdict must be
        # LOSSLESS (or a NUMERIC_TIE) rather than a prefix verdict: a prefix here
        # would mean the partner stopped short, which for this stage is itself a
        # failure.
        cap = int(r["max_new_tokens"])
        res = compare_generations(a["generated_token_ids"],
                                  b["generated_token_ids"],
                                  full_length=cap, min_prefix=cap,
                                  # Both arms' gaps, so a divergence at an
                                  # indifferent target is reported as the tie it
                                  # is rather than as an implementation defect.
                                  spec_gaps=a.get("token_gaps"),
                                  target_gaps=b.get("token_gaps"))
        seen += 1
        # PLAN RULE (B4 revision): a stage fails ONLY on MISMATCH. NUMERIC_TIE
        # passes and is REPORTED with its positions, because the target was
        # indifferent at the divergence and no implementation defect is implied.
        # Treating a tie as a failure here would fail a stage for being honest
        # about a numerics artefact.
        if res["verdict"] == "MISMATCH":
            problems.append(f"{r['run_id']}: MISMATCH vs {partner['run_id']} — "
                            f"{res['detail']}")
        elif res["verdict"] == "NUMERIC_TIE":
            notes.append(
                f"NUMERIC_TIE at {res['tie_positions']} vs "
                f"{partner['run_id']}: gaps {res['gap_at_divergence_spec']}/"
                f"{res['gap_at_divergence_target']} below "
                f"{res['tie_gap_threshold']} — reported, not a failure "
                f"({res['compared_tokens']} tokens compared)")
        elif res["verdict"] != "LOSSLESS":
            problems.append(f"{r['run_id']}: {res['verdict']} vs "
                            f"{partner['run_id']} — {res['detail']}")
        else:
            notes.append(f"losslessness OK at cap {cap} "
                         f"({res['compared_tokens']} tokens)")
    if seen == 0:
        problems.append("no speculative/target-only pair was checked")
    return problems, notes


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True)
    ap.add_argument("--tokens-dir", default=None)
    args = ap.parse_args()

    problems, notes = check(Path(args.results),
                            Path(args.tokens_dir) if args.tokens_dir else None)
    for n in notes:
        print(f"  ok    {n}")
    if problems:
        print("\nCAP SMOKE FAILED — the manifest must not start any speculative "
              "stage:")
        for p in problems:
            print(f"  FAIL  {p}")
        return 1
    print("\nCAP SMOKE PASSED: the cap is enforced in both arms and the pair is "
          "lossless.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

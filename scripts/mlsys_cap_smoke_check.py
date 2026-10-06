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
        if gen != cap:
            problems.append(
                f"{rid}: tokens_generated={gen} but max_new_tokens={cap} "
                f"({gen - cap:+d}); the cap is not enforced"
            )
        else:
            notes.append(f"{rid}: generated exactly {cap}")

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

        tr = _trace(results_dir, rid)
        if not tr:
            problems.append(f"{rid}: no per-round trace, so the cap "
                            f"bookkeeping cannot be checked")
            continue
        emitted = sum(int(x.get("n_emitted", x["n_acc"])) for x in tr)
        # The bonus token: one per round unless the budget cut the round short.
        # One bonus/resampled token per round that was NOT cut short: the
        # truncated final round has no room for it (see _round_commit_plan).
        bonuses = sum(1 for x in tr if not x.get("round_truncated"))
        if emitted + bonuses != cap:
            problems.append(
                f"{rid}: trace accounts for {emitted} accepted + {bonuses} bonus "
                f"= {emitted + bonuses} tokens, expected {cap}"
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
        notes.append(
            f"{rid}: {len(tr)} rounds, {emitted} accepted + {bonuses} bonus = {cap}, "
            f"final round n_acc={last['n_acc']} n_emitted={last.get('n_emitted')} "
            f"truncated={last.get('round_truncated')}"
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
        res = compare_generations(a["generated_token_ids"],
                                  b["generated_token_ids"],
                                  int(r["max_new_tokens"]))
        seen += 1
        if not res["lossless"]:
            problems.append(f"{r['run_id']}: NOT LOSSLESS vs "
                            f"{partner['run_id']} — {res['detail']}")
        else:
            notes.append(f"losslessness OK at cap {r['max_new_tokens']} "
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

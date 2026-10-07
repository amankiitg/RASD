#!/usr/bin/env python3
"""Check speculative decoding is lossless against the target-only run.

Pairs each speculative run with the target-only run for the *same document and
rung*, compares the two generated token streams, and reports the first mismatch
position. A mismatch fails the cell: its acceptance and speedup are not results.

Why the pairing is checked and not assumed: a spec run compared against a
target-only run for a different document, context or decoding contract would
still produce a green tick, and the check would be worthless. `require_same_request`
rejects a pair whose request fields disagree.

Input: the `tokens/` sidecars written by `run_experiment.py`, plus the result CSV
for the request fields.

Usage:
    python scripts/mlsys_losslessness.py --results results/mlsys/arm.csv \
        --tokens-dir results/mlsys/tokens --out results/mlsys/losslessness.csv
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from src.analysis.losslessness import (
    TIE_GAP, compare_generations, require_same_request, stage_requirement,
)


def pair_key(row: dict) -> tuple:
    """Identity of the *request*: same document, rung and prompt.

    `max_new_tokens` is deliberately NOT part of the key. The campaign's
    target-only baselines generate 128 tokens for 7 of the 10 documents and 1024
    for the other 3, so keying on the generation length would report NO_PAIR for
    seven correct cells — and NO_PAIR reads as "nothing to check" rather than as
    "the check is weaker than intended".
    """
    return (
        str(row.get("doc_id", "")),
        str(row.get("prompt_sha256", "")),
        str(row.get("context_length", "")),
    )


def is_target_only(row: dict) -> bool:
    return str(row.get("spec_steps", "")).strip() in ("", "0")


def load_rows(csv_path: Path) -> list[dict]:
    with csv_path.open() as fh:
        return [r for r in csv.DictReader(fh) if r.get("status") == "ok"]


def load_tokens(tokens_dir: Path, run_id: str) -> dict | None:
    path = tokens_dir / f"{run_id}.json"
    if not path.exists():
        return None
    return json.loads(path.read_text())


def check(csv_path: Path, tokens_dir: Path, full_length: int = 1024,
          min_prefix: int = 128) -> tuple:
    rows = load_rows(csv_path)
    # Keep the LONGEST partner per request: when both a full-length and a short
    # partner exist for the same document, the full one is the stronger check.
    by_key: dict[tuple, dict] = {}
    for r in rows:
        if not is_target_only(r):
            continue
        k = pair_key(r)
        cur = by_key.get(k)
        if cur is None or int(r.get("max_new_tokens") or 0) > int(
                cur.get("max_new_tokens") or 0):
            by_key[k] = r

    out = []
    for r in rows:
        if is_target_only(r):
            continue
        base = {
            "spec_run_id": r["run_id"],
            "doc_id": r.get("doc_id", ""),
            "context_length": r.get("context_length", ""),
            "spec_acceptance": r.get("acceptance_rate", ""),
            "spec_throughput_tps": r.get("throughput_tps", ""),
        }
        tgt = by_key.get(pair_key(r))
        if tgt is None:
            out.append({**base, "target_run_id": "", "lossless": "",
                        "verdict": "NO_PAIR",
                        "detail": "no target-only run with the same request"})
            continue
        base["target_run_id"] = tgt["run_id"]
        base["target_throughput_tps"] = tgt.get("throughput_tps", "")
        a = load_tokens(tokens_dir, r["run_id"])
        b = load_tokens(tokens_dir, tgt["run_id"])
        if a is None or b is None:
            missing = [n for n, t in ((r["run_id"], a), (tgt["run_id"], b))
                       if t is None]
            out.append({**base, "lossless": "", "verdict": "NO_TOKENS",
                        "detail": f"missing token sidecar for {missing}"})
            continue
        base["target_tokens_requested"] = tgt.get("max_new_tokens", "")
        # The guard needs the actual lengths to enforce "partner is not longer".
        r2, t2 = dict(r), dict(tgt)
        r2["_n_tokens"] = len(a["generated_token_ids"])
        t2["_n_tokens"] = len(b["generated_token_ids"])
        problems = require_same_request(r2, t2)
        if problems:
            out.append({**base, "verdict": "BAD_PAIR", "lossless": "",
                        "detail": "; ".join(problems)})
            continue
        res = compare_generations(
            a["generated_token_ids"], b["generated_token_ids"],
            full_length=full_length, min_prefix=min_prefix,
            # The tie rule needs the target's indifference at the divergence in
            # BOTH arms: it is the target's own top1-top2 gap, recorded per
            # emitted position by the engine. A sidecar written before that
            # existed has no `token_gaps`, which is reported as "no gap" (and
            # therefore a MISMATCH) rather than assumed indifferent.
            spec_gaps=a.get("token_gaps"), target_gaps=b.get("token_gaps"))
        if res["verdict"] == "NUMERIC_TIE":
            base["tie_positions"] = res["tie_positions"]
        out.append({**base, **res})
    return out, stage_requirement(out, full_length, min_prefix)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--results", required=True)
    p.add_argument("--tokens-dir", default=None,
                   help="Default: <results dir>/tokens")
    p.add_argument("--full-length", type=int, default=1024,
                   help="Generation length of the full-length target-only pairs")
    p.add_argument("--min-prefix", type=int, default=128,
                   help="Shortest verified prefix a cell may settle for")
    p.add_argument("--out", default=None)
    args = p.parse_args()

    csv_path = Path(args.results)
    tokens_dir = Path(args.tokens_dir) if args.tokens_dir else (
        csv_path.resolve().parent / "tokens")
    rows, req = check(csv_path, tokens_dir, args.full_length,
                      args.min_prefix)

    fields = ["spec_run_id", "target_run_id", "doc_id", "context_length",
              "verdict", "lossless", "lossless_full_prefix",
              "verified_prefix", "meets_min_prefix", "min_prefix",
              "first_mismatch_position",
              "spec_tokens", "target_tokens", "target_tokens_requested",
              "compared_tokens", "numeric_tie", "tie_gap_threshold",
              "tie_positions", "gap_at_divergence_spec",
              "gap_at_divergence_target",
              "spec_acceptance", "spec_throughput_tps",
              "target_throughput_tps", "detail"]
    out_path = Path(args.out) if args.out else csv_path.with_name(
        csv_path.stem + "_losslessness.csv")
    with out_path.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)

    for r in rows:
        print(f"  {r['verdict']:<20} {r['spec_run_id']:<42} "
              f"prefix={r.get('verified_prefix','')} {r.get('detail','')[:44]}")
    n_loss = sum(1 for r in rows if r["verdict"] == "LOSSLESS")
    n_pref = sum(1 for r in rows if str(r["verdict"]).startswith("LOSSLESS_PREFIX"))
    n_tie = sum(1 for r in rows if r["verdict"] == "NUMERIC_TIE")
    print(f"\n{n_loss} full-length LOSSLESS, {n_pref} prefix-verified, "
          f"{n_tie} NUMERIC_TIE, {len(rows) - n_loss - n_pref - n_tie} failed; "
          f"wrote {out_path}")
    if n_tie:
        print(f"  {n_tie} divergence(s) at a target top1-top2 gap below "
              f"{TIE_GAP}: reported, not counted as failures. A stage passing "
              f"on ties alone is not a clean pass.")
        for r in rows:
            if r["verdict"] == "NUMERIC_TIE":
                print(f"    tie at {r['tie_positions']} gaps "
                      f"{r.get('gap_at_divergence_spec')}/"
                      f"{r.get('gap_at_divergence_target')} "
                      f"({r['spec_run_id']})")
    print(f"  requirement: every cell verified over >= {args.min_prefix} "
          f"tokens, and all {req['full_length_cells']} full-length pairs "
          f"LOSSLESS over {args.full_length}")
    if not req["ok"]:
        for f in req["failures"]:
            print(f"  FAIL  {f}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

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

from src.analysis.losslessness import compare_generations, require_same_request


def pair_key(row: dict) -> tuple:
    """Identity of the *request*: same document, rung, prompt and generation."""
    return (
        str(row.get("doc_id", "")),
        str(row.get("prompt_sha256", "")),
        str(row.get("context_length", "")),
        str(row.get("max_new_tokens", "")),
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


def check(csv_path: Path, tokens_dir: Path) -> list[dict]:
    rows = load_rows(csv_path)
    by_key: dict[tuple, dict] = {}
    for r in rows:
        if is_target_only(r):
            by_key[pair_key(r)] = r

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
        problems = require_same_request(r, tgt)
        if problems:
            out.append({**base, "lossless": "", "verdict": "BAD_PAIR",
                        "detail": "; ".join(problems)})
            continue
        a = load_tokens(tokens_dir, r["run_id"])
        b = load_tokens(tokens_dir, tgt["run_id"])
        if a is None or b is None:
            missing = [n for n, t in ((r["run_id"], a), (tgt["run_id"], b))
                       if t is None]
            out.append({**base, "lossless": "", "verdict": "NO_TOKENS",
                        "detail": f"missing token sidecar for {missing}"})
            continue
        req = None
        if str(r.get("max_new_tokens", "")).strip():
            req = int(r["max_new_tokens"])
        res = compare_generations(
            a["generated_token_ids"], b["generated_token_ids"], req)
        out.append({**base, **res,
                    "verdict": "LOSSLESS" if res["lossless"] else "MISMATCH"})
    return out


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--results", required=True)
    p.add_argument("--tokens-dir", default=None,
                   help="Default: <results dir>/tokens")
    p.add_argument("--out", default=None)
    args = p.parse_args()

    csv_path = Path(args.results)
    tokens_dir = Path(args.tokens_dir) if args.tokens_dir else (
        csv_path.resolve().parent / "tokens")
    rows = check(csv_path, tokens_dir)

    fields = ["spec_run_id", "target_run_id", "doc_id", "context_length",
              "verdict", "lossless", "first_mismatch_position",
              "spec_tokens", "target_tokens", "compared_tokens",
              "spec_acceptance", "spec_throughput_tps",
              "target_throughput_tps", "detail"]
    out_path = Path(args.out) if args.out else csv_path.with_name(
        csv_path.stem + "_losslessness.csv")
    with out_path.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)

    bad = [r for r in rows if r["verdict"] != "LOSSLESS"]
    for r in rows:
        print(f"  {r['verdict']:<9} {r['spec_run_id']:<44} "
              f"{r.get('detail','')[:60]}")
    print(f"\n{len(rows) - len(bad)}/{len(rows)} cells lossless; wrote {out_path}")
    if bad:
        print(f"NOT LOSSLESS: {[r['spec_run_id'] for r in bad]}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

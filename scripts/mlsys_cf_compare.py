#!/usr/bin/env python3
"""Compare two rows' generated token streams: first divergence, and the top-2 gap there.

WHY THIS IS A SCRIPT AND NOT A SENTENCE IN A REPORT. A counterfactual that changes the
reduction order is expected to diverge from its baseline; that is not a bug, and the only
question a reader can actually check is *where* it diverges and *whether the position was a
near-tie*. Both are properties of the recorded sidecars, so they are computed rather than
asserted: `generated_token_ids` gives the streams, and `token_gaps` -- recorded per emitted
token by the engine -- gives the top-2 logit margin at each position, which is the evidence
that separates "floating-point tie broken the other way" from "this changed the model's
behaviour".

Usage:
    python scripts/mlsys_cf_compare.py TOKENS_DIR RUN_A RUN_B
Exit code 0 always; the verdict is printed, not thrown.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def load(tokens_dir: Path, run_id: str) -> dict:
    p = tokens_dir / f"{run_id}.json"
    if not p.exists():
        alt = sorted(tokens_dir.glob(f"{run_id}*.json"))
        if not alt:
            raise SystemExit(f"no sidecar for {run_id} in {tokens_dir}")
        p = alt[0]
    return json.loads(p.read_text())


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("tokens_dir", type=Path)
    ap.add_argument("run_a")
    ap.add_argument("run_b")
    args = ap.parse_args()

    a, b = load(args.tokens_dir, args.run_a), load(args.tokens_dir, args.run_b)
    ia = a.get("generated_token_ids") or []
    ib = b.get("generated_token_ids") or []
    print(f"A {a.get('run_id')}: {len(ia)} tokens, sha {(a.get('generated_tokens_sha256') or '')[:16]}")
    print(f"B {b.get('run_id')}: {len(ib)} tokens, sha {(b.get('generated_tokens_sha256') or '')[:16]}")
    for label, rec in (("A", a), ("B", b)):
        print(f"   {label} recorded: weight_precision={rec.get('weight_precision')} "
              f"kv_dtype={rec.get('kv_dtype')} T={rec.get('temperature')}")

    if ia == ib:
        print("\nVERDICT: BIT-IDENTICAL token streams.")
        return 0

    first = next((i for i, (x, y) in enumerate(zip(ia, ib)) if x != y), min(len(ia), len(ib)))
    print(f"\nVERDICT: DIVERGENT at token index {first} "
          f"(A={ia[first] if first < len(ia) else None}, B={ib[first] if first < len(ib) else None}) "
          f"after {first} identical tokens.")
    ga = (a.get("token_gaps") or [])
    gb = (b.get("token_gaps") or [])
    for label, g in (("A", ga), ("B", gb)):
        window = g[max(0, first - 1): first + 2]
        print(f"   {label} token_gaps around the divergence (idx {max(0, first-1)}..{first+1}): {window}")
    for label, g in (("A", ga), ("B", gb)):
        vals = sorted(abs(float(x)) for x in g if isinstance(x, (int, float)))
        if not vals:
            continue
        gd = abs(float(g[first])) if first < len(g) and isinstance(g[first], (int, float)) else None
        med = vals[len(vals) // 2]
        pct = (100.0 * sum(1 for v in vals if v <= (gd if gd is not None else -1)) / len(vals)
               if gd is not None else float("nan"))
        print(f"   {label} row's own gap distribution: median {med:.3f}, min {vals[0]:.3f}, "
              f"max {vals[-1]:.3f}; the divergent position sits at {pct:.0f}th percentile")
    print("   reading: the unit of token_gaps is the engine's recorded top-2 margin and is NOT "
          "calibrated here, so the check is RELATIVE -- a divergence at a gap far below that "
          "row's own median is a near-tie being broken; a divergence at a typical or large gap "
          "means the arm moved the distribution, not just the rounding.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Score RULER niah_single runs by checking whether the generated text
contains the seeded magic number.

Reads from a directory containing paired sidecars per run:
    <run_id>.ruler_niah.json   (needle metadata, written at prompt-build time)
    <run_id>.generated.txt     (decoded model output, written post-generation)

Writes a CSV with columns:
    run_id, context_length, seed, magic_number, needle_position_frac,
    found, generated_excerpt
where `found` is 1 if the magic number string appears in the generated
text, 0 otherwise.

Usage:
    python scripts/score_ruler_niah.py \\
        --sidecar-dir results/final/ruler \\
        --out results/final/ruler_niah_scores.csv
"""
from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path


def score_one(needle_json: Path) -> dict:
    """Score a single run by checking generated.txt for the magic number."""
    meta = json.loads(needle_json.read_text())
    run_id = meta["run_id"]
    magic = str(meta["magic_number"])
    gen_path = needle_json.parent / f"{run_id}.generated.txt"
    if not gen_path.exists():
        return {
            "run_id": run_id,
            "context_length": meta["context_length"],
            "seed": meta["seed"],
            "magic_number": magic,
            "needle_position_frac": meta["needle_position_frac"],
            "found": -1,  # generated text missing
            "generated_excerpt": "",
        }
    gen_text = gen_path.read_text()
    # Match the magic number as a whole-number token (not as a substring
    # of a longer number — "42" should not match "421").
    found = 1 if re.search(rf"(?<!\d){re.escape(magic)}(?!\d)", gen_text) else 0
    return {
        "run_id":               run_id,
        "context_length":       meta["context_length"],
        "seed":                 meta["seed"],
        "magic_number":         magic,
        "needle_position_frac": round(meta["needle_position_frac"], 4),
        "found":                found,
        "generated_excerpt":    gen_text[:200].replace("\n", " "),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--sidecar-dir", required=True,
                   help="Directory with <run_id>.ruler_niah.json + "
                        "<run_id>.generated.txt files from run_experiment "
                        "with --prompt-source=ruler_niah.")
    p.add_argument("--out", default="results/final/ruler_niah_scores.csv")
    args = p.parse_args()

    sidecar_dir = Path(args.sidecar_dir)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    needle_jsons = sorted(sidecar_dir.glob("*.ruler_niah.json"))
    if not needle_jsons:
        raise SystemExit(f"No *.ruler_niah.json files in {sidecar_dir}")

    rows = [score_one(p) for p in needle_jsons]

    with open(out_path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=[
            "run_id", "context_length", "seed", "magic_number",
            "needle_position_frac", "found", "generated_excerpt",
        ])
        w.writeheader()
        for r in rows:
            w.writerow(r)

    n_found = sum(r["found"] == 1 for r in rows)
    n_total = sum(r["found"] in (0, 1) for r in rows)
    n_missing = sum(r["found"] == -1 for r in rows)
    print(f"Scored {len(rows)} runs -> {out_path}")
    print(f"  found:   {n_found}/{n_total}")
    if n_missing:
        print(f"  missing generated.txt: {n_missing}")

    # Per-context breakdown
    by_ctx: dict[int, list[int]] = {}
    for r in rows:
        if r["found"] in (0, 1):
            by_ctx.setdefault(r["context_length"], []).append(r["found"])
    print()
    print(f"{'ctx':>10}  {'found':>6}  {'accuracy':>9}")
    for ctx in sorted(by_ctx):
        hits = sum(by_ctx[ctx])
        n = len(by_ctx[ctx])
        print(f"{ctx:>10}  {hits:>3}/{n:<2}   {hits/n:>9.2%}")


if __name__ == "__main__":
    raise SystemExit(main())

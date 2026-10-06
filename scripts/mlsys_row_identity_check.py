#!/usr/bin/env python3
"""Every result row must satisfy one identity:

    prompt_tokens + 1 (leading BOS) + tokens_generated == sequence_tokens

`sequence_tokens` is the number of positions the engine actually built and
attended over: the prompt the run was given, the BOS `generate_text` prepends,
and the tokens that were ACTUALLY emitted. It is what makes `context_length` and
"the rung" mean the same thing across tables, and it is the number a reader
compares against the native window.

It has been wrong in two ways on this project, both silent:

  * it was recorded from the DOCUMENT PLAN (`len(prompt) + 1 + gen_tokens`),
    where `gen_tokens` is the rung's generation length. An arm that stops early,
    or a short arm that generates 128 tokens into the rung's prompt, therefore
    claimed a sequence length it never produced;
  * nothing checked it, so a row whose sequence disagreed with its own parts was
    reported as a normal rung.

This runs on every stage's CSV, not just the cap smoke, because the identity
holds for every arm: a target-only row and a speculative row at the same rung
must agree, and a short baseline must say what it really did.

Exit codes: 0 all rows satisfy the identity; 1 at least one row does not, or a
row is missing the fields needed to check it; 2 the CSV could not be read.
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

INITIAL_TOKEN = 1          # the leading BOS generate_text adds


def check(path: Path) -> tuple[list[str], int]:
    rows = list(csv.DictReader(path.open()))
    problems: list[str] = []
    checked = 0
    for r in rows:
        rid = r.get("run_id", "?")
        if str(r.get("status", "")).strip() != "ok":
            # An error row has no metrics to check. It is already a failure for
            # the stage's row check; not double-reporting it here keeps the
            # message about the identity.
            continue
        try:
            prompt = int(str(r.get("prompt_tokens", "")).strip())
            gen = int(str(r.get("tokens_generated", "")).strip())
            seq = int(str(r.get("sequence_tokens", "")).strip())
        except (TypeError, ValueError):
            problems.append(
                f"{rid}: cannot check the identity -- prompt_tokens="
                f"{r.get('prompt_tokens')!r} tokens_generated="
                f"{r.get('tokens_generated')!r} sequence_tokens="
                f"{r.get('sequence_tokens')!r}")
            continue
        checked += 1
        want = prompt + INITIAL_TOKEN + gen
        if seq != want:
            problems.append(
                f"{rid}: sequence_tokens={seq} but prompt {prompt} + "
                f"{INITIAL_TOKEN} BOS + generated {gen} = {want} "
                f"({seq - want:+d})")
    return problems, checked


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True)
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()
    path = Path(args.results)
    if not path.exists():
        print(f"no CSV at {path}", file=sys.stderr)
        return 2
    problems, checked = check(path)
    if problems:
        print(f"SEQUENCE IDENTITY FAILED in {path}: {len(problems)} row(s) of "
              f"{checked} checked")
        for p in problems[:20]:
            print(f"  {p}")
        return 1
    if not args.quiet:
        print(f"  sequence identity holds for all {checked} ok row(s) in "
              f"{path.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

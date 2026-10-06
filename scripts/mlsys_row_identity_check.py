#!/usr/bin/env python3
"""Every result row must satisfy one identity:

    prompt_tokens + 1 (leading BOS) + tokens_generated == sequence_tokens

`sequence_tokens` is the number of positions the engine actually held and
attended over: the prompt it was given, the BOS `generate_text` prepends, and the
tokens it emitted. It is what makes `context_length` and "the rung" mean the same
thing across tables, and it is the number a reader compares against the native
window.

THE THREE NUMBERS MUST COME FROM THREE PLACES
---------------------------------------------
An identity is only a check if its two sides are obtained independently. Written
as `sequence_tokens = prompt_tokens + 1 + tokens_generated` inside the runner,
and then asserted here as the same expression, it is a tautology: it cannot fail,
so it verifies nothing. The three sources are therefore:

  * `prompt_tokens`     -- recorded by the runner from the prompt's token ids;
  * `tokens_generated`  -- taken HERE from the run's token SIDECAR
    (`<results>/tokens/<run_id>.json`), the list of ids the engine emitted, and
    cross-checked against the CSV's own count;
  * `sequence_tokens`   -- recorded by the ENGINE, from the length of the final
    sequence tensor (`int(generated_ids.shape[1])`), before any slicing.

A divergence between them is a real defect: a BOS counted twice, a prompt that is
not the prompt the engine saw, or a generation loop that stopped somewhere other
than where it reported.

Exit codes: 0 the identity holds for every checked row; 1 at least one row does
not, or a row lacks the evidence to be checked; 2 the CSV could not be read.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

INITIAL_TOKEN = 1          # the leading BOS generate_text adds


def _sidecar(tokens_dir: Path, run_id: str):
    p = tokens_dir / f"{run_id}.json"
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text())
    except Exception:                                  # noqa: BLE001
        return None


def check(path: Path, tokens_dir: Path | None = None) -> tuple[list[str], int]:
    rows = list(csv.DictReader(path.open()))
    tokens_dir = tokens_dir or (path.resolve().parent / "tokens")
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
            gen_csv = int(str(r.get("tokens_generated", "")).strip())
            seq = int(str(r.get("sequence_tokens", "")).strip())
        except (TypeError, ValueError):
            problems.append(
                f"{rid}: cannot check the identity -- prompt_tokens="
                f"{r.get('prompt_tokens')!r} tokens_generated="
                f"{r.get('tokens_generated')!r} sequence_tokens="
                f"{r.get('sequence_tokens')!r}")
            continue
        checked += 1

        # The emitted count comes from the sidecar, not from the CSV: the CSV
        # number is one of the things being checked.
        side = _sidecar(tokens_dir, rid)
        if side is None:
            problems.append(
                f"{rid}: no token sidecar in {tokens_dir}, so the emitted count "
                f"is not independently available and the identity cannot be "
                f"checked")
            continue
        ids = side.get("generated_token_ids")
        if ids is None:
            problems.append(f"{rid}: the sidecar has no generated_token_ids")
            continue
        gen = len(ids)
        if gen != gen_csv:
            problems.append(
                f"{rid}: the CSV reports tokens_generated={gen_csv} but the "
                f"sidecar holds {gen} ids")

        want = prompt + INITIAL_TOKEN + gen
        if seq != want:
            problems.append(
                f"{rid}: sequence_tokens={seq} (engine) vs prompt {prompt} + "
                f"{INITIAL_TOKEN} BOS + {gen} emitted (sidecar) = {want} "
                f"({seq - want:+d})")
    return problems, checked


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True)
    ap.add_argument("--tokens-dir", default=None,
                    help="Default: <results dir>/tokens")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()
    path = Path(args.results)
    if not path.exists():
        print(f"no CSV at {path}", file=sys.stderr)
        return 2
    problems, checked = check(path,
                              Path(args.tokens_dir) if args.tokens_dir else None)
    if problems:
        print(f"SEQUENCE IDENTITY FAILED in {path}: {len(problems)} problem(s) "
              f"among {checked} checked row(s)")
        for p in problems[:20]:
            print(f"  {p}")
        return 1
    if not args.quiet:
        print(f"  sequence identity holds for all {checked} ok row(s) in "
              f"{path.name} (engine length vs prompt + BOS + sidecar ids)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

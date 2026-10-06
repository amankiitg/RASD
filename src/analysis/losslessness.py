"""Speculative decoding must be exactly lossless under greedy decoding.

With greedy verification, speculative decoding is an exact optimisation: the
output token sequence must be **identical** to what the target alone would have
produced for the same prompt. Any divergence is an implementation defect, not a
quality trade-off, because the target's choice at every position is what defines
the output.

This matters here specifically. This project has already published acceptance
numbers for cells whose *target was broken* — the ARM4 f2/f4 rungs replaced
Llama-3.1-8B's shipped rope block and the target emitted EOS at token 0 — and
those numbers looked like ordinary acceptance collapse. Acceptance alone cannot
distinguish "the draft agrees with a healthy target" from "the target and draft
are both broken in the same way". Token-level agreement against the target's own
greedy output can.

Comparison is on token IDs, never on decoded text: text is not injective
(`decode` then `encode` need not round-trip), so matching strings would neither
prove nor disprove agreement.
"""
from __future__ import annotations

from typing import List, Optional, Sequence


def first_mismatch(a: Sequence[int], b: Sequence[int]) -> Optional[int]:
    """Index of the first differing token, or None if one is a prefix of the other.

    A length difference is a mismatch when the shorter is an exact prefix: the
    runs were asked for the same number of tokens, so a short run means
    generation stopped early, which is itself a divergence. The returned index
    is then the first position the shorter run did not produce.
    """
    n = min(len(a), len(b))
    for i in range(n):
        if a[i] != b[i]:
            return i
    if len(a) != len(b):
        return n
    return None


def compare_generations(
    spec_ids: Sequence[int],
    target_ids: Sequence[int],
    full_length: int | None = None,
    min_prefix: int = 128,
) -> dict:
    """Losslessness verdict for one (speculative, target-only) pair.

    The two arms do NOT have to generate the same number of tokens. The campaign
    uses three target-only partners at the full 1024 tokens and seven at 128
    tokens, because a throughput RATE needs a steady state while losslessness
    needs an equal length. So the comparison is over the common prefix and the
    verdict says which one it is:

      LOSSLESS           the partner reached `full_length` and every token matches
      LOSSLESS_PREFIX_n  the partner produced n < full_length tokens and the first
                         n tokens match
      MISMATCH           the first divergence position, with both tokens named

    `verified_prefix` is the length actually checked, and `meets_min_prefix` says
    whether it clears `min_prefix`. A stage requiring losslessness needs every
    speculative cell to clear the minimum and the full-length pairs to be
    lossless over `full_length`.

    Comparison is on token IDs, never on decoded text: text is not injective.
    """
    n_common = min(len(spec_ids), len(target_ids))
    pos = first_mismatch(spec_ids[:n_common], target_ids[:n_common])

    if pos is not None:
        verdict = "MISMATCH"
    elif full_length is not None and len(target_ids) >= full_length \
            and len(spec_ids) >= full_length:
        verdict = "LOSSLESS"
    elif full_length is None and len(spec_ids) == len(target_ids):
        verdict = "LOSSLESS"
    else:
        verdict = f"LOSSLESS_PREFIX_{n_common}"

    out = {
        "verdict": verdict,
        "lossless": bool(verdict == "LOSSLESS"),
        "lossless_full_prefix": bool(verdict.startswith("LOSSLESS")),
        "verified_prefix": int(n_common),
        "meets_min_prefix": bool(n_common >= min_prefix and verdict.startswith("LOSSLESS")),
        "first_mismatch_position": "" if pos is None else int(pos),
        "spec_tokens": len(spec_ids),
        "target_tokens": len(target_ids),
        "compared_tokens": int(n_common),
        "min_prefix": int(min_prefix),
    }
    if pos is not None:
        out["detail"] = (
            f"token {pos}: spec={spec_ids[pos]} target={target_ids[pos]}"
            if pos < n_common else f"runs diverged by length at {pos}"
        )
    elif verdict == "LOSSLESS":
        out["detail"] = f"identical over the full {full_length or n_common} tokens"
    else:
        out["detail"] = (
            f"identical over the {n_common}-token common prefix "
            f"(partner produced {len(target_ids)})"
        )
    return out


def stage_requirement(rows: list[dict], full_length: int,
                      min_prefix: int = 128) -> dict:
    """Apply the plan's rule for a stage that declares `losslessness: required`.

    Every speculative cell must have a verified prefix of at least `min_prefix`,
    AND every cell whose partner reached `full_length` must be LOSSLESS over
    `full_length`. Returns the failures, so a stage can report them rather than
    trusting an aggregate.
    """
    failures = []
    full_cells = 0
    for r in rows:
        v = str(r.get("verdict", ""))
        if not v:
            continue
        if not v.startswith("LOSSLESS"):
            failures.append(f"{r.get('spec_run_id')}: {v} at "
                            f"{r.get('first_mismatch_position')}")
            continue
        if int(r.get("verified_prefix") or 0) < min_prefix:
            failures.append(f"{r.get('spec_run_id')}: verified prefix "
                            f"{r.get('verified_prefix')} < {min_prefix}")
        if int(r.get("target_tokens") or 0) >= full_length:
            full_cells += 1
            if v != "LOSSLESS":
                failures.append(f"{r.get('spec_run_id')}: partner reached "
                                f"{full_length} but verdict is {v}")
    if full_cells == 0:
        failures.append(f"no pair had a {full_length}-token partner, so the "
                        f"full-length check did not happen")
    return {"failures": failures, "full_length_cells": full_cells,
            "ok": not failures}


def require_same_request(spec_row: dict, target_row: dict) -> list[str]:
    """Reasons a spec/target-only pair may not be compared at all.

    Comparing a speculative run against a target-only run with a different
    prompt, context or decoding contract would make the losslessness result
    meaningless while still producing a green tick, so the request fields are
    checked explicitly rather than assumed.
    """
    problems: List[str] = []
    # The contract fields, not just the request identity. Losslessness is only
    # defined for greedy decoding with EOS ignored, and a pair that differs in
    # sampling or rope is not the same experiment even when the prompt matches.
    #
    # `max_new_tokens` is deliberately NOT compared: the campaign's short
    # target-only baselines generate 128 tokens against a 1024-token speculative
    # run, which is exactly the case the prefix verdict exists for. What matters
    # is that the partner is not LONGER than the run it checks.
    for field in ("prompt_sha256", "prompt_tokens", "context_length",
                  "temperature", "top_p", "ignore_eos",
                  "rope_type", "rope_factor", "rope_anchor_base",
                  "target_revision", "draft_revision"):
        a, b = spec_row.get(field), target_row.get(field)
        if a != b:
            problems.append(f"{field}: spec={a!r} target={b!r}")
    if str(spec_row.get("spec_steps")) in ("", "0"):
        problems.append("spec row has spec_steps=0 (it is a target-only run)")
    if str(target_row.get("spec_steps")) not in ("", "0"):
        problems.append(f"target row has spec_steps={target_row.get('spec_steps')}")

    spec_len, tgt_len = spec_row.get("_n_tokens"), target_row.get("_n_tokens")
    if spec_len is not None and tgt_len is not None and int(tgt_len) > int(spec_len):
        problems.append(
            f"partner generated more tokens than the run it checks "
            f"({tgt_len} > {spec_len}); the prefix comparison would be over a "
            f"window the speculative run does not have"
        )
    return problems

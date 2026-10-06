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
    requested_tokens: int | None = None,
) -> dict:
    """Losslessness verdict for one (speculative, target-only) pair.

    `requested_tokens` is the generation length both runs were asked for. A run
    that produced FEWER tokens than requested is incomplete: the comparison
    window would be shorter than the plan fixed, and the shortfall usually means
    generation stopped early, which is itself a divergence. A run that produced
    more is not treated as a failure — see below — but is reported.
    """
    pos = first_mismatch(spec_ids, target_ids)
    short = None
    if requested_tokens is not None:
        if len(spec_ids) < requested_tokens or len(target_ids) < requested_tokens:
            short = min(len(spec_ids), len(target_ids))
    # A speculative run may legitimately emit more than the cap if the engine
    # commits a whole verify round; the target-only arm stops exactly on the
    # cap. Comparing over their common prefix is the correct comparison, and
    # the overrun is recorded because it also shifts the throughput denominator.
    overrun = (max(len(spec_ids), len(target_ids)) - requested_tokens
               if requested_tokens is not None else 0)
    lossless = (pos is None) and (short is None)
    out = {
        "lossless": bool(lossless),
        "first_mismatch_position": "" if pos is None else int(pos),
        "spec_tokens": len(spec_ids),
        "target_tokens": len(target_ids),
        "compared_tokens": min(len(spec_ids), len(target_ids)),
        "spec_overrun_tokens": max(0, len(spec_ids) - (requested_tokens or len(spec_ids))),
        "target_overrun_tokens": max(0, len(target_ids) - (requested_tokens or len(target_ids))),
        "length_overrun": max(0, overrun),
    }
    if pos is not None:
        out["detail"] = (
            f"token {pos}: spec={spec_ids[pos]} target={target_ids[pos]}"
            if pos < min(len(spec_ids), len(target_ids))
            else f"runs diverged by length at {pos}"
        )
    elif short is not None:
        out["detail"] = (
            f"incomplete generation: asked for {requested_tokens}, "
            f"shortest run produced {short}"
        )
    else:
        out["detail"] = "identical over the full generation"
    return out


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
    for field in ("prompt_sha256", "prompt_tokens", "context_length",
                  "max_new_tokens", "temperature", "top_p", "ignore_eos",
                  "rope_type", "rope_factor", "rope_anchor_base",
                  "target_revision", "draft_revision"):
        a, b = spec_row.get(field), target_row.get(field)
        if a != b:
            problems.append(f"{field}: spec={a!r} target={b!r}")
    if str(spec_row.get("spec_steps")) in ("", "0"):
        problems.append("spec row has spec_steps=0 (it is a target-only run)")
    if str(target_row.get("spec_steps")) not in ("", "0"):
        problems.append(f"target row has spec_steps={target_row.get('spec_steps')}")
    return problems

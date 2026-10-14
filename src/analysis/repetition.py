"""Repetition and periodicity of a greedy generation, and acceptance by window.

WHY THIS EXISTS. On 2026-10-10 an audit of the campaign's own per-token traces found
that the headline acceptance rate was measuring a greedy degeneracy rather than the
speculation mechanism:

  * acceptance reached EXACTLY 1.000 in every 128-token window from token 128 on, for
    the top rows of natural_f1_128k (0.970, 0.963, 0.960, 0.944);
  * those were exactly the windows where 100% of tokens sat inside a repeated
    16-gram, and the whole-row repeat share reached 0.974;
  * dominant cycle periods were 14, 20, 26;
  * the highest-acceptance row's last 200 tokens were " He was deceived." fifty
    times.

None of that was caught, for two separate reasons. The cap-smoke checker tests
repetition not at all, and where a rule DOES exist -- the coherence gate's
`_share_reason` -- it is purely RELATIVE to the paired baseline (GEN_SHARE_TOLERANCE
= 2.0) with no absolute ceiling on repetition. A greedy loop repeats in BOTH arms, so
that ratio sits near 1 and the rule cannot fire by construction. Measured on the real
pairs: 1.07x, 1.32x, 1.80x -- all PASS.

So this module adds the two things that were missing: an ABSOLUTE ceiling on
repetition, and a periodicity statistic that says "this generation is a p-periodic
cycle from token N to the end" instead of "repeat share 0.97", because the second is
legible and the first is not.

It is deliberately dependency-free (no torch, no numpy) so the gate, the stage
checkers and the offline rescore all use ONE implementation -- three sites computing
"repeat share" under three conventions is how a campaign comes to disagree with
itself about what a loop is.
"""
from __future__ import annotations

import collections
from typing import Iterable, Sequence

#: Absolute ceiling on word-trigram repetition, as a share. Measured: real PG-19
#: prose sits at 0.00-0.30 even for the model's worst non-degenerate rows, and the
#: loop rows sit at 0.65-0.97. 0.60 separates them with room on both sides and it
#: does NOT depend on a baseline, which is the whole point -- the relative rule is
#: blind when both arms loop.
REPEAT_CEILING = 0.60

#: Absolute ceiling on the share of a row that lies inside a repeated 16-gram.
#: Measured over the 26 real rows pulled at the repetition stop: clean rows top out at
#: 0.570 (NATURAL_f1_128k_pg19_train_0) and loops start at 0.646
#: (targetfull_pg19_train_1), so the two populations are 0.076 apart and this ceiling
#: sits between them with 0.030 below and 0.046 above. That margin is THIN and is
#: stated as thin -- it is the discriminating rule of last resort, not the first.
IN_REPEAT_CEILING = 0.60

#: A verbatim repeat of at least this many tokens. This is the primary detector
#: because its margin is not thin: across the same 26 rows, clean generations never
#: exceeded 64 (and that one is a 128-token row) while every loop exceeded 226.
VERBATIM_SPAN_CEILING = 128

#: Absolute ceiling on the share of a row that lies inside a periodic tail. A
#: generation that is p-periodic from token N onward for most of its length is a
#: loop whatever its n-gram statistics say. Clean rows: max 0.054. Loops: min 0.516.
PERIODIC_TAIL_CEILING = 0.50

#: Period search bounds. min 1 so a single repeated token is found (that is the most
#: degenerate loop of all); max is capped by the sequence length as well.
MIN_PERIOD = 1
MAX_PERIOD = 512


def repeat_share(text: str) -> float:
    """1 - distinct word-trigrams / total word-trigrams, EXACTLY as the coherence
    gate has always computed `gen_repeat_share` (mlsys_coherence_gate.py's
    generation_metrics). Same definition, one implementation, so an existing CSV
    column and this module cannot drift apart."""
    toks = text.split()
    grams = [tuple(toks[i:i + 3]) for i in range(max(0, len(toks) - 2))]
    if not grams:
        return 0.0
    return 1.0 - (len(set(grams)) / len(grams))


def periodicity(ids: Sequence[int], min_period: int = MIN_PERIOD,
                max_period: int = MAX_PERIOD) -> dict:
    """The dominant cycle of `ids`, and how much of the row obeys it.

    Returns `period`, `agreement` (the share of positions i with ids[i] ==
    ids[i+period]), `first_periodic_token` (the earliest index from which the row is
    period-consistent to the end, or len(ids) if it never is), and
    `periodic_tail_share` = (len - first_periodic_token) / len.

    `first_periodic_token` is computed by walking BACK from the end while
    ids[i] == ids[i+period], which is O(n) and needs no search. A row that is a loop
    for its whole length returns 0; a row with no cycle returns len(ids).
    """
    n = len(ids)
    if n < 2 * min_period:
        return {"period": 0, "agreement": 0.0, "first_periodic_token": n,
                "periodic_tail_share": 0.0}
    hi = min(max_period, n // 2)
    best_p, best_a = 0, 0.0
    for p in range(min_period, hi + 1):
        agree = sum(1 for i in range(n - p) if ids[i] == ids[i + p]) / (n - p)
        if agree > best_a:
            best_p, best_a = p, agree
    if best_p <= 0:
        return {"period": 0, "agreement": 0.0, "first_periodic_token": n,
                "periodic_tail_share": 0.0}
    first = n
    for i in range(n - 1 - best_p, -1, -1):
        if ids[i] == ids[i + best_p]:
            first = i
        else:
            break
    return {"period": int(best_p), "agreement": round(float(best_a), 4),
            "first_periodic_token": int(first),
            "periodic_tail_share": round((n - first) / n, 4)}


def _dup_of_length(ids: Sequence[int], L: int) -> bool:
    seen = set()
    for i in range(len(ids) - L + 1):
        g = tuple(ids[i:i + L])
        if g in seen:
            return True
        seen.add(g)
    return False


def longest_repeated_span(ids: Sequence[int], cap: int = 200_000) -> int:
    """The longest span of `ids` that occurs at least twice, by binary search on
    the length. Used to be a full O(n^2) DP; the search is O(n log n) and these rows
    are short, but the cap keeps a pathological 1M-token row from stalling a stage."""
    n = len(ids)
    if n < 2 or n > cap:
        return 0
    lo, hi, best = 1, n // 2, 0
    while lo <= hi:
        mid = (lo + hi) // 2
        if _dup_of_length(ids, mid):
            best, lo = mid, mid + 1
        else:
            hi = mid - 1
    return best


def token_repeat_stats(ids: Sequence[int], n: int = 16) -> dict:
    """Repetition measured on TOKENS, so it needs no tokenizer and is comparable
    across stages. This is the statistic the 2026-10-10 audit used by hand:
    distinct n-grams / total, the longest repeated span, and the share of tokens that
    lie inside some repeated n-gram."""
    if len(ids) < n:
        return {"rep_tok_repeat_share": 0.0, "rep_longest_repeat_span": 0,
                "rep_tokens_in_repeat": 0.0}
    grams = collections.Counter(tuple(ids[i:i + n]) for i in range(len(ids) - n + 1))
    repeated = {g for g, c in grams.items() if c > 1}
    total = sum(grams.values())
    covered = set()
    for i in range(len(ids) - n + 1):
        if tuple(ids[i:i + n]) in repeated:
            covered.update(range(i, i + n))
    return {
        "rep_tok_repeat_share": round(1.0 - len(grams) / total, 4),
        "rep_longest_repeat_span": longest_repeated_span(ids),
        "rep_tokens_in_repeat": round(len(covered) / len(ids), 4),
    }


def row_stats(ids: Sequence[int], text: str | None = None) -> dict:
    """Everything a row's repetition verdict needs, in one call.

    `text` must be the CONTINUATION ONLY, never the prompt. The natural stages write
    `generated/*.txt` as prompt + continuation (526 KB for a 1024-token generation
    over a 130k-token prompt), and measuring repetition over that returned a maximum
    repeat share of 0.295 across all 20 real rows -- the loop was diluted 128x. The
    gate writes its `gen_*.txt` as the continuation alone, which is why its numbers
    were right and the stage's were absent."""
    p = periodicity(ids)
    out = {
        "rep_repeat_share": round(repeat_share(text) if text is not None else 0.0, 4),
        "rep_period": p["period"],
        "rep_period_agreement": p["agreement"],
        "rep_first_periodic_token": p["first_periodic_token"],
        "rep_periodic_tail_share": p["periodic_tail_share"],
    }
    if len(ids) >= 2:
        out.update(token_repeat_stats(ids))
    out["rep_degenerate"] = row_reasons(out) != []
    return out


def row_reasons(stats: dict) -> list[str]:
    """The ABSOLUTE repetition verdict for one row: reasons, empty when clean.

    Each reason names the measurement and both the observed and the ceiling value,
    because "repeated n-grams 97% > 60% absolute" is a finding the reader can audit
    and "degenerate" is not.
    """
    reasons = []
    ls = stats.get("rep_longest_repeat_span")
    if isinstance(ls, (int, float)) and ls >= VERBATIM_SPAN_CEILING:
        reasons.append(f"a {ls}-token span repeats verbatim >= "
                       f"{VERBATIM_SPAN_CEILING} absolute")
    rs = stats.get("rep_repeat_share")
    if isinstance(rs, (int, float)) and rs > REPEAT_CEILING:
        reasons.append(f"repeated n-grams {rs:.0%} > {REPEAT_CEILING:.0%} absolute")
    tr = stats.get("rep_tokens_in_repeat")
    if isinstance(tr, (int, float)) and tr > IN_REPEAT_CEILING:
        reasons.append(f"{tr:.0%} of tokens inside a repeated {16}-gram > "
                       f"{IN_REPEAT_CEILING:.0%} absolute")
    ts = stats.get("rep_periodic_tail_share")
    per = stats.get("rep_period")
    if isinstance(ts, (int, float)) and ts > PERIODIC_TAIL_CEILING and per:
        reasons.append(f"periodic tail {ts:.0%} of the row (period {per}) > "
                       f"{PERIODIC_TAIL_CEILING:.0%} absolute")
    return reasons


def window_acceptance(rounds: Iterable[dict], n_tokens: int,
                      window: int = 128) -> list[dict]:
    """Acceptance per token window, from a stage's per-round trace.

    Each round declares `n_acc` (accepted prefix length), `spec_steps` (gamma) and
    `n_emitted` (tokens it contributed), so a round maps to the token positions it
    emitted and the acceptance it contributes is n_acc / gamma -- the PER-ROUND
    quantity the plan requires, not the i.i.d. per-token parameter.

    Returns one dict per window that any round landed in: {index, lo, hi, rounds,
    accepted, drafted, acceptance}.
    """
    bins: dict[int, dict] = {}
    pos = 0
    for r in rounds:
        k = int(r.get("spec_steps") or 0)
        # A round APPENDS `n_committed` tokens to the sequence (the accepted
        # draft tokens plus the bonus), while `n_acc` counts only the accepted
        # draft tokens. Using n_emitted to advance the position walks the window
        # boundaries backwards by the bonus token per round -- 25% of the row at
        # gamma=4 -- which would attribute a loop's windows to the wrong tokens.
        n = int(r.get("n_committed") or 0) or int(r.get("n_emitted") or 0)
        if k <= 0 or n <= 0:
            continue
        w = min(pos // window, max(0, (n_tokens - 1) // window))
        b = bins.setdefault(w, {"index": w, "lo": w * window,
                                "hi": min((w + 1) * window, n_tokens),
                                "rounds": 0, "accepted": 0, "drafted": 0})
        b["rounds"] += 1
        b["accepted"] += int(r.get("n_acc") or 0)
        b["drafted"] += k
        pos += n
    for b in bins.values():
        b["acceptance"] = round(b["accepted"] / b["drafted"], 4) if b["drafted"] else 0.0
    return [bins[k] for k in sorted(bins)]


def non_degenerate_acceptance(rounds: Iterable[dict], ids: Sequence[int],
                              window: int = 128) -> dict:
    """Acceptance over the windows that are NOT part of a loop, with its n.

    This is the provisional protocol in one function: a window is discarded when its
    own tokens are already periodic (its periodicity tail covers most of the window),
    and the headline is the mean of the surviving windows' `acceptance`, reported
    with the number of windows it was computed from. A row with no surviving window
    returns acceptance None and `windows_used` 0 -- an honest "no measurement" rather
    than a number drawn from a loop.
    """
    wins = window_acceptance(rounds, len(ids), window)
    used, dropped = [], []
    for b in wins:
        seg = list(ids[b["lo"]:b["hi"]])
        st = periodicity(seg) if len(seg) >= 4 else {
            "periodic_tail_share": 1.0, "period": 0}
        (dropped if st["periodic_tail_share"] > PERIODIC_TAIL_CEILING else used).append(b)
    return {
        "windows_total": len(wins),
        "windows_used": len(used),
        "windows_dropped": len(dropped),
        "acceptance": (round(sum(b["acceptance"] for b in used) / len(used), 4)
                       if used else None),
        "per_window": wins,
        "dropped": [b["index"] for b in dropped],
    }

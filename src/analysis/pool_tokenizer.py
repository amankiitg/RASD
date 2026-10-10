"""Is this prompt pool really tokenized with the candidate's tokenizer?

THE TRAP, MEASURED 2026-10-10. `data/processed/pg19` was built with the Llama-2
tokenizer (max id 29968) and `data/processed/pg19_docs` with Llama-3.1. Pointing a
Llama-3.1 candidate at the Llama-2 pool does NOT fail: every Llama-2 id is inside
Llama-3.1's 128k vocabulary, so an id-range check passes, the model loads, the
forward runs, and the perplexity comes back as a plausible number. The intended
memorization control would have returned a FALSE "memorized" verdict silently, in
the direction that flatters the paper's story.

`assert_prompt_in_vocab` cannot catch this because the ids ARE in vocab. What can
is the round trip: decode the ids with the candidate's tokenizer and re-encode. If
the pool was built with the same tokenizer, `encode(decode(ids))` returns `ids`
itself for all but a handful of positions; if it was built with a different
tokenizer, the decoded text is re-segmented differently and the match share
collapses.

This is evidence about the data, not a claim about a filename, so it holds even
when the pool carries no tokenizer metadata -- and the metadata (when present) is
checked too, because a name mismatch is the cheaper, more legible signal.
"""
from __future__ import annotations

import re

#: Below this share of positions that survive the round trip, the pool was built
#: with a different tokenizer. A matched pair lands at 1.000; the Llama-2 pool read
#: by a Llama-3.1 tokenizer lands far below this.
ROUNDTRIP_MIN = 0.90

#: Positions checked. A contiguous middle slice is enough: a tokenizer mismatch
#: breaks the segmentation everywhere, not only at the edges, and encoding a full
#: 1M-token prompt to run this check would cost more than the measurement it
#: protects.
ROUNDTRIP_PROBE = 32768


def tokenizer_family(name: str) -> str:
    """`meta-llama/Llama-3.1-8B` -> `llama3`. Org, size suffix and the minor
    version are dropped: Llama-3.1-8B and Llama-3.2-1B share a vocabulary, and a
    check that called them different families would block the arm the paper needs."""
    if not name:
        return ""
    base = str(name).rsplit("/", 1)[-1].lower()
    m = re.search(r"([a-z]+)[-_.]?(\d+)", base)
    if not m:
        return re.sub(r"[^a-z]", "", base)
    return f"{m.group(1)}{m.group(2)}"


def roundtrip_match_share(tok, ids, probe: int = ROUNDTRIP_PROBE) -> tuple:
    """(share of positions whose id survives decode -> encode, positions checked).

    Compared positionally over the common prefix of `ids` and `encode(decode(ids))`.
    A tokenizer swap does not merely permute ids, it changes the segmentation
    length, so the share is computed against the shorter of the two.
    """
    ids = [int(i) for i in ids]
    if not ids:
        return 1.0, 0
    if len(ids) > probe:
        start = (len(ids) - probe) // 2
        ids = ids[start:start + probe]
    # TWO DEFAULTS THAT BOTH HAVE TO BE TURNED OFF, and each one was a false
    # positive that only a real document exposed:
    #
    #   clean_up_tokenization_spaces=False -- decode() defaults to True, which
    #   rewrites punctuation spacing (" ." -> ".", "  " -> " "). Re-encoding the
    #   CLEANED text re-segments it, so the check reported a tokenizer mismatch
    #   on pools that were tokenized perfectly. Measured on data/processed/
    #   pg19_1m, 2026-10-10: with cleanup on, pg19_train_1 matched 1.3%,
    #   pg19_train_1726 7.9%, pg19_train_2204 73.2%, pg19_train_2768 3.6% -- and
    #   with it off ALL SIX matched 100.0%. The Bible matched either way, which
    #   is why a positive control built from one document would have missed it.
    #
    #   add_special_tokens=False -- the default prepends a BOS, shifting every
    #   position by one and reporting a 0.1% match on an identical tokenizer.
    #
    # A detector that fails its own positive control is worse than no detector:
    # this one is wired into the gate verdict, so a false positive would have
    # failed legitimate rows and stopped a campaign.
    text = tok.decode(ids, clean_up_tokenization_spaces=False)
    back = tok(text, add_special_tokens=False)["input_ids"]
    n = min(len(ids), len(back))
    if n == 0:
        return 0.0, 0
    same = sum(1 for i in range(n) if ids[i] == back[i])
    return same / n, n


def pool_check(tok, ids, pool_tokenizer: str | None = None,
               candidate_tokenizer: str | None = None,
               probe: int = ROUNDTRIP_PROBE) -> dict:
    """Everything a row needs to say whether its pool matches its tokenizer.

    Returns both tokenizer names, both families, the name `pool_tokenizer_match`,
    the `pool_roundtrip_share`, and `pool_probe_tokens`. The caller decides whether
    a failure is fatal; this reports.
    """
    share, n = roundtrip_match_share(tok, ids, probe)
    out = {
        "pool_tokenizer": pool_tokenizer or "",
        "candidate_tokenizer": candidate_tokenizer or "",
        "pool_family": tokenizer_family(pool_tokenizer or ""),
        "candidate_family": tokenizer_family(candidate_tokenizer or ""),
        "pool_roundtrip_share": round(share, 4),
        "pool_probe_tokens": n,
    }
    if pool_tokenizer and candidate_tokenizer:
        out["pool_tokenizer_match"] = (
            out["pool_family"] == out["candidate_family"])
    else:
        out["pool_tokenizer_match"] = ""
    return out


def pool_problem(chk: dict) -> str:
    """The failure text, or "" when the pool is usable for this tokenizer."""
    share = chk.get("pool_roundtrip_share")
    if isinstance(share, (int, float)) and share < ROUNDTRIP_MIN:
        return (f"prompt ids do not round-trip through the candidate's tokenizer "
                f"({share:.1%} of {chk.get('pool_probe_tokens')} positions match, "
                f"below {ROUNDTRIP_MIN:.0%}): the pool was built with a different "
                f"tokenizer (pool={chk.get('pool_tokenizer') or 'unstated'}, "
                f"candidate={chk.get('candidate_tokenizer') or 'unstated'}), so "
                f"every perplexity from it is a measurement of the wrong text")
    if chk.get("pool_tokenizer_match") is False:
        return (f"pool tokenizer {chk.get('pool_tokenizer')} "
                f"({chk.get('pool_family')}) is a different family from the "
                f"candidate's {chk.get('candidate_tokenizer')} "
                f"({chk.get('candidate_family')})")
    return ""

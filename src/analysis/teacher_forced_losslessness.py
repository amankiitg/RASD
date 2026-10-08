"""Teacher-forced losslessness: is an emitted token the target's own argmax?

WHY THIS EXISTS

The campaign's losslessness gate originally required a speculative arm's token
stream to be *token-identical* to its target-only partner's. On 2026-10-08 that
failed at 8 ranks and 128k, and a $0.54 1x reproduction showed why: the target
is run in two different forward shapes -- a packed (gamma+1)-token verify in one
forward, versus one token per step in target-only decode -- and its argmax is
not invariant to that choice. Two independent runs of the SAME arm are
byte-identical, so this is not run-to-run noise; it is a deterministic
difference between the two shapes, amplified by FP4 weights, which flips
near-tie argmax decisions.

Token identity between the two arms is therefore not a test of the speculative
implementation at all: it is a test of whether two different bf16 kernels agree
bit-for-bit, and they do not. The question that IS about the implementation is:

    for each token the speculative arm emitted, is it the token the target
    itself would have chosen, having been shown the same prefix?

That is teacher forcing on the spec arm's own stream, and that is what this
module decides. A token that is the target's argmax is correct by definition,
regardless of which shape computed it. A token that is merely *close* to the
argmax is admissible only within a tolerance derived from the measured
shape-difference (see `tol_from_noise`), because a decision legitimately
overturned by that difference has a bounded logit shortfall.

WHY A SHORTFALL BOUND, AND WHY 2x

Work in logit space. Let the emitted token be the argmax under shape A, so
`L_A[g] >= L_A[j]` for every j. Shape B differs by at most epsilon per logit,
`L_B = L_A + d` with `|d| <= epsilon`. Then

    L_B[g] >= L_A[g] - eps >= L_A[argmax_B] - eps >= L_B[argmax_B] - 2*eps

so the emitted token can sit at most `2*eps` below the maximum under shape B.
`2*eps` is therefore the largest shortfall a *correct* decision can show, and
that is the tolerance: `tol = 2 * max|delta|` measured between the two shapes.
The factor 2 is not a safety margin and not a fudge -- it is the bound.

A cap keeps a bad measurement from turning the gate into a rubber stamp: if the
measured noise is larger than the cap the rule REFUSES rather than widening
until everything passes.
"""
from __future__ import annotations

from typing import Any, Sequence

import numpy as np

#: Hard ceiling on the tolerance, in logits. A measured noise floor above this
#: means the engine is too imprecise for the gate to mean anything, so the rule
#: that uses it must refuse (see `tol_from_noise`) instead of admitting it.
TOL_CAP = 2.0


def tol_from_noise(max_abs_delta: float, cap: float = TOL_CAP) -> float:
    """Tolerance implied by a measured packed-vs-stepwise noise floor.

    `min(2 * max_abs_delta, cap)`. The 2x is the bound derived above; the cap is
    the operator's standing rule that the gate never admits a token more than
    `cap` logits below the target's argmax, however large the measured noise
    turns out to be.

    WHEN THE CAP BINDS, THE TOLERANCE IS IMPOSED, NOT DERIVED -- and that is the
    strict (safer) direction: the true 2x bound would be wider, so the gate
    rejects more than the noise strictly excuses. Callers should report it (see
    `tol_provenance`) because a binding cap means this gate is not a claim that
    every admitted token is noise-consistent.
    """
    return tol_provenance(max_abs_delta, cap)["tol"]


def tol_provenance(max_abs_delta: float, cap: float = TOL_CAP) -> dict:
    """The tolerance, plus whether the cap or the measurement decided it."""
    if max_abs_delta < 0:
        raise ValueError(f"max_abs_delta must be >= 0, got {max_abs_delta}")
    raw = 2.0 * float(max_abs_delta)
    tol = min(raw, float(cap))
    return {
        "raw_2x_noise": raw,
        "cap": float(cap),
        "tol": tol,
        "cap_binds": bool(raw > float(cap)),
        "note": (
            f"cap binds: 2 x measured noise = {raw:.4f} exceeds the {cap} cap, "
            f"so TOL={tol:.4f} is IMPOSED by the cap, not derived from the "
            f"measurement. The gate is therefore stricter than the noise floor "
            f"requires -- legitimate flips with a shortfall between {tol:.4f} "
            f"and {raw:.4f} would fail."
            if raw > float(cap) else
            f"tol derived from the measurement: 2 x {max_abs_delta:.4f} = {tol:.4f}"
        ),
    }


def _as_numpy(logits: Any) -> np.ndarray:
    """One position's logits as a host float32 array.

    Accepts numpy, or anything torch-like. The conversion is duck-typed rather
    than `import torch` so this module stays importable without torch (the CPU
    test suite imports it), and so a caller cannot accidentally leave a CUDA
    tensor where numpy is expected -- which fails with a message about numpy
    rather than about the device, sending the reader to the wrong place.
    """
    if hasattr(logits, "detach") and hasattr(logits, "cpu"):
        logits = logits.detach().cpu().numpy()
    return np.asarray(logits, dtype=np.float32).reshape(-1)


def logit_shortfall(logits: Any, token: int) -> float:
    """How far `token` sits below the argmax, in logits (0.0 if it IS the argmax).

    Always >= 0. `logits` is one position's vector over the vocabulary.
    """
    row = _as_numpy(logits)
    if row.size == 0:
        raise ValueError("empty logits row")
    if not 0 <= token < row.size:
        raise IndexError(f"token {token} outside the {row.size}-token vocabulary")
    return float(row.max() - row[token])


def evaluate_stream(emitted_ids: Sequence[int], teacher_logits: Sequence[Any],
                    tol: float, min_checked: int = 1) -> dict:
    """Decide an emitted stream against the target's teacher-forced logits.

    `teacher_logits[i]` must be the target's logits for position `i` AFTER
    being shown `emitted_ids[:i]` -- i.e. the caller teacher-forces the target
    on the stream's own prefix, stepwise. Passing the prefix-conditioned logits
    is what makes this a statement about the implementation rather than about
    two kernels agreeing.

    A position PASSES when either
      * the emitted token IS the argmax (`shortfall == 0`), or
      * its shortfall is <= `tol`, i.e. the decision is inside the measured
        shape-difference and cannot be told apart from a correct one.

    Verdicts:
      TF_LOSSLESS      every checked position passed
      TF_MISMATCH      at least one position sat further below the argmax than
                       `tol`; the worst position and its shortfall are reported
      TF_UNCHECKED     fewer than `min_checked` positions had logits, so the
                       check did not actually run
    """
    if tol < 0:
        raise ValueError(f"tol must be >= 0, got {tol}")
    n = min(len(emitted_ids), len(teacher_logits))
    records = []
    failures = []
    for i in range(n):
        tok = int(emitted_ids[i])
        short = logit_shortfall(teacher_logits[i], tok)
        is_argmax = short <= 0.0
        ok = is_argmax or short <= tol
        rec = {
            "position": i,
            "emitted_token": tok,
            "is_argmax": bool(is_argmax),
            "shortfall": short,
            "within_tol": bool(ok),
        }
        records.append(rec)
        if not ok:
            failures.append(rec)

    if n < min_checked:
        return {
            "verdict": "TF_UNCHECKED",
            "checked": n,
            "tol": float(tol),
            "positions": records,
            "failures": failures,
            "reason": (f"only {n} position(s) had teacher-forced logits, "
                       f"need >= {min_checked}; the check did not run"),
        }
    if failures:
        worst = max(failures, key=lambda r: r["shortfall"])
        return {
            "verdict": "TF_MISMATCH",
            "checked": n,
            "tol": float(tol),
            "positions": records,
            "failures": failures,
            "first_failure_position": failures[0]["position"],
            "worst_position": worst["position"],
            "worst_shortfall": worst["shortfall"],
            "n_argmax": sum(1 for r in records if r["is_argmax"]),
            "n_within_tol": sum(1 for r in records if r["within_tol"]),
        }
    return {
        "verdict": "TF_LOSSLESS",
        "checked": n,
        "tol": float(tol),
        "positions": records,
        "failures": [],
        "n_argmax": sum(1 for r in records if r["is_argmax"]),
        "n_within_tol": sum(1 for r in records if r["within_tol"]),
        "max_shortfall": max((r["shortfall"] for r in records), default=0.0),
    }


def describe(result: dict) -> str:
    """One line for a log or a report."""
    v = result["verdict"]
    if v == "TF_UNCHECKED":
        return "teacher-forced check DID NOT RUN: %s" % result.get("reason", "")
    if v == "TF_MISMATCH":
        return ("teacher-forced MISMATCH: %d/%d positions below the target's "
                "argmax by more than tol=%.4g; worst at position %d, shortfall "
                "%.4g (first failure at %d)"
                % (len(result["failures"]), result["checked"], result["tol"],
                   result["worst_position"], result["worst_shortfall"],
                   result["first_failure_position"]))
    return ("teacher-forced LOSSLESS: %d positions checked, %d exactly the "
            "argmax, %d within tol=%.4g (max shortfall %.4g)"
            % (result["checked"], result["n_argmax"], result["n_within_tol"],
               result["tol"], result["max_shortfall"]))

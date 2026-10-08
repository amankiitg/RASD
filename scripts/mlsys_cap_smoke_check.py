#!/usr/bin/env python3
"""Validate the generation cap on real output before the campaign depends on it.

The engine change that stops a verify round overshooting `max_new_tokens` has
never executed on a GPU. If it is wrong it fails QUIETLY: the generated length
changes, which breaks the `prompt + BOS + generated == context` identity,
invalidates the losslessness comparison (the target-only arm stops exactly on the
cap), and makes the acceptance denominator a function of acceptance.

This script is the gate on that change. It asserts, for every cell of
`configs/mlsys_engine_cap_smoke.yml`:

  1. `tokens_generated == max_new_tokens` EXACTLY, in both arms, at both caps.
  2. the final per-round trace record's `n_emitted` / `round_truncated` are
     consistent with the cap: the emitted tokens across all rounds sum to the
     cap, and a round is flagged truncated only when the budget cut it short.
  3. the speculative/target-only pair is lossless. The GATE is the
     teacher-forced check on the bf16-KV pair: each token the speculative arm
     emitted must be the token the target itself would have chosen, given the
     stream's own prefix, within TOL_bf16 logits. Token identity between the two
     arms is still computed and reported, but it does NOT gate -- the two arms
     run the target in different forward SHAPES (a packed (gamma+1)-token verify
     versus one token per step), and in bf16 those shapes do not agree
     bit-for-bit, so identity tested the kernels rather than the implementation.
     See docs/mlsys_analysis_plan.md 6.1a-6.1c for the measurement and the rule.

Exit non-zero if any assertion fails. The manifest stops before the
speculative stages on a non-zero exit.

Usage:
    python scripts/mlsys_cap_smoke_check.py --results results/mlsys/engine_cap_smoke.csv
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

#: The teacher-forced tolerance for the bf16-KV pair, in logits.
#:
#: DERIVED, not chosen: 2 x the measured max |delta logit| between the packed and
#: stepwise forward shapes on a 63-position-per-cell floor at 8k and 32k, which
#: is 2 x 0.9375. The 2x is the bound on the shortfall a CORRECT decision can
#: show (docs/mlsys_analysis_plan.md 6.1a), and there is no cap. It is valid for
#: as long as the bf16 negatives' best case (the 6.75-logit unverified draft at
#: 8k) stays at or above 2 x TOL_bf16, i.e. at or above 3.75 -- it does, by
#: 1.80x. Re-derive it if the engine's precision or the shapes change.
TOL_BF16 = 1.875

#: KV precisions whose pairs must PASS the teacher-forced gate. NF4 is measured
#: and reported but never gated: at this context its own noise flips 20% of
#: argmaxes, which is the same order as the defects a gate has to catch, and its
#: shortfall separation is only 1.31x (6.1b).
GATED_KV = ("bfloat16",)

#: KV precisions this stage declares. `nf4` is the campaign's default and the
#: baseline every comparison rests on; the `bfloat16` pair exists because NF4's
#: own noise cannot support the gate (6.1c). Naming both keeps the original
#: guard -- a blank or unknown measured dtype is still a failure -- while
#: admitting the one deliberate exception, which is declared rather than
#: inferred. A dtype outside this tuple is a problem, not a new configuration.
DECLARED_KV = ("nf4", "bfloat16")


def _read_csv(path: Path) -> list[dict]:
    with path.open() as fh:
        return list(csv.DictReader(fh))


def _trace(results_dir: Path, run_id: str) -> list[dict]:
    p = results_dir / "per_token" / f"{run_id}.jsonl"
    if not p.exists():
        return []
    return [json.loads(line) for line in p.read_text().splitlines() if line.strip()]


# `generated` holds one token before the verify loop starts: the seed
# `cur_token` that the first round verifies. Every token-count identity in this
# file has to include it.
INITIAL_TOKEN = 1


def _sidecar(tokens_dir: Path, run_id: str) -> dict | None:
    """The token sidecar for a run, or None. Malformed counts as missing."""
    p = tokens_dir / f"{run_id}.json"
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text())
    except Exception:                                  # noqa: BLE001
        return None


def _read_probes(tokens_dir: Path, rows: list[dict]) -> dict:
    """The teacher-forced probe sidecars, keyed by run_id.

    A missing probe is NOT an empty probe: the caller must be able to tell
    "measured and clean" from "never measured", and only the first is a pass.
    """
    out = {}
    for r in rows:
        p = tokens_dir / f"{r['run_id']}.tflossless.json"
        if not p.exists():
            continue
        try:
            out[r["run_id"]] = json.loads(p.read_text())
        except Exception as e:                                  # noqa: BLE001
            out[r["run_id"]] = {"error": f"unreadable probe sidecar: {e}"}
    return out


def check(results_csv: Path, tokens_dir: Path | None = None) -> tuple[list[str], list[str]]:
    """Returns (problems, notes)."""
    problems: list[str] = []
    notes: list[str] = []
    rows = _read_csv(results_csv)
    results_dir = results_csv.resolve().parent
    tokens_dir = tokens_dir or (results_dir / "tokens")

    ok_rows = [r for r in rows if r.get("status") == "ok"]

    if not ok_rows:
        return ([f"no row completed successfully in {results_csv}"], notes)

    for r in ok_rows:
        rid, cap = r["run_id"], int(r["max_new_tokens"])
        gen = int(r["tokens_generated"])
        # A row with no verify rounds is a target-only run: there is nothing for
        # a per-round trace to record, so it is checked on the cap and on its
        # token sidecar instead. Demanding a trace of it asked the generator to
        # have produced a verify loop it does not have.
        is_spec = str(r.get("spec_steps", "")).strip() not in ("", "0")

        if gen != cap:
            problems.append(
                f"{rid}: tokens_generated={gen} but max_new_tokens={cap} "
                f"({gen - cap:+d}); the cap is not enforced"
            )
        else:
            notes.append(f"{rid}: generated exactly {cap}")

        # GAP ALIGNMENT, on every row of BOTH arms. `token_gaps` is a parallel
        # array to the emitted ids and the losslessness tie rule reads it by
        # position: a list one short reports the NEXT token's indifference, which
        # is how a real mismatch gets excused as a numerics tie. The engine
        # cannot be run on CPU (it requires CUDA streams), so this is the check
        # that proves the alignment on real hardware -- it runs on the first GPU
        # stage, before any expensive one.
        sidecar = _sidecar(tokens_dir, rid)
        if sidecar is None:
            problems.append(f"{rid}: no token sidecar, so the gap array cannot "
                            f"be checked for alignment")
        else:
            ids = sidecar.get("generated_token_ids") or []
            gaps = sidecar.get("token_gaps")
            if gaps is None:
                problems.append(f"{rid}: the sidecar carries no token_gaps; the "
                                f"tie rule would treat every divergence as a "
                                f"MISMATCH with no evidence either way")
            elif len(gaps) != len(ids):
                problems.append(
                    f"{rid}: token_gaps has {len(gaps)} entries for {len(ids)} "
                    f"emitted ids; the tie rule would read the wrong position"
                )
            else:
                notes.append(f"{rid}: {len(gaps)} gaps aligned with "
                             f"{len(ids)} emitted ids")

            # NUMERICS, measured by the engine off the loaded models and the
            # live cache. The campaign's stages run FP4 weights and an NF4 KV
            # cache, and every comparison in the plan rests on that: the vLLM
            # cross-check reports token agreement instead of scoring it BECAUSE
            # vLLM cannot reproduce this cache. A row whose config claimed 4-bit
            # while the loader skipped it would invalidate that reasoning
            # silently, so the values are asserted rather than assumed.
            wp = sidecar.get("weight_precision")
            kvd = sidecar.get("kv_dtype")
            if not wp or not kvd:
                problems.append(
                    f"{rid}: the sidecar carries no measured precision "
                    f"(weight_precision={wp!r} kv_dtype={kvd!r}); the numerics "
                    f"every comparison depends on are unstated")
            else:
                if wp != "fp4":
                    problems.append(
                        f"{rid}: weight_precision={wp!r}, expected 'fp4' — the "
                        f"campaign's cells are FP4 and a different precision "
                        f"makes them incomparable with the published numbers")
                if kvd not in DECLARED_KV:
                    problems.append(
                        f"{rid}: kv_dtype={kvd!r}, expected one of "
                        f"{DECLARED_KV} — the KV cache is what the vLLM "
                        f"comparison cannot reproduce, and an unlisted dtype "
                        f"means the run is not the configuration this stage "
                        f"declares")
                if wp == "fp4" and kvd in DECLARED_KV:
                    notes.append(f"{rid}: numerics fp4 weights + {kvd} KV "
                                 f"(measured)")

        # The sequence the engine built must be the rung.
        pt, seq = r.get("prompt_tokens"), r.get("sequence_tokens")
        if pt and seq and int(seq) != int(r["context_length"]):
            problems.append(
                f"{rid}: sequence_tokens={seq} != context_length="
                f"{r['context_length']}"
            )
        elif pt and seq:
            if int(pt) + 1 + gen != int(seq):
                problems.append(
                    f"{rid}: {pt} prompt + 1 BOS + {gen} generated != {seq}"
                )

        # The token sidecar, for BOTH arms: it is the record of what was
        # actually emitted, and for a target-only row it is the only evidence
        # available.
        side = _sidecar(tokens_dir, rid)
        if side is None:
            problems.append(f"{rid}: no token sidecar, so the emitted tokens "
                            f"cannot be checked")
        else:
            ids = side.get("generated_token_ids") or []
            if len(ids) != cap:
                problems.append(
                    f"{rid}: token sidecar holds {len(ids)} ids, expected the "
                    f"cap ({cap})"
                )
            if not is_spec:
                notes.append(
                    f"{rid}: target-only, {gen} generated == cap, sidecar has "
                    f"{len(ids)} ids"
                )
                continue

        tr = _trace(results_dir, rid)
        if not tr:
            problems.append(f"{rid}: no per-round trace, so the cap "
                            f"bookkeeping cannot be checked")
            continue

        # `generated` already holds ONE token before the loop starts (the seed
        # `cur_token` the first round verifies). So the cap is
        #   1 + (sum of accepted tokens emitted) + (one bonus per untruncated
        #       round)
        # and NOT `emitted + bonuses`, which was short by exactly that first
        # token.
        emitted = sum(int(x.get("n_emitted", x["n_acc"])) for x in tr)
        # The bonus token: one per round unless the budget cut the round short
        # (see _round_commit_plan). The truncated final round has no room for it.
        bonuses = sum(1 for x in tr if not x.get("round_truncated"))
        if emitted + bonuses + INITIAL_TOKEN != cap:
            problems.append(
                f"{rid}: trace accounts for {emitted} accepted + {bonuses} bonus "
                f"+ {INITIAL_TOKEN} seed = {emitted + bonuses + INITIAL_TOKEN} "
                f"tokens, expected {cap}"
            )
        truncated = [i for i, x in enumerate(tr) if x.get("round_truncated")]
        if truncated and truncated != [len(tr) - 1]:
            problems.append(
                f"{rid}: rounds {truncated} flagged truncated; only the FINAL "
                f"round can be cut short by the budget"
            )
        last = tr[-1]
        if int(last.get("n_emitted", last["n_acc"])) + (
                0 if last.get("round_truncated") else 1) < 1:
            problems.append(f"{rid}: final round committed nothing")

        # KV geometry. The context the next round reads must cover exactly the
        # tokens this round emitted -- no less (the next cur_token would then sit
        # at the wrong positional offset, silently corrupting every later round)
        # and no more (it would keep the verified-but-unemitted tail). A
        # truncated round is where this can go wrong, which is why it is
        # asserted here rather than assumed: the engine needs CUDA, so this is
        # the only place the arithmetic is ever checked end to end.
        kv_expected_prev = None
        for i, x in enumerate(tr):
            if "kv_len_after" not in x:
                problems.append(
                    f"{rid}: round {i} records no kv_len_after, so the KV "
                    f"length cannot be verified"
                )
                break
            kb, ka = int(x.get("kv_len_before", -1)), int(x["kv_len_after"])
            committed = int(x.get("n_committed", -1))
            emitted_i = int(x.get("n_emitted", x["n_acc"]))
            truncated_i = bool(x.get("round_truncated"))
            if committed < 0:
                problems.append(f"{rid}: round {i} records no n_committed")
                continue
            if ka != kb + committed:
                problems.append(
                    f"{rid}: round {i} KV {kb} -> {ka} but committed "
                    f"{committed} tokens"
                )
            # A truncated round commits nothing for the unemitted tail and gets
            # no bonus; an untruncated round commits the emitted prefix plus one
            # bonus token.
            want = emitted_i if truncated_i else emitted_i + 1
            if committed != want:
                problems.append(
                    f"{rid}: round {i} committed {committed}, expected {want} "
                    f"(emitted {emitted_i}, truncated={truncated_i})"
                )
            if truncated_i and ka - kb != emitted_i:
                problems.append(
                    f"{rid}: truncated round {i} covers {ka - kb} positions but "
                    f"emitted {emitted_i}; the KV must cover exactly the "
                    f"verified emitted prefix"
                )
            if kv_expected_prev is not None and kb != kv_expected_prev:
                problems.append(
                    f"{rid}: round {i} starts at KV {kb} but the previous round "
                    f"left {kv_expected_prev}"
                )
            kv_expected_prev = ka
        notes.append(
            f"{rid}: spec, {len(tr)} rounds, 1 seed + {emitted} accepted + "
            f"{bonuses} bonus = {cap}, final round n_acc={last['n_acc']} "
            f"n_emitted={last.get('n_emitted')} "
            f"truncated={last.get('round_truncated')} "
            f"kv {tr[0].get('kv_len_before')}->{kv_expected_prev}"
        )

    # ------------------------------------------------------------------
    # Losslessness between each spec row and its target-only partner.
    #
    # The pairing key INCLUDES the measured KV precision. Without it, two pairs
    # that differ only in kv_quant land on the same key and a spec arm is
    # silently paired with the wrong baseline -- comparing an NF4 stream against
    # a bf16 one, which is exactly the numerics difference this stage exists to
    # quantify. The precision is read from each sidecar's MEASURED `kv_dtype`,
    # never from the run's config, for the same reason the numerics block is
    # measured: `kv_quant` is an instruction and an instruction can be inert.
    # ------------------------------------------------------------------
    side = {}
    for r in ok_rows:
        p = tokens_dir / f"{r['run_id']}.json"
        if p.exists():
            try:
                side[r["run_id"]] = json.loads(p.read_text())
            except Exception:                                  # noqa: BLE001
                side[r["run_id"]] = None

    probes = _read_probes(tokens_dir, ok_rows)

    def _pair_key(row):
        return (row.get("doc_id"), row.get("context_length"),
                row.get("max_new_tokens"),
                str((side.get(row["run_id"]) or {}).get("kv_dtype", "")))

    by_key = {}
    for r in ok_rows:
        if str(r.get("spec_steps", "")).strip() in ("", "0"):
            by_key[_pair_key(r)] = r


    seen = 0
    for r in ok_rows:
        if str(r.get("spec_steps", "")).strip() in ("", "0"):
            continue
        key = _pair_key(r)
        partner = by_key.get(key)
        if partner is None:
            problems.append(f"{r['run_id']}: no target-only partner at the same "
                            f"doc/context/cap/KV ({key[1]} ctx, {key[2]} tokens, "
                            f"kv={key[3] or 'unknown'}), so losslessness is "
                            f"unverified")
            continue
        bad = require_same_request(r, partner)
        if bad:
            problems.append(f"{r['run_id']}: pair refused: {'; '.join(bad)}")
            continue
        a, b = side.get(r["run_id"]), side.get(partner["run_id"])
        if a is None or b is None:
            problems.append(f"{r['run_id']}: missing token sidecar for a pair "
                            f"member")
            continue
        seen += 1
        cap = int(r["max_new_tokens"])
        kv = str(a.get("kv_dtype", ""))

        # --- REPORTED, not gated: token identity between the arms ----------
        res = compare_generations(a["generated_token_ids"],
                                  b["generated_token_ids"],
                                  full_length=cap, min_prefix=cap,
                                  spec_gaps=a.get("token_gaps"),
                                  target_gaps=b.get("token_gaps"))
        notes.append(f"{r['run_id']}: token identity vs {partner['run_id']} -> "
                     f"{res['verdict']} ({res.get('detail', '')}) [REPORTED, "
                     f"not a gate: the two arms run the target in different "
                     f"forward shapes]")

        # --- the GATE: the teacher-forced check on the bf16 pair -----------
        pr = probes.get(r["run_id"])
        if kv in GATED_KV:
            if pr is None:
                problems.append(
                    f"{r['run_id']}: kv={kv} is gated but has NO "
                    f"teacher-forced probe sidecar (<run_id>.tflossless.json); "
                    f"losslessness is UNVERIFIED, which is not a pass")
            elif pr.get("error"):
                problems.append(f"{r['run_id']}: teacher-forced probe failed: "
                                f"{pr['error']} -- UNVERIFIED, not a pass")
            else:
                ms = float(pr["max_shortfall"])
                notes.append(
                    f"{r['run_id']}: teacher-forced max shortfall {ms:.4f} <= "
                    f"TOL_bf16 {TOL_BF16} ({pr['non_argmax']}/{pr['positions']} "
                    f"positions not the argmax, worst at "
                    f"{pr['worst_position']})")
                if ms > TOL_BF16:
                    problems.append(
                        f"{r['run_id']}: teacher-forced MISMATCH -- max "
                        f"shortfall {ms:.4f} > TOL_bf16 {TOL_BF16}, worst at "
                        f"position {pr['worst_position']}; "
                        f"{pr['non_argmax']}/{pr['positions']} positions are not "
                        f"the target's argmax")
        else:
            # REPORT ONLY for everything else.
            if pr is None:
                notes.append(f"{r['run_id']}: kv={kv or 'unknown'} not gated, and "
                             f"no teacher-forced probe was recorded")
            elif pr.get("error"):
                notes.append(f"{r['run_id']}: kv={kv} probe failed ({pr['error']}) "
                             f"[REPORT ONLY]")
            else:
                notes.append(
                    f"{r['run_id']}: kv={kv} [REPORT ONLY, not gated] "
                    f"max shortfall {float(pr['max_shortfall']):.4f}, "
                    f"non-argmax {pr['non_argmax']}/{pr['positions']} "
                    f"({100 * float(pr['non_argmax_fraction']):.1f}%), "
                    f"noise floor max|delta| "
                    f"{float(pr['noise_floor']['max_abs_delta']):.4f}, "
                    f"control {float(pr['noise_floor']['control_max_abs_delta']):.4f}")

    # The 8-rank noise floor, reported per arm and compared across arms, so a
    # rank-dependent blow-up is visible in the gate's own output rather than
    # only in a sidecar nobody reads.
    for rid, pr in sorted(probes.items()):
        if pr.get("error"):
            continue
        nf = pr.get("noise_floor") or {}
        notes.append(f"noise_floor {rid}: world_size={pr.get('world_size')} "
                     f"kv={pr.get('measured_kv')} "
                     f"max|delta|={nf.get('max_abs_delta')} over "
                     f"{nf.get('positions')} positions, "
                     f"control={nf.get('control_max_abs_delta')}")
    # The gate must not be able to VANISH. If no speculative pair measured a
    # GATED_KV dtype -- e.g. the bf16 level's kv_quant override was silently
    # inert so it ran as NF4 and was treated as report-only -- then every pair
    # was reported and none was gated, and the stage would pass having tested
    # nothing. Same silent-no-op class the measured-dtype assertion exists for.
    gated_pairs = [
        r["run_id"] for r in ok_rows
        if str(r.get("spec_steps", "")).strip() not in ("", "0")
        and str((side.get(r["run_id"]) or {}).get("kv_dtype", "")) in GATED_KV
    ]
    if not gated_pairs:
        problems.append(
            f"no speculative pair measured a GATED_KV dtype {GATED_KV}; every "
            f"pair is report-only, so this stage would pass without gating "
            f"anything")

    if seen == 0:
        problems.append("no speculative/target-only pair was checked")

    return problems, notes


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True)
    ap.add_argument("--tokens-dir", default=None)
    args = ap.parse_args()

    problems, notes = check(Path(args.results),
                            Path(args.tokens_dir) if args.tokens_dir else None)
    for n in notes:
        print(f"  ok    {n}")
    if problems:
        print("\nCAP SMOKE FAILED — the manifest must not start any speculative "
              "stage:")
        for p in problems:
            print(f"  FAIL  {p}")
        return 1
    print("\nCAP SMOKE PASSED: the cap is enforced in both arms and the pair is "
          "lossless.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""GPU-free stand-in for `scripts/mlsys_vllm_baseline.py`.

Only the `vllm`-importing worker is replaced. The row assembly
(`build_row`, which sets `prompt_ids_from_engine` from the RASD cell), the
pairing identity (`prompt_sha256` recomputed from the ids the sidecar supplies),
the `unit_matched` verdict and the CSV writer all come from the real module, so
a missing revision pin or a mismatched prompt id is decided by production code.
An earlier version of this stub set `prompt_ids_from_engine` itself, which hid a
real bug: the production parent copied only the worker's result fields, and the
worker cannot report that field at all, so no real row could ever be
unit-matched.

The one thing it must NOT do is take the verdict for granted. The real worker
records the ids vLLM REPORTED consuming -- `outs[0].prompt_token_ids`, a value
that comes back from the engine -- and the row is unit-matched only when those
equal the ids supplied. A stub that simply echoed the sidecar would make
`prompt_ids_verified` a restatement of its own input and the rehearsal would
pass on a check that can never fail.

So the reported ids come from a SEPARATE code path here, controlled by
`MLSYS_REHEARSAL_VLLM_IDS`, which the rehearsal uses to exercise all three
outcomes the real worker can produce:

  match        the engine reported exactly the ids it was given
  mismatch     the engine reported something else (a dropped final token)
  unavailable  the engine reported nothing at all

Only the first may be unit-matched.
"""
from __future__ import annotations

import importlib.util
import json
import os
import pathlib
import random
import sys
import zlib

HERE = pathlib.Path(__file__).resolve()
REAL = HERE.parent / "_real_vllm_baseline.py"


def _load_real():
    spec = importlib.util.spec_from_file_location("_real_vllm_baseline", REAL)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


vb = _load_real()


def _reported_ids(given):
    """The ids the engine "reported", from a path independent of the input.

    `mlsys_vllm_baseline.worker_main` compares vLLM's own
    `outs[0].prompt_token_ids` against the ids it was handed, so this stands in
    for that check rather than restating it.
    """
    mode = os.environ.get("MLSYS_REHEARSAL_VLLM_IDS", "match")
    if given is None:
        return None
    if mode == "match":
        return list(given)
    if mode == "mismatch":
        # A decode/re-encode round trip is not injective; the realistic failure
        # is a prompt that differs by a token or two.
        return list(given[:-1]) + [7]
    if mode == "unavailable":
        return None
    raise SystemExit(f"unknown MLSYS_REHEARSAL_VLLM_IDS={mode!r}")


def _perturbed(ids):
    """Optionally move ONE emitted id, to exercise the cross-check's verdicts.

    `MLSYS_REHEARSAL_VLLM_DIVERGE=<pos>` makes this "engine" disagree with the
    RASD target-only run at that position; `MLSYS_REHEARSAL_VLLM_GAP=<f>` sets
    the gap it reports there, so the rehearsal can drive both outcomes of the
    tie rule (a decisive divergence is a MISMATCH; an indifferent one is a
    NUMERIC_TIE) through the production comparison.
    """
    pos = os.environ.get("MLSYS_REHEARSAL_VLLM_DIVERGE")
    if pos in (None, "") or not ids:
        return list(ids)
    out = list(ids)
    i = min(int(pos), len(out) - 1)
    out[i] = 12345 if out[i] != 12345 else 54321
    return out


def _gap():
    return float(os.environ.get("MLSYS_REHEARSAL_VLLM_GAP", "3.0"))


def _target_ids(cell, toks):
    """The ids a correct target-only implementation would emit.

    Taken from the RASD target-only sidecar for this document when one exists,
    because the cross-check compares against exactly that; otherwise a
    deterministic stand-in so the stub still produces a row.
    """
    sc = (cell.get("rasd_target") or {})
    ids = sc.get("ids") or []
    if len(ids) >= toks:
        return [int(i) for i in ids[:toks]]
    seed = zlib.crc32(f"vllm|{cell.get('doc_id')}|{toks}".encode())
    rng = random.Random(seed)
    return [int(i) for i in ids] + [rng.randrange(1, 30000)
                                    for _ in range(toks - len(ids))]


def main() -> int:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--prompt-ids-from-sidecars")
    ap.add_argument("--prompt-ids")
    ap.add_argument("--documents")
    ap.add_argument("--quantizations", nargs="+", default=[])
    ap.add_argument("--rasd-target-sidecars")
    ap.add_argument("--compare-out")
    ap.add_argument("--context-lengths", nargs="+", type=int, default=[])
    ap.add_argument("--max-new-tokens", type=int, default=1024)
    ap.add_argument("--matched-max-new-tokens", type=int, default=1024)
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--tensor-parallel-size", type=int, default=8)
    ap.add_argument("--target-revision", default="")
    ap.add_argument("--draft-revision", default="")
    ap.add_argument("--target-revisions", default="")
    ap.add_argument("--draft-revisions", default="")
    args = ap.parse_args()

    cells = (vb.load_rasd_cells(pathlib.Path(args.prompt_ids_from_sidecars))
             if args.prompt_ids_from_sidecars else [])
    if args.documents:
        want = {d.strip() for d in args.documents.split(",") if d.strip()}
        cells = [c for c in cells if c["doc_id"] in want]
    if not cells:
        raise SystemExit(
            "the rehearsal's vLLM stub found no RASD token sidecars to pair "
            "against; without them every row is unit_matched=no and the stage "
            "would rehearse a comparison that never happened")

    tmap = vb._parse_revision_map(args.target_revisions)
    dmap = vb._parse_revision_map(args.draft_revisions)

    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for model in args.models:
        for ctx in args.context_lengths:
            # Mirror the real script: a rung with no sidecar still produces a
            # row, with the prompt generated from the model and the row flagged
            # not unit-matched. Silently emitting no row would hide the case.
            here = [c for c in cells if c["context_length"] == ctx]
            if not here:
                here = [{"doc_id": "", "prompt_ids": None,
                         "prompt_sha256": None, "sidecar": "",
                         "context_length": ctx}]
            for cell in here:
                rng = random.Random(zlib.crc32(
                    f"{model}|{ctx}|{cell['doc_id']}".encode()))
                toks = args.max_new_tokens
                # The production-stack reference is the FAST one: a plausible
                # steady-state decode rate above the RASD target-only rate.
                tps = 30.0 + rng.random() * 8.0
                ttft = 0.9 + rng.random() * 0.2
                wall = ttft + (toks - 1) / tps
                tgt_rev = tmap.get(model) or args.target_revision
                drf_rev = dmap.get(model) or args.draft_revision
                given = (list(cell["prompt_ids"])
                         if cell["prompt_ids"] is not None else None)
                reported = _reported_ids(given)
                # The ids vLLM "reported" consuming: the equality the real
                # worker asserts on vLLM's own `prompt_token_ids`, produced here
                # by a path independent of the input. Never echo the input:
                # `prompt_ids_verified` would then be a tautology.
                result = {
                    "status": "ok",
                    "eos_policy": vb.EOS_POLICY,
                    "vllm_version": vb.VLLM_PIN,
                    "quantization": "bfloat16",
                    "prompt_tokens": len(cell["prompt_ids"] or []),
                    "prompt_sha256": (vb._ids_sha(cell["prompt_ids"])
                                      if cell["prompt_ids"] else ""),
                    "prompt_ids_used_sha256": (
                        vb._ids_sha(reported) if reported else ""),
                    "prompt_ids_verified": (
                        "" if cell["prompt_ids"] is None
                        else ("yes" if reported == list(cell["prompt_ids"])
                              else ("no" if reported is not None
                                    else "unavailable"))),
                    "target_revision": tgt_rev, "draft_revision": drf_rev,
                    "output_tokens": toks,
                    "end_to_end_wall_s": round(wall, 4),
                    "ttft_s": round(ttft, 4),
                    "decode_only_wall_s": round((toks - 1) / tps, 4),
                    "throughput_tps_end_to_end": round(toks / wall, 4),
                    "throughput_tps_decode_only": round(
                        vb.rasd_decode_rate(toks, (toks - 1) / tps), 4),
                    "peak_mem_mb": 78000.0, "error": "", "error_class": "",
                    # The numerics this engine ran, as the production worker
                    # reports them: vLLM has no NF4 KV path, which is why the
                    # cross-check reports token agreement instead of scoring it.
                    "weight_precision": "int4_bnb" if args.quantizations
                    and args.quantizations != ["bfloat16"] else "bfloat16",
                    "kv_dtype": "bfloat16",
                    # What this "engine" emitted, and its own gap per position.
                    # The ids mirror the RASD target-only sidecar for the same
                    # document -- that is what a working cross-check looks like --
                    # and the gaps are decisive, so a divergence would be a
                    # MISMATCH rather than excused as a tie.
                    "output_token_ids": _perturbed(_target_ids(cell, toks)),
                    "output_gaps": [round(_gap(), 6) for _ in range(toks)],
                }
                # The PRODUCTION row path: `prompt_ids_from_engine` and the
                # unit-match verdict are decided by build_row, exactly as in the
                # real parent. The stub must not set the marker itself -- doing
                # so masked a bug where no real row could ever be unit-matched.
                rows.append(vb.build_row(
                    cell, model, ctx, cell["prompt_ids"], result,
                    attempt_idx=1, config_name="stub",
                    log_path=f"stub/{model}/{ctx}/{cell['doc_id']}", rope=None,
                    args=args, max_model_len=ctx, target_revision=tgt_rev,
                    draft_revision=drf_rev))

    # Attach the RASD target-only ids the cross-check will compare against, so
    # the stub's emitted ids agree with them (the production worker gets its ids
    # from vLLM, not from the sidecar).
    if args.rasd_target_sidecars:
        try:
            refs = vb.load_rasd_target_sidecars(args.rasd_target_sidecars)
        except SystemExit:
            refs = {}
        for r in rows:
            key = (r.get("doc_id") or "", int(r.get("context_length") or 0))
            ref = refs.get(key)
            if ref:
                # Same document, so a correct implementation emits the reference
                # ids -- and then the divergence knob is applied ON TOP, so the
                # three outcomes the rehearsal drives are the comparison's own
                # verdicts rather than the stub's assertions.
                r["output_token_ids"] = _perturbed(
                    ref["ids"][:args.max_new_tokens])
                r["_output_token_ids"] = r["output_token_ids"]
    vb.write_rows(out, rows, append=False)
    n_unit = sum(1 for r in rows if r["unit_matched"] == "yes")
    print(f"stub wrote {len(rows)} vLLM rows to {out}")
    print(json.dumps({"unit_matched_yes": n_unit}))
    if n_unit == 0:
        # Mirror the production exit contract: a stage with no usable baseline
        # is FAILED, so the rehearsal's mismatch/unavailable cases exercise the
        # same non-zero path the real script takes.
        print("[FAIL] no row is unit_matched=yes: no usable baseline")
        return 1
    if args.compare_out:
        # The PRODUCTION cross-check, on the production rows: the stub does not
        # decide whether two engines agree, it only supplies a worker result.
        vdir = vb.write_vllm_token_sidecars(rows, args.compare_out)
        return vb.compare_targets(rows, args.rasd_target_sidecars,
                                  args.compare_out, vllm_sidecars_dir=vdir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

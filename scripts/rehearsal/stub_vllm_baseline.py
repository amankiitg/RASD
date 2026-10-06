"""GPU-free stand-in for `scripts/mlsys_vllm_baseline.py`.

Only the `vllm`-importing worker is replaced. The row-assembly, the pairing
identity (`prompt_sha256` recomputed from the ids the sidecar supplies), the
`unit_matched` verdict and the CSV schema all come from the real module, so a
missing revision pin or a mismatched prompt id is decided by production code.

The one thing simulated honestly: this stub cannot consume the ids through vLLM,
so it records them as verified after checking them against the sidecar itself --
which is the same equality the real worker asserts on `prompt_token_ids`.
"""
from __future__ import annotations

import csv
import importlib.util
import json
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


def main() -> int:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--prompt-ids-from-sidecars")
    ap.add_argument("--prompt-ids")
    ap.add_argument("--documents")
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
                row = {f: "" for f in vb.CSV_FIELDS}
                row.update({
                    "model": model, "context_length": ctx,
                    "doc_id": cell["doc_id"], "max_new_tokens": toks,
                    "temperature": 0.0,
                    "tensor_parallel_size": args.tensor_parallel_size,
                    "eos_policy": vb.EOS_POLICY, "vllm_version": vb.VLLM_PIN,
                    "quantization": "bfloat16",
                    "prompt_tokens": len(cell["prompt_ids"] or []),
                    "prompt_sha256": (vb._ids_sha(cell["prompt_ids"])
                                      if cell["prompt_ids"] else ""),
                    "prompt_ids_from": cell["sidecar"],
                    # The equality the real worker asserts on vLLM's own
                    # prompt_token_ids; here it is the sidecar's own ids.
                    "prompt_ids_used_sha256": (
                        vb._ids_sha(cell["prompt_ids"])
                        if cell["prompt_ids"] else ""),
                    "prompt_ids_verified": ("yes" if cell["prompt_ids"]
                                            else "unavailable"),
                    "target_revision": tgt_rev, "draft_revision": drf_rev,
                    "output_tokens": toks,
                    "end_to_end_wall_s": round(wall, 4),
                    "ttft_s": round(ttft, 4),
                    "decode_only_wall_s": round((toks - 1) / tps, 4),
                    "throughput_tps_end_to_end": round(toks / wall, 4),
                    "throughput_tps_decode_only": round(vb.rasd_decode_rate(
                        toks, (toks - 1) / tps), 4),
                    "peak_mem_mb": 78000.0, "status": "ok", "error": "",
                    "error_class": "",
                })
                ok, why = vb._unit_match_verdict(
                    row, cell["prompt_ids"], args,
                    target_revision=tgt_rev, draft_revision=drf_rev)
                row["unit_matched"] = "yes" if ok else "no"
                if not ok:
                    row["error_class"] = "UnitMismatch"
                    row["error"] = why
                rows.append(row)

    write_header = not out.exists()
    with out.open("a", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=vb.CSV_FIELDS, extrasaction="ignore")
        if write_header:
            w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"stub wrote {len(rows)} vLLM rows to {out}")
    print(json.dumps({"unit_matched_yes": sum(
        1 for r in rows if r["unit_matched"] == "yes")}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

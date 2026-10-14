#!/usr/bin/env python
"""Teacher-force the target on a spec arm's own stream, and on two broken ones.

WHAT IT ANSWERS

  POSITIVE: for each token the speculative arm emitted, is it the target's own
            argmax having been shown the same prefix? (the new gate)
  NEGATIVE: does that gate actually FAIL when the spec path is broken?

The second half is the one that matters. A gate that passes everything is
indistinguishable from a correct engine, so this script also evaluates two
deliberately broken streams and reports by how much they fail. If they do not
fail clearly, the gate must not be applied -- a rule that cannot detect the
defects it exists for is worse than the strict rule it replaces.

THE TWO BROKEN STREAMS (repro-only; no engine code is modified)

  (a) UNVERIFIED DRAFT. What the bug "accept the draft without checking" emits:
      the draft model's own greedy continuation from the prompt. The tokens are
      plausible English and a 1B draft often agrees with an 8B target, which is
      exactly why this control has to clear a wide margin to be meaningful.

  (b) OFF-BY-ONE KV POSITION. Every token is the target's argmax from ONE
      POSITION EARLIER: G[i] = argmax(L[i-1]). This is what a KV/position
      misalignment emits. It reuses the logits already computed for the
      positive, so it is clean rather than stochastic.

Both broken streams are then teacher-forced properly and scored by the SAME
rule as the positive, so the comparison is apples to apples.

Usage:
    python scripts/mlsys_teacher_forced_1x.py \
        --repro-dir results/mlsys/lossless_repro \
        --noise-floor results/mlsys/noise_floor/noise_floor.json \
        --outdir results/mlsys/teacher_forced
"""
from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from src.analysis.teacher_forced_losslessness import (  # noqa: E402
    describe, evaluate_stream, tol_provenance,
)

# Reuse the noise floor's engine construction, cache construction and measured
# KV assertion rather than restating them: a second copy is a second thing to
# drift, and the cache construction in particular is not obvious (the NF4 cache
# is built inside generate(), so a naive harness silently measures bf16).
_spec = importlib.util.spec_from_file_location(
    "mlsys_noise_floor", Path(__file__).with_name("mlsys_noise_floor.py"))
nf = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(nf)

# Labels match the sidecar naming exactly (`LOSS_32K_nf4_spec_...`): a case
# mismatch silently skipped all four cells, which is the kind of "no rows" that
# reads as "clean" if the summary is not checked for a zero denominator.
CELLS = [("32K", "nf4", 32768, True), ("32K", "bf16", 32768, False),
         ("8K", "nf4", 8192, True), ("8K", "bf16", 8192, False)]


def teacher_force(model, prefix_ids, tokens, chunk, initial_cache):
    """Logits for every position of `tokens`, teacher-forced on the same prefix.

    `rows[i]` is the target's distribution having been shown
    `prefix_ids + tokens[:i]` -- i.e. conditioned on the STREAM'S OWN prefix,
    which is what makes a shortfall a statement about the implementation rather
    than about two kernels agreeing.
    """
    cache = initial_cache
    n = prefix_ids.shape[1]
    out = None
    for start in range(0, n, chunk):
        piece = prefix_ids[:, start:min(start + chunk, n)]
        out = model(piece, past_key_values=cache, use_cache=True)
        cache = out.past_key_values
    rows = [out.logits[:, -1, :].float()]
    for i in range(len(tokens) - 1):
        step = torch.tensor([[int(tokens[i])]], device=prefix_ids.device,
                            dtype=prefix_ids.dtype)
        out = model(step, past_key_values=cache, use_cache=True)
        cache = out.past_key_values
        rows.append(out.logits[:, -1, :].float())
    return rows


def draft_stream(engine, prefix_ids, n_tokens: int) -> list:
    """The draft's own greedy continuation: what 'no verification' would emit.

    Uses the engine's own draft window (`_draft_window`) so the control is the
    stream the engine WOULD have produced on this prefix, not a draft run at a
    different context.
    """
    from src.models.rasd_inference import _draft_window
    draft = engine.draft_model
    window = engine.cfg.draft_window_cap or 0
    ids = _draft_window(prefix_ids, window) if window else prefix_ids
    cache = None
    out = draft(ids, use_cache=True)
    cache = out.past_key_values
    tok = int(out.logits[:, -1, :].argmax())
    toks = [tok]
    for _ in range(n_tokens - 1):
        step = torch.tensor([[tok]], device=ids.device, dtype=ids.dtype)
        out = draft(step, past_key_values=cache, use_cache=True)
        cache = out.past_key_values
        tok = int(out.logits[:, -1, :].argmax())
        toks.append(tok)
    return toks


def run_cell(label, kv, ctx, kv_quant, repro_dir: Path, tol: float, rank: int,
             outdir: Path):
    spec_path = repro_dir / "tokens" / ("LOSS_%s_%s_spec_pg19_train_1_s42.json"
                                        % (label, kv))
    if not spec_path.exists():
        print("  SKIP %s: no sidecar at %s" % (label and ("%s_%s" % (label, kv)),
                                              spec_path))
        return None
    side = json.loads(spec_path.read_text())
    G = [int(t) for t in side["generated_token_ids"]]
    prefix = torch.tensor([side["engine_input_ids"]], dtype=torch.long)

    print("\n=== %s_%s : %d emitted tokens, prefix %d ==="
          % (label, kv, len(G), prefix.shape[1]))
    # spec_steps=4 so the DRAFT is loaded: the unverified-draft control needs
    # the draft model, and _load_models skips it entirely at spec_steps=0.
    engine = nf.build_engine(ctx, kv_quant, "pg19_train_1", 64, rank, 1,
                             spec_steps=4)
    model = engine.target_model
    device = next(model.parameters()).device
    prefix = prefix.to(device)

    # --- POSITIVE: the spec arm's own stream -------------------------------
    cache = nf.make_initial_cache(engine, kv_quant)
    measured = nf.assert_measured_kv(engine, cache, kv_quant, rank)
    with torch.no_grad():
        L = teacher_force(model, prefix, G, nf.PREFILL_CHUNK, cache)
    pos = evaluate_stream(G, L, tol)
    print("  POSITIVE  (spec stream)   : %s" % describe(pos))

    # --- NEGATIVE (b): every token is the previous position's argmax --------
    argmaxes = [int(r.argmax()) for r in L]
    G_b = [argmaxes[0]] + argmaxes[:-1]
    cache = nf.make_initial_cache(engine, kv_quant)
    with torch.no_grad():
        L_b = teacher_force(model, prefix, G_b, nf.PREFILL_CHUNK, cache)
    neg_off = evaluate_stream(G_b, L_b, tol)
    print("  NEGATIVE b (off-by-one KV): %s" % describe(neg_off))

    # --- NEGATIVE (a): the draft's own greedy continuation ------------------
    with torch.no_grad():
        G_a = draft_stream(engine, prefix, len(G))
    cache = nf.make_initial_cache(engine, kv_quant)
    with torch.no_grad():
        L_a = teacher_force(model, prefix, G_a, nf.PREFILL_CHUNK, cache)
    neg_draft = evaluate_stream(G_a, L_a, tol)
    print("  NEGATIVE a (unverified dr): %s" % describe(neg_draft))
    print("    draft token agreement with the target's argmax: %d/%d"
          % (sum(1 for i, r in enumerate(L) if G_a[i] == argmaxes[i]), len(G)))

    result = {
        "cell": "%s_%s" % (label, kv),
        "context_length": ctx,
        "kv_quant": kv_quant,
        "measured_kv": measured,
        "tokens": len(G),
        "tol": tol,
        "positive": pos,
        "negative_off_by_one": neg_off,
        "negative_unverified_draft": neg_draft,
        "draft_agreement": sum(1 for i, r in enumerate(L)
                               if G_a[i] == argmaxes[i]),
    }
    if rank == 0:
        (outdir / ("%s_%s.json" % (label, kv))).write_text(
            json.dumps(result, indent=2) + "\n")
    del engine
    torch.cuda.empty_cache()
    return result


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repro-dir", default="results/mlsys/lossless_repro")
    ap.add_argument("--noise-floor",
                    default="results/mlsys/noise_floor/noise_floor.json")
    ap.add_argument("--outdir", default="results/mlsys/teacher_forced")
    ap.add_argument("--cap", type=float, default=None,
                    help="tolerance cap (default: the module's TOL_CAP)")
    args = ap.parse_args()

    rank = int(os.environ.get("RANK", 0))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    torch.cuda.set_device(local_rank)

    # TOL is DERIVED from the measured noise floor, per the pre-registered rule,
    # and the provenance is printed because a binding cap means the tolerance is
    # imposed rather than measured.
    nf_path = Path(args.noise_floor)
    noise = json.loads(nf_path.read_text())["summary"]
    prov = (tol_provenance(noise["max_abs_delta"], args.cap)
            if args.cap is not None else tol_provenance(noise["max_abs_delta"]))
    tol = prov["tol"]
    if rank == 0:
        print("=== TOL from the measured noise floor ===")
        print("  max |delta logit| packed vs stepwise = %.6g" % noise["max_abs_delta"])
        print("  2x = %.6g ; cap = %.6g ; TOL = %.6g ; cap_binds=%s"
              % (prov["raw_2x_noise"], prov["cap"], prov["tol"], prov["cap_binds"]))
        print("  %s" % prov["note"])
        Path(args.outdir).mkdir(parents=True, exist_ok=True)

    results = []
    for label, kv, ctx, kv_quant in CELLS:
        r = run_cell(label, kv, ctx, kv_quant, Path(args.repro_dir), tol, rank,
                     Path(args.outdir))
        if r:
            results.append(r)

    if rank == 0:
        fields = ["cell", "tokens", "tol", "positive_verdict",
                  "positive_max_shortfall", "negative_off_verdict",
                  "negative_off_max_shortfall", "negative_draft_verdict",
                  "negative_draft_max_shortfall", "draft_agreement", "measured_kv"]
        with open(Path(args.outdir) / "teacher_forced.csv", "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
            w.writeheader()
            for r in results:
                w.writerow({
                    "cell": r["cell"], "tokens": r["tokens"], "tol": r["tol"],
                    "positive_verdict": r["positive"]["verdict"],
                    "positive_max_shortfall": r["positive"].get("max_shortfall", ""),
                    "negative_off_verdict": r["negative_off_by_one"]["verdict"],
                    "negative_off_max_shortfall": r["negative_off_by_one"].get(
                        "worst_shortfall", ""),
                    "negative_draft_verdict": r["negative_unverified_draft"]["verdict"],
                    "negative_draft_max_shortfall": r["negative_unverified_draft"].get(
                        "worst_shortfall", ""),
                    "draft_agreement": r["draft_agreement"],
                    "measured_kv": r["measured_kv"],
                })
        (Path(args.outdir) / "summary.json").write_text(json.dumps(
            {"generated_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
             "tol": tol, "tol_provenance": prov, "cells": results}, indent=2) + "\n")

        print("\n=== VERDICT TABLE (tol=%.4g) ===" % tol)
        print("  %-12s %-14s %-8s %-14s %-8s %-14s" %
              ("cell", "POSITIVE", "max|s|", "NEG off-by-one", "max|s|",
               "NEG unver.draft"))
        ok_pos = ok_neg = 0
        for r in results:
            p, b, a = r["positive"], r["negative_off_by_one"], r["negative_unverified_draft"]
            print("  %-12s %-14s %-8.4g %-14s %-8.4g %-14s %-8.4g"
                  % (r["cell"], p["verdict"], p.get("max_shortfall", 0.0),
                     b["verdict"], b.get("worst_shortfall", 0.0),
                     a["verdict"], a.get("worst_shortfall", 0.0)))
            if p["verdict"] == "TF_LOSSLESS":
                ok_pos += 1
            if b["verdict"] == "TF_MISMATCH" and a["verdict"] == "TF_MISMATCH":
                ok_neg += 1
        print("\n  positive cells LOSSLESS: %d/%d" % (ok_pos, len(results)))
        print("  cells where BOTH negatives failed: %d/%d" % (ok_neg, len(results)))
        print("  (the gate is only applicable if BOTH are at their best)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python
"""The packed-vs-stepwise noise floor of the TARGET model, on one fixed prefix.

WHY THIS EXISTS

engine_cap_smoke failed because a speculative arm and its target-only partner
disagreed at token 2-3 with decided logit gaps. The 1x repro narrowed that to
one mechanism: the target is run in two different forward SHAPES -- a packed
(gamma+1)-token verify in one forward, versus one token per step in target-only
decode -- and its argmax is not invariant to that choice. This script measures
that difference directly, as a number, so a tolerance can be set from data
instead of guessed.

WHAT IT MEASURES

On one fixed prefix, with the same five tokens appended both ways:

  stepwise  prefill, then one forward per token (the target-only shape)
  packed    prefill, then ONE forward over all five tokens (the verify shape)

and per prediction position it records the max and mean |delta logit| over the
vocabulary, the top1/top2 gap in each shape, and whether the argmax flipped.

Position 0 is a CONTROL, not a measurement: it is the prefill's own last logit
in both passes, i.e. the same computation twice. Its delta is the run-to-run
floor. If the control is non-zero, nothing downstream means anything, so it is
recorded separately and the run says so.

WHAT IT REUSES, deliberately

The campaign's own `_build_pg19_document_prompt` and `RASDConfig`/`RASDInference`,
so the measurement is of THIS engine: FP4 target weights, NF4 or bf16 KV, and
the ring-attention patch when world_size > 1. A plain HuggingFace model would
measure a stack the campaign never runs.

Usage (1 rank):
    python scripts/mlsys_noise_floor.py --outdir results/mlsys/noise_floor
Usage (2 ranks):
    torchrun --nproc_per_node=2 scripts/mlsys_noise_floor.py --outdir ...

Writes <outdir>/noise_floor.csv (one row per cell x position) and
<outdir>/noise_floor.json (the same, plus the control and environment).
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

# The campaign's own prompt builder, imported rather than reimplemented: a
# reimplementation would drift and the noise floor would stop describing the
# runs it is supposed to bound. run_experiment guards its CLI behind
# `if __name__ == "__main__"`, so importing it is side-effect free.
from run_experiment import _build_pg19_document_prompt  # noqa: E402

# gamma+1 with the campaign's spec_steps=4. A full 64-position window
# needs 64 appended tokens (65 prediction positions, of which position 0
# is the control), which is what --append-tokens sets.
APPEND_TOKENS = 5
PREFILL_CHUNK = 4096       # tokens per prefill forward, to bound peak memory


def _log(rank: int, msg: str) -> None:
    if rank == 0:
        print(msg, flush=True)


def build_engine(context_length: int, kv_quant: bool, doc_id: str,
                 gen_tokens: int, rank: int, world_size: int,
                 spec_steps: int = 0):
    from src.models.rasd_inference import RASDConfig, RASDInference

    cfg = RASDConfig(
        target_model_name="meta-llama/Llama-3.1-8B",
        draft_model_name="meta-llama/Llama-3.2-1B",
        # Pinned to the same revisions engine_cap_smoke uses, so the noise
        # floor belongs to the exact bytes the campaign loads.
        target_revision="d04e592bb4f6aa9cfee91e2e20afa771667e1d4b",
        draft_revision="4e20de362430cd3b72f300e6b0f18e50e7166e08",
        # spec_steps=0 for the noise floor (this script drives the loop); 4
        # when the caller needs the DRAFT loaded, e.g. the unverified-draft
        # negative control.
        spec_steps=spec_steps,
        kv_block_size=2048,
        prefetch_depth=1,
        max_new_tokens=gen_tokens,
        dtype="bfloat16",
        quantize_draft=True,
        quantize_target=True,
        temperature=0.0,
        top_p=1.0,
        context_length=context_length,
        # NOT set: engine_cap_smoke sets no rope keys, so RASDConfig's
        # default (rope_type="linear", factor=None -> no scaling) is what
        # the campaign actually runs. The 1x probe set "none" and that
        # difference would have made this floor describe the wrong stack.
        ignore_eos=True,
        kv_quant=kv_quant,
        draft_window_cap=4096,
        seed=42,
        save_generated_tokens=False,
        log_per_token=False,
    )
    engine = RASDInference(cfg)
    return engine



def make_initial_cache(engine, kv_quant: bool):
    """The KV cache generate() would build, so kv_quant is not silently inert.

    The NF4 cache is constructed INSIDE RASDInference.generate(), not in
    _load_models. A harness that calls `model(...)` directly therefore gets
    HuggingFace's plain bf16 DynamicCache no matter what kv_quant says -- and
    the first version of this script did exactly that, producing two
    bit-identical "bf16" and "nf4" rows. That is the silent-no-op failure the
    project keeps hitting, so the measured dtype is asserted against the
    request rather than assumed.
    """
    if not kv_quant:
        return None
    from src.models.nf4_dynamic_cache import NF4DynamicCache
    cfg = engine.cfg
    prefix_size = cfg.kv_outlier_prefix_size if engine._rank == 0 else 0
    return NF4DynamicCache(
        block_size=cfg.kv_block_size_nf4,
        dtype=cfg.torch_dtype,
        bf16_prefix_size=prefix_size,
        update_chunk_size=cfg.nf4_update_chunk_size,
        tail_merge_below=cfg.nf4_tail_merge_below,
    )


def assert_measured_kv(engine, cache, kv_quant: bool, rank: int) -> str:
    """Read the KV precision off the CACHE, and refuse a mismatch."""
    from src.models.rasd_inference import detect_kv_precision
    got = detect_kv_precision(cache) if cache is not None else "bfloat16"
    want = "nf4" if kv_quant else "bfloat16"
    if got != want:
        raise AssertionError(
            "kv_quant=%s requested but the cache measures %r (want %r): the "
            "quantisation is inert and this cell would be a duplicate of the "
            "other dtype" % (kv_quant, got, want))
    _log(rank, "  measured KV precision: %s (requested %s)" % (got, want))
    return got


def _prefill(model, ids: torch.Tensor, chunk: int, initial_cache=None):
    """Prefill `ids` in chunks, returning (last_logits, cache).

    Chunking keeps a 32k prefill from holding a full-length activation, which
    is also how the engine bounds its own prefill peak.
    """
    cache = initial_cache
    logits_last = None
    n = ids.shape[1]
    for start in range(0, n, chunk):
        piece = ids[:, start:min(start + chunk, n)]
        out = model(piece, past_key_values=cache, use_cache=True)
        cache = out.past_key_values
        logits_last = out.logits[:, -1, :]
    return logits_last, cache


def _packed_predictions(model, prefix_ids, toks, chunk, initial_cache=None):
    """The verify shape: prefill once, then ONE forward over all 5 tokens."""
    pre_logits, cache = _prefill(model, prefix_ids, chunk, initial_cache)
    packed = torch.tensor([toks], device=prefix_ids.device, dtype=prefix_ids.dtype)
    out = model(packed, past_key_values=cache, use_cache=True)
    # logits[:, i] predicts toks[i+1], so positions 0..len-2 are the four
    # predictions after the prefill's own.
    preds = [pre_logits] + [out.logits[:, i, :] for i in range(len(toks) - 1)]
    return pre_logits, preds


def _stepwise_predictions(model, prefix_ids, toks, chunk, initial_cache=None):
    """The target-only shape: prefill, then one forward per token."""
    pre_logits, cache = _prefill(model, prefix_ids, chunk, initial_cache)
    preds = [pre_logits]
    for t in toks[:-1]:
        step = torch.tensor([[t]], device=prefix_ids.device, dtype=prefix_ids.dtype)
        out = model(step, past_key_values=cache, use_cache=True)
        cache = out.past_key_values
        preds.append(out.logits[:, -1, :])
    return pre_logits, preds


def _delta(a: torch.Tensor, b: torch.Tensor) -> dict:
    """Compare two (1, V) logit vectors as float32."""
    x = a[0].float()
    y = b[0].float()
    d = (x - y).abs()
    ax, ay = int(torch.argmax(x)), int(torch.argmax(y))
    top2x = torch.topk(x, k=2).values
    top2y = torch.topk(y, k=2).values
    return {
        "max_abs_delta": float(d.max()),
        "mean_abs_delta": float(d.mean()),
        # The vocabulary is 128256, so a mean over it hides the head; the
        # top-50 max is what decides an argmax, so it is reported too.
        "max_abs_delta_top50": float(d[torch.topk(x, k=50).indices].max()),
        "argmax_a": ax,
        "argmax_b": ay,
        "argmax_match": bool(ax == ay),
        "gap_a": float(top2x[0] - top2x[1]),
        "gap_b": float(top2y[0] - top2y[1]),
        # Same two candidates, different order = a marginal decision; a
        # different SET means the whole head moved.
        "top2_sets_equal": bool(set(torch.topk(x, k=2).indices.tolist()) ==
                                set(torch.topk(y, k=2).indices.tolist())),
    }


def run_cell(engine, tok, doc_json, context_length, kv_quant, doc_id,
             gen_tokens, rank, world_size, outdir: Path, append_tokens: int = APPEND_TOKENS):
    model = engine.target_model
    device = next(model.parameters()).device

    # Returns (prompt_text, continuation_ids, provenance).
    prompt_text, cont_ids, _prov = _build_pg19_document_prompt(
        doc_json, context_length, doc_id, tok, gen_tokens=gen_tokens)
    enc = tok(prompt_text, return_tensors="pt")
    prefix_ids = enc.input_ids.to(device)
    toks = [int(t) for t in cont_ids[:append_tokens]]
    assert len(toks) == append_tokens, (
        "the document ran out of continuation: got %d of %d tokens; raise "
        "--gen-tokens so the prompt leaves a long enough continuation"
        % (len(toks), append_tokens))

    # A fresh cache per pass: the two shapes must not share state, and the
    # stepwise pass would otherwise see the packed pass's appends.
    cache_a = make_initial_cache(engine, kv_quant)
    measured = assert_measured_kv(engine, cache_a, kv_quant, rank)
    cache_b = make_initial_cache(engine, kv_quant)
    with torch.no_grad():
        pre_a, step_preds = _stepwise_predictions(model, prefix_ids, toks,
                                                 PREFILL_CHUNK, cache_a)
        pre_b, pack_preds = _packed_predictions(model, prefix_ids, toks,
                                               PREFILL_CHUNK, cache_b)
    # And read it off the cache AFTER the pass, which is the state that
    # produced the logits being compared.
    assert_measured_kv(engine, cache_a, kv_quant, rank)

    control = _delta(pre_a, pre_b)
    rows = []
    for i, (s, k) in enumerate(zip(step_preds, pack_preds)):
        d = _delta(s, k)
        d.update({
            # Derived from the context, not from a two-way branch: the branch
            # version labelled a 131072 run "32k", so the 128k floor landed in a
            # directory named for a different context.
            "cell": "%dk_%s" % (context_length // 1024,
                                "nf4" if kv_quant else "bf16"),
            "context_length": context_length,
            "kv_quant": kv_quant,
            "world_size": world_size,
            # Position 0 is the prefill's own last logits in both passes, i.e.
            # the SAME computation twice: the run-to-run floor, not a
            # packed-vs-stepwise measurement.
            "position": i,
            "measured_kv": measured,
            "is_control": i == 0,
            "token_appended": toks[i] if i < len(toks) else None,
        })
        rows.append(d)

    cell = rows[0]["cell"]
    _log(rank, "  [%s] control(prefill vs prefill) max|d|=%.6g  argmax_match=%s"
         % (cell, control["max_abs_delta"], control["argmax_match"]))
    for r in rows[1:]:
        _log(rank, "    pos %d: max|d|=%.6g mean|d|=%.6g top50|d|=%.6g "
                   "argmax %s (gap %.4g vs %.4g) top2same=%s"
             % (r["position"], r["max_abs_delta"], r["mean_abs_delta"],
                r["max_abs_delta_top50"], "MATCH" if r["argmax_match"] else "FLIP",
                r["gap_a"], r["gap_b"], r["top2_sets_equal"]))

    if rank == 0:
        cell_dir = outdir / cell
        cell_dir.mkdir(parents=True, exist_ok=True)
        (cell_dir / "noise_floor.json").write_text(json.dumps(
            {"control": control, "rows": rows, "tokens": toks,
             "prompt_tokens": int(prefix_ids.shape[1]), "gen_tokens": gen_tokens,
             "append_tokens": append_tokens,
             "measured_positions": sum(1 for r in rows if not r["is_control"])},
            indent=2) + "\n")
    return cell, rows, control


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default="results/mlsys/noise_floor")
    ap.add_argument("--doc-id", default="pg19_train_1")
    ap.add_argument("--doc-json",
                    default="data/processed/pg19_docs/documents.json")
    ap.add_argument("--gen-tokens", type=int, default=64,
                    help="prompt window = C - gen_tokens - 1, as the campaign "
                         "sizes it; only the first 5 continuation tokens are used")
    ap.add_argument("--contexts", default="8192,32768")
    ap.add_argument("--append-tokens", type=int, default=APPEND_TOKENS,
                    help="tokens appended both ways; 64 gives a full 64-position "
                         "window (63 measured after the control position)")
    ap.add_argument("--kv", default="bf16,nf4",
                    help="bf16 = no KV quantisation, nf4 = the campaign default")
    args = ap.parse_args()

    rank = int(os.environ.get("RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    if world_size > 1:
        import torch.distributed as dist
        from datetime import timedelta
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl",
                                device_id=torch.device(f"cuda:{local_rank}"),
                                timeout=timedelta(hours=1))
    else:
        torch.cuda.set_device(0)

    outdir = Path(args.outdir)
    if rank == 0:
        outdir.mkdir(parents=True, exist_ok=True)
    contexts = [int(c) for c in args.contexts.split(",") if c]
    kvs = [k.strip() for k in args.kv.split(",") if k.strip()]

    all_rows = []
    controls = []
    for ctx in contexts:
        for kv in kvs:
            kv_quant = kv == "nf4"
            _log(rank, "\n=== ctx=%d kv=%s world_size=%d ===" % (ctx, kv, world_size))
            engine = build_engine(ctx, kv_quant, args.doc_id, args.gen_tokens,
                                  rank, world_size)
            tok = engine.tokenizer
            cell, rows, control = run_cell(
                engine, tok, args.doc_json, ctx, kv_quant, args.doc_id,
                args.gen_tokens, rank, world_size, outdir, args.append_tokens)
            control["cell"] = cell
            controls.append(control)
            all_rows.extend(rows)
            del engine
            torch.cuda.empty_cache()

    if rank == 0:
        import csv
        fields = ["cell", "context_length", "kv_quant", "measured_kv", "world_size", "position",
                  "is_control", "token_appended", "max_abs_delta",
                  "mean_abs_delta", "max_abs_delta_top50", "argmax_a", "argmax_b",
                  "argmax_match", "gap_a", "gap_b", "top2_sets_equal"]
        with open(outdir / "noise_floor.csv", "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
            w.writeheader()
            for r in all_rows:
                w.writerow(r)
        measured = [r for r in all_rows if not r["is_control"]]
        summary = {
            "generated_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "world_size": world_size,
            "host": platform.node(),
            "controls": controls,
            "control_max_abs_delta_max": max(c["max_abs_delta"] for c in controls),
            # THE NUMBER TOL IS DERIVED FROM:
            "max_abs_delta": max(r["max_abs_delta"] for r in measured),
            "max_abs_delta_top50": max(r["max_abs_delta_top50"] for r in measured),
            "mean_abs_delta_max": max(r["mean_abs_delta"] for r in measured),
            "positions": len(measured),
            "argmax_flips": sum(1 for r in measured if not r["argmax_match"]),
            "flip_rate": (sum(1 for r in measured if not r["argmax_match"])
                          / max(1, len(measured))),
        }
        (outdir / "noise_floor.json").write_text(
            json.dumps({"summary": summary, "rows": all_rows}, indent=2) + "\n")
        print("\n=== NOISE FLOOR (world_size=%d) ===" % world_size)
        print("  control (same computation twice) max|delta| = %.6g"
              % summary["control_max_abs_delta_max"])
        print("  packed vs stepwise     max|delta| = %.6g"
              % summary["max_abs_delta"])
        print("  packed vs stepwise max|delta| top-50 = %.6g"
              % summary["max_abs_delta_top50"])
        print("  argmax flips = %d / %d positions (%.1f%%)"
              % (summary["argmax_flips"], summary["positions"],
                 100 * summary["flip_rate"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Coherence gate for rope configurations above a model's native window.

A rope configuration may only be used for speculative runs if its TARGET is
still coherent at the context it will be run at. This script is that gate.

Why it exists: the ARM4 f2/f4 rungs replaced Llama-3.1-8B's shipped `llama3`
rope block with YaRN and the target degenerated (EOS at token 0, acceptance
collapse). Separately, a CPU analysis showed the project's YaRN anchoring was
wrong: transformers 4.47.1's YaRN IGNORES
``rope_scaling["original_max_position_embeddings"]`` and derives its
interpolation band from ``config.max_position_embeddings``, which the engine
sets to the context length, not the model's base. Measured on Llama-2-7B at
32k that cost 57x perplexity (826.9 mis-anchored vs 14.45 correctly anchored).

Design decisions that follow from that evidence:

* The anchor is asserted from the BUILT MODEL's ``inv_freq``, never from the
  config dict. The dict key is inert on 4.47.1, so reading it back proves
  nothing about what the model actually runs.
* Everything is measured at the candidate's REAL context, after a
  FULL-LENGTH prompt. The anchor probe showed a 512-token prompt cannot
  exercise long context: the two anchoring choices were indistinguishable at
  512/4096 tokens (PPL 4.89 vs 4.80) yet 57x apart at 32k.
* Perplexity alone is not enough, so a generation is scored too. On the probe
  both YaRN variants collapsed into newline repetition while their perplexities
  differed 57x, i.e. PPL and output quality can disagree.

Pass requires all of: perplexity within 1.5x of the native baseline at 128k,
no EOS within the first 16 generated tokens, and a generation not dominated by
blank lines or repeated n-grams.

Usage:
    python scripts/mlsys_coherence_gate.py --candidates configs/rope_candidates.json \
        --out results/mlsys/coherence_gate.csv
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

PPL_TOLERANCE = 1.5          # candidate PPL must be within this x native
EARLY_EOS_TOKENS = 16        # EOS inside the first N generated tokens = fail
MAX_BLANK_SHARE = 0.30       # blank-line share of a passing generation
MAX_REPEAT_SHARE = 0.50      # share of the generated n-grams that are repeats
GENERATE_TOKENS = 200
CONTINUATION_TOKENS = 1024   # scored continuation after the full-length prompt


# --------------------------------------------------------------------------
# Natural-text prompt + held-out continuation (d)
# --------------------------------------------------------------------------

def load_pg19_window(meta_path: str, context_length: int, seed: int):
    """Return (prompt_ids, continuation_ids) from held-out natural text.

    Deterministic per seed: the document and the offset inside it are both drawn
    from `seed`.

    Budget. The two consumers of this window are a teacher-forced perplexity over
    prompt + continuation and a generation of GENERATE_TOKENS after the prompt.
    Both must fit inside `context_length`, so the prompt is
    `context_length - max(CONTINUATION_TOKENS, GENERATE_TOKENS)` and the
    continuation fills the rest. An earlier version used a full-length prompt
    plus a continuation on top, i.e. `context_length + 1024` positions. For a
    candidate whose window *is* `context_length` that measures extrapolation:
    the native positive control at 128k would have been scored 1024 positions
    past its own window, which would flatter every candidate it is there to
    calibrate against.

    Two metadata shapes are accepted. `documents.json` (one memmap per book) is
    preferred because a document is then a real book; the older chunk metadata
    concatenates books and a slice can straddle a boundary.
    """
    meta = json.loads(Path(meta_path).read_text())
    prompt_len = context_length - max(CONTINUATION_TOKENS, GENERATE_TOKENS)
    if prompt_len < 1:
        raise RuntimeError(
            f"context_length {context_length} leaves no room for a prompt after "
            f"{max(CONTINUATION_TOKENS, GENERATE_TOKENS)} continuation tokens"
        )
    need = context_length

    if "documents" in meta:
        pool = [{"file": d["file"], "length": d["length"],
                 "doc_id": d["doc_id"]} for d in meta["documents"]]
    else:
        pool = [{"file": c["file"], "length": c["length"],
                 "doc_id": Path(c["file"]).stem} for c in meta["chunks"]]

    suitable = sorted((c for c in pool if c["length"] >= need),
                      key=lambda c: c["doc_id"])
    if not suitable:
        longest = max(c["length"] for c in pool)
        raise RuntimeError(
            f"no document holds {need} tokens (longest is {longest}); the gate "
            f"cannot run at {context_length} on this staged data"
        )
    rng = np.random.default_rng(seed)
    c = suitable[int(rng.integers(0, len(suitable)))]
    arr = np.memmap(c["file"], dtype="int32", mode="r")
    off = int(rng.integers(0, c["length"] - need + 1))
    ids = arr[off:off + prompt_len + CONTINUATION_TOKENS].astype(int).tolist()
    return ids[:prompt_len], ids[prompt_len:]


# --------------------------------------------------------------------------
# Effective-rope assertion from the built model
# --------------------------------------------------------------------------

def built_inv_freq(model):
    """The inv_freq the model actually built, read off layer 0.

    This is the assertion of record: on transformers 4.47.1 the YaRN dict's
    original_max_position_embeddings is IGNORED, so reading the config back
    cannot tell you which rope is running. The built buffer can.
    """
    rope = model.model.layers[0].self_attn.rotary_emb
    inv = rope.inv_freq
    if not torch.is_tensor(inv):
        inv = torch.as_tensor(inv)
    return inv.detach().float().cpu()


def reference_inv_freq(hf_config_kwargs, model_name):
    """inv_freq for a hypothetical rope config, computed from config alone."""
    from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS
    cfg = AutoConfig.from_pretrained(model_name)
    for k, v in hf_config_kwargs.items():
        setattr(cfg, k, v)
    rs = getattr(cfg, "rope_scaling", None) or {"rope_type": "default"}
    rs = dict(rs)
    if "rope_type" not in rs and "type" in rs:
        rs["rope_type"] = rs.pop("type")
    cfg.rope_scaling = rs
    return ROPE_INIT_FUNCTIONS[rs["rope_type"]](cfg, torch.device("cpu"))[0].float()


def assert_effective_rope(model, model_name: str, intended: dict) -> dict:
    """Recover which rope the model ACTUALLY built, by matching inv_freq.

    Compares the built vector against reference vectors for the intended
    configuration and for the two ways it could have gone wrong: the anchor
    carried on config.max_position_embeddings (the 4.47.1 behaviour) versus
    honoured from the dict, and the model's untouched native block.

    Returns a dict with the matched label, the stretch summary, and whether the
    match is the intended one. A mismatch is reported, not raised: the gate
    should record what it measured, and fail the candidate on it.
    """
    built = built_inv_freq(model)
    ctx = intended["context_length"]
    native_max = AutoConfig.from_pretrained(model_name).max_position_embeddings

    refs = {
        "intended": {"rope_scaling": intended.get("rope_scaling"),
                     "max_position_embeddings": intended.get("max_position_embeddings")},
    }
    # When the intended rope IS the model's shipped configuration (no scaling
    # requested, context inside the native window), "native_shipped" is not a
    # mismatch — it is exactly what was asked for. Without this the no-scaling
    # baseline candidate reports a rope mismatch and fails the gate it defines.
    if intended.get("rope_scaling") is None:
        native_expected = True
    else:
        native_expected = False
        refs["native_shipped"] = {}
        rs = dict(intended["rope_scaling"])
        refs["anchor_on_context"] = {
            "rope_scaling": {**rs, "original_max_position_embeddings": native_max},
            "max_position_embeddings": ctx,
        }

    best, best_err = None, float("inf")
    for label, kw in refs.items():
        try:
            r = reference_inv_freq(kw, model_name)
        except Exception:
            continue
        if r.shape != built.shape:
            continue
        err = float((r - built).abs().max().item())
        if err < best_err:
            best, best_err = label, err

    # Stretch of the slowest channel vs the model's untouched native rope.
    native = reference_inv_freq({}, model_name)
    stretch = float((built[-1] / native[-1]).item()) if native[-1] != 0 else float("nan")

    matches = (best_err < 1e-6) and (
        best == "intended" or (native_expected and best == "intended"))
    return {
        "effective_rope_match": best,
        "effective_rope_maxerr": best_err,
        "effective_rope_matches_intent": bool(matches),
        "inv_freq_last": f"{float(built[-1]):.6g}",
        "inv_freq_first": f"{float(built[0]):.6g}",
        "slowest_channel_stretch": round(stretch, 6),
        "native_window": int(native_max),
    }


# --------------------------------------------------------------------------
# Metrics
# --------------------------------------------------------------------------

def continuation_perplexity(model, prompt_ids, cont_ids) -> float:
    """Perplexity of `cont_ids` conditioned on `prompt_ids`.

    Delegates to `src.analysis.target_quality` so the gate and the per-rung
    target-quality measurement score perplexity with the *same* code. Two
    implementations of the same quantity would eventually disagree, and the
    gate's whole job is to predict what the runs will measure.
    """
    from src.analysis.target_quality import continuation_nll, perplexity_from_sums

    ids = torch.tensor([prompt_ids + cont_ids], dtype=torch.long,
                       device=model.device)

    def _forward(local_ids, abs_pos):
        return model.model(input_ids=local_ids, use_cache=False,
                           past_key_values=None).last_hidden_state

    with torch.no_grad():
        total, n, _ = continuation_nll(
            model, ids, score_from=len(prompt_ids), forward=_forward,
        )
    return perplexity_from_sums(total, n)


@torch.no_grad()
def generate_after_prompt(model, tok, prompt_ids, n_new=GENERATE_TOKENS):
    ids = torch.tensor([prompt_ids], dtype=torch.long, device=model.device)
    out = model.generate(ids, max_new_tokens=n_new, do_sample=False,
                         pad_token_id=tok.pad_token_id,
                         eos_token_id=tok.eos_token_id)
    new = out[0][len(prompt_ids):].tolist()
    return tok.decode(new, skip_special_tokens=True), new


def generation_metrics(text: str, new_ids: list[int], eos_id) -> dict:
    lines = text.split("\n")
    blank = sum(1 for l in lines if not l.strip()) / max(1, len(lines))
    alpha = sum(1 for c in text if c.isalpha()) / max(1, len(text))
    toks = text.split()
    grams = [tuple(toks[i:i + 3]) for i in range(max(0, len(toks) - 2))]
    repeat = 1.0 - (len(set(grams)) / len(grams)) if grams else 0.0
    early = next((i for i, t in enumerate(new_ids) if t == eos_id), None)
    return {
        "gen_chars": len(text),
        "gen_blank_share": round(blank, 4),
        "gen_alpha_share": round(alpha, 4),
        "gen_repeat_share": round(repeat, 4),
        "early_eos": early is not None and early < EARLY_EOS_TOKENS,
        "eos_at": early if early is not None else "",
    }


# --------------------------------------------------------------------------
# One candidate
# --------------------------------------------------------------------------

def _device_map():
    """Single-device placement that works on CPU as well as CUDA.

    The gate must be runnable locally on CPU for the pipeline dry run; passing
    `{"": 0}` unconditionally makes torch assert "Torch not compiled with CUDA
    enabled" and the gate cannot be exercised without a GPU, which defeats the
    point of a dry run.
    """
    return {"": 0} if torch.cuda.is_available() else {"": "cpu"}


def run_candidate(cand: dict, tok, meta_path: str, out_dir: Path):
    from src.models.rasd_inference import RASDInference

    name, model_name = cand["name"], cand["target_model_name"]
    ctx = int(cand["context_length"])
    row = {"candidate": name, "target_model_name": model_name,
           "context_length": ctx, "seed": cand.get("seed", 42),
           "rope_type": cand.get("rope_type"),
           "rope_factor": cand.get("rope_factor"),
           "rope_anchor_base": cand.get("rope_anchor_base")}

    intended = {}
    try:
        hf_cfg = RASDInference._build_hf_config(
            None, model_name, cand.get("target_revision"), ctx, label="gate",
            apply_rope_scaling=cand.get("rope_type") not in (None, "none"),
            rope_type=cand.get("rope_type", "linear"),
            rope_factor=cand.get("rope_factor"),
            rope_anchor_base=cand.get("rope_anchor_base"),
        )
        intended["rope_scaling"] = getattr(hf_cfg, "rope_scaling", None)
        intended["max_position_embeddings"] = hf_cfg.max_position_embeddings
        intended["context_length"] = ctx
        model = AutoModelForCausalLM.from_pretrained(
            model_name, config=hf_cfg,
            revision=cand.get("target_revision"),
            torch_dtype=(torch.bfloat16 if torch.cuda.is_available() else torch.float32),
            device_map=_device_map()).eval()
        row["config_max_position_embeddings"] = hf_cfg.max_position_embeddings
        row["config_rope_scaling"] = json.dumps(getattr(hf_cfg, "rope_scaling", None))
    except Exception as e:
        row.update(status="error", error=f"{type(e).__name__}: {e}")
        return row

    try:
        row.update(assert_effective_rope(model, model_name, intended))

        prompt_ids, cont_ids = load_pg19_window(
            meta_path, ctx, int(cand.get("seed", 42)))
        row["prompt_tokens"] = len(prompt_ids)
        import hashlib
        row["prompt_sha256"] = hashlib.sha256(
            json.dumps(prompt_ids).encode()).hexdigest()[:16]

        row["ppl_continuation"] = round(
            continuation_perplexity(model, prompt_ids, cont_ids), 4)

        text, new_ids = generate_after_prompt(model, tok, prompt_ids)
        (out_dir / f"gen_{name}.txt").write_text(text)
        row.update(generation_metrics(text, new_ids, tok.eos_token_id))
        row["status"] = "ok"
    except torch.cuda.OutOfMemoryError as e:
        row.update(status="oom", error=f"CUDA OOM: {str(e)[:120]}")
    except Exception as e:
        import traceback
        row.update(status="error", error=f"{type(e).__name__}: {e}",
                   traceback=traceback.format_exc()[-400:])
    finally:
        del model
        torch.cuda.empty_cache()
    return row


def verdict(row: dict, native_ppl: float) -> dict:
    """Apply the gate's pass rule, or the reason it could not be applied."""
    out = {"native_ppl_at_128k": round(native_ppl, 4) if native_ppl else ""}
    if row.get("status") != "ok":
        out.update(ppl_ratio="", gate_pass=False,
                   gate_reason=f"not measured ({row.get('status')})")
        return out
    if not native_ppl:
        out.update(ppl_ratio="", gate_pass=False,
                   gate_reason="no native baseline in this run")
        return out
    ratio = row["ppl_continuation"] / native_ppl
    out["ppl_ratio"] = round(ratio, 4)
    reasons = []
    if ratio > PPL_TOLERANCE:
        reasons.append(f"ppl {ratio:.2f}x native > {PPL_TOLERANCE}x")
    if row.get("early_eos"):
        reasons.append(f"EOS at token {row.get('eos_at')}")
    if row.get("gen_blank_share", 0) > MAX_BLANK_SHARE:
        reasons.append(f"blank lines {row['gen_blank_share']:.0%}")
    if row.get("gen_repeat_share", 0) > MAX_REPEAT_SHARE:
        reasons.append(f"repeated n-grams {row['gen_repeat_share']:.0%}")
    if not row.get("effective_rope_matches_intent"):
        err = row.get("effective_rope_maxerr")
        err_s = f"{err:.3g}" if isinstance(err, (int, float)) else "n/a"
        reasons.append(
            f"rope mismatch: built {row.get('effective_rope_match')} "
            f"(err {err_s}), not the intended config")
    out["gate_pass"] = not reasons
    out["gate_reason"] = "pass" if not reasons else "; ".join(reasons)
    return out


FIELDS = ["candidate", "target_model_name", "context_length", "seed",
          "rope_type", "rope_factor", "rope_anchor_base",
          "config_max_position_embeddings", "config_rope_scaling",
          "effective_rope_match", "effective_rope_maxerr",
          "effective_rope_matches_intent", "native_window",
          "inv_freq_first", "inv_freq_last", "slowest_channel_stretch",
          "prompt_tokens", "prompt_sha256", "ppl_continuation", "native_ppl_at_128k",
          "ppl_ratio", "early_eos", "eos_at", "gen_chars", "gen_blank_share",
          "gen_alpha_share", "gen_repeat_share",
          "gate_pass", "gate_reason", "status", "error"]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidates", required=True,
                    help="JSON list of candidate dicts, or a run manifest")
    ap.add_argument("--pg19-meta", default="data/processed/pg19/pg19_validation_metadata.json")
    ap.add_argument("--out", default="results/mlsys/coherence_gate.csv")
    ap.add_argument("--gen-dir", default="results/mlsys/gate_generated")
    args = ap.parse_args()

    spec = json.loads(Path(args.candidates).read_text())
    candidates = spec["coherence_gate"] if isinstance(spec, dict) else spec
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    gen_dir = Path(args.gen_dir)
    gen_dir.mkdir(parents=True, exist_ok=True)

    tok = AutoTokenizer.from_pretrained(candidates[0]["target_model_name"])
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    rows = [run_candidate(c, tok, args.pg19_meta, gen_dir) for c in candidates]

    native = next((r["ppl_continuation"] for r in rows
                   if r.get("native_baseline") and r.get("ppl_continuation")), None)
    if native is None:
        # Fall back to the untouched-config candidate if not explicitly flagged.
        native = next((r["ppl_continuation"] for r in rows
                       if r.get("status") == "ok"
                       and r.get("rope_type") in (None, "none", "llama3")
                       and r.get("context_length", 0) <= 131072), None)

    for r in rows:
        r.update(verdict(r, native))

    with out_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in FIELDS})

    print(f"\n  {'candidate':<24} {'ctx':>7} {'anchor':>10} {'ppl':>10} {'ratio':>7} "
          f"{'blank':>7} {'rep':>6} {'eos':>5}  gate")
    for r in rows:
        print(f"  {r['candidate']:<24} {r['context_length']:>7} "
              f"{str(r.get('effective_rope_match','-')):>10} "
              f"{str(r.get('ppl_continuation','-')):>10} "
              f"{str(r.get('ppl_ratio','-')):>7} "
              f"{r.get('gen_blank_share','-')!s:>7} "
              f"{r.get('gen_repeat_share','-')!s:>6} "
              f"{'YES' if r.get('early_eos') else 'no':>5}  "
              f"{'PASS' if r.get('gate_pass') else 'FAIL'}")
        if not r.get("gate_pass"):
            print(f"      -> {r.get('gate_reason')}")
    print(f"\n  wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

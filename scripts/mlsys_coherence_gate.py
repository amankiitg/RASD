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

# The pass tolerance, FIXED by the 2026-10-06 plan revision ("the gate tolerance
# is FIXED at 1.5x, not derived"). The earlier rule that derived it from the
# positive controls' measured spread was withdrawn before any run: three
# single-context measurements do not estimate the gate's noise, and every
# available estimator can only loosen the gate. This constant and the plan must
# agree; the dry run asserts it.
PPL_TOLERANCE = 1.5          # candidate PPL must be within this x native
EARLY_EOS_TOKENS = 16        # EOS inside the first N generated tokens = fail
MAX_BLANK_SHARE = 0.30       # blank-line share of a passing generation
MAX_REPEAT_SHARE = 0.50      # share of the generated n-grams that are repeats
GENERATE_TOKENS = 200
CONTINUATION_TOKENS = 1024   # scored continuation after the full-length prompt


# --------------------------------------------------------------------------
# Natural-text prompt + held-out continuation (d)
# --------------------------------------------------------------------------

def load_pg19_window(meta_path: str, context_length: int, seed: int,
                     bos_id: int | None = None):
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
    # The engine prepends a leading BOS at encode time, so the gate's prompt
    # must include one too, and the text budget shrinks by that token. Without
    # this the gate measures a sequence one token shorter than the runs and
    # (before the fix) 1024 tokens LONGER, which put the 128k native control
    # past its own window.
    bos = 1 if bos_id is not None else 0
    prompt_len = context_length - max(CONTINUATION_TOKENS, GENERATE_TOKENS) - bos
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
    prompt = ids[:prompt_len]
    if bos_id is not None:
        prompt = [int(bos_id)] + prompt
    return prompt, ids[prompt_len:]


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


def _declared_reference(intended: dict, model_name: str, anchor_override=None):
    """inv_freq implied by the candidate's DECLARED parameters.

    This is the whole point of the assertion, so it must not be a replay of the
    configuration the model was built with. An earlier version compared the
    built `inv_freq` against a reference recomputed from the very `hf_cfg` used
    to build the model, via the same `ROPE_INIT_FUNCTIONS`. That is a tautology:
    it always matched, so it could not detect the historical mis-anchoring
    (routing the YaRN anchor onto `config.max_position_embeddings = ctx`),
    because the reference inherited the same mis-anchoring.

    Here the reference is built from what the CANDIDATE asked for — rope type,
    factor, and anchor — so a configuration that silently anchors somewhere else
    matches a *different* reference and is reported as a mismatch.

    `anchor_override` forces the anchor (used to build the deliberately
    mis-anchored reference for comparison).
    """
    from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS
    from types import SimpleNamespace

    cfg = AutoConfig.from_pretrained(model_name)
    native_max = int(cfg.max_position_embeddings)
    rtype = intended.get("rope_type")
    anchor = anchor_override
    if anchor is None:
        anchor = intended.get("rope_anchor_base")
    if anchor is None:
        # No declared anchor: use the model's SHIPPED
        # `original_max_position_embeddings`, not `max_position_embeddings`.
        # Llama-3.1-8B ships original_max_position_embeddings = 8192 while
        # max_position_embeddings = 131072, so taking the window as the anchor
        # rebuilt the historical mis-anchoring INSIDE the reference: the
        # reference then agreed with a wrongly anchored model, and the
        # assertion passed for exactly the bug it exists to catch.
        shipped_anchor = (getattr(cfg, "rope_scaling", None) or {}).get(
            "original_max_position_embeddings")
        anchor = native_max if shipped_anchor is None else shipped_anchor
    anchor = int(anchor)

    # Start from the model's SHIPPED rope dict, not a minimal one. Some rope
    # types need keys beyond type/factor/anchor — `llama3` reads
    # low_freq_factor and high_freq_factor — so a hand-built dict raises
    # KeyError and the gate would report an error instead of a verdict for
    # exactly the configurations it exists to check.
    shipped = dict(getattr(cfg, "rope_scaling", None) or {})
    if "rope_type" not in shipped and "type" in shipped:
        shipped["rope_type"] = shipped.pop("type")
    probe = SimpleNamespace(**{k: getattr(cfg, k) for k in
                               ("hidden_size", "num_attention_heads",
                                "head_dim", "rope_theta")
                               if hasattr(cfg, k)})

    if rtype in (None, "none"):
        # No scaling declared: the expectation is the model's own rope block.
        rs = shipped or {"rope_type": "default"}
        probe.max_position_embeddings = native_max
        probe.rope_scaling = rs
        return ROPE_INIT_FUNCTIONS[rs["rope_type"]](probe, torch.device("cpu"))[0].float()

    rs = dict(shipped)
    rs.update({
        "rope_type": rtype,
        "type": rtype,
        "factor": float(intended.get("rope_factor") or 1.0),
        "original_max_position_embeddings": anchor,
    })
    # transformers 4.47.1 YaRN reads the band from max_position_embeddings and
    # llama3 reads it from the dict; setting both keeps this honest for either.
    probe.max_position_embeddings = anchor
    probe.rope_scaling = rs
    return ROPE_INIT_FUNCTIONS[rtype](probe, torch.device("cpu"))[0].float()


def assert_effective_rope(model, model_name: str, intended: dict) -> dict:
    """Recover which rope the model ACTUALLY built, and whether it was asked for.

    Compares the built `inv_freq` against references derived from the
    candidate's declared parameters:

      `declared`         rope type / factor / anchor the candidate asked for
      `anchor_on_context` the same, but with the anchor replaced by the context
                         length — the historical 4.47.1 failure mode
      `native_shipped`   the model's untouched block

    `effective_rope_matches_intent` requires matching `declared`. Matching the
    mis-anchored reference instead is reported as a mismatch even though the
    model built successfully, which is the case that previously went unnoticed.

    A mismatch is reported, not raised: the gate should record what it measured
    and fail the candidate on it.
    """
    built = built_inv_freq(model)
    ctx = int(intended["context_length"])
    native_max = int(AutoConfig.from_pretrained(model_name).max_position_embeddings)
    rtype = intended.get("rope_type")

    refs = {"declared": _declared_reference(intended, model_name)}
    if rtype in (None, "none"):
        # Nothing was scaled, so `native_shipped` IS the declaration and the
        # tie is not a mismatch.
        refs["native_shipped"] = _declared_reference({}, model_name)
    elif rtype == "llama3":
        # llama3 reads original_max_position_embeddings from the dict, so
        # replacing the anchor with the context does not produce a different
        # rope: there is no context-anchoring failure mode to test for. Say so
        # rather than emitting a reference identical to `declared` under a
        # second label, which would make the label meaningless.
        refs["native_shipped"] = _declared_reference({}, model_name)
    else:
        refs["native_shipped"] = _declared_reference({}, model_name)
        refs["anchor_on_context"] = _declared_reference(
            intended, model_name, anchor_override=ctx)

    best, best_err = None, float("inf")
    errors = {}
    for label, r in refs.items():
        if r is None or r.shape != built.shape:
            continue
        err = float((r - built).abs().max().item())
        errors[label] = err
        if err < best_err:
            best, best_err = label, err

    native = _declared_reference({}, model_name)
    stretch = float((built[-1] / native[-1]).item()) if native[-1] != 0 else float("nan")

    return {
        "effective_rope_match": best,
        "effective_rope_maxerr": best_err,
        "effective_rope_errors": json.dumps(errors),
        "effective_rope_matches_intent": bool(best_err < 1e-6 and best == "declared"),
        "declared_anchor": int(intended.get("rope_anchor_base") or native_max),
        "built_anchor_is_context": bool(
            errors.get("anchor_on_context", float("inf")) < 1e-6
            and errors.get("declared", 0.0) > 1e-6),
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


def eos_ids(tok) -> set:
    """Every token id that terminates a generation.

    `tok.eos_token_id` is only `<|end_of_text|>` (128001) for Llama-3.1;
    `<|eot_id|>` (128009) is a separate special token that also stops
    generation. Checking one of them lets a generation that immediately emits
    the other pass the early-EOS test.
    """
    ids = set()
    for attr in ("eos_token_id",):
        v = getattr(tok, attr, None)
        if v is not None:
            ids.add(int(v))
    for name in ("<|eot_id|>", "<|end_of_text|>", "<|end_of_turn|>",
                 "<|im_end|>", "</s>"):
        try:
            i = tok.convert_tokens_to_ids(name)
            if i is not None and int(i) >= 0 and i != tok.unk_token_id:
                ids.add(int(i))
        except Exception:
            pass
    return ids


def generation_metrics(text: str, new_ids: list[int], eos_id) -> dict:
    lines = text.split("\n")
    blank = sum(1 for l in lines if not l.strip()) / max(1, len(lines))
    alpha = sum(1 for c in text if c.isalpha()) / max(1, len(text))
    toks = text.split()
    grams = [tuple(toks[i:i + 3]) for i in range(max(0, len(toks) - 2))]
    repeat = 1.0 - (len(set(grams)) / len(grams)) if grams else 0.0
    eos_set = eos_id if isinstance(eos_id, (set, frozenset)) else {eos_id}
    early = next((i for i, t in enumerate(new_ids) if t in eos_set), None)
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
           "rope_anchor_base": cand.get("rope_anchor_base"),
           "reference_context": cand.get("reference_context"),
           "target_revision": cand.get("target_revision"),
           # Carried onto the row because the baseline lookup selects on it.
           # Without this the flag was read from the candidate dict list, which
           # worked only because `rows` happened to be built from the same dicts.
           "native_baseline": bool(cand.get("native_baseline", False)),
           "role": cand.get("role", ""),
           "expect": cand.get("expect", "")}

    intended = {}
    try:
        hf_cfg = RASDInference._build_hf_config(
            None, model_name, cand.get("target_revision"), ctx, label="gate",
            apply_rope_scaling=cand.get("rope_type") not in (None, "none"),
            rope_type=cand.get("rope_type", "linear"),
            rope_factor=cand.get("rope_factor"),
            rope_anchor_base=cand.get("rope_anchor_base"),
        )
        # The DECLARED parameters, deliberately not read back off `hf_cfg`. A
        # reference built from the config the model was built with cannot detect
        # a mis-anchored build; see _declared_reference.
        intended["rope_type"] = cand.get("rope_type")
        intended["rope_factor"] = cand.get("rope_factor")
        intended["rope_anchor_base"] = cand.get("rope_anchor_base")
        intended["context_length"] = ctx
        # What the builder actually produced, recorded for the audit only.
        intended["built_rope_scaling"] = json.dumps(getattr(hf_cfg, "rope_scaling", None))
        intended["built_max_position_embeddings"] = hf_cfg.max_position_embeddings
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
        row.update(generation_metrics(text, new_ids, eos_ids(tok)))
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
    out = {"native_ppl_reference": round(native_ppl, 4) if native_ppl else ""}
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


FIELDS = ["candidate", "target_model_name", "target_revision",
          "context_length", "reference_context", "seed",
          "rope_type", "rope_factor", "rope_anchor_base",
          "config_max_position_embeddings", "config_rope_scaling",
          "effective_rope_match", "effective_rope_maxerr",
          "effective_rope_matches_intent", "native_window",
          "inv_freq_first", "inv_freq_last", "slowest_channel_stretch",
          "prompt_tokens", "prompt_sha256", "ppl_continuation", "native_ppl_reference", "baseline_context",
          "ppl_ratio", "early_eos", "eos_at", "gen_chars", "gen_blank_share",
          "gen_alpha_share", "gen_repeat_share",
          "gate_pass", "gate_reason", "status", "error"]


def load_candidates(spec):
    """Accept a bare list, or a dict under any of the documented keys.

    The gate used to require `{"coherence_gate": [...]}` and would raise
    KeyError on anything else. The control and correction-evidence files use
    `{"candidates": [...]}`, so S0 (the stage that must run first) would have
    crashed on its own config rather than reporting a calibration failure. A
    loader that silently accepts one shape and not the other is a trap.
    """
    if isinstance(spec, list):
        return spec
    for key in ("coherence_gate", "candidates", "controls"):
        if key in spec:
            return spec[key]
    raise SystemExit(
        f"candidate file has no candidate list; expected one of "
        f"'coherence_gate', 'candidates', 'controls' or a bare JSON list, "
        f"found keys {sorted(spec)}"
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidates", required=True,
                    help="JSON list of candidate dicts, or a run manifest")
    ap.add_argument("--pg19-meta", default="data/processed/pg19/pg19_validation_metadata.json")
    ap.add_argument("--out", default="results/mlsys/coherence_gate.csv")
    ap.add_argument("--gen-dir", default="results/mlsys/gate_generated")
    args = ap.parse_args()

    spec = json.loads(Path(args.candidates).read_text())
    candidates = load_candidates(spec)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    gen_dir = Path(args.gen_dir)
    gen_dir.mkdir(parents=True, exist_ok=True)

    # One tokenizer PER MODEL, cached. A single tokenizer taken from
    # candidates[0] was used for every row, so the Llama-2 controls were
    # generated and EOS-checked with Llama-3.1's tokenizer -- a different vocab
    # and a different eos id, which corrupts exactly the two numbers
    # (early-EOS, degeneration share) that decide those controls' verdicts.
    _tok_cache: dict = {}

    def _tok_for(model_name: str):
        if model_name not in _tok_cache:
            t = AutoTokenizer.from_pretrained(model_name)
            if t.pad_token is None:
                t.pad_token = t.eos_token
            _tok_cache[model_name] = t
        return _tok_cache[model_name]

    rows = [run_candidate(c, _tok_for(c["target_model_name"]), args.pg19_meta,
                          gen_dir) for c in candidates]

    # The reference every candidate is judged against. It must be DECLARED, and
    # it must be measured at the candidate's own context.
    #
    # The previous fallback took the first ok row with rope_type in
    # (none, llama3) and context <= 131072. At S0 that selected P1_native_32k,
    # so the 128k positive control was judged against the 32k perplexity and the
    # column was still labelled `native_ppl_at_128k`. Since perplexity changes
    # with context and the loader picks a different offset per context, the whole
    # comparison was against the wrong measurement.
    baseline_rows = [r for r in rows
                     if r.get("native_baseline") and r.get("ppl_continuation")]
    if not baseline_rows:
        # Refuse rather than guess. Guessing here silently invalidates every
        # ratio in the file.
        raise SystemExit(
            "no candidate is flagged native_baseline=true, so there is no "
            "reference perplexity; refusing to compute ratios against a "
            "baseline nobody declared. Add native_baseline: true to the "
            "configuration that should serve as the reference."
        )

    def _baseline_for(r):
        """The baseline this candidate is to be judged against.

        `reference_context` decides, and it is DECLARED rather than inferred:

          * a candidate that declares one (an EXTENSION: 256k, 512k) is judged
            against the declared native baseline at that context. The plan
            states it this way -- the question for an extension is whether
            extending the window keeps the target coherent, and "coherent" is
            defined by the native configuration at the native window;
          * a candidate that declares none is judged against a baseline at its
            OWN context. That is what a calibration CONTROL needs: its job is to
            show the gate passes a configuration known to be correct at its own
            context, so a same-context reference is the right one there.

        A missing reference is reported as `no_baseline`, never substituted:
        a different context or model is a different reference, and guessing one
        turns "we did not measure this" into a number.
        """
        want_ctx = r.get("reference_context")
        if want_ctx in ("", None):
            want_ctx = r.get("context_length")
        want_ctx = int(want_ctx)
        same = [b for b in baseline_rows
                if b.get("context_length") == want_ctx
                and b.get("target_model_name") == r.get("target_model_name")]
        if same:
            return same[0]
        return None

    for r in rows:
        b = _baseline_for(r)
        r["native_ppl_reference"] = b["ppl_continuation"] if b else ""
        r["baseline_context"] = b["context_length"] if b else ""
        # Which reference this candidate declared. A 256k candidate judged
        # against a 128k baseline and one judged against a 256k baseline are
        # different measurements, and the CSV has to say which happened.
        r["reference_context"] = (r.get("reference_context")
                                  or r.get("context_length"))
        if b is None:
            r["gate_pass"] = False
            r["gate_reason"] = (
                f"no declared native baseline at reference context "
                f"{r.get('reference_context') or r.get('context_length')} for "
                f"{r.get('target_model_name')}; cannot compute a ratio against "
                f"one")
            r["status"] = r.get("status") or "no_baseline"
        else:
            r.update(verdict(r, b["ppl_continuation"]))

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

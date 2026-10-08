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
# Generation shares are judged RELATIVE to the candidate's own PAIRED baseline
# (2026-10-08 plan revision), not against an absolute threshold: PG-19 is
# hard-wrapped Gutenberg text, so a continuation's blank-line share is a fact
# about the passage that got drawn (dialogue vs narrative) rather than about the
# model. The band is deliberately WIDER than PPL_TOLERANCE because a 200-token
# generation is the noisier statistic, and deliberately no wider than 2.0
# because the one mis-anchored build that has to be caught sits at 2.53x.
GEN_SHARE_TOLERANCE = 2.0    # candidate blank/repeat share vs its baseline's
# Degeneration, not quality. These are the only ABSOLUTE generation thresholds,
# and they exist so a collapsed generation cannot pass by standing next to a
# collapsed baseline.
CATASTROPHIC_BLANK_SHARE = 0.90   # above this the generation has collapsed
CATASTROPHIC_EOS_TOKENS = 10      # EOS before this emitted token = collapsed
GENERATE_TOKENS = 200
CONTINUATION_TOKENS = 1024   # scored continuation after the full-length prompt


# --------------------------------------------------------------------------
# Natural-text prompt + held-out continuation (d)
# --------------------------------------------------------------------------

def _norm_tok_name(name) -> str:
    """Normalize a tokenizer name for comparison.

    `meta-llama/Llama-3.1-8B` and `meta-llama/Llama-3.1-8B/` are the same
    tokenizer, and a comparison that says otherwise would re-tokenize a window
    that did not need it -- changing a number that was already valid.
    """
    return str(name or "").strip().rstrip("/").lower()


def _tokenize_window_for(cand_tok, pool_tok, arr, off, span):
    """`span` ids in `cand_tok` for the text `arr[off:off+L]` covers.

    The pool is tokenized once, by one model's tokenizer. A token id is only
    meaningful relative to the embedding that looks it up, so a candidate whose
    tokenizer is not the pool's must be shown the text in its OWN tokenization.
    This is not a numerical nicety: 8.7% of the ids in the staged pool exceed
    Llama-2-7B's 32000-row embedding, and the first one that appears aborts the
    whole CUDA context in `indexSelectLargeIndex` (measured 2026-10-08).

    The source length is found by BISECTION, not by correcting one estimate.
    `f(L) = candidate tokens for arr[off:off+L]` is nondecreasing but only
    piecewise constant, and its slope is ~1.13 for Llama-2 against a Llama-3.1
    pool: the correction `L += span - f(L)` has a contraction factor of -0.13,
    so it oscillates around the fixed point without ever landing on it (measured:
    32767 -> 28403 -> 29001 -> 28928 -> ... and no exact hit in eight probes).
    Bisection over a monotone f has no such failure mode.

    Falls back, if `span` is not attainable at this offset, to nudging the offset
    by up to 8 tokens and retrying. `span` can be skipped when one pool id
    decodes to text worth two candidate tokens. The offset the seed chose moves
    by at most 8 out of hundreds of thousands, and adjusting the source span is
    the same rule `run_experiment._exact_token_window` already uses.

    Returns None if no exact-length window exists; the caller then keeps the pool
    ids and `assert_prompt_in_vocab` records the mismatch instead of the forward
    aborting.
    """
    if pool_tok is None:
        return None

    def _candidate_ids(local_off, L):
        text = pool_tok.decode(arr[local_off:local_off + L].astype(int).tolist())
        return cand_tok.encode(text, add_special_tokens=False)

    for delta in range(9):
        o = off + delta
        lo, hi = 1, min(2 * span, len(arr) - o)
        if hi < 1:
            return None
        while lo <= hi:
            mid = (lo + hi) // 2
            got = _candidate_ids(o, mid)
            if len(got) == span:
                return got
            if len(got) < span:
                lo = mid + 1
            else:
                hi = mid - 1
    return None


def load_pg19_window(meta_path: str, context_length: int, seed: int,
                     bos_id: int | None = None, tokenizer=None):
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
    span = prompt_len + CONTINUATION_TOKENS
    ids = arr[off:off + span].astype(int).tolist()
    # The pool carries ONE model's tokenization. When the candidate's tokenizer
    # is not that one, re-tokenize the text at the same document and offset with
    # the candidate's tokenizer, so the target is shown ids its own embedding
    # can look up. The seed still selects the same document and the same offset,
    # so the sample stays paired; the source span adjusts (a smaller vocabulary
    # needs more tokens for the same text) so the token budget is still exactly
    # right -- the same rule the runs use.
    #
    # When the two tokenizers agree -- every Llama-3.1 candidate against this
    # pool -- `ids` is byte-for-byte what it was before this branch existed, so
    # numbers already recorded for those controls stay comparable.
    pool_tok = None
    pool_name = meta.get("tokenizer")
    if (tokenizer is not None and pool_name
            and _norm_tok_name(pool_name)
            != _norm_tok_name(getattr(tokenizer, "name_or_path", ""))):
        pool_tok = AutoTokenizer.from_pretrained(pool_name)
        retok = _tokenize_window_for(tokenizer, pool_tok, arr, off, span)
        if retok is not None:
            ids = retok
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

def assert_prompt_in_vocab(ids: list[int], vocab_size: int) -> dict:
    """Ids the candidate's embedding cannot look up, as a REPORTED outcome.

    An out-of-range index is not a numerical nuisance to be worked around. On
    CUDA, `indexSelectLargeIndex` asserts and the whole context dies, taking
    every later candidate with it and leaving the stage with no output at all
    (measured 2026-10-08: rows >= Llama-2-7B's 32000-row embedding, from a pool
    tokenized by Llama-3.1). It also means the candidate was being shown a
    DIFFERENT model's tokenization, which is a finding about the input pool
    rather than about the rope under test.

    Returned, never raised: the gate's contract is to record what it measured.
    """
    bad = [i for i, t in enumerate(ids) if t < 0 or t >= int(vocab_size)]
    return {
        "prompt_ids_out_of_vocab": len(bad),
        "prompt_first_out_of_vocab_index": bad[0] if bad else "",
        "prompt_first_out_of_vocab_id": int(ids[bad[0]]) if bad else "",
        "prompt_max_id": int(max(ids)) if ids else "",
        "prompt_vocab_size": int(vocab_size),
    }


def _cuda_died(row: dict) -> bool:
    """Did this row's CUDA CONTEXT die, as opposed to the row merely failing?

    The first version matched the bare substring "CUDA", which is wrong in both
    directions: it reads a per-row `CUDA OOM` (a resource limit that the row
    records and the stage can continue past) as a dead context, and it reads any
    message that merely MENTIONS CUDA the same way -- including the guard's own
    note that a forward would abort on the GPU. Matching the specific abort
    signatures keeps "this configuration measured badly" and "the context is
    gone, so no later row means anything" told apart.
    """
    if row.get("cuda_poisoned"):
        return True
    blob = f"{row.get('error', '')} {row.get('cleanup_error', '')}"
    return any(m in blob for m in (
        "device-side assert", "CUDA error", "an illegal memory access",
        "unspecified launch failure", "cudaErrorAssert",
    ))


def _first_non_finite_logit_position(model, pred, chunk: int = 8) -> int | None:
    """First scored position whose logits are not finite, or None.

    The hidden state can stay finite while the head overflows, and it matters
    which: `torch.multinomial` on a non-finite softmax aborts the CUDA context
    in the SAMPLING arms, an abort that looks exactly like a hardware fault.
    The head is applied in small slices so this stays cheap on a 128k window.
    """
    if pred is None or pred.shape[1] == 0:
        return None
    for start in range(0, pred.shape[1], chunk):
        logits = model.lm_head(pred[:, start:start + chunk, :])
        finite = torch.isfinite(logits)
        if not bool(finite.all()):
            bad = (~finite).any(dim=-1)[0].nonzero()
            if bad.numel():
                return int(start + int(bad[0]))
    return None


def continuation_perplexity(model, prompt_ids, cont_ids):
    """(perplexity of `cont_ids` given `prompt_ids`, first non-finite position).

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
        total, n, pred = continuation_nll(
            model, ids, score_from=len(prompt_ids), forward=_forward,
        )
        bad_pos = _first_non_finite_logit_position(model, pred)
    return perplexity_from_sums(total, n), bad_pos


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
        # `early_eos` is the DEGENERATE case, not merely an early one: an EOS at
        # token 12 of 200 is a short answer, an EOS at token 5 is a dead model.
        "early_eos": early is not None and early < CATASTROPHIC_EOS_TOKENS,
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


def gate_sample(meta_path: str, ctx: int, seed: int, tok):
    """The (prompt_ids, continuation_ids) the gate scores.

    ONE function, so every path that needs a window goes through the same call
    and passes the BOS. The engine prepends one at encode time; a gate that
    measured the prompt without it would score a sequence one token shorter than
    the runs it is calibrating against -- and, because the text budget is fixed,
    it would end 1024 tokens further into the book, i.e. a different sample as
    well as a different length.
    """
    return load_pg19_window(meta_path, ctx, seed,
                            bos_id=getattr(tok, "bos_token_id", None),
                            tokenizer=tok)


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
           "expect": cand.get("expect", ""),
           # WHAT KIND of reference this row is. `native_baseline` is the
           # mechanical flag the ratio machinery looks for; it is NOT a claim
           # that the row is in-distribution. Llama-2 at 32k with its shipped
           # rope is the unscaled model at 4x its training window, and calling
           # that "native" invites reading the ratio as quality against an
           # in-distribution model, which it is not.
           "reference_role": cand.get("reference_role", "")}

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

        prompt_ids, cont_ids = gate_sample(
            meta_path, ctx, int(cand.get("seed", 42)), tok)
        row["prompt_tokens"] = len(prompt_ids)
        import hashlib

        def _sample_sha(ids):
            """The same spelling run_experiment uses for its prompt and
            continuation hashes, so the gate and the runs agree about what "the
            same sample" means.

            Both halves are hashed and recorded. Two rows can share a prompt and
            score different continuations, and a ratio between them would then be
            a difference between two samples rather than a measurement of the
            rope's anchoring -- which is the only thing this file is allowed to
            report.
            """
            return hashlib.sha256(
                ",".join(str(int(i)) for i in ids).encode()).hexdigest()

        row["prompt_sha256"] = _sample_sha(prompt_ids)
        row["continuation_sha256"] = _sample_sha(cont_ids)
        row["seed"] = int(cand.get("seed", 42))

        # WHICH TOKENIZER EACH SIDE USES, and whether the prompt is even in the
        # candidate's embedding. Checked BEFORE any forward: an out-of-range id
        # asserts in indexSelectLargeIndex and kills the CUDA context, which
        # takes every remaining candidate with it and leaves the stage with no
        # output at all (measured 2026-10-08). Recorded as the candidate's
        # outcome instead, so the CSV says what happened rather than the stage
        # dying with nothing written.
        try:
            pool_tok_name = (json.loads(Path(meta_path).read_text())
                             or {}).get("tokenizer") or ""
        except Exception:                                   # noqa: BLE001
            pool_tok_name = ""
        row["pool_tokenizer"] = pool_tok_name
        row["candidate_tokenizer"] = getattr(tok, "name_or_path", "")
        vocab = int(getattr(model.config, "vocab_size", 0) or 0)
        if vocab:
            row.update(assert_prompt_in_vocab(prompt_ids, vocab))
            n_bad = row["prompt_ids_out_of_vocab"]
            if n_bad:
                row["status"] = "invalid_prompt_tokens"
                row["error"] = (
                    f"{n_bad} of {len(prompt_ids)} prompt ids are outside "
                    f"{model_name}'s embedding (vocab_size={vocab}, max id "
                    f"{row['prompt_max_id']}); the pool is tokenized by "
                    f"{pool_tok_name or 'an unknown tokenizer'} and the "
                    f"candidate by {row['candidate_tokenizer']}. The forward is "
                    f"not attempted: it would abort the device context in "
                    f"indexSelectLargeIndex."
                )
                # Nothing was measured, so no perplexity exists to report. An
                # empty field says "not measured"; a 0.0 would read as a number.
                row["ppl_continuation"] = ""
                return row

        ppl, bad_pos = continuation_perplexity(model, prompt_ids, cont_ids)
        row["ppl_continuation"] = round(ppl, 4)
        # A non-finite logit is a property of the configuration under test, not
        # a gate malfunction: the f2 construction is a KNOWN-BROKEN negative
        # control, and recording it is the finding. The position is kept so a
        # degeneration can be told apart from an arithmetic accident.
        row["non_finite_logits"] = bad_pos is not None
        row["non_finite_first_position"] = (
            bad_pos if bad_pos is not None else "")

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
        # CLEANUP MUST NOT BE ABLE TO DESTROY THE STAGE'S OUTPUT.
        #
        # A CUDA device-side assert is reported at the next synchronising call,
        # and empty_cache() is one. Raising from `finally` BYPASSES the handlers
        # above -- so on 2026-10-08 one candidate poisoned the context, the
        # assert surfaced here, and the exception escaped run_candidate, killed
        # the list comprehension in main(), and left the gate with NO CSV AT ALL.
        # That is the worst possible outcome for a stage whose entire purpose is
        # to run a candidate that is known to be broken.
        try:
            torch.cuda.empty_cache()
        except Exception as e:                      # noqa: BLE001
            row.setdefault("status", "error")
            row["cleanup_error"] = f"{type(e).__name__}: {str(e)[:160]}"
            row["cuda_poisoned"] = True
    return row


# A reference row that is only DESCRIPTIVE is not a denominator: it answers
# "what does this model look like where it is valid", and it is reported beside
# the ratios rather than used to divide anything.
DESCRIPTIVE_ROLES = ("in_distribution_reference",)


def pairing_verdict(cand: dict, base: dict) -> tuple[str, str]:
    """Is this candidate scored on the SAME SAMPLE as its reference?

    Returns `(pairing, reason)` where pairing is one of:

      paired         prompt and continuation hashes agree: the ratio is a paired
                     contrast on ONE sample
      unpaired       the hashes disagree: NO ratio is computed, and the row is
                     reported as an error
      cross_context  an extension judged against the declared native baseline at
                     a DIFFERENT context, so the windows differ by construction:
                     the plan's coherence comparison, labelled as such
      descriptive    an in-distribution measurement, reported beside the ratios
                     and never used as their denominator

    A ratio between two different samples is not a measurement of the rope's
    anchoring: it is a difference between two samples, and the document, offset,
    prompt and continuation all move with the seed.
    """
    role = str(cand.get("reference_role") or "")
    if role in DESCRIPTIVE_ROLES:
        return "descriptive", ("in-distribution reference: reported beside the "
                               "ratios, not a denominator")
    if base is None:
        return "unpaired", "no declared reference at this context"
    # An EXTENSION is judged against the native configuration at the native
    # window, so the two windows are different lengths and the samples cannot be
    # the same. That comparison is what the plan declares for >128k -- "does
    # extending the window keep the target coherent, where coherent is defined by
    # the native configuration" -- so it is labelled rather than refused.
    want_ctx = cand.get("reference_context")
    if want_ctx not in (None, ""):
        if int(want_ctx) != int(cand.get("context_length", -1)):
            return "cross_context", (
                f"extension at {cand.get('context_length')} judged against the "
                f"declared native baseline at {want_ctx}: a cross-window "
                f"coherence ratio, not a same-sample contrast")
    for field in ("prompt_sha256", "continuation_sha256"):
        a, b = str(cand.get(field) or ""), str(base.get(field) or "")
        if not a or not b or a != b:
            return "unpaired", (
                f"{field} differs between the candidate and its reference "
                f"({a[:12] or '<none>'} vs {b[:12] or '<none>'}): they are not "
                f"the same sample, so no ratio is computed")
    if int(cand.get("seed", -1)) != int(base.get("seed", -2)):
        return "unpaired", (f"seeds differ ({cand.get('seed')} vs "
                            f"{base.get('seed')})")
    return "paired", "prompt and continuation hashes agree"


def _share_reason(label: str, field: str, row: dict, baseline) -> tuple:
    """The relative half of the generation rule, for one share field.

    Returns (reason_or_None, baseline_share, ratio). Both the reason and the
    numbers are returned because the CSV has to carry the ratio that was
    applied: a reader must be able to see 0.12 vs a 0.05 baseline was rejected
    by a factor of 2.4, not by a threshold nobody can recompute.

    A baseline share of 0 leaves the ratio undefined, so the relative test does
    not apply and only the absolute ceiling judges this statistic. That is a
    real hole and it is the documented one: the alternative is dividing by zero,
    or inventing an absolute threshold for a quantity whose absolute value is
    the thing this revision established is not a property of the model.
    """
    cand = row.get(field)
    base = (baseline or {}).get(field)
    if not isinstance(cand, (int, float)) or not isinstance(base, (int, float)):
        return None, "", ""
    if base <= 0:
        return None, round(base, 4), ""
    ratio = cand / base
    if ratio > GEN_SHARE_TOLERANCE:
        return (f"{label} {cand:.0%} = {ratio:.2f}x the baseline's {base:.0%} "
                f"> {GEN_SHARE_TOLERANCE}x"), round(base, 4), round(ratio, 4)
    return None, round(base, 4), round(ratio, 4)


def verdict(row: dict, baseline: dict) -> dict:
    """Apply the gate's pass rule, or the reason it could not be applied.

    `baseline` is the PAIRED BASELINE ROW — the same sample measured on the
    reference configuration — and not merely its perplexity, because the
    generation shares are judged against the baseline's OWN shares (2026-10-08
    revision). Passing just a number is what made the old absolute
    MAX_BLANK_SHARE possible.
    """
    native_ppl = (baseline or {}).get("ppl_continuation") or 0.0
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
    if row.get("gen_blank_share", 0) > CATASTROPHIC_BLANK_SHARE:
        reasons.append(f"blank lines {row['gen_blank_share']:.0%} = degenerate "
                       f"(> {CATASTROPHIC_BLANK_SHARE:.0%} absolute)")
    if row.get("early_eos"):
        reasons.append(f"EOS at token {row.get('eos_at')} = degenerate "
                       f"(< {CATASTROPHIC_EOS_TOKENS})")
    for label, field, bkey, rkey in (
            ("blank lines", "gen_blank_share", "baseline_blank_share",
             "blank_share_ratio"),
            ("repeated n-grams", "gen_repeat_share", "baseline_repeat_share",
             "repeat_share_ratio")):
        why, base_share, share_ratio = _share_reason(label, field, row, baseline)
        out[bkey] = base_share
        out[rkey] = share_ratio
        if why:
            reasons.append(why)
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
          # What this candidate is (positive/negative control, baseline,
          # candidate) and, for a reference row, WHICH KIND of reference it is.
          "role", "expect", "reference_role",
          # The role of the baseline this row was actually judged against, so
          # the CSV says "this ratio is against an unscaled-OOD reference"
          # rather than leaving a reader to infer it from the row name.
          "baseline_role",
          "context_length", "reference_context", "seed",
          "rope_type", "rope_factor", "rope_anchor_base",
          "config_max_position_embeddings", "config_rope_scaling",
          "effective_rope_match", "effective_rope_maxerr",
          "effective_rope_matches_intent", "native_window",
          "inv_freq_first", "inv_freq_last", "slowest_channel_stretch",
          "prompt_tokens", "prompt_sha256", "continuation_sha256",
          "ppl_continuation", "native_ppl_reference", "baseline_context",
          # Whether the ratio is on the SAME SAMPLE as its reference, and the
          # reference's role. A ratio across two samples is not a measurement.
          "pairing", "baseline_prompt_sha256", "baseline_continuation_sha256",
          "ppl_ratio", "early_eos", "eos_at", "gen_chars", "gen_blank_share",
          "gen_alpha_share", "gen_repeat_share",
          # The baseline's own shares and the ratios that were actually applied,
          # so the share criterion can be recomputed from the CSV instead of
          # being taken on trust (2026-10-08 revision).
          "baseline_blank_share", "baseline_repeat_share",
          "blank_share_ratio", "repeat_share_ratio",
          # Which tokenizer the INPUT POOL was built with, and which one the
          # candidate uses. A mismatch is the whole cause of the 2026-10-08
          # abort, and it has to be legible from the CSV alone.
          "pool_tokenizer", "candidate_tokenizer",
          # Ids the candidate's embedding cannot look up. Recorded, never
          # raised: see assert_prompt_in_vocab.
          "prompt_ids_out_of_vocab", "prompt_first_out_of_vocab_index",
          "prompt_first_out_of_vocab_id", "prompt_max_id", "prompt_vocab_size",
          # Non-finite logits, with the first scored position that had them.
          # Reported rather than crashed on, because a NaN logit is a property
          # of the configuration under test (the ARM4 f2 construction is a
          # KNOWN-BROKEN negative control), not a gate malfunction.
          "non_finite_logits", "non_finite_first_position",
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

    # A DEAD GPU IS NOT A MISCALIBRATED GATE.
    #
    # If the CUDA context died part-way through, every row after that point fails
    # for a reason that has nothing to do with the controls, and the verdict
    # logic below would read those failures as "the positives failed, so the gate
    # is wrong" -- the exact opposite of what happened. Say so instead, and still
    # write the rows: the evidence is worth more than the verdict.
    cuda_dead = [r for r in rows if _cuda_died(r)]
    if cuda_dead:
        print(f"CUDA CONTEXT DIED during {len(cuda_dead)} of {len(rows)} rows; "
              f"first at '{cuda_dead[0].get('candidate', cuda_dead[0].get('name', '?'))}'.")
        print("This is NOT a calibration result: the gate never got to judge the "
              "controls that follow it.")
        for r in cuda_dead[:5]:
            print(f"  {r.get('candidate', r.get('name', '?'))}: "
                  f"{r.get('error') or r.get('cleanup_error')}")

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
        r["baseline_role"] = ((b.get("reference_role") or b.get("role") or "")
                              if b else "")
        # Which reference this candidate declared. A 256k candidate judged
        # against a 128k baseline and one judged against a 256k baseline are
        # different measurements, and the CSV has to say which happened.
        r["reference_context"] = (r.get("reference_context")
                                  or r.get("context_length"))
        r["baseline_prompt_sha256"] = (b.get("prompt_sha256") or "") if b else ""
        r["baseline_continuation_sha256"] = (
            (b.get("continuation_sha256") or "") if b else "")

        # PAIRING FIRST. A ratio is only a measurement of the rope's anchoring
        # if the candidate and the reference scored the SAME sample; otherwise
        # the first thing a "damage" number contains is the difference between
        # two documents, which is far larger than the effect under test.
        pairing, reason = pairing_verdict(r, b)
        r["pairing"] = pairing
        if pairing == "descriptive":
            # An in-distribution measurement, reported beside the ratios and
            # never used to divide anything.
            r["ppl_ratio"] = ""
            r["native_ppl_reference"] = ""
            r["baseline_context"] = ""
            r["gate_pass"] = ""
            r["gate_reason"] = reason
            continue
        if pairing == "unpaired":
            r["ppl_ratio"] = ""
            r["gate_pass"] = False
            r["gate_reason"] = f"UNPAIRED: {reason}"
            r["status"] = "unpaired_reference"
            r["error"] = reason
            continue
        if b is None:
            r["gate_pass"] = False
            r["gate_reason"] = (
                f"no declared reference at reference context "
                f"{r.get('reference_context') or r.get('context_length')} for "
                f"{r.get('target_model_name')}; cannot compute a ratio against "
                f"one")
            r["status"] = r.get("status") or "no_baseline"
        else:
            r.update(verdict(r, b))

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
    if cuda_dead:
        # Non-zero, but distinct from a calibration failure: the watcher's
        # fail-fast treats any non-zero as an incident and pulls the logs, and
        # the CSV above is what makes the incident diagnosable.
        print(f"\n  CUDA CONTEXT DIED ({len(cuda_dead)} rows); the rows were "
              f"written for diagnosis, but the gate did not run to completion.")
        return 8
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

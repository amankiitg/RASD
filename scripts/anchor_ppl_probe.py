#!/usr/bin/env python3
"""Does correct YaRN anchoring remove the degeneration?

Measures Llama-2-7B on a held-out PG-19 continuation under:

  native_4k        no rope scaling at all, 4096-token window (reference)
  yarn_f8_anchor_ctx   YaRN factor 8, config.max_position_embeddings = 32768
                       (== the project's current behaviour: the anchor travels
                        on config.max_position_embeddings because transformers
                        4.47.1's YaRN ignores the dict key)
  yarn_f8_anchor_base  YaRN factor 8, config.max_position_embeddings = 4096
                       (the true pretraining base)

Everything is measured twice: at the full 32768-token context, and at 4096 so
the two YaRN variants can be compared to native at IDENTICAL context length.
That 4096 row is the controlled comparison — at 32k the context length itself
differs from the native baseline and confounds the read.

Also reads 200 greedy tokens per configuration so the output can be judged
directly, not just through a perplexity scalar.

Writes results/anchor_ppl/{anchor_ppl.csv,gen_<name>.txt} and prints a table.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

MODEL = "meta-llama/Llama-2-7b-hf"
PG19_META = "data/processed/pg19/pg19_validation_metadata.json"
BASE_ANCHOR = 4096          # Llama-2-7B's true pretraining window
FACTOR = 8                  # the LCFM paper's 32k setting (4096 * 8)

CONFIGS = [
    # name,                 rope_scaling,                        ctx,   anchor
    ("native_4k",           None,                                4096,  None),
    ("yarn_f8_anchor_ctx",  {"type": "yarn", "factor": FACTOR},  32768, 32768),
    ("yarn_f8_anchor_base", {"type": "yarn", "factor": FACTOR},  32768, BASE_ANCHOR),
]


def load_pg19_tokens(n: int) -> list[int]:
    """First n tokens of the largest held-out PG-19 chunk."""
    meta = json.loads(Path(PG19_META).read_text())
    chunks = sorted(meta["chunks"], key=lambda c: -c["length"])
    c = chunks[0]
    assert c["length"] >= n, f"need {n} tokens, chunk has {c['length']}"
    arr = np.memmap(c["file"], dtype="int32", mode="r")
    return arr[:n].astype(int).tolist()


def perplexity(model, ids: "torch.Tensor") -> float:
    """Exact token-mean NLL over one window, chunked over the sequence.

    Cannot use `labels=` here: transformers' ForCausalLMLoss upcasts the full
    (1, 32768, 32000) logit tensor to fp32, which is 4.2 GiB on top of a
    30 GiB bf16 forward and OOMs a 40 GiB A100. Applying lm_head in position
    chunks produces the identical shifted cross-entropy, just without
    materialising the whole logit tensor at once.
    """
    with torch.no_grad():
        hidden = model.model(input_ids=ids).last_hidden_state   # (1, L, H) bf16
        total_nll = torch.zeros((), dtype=torch.float64, device=ids.device)
        n_scored = 0
        chunk = 2048
        for start in range(0, hidden.shape[1] - 1, chunk):
            stop = min(start + chunk, hidden.shape[1] - 1)
            logits = model.lm_head(hidden[:, start:stop, :]).float()
            tgt = ids[:, start + 1:stop + 1]
            nll = torch.nn.functional.cross_entropy(
                logits.reshape(-1, logits.shape[-1]),
                tgt.reshape(-1),
                reduction="sum",
            )
            total_nll += nll.double()
            n_scored += int(tgt.numel())
            del logits, nll
        return float(torch.exp(total_nll / max(1, n_scored)).item())


@torch.no_grad()
def generate(model, tok, prompt_ids, n_new: int = 200) -> str:
    ids = torch.tensor([prompt_ids], dtype=torch.long, device=model.device)
    out = model.generate(
        ids, max_new_tokens=n_new, do_sample=False,
        pad_token_id=tok.pad_token_id, eos_token_id=tok.eos_token_id,
    )
    return tok.decode(out[0][len(prompt_ids):], skip_special_tokens=True)


def run_one(name, rope_scaling, ctx, anchor, tokens, tok, out_dir: Path) -> dict:
    """Build the config exactly as the engine would, load, measure, generate."""
    cfg = AutoConfig.from_pretrained(MODEL)
    native_max = cfg.max_position_embeddings            # 4096 for Llama-2
    if rope_scaling is None:
        assert ctx <= native_max
        applied, eff_anchor = None, None
    else:
        rs = dict(rope_scaling)
        # Mirror src/models/rasd_inference._build_rope_scaling_dict: the anchor
        # goes in the dict (inert for YaRN on 4.47) AND, when the caller asks
        # for a specific base, on config.max_position_embeddings (the channel
        # YaRN actually reads).
        rs["original_max_position_embeddings"] = BASE_ANCHOR
        cfg.rope_scaling = rs
        cfg.max_position_embeddings = anchor
        applied, eff_anchor = rs, anchor

    model = AutoModelForCausalLM.from_pretrained(
        MODEL, config=cfg, torch_dtype=torch.bfloat16, device_map={"": 0},
    ).eval()

    # Report the inv_freq the model actually built, so the anchor claim is
    # verified rather than assumed.
    inv = None
    try:
        rope = model.model.layers[0].self_attn.rotary_emb
        inv = rope.inv_freq.detach().float().cpu()
    except Exception as e:                                  # pragma: no cover
        print(f"    (could not read inv_freq: {e})")

    row = {"config": name, "context": ctx, "rope_scaling": json.dumps(applied),
           "effective_anchor": eff_anchor}

    ids = torch.tensor([tokens[:ctx]], dtype=torch.long, device=model.device)
    t0 = time.time()
    row["ppl"] = perplexity(model, ids)
    row["ppl_seconds"] = round(time.time() - t0, 1)

    # Controlled comparison: the same 4096-token window under every config.
    ids4 = torch.tensor([tokens[:4096]], dtype=torch.long, device=model.device)
    row["ppl_at_4096"] = perplexity(model, ids4)

    if inv is not None:
        row["inv_freq_last"] = f"{float(inv[-1]):.6g}"
        row["inv_freq_0"] = f"{float(inv[0]):.6g}"

    # 200 greedy tokens from a 512-token prompt.
    gen = generate(model, tok, tokens[:512], 200)
    (out_dir / f"gen_{name}.txt").write_text(gen)
    row["gen_chars"] = len(gen)

    del model
    torch.cuda.empty_cache()
    return row


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/anchor_ppl")
    args = ap.parse_args()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    tok = AutoTokenizer.from_pretrained(MODEL)
    tokens = load_pg19_tokens(32768)
    print(f"PG-19 held-out tokens: {len(tokens)}  first 8 ids: {tokens[:8]}")

    rows = []
    for name, rs, ctx, anchor in CONFIGS:
        print(f"\n=== {name}  ctx={ctx}  anchor={anchor} ===")
        try:
            r = run_one(name, rs, ctx, anchor, tokens, tok, out_dir)
            r["status"] = "ok"
            print(f"    PPL@ctx={r['ppl']:.4f}   PPL@4096={r['ppl_at_4096']:.4f}   "
                  f"inv_freq[0]={r.get('inv_freq_0')} inv_freq[-1]={r.get('inv_freq_last')}")
        except Exception as e:
            import traceback
            r = {"config": name, "context": ctx, "status": "error",
                 "error": f"{type(e).__name__}: {e}"}
            print(traceback.format_exc())
        rows.append(r)

    fields = ["config", "context", "effective_anchor", "ppl", "ppl_at_4096",
              "inv_freq_0", "inv_freq_last", "gen_chars", "status", "error",
              "rope_scaling", "ppl_seconds"]
    import csv
    with (out_dir / "anchor_ppl.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fields})

    print("\n=== SUMMARY ===")
    print(f"  {'config':<22} {'ctx':>6} {'anchor':>7} {'PPL':>10} {'PPL@4096':>10}")
    for r in rows:
        ppl = f"{r['ppl']:.4f}" if r.get("ppl") else f"[{r.get('status')}]"
        p4 = f"{r['ppl_at_4096']:.4f}" if r.get("ppl_at_4096") else "-"
        print(f"  {r['config']:<22} {str(r['context']):>6} "
              f"{str(r.get('effective_anchor')):>7} {ppl:>10} {p4:>10}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

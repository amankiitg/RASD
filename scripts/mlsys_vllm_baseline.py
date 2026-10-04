#!/usr/bin/env python3
"""MLSys Phase 5 — production-stack (vLLM) long-context decode baseline.

Roadmap T2.1: the paper's speedup numbers had no production-stack
reference point. This measures vLLM decoding at the same context length
on the same hardware so the RASD overhead can be read against the
strongest available serving stack.

UNIT MATCHING — read before comparing anything
----------------------------------------------
RASD's `throughput_tps` (rasd_inference.generate) is:

    throughput_tps = tokens_generated / elapsed

where `elapsed` is wall-clock from *before prefill* to the end of the
last decode round, and `tokens_generated` counts OUTPUT tokens only.
It is therefore END-TO-END GENERATION throughput, prefill included.

vLLM reports several different numbers and only some are comparable:

    * end-to-end generation tok/s = output_tokens / total_wall
      (prefill + decode)          <-- THE COMPARABLE ONE
    * decode-only tok/s = output_tokens / decode_wall
      (excludes TTFT)             <-- NOT comparable, larger
    * per-forward tok/s           <-- NOT comparable at all

This script always measures and writes the end-to-end figure, and records
both so the gap is visible. If `--unit-check` cannot confirm that the
measured quantity includes prefill it prints a loud warning and marks the
row `unit_matched=no`, so a non-comparable row can never silently enter
the comparison table.

ENVIRONMENT
-----------
vLLM pins its own transformers and conflicts with the RASD pod pins
(transformers>=4.46, accelerate==0.33.0). Install it in a DEDICATED venv:

    python -m venv ~/venv-vllm && source ~/venv-vllm/bin/activate
    pip install vllm

EXIT / STATUS VALUES
--------------------
Every attempted configuration produces a CSV row, including the failures:
`status` is one of ok | oom | unsupported | error. An OOM at 128k on a
single rank is a *result* (it is exactly why RASD exists), so it is
recorded rather than raised.

Usage:
    python scripts/mlsys_vllm_baseline.py \
        --models meta-llama/Llama-2-7b-hf meta-llama/Llama-3.1-8B \
        --context-lengths 131072 \
        --max-new-tokens 64 \
        --out results/mlsys/vllm_baseline.csv
"""
from __future__ import annotations

import argparse
import csv
import gc
import json
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent

CSV_FIELDS = [
    "model", "context_length", "max_new_tokens", "tensor_parallel_size",
    "rope_scaling", "prompt_tokens", "output_tokens",
    "end_to_end_wall_s", "decode_only_wall_s", "ttft_s",
    # the unit-matched metric, named to make the comparison explicit
    "throughput_tps_end_to_end", "throughput_tps_decode_only",
    "unit_matched", "peak_mem_mb", "status", "error",
]

# Llama-2 needs YaRN to reach 128k, exactly as the RASD arm-1 runs do.
# Keyed by the target's native window so the same scaling RASD uses is
# applied here; without it vLLM would silently extrapolate or refuse.
DEFAULT_ROPE = {
    "meta-llama/Llama-2-7b-hf": {"type": "yarn", "factor": 32.0,
                                 "original_max_position_embeddings": 4096},
}


def build_prompt(tokenizer, n_tokens: int) -> str:
    """A prompt of ~n_tokens tokens.

    Uses the same repeated technical-English paragraph RASD's synthetic
    prompt uses, so the content distribution is as close as vLLM allows
    without importing the RASD stack (which would defeat the point of a
    separate-environment baseline).
    """
    paragraph = (
        "The following is a technical description of a distributed "
        "inference system designed for long-context language model "
        "decoding across multiple accelerators. "
    )
    unit = tokenizer.encode(paragraph, add_special_tokens=False)
    if not unit:
        unit = [0]
    reps = max(1, n_tokens // len(unit))
    ids = (unit * reps)[:n_tokens]
    return tokenizer.decode(ids)


def run_one(model: str, ctx: int, max_new: int, tp: int, out_rows: list[dict],
            gpu_mem_util: float) -> None:
    """Attempt one (model, ctx) cell; append exactly one row."""
    row = {f: "" for f in CSV_FIELDS}
    row.update({"model": model, "context_length": ctx,
                "max_new_tokens": max_new, "tensor_parallel_size": tp,
                "peak_mem_mb": ""})

    rope = DEFAULT_ROPE.get(model)
    row["rope_scaling"] = json.dumps(rope) if rope else ""

    try:
        import torch
        from transformers import AutoTokenizer
        from vllm import LLM, SamplingParams
    except Exception as e:  # noqa: BLE001
        row.update({"status": "error", "unit_matched": "no",
                    "error": f"vLLM import failed ({e}); "
                             f"did you activate the dedicated venv?"})
        out_rows.append(row)
        return

    try:
        tok_kwargs = {}
        if rope:
            tok_kwargs["rope_scaling"] = rope
        llm_kwargs = dict(
            model=model,
            tensor_parallel_size=tp,
            gpu_memory_utilization=gpu_mem_util,
            max_model_len=ctx + max_new,
            trust_remote_code=True,
            enforce_eager=False,
        )
        if rope:
            llm_kwargs["rope_scaling"] = rope
        llm = LLM(**llm_kwargs)
    except torch.cuda.OutOfMemoryError as e:  # type: ignore[attr-defined]
        row.update({"status": "oom", "unit_matched": "no",
                    "error": f"OOM constructing engine: {str(e)[:300]}"})
        out_rows.append(row)
        return
    except Exception as e:  # noqa: BLE001
        msg = str(e)
        status = "unsupported" if any(
            k in msg.lower() for k in ("max_model_len", "longer than",
                                       "rope", "not supported", "exceed")
        ) else "error"
        row.update({"status": status, "unit_matched": "no",
                    "error": msg[:300]})
        out_rows.append(row)
        return

    try:
        tokenizer = AutoTokenizer.from_pretrained(model)
        prompt = build_prompt(tokenizer, ctx)
        row["prompt_tokens"] = len(
            tokenizer(prompt, add_special_tokens=False)["input_ids"])
        params = SamplingParams(temperature=1.0, top_p=1.0,
                                max_tokens=max_new, ignore_eos=True)

        # Warm-up is deliberately NOT done: a cold first call is what the
        # RASD number also includes (RASD measures from before its own
        # prefill on a freshly constructed engine).
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        t0 = time.perf_counter()
        outs = llm.generate([prompt], params, use_tqdm=False)
        t1 = time.perf_counter()

        end_to_end = t1 - t0
        out_tokens = len(outs[0].outputs[0].token_ids)
        # vLLM exposes TTFT through metrics only in async/server mode, so
        # decode-only time is derived from the reported prefill if
        # available, else left blank rather than silently approximated.
        ttft = _ttft_from_metrics(outs[0])
        decode_only = (end_to_end - ttft) if ttft is not None else float("nan")

        unit_ok = ttft is not None and end_to_end > 0
        row.update({
            "output_tokens":              out_tokens,
            "end_to_end_wall_s":          f"{end_to_end:.4f}",
            "decode_only_wall_s":         (f"{decode_only:.4f}"
                                           if decode_only == decode_only else ""),
            "ttft_s":                     f"{ttft:.4f}" if ttft is not None else "",
            "throughput_tps_end_to_end":  (f"{out_tokens / end_to_end:.4f}"
                                           if end_to_end > 0 else ""),
            "throughput_tps_decode_only": (f"{out_tokens / decode_only:.4f}"
                                           if decode_only and decode_only == decode_only
                                           else ""),
            "unit_matched":               "yes" if unit_ok else "no",
            "peak_mem_mb":                (f"{torch.cuda.max_memory_allocated() / 1024 ** 2:.1f}"
                                           if torch.cuda.is_available() else ""),
            "status":                     "ok",
            "error":                      "",
        })
        if not unit_ok:
            print("[WARN] could not measure TTFT; throughput_tps_end_to_end "
                  "is still output/total_wall (comparable), but "
                  "throughput_tps_decode_only is blank. Marking "
                  "unit_matched=no so the row is not silently compared.")
    except torch.cuda.OutOfMemoryError as e:  # type: ignore[attr-defined]
        row.update({"status": "oom", "unit_matched": "no",
                    "error": f"OOM during generate: {str(e)[:300]}"})
    except Exception as e:  # noqa: BLE001
        row.update({"status": "error", "unit_matched": "no",
                    "error": str(e)[:300]})
    finally:
        try:
            del llm  # noqa: F821
        except Exception:  # noqa: BLE001
            pass
        gc.collect()
        try:
            import torch  # noqa: F811
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:  # noqa: BLE001
            pass

    out_rows.append(row)


def _ttft_from_metrics(request_output) -> float | None:
    """Best-effort TTFT from a vLLM RequestOutput.

    vLLM only populates detailed timings in async/server mode; offline
    `generate()` typically exposes `metrics` as None. Return None rather
    than inventing a value — the caller marks the row accordingly.
    """
    m = getattr(request_output, "metrics", None)
    if m is None:
        return None
    val = getattr(m, "first_token_time", None)
    if val is None:
        return None
    try:
        return float(val)
    except (TypeError, ValueError):
        return None


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models", nargs="+",
                    default=["meta-llama/Llama-2-7b-hf"],
                    help="HF model ids to benchmark.")
    ap.add_argument("--context-lengths", nargs="+", type=int,
                    default=[131072],
                    help="Prompt token counts, matched to the RASD runs.")
    ap.add_argument("--max-new-tokens", type=int, default=64,
                    help="Matches RASD's max_new_tokens=64.")
    ap.add_argument("--tensor-parallel-size", type=int, default=1,
                    help="1 = single-rank. Bump and re-run if 128k OOMs; "
                         "the OOM row itself is a reportable result.")
    ap.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    ap.add_argument("--out", default=str(REPO / "results" / "mlsys"
                                         / "vllm_baseline.csv"))
    args = ap.parse_args()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []
    for model in args.models:
        for ctx in args.context_lengths:
            print(f"\n=== vLLM {model} @ {ctx} tokens (TP="
                  f"{args.tensor_parallel_size}) ===")
            run_one(model, ctx, args.max_new_tokens,
                    args.tensor_parallel_size, rows,
                    args.gpu_memory_utilization)

    # Rewrite the whole file so a partial failure still yields a usable
    # CSV with the failures recorded as rows.
    with out_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CSV_FIELDS, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)

    print(f"\n[write] {out_path}  ({len(rows)} rows)")
    n_ok = sum(1 for r in rows if r.get("status") == "ok")
    n_unmatched = sum(1 for r in rows if r.get("unit_matched") == "no")
    print(f"        {n_ok}/{len(rows)} ok, "
          f"{len(rows) - n_ok} failed/unsupported")
    if n_unmatched:
        print(f"[WARN] {n_unmatched} row(s) have unit_matched=no — do NOT "
              f"place them in a speedup comparison against RASD's "
              f"throughput_tps without handling the unit difference.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

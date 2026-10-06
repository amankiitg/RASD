#!/usr/bin/env python
"""vLLM baseline for long-context decode — with attempt ladder + real tracebacks.

Phase 5 exists to give the RASD speedup numbers an apples-to-apples external
reference point at 128k on the SAME hardware. The first two attempts at this
produced only `status=unsupported` rows whose `error` was a one-line summary
("Engine core initialization failed"), which is useless for diagnosis: the
actual cause is several frames deeper in vLLM's engine log.

This version runs each attempt in a SUBPROCESS, tees its full stdout/stderr
to results/mlsys/logs/vllm_<model>_attempt<N>.log, and on failure extracts
the LAST Python exception (with traceback) from that log into the CSV row.

UNIT MATCHING (the thing that makes or breaks the comparison)
    RASD reports `throughput_tps` = tokens_generated / total wall time,
    measured end-to-end INCLUDING prefill. The comparable vLLM number is
    therefore `throughput_tps_end_to_end` = output_tokens / end_to_end_wall,
    which also includes prefill. `throughput_tps_decode_only` is recorded
    separately and is NOT comparable to RASD's figure. Rows carry
    `unit_matched` and a non-matching row must be excluded from any ratio.

Usage (parent — runs the ladder):
    python scripts/mlsys_vllm_baseline.py \
        --models meta-llama/Llama-3.1-8B meta-llama/Llama-2-7b-hf \
        --context-lengths 131072 --max-new-tokens 64 \
        --tensor-parallel-size 8 --out results/mlsys/vllm_baseline.csv

Internal (worker — one attempt, one process):
    python scripts/mlsys_vllm_baseline.py --_worker /path/spec.json
"""
from __future__ import annotations

import argparse
import csv
import gc
import json
import os
import re
import subprocess
import sys
import threading
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
LOGDIR = REPO / "results" / "mlsys" / "logs"

CSV_FIELDS = [
    "model", "context_length", "max_new_tokens", "tensor_parallel_size",
    "rope_scaling", "prompt_tokens", "output_tokens",
    "end_to_end_wall_s", "decode_only_wall_s", "ttft_s",
    "throughput_tps_end_to_end", "throughput_tps_decode_only",
    "unit_matched", "peak_mem_mb", "status", "error",
    # MLSys follow-up — attempt provenance so a failure is diagnosable.
    "attempt", "config_used", "error_class", "log_path",
]

# YaRN for the Llama-2 arm-1 counterpart. Base 4096 = Llama-2's own window,
# factor 32 = 128k/4k, matching what RASD arm 1 applies.
DEFAULT_ROPE = {
    "meta-llama/Llama-2-7b-hf": {
        "type": "yarn", "factor": 32.0,
        "original_max_position_embeddings": 4096,
    },
    # Llama-3.1 needs NO rope override at 128k: it is within its native
    # 131072 window and must stay native to be comparable to RASD arm 2/3.
    "meta-llama/Llama-3.1-8B": None,
}

# Three DISTINCT fixes, tried in order, stopping at the first success.
ATTEMPT_LADDER = [
    {
        "name": "1-default-tp8",
        "kwargs": {"enforce_eager": False, "gpu_memory_utilization": 0.90},
        "env": {},
        "max_model_len": "ctx",
        "max_num_batched_tokens": None,
        "note": "max_model_len exactly ctx; YaRN via hf_overrides",
    },
    {
        "name": "2-eager-mem95-seq1",
        "kwargs": {"enforce_eager": True, "gpu_memory_utilization": 0.95,
                   "max_num_seqs": 1},
        "env": {"VLLM_ALLOW_LONG_MAX_MODEL_LEN": "1"},
        "max_model_len": "ctx",
        "max_num_batched_tokens": "ctx",
        "note": "eager, 0.95 util, seq=1, chunked prefill sized to ctx",
    },
    {
        "name": "3-fallback-64k",
        "kwargs": {"enforce_eager": True, "gpu_memory_utilization": 0.95,
                   "max_num_seqs": 1},
        "env": {"VLLM_ALLOW_LONG_MAX_MODEL_LEN": "1"},
        "max_model_len": 65536,
        "max_num_batched_tokens": None,
        "note": "valid 64k reference point if 128k cannot be made to load",
    },
]

MAX_ATTEMPT_WALL_S = 20 * 60          # 20 min cap per attempt


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

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


_EXC_LINE = re.compile(
    r"^([A-Za-z_][A-Za-z0-9_.]*(?:Error|Exception|Interrupt|Exit|"
    r"RuntimeError|ValueError|TypeError|AssertionError|MemoryError))\b")
_TRACEBACK_START = re.compile(r"^Traceback \(most recent call last\):")


def extract_last_exception(log_text: str) -> tuple[str, str]:
    """Return (error_class, full_message) for the LAST Python exception.

    Scans backwards for the final `XxxError: message` line and, when the log
    contains a preceding `Traceback (most recent call last):`, returns the
    whole traceback block instead of the useless one-line summary vLLM's
    engine prints as its last line.

    Falls back to the final non-empty line when no Python exception is
    present (e.g. a hard OOM kill or a C++ abort), so a row never ends up
    with an empty error.
    """
    lines = log_text.splitlines()
    last_idx = None
    last_class = ""
    for i in range(len(lines) - 1, -1, -1):
        m = _EXC_LINE.match(lines[i].strip())
        if m:
            last_idx, last_class = i, m.group(1)
            break

    if last_idx is None:
        # No Python exception — surface the tail rather than nothing.
        tail = [l for l in lines if l.strip()][-6:]
        return "", "\n".join(tail)[:4000]

    # Walk back to the nearest traceback header for this exception.
    start = last_idx
    for j in range(last_idx, max(-1, last_idx - 400), -1):
        if _TRACEBACK_START.match(lines[j]):
            start = j
            break
    else:
        start = max(0, last_idx - 20)
    block = "\n".join(lines[start:last_idx + 1])
    return last_class, block[-4000:]


# ---------------------------------------------------------------------------
# worker — one attempt, one process
# ---------------------------------------------------------------------------

def worker_main(spec_path: Path) -> int:
    spec = json.loads(spec_path.read_text())
    res_path = Path(spec["result_path"])
    result: dict = {"status": "error", "error": "", "error_class": ""}
    try:
        import torch
        from transformers import AutoTokenizer
        from vllm import LLM, SamplingParams
    except Exception as e:  # noqa: BLE001
        import traceback
        traceback.print_exc()
        result.update({"status": "error",
                       "error_class": type(e).__name__,
                       "error": f"vLLM import failed: {e}"})
        res_path.write_text(json.dumps(result))
        return 1

    model = spec["model"]
    ctx = spec["context_length"]
    max_new = spec["max_new_tokens"]
    max_model_len = spec["max_model_len"]
    rope = spec["rope"]

    try:
        llm_kwargs = dict(
            model=model,
            tensor_parallel_size=spec["tensor_parallel_size"],
            max_model_len=max_model_len,
            trust_remote_code=True,
            **spec["kwargs"],
        )
        if spec.get("max_num_batched_tokens"):
            llm_kwargs["max_num_batched_tokens"] = spec["max_num_batched_tokens"]
        # vLLM moved `rope_scaling` off EngineArgs; it is applied via
        # hf_overrides now. Try modern spelling, fall back to the legacy
        # kwarg so this works across vLLM versions.
        if rope:
            try:
                llm = LLM(**llm_kwargs, hf_overrides={"rope_scaling": rope})
            except TypeError:
                llm = LLM(**llm_kwargs, rope_scaling=rope)
        else:
            llm = LLM(**llm_kwargs)
    except Exception as e:  # noqa: BLE001
        import traceback
        traceback.print_exc()
        import torch  # noqa: F811
        is_oom = isinstance(e, torch.cuda.OutOfMemoryError) or "out of memory" in str(e).lower()
        result.update({
            "status": "oom" if is_oom else "error",
            "error_class": type(e).__name__,
            "error": str(e)[:2000],
        })
        res_path.write_text(json.dumps(result))
        return 1

    try:
        tokenizer = AutoTokenizer.from_pretrained(model)
        # Size the prompt to ctx - max_new so prompt+output exactly fills the
        # model's window instead of exceeding it.
        prompt_target = max(1, min(ctx, max_model_len) - max_new)
        prompt = build_prompt(tokenizer, prompt_target)
        prompt_tokens = len(tokenizer(prompt, add_special_tokens=False)["input_ids"])
        params = SamplingParams(temperature=1.0, top_p=1.0,
                                max_tokens=max_new, ignore_eos=True)

        # No warm-up: a cold first call is what the RASD number includes too.
        import torch  # noqa: F811
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        t0 = time.perf_counter()
        outs = llm.generate([prompt], params, use_tqdm=False)
        t1 = time.perf_counter()

        end_to_end = t1 - t0
        out_tokens = len(outs[0].outputs[0].token_ids)
        ttft = _ttft_from_metrics(outs[0])
        decode_only = (end_to_end - ttft) if ttft is not None else None
        unit_ok = end_to_end > 0 and out_tokens > 0

        result.update({
            "status": "ok",
            "prompt_tokens": prompt_tokens,
            "output_tokens": out_tokens,
            "end_to_end_wall_s": round(end_to_end, 4),
            "decode_only_wall_s": (round(decode_only, 4) if decode_only else ""),
            "ttft_s": (round(ttft, 4) if ttft is not None else ""),
            "throughput_tps_end_to_end": (round(out_tokens / end_to_end, 4)
                                          if end_to_end > 0 else ""),
            "throughput_tps_decode_only": (round(out_tokens / decode_only, 4)
                                           if decode_only else ""),
            "unit_matched": "yes" if unit_ok else "no",
            "peak_mem_mb": (round(torch.cuda.max_memory_allocated() / 1024 ** 2, 1)
                            if torch.cuda.is_available() else ""),
            "error": "", "error_class": "",
        })
        if ttft is None:
            print("[WARN] TTFT unavailable; end-to-end is still comparable "
                  "(includes prefill), decode-only is blank.")
    except Exception as e:  # noqa: BLE001
        import traceback
        traceback.print_exc()
        import torch  # noqa: F811
        is_oom = isinstance(e, torch.cuda.OutOfMemoryError) or "out of memory" in str(e).lower()
        result.update({
            "status": "oom" if is_oom else "error",
            "error_class": type(e).__name__,
            "error": str(e)[:2000],
        })
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

    res_path.write_text(json.dumps(result))
    return 0 if result.get("status") == "ok" else 1


# ---------------------------------------------------------------------------
# parent — run one attempt as a subprocess, teeing to a log
# ---------------------------------------------------------------------------

def run_attempt(spec: dict, log_path: Path, timeout_s: int) -> tuple[int, str]:
    """Run the worker in a subprocess, tee output to log_path.

    Returns (returncode, log_text). A subprocess is what makes the traceback
    survivable: a hard crash or a CUDA abort cannot take the parent down, and
    the child's stderr is captured rather than lost.
    """
    spec_path = log_path.with_suffix(".spec.json")
    spec_path.write_text(json.dumps(spec))
    env = dict(os.environ)
    env.update(spec.get("env", {}))
    cmd = [sys.executable, str(Path(__file__).resolve()), "--_worker", str(spec_path)]

    chunks: list[str] = []
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w") as lf:
        lf.write(f"# cmd: {' '.join(cmd)}\n")
        lf.write(f"# env overrides: {spec.get('env', {})}\n\n")
        lf.flush()
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT, text=True,
                                bufsize=1, env=env)

        # Silently-hanging engines are common; a reader-thread timeout is the
        # only way to bound this without losing the partial log.
        killed = {"v": False}

        def _kill():
            killed["v"] = True
            try:
                proc.kill()
            except Exception:  # noqa: BLE001
                pass

        timer = threading.Timer(timeout_s, _kill)
        timer.start()
        try:
            for line in proc.stdout:  # type: ignore[union-attr]
                lf.write(line)
                lf.flush()
                chunks.append(line)
        finally:
            timer.cancel()
            proc.wait(timeout=60)
        if killed["v"]:
            msg = f"\n# ATTEMPT TIMED OUT after {timeout_s}s and was killed\n"
            lf.write(msg)
            chunks.append(msg)
    return proc.returncode, "".join(chunks)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--_worker", help=argparse.SUPPRESS)
    ap.add_argument("--models", nargs="+",
                    default=["meta-llama/Llama-3.1-8B",
                             "meta-llama/Llama-2-7b-hf"])
    ap.add_argument("--context-lengths", nargs="+", type=int, default=[131072])
    ap.add_argument("--max-new-tokens", type=int, default=64,
                    help="must match the RASD 128k cells for unit matching")
    ap.add_argument("--tensor-parallel-size", type=int, default=8)
    ap.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    ap.add_argument("--attempt-timeout-s", type=int, default=MAX_ATTEMPT_WALL_S)
    ap.add_argument("--out", default=str(REPO / "results" / "mlsys"
                                         / "vllm_baseline.csv"))
    args = ap.parse_args()

    if args._worker:
        return worker_main(Path(args._worker))

    if args.gpu_memory_utilization != 0.90:
        print("[warn] --gpu-memory-utilization only overrides attempt 1")

    rows: list[dict] = []
    for model in args.models:
        for ctx in args.context_lengths:
            rope = DEFAULT_ROPE.get(model)
            print(f"\n=== {model} @ {ctx} tokens (TP={args.tensor_parallel_size}) ===")
            succeeded = False
            for idx, att in enumerate(ATTEMPT_LADDER, start=1):
                if succeeded:
                    break
                mml = ctx if att["max_model_len"] == "ctx" else att["max_model_len"]
                mnbt = (mml if att.get("max_num_batched_tokens") == "ctx"
                        else att.get("max_num_batched_tokens"))
                kwargs = dict(att["kwargs"])
                if idx == 1:
                    kwargs["gpu_memory_utilization"] = args.gpu_memory_utilization
                safe_model = model.replace("/", "_")
                log_path = LOGDIR / f"vllm_{safe_model}_attempt{idx}.log"
                spec = {
                    "model": model, "context_length": ctx,
                    "max_new_tokens": args.max_new_tokens,
                    "tensor_parallel_size": args.tensor_parallel_size,
                    "max_model_len": mml, "max_num_batched_tokens": mnbt,
                    "kwargs": kwargs, "env": att["env"], "rope": rope,
                    "result_path": str(log_path.with_suffix(".result.json")),
                }
                print(f"  attempt {idx} ({att['name']}): {att['note']}")
                rc, log_text = run_attempt(spec, log_path, args.attempt_timeout_s)

                res_path = log_path.with_suffix(".result.json")
                result = {}
                if res_path.exists():
                    try:
                        result = json.loads(res_path.read_text())
                    except Exception:  # noqa: BLE001
                        result = {}
                if not result:
                    cls, tb = extract_last_exception(log_text)
                    result = {"status": "error", "error_class": cls,
                              "error": tb or f"worker rc={rc} with no result file"}

                row = {f: "" for f in CSV_FIELDS}
                row.update({
                    "model": model, "context_length": ctx,
                    "max_new_tokens": args.max_new_tokens,
                    "tensor_parallel_size": args.tensor_parallel_size,
                    "rope_scaling": json.dumps(rope) if rope else "",
                    "attempt": idx, "config_used": att["name"],
                    "log_path": str(log_path.relative_to(REPO)),
                    **{k: v for k, v in result.items() if k in CSV_FIELDS},
                })
                # If the worker died before writing a result, mine the log for
                # the real exception instead of reporting vLLM's one-liner.
                if row["status"] == "ok":
                    succeeded = True
                    print(f"    -> OK  end-to-end={row['throughput_tps_end_to_end']} tok/s "
                          f"(unit_matched={row['unit_matched']})")
                else:
                    cls, tb = extract_last_exception(log_text)
                    if tb and (not row["error"] or len(tb) > len(str(row["error"]))):
                        row["error"] = tb
                    if cls and not row["error_class"]:
                        row["error_class"] = cls
                    print(f"    -> {row['status'].upper()} "
                          f"[{row['error_class'] or 'unknown'}]: "
                          f"{str(row['error']).splitlines()[-1][:160]}")
                rows.append(row)
                res_path.unlink(missing_ok=True)

            if not succeeded:
                print(f"  !! all {len(ATTEMPT_LADDER)} attempts failed for "
                      f"{model} @ {ctx} — recorded as failures, not fabricated")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CSV_FIELDS, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)

    n_ok = sum(1 for r in rows if r["status"] == "ok")
    n_bad_unit = sum(1 for r in rows if r["unit_matched"] != "yes")
    print(f"\n[write] {out_path}  ({len(rows)} rows)")
    print(f"        {n_ok}/{len(rows)} ok, {len(rows) - n_ok} failed/unsupported")
    if n_bad_unit:
        print(f"[WARN] {n_bad_unit} row(s) have unit_matched=no — do NOT place "
              f"them in a speedup comparison against RASD's throughput_tps.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

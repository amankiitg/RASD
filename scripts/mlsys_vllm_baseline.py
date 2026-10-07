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
import hashlib
import json
import os
import re
import subprocess
import sys
import threading
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
# The target cross-check imports the campaign's analysis rule
# (src.analysis.losslessness) rather than restating it, so the repo root has to
# be importable: this script lives in scripts/, and running it as
# `python scripts/...` puts scripts/ on sys.path, not the repo root.
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
LOGDIR = REPO / "results" / "mlsys" / "logs"

CSV_FIELDS = [
    "model", "context_length", "max_new_tokens", "tensor_parallel_size",
    "rope_scaling", "prompt_tokens", "output_tokens",
    "end_to_end_wall_s", "decode_only_wall_s", "ttft_s",
    "throughput_tps_end_to_end", "throughput_tps_decode_only",
    "unit_matched", "peak_mem_mb", "status", "error",
    # MLSys follow-up — attempt provenance so a failure is diagnosable.
    "attempt", "config_used", "error_class", "log_path",
    # C1-C5 fairness provenance. Without these a reader cannot tell whether
    # the comparison was actually like-for-like: which vLLM, which precision,
    # which EOS policy, and — most importantly — WHICH PROMPT.
    "vllm_version", "quantization", "eos_policy", "prompt_source",
    "prompt_sha256",
    # Which RASD cell this vLLM cell is the counterpart of. The comparison is
    # paired per document, so a row that does not name its document cannot be
    # paired with anything.
    "doc_id", "temperature", "prompt_ids_from",
    # vLLM's own ids for the prompt, so "we passed the RASD ids" is checked
    # against what it actually consumed rather than asserted.
    "prompt_ids_used_sha256", "prompt_ids_verified", "prompt_ids_from_engine",
    "target_revision", "draft_revision",
    # The model length the row actually ran at. The ladder's last resort is a
    # 64k fallback, and without this column a 64k run is indistinguishable from
    # the 128k rung it claims to be the counterpart of.
    "max_model_len",
    # Which interpreter ran the engine. The baseline lives in its own venv with
    # its own torch (R7), so "which vLLM" is only half the provenance: the row
    # also has to say which interpreter produced it.
    "interpreter",
]

# C1: pin vLLM. A speedup ratio is only meaningful against a named release;
# "whatever pip resolved that afternoon" cannot be reproduced or cited.
VLLM_PIN = "0.6.3"

# C3: ONE EOS policy on BOTH systems. RASD's ARM4 cells run with
# ignore_eos=True and emit exactly max_new_tokens, so vLLM must do the same.
# If vLLM stopped early on EOS it would generate fewer tokens over less wall
# time and the tok/s ratio would flatter whichever side stopped sooner.
EOS_POLICY = "ignore_eos_exact_max_new_tokens"

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
        import vllm as _vllm
        from vllm import LLM, SamplingParams
    except Exception as e:  # noqa: BLE001
        import traceback
        traceback.print_exc()
        result.update({"status": "error",
                       "error_class": type(e).__name__,
                       "error": f"vLLM import failed: {e}"})
        res_path.write_text(json.dumps(result))
        return 1

    # C1: record the ACTUAL vLLM that ran, not the pin we hoped for.
    # ENFORCE the pin. The previous line fell back to reporting the pin when the
    # installed version was unreadable, so a row could claim the pinned release
    # while running something else -- and the pin is the one thing that makes a
    # speedup ratio citable. Refuse instead.
    vllm_version = getattr(_vllm, "__version__", "") or ""
    if vllm_version != VLLM_PIN:
        raise SystemExit(
            f"vLLM version is {vllm_version!r}, pin is {VLLM_PIN!r}. "
            f"A throughput ratio is only comparable against a named release; "
            f"install the pin or update VLLM_PIN deliberately."
        )

    model = spec["model"]
    ctx = spec["context_length"]
    max_new = spec["max_new_tokens"]
    max_model_len = spec["max_model_len"]
    rope = spec["rope"]
    quant = spec.get("quantization") or None

    try:
        llm_kwargs = dict(
            model=model,
            tensor_parallel_size=spec["tensor_parallel_size"],
            max_model_len=max_model_len,
            trust_remote_code=True,
            **spec["kwargs"],
        )
        # C4: the RASD side that the draft story rests on runs 4-bit
        # bitsandbytes weights, so the baseline needs BOTH a matched-quant row
        # and a bf16 row. They are separate rows and are never averaged.
        if quant == "bitsandbytes":
            llm_kwargs["quantization"] = "bitsandbytes"
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
        # C2: the EXACT token ids RASD generated with, passed as IDS.
        #
        # The previous version decoded them to text and let vLLM re-encode,
        # which reintroduces the very doubt the ids were meant to remove: decode
        # is not injective and re-encoding need not return the same sequence, so
        # "same prompt" was an assumption about the tokenizer rather than a fact
        # about the run. vLLM takes `prompt_token_ids` directly; use it.
        prompt_ids_out = None
        if spec.get("prompt_ids"):
            ids = list(spec["prompt_ids"])
            prompt = {"prompt_token_ids": [int(i) for i in ids]}
            prompt_tokens = len(ids)
            prompt_source = spec.get("prompt_source", "rasd_token_ids")
        else:
            # Fallback: right length, explicitly flagged NOT token-identical
            # so it can never be silently reported as unit-matched.
            prompt_target = max(1, min(ctx, max_model_len) - max_new)
            prompt = build_prompt(tokenizer, prompt_target)
            ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
            prompt_tokens = len(ids)
            prompt_source = "synthetic_same_paragraph_NOT_rasd_ids"
        prompt_sha = hashlib.sha256(
            json.dumps(list(ids)).encode()).hexdigest()[:16]
        # GREEDY. The RASD cells run at temperature 0.0; sampling here would
        # make the two systems answer different questions, and the acceptance
        # comparison would be between a sampled path and a greedy one.
        # logprobs=2 is what makes the TIE rule usable on this side: vLLM then
        # returns its own top-2 per position, and the difference of two logprobs
        # is the difference of the two logits (the partition function cancels),
        # so the gap is on the same scale as the target's recorded one.
        params = SamplingParams(temperature=0.0, top_p=1.0,
                                max_tokens=max_new, ignore_eos=True,
                                logprobs=2)

        # No warm-up: a cold first call is what the RASD number includes too.
        import torch  # noqa: F811
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        t0 = time.perf_counter()
        outs = llm.generate([prompt], params, use_tqdm=False)
        t1 = time.perf_counter()

        # Verify the ids vLLM actually consumed are the ids we handed it. If
        # they are not, the row is not comparable to the RASD cell no matter how
        # the other flags look, and the failure is silent.
        try:
            prompt_ids_out = list(outs[0].prompt_token_ids)
        except Exception:                              # noqa: BLE001
            prompt_ids_out = None

        end_to_end = t1 - t0
        out_tokens = len(outs[0].outputs[0].token_ids)
        ttft = _ttft_from_metrics(outs[0])
        decode_only = (end_to_end - ttft) if ttft is not None else None
        # C5: raw success is not enough; main() re-checks the fairness rules.
        unit_ok = end_to_end > 0 and out_tokens > 0

        result.update({
            "status": "ok",
            "prompt_tokens": prompt_tokens,
            "vllm_version": vllm_version,
            "quantization": quant or "bfloat16",
            "eos_policy": EOS_POLICY,
            "prompt_source": prompt_source,
            "prompt_sha256": prompt_sha,
            "prompt_ids_used_sha256": (
                hashlib.sha256(",".join(str(int(i))
                                        for i in prompt_ids_out).encode()).hexdigest()
                if prompt_ids_out else ""),
            "prompt_ids_verified": (
                "yes" if prompt_ids_out
                and [int(i) for i in prompt_ids_out] == [int(i) for i in ids]
                else ("no" if prompt_ids_out else "unavailable")),
            # The revisions the comparison is claimed against.
            "target_revision": spec.get("target_revision") or "",
            "draft_revision": spec.get("draft_revision") or "",
            "output_tokens": out_tokens,
            # What this engine actually emitted, and its own top1-top2 gap at
            # each of those positions. The parent compares these against the
            # RASD target-only sidecar for the same document under the tie rule;
            # without them the cross-check would be a throughput table with the
            # word "validation" in the title.
            "output_token_ids": [int(t) for t in outs[0].outputs[0].token_ids],
            "output_gaps": _position_gaps(outs[0].outputs[0]),
            "end_to_end_wall_s": round(end_to_end, 4),
            "decode_only_wall_s": (round(decode_only, 4) if decode_only else ""),
            "ttft_s": (round(ttft, 4) if ttft is not None else ""),
            "throughput_tps_end_to_end": (round(out_tokens / end_to_end, 4)
                                          if end_to_end > 0 else ""),
            # Same convention as RASD's decode_tps: the first token is a
            # prefill product, so the decode wall produces out_tokens - 1.
            # Using out_tokens here would bias the ratio toward 1.0.
            "throughput_tps_decode_only": (round(rasd_decode_rate(out_tokens,
                                                                  decode_only), 4)
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


def rasd_decode_rate(tokens: int, decode_wall_s) -> float:
    """RASD's decode-only rate: (tokens - 1) / decode wall."""
    if tokens < 1 or decode_wall_s is None:
        return 0.0
    return (int(tokens) - 1) / max(float(decode_wall_s), 1e-9)


def _position_gaps(output) -> list:
    """Top-1 minus top-2 of vLLM's own logprobs at each emitted position.

    `SamplingParams(logprobs=2)` returns a dict per position keyed by token id.
    Two entries are the minimum for a gap; fewer means the engine did not report
    enough, and the position gets `None` -- which the verdict treats as "no
    evidence of indifference", i.e. NOT a tie.
    """
    try:
        per_position = output.logprobs
    except Exception:                                      # noqa: BLE001
        return []
    if not per_position:
        return []
    gaps = []
    for entry in per_position:
        try:
            vals = sorted((float(v.logprob) for v in entry.values()),
                          reverse=True)
        except Exception:                                  # noqa: BLE001
            vals = []
        gaps.append(round(vals[0] - vals[1], 6) if len(vals) >= 2 else None)
    return gaps


def load_rasd_target_sidecars(tokens_dir) -> dict:
    """Target-only RASD sidecars, keyed by (doc_id, context_length).

    `spec_steps == 0` is what makes a sidecar a target-only run: a speculative
    cell's ids are a different claim (they depend on the draft), and comparing
    those against a plain vLLM decode would measure the draft, not the engine.
    """
    out = {}
    d = Path(tokens_dir)
    if not d.is_dir():
        raise SystemExit(f"--rasd-target-sidecars: no such directory: {d}")
    for f in sorted(d.glob("*.json")):
        try:
            sc = json.loads(f.read_text())
        except Exception:                                  # noqa: BLE001
            continue
        if int(sc.get("spec_steps") or 0) != 0:
            continue
        ids = sc.get("generated_token_ids")
        if not ids:
            continue
        key = (sc.get("doc_id") or "", int(sc.get("context_length") or 0))
        out[key] = {"run_id": sc.get("run_id"), "ids": [int(t) for t in ids],
                    "gaps": sc.get("token_gaps"), "sidecar": f.name}
    return out


def write_vllm_token_sidecars(rows: list, out_path) -> Path:
    """Per-cell emitted ids and gaps, so the cross-check can be re-run later.

    Written next to the comparison table rather than into it: the CSV stays a
    table, and the ids stay re-checkable from artifacts without a GPU.
    """
    d = Path(str(out_path)).with_suffix("")
    d = d.parent / (d.name + "_tokens")
    d.mkdir(parents=True, exist_ok=True)
    for r in rows:
        key = (f"{r.get('doc_id') or 'nodoc'}_"
               f"{r.get('context_length') or 0}_{r.get('attempt') or 1}")
        (d / f"{key}.json").write_text(json.dumps({
            "run_id": r.get("run_id") or "",
            "model": r.get("model"), "doc_id": r.get("doc_id"),
            "context_length": r.get("context_length"),
            "status": r.get("status"), "unit_matched": r.get("unit_matched"),
            "output_token_ids": r.get("_output_token_ids") or [],
            "output_gaps": r.get("_output_gaps") or [],
        }))
    return d


def _load_vllm_sidecars(d) -> dict:
    out = {}
    if d is None or not Path(d).is_dir():
        return out
    for f in sorted(Path(d).glob("*.json")):
        try:
            sc = json.loads(f.read_text())
        except Exception:                                  # noqa: BLE001
            continue
        out[(sc.get("doc_id") or "",
             int(sc.get("context_length") or 0))] = sc
    return out


def compare_targets(rows: list, tokens_dir, out_path,
                    vllm_sidecars_dir=None) -> int:
    """vLLM's greedy target-only output vs RASD's, under the tie rule (R7).

    This is what `impl_validation` now claims: two independent implementations of
    the same target at the same revision and context produce the same greedy
    continuation, and here is the throughput of both. It is NOT an acceptance
    cross-check -- vLLM cannot be given RASD's draft model, draft window, ring
    sharding or NF4 cache, so an acceptance agreement would be a statement about
    two different speculative implementations.
    """
    from src.analysis.losslessness import (TIE_GAP, compare_generations,
                                           stage_requirement)

    sidecars = load_rasd_target_sidecars(tokens_dir)
    vside = _load_vllm_sidecars(vllm_sidecars_dir)
    out_rows = []
    for r in rows:
        key = (r.get("doc_id") or "", int(r.get("context_length") or 0))
        ref = sidecars.get(key)
        base = {
            "model": r.get("model"), "doc_id": r.get("doc_id"),
            "context_length": r.get("context_length"),
            "max_model_len": r.get("max_model_len"),
            "unit_matched": r.get("unit_matched"),
            "vllm_version": r.get("vllm_version"),
            "interpreter": r.get("interpreter"),
            "vllm_run_id": r.get("run_id") or "",
            "rasd_run_id": (ref or {}).get("run_id", ""),
            "vllm_throughput_tps_end_to_end": r.get("throughput_tps_end_to_end"),
        }
        if r.get("status") != "ok":
            out_rows.append({**base, "verdict": "VLLM_FAILED",
                             "detail": f"status={r.get('status')}"})
            continue
        if r.get("unit_matched") != "yes":
            out_rows.append({**base, "verdict": "NOT_UNIT_MATCHED",
                             "detail": r.get("error") or "unit_matched=no"})
            continue
        if ref is None:
            out_rows.append({**base, "verdict": "NO_PAIR",
                             "detail": "no RASD target-only sidecar for this "
                                       "document and context"})
            continue
        mine = vside.get(key, {})
        v_ids = r.get("_output_token_ids") or mine.get("output_token_ids") or []
        v_gaps = r.get("_output_gaps") or mine.get("output_gaps")
        if not v_ids:
            out_rows.append({**base, "verdict": "NO_IDS",
                             "detail": "the vLLM row carries no emitted ids "
                                       "(no result file, or an older run)"})
            continue
        res = compare_generations(
            v_ids, ref["ids"],
            full_length=int(r.get("output_tokens") or 0) or None,
            spec_gaps=v_gaps, target_gaps=ref["gaps"],
        )
        out_rows.append({**base, **res})

    p = Path(out_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    fields = ["model", "doc_id", "context_length", "max_model_len",
              "unit_matched", "vllm_version", "interpreter", "vllm_run_id",
              "rasd_run_id", "verdict", "lossless", "numeric_tie",
              "tie_gap_threshold", "tie_positions", "verified_prefix",
              "first_mismatch_position", "gap_at_divergence_spec",
              "gap_at_divergence_target", "vllm_throughput_tps_end_to_end",
              "detail"]
    with p.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(out_rows)

    req = stage_requirement(out_rows,
                            full_length=max((int(x.get("verified_prefix") or 0)
                                             for x in out_rows), default=0) or 1)
    n_tie = sum(1 for x in out_rows if x.get("verdict") == "NUMERIC_TIE")
    n_mis = sum(1 for x in out_rows if x.get("verdict") == "MISMATCH")
    print(f"\n[compare] {p} ({len(out_rows)} cell(s))")
    for x in out_rows:
        print(f"    {x.get('verdict',''):<16} {x.get('doc_id','')}@"
              f"{x.get('context_length','')} ties={x.get('tie_positions','')} "
              f"{(x.get('detail') or '')[:60]}")
    print(f"    {n_tie} NUMERIC_TIE (reported, not failures), "
          f"{n_mis} MISMATCH (failures)")
    if n_mis:
        return 1
    # Only MISMATCH fails (plan revision, 2026-10-07). A cell that agreed, or
    # that diverged only where the target was indifferent, is a usable
    # cross-check; a cell that was never compared is not.
    usable = sum(1 for x in out_rows if str(x.get("verdict", "")).startswith(
        "LOSSLESS") or x.get("verdict") == "NUMERIC_TIE")
    if usable == 0:
        print("[FAIL] no cell produced a token-level comparison: this stage "
              "produced no usable cross-check, so it is recorded FAILED.")
        return 1
    return 0


def _ids_sha(ids) -> str:
    """sha256 of a prompt id list, in RASD's spelling.

    run_experiment hashes prompt ids as sha256(",".join(str(i) for i in ids))
    and records that in the sidecar's `prompt_sha256`. Hashing them differently
    here (json.dumps, truncated to 16 hex) meant an equality test between the
    two could NEVER hold: every row would have been flagged "not the RASD ids"
    while looking like a genuine prompt mismatch, and the one check that
    establishes the two systems saw the same prompt would have been inert.
    """
    return hashlib.sha256(
        ",".join(str(int(i)) for i in ids).encode()).hexdigest()


def load_rasd_cells(tokens_dir) -> list[dict]:
    """One cell per (context, document) from the RASD token sidecars.

    The sidecars are written by run_experiment for every run that ran with
    --save-generated-tokens. The ids used here are `engine_input_ids`: the ids the
    TARGET was actually fed, INCLUDING the leading BOS. `prompt_token_ids` is the
    same prompt WITHOUT the BOS -- it is what `prompt_tokens` counts and what the
    engine's prompt builder is defined over -- so replaying it into vLLM replays
    a sequence the target never saw. One token at the front changes every
    position's rotary phase, which is exactly the kind of difference this
    comparison exists to eliminate.

    So a sidecar with no `engine_input_ids` yields NO prompt ids, and a row
    without prompt ids is never unit-matched. `prompt_tokens + 1 ==
    len(engine_input_ids)` is checked too: the BOS is the one-token difference,
    and a sidecar whose engine ids do not account for it is a sidecar written by
    something other than the engine.

    A sidecar whose ids do not hash to its own recorded sha256 is a corrupt
    sidecar; it is reported and skipped, never used.
    """
    cells: list[dict] = []
    seen_cells: set = set()
    d = Path(tokens_dir)
    if not d.is_dir():
        raise SystemExit(f"--prompt-ids-from-sidecars: no such directory: {d}")
    for f in sorted(d.glob("*.json")):
        try:
            sc = json.loads(f.read_text())
        except Exception as exc:  # noqa: BLE001
            print(f"[warn] unreadable sidecar {f.name}: {exc}")
            continue
        ids = sc.get("engine_input_ids")
        if not ids:
            print(f"[warn] sidecar {f.name}: no engine_input_ids (the ids the "
                  f"target was actually fed, BOS included); this document "
                  f"cannot be given the target's prompt and will not be "
                  f"unit-matched")
            continue
        # The no-BOS prompt ids are still checked, because their hash is what the
        # sidecar records as the prompt's identity.
        pids = sc.get("prompt_token_ids")
        if pids and _ids_sha(pids) != (sc.get("prompt_sha256") or ""):
            print(f"[warn] sidecar {f.name}: prompt ids do not match the "
                  f"recorded sha256; skipping")
            continue
        ptok = sc.get("prompt_tokens")
        if ptok is not None and int(ptok) + 1 != len(ids):
            print(f"[warn] sidecar {f.name}: prompt_tokens={ptok} but the "
                  f"engine fed {len(ids)} ids; the difference must be exactly "
                  f"the BOS, so this sidecar did not come from the engine")
            continue
        # One cell per (document, context). The token directory holds a sidecar
        # for EVERY run of that document -- the speculative cell and its
        # target-only partner -- and they share a prompt, so without this the
        # same document is launched twice and appears twice in the comparison.
        key = (sc.get("doc_id") or "", int(sc.get("context_length") or 0))
        if key in seen_cells:
            continue
        seen_cells.add(key)
        cells.append({
            "doc_id": sc.get("doc_id") or "",
            "context_length": int(sc.get("context_length") or 0),
            "prompt_ids": [int(i) for i in ids],
            "prompt_sha256": sc.get("prompt_sha256"),
            "prompt_tokens": (int(ptok) if ptok is not None else None),
            "engine_input_ids": True,
            "sidecar": f.name,
        })
    if not cells:
        raise SystemExit(
            f"--prompt-ids-from-sidecars: no usable sidecar in {d} (a usable "
            f"one carries `engine_input_ids`). The vLLM rows would otherwise be "
            f"compared against prompts the RASD runs never used."
        )
    return cells


def _lookup_prompt_ids(mapping: dict, model: str, ctx: int):
    """Find the RASD token ids for one cell.

    Several key spellings are accepted so the producer does not have to guess
    ours. A "*" key means one prompt shared by every cell. Returning None is
    a legitimate outcome: it means "we could not prove this was the same
    prompt", and the row is then ineligible for unit matching.
    """
    if not mapping:
        return None
    for key in (f"{model}@{ctx}", f"{model}_{ctx}", str(ctx), "*"):
        if key in mapping:
            return mapping[key]
    return None


def _parse_revision_map(spec: str) -> dict:
    """`model=rev,model=rev` -> {model: rev}.

    A single `--target-revision` cannot describe `vllm_ladder`, which runs two
    different targets: the pin has to name which model it belongs to, or the row
    is compared against a revision chosen by position.
    """
    out: dict = {}
    for part in (spec or "").split(","):
        part = part.strip()
        if not part:
            continue
        if "=" not in part:
            raise SystemExit(
                f"revision map entry {part!r} is not 'model=revision'; a bare "
                f"revision cannot be attributed to a model")
        model, rev = part.split("=", 1)
        out[model.strip()] = rev.strip()
    return out


def _unit_match_verdict(row: dict, prompt_ids, args,
                        target_revision: str = "",
                        draft_revision: str = "") -> tuple[bool, str]:
    """C2/C3/C5: is this row genuinely comparable to a RASD 128k cell?

    All four conditions must hold. Failing any one makes the ratio
    meaningless, so the row is kept — the failure is itself a result — but it
    is flagged rather than quietly averaged into a speedup number.
    """
    why: list[str] = []
    if not prompt_ids:
        why.append("prompt token ids are not the RASD ids (C2)")
    elif row.get("prompt_ids_from_engine") != "yes":
        # The ids handed to vLLM must be the ids the TARGET was fed -- BOS and
        # all. A sidecar that only carries the no-BOS prompt ids would have this
        # empty, and the row is then a comparison against a sequence the target
        # never saw.
        why.append("prompt ids are not the engine's input ids (the BOS-carrying "
                   "`engine_input_ids`), so they are not the sequence the target "
                   "conditioned on")
    elif _ids_sha(prompt_ids) != (row.get("prompt_sha256") or ""):
        # The ids must be the ones the row says it used, or "we passed the RASD
        # prompt" is an assertion about a variable, not about the run.
        why.append("prompt sha256 does not match the ids actually used (C2)")
    if row.get("eos_policy") != EOS_POLICY:
        why.append(f"eos policy {row.get('eos_policy')!r} != {EOS_POLICY!r} (C3)")
    if int(row.get("tensor_parallel_size") or 0) != 8:
        why.append("tensor_parallel_size != 8 (C5)")
    if int(row.get("max_new_tokens") or 0) != args.matched_max_new_tokens:
        why.append("max_new_tokens != the matched RASD cell's "
                   f"{args.matched_max_new_tokens} (C5)")
    # NOTE the explicit None/"" test: `row.get("temperature") or -1` reads a
    # correct 0.0 as -1, because 0.0 is falsy, so every greedy row would have
    # been reported as a temperature mismatch.
    temp = row.get("temperature")
    if temp in ("", None) or float(temp) != 0.0:
        why.append(f"temperature {temp!r} != 0.0: the RASD cells are greedy, so "
                   f"a sampled vLLM row answers a different question (C5)")
    if (row.get("vllm_version") or "") != VLLM_PIN:
        why.append(f"vllm_version {row.get('vllm_version')!r} != pin "
                   f"{VLLM_PIN!r} (C1)")
    if not row.get("doc_id"):
        why.append("no doc_id, so the row cannot be paired with its RASD cell")
    # Context. The ladder falls back to a 64k model length when 128k will not
    # load. That fallback is a legitimate row to REPORT -- it is the honest
    # "vLLM could not match this rung" result -- but it is not the 128k cell it
    # would be paired with, and certifying it as unit-matched would put a 64k
    # throughput in the 128k speedup column.
    ctx_rung = int(row.get("context_length") or 0)
    ctx_ran = int(row.get("max_model_len") or 0)
    if ctx_rung and ctx_ran < ctx_rung:
        why.append(f"ran at max_model_len={ctx_ran} but the rung is "
                   f"{ctx_rung}: a shorter-context fallback is not this rung's "
                   f"baseline")
    # The ids vLLM consumed must be the ids supplied. "We passed them" is not
    # the same claim, and only the second one is about the run.
    if prompt_ids:
        if row.get("prompt_ids_verified") == "no":
            why.append("vLLM's prompt ids differ from the ones supplied")
        elif row.get("prompt_ids_verified") != "yes":
            why.append("vLLM did not report the prompt ids it consumed, so "
                       "'same prompt' is unverified")
    # Which revision was compared against which. A unit match against an
    # unpinned or different revision is a comparison across two models.
    for field, want in (("target_revision",
                         target_revision or args.target_revision),
                        ("draft_revision",
                         draft_revision or args.draft_revision)):
        if not want:
            continue
        if (row.get(field) or "") != want:
            why.append(f"{field} {row.get(field) or '<none>'!r} != the RASD "
                       f"cell's {want!r}")
    return (not why), "; ".join(why)


def build_row(cell: dict, model: str, ctx: int, prompt_ids, result: dict, *,
              attempt_idx: int, config_name: str, log_path: str, rope,
              args, max_model_len: int, target_revision: str = "",
              draft_revision: str = "") -> dict:
    """Assemble one CSV row from a worker result — the parent's only row path.

    `prompt_ids_from_engine` is decided HERE, from the RASD cell, never from the
    worker's result. Whether the ids fed to vLLM are the ids the target engine
    was actually given (BOS included) is a fact about the SIDECAR, and the
    worker never sees the sidecar, so it cannot report it. A parent that only
    copied the worker's fields therefore left the marker empty on every row and
    no vLLM row could ever be unit-matched -- the comparison would fail closed
    silently, and every lookup in a speedup table would come back empty. The
    rehearsal's stub hid this by setting the field itself; it now calls this
    function, so the rehearsal exercises the production path.
    """
    row = {f: "" for f in CSV_FIELDS}
    row.update({
        "model": model, "context_length": ctx,
        "max_new_tokens": args.max_new_tokens,
        "tensor_parallel_size": args.tensor_parallel_size,
        "rope_scaling": json.dumps(rope) if rope else "",
        "attempt": attempt_idx, "config_used": config_name,
        "log_path": log_path,
        # Which interpreter ran the engine (R7: the baseline has its own venv).
        "interpreter": sys.executable,
        **{k: v for k, v in (result or {}).items() if k in CSV_FIELDS},
    })
    # Authoritative pairing identity, set AFTER the worker's result so a worker
    # that did not report it cannot leave the row un-pairable.
    row["doc_id"] = cell.get("doc_id", "")
    row["prompt_ids_from"] = cell.get("sidecar", "")
    row["temperature"] = 0.0
    # What the ladder actually ran at, taken from the attempt rather than from
    # the worker: the fallback attempt runs a 64k model length and must be
    # distinguishable from the rung it was launched for.
    row["max_model_len"] = max_model_len
    # The RASD loader sets `engine_input_ids` on the cells it built from the
    # sidecars the engine wrote; anything else (a synthetic prompt, a bare
    # `--prompt-ids` list) has no such proof and stays ineligible.
    row["prompt_ids_from_engine"] = "yes" if cell.get("engine_input_ids") else ""
    if prompt_ids:
        row["prompt_sha256"] = _ids_sha(prompt_ids)
    # What the engine emitted, and its own gaps. Kept OUT of the CSV (a 1024-id
    # list per row would make the table unreadable) but ON the row, so the
    # cross-check compares what ran rather than re-reading a log.
    for k in ("output_token_ids", "output_gaps"):
        if isinstance(result, dict) and k in result:
            row["_" + k] = result[k]
    # C5: downgrade unless EVERY fairness criterion holds. A row that merely ran
    # successfully is not a comparable row.
    if row["status"] == "ok":
        ok_unit, why = _unit_match_verdict(
            row, prompt_ids, args, target_revision=target_revision,
            draft_revision=draft_revision)
        row["unit_matched"] = "yes" if ok_unit else "no"
        if not ok_unit:
            row["error_class"] = "UnitMismatch"
            row["error"] = f"ran ok but NOT comparable: {why}"
    return row


def write_rows(out_path, rows: list, append: bool = False) -> None:
    """Write the comparison CSV. `append` keeps one header across models."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not (append and out_path.exists()
                        and out_path.stat().st_size > 0)
    with out_path.open("a" if append else "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CSV_FIELDS, extrasaction="ignore")
        if write_header:
            w.writeheader()
        w.writerows(rows)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--_worker", help=argparse.SUPPRESS)
    ap.add_argument("--models", nargs="+",
                    default=["meta-llama/Llama-3.1-8B",
                             "meta-llama/Llama-2-7b-hf"])
    ap.add_argument("--context-lengths", nargs="+", type=int, default=[131072])
    ap.add_argument("--max-new-tokens", type=int, default=128,
                    help="C3: must equal the matched RASD cell's "
                         "max_new_tokens or the row is not unit-matched")
    ap.add_argument("--matched-max-new-tokens", type=int, default=128,
                    help="C5: max_new_tokens of the RASD cell we compare "
                         "against (ARM4 cells run 128)")
    ap.add_argument("--quantizations", nargs="+",
                    default=["bitsandbytes", "bfloat16"],
                    help="C4: emit a 4-bit bitsandbytes row AND a bf16 row")
    ap.add_argument("--target-revision", default="",
                    help="the RASD cell's target revision; a row is not "
                         "unit-matched unless it ran the same one")
    ap.add_argument("--draft-revision", default="",
                    help="the RASD cell's draft revision")
    ap.add_argument("--target-revisions", default="",
                    help="'model=rev,model=rev'. Required for a stage that "
                         "runs more than one target: a bare revision cannot "
                         "say which model it pins")
    ap.add_argument("--draft-revisions", default="",
                    help="'model=rev,model=rev' for the drafts")
    ap.add_argument("--documents", default=None,
                    help="comma list of doc_ids to run; the paired comparison "
                         "is per document, and a subset keeps a cross-check "
                         "cheap without silently dropping documents from a "
                         "larger stage")
    ap.add_argument("--prompt-ids-from-sidecars", default=None,
                    help="directory of RASD token sidecars; gives vLLM the "
                         "EXACT prompt ids the RASD cells fed, per document")
    ap.add_argument("--prompt-ids",
                    help="C2: JSON of the EXACT token ids RASD used. A flat "
                         "list, or a map keyed '<model>@<ctx>'. Without it "
                         "rows cannot be unit-matched.")
    ap.add_argument("--tensor-parallel-size", type=int, default=8)
    ap.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    ap.add_argument("--attempt-timeout-s", type=int, default=MAX_ATTEMPT_WALL_S)
    ap.add_argument("--out", default=str(REPO / "results" / "mlsys"
                                         / "vllm_baseline.csv"))
    ap.add_argument("--rasd-target-sidecars", default=None,
                    help="directory of RASD target-only token sidecars; with "
                         "--compare-out, the stage cross-checks vLLM's greedy "
                         "output against RASD's target-only output on the same "
                         "document, under the tie rule (R7)")
    ap.add_argument("--compare-out", default=None,
                    help="where to write the target cross-check table")
    ap.add_argument("--append", action="store_true",
                    help="append to --out instead of overwriting, so the two "
                         "targets can be run with their OWN matched "
                         "max_new_tokens (Llama-3.1 -> 128, Llama-2 -> 64) "
                         "and still land in one comparison table")
    args = ap.parse_args()

    if args._worker:
        return worker_main(Path(args._worker))

    prompt_ids_map: dict = {}
    if args.prompt_ids:
        raw = json.loads(Path(args.prompt_ids).read_text())
        prompt_ids_map = raw if isinstance(raw, dict) else {"*": raw}

    # The paired comparison is per document, so the cells are per document too.
    # A single prompt shared by every cell cannot be paired with anything.
    rasd_cells: list[dict] = []
    if args.prompt_ids_from_sidecars:
        rasd_cells = load_rasd_cells(args.prompt_ids_from_sidecars)
        if args.documents:
            want = [d.strip() for d in args.documents.split(",") if d.strip()]
            have = {c["doc_id"] for c in rasd_cells}
            missing = [d for d in want if d not in have]
            if missing:
                raise SystemExit(
                    f"--documents names {missing}, which have no sidecar in "
                    f"{args.prompt_ids_from_sidecars}; refusing to substitute "
                    f"a different document")
            rasd_cells = [c for c in rasd_cells if c["doc_id"] in want]
        ctxs = sorted({c["context_length"] for c in rasd_cells})
        docs = sorted({c["doc_id"] for c in rasd_cells})
        print(f"  RASD cells from sidecars: {len(rasd_cells)} "
              f"({len(docs)} documents x {len(ctxs)} context(s)): "
              f"{', '.join(docs)}")

    if args.gpu_memory_utilization != 0.90:
        print("[warn] --gpu-memory-utilization only overrides attempt 1")

    target_revisions = _parse_revision_map(args.target_revisions)
    draft_revisions = _parse_revision_map(args.draft_revisions)
    missing_rev = [m for m in args.models
                   if args.target_revisions and m not in target_revisions]
    if missing_rev:
        raise SystemExit(
            f"--target-revisions does not pin {missing_rev}. A row whose model "
            f"has no declared revision cannot be unit-matched, and defaulting "
            f"it to another model's pin would compare across revisions.")
    rows: list[dict] = []
    for model in args.models:
        for ctx in args.context_lengths:
            rope = DEFAULT_ROPE.get(model)
            for quant in args.quantizations:
              quant_label = quant or "bfloat16"
              # One iteration per document when the sidecars are available.
              cells_here = [c for c in rasd_cells if c["context_length"] == ctx]
              if not cells_here:
                  cells_here = [{"doc_id": "", "prompt_ids": None,
                                 "prompt_sha256": None, "sidecar": ""}]
              for _cell in cells_here:
                doc_id = _cell["doc_id"]
                print(f"\n=== {model} @ {ctx} tokens "
                      f"(TP={args.tensor_parallel_size}, {quant_label}, "
                      f"doc={doc_id or '<none>'}) ===")
                prompt_ids = _cell["prompt_ids"]
                if prompt_ids is None:
                    prompt_ids = _lookup_prompt_ids(prompt_ids_map, model, ctx)
                if prompt_ids:
                    print(f"    prompt: {len(prompt_ids)} exact RASD token ids "
                          f"(sha256 {_ids_sha(prompt_ids)}) "
                          f"from {_cell['sidecar'] or '--prompt-ids'}")
                else:
                    print("    prompt: synthetic, NOT the RASD ids — row will "
                          "be ineligible for unit_matched=yes")
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
                    log_path = (LOGDIR
                                / f"vllm_{safe_model}_{quant_label}_attempt{idx}.log")
                    spec = {
                        "model": model, "context_length": ctx,
                        "max_new_tokens": args.max_new_tokens,
                        "tensor_parallel_size": args.tensor_parallel_size,
                        "max_model_len": mml, "max_num_batched_tokens": mnbt,
                        "kwargs": kwargs, "env": att["env"], "rope": rope,
                        "quantization": quant,
                        "prompt_ids": prompt_ids,
                        # NOTE: `prompt_ids_from_engine` is deliberately NOT
                        # here. It is not an instruction to the worker -- the
                        # worker cannot know it, because it never sees the
                        # sidecar. build_row() sets it from `_cell`.
                        "target_revision": (target_revisions.get(model)
                                            or args.target_revision),
                        "draft_revision": (draft_revisions.get(model)
                                           or args.draft_revision),
                        "prompt_source": ("rasd_token_ids"
                                          if prompt_ids else "synthetic"),
                        "doc_id": doc_id,
                        "prompt_ids_from": _cell["sidecar"],
                        "prompt_sha256": _ids_sha(prompt_ids) if prompt_ids else "",
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

                    row = build_row(
                        _cell, model, ctx, prompt_ids, result,
                        attempt_idx=idx, config_name=att["name"],
                        log_path=str(log_path.relative_to(REPO)), rope=rope,
                        args=args, max_model_len=int(mml),
                        target_revision=(target_revisions.get(model)
                                         or args.target_revision),
                        draft_revision=(draft_revisions.get(model)
                                        or args.draft_revision))
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
    # --append keeps the header once and concatenates rows, so a second model
    # with a different matched max_new_tokens does not wipe the first.
    write_rows(out_path, rows, append=args.append)

    n_ok = sum(1 for r in rows if r["status"] == "ok")
    n_unit = sum(1 for r in rows if r["unit_matched"] == "yes")
    n_bad_unit = len(rows) - n_unit
    print(f"\n[write] {out_path}  ({len(rows)} rows)")
    print(f"        {n_ok}/{len(rows)} ok, {len(rows) - n_ok} failed/unsupported")
    if n_bad_unit:
        print(f"[WARN] {n_bad_unit} row(s) have unit_matched=no — do NOT place "
              f"them in a speedup comparison against RASD's throughput_tps.")
    if n_unit == 0:
        # A reference row that is not unit-matched is not a baseline. Exiting 0
        # here would let the stage be recorded ok and a speedup table be built
        # on zero comparable rows -- the exact failure the unit match exists to
        # prevent.
        print("[FAIL] no row is unit_matched=yes: this stage produced no usable "
              "baseline, so it is recorded FAILED rather than ok.")
        return 1
    if args.compare_out:
        if not args.rasd_target_sidecars:
            raise SystemExit("--compare-out needs --rasd-target-sidecars: the "
                             "cross-check is against RASD's target-only runs")
        vdir = write_vllm_token_sidecars(rows, args.compare_out)
        return compare_targets(rows, args.rasd_target_sidecars,
                               args.compare_out, vllm_sidecars_dir=vdir)
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""GPU-free stand-in for `run_experiment.py`, used by scripts/mlsys_rehearsal.sh.

WHY THIS EXISTS
---------------
The campaign has failed on plumbing, not on physics: a missing metadata file, a
CSV with no `paired_speedup` row, a stage refused on cost, a partial pull merged
into results/. None of those need a GPU; all of them need the WHOLE pipeline to
run, because each failure was only visible two stages later.

So this module replaces exactly one thing -- model execution -- and nothing else:

  * planning comes from the real `run_experiment` (`load_config`,
    `build_run_configs`), so the run count the manifest checks against is the
    real planner's count rather than a number this stub invented;
  * every artifact is written by the REAL writer (`append_csv`,
    `_build_per_token_record`, `write_per_token_sidecar`,
    `write_generated_tokens_sidecar`, `_guard_output_collision`), so a schema
    change in run_experiment propagates here instead of being papered over by a
    copy that drifts;
  * the values are synthetic and boring EXCEPT where the analysis code makes a
    structural demand -- the generation cap, the KV bookkeeping, the token-level
    agreement between a speculative run and its target-only partner, and the
    pairing identity. Those are the parts a stub must get exactly right, because
    exercising the checks that read them is the entire point.

The rehearsal installs this as the sandbox's `run_experiment.py` and renames the
real file to `_real_run_experiment.py`. It refuses any CLI flag it does not know:
a manifest that passes a flag this stub ignores would rehearse green while the
real run did something else.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import pathlib
import random
import sys
import zlib

SANDBOX = pathlib.Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(SANDBOX))

import _real_run_experiment as real          # noqa: E402
# The trace record builder lives with the engine, which is where the record is
# produced during a real run. Importing it (rather than copying the dict) means
# a schema change there reaches this stub.
from src.models.rasd_inference import (                          # noqa: E402
    _build_per_token_record, _round_commit_plan)

CSV_FIELDS = real.CSV_FIELDS

# Vocabulary ceiling for the synthetic ids; only has to be a legal token index
# for the models in the campaign.
VOCAB = 128256
BOS_ID = 128000          # Llama-3's <|begin_of_text|>


# ---------------------------------------------------------------------------
# Deterministic synthetic content
# ---------------------------------------------------------------------------

def _seed_of(*parts) -> int:
    """A process-stable seed.

    `hash()` is salted per interpreter, so a sidecar written in one process
    could not be reproduced by the next and the pairing would depend on which
    process wrote what.
    """
    return zlib.crc32("|".join(str(p) for p in parts).encode())


def doc_token_sequence(doc_id: str) -> list[int]:
    """The generated token ids for a document: the SAME for every arm.

    Keyed on the document alone, so a speculative run and its target-only
    partner emit identical ids by construction, and a short partner is a prefix
    of the same sequence. That is what makes the losslessness check exercise the
    real comparison instead of being handed a match, and what produces the
    LOSSLESS_PREFIX_128 verdict the plan requires for the seven short baselines.
    """
    rng = random.Random(_seed_of("doc", doc_id))
    return [rng.randrange(0, VOCAB) for _ in range(4096)]


def prompt_ids_for(run: dict) -> list[int]:
    """Synthetic prompt ids of exactly the length this run needs.

    The window is `[0, C - prompt_gen_tokens - 1)` -- the RUNG's generation
    length, not this arm's own cap -- because every arm of a rung has to see the
    same prompt for the losslessness pairing to mean anything. That is the same
    rule `run_experiment._prompt_gen_tokens` applies.
    """
    ctx = int(run.get("context_length", 0))
    gen = int(run.get("prompt_gen_tokens") or run.get("max_new_tokens", 0))
    n = max(1, ctx - 1 - gen)
    rng = random.Random(_seed_of("prompt", run.get("doc_id") or run["run_id"],
                                 ctx, gen))
    return [rng.randrange(0, VOCAB) for _ in range(n)]


def _ids_sha(ids: list[int]) -> str:
    """run_experiment's exact spelling for `prompt_sha256`, so the pairing guard
    compares like with like."""
    return hashlib.sha256(",".join(str(int(i)) for i in ids).encode()).hexdigest()


# ---------------------------------------------------------------------------
# The generation accounting the cap-smoke check asserts
# ---------------------------------------------------------------------------

def round_plan(cap: int, k: int) -> list[dict]:
    """Per-round commitments, produced BY THE ENGINE'S OWN PLANNER.

    This used to be a local reimplementation of the cap arithmetic, which is
    exactly the kind of copy a rehearsal must not have: the stub would have been
    checking its own arithmetic instead of the engine's, and a change to
    `_round_commit_plan` would have been rehearsed as green.

    Each round proposes `k` accepted drafts (`n_acc = k`); the real planner says
    how many of them fit in the remaining budget and whether that cut the round
    short, and the loop advances by what was committed. The result satisfies
    `1 + sum(committed) == cap` by construction, with the seed token counted
    once, because the budget is exhausted exactly.
    """
    if cap < 1:
        return []
    out: list[dict] = []
    generated = 1                      # the seed `cur_token` is emitted first
    while generated < cap:
        n_emit, committed, with_bonus, truncated = _round_commit_plan(
            cap - generated, k, k)
        out.append({"n_acc": k, "n_emitted": int(n_emit),
                    "n_committed": int(committed),
                    "round_truncated": bool(truncated),
                    "with_bonus": bool(with_bonus)})
        generated += int(committed)
    return out


class _Row(list):
    """A one-row tensor stand-in: `_build_per_token_record` calls `.tolist()`.

    Passing a plain list fails on that call, so the shape is matched rather than
    the function reimplemented.
    """

    def tolist(self):
        return list(self)


def engine_input_ids(prompt_ids: list[int]) -> list[int]:
    """The engine's own prompt tensor: the runner's prompt ids WITH the BOS.

    The engine sees `tokenizer(prompt)` (special tokens added) while the runner
    records `prompt_tokens` from `tokenizer(prompt, add_special_tokens=False)`.
    The two differ by exactly the BOS, which is why the identity has a `+ 1` in
    it and why the check can fail.
    """
    return [BOS_ID] + list(prompt_ids)


def engine_held_sequence(run: dict, prompt_ids: list[int], cap: int,
                         trace: list[dict], gen_ids: list[int]) -> list[int]:
    """The final token sequence the stub's engine-side loop holds.

    Mirrors `generated_ids = torch.cat([input_ids] + generated, dim=1)`: the
    engine's prompt tensor, then whatever the loop appended -- the seed the first
    round verifies, then each round's accepted prefix and, when the round was not
    cut short by the budget, its bonus token. Returned as an OBJECT so its length
    is MEASURED, exactly as the engine measures `int(generated_ids.shape[1])`.

    The sequence is NOT the trace's final `kv_len_after`. The KV covers the
    positions that have been forwarded, which is one fewer than the tokens
    emitted: the last emitted token's keys and values are computed by the next
    forward, and there is no next forward. Taking the KV length as the sequence
    length is an off-by-one, and the identity check is what caught it.
    """
    base = engine_input_ids(prompt_ids)
    held = list(base)
    consumed = 0
    held.append(int(gen_ids[consumed]))                 # the seed
    consumed += 1
    if not trace:
        # TARGET-ONLY: no verify rounds, so the loop is a plain decode -- one
        # token per step until the cap. Same held object, same measurement.
        held.extend(int(x) for x in gen_ids[consumed:int(cap)])
        consumed = int(cap)
    for p in trace:
        n_emit = int(p["n_emitted"])
        held.extend(int(x) for x in gen_ids[consumed:consumed + n_emit])
        consumed += n_emit
        if not p["round_truncated"]:
            held.append(int(gen_ids[consumed]))         # the bonus token
            consumed += 1
    emitted = len(held) - len(base)
    if emitted != int(cap):
        # The loop and the cap disagree. The measurement is returned as it is:
        # the identity check reports it, and inventing a value here would hide
        # exactly the defect the check exists for.
        print(f"  [stub] WARNING {run['run_id']}: the loop held {emitted} "
              f"tokens, the cap is {cap}", flush=True)
    return held


def build_trace(run: dict, prompt_tokens: int, cap: int) -> list[dict]:
    k = max(1, int(run.get("spec_steps") or 0))
    plan = round_plan(cap, k)
    rng = random.Random(_seed_of("trace", run["run_id"]))
    traces: list[dict] = []
    kv_before = prompt_tokens + 1          # prompt + the seed token
    for i, p in enumerate(plan):
        n_acc = int(p["n_acc"])
        # The REAL record builder, so the trace schema is run_experiment's.
        rec = _build_per_token_record(
            round_idx=i,
            global_pos_start=kv_before - 1,
            spec_steps=k,
            n_acc=n_acc,
            draft_seq=[_Row(rng.randrange(0, VOCAB) for _ in range(k))],
            accepted=[_Row([True] * n_acc + [False] * (k - n_acc))],
        )
        committed = int(p["n_committed"])
        rec["n_emitted"] = int(p["n_emitted"])
        rec["round_truncated"] = bool(p["round_truncated"])
        rec["n_committed"] = int(committed)
        rec["kv_len_before"] = int(kv_before)
        rec["kv_len_after"] = int(kv_before + committed)
        kv_before += committed
        traces.append(rec)
    return traces


# ---------------------------------------------------------------------------
# One run
# ---------------------------------------------------------------------------

def emit_run(run: dict, output_csv: pathlib.Path, log_per_token: bool,
             memory_trace: bool, save_text: bool,
             save_tokens: bool) -> dict:
    cap = int(run.get("max_new_tokens", 0))
    ctx = int(run.get("context_length", 0))
    is_spec = str(run.get("spec_steps", "")).strip() not in ("", "0")

    prompt_ids = prompt_ids_for(run)
    gen_ids = doc_token_sequence(run.get("doc_id") or run["run_id"])[:cap]

    row = {f: "" for f in CSV_FIELDS}
    row.update({
        "run_id": run["run_id"], "group": run["group"],
        "level_id": run["level_id"], "seed": run.get("seed", 42),
        "target_model_name": run.get("target_model_name", ""),
        "draft_model_name": run.get("draft_model_name", ""),
        "spec_steps": run.get("spec_steps", 0),
        "kv_block_size": run.get("kv_block_size", ""),
        "prefetch_depth": run.get("prefetch_depth", ""),
        "context_length": ctx, "dtype": run.get("dtype", "bfloat16"),
        "tokens_generated": cap, "max_new_tokens": cap,
        "draft_window_cap": run.get("draft_window_cap") or "",
        "draft_dtype": run.get("draft_dtype") or "nf4",
        "target_revision": run.get("target_revision") or "",
        "draft_revision": run.get("draft_revision") or "",
        "prompt_tokens": len(prompt_ids),
        "prompt_sha256": _ids_sha(prompt_ids),
        "prompt_source": run.get("prompt_source", "synthetic"),
        "doc_id": run.get("doc_id", "") or "",
        "temperature": run.get("temperature", 1.0),
        "top_p": run.get("top_p", 1.0),
        "ignore_eos": bool(run.get("ignore_eos", False)),
        "rope_arm": str(run.get("rope_arm", "") or ""),
        "status": "ok", "error": "",
    })
    # Pairing identity from the real helpers, so it cannot drift from what the
    # analysis code expects.
    row["pair_id"] = f"{ctx}:{row['doc_id']}" if row["doc_id"] else ""
    row["arm_role"] = real._arm_role(run)               # noqa: SLF001

    trace = build_trace(run, len(prompt_ids), cap) if is_spec else []
    n_rounds = len(trace)
    # From the engine-side OBJECT, never from the row's arithmetic.
    held = engine_held_sequence(run, prompt_ids, cap, trace, gen_ids)
    row["sequence_tokens"] = len(held)
    # The sidecar is the engine's own slice of the same object:
    # `generated_ids[0, input_ids.shape[1]:]`, i.e. everything after the
    # engine's prompt tensor. The checker compares the CSV's count against THIS.
    emitted_ids = held[len(engine_input_ids(prompt_ids)):]
    # The hash of the ids the engine emitted -- its own slice of the held object,
    # so it cannot disagree with the sidecar written from the same list.
    row["generated_tokens_sha256"] = hashlib.sha256(
        ",".join(str(i) for i in emitted_ids).encode()).hexdigest()

    # A plausible steady state: speculation decodes faster than the target
    # alone, which is the effect the campaign exists to measure.
    rng = random.Random(_seed_of("metrics", run["run_id"]))
    target_only_tps = 22.0 + rng.random() * 6.0
    if is_spec:
        # alpha_round = accepted-prefix / gamma, per round, rounds that were
        # truncated by the cap excluded -- the estimand the plan pre-registers.
        alpha = 0.55 + rng.random() * 0.25
        decode_tps = target_only_tps * (1.0 + 1.05 * alpha + 0.05 * rng.random())
        row["acceptance_rate"] = round(alpha, 4)
        row["n_rounds"] = n_rounds
    else:
        # A target-only run has no verify rounds, so there is no acceptance to
        # report. 0.0 is the structural zero the arm_role split keeps out of the
        # speculative marginal.
        decode_tps = target_only_tps
        row["acceptance_rate"] = 0.0
        row["n_rounds"] = 0
    time_sec = (cap - 1) / decode_tps
    ttft_ms = 900.0 + rng.random() * 200.0
    row.update({
        "time_sec": round(time_sec, 4),
        "ttft_ms": round(ttft_ms, 3),
        "throughput_tps": round(cap / time_sec, 2),
        "decode_tps": round(decode_tps, 3),
        "mean_latency_ms": round(time_sec * 1000.0 / max(1, n_rounds or cap), 3),
        "gpu_peak_mem_mb": round(62000.0 + rng.random() * 3000.0, 1),
    })
    if run.get("measure_target_ppl") and not is_spec:
        ppl_rng = random.Random(_seed_of("ppl", row["doc_id"], ctx))
        row["target_ppl"] = round(7.0 + ppl_rng.random() * 1.5, 6)
        row["ppl_tokens"] = 512

    if log_per_token and trace:
        real.write_per_token_sidecar(trace, output_csv, run["run_id"])
    if save_tokens:
        real.write_generated_tokens_sidecar(
            output_csv, run["run_id"], emitted_ids,
            {"doc_id": row["doc_id"], "prompt_tokens": row["prompt_tokens"],
             # The ids the target was fed, BOS included -- the engine's own
             # tensor, mirrored by the held-sequence object above.
             "engine_input_ids": list(held[:len(engine_input_ids(prompt_ids))]),
             "prompt_token_ids": list(prompt_ids),
             # The exact ids the engine fed. run_experiment's pg19_document path
             # records these from the document pool, and the vLLM comparison
             # reads them -- without the field the vLLM stage refuses to run,
             # which is precisely how this was found.
             "prompt_token_ids": prompt_ids,
             "prompt_sha256": row["prompt_sha256"],
             "context_length": ctx, "max_new_tokens": cap,
             "spec_steps": row["spec_steps"], "temperature": row["temperature"],
             "top_p": row["top_p"], "ignore_eos": row["ignore_eos"],
             "arm_role": row["arm_role"], "pair_id": row["pair_id"],
             # The target's top1-top2 gap at each emitted position, in the same
             # order as the ids. The rehearsal generates a decisive gap (well
             # above the 0.1 tie threshold) so a real divergence would be a
             # MISMATCH rather than silently excused as a numerics tie.
             "token_gaps": [round(3.0 + 0.001 * i, 6)
                            for i in range(len(emitted_ids))],
             # The numerics this run was measured under, exactly as
             # run_experiment records them. The vLLM cross-check reports these
             # beside its own, because the two engines cannot share an NF4 KV
             # path and their token divergence measures precision.
             "weight_precision": (run.get("weight_quant")
                                  or run.get("quantization")
                                  or run.get("dtype") or "bfloat16"),
             "kv_dtype": ("nf4" if run.get("kv_quant")
                          else (run.get("dtype") or "bfloat16")),
             "throughput_tps": row.get("throughput_tps"),
             "target_revision": row["target_revision"],
             "draft_revision": row["draft_revision"]})
    if save_text:
        gen_dir = output_csv.resolve().parent / "generated"
        gen_dir.mkdir(parents=True, exist_ok=True)
        (gen_dir / f"{run['run_id']}.txt").write_text(
            "rehearsal stub generation " * 8)
    if memory_trace:
        # Written here rather than through MemoryTracer: that class returns
        # early without CUDA, so a rehearsal relying on it would silently write
        # no memory trace at all -- the failure mode the flag exists to prevent.
        mt_dir = pathlib.Path(run.get("memory_trace_dir")
                              or output_csv.resolve().parent / "memory_trace")
        mt_dir.mkdir(parents=True, exist_ok=True)
        snaps = []
        for label, frac in (("post_load", 0.55), ("post_prefill", 0.86),
                            ("round1", 0.88), ("round2", 0.89), ("end", 0.90)):
            alloc = 80000.0 * frac
            snaps.append({"label": label, "allocated_mb": round(alloc, 3),
                          "reserved_mb": round(alloc * 1.03, 3),
                          "max_alloc_mb": round(alloc, 3),
                          "max_reserved_mb": round(alloc * 1.03, 3)})
        (mt_dir / f"{run['run_id']}.rank0.json").write_text(json.dumps(
            {"run_id": run["run_id"], "rank": 0, "device": "cuda:0",
             "snapshots": snaps}, indent=2))
    return row


# ---------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--groups", nargs="+", default=None)
    ap.add_argument("--seeds", nargs="+", type=int, default=None)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--debug", action="store_true")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--overwrite-stage", action="store_true")
    ap.add_argument("--nproc", type=int, default=8)
    ap.add_argument("--checkpoint-every", type=int, default=0)
    ap.add_argument("--abort-on-failure", action="store_true")
    ap.add_argument("--timeout-per-run-s", type=int, default=3600)
    ap.add_argument("--log-per-token", action="store_true")
    ap.add_argument("--profile", action="store_true")
    ap.add_argument("--memory-trace", action="store_true")
    ap.add_argument("--prompt-source", default="synthetic")
    ap.add_argument("--prompt-pg19-meta", default=None)
    ap.add_argument("--ruler-sidecar-dir", default=None)
    ap.add_argument("--save-generated-tokens", action="store_true")
    ap.add_argument("--save-generated-text", action="store_true")
    ap.add_argument("--draft-window-cap", type=int, default=None)
    ap.add_argument("--draft-dtype", choices=["auto", "nf4", "bf16"],
                    default="auto")
    ap.add_argument("--output", default="results/mlsys/stub.csv")
    ap.add_argument("--stage-id", default=None)
    ap.add_argument("--wandb-project", default="rehearsal")
    ap.add_argument("--_worker", default=None)
    args, unknown = ap.parse_known_args(sys.argv[1:])
    if unknown:
        # Not a warning: a flag the manifest passes and this stub ignores is a
        # difference between the rehearsal and the real run, which is exactly
        # what a rehearsal must not have.
        raise SystemExit(
            f"STUB: unknown flag(s) {unknown}. The rehearsal cannot stand in "
            f"for a command line it does not understand; teach the stub the "
            f"flag (and what the real run does with it) or fix the manifest.")

    cfg = real.load_config(args.config)
    runs = real.build_run_configs(cfg, args.groups, args.debug,
                                  seed_filter=args.seeds)
    # The real propagation, not a copy of it: --profile,
    # prompt-source validation and the RULER setup all live there.
    real.apply_cli_to_runs(args, runs)

    if args.dry_run:
        # THE REAL PRINTER, character for character. `expected_rows` in the
        # manifest counts the run lines of this output; a second printer that
        # differed by one space in the separator would make the manifest's row
        # check agree with the stub and with nothing else -- it would pass while
        # the real run's plan was never examined.
        print(real.format_dry_run(runs))
        return 0

    output_csv = pathlib.Path(args.output)
    # The real collision guard: a stage writing into another stage's file is one
    # of the failure modes this rehearsal exists to catch.
    real._guard_output_collision(                       # noqa: SLF001
        output_csv, args.stage_id or "unnamed", overwrite=args.overwrite_stage)

    # ---- rehearsal-only fault injection ---------------------------------
    # Two knobs, read from the environment so the MANIFEST is unchanged: the
    # failure has to arrive through the same channel a real failure would (a
    # non-zero exit and an error row), or the rehearsal would be testing its own
    # shortcut instead of the runner's abort path.
    #
    #   MLSYS_REHEARSAL_FAIL_RUN=<substring>   that run is recorded as an error
    #   MLSYS_REHEARSAL_SLOW_RUN=<substring>   that run's simulated wall exceeds
    #                                          --timeout-per-run-s, so it times out
    fail_pat = os.environ.get("MLSYS_REHEARSAL_FAIL_RUN", "")
    slow_pat = os.environ.get("MLSYS_REHEARSAL_SLOW_RUN", "")
    wrote = 0
    for run in runs:
        rid = run["run_id"]
        if fail_pat and fail_pat in rid:
            row = {f: "" for f in CSV_FIELDS}
            row.update({"run_id": rid, "group": run.get("group", ""),
                        "level_id": run.get("level_id", ""),
                        "seed": run.get("seed", 42),
                        "context_length": run.get("context_length", ""),
                        "spec_steps": run.get("spec_steps", 0),
                        "max_new_tokens": run.get("max_new_tokens", 0),
                        "status": "error",
                        "error": "rehearsal-injected failure"})
            real.append_csv(output_csv, row)
            wrote += 1
            print(f"  stub run {rid}: INJECTED FAILURE")
            if args.abort_on_failure:
                print("--abort-on-failure: stopping at the first error row", flush=True)
                return 1
            continue
        row = emit_run(run, output_csv, args.log_per_token, args.memory_trace,
                       args.save_generated_text, args.save_generated_tokens)
        if slow_pat and slow_pat in rid:
            # Honour --timeout-per-run-s the way the real runner does: a run
            # whose wall exceeds the budget is a timeout, not a result. The stub
            # does not actually sleep -- the point is that the timeout PATH is
            # exercised, not that a CPU stands in for an A100.
            row["status"] = "error"
            row["error"] = (f"timeout after {args.timeout_per_run_s}s "
                            f"(simulated wall {args.timeout_per_run_s + 60}s)")
            print(f"  stub run {rid}: TIMEOUT at {args.timeout_per_run_s}s")
        real.append_csv(output_csv, row)
        wrote += 1
        if row["status"] != "ok" and args.abort_on_failure:
            print("--abort-on-failure: stopping at the first error row", flush=True)
            return 1
    print(f"stub wrote {wrote} rows to {output_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

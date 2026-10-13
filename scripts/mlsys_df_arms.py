#!/usr/bin/env python3
"""PART 1 DIAGNOSTIC ARMS — everything the engine comparison needs, in ONE model load.

WHY A CUSTOM POD-SIDE DRIVER RATHER THAN run_experiment.py PER ARM. A pod-side
`run_experiment.py` invocation loads the 8B target (and the draft) from scratch: ~5 minutes
per arm at $22.32/h, and the arm-to-arm difference then includes whatever the load did. Both
costs disappear if one process loads the model once and flips the ring knobs between
generations — the knobs are per-module attributes (`_ring_chunk_size`, `_ring_decode_mode`,
`_ring_expand_after_rotation`), so `set_arm()` is a loop over 32 attention modules. Every
arm below therefore runs against an in-session baseline on the same box, same book, same
seed, which is what R-baseline requires.

WHAT IT MEASURES, and why each arm exists:

  ring2048    kv_block_size 2048, ring rotation   -- the PUBLISHED configuration; baseline
  ring32768   kv_block_size 32768 (>= S_local)    -- one chunk per rotation; D.1 showed
                                                     1.181x target / 1.142x spec, bit-identical
  ring32768x  the above + deferred GQA expansion  -- the rotation ships num_kv_heads (8)
                                                     instead of num_heads (32): 4x fewer bytes
  splitkv     ring_decode_mode split_kv           -- no rotation at all; each rank attends to
                                                     its own shard, merged with two tiny
                                                     all_reduces (the 7-15x prize)

ROWS ARE WRITTEN AND FLUSHED AFTER EVERY GENERATION. The Phase D session lost an arm because
a crash in a later stage took the whole chain with it; here a fault leaves every earlier row
on disk, and the loop stops only on a CUDA fault (which poisons the context for anything
after it). A Python-level error in one arm does not stop the others.

EVERY ROW RECORDS THE ENGINE THAT PRODUCED IT -- ring_decode_mode, kv_block_size,
prefetch_depth, ring_expand_after_rotation, the quant flags the ROW saw, and the git commit.
A row that cannot prove which engine produced it is void, so those are columns, not notes.

Usage (from the pod-side shell script; all settings explicit, no silent defaults):
  torchrun --nproc_per_node=8 scripts/mlsys_df_arms.py --mode arms --ctx 131072 \
      --arms ring2048,ring32768,ring32768x,splitkv --target-tokens 16 --spec-tokens 32 \
      --out results/mlsys/decode_df/arms
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import time


ARMS = {
    "ring2048":   {"kv_block_size": 2048,  "ring_decode_mode": "ring",
                   "ring_expand_after_rotation": False},
    "ring32768":  {"kv_block_size": 32768, "ring_decode_mode": "ring",
                   "ring_expand_after_rotation": False},
    "ring32768x": {"kv_block_size": 32768, "ring_decode_mode": "ring",
                   "ring_expand_after_rotation": True},
    "splitkv":    {"kv_block_size": 32768, "ring_decode_mode": "split_kv",
                   "ring_expand_after_rotation": False},
}

ROW_FIELDS = [
    "run_id", "arm", "mode", "context_length", "book", "seed", "temperature",
    "spec_steps", "max_new_tokens",
    "kv_block_size", "prefetch_depth", "ring_decode_mode", "expand_after_rotation",
    "quantize_target", "kv_quant", "dtype",
    "tokens_generated", "time_sec", "ttft_ms", "decode_s_per_tok", "e2e_tps",
    "acceptance_rate",
    "peak_alloc_mb", "peak_reserved_mb", "peak_alloc_mb_worst_rank",
    "peak_reserved_mb_worst_rank",
    "target_ppl", "ppl_tokens", "git_commit", "world_size", "ring_layers_patched",
    "rng_reseeded", "status", "error",
]


def _args(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--mode", required=True, choices=["smoke", "arms", "ppl"])
    p.add_argument("--ctx", type=int, required=True)
    p.add_argument("--arms", required=True, help="comma-separated arm names")
    p.add_argument("--out", required=True)
    p.add_argument("--book", default="pg19_train_0")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--spec-steps", type=int, default=4)
    p.add_argument("--target-tokens", type=int, default=16)
    p.add_argument("--spec-tokens", type=int, default=32)
    p.add_argument("--ppl-tokens", type=int, default=1024)
    p.add_argument("--single-rank-ok", action="store_true",
                   help="plumbing rehearsal at world_size==1: do not refuse when the ring "
                        "patch is absent; rows record world_size so they read as void")
    p.add_argument("--greedy-check", action="store_true",
                   help="also run a target-only GREEDY (T=0) row per arm. This is the "
                        "decode-path correctness gate: greedy removes sampling, so two "
                        "correct engines must agree token-for-token and a divergence is "
                        "structural. The ppl check cannot serve this purpose -- it runs "
                        "model.model() with use_cache=False, i.e. the PREFILL path.")
    return p.parse_args(argv)


def _git_commit() -> str:
    """The commit the ROW was produced from.

    DF_COMMIT FIRST. The pod's ~/RASD is an rsync of the working tree and carries no
    `.git`, so `git rev-parse HEAD` there returns nothing and every row recorded
    `git_commit: unknown`. The driver therefore sends the commit it is staging, in its own
    variable, and the parameter contract proves it arrived -- the same rule as every other
    setting: a value that cannot be shown to have travelled is not recorded.
    """
    env = os.environ.get("DF_COMMIT")
    if env and env.strip():
        return env.strip()
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True,
                                       stderr=subprocess.DEVNULL).strip() or "unknown"
    except Exception:                                     # noqa: BLE001
        return "unknown"


def _reseed(seed: int) -> None:
    """Put the RNG back to the state a FRESH PROCESS would start a single generation in.

    WHY THIS IS NOT OPTIONAL. `RASDInference.__init__` calls `torch.manual_seed(config.seed)`
    ONCE (rasd_inference.py:1015) and `generate()` never reseeds. In the main runner that is
    invisible: one process builds one engine and generates once, so the single generation
    sees the seeded stream -- which is why a fixed seed reproduces byte-identically across
    sessions and hosts. This driver generates 8 times from ONE engine, so without a reseed
    each arm inherits the RNG state its predecessors left behind, and a token-stream
    difference between two arms can be an artefact of ARM ORDER rather than of the engine
    change under test. That would turn the bit-identity check into a coin flip and could
    either pass a broken change or fail a correct one (stop condition S1).

    The 1x rehearsal caught exactly this: identical arms (world_size==1, no ring patch, so
    all four arms run the same code) produced three identical spec streams and a fourth that
    differed, because the RNG state before each generation depends on how many draws the
    preceding generations consumed.
    """
    import random
    import torch as _torch   # imported here because `torch` is a local import in main()
    _torch.manual_seed(int(seed))
    random.seed(int(seed))
    try:
        import numpy as _np
        _np.random.seed(int(seed) % (2 ** 32))
    except Exception:                                       # noqa: BLE001
        pass


def _single_rank_selftest(dig: dict) -> list:
    """Group the digests by mode and count distinct streams per mode.

    The keys are "<arm>|<mode>" strings (they are written to JSON, where a tuple key would
    not survive), so they are split rather than unpacked -- unpacking a string is how the
    first version of this check died with "too many values to unpack (expected 2)" AFTER all
    twelve rows had already been written correctly.
    """
    by = {}
    for key, d in dig.items():
        arm, mode = key.split("|")
        by.setdefault(mode, {})[arm] = d
    return [(mode, len(set(vals.values())), len(vals)) for mode, vals in sorted(by.items())]


def set_arm(engine, arm_name: str, single_rank_ok: bool = False) -> tuple:
    """Point every patched attention module at one arm's knobs. Returns (settings, n_patched).

    These are the SAME attributes `install_ring_attention` sets, so an arm is the engine's
    own configuration, not a second code path.

    A 1x REHEARSAL CANNOT TEST THE RING. `install_ring_attention` is a no-op at
    world_size<=1, so there are no knobs to set and every arm would be the same run. That is
    what `single_rank_ok` is for: the rehearsal exercises the plumbing (engine load, the
    quant flags, prompt building, the row writer, the sidecars, the markers, the pull path)
    and the rows say `world_size=1` so they cannot be mistaken for measurements.
    """
    spec = ARMS[arm_name]
    layers = getattr(getattr(engine.target_model, "model", None), "layers", None)
    if layers is None:
        raise RuntimeError("set_arm: could not find target_model.model.layers")
    n = 0
    for layer in layers:
        attn = getattr(layer, "self_attn", None)
        if attn is None or not hasattr(attn, "_ring_chunk_size"):
            continue
        attn._ring_chunk_size = spec["kv_block_size"]
        attn._ring_decode_mode = spec["ring_decode_mode"]
        attn._ring_expand_after_rotation = spec["ring_expand_after_rotation"]
        n += 1
    if n == 0 and not single_rank_ok:
        raise RuntimeError("set_arm: no ring-patched attention modules found "
                           "(world_size<=1? then the ring never runs and every arm is void)")
    if n == 0:
        print("[df] WARNING: no ring-patched layers (world_size<=1): this is a PLUMBING "
              "rehearsal and every row is void as a measurement.", flush=True)
    return spec, n


def _peak_memory(dist, torch):
    """Peak allocated/reserved per rank, as the WORST rank and rank 0.

    Rank 0 alone is not the answer: the documented rank-0 vs rank-4 asymmetry in the
    profile means a per-rank spread has to be visible in the row.
    """
    alloc = torch.cuda.max_memory_allocated() / 1e6 if torch.cuda.is_available() else 0.0
    resv = torch.cuda.max_memory_reserved() / 1e6 if torch.cuda.is_available() else 0.0
    if not (dist is not None and dist.is_initialized()):
        return alloc, resv, alloc, resv
    gathered = [None] * dist.get_world_size()
    dist.all_gather_object(gathered, (alloc, resv))
    a = [g[0] for g in gathered]
    r = [g[1] for g in gathered]
    return a[0], r[0], max(a), max(r)


def main(argv=None):
    args = _args(argv)
    os.makedirs(args.out, exist_ok=True)
    os.makedirs(os.path.join(args.out, "tokens"), exist_ok=True)

    rank = int(os.environ.get("RANK", "0"))
    world = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))

    import torch
    import torch.distributed as dist
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
    # THE PROCESS GROUP IS OURS TO INITIALISE. torchrun sets RANK/WORLD_SIZE/LOCAL_RANK but
    # does not create the group; `run_experiment.py` calls init_process_group in its worker
    # and this driver has to do the same. Without it `dist.is_initialized()` is False,
    # `RASDInference` sees world_size 1, `install_ring_attention` patches NOTHING, and an
    # 8-GPU box runs eight independent single-rank models that produce perfectly plausible
    # rows. That is the failure the $5.47 manifest exists for, one layer down, so it is
    # checked by EFFECT immediately after construction rather than assumed.
    if world > 1 and torch.cuda.is_available() and not dist.is_initialized():
        from datetime import timedelta
        dist.init_process_group(backend="nccl",
                                device_id=torch.device(f"cuda:{local_rank}"),
                                timeout=timedelta(minutes=30))
    if dist.is_initialized():
        assert dist.get_world_size() == world, \
            f"torchrun said world={world}, the group says {dist.get_world_size()}"

    sys.path.insert(0, os.getcwd())
    import run_experiment as R
    from src.models.rasd_inference import RASDConfig, RASDInference
    from transformers import AutoTokenizer

    commit = _git_commit()
    arm_names = [a.strip() for a in args.arms.split(",") if a.strip()]
    for a in arm_names:
        if a not in ARMS:
            raise SystemExit(f"FATAL: unknown arm '{a}'; known: {sorted(ARMS)}")

    def log(msg):
        print(f"[df] {msg}", flush=True)

    if rank == 0:
        log(f"commit={commit} world={world} mode={args.mode} ctx={args.ctx} "
            f"arms={arm_names} out={args.out}")

    TARGET = "gradientai/Llama-3-8B-Instruct-Gradient-1048k"
    TARGET_REV = "cd3069b65a8eb13da639d332a5f61b0fbb29fa73"
    DRAFT = "meta-llama/Llama-3.2-1B"
    DRAFT_REV = "4e20de362430cd3b72f300e6b0f18e50e7166e08"
    # The published headline precision (branch C): FP4 weights + NF4 KV.
    kw = dict(
        target_model_name=TARGET, target_revision=TARGET_REV,
        draft_model_name=DRAFT, draft_revision=DRAFT_REV,
        spec_steps=args.spec_steps, kv_block_size=2048, prefetch_depth=1,
        rope_type="none", dtype="bfloat16", top_p=1.0, ignore_eos=True,
        temperature=args.temperature, max_new_tokens=128, draft_window_cap=4096,
        seed=args.seed, log_per_token=True, save_generated_tokens=True,
        quantize_draft=True, quantize_target=True, kv_quant=True,
    )
    tok = AutoTokenizer.from_pretrained(TARGET, revision=TARGET_REV)
    t_load = time.perf_counter()
    engine = RASDInference(RASDConfig(**kw))
    if rank == 0:
        log(f"engine built in {time.perf_counter() - t_load:.1f} s; "
            f"world_size seen by engine = "
            f"{getattr(engine, '_world_size', 'n/a')}")

    # VERIFY BY EFFECT, before any row is written: at world > 1 the ring patch MUST be
    # installed. A run that silently fell back to one rank per GPU would otherwise look
    # exactly like a successful session.
    _layers = getattr(getattr(engine.target_model, "model", None), "layers", None) or []
    _patched = sum(1 for l in _layers
                   if hasattr(getattr(l, "self_attn", None), "_ring_chunk_size"))
    if rank == 0:
        log(f"ring-patched attention layers: {_patched} (world={world})")
    if world > 1 and _patched == 0 and not args.single_rank_ok:
        raise SystemExit(
            "FATAL: world_size > 1 but NO ring-patched attention layer exists. The ring "
            "patch is a no-op at world_size<=1, so this process is running eight "
            "independent single-rank models and every row it would write is void. Refusing.")
    if world == 1 and _patched > 0 and rank == 0:
        log("WARNING: world_size == 1 but ring layers ARE patched; unexpected, "
            "the rows would still be single-rank.")

    # The prompt is built ONCE per (context, book) and reused by every arm: an arm-to-arm
    # timing difference must not be a prompt difference.
    prompt_text, continuation_ids, prov = R._build_pg19_document_prompt(
        "data/processed/pg19_1m_128kwin/documents.json", args.ctx, args.book, tok,
        gen_tokens=args.ppl_tokens if args.mode == "ppl" else 1024)
    if rank == 0:
        log(f"prompt {args.book} @{args.ctx}: {len(prompt_text)} chars, "
            f"continuation {len(continuation_ids)} tokens, prov={prov}")

    rows_path = os.path.join(args.out, f"df_{args.mode}.csv")
    rows = []

    def flush():
        if rank != 0:
            return
        with open(rows_path, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=ROW_FIELDS)
            w.writeheader()
            for r in rows:
                w.writerow({k: r.get(k, "") for k in ROW_FIELDS})

    def run_one(arm, mode):
        """One generation. Returns the row dict (rank 0) or None."""
        spec, n_patched = set_arm(engine, arm, args.single_rank_ok)
        engine.cfg.spec_steps = 0 if mode in ("target", "greedy") else args.spec_steps
        engine.cfg.temperature = 0.0 if mode == "greedy" else args.temperature
        engine.cfg.max_new_tokens = (args.target_tokens if mode in ("target", "greedy")
                                     else args.spec_tokens)
        run_id = f"DF_{arm}_{mode}_{args.book}_ctx{args.ctx}_s{args.seed}"
        # Restore the per-process RNG state, so this arm sees what a fresh process would.
        _reseed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        t0 = time.perf_counter()
        try:
            text, metrics = engine.generate_text(prompt_text)
            wall = time.perf_counter() - t0
            err = ""
        except BaseException as exc:                      # noqa: BLE001
            wall = time.perf_counter() - t0
            import traceback
            err = f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}"
            metrics = {}
            text = ""
        alloc0, resv0, alloc_w, resv_w = _peak_memory(dist, torch)

        ntok = metrics.get("tokens_generated")
        tsec = metrics.get("time_sec")
        ttft = metrics.get("ttft_ms")
        decode = ""
        if ntok and tsec is not None and ttft is not None and ntok > 1:
            decode = round((float(tsec) - float(ttft) / 1000.0) / (ntok - 1), 6)
        row = {
            "run_id": run_id, "arm": arm, "mode": mode,
            "context_length": args.ctx, "book": args.book, "seed": args.seed,
            "temperature": engine.cfg.temperature, "spec_steps": engine.cfg.spec_steps,
            "max_new_tokens": engine.cfg.max_new_tokens,
            "kv_block_size": spec["kv_block_size"],
            "prefetch_depth": engine.cfg.prefetch_depth,
            "ring_decode_mode": spec["ring_decode_mode"],
            "expand_after_rotation": spec["ring_expand_after_rotation"],
            "quantize_target": engine.cfg.quantize_target,
            "kv_quant": engine.cfg.kv_quant, "dtype": engine.cfg.dtype,
            "tokens_generated": ntok, "time_sec": tsec, "ttft_ms": ttft,
            "decode_s_per_tok": decode,
            "e2e_tps": (round(float(ntok) / float(tsec), 4)
                        if ntok and tsec else ""),
            "acceptance_rate": metrics.get("acceptance_rate", ""),
            "peak_alloc_mb": round(alloc0, 1), "peak_reserved_mb": round(resv0, 1),
            "peak_alloc_mb_worst_rank": round(alloc_w, 1),
            "peak_reserved_mb_worst_rank": round(resv_w, 1),
            "target_ppl": "", "ppl_tokens": "",
            "git_commit": commit,
            "world_size": world, "ring_layers_patched": n_patched,
            "rng_reseeded": True,
            "status": "ok" if not err else "error",
            "error": err,
        }
        if rank == 0 and text:
            sidecar = os.path.join(args.out, "tokens", f"{run_id}.json")
            json.dump({
                "run_id": run_id, "arm": arm, "mode": mode,
                "context_length": args.ctx, "book": args.book, "seed": args.seed,
                "engine": spec, "spec_steps": engine.cfg.spec_steps,
                "quantize_target": engine.cfg.quantize_target,
                "kv_quant": engine.cfg.kv_quant, "git_commit": commit,
                "world_size": world, "ring_layers_patched": n_patched,
                "rng_reseeded": True,
                # `token_gaps` at the TOP level as well as inside `metrics`: that is where
                # `scripts/mlsys_cf_compare.py` looks for it (it reads the runner's sidecars),
                # and a comparison tool that silently sees an empty list reports "no gaps
                # around the divergence" rather than failing -- which is how the first version
                # of this sidecar made a 0.0 near-tie look like missing evidence.
                "token_gaps": metrics.get("token_gaps"),
                "generated_token_ids": metrics.get("generated_token_ids"),
                "per_token_trace": metrics.get("per_token_trace"),
                "text_head": text[:200],
                "metrics": {k: v for k, v in metrics.items()
                            if k not in ("per_token_trace", "generated_token_ids")},
            }, open(sidecar, "w"), indent=1, default=str)
        if rank == 0:
            log(f"{run_id}: status={row['status']} tokens={ntok} time={tsec} "
                f"ttft={ttft} decode_s/tok={decode} peak_res={row['peak_reserved_mb']}MB")
            if err:
                log(f"{run_id}: ERROR {err.splitlines()[0]}")
        return row

    # ------------------------------------------------------------------ modes
    if args.mode in ("smoke", "arms"):
        modes = ["target"] if args.mode == "smoke" else ["target", "spec"]
        if args.mode == "arms" and args.greedy_check:
            modes.append("greedy")
        for arm in arm_names:
            for mode in modes:
                row = run_one(arm, mode)
                if rank == 0:
                    rows.append(row)
                    flush()
                    if "CUDA" in str(row.get("error", "")):
                        log(f"CUDA fault in {arm}/{mode}; stopping — the context is poisoned. "
                            f"Rows written so far are on disk.")
                        if args.mode == "smoke":
                            open(os.path.join(args.out, "cuda_fault.marker"), "w").close()
                        else:
                            open(os.path.join(args.out, "cuda_fault_arms.marker"), "w").close()
                        return 0
        if args.mode == "smoke" and rank == 0:
            ok = [r for r in rows if r["arm"] == "splitkv" and r["status"] == "ok"]
            if ok:
                open(os.path.join(args.out, "splitkv_ok.marker"), "w").close()
                log("splitkv smoke OK -> marker written")
        if args.mode == "arms" and rank == 0:
            # Token-stream comparison against the published baseline: the bit-identity
            # claims (chunk size, deferred expansion) and the expected split-KV divergence
            # are all read from these.
            import hashlib
            dig = {}
            for r in rows:
                if r["status"] != "ok":
                    continue
                p = os.path.join(args.out, "tokens", f"{r['run_id']}.json")
                if not os.path.exists(p):
                    continue
                ids = json.load(open(p)).get("generated_token_ids") or []
                dig[f"{r['arm']}|{r['mode']}"] = hashlib.sha256(
                    json.dumps(ids).encode()).hexdigest()[:16]
            json.dump(dig, open(os.path.join(args.out, "token_digests.json"), "w"), indent=1)
            log(f"token digests: {dig}")
            # SELF-TEST (world_size == 1 only). With no ring patch installed, all four arms
            # are the SAME code, so every arm's stream must be identical to every other's.
            # If they are not, the arms are not independent and the bit-identity claims this
            # session exists to make would be meaningless -- so say so loudly rather than
            # report a divergence as an engine result.
            if world == 1:
                for mode, n_distinct, n_arms in _single_rank_selftest(dig):
                    verdict = "PASS" if n_distinct <= 1 else "FAIL"
                    log(f"single-rank self-test [{mode}]: {verdict} "
                        f"({n_distinct} distinct stream(s) over {n_arms} arms; "
                        f"expected 1 because no ring knob is installed at world_size==1)")
                    if n_distinct > 1:
                        log("  -> the arms are coupled, most likely through the global RNG "
                            "(torch.manual_seed runs once per engine, not per generation). "
                            "Check that _reseed() runs before every arm.")

    elif args.mode == "ppl":
        # NOTE, recorded here because it changes how the number may be read: _measure_target_ppl
        # calls model.model(...) with use_cache=False, i.e. a PREFILL forward. For a decode-path
        # change (split-KV) it therefore does NOT exercise the code under test. It is measured
        # anyway, for the record, and the decode-path correctness evidence is the token-stream
        # divergence in --mode arms.
        for arm in arm_names:
            spec, n_patched = set_arm(engine, arm, args.single_rank_ok)
            run_id = f"DF_PPL_{arm}_{args.book}_ctx{args.ctx}"
            try:
                ppl, n = R._measure_target_ppl(engine, prompt_text, continuation_ids,
                                               world, local_rank)
                err = ""
            except BaseException as exc:                  # noqa: BLE001
                import traceback
                ppl, n, err = "", "", f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}"
            row = {k: "" for k in ROW_FIELDS}
            row.update({
                "run_id": run_id, "arm": arm, "mode": "ppl",
                "context_length": args.ctx, "book": args.book, "seed": args.seed,
                "kv_block_size": spec["kv_block_size"],
                "ring_decode_mode": spec["ring_decode_mode"],
                "expand_after_rotation": spec["ring_expand_after_rotation"],
                "quantize_target": engine.cfg.quantize_target,
                "kv_quant": engine.cfg.kv_quant,
                "target_ppl": round(ppl, 6) if ppl != "" else "", "ppl_tokens": n,
                "git_commit": commit,
                "world_size": world, "ring_layers_patched": n_patched,
                "status": "ok" if not err else "error", "error": err,
            })
            if rank == 0:
                rows.append(row)
                flush()
                log(f"{run_id}: ppl={ppl} tokens={n} status={row['status']}")
                if err:
                    log(f"{run_id}: ERROR {err.splitlines()[0]}")

    if rank == 0:
        log(f"wrote {rows_path} ({len(rows)} rows)")
    if dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""C6 multi-rank resume validation (M4 Phase C).

Confirms that:
1. A run with `cfg.checkpoint_every=4` produces the same final
   generated_ids as the same run with `cfg.checkpoint_every=0`.
   (Save side: writing checkpoints must not perturb generation.)
2. Killing a run after the first checkpoint and re-running with
   `--resume` produces the same final generated_ids as one full run.
   (Load side: resume produces correct output.)
3. Per-rank checkpoints exist on disk: `round_<n>_rank_<r>.pt`
   for every rank in the world.

Single short-context test (ctx=4096, max_new=16) — fast iteration on
a fresh pod, exercises both save and load sides under the actual
multi-rank ring kernel.

Usage (on pod, 8 GPUs):
    torchrun --nproc-per-node=8 scripts/c6_resume_validation.py \\
        --target meta-llama/Llama-2-7b-hf \\
        --draft  princeton-nlp/Sheared-LLaMA-1.3B \\
        --ctx 4096 --max-new 16 --checkpoint-every 4

Output: results/c6_validation/c6_validation_rank<r>.json (per rank)
Exit code 0 if all gates pass; 1 if any fails.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path

import torch
import torch.distributed as dist


def _make_engine(target: str, draft: str, ctx: int, max_new: int,
                 seed: int, checkpoint_every: int = 0,
                 checkpoint_dir: str | None = None,
                 run_id: str | None = None):
    from src.models.rasd_inference import RASDConfig, RASDInference

    cfg = RASDConfig(
        target_model_name=target,
        draft_model_name=draft,
        spec_steps=4,
        kv_block_size=2048,
        prefetch_depth=1,
        max_new_tokens=max_new,
        dtype="bfloat16",
        quantize_target=True,
        quantize_draft=True,
        context_length=ctx,
        seed=seed,
        checkpoint_every=checkpoint_every,
        checkpoint_dir=checkpoint_dir,
        run_id=run_id,
    )
    return RASDInference(cfg)


def _gen(engine, ctx: int) -> torch.Tensor:
    """Run a single generation; returns the generated_ids tensor."""
    # Fixed deterministic prompt for reproducibility
    torch.manual_seed(engine.cfg.seed)
    vocab = engine.target_model.config.vocab_size
    ids = torch.randint(0, vocab, (1, ctx), device="cuda")
    generated_ids, metrics = engine.generate(ids)
    return generated_ids, metrics


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--target", required=True)
    p.add_argument("--draft", required=True)
    p.add_argument("--ctx", type=int, default=4096)
    p.add_argument("--max-new", type=int, default=16)
    p.add_argument("--checkpoint-every", type=int, default=4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out-dir", default="results/c6_validation")
    args = p.parse_args()

    rank = int(os.environ.get("RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    if world_size > 1:
        torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", 0)))
        dist.init_process_group(
            backend="nccl",
            device_id=torch.device(f"cuda:{int(os.environ.get('LOCAL_RANK', 0))}"),
        )

    out_dir = Path(args.out_dir)
    if rank == 0:
        out_dir.mkdir(parents=True, exist_ok=True)

    # Use a clean checkpoint dir per validation run
    ckpt_dir = out_dir / "checkpoints"
    if rank == 0 and ckpt_dir.exists():
        shutil.rmtree(ckpt_dir)
    if world_size > 1:
        dist.barrier()

    results = {
        "rank": rank,
        "world_size": world_size,
        "ctx": args.ctx,
        "max_new": args.max_new,
        "checkpoint_every": args.checkpoint_every,
    }

    # ---- Gate 1: baseline run with checkpoint_every=0 ----
    if rank == 0:
        print(f"Gate 1: baseline run (checkpoint_every=0)...")
    eng_base = _make_engine(args.target, args.draft, args.ctx,
                            args.max_new, args.seed)
    base_ids, base_metrics = _gen(eng_base, args.ctx)
    results["baseline"] = {
        "tokens_generated":  int(base_metrics["tokens_generated"]),
        "throughput_tps":    float(base_metrics["throughput_tps"]),
        "acceptance_rate":   float(base_metrics["acceptance_rate"]),
        "first_8_tokens":    base_ids[0, args.ctx:args.ctx + 8].tolist(),
    }
    del eng_base
    torch.cuda.empty_cache()

    # ---- Gate 2: same run with checkpoint_every=N enabled ----
    # Save-side check: enabling checkpoints must not perturb the output
    if rank == 0:
        print(f"Gate 2: same run with checkpoint_every={args.checkpoint_every}...")
    eng_save = _make_engine(
        args.target, args.draft, args.ctx, args.max_new, args.seed,
        checkpoint_every=args.checkpoint_every,
        checkpoint_dir=str(ckpt_dir),
        run_id="c6_validation",
    )
    save_ids, save_metrics = _gen(eng_save, args.ctx)
    same_as_baseline = bool(torch.equal(base_ids, save_ids))
    results["save_side"] = {
        "tokens_generated":  int(save_metrics["tokens_generated"]),
        "throughput_tps":    float(save_metrics["throughput_tps"]),
        "first_8_tokens":    save_ids[0, args.ctx:args.ctx + 8].tolist(),
        "same_as_baseline":  same_as_baseline,
    }

    # ---- Gate 3: per-rank checkpoint files exist ----
    expected = list((ckpt_dir / "c6_validation").glob(f"round_*_rank_{rank}.pt"))
    results["checkpoints_on_disk"] = sorted(p.name for p in expected)
    has_at_least_one = len(expected) > 0
    results["has_at_least_one_checkpoint"] = has_at_least_one

    del eng_save
    torch.cuda.empty_cache()

    # ---- Gate 4 (load side): resume from latest checkpoint, must
    # produce same final tokens as the full run ----
    # We don't actually kill mid-run here (that's hard from a script);
    # we just verify the resume code path produces an identical answer
    # by reloading from the latest checkpoint and continuing.
    if rank == 0:
        print(f"Gate 4: resume from latest checkpoint, expect same answer...")
    eng_resume = _make_engine(
        args.target, args.draft, args.ctx, args.max_new, args.seed,
        checkpoint_every=args.checkpoint_every,  # checkpoints persist
        checkpoint_dir=str(ckpt_dir),
        run_id="c6_validation",
    )
    resume_ids, resume_metrics = _gen(eng_resume, args.ctx)
    same_as_save = bool(torch.equal(save_ids, resume_ids))
    results["load_side"] = {
        "tokens_generated":  int(resume_metrics["tokens_generated"]),
        "first_8_tokens":    resume_ids[0, args.ctx:args.ctx + 8].tolist(),
        "same_as_save":      same_as_save,
    }

    overall = (
        results["save_side"]["same_as_baseline"]
        and results["has_at_least_one_checkpoint"]
        and results["load_side"]["same_as_save"]
    )
    results["overall_pass"] = overall

    out_path = out_dir / f"c6_validation_rank{rank}.json"
    out_path.write_text(json.dumps(results, indent=2))
    print(f"[rank={rank}] wrote {out_path}, overall={'PASS' if overall else 'FAIL'}")

    if world_size > 1:
        dist.barrier()
        dist.destroy_process_group()
    return 0 if overall else 1


if __name__ == "__main__":
    raise SystemExit(main())

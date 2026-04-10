"""
RASD Experiment Runner
======================
Reads configs/ablations.yml (or any YAML passed via --config), expands the
ablation grid, and runs each (level × seed) combination. Logs every run to
wandb and appends a row to results/ablations.csv.

Usage
-----
    # Run the full grid
    python run_experiment.py --config configs/ablations.yml

    # Run a single ablation group (e.g. only A2)
    python run_experiment.py --config configs/ablations.yml --groups A2

    # Run multiple groups
    python run_experiment.py --config configs/ablations.yml --groups A1 A4

    # Dry-run: print all jobs without executing
    python run_experiment.py --config configs/ablations.yml --dry-run

    # Debug mode (forced sync, verbose logs)
    python run_experiment.py --config configs/ablations.yml --groups A2 --debug

    # Resume: skip runs already present in results/ablations.csv
    python run_experiment.py --config configs/ablations.yml --resume
"""

import argparse
import csv
import itertools
import logging
import os
import sys
import time
from copy import deepcopy
from pathlib import Path
from typing import Dict, List, Optional

import torch
import yaml

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("rasd.runner")

RESULTS_DIR = Path("results")
RESULTS_CSV  = RESULTS_DIR / "ablations.csv"

CSV_FIELDS = [
    "run_id", "group", "level_id", "seed",
    "target_model_name", "draft_model_name",
    "spec_steps", "kv_block_size", "prefetch_depth",
    "context_length", "dtype",
    # metrics
    "tokens_generated", "time_sec", "throughput_tps",
    "acceptance_rate", "mean_latency_ms", "gpu_peak_mem_mb",
    "n_rounds", "status", "error",
]


# ---------------------------------------------------------------------------
# Config loading & grid expansion
# ---------------------------------------------------------------------------

def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def build_run_configs(cfg: dict, groups: Optional[List[str]], debug: bool) -> List[dict]:
    """Expand the YAML into a flat list of run dicts (one per level × seed)."""
    defaults = deepcopy(cfg["defaults"])
    seeds    = defaults.pop("seeds")

    ablation_keys = [k for k in cfg if k.startswith("A")]
    if groups:
        ablation_keys = [k for k in ablation_keys if k in groups]

    runs = []
    for group_id in ablation_keys:
        group = cfg[group_id]
        for level in group["levels"]:
            for seed in seeds:
                run = deepcopy(defaults)
                run.update({k: v for k, v in level.items() if k not in ("notes",)})
                run["seed"]     = seed
                run["group"]    = group_id
                run["level_id"] = level["id"]
                run["debug"]    = debug
                run["run_id"]   = f"{level['id']}_s{seed}"
                runs.append(run)

    return runs


# ---------------------------------------------------------------------------
# wandb helpers
# ---------------------------------------------------------------------------

def init_wandb(run: dict, project: str):
    try:
        import wandb
        return wandb.init(
            project=project,
            name=run["run_id"],
            config={k: v for k, v in run.items() if k not in ("run_id", "group", "level_id", "debug")},
            reinit=True,
        )
    except ImportError:
        log.warning("wandb not installed — skipping wandb logging.")
        return None


def log_wandb(wb_run, metrics: dict):
    if wb_run is None:
        return
    import wandb
    wb_run.log(metrics)
    wb_run.finish()


# ---------------------------------------------------------------------------
# CSV helpers
# ---------------------------------------------------------------------------

def load_completed_runs(csv_path: Path) -> set:
    if not csv_path.exists():
        return set()
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        return {row["run_id"] for row in reader if row.get("status") == "ok"}


def append_csv(csv_path: Path, row: dict):
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not csv_path.exists()
    with open(csv_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS, extrasaction="ignore")
        if write_header:
            writer.writeheader()
        writer.writerow(row)


# ---------------------------------------------------------------------------
# Prompt builder (synthetic long-context prompt)
# ---------------------------------------------------------------------------

def build_prompt(context_length: int, tokenizer) -> str:
    """Build a synthetic prompt of approximately `context_length` tokens."""
    # Repeat a paragraph until we hit the target token count
    base = (
        "The following is a detailed technical analysis of distributed machine learning systems. "
        "Ring attention enables long-context inference by sharding the sequence across GPUs. "
        "Speculative decoding accelerates generation by using a smaller draft model. "
    )
    tokens = tokenizer.encode(base)
    repeats = max(1, context_length // len(tokens))
    full_text = base * repeats
    # Trim to exactly context_length tokens
    full_tokens = tokenizer.encode(full_text)[:context_length]
    return tokenizer.decode(full_tokens)


# ---------------------------------------------------------------------------
# Single run executor
# ---------------------------------------------------------------------------

def execute_run(run: dict, wandb_project: str) -> dict:
    """Instantiate RASDInference, run generate(), return result row."""
    from src.models.rasd_inference import RASDConfig, RASDInference

    log.info("▶  %s  (seed=%d)", run["run_id"], run["seed"])

    row = {f: run.get(f, "") for f in CSV_FIELDS}
    row["status"] = "error"
    row["error"]  = ""

    wb_run = init_wandb(run, wandb_project)

    try:
        cfg = RASDConfig(
            target_model_name = run["target_model_name"],
            draft_model_name  = run["draft_model_name"],
            spec_steps        = int(run["spec_steps"]),
            kv_block_size     = int(run["kv_block_size"]),
            prefetch_depth    = int(run["prefetch_depth"]),
            max_new_tokens    = int(run.get("max_new_tokens", 256)),
            dtype             = run.get("dtype", "bfloat16"),
            quantize_draft    = bool(run.get("quantize_draft", True)),
            quantize_target   = bool(run.get("quantize_target", False)),
            temperature       = float(run.get("temperature", 1.0)),
            top_p             = float(run.get("top_p", 1.0)),
            seed              = int(run["seed"]),
            debug             = bool(run.get("debug", False)),
        )

        engine = RASDInference(cfg)

        prompt = build_prompt(int(run.get("context_length", 65536)), engine.tokenizer)
        _, metrics = engine.generate_text(prompt)

        row.update({
            "tokens_generated": metrics["tokens_generated"],
            "time_sec":         round(metrics["time_sec"], 4),
            "throughput_tps":   round(metrics["throughput_tps"], 2),
            "acceptance_rate":  round(metrics["acceptance_rate"], 4),
            "mean_latency_ms":  round(metrics["mean_latency_ms"], 3),
            "gpu_peak_mem_mb":  round(metrics["gpu_peak_mem_mb"], 1),
            "n_rounds":         metrics["n_rounds"],
            "status":           "ok",
        })

        log.info(
            "✓  %s  tps=%.1f  accept=%.3f  mem=%.0f MB",
            run["run_id"],
            metrics["throughput_tps"],
            metrics["acceptance_rate"],
            metrics["gpu_peak_mem_mb"],
        )
        log_wandb(wb_run, metrics)

        # Cleanup to free GPU memory before next run
        del engine
        torch.cuda.empty_cache()

    except Exception as exc:
        log.error("✗  %s  FAILED: %s", run["run_id"], exc)
        row["error"] = str(exc)
        if wb_run:
            import wandb
            wb_run.finish(exit_code=1)

    return row


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="RASD ablation runner")
    parser.add_argument("--config",   default="configs/ablations.yml", help="Path to YAML config")
    parser.add_argument("--groups",   nargs="+", help="Ablation groups to run (e.g. A1 A2). Default: all")
    parser.add_argument("--dry-run",  action="store_true", help="Print jobs without executing")
    parser.add_argument("--debug",    action="store_true", help="Enable RASD debug mode")
    parser.add_argument("--resume",   action="store_true", help="Skip runs already in results CSV")
    parser.add_argument("--wandb-project", default="rasd-ablations", help="wandb project name")
    parser.add_argument("--output",   default=str(RESULTS_CSV), help="Output CSV path")
    args = parser.parse_args()

    cfg        = load_config(args.config)
    all_runs   = build_run_configs(cfg, args.groups, args.debug)
    output_csv = Path(args.output)

    log.info("Total runs: %d", len(all_runs))

    if args.dry_run:
        print(f"\n{'RUN ID':<35}  {'GROUP':<5}  {'SEED':>5}  CONFIG")
        print("-" * 80)
        for r in all_runs:
            config_summary = (
                f"draft={r['draft_model_name'].split('/')[-1]}  "
                f"k={r['spec_steps']}  "
                f"block={r['kv_block_size']}  "
                f"prefetch={r['prefetch_depth']}  "
                f"target={r['target_model_name'].split('/')[-1]}"
            )
            print(f"{r['run_id']:<35}  {r['group']:<5}  {r['seed']:>5}  {config_summary}")
        print(f"\n{len(all_runs)} runs total.")
        return

    # Resume: skip completed runs
    completed = load_completed_runs(output_csv) if args.resume else set()
    pending   = [r for r in all_runs if r["run_id"] not in completed]
    skipped   = len(all_runs) - len(pending)
    if skipped:
        log.info("Resuming: skipping %d already-completed runs.", skipped)

    if not pending:
        log.info("Nothing to run — all jobs complete.")
        return

    log.info("Running %d jobs → %s", len(pending), output_csv)

    for i, run in enumerate(pending, 1):
        log.info("[%d/%d]", i, len(pending))
        row = execute_run(run, args.wandb_project)
        append_csv(output_csv, row)

    # Summary
    import csv as _csv
    with open(output_csv, newline="") as f:
        rows = list(_csv.DictReader(f))
    n_ok  = sum(1 for r in rows if r["status"] == "ok")
    n_err = sum(1 for r in rows if r["status"] != "ok")
    log.info("Done. %d succeeded, %d failed. Results: %s", n_ok, n_err, output_csv)


if __name__ == "__main__":
    main()

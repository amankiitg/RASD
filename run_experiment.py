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
import signal
import subprocess
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

def _ring_peer_loop(local_rank: int, world_size: int, kv_block_size: int,
                    max_rounds: int):
    """Non-rank-0 peer process for the ring KV communication.

    Runs for exactly `max_rounds` P2P rounds (= max_new_tokens from config).
    No collective signalling — eliminates the NCCL broadcast/all_reduce
    deadlock that occurred at kv_block_size≥1024.

    Ordering contract (mirrors generate() in rasd_inference.py):
      wait(prev_reqs) → batch_isend_irecv → ...

    Buffer shape: (num_layers=32, B=1, H=32, kv_block_size, head_dim=128)
    flat_size = 32 * 32 * kv_block_size * 128 elements (matches rank 0's k_send).
    """
    import torch.distributed as dist

    device = f"cuda:{local_rank}"
    torch.cuda.set_device(local_rank)
    send_to   = (local_rank + 1) % world_size
    recv_from = (local_rank - 1) % world_size

    flat_size = 32 * 32 * kv_block_size * 128   # num_layers * H * block * head_dim
    k_send = torch.zeros(flat_size, dtype=torch.bfloat16, device=device)
    v_send = torch.zeros(flat_size, dtype=torch.bfloat16, device=device)
    k_recv = torch.zeros(flat_size, dtype=torch.bfloat16, device=device)
    v_recv = torch.zeros(flat_size, dtype=torch.bfloat16, device=device)

    prev_reqs = []
    for _round in range(max_rounds):
        # Wait on prev P2P, relay received data into send buffer
        print(f"[TRACE peer rank={local_rank}] round={_round} draining {len(prev_reqs)} prev reqs", flush=True)
        for r in prev_reqs:
            r.wait()
        prev_reqs = []
        k_send.copy_(k_recv)
        v_send.copy_(v_recv)

        ops = [
            dist.P2POp(dist.isend, k_send, send_to),
            dist.P2POp(dist.isend, v_send, send_to),
            dist.P2POp(dist.irecv, k_recv, recv_from),
            dist.P2POp(dist.irecv, v_recv, recv_from),
        ]
        prev_reqs = dist.batch_isend_irecv(ops)
        print(f"[TRACE peer rank={local_rank}] round={_round} P2P submitted", flush=True)

    # Drain final round
    for r in prev_reqs:
        r.wait()
    print(f"[TRACE peer rank={local_rank}] all {max_rounds} rounds done", flush=True)


def _run_single_worker(run: dict, wandb_project: str, output_csv: str):
    """Worker executed in a subprocess — full isolation, fresh CUDA context.

    When launched via torchrun (nproc>1):
      - Rank 0 runs the full RASDInference pipeline and logs metrics.
      - Ranks 1..N-1 run _ring_peer_loop(), participating in P2P KV ring
        sends/receives so that rank 0's AsyncKVRingPrefetcher has live peers.
        This is what makes A3 (kv_block_size) and A4 (prefetch_depth) meaningful.
    Only rank 0 writes results to the CSV.
    """
    import json, sys
    import torch.distributed as dist
    from src.models.rasd_inference import RASDConfig, RASDInference

    # Init distributed if torchrun set the env vars
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    if world_size > 1:
        torch.cuda.set_device(local_rank)
        # device_id is required for PyTorch 2.x+ to eagerly bind the NCCL
        # communicator to a specific device. Without it, NCCL initialises
        # lazily on the first collective, which has caused subtle ordering
        # bugs (SeqNum mismatch deadlock) in the kv_block_size=1024/2048 runs.
        dist.init_process_group(
            backend="nccl",
            device_id=torch.device(f"cuda:{local_rank}"),
        )
        if local_rank != 0:
            # Non-rank-0: participate in ring comms only, then exit
            _ring_peer_loop(
                local_rank, world_size,
                int(run.get("kv_block_size", 512)),
                max_rounds=int(run.get("max_new_tokens", 256)),
            )
            dist.destroy_process_group()
            return
    else:
        torch.cuda.set_device(0)

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
        log_wandb(wb_run, metrics)
    except Exception as exc:
        row["error"] = str(exc)
        if wb_run:
            import wandb; wb_run.finish(exit_code=1)

    # Write results BEFORE destroy_process_group — destroy can hang if peers
    # are slow to reach the same barrier, and we don't want to lose metrics.
    if local_rank == 0:
        with open(output_csv + f".{run['run_id']}.tmp", "w") as f:
            json.dump(row, f)

    if world_size > 1:
        dist.destroy_process_group()


def execute_run(run: dict, wandb_project: str, output_csv: str, nproc: int = 1) -> dict:
    """Run a single ablation in an isolated subprocess to keep CUDA context clean.

    GPU state is contaminated after a CUDA error — running each job in its own
    process guarantees a fresh device context regardless of previous failures.

    When nproc > 1, launches via torchrun so the ring attention distributed
    primitives (A3 kv_block_size, A4 prefetch_depth) actually exercise multi-GPU
    communication. With nproc=1 the ring is a no-op (world_size=1).
    """
    import json, subprocess, sys

    log.info("▶  %s  (seed=%d, nproc=%d)", run["run_id"], run["seed"], nproc)

    tmp_result = output_csv + f".{run['run_id']}.tmp"

    worker_args = ["--_worker", json.dumps(run),
                   "--wandb-project", wandb_project, "--output", output_csv]

    if nproc > 1:
        # torchrun spawns nproc processes, each gets RANK/LOCAL_RANK/WORLD_SIZE set
        cmd = [
            "torchrun",
            f"--nproc_per_node={nproc}",
            "--master_port=29500",
            __file__,
        ] + worker_args
    else:
        cmd = [sys.executable, __file__] + worker_args

    env = {
        **os.environ,
        "WANDB_API_KEY": os.environ.get("WANDB_API_KEY", ""),
        "HF_TOKEN":      os.environ.get("HF_TOKEN", ""),
    }
    # Remove any stale CUDA_VISIBLE_DEVICES override — let torchrun/PyTorch manage devices
    env.pop("CUDA_VISIBLE_DEVICES", None)

    try:
        proc = subprocess.Popen(cmd, env=env, start_new_session=True)
        proc.wait(timeout=3600)
    except subprocess.TimeoutExpired:
        log.error("✗  %s  TIMEOUT after %ds", run["run_id"], 3600)
        # Kill entire process group (torchrun + all spawned workers)
        try:
            os.killpg(proc.pid, signal.SIGTERM)
        except OSError:
            pass
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except OSError:
                pass
            proc.wait()

    if os.path.exists(tmp_result):
        with open(tmp_result) as f:
            row = json.load(f)
        os.unlink(tmp_result)
    else:
        row = {f: run.get(f, "") for f in CSV_FIELDS}
        row["status"] = "error"
        row["error"]  = "subprocess produced no output"

    if row.get("status") == "ok":
        log.info("✓  %s  tps=%.1f  accept=%.3f  mem=%.0f MB",
                 run["run_id"], float(row["throughput_tps"]),
                 float(row["acceptance_rate"]), float(row["gpu_peak_mem_mb"]))
    else:
        log.error("✗  %s  FAILED: %s", run["run_id"], row.get("error", "")[:120])

    return row


# ---------------------------------------------------------------------------
# GPU cleanup between runs
# ---------------------------------------------------------------------------

def _wait_gpu_idle(pause: float = 5.0):
    """Wait for GPU processes from the previous run to fully exit.

    Runs `nvidia-smi` to check for any lingering Python processes on all GPUs.
    Polls until clear (or up to 60s), then sleeps an extra `pause` seconds so
    the NCCL store and CUDA context are fully torn down before the next torchrun.
    """
    import subprocess as _sp
    deadline = time.time() + 60
    while time.time() < deadline:
        try:
            out = _sp.check_output(
                ["nvidia-smi", "--query-compute-apps=pid,used_memory", "--format=csv,noheader"],
                text=True, stderr=_sp.DEVNULL,
            ).strip()
        except Exception:
            break   # nvidia-smi not available (e.g. local dev) — skip
        if not out:
            break   # no processes on any GPU
        log.debug("Waiting for GPU processes to exit:\n%s", out)
        time.sleep(3)
    time.sleep(pause)   # extra buffer for NCCL store teardown


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
    parser.add_argument("--nproc",    type=int, default=8,
                        help="GPUs per run (torchrun nproc_per_node). Use 1 for single-GPU. "
                             "Default 8 — required for ring attention A3/A4 ablations.")
    # Internal: subprocess worker mode
    parser.add_argument("--_worker",  default=None, help=argparse.SUPPRESS)
    args = parser.parse_args()

    # ---- Subprocess worker path ----
    if args._worker:
        import json
        run = json.loads(args._worker)
        output_csv = args.output
        _run_single_worker(run, args.wandb_project, output_csv)
        return

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

    # ---- Canary run: execute default config before the full grid ----
    canary_cfg = cfg.get("canary")
    if canary_cfg and not args.dry_run:
        completed_so_far = load_completed_runs(output_csv) if output_csv.exists() else set()
        canary_id = canary_cfg["id"]
        if canary_id in completed_so_far:
            log.info("Canary '%s' already passed — skipping.", canary_id)
        else:
            log.info("=== CANARY RUN: %s ===", canary_id)
            defaults = deepcopy(cfg["defaults"])
            defaults.pop("seeds")
            canary_run = {**defaults,
                          "run_id":   canary_id,
                          "group":    "canary",
                          "level_id": canary_id,
                          "seed":     int(canary_cfg.get("seed", 42)),
                          "max_new_tokens": int(canary_cfg.get("max_new_tokens", 32)),
                          "debug":    args.debug}
            canary_row = execute_run(canary_run, args.wandb_project, str(output_csv), nproc=args.nproc)
            append_csv(output_csv, canary_row)
            _wait_gpu_idle()
            if canary_row.get("status") != "ok":
                log.error("CANARY FAILED — aborting ablation grid. Error: %s",
                          canary_row.get("error", "unknown"))
                log.error("Fix the issue and re-run. The canary will be skipped once it passes.")
                return
            log.info("=== CANARY PASSED (tps=%.1f, accept=%.3f) — starting ablation grid ===",
                     float(canary_row.get("throughput_tps", 0)),
                     float(canary_row.get("acceptance_rate", 0)))

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
        row = execute_run(run, args.wandb_project, str(output_csv), nproc=args.nproc)
        append_csv(output_csv, row)
        # Wait for all GPU processes from this run to fully exit before starting
        # the next one — prevents CUDA/NCCL state pollution across runs.
        _wait_gpu_idle()

    # Summary
    import csv as _csv
    with open(output_csv, newline="") as f:
        rows = list(_csv.DictReader(f))
    n_ok  = sum(1 for r in rows if r["status"] == "ok")
    n_err = sum(1 for r in rows if r["status"] != "ok")
    log.info("Done. %d succeeded, %d failed. Results: %s", n_ok, n_err, output_csv)


if __name__ == "__main__":
    main()
